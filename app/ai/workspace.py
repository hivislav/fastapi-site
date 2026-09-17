"""Файловое хранилище рабочего пространства режима «AI-агент»: задачи и сессии.

В режиме «AI-агент» диалог ведётся не «вообще», а в рамках ЗАДАЧИ: у каждой
задачи свой список диалогов-сессий, а у каждой сессии — своё независимое
состояние агентского диалога (память сообщений, замеры расхода токенов, резюме
стратегии «summary» вместе с границей covered, блок фактов «sticky facts»,
ветви плана и активная ветка «branching»). Переключение задачи или сессии
заменяет всю эту «начинку» целиком — диалоги разных задач и сессий не
смешиваются.

Формат файла (по умолчанию data/agent_workspace.json):

    {"version": 1,
     "active_task": "t-1a2b3c4d",
     "active_tasks": {"user_13213123": "t-1a2b3c4d"},
     "tasks": [
       {"id": "t-1a2b3c4d",
        "profile": "user_13213123",
        "name": "Задача 1",
        "created": "2025-01-01T12:00:00",
        "active_session": "s-9f8e7d6c",
        "sessions": [
          {"id": "s-9f8e7d6c",
           "title": "",
           "created": "2025-01-01T12:00:10",
           "dialog": {"messages": [], "usage": [], "summary": [],
                      "covered": 0, "facts": {}, "branches": {},
                      "active_branch": null}}
        ]}
     ]}

У каждого профиля СВОЯ долговременная память ("long_term_by_profile": {"<id
профиля>": {"long_term": [...]}}), поэтому её записи видны и удаляются только в
своём профиле. Список "long_term" в корне — прежняя общая база знаний (до
профилей): при старте приложения он один раз передаётся текущему профилю и
очищается (см. migrate_legacy_long_term), чтобы «вечных» записей не оставалось.

Каждая задача (а значит и все её диалоги-сессии) принадлежит КОНКРЕТНОМУ
ПРОФИЛЮ пользователя (поле "profile" — id профиля из app/ai/profiles.py):
профили полностью изолированы друг от друга и не видят чужие задачи и
диалоги. "active_tasks" хранит текущую задачу КАЖДОГО профиля, поэтому
переключение профиля возвращает его собственную задачу (поле "active_task" —
прежний одиночный указатель: нужен только для чтения файлов прошлых версий).

"title" пустой — заголовок сессии берётся из ПЕРВЫХ СЛОВ первого запроса
пользователя (см. session_title); как только пользователь переименовал сессию
(карандаш в панели), хранится своё название.

Запись атомарная (временный файл + os.replace) — как в agent_memory.py.
Подразумевается один процесс-писатель: одновременные запросы внутри процесса
сериализованы блокировкой в chat.py.
"""

import json
import logging
import os
import tempfile
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from app import config
from app.ai import task_state

logger = logging.getLogger(__name__)

# Версия формата файла (на будущее — для миграций структуры).
_VERSION = 1

# Страховочные лимиты на загрузку: срабатывают только для повреждённого или
# вручную разросшегося файла, обычная работа до них не доходит.
_MAX_TASKS = 50
_MAX_SESSIONS = 100
_MAX_NAME = 120
_MAX_TITLE = 200
# Длины содержимого сессии — те же, что у одиночной истории агента
# (app/ai/agent_memory.py): память диалога, замеры токенов, части резюме,
# ветви плана с их независимыми историями и блок фактов.
_MAX_MESSAGES = 500
_MAX_TOTAL_CHARS = 200_000
_MAX_USAGE = 500
_MAX_SUMMARY = 500
_MAX_BRANCHES = 5
_MAX_BRANCH_MESSAGES = 500
_MAX_FACTS = 40
# Журнал чата сессии (dialog["log"]): что пользователь ВИДЕЛ в окне чата — его
# сообщения, ответы, показанный план и служебные debug/error-строки агента, в
# порядке появления. Нужен, чтобы при переключении сессии окно чата
# восстанавливалось полностью: память диалога (messages) хранит только реплики
# для модели, а debug-вывод и текст запроса в неё не попадают.
_MAX_LOG = 400
_MAX_LOG_CHARS = 200_000
_MAX_LOG_TEXT = 8000
# Виды записей журнала (совпадают с ролями узлов чата на фронте).
LOG_USER = "user"
LOG_ASSISTANT = "assistant"
LOG_DEBUG = "debug"
LOG_ERROR = "error"
LOG_KINDS = (LOG_USER, LOG_ASSISTANT, LOG_DEBUG, LOG_ERROR)

# Сколько символов первого запроса пользователя попадает в заголовок сессии.
_TITLE_CHARS = 120

# --- Слои памяти, которые наполняет пользователь ---------------------------
# Рабочая память (данные текущей ЗАДАЧИ) живёт в задаче — ключ "working";
# долговременная (глобальная база знаний) — в корне workspace, ключ "long_term":
# она одна на всё приложение, а не на задачу. Краткосрочная память — это
# сессии (dialog) со стратегиями, см. app/ai/agent.py.
MAX_MEMORY_ENTRIES = 100
MEMORY_ENTRY_LIMIT = 4000
# Ключи слоёв памяти (совпадают с именами в API: "work" | "long").
MEMORY_WORKING = "working"
MEMORY_LONG_TERM = "long_term"

# Заголовок пустой сессии (пока пользователь не отправил ни одного запроса).
EMPTY_SESSION_TITLE = "Новая задача"


def _now() -> str:
    """Метка времени создания задачи/сессии (ISO, до секунд)."""
    return datetime.now().isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    """Идентификатор задачи («t-…») или сессии («s-…»)."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def empty_dialog(task_id: str = "") -> Dict[str, Any]:
    """Пустое состояние агентского диалога (одна сессия).

    Кроме памяти, замеров, резюме, фактов и ветвей плана здесь лежит состояние
    конечного автомата задачи (`state`, см. app/ai/task_state.py): этап, текущий
    шаг и ожидаемое действие. Диалог без автомата не бывает — новое состояние
    создаётся сразу (этап planning).
    """
    return {
        "messages": [],
        "usage": [],
        "summary": [],
        "covered": 0,
        "facts": {},
        "branches": {},
        "active_branch": None,
        # Журнал чата (см. _MAX_LOG): порядок узлов окна чата этой сессии.
        "log": [],
        # Конечный автомат задачи: этап → шаг → ожидаемое действие + история
        # переходов. Живёт В СЕССИИ: у каждого диалога свой ход автомата.
        "state": task_state.to_dict(task_state.new_state(task_id)),
    }


# ---------------------------------------------------------------------------
# Нормализация прочитанных данных
# ---------------------------------------------------------------------------
def _clean_messages(raw: Any, limit: int = _MAX_MESSAGES) -> List[Dict[str, str]]:
    """Оставляет только пары {"role": user|assistant, "content": непустое}.

    Реплика конечного автомата задачи (запрос «выполни текущий шаг плана», см.
    ChatMessage.continue_step) помечена полем "source": "machine" — интерфейс
    рисует её служебным сообщением, а не репликой пользователя. Пометка
    сохраняется здесь, иначе после перезагрузки страницы такая реплика
    выглядела бы как написанная человеком.
    """
    clean: List[Dict[str, str]] = []
    for msg in (raw if isinstance(raw, list) else []):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = str(msg.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        item: Dict[str, str] = {"role": role, "content": content}
        source = str(msg.get("source") or "").strip().lower()
        if source:
            item["source"] = source[:20]
        clean.append(item)
    if len(clean) > limit:
        clean = clean[-limit:]
    total_chars = sum(len(m["content"]) for m in clean)
    while total_chars > _MAX_TOTAL_CHARS and clean:
        total_chars -= len(clean[0]["content"])
        clean.pop(0)
    return clean


def _clean_usage(raw: Any) -> List[Dict[str, Any]]:
    """Замеры расхода токенов по запросам пользователя (панель токенов)."""
    clean: List[Dict[str, Any]] = []
    for item in (raw if isinstance(raw, list) else [])[:_MAX_USAGE]:
        if not isinstance(item, dict):
            continue
        record: Dict[str, Any] = {}
        for key in ("requests", "input", "output", "summary_requests", "summary_input",
                    "summary_output"):
            if key in item:
                try:
                    record[key] = max(0, int(item.get(key) or 0))
                except (TypeError, ValueError):
                    continue
        if "limit" in item:
            try:
                record["limit"] = int(item["limit"]) if item.get("limit") else None
            except (TypeError, ValueError):
                record["limit"] = None
        if "overflow" in item:
            record["overflow"] = item.get("overflow") is True
        # kind="plan" — замер запроса, на котором построен только ПЛАН (ответа
        # пользователю не было): такие записи не привязаны к реплике диалога и
        # не отбрасываются при синхронизации замеров (см. _usage_matches_history).
        kind = str(item.get("kind") or "").strip().lower()
        if kind:
            record["kind"] = kind[:20]
        clean.append(record)
    return clean


def _clean_summary(raw: Any) -> List[str]:
    """Части резюме старой части переписки (стратегия «summary»)."""
    clean = [str(part).strip() for part in (raw if isinstance(raw, list) else [])]
    parts = [part for part in clean if part]
    return parts[-_MAX_SUMMARY:]


def _clean_facts(raw: Any) -> Dict[str, str]:
    """Блок фактов (стратегия «sticky facts»): «ключ: значение»."""
    if not isinstance(raw, dict):
        return {}
    facts: Dict[str, str] = {}
    for key, value in list(raw.items())[-_MAX_FACTS:]:
        name = str(key).strip()[:100]
        if not name:
            continue
        facts[name] = str(value).strip()[:600]
    return facts


def _clean_log(raw: Any) -> List[Dict[str, str]]:
    """Приводит журнал чата сессии к списку {"kind", "text"}.

    kind — user | assistant | debug | error (LOG_KINDS). Пустые записи
    отбрасываются, длинные тексты обрезаются, старые вытесняются
    (_MAX_LOG записей и _MAX_LOG_CHARS символов).
    """
    clean: List[Dict[str, str]] = []
    for item in (raw if isinstance(raw, list) else []):
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "").strip().lower()
        text = str(item.get("text") or "").strip()
        if kind not in LOG_KINDS or not text:
            continue
        clean.append({"kind": kind, "text": text[:_MAX_LOG_TEXT]})
    if len(clean) > _MAX_LOG:
        clean = clean[-_MAX_LOG:]
    total = sum(len(item["text"]) for item in clean)
    while total > _MAX_LOG_CHARS and clean:
        total -= len(clean[0]["text"])
        clean.pop(0)
    return clean


def add_log(dialog: Dict[str, Any], kind: str, text: str) -> None:
    """Добавляет узел в журнал чата сессии (что пользователь видит в окне).

    Пишется веб-слоем по ходу ответа: реплика пользователя, ответ агента,
    показанный план и служебные debug/error-строки. Так окно чата можно
    восстановить целиком при переключении сессии (см. GET /api/agent/history).
    """
    value = str(text or "").strip()
    if kind not in LOG_KINDS or not value:
        return
    log = dialog.setdefault("log", [])
    log.append({"kind": kind, "text": value[:_MAX_LOG_TEXT]})
    if len(log) > _MAX_LOG:
        del log[:len(log) - _MAX_LOG]
    total = sum(len(item.get("text") or "") for item in log)
    while total > _MAX_LOG_CHARS and len(log) > 1:
        total -= len(log[0].get("text") or "")
        log.pop(0)


def _clean_branches(raw: Any) -> Dict[str, Dict[str, Any]]:
    """Ветви плана (стратегия «branching») — у каждой своя история сообщений."""
    if not isinstance(raw, dict):
        return {}
    branches: Dict[str, Dict[str, Any]] = {}
    for branch_id, branch in list(raw.items())[:_MAX_BRANCHES]:
        if not isinstance(branch, dict):
            continue
        item = {key: value for key, value in branch.items() if key != "messages"}
        item["messages"] = _clean_messages(branch.get("messages"), _MAX_BRANCH_MESSAGES)
        branches[str(branch_id)] = item
    return branches


def normalize_dialog(raw: Any, task_id: str = "") -> Dict[str, Any]:
    """Приводит состояние диалога сессии к безопасному виду."""
    dialog = empty_dialog(task_id)
    if not isinstance(raw, dict):
        return dialog
    dialog["messages"] = _clean_messages(raw.get("messages"))
    dialog["usage"] = _clean_usage(raw.get("usage"))
    dialog["summary"] = _clean_summary(raw.get("summary"))
    try:
        covered = int(raw.get("covered") or 0)
    except (TypeError, ValueError):
        covered = 0
    dialog["covered"] = max(0, min(covered, len(dialog["messages"])))
    dialog["facts"] = _clean_facts(raw.get("facts"))
    dialog["branches"] = _clean_branches(raw.get("branches"))
    active = raw.get("active_branch")
    active = str(active) if active else None
    dialog["active_branch"] = active if active in dialog["branches"] else None
    # Состояние автомата задачи: битое/отсутствующее поле даёт новое состояние
    # (этап planning) — диалог без автомата работать не должен.
    dialog["state"] = task_state.to_dict(task_state.from_dict(raw.get("state"), task_id))
    dialog["log"] = _clean_log(raw.get("log"))
    return dialog


def _normalize_session(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    session_id = str(raw.get("id") or "").strip() or new_id("s")
    return {
        "id": session_id,
        "title": str(raw.get("title") or "").strip()[:_MAX_TITLE],
        "created": str(raw.get("created") or _now()),
        # task_id состояния — это id САМОЙ сессии: задача пользователя в режиме
        # «AI-агент» ведётся в диалоге (сессии) и своей сессии не имеет.
        "dialog": normalize_dialog(raw.get("dialog"), session_id),
    }


# ---------------------------------------------------------------------------
# Состояние автомата задачи (Task State Machine) внутри сессии
# ---------------------------------------------------------------------------
def dialog_state(session: Optional[Dict[str, Any]]) -> "task_state.TaskState":
    """Состояние автомата задачи из диалога сессии.

    Сессии нет — новое состояние (этап planning) без имени задачи: вызывающий
    код (GET /api/agent/state) отдаёт его фронту, чтобы полоса этапов была
    видна даже до создания диалога.
    """
    dialog = (session or {}).get("dialog") or {}
    return task_state.from_dict(dialog.get("state"), str((session or {}).get("id") or ""))


def set_dialog_state(session: Dict[str, Any], state: "task_state.TaskState") -> Dict[str, Any]:
    """Записывает состояние автомата обратно в диалог сессии.

    Возвращает JSON-словарь состояния — его и отдаём фронту.
    """
    if not state.task_id:
        state.task_id = str(session.get("id") or "")
    state.steps = task_state.clean_steps(state.steps)
    session.setdefault("dialog", {})["state"] = task_state.to_dict(state)
    return session["dialog"]["state"]


def _normalize_entries(raw: Any) -> List[Dict[str, Any]]:
    """Приводит слой памяти к списку записей {id, text, created, source}.

    Принимает как готовые словари (формат файла), так и голые строки —
    на случай ручной правки JSON. Пустые записи отбрасываются, длинные
    обрезаются, число записей ограничено MAX_MEMORY_ENTRIES (старые вытесняются).
    """
    entries: List[Dict[str, Any]] = []
    for item in (raw if isinstance(raw, list) else []):
        if isinstance(item, dict):
            text = str(item.get("text") or "").strip()
            entry_id = str(item.get("id") or "").strip()
            created = str(item.get("created") or "")
            source = str(item.get("source") or "")
        else:
            text = str(item or "").strip()
            entry_id, created, source = "", "", ""
        if not text:
            continue
        entries.append({
            "id": entry_id or new_id("m"),
            "text": text[:MEMORY_ENTRY_LIMIT],
            "created": created or _now(),
            "source": source,
        })
    return entries[-MAX_MEMORY_ENTRIES:]


def memory(container: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    """Список записей слоя памяти в контейнере.

    key = MEMORY_WORKING — рабочая память задачи (контейнер — задача),
    key = MEMORY_LONG_TERM — долговременная память (контейнер — workspace).
    Нормализует поле на месте, если его нет или оно повреждено.
    """
    if not isinstance(container.get(key), list):
        container[key] = _normalize_entries(container.get(key))
    return container[key]


def memory_texts(container: Dict[str, Any], key: str) -> List[str]:
    """Тексты записей слоя памяти — в таком виде их получает агент."""
    return [entry["text"] for entry in memory(container, key)]


def _normalize_long_term_by_profile(raw: Any) -> Dict[str, Dict[str, Any]]:
    """Приводит долговременную память по профилям к безопасному виду.

    Формат: {"<id профиля>": [записи слоя], ...} — у каждого профиля своя база
    знаний. Пустые профили отбрасываются.
    """
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for profile_id, entries in list(raw.items())[:_MAX_TASKS]:
        key = str(profile_id or "").strip()[:_MAX_NAME]
        if not key:
            continue
        # Значение может быть как списком записей, так и контейнером
        # {"id", "long_term": [...]} — принимаем оба вида.
        if isinstance(entries, dict):
            entries = entries.get(MEMORY_LONG_TERM)
        normalized = _normalize_entries(entries)
        if normalized:
            out[key] = {MEMORY_LONG_TERM: normalized}
    return out


def long_term_container(workspace: Dict[str, Any],                        profile_id: Optional[str] = None) -> Dict[str, Any]:
    """Контейнер долговременной памяти ПРОФИЛЯ.

    Профили изолированы, поэтому долговременная память (глобальная база знаний
    профиля) живёт в workspace по ключу "long_term_by_profile": у каждого профиля
    свой список записей, и чужие он не видит. Профиль не передан — отдаём сам
    workspace (прежнее поведение: слой в корне) вместе с унаследованными
    записями прежних версий.
    """
    key = str(profile_id or "").strip()
    if not key:
        return workspace
    by_profile = workspace.setdefault("long_term_by_profile", {})
    if not isinstance(by_profile, dict):
        by_profile = {}
        workspace["long_term_by_profile"] = by_profile
    container = by_profile.get(key)
    if not isinstance(container, dict):
        container = {"id": key, "long_term": []}
        by_profile[key] = container
    return container


def long_term_texts(workspace: Dict[str, Any],
                    profile_id: Optional[str] = None) -> List[str]:
    """Тексты долговременной памяти ПРОФИЛЯ (только его собственные записи).

    Записи в корне workspace («long_term») — это база знаний ВРЕМЁН ДО ПРОФИЛЕЙ:
    они не принадлежат ни одному профилю и раньше примешивались к каждому из них,
    из-за чего их нельзя было удалить (см. migrate_legacy_long_term — при старте
    они один раз переезжают в текущий профиль, а корень очищается).
    """
    return memory_texts(long_term_container(workspace, profile_id), MEMORY_LONG_TERM)


def migrate_legacy_long_term(workspace: Dict[str, Any],
                             profile_id: Optional[str]) -> bool:
    """Переносит прежнюю (общую) долговременную память в текущий профиль.

    База знаний до появления профилей лежала в корне workspace и была видна всем
    профилям сразу: удалить её из интерфейса было нельзя (в панели видны только
    записи профиля), поэтому у пользователя оставались «вечные» записи вроде
    любимого фильма. Здесь она ОДИН РАЗ передаётся текущему профилю (он станет её
    владельцем, а значит сможет удалить записи в панели памяти и они уйдут вместе
    с профилем), после чего корень очищается — в контекст агента эти записи
    больше не попадают ни у одного профиля, кроме владельца.

    True — что-то перенесли (workspace нужно сохранить).
    """
    legacy = memory(workspace, MEMORY_LONG_TERM)
    if not legacy:
        return False
    key = str(profile_id or "").strip()
    if not key:
        # Профиля нет (не должно случаться: он заводится при старте) — оставляем
        # записи в корне, чтобы ничего не потерять.
        return False
    own = memory(long_term_container(workspace, key), MEMORY_LONG_TERM)
    own[:0] = list(legacy)          # прежние записи идут первыми (они старше)
    del own[:-MAX_MEMORY_ENTRIES]
    workspace[MEMORY_LONG_TERM] = []
    logger.info(
        "Workspace AI-агента: прежняя долговременная память (%d записей) передана "
        "профилю %s и убрана из корня", len(legacy), key,
    )
    return True


def add_memory(container: Dict[str, Any], key: str, text: str,
               source: str = "") -> Optional[Dict[str, Any]]:
    """Добавляет запись в слой памяти (кнопки «добавить в … память»).

    source — откуда запись: "user" (сообщение пользователя) или "assistant"
    (ответ агента). Пустой текст не добавляется (None).
    """
    text = str(text or "").strip()
    if not text:
        return None
    entry = {
        "id": new_id("m"),
        "text": text[:MEMORY_ENTRY_LIMIT],
        "created": _now(),
        "source": str(source or ""),
    }
    entries = memory(container, key)
    entries.append(entry)
    del entries[:-MAX_MEMORY_ENTRIES]
    return entry


def delete_memory(container: Dict[str, Any], key: str, entry_id: str) -> bool:
    """Удаляет запись слоя памяти по id. True — запись была и удалена."""
    entries = memory(container, key)
    for index, entry in enumerate(entries):
        if entry.get("id") == entry_id:
            entries.pop(index)
            return True
    return False


def _normalize_task(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name") or "").strip()[:_MAX_NAME]
    if not name:
        return None
    sessions: List[Dict[str, Any]] = []
    for item in (raw.get("sessions") if isinstance(raw.get("sessions"), list) else []):
        session = _normalize_session(item)
        if session is not None:
            sessions.append(session)
    sessions = sessions[-_MAX_SESSIONS:]
    active = str(raw.get("active_session") or "") or None
    if active not in [s["id"] for s in sessions]:
        active = sessions[-1]["id"] if sessions else None
    return {
        "id": str(raw.get("id") or "").strip() or new_id("t"),
        # Профиль-владелец задачи: задачи (и их диалоги) разных профилей
        # полностью изолированы. Пустая строка — задача из файла прежней версии
        # (до появления профилей): её подберёт текущий профиль при первой
        # миграции, см. adopt_orphan_tasks.
        "profile": str(raw.get("profile") or "").strip()[:_MAX_NAME],
        "name": name,
        "created": str(raw.get("created") or _now()),
        "sessions": sessions,
        "active_session": active,
        # Рабочая память задачи — записи, добавленные пользователем вручную
        # (кнопка «добавить в рабочую память»); стратегиям контекста не подчиняется.
        "working": _normalize_entries(raw.get("working")),
    }


# ---------------------------------------------------------------------------
# Чтение/запись файла
# ---------------------------------------------------------------------------
def _read_payload(path: Optional[str] = None) -> Any:
    file_path = path or config.AGENT_WORKSPACE_FILE
    try:
        if not os.path.isfile(file_path):
            return None
        with open(file_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("Workspace AI-агента: не удалось прочитать %s: %s", file_path, exc)
        return None


def _migrate_legacy() -> Optional[Dict[str, Any]]:
    """Переносит прежнюю единую историю агента в первую задачу/сессию.

    До появления задач история режима «AI-агент» была одна на приложение
    (data/agent_memory.json). Если файла workspace ещё нет, а старая история
    непуста — заводим задачу «Задача 1» с одной сессией, чтобы прежняя
    переписка (вместе с панелью токенов, резюме, фактами и ветвями плана) не
    потерялась.
    """
    from app.ai import agent_memory

    messages = agent_memory.load_agent_memory()
    if not messages:
        return None
    legacy_branches = agent_memory.load_agent_branches()
    task = {
        "id": new_id("t"),
        "name": "Задача 1",
        "created": _now(),
        "sessions": [{
            "id": new_id("s"),
            "title": "",
            "created": _now(),
            "dialog": {
                "messages": messages,
                "usage": agent_memory.load_agent_usage(),
                "summary": agent_memory.load_agent_summary(),
                "covered": agent_memory.load_agent_covered(),
                "facts": agent_memory.load_agent_facts(),
                "branches": legacy_branches.get("branches") or {},
                "active_branch": legacy_branches.get("active"),
            },
        }],
    }
    task["active_session"] = task["sessions"][0]["id"]
    logger.info("Workspace AI-агента: перенёс прежнюю историю диалога в задачу «Задача 1»")
    return normalize_workspace({"version": _VERSION, "active_task": task["id"], "tasks": [task]})


def normalize_workspace(raw: Any) -> Dict[str, Any]:
    """Приводит прочитанный workspace к безопасному виду."""
    if not isinstance(raw, dict):
        raw = {}
    tasks: List[Dict[str, Any]] = []
    for item in (raw.get("tasks") if isinstance(raw.get("tasks"), list) else []):
        task = _normalize_task(item)
        if task is not None:
            tasks.append(task)
    tasks = tasks[-_MAX_TASKS:]
    ids = [t["id"] for t in tasks]
    # Текущая задача КАЖДОГО профиля: у профилей свои задачи и диалоги.
    raw_active = raw.get("active_tasks") if isinstance(raw.get("active_tasks"), dict) else {}
    active_tasks: Dict[str, str] = {}
    for profile_id, task_id in list(raw_active.items())[:_MAX_TASKS]:
        key = str(profile_id or "").strip()[:_MAX_NAME]
        value = str(task_id or "").strip()
        if key and value in ids:
            active_tasks[key] = value
    # Прежний одиночный указатель (файл прошлой версии): приписываем его той
    # задаче, к которой он относился, — иначе он потерялся бы после появления
    # профилей. Новый формат его больше не обновляет.
    legacy_active = str(raw.get("active_task") or "").strip()
    if legacy_active in ids:
        legacy_task = next(t for t in tasks if t["id"] == legacy_active)
        owner = legacy_task.get("profile") or ""
        if owner and owner not in active_tasks:
            active_tasks[owner] = legacy_active
    else:
        # Указатель на несуществующую задачу (остался от прежних версий или от
        # удалённых задач) — просто мусор в файле, не храним его.
        legacy_active = ""
    return {
        "version": _VERSION,
        "active_task": legacy_active or None,
        "active_tasks": active_tasks,
        "tasks": tasks,
        # Долговременная память — база знаний ПРОФИЛЯ (профили изолированы):
        # словарь «id профиля -> записи». Записи в корне (прежних версий, когда
        # профилей ещё не было) остаются как общее наследие.
        "long_term": _normalize_entries(raw.get("long_term")),
        "long_term_by_profile": _normalize_long_term_by_profile(raw.get("long_term_by_profile")),
    }


def load_workspace(path: Optional[str] = None) -> Dict[str, Any]:
    """Загружает workspace из JSON-файла (или создаёт пустой).

    Файла нет / он повреждён — пустой workspace (задач нет), без исключений.
    Если файла нет, но есть прежняя единая история агента — она переносится в
    задачу «Задача 1» (см. _migrate_legacy).
    """
    data = _read_payload(path)
    if data is None:
        migrated = _migrate_legacy() if path is None else None
        if migrated is None:
            return normalize_workspace({})
        # Перенесённую историю сразу записываем в файл workspace: иначе при
        # следующем старте перенос выполнялся бы снова.
        try:
            save_workspace(migrated)
        except OSError as exc:
            logger.warning("Workspace AI-агента: не удалось записать перенесённую историю: %s", exc)
        return migrated
    return normalize_workspace(data)


def workspace_payload(workspace: Dict[str, Any]) -> str:
    """Сериализует workspace в JSON (синхронно, без await).

    Отдельно от записи намеренно: сериализация идёт в ПОТОКЕ EVENT LOOP, а файл
    пишется в отдельном потоке. Пока агент отвечает, workspace меняют и другие
    маршруты (переключение диалога, пауза, запись журнала), поэтому снимок для
    файла нужно снимать атомарно — иначе json.dumps мог бы поймать «изменение
    словаря во время итерации» (см. _persist в chat.py).
    """
    return json.dumps(normalize_workspace(workspace), ensure_ascii=False, indent=2)


def write_payload(payload: str, path: Optional[str] = None) -> None:
    """Пишет готовый JSON в файл атомарно (временный файл + os.replace)."""
    file_path = path or config.AGENT_WORKSPACE_FILE
    directory = os.path.dirname(file_path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".workspace-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp_path, file_path)
    except OSError:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def save_workspace(workspace: Dict[str, Any], path: Optional[str] = None) -> None:
    """Сохраняет workspace в JSON-файл атомарно (временный файл + os.replace).

    Обёртка над workspace_payload + write_payload — для инструментов и тестов,
    которым не важна развязка сериализации и записи (в веб-слое см. _persist).
    """
    write_payload(workspace_payload(workspace), path)


# ---------------------------------------------------------------------------
# Доступ к задачам и сессиям
# ---------------------------------------------------------------------------
def find_task(workspace: Dict[str, Any], task_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """Задача по id (None — такой задачи нет)."""
    for task in workspace.get("tasks", []):
        if task.get("id") == task_id:
            return task
    return None


def task_owner(task: Optional[Dict[str, Any]]) -> str:
    """id профиля-владельца задачи ("" — задача без профиля, старая версия файла)."""
    return str((task or {}).get("profile") or "").strip()


def task_belongs(task: Optional[Dict[str, Any]], profile_id: Optional[str]) -> bool:
    """True, если задача принадлежит этому профилю.

    Профили полностью изолированы: задача (и все её диалоги) видны только своему
    профилю. Задача без владельца (файл прежней версии) доступна любому профилю,
    пока её не подберёт adopt_orphan_tasks.
    """
    if task is None:
        return False
    owner = task_owner(task)
    if not owner:
        return True
    return owner == str(profile_id or "").strip()


def profile_tasks(workspace: Dict[str, Any], profile_id: Optional[str]) -> List[Dict[str, Any]]:
    """Задачи одного профиля (в порядке создания)."""
    return [task for task in workspace.get("tasks", [])
            if task_belongs(task, profile_id)]


def active_task(workspace: Dict[str, Any],
                profile_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Текущая задача ПРОФИЛЯ (None — у профиля ещё нет задач).

    profile_id=None — задачи без владельца (для служебных вызовов и тестов);
    обычная работа всегда передаёт id профиля, поэтому профили не видят чужие
    задачи и диалоги.
    """
    key = str(profile_id or "").strip()
    candidates = [task for task in workspace.get("tasks", [])
                  if task_owner(task) == key]
    if not candidates:
        return None
    active_id = (workspace.get("active_tasks") or {}).get(key)
    task = find_task(workspace, active_id)
    if task is not None and task_owner(task) == key:
        return task
    return candidates[-1]


def find_session(task: Optional[Dict[str, Any]], session_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """Сессия задачи по id."""
    if not task:
        return None
    for session in task.get("sessions", []):
        if session.get("id") == session_id:
            return session
    return None


def active_session(workspace: Dict[str, Any],
                   task: Optional[Dict[str, Any]] = None,
                   profile_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Текущая сессия текущей задачи профиля (None — сессии ещё нет)."""
    task = task if task is not None else active_task(workspace, profile_id)
    if not task:
        return None
    return find_session(task, task.get("active_session"))


def create_task(workspace: Dict[str, Any], name: str,
                profile_id: Optional[str] = None) -> Dict[str, Any]:
    """Создаёт задачу ПРОФИЛЯ (и делает её текущей у этого профиля).

    Задача создаётся без сессий — первый запрос пользователя заведёт сессию сам.
    Задача привязана к профилю (поле "profile"), поэтому другие профили её не
    видят.
    """
    key = str(profile_id or "").strip()
    task = {
        "id": new_id("t"),
        "profile": key,
        "name": str(name).strip()[:_MAX_NAME],
        "created": _now(),
        "sessions": [],
        "active_session": None,
        # Рабочая память новой задачи пуста: её наполняет пользователь кнопкой
        # «добавить в рабочую память».
        "working": [],
    }
    workspace.setdefault("tasks", []).append(task)
    workspace["tasks"] = workspace["tasks"][-_MAX_TASKS:]
    workspace.setdefault("active_tasks", {})[key] = task["id"]
    return task


def create_session(task: Dict[str, Any], title: str = "") -> Dict[str, Any]:
    """Создаёт в задаче новую сессию-диалог и делает её текущей."""
    session_id = new_id("s")
    session = {
        "id": session_id,
        "title": str(title).strip()[:_MAX_TITLE],
        "created": _now(),
        # Новый диалог — новая задача для автомата: состояние создаётся сразу
        # (этап planning), а его task_id — id этой сессии.
        "dialog": empty_dialog(session_id),
    }
    task.setdefault("sessions", []).append(session)
    task["sessions"] = task["sessions"][-_MAX_SESSIONS:]
    task["active_session"] = session["id"]
    return session


def set_active_task(workspace: Dict[str, Any], task: Dict[str, Any]) -> None:
    """Делает задачу текущей у ЕЁ профиля (у каждого профиля своя текущая)."""
    key = task_owner(task)
    workspace.setdefault("active_tasks", {})[key] = task["id"]


def delete_task(workspace: Dict[str, Any], task_id: str) -> bool:
    """Удаляет задачу вместе со всеми её диалогами. True — задача была."""
    tasks = workspace.get("tasks", [])
    for index, task in enumerate(tasks):
        if task.get("id") == task_id:
            tasks.pop(index)
            active_tasks = workspace.setdefault("active_tasks", {})
            key = task_owner(task)
            if active_tasks.get(key) == task_id:
                # Текущей становится последняя оставшаяся задача этого профиля.
                rest = [t for t in tasks if task_owner(t) == key]
                if rest:
                    active_tasks[key] = rest[-1]["id"]
                else:
                    active_tasks.pop(key, None)
            return True
    return False


def delete_session(task: Dict[str, Any], session_id: str) -> bool:
    """Удаляет сессию задачи. Активной становится соседняя (или ни одной)."""
    sessions = task.get("sessions", [])
    for index, session in enumerate(sessions):
        if session.get("id") == session_id:
            sessions.pop(index)
            if task.get("active_session") == session_id:
                if sessions:
                    neighbour = sessions[min(index, len(sessions) - 1)]
                    task["active_session"] = neighbour["id"]
                else:
                    task["active_session"] = None
            return True
    return False


def purge_profile(workspace: Dict[str, Any], profile_id: Optional[str]) -> int:
    """Убирает профиль из workspace ЦЕЛИКОМ: его задачи, диалоги и память.

    Профиль удалён — значит его данных в файле быть не должно: задачи вместе с
    сессиями (и всей их начинкой: память диалога, замеры токенов, резюме, факты,
    ветви плана) удаляются, указатель на текущую задачу и долговременная память —
    тоже. Возвращает число удалённых ЗАДАЧ (0 — удалять было нечего), чтобы
    веб-слой мог сказать об этом пользователю.
    """
    key = str(profile_id or "").strip()
    if not key:
        return 0
    tasks = workspace.get("tasks", [])
    kept = [task for task in tasks if task_owner(task) != key]
    purged = len(tasks) - len(kept)
    if purged:
        workspace["tasks"] = kept
    workspace.get("active_tasks", {}).pop(key, None)
    workspace.get("long_term_by_profile", {}).pop(key, None)
    if purged:
        logger.info("Workspace AI-агента: профиль %s удалён вместе с %d задачами",
                    key, purged)
    return purged


def session_title(session: Optional[Dict[str, Any]]) -> str:
    """Заголовок сессии для панели истории.

    Своё название (пользователь переименовал карандашом) — как есть. Иначе —
    первые слова ПЕРВОГО запроса пользователя в этой сессии (как в панели
    истории введённых данных); сессия без единого запроса — «Новый диалог».
    """
    if not session:
        return EMPTY_SESSION_TITLE
    title = str(session.get("title") or "").strip()
    if title:
        return title
    for message in session.get("dialog", {}).get("messages", []):
        # Реплика автомата (source="machine", см. continue_step) — не запрос
        # пользователя: по ней заголовок не строим.
        if message.get("role") == "user" and message.get("source") != "machine":
            text = " ".join(str(message.get("content") or "").split())
            if text:
                return text[:_TITLE_CHARS] + ("…" if len(text) > _TITLE_CHARS else "")
    return EMPTY_SESSION_TITLE


def adopt_orphan_tasks(workspace: Dict[str, Any], profile_id: Optional[str],
                       profile_name: str = "") -> List[Dict[str, Any]]:
    """Отдаёт текущему профилю задачи без владельца (файл прежней версии).

    До появления профилей workspace был общим для всех: задачи не имели поля
    "profile" и были видны всем. При первом запуске новой версии такие задачи
    достаются ТЕКУЩЕМУ профилю (обычно это профиль, созданный автоматически) —
    иначе вся прежняя переписка осталась бы ничей. Возвращает принятые задачи.
    """
    key = str(profile_id or "").strip()
    if not key:
        return []
    adopted = [task for task in workspace.get("tasks", []) if not task_owner(task)]
    if not adopted:
        return []
    for task in adopted:
        task["profile"] = key
    # Текущей у профиля становится та задача, что была текущей у старого
    # одиночного указателя, иначе — последняя принятая.
    legacy = str(workspace.get("active_task") or "").strip()
    adopted_ids = [task["id"] for task in adopted]
    active = legacy if legacy in adopted_ids else adopted_ids[-1]
    workspace.setdefault("active_tasks", {})[key] = active
    logger.info(
        "Workspace AI-агента: задачи без профиля (%d) закреплены за профилем %s%s",
        len(adopted), key, f" «{profile_name}»" if profile_name else "",
    )
    return adopted


def profile_ids_with_data(workspace: Dict[str, Any]) -> List[str]:
    """Список id профилей, данные которых лежат в workspace (задачи/память).

    Нужен веб-слою при старте: если в файле остались данные профилей, которых
    в data/profiles.json уже нет (наследие прежних версий, когда удаление профиля
    их не убирало), это НЕ удаляется автоматически — об этом пишется
    предупреждение в лог, а убрать их можно инструментом
    tools/cleanup_workspace.py. Никакие данные существующих профилей не трогаем.
    """
    owners = {task_owner(task) for task in workspace.get("tasks", [])}
    owners |= set(workspace.get("long_term_by_profile") or {})
    owners |= set(workspace.get("active_tasks") or {})
    owners.discard("")
    return sorted(owners)


def _task_brief(task: Dict[str, Any]) -> Dict[str, Any]:
    """Краткая запись задачи для снимка фронтенда."""
    return {"id": task["id"], "name": task["name"]}


def snapshot(workspace: Dict[str, Any], profile_id: Optional[str] = None) -> Dict[str, Any]:
    """Снимок для фронтенда: задачи ПРОФИЛЯ, его текущая задача и её диалоги.

    Профили изолированы: в снимке только задачи переданного профиля (и его
    текущая задача из active_tasks), поэтому чужой профиль их не видит. В каждой
    задаче есть поле "profile" — id профиля-владельца.
    """
    tasks = profile_tasks(workspace, profile_id)
    task = active_task(workspace, profile_id)
    sessions = [
        {"id": session["id"], "title": session_title(session)}
        for session in (task.get("sessions", []) if task else [])
    ]
    return {
        "tasks": [_task_brief(item) for item in tasks],
        "active_task": task["id"] if task else None,
        "sessions": sessions,
        "active_session": task.get("active_session") if task else None,
        # Число записей в слоях памяти — в панели «Состояние памяти».
        "memory_count": {
            "working": len(memory(task, MEMORY_WORKING)) if task else 0,
            "long_term": len(memory(workspace, MEMORY_LONG_TERM)),
        },
    }


def memory_snapshot(workspace: Dict[str, Any],
                    profile_id: Optional[str] = None) -> Dict[str, Any]:
    """Содержимое слоёв памяти, наполняемых пользователем (панель справа).

    Рабочая память — данные ТЕКУЩЕЙ задачи ТЕКУЩЕГО профиля, долговременная —
    база знаний ЭТОГО профиля (профили изолированы). Краткосрочная память
    (диалоги сессий) в этот снимок не входит: она живёт в dialog каждой сессии и
    в панель токенов.
    """
    task = active_task(workspace, profile_id)
    return {
        "task": ({"id": task["id"], "name": task["name"]} if task else None),
        "working": [dict(entry) for entry in (memory(task, MEMORY_WORKING) if task else [])],
        "long_term": [dict(entry) for entry in memory(
            long_term_container(workspace, profile_id), MEMORY_LONG_TERM)],
    }
