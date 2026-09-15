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
     "tasks": [
       {"id": "t-1a2b3c4d",
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
EMPTY_SESSION_TITLE = "Новый диалог"


def _now() -> str:
    """Метка времени создания задачи/сессии (ISO, до секунд)."""
    return datetime.now().isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    """Идентификатор задачи («t-…») или сессии («s-…»)."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def empty_dialog() -> Dict[str, Any]:
    """Пустое состояние агентского диалога (одна сессия)."""
    return {
        "messages": [],
        "usage": [],
        "summary": [],
        "covered": 0,
        "facts": {},
        "branches": {},
        "active_branch": None,
    }


# ---------------------------------------------------------------------------
# Нормализация прочитанных данных
# ---------------------------------------------------------------------------
def _clean_messages(raw: Any, limit: int = _MAX_MESSAGES) -> List[Dict[str, str]]:
    """Оставляет только пары {"role": user|assistant, "content": непустое}."""
    clean: List[Dict[str, str]] = []
    for msg in (raw if isinstance(raw, list) else []):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = str(msg.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        clean.append({"role": role, "content": content})
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


def normalize_dialog(raw: Any) -> Dict[str, Any]:
    """Приводит состояние диалога сессии к безопасному виду."""
    dialog = empty_dialog()
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
    return dialog


def _normalize_session(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    session_id = str(raw.get("id") or "").strip() or new_id("s")
    return {
        "id": session_id,
        "title": str(raw.get("title") or "").strip()[:_MAX_TITLE],
        "created": str(raw.get("created") or _now()),
        "dialog": normalize_dialog(raw.get("dialog")),
    }


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
    active = str(raw.get("active_task") or "") or None
    if active not in [t["id"] for t in tasks]:
        active = tasks[-1]["id"] if tasks else None
    return {
        "version": _VERSION,
        "active_task": active,
        "tasks": tasks,
        # Долговременная память — глобальная база знаний (одна на приложение,
        # не на задачу); наполняется пользователем вручную.
        "long_term": _normalize_entries(raw.get("long_term")),
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


def save_workspace(workspace: Dict[str, Any], path: Optional[str] = None) -> None:
    """Сохраняет workspace в JSON-файл атомарно (временный файл + os.replace)."""
    file_path = path or config.AGENT_WORKSPACE_FILE
    directory = os.path.dirname(file_path) or "."
    os.makedirs(directory, exist_ok=True)
    payload = json.dumps(
        normalize_workspace(workspace), ensure_ascii=False, indent=2
    )
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


# ---------------------------------------------------------------------------
# Доступ к задачам и сессиям
# ---------------------------------------------------------------------------
def find_task(workspace: Dict[str, Any], task_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """Задача по id (None — такой задачи нет)."""
    for task in workspace.get("tasks", []):
        if task.get("id") == task_id:
            return task
    return None


def active_task(workspace: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Текущая задача (None — ни одной задачи ещё не создано)."""
    return find_task(workspace, workspace.get("active_task"))


def find_session(task: Optional[Dict[str, Any]], session_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """Сессия задачи по id."""
    if not task:
        return None
    for session in task.get("sessions", []):
        if session.get("id") == session_id:
            return session
    return None


def active_session(workspace: Dict[str, Any],
                   task: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Текущая сессия текущей задачи (None — сессии ещё нет)."""
    task = task if task is not None else active_task(workspace)
    if not task:
        return None
    return find_session(task, task.get("active_session"))


def create_task(workspace: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Создаёт задачу (и делает её текущей). Задача создаётся без сессий —
    первый запрос пользователя заведёт сессию сам."""
    task = {
        "id": new_id("t"),
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
    workspace["active_task"] = task["id"]
    return task


def create_session(task: Dict[str, Any], title: str = "") -> Dict[str, Any]:
    """Создаёт в задаче новую сессию-диалог и делает её текущей."""
    session = {
        "id": new_id("s"),
        "title": str(title).strip()[:_MAX_TITLE],
        "created": _now(),
        "dialog": empty_dialog(),
    }
    task.setdefault("sessions", []).append(session)
    task["sessions"] = task["sessions"][-_MAX_SESSIONS:]
    task["active_session"] = session["id"]
    return session


def delete_task(workspace: Dict[str, Any], task_id: str) -> bool:
    """Удаляет задачу вместе со всеми её диалогами. True — задача была."""
    tasks = workspace.get("tasks", [])
    for index, task in enumerate(tasks):
        if task.get("id") == task_id:
            tasks.pop(index)
            if workspace.get("active_task") == task_id:
                workspace["active_task"] = tasks[-1]["id"] if tasks else None
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
        if message.get("role") == "user":
            text = " ".join(str(message.get("content") or "").split())
            if text:
                return text[:_TITLE_CHARS] + ("…" if len(text) > _TITLE_CHARS else "")
    return EMPTY_SESSION_TITLE


def snapshot(workspace: Dict[str, Any]) -> Dict[str, Any]:
    """Снимок для фронтенда: задачи, текущая задача, её диалоги и текущий диалог."""
    task = active_task(workspace)
    sessions = [
        {"id": session["id"], "title": session_title(session)}
        for session in (task.get("sessions", []) if task else [])
    ]
    return {
        "tasks": [
            {"id": item["id"], "name": item["name"]}
            for item in workspace.get("tasks", [])
        ],
        "active_task": task["id"] if task else None,
        "sessions": sessions,
        "active_session": task.get("active_session") if task else None,
        # Число записей в слоях памяти — в панели «Состояние памяти».
        "memory_count": {
            "working": len(memory(task, MEMORY_WORKING)) if task else 0,
            "long_term": len(memory(workspace, MEMORY_LONG_TERM)),
        },
    }


def memory_snapshot(workspace: Dict[str, Any]) -> Dict[str, Any]:
    """Содержимое слоёв памяти, наполняемых пользователем (панель справа).

    Рабочая память — данные ТЕКУЩЕЙ задачи, долговременная — глобальная база
    знаний (одна на приложение). Краткосрочная память (диалоги сессий) в этот
    снимок не входит: она живёт в dialog каждой сессии и в панель токенов.
    """
    task = active_task(workspace)
    return {
        "task": ({"id": task["id"], "name": task["name"]} if task else None),
        "working": [dict(entry) for entry in (memory(task, MEMORY_WORKING) if task else [])],
        "long_term": [dict(entry) for entry in memory(workspace, MEMORY_LONG_TERM)],
    }
