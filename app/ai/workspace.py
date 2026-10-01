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
           "periodic": {"enabled": true, "interval": 3600, "request": "…",
                        "next_run": "2025-01-01T13:00:10", "last_run": "",
                        "runs": 0, "error": ""},
           "dialog": {"messages": [], "usage": [], "summary": [],
                      "covered": 0, "facts": {}, "branches": {},
                      "active_branch": null}}
        ]}
     ]}

Поле "periodic" есть только у ПЕРИОДИЧЕСКОЙ задачи (кнопка «Новая периодическая
задача»): это расписание, по которому сервер сам повторяет запрос задачи и кладёт
ответ в её чат (см. app/ai/periodic.py и app/periodic_runner.py). Задача без
расписания выполняется один раз и поля "periodic" не имеет.

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
import shutil
import tempfile
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from app import config
from app.ai import attachments as attach_store
from app.ai import invariants as invariants_store
from app.ai import mcp as mcp_store
from app.ai import periodic as periodic_store
from app.ai import rag_chunking
from app.ai import rag_search
from app.ai import rag_store
from app.ai import task_state

logger = logging.getLogger(__name__)

# Версия формата файла (на будущее — для миграций структуры).
_VERSION = 1

# Страховочные лимиты на загрузку: срабатывают только для повреждённого или
# вручную разросшегося файла, обычная работа до них не доходит.
_MAX_TASKS = 50
# Режимы экспертной страницы статистики («Статистика ответов»): по каждому
# копятся верные и неверные ответы. Счётчики живут в workspace ПРОФИЛЯ, поэтому
# переживают переключение режима, смену задачи и перезапуск приложения.
EXPERT_MODES = ("direct", "stepwise", "prompt", "group")
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
# Показ разбора инвариантов: текст объяснения + КЛИКАБЕЛЬНЫЕ варианты решения
# (анализ запроса: нарушение требования инвариантом или конфликт правил). Отдельный
# вид узла, потому что у него есть структура, а не только текст.
LOG_SUGGESTIONS = "suggestions"
# Автозапуск ПЕРИОДИЧЕСКОЙ задачи: реплика отправлена не пользователем, а
# планировщиком по расписанию (см. app/periodic_runner.py). Отдельный вид узла,
# чтобы в окне чата было видно, что задача повторилась САМА, и с каким периодом.
LOG_PERIODIC = "periodic"
LOG_KINDS = (LOG_USER, LOG_ASSISTANT, LOG_DEBUG, LOG_ERROR, LOG_SUGGESTIONS,
             LOG_PERIODIC)
# Разбор инвариантов в журнале: сколько вариантов и полей храним.
_MAX_SUGGESTIONS = 4
_SUGGESTION_TEXT = 600

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

# Инварианты (правила, которые агент не имеет права нарушить) — см.
# app/ai/invariants.py. Хранятся ОТДЕЛЬНО от диалога: у ПРОЕКТА (задача
# workspace) свои правила — поле "invariants" задачи, у ЗАДАЧИ-диалога (сессия)
# свои — поле "invariants" диалога. В переписку (messages) они не попадают.
MAX_INVARIANTS = 50
INVARIANT_ENTRY_LIMIT = 1000
# Проверки пар «инвариант проекта × инвариант задачи» и оставшиеся без проверки
# пары — в диалоге задачи (см. app/ai/invariants.py).
_MAX_CONFLICTS = 250
_MAX_UNCHECKED = 250

# MCP (внешние инструменты агента) — см. app/ai/mcp.py. Настройка ПРОЕКТА: поле
# "mcp" задачи-workspace хранит, какие серверы включены (у каждой задачи-проекта
# свой набор). Включает и выключает их пользователь в диалоге «MCP» по кнопке
# рядом с шестерёнкой проекта; включённые серверы уходят в каждый запрос агента.
MCP_FIELD = "mcp"
# Сколько ВНЕШНИХ СБОРОВ (наблюдений, подписок) помним на задачу: список нужен,
# чтобы отмена или удаление задачи остановили их на сервере (см. app/ai/mcp.py).
_MAX_MCP_STARTED = 20
# Сколько серверов может быть включено одновременно (серверов в реестре
# app/ai/mcp.py сейчас три; запас на будущее — как MAX_INVARIANTS).
MAX_MCP_SERVERS = 10

# RAG (базы знаний проекта) — см. app/ai/rag.py, кнопка «RAG» рядом с
# кнопкой «MCP». Настройка ПРОЕКТА, как и MCP: поле "rag" задачи-workspace хранит, какие
# базы знаний включены у проекта, и последние выбранные параметры разбиения на
# чанки (стратегия, размер чанка, перекрытие) — их диалог подставляет при
# следующем открытии. САМИ базы знаний живут отдельно (data/rag, см.
# app/ai/rag_store.py) и принадлежат профилю: включить можно только свою базу,
# поэтому здесь проверяется ФОРМА идентификатора, а существование и владельца
# проверяет маршрут (app/ai/rag.py: filter_enabled).
RAG_FIELD = "rag"
MAX_RAG_BASES = 50

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
        # Инварианты ЗАДАЧИ (правила этого диалога) и проверки их на
        # противоречие с инвариантами проекта: см. app/ai/invariants.py.
        "invariants": [],
        # Решения пользователя по противоречиям правил («главнее проект/задача»),
        # принятые В ДИАЛОГЕ: {"key", "project_id", "task_id", "winner"}.
        "exceptions": [],
        "conflicts": [],
        "unchecked": [],
        # Конечный автомат задачи: этап → шаг → ожидаемое действие + история
        # переходов. Живёт В СЕССИИ: у каждого диалога свой ход автомата.
        "state": task_state.to_dict(task_state.new_state(task_id)),
        # Подпись плана: для какого запроса и при каком наборе правил он
        # построен. Нужна, чтобы НЕ строить план заново (это платный служебный
        # вызов) и НЕ проверять его повторно, когда ни запрос, ни правила не
        # изменились, — например, при перезапуске задачи после ошибки.
        "plan_signature": {},
        # Данные внешних инструментов MCP по ТЕКУЩЕМУ запросу задачи (см.
        # app/ai/mcp.py): подпись «включённые серверы + исходный запрос» и
        # результаты вызовов. Шаги плана и проверка результата идут отдельными
        # запросами и берут данные отсюда — иначе второй шаг отвечал бы уже без
        # них. Новый запрос пользователя меняет подпись, и данные собираются
        # заново.
        "mcp": {},
        # ВНЕШНИЕ СБОРЫ, ЗАПУЩЕННЫЕ этой задачей (наблюдения, подписки, задания):
        # по ним отмена/удаление задачи останавливает работу на САМОМ сервере —
        # иначе сбор остался бы висеть там навсегда.
        "mcp_started": [],
        # ФРАГМЕНТЫ БАЗ ЗНАНИЙ (RAG) по ТЕКУЩЕМУ запросу задачи (см.
        # app/ai/rag_search.py): подпись «базы + их отпечаток + запрос» и сами
        # найденные фрагменты. Живут в диалоге по той же причине, что и данные
        # MCP: шаг плана и проверка результата приходят ОТДЕЛЬНЫМИ запросами, а
        # векторный поиск стоит времени — искать заново на каждом шаге незачем.
        # Новый запрос (или переиндексация базы) меняет подпись, и поиск идёт
        # заново.
        "rag": {},
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
    """Замеры расхода токенов по запросам пользователя (панель токенов).

    Берём ПОСЛЕДНИЕ замеры (_MAX_USAGE): панель показывает текущий диалог, а
    срез [:N] отбрасывал бы как раз свежие запросы при длинной переписке.
    """
    clean: List[Dict[str, Any]] = []
    for item in (raw if isinstance(raw, list) else [])[-_MAX_USAGE:]:
        if not isinstance(item, dict):
            continue
        record: Dict[str, Any] = {}
        for key in ("requests", "input", "output", "summary_requests", "summary_input",
                    "summary_output", "failed_requests"):
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
        if "cost_rub" in item:
            try:
                record["cost_rub"] = round(max(0.0, float(item.get("cost_rub") or 0.0)), 5)
            except (TypeError, ValueError):
                record["cost_rub"] = 0.0
        # Разбивка служебных вызовов по видам (план, гейт, разбор запроса,
        # проверка результата, сжатие, факты, ветвление): панель показывает
        # вклад каждого вида отдельной строкой.
        service = item.get("service")
        if isinstance(service, dict):
            buckets: Dict[str, Dict[str, int]] = {}
            for kind, bucket in service.items():
                if not isinstance(bucket, dict):
                    continue
                clean_bucket: Dict[str, int] = {}
                for field in ("requests", "input", "output", "failed"):
                    try:
                        clean_bucket[field] = max(0, int(bucket.get(field) or 0))
                    except (TypeError, ValueError):
                        clean_bucket[field] = 0
                buckets[str(kind)[:20]] = clean_bucket
            if buckets:
                record["service"] = buckets
        # kind="plan" — замер запроса, на котором построен только ПЛАН (ответа
        # пользователю не было); kind="service" — другой служебный запрос без
        # ответа (например, отказ по инвариантам). Такие записи не привязаны к
        # реплике диалога и не отбрасываются при синхронизации замеров
        # (см. _usage_matches_history в app/routers/chat.py).
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


def _clean_log(raw: Any) -> List[Dict[str, Any]]:
    """Приводит журнал чата сессии к списку записей журнала.


    kind — user | assistant | debug | error | suggestions | periodic (LOG_KINDS).
    Пустые записи отбрасываются, длинные тексты обрезаются, старые вытесняются
    (_MAX_LOG записей и _MAX_LOG_CHARS символов). У узла suggestions дополнительно
    лежит разбор инвариантов (объяснение + варианты решения), по нему интерфейс
    рисует кликабельные варианты — в том числе после перезагрузки страницы.
    """
    clean: List[Dict[str, Any]] = []
    for item in (raw if isinstance(raw, list) else []):
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "").strip().lower()
        text = str(item.get("text") or "").strip()
        if kind not in LOG_KINDS or not text:
            continue
        entry: Dict[str, Any] = {
            "kind": kind, "text": text[:_MAX_LOG_TEXT],
            # Время узла (у старых файлов его нет — тогда интерфейс времени не
            # показывает: выдумывать его нельзя).
            "at": str(item.get("at") or "").strip()[:40],
        }
        if kind == LOG_SUGGESTIONS:
            # Узел разбора хранится вместе с текстом сообщения: даже если вариантов
            # нет (старая запись без analysis или разбор без альтернатив), текст
            # сообщения агента терять нельзя — он уже показан пользователю.
            analysis = _clean_analysis(item.get("analysis"))
            if analysis["suggestions"] or analysis["explanation"]:
                entry["analysis"] = analysis
        files = _clean_files(item.get("files"))
        if files:
            # ФАЙЛЫ, полученные от MCP-инструмента: карточки со ссылкой на
            # скачивание под сообщением. Хранятся в журнале, поэтому видны и
            # после переключения задачи/перезагрузки страницы.
            entry["files"] = files
        sources = _clean_sources(item.get("sources"))
        if sources:
            # ФРАГМЕНТЫ баз знаний, которые были у модели в этом ответе: под
            # сообщением рисуются карточки источников (файл, раздел, близость).
            entry["sources"] = sources
        clean.append(entry)
    if len(clean) > _MAX_LOG:
        clean = clean[-_MAX_LOG:]
    total = sum(len(item["text"]) for item in clean)
    while total > _MAX_LOG_CHARS and clean:
        total -= len(clean[0]["text"])
        clean.pop(0)
    return clean


def add_log(dialog: Dict[str, Any], kind: str, text: str,
            at: str = "", files: Any = None, sources: Any = None) -> None:
    """Добавляет узел в журнал чата сессии (что пользователь видит в окне).

    Пишется веб-слоем по ходу ответа: реплика пользователя, ответ агента,
    показанный план и служебные debug/error-строки. Так окно чата можно
    восстановить целиком при переключении сессии (см. GET /api/agent/history).

    `at` — время узла (наивное локальное, как метки задач). Интерфейс показывает
    его у реплик пользователя и ответов агента, как в мессенджерах; служебные
    строки (debug/error) времени не показывают, но оно всё равно хранится —
    по нему видно порядок и когда именно шла работа.

    `files` — вложения MCP (xlsx и т. п.), полученные по этому запросу: узел
    рисуется с карточками файлов и ссылками на скачивание.
    `sources` — фрагменты баз знаний, которые были у модели в этом ответе:
    узел рисуется с карточками источников (файл, раздел, близость к запросу).
    """
    value = str(text or "").strip()
    if kind not in LOG_KINDS or not value:
        return
    log = dialog.setdefault("log", [])
    entry: Dict[str, Any] = {"kind": kind, "text": value[:_MAX_LOG_TEXT],
                             "at": str(at or "").strip()[:40] or _now()}
    clean_files = _clean_files(files)
    if clean_files:
        entry["files"] = clean_files
    clean_sources = _clean_sources(sources)
    if clean_sources:
        entry["sources"] = clean_sources
    log.append(entry)
    if len(log) > _MAX_LOG:
        del log[:len(log) - _MAX_LOG]
    total = sum(len(item.get("text") or "") for item in log)
    while total > _MAX_LOG_CHARS and len(log) > 1:
        total -= len(log[0].get("text") or "")
        log.pop(0)


def add_log_event(dialog: Dict[str, Any], kind: str, text: str,
                  analysis: Any = None, at: str = "") -> None:
    """Добавляет в журнал узел со структурой (разбор инвариантов).

    Отличается от add_log() только тем, что вместе с текстом сохраняет сам
    разбор: по нему интерфейс рисует кликабельные варианты решения и после
    переключения задачи/перезагрузки страницы.
    """
    add_log(dialog, kind, text, at=at)
    log = dialog.get("log") or []
    if kind != LOG_SUGGESTIONS or not log:
        return
    clean = _clean_analysis(analysis)
    if clean["suggestions"] or clean["explanation"]:
        log[-1]["analysis"] = clean


def _clean_analysis(raw: Any) -> Dict[str, Any]:
    """Разбор инвариантов для журнала: объяснение + варианты решения."""
    from app.ai import invariants as invariants_store

    data = invariants_store.normalize_analysis(raw)
    suggestions: List[Dict[str, Any]] = []
    for item in (data.get("suggestions") or [])[:_MAX_SUGGESTIONS]:
        title = str(item.get("title") or "").strip()[:_SUGGESTION_TEXT]
        send = str(item.get("send") or "").strip()[:_SUGGESTION_TEXT]
        if not title or not send:
            continue
        suggestions.append({
            "title": title,
            "details": str(item.get("details") or "").strip()[:_SUGGESTION_TEXT],
            "send": send,
        })
    explanation = str(data.get("explanation") or "").strip()
    return {
        "kind": str(data.get("kind") or "").strip().lower()[:20],
        "explanation": explanation[:_MAX_LOG_TEXT],
        # Текст сообщения агента: у снимка для фронта он в поле message, у
        # «сырого» разбора его нет — тогда повторяем объяснение.
        "message": (str(data.get("message") or "").strip() or explanation)[:_MAX_LOG_TEXT],
        "suggestions": suggestions,
        # Отметка, что варианты ПРОВЕРЕНЫ по правилам (см. invariants.analyze):
        # без неё клик по варианту после перезагрузки считался бы непроверенным
        # (см. POST /api/agent/invariants/choose). Пустой разбор проверять нечего.
        "suggestions_checked": (bool(data.get("suggestions_checked"))
                                if suggestions else True),
        # Номера правил задачи, противоречащих правилам проекта (не действуют):
        # по ним текст отказа говорит, что правило задачи поправляется в
        # «Инварианты», а снимок инвариантов помечает его недействующим.
        "overridden": [int(number) for number in (data.get("overridden") or [])
                       if str(number).strip().isdigit()][:invariants_store.MAX_OVERRIDDEN],
        # Подпись ПРАВИЛ, при которых варианты проверены: если правила не
        # менялись, повторно судить выбранный вариант не нужно (см.
        # `_verified_choice` в chat.py).
        "rules_signature": str(data.get("rules_signature") or "")[:64],
    }


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
    # Инварианты ЗАДАЧИ (правила этого диалога) и проверки их на противоречие с
    # инвариантами проекта: живут отдельно от переписки (см. app/ai/invariants.py).
    dialog["invariants"] = _normalize_invariants(raw.get("invariants"))
    dialog["conflicts"] = _clean_conflicts(raw.get("conflicts"))
    dialog["unchecked"] = _clean_unchecked(raw.get("unchecked"))
    # Решения по противоречиям правил: пишутся, когда пользователь выбрал в
    # диалоге, чьё правило главнее (проверок правил при их записи нет).
    decisions = invariants_store.merge_exceptions(
        raw.get("conflicts"), raw.get("exceptions"))
    dialog["exceptions"] = [dict(item) for item in decisions]
    dialog["plan_signature"] = _clean_plan_signature(raw.get("plan_signature"))
    # Данные внешних инструментов MCP по текущему запросу задачи (см.
    # app/ai/mcp.py): подпись + результаты вызовов.
    dialog["mcp"] = _normalize_dialog_mcp(raw.get("mcp"))
    # Фрагменты баз знаний (RAG) по текущему запросу задачи: подпись + сами
    # фрагменты (см. app/ai/rag_search.py). Без подписи запись не хранится —
    # значит, неизвестно, к какому запросу эти фрагменты.
    dialog["rag"] = _normalize_dialog_rag(raw.get("rag"))
    # Внешние сборы, запущенные задачей: их отменяет отмена/удаление задачи.
    dialog["mcp_started"] = _normalize_started(raw.get("mcp_started"))
    return dialog


def _normalize_dialog_mcp(raw: Any) -> Dict[str, Any]:
    """Данные MCP диалога: {"signature", "request", "calls", "results", "chain"}.

    Хранится то, что уже получено: подпись (включённые серверы + исходный запрос
    задачи), САМИ вызовы и их результаты. Вызовы нужны повторам периодической
    задачи: тот же запрос — те же ЧИТАЮЩИЕ вызовы, только данные свежие
    (см. _preflight_mcp); выбор инструментов моделью заново не оплачивается и не
    «плывёт» от повтора к повтору.

    `chain` — состояние ЦЕПОЧКИ вызовов (идентификаторы, ключи вызовов, сколько
    раундов сделано). Нужно повторам периодической задачи: они перезаписывают ТОТ
    ЖЕ набор данных (тот же dataset_id + флаг перезаписи), а не создают новый на
    каждом повторе. Битое — пусто.
    """
    if not isinstance(raw, dict):
        return {}
    signature = str(raw.get("signature") or "").strip()
    if not signature:
        return {}
    out = {
        "signature": signature[:600],
        "request": str(raw.get("request") or "")[:400],
        "calls": mcp_store.normalize_calls(raw.get("calls"),
                                           limit=mcp_store.MAX_TOTAL_CALLS_PER_REQUEST),
        "results": mcp_store.normalize_results(raw.get("results")),
    }
    chain = _clean_chain(raw.get("chain"))
    if chain:
        out["chain"] = chain
    return out


def _clean_chain(raw: Any) -> Dict[str, Any]:
    """Состояние цепочки: {"ids", "keys", "iterations", "pending"}.

    `pending` — цепочка НЕ ДОИГРАНА: до подтверждения плана выполнялись только
    чтения, а вызовы, меняющие что-то на сервере (сохранение набора, выгрузка
    файла), отложены до первого шага выполнения. По этому признаку `_preflight_mcp`
    продолжает цепочку после «ок».
    """
    if not isinstance(raw, dict):
        return {}
    ids: Dict[str, str] = {}
    for key, value in list((raw.get("ids") or {}).items())[:20]:
        name = str(key or "").strip()[:60]
        text = str(value or "").strip()[:80]
        if name and text:
            ids[name] = text
    keys = [str(item)[:300] for item in (raw.get("keys") or []) if str(item or "").strip()]
    try:
        iterations = max(0, min(20, int(raw.get("iterations") or 0)))
    except (TypeError, ValueError):
        iterations = 0
    pending = bool(raw.get("pending"))
    if not ids and not keys and not pending:
        return {}
    return {"ids": ids, "keys": list(dict.fromkeys(keys))[:20],
            "iterations": iterations, "pending": pending}


def _clean_files(raw: Any) -> List[Dict[str, Any]]:
    """Вложения MCP для журнала чата (карточки файлов со ссылкой на скачивание)."""
    return attach_store.normalize(raw)


def dialog_mcp(dialog: Dict[str, Any]) -> Dict[str, Any]:
    """Данные MCP диалога (нормализует поле на месте, если его нет)."""
    if not isinstance(dialog.get("mcp"), dict):
        dialog["mcp"] = _normalize_dialog_mcp(dialog.get("mcp"))
    return dialog["mcp"]


def _normalize_started(raw: Any) -> List[Dict[str, Any]]:
    """Внешние сборы, ЗАПУЩЕННЫЕ задачей (наблюдения, подписки; см. app/ai/mcp.py).

    Хранится вместе с диалогом задачи, потому что отменять их нужно и ПОСЛЕ того,
    как диалог переписан: отмена или удаление задачи обязаны остановить то, что
    она запустила на серверах, — иначе там останется висеть вечный сбор, о
    котором задача уже забыла.
    """
    out: List[Dict[str, Any]] = []
    for item in (raw if isinstance(raw, list) else []):
        if not isinstance(item, dict):
            continue
        server_id = str(item.get("server") or "").strip()[:80]
        tool = str(item.get("tool") or "").strip()[:120]
        stop_tool = str(item.get("stop_tool") or "").strip()[:120]
        if not server_id or not tool or not stop_tool:
            continue
        arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
        entry = {
            "server": server_id,
            "server_name": str(item.get("server_name") or server_id)[:120],
            "tool": tool,
            "stop_tool": stop_tool,
            "arguments": {str(key)[:60]: str(value)[:200]
                          for key, value in list(arguments.items())[:10]},
        }
        if entry not in out:
            out.append(entry)
    return out[-_MAX_MCP_STARTED:]


def mcp_started(dialog: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Внешние сборы задачи (нормализует поле на месте, если его нет)."""
    if not isinstance(dialog.get("mcp_started"), list):
        dialog["mcp_started"] = _normalize_started(dialog.get("mcp_started"))
    return dialog["mcp_started"]


def add_mcp_started(dialog: Dict[str, Any], entries: Any) -> List[Dict[str, Any]]:
    """Запоминает внешние сборы, запущенные задачей (дубликаты отбрасываются)."""
    current = mcp_started(dialog)
    for entry in _normalize_started(entries):
        if entry not in current:
            current.append(entry)
    del current[:-_MAX_MCP_STARTED]
    return current


def clear_mcp_started(dialog: Dict[str, Any], entries: Any = None) -> List[Dict[str, Any]]:
    """Убирает обязательства из памяти задачи (отменённые — отменять не нужно).

    Возвращает снятые записи. `entries` — какие именно сняты; None — снять все.
    """
    current = mcp_started(dialog)
    if entries is None:
        dialog["mcp_started"] = []
        return current
    drop = _normalize_started(entries)
    kept = [item for item in current if item not in drop]
    dialog["mcp_started"] = kept
    return [item for item in current if item in drop]


def set_dialog_mcp(dialog: Dict[str, Any], signature: str, request: str,
                   results: Any, calls: Any = None,
                   chain: Any = None) -> Dict[str, Any]:
    """Запоминает данные MCP текущего запроса задачи: подпись, вызовы, результаты.

    Вызовы (`calls`) хранятся, чтобы повтор периодической задачи выполнил те же
    ЧИТАЮЩИЕ вызовы со свежими данными, а не спрашивал модель заново: выбор
    инструментов иначе «плыл» бы от повтора к повтору (в живой задаче повтор
    решил, что данные не нужны, и ответ ушёл без погоды).

    `chain` — состояние цепочки (идентификаторы сохранённых наборов, ключи вызовов
    и признак «цепочка не доиграна, ждёт подтверждения плана»). None — не трогаем
    прежнее состояние: промежуточные сохранения раундов не должны его терять.

    Состояние цепочки НЕ переносится на другой запрос: если подпись сменилась
    (пользователь поправил запрос), отложенная часть прежнего запроса не должна
    «дотянуться» до нового — иначе после «ок» по новому запросу выполнялись бы
    сохранение и выгрузка прежнего.
    """
    previous = dialog.get("mcp") if isinstance(dialog.get("mcp"), dict) else {}
    same_request = str(previous.get("signature") or "") == str(signature or "")
    keep_chain = chain if chain is not None else (
        previous.get("chain") if same_request else None)
    dialog["mcp"] = _normalize_dialog_mcp({
        "signature": signature,
        "request": request,
        "calls": calls if calls is not None else [],
        "results": results,
        "chain": keep_chain,
    })
    return dialog["mcp"]


def _normalize_dialog_rag(raw: Any) -> Dict[str, Any]:
    """Фрагменты баз знаний диалога: {"signature", "request", "query", "bases", "hits", "notes"}.

    Ровно то, что уже найдено поиском (см. app/ai/rag_search.py): подпись
    (включённые базы + отпечаток их индексов + запрос задачи) и сами фрагменты с
    адресами файлов и разделов. Держится в диалоге, чтобы шаг плана и проверка
    результата — а это ОТДЕЛЬНЫЕ HTTP-запросы — видели те же документы, а не
    искали заново и не отвечали «в документах этого нет».

    Запись БЕЗ подписи не хранится: по ней не понять, к какому запросу относятся
    фрагменты, и она «переехала» бы на чужой запрос.
    """
    if not isinstance(raw, dict):
        return {}
    clean = rag_search.normalize(raw)
    return clean if clean.get("signature") else {}


def dialog_rag(dialog: Dict[str, Any]) -> Dict[str, Any]:
    """Фрагменты баз знаний диалога (нормализует поле на месте, если оно битое).

    Наружу отдаётся либо ПУСТАЯ запись, либо запись с подписью: по записи без
    подписи нельзя понять, к какому запросу относятся фрагменты, и она «переехала»
    бы на чужой запрос. Поэтому проверка стоит и на чтении, а не только при
    загрузке файла (диалог мог прийти из памяти процесса, а не с диска).
    """
    current = dialog.get("rag")
    if not isinstance(current, dict) or not str(current.get("signature") or "").strip():
        dialog["rag"] = _normalize_dialog_rag(current)
    return dialog["rag"]


def set_dialog_rag(dialog: Dict[str, Any], signature: str, request: str,
                   result: Any) -> Dict[str, Any]:
    """Запоминает фрагменты RAG текущего запроса задачи (подпись + что нашлось).

    `result` — результат `rag_search.search` (или уже нормализованная запись):
    запрос, состояние каждой базы и найденные фрагменты. Пустой результат —
    законный случай («искали, ничего не нашлось»): он тоже хранится под подписью,
    иначе на каждом шаге поиск повторялся бы впустую.
    """
    data = rag_search.normalize(result)
    dialog["rag"] = _normalize_dialog_rag({
        "signature": signature,
        "request": request,
        "query": data.get("query") or request,
        "bases": data.get("bases") or [],
        "hits": data.get("hits") or [],
        "notes": data.get("notes") or [],
    })
    return dialog["rag"]


def _clean_sources(raw: Any) -> List[Dict[str, Any]]:
    """Источники под ответом агента (список «что подобрано из баз знаний»).

    Хранятся в журнале чата вместе с ответом: так они видны и после переключения
    задачи или перезагрузки страницы. Отрывок фрагмента обрезается — журнал не
    должен расти вместе с документами пользователя.

    Хранится и НОМЕР ЧАНКА, и разбор оценки (вектор + текст): без них после
    перезагрузки список терял бы самое полезное — по какому именно фрагменту базы
    получен ответ и почему он оказался вверху (эту потерю нашла живая проверка на
    настоящей базе: в событии номер был, а в журнале — уже нет).
    """
    out: List[Dict[str, Any]] = []
    for item in (raw if isinstance(raw, list) else [])[:rag_search.MAX_HITS]:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or "").strip()[:160]
        if not source:
            continue
        out.append({
            "base": str(item.get("base") or "").strip()[:120],
            "source": source,
            "section": str(item.get("section") or "").strip()[:200],
            "number": _positive_int(item.get("number")),
            "score": _round3(item.get("score")),
            "vector_score": _round3(item.get("vector_score")),
            "lexical": _round3(item.get("lexical")),
            "chars": _positive_int(item.get("chars")),
            "snippet": str(item.get("snippet") or "").strip()[:400],
        })
    return out


def _round3(value: Any) -> float:
    """Число оценки с тремя знаками (битое — 0.0)."""
    try:
        return round(float(value or 0.0), 3)
    except (TypeError, ValueError):
        return 0.0


def _positive_int(value: Any) -> int:
    """Неотрицательное целое из значения любого вида (битое — 0)."""
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _clean_plan_signature(raw: Any) -> Dict[str, str]:
    """Подпись построенного плана: {"request", "rules", "basis"} (битое — пусто).

    `basis` — ОСНОВА ДАННЫХ плана (какие внешние данные получены по запросу, см.
    _data_basis в chat.py): смена основы означает другой план, а ключ обязан
    переживать перезагрузку — иначе после каждого запуска приложения план
    пересобирался бы заново (лишний вызов планировщика и гейт).
    """
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, str] = {}
    for key in ("request", "rules", "basis"):
        value = str(raw.get(key) or "").strip()
        if value:
            out[key] = value[:400]
        elif key == "basis":
            # Пустая основа — законное значение («данных MCP нет»): подпись без
            # ключа не совпала бы с подписью с пустым ключом, и план пересобирался
            # бы на каждой перезагрузке.
            out[key] = ""
    return out


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
        # Расписание ПЕРИОДИЧЕСКОЙ задачи (см. app/ai/periodic.py): период,
        # следующий срок, счётчик повторов. Пусто — задача обычная, одноразовая.
        "periodic": periodic_store.normalize(raw.get("periodic")),
    }


# ---------------------------------------------------------------------------
# Периодические задачи (расписание рядом с сессией, см. app/ai/periodic.py)
# ---------------------------------------------------------------------------
def periodic_meta(session: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Расписание сессии ({} — задача не периодическая).

    Поле нормализуется на месте: файл мог быть записан прежней версией или
    поправлен руками — расписание всё равно читается безопасно.
    """
    if not session:
        return {}
    meta = periodic_store.normalize(session.get("periodic"))
    session["periodic"] = meta
    return meta


def set_periodic(session: Dict[str, Any], meta: Any) -> Dict[str, Any]:
    """Записывает расписание в сессию ({} — задача снова обычная)."""
    normalized = periodic_store.normalize(meta)
    session["periodic"] = normalized
    return normalized


def make_periodic(session: Dict[str, Any], interval: Any = None,
                  request: str = "", enabled: bool = True) -> Dict[str, Any]:
    """Заводит (или заменяет) расписание ГОТОВОЙ сессии.

    Нужен, когда периодической становится уже существующая задача (сейчас такой
    путь есть только у проверок: интерфейс заводит расписание вместе с задачей —
    см. create_session). Период не задан — берётся сутки по умолчанию.
    """
    meta = periodic_store.make(
        periodic_store.DEFAULT_INTERVAL if interval is None else interval,
        request=request, enabled=enabled)
    session["periodic"] = meta
    return meta


def session_periodic_brief(session: Dict[str, Any], running: bool = False,
                           moment: Optional[datetime] = None
                           ) -> Optional[Dict[str, Any]]:
    """Расписание сессии для снимка фронтенда (None — задача не периодическая).

    Вместе с расписанием отдаётся ПРИЧИНА, по которой повторы сейчас не идут
    (`hold`): «paused» — задача на паузе, «cancelled» — задача отменена (отмена и
    есть остановка периодической задачи). Интерфейс показывает это в списке задач
    и в модалке расписания.
    """
    meta = periodic_meta(session)
    if not meta:
        return None
    dialog = session.get("dialog") or {}
    state = dialog_state(session)
    hold = "paused" if state.paused else ("cancelled" if state.stage == "cancelled" else "")
    return periodic_store.brief(
        meta, running=running, hold=hold,
        log_len=len(dialog.get("log") or []), moment=moment)


def periodic_sessions(workspace: Dict[str, Any],
                      profile_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Все периодические задачи профиля: пары (задача, сессия).

    Периодическая задача — всегда ЗАДАЧА-ДИАЛОГ (сессия) конкретного проекта:
    свой запрос, свой автомат, своё расписание. Выключенный повтор (enabled:
    False) тоже здесь — задача остаётся периодической, просто не повторяется.
    """
    found: List[Dict[str, Any]] = []
    for task in profile_tasks(workspace, profile_id):
        for session in task.get("sessions", []):
            if periodic_meta(session):
                found.append((task, session))
    return found


def due_periodic_sessions(workspace: Dict[str, Any],
                          moment: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """Периодические задачи, у которых наступил срок повтора (по всему файлу).

    Планировщик работает для ВСЕХ профилей: задача профиля, который сейчас не
    открыт, тоже должна повторяться (профиль подставляется планировщиком, см.
    PROFILE_OVERRIDE в chat.py).
    """
    moment = moment or periodic_store.now()
    found: List[Dict[str, Any]] = []
    for task in workspace.get("tasks", []):
        for session in task.get("sessions", []):
            meta = periodic_meta(session)
            if meta and periodic_store.is_due(meta, moment):
                found.append((task, session))
    found.sort(key=lambda pair: str(periodic_meta(pair[1]).get("next_run") or ""))
    return found


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


def _normalize_invariants(raw: Any) -> List[Dict[str, Any]]:
    """Приводит список инвариантов к записям {id, text, created}.

    Инварианты проекта лежат в задаче ("invariants"), инварианты задачи-диалога —
    в её диалоге. Принимаются и голые строки (ручная правка файла), пустые
    отбрасываются, слишком длинные обрезаются, число записей ограничено
    MAX_INVARIANTS (старые вытесняются).
    """
    entries: List[Dict[str, Any]] = []
    for item in (raw if isinstance(raw, list) else []):
        if isinstance(item, dict):
            text = str(item.get("text") or "").strip()
            entry_id = str(item.get("id") or "").strip()
            created = str(item.get("created") or "")
        else:
            text = str(item or "").strip()
            entry_id, created = "", ""
        if not text:
            continue
        entries.append({
            "id": entry_id or new_id("i"),
            "text": text[:INVARIANT_ENTRY_LIMIT],
            "created": created or _now(),
        })
    return entries[-MAX_INVARIANTS:]


def _normalize_mcp(raw: Any) -> Dict[str, Any]:
    """Приводит настройку MCP проекта к виду {"enabled": [...]}.

    Включёнными считаются только известные реестру id (app/ai/mcp.py): запись,
    оставшаяся в файле от прежней версии или от другого набора серверов, ничего
    не ломает — она просто не попадает в список и в запросы агента.
    """
    if isinstance(raw, list):  # запасной формат: голый список включённых серверов
        raw = {"enabled": raw}
    data = raw if isinstance(raw, dict) else {}
    enabled: List[str] = []
    for item in (data.get("enabled") if isinstance(data.get("enabled"), list) else []):
        key = str(item or "").strip().lower()
        if key and key in mcp_store.SERVER_IDS and key not in enabled:
            enabled.append(key)
    if not enabled and data.get("enabled"):
        # Ни одного известного сервера не осталось (реестр изменился) — набор пуст.
        logger.info("MCP: в настройках проекта нет известных серверов")
    return {"enabled": enabled[-MAX_MCP_SERVERS:]}


def mcp_settings(task: Dict[str, Any]) -> Dict[str, Any]:
    """Настройка MCP проекта (нормализует поле на месте, если его нет).

    Как invariants(): задача, созданная в памяти и ещё не записанная в файл, не
    имеет поля — оно появляется при первом обращении, а не падает с KeyError.
    """
    if not isinstance(task.get(MCP_FIELD), dict):
        task[MCP_FIELD] = _normalize_mcp(task.get(MCP_FIELD))
    return task[MCP_FIELD]


def mcp_enabled(task: Optional[Dict[str, Any]]) -> List[str]:
    """Включённые серверы MCP проекта (пустой список — MCP у проекта выключен)."""
    if not task:
        return []
    return list(mcp_settings(task)["enabled"])


def set_mcp_enabled(task: Dict[str, Any], enabled: Any) -> List[str]:
    """Записывает набор включённых серверов MCP и возвращает его.

    Пишет только известные реестру id: включить сервер, которого нет, нельзя —
    иначе агент «включённым» инструментом пользоваться не сможет, а интерфейс
    покажет включённым то, чего в диалоге нет.
    """
    task[MCP_FIELD] = _normalize_mcp({"enabled": enabled})
    return list(task[MCP_FIELD]["enabled"])


def _normalize_rag(raw: Any) -> Dict[str, Any]:
    """Приводит настройку баз знаний проекта к рабочему виду.

    Формат: {"enabled": [<id базы>], "strategy": "structure", "chunk_size": 1000,
    "overlap": 150}. Стратегия и размеры проверяются по общим правилам разбиения
    (app/ai/rag_chunking.py), поэтому «настройка из файла» не может дать чанк
    нулевой длины или перекрытие больше размера.

    Существование базы здесь НЕ проверяется: список баз читается с диска, а
    нормализация вызывается на каждой записи workspace. Отсеивает чужие и
    удалённые базы маршрут — там же, где известен профиль (rag.filter_enabled).
    """
    if isinstance(raw, list):        # запасной формат: голый список включённых баз
        raw = {"enabled": raw}
    data = raw if isinstance(raw, dict) else {}
    enabled: List[str] = []
    for item in (data.get("enabled") if isinstance(data.get("enabled"), list) else []):
        key = str(item or "").strip().lower()
        if rag_store.valid_id(key) and key not in enabled:
            enabled.append(key)
    settings = rag_chunking.chunk_settings(
        data.get("strategy"), data.get("chunk_size"), data.get("overlap"))
    return {
        "enabled": enabled[-MAX_RAG_BASES:],
        "strategy": settings["strategy"],
        "chunk_size": settings["chunk_size"],
        "overlap": settings["overlap"],
    }


def rag_settings(task: Dict[str, Any]) -> Dict[str, Any]:
    """Настройка баз знаний проекта (нормализует поле на месте, если его нет).

    Как mcp_settings(): задача, созданная в памяти и ещё не записанная в файл,
    не имеет поля — оно появляется при первом обращении, а не падает с KeyError.
    """
    if not isinstance(task.get(RAG_FIELD), dict):
        task[RAG_FIELD] = _normalize_rag(task.get(RAG_FIELD))
    return task[RAG_FIELD]


def rag_enabled(task: Optional[Dict[str, Any]]) -> List[str]:
    """Включённые базы знаний проекта (пустой список — RAG у проекта выключен)."""
    if not task:
        return []
    return list(rag_settings(task)["enabled"])


def set_rag_enabled(task: Dict[str, Any], enabled: Any) -> List[str]:
    """Записывает набор включённых баз знаний и возвращает его.

    Идентификаторы уже отфильтрованы маршрутом по фактическому списку баз
    профиля (rag.filter_enabled) — здесь остаётся нормализация и хранение.
    """
    settings = rag_settings(task)
    task[RAG_FIELD] = _normalize_rag({
        "enabled": enabled,
        "strategy": settings.get("strategy"),
        "chunk_size": settings.get("chunk_size"),
        "overlap": settings.get("overlap"),
    })
    return list(task[RAG_FIELD]["enabled"])


def set_rag_chunking(task: Dict[str, Any], strategy: Any = None,
                     chunk_size: Any = None, overlap: Any = None) -> Dict[str, Any]:
    """Запоминает параметры разбиения, выбранные в диалоге «База знаний».

    Настройка ЗАПОМИНАЕТСЯ на проекте: пользователь задаёт размер чанка один раз,
    а не заново при каждой загрузке. Незаданные поля (None) остаются прежними.
    """
    settings = rag_settings(task)
    next_settings = rag_chunking.chunk_settings(
        settings.get("strategy") if strategy is None else strategy,
        settings.get("chunk_size") if chunk_size is None else chunk_size,
        settings.get("overlap") if overlap is None else overlap)
    task[RAG_FIELD] = _normalize_rag({
        "enabled": settings.get("enabled"),
        "strategy": next_settings["strategy"],
        "chunk_size": next_settings["chunk_size"],
        "overlap": next_settings["overlap"],
    })
    return dict(task[RAG_FIELD])


def exceptions(dialog: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Решения пользователя по противоречиям правил (нормализует поле на месте).

    Контейнер — диалог задачи. Запись: {"key", "project_id", "task_id", "winner"}.
    """
    if not isinstance(dialog.get("exceptions"), list):
        dialog["exceptions"] = [
            dict(item) for item in invariants_store.clean_exceptions(dialog.get("exceptions"))]
    return dialog["exceptions"]


def add_exception(dialog: Dict[str, Any], key: str, winner: str) -> Dict[str, Any]:
    """Фиксирует решение по противоречию: чьё правило главнее в этой задаче."""
    entry = invariants_store.clean_exceptions([{"key": key, "winner": winner}])
    if not entry:
        return {}
    item = entry[0]
    items = [existing for existing in exceptions(dialog)
             if existing.get("key") != item["key"]]
    items.append(item)
    dialog["exceptions"] = items
    return item


def _clean_conflicts(raw: Any) -> List[Dict[str, Any]]:
    """Проверки пар «инвариант проекта × инвариант задачи» (dialog["conflicts"])."""
    records = invariants_store.clean_records(raw)
    return records[-_MAX_CONFLICTS:]


def _clean_unchecked(raw: Any) -> List[str]:
    """Пары, которые проверить не удалось: их перепроверяют при следующем шансе."""
    if not isinstance(raw, list):
        return []
    keys: List[str] = []
    for item in raw:
        key = str(item or "").strip()
        if key and key not in keys:
            keys.append(key)
    return keys[-_MAX_UNCHECKED:]


def invariants(container: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Записи инвариантов контейнера (задача-проект или диалог задачи).

    Нормализует поле на месте, если его нет или оно повреждено, — как memory().
    Контейнером может быть как задача workspace (инварианты ПРОЕКТА), так и её
    диалог (инварианты ЗАДАЧИ-диалога).
    """
    if not isinstance(container.get("invariants"), list):
        container["invariants"] = _normalize_invariants(container.get("invariants"))
    return container["invariants"]


def invariants_texts(container: Dict[str, Any]) -> List[str]:
    """Тексты инвариантов контейнера — в таком виде их получает агент."""
    return [entry["text"] for entry in invariants(container)]


def add_invariant(container: Dict[str, Any], text: str) -> Dict[str, Any]:
    """Добавляет инвариант (одно поле — один инвариант) и возвращает запись."""
    entries = invariants(container)
    entry = {"id": new_id("i"), "text": str(text or "").strip()[:INVARIANT_ENTRY_LIMIT],
             "created": _now()}
    entries.append(entry)
    if len(entries) > MAX_INVARIANTS:
        del entries[:len(entries) - MAX_INVARIANTS]
    return entry


def delete_invariant(container: Dict[str, Any], entry_id: str) -> bool:
    """Удаляет инвариант по id (False — такого инварианта нет)."""
    entries = invariants(container)
    for index, entry in enumerate(entries):
        if entry["id"] == entry_id:
            del entries[index]
            return True
    return False


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


def _normalize_stats_by_profile(raw: Any) -> Dict[str, Dict[str, Dict[str, int]]]:
    """Приводит статистику экспертных режимов по профилям к безопасному виду.

    Формат: {"<id профиля>": {"direct": {"correct": 0, "incorrect": 0}, …}}.
    Неизвестные режимы и отрицательные значения отбрасываются.
    """
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Dict[str, Dict[str, int]]] = {}
    for profile_id, modes in list(raw.items())[:_MAX_TASKS]:
        key = str(profile_id or "").strip()[:_MAX_NAME]
        if not key or not isinstance(modes, dict):
            continue
        clean: Dict[str, Dict[str, int]] = {}
        for mode in EXPERT_MODES:
            counters = modes.get(mode)
            if not isinstance(counters, dict):
                continue
            try:
                correct = max(0, int(counters.get("correct") or 0))
                incorrect = max(0, int(counters.get("incorrect") or 0))
            except (TypeError, ValueError):
                continue
            clean[mode] = {"correct": correct, "incorrect": incorrect}
        if clean:
            out[key] = clean
    return out


def stats_container(workspace: Dict[str, Any],
                    profile_id: Optional[str] = None) -> Dict[str, Dict[str, int]]:
    """Счётчики экспертной статистики ПРОФИЛЯ (создаются при первом обращении).

    Профили изолированы: у каждого своя статистика ответов, как и свои задачи.
    Профиль не передан — статистика не ведётся (возвращаем пустой словарь без
    записи в workspace).
    """
    key = str(profile_id or "").strip()
    if not key:
        return {}
    by_profile = workspace.setdefault("stats_by_profile", {})
    if not isinstance(by_profile, dict):
        by_profile = {}
        workspace["stats_by_profile"] = by_profile
    modes = by_profile.setdefault(key, {})
    if not isinstance(modes, dict):
        modes = {}
        by_profile[key] = modes
    return modes


def add_expert_result(workspace: Dict[str, Any], mode: str, correct: bool,
                      profile_id: Optional[str] = None) -> None:
    """Учитывает вердикт экспертного режима: «Верно» или «Неверно».

    Счётчик режима, которого нет в списке (новая настройка интерфейса),
    инициализируется на месте — статистика не должна терять ответы.
    """
    modes = stats_container(workspace, profile_id)
    if not modes:
        return
    name = str(mode or "").strip()
    if not name:
        return
    counters = modes.setdefault(name, {"correct": 0, "incorrect": 0})
    counters["correct" if correct else "incorrect"] += 1


def expert_stats(workspace: Dict[str, Any],
                 profile_id: Optional[str] = None) -> Dict[str, Dict[str, int]]:
    """Снимок статистики экспертных режимов профиля для интерфейса."""
    modes = stats_container(workspace, profile_id) if str(profile_id or "").strip() else {}
    out: Dict[str, Dict[str, int]] = {}
    for mode in EXPERT_MODES:
        counters = modes.get(mode) if isinstance(modes, dict) else None
        out[mode] = {
            "correct": int((counters or {}).get("correct") or 0),
            "incorrect": int((counters or {}).get("incorrect") or 0),
        }
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
        # Инварианты ПРОЕКТА (в терминах интерфейса «проект» — это задача):
        # правила, действующие во всех задачах-диалогах этого проекта.
        "invariants": _normalize_invariants(raw.get("invariants")),
        # MCP ПРОЕКТА: какие внешние инструменты (погода, курсы валют, …) агент
        # может вызывать в задачах этого проекта (см. app/ai/mcp.py).
        "mcp": _normalize_mcp(raw.get("mcp")),
        # RAG ПРОЕКТА: какие базы знаний включены у проекта и с какими
        # параметрами разбиения на чанки (см. app/ai/rag.py). Без этой строки
        # ключ молча терялся бы при первой же записи workspace — как это было бы
        # с "mcp".
        "rag": _normalize_rag(raw.get("rag")),
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
        # Статистика экспертных режимов по профилям (страница «Статистика ответов»):
        # живёт на сервере, поэтому не обнуляется при переключении режима.
        "stats_by_profile": _normalize_stats_by_profile(raw.get("stats_by_profile")),
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

    Пишем КОМПАКТНО (без indent): файл машинный, читает его приложение, а
    отступы в 2 пробела давали +25 % размера и впятеро больше времени на
    сериализацию — а она идёт в event loop на каждом шаге задачи.
    """
    return json.dumps(normalize_workspace(workspace), ensure_ascii=False,
                      separators=(",", ":"))


# При каком уменьшении записи сохраняется копия прежнего файла. Файл workspace —
# единственное место, где живёт вся история задач и диалогов: он не в git и не в
# резервных копиях, поэтому РЕЗКОЕ уменьшение (больше чем втрое) считается
# подозрительным и оставляет .bak рядом. Обычные правки (удаление одной задачи,
# добавление реплики) сюда не попадают.
BACKUP_RATIO = 3.0
BACKUP_MIN_BYTES = 4096


def _keep_backup(file_path: str, payload: str) -> str:
    """Сохраняет копию прежнего файла, если новая запись РЕЗКО меньше.

    Возвращает путь к копии ("" — копия не нужна). Зачем: история задач и
    диалогов пользователя не восстановима ничем (data/ в .gitignore, файл один),
    а путей, которыми запись может «похудеть» сразу в разы, несколько — включая
    разбор битого файла как пустого. Одна копия прежнего состояния делает такую
    потерю видимой и обратимой, а места занимает столько же, сколько сам файл.
    """
    try:
        if not os.path.isfile(file_path):
            return ""
        previous = os.path.getsize(file_path)
        if previous < BACKUP_MIN_BYTES or len(payload) * BACKUP_RATIO > previous:
            return ""
        backup = file_path + ".bak"
        shutil.copy2(file_path, backup)
        logger.warning(
            "Workspace AI-агента: запись уменьшилась в %.1f раз (%d → %d байт) — "
            "прежнее состояние сохранено в %s",
            previous / float(max(1, len(payload))), previous, len(payload), backup)
        return backup
    except OSError as exc:              # копия не должна мешать самой записи
        logger.warning("Workspace AI-агента: копию сделать не удалось: %s", exc)
        return ""


def write_payload(payload: str, path: Optional[str] = None) -> None:
    """Пишет готовый JSON в файл атомарно (временный файл + os.replace).

    Перед записью проверяется, не стала ли запись РЕЗКО меньше прежнего файла: в
    этом случае рядом остаётся копия `.bak` (см. _keep_backup).
    """
    file_path = path or config.AGENT_WORKSPACE_FILE
    directory = os.path.dirname(file_path) or "."
    os.makedirs(directory, exist_ok=True)
    _keep_backup(file_path, payload)
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


def create_session(task: Dict[str, Any], title: str = "",
                   periodic: Any = None) -> Dict[str, Any]:
    """Создаёт в задаче новую сессию-диалог и делает её текущей.

    `periodic` — период ПЕРИОДИЧЕСКОЙ задачи в секундах (кнопка «Новая
    периодическая задача»): None — обычная задача, которая выполняется один раз.
    Для периодической сразу заводится расписание (по умолчанию — сутки) и первый
    срок повтора; сам запрос задачи появится с первым сообщением пользователя.
    """
    session_id = new_id("s")
    session = {
        "id": session_id,
        "title": str(title).strip()[:_MAX_TITLE],
        "created": _now(),
        # Новый диалог — новая задача для автомата: состояние создаётся сразу
        # (этап planning), а его task_id — id этой сессии.
        "dialog": empty_dialog(session_id),
    }
    if periodic is not None:
        session["periodic"] = periodic_store.make(periodic)
    else:
        session["periodic"] = {}
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
    # Статистика экспертных режимов — тоже данные профиля.
    workspace.get("stats_by_profile", {}).pop(key, None)
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
    owners |= set(workspace.get("stats_by_profile") or {})
    owners |= set(workspace.get("active_tasks") or {})
    owners.discard("")
    return sorted(owners)


def _task_brief(task: Dict[str, Any]) -> Dict[str, Any]:
    """Краткая запись задачи для снимка фронтенда."""
    return {"id": task["id"], "name": task["name"]}


def snapshot(workspace: Dict[str, Any], profile_id: Optional[str] = None,
             running: Optional[Any] = None,
             moment: Optional[datetime] = None) -> Dict[str, Any]:
    """Снимок для фронтенда: задачи ПРОФИЛЯ, его текущая задача и её диалоги.

    Профили изолированы: в снимке только задачи переданного профиля (и его
    текущая задача из active_tasks), поэтому чужой профиль их не видит. В каждой
    задаче есть поле "profile" — id профиля-владельца.

    У периодической задачи-диалога вместе с заголовком отдаётся её расписание
    ("periodic": период, срок следующего повтора, счётчик, ошибка) — по нему
    интерфейс помечает такую задачу в списке. `running` — id задач, повтор
    которых выполняется ПРЯМО СЕЙЧАС (ведёт планировщик, см. chat.py).
    """
    running_ids = running or set()
    tasks = profile_tasks(workspace, profile_id)
    task = active_task(workspace, profile_id)
    sessions = []
    for session in (task.get("sessions", []) if task else []):
        brief = {"id": session["id"], "title": session_title(session)}
        periodic = session_periodic_brief(
            session, running=str(session["id"]) in running_ids, moment=moment)
        if periodic:
            brief["periodic"] = periodic
        sessions.append(brief)
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
