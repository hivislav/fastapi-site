"""Маршруты чата — приём сообщений и возврат ответа бота.

Классический POST /api/chat (обычный/экспертный/температура/тест моделей) и
потоковый POST /api/agent/chat для режима «AI-агент»: сервер отдаёт NDJSON,
каждая строка — событие агента (state/debug/bot/error), которое фронтенд
выводит в чат отдельным сообщением по мере появления.

Диалог режима «AI-агент» живёт в «рабочем пространстве» (workspace): задачи →
сессии (диалоги) → состояние диалога (память сообщений, замеры токенов, резюме
стратегии «summary» с границей covered, блок фактов «sticky facts», ветви плана
и активная ветка «branching», состояние конечного автомата задачи). Диалог
нельзя начать, пока не создана задача; у каждой задачи своя история диалогов, а
внутри задачи можно переключаться между сессиями — у каждой своё независимое
состояние. Всё это переживает перезапуск приложения: workspace загружается из
JSON-файла при старте процесса и сохраняется после каждого изменения
(см. app/ai/workspace.py).

КОНЕЧНЫЙ АВТОМАТ ЗАДАЧИ (Task State Machine, см. app/ai/task_state.py) ведёт
именно этот модуль: каждый запрос пользователя проходит этапы planning →
execution → validation → done (расширения: awaiting_user, failed, cancelled).
Состояние лежит в dialog["state"] текущей сессии, переходы делает ТОЛЬКО
chat.py (агент получает состояние готовым и показывает его модели системным
блоком), а интерфейс рисует его полосой этапов над окном чата и меняет кнопками
«Пауза»/«Продолжить», «Подтвердить план» и правкой плана.
"""

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from app.ai import client as llm_client
from app.ai import service
from app.ai import invariants as invariants_store
from app.ai import profiles as profile_store
from app.ai import task_state
from app.ai import workspace as workspace_store
from app.ai.agent import (
    Agent, AgentConfig, DEFAULT_SUMMARY_SIZE, DEFAULT_WINDOW_SIZE, merge_usage,
)
from app.schemas import (
    ChatMessage, InvariantCreate, InvariantDelete, InvariantPick, InvariantResolve,
    MemoryEntryCreate, NameUpdate, PlanUpdate, ProfileCreate, ProfileFields,
    TaskCreate,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# Рабочее пространство режима «AI-агент»: задачи с их сессиями-диалогами.
# При старте процесса восстанавливается из JSON-файла (data/agent_workspace.json),
# поэтому после перезапуска задачи, диалоги, панель токенов и ветви плана
# остаются на месте. Изменения сериализуются блокировкой: агент работает
# асинхронно, а workspace у нас один на приложение.
_workspace: Dict[str, Any] = workspace_store.load_workspace()
# Блокировки ПО ЗАДАЧАМ (сессиям): шаг одной задачи держит свою блокировку, и
# задачи друг друга НЕ ждут — параллельные задачи действительно работают
# параллельно (вызовы LLM уходят в отдельные потоки). Одна задача по-прежнему
# выполняет шаги по одному: шаг — это пара «запрос + ответ» в её диалоге.
_session_locks: Dict[str, asyncio.Lock] = {}
# Общая блокировка — только для редких операций над ВСЕМ workspace (удаление
# задач и проектов, слои памяти, профили). Вызовы LLM её не держат, поэтому
# такие операции не ждут ответа модели.
_workspace_lock = asyncio.Lock()


def _session_lock(session_id: str) -> asyncio.Lock:
    """Блокировка конкретной задачи (создаётся по требованию)."""
    lock = _session_locks.get(str(session_id))
    if lock is None:
        lock = asyncio.Lock()
        _session_locks[str(session_id)] = lock
    return lock

# Намерения остановки, полученные ПОКА идёт обработка шага: «Пауза» и «Отменить»
# не ждут блокировку (её держит поток ответа), а мгновенно отвечают интерфейсу и
# кладут намерение сюда; поток применяет его, как только закончит текущий вызов
# LLM. Ключ — id сессии, значение — "pause" | "cancel".
# Запись/чтение — простое присваивание без await (как в GET /api/agent/memory),
# поэтому гонки с потоком нет.
_pending_stops: Dict[str, str] = {}
PENDING_PAUSE = "pause"
PENDING_CANCEL = "cancel"
# id задач, шаг которых выполняется ПРЯМО СЕЙЧАС (их может быть НЕСКОЛЬКО:
# параллельные задачи работают одновременно). Нужен «Паузе»/«Отмене»: отложить
# команду имеет смысл только для выполняющейся задачи — у остальных диалог никто
# не пишет, и переход применяется сразу.
_running_sessions: set = set()
# Ключ снимка состояния: интерфейс показывает по нему «остановлю после текущего
# шага» (запрос принят, но применён будет по завершении шага).
PENDING_KEY = "pending"

# Профили пользователя (data/profiles.json): сведения о пользователе, которые
# уходят в системный промпт сессии агента. Профилей может быть несколько —
# пользователь создаёт, удаляет и переключает их в меню профиля.
# Если профиля нет ни одного (файла ещё нет), он создаётся здесь с
# автоматическим идентификатором «user_<цифры>» (напр. user_13213123) — так
# профиль появляется к моменту первого диалога/задачи.
_profiles: Dict[str, Any] = profile_store.load_profiles()
if not _profiles.get("profiles"):
    # Профиля нет — заводим его сразу (идентификатор «user_<цифры>», название
    # «user<цифры>») и записываем в файл: так профиль существует уже к первому
    # диалогу/задаче.
    profile_store.ensure_profile(_profiles)
    try:
        profile_store.save_profiles(_profiles)
    except OSError:  # сбой записи не мешает работе в памяти
        logger.warning("Не удалось записать профили пользователя", exc_info=True)

# После появления профилей workspace стал профильным: задачи (и их диалоги)
# принадлежат конкретному профилю. Задачи из файла прежней версии (без владельца)
# отдаём текущему профилю — иначе прежняя переписка осталась бы ничей.
_workspace_changed = bool(workspace_store.adopt_orphan_tasks(
    _workspace,
    _profiles.get("active"),
    profile_store.profile_label(profile_store.active_profile(_profiles)),
))
# То же для долговременной памяти: прежняя общая база знаний лежала в корне и
# была видна всем профилям, а удалить её было нельзя (в панели памяти видны
# только записи профиля). Передаём её текущему профилю — теперь её можно
# удалить, и она уйдёт вместе с профилем.
_workspace_changed |= workspace_store.migrate_legacy_long_term(
    _workspace, _profiles.get("active"))
if _workspace_changed:
    try:
        workspace_store.save_workspace(_workspace)
    except OSError:
        logger.warning("Не удалось записать workspace AI-агента", exc_info=True)

# В файле могли остаться данные профилей, которых уже нет (прежние версии не
# удаляли их вместе с профилем). Сами по себе они ничего не ломают — они никому не
# видны, — но и не удаляются через интерфейс: подсказываем в логе, чем убрать.
_orphan_profiles = [pid for pid in workspace_store.profile_ids_with_data(_workspace)
                    if profile_store.find_profile(_profiles, pid) is None]
if _orphan_profiles:
    logger.warning(
        "Workspace AI-агента: в файле остались данные профилей, которых больше нет (%s) — "
        "они никому не видны и в модель не попадают. Убрать: "
        "./venv/bin/python tools/cleanup_workspace.py --apply",
        ", ".join(_orphan_profiles),
    )


def _current_profile() -> Optional[Dict[str, Any]]:
    """Текущий профиль пользователя (None — профилей нет: не должно случаться,
    профиль заводится при старте)."""
    return profile_store.active_profile(_profiles)


def _current_profile_id() -> Optional[str]:
    """id текущего профиля — им ограничены задачи, диалоги и память."""
    profile = _current_profile()
    return profile["id"] if profile else None


def _current_task() -> Optional[Dict[str, Any]]:
    """Текущая задача ТЕКУЩЕГО ПРОФИЛЯ (None — у профиля ещё нет задач)."""
    return workspace_store.active_task(_workspace, _current_profile_id())


def _current_session() -> Optional[Dict[str, Any]]:
    """Текущая сессия-диалог текущей задачи (None — диалогов ещё нет)."""
    return workspace_store.active_session(
        _workspace, profile_id=_current_profile_id())


def _own_task(task: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Задача, если она принадлежит ТЕКУЩЕМУ профилю, иначе None.

    Профили изолированы: чужая задача для запроса не существует (для фронта это
    «задача не найдена»).
    """
    if task is None:
        return None
    if not workspace_store.task_belongs(task, _current_profile_id()):
        return None
    return task


def _find_session_anywhere(session_id: str) -> Tuple[Optional[Dict[str, Any]],
                                                     Optional[Dict[str, Any]]]:
    """Ищет сессию по id СРЕДИ ЗАДАЧ ТЕКУЩЕГО ПРОФИЛЯ: (задача, сессия).

    Диалог чужого профиля найти нельзя — вернётся (None, None), фронт покажет
    «диалог не найден».
    """
    for task in workspace_store.profile_tasks(_workspace, _current_profile_id()):
        session = workspace_store.find_session(task, session_id)
        if session is not None:
            return task, session
    return None, None


async def _persist() -> None:
    """Сохраняет workspace в файл (сбой записи не рвёт диалог).

    Снимок (json.dumps) снимается СИНХРОННО, до первого await: пока ответ
    агента пишется в отдельном потоке, workspace меняют и другие маршруты
    (переключение диалога, «Пауза», запись журнала чата) — иначе сериализация в
    потоке могла бы поймать изменение структуры «на лету». В отдельный поток
    уходит только запись файла.
    """
    try:
        payload = workspace_store.workspace_payload(_workspace)
        await asyncio.to_thread(workspace_store.write_payload, payload)
    except Exception:  # noqa: BLE001 — сбой записи логируем, работу продолжаем
        logger.warning("Не удалось сохранить workspace AI-агента", exc_info=True)


async def _persist_profiles() -> None:
    """Сохраняет профили пользователя в файл (сбой записи не рвёт диалог)."""
    try:
        await asyncio.to_thread(profile_store.save_profiles, _profiles)
    except Exception:  # noqa: BLE001
        logger.warning("Не удалось сохранить профили пользователя", exc_info=True)


def _profile_block() -> str:
    """Системный блок профиля пользователя для текущего запроса к агенту.

    Собирается на каждый запрос: пользователь мог переключить профиль или
    изменить поля, а агент создаётся заново под каждый запрос.
    """
    return profile_store.profile_block(profile_store.active_profile(_profiles))


def _snapshot() -> dict:
    """Снимок workspace для фронтенда: задачи ПРОФИЛЯ, его текущая задача,
    диалоги и сам профиль."""
    snapshot = workspace_store.snapshot(_workspace, _current_profile_id())
    # Профиль — глобальная сущность (не на задачу), но фронту удобно получать
    # его вместе со снимком workspace: иконка профиля и его поля обновляются
    # одним ответом на любую операцию с задачами/диалогами.
    snapshot["profile"] = profile_store.snapshot(_profiles)
    return snapshot


def _usage_matches_history(dialog: Dict[str, Any]) -> None:
    """Синхронизирует число замеров токенов с числом запросов в диалоге.

    Замеры идут по одному на запрос пользователя, поэтому лишние (более
    старые, чем сам диалог) отбрасываем — панель «Токены диалога» показывает
    ТЕКУЩИЙ диалог. Считаем запросы и в корневом диалоге, и в ветках плана
    (стратегия «branching»): диалог в ветке — тоже запросы.
    """
    requests = sum(1 for m in dialog.get("messages", []) if m.get("role") == "user")
    for branch in (dialog.get("branches") or {}).values():
        messages = branch.get("messages") if isinstance(branch, dict) else None
        requests += sum(1 for m in (messages or []) if m.get("role") == "user")
    usage = dialog.setdefault("usage", [])
    # Замеры служебных запросов планирования (kind="plan") реплик в диалоге не
    # имеют — их синхронизация не касается, иначе расход на план пропадал бы из
    # панели токенов после перезагрузки страницы.
    plain = [item for item in usage if item.get("kind") != "plan"]
    while len(plain) > requests:
        for index, item in enumerate(usage):
            if item.get("kind") != "plan":
                usage.pop(index)
                break
        else:
            break
        plain = [item for item in usage if item.get("kind") != "plan"]


def _exchange_stored(memory: List[Dict[str, str]], user_text: str) -> bool:
    """True, если агент записал в память пару «запрос + ответ».

    Нужно, чтобы список замеров токенов не разъезжался с историей: пустой
    запрос, отказ по безопасности и прочие ранние выходы в память не пишут.
    """
    text = (user_text or "").strip()
    return (
        len(memory) >= 2
        and memory[-2].get("role") == "user"
        and memory[-2].get("content") == text
        and memory[-1].get("role") == "assistant"
    )


def _exchange_memory(agent: Agent, branch: Optional[str],
                     memory: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """История, в которую агент записал текущий обмен репликами.

    В стратегии «branching» диалог ведётся внутри выбранной ветки — там своя
    история сообщений, и замер токенов нужно сверять именно с ней, а не с
    корневой перепиской.
    """
    branch_id = str(branch or "").strip()
    if branch_id and branch_id in agent.branches:
        return agent.branches[branch_id].get("messages") or []
    return memory


def _memory_target(layer: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """Контейнер и ключ слоя памяти для записи.

    Рабочая память привязана к ТЕКУЩЕЙ ЗАДАЧЕ текущего профиля (её нет —
    добавлять некуда, вернётся None), долговременная — к ПРОФИЛЮ: профили
    изолированы, поэтому база знаний у каждого своя.
    """
    normalized = str(layer or "").strip().lower()
    if normalized in ("long", "long_term", "long-term", "долговременная"):
        return workspace_store.long_term_container(_workspace, _current_profile_id()), \
            workspace_store.MEMORY_LONG_TERM
    if normalized in ("work", "working", "рабочая"):
        return _current_task(), workspace_store.MEMORY_WORKING
    raise HTTPException(status_code=400, detail="Неизвестный слой памяти")


# ---------------------------------------------------------------------------
# Конечный автомат задачи (Task State Machine)
# ---------------------------------------------------------------------------
# Состояние задачи (этап → текущий шаг → ожидаемое действие) живёт в диалоге
# ТЕКУЩЕЙ сессии (dialog["state"], см. app/ai/task_state.py), а переходы
# выполняет ТОЛЬКО этот модуль: агент состояние получает готовым и показывает
# его модели системным блоком, но этапы не меняет.
#
# Подтверждение плана приходит и кнопкой («Подтвердить план» → POST
# /api/agent/state/confirm), и текстом сообщения: «ок», «поехали», «работай
# автономно». Слова-подтверждения сверяются ЦЕЛИКОМ (сообщение из одних
# подтверждений, не длиннее _CONFIRM_LIMIT слов), иначе «ок, но шаг 2 переставь»
# считалось бы подтверждением, хотя это правки плана.
_CONFIRM_WORDS = frozenset((
    "ок", "окей", "океюшки", "ok", "okay", "да", "ага", "угу", "yes", "y",
    "давай", "поехали", "погнали", "старт", "стартуем", "начинай", "начинаем",
    "запускай", "работай", "делай", "выполняй", "действуй", "go",
    "принято", "принимаю", "подтверждаю", "подтверждение", "согласен",
    "согласна", "верно", "хорошо", "отлично", "супер", "норм", "нормально",
    "план", "плану", "плана", "подходит", "устраивает", "годится", "принят",
    "принято",
))
_CONFIRM_LIMIT = 4          # слов в сообщении-подтверждении
# «Работай автономно» и «перезапусти» ищем как фразы внутри сообщения.
_AUTONOMOUS_RE = re.compile(r"(автономн|без подтверждени|не спрашивай|сам решай)",
                            re.IGNORECASE)
_RESTART_RE = re.compile(r"(перезапуст|заново|сначала|с нуля|сброс|restart)",
                         re.IGNORECASE)
# «Работай автономно» — это и подтверждение плана, и переключение режима, но
# только когда это ВСЁ сообщение: в длинной фразе («сделай отчёт и работай
# автономно») пользователь даёт новый запрос, и план нужно строить заново.
_AUTONOMOUS_CONFIRM_LIMIT = 3
# Предупреждение о переполнении лимита токенов — не провал шага (сам ответ
# пользователь получает), поэтому в самопроверке оно не считается ошибкой.
_LIMIT_WARNING_MARK = "Лимит токенов превышен"
# Префикс служебных сообщений автомата в debug-чате.
_MACHINE = "Автомат задачи"


def _normalized(text: str) -> str:
    """Текст без пунктуации и лишних пробелов, в нижнем регистре."""
    return " ".join(re.sub(r"[^\w\s]+", " ", str(text or "").lower()).split())


def _is_plan_confirmation(text: str) -> bool:
    """True, если сообщение — подтверждение плана («ок», «поехали», …).

    Подтверждением считается короткое сообщение из одних подтверждающих слов.
    «Работай автономно» подтверждает план и включает автономный режим — но
    только если это ВСЁ сообщение: в длинной фразе пользователь дал новый
    запрос, и план строится заново.
    """
    normalized = _normalized(text)
    if not normalized:
        return False
    tokens = normalized.split()
    if _AUTONOMOUS_RE.search(normalized):
        return len(tokens) <= _AUTONOMOUS_CONFIRM_LIMIT
    if len(tokens) > _CONFIRM_LIMIT:
        return False
    return all(token in _CONFIRM_WORDS for token in tokens)


def _wants_autonomous(text: str) -> bool:
    """True, если пользователь просит работать автономно (без подтверждений)."""
    return bool(_AUTONOMOUS_RE.search(_normalized(text)))


def _wants_restart(text: str) -> bool:
    """True, если пользователь просит перезапустить задачу после ошибки."""
    return bool(_RESTART_RE.search(_normalized(text)))


def _plan_message(state: "task_state.TaskState", autonomous: bool = False) -> str:
    """Текст плана для чата (этап planning): шаги и что делать дальше."""
    steps = state.steps or []
    plural = task_state.steps_word(len(steps))
    lines = [f"📋 План задачи — {len(steps)} {plural}:"]
    lines.extend(f"{number}. {step}" for number, step in enumerate(steps, 1))
    if autonomous:
        lines.append(
            "Режим «работай автономно»: подтверждение плана не требуется — "
            "начинаю выполнение с первого шага."
        )
    else:
        lines.append(
            "Подтвердите план — кнопка «Подтвердить план» в полосе состояния или слово «ок». "
            "Если нужно иначе — напишите правки сообщением, план перестрою."
        )
    return "\n".join(lines)


def _self_check(answered: bool, errors: List[str], steps_total: int,
                steps_done: int, stored: bool = True) -> Tuple[bool, str]:
    """Самопроверка результата на этапе validation (без сети и LLM).

    Проверяем то, что видно серверу: содержательный ответ модели получен,
    критических ошибок при выполнении не было, обмен репликами записан в память
    диалога и все шаги плана пройдены. Проблема → возврат в execution на текущий
    шаг (validation → execution), всё в порядке → validation → done.
    """
    problems: List[str] = []
    if not answered:
        problems.append("содержательного ответа от модели нет")
    if errors:
        problems.append("ошибки выполнения: " + "; ".join(errors[:2]))
    if not stored:
        problems.append("обмен репликами не записан в память диалога")
    if steps_total and steps_done < steps_total:
        problems.append(f"выполнено шагов: {steps_done} из {steps_total}")
    if problems:
        return False, ", ".join(problems)
    return True, (f"ответ получен, ошибок нет, обмен сохранён, "
                  f"шагов выполнено: {steps_total or 1}")


def _restore_message_sources(previous: List[Dict[str, Any]],
                             current: List[Dict[str, Any]]) -> None:
    """Возвращает репликам пометки `source` после обхода через память агента.

    Память агента хранит только пару role/content (лишние поля нельзя отправлять
    в API), поэтому пометка «эту реплику сгенерировал автомат» (`source ==
    "machine"`) терялась уже на следующем запросе: в файле workspace все
    служебные реплики выглядели как написанные пользователем, а `_task_request`
    (поиск исходного запроса для проверки результата) мог принять за запрос
    служебную фразу «Продолжай по плану: …». Сопоставляем по роли и тексту.
    """
    marks = {
        (message.get("role"), message.get("content")): message.get("source")
        for message in (previous or [])
        if isinstance(message, dict) and message.get("source")
    }
    if not marks:
        return
    for message in (current or []):
        if isinstance(message, dict) and not message.get("source"):
            source = marks.get((message.get("role"), message.get("content")))
            if source:
                message["source"] = source


def _task_request(messages: List[Dict[str, Any]], fallback: str = "") -> str:
    """Запрос пользователя, с которого началась ТЕКУЩАЯ задача.

    Ищем последнее сообщение пользователя, написанное ЧЕЛОВЕКОМ: реплики,
    сгенерированные автоматом, помечены `source == "machine"` (см.
    continue_step) и запросом не считаются. Нужно для содержательной проверки
    результата — она сверяет работу именно с тем, что просил пользователь.
    """
    for message in reversed(messages or []):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "user" and message.get("source") != "machine":
            content = str(message.get("content") or "").strip()
            if content:
                return content
    return str(fallback or "").strip()


def _log_event(dialog: Optional[Dict[str, Any]], event: Dict[str, Any]) -> None:
    """Пишет событие агента в журнал чата сессии (что видит пользователь).

    В журнал идут только узлы окна чата: ответ (bot) и служебные сообщения
    (debug/error). Служебные события (state/usage/done/branches) не пишутся:
    это не сообщения, а состояние и замеры.
    """
    if not dialog:
        return
    kind = event.get("type")
    if kind == "debug":
        workspace_store.add_log(dialog, workspace_store.LOG_DEBUG, event.get("text"))
    elif kind == "error":
        workspace_store.add_log(dialog, workspace_store.LOG_ERROR, event.get("text"))
    elif kind == "bot":
        workspace_store.add_log(dialog, workspace_store.LOG_ASSISTANT, event.get("text"))


def _defer_stop(session: Dict[str, Any], action: str) -> Optional[Dict[str, Any]]:
    """Откладывает «Паузу»/«Отмену», если шаг ЭТОЙ сессии выполняется сейчас.

    Возвращает оптимистичный снимок (кнопка и полоса в интерфейсе меняются
    мгновенно, пометка `pending` объясняет, что команда применится после ответа
    модели) либо None — тогда вызывающий маршрут применяет переход СРАЗУ, не
    ожидая блокировку: диалог этой сессии сейчас никто не пишет (шаг идёт в
    другой сессии или не идёт вовсе), а раньше такой запрос ждал чужой вызов LLM
    и кнопка «зависала».
    """
    if str(session.get("id") or "") not in _running_sessions:
        return None
    _pending_stops[str(session["id"])] = action
    state = workspace_store.dialog_state(session)
    return {
        "state": _pending_state(session, state, action),
        "session": _session_payload(session),
    }


def _pending_state(session: Dict[str, Any], state: "task_state.TaskState",
                   action: str) -> Dict[str, Any]:
    """Снимок состояния с ОПТИМИСТИЧНО применённой отложенной остановкой.

    Нужен, чтобы «Пауза»/«Отменить», нажатые во время шага, отвечали мгновенно:
    интерфейс сразу видит новое состояние и пометку pending, а сам переход
    применяется потоком по завершении текущего вызова LLM. Копия состояния —
    чтобы не менять то, что поток сейчас пишет в диалог.
    """
    preview = task_state.from_dict(task_state.to_dict(state), state.task_id)
    try:
        if action == PENDING_CANCEL:
            task_state.cancel(preview, "пользователь отменил задачу")
        else:
            task_state.pause(preview, "пользователь нажал «Пауза»")
    except task_state.IllegalTransition:
        pass   # этап уже терминальный — отменять/паузить нечего
    snapshot = task_state.snapshot(preview)
    snapshot[PENDING_KEY] = action
    return snapshot


def _apply_pending_stop(session: Dict[str, Any], state: "task_state.TaskState",
                        force: Optional[str] = None) -> bool:
    """Применяет отложенную остановку («Пауза»/«Отменить»), если она есть.

    Вызывается потоком, который держал блокировку: пользователь нажал кнопку во
    время шага, и переход не мог быть применён сразу. force — применить именно
    это действие (поток уже снял намерение из очереди). True — остановка
    применена (состояние изменено, причина записана в историю).
    """
    action = force or _pending_stops.pop(str(session.get("id") or ""), None)
    if action is None:
        return False
    try:
        if action == PENDING_CANCEL:
            task_state.cancel(state, "пользователь отменил задачу во время выполнения шага")
        else:
            task_state.pause(state, "пользователь нажал «Пауза» во время выполнения шага")
    except task_state.IllegalTransition as exc:
        logger.info("Отложенная остановка %s не применена: %s", action, exc)
        return False
    return True


def _state_event(state: "task_state.TaskState") -> dict:
    """Событие потока с состоянием автомата (фронт рисует полосу этапов)."""
    return {"type": "state", "state": task_state.snapshot(state)}


def _session_payload(session: Optional[Dict[str, Any]]) -> Optional[dict]:
    """Краткая ссылка на текущую сессию (для ответов маршрутов состояния)."""
    if session is None:
        return None
    return {"id": session["id"], "title": workspace_store.session_title(session)}


def _require_session() -> Dict[str, Any]:
    """Текущая сессия-диалог; диалогов нет — создаём (как в agent_chat)."""
    task = _current_task()
    if task is None:
        raise HTTPException(status_code=400, detail="Сначала создайте проект")
    session = workspace_store.active_session(_workspace, task)
    if session is None:
        session = workspace_store.create_session(task)
    return session


def _state_response(session: Dict[str, Any]) -> dict:
    """Ответ маршрутов состояния: снимок автомата + текущая сессия.

    Если для сессии ждёт отложенная остановка («Пауза»/«Отменить», нажатая во
    время шага), в снимке появляется пометка `pending` — интерфейс показывает по
    ней «остановлю после текущего шага».
    """
    snapshot = task_state.snapshot(workspace_store.dialog_state(session))
    pending = _pending_stops.get(str(session.get("id") or ""))
    if pending:
        snapshot[PENDING_KEY] = pending
    return {"state": snapshot, "session": _session_payload(session)}


async def _persist_state(session: Dict[str, Any], state: "task_state.TaskState") -> None:
    """Записывает состояние автомата в диалог сессии и сохраняет workspace."""
    workspace_store.set_dialog_state(session, state)
    await _persist()


@router.post("/agent/state/cancel")
async def state_cancel() -> dict:
    """Кнопка «Отменить задачу»: автомат останавливается на этапе cancelled.

    Отмена — не переход автомата, а остановка задачи (см. task_state.cancel):
    разрешена из любого незавершённого этапа, cancelled — терминальный.
    Прогресс по плану и текущий шаг сбрасываются, а сам диалог, его память и
    история остаются; следующее сообщение пользователя начинает НОВУЮ задачу
    (done|cancelled → новый автомат, см. agent_chat).
    """
    session = _current_session()
    if session is None:
        raise HTTPException(status_code=400, detail="Отменять нечего: задач ещё нет")
    deferred = _defer_stop(session, PENDING_CANCEL)
    if deferred is not None:
        return deferred
    # См. state_pause: применяем сразу, не дожидаясь чужого шага.
    state = workspace_store.dialog_state(session)
    try:
        task_state.cancel(state, "пользователь отменил задачу (кнопка «Отменить задачу»)")
    except task_state.IllegalTransition as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    await _persist_state(session, state)
    return _state_response(session)


@router.get("/agent/state")
async def state_get() -> dict:
    """Состояние конечного автомата ТЕКУЩЕЙ сессии (полоса этапов над чатом).

    Отдаёт {"state": {stage, current_step, expected_action, steps, paused,
    reason, history, …}, "session": {"id", "title"}|None}. Фронт вызывает
    маршрут при включении режима агента и при переключении задачи/диалога (а
    также получает то же состояние событиями "state" в потоке /api/agent/chat).
    """
    # Блокировку не берём: снимок состояния — чистое чтение (см. state_pause).
    session = _current_session()
    if session is None:
        # Диалогов ещё нет — отдаём состояние только что созданной задачи
        # (этап planning, план пуст): полоса этапов видна сразу.
        empty = task_state.new_state("")
        return {"state": task_state.snapshot(empty), "session": None}
    return _state_response(session)


@router.post("/agent/state/pause")
async def state_pause() -> dict:
    """Кнопка «Пауза» в полосе состояния: автомат останавливается.

    Этап и текущий шаг НЕ меняются — задача просто ждёт нажатия «Продолжить»;
    запросы к агенту в это время отклоняются (400), чтобы шаг не выполнялся
    «через паузу».

    Если прямо сейчас выполняется шаг (блокировку держит поток ответа), запрос
    НЕ ждёт его: намерение кладётся в `_pending_stops`, а интерфейс мгновенно
    получает снимок с `paused = true` и пометкой `pending = "pause"` («остановлю
    после текущего шага»). Поток применит паузу, как только ответит модель.
    """
    session = _current_session()
    if session is None:
        raise HTTPException(status_code=400, detail="Сначала создайте проект")
    deferred = _defer_stop(session, PENDING_PAUSE)
    if deferred is not None:
        return deferred
    # Сразу и без блокировки: её держит поток ДРУГОЙ сессии, а диалог этой никто
    # не пишет (см. _defer_stop). Иначе кнопка ждала бы чужого ответа модели.
    state = workspace_store.dialog_state(session)
    try:
        task_state.pause(state, "пользователь нажал «Пауза»")
    except task_state.IllegalTransition as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    await _persist_state(session, state)
    return _state_response(session)


@router.post("/agent/state/resume")
async def state_resume() -> dict:
    """Кнопка «Продолжить»: снимает паузу, работа идёт с того же шага.

    Заодно снимает неотработанное намерение «Пауза» (пользователь передумал).
    """
    session = _current_session()
    if session is not None:
        _pending_stops.pop(str(session["id"]), None)
    # Блокировка СВОЕЙ задачи (задача на паузе, шаг не идёт — берётся сразу);
    # другие задачи на своих блокировках и друг друга не ждут.
    session = _require_session()
    async with _session_lock(session["id"]):
        state = workspace_store.dialog_state(session)
        task_state.resume(state, "пользователь нажал «Продолжить»")
        await _persist_state(session, state)
        return _state_response(session)


@router.post("/agent/state/confirm")
async def state_confirm() -> dict:
    """Подтверждает план кнопкой: planning | awaiting_user → execution.

    План должен быть построен (приходит с первым запросом пользователя) —
    подтверждать пустой план нечего, это 400.
    """
    session = _require_session()
    async with _session_lock(session["id"]):
        state = workspace_store.dialog_state(session)
        if not state.steps:
            raise HTTPException(status_code=400,
                                detail="План ещё не построен — отправьте запрос")
        if state.paused:
            raise HTTPException(status_code=400, detail="Задача на паузе — нажмите «Продолжить»")
        if state.stage == "awaiting_user":
            task_state.start_planning(state, "пользователь подтвердил план кнопкой")
        if state.stage != "planning":
            raise HTTPException(
                status_code=400,
                detail=f"Подтвердить план можно на этапах planning/awaiting_user, сейчас {state.stage}",
            )
        task_state.plan_ready(state, state.steps, "план подтверждён пользователем (кнопка)")
        await _persist_state(session, state)
        return _state_response(session)


@router.put("/agent/state/plan")
async def state_plan(payload: PlanUpdate) -> dict:
    """Правка плана на этапе planning: пользователь задаёт свои шаги.

    Шаги заменяются целиком и задача снова ждёт подтверждения (planning →
    awaiting_user) — переход в execution делает кнопка «Подтвердить план» или
    ответ «ок».
    """
    steps = task_state.clean_steps(payload.steps)
    if not steps:
        raise HTTPException(status_code=400, detail="План не может быть пустым")
    session = _require_session()
    async with _session_lock(session["id"]):
        state = workspace_store.dialog_state(session)
        if state.stage not in ("planning", "awaiting_user"):
            raise HTTPException(
                status_code=400,
                detail=f"План правится на этапе planning, сейчас {state.stage}",
            )
        if state.stage == "awaiting_user":
            task_state.start_planning(state, "пользователь правит план — возвращаюсь к планированию")
        task_state.await_confirmation(state, steps, "план обновлён пользователем — жду подтверждения")
        await _persist_state(session, state)
        return _state_response(session)


@router.post("/chat")
def chat(msg: ChatMessage) -> dict:
    """Принимает сообщение пользователя и возвращает ответ бота."""
    if not msg.content.strip():
        return {"user": msg.content, "bot": "Пожалуйста, введите сообщение."}
    answer, correct = service.generate_response(
        msg.content,
        msg.format,
        msg.max_tokens,
        msg.stop,
        msg.expert_mode,
        msg.expert_mode_type,
        msg.expert_roles,
        msg.temperatures,
        msg.models,
    )
    result = {"user": msg.content, "bot": answer, "correct": correct}
    # Настройка «Тест моделей»: ответы каждой модели по отдельности.
    if isinstance(answer, dict) and "model_responses" in answer:
        model_responses = answer["model_responses"]
        result["model_responses"] = model_responses
        result["bot"] = "\n".join(
            f"Ответ модели {r['model']}: {r['text']}" for r in model_responses
        ) if model_responses else "Пожалуйста, введите сообщение."
        # Аналитика судьи-аналитика (время, токены, стоимость) — отдельным полем.
        if answer.get("analytics"):
            result["analytics"] = answer["analytics"]
        return result
    # Настройка «Температура»: несколько независимых ответов + резюме судьи.
    # Фронтенд выводит каждый ответ с пометкой «Ответ при значении temperature …»,
    # а затем — резюме судьи-аналитика.
    if isinstance(answer, dict) and "responses" in answer:
        responses = answer["responses"]
        result["responses"] = responses
        result["bot"] = "\n".join(
            f"Ответ при значении temperature {r['temperature']}: {r['text']}"
            for r in responses
        ) if responses else "Пожалуйста, введите сообщение."
        if answer.get("judge"):
            result["judge"] = answer["judge"]
    return result


# ---------------------------------------------------------------------------
# Рабочее пространство: задачи и сессии (диалоги)
# ---------------------------------------------------------------------------
@router.get("/agent/workspace")
async def workspace_get() -> dict:
    """Снимок workspace: задачи, текущая задача, её диалоги и текущий диалог.

    Отдаёт {"tasks": [{"id", "name"}], "active_task": id|None,
    "sessions": [{"id", "title"}], "active_session": id|None}. Заголовок сессии —
    первые слова первого запроса пользователя в этой сессии (или своё название,
    если пользователь переименовал её карандашом). Фронтенд рисует по этому
    снимку панель Workspace: список задач в выпадающем списке и список сессий
    в истории запросов.
    """
    # Блокировку НЕ берём: снимок — чистая операция без await, а панель
    # workspace не должна ждать конца шага агента (он держит блокировку СВОЕЙ
    # задачи весь стрим) — иначе переключение задач «зависало».
    return _snapshot()


@router.post("/agent/tasks")
async def task_create(payload: TaskCreate) -> dict:
    """Создаёт задачу (кнопка «Новая задача») и делает её текущей.

    Задача создаётся без диалогов: первый запрос пользователя заведёт сессию
    сам. Пока задачи нет, диалог в режиме агента начать нельзя.
    """
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Введите название проекта")
    # Блокировку не берём: создание проекта — правка workspace и запись файла без
    # await (как и переключение задач), а ждать конца шага агента кнопке незачем.
    # Задача создаётся ДЛЯ ТЕКУЩЕГО ПРОФИЛЯ: у профилей свои задачи и диалоги,
    # чужой профиль их не увидит.
    workspace_store.create_task(_workspace, name, _current_profile_id())
    await _persist()
    return _snapshot()


@router.put("/agent/tasks/{task_id}")
async def task_rename(task_id: str, payload: NameUpdate) -> dict:
    """Переименовывает задачу (карандаш рядом со списком задач)."""
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Введите название проекта")
    # Блокировку не берём (см. task_create): переименование — правка строки и
    # запись файла, ждать шага агента незачем.
    task = _own_task(workspace_store.find_task(_workspace, task_id))
    if task is None:
        raise HTTPException(status_code=404, detail="Проект не найден")
    task["name"] = name[:120]
    await _persist()
    return _snapshot()


@router.delete("/agent/tasks/{task_id}")
async def task_delete(task_id: str) -> dict:
    """Удаляет задачу вместе со всеми её диалогами (корзина у списка задач)."""
    async with _workspace_lock:
        if _own_task(workspace_store.find_task(_workspace, task_id)) is None \
                or not workspace_store.delete_task(_workspace, task_id):
            raise HTTPException(status_code=404, detail="Проект не найден")
        await _persist()
        return _snapshot()


@router.post("/agent/tasks/{task_id}/select")
async def task_select(task_id: str) -> dict:
    """Переключает текущую задачу (выпадающий список «Текущая задача»).

    У каждой задачи своя история диалогов: фронт после переключения заново
    запрашивает диалог текущей сессии (GET /api/agent/history).
    """
    # Блокировку не берём: переключение задачи — правка указателя (без await) и
    # запись файла; иначе клик ждал бы конца ответа агента.
    task = _own_task(workspace_store.find_task(_workspace, task_id))
    if task is None:
        raise HTTPException(status_code=404, detail="Проект не найден")
    # Текущая задача — у КАЖДОГО профиля своя: чужой указатель не трогаем.
    workspace_store.set_active_task(_workspace, task)
    await _persist()
    return _snapshot()


@router.post("/agent/sessions")
async def session_create() -> dict:
    """Создаёт в текущей задаче новый пустой диалог (кнопка «Новая сессия»).

    Сессия сразу становится текущей — её диалог пуст, а заголовок в истории
    появится по первому запросу пользователя.
    """
    # Блокировку не берём (см. task_create): новая задача отвечает сразу, даже
    # если прямо сейчас выполняется шаг агента — шаг пишет в СВОЮ задачу.
    task = _current_task()
    if task is None:
        raise HTTPException(status_code=400, detail="Сначала создайте проект")
    workspace_store.create_session(task)
    await _persist()
    return _snapshot()


@router.put("/agent/sessions/{session_id}")
async def session_rename(session_id: str, payload: NameUpdate) -> dict:
    """Переименовывает диалог (карандаш в элементе истории сессий)."""
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Введите название задачи")
    # См. task_rename: без ожидания шага агента.
    _, session = _find_session_anywhere(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    session["title"] = name[:200]
    await _persist()
    return _snapshot()


@router.delete("/agent/sessions/{session_id}")
async def session_delete(session_id: str) -> dict:
    """Удаляет диалог (корзина в элементе истории сессий).

    Если удалён текущий диалог, текущим становится соседний — или ни одного,
    и тогда следующий запрос заведёт новый.
    """
    async with _workspace_lock:
        task, session = _find_session_anywhere(session_id)
        if task is None or session is None:
            raise HTTPException(status_code=404, detail="Задача не найдена")
        workspace_store.delete_session(task, session_id)
        await _persist()
        return _snapshot()


@router.post("/agent/sessions/{session_id}/select")
async def session_select(session_id: str) -> dict:
    """Переключает текущий диалог внутри его задачи (клик по элементу истории)."""
    # Блокировку не берём (см. task_select): переключение диалога должно быть
    # мгновенным даже когда агент отвечает. Поток ответа пишет ТОЛЬКО в свой
    # диалог (он разрешён под блокировкой при старте), поэтому указатель
    # активной сессии ему не мешает.
    task, session = _find_session_anywhere(session_id)
    if task is None or session is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    # Переключение диалога внутри задачи своего профиля.
    workspace_store.set_active_task(_workspace, task)
    task["active_session"] = session["id"]
    await _persist()
    return _snapshot()


# ---------------------------------------------------------------------------
# Слои памяти агента: рабочая (задача) и долговременная (глобальная)
# ---------------------------------------------------------------------------
@router.get("/agent/memory")
async def memory_get() -> dict:
    """Содержимое слоёв памяти для панели «Состояние памяти».

    Отдаёт {"task": {"id", "name"}|None,
    "working": [{"id", "text", "created", "source"}, ...],
    "long_term": [{"id", "text", "created", "source"}, ...]} —
    рабочая память (данные ТЕКУЩЕЙ задачи) и долговременная (глобальная база
    знаний) отдельными разбивками. Обе наполняет пользователь кнопками
    «добавить в рабочую память» / «добавить в долговременную память» под
    сообщениями; агент получает их на каждый запрос целиком, стратегиям
    управления контекстом они не подчиняются.
    """
    # Блокировку здесь НЕ берём намеренно: снимок памяти — чистая операция без
    # await, поэтому гонки с записью в workspace нет, зато панель не ждёт
    # окончания шага агента (он держит блокировку СВОЕЙ задачи весь стрим).
    return workspace_store.memory_snapshot(_workspace, _current_profile_id())


@router.post("/agent/memory")
async def memory_add(payload: MemoryEntryCreate) -> dict:
    """Добавляет запись в слой памяти (кнопки под сообщениями в чате)."""
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Нечего добавлять в память")
    async with _workspace_lock:
        container, key = _memory_target(payload.layer)
        if container is None:
            raise HTTPException(
                status_code=400,
                detail="Сначала создайте проект — рабочая память привязана к проекту",
            )
        workspace_store.add_memory(container, key, text, payload.source)
        await _persist()
        return workspace_store.memory_snapshot(_workspace, _current_profile_id())


@router.delete("/agent/memory/{layer}/{entry_id}")
async def memory_delete(layer: str, entry_id: str) -> dict:
    """Удаляет запись из слоя памяти (корзина в панели «Состояние памяти»)."""
    async with _workspace_lock:
        container, key = _memory_target(layer)
        if container is None or not workspace_store.delete_memory(container, key, entry_id):
            raise HTTPException(status_code=404, detail="Запись не найдена")
        await _persist()
        return workspace_store.memory_snapshot(_workspace, _current_profile_id())


# ---------------------------------------------------------------------------
# Инварианты: правила, которые агент не имеет права нарушить
# ---------------------------------------------------------------------------
# Инвариант — короткое текстовое правило (выбранная архитектура, принятые
# технические решения, ограничения стека, бизнес-правила). Хранятся ОТДЕЛЬНО ОТ
# ДИАЛОГА: у ПРОЕКТА (задача workspace) свои правила, у ЗАДАЧИ-диалога (сессия)
# свои; в messages они не пишутся. Агент получает их на каждый запрос и уводит в
# модель отдельным системным блоком (см. app/ai/invariants.py), поэтому правила
# действуют в любой стратегии контекста. Правка — по кнопке «Инварианты» в шапке
# чата (одно текстовое поле = один инвариант).
#
# ЗАПИСЬ ПРАВИЛ ОБРАЩЕНИЙ К МОДЕЛИ НЕ ДЕЛАЕТ: инвариант — это данные. Всё
# взаимодействие с LLM идёт В ДИАЛОГЕ: по новому запросу пользователя агент
# разбирает его на соответствие правилам ДО планирования (см.
# _preflight_invariants). ПРИОРИТЕТ ВСЕГДА У ПРАВИЛА ПРОЕКТА: если правило задачи
# просит то, что проект запрещает, это нарушение правила проекта — агент
# отказывается от запроса и предлагает альтернативы (никакого выбора «какое
# правило главнее» пользователю не даём).
def _invariants_session(session_id: str = "") -> Tuple[Optional[Dict[str, Any]],
                                                       Optional[Dict[str, Any]]]:
    """(проект, задача-диалог) для операций с инвариантами.

    Область «project» живёт в проекте, область «task» — в диалоге задачи.
    Задача-диалог при необходимости создаётся (как при первом запросе к агенту).
    """
    if session_id:
        return _find_session_anywhere(session_id)
    task = _current_task()
    if task is None:
        return None, None
    session = workspace_store.active_session(_workspace, task)
    if session is None:
        session = workspace_store.create_session(task)
    return task, session


def _invariants_snapshot(task: Optional[Dict[str, Any]],
                         session: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Снимок инвариантов для агента: правила проекта, правила задачи, исключения.

    Исключения — решения пользователя «главнее инвариант задачи»: они снимают
    противоречие, поэтому уходят в контекст отдельным списком.

    `overridden` — правила ЗАДАЧИ, НЕ ДЕЙСТВУЮЩИЕ из-за противоречия правилам
    проекта: их номера вернул разбор запроса (`недействующие`), а в блок правил
    уходят тексты, чтобы агент видел, какое именно требование задачи выполнять
    нельзя. Без этого блок называл противоречащее правило задачи «нарушать
    нельзя», и план строился по нему (запрет проекта не срабатывал).
    """
    dialog = (session or {}).get("dialog") or {}
    task_items = [dict(entry) for entry in
                  (workspace_store.invariants(dialog) if dialog else [])]
    analysis = dialog.get("analysis") if isinstance(dialog.get("analysis"), dict) else {}
    return {
        "project": [dict(entry) for entry in
                    (workspace_store.invariants(task) if task else [])],
        "task": task_items,
        # Решений «главнее задача» больше не бывает: правило проекта всегда в
        # силе. Поле оставлено пустым для совместимости формата снимка.
        "exceptions": [],
        "overridden": invariants_store.overridden_texts(
            task_items, (analysis or {}).get("overridden")),
    }


def _invariants_view(task: Optional[Dict[str, Any]],
                     session: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Снимок инвариантов для интерфейса (модалка «Инварианты»).

    Отдаёт правила проекта и задачи, утверждённые решения по противоречиям
    (`exceptions`) и счётчики для шестерёнок. Проверок пар правил здесь НЕТ:
    инварианты записываются без обращения к модели, а противоречия выявляются в
    диалоге (см. _preflight_invariants).
    """
    dialog = (session or {}).get("dialog") or {}
    project = [dict(entry) for entry in (workspace_store.invariants(task) if task else [])]
    task_items = [dict(entry) for entry in
                  (workspace_store.invariants(dialog) if dialog else [])]
    return {
        "project": project,
        "task": task_items,
        "task_id": (session or {}).get("id"),
        "project_id": (task or {}).get("id"),
        "pairs": len(invariants_store.pairs(project, task_items)),
        "exceptions": [],
        "has_conflict": False,
        "counts": {"project": len(project), "task": len(task_items),
                   "conflict": 0, "exceptions": 0},
    }

def _flat(text: Any) -> str:
    """Текст в одну строку без краевых пробелов (сверка текстов вариантов)."""
    return " ".join(str(text or "").split())


def _verified_choice(dialog: Optional[Dict[str, Any]], text: str,
                     snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Разбор с проверенным вариантом, если это сообщение — ОН САМ (иначе None).

    Варианты под сообщением об отказе прошли проверку по правилам
    (`invariants_store.check_suggestions`), поэтому повторно «судить» свой же
    проверенный текст нельзя: иначе пользователь выбирает вариант и снова
    получает отказ — цикл, в котором задача не выполняется никогда. Сверяем
    текст варианта И набор правил (`rules_signature`): правила изменились —
    разбор делаем заново, обычным порядком.

    Возвращает найденный разбор (из него вызывающий забирает `overridden` —
    пометку о недействующих правилах задачи, которая нужна блоку правил).

    Берём только ПОСЛЕДНИЙ узел с вариантами, и только если между ним и текущим
    запросом нет других содержательных узлов (ответа агента, плана): старая
    запись журнала не должна отменять проверку нового запроса.
    """
    log = list((dialog or {}).get("log") or [])
    index = len(log) - 1
    # Реплика пользователя и строка дебага ЭТОГО запроса уже записаны в журнал
    # (дебаг «сверяю запрос с инвариантами» уходит до вызова разбора): ищем
    # последний СОДЕРЖАТЕЛЬНЫЙ узел, пропуская служебные записи текущего запроса.
    while index >= 0 and log[index].get("kind") in (workspace_store.LOG_USER,
                                                    workspace_store.LOG_DEBUG):
        index -= 1
    if index < 0 or log[index].get("kind") != workspace_store.LOG_SUGGESTIONS:
        return None
    data = invariants_store.normalize_analysis(log[index].get("analysis"))
    if not data["suggestions_checked"] or not data["suggestions"]:
        return None
    if data["rules_signature"] != invariants_store.rules_signature(snapshot):
        return None
    wanted = _flat(text)
    if any(_flat(item.get("send")) == wanted for item in data["suggestions"]):
        return data
    return None


async def _preflight_invariants(task: Dict[str, Any], session: Dict[str, Any],
                                text: str, analyzer: Agent,
                                snapshot_now: Optional[Dict[str, Any]] = None
                                ) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
    """Сверяет запрос пользователя с инвариантами ДО планирования.

    Возвращает (разбор, расход служебного вызова, выбран_проверенный_вариант).
    Разбор кладётся в dialog["analysis"]: по подписи (запрос + действующие
    правила) видно, что он ещё актуален — тот же запрос при тех же правилах
    повторно модель не спрашивает. Правил нет вовсе — вызова LLM нет.

    Отдельный случай — пользователь отправил вариант, который агент САМ показал
    под отказом и проверил по правилам (см. `_verified_choice`): такой текст уже
    проверен при тех же правилах, поэтому служебного вызова нет и отказа быть не
    может — иначе выбор варианта приводил бы к новому отказу.
    """
    dialog = session["dialog"]
    snapshot = snapshot_now if snapshot_now is not None else _invariants_snapshot(task, session)
    choice = _verified_choice(dialog, text, snapshot)
    if choice is not None:
        analysis = dict(invariants_store.empty_analysis())
        analysis["verdict"] = invariants_store.COMPLIANCE_CLEAR
        analysis["kind"] = invariants_store.COMPLIANCE_CLEAR
        analysis["request"] = str(text or "")
        # Пометка о недействующих правилах задачи переносится из разбора, по
        # которому вариант и был предложен: иначе блок правил для планировщика
        # снова назвал бы противоречащее правило задачи обязательным.
        analysis["overridden"] = list(choice.get("overridden") or [])
        analysis["signature"] = invariants_store.analysis_signature(text, snapshot)
        analysis["rules_signature"] = invariants_store.rules_signature(snapshot)
        dialog["analysis"] = dict(analysis)
        return analysis, {}, True
    if not invariants_store.has_rules(snapshot):
        dialog["analysis"] = dict(invariants_store.empty_analysis())
        return dict(dialog["analysis"]), {}, False
    signature = invariants_store.analysis_signature(text, snapshot)
    saved = dialog.get("analysis")
    if isinstance(saved, dict) and saved.get("signature") == signature:
        return invariants_store.normalize_analysis(saved), {}, False
    analysis = await analyzer.check_invariants(text, snapshot)
    dialog["analysis"] = dict(analysis)
    return analysis, dict(analyzer.last_usage or {}), False


def _analysis_view(analysis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Разбор для интерфейса: сообщение отказа и варианты-альтернативы.

    Варианты — готовые тексты запросов, которые правила НЕ нарушают: каждый
    прошёл служебную проверку (`check_suggestions`) перед показом, а непроверенные
    варианты в интерфейс не попадают. Страница отправляет выбранный как новое
    сообщение пользователя. Выбора «какое правило главнее» нет: правило проекта
    всегда в силе, поэтому конфликт правила задачи с правилом проекта — обычное
    нарушение с альтернативами.
    """
    data = invariants_store.normalize_analysis(analysis)
    return {
        "verdict": data["verdict"],
        "kind": data["kind"],
        "message": _analysis_message(data),
        "explanation": data["explanation"],
        "suggestions": list(data["suggestions"]),
        "suggestions_checked": data["suggestions_checked"],
        "overridden": list(data["overridden"]),
        "rules_signature": data["rules_signature"],
    }


def _analysis_message(analysis: Dict[str, Any]) -> str:
    """Текст сообщения агента: какое правило нарушено и что делать дальше.

    Хвост про варианты зависит от того, что реально показано: обещать «варианты,
    которые правила не нарушают» можно ТОЛЬКО когда они прошли проверку
    (`suggestions_checked`), иначе пользователь будет выбирать из нарушающих.
    Правило задачи, противоречащее правилам проекта, в объяснении названо
    недействующим — и это видно пользователю (иначе правило выглядит молча
    проигнорированным).
    """
    data = invariants_store.normalize_analysis(analysis)
    explanation = str(data.get("explanation") or "").strip()
    head = ("⛔ Запрос нарушает инвариант (правило, которое нарушать нельзя) — "
            "выполнять его не буду. Инвариант проекта всегда главнее правила "
            "задачи: если задача просит то, что проект запрещает, действует "
            "запрет проекта.")
    if explanation:
        head += "\n\n" + explanation
    return head + _analysis_tail(data)


def _analysis_tail(data: Dict[str, Any]) -> str:
    """Хвост сообщения отказа: что показано вместо запрещённого требования."""
    variants = list(data.get("suggestions") or [])
    checked = bool(data.get("suggestions_checked"))
    overridden = list(data.get("overridden") or [])
    rules_hint = ""
    if overridden:
        rules_hint = ("\n\nПравила задачи, противоречащие правилам проекта, не "
                      "действуют. Если правило задачи устарело — поправьте или "
                      "удалите его в «Инварианты» (⚙️ у задачи).")
    if variants and checked:
        return ("\n\nНиже — варианты, которые правила не нарушают (каждый проверен "
                "по правилам). Выберите вариант, и я продолжу по нему."
                + rules_hint)
    if variants:
        return ("\n\n⚠️ Варианты не проверены: служебная проверка не подтвердила, "
                "что они укладываются в правила." + rules_hint)
    if not checked:
        return ("\n\nПодходящих вариантов не показываю: служебная проверка не "
                "удалась, а непроверенные варианты предлагать нельзя. "
                "Сформулируйте требование иначе." + rules_hint)
    return ("\n\nВариантов, не нарушающих правила, не нашлось — сформулируйте "
            "требование иначе." + rules_hint)


def _analysis_debug(analysis: Dict[str, Any]) -> str:
    """Строка дебага: вердикт разбора и число проверенных вариантов."""
    data = invariants_store.normalize_analysis(analysis)
    names = {"violation": "запрос нарушает инвариант",
             "clear": "нарушений нет"}
    kind = data["kind"] or data["verdict"] or "без вердикта"
    number = len(data["suggestions"])
    if data["verdict"] == "violation":
        note = " (варианты проверены по правилам)" if data["suggestions_checked"] \
            else " (варианты не проверены — проверка не дала вердикта)"
    else:
        note = ""
    return (
        f"разбор до планирования: {names.get(kind, kind)} — "
        f"вариантов-альтернатив: {number}{note}. "
        "План в этом случае не строится."
    )


async def _plan_gate(steps: List[str], analyzer: Agent, snapshot: Dict[str, Any],
                     replan: Any) -> Dict[str, Any]:
    """КОД-ГЕЙТ ПЛАНА: шаги проверяются по правилам, нарушающие не принимаются.

    Блок правил в контексте планировщика — ПРОСЬБА, а не гарантия: при правиле
    проекта «только нативная платформа android, никакой мультиплатформы» план
    приходил с шагами «UI на Compose для обоих платформ» и «сетевой слой в KMP»,
    и задача выполнялась в обход запрета. Поэтому готовые шаги уходят арбитру
    отдельным служебным вызовом (`analyzer.check_plan`), и:

      * нарушений нет — план принимается;
      * есть нарушающие шаги — ОДНА попытка перепланирования (`replan`): в
        пометке перечислены шаги, которые повторять нельзя; новый план
        проверяется так же;
      * проверка не удалась (`None`) — план НЕ принимается: «не проверено» не
        значит «совместимо» (как у вариантов-альтернатив);
      * после попытки нарушения остались — план НЕ принимается.

    Возвращает {"steps", "usage", "lines", "error"}: steps — принятые шаги (пусто,
    если план не принят), lines — строки дебага, error — текст ошибки для чата.
    """
    usage: Dict[str, Any] = {}
    lines: List[str] = []
    result: Dict[str, Any] = {"steps": [], "usage": usage, "lines": lines,
                              "error": None}
    if not steps or not invariants_store.has_rules(snapshot):
        result["steps"] = list(steps or [])
        return result
    attempt = 0
    while True:
        keep = await analyzer.check_plan(steps, snapshot)
        usage = merge_usage(usage, dict(analyzer.last_usage or {}))
        if keep is None:
            lines.append("проверка шагов плана по инвариантам не удалась "
                         "(служебный вызов не дал вердикта) — план не принимаю.")
            result["error"] = (
                "⚠️ План не принят: не удалось проверить шаги по инвариантам "
                "(служебная проверка не дала вердикта). Повторите запрос — "
                "выполнять непроверенный план нельзя."
            )
            result["usage"] = usage
            return result
        kept = set(keep)
        bad = [step for index, step in enumerate(steps) if index not in kept]
        if not bad:
            lines.append("шаги плана проверены по инвариантам — нарушений нет."
                         if attempt == 0 else
                         "новый план проверен по инвариантам — нарушений нет.")
            result["steps"] = list(steps)
            result["usage"] = usage
            return result
        lines.append("план нарушает инварианты — шаги: " + "; ".join(bad[:3]) + ".")
        if attempt >= invariants_store.PLAN_RETRIES:
            result["error"] = (
                "⚠️ План нарушает инварианты — выполнять его нельзя. Шаги, "
                "которые требуют запрещённого: " + "; ".join(bad[:4]) + ". "
                "Задача остановлена: переформулируйте запрос или поправьте "
                "правила в «Инварианты»."
            )
            result["usage"] = usage
            return result
        attempt += 1
        steps, more = await replan(bad)
        usage = merge_usage(usage, more)
        if not steps:
            lines.append("перепланирование не дало шагов — план не принимаю.")
            result["error"] = (
                "⚠️ План не принят: перепланирование по правилам не дало шагов. "
                "Повторите запрос."
            )
            result["usage"] = usage
            return result


def _last_analysis(dialog: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Последний разбор инвариантов из журнала чата задачи (None — разбора нет)."""
    for item in reversed((dialog or {}).get("log") or []):
        if item.get("kind") == workspace_store.LOG_SUGGESTIONS and item.get("analysis"):
            return dict(item["analysis"])
    return None


@router.post("/agent/invariants/choose")
async def invariant_choose(payload: InvariantPick) -> dict:
    """Клик по варианту-альтернативе под сообщением агента (разбор инвариантов).

    Вариант берётся из ПОСЛЕДНЕГО разбора в журнале чата задачи по номеру: с
    фронта приходит только номер, поэтому подменить текст нельзя. Сервер
    возвращает готовый текст запроса, а страница отправляет его как новое
    сообщение пользователя — правила он не нарушает, поэтому агент продолжит
    работу (план и шаги).

    Непроверенный вариант (запись журнала до появления проверки) тоже можно
    отправить: его текст уйдёт ОБЫЧНЫМ разбором инвариантов, как любой другой
    запрос — «не проверено» не значит «разрешено». А вот ПРОВЕРЕННЫЙ вариант
    разбора не требует вовсе (`_verified_choice`): правила менять не успели,
    поэтому отказ по нему был бы циклом.

    Выбора «какое правило главнее» здесь нет: правило проекта всегда в силе.
    """
    task, session = _invariants_session(str(payload.session_id or "").strip())
    if task is None or session is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    data = invariants_store.normalize_analysis(_last_analysis(session["dialog"]))
    suggestions = list(data["suggestions"])
    index = int(payload.index or 0)
    if index < 0 or index >= len(suggestions):
        raise HTTPException(status_code=404, detail="Вариант не найден — обновите сообщение")
    text = str(suggestions[index].get("send") or "").strip()
    if not text:
        raise HTTPException(status_code=404, detail="Вариант не найден — обновите сообщение")
    return {"action": "send", "text": text, "resume": False}


def _last_analysis(dialog: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Последний разбор инвариантов из журнала чата задачи (None — разбора нет)."""
    for item in reversed((dialog or {}).get("log") or []):
        if item.get("kind") == workspace_store.LOG_SUGGESTIONS and item.get("analysis"):
            return dict(item["analysis"])
    return None


async def invariants_get(session_id: str = "") -> dict:
    """Инварианты проекта и задачи для модалки (шестерёнки ⚙️).

    Отдаёт {"project": [{"id", "text", "created"}, ...], "task": [...],
    "pairs": N, "counts": {"project", "task"}, "project_id": ..., "task_id": ...}.
    Проверок пар здесь НЕТ: правила записываются без обращений к модели, а
    нарушения и противоречия выясняются в диалоге разбором запроса (§5.10).

    session_id задан — снимок по КОНКРЕТНОЙ задаче (шестерёнка в списке задач:
    правила правятся, даже если диалог не открыт). Чужая/неизвестная задача —
    404, как и у прочих маршрутов задач.
    """
    if session_id:
        task, session = _find_session_anywhere(session_id)
        if task is None or session is None:
            raise HTTPException(status_code=404, detail="Задача не найдена")
        return _invariants_view(task, session)
    task = _current_task()
    session = workspace_store.active_session(_workspace, task) if task else None
    return _invariants_view(task, session)


async def invariant_create(payload: InvariantCreate) -> dict:
    """Добавляет инвариант (одно поле — один инвариант) и проверяет противоречия.

    scope="project" — правило всего проекта (все его задачи-диалоги),
    scope="task" — правило конкретной задачи-диалога. После добавления пары
    «инвариант проекта × инвариант задачи» проверяются служебным вызовом LLM:
    противоречие пользователь видит сразу и решает, какое правило главнее
    (см. POST /api/agent/invariants/conflicts/resolve).
    """
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Нечего добавлять: инвариант пуст")
    session_id = str(payload.session_id or "").strip()
    task, session = _invariants_session(session_id)
    if task is None:
        raise HTTPException(
            status_code=400,
            detail="Сначала создайте проект — инварианты привязаны к проекту",
        )
    if payload.scope == "task" and session is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    async with _workspace_lock:
        container = task if payload.scope == "project" else session["dialog"]
        workspace_store.add_invariant(container, text)
        # Никаких обращений к модели при записи правил: инвариант — это данные.
        # Противоречия и нарушения выясняются в диалоге (разбор запроса).
        await _persist()
        return _invariants_view(task, session)


async def invariant_delete(scope: str, invariant_id: str,
                           session_id: str = "") -> dict:
    """Удаляет инвариант (корзина рядом с правилом в модалке «Инварианты»).

    Обращений к модели нет: правило — это данные. Разбор запроса сбрасывается —
    правила изменились, следующий запрос сверяется заново.
    """
    scope = str(scope or "").strip().lower()
    if scope not in invariants_store.SCOPES:
        raise HTTPException(status_code=400, detail="Неизвестная область инварианта")
    task, session = _invariants_session(session_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Проект не найден")
    if scope == "task" and session is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    async with _workspace_lock:
        container = task if scope == "project" else session["dialog"]
        if not workspace_store.delete_invariant(container, invariant_id):
            raise HTTPException(status_code=404, detail="Инвариант не найден")
        # Правила изменились — сохранённый разбор запроса больше не актуален.
        dialog = session["dialog"]
        dialog["analysis"] = dict(invariants_store.empty_analysis())
        await _persist()
        return _invariants_view(task, session)


async def invariant_resolve(payload: InvariantResolve) -> dict:
    """Устаревший маршрут решения противоречия (оставлен для совместимости).

    Приоритет всегда у правила ПРОЕКТА, поэтому «главнее задача» больше не
    принимается: такой выбор означал бы работу вопреки правилу проекта. Ответ —
    409 с объяснением; интерфейс этот маршрут не использует (конфликт правила
    задачи с правилом проекта — обычное нарушение с альтернативами).
    """
    task, session = _invariants_session()
    if task is None or session is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    if str(payload.winner or "").strip().lower() == invariants_store.WINNER_TASK:
        # «Главнее задача» означало бы работу вопреки правилу проекта.
        raise HTTPException(
            status_code=409,
            detail="Инвариант проекта всегда главнее: выберите один из предложенных "
                   "вариантов, которые правила не нарушают",
        )
    # Правило проекта и так в силе — состояние не меняем, просто сбрасываем разбор.
    async with _workspace_lock:
        session["dialog"]["analysis"] = dict(invariants_store.empty_analysis())
        await _persist()
        return _invariants_view(task, session)


@router.get("/agent/invariants")
async def invariants_get(session_id: str = "") -> dict:
    """Инварианты проекта и задачи + проверки на противоречие.

    Отдаёт {"project": [{"id", "text", "created"}, ...],
    "task": [...], "pairs": N, "checks": [...], "has_conflict": bool,
    "counts": {"project", "task", "conflict"},
    "project_id": ..., "task_id": ...} — этим снимком живут шестерёнки
    инвариантов (у проекта и у каждой задачи) и сама модалка.

    session_id задан — снимок по КОНКРЕТНОЙ задаче (шестерёнка в списке задач:
    правила правятся, даже если диалог не открыт). Чужая/неизвестная задача —
    404, как и у прочих маршрутов задач.
    """
    if session_id:
        task, session = _find_session_anywhere(session_id)
        if task is None or session is None:
            raise HTTPException(status_code=404, detail="Задача не найдена")
        return _invariants_view(task, session)
    task = _current_task()
    session = workspace_store.active_session(_workspace, task) if task else None
    return _invariants_view(task, session)


@router.post("/agent/invariants")
async def invariant_create(payload: InvariantCreate) -> dict:
    """Добавляет инвариант (одно поле — один инвариант) и проверяет противоречия.

    scope="project" — правило всего проекта (все его задачи-диалоги),
    scope="task" — правило конкретной задачи-диалога. После добавления пары
    «инвариант проекта × инвариант задачи» проверяются служебным вызовом LLM:
    противоречие пользователь видит сразу и решает, какое правило главнее
    (см. POST /api/agent/invariants/conflicts/resolve).
    """
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Нечего добавлять: инвариант пуст")
    session_id = str(payload.session_id or "").strip()
    task, session = _invariants_session(session_id)
    if task is None:
        raise HTTPException(
            status_code=400,
            detail="Сначала создайте проект — инварианты привязаны к проекту",
        )
    if payload.scope == "task" and session is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    async with _workspace_lock:
        container = task if payload.scope == "project" else session["dialog"]
        workspace_store.add_invariant(container, text)
        # Никаких обращений к модели при записи правил: инвариант — это данные.
        # Противоречия и нарушения выясняются в диалоге (разбор запроса).
        await _persist()
        return _invariants_view(task, session)


@router.delete("/agent/invariants/{scope}/{invariant_id}")
async def invariant_delete(scope: str, invariant_id: str,
                           session_id: str = "") -> dict:
    """Удаляет инвариант (корзина рядом с правилом в модалке «Инварианты»).

    Обращений к модели нет: правило — это данные. Разбор запроса сбрасывается —
    правила изменились, следующий запрос сверяется заново.
    """
    scope = str(scope or "").strip().lower()
    if scope not in invariants_store.SCOPES:
        raise HTTPException(status_code=400, detail="Неизвестная область инварианта")
    task, session = _invariants_session(session_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Проект не найден")
    if scope == "task" and session is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    async with _workspace_lock:
        container = task if scope == "project" else session["dialog"]
        if not workspace_store.delete_invariant(container, invariant_id):
            raise HTTPException(status_code=404, detail="Инвариант не найден")
        # Правила изменились — сохранённый разбор запроса больше не актуален.
        dialog = session["dialog"]
        dialog["analysis"] = dict(invariants_store.empty_analysis())
        await _persist()
        return _invariants_view(task, session)


@router.post("/agent/invariants/conflicts/resolve")
async def invariant_resolve(payload: InvariantResolve) -> dict:
    """Устаревший маршрут решения противоречия (только «главнее проект»).

    Приоритет ВСЕГДА у правила проекта, поэтому выбор «главнее задача» не
    принимается (409): он означал бы работу вопреки правилу проекта. Интерфейс
    этот маршрут не использует — конфликт правила задачи с правилом проекта
    обрабатывается как обычное нарушение с альтернативами (см. §5.10).
    """
    task = _current_task()
    session = workspace_store.active_session(_workspace, task) if task else None
    if task is None or session is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    if str(payload.winner or "").strip().lower() == invariants_store.WINNER_TASK:
        raise HTTPException(
            status_code=409,
            detail="Инвариант проекта всегда главнее: выберите один из предложенных "
                   "вариантов, которые правила не нарушают",
        )
    # Правило проекта и так в силе — состояние не меняем, сбрасываем только разбор.
    async with _workspace_lock:
        session["dialog"]["analysis"] = dict(invariants_store.empty_analysis())
        await _persist()
        return _invariants_view(task, session)


# ---------------------------------------------------------------------------
# Профиль пользователя: сведения о юзере уходят в системный промпт сессии
# ---------------------------------------------------------------------------
def _profile_fields(payload: ProfileFields) -> Dict[str, Any]:
    """Значения полей профиля из запроса (обновляются только переданные ключи).

    Это название профиля (profile_name — только для интерфейса) и пять полей
    сведений о пользователе, которые уходят в системный промпт сессии.
    """
    data = payload.model_dump(exclude_unset=True) if hasattr(payload, "model_dump") \
        else payload.dict(exclude_unset=True)
    return {key: data[key] for key in profile_store.EDITABLE_KEYS if key in data}


@router.get("/agent/profiles")
async def profiles_get() -> dict:
    """Профили пользователя для меню профиля (иконка рядом с заголовком панели).

    Отдаёт {"active": id|None,
    "profiles": [{"id", "profile_name", "label", "user_name", "occupation",
    "style", "answer_format", "limits", "created"}, ...],
    "profile": {...}|None} —
    текущий профиль и список всех профилей ("label" — название профиля; id в
    списке не показывается). Пять полей сведений о пользователе (имя, род
    деятельности, стиль общения, формат ответа, ограничения) уходят в системный
    промпт сессии режима «AI-агент» (см. POST /api/agent/chat), название профиля
    — только для интерфейса. Профилей нет — сервер создаёт профиль с
    автоматическими id «user_<цифры>» и названием «user<цифры>».
    """
    # Блокировку не берём: снимок — чистая операция без await, а панель профиля
    # не должна ждать, пока агент отвечает (он держит блокировку своей задачи).
    if not _profiles.get("profiles"):
        profile_store.ensure_profile(_profiles)
        await _persist_profiles()
    return profile_store.snapshot(_profiles)


@router.post("/agent/profiles")
async def profile_create(payload: ProfileCreate) -> dict:
    """Создаёт профиль (кнопка «Создать профиль») и делает его текущим.

    Название профиля (profile_name) ОБЯЗАТЕЛЬНО — пустое → 400: по нему профиль
    выбирается в списке. Идентификатор генерирует сервер: «user_<цифры>» (напр.
    user_13213123), цифры — номер конкретного профиля. Поля сведений о
    пользователе можно заполнить и позже.
    """
    fields = _profile_fields(payload)
    if not str(fields.get(profile_store.NAME_KEY) or "").strip():
        raise HTTPException(status_code=400, detail="Введите название профиля")
    async with _workspace_lock:
        profile_store.create_profile(_profiles, fields)
        await _persist_profiles()
        return profile_store.snapshot(_profiles)


@router.put("/agent/profiles/{profile_id}")
async def profile_update(profile_id: str, payload: ProfileFields) -> dict:
    """Сохраняет поля профиля (кнопка «Сохранить» в меню профиля).

    Обновляются только переданные поля: название профиля (пустое игнорируется —
    название обязательно) и пять полей сведений о пользователе (пустая строка
    очищает поле). Пустые поля в системный промпт не попадают — если профиль не
    заполнен целиком, агент работает как раньше, без блока профиля.
    """
    async with _workspace_lock:
        profile = profile_store.find_profile(_profiles, profile_id)
        if profile is None:
            raise HTTPException(status_code=404, detail="Профиль не найден")
        fields = _profile_fields(payload)
        if fields:
            profile_store.apply_fields(profile, fields)
            await _persist_profiles()
        return profile_store.snapshot(_profiles)


@router.delete("/agent/profiles/{profile_id}")
async def profile_delete(profile_id: str) -> dict:
    """Удаляет профиль (корзина в меню профиля).

    Если удалён текущий профиль, текущим становится соседний; когда профилей не
    осталось, создаётся новый пустой с идентификатором «user_<цифры>» — чтобы
    диалог в режиме агента всегда был с профилем. Данные профиля удаляются ВМЕСТЕ
    с ним: его задачи со всеми диалогами, рабочая и долговременная память.
    """
    async with _workspace_lock:
        if not profile_store.delete_profile(_profiles, profile_id):
            raise HTTPException(status_code=404, detail="Профиль не найден")
        profile_store.ensure_profile(_profiles)
        await _persist_profiles()
        # Данные профиля уходят вместе с ним: его задачи с диалогами, рабочая и
        # долговременная память. Иначе в файле оставался бы мусор, который уже
        # никому не виден и не удаляется через интерфейс.
        workspace_store.purge_profile(_workspace, profile_id)
        await _persist()
        return profile_store.snapshot(_profiles)


@router.post("/agent/profiles/{profile_id}/select")
async def profile_select(profile_id: str) -> dict:
    """Переключает текущий профиль (выпадающий список в меню профиля).

    Профили изолированы: вместе с профилем меняются его задачи, диалоги,
    рабочая и долговременная память, а системный промпт сессии собирается заново
    (см. POST /api/agent/chat). Ответ — снимок профилей; фронт после него
    перечитывает снимок workspace, и панель показывает задачи нового профиля.
    """
    async with _workspace_lock:
        profile = profile_store.find_profile(_profiles, profile_id)
        if profile is None:
            raise HTTPException(status_code=404, detail="Профиль не найден")
        _profiles["active"] = profile["id"]
        await _persist_profiles()
        return profile_store.snapshot(_profiles)


@router.post("/agent/chat")
async def agent_chat(msg: ChatMessage) -> StreamingResponse:
    """Режим «AI-агент»: потоковый ответ NDJSON через конечный автомат задачи.

    Каждая строка ответа — JSON-событие агента:
      {"type": "debug", "text": "чем агент сейчас занят"} — отдельное
        сообщение в чате (выводится по мере появления);
      {"type": "state", "state": {...}} — СОСТОЯНИЕ АВТОМАТА ЗАДАЧИ (этап,
        текущий шаг, ожидаемое действие, шаги плана, пауза): по нему фронт
        рисует полосу этапов над окном чата;
      {"type": "branches", "analysis": {...}, "active": "A"} — план ветвления
        (стратегия «branching»): из него фронтенд рисует блок плана и список
        веток для переключения;
      {"type": "bot", "text": "..."} — ответ модели, а на этапе planning —
        показанный пользователю план задачи;
      {"type": "error", "text": "..."} — понятное сообщение об ошибке;
      {"type": "done", "usage": {...}} — служебный конец потока.

    Запрос может быть не от пользователя, а от самого автомата: поле
    `continue_step=true` (пустой content) означает «выполни текущий шаг плана» —
    так интерфейс после подтверждения плана и между шагами ведёт задачу сам, без
    «придумайте сообщение». Такая реплика сохраняется в диалоге с пометкой
    `source: "machine"` и рисуется служебным сообщением.

    Этапы автомата (см. app/ai/task_state.py) ведёт ЭТОТ маршрут:
      planning   — запрос разбивается на шаги служебным вызовом LLM
                   (Agent.build_plan); план показывается пользователю, и задача
                   ждёт «ок» (planning → awaiting_user). Режим «работай
                   автономно» подтверждения не требует (planning → execution);
      execution  — выполняется ТЕКУЩИЙ шаг плана: он уходит агенту системным
                   блоком состояния, ответ модели — результат шага. Шаг
                   выполнен → следующий шаг плана (execution), а после
                   последнего — проверка (execution → validation);
      validation — самопроверка результата: пройдена → done, не пройдена →
                   возврат в execution на нужный шаг (validation → execution);
      failed     — шаг не выполнен (модель не дала ответа); следующее сообщение
                   пользователя перезапускает задачу (failed → planning);
      awaiting_user/failed → planning — по сообщению пользователя («ок» или
                   правки плана);
      пауза      — этап и текущий шаг НЕ меняются, а запросы к агенту
                   отклоняются, пока не нажата кнопка «Продолжить»
                   (см. POST /api/agent/state/pause|resume).

    Запрос обрабатывается в диалоге ТЕКУЩЕЙ сессии текущей задачи: её память и
    состояние стратегий уходят агенту на вход и обновляются по завершении
    обмена. Без созданной задачи диалог начать нельзя — 400 (фронт показывает
    подсказку «сначала создайте задачу»). Если у задачи ещё нет ни одного
    диалога, сессия создаётся автоматически под первый запрос.

    Агент берёт из сообщения длину (max_tokens) и условие завершения (stop),
    стратегию работы с контекстом (agent_strategy) с её полями (summary/window)
    и выбранную ветку плана (branch). Плюс системные блоки сессии: профиль
    пользователя (см. /api/agent/profiles) и состояние задачи.
    """
    # Формат ответа агенту не передаём: системные блоки сессии — это профиль
    # пользователя и состояние задачи (см. ниже), а сам запрос уходит в модель
    # как есть (см. app/ai/agent.py).
    # Задача запроса: обычно ТЕКУЩАЯ, но авто-прогон может назвать её явно
    # (session_id) — так задача продолжает выполняться ФОНОМ, пока пользователь
    # смотрит другую. Чужую задачу (другой профиль) взять нельзя.
    explicit_session = str(getattr(msg, "session_id", "") or "").strip()
    if explicit_session:
        task, session = _find_session_anywhere(explicit_session)
        if task is None or session is None:
            return JSONResponse({"detail": "Задача не найдена"}, status_code=404)
    else:
        task = _current_task()
        if task is None:
            return JSONResponse({"detail": "Сначала создайте проект"}, status_code=400)
        session = workspace_store.active_session(_workspace, task)
        if session is None:
            session = workspace_store.create_session(task)
    # Слои памяти пользователя (memory layers): рабочая память задачи и
    # долговременная (глобальная) база знаний. Это снимки — агент получает их
    # на каждый запрос и кладёт в контекст ВСЕГДА, независимо от стратегии.
    working_memory = workspace_store.memory_texts(task, workspace_store.MEMORY_WORKING)
    # Долговременная память — база знаний ТЕКУЩЕГО профиля: профили изолированы.
    long_term_memory = workspace_store.long_term_texts(_workspace, _current_profile_id())
    # Профиль пользователя — системный промпт СЕССИИ: сведения о юзере (имя,
    # род деятельности, стиль общения, формат ответа, ограничения) уходят в
    # модель первым системным блоком; собирается заново на каждый запрос,
    # поэтому смена профиля действует сразу.
    profile = _profile_block()
    # Настройки генерации общие для ответа и для служебного плана.
    agent_settings = dict(
        max_tokens=msg.max_tokens,
        stop=msg.stop,
        strategy=msg.agent_strategy,
        summary_size=msg.summary or DEFAULT_SUMMARY_SIZE,
        window_size=msg.window or DEFAULT_WINDOW_SIZE,
    )
    # Инварианты (правила, которые агент не имеет права нарушить): снимок
    # проекта и диалога + утверждённые исключения. Собирается ДО потока:
    # значения читаются генератором при каждом шаге, а диалог за время ответа
    # мог быть переключён.
    invariants_now = _invariants_snapshot(task, session)
    agent = Agent(AgentConfig(**agent_settings))
    # Планировщик — отдельный агент с теми же настройками: он делает служебный
    # вызов плана (Agent.build_plan), и его токены складываются с токенами
    # ответа (merge_usage), попадая в панель «Токены диалога».
    planner = Agent(AgentConfig(**agent_settings))
    # Приёмщик — ещё один агент для содержательной проверки результата на этапе
    # validation (Agent.review_result): один служебный вызов на завершённую
    # задачу, токены так же складываются в общий замер запроса.
    reviewer = Agent(AgentConfig(**agent_settings))
    # Арбитр инвариантов — ещё один агент для служебного разбора ЗАПРОСА до
    # планирования (Agent.check_invariants): нарушает ли запрос правило проекта
    # или задачи и не противоречат ли правила друг другу. Его токены тоже
    # складываются в общий замер запроса.
    analyzer = Agent(AgentConfig(**agent_settings))

    # Диалог текущей сессии для журнала чата: заполняется внутри потока, когда
    # сессия разрешена под блокировкой. Через него encode() пишет в журнал все
    # события (иначе пришлось бы логировать каждую точку yield).
    log_target: Dict[str, Any] = {}

    def encode(event: dict) -> str:
        _log_event(log_target.get("dialog"), event)
        return json.dumps(event, ensure_ascii=False) + "\n"

    async def event_stream():
        # Переменные объявлены заранее: в аварийной ветке (except) нужно
        # сохранить состояние и закрыть поток событием "done".
        state: Optional[task_state.TaskState] = None
        session_now: Optional[Dict[str, Any]] = None
        try:
            async with _session_lock(session["id"]):
                # Сессию, диалог и состояние берём ПОД блокировкой СВОЕЙ задачи:
                # между
                # подготовкой ответа и стримом задачу могли переключить.
                if explicit_session:
                    # Фоновый шаг: задача задана явно — текущую не подставляем.
                    task_now, found = _find_session_anywhere(explicit_session)
                    session_now = found or session
                    task_now = task_now or task
                else:
                    task_now = _current_task() or task
                    session_now = workspace_store.active_session(_workspace, task_now) or session
                dialog_now: Dict[str, Any] = session_now["dialog"]
                state = workspace_store.dialog_state(session_now)
                if not state.task_id:
                    state.task_id = session_now["id"]
                # Все события этого ответа попадут в журнал чата сессии — по нему
                # окно чата восстанавливается при переключении диалога.
                log_target["dialog"] = dialog_now
                _running_sessions.add(str(session_now["id"]))
                # «Пауза»/«Отменить», нажатые в предыдущем шаге, могли не успеть
                # примениться (блокировку держал поток) — применяем до работы.
                _apply_pending_stop(session_now, state)
                text = (msg.content or "").strip()
                # Запрос от АВТОМАТА (continue_step): пользователь ничего не
                # писал — выполняется текущий шаг плана. Так интерфейс ведёт
                # задачу сам (после «Подтвердить план» и между шагами), и
                # пользователю не нужно придумывать сообщение ради шага.
                machine_step = bool(getattr(msg, "continue_step", False))
                # Команда остановки, полученная во время текущего шага (см. ниже).
                stopped: Optional[str] = None
                # Проверку отложили в ЭТОМ ЖЕ запросе («Пауза» на последнем шаге):
                # выполнять её сразу нельзя — задача должна остаться на этапе
                # validation и ждать «Продолжить».
                validation_deferred_here = False
                if text and not machine_step:
                    # Реплика пользователя — в журнал чата: в память диалога она
                    # попадает только вместе с ответом, а при построении плана
                    # ответа нет, и текст запроса терялся (в восстановленном
                    # диалоге его не было видно).
                    workspace_store.add_log(dialog_now, workspace_store.LOG_USER, text)
                plan_usage: Dict[str, Any] = {}
                # Расход текущего запроса: заполняется ответом шага и служебными
                # вызовами (план/проверка). Объявлен заранее — помощник проверки
                # обращается к нему и в ветке ОТЛОЖЕННОЙ проверки, где шага не было.
                usage: Dict[str, Any] = {}
                # Результат шага этого запроса (в ветке ОТЛОЖЕННОЙ проверки шага
                # не было — значения остаются пустыми): нужны и переходам, и
                # записи расхода в хвосте потока.
                answered = False
                fallback_answer = False
                errors: List[str] = []
                stored = False

                # 1. ПАУЗА. Автомат остановлен кнопкой «Пауза»: шаг не
                #    выполняем, просим нажать «Продолжить».
                if state.paused:
                    yield encode(_state_event(state))
                    yield encode({"type": "error", "text": (
                        "Задача на паузе: нажмите «Продолжить» в полосе состояния, "
                        "чтобы продолжить работу."
                    )})
                    yield encode({"type": "done", "usage": {}, "state": task_state.snapshot(state)})
                    return

                if not text and not machine_step:
                    yield encode({"type": "bot", "text": "Пожалуйста, введите сообщение."})
                    yield encode({"type": "done", "usage": {}, "state": task_state.snapshot(state)})
                    return
                if machine_step and state.stage not in ("execution", "validation"):
                    # Автомат сам приходит только за шагом (execution) или за
                    # ОТЛОЖЕННОЙ проверкой результата (validation). В остальных
                    # этапах он ждёт пользователя: план, ошибка, завершение.
                    yield encode(_state_event(state))
                    yield encode({"type": "error", "text": (
                        "Работать нечего: задача на этапе "
                        f"«{task_state.STAGE_LABELS.get(state.stage, state.stage)}». "
                        "Отправьте сообщение или подтвердите план."
                    )})
                    yield encode({"type": "done", "usage": {}, "state": task_state.snapshot(state)})
                    return

                # 1а. ИНВАРИАНТЫ. Правило проекта и правило задачи могут
                #     противоречить друг другу: выбор («главнее проект» или
                #     «главнее задача») делает пользователь, и пока решения нет,
                #     агент не работает — иначе он молча нарушил бы одно из
                #     правил. Проверка и решение — в модалке «Инварианты».
                # 2. ЭТАП: смотрим, где задача, и что означает это сообщение.
                confirmed = _is_plan_confirmation(text)
                autonomous = _wants_autonomous(text)
                restart = _wants_restart(text)

                if state.stage in ("done", "cancelled"):
                    # Предыдущая задача завершена — это НОВАЯ задача: автомат
                    # рождается заново (planning) с записью о сбросе в истории.
                    state = task_state.reset(
                        session_now["id"], state,
                        "предыдущая задача завершена — начинаю новую по новому запросу",
                        autonomous=autonomous,
                    )
                if state.stage == "failed":
                    task_state.start_planning(
                        state,
                        "пользователь перезапустил задачу после ошибки"
                        + (" (сообщение «перезапусти»)" if restart else ""),
                    )
                # ВАЖНО: этап validation в начале запроса — это ОТЛОЖЕННАЯ
                # проверка (пауза была на последнем шаге), а не «зависшая»:
                # её выполняет ветка ниже, поэтому конвертировать validation в
                # execution здесь нельзя.
                if state.stage == "awaiting_user":
                    task_state.start_planning(
                        state,
                        "пользователь ответил на вопрос о плане"
                        + (" — план подтверждён" if confirmed else " — вношу правки"),
                    )
                if autonomous:
                    state.autonomous = True
                yield encode(_state_event(state))

                # 1б. ИНВАРИАНТЫ И ЗАПРОС. Самое первое, что делает агент, —
                #     сверяет ЗАПРОС пользователя с правилами, которые нарушать
                #     нельзя (правила проекта + правила задачи), ДО планирования.
                #     Нарушение требования или противоречие правил — агент
                #     ОТКАЗЫВАЕТСЯ работать по этому запросу и показывает варианты
                #     решения: варианты ПРОВЕРЕНЫ по правилам (нарушающие
                #     отброшены, см. invariants_store.check_suggestions), они
                #     кликабельные (событие suggestions, рисуется под сообщением).
                #     Плана в этом случае нет — задача остаётся на этапе
                #     планирования.
                if text and not machine_step and not confirmed and not autonomous:
                    yield encode({"type": "debug", "text": (
                        f"{_MACHINE}: сверяю запрос с инвариантами (правила проекта "
                        "и задачи) до планирования."
                    )})
                    analysis, pre_usage, preverified = await _preflight_invariants(
                        task_now, session_now, text, analyzer, invariants_now)
                    if pre_usage:
                        usage = merge_usage(usage, pre_usage)
                    if preverified:
                        # Текст — вариант, который агент сам показал и проверил по
                        # правилам: повторный разбор не нужен (и отказа быть не
                        # может, иначе выбор варианта зацикливал бы пользователя).
                        yield encode({"type": "debug", "text": (
                            f"{_MACHINE}: запрос — проверенный вариант-альтернатива "
                            "(правила не менялись): разбор не нужен, строю план."
                        )})
                    if invariants_store.blocks(analysis):
                        yield encode({"type": "debug", "text": (
                            f"{_MACHINE}: " + _analysis_debug(analysis)
                        )})
                        # Варианты и объяснение — в журнал чата: по нему окно
                        # восстанавливается вместе с кликабельными вариантами.
                        view = _analysis_view(analysis)
                        workspace_store.add_log_event(
                            dialog_now, workspace_store.LOG_SUGGESTIONS,
                            view["message"], view)
                        yield encode({"type": "suggestions", "analysis": view,
                                      "text": view["message"]})
                        # Задача ждёт выбора: план не строим, шаги не выполняем.
                        if state.stage == "planning" and not state.steps:
                            task_state.await_confirmation(
                                state, [],
                                "запрос нарушает инварианты — жду решения пользователя")
                        yield encode(_state_event(state))
                        yield encode({"type": "done", "usage": dict(usage),
                                      "state": task_state.snapshot(state)})
                        return

                async def run_validation(answered_step: bool, step_errors: List[str],
                                         stored_exchange: bool, resumed: bool):
                    """Проверка результата: локальная самопроверка + содержательная
                    проверка моделью (события отдаём в поток).

                    Вызывается дважды: сразу после последнего шага и при ОТЛОЖЕННОЙ
                    проверке (пользователь поставил «Паузу» на последнем шаге —
                    задача остановилась на этапе validation, проверку выполняем
                    после «Продолжить»). Возвращает расход вызова проверки.
                    """
                    nonlocal usage
                    if not resumed:
                        task_state.to_validation(
                            state, "все шаги плана выполнены — проверяю результат")
                        yield encode(_state_event(state))
                    yield encode({"type": "debug", "text": (
                        f"{_MACHINE}: этап validation — самопроверка полученного ответа "
                        f"({state.steps_total or 1} "
                        f"{task_state.steps_word(state.steps_total or 1)}, "
                        + ("ответ и обмен взяты из диалога)." if resumed
                           else "ответ получен, обмен сохранён).")
                    )})
                    ok, note = _self_check(
                        answered=answered_step,
                        errors=step_errors,
                        steps_total=state.steps_total,
                        steps_done=state.step_index + 1,
                        stored=stored_exchange,
                    )
                    redo_step = state.step_index
                    if stopped is not None:
                        # Пользователь просил остановиться: служебный вызов
                        # проверки не делаем (это лишний вызов LLM после «стоп»).
                        yield encode({"type": "debug", "text": (
                            f"{_MACHINE}: поступила команда остановки — содержательную "
                            "проверку результата пропускаю (это лишний служебный вызов)."
                        )})
                    elif ok:
                        yield encode({"type": "debug", "text": (
                            f"{_MACHINE}: самопроверка пройдена — отправляю результат на "
                            "содержательную проверку (служебный вызов LLM)."
                        )})
                        review = await reviewer.review_result(
                            state.request or _task_request(dialog_now["messages"], text),
                            plan=list(state.steps),
                            history=dialog_now["messages"],
                            working_memory=working_memory,
                            long_term_memory=long_term_memory,
                            profile=profile,
                            invariants=_invariants_snapshot(task_now, session_now),
                        )
                        if reviewer.last_usage:
                            usage = merge_usage(usage, reviewer.last_usage)
                            # Расход запроса вырос (проверка — служебный вызов):
                            # сообщаем интерфейсу ОБНОВЛЁННЫЙ замер.
                            yield encode({"type": "usage", "usage": dict(usage)})
                        if review is None:
                            yield encode({"type": "debug", "text": (
                                f"{_MACHINE}: содержательная проверка не получена "
                                "(модель не ответила) — опираюсь на самопроверку."
                            )})
                        elif review["ok"]:
                            note = f"{note}; модель: {review['comment']}"
                            yield encode({"type": "debug", "text": (
                                f"{_MACHINE}: содержательная проверка пройдена — "
                                f"{review['comment']}."
                            )})
                        else:
                            ok = False
                            note = f"модель не приняла результат: {review['comment']}"
                            if review["step"]:
                                redo_step = review["step"] - 1
                    if ok:
                        task_state.validation_ok(state, f"проверка пройдена: {note}")
                        yield encode({"type": "debug", "text": (
                            f"{_MACHINE}: проверка пройдена ({note}) — этап done, задача готова."
                        )})
                    elif task_state.can_redo(state):
                        task_state.validation_failed(
                            state, f"проверка не пройдена: {note}", redo_step)
                        yield encode({"type": "error", "text": (
                            f"⚠️ Проверка не пройдена: {note}. Возвращаю задачу на "
                            f"{state.step_label()} — автомат доработает его"
                            + (f" (доработок: {state.redo_count} из {task_state.MAX_REDO})."
                               if state.redo_count else ".")
                        )})
                    else:
                        task_state.fail(
                            state,
                            f"проверка не приняла результат после {state.redo_count} "
                            f"{task_state.steps_word(state.redo_count)} доработок: {note}",
                        )
                        yield encode({"type": "error", "text": (
                            f"⚠️ Проверка не приняла результат после {state.redo_count} доработок "
                            f"({note}). Задача переведена в этап «ошибка» — уточните запрос "
                            "или поправьте план, и отправьте сообщение."
                        )})

                if state.stage == "planning":
                    # 3. PLANNING: план задачи. Подтверждённый план уже есть —
                    #    строить заново нечего; иначе спрашиваем модель.
                    if confirmed and state.steps:
                        task_state.plan_ready(state, state.steps, "план подтверждён пользователем")
                        yield encode({"type": "debug", "text": (
                            f"{_MACHINE}: план подтверждён — этап planning → execution, "
                            f"{state.step_label()}."
                        )})
                    else:
                        yield encode({"type": "debug", "text": (
                            f"{_MACHINE}: этап planning — разбиваю запрос на шаги "
                            "(служебный вызов LLM)."
                        )})
                        steps = await planner.build_plan(
                            text,
                            history=dialog_now["messages"],
                            working_memory=working_memory,
                            long_term_memory=long_term_memory,
                            profile=profile,
                            # Снимок берём СВЕЖИЙ: разбор запроса уже мог
                            # пометить правило задачи недействующим (см.
                            # `overridden` в _invariants_snapshot), и планировщик
                            # обязан это видеть — иначе он строит план по
                            # требованию задачи в обход правила проекта.
                            invariants=_invariants_snapshot(task_now, session_now),
                        )
                        plan_usage = dict(planner.last_usage or {})

                        # КОД-ГЕЙТ ПЛАНА. Промпт-блок правил — просьба; гарантию
                        # даёт проверка шагов арбитром: нарушающие шаги в работу
                        # не уходят, при нарушении — одна попытка перепланирования.
                        async def replan(bad_steps: List[str]) -> Any:
                            """Перепланирование с пометкой «эти шаги повторять нельзя»."""
                            fresh = await planner.build_plan(
                                text,
                                history=dialog_now["messages"],
                                working_memory=working_memory,
                                long_term_memory=long_term_memory,
                                profile=profile,
                                invariants=_invariants_snapshot(task_now, session_now),
                                note=(
                                    "ПРОВЕРКА ПО ПРАВИЛАМ ОТКЛОНИЛА предыдущий план. "
                                    "Эти шаги нарушают правила, повторять их НЕЛЬЗЯ:\n"
                                    + "\n".join(f"- {step}" for step in bad_steps)
                                    + "\nДай ДРУГОЙ план (шаги), который не нарушает "
                                      "правила: запрещённое требование выполнять нельзя."
                                ),
                            )
                            return fresh, dict(planner.last_usage or {})

                        gate = await _plan_gate(
                            steps, planner, _invariants_snapshot(task_now, session_now),
                            replan)
                        # Расход проверки — в тот же замер, что и построение плана:
                        # обе траты этого запроса уходят в панель строкой
                        # «из них служебные вызовы» (kind="plan").
                        plan_usage = merge_usage(plan_usage, gate["usage"])
                        for line in gate["lines"]:
                            yield encode({"type": "debug", "text": f"{_MACHINE}: {line}"})
                        if gate["error"]:
                            # План не принят: шаги не выполняем, задача ждёт
                            # пользователя (переформулировать запрос/поправить
                            # правила), плана в состоянии нет.
                            yield encode({"type": "error", "text": gate["error"]})
                            if state.stage == "planning" and not state.steps:
                                task_state.await_confirmation(
                                    state, [],
                                    "план не прошёл проверку по инвариантам")
                            dialog_now.setdefault("usage", []).append(
                                dict(plan_usage, kind="plan"))
                            yield encode(_state_event(state))
                            yield encode({"type": "done", "usage": dict(plan_usage),
                                          "state": task_state.snapshot(state)})
                            return
                        usage = merge_usage(usage, gate["usage"])
                        steps = gate["steps"]
                        state.steps = steps
                        # Исходный запрос задачи — в состоянии: по нему проверка
                        # результата сверяет работу (последнее сообщение может
                        # быть подтверждением «ок» или служебной фразой шага).
                        if text and not machine_step:
                            state.request = text[:task_state.REQUEST_LIMIT]
                        steps_plural = task_state.steps_word(len(steps))
                        yield encode({"type": "debug", "text": (
                            f"{_MACHINE}: план готов — {len(steps)} {steps_plural}"
                            + (": " + "; ".join(steps) if steps else ".")
                        )})
                        plan_text = _plan_message(state, autonomous=state.autonomous)
                        # Запрос пользователя и показанный план — в память диалога
                        # сессии: иначе после переключения сессии/перезагрузки их
                        # не было бы видно в окне чата (память хранила только
                        # обмены «шаг → ответ»). Текст запроса нужен и модели как
                        # контекст, поэтому он идёт обычной репликой пользователя.
                        if text and not machine_step:
                            dialog_now.setdefault("messages", []).append(
                                {"role": "user", "content": text})
                        dialog_now.setdefault("messages", []).append(
                            {"role": "assistant", "content": plan_text})
                        if state.autonomous:
                            task_state.plan_ready(
                                state, steps,
                                "режим «работай автономно» — подтверждение плана не требуется",
                            )
                            yield encode({"type": "bot", "text": plan_text})
                        else:
                            task_state.await_confirmation(
                                state, steps, "план показан пользователю — жду «ок» или правок")
                            yield encode({"type": "bot", "text": plan_text})
                            yield encode(_state_event(state))
                            # «Пауза»/«Отменить», нажатые пока строился план:
                            # применяем СРАЗУ (это тот самый «этот этап», после
                            # которого пользователь просил остановиться) — иначе
                            # намерение висело бы до следующего запроса, а
                            # интерфейс показывал несогласованный набор кнопок.
                            pending_here = _pending_stops.pop(str(session_now["id"]), None)
                            if pending_here is not None \
                                    and _apply_pending_stop(session_now, state,
                                                            force=pending_here):
                                yield encode({"type": "debug", "text": (
                                    f"{_MACHINE}: во время построения плана нажали "
                                    + ("«Отменить»" if pending_here == PENDING_CANCEL
                                       else "«Пауза»")
                                    + " — план показан, задача остановлена."
                                )})
                                yield encode(_state_event(state))
                            if plan_usage:
                                # Расход на построение плана — отдельной записью
                                # (kind="plan"): ответа пользователю в этом
                                # запросе не было, но токены потрачены.
                                dialog_now.setdefault("usage", []).append(
                                    dict(plan_usage, kind="plan"))
                            workspace_store.set_dialog_state(session_now, state)
                            await _persist()
                            yield encode({"type": "done", "usage": plan_usage,
                                          "state": task_state.snapshot(state)})
                            return

                if state.stage == "execution":
                    # 4. EXECUTION: выполняем ТЕКУЩИЙ шаг плана (один шаг на
                    #    запрос пользователя). Шаг уходит агенту системным
                    #    блоком состояния — модель видит план и своё место в нём.
                    step_text = state.step_text() or text
                    # Что уйдёт в модель репликой пользователя: сообщение
                    # пользователя как есть либо (для авто-продолжения) текст
                    # текущего шага — план и место в нём модель видит ещё и в
                    # системном блоке состояния.
                    prompt = text or (
                        f"Продолжай по плану: {state.step_label()} — {step_text}."
                    )
                    yield encode({"type": "debug", "text": (
                        f"{_MACHINE}: этап execution, {state.step_label()} — «{step_text}». "
                        f"Ожидается: {state.expected_action}."
                    )})
                    answered = False
                    fallback_answer = False
                    errors = []
                    async for event in agent.stream_generate(
                        prompt,
                        history=dialog_now["messages"],
                        summary=dialog_now["summary"],
                        covered=dialog_now["covered"],
                        facts=dialog_now["facts"],
                        branches=dialog_now["branches"],
                        branch=msg.branch,
                        working_memory=working_memory,
                        long_term_memory=long_term_memory,
                        profile=profile,
                        state=state,
                        invariants=_invariants_snapshot(task_now, session_now),
                    ):
                        # Событие "done" несёт расход токенов текущего запроса.
                        kind = event.get("type")
                        if kind == "done" and isinstance(event.get("usage"), dict):
                            usage = merge_usage(plan_usage, event["usage"])
                        elif kind == "bot" and str(event.get("text") or "").strip():
                            answered = True
                            # Запасной ответ (модель не ответила) — шаг НЕ
                            # выполнен: execution → failed (см. ниже).
                            if event.get("fallback"):
                                fallback_answer = True
                        elif kind == "error":
                            note = str(event.get("text") or "")
                            # Предупреждение о лимите токенов — не провал шага.
                            if _LIMIT_WARNING_MARK not in note:
                                errors.append(note)
                        yield encode(event)
                    # Обмен завершён — запоминаем реплики и состояние стратегии
                    # (резюме и его границу, факты, ветви плана, активную ветку)
                    # В ДИАЛОГЕ ЭТОЙ СЕССИИ.
                    memory = list(agent.memory)
                    # Пометки source возвращаем ДО записи в диалог: агент хранит
                    # только role/content (см. _restore_message_sources).
                    _restore_message_sources(dialog_now["messages"], memory)
                    exchange = _exchange_memory(agent, msg.branch, memory)
                    stored = _exchange_stored(exchange, prompt)
                    if stored and machine_step:
                        # Реплика сгенерирована автоматом: в интерфейсе это
                        # служебное сообщение, а не запрос пользователя.
                        exchange[-2]["source"] = "machine"
                    dialog_now["messages"] = memory
                    dialog_now["summary"] = list(agent.summary)
                    dialog_now["covered"] = agent.covered
                    dialog_now["facts"] = dict(agent.facts)
                    dialog_now["branches"] = {
                        bid: dict(branch) for bid, branch in agent.branches.items()}
                    dialog_now["active_branch"] = agent.active_branch

                    # 4а. Отложенная остановка: пользователь нажал «Пауза» или
                    #     «Отменить» ПОКА шёл этот шаг. Ответ модели уже получен
                    #     и сейчас будет записан, поэтому сначала фиксируем
                    #     результат шага (переходы ниже), а остановку применяем
                    #     в самом конце — иначе вместе с переходом потерялся бы
                    #     и сам ответ.
                    stopped = _pending_stops.get(str(session_now["id"]))

                    # 5. Переход по результату шага (единственное место, где
                    #    меняется этап: шаг выполнен / не выполнен).
                    if not answered or fallback_answer or errors:
                        if not answered:
                            reason = f"шаг {state.step_number} не выполнен: ответа от модели нет"
                        elif fallback_answer:
                            reason = (f"шаг {state.step_number} не выполнен: модель не ответила, "
                                      "ответ отдан по запасному сценарию")
                        else:
                            reason = f"шаг {state.step_number} не выполнен: {errors[0]}"
                        task_state.fail(state, reason)
                        yield encode({"type": "error", "text": (
                            "⚠️ Шаг не выполнен: " + reason.split(": ", 1)[-1] + ". Задача переведена "
                            "в этап «ошибка» — отправьте сообщение, чтобы перезапустить её."
                        )})
                    elif state.step_number < state.steps_total:
                        task_state.next_step(
                            state,
                            f"шаг {state.step_number} выполнен — перехожу к шагу "
                            f"{state.step_number + 1}",
                        )
                        # Короткая строка без повтора шага: подробности («что
                        # делаем сейчас, что ожидается») даёт начало следующего
                        # шага и событие состояния — иначе в дебаге появлялись
                        # две почти одинаковые строки подряд.
                        yield encode({"type": "debug", "text": (
                            f"{_MACHINE}: шаг {state.step_number - 1} из "
                            f"{state.steps_total} выполнен."
                        )})
                    elif stopped == PENDING_PAUSE:
                        # Последний шаг выполнен, НО пользователь просил паузу:
                        # проверку не запускаем — переводим задачу на этап
                        # validation и останавливаемся. Проверка выполнится после
                        # «Продолжить» (см. ветку validation ниже).
                        task_state.to_validation(
                            state,
                            "шаг выполнен, но пользователь нажал «Пауза» — проверку отложил",
                        )
                        validation_deferred_here = True
                        yield encode(_state_event(state))
                        yield encode({"type": "debug", "text": (
                            f"{_MACHINE}: «Пауза» на последнем шаге — проверку результата "
                            "отложил: задача остановится НА ЭТАПЕ validation, проверка "
                            "выполнится после «Продолжить»."
                        )})
                    else:
                        # Последний шаг выполнен — проверяем результат (общий
                        # помощник: тот же путь используется для отложенной
                        # проверки после «Продолжить»).
                        async for chunk in run_validation(
                            answered and not fallback_answer, errors, stored, resumed=False
                        ):
                            yield chunk

                if state.stage == "validation" and not validation_deferred_here:
                    # ОТЛОЖЕННАЯ ПРОВЕРКА: пользователь поставил «Паузу» на
                    # последнем шаге, задача остановилась на этом этапе, а теперь
                    # её продолжили (запрос пришёл — значит не на паузе). Проверяем результат по сохранённому диалогу
                    # (ответ и обмен уже в нём) и завершаем задачу.
                    yield encode({"type": "debug", "text": (
                        f"{_MACHINE}: этап validation — выполняю ОТЛОЖЕННУЮ проверку "
                        "результата (пауза была на последнем шаге)."
                    )})
                    last_role = (dialog_now["messages"][-1].get("role")
                                 if dialog_now["messages"] else "")
                    async for chunk in run_validation(
                        last_role == "assistant", [], last_role == "assistant",
                        resumed=True,
                    ):
                        yield chunk

                # 6. Остановка, нажатая во время шага: шаг уже зафиксирован, а
                #    теперь останавливаем автомат (пауза) или отменяем задачу.
                if stopped is not None:
                    _pending_stops.pop(str(session_now["id"]), None)
                    if _apply_pending_stop(session_now, state, force=stopped):
                        yield encode({"type": "debug", "text": (
                            f"{_MACHINE}: во время выполнения шага нажали "
                            + ("«Отменить»" if stopped == PENDING_CANCEL else "«Пауза»")
                            + " — применил сразу после ответа."
                        )})
                    yield encode(_state_event(state))

                # Расход токенов запроса — ОДНОЙ записью на запрос пользователя,
                # уже с учётом служебных вызовов (план и проверка результата).
                if state.stage in ("execution", "validation", "done", "failed") \
                        and usage and stored:
                    dialog_now["usage"].append(dict(usage))
                    _usage_matches_history(dialog_now)

                # Состояние автомата — в диалог сессии и в файл workspace: этап,
                # шаг и история переходов переживают перезапуск приложения.
                workspace_store.set_dialog_state(session_now, state)
                await _persist()
                yield encode(_state_event(state))

                # Ещё раз проверяем отложенную команду: «Пауза»/«Отменить» могла
                # прийти, пока шаг дорабатывал и сохранялся (окно между чтением
                # намерения и концом потока). Без этого интерфейс оставался с
                # пометкой «остановлю после текущего шага», хотя шаг уже закончился.
                late = _pending_stops.get(str(session_now["id"]))
                if late is not None:
                    _pending_stops.pop(str(session_now["id"]), None)
                    if _apply_pending_stop(session_now, state, force=late):
                        workspace_store.set_dialog_state(session_now, state)
                        await _persist()
                        yield encode({"type": "debug", "text": (
                            f"{_MACHINE}: команда "
                            + ("«Отменить»" if late == PENDING_CANCEL else "«Пауза»")
                            + " пришла в самом конце шага — применил сразу."
                        )})
                    else:
                        # Этап терминальный (задача успела завершиться) — пауза
                        # уже не нужна: объясняем это, чтобы не выглядело сбоем.
                        yield encode({"type": "debug", "text": (
                            f"{_MACHINE}: команда "
                            + ("«Отменить»" if late == PENDING_CANCEL else "«Пауза»")
                            + " не потребовалась — задача уже завершена."
                        )})
                    yield encode(_state_event(state))
        except asyncio.CancelledError:
            raise  # клиент отключился — просто останавливаем поток
        except Exception:  # noqa: BLE001
            logger.exception("Режим AI-агента: ошибка обработки запроса")
            yield encode({"type": "error", "text": "Внутренняя ошибка агента. Попробуйте ещё раз."})
            # Аварийный выход: состояние (этап при этом НЕ меняем) стараемся
            # сохранить, а поток закрываем событием "done" — иначе автомат
            # остался бы в памяти без записи в файл.
            if state is not None and session_now is not None:
                try:
                    workspace_store.set_dialog_state(session_now, state)
                    await _persist()
                except Exception:  # noqa: BLE001
                    logger.warning("Не удалось сохранить состояние задачи", exc_info=True)
                yield encode({"type": "done", "usage": {},
                              "state": task_state.snapshot(state)})
        finally:
            # Пометку «задача выполняется» снимаем ВСЕГДА: иначе сбой в потоке
            # оставил бы её висеть и «Пауза» вечно откладывалась бы.
            if session_now is not None:
                _running_sessions.discard(str(session_now["id"]))

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@router.get("/agent/history")
async def agent_history() -> dict:
    """Сохранённый диалог ТЕКУЩЕЙ сессии режима «AI-агент».

    Отдаёт {"messages": [{"role": "user"|"assistant", "content": "..."}, ...],
    "usage": [{"requests", "input", "output", "limit", "overflow"}, ...],
    "summary": ["резюме старой части переписки", ...],
    "facts": {"ключ": "значение"},
    "branches": {"A": {"title", "approach", "messages", ...}},
    "active_branch": "A",
    "session": {"id", "title"}|None,
    "log": [{"kind": "user"|"assistant"|"debug"|"error", "text": "..."}, ...],
    "state": {"stage", "current_step", "expected_action", "steps", "paused", …}} —
    сообщения агент использует как память диалога, log — ЖУРНАЛ ЧАТА (что
    пользователь видел в окне: его реплики, ответы, показанный план и служебные
    debug/error-строки, в порядке появления), usage — расход токенов по
    запросам пользователя (панель «Токены диалога»), summary — резюме прежней
    переписки (стратегия «summary»), facts — блок фактов (стратегия «sticky
    facts»), branches/active_branch — ветви плана и активная ветка (стратегия
    «branching»; у каждой ветки своя история сообщений), state — состояние
    конечного автомата задачи этой сессии (по нему фронт рисует полосу этапов).
    Фронтенд вызывает этот маршрут при включении режима агента и при каждом
    переключении задачи или сессии, чтобы нарисовать реплики, панель токенов,
    список веток и полосу состояния — диалог выглядит так, будто агент не
    выключался. Задачи нет / диалогов нет — пустой диалог.
    """
    # Блокировку не берём: снимок диалога — чистое чтение без await (как и
    # GET /api/agent/memory), а окно чата иначе ждало бы конца ответа агента.
    session = _current_session()
    dialog = session["dialog"] if session else workspace_store.empty_dialog()
    return {
            "messages": [dict(m) for m in dialog["messages"]],
            "log": [dict(item) for item in dialog.get("log", [])],
            "usage": [dict(item) for item in dialog["usage"]],
            "summary": list(dialog["summary"]),
            "facts": dict(dialog["facts"]),
            "branches": {bid: dict(branch) for bid, branch in dialog["branches"].items()},
            "active_branch": dialog["active_branch"],
            "session": _session_payload(session),
            "state": task_state.snapshot(workspace_store.dialog_state(session)),
            "pending": _pending_stops.get(str(session["id"])) if session else None,
        }


@router.delete("/agent/history")
async def agent_history_clear() -> dict:
    """Очищает диалог ТЕКУЩЕЙ сессии режима «AI-агент».

    Стирает память диалога, замеры токенов, резюме (стратегия «summary») с его
    границей, блок фактов (стратегия «sticky facts») и ветви плана со всеми
    диалогами внутри них (стратегия «branching») — сама сессия (и её название)
    остаётся в истории задач. Вызывается по кнопке «Очистить историю» рядом с
    «Отправить» в режиме агента. Диалогов нет — ничего не меняется.

    Конечный автомат задачи при этом начинается ЗАНОВО (этап planning, план и
    история переходов пустые): очищенный диалог — это новая задача, а прежний
    план к ней отношения не имеет.
    """
    session = _current_session()
    async with _session_lock(session["id"]):
        session["dialog"] = workspace_store.empty_dialog(session["id"])
        await _persist()
    return {"ok": True}
