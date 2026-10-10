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
import base64
import contextlib
import contextvars
import json
import logging
import os
import re
import tempfile
import time
from typing import (Any, AsyncIterator, Callable, Dict, List, Optional,
                    Tuple)

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from app import config
from app.ai import attachments as attach_store
from app.ai import client as llm_client
from app.ai import service
from app.ai import invariants as invariants_store
from app.ai import local_llm
from app.ai import mcp as mcp_store
from app.ai import periodic as periodic_store
from app.ai import profiles as profile_store
from app.ai import rag
from app.ai import rag_dialog
from app.ai import rag_documents
from app.ai import rag_jobs
from app.ai import rag_query
from app.ai import rag_rerank
from app.ai import rag_search
from app.ai import rag_suite
from app.ai import rag_store
from app.ai import task_memory as memory_store
from app.ai import task_state
from app.ai import workspace as workspace_store
from app.ai.agent import (
    Agent, AgentConfig, DEFAULT_SUMMARY_SIZE, DEFAULT_WINDOW_SIZE, merge_usage,
)
from app.schemas import (
    ChatMessage, InvariantCreate, InvariantDelete, InvariantPick, InvariantResolve,
    LlmParams, LlmServerAction, LlmSourceUpdate,
    McpApply, MemoryEntryCreate, NameUpdate, PeriodicUpdate, PlanUpdate, ProfileCreate,
    ProfileFields, RagApply, RagDialogTest, RagJobDone, RagRelax, RagUpload,
    SessionCreate, SessionMode, TaskCreate,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# Каталог незавершённых ПОТОКОВЫХ загрузок внутри RAG_DIR. Имя намеренно не
# похоже на идентификатор базы («kb-…»): список баз проверяет шаблон и этот
# каталог просто не увидит, а мусор от оборванной загрузки не станет «базой».
_INCOMING_DIR = ".incoming"

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
# Очередь записи файла workspace: снимки не должны обгонять друг друга.
_persist_lock = asyncio.Lock()


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
# id ПЕРИОДИЧЕСКИХ задач, повтор которых выполняет планировщик прямо сейчас (см.
# app/periodic_runner.py): по этому признаку интерфейс показывает «⏳ автозапуск»,
# а сам планировщик не запускает повтор дважды.
_periodic_running: set = set()
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


# ПРОФИЛЬ, В КОНТЕКСТЕ КОТОРОГО ИДЁТ ЗАПРОС. Обычно None — берётся активный
# профиль пользователя. Ставит его ТОЛЬКО планировщик периодических задач
# (app/periodic_runner.py): повтор задачи профиля, который сейчас не открыт,
# должен идти в СВОЁМ профиле (свой системный блок профиля, своя долговременная
# память, свои задачи). ContextVar, а не общая переменная, потому что она
# локальна для asyncio-задачи: значение видит только тот прогон, который его
# поставил, а параллельные запросы пользователя продолжают работать с активным
# профилем.
_ACTIVE_PROFILE_OVERRIDE: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar(
    "agent_profile_override", default=None)


def _current_profile() -> Optional[Dict[str, Any]]:
    """Текущий профиль пользователя (None — профилей нет: не должно случаться,
    профиль заводится при старте).

    Для фонового повтора периодической задачи профиль берётся из переопределения
    (см. _ACTIVE_PROFILE_OVERRIDE) — это профиль ВЛАДЕЛЬЦА задачи.
    """
    override = _ACTIVE_PROFILE_OVERRIDE.get()
    if override:
        return profile_store.find_profile(_profiles, override)
    return profile_store.active_profile(_profiles)


def _current_profile_id() -> Optional[str]:
    """id текущего профиля — им ограничены задачи, диалоги и память."""
    profile = _current_profile()
    return profile["id"] if profile else None


@contextlib.contextmanager
def profile_override(profile_id: Optional[str]):
    """Выполняет блок в профиле ВЛАДЕЛЬЦА задачи (для планировщика повторов).

    Ставит переопределение профиля (см. _ACTIVE_PROFILE_OVERRIDE) на время
    фонового прогона периодической задачи: задача профиля, который сейчас не
    открыт, должна идти в СВОЁМ профиле — с его системным блоком и его
    долговременной памятью. Значение живёт в asyncio-задаче планировщика, поэтому
    запросы пользователя в это же время работают с активным профилем.
    """
    token = _ACTIVE_PROFILE_OVERRIDE.set(str(profile_id or "").strip() or None)
    try:
        yield
    finally:
        _ACTIVE_PROFILE_OVERRIDE.reset(token)


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

    Записи идут ПО ОЧЕРЕДИ (_persist_lock): часть маршрутов сохраняет workspace
    без _workspace_lock, и два параллельных снимка могли лечь на диск в обратном
    порядке — более старый затирал свежий.
    """
    async with _persist_lock:
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
    # Занятые задачи (шаг агента или автозапуск периодической задачи) — по ним
    # интерфейс помечает задачи в списке: «⏳ идут шаги» / «⏳ автозапуск».
    running = set(_running_sessions) | set(_periodic_running)
    snapshot = workspace_store.snapshot(_workspace, _current_profile_id(), running=running)
    # Профиль — глобальная сущность (не на задачу), но фронту удобно получать
    # его вместе со снимком workspace: иконка профиля и его поля обновляются
    # одним ответом на любую операцию с задачами/диалогами.
    snapshot["profile"] = profile_store.snapshot(_profiles)
    # Статистика экспертных режимов: живёт на сервере (у профиля), поэтому
    # страница «Статистика ответов» её не теряет.
    snapshot["stats"] = workspace_store.expert_stats(_workspace, _current_profile_id())
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
    # Замеры СЛУЖЕБНЫХ запросов (kind="plan" — построен только план;
    # kind="service" — например, отказ по инвариантам) реплик в диалоге не
    # имеют: синхронизация их не касается, иначе расход пропадал бы из панели
    # токенов после перезагрузки страницы.
    plain = [item for item in usage if not item.get("kind")]
    while len(plain) > requests:
        for index, item in enumerate(usage):
            if not item.get("kind"):
                usage.pop(index)
                break
        else:
            break
        plain = [item for item in usage if not item.get("kind")]


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


def _plan_message(state: "task_state.TaskState", autonomous: bool = False,
                  delivered: bool = False) -> str:
    """Текст плана для чата (этап planning): шаги и что делать дальше.

    `delivered` — результат по запросу УЖЕ получен внешними инструментами (данные
    собраны, файл выгружен): подтверждение не спрашивается, и текст плана обязан
    говорить то же самое. Иначе чат просил «нажмите ок», а задача уже шла.
    """
    steps = state.steps or []
    plural = task_state.steps_word(len(steps))
    lines = [f"📋 План задачи — {len(steps)} {plural}:"]
    lines.extend(f"{number}. {step}" for number, step in enumerate(steps, 1))
    if autonomous:
        lines.append(
            "Режим «работай автономно»: подтверждение плана не требуется — "
            "начинаю выполнение с первого шага."
        )
    elif delivered:
        lines.append(
            "Подтверждение не требуется: результат по запросу уже получен внешними "
            "инструментами (данные собраны, файл приложен) — начинаю выполнять шаги."
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


def _plan_needed(text: str, session: Dict[str, Any],
                 force_plan: bool = False,
                 periodic: bool = False) -> Tuple[bool, str]:
    """НУЖЕН ЛИ ПЛАН для этого запроса: (нужен, причина).

    ОДИН РЕЖИМ, ДВА ПУТИ. Источники у обоих путей ОДНИ (правила проекта, внешние
    инструменты, фрагменты баз знаний) — разница только в том, нужна ли
    ПРОЦЕДУРА. Поэтому выбор делает КОД по признакам запроса
    (`task_state.needs_plan`), а человек переопределяет его:

      * `force_plan` (поле запроса) — кнопка «⚙ Разложить работу на шаги»
        под прямым ответом: то же самое, но как задача;
      * тип задачи (`session["mode"]`): `plan` — всегда план, `answer` — всегда
        прямой ответ, `auto` — по обстоятельствам (по умолчанию);
      * ПЕРИОДИЧЕСКАЯ задача всегда идёт планом: её повтор — это процедура, и
        выбор тут не при чём.

    Причина возвращается человеческой фразой — чат говорит, ПОЧЕМУ выбран путь:
    «промпт — просьба, гарантия — код» требует, чтобы решение было видно, а не
    выглядело случайным.
    """
    if periodic:
        return True, "периодическая задача идёт по плану"
    if force_plan:
        return True, "просили выполнить как задачу"
    mode = workspace_store.session_mode(session)
    if mode == workspace_store.MODE_PLAN:
        return True, "у задачи тип «всегда по плану»"
    if mode == workspace_store.MODE_ANSWER:
        return False, "у задачи тип «всегда сразу ответ»"
    return task_state.needs_plan(text)


def _log_event(dialog: Optional[Dict[str, Any]], event: Dict[str, Any]) -> None:
    """Пишет событие агента в журнал чата сессии (что видит пользователь).

    В журнал идут только узлы окна чата: ответ (bot) и служебные сообщения
    (debug/error). Служебные события (state/usage/done/branches) не пишутся:
    это не сообщения, а состояние и замеры.

    У узла ответа могут быть ВЛОЖЕНИЯ (`files`): файлы, которые вернули
    MCP-инструменты. Они хранятся в журнале вместе с текстом — карточки со
    ссылкой на скачивание видны и после переключения задачи.
    """
    if not dialog:
        return
    kind = event.get("type")
    if kind == "debug":
        workspace_store.add_log(dialog, workspace_store.LOG_DEBUG, event.get("text"))
    elif kind == "error":
        workspace_store.add_log(dialog, workspace_store.LOG_ERROR, event.get("text"))
    elif kind == "bot":
        workspace_store.add_log(dialog, workspace_store.LOG_ASSISTANT, event.get("text"),
                                files=event.get("files"),
                                sources=event.get("sources"))


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
    snapshot = _state_snapshot(session, preview)
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


def _periodic_session(session: Optional[Dict[str, Any]]) -> bool:
    """Периодическая ли задача (у неё своя полоса этапов: см. task_state.snapshot)."""
    return bool(session) and bool(workspace_store.periodic_meta(session))


def _state_snapshot(session: Optional[Dict[str, Any]],
                    state: "task_state.TaskState") -> Dict[str, Any]:
    """Снимок состояния для интерфейса — с учётом ПЕРИОДИЧНОСТИ задачи.

    У периодической задачи в полосе нет блоков «проверка» и «готово»: итоговой
    проверки она не проходит, а завершённой не бывает (повторяется, пока её не
    остановит пользователь). Один помощник на все ответы и события, чтобы полоса
    не «мигала» четырьмя блоками в одних ответах и двумя в других.
    """
    return task_state.snapshot(state, periodic=_periodic_session(session))


def _state_event(session: Optional[Dict[str, Any]],
                 state: "task_state.TaskState") -> dict:
    """Событие потока с состоянием автомата (фронт рисует полосу этапов)."""
    return {"type": "state", "state": _state_snapshot(session, state)}


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
    snapshot = _state_snapshot(session, workspace_store.dialog_state(session))
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

    У ПЕРИОДИЧЕСКОЙ задачи отмена означает остановку повторов: её автозапуск
    выключает планировщик (см. app/periodic_runner.py), а ВНЕШНИЕ СБОРЫ, которые
    задача завела на серверах (наблюдения и т. п.), останавливаются ЗДЕСЬ —
    задача остановлена, значит и работа на сервере ей больше не нужна.
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
    # Внешние сборы снимаем только у ПЕРИОДИЧЕСКОЙ задачи: у обычной «Отмена»
    # останавливает текущий заход (задачу можно продолжить сообщением), а
    # удалять накопленные на сервере данные было бы потерей без спроса.
    if workspace_store.periodic_meta(session):
        await _stop_mcp_started(session, "периодическая задача отменена")
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


@router.post("/agent/state/accept")
async def state_accept() -> dict:
    """Кнопка «Принять вручную»: результат принимается без проверки модели.

    Нужна, когда проверку результата выполнить не удалось (`check_blocked`):
    задача НЕ объявляется готовой сама и на доработку не возвращается — решение
    за пользователем. Принять можно только такую задачу (этап validation со
    снятой проверкой), иначе 400: обычный путь закрытия задачи — проверка.
    """
    session = _require_session()
    async with _session_lock(session["id"]):
        state = workspace_store.dialog_state(session)
        if state.paused:
            raise HTTPException(status_code=400, detail="Задача на паузе — нажмите «Продолжить»")
        if state.stage != "validation" or not state.check_blocked:
            raise HTTPException(
                status_code=400,
                detail="Принимать вручную нечего: проверка результата не ждёт решения",
            )
        task_state.validation_ok(
            state, "результат принят пользователем вручную (проверка не удалась)")
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
async def chat(msg: ChatMessage) -> dict:
    """Принимает сообщение пользователя и возвращает ответ бота."""
    if not msg.content.strip():
        return {"user": msg.content, "bot": "Пожалуйста, введите сообщение."}
    # Выбрана локальная модель — сервер MLX обязан отвечать ДО вызова: иначе
    # обычный режим молча подменил бы недоступную модель ДЕМО-ОТВЕТОМ, и это
    # выглядело бы как «модель ответила ерунду». Недоступный источник — понятная
    # причина текстом (как и другие сообщения этого маршрута), а не пустой ответ.
    reason = await local_source_guard()
    if reason:
        return {
            "user": msg.content,
            "bot": LOCAL_SOURCE_NOT_READY % reason,
            "correct": None,
        }
    answer, correct, analytics = service.generate_response(
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
    # Метрики обращений к модели за этот запрос (время, токены, стоимость):
    # страница статистики показывает расход и в обычном, и в экспертном режиме.
    if analytics:
        result["analytics"] = analytics
    # Страница «Статистика ответов» (экспертный режим): вердикт модели копится в
    # workspace ПРОФИЛЯ, поэтому статистика не обнуляется при переключении
    # режима, переходе между задачами и перезапуске приложения. Запись файла —
    # через to_thread, чтобы не блокировать event loop.
    if msg.expert_mode and isinstance(correct, bool):
        workspace_store.add_expert_result(
            _workspace, msg.expert_mode_type, correct, _current_profile_id())
        # Обновлённый счётчик отдаём тем же ответом: интерфейсу не нужен
        # отдельный запрос снимка, чтобы показать статистику.
        result["stats"] = workspace_store.expert_stats(_workspace, _current_profile_id())
        # Пишем через общий _persist: снимок файла идёт в одной очереди с
        # остальными записями workspace.
        await _persist()
    # Настройка «Тест моделей»: ответы каждой модели по отдельности.
    if isinstance(answer, dict) and "model_responses" in answer:
        model_responses = answer["model_responses"]
        result["model_responses"] = model_responses
        result["bot"] = "\n".join(
            f"Ответ модели {r['model']}: {r['text']}" for r in model_responses
        ) if model_responses else "Пожалуйста, введите сообщение."
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


# ---------------------------------------------------------------------------
# Источник ответа: локальная модель (MLX на этом же Mac) или удалённая
#
# Переключатель в панели workspace. Источник ОДИН на всё приложение — это
# адрес и модель по умолчанию, а не настройка одного режима (см.
# app/ai/local_llm.py и config.active_model). Опрос состояния — короткий
# HTTP-запрос к локальному серверу, поэтому и он, и команды серверу уходят в
# поток: event loop не должен ждать ни опроса, ни запуска.
# ---------------------------------------------------------------------------
@router.get("/agent/llm")
async def llm_get() -> dict:
    """Состояние действующего источника ответа.

    Возвращает {"source", "provider", "title", "model", "remote", "installed",
    "server", "ready", "hint"}: что выбрано, отвечает ли локальный сервер, что
    установлено (venv, веса) и что делать, если ответить нечем. К модели этот
    маршрут НЕ обращается — только короткий опрос «жив ли сервер», иначе
    открытие панели стоило бы токенов.
    """
    return await asyncio.to_thread(local_llm.status)


@router.post("/agent/llm/source")
async def llm_source_set(payload: LlmSourceUpdate) -> dict:
    """Переключает источник ответа («локальная / удалённая») и сохраняет выбор.

    Выбор переживает перезапуск приложения (config.LLM_SOURCE_FILE). При
    переходе на локальную модель сервер поднимается ОТДЕЛЬНЫМ процессом
    (autostart) и загружает веса десятки секунд — маршрут этого не ждёт: в
    ответе `server.starting` = true, готовность интерфейс видит опросом
    GET /api/agent/llm. Неизвестное имя источника — 400, а не «тихий remote»:
    человек не должен думать, что запросы больше не уходят в сеть, когда уходят.
    """
    try:
        state = await asyncio.to_thread(local_llm.switch, payload.source,
                                        payload.autostart)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return state


def _llm_params_view(task: Optional[Dict[str, Any]], saved: bool = False) -> dict:
    """Снимок параметров модели проекта (окно шестерёнки ⚙️ проекта).

    Отдаёт {"project_id", "temperature", "max_tokens", "limits", "saved"}:
    действующие значения проекта (None — «не задано»: параметр в запрос к API не
    отправляется, работает значение провайдера), границы полей и признак
    `saved` — «значения только что сохранены» (интерфейс по нему показывает
    подтверждение, а не догадывается по разнице чисел).
    """
    settings = workspace_store.llm_settings(task)
    return {
        "project_id": (task or {}).get("id"),
        "temperature": settings["temperature"],
        "max_tokens": settings["max_tokens"],
        "limits": {
            "temperature": [workspace_store.LLM_TEMPERATURE_MIN,
                            workspace_store.LLM_TEMPERATURE_MAX],
            "max_tokens": [workspace_store.LLM_MAX_TOKENS_MIN,
                           workspace_store.LLM_MAX_TOKENS_MAX],
        },
        "saved": bool(saved),
    }


@router.get("/agent/llm/params")
async def llm_params_get() -> dict:
    """Параметры модели проекта (temperature и предел длины ответа).

    Настройка ПРОЕКТА, как «MCP» и «RAG»: правится в окне шестерёнки ⚙️ проекта и
    действует во всех его задачах-диалогах. Значения уходят в модель
    ДЕЙСТВУЮЩЕГО источника — и в локальную, и в удалённую. К модели маршрут НЕ
    обращается: это данные.
    """
    return _llm_params_view(_current_task())


@router.post("/agent/llm/params")
async def llm_params_set(payload: LlmParams) -> dict:
    """Запоминает параметры модели проекта (полное состояние, а не правка полей).

    Пустое поле — «настройка снята» (None): параметр перестаёт уходить в запрос к
    API, и действует значение провайдера. Так «пусто» и «0» не путаются:
    temperature=0 — осознанный выбор пользователя.

    Значения проверяются ЗДЕСЬ и отказ называет причину (текст вместо числа,
    число вне границ) — тихое зажатие до границы выглядело бы как «настройка не
    применилась». Файл при чтении зажимается молча (app/ai/workspace.py:
    _normalize_llm): битая настройка в файле ломать приложение не должна.
    """
    task = _current_task()
    if task is None:
        raise HTTPException(
            status_code=400,
            detail="Сначала создайте проект — параметры модели привязаны к проекту")
    settings, error = workspace_store.llm_params_from_form(payload.temperature,
                                                           payload.max_tokens)
    if error:
        raise HTTPException(status_code=400, detail=error)
    async with _workspace_lock:
        workspace_store.set_llm_params(task, settings["temperature"],
                                       settings["max_tokens"])
        await _persist()
    return _llm_params_view(task, saved=True)


@router.post("/agent/llm/server")
async def llm_server_action(payload: LlmServerAction) -> dict:
    """Запускает или останавливает локальный сервер модели.

    Сервер живёт отдельным процессом и переживает перезапуск приложения,
    поэтому его жизнь управляется явно. `start` не ждёт загрузки весов (в ответе
    `server.starting`), `stop` ждёт завершения процесса: иначе следующий запуск
    упёрся бы в занятый порт. Неустановленное окружение — 409 с причиной.
    """
    try:
        if payload.action == "start":
            state = await asyncio.to_thread(local_llm.start)
        else:
            state = await asyncio.to_thread(local_llm.stop)
    except local_llm.LocalLlmError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return state


# ---------------------------------------------------------------------------
# ГОТОВНОСТЬ ИСТОЧНИКА ОТВЕТА ПЕРЕД ВЫЗОВОМ МОДЕЛИ
#
# Выбрана локальная модель — сервер MLX обязан ОТВЕЧАТЬ до первого вызова
# (см. §5.14 в SESSION_PROMPT): иначе запрос уходит в закрытую дверь, а ответ
# возвращается пустым и выглядит как «модель промолчала». Гейт ОДИН и общий для
# ВСЕХ маршрутов, которые зовут модель: чат агента, обычный режим, а также
# мини-чат по документам, `/test_rag` и контрольные диалоги RAG — они ходят к
# модели напрямую, своими вызовами, и раньше готовность источника не проверяли
# вовсе: при остановленном сервере мини-чат отвечал «не получил ответа от
# модели», хотя сервер достаточно было поднять.
#
# Строки тоже ОДНИ на все маршруты: причина отказа читается одинаково, откуда бы
# запрос ни пришёл.
# ---------------------------------------------------------------------------
LOCAL_SOURCE_START_NOTE = (
    "🧠 Локальная модель: запускаю сервер и читаю веса — это занимает десятки "
    "секунд.")
LOCAL_SOURCE_NOT_READY = (
    "⚠ Локальная модель не готова: %s. Источник ответа переключается в панели "
    "слева («Локальная модель» / «Удалённая модель»).")


async def local_source_guard(
        notify: Optional[Callable[[str], None]] = None) -> str:
    """Готовит ИСТОЧНИК ОТВЕТА перед работой: пустая строка — можно звать модель.

    Выбран удалённый источник — ждать нечего (сеть и ключ — забота клиента).
    Выбран локальный — сервер поднимается ЗДЕСЬ, и ожидание загрузки весов идёт
    в отдельном потоке, а не в цикле событий. Не дождались — возвращается
    понятная причина (`local_llm.status()["hint"]`), и маршрут выдаёт её
    человеку вместо пустого ответа модели.

    `notify` — необязательный обработчик строки прогресса: он вызывается ПЕРЕД
    ожиданием весов (только если сервер придётся запускать), чтобы потоковый
    маршрут успел сказать «запускаю сервер» до многосекундной тишины.
    """
    if config.llm_source() != "local":
        return ""
    state = await asyncio.to_thread(local_llm.status)
    # `ready` у локального источника — это «сервер отвечает И объявил модель»:
    # отвечающий сервер, которому не нашлись веса, генерировать не может, и
    # пропустить в него вызов значило бы получить пустой ответ без причины.
    if state["ready"]:
        return ""
    if notify is not None and not state["server"]["running"]:
        notify(LOCAL_SOURCE_START_NOTE)
    try:
        await asyncio.to_thread(local_llm.prepare)
    except local_llm.LocalLlmError as exc:
        return str(exc)
    return ""


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
    """Удаляет задачу вместе со всеми её диалогами (корзина у списка задач).

    Перед удалением останавливаются ВНЕШНИЕ СБОРЫ ВСЕХ диалогов проекта
    (наблюдения, подписки MCP — см. _stop_mcp_started): проект удаляется целиком,
    и работа, которую его задачи завели на серверах, должна прекратиться.
    """
    async with _workspace_lock:
        task = _own_task(workspace_store.find_task(_workspace, task_id))
        if task is None:
            raise HTTPException(status_code=404, detail="Проект не найден")
        for session in list(task.get("sessions", [])):
            await _stop_mcp_started(session, f"проект «{task.get('name') or ''}» удалён")
        if not workspace_store.delete_task(_workspace, task_id):
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
async def session_create(payload: Optional[SessionCreate] = None) -> dict:
    """Создаёт в текущей задаче новый пустой диалог (кнопки панели Workspace).

    «Новая задача» — обычная задача: выполняется один раз по запросу
    пользователя. «Новая периодическая задача» (`periodic=true`) — задача с
    расписанием: сервер сам повторяет её запрос через период (по умолчанию
    сутки) и кладёт результат в чат этой задачи (см. app/ai/periodic.py и
    app/periodic_runner.py). Период можно назвать и в самом запросе («Сводка
    погоды в Москве за последние сутки, раз в час») — сервер распознаёт его,
    когда пользователь отправит первое сообщение.

    Сессия сразу становится текущей — её диалог пуст, а заголовок в истории
    появится по первому запросу пользователя.
    """
    # Блокировку не берём (см. task_create): новая задача отвечает сразу, даже
    # если прямо сейчас выполняется шаг агента — шаг пишет в СВОЮ задачу.
    task = _current_task()
    if task is None:
        raise HTTPException(status_code=400, detail="Сначала создайте проект")
    periodic = None
    if payload is not None and payload.periodic:
        periodic = payload.interval if payload.interval else periodic_store.DEFAULT_INTERVAL
    session = workspace_store.create_session(task, periodic=periodic)
    # ТИП задачи можно задать сразу при создании («разговор по документам» —
    # мини-чат по базам вместо плана). У периодической задачи тип не меняется:
    # её повтор ведёт автомат по сохранённому плану (см. app/periodic_runner.py).
    if payload is not None and payload.mode and periodic is None:
        workspace_store.set_session_mode(session, payload.mode)
    await _persist()
    return _snapshot()


@router.post("/agent/sessions/{session_id}/mode")
async def session_mode_set(session_id: str, payload: SessionMode) -> dict:
    """Переключает ТИП задачи-диалога: «разговор по документам» / «задача с планом».

    Тип принадлежит ЗАДАЧЕ, а не режиму окна: у каждой задачи свой диалог, своя
    память задачи и свой замер, поэтому и способ ответа — её свойство. «Разговор
    по документам» — мини-чат по базам знаний (история, поиск на каждый вопрос,
    ответ по фрагментам, источники, память задачи) БЕЗ плана и шагов;
    «задача с планом» — обычный режим «AI-агент» с конечным автоматом.

    Переключение НИЧЕГО не сбрасывает: диалог (messages, журнал, память задачи)
    общий для обоих типов, а состояние автомата живёт своей жизнью — вернувшись к
    «задаче с планом», пользователь продолжит её с того же места. У периодической
    задачи тип не переключается: её повтор ведёт автомат по сохранённому плану.
    """
    task, session = _find_session_anywhere(session_id)
    if task is None or session is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    if workspace_store.periodic_meta(session):
        raise HTTPException(status_code=409, detail=(
            "У периодической задачи тип не переключается: её повтор ведёт автомат "
            "по сохранённому плану"))
    async with _workspace_lock:
        workspace_store.set_session_mode(session, payload.mode)
        await _persist()
        return _snapshot()


@router.get("/agent/periodic")
async def periodic_get() -> dict:
    """Периодические задачи профиля: расписание для списка задач и опроса.

    Отдаёт {"tasks": [{"task_id", "session_id", "title", "periodic": {...}}],
    "now": "…"}: `periodic` — период, срок следующего повтора, сколько раз задача
    уже повторялась, ошибка последнего повтора и размер журнала задачи
    (`log_len`). Интерфейс опрашивает этот маршрут, пока включён режим агента: по
    `log_len` открытой задачи видно, что автозапуск дописал в чат новое, и окно
    чата обновляется само (см. pollPeriodic в chat.web/chat.html).

    Задачи без расписания в ответ не попадают; выключенный повтор попадает — он
    остаётся периодической задачей и его можно снова включить (🔁 в списке).
    """
    now = periodic_store.now()
    tasks: List[Dict[str, Any]] = []
    for task, session in workspace_store.periodic_sessions(_workspace, _current_profile_id()):
        brief = workspace_store.session_periodic_brief(
            session,
            running=str(session["id"]) in _periodic_running
            or str(session["id"]) in _running_sessions,
            moment=now)
        tasks.append({
            "task_id": task.get("id"),
            "session_id": session.get("id"),
            "title": workspace_store.session_title(session),
            "periodic": brief,
        })
    return {"tasks": tasks, "now": periodic_store.to_iso(now)}


@router.post("/agent/periodic/{session_id}")
async def periodic_update(session_id: str, payload: PeriodicUpdate) -> dict:
    """Правка расписания периодической задачи (кнопка 🔁 у задачи в списке).

    Меняет период и/или включает-выключает автозапуск. Обычную (одноразовую)
    задачу сделать периодической этим маршрутом нельзя — 409: расписание заводит
    только кнопка «Новая периодическая задача», а здесь правится уже заведённое.

    Период задаётся числом секунд (`interval`) или текстом (`period` — «раз в
    час», «каждые 30 минут»); число важнее текста, значение приводится к
    допустимым границам (см. app/ai/periodic.py). Выключенный повтор сам не
    возвращается: даже если в задаче написать ещё сообщение, автозапуск останется
    выключенным, пока его не включат снова.
    """
    task, session = _find_session_anywhere(session_id)
    if task is None or session is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    meta = workspace_store.periodic_meta(session)
    if not meta:
        raise HTTPException(
            status_code=409,
            detail="Задача не периодическая: её расписание заводит кнопка "
                   "«Новая периодическая задача»")
    interval: Optional[int] = None
    if payload.interval:
        interval = periodic_store.clamp(payload.interval)
    elif payload.period.strip():
        parsed, phrase = periodic_store.parse_request(payload.period)
        if parsed is None:
            raise HTTPException(
                status_code=400,
                detail="Не понял период — напишите его как «раз в час» или "
                       "«каждые 30 минут»")
        interval = parsed
    if interval is not None:
        workspace_store.set_periodic(session, periodic_store.reschedule(
            meta, interval=interval, moment=periodic_store.now()))
    if payload.enabled != bool(meta.get("enabled")):
        workspace_store.set_periodic(session, periodic_store.set_enabled(
            meta, payload.enabled, moment=periodic_store.now()))
    await _persist()
    snapshot = _snapshot()
    snapshot["periodic"] = await periodic_get()
    return snapshot


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

    Перед удалением останавливаются ВНЕШНИЕ СБОРЫ задачи (наблюдения, подписки —
    см. _stop_mcp_started): задачи больше нет, и работа, которую она завела на
    серверах, должна прекратиться.
    """
    async with _workspace_lock:
        task, session = _find_session_anywhere(session_id)
        if task is None or session is None:
            raise HTTPException(status_code=404, detail="Задача не найдена")
        await _stop_mcp_started(session, "задача удалена")
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


# ---------------------------------------------------------------------------
# MCP: внешние инструменты агента (кнопка «MCP» рядом с шестерёнкой проекта)
# ---------------------------------------------------------------------------
# MCP (Model Context Protocol) — протокол внешних инструментов: сервер объявляет
# инструменты (погода, курсы валют, цены), а агент вызывает их сам, когда для
# ответа не хватает данных (см. app/ai/mcp.py). Набор серверов — настройка
# ПРОЕКТА (task["mcp"]): включённые серверы видны в КАЖДОМ запросе агента.
#
# Что происходит в запросе (ДО планирования, как и разбор инвариантов):
#   1) по запросу пользователя служебный вызов выбирает нужные инструменты и их
#      аргументы (Agent.choose_mcp_tools);
#   2) инструменты выполняются ЛОКАЛЬНЫМИ процессами-серверами (tools/call);
#   3) полученные данные уходят в модель отдельным системным блоком — вместе с
#      планом, ответом и проверкой результата.
# Данные привязаны к ЗАПРОСУ ЗАДАЧИ (подпись «серверы + исходный запрос»): шаги
# плана и проверка результата идут отдельными HTTP-запросами, и без сохранённых
# данных второй шаг отвечал бы уже без них. Новый запрос пользователя — новая
# подпись, инструменты выбираются заново.
def _mcp_signature(enabled: List[str], request: str) -> str:
    """Подпись набора данных MCP: включённые серверы + запрос задачи."""
    return "|".join(list(enabled) + [_flat(request)[:400]])


async def _mcp_view_async(task: Optional[Dict[str, Any]],
                          force: bool = False) -> Dict[str, Any]:
    """Снимок MCP для интерфейса (модалка «MCP» рядом с шестерёнкой проекта).

    Отдаёт серверы проекта с описанием, инструментами, состоянием галочек и
    доступностью. Серверы опрашиваются по-настоящему (initialize + tools/list) в
    отдельном потоке — цикл событий на это время не блокируется. Пользователь
    видит, что «включено» — это работающий инструмент, а не просто галочка:
    сбой сервера показывается причиной, а не молчанием.
    """
    enabled = workspace_store.mcp_enabled(task)
    data = await mcp_store.async_view(enabled, force=force)
    data["project_id"] = (task or {}).get("id")
    return data


def _mcp_tools(found: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Плоский список инструментов доступных серверов — для выбора моделью."""
    tools: List[Dict[str, Any]] = []
    for item in found:
        if not item.get("ok"):
            continue
        for tool in (item.get("tools") or []):
            tools.append({
                "server": item["id"],
                "server_name": item.get("server_name") or item["id"],
                "tool": tool["name"],
                "description": tool.get("description") or tool.get("title") or "",
                "schema": tool.get("schema") or {},
            })
    return tools


async def _preflight_mcp(task: Dict[str, Any], session: Dict[str, Any], text: str,
                         analyzer: Agent, state: "task_state.TaskState",
                         machine_step: bool = False, reuse: bool = False,
                         fresh: bool = False, approved: bool = False,
                         resume_chain: bool = False, direct: bool = False
                         ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[str], bool]:
    """Данные внешних инструментов MCP по запросу — ДО этапа планирования.

    Возвращает (данные вызовов, расход служебного вызова, строки диагностики).
    MCP у проекта выключен — ни вызовов, ни расхода: запрос идёт как раньше.

    Данные кладутся в dialog["mcp"] вместе с подписью «серверы + запрос задачи»:
    шаги плана выполняются отдельными запросами, и каждый из них должен видеть те
    же факты. `reuse` (служебная реплика шага, подтверждение плана, «работай
    автономно», «перезапусти») означает «это НЕ новый запрос»: инструменты не
    выбираются заново ни по служебной фразе, ни по фразе управления, а данные
    берутся из сохранённых. Новый содержательный текст пользователя (в том числе
    правка запроса на этапе awaiting_user) — наоборот, новый запрос: инструменты
    выбираются по НЕМУ, иначе агент отвечал бы по данным прежнего запроса.

    `fresh` — АВТОЗАПУСК периодической задачи: запрос у повтора ТОТ ЖЕ, поэтому
    подпись совпадает с прежней, но данные обязаны быть новыми — за прошедший
    период они изменились. Сохранённый набор при `fresh` не переиспользуется:
    инструменты выбираются и вызываются заново, а результат кладётся под ту же
    подпись — шаги этого повтора берут уже свежие данные. Какие это инструменты —
    неважно: набор объявляет сам сервер, и он у проекта может меняться.

    `approved` — ПЛАН ПОДТВЕРЖДЁН (или задача автономная/периодическая): только
    тогда выполняются вызовы, меняющие что-то на сервере (сохранение набора,
    выгрузка файла, запуск сбора). До подтверждения выполняется ТОЛЬКО ЧТЕНИЕ —
    данные для плана, без побочных эффектов и без файла в чате; отложенная часть
    доигрывается на первом шаге выполнения (см. `_resume_mcp_chain`).

    `resume_chain` — план подтверждён, поэтому НЕДОИГРАННУЮ цепочку можно
    продолжать (добирать данные по зависимостям) уже на ТЕКУЩЕМ шаге. Это
    отдельный флаг, а не тот же `approved`: «эффекты разрешены» и «цепочку можно
    продолжать» — разные вещи. Продолжение идёт в режиме чтений, а сохранение и
    выгрузка по-прежнему ждут последнего шага.
    """
    dialog = session["dialog"]
    enabled = workspace_store.mcp_enabled(task)
    if not enabled:
        return [], {}, [], False
    # Служебные фразы описывают ПРЕЖНИЙ запрос задачи (state.request), новый текст
    # пользователя — сам является запросом.
    request_text = ((state.request or text or "").strip() if (reuse or machine_step)
                    else (text or "").strip())
    # ДЕДЛАЙН ЦЕПОЧКИ. У обычной задачи это общий предел времени на цепочку
    # (CHAIN_DEADLINE_S); у ПЕРИОДИЧЕСКОЙ — не больше половины её периода: повтор
    # приходит по расписанию, и цепочка не имеет права «переехать» в следующий
    # цикл (у задачи с периодом 60 с дедлайн получается ~30 с).
    deadline = time.monotonic() + mcp_store.CHAIN_DEADLINE_S
    meta = session.get("periodic") if isinstance(session.get("periodic"), dict) else {}
    try:
        interval = int((meta or {}).get("interval") or 0)
    except (TypeError, ValueError):
        interval = 0
    if interval > 0:
        deadline = time.monotonic() + min(mcp_store.CHAIN_DEADLINE_S,
                                          max(20.0, interval * 0.5))
    signature = _mcp_signature(enabled, request_text)
    stored = workspace_store.dialog_mcp(dialog)
    chain_state = stored.get("chain") or {}
    same_request = bool(stored.get("signature")) and stored.get("signature") == signature
    # ЦЕПОЧКА НЕ ДОИГРАНА: до подтверждения плана выполнены только чтения, а
    # зависимые вызовы (в том числе ЧТЕНИЯ по найденным данным) ждали «ок».
    # Доигрываем её сразу после подтверждения — на ПЕРВОМ же шаге выполнения, а не
    # только на последнем: шаг плана «получить данные» обязан идти с данными.
    # Иначе он отвечал «данных нет», этот ответ попадал в контекст следующего шага,
    # и модель противоречила уже полученным данным (живая задача: «проверь прогноз в
    # городах Славы и Ивана» — прогноз добывался на последнем шаге, шаг 1 отвечал
    # без погоды, шаг 2 повторил его вывод и проверка отклонила оба шага).
    # ПОБОЧНЫЕ ЭФФЕКТЫ по-прежнему только при `approved` (последний шаг): цепочка
    # продолжается в режиме чтений, а сохранение и выгрузка ждут своего шага.
    resume = bool((approved or resume_chain) and chain_state.get("pending")
                  and same_request)
    if not fresh and same_request and not resume:
        # Данные по этому запросу уже собраны (шаг плана, проверка результата или
        # повтор того же запроса) — служебного вызова и обращений к серверам нет.
        return list(stored.get("results") or []), {}, [], False
    if (reuse or machine_step) and not fresh and not resume:
        # Данных по запросу задачи в диалоге нет (например, задача пришла из
        # файла до первого шага): выбирать инструменты по служебной фразе нельзя.
        return [], {}, [], False
    # ПОВТОР периодической задачи по ТОМУ ЖЕ запросу: спрашивать диспетчера заново
    # не нужно — вызовы уже выбраны, меняются только данные. Повторяем ЧИТАЮЩИЕ
    # вызовы (запуски сбора НЕ повторяем: сбор уже идёт, а второй запуск завёл бы
    # дубль на сервере). Так ответ повтора не остаётся без данных и не «плывёт»:
    # в живой задаче повтор решил, что данные не нужны, и погоды в ответе не было.
    replay = []
    if fresh and not resume and str(stored.get("request") or "") == request_text:
        # Повторяем только те вызовы, которые В ПРОШЛЫЙ РАЗ ДАЛИ ДАННЫЕ: сломанный
        # вызов (например, отчёт по придуманному id) повторять бессмысленно — он
        # снова откажет, а модель, оставшись без данных, начинает их выдумывать
        # (живая задача: отчёт отказал, а суточная таблица всё равно появилась).
        # Такой повтор выбирает инструменты заново — уже с блоком про идущие сборы.
        replay = _replayable_calls(stored)
    if replay:
        # Повтор выполняет ВСЮ цепочку прошлого раза, а не один её раунд: у
        # `async_run_calls` умолчание — предел ОДНОГО раунда (3 вызова), и без
        # явного предела последний вызов цепочки (выгрузка файла) молча терялся —
        # повтор оставался без свежего файла.
        chain_state = stored.get("chain") or {}
        # Повторная запись того же набора: сервер отказывает без флага перезаписи,
        # а в повторе диспетчер не спрашивается — ставим флаг сами (см.
        # mcp_store.enforce_overwrite). Схемы инструментов берём из объявленного
        # списка: он кэширован в памяти, отдельного обращения к серверу нет.
        try:
            found_now = await mcp_store.async_discover(enabled)
            known_tools = _mcp_tools(found_now)
        except Exception:  # noqa: BLE001 — без схем повтор просто идёт как раньше
            logger.warning("MCP: список инструментов для повтора не получен",
                           exc_info=True)
            known_tools = []
        overwrites = mcp_store.enforce_overwrite(
            replay, known_tools, chain_state.get("ids") or {})
        results = await mcp_store.async_run_calls(
            replay, limit=mcp_store.MAX_TOTAL_CALLS_PER_REQUEST)
        workspace_store.set_dialog_mcp(dialog, signature, request_text, results,
                                       calls=replay)
        lines = [
            "MCP: повтор — те же запросы данных, что и в прошлый раз ("
            + ", ".join(f"{call['server']} · {call['tool']}" for call in replay)
            + "), только свежие."
        ]
        for note in overwrites:
            lines.append("MCP: повторная запись того же ключа — ставлю флаг "
                         f"перезаписи ({note}).")
        return results, {}, lines, True
    if fresh and str(stored.get("request") or "") == request_text and stored.get("calls"):
        # Повторять нечего (прошлые вызовы — только запуски сбора или все отказали):
        # выбираем инструменты заново и говорим об этом в чате.
        lines_pre = ["MCP: прошлый повтор не дал данных для чтения — "
                     "выбираю инструменты заново (с учётом уже идущих сборов)."]
    else:
        lines_pre = []
    found = await mcp_store.async_discover(enabled)
    tools = _mcp_tools(found)
    if not tools:
        lines = ["MCP: ни один включённый сервер не ответил — "
                 + "; ".join(f"{item['id']}: {item['error'] or 'нет инструментов'}"
                             for item in found)]
        workspace_store.set_dialog_mcp(dialog, signature, request_text, [])
        return [], {}, lines, False
    # Уже идущие внешние сборы ЭТОЙ задачи (наблюдения, подписки): модель видит их
    # и не начинает заново, а данные читает по их id — иначе каждый повтор заводил
    # бы новое наблюдение на сервере (см. started_calls).
    started = workspace_store.mcp_started(dialog)
    if resume:
        # ЦЕПОЧКА ОСТАЛАСЬ НЕ ДОИГРАННОЙ: до «ок» выполнялись только чтения, а
        # зависимые вызовы ждали подтверждения. Доигрываем её — с этого шага.
        # `approved` решает, разрешены ли ПОБОЧНЫЕ ЭФФЕКТЫ (последний шаг плана):
        # продолжение цепочки на промежуточном шаге идёт в режиме чтений.
        return await _resume_mcp_chain(dialog, signature, request_text, tools,
                                       analyzer, stored, deadline, started,
                                       lines_pre, approved=approved)
    # РЕШЕНИЕ ДИСПЕТЧЕРА + МАРШРУТИЗАЦИЯ (гибридная схема). Разовый запрос идёт
    # одним раундом — дешёвым путём, как раньше. Многошаговый (следующий вызов
    # зависит от результата предыдущего: сохранить прочитанное, выгрузить файл по
    # полученному идентификатору) уходит в ограниченный цикл. Режим объявляет сам
    # диспетчер, а дешёвые признаки (маркеры в запросе + схемы инструментов)
    # подтверждают его: так «какая погода в Казани» не платит за цикл, а «получи,
    # сохрани и отдай Excel» доходит до конца.
    decision = await analyzer.decide_mcp_tools(
        request_text, tools, started=started, chain_state=stored.get("chain"))
    usage = dict(analyzer.last_usage or {})
    calls = list(decision.get("calls") or [])
    # Список вызовов ПОСЛЕ переспроса (см. ниже): None — переспроса ещё не было.
    # Объявляем заранее: иначе ветка «диспетчер ничего не выбрал» обращалась бы к
    # неопределённому имени, и запрос падал бы вместо ответа (поймано проверкой
    # «сбой выбора не ломает ответ»).
    retry: Optional[List[Dict[str, Any]]] = None
    signals = mcp_store.chain_signals(tools, request_text, expected=len(calls))
    # В цикл уходим и по СТРУКТУРНОМУ признаку — «остались невызванные
    # инструменты» (см. mcp_store.chain_possible). Раньше вход требовал текстовых
    # маркеров или объявления модели, и цепочки с зависимостью по данным без
    # слов-подсказок не начинались вовсе: «погода в городе человека» вызывала
    # только список жителей, а второй вызов (прогноз по найденному городу) не
    # делался — модель отвечала «погоды в данных нет». Признак общий: он не
    # знает ни про погоду, ни про города, ни про конкретный сервер.
    # Цена — один вопрос диспетчеру в первом раунде цикла; он же его и закрывает
    # ответом «готово», поэтому лишних раундов не будет.
    declared_chain = str(decision.get("mode") or "") == "chain"
    # СТРУКТУРНАЯ проверка — для ИНТЕРАКТИВНОГО запроса. У периодических задач
    # (fresh=True) своя отлаженная механика: повтор доигрывает СОХРАНЁННЫЕ вызовы
    # без диспетчера, а автономный запуск отдельно просит ЧИТАЮЩИЙ вызов после
    # запуска сбора. Лишний раунд цепочки там ломал бы и счёт служебных вызовов, и
    # поведение повторов (проверено check_periodic: он ловит и то, и другое).
    # ПЕРИОДИЧЕСКАЯ ЗАДАЧА — отдельный случай: у неё своя отлаженная механика
    # (повтор доигрывает СОХРАНЁННЫЕ вызовы без диспетчера; автономный запуск
    # отдельно просит читающий вызов после запуска сбора), и лишние раунды ломают
    # и счёт служебных вызовов, и состав повтора (ловят check_periodic).
    periodic_task = bool(session.get("periodic"))
    structural_chain = (mcp_store.chain_possible(tools, calls)
                        and not fresh and not periodic_task)
    # ПИШУЩАЯ цепочка (просили сохранить/выгрузить, или так объявила модель) и
    # ЧИТАЮЩАЯ (возможен следующий вызов-чтение по найденному значению).
    # ПИШУЩЕЙ считается цепочка, которую просит САМ ЗАПРОС (маркеры «сохрани»,
    # «выгрузи» + инструменты с обязательным идентификатором). Объявление модели
    # (`mode: chain`) НЕ признак записи: в живом случае «погода в городе человека»
    # модель объявила chain потому, что нужен второй ВЫЗОВ-ЧТЕНИЕ, а не запись —
    # и, отложив такую цепочку до «ок», агент в пути прямого ответа (где
    # подтверждения нет вовсе) оставался без погоды: город человека найден, а
    # погоды по нему нет.
    chain_mode = declared_chain or bool(signals.get("needed")) or structural_chain
    if chain_mode:
        if mcp_store.chain_reason(signals):
            why = mcp_store.chain_reason(signals)
        elif declared_chain:
            why = ("диспетчер объявил режим chain: следующий вызов зависит "
                   "от результата предыдущего")
        else:
            why = ("остались невызванные инструменты — проверяю по результатам, "
                   "не нужен ли следующий вызов (могут быть зависимости по "
                   "данным: значение из результата нужно другому инструменту)")
        lines_pre.append(
            "MCP: задача может быть многошаговой — работаю цепочкой (до "
            f"{mcp_store.MAX_CHAIN_ITERATIONS} раундов и "
            f"{mcp_store.MAX_TOTAL_CALLS_PER_REQUEST} вызовов). Причина: " + why)
    # ВЫБОР ТОЛЬКО ИЗ ДЕЙСТВИЙ — это не данные: агенту нечего сказать по такому
    # ответу. Спрашиваем диспетчера ещё раз, прямо требуя ЧИТАЮЩИЙ вызов: запрос
    # «проверяй погоду раз в минуту» превращался в «зарегистрировать наблюдение»,
    # и пользователь не получал ни погоды, ни объяснения про период.
    if calls and not mcp_store.has_reads(calls, tools) \
            and not mcp_store.looks_aggregate(request_text):
        retry = await analyzer.choose_mcp_tools(
            request_text, tools, started=started, need_reads=True)
        usage = merge_usage(usage, dict(analyzer.last_usage or {}))
        if mcp_store.has_reads(retry, tools):
            lines_pre.append(
                "MCP: первые вызовы только запускали сбор — переспросил и добавил "
                "чтение данных (" + ", ".join(
                    f"{call['server']} · {call['tool']}"
                    for call in mcp_store.read_calls(retry, tools)) + ").")
            calls = retry
    if not calls and (chain_mode or mcp_store.requested_kinds(request_text)):
        # ДИСПЕТЧЕР ОТВЕТИЛ «ДАННЫЕ НЕ НУЖНЫ» на запрос, который явно просит
        # сохранить/выгрузить. Это живой случай: на один и тот же запрос модель то
        # строит цепочку, то возвращает пустой список — и задача остаётся без
        # данных и без файла. Переспрашиваем ОДИН раз, назвав невызванные
        # инструменты, которые создают результат (см. deliver_note): нужен
        # читающий вызов и режим chain, дальше цепочка дойдёт сама.
        #
        # Условие — ВИД просьбы, а не режим цепочки: запрос «запиши, что Иван
        # живёт в Москве» закрывается одним вызовом и цепочки не требует, но
        # страховка «просили записать, а не вызвано ничего» нужна ему ровно так
        # же. Иначе флаки-диспетчер оставил бы такую задачу без записи.
        candidates = mcp_store.delivery_candidates(
            tools, {(str(c.get("server") or ""), str(c.get("tool") or "")) for c in calls},
            kinds=mcp_store.unsatisfied_kinds(request_text, calls))
        if candidates:
            retry = await analyzer.decide_mcp_tools(
                request_text, tools, started=started,
                extra=mcp_store.deliver_note(candidates, request_text, initial=True))
            usage = merge_usage(usage, dict(analyzer.last_usage or {}))
            if retry.get("calls"):
                calls = list(retry["calls"])
                lines_pre.append(
                    "MCP: диспетчер ответил «данные не нужны», хотя запрос просит "
                    "сохранить/выгрузить — переспросил ("
                    + ", ".join(f"{call['server']} · {call['tool']}" for call in calls)
                    + ").")
                if str(retry.get("mode") or "") == "chain":
                    chain_mode = True
        # ВТОРОЙ ШАНС ОБЩЕГО ВИДА: диспетчер не выбрал ни одного вызова, а
        # инструменты есть. Это не «запрос без данных», а решение, которое
        # локальная модель принимает по-разному на один и тот же запрос: живой
        # случай — «погода в городе человека» осталась без данных и без погоды,
        # хотя для неё есть и список людей, и прогноз. Переспрашиваем ОДИН раз,
        # объяснив, когда пустой ответ уместен (см. EMPTY_DECISION_NOTE).
        # ... и только в ПРЯМОМ ОТВЕТЕ. У пути с планом и у периодических задач
        # свои механизмы переспроса: «уговор» про результат (`deliver_note`,
        # «ЗАПРОС ВЫПОЛНЕН НЕ ПОЛНОСТЬЮ») и просьба ЧИТАЮЩЕГО вызова после запуска
        # сбора. Общий переспрос перехватывал у них очередь решений диспетчера и
        # ломал их поведение (живой отказ check_mcp: «уточнений: 0» там, где
        # уточнение обязано быть).
        if not calls and retry is None and not fresh and not periodic_task and direct:
            again = await analyzer.decide_mcp_tools(
                request_text, tools, started=started,
                extra=mcp_store.EMPTY_DECISION_NOTE)
            usage = merge_usage(usage, dict(analyzer.last_usage or {}))
            if again.get("calls"):
                calls = list(again["calls"])
                lines_pre.append(
                    "MCP: диспетчер ответил «данные не нужны», но запрос просит "
                    "внешние данные — переспросил ("
                    + ", ".join(f"{call['server']} · {call['tool']}" for call in calls)
                    + ").")
                if str(again.get("mode") or "") == "chain":
                    chain_mode = True
    if not calls:
        workspace_store.set_dialog_mcp(dialog, signature, request_text, [])
        return [], usage, lines_pre + [
            f"MCP: проверил внешние инструменты ({len(enabled)} "
            f"{'сервер' if len(enabled) == 1 else 'сервера'}, {len(tools)} "
            "инструментов) — для этого запроса данные не нужны."
        ], False
    # ФАЗА ДО ПОДТВЕРЖДЕНИЯ ПЛАНА: выполняем ТОЛЬКО ЧТЕНИЯ. Вызовы, которые что-то
    # меняют на сервере (сохранение набора, выгрузка файла, запуск сбора),
    # откладываются: пользователь утверждает план до любых побочных эффектов, а
    # файл не появляется в чате раньше плана.
    deferred_note = ""
    if not approved:
        reads, effects = mcp_store.split_calls(calls, tools)
        if effects:
            deferred_note = mcp_store.calls_note(effects)
            lines_pre.append(
                "MCP: до подтверждения плана выполняю только чтение — изменения "
                f"отложены до «ок»: {deferred_note}.")
        calls = reads
        if not calls:
            # Читать нечего, а изменения ждут подтверждения: цепочка остаётся
            # не доигранной и продолжится на первом шаге выполнения.
            workspace_store.set_dialog_mcp(dialog, signature, request_text, [],
                                           calls=[], chain={"pending": True})
            return [], usage, lines_pre + [
                "MCP: запрос выполняется внешними инструментами — изменения "
                "выполню после подтверждения плана."], False
    # Помечаем вызовы (действие или чтение) ДО сохранения: повтору это скажет,
    # что можно повторять, а что нет.
    mcp_store.mark_call_kinds(calls, tools)
    # Повторная запись под ключом, который задача уже использовала (правка запроса,
    # второй прогон той же задачи): флаг перезаписи ставим сами — иначе сервер
    # откажет, а данные не обновятся.
    for note in mcp_store.enforce_overwrite(
            calls, tools, (stored.get("chain") or {}).get("ids") or {}):
        lines_pre.append("MCP: повторная запись того же ключа — ставлю флаг "
                         f"перезаписи ({note}).")
    results = await mcp_store.async_run_calls(calls)
    workspace_store.set_dialog_mcp(dialog, signature, request_text, results,
                                   calls=calls)
    lines = lines_pre + [mcp_store.results_note(results)]
    # Результаты, которые легли в диалог: ИМЕННО они уходят модели системным
    # блоком и именно их читает отмена задачи. Помечаем в них ДЕЙСТВИЯ (запуск
    # сбора, подписки): модель видит строку «ДЕЙСТВИЯ … ВЫПОЛНЕНО/НЕ ВЫПОЛНЕНО»
    # и не выдаёт словами то, чего вызов не делал (см. mcp_store.block).
    stored_results = list(workspace_store.dialog_mcp(dialog).get("results") or [])
    mcp_store.mark_actions(stored_results, tools)
    # Что задача ЗАПУСТИЛА на серверах (сбор наблюдений и т. п.): запоминаем, чтобы
    # отмена или удаление задачи остановили это (см. _stop_mcp_started).
    fresh_started = mcp_store.started_calls(stored_results, tools)
    if fresh_started:
        known = workspace_store.add_mcp_started(dialog, fresh_started)
        lines.append(
            "MCP: задача запустила внешний сбор (" + "; ".join(
                f"{item['server']} · {item['tool']}" for item in fresh_started)
            + ") — он остановится вместе с задачей (отмена или удаление). "
            f"Всего идущих сборов у задачи: {len(known)}.")
    # ВТОРОЙ ВОПРОС ДИСПЕТЧЕРУ — когда первый выбор данных не дал:
    #   (а) вызовы только ЗАПУСТИЛИ сбор: id стал известен лишь сейчас, и данные
    #       надо прочитать по нему (в живой задаче запуск возобновил наблюдение с
    #       7 образцами, отчёт так и не был запрошен — модель пересказала ответ
    #       запуска как «результат отчёта»);
    #   (б) все вызовы ОТКАЗАЛИ (например, отчёт по придуманному id): без данных
    #       ответ либо пустой, либо выдуманный — выбор делаем заново, уже с
    #       блоком про идущие сборы этой задачи.
    started_now = workspace_store.mcp_started(dialog)
    missed_reads = bool(calls) and mcp_store.has_reads(calls, tools) \
        and not any(item.get("ok") for item in results)
    if (fresh_started and not mcp_store.has_reads(calls, tools)) or missed_reads:
        reason = ("сбор запущен — читаю данные по нему"
                  if fresh_started and not missed_reads
                  else "прошлые вызовы отказали — выбираю инструменты заново")
        more = await analyzer.choose_mcp_tools(
            request_text, tools, started=started_now, need_reads=not fresh_started)
        usage = merge_usage(usage, dict(analyzer.last_usage or {}))
        reads = mcp_store.read_calls(more, tools)
        if reads:
            mcp_store.mark_call_kinds(reads, tools)
            # ВАЖНО: заметка строится по РЕЗУЛЬТАТАМ, а не по вызовам — иначе в
            # чат уходило «источник отказал» про вызов, который на самом деле
            # отработал (такую строку видел пользователь при живом отчёте).
            fresh_results = await mcp_store.async_run_calls(reads)
            results = results + fresh_results
            calls = calls + reads
            workspace_store.set_dialog_mcp(dialog, signature, request_text,
                                           results, calls=calls)
            stored_results = list(
                workspace_store.dialog_mcp(dialog).get("results") or [])
            mcp_store.mark_actions(stored_results, tools)
            lines.append(
                f"MCP: {reason} ("
                + ", ".join(f"{call['server']} · {call['tool']}" for call in reads)
                + ").")
            lines.append(mcp_store.results_note(fresh_results))
    # ГДЕ ЖДАТЬ «ок», А ГДЕ НЕТ. В пути с планом поведение прежнее: цепочка до
    # подтверждения только ЧИТАЕТ и помечается не доигранной, а зависимые чтения
    # продолжаются на первом шаге выполнения (проверяется check_mcp, раздел [10]).
    # А в ПРЯМОМ ОТВЕТЕ плана и подтверждения нет вовсе, поэтому откладывать там
    # — значит не выполнить никогда: ровно на этом споткнулся живой случай
    # «погода в городе человека» (агент нашёл город по реестру и на этом закончил).
    # Исключение — запрос, который ПРОСИТ РЕЗУЛЬТАТ (сохранить, выгрузить): у него
    # есть своя механика «уговора» и доигрывания, и ломать её нельзя.
    write_chain = bool(signals.get("needed"))
    if chain_mode and not approved and not (direct and not write_chain):
        # ДО ПОДТВЕРЖДЕНИЯ ПЛАНА ПИШУЩИЕ РАУНДЫ ЦЕПОЧКИ НЕ НУЖНЫ: данные для плана
        # уже прочитаны, а результат (сохранение, выгрузка) всё равно откладывается
        # до «ок» — значит спрашивать диспетчера не о чем, его предложения были бы
        # тут же отложены.
        # А ВОТ ЧИТАЮЩАЯ цепочка (структурный признак: остались невызванные
        # инструменты) идёт СРАЗУ, в том числе до «ок»: таков путь прямого ответа,
        # где подтверждения плана нет вовсе. Иначе «погода в городе человека»
        # отвечала по списку жителей, не сделав второго вызова, — а второго шанса
        # у неё не было. Побочные эффекты внутри цикла откладываются сами
        # (`split_calls` при `approved=False`), поэтому читать безопасно. Каждый такой вопрос — целый вызов LLM с самым дорогим
        # промптом (живой замер: 5–8 тыс. входных токенов), и он тратился впустую.
        # Помечаем цепочку не доигранной: после «ок» её продолжит _resume_mcp_chain.
        keep_ids = mcp_store.extract_ids(results)
        for key, value in mcp_store.produced_ids(calls).items():
            keep_ids.setdefault(key, value)
        workspace_store.set_dialog_mcp(
            dialog, signature, request_text, results, calls=calls,
            chain={"ids": keep_ids,
                   "keys": sorted(mcp_store.call_key(call) for call in calls),
                   "iterations": 0, "pending": True})
        lines.append("MCP: цепочка продолжится после подтверждения плана — результат "
                     "(сохранение и файл) выдаётся на последнем шаге.")
    elif chain_mode:
        # ЦЕПОЧКА: следующие вызовы строятся по РЕЗУЛЬТАТАМ (сохранить данные,
        # выгрузить файл по id из ответа сервера). Границы — итерации, суммарные
        # вызовы и время (см. _mcp_chain).
        results, calls, usage, lines = await _mcp_chain(
            dialog, request_text, tools, results, calls, analyzer, usage, signals,
            signature, deadline=deadline,
            started=workspace_store.mcp_started(dialog),
            chain_state=stored.get("chain"), lines=lines, approved=approved)
    return results, usage, lines, True


async def _resume_mcp_chain(dialog: Dict[str, Any], signature: str, request_text: str,
                            tools: List[Dict[str, Any]], analyzer: Agent,
                            stored: Dict[str, Any], deadline: float,
                            started: Optional[List[Dict[str, Any]]] = None,
                            lines_pre: Optional[List[str]] = None,
                            approved: bool = True
                            ) -> Tuple[List[Dict[str, Any]], Dict[str, Any],
                                       List[str], bool]:
    """Доигрывает цепочку после подтверждения плана — на текущем шаге выполнения.

    Гибридная схема: до «ок» выполнялись только ЧТЕНИЯ (по ним построен план), а
    зависимые вызовы остались отложенными. План подтверждён — цепочка продолжается
    с того же места: диспетчер видит прежние результаты и выбирает следующие вызовы
    сам (прогноз по городам из реестра, сохранение, затем выгрузка по id из ответа
    сервера). Границы, защита и идемпотентность — те же (см. `_mcp_chain`).

    Доигрывается на ПЕРВОМ шаге выполнения, а не на последнем: шаг «получить
    данные» обязан идти с данными. `approved` решает только одно — разрешены ли
    вызовы с побочными эффектами (сохранение, выгрузка): они ждут последнего шага
    плана, а чтения добываются сразу. Если эффект выбран на промежуточном шаге, он
    снова откладывается, и цепочка остаётся не доигранной до последнего шага.

    Результаты и вызовы прежней фазы берутся из диалога, поэтому чтения не
    повторяются: их ключи уже в `done_keys`.
    """
    chain_state = stored.get("chain") or {}
    results = list(stored.get("results") or [])
    calls = list(stored.get("calls") or [])
    signals = mcp_store.chain_signals(tools, request_text, expected=len(calls))
    lines = list(lines_pre or [])
    lines.append("MCP: план подтверждён — доигрываю цепочку: добираю данные по "
                 "зависимостям" + ("" if approved else
                                   ", побочные эффекты ждут последнего шага") + ".")
    results, calls, usage, lines = await _mcp_chain(
        dialog, request_text, tools, results, calls, analyzer, {}, signals,
        signature, deadline=deadline, started=started,
        chain_state=chain_state, lines=lines, approved=approved)
    return results, usage, lines, True


async def _mcp_chain(dialog: Dict[str, Any], request_text: str,
                     tools: List[Dict[str, Any]], results: List[Dict[str, Any]],
                     calls: List[Dict[str, Any]], analyzer: Agent,
                     usage: Dict[str, Any], signals: Dict[str, Any],
                     signature: str, deadline: float,
                     started: Optional[List[Dict[str, Any]]] = None,
                     chain_state: Optional[Dict[str, Any]] = None,
                     lines: Optional[List[str]] = None,
                     approved: bool = True
                     ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]],
                                Dict[str, Any], List[str]]:
    """ЦЕПОЧКА вызовов по результатам (agent loop) — вторая половина гибридной схемы.

    Каждый раунд: диспетчер получает запрос, список инструментов и ДАЙДЖЕСТ уже
    выполненных вызовов (короткие результаты, найденные идентификаторы, подсказки
    «что теперь возможно») и решает, нужен ли СЛЕДУЮЩИЙ вызов. Так собираются
    задачи с зависимостью по данным: «получи прогноз → сохрани → выгрузи Excel»,
    где идентификатор сохранённого набора приходит только из ответа сервера.

    Останавливаемся, когда: диспетчер сказал «готово»; новых вызовов нет;
    исчерпан бюджет вызовов или раундов; вышло время. Недопустимые вызовы
    (выдуманный идентификатор, разрушительный без просьбы пользователя, повтор
    уже выполненного) отбрасываются ДО выполнения — см. mcp_store.guard_calls.

    Возвращает накопленные результаты, вызовы, расход и строки диагностики.
    """
    lines = list(lines or [])
    done_keys = {mcp_store.call_key(call) for call in calls}
    chain_ids = mcp_store.extract_ids(results)
    for key, value in ((chain_state or {}).get("ids") or {}).items():
        chain_ids.setdefault(str(key), str(value))
    # Идентификаторы, которые НАЗВАЛИ создающие вызовы (save_/create_/…): их можно
    # переиспользовать дальше (выгрузить файл по сохранённому набору, обновить его
    # на повторе), хотя в результатах их ещё не было.
    issued = dict(mcp_store.produced_ids(calls))
    issued.update(chain_ids)
    # Уточнение «запрос просил сохранить/выгрузить — вызова не было»: уходит в
    # условие СЛЕДУЮЩЕГО раунда (см. deliver_note). `nudged` не даёт уговаривать
    # диспетчера повторно — иначе это превратилось бы в цикл уговоров.
    nudge_pending = ""
    nudged = False
    # До подтверждения плана изменения откладываются: признак «цепочка не
    # доиграна» уходит в диалог, и после «ок» её продолжит `_resume_mcp_chain`.
    pending = False
    deferred_names: List[str] = []
    # Очевидная достройка (см. auto_followup): вычисляется после раунда и
    # выполняется следующим раундом БЕЗ вызова диспетчера.
    auto_call: Optional[Dict[str, Any]] = None
    for round_number in range(1, mcp_store.MAX_CHAIN_ITERATIONS + 1):
        if time.monotonic() > deadline:
            lines.append("MCP: время, отведённое на цепочку, вышло — останавливаюсь "
                         "на том, что уже получено.")
            break
        remaining = mcp_store.MAX_TOTAL_CALLS_PER_REQUEST - len(calls)
        if remaining <= 0:
            lines.append(f"MCP: предел вызовов за запрос "
                         f"({mcp_store.MAX_TOTAL_CALLS_PER_REQUEST}) достигнут — "
                         "цепочка остановлена.")
            break
        if auto_call is not None:
            # ОЧЕВИДНАЯ ДОСТРОЙКА БЕЗ МОДЕЛИ: остался ровно один невызванный
            # инструмент результата, и все его обязательные аргументы — уже
            # известные идентификаторы (см. auto_followup). Спрашивать
            # диспетчера незачем: это целый вызов LLM с самым дорогим промптом.
            allowed, rejected = [auto_call], []
            lines.append("MCP: остался один очевидный шаг результата — вызываю "
                         f"без диспетчера ({auto_call['server']} · {auto_call['tool']}).")
            auto_call = None
        else:
            extra = mcp_store.chain_note(round_number, mcp_store.MAX_CHAIN_ITERATIONS,
                                         remaining, signals)
            if nudge_pending:
                # ПРИНУДИТЕЛЬНОЕ УТОЧНЕНИЕ: диспетчер непостоянен — на один и тот
                # же запрос он то строит цепочку, то решает «данных хватает». Если
                # запрос просит сохранить/выгрузить, а вызовов для этого не было,
                # называем подходящие инструменты прямо (см. deliver_note).
                # Уточнение уходит РОВНО в один следующий раунд.
                extra += "\n\n" + nudge_pending
                nudge_pending = ""
            decision = await analyzer.decide_mcp_tools(
                request_text, tools, started=started,
                extra=extra,
                results=results,
                remaining=remaining,
                chain_state={"ids": chain_ids} if chain_ids else None)
            usage = merge_usage(usage, dict(analyzer.last_usage or {}))
            allowed, rejected = mcp_store.guard_calls(
                decision.get("calls"), results, tools,
                user_text=request_text, done_keys=done_keys, issued_ids=issued)
        for item in rejected:
            lines.append(f"MCP: вызов {item['server']} · {item['tool']} отброшен — "
                         f"{item['reason']}.")
        if not approved:
            # ДО ПОДТВЕРЖДЕНИЯ ПЛАНА раунд выполняет только ЧТЕНИЯ: вызовы,
            # меняющие что-то на сервере (сохранение, выгрузка, запуск сбора),
            # откладываются до «ок» — пользователь утверждает план до побочных
            # эффектов, а файл не появляется в чате раньше плана.
            reads, effects = mcp_store.split_calls(allowed, tools)
            if effects:
                names = mcp_store.calls_note(effects)
                deferred_names.extend(str(call.get("tool") or "") for call in effects)
                # Формулировка честна в обоих случаях: до «ок» (плана ещё нет) и на
                # промежуточном шаге после «ок» — изменения в любом случае ждут
                # ПОСЛЕДНЕГО шага плана.
                lines.append("MCP: изменения сейчас не выполняю — сохранение и "
                             f"выгрузка идут на последнем шаге плана: {names}.")
            allowed = reads
        allowed = allowed[:remaining]
        if not allowed:
            reason = str((decision or {}).get("reason") or "").strip()
            if not approved and deferred_names:
                # Отложенная часть есть, но выполнять её до «ок» нельзя: цепочка
                # остаётся НЕ ДОИГРАННОЙ и продолжится на первом шаге выполнения.
                pending = True
                lines.append("MCP: цепочка продолжится сразу после подтверждения плана.")
                keep_ids = dict(chain_ids)
                for key, value in issued.items():
                    keep_ids.setdefault(key, value)
                workspace_store.set_dialog_mcp(
                    dialog, signature, request_text, results, calls=calls,
                    chain={"ids": keep_ids, "keys": sorted(done_keys),
                           "iterations": round_number, "pending": True})
                break
            # Запрос просил «сделать с данными ещё шаг», а вызовов для этого нет:
            # спрашиваем диспетчера ЕЩЁ РАЗ, назвав невызванные инструменты, которые
            # создают результат (файл, набор, отчёт). Один раз на цепочку — иначе
            # это превратилось бы в цикл уговоров.
            #
            # ТОЛЬКО ПРИ `approved` (последний шаг плана). Уточнение требует
            # СОЗДАТЬ результат, а эффекты на промежуточном шаге всё равно
            # откладываются: уточнение там просило бы то, что нельзя выполнить, и —
            # так как `nudged` живёт в пределах одного запроса — повторялось бы на
            # КАЖДОМ шаге, сжигая самый дорогой вызов LLM.
            if approved and not nudged and not rejected:
                done_pairs = {(str(call.get("server") or ""), str(call.get("tool") or ""))
                              for call in calls}
                candidates = mcp_store.delivery_candidates(
                    tools, done_pairs,
                    kinds=mcp_store.unsatisfied_kinds(request_text, calls))
                if candidates:
                    nudged = True
                    nudge_pending = mcp_store.deliver_note(candidates, request_text)
                    lines.append(
                        "MCP: запрос просит сохранить/выгрузить, а вызовов для этого "
                        "не было — уточняю задачу диспетчеру ("
                        + ", ".join(item["tool"] for item in candidates[:5]) + ").")
                    continue
            if (decision or {}).get("done") and not rejected:
                lines.append("MCP: цепочка завершена — "
                             + (reason or "всё нужное выполнено") + ".")
            else:
                lines.append("MCP: допустимых новых вызовов нет — цепочка закончена."
                             if rejected else
                             "MCP: новых вызовов не требуется — цепочка закончена.")
            break
        mcp_store.mark_call_kinds(allowed, tools)
        # Раунд цепочки повторно пишет под уже известным ключом? Ставим флаг
        # перезаписи сами (сервер иначе отказывает) — до выполнения вызовов.
        known_ids = dict(chain_ids)
        for key, value in issued.items():
            known_ids.setdefault(key, value)
        for note in mcp_store.enforce_overwrite(allowed, tools, known_ids):
            lines.append("MCP: повторная запись того же ключа — ставлю флаг "
                         f"перезаписи ({note}).")
        fresh_results = await mcp_store.async_run_calls(allowed, limit=remaining)
        results = list(results) + fresh_results
        calls = list(calls) + allowed
        done_keys |= {mcp_store.call_key(call) for call in allowed}
        chain_ids.update(mcp_store.extract_ids(fresh_results))
        # Идентификаторы, названные создающими вызовами (save_/create_/…): сервер их
        # принял, значит они существуют. Храним их вместе с цепочкой — повтор
        # периодической задачи обновляет ТОТ ЖЕ набор, а не заводит новый.
        issued.update(mcp_store.produced_ids(allowed))
        keep_ids = dict(chain_ids)
        for key, value in issued.items():
            keep_ids.setdefault(key, value)
        # Сохраняем ПОСЛЕ каждого раунда: шаги плана и проверка результата видят
        # всю цепочку, а повтор периодической задачи — её идентификаторы.
        # `pending` остаётся True, если раунд отложил СОЗДАЮЩИЕ вызовы: иначе
        # отложенный эффект потерялся бы — цепочка считалась бы доигранной, и на
        # последнем шаге её уже никто не продолжил бы (живой случай: раунд прочитал
        # данные И попросил сохранение — сохранение не выполнялось никогда).
        stored = workspace_store.set_dialog_mcp(
            dialog, signature, request_text, results, calls=calls,
            chain={"ids": keep_ids, "keys": sorted(done_keys),
                   "iterations": round_number,
                   "pending": bool(deferred_names) and not approved})
        mcp_store.mark_actions(stored.get("results") or [], tools)
        fresh_started = mcp_store.started_calls(stored.get("results") or [], tools)
        if fresh_started:
            workspace_store.add_mcp_started(dialog, fresh_started)
            lines.append("MCP: задача запустила внешний сбор ("
                         + "; ".join(f"{item['server']} · {item['tool']}"
                                     for item in fresh_started)
                         + ") — он остановится вместе с задачей.")
        lines.append(f"MCP: раунд {round_number} цепочки — "
                     + ", ".join(f"{call['server']} · {call['tool']}"
                                 for call in allowed) + ".")
        lines.append(mcp_store.results_note(fresh_results))
        # ФАЙЛ ВЫДАН — ЦЕПОЧКА ЗАКОНЧЕНА. Если раунд принёс вложение (файл), а
        # невызванных инструментов, создающих результат, больше нет, спрашивать
        # диспетчера «всё ли готово» незачем: это ещё один ПОЛНЫЙ вызов LLM
        # (промпт диспетчера — самая дорогая часть задачи), а решение очевидно.
        if any(item.get("attachments") for item in fresh_results):
            done_pairs = {(str(call.get("server") or ""), str(call.get("tool") or ""))
                          for call in calls}
            if not mcp_store.delivery_candidates(tools, done_pairs):
                lines.append("MCP: файл выдан, других инструментов результата нет — "
                             "цепочка завершена.")
                break
        # Осталась одна очевидная достройка результата? Выполним её без модели
        # (см. auto_followup): решение однозначно, а вызов диспетчера — самый
        # дорогой вызов задачи. Берём только те виды результата, которые запрос
        # ещё ждёт: иначе второй включённый сервер со своим невызванным
        # save-инструментом делал кандидатов «двумя», и достройка не срабатывала.
        if approved and not pending:
            done_pairs = {(str(call.get("server") or ""), str(call.get("tool") or ""))
                          for call in calls}
            auto_call = mcp_store.auto_followup(
                tools, done_pairs, keep_ids,
                kinds=mcp_store.unsatisfied_kinds(request_text, calls))
    else:
        lines.append(f"MCP: исчерпан предел раундов цепочки "
                     f"({mcp_store.MAX_CHAIN_ITERATIONS}) — останавливаюсь на "
                     "достигнутом.")
    return results, calls, usage, lines


def _looks_like_data_table(text: str) -> bool:
    """Похож ли ответ на ТАБЛИЦУ с числами (markdown-строки «| … | … |»).

    Нужно только для защиты от выдумки: если внешние данные не получены вовсе,
    таблица значений — это не «сводка», а придуманные числа (регресс живой задачи:
    отчёт отказал, а таблица за сутки по часам всё равно появилась).
    """
    rows = [line for line in str(text or "").splitlines()
            if line.strip().startswith("|") and line.count("|") >= 2]
    if len(rows) < 2:
        return False
    return any(any(char.isdigit() for char in row) for row in rows)


def _mcp_fabrication_guard(text: str, mcp_data: Any) -> str:
    """Заменяет выдуманную таблицу честным отказом ("" — заменять не надо).

    Срабатывает УЗКО: только когда внешние инструменты вызывались, но НИ ОДИН не
    дал данных, а в ответе таблица с числами. Такой ответ не показываем: вместо
    него пользователь видит причину отказа инструмента. Обычный текст про
    отсутствие данных под правило не попадает (таблицы в нём нет) и уходит как есть.
    """
    data = mcp_store.normalize_results(mcp_data)
    if not data or any(item.get("ok") for item in data):
        return ""
    if not _looks_like_data_table(text):
        return ""
    reasons = "; ".join(
        f"{item['server']} · {item['tool']}: {item['error'] or 'данных нет'}"
        for item in data)
    return ("⚠️ Данные от внешних инструментов не получены, поэтому таблицу не "
            "привожу: значения пришлось бы выдумать.\n"
            f"Причина: {reasons}")


def _replayable_calls(stored: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Вызовы прошлого повтора, которые МОЖНО повторить: чтение и успех.

    Повтор берёт те же запросы данных (выбор инструментов не оплачивается заново),
    но только те, что реально дали данные: отказавший вызов (например, отчёт по
    несуществующему id) при повторе откажет снова, а ответ уйдёт без фактов.
    Успех сверяем по сохранённым результатам: сервер + инструмент + аргументы.
    """
    results = [item for item in (stored.get("results") or []) if isinstance(item, dict)]
    ok_keys = {
        (str(item.get("server") or ""), str(item.get("tool") or ""),
         mcp_store.arguments_text(item.get("arguments")))
        for item in results if item.get("ok")
    }
    replay: List[Dict[str, Any]] = []
    for call in mcp_store.read_calls(stored.get("calls")):
        key = (str(call.get("server") or ""), str(call.get("tool") or ""),
               mcp_store.arguments_text(call.get("arguments")))
        if key in ok_keys:
            replay.append(call)
    return replay


def _mcp_debug() -> str:
    """Строка диагностики перед выбором инструментов (что именно происходит)."""
    return ("MCP: проверяю внешние инструменты проекта (служебный вызов LLM) — "
            "какие данные нужны для этого запроса.")


def _rag_alternatives(task: Dict[str, Any]) -> List[str]:
    """Чем агент может продолжить, если в базах знаний ответа нет.

    Список уходит в блок «в документах этого нет» (`rag_search.NO_HITS_NOTE`), и
    модель обязана ПРЕДЛОЖИТЬ эти варианты пользователю — иначе на вопрос, ответа
    на который в документах нет, агент просто молчит или уходит в общие знания.

    Варианты берутся только из того, что у ПРОЕКТА действительно есть: внешние
    инструменты (MCP) — из КЭША обнаружения, без подключения к серверам (тянуть
    живое соединение ради подсказки нельзя: это замедлило бы каждый запрос).
    Плюс два варианта, которые есть всегда: ответ по общим знаниям с честной
    пометкой и уточнение запроса/пополнение базы.
    """
    out: List[str] = []
    for server_id in workspace_store.mcp_enabled(task):
        entry = mcp_store.find_server(server_id) or {}
        tools = mcp_store.cached_tools(server_id)[:6]
        names = [str(item.get("name") or "").strip() for item in tools]
        names = [name for name in names if name]
        if not names:
            continue
        out.append("поискать во внешних инструментах проекта (MCP, сервер «%s»): %s "
                   "— для этого напишите, что искать, и я вызову их следующим запросом"
                   % (entry.get("title") or entry.get("name") or server_id,
                      ", ".join(names)))
    # Варианты, которые есть всегда, живут в модуле поиска: они не зависят от
    # настроек проекта, и без них блок «в документах этого нет» снова был бы
    # тупиком (в том числе при прямых вызовах поиска, мимо веб-слоя).
    out.extend(rag_search.DEFAULT_ALTERNATIVES)
    return out


def _rag_general_state(task: Dict[str, Any], text: str, *, reuse: bool,
                       machine_step: bool) -> str:
    """Можно ли по ЭТОМУ запросу отвечать без документов. ЧЕТЫРЕ состояния:

      * "allowed" — пользователь разрешил отвечать ПО ОБЩИМ ЗНАНИЯМ сам (список
        формулировок — `rag_search.allows_general_answer`): он выбрал вариант,
        который агент предложил, и вопрос задавать больше не о чем;
      * "continue" — он выбрал ПРОДОЛЖЕНИЕ ВНЕ ДОКУМЕНТОВ (внешние инструменты
        проекта, `rag_search.allows_continuation`): данные инструментов — не общие
        знания модели, и правила для них другие. Состояние нужно ровно затем,
        чтобы не спросить «как продолжить» ВТОРОЙ раз (это зациклило бы задачу);
      * "ask" — базы включены, настройка проекта «спрашивать, если в документах
        ничего нет» включена, и это НОВЫЙ запрос пользователя: отвечать без его
        согласия нельзя (см. `_rag_choice_view`);
      * "" — спрашивать некого или нечего: служебный шаг плана, подтверждение
        плана, автономный прогон периодической задачи, выключенная настройка.
        Тогда агент отвечает, но ОБЯЗАН пометить, что это не из документов.

    Служебные реплики (`reuse`, `machine_step`) не спрашивают намеренно: шаг плана
    и проверка выполняются отдельными запросами, и вопрос «как продолжить» на
    каждом из них превратился бы в цикл.
    """
    if reuse or machine_step:
        return ""
    if rag_search.allows_general_answer(text):
        return "allowed"
    if rag_search.allows_continuation(text):
        return "continue"
    return "ask" if workspace_store.rag_settings(task).get("ask_when_empty", True) else ""


def _rag_choice_view(data: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
    """Сообщение о том, что в документах ничего нет, и варианты продолжения.

    Отправляется ВМЕСТО ответа: агент не уходит сам в общие знания, а
    останавливается и ждёт решения пользователя — ровно то, что просили («конечное
    решение же должен предоставлять юзер, если в базе ничего не найдено»).

    ВАРИАНТЫ — ДЕЙСТВИЯ, А НЕ ЗАГОТОВКИ. У каждого варианта есть ровно один способ
    сработать, и он описан данными:

      * `send` — готовая ФРАЗА, которую интерфейс отправляет как запрос
        пользователя: по ней сервер узнаёт решение (`rag_search.allows_general_answer`
        / `allows_continuation`), поэтому формулировки задаёт код, а не модель;
      * `apply` — ПРАВКА НАСТРОЙКИ перед повторным поиском (сейчас это снижение
        порога уверенности до значения, которое пропустит лучший из отсечённых
        фрагментов): интерфейс сначала применяет настройку, потом отправляет
        `send` — и поиск идёт заново, уже с новым порогом;
      * `action` — действие в интерфейсе, когда отправлять нечего (`clarify` —
        поставить курсор в поле ввода: уточнять формулировку может только человек).

    Мёртвых кнопок здесь быть не должно: раньше варианты без готовой фразы просто
    гасились, и предложение «снизить порог» ничего не делало.
    """
    bases = "; ".join(
        "%s — %s" % (item.get("name") or "база", rag_search._base_outcome(item))
        for item in (data.get("bases") or [])) or "нет"
    stages = data.get("stages") or {}
    cut_kind = rag_search.cut_kind(data)
    cut = rag_search._cut_by_threshold(data)
    # Запрос, по которому шёл поиск: его повторяют варианты, меняющие настройку —
    # «снизить порог и поискать снова» обязано искать то же самое, иначе это была
    # бы подмена вопроса.
    request = str(data.get("request") or data.get("query") or "").strip()
    # У ДВУХ ПОРОГОВ ДВЕ ШКАЛЫ, и в сообщении это названо прямо: путать «порог
    # первичной релевантности» (0…2, косинус + слова запроса — то число, что видно
    # в карточке источника) и «порог уверенности модели» (0…1, второй этап) нельзя
    # — ровно на этой путанице и строился живой случай 03.10.
    if cut_kind == "ce":
        head = ("⚠ В документах проекта есть близкие фрагменты, но все они ниже "
                "порога УВЕРЕННОСТИ МОДЕЛИ (порог %s) — это порог ВТОРОГО этапа, "
                "реранкинга, поэтому в ответ они не пошли."
                % _fmt_score(stages.get("min_ce")))
    elif cut_kind == "score":
        head = ("⚠ В документах проекта есть близкие фрагменты, но все они ниже "
                "порога ПЕРВИЧНОЙ РЕЛЕВАНТНОСТИ (порог %s — косинус + слова "
                "запроса, то число, что видно в карточке источника)."
                % _fmt_score(stages.get("min_score")))
    else:
        head = "⚠ В документах проекта по этому запросу ничего не нашлось."
    text = (head + "\n\nПоиск уже выполнен по всем включённым базам: " + bases
            + ".\n\nОтвечать по общим знаниям без вашего решения я не буду — "
              "выберите, как продолжить (или напишите свой вариант):")
    options: List[Dict[str, Any]] = []
    if cut:
        # СНИЖАЕМ ТОТ ПОРОГ, КОТОРЫЙ ОТСЁК: у каждого своя настройка, и совет
        # «снизить порог» без указания, какой именно, отправлял бы человека не туда.
        if cut_kind == "ce":
            field, best = "min_ce", float(stages.get("cut_best_ce") or 0.0)
            which = "уверенности модели"
            limit = 1.0
        else:
            field, best = "min_score", float(stages.get("cut_best_score") or 0.0)
            which = "первичной релевантности"
            limit = rag_search.MIN_SCORE_LIMIT
        target = _relax_target(best)
        # Слабый лучший фрагмент: снижать нечего, и это сказано словами, а не
        # мёртвой кнопкой. Для порога уверенности «слабо» — ниже 0,2 (у мусора
        # 0,00–0,02), для первичной релевантности — ниже базового отсева шума.
        weak = best < (RELAX_MIN_BEST if field == "min_ce"
                       else float(stages.get("floor") or 0.0))
        if not weak and request:
            options.append({
                "title": "Снизить порог %s до %s и повторить поиск"
                         % (which, _fmt_score(target)),
                "details": "лучший из отсечённых фрагментов имел %s — порог "
                           "опустится чуть ниже него, и поиск пойдёт заново по "
                           "тому же запросу" % _fmt_score(best),
                "send": request,
                "apply": {field: target},
            })
        else:
            options.append({
                "title": "Снизить порог %s и поискать снова" % which,
                "details": "порог %s отсёк всё, но и лучший из отсечённых "
                           "фрагментов слабый (%s): похоже, в документах ответа "
                           "действительно нет — снижать порог нечего"
                           % (_fmt_score(stages.get(field)), _fmt_score(best)),
                "send": "",
            })
    options.append({
        "title": "Ответить по общим знаниям",
        "details": "ответ будет помечен как «не из ваших документов»",
        "send": rag_search.GENERAL_CHOICE,
    })
    for item in (data.get("alternatives") or []):
        low = str(item).lower()
        if "mcp" not in low and "инструмент" not in low:
            continue
        mcp_phrase = ("%s: %s" % (rag_search.MCP_CHOICE, request or "")).strip()
        options.append({
            "title": "Поискать во внешних инструментах проекта",
            "details": "данные придут от инструментов проекта (MCP), а не из "
                       "документов; поиск пойдёт по тому же запросу",
            "send": mcp_phrase,
        })
    options.append({
        "title": "Уточнить вопрос",
        "details": "формулировку, термин или контекст — напишите уточнение, и я "
                   "начну новый поиск по нему (можно и добавить документ в базу)",
        "send": "",
        "action": "clarify",
    })
    return {"message": text, "options": options, "kind": "rag_empty",
            "bases": data.get("bases") or []}


# Снижать порог есть смысл, только если отсечённое было ПОХОЖЕ на ответ: у мусора
# оценка 0,0–0,2, у нужного фрагмента — 0,45–1,00 (замер cross-encoder). Ниже
# этого значения предложение «снизить порог» было бы советом показать модели шум.
RELAX_MIN_BEST = 0.2


def _relax_target(best: Any) -> float:
    """До какого значения снизить порог, чтобы пропустить лучший отсечённый фрагмент.

    Шаг — 0,05 вниз от его оценки: попап «снизить до 0» показывал бы модели весь
    шум, а попап «снизить на волосок» не пропустил бы ничего. Ноль остаётся
    законным значением («не отсекать»), но получается только из нулевой оценки.
    Одна функция на ОБА порога: шкалы у них разные, а правило шага — одно.
    """
    try:
        value = float(best or 0.0)
    except (TypeError, ValueError):
        value = 0.0
    return max(0.0, round(value - 0.05, 2))


def _fmt_score(value: Any) -> str:
    """Оценка для человека: два знака и запятая (как в интерфейсе)."""
    try:
        return ("%.2f" % float(value or 0.0)).replace(".", ",")
    except (TypeError, ValueError):
        return "0,00"


def _rag_debug(settings: Any = None) -> str:
    """Строка диагностики перед поиском: что произойдёт С ЭТИМИ настройками.

    Раньше строка описывала архитектуру вообще («ДВА ЭТАПА… реранкинг… порог»),
    и при снятых галочках она обещала то, чего в поиске не будет — живое
    замечание 02.10. Поэтому текст собирается по ДЕЙСТВУЮЩИМ настройкам проекта:
    два этапа упоминаются только когда реранкинг включён, порог — только когда
    он задан и есть чем фильтровать, а переформулировка — только когда включена.

    Переформулировка запроса — единственное место, где RAG обращается к модели
    (служебный вызов, вид "rewrite"): без неё поиск идёт локально — вектор
    запроса считается локальными эмбеддингами, дальше перебор индекса.
    """
    data = settings if isinstance(settings, dict) else {}
    rerank = bool(data.get("rerank"))
    filter_on = bool(data.get("filter"))
    min_score = float(data.get("min_score") or 0.0)
    min_ce = float(data.get("min_ce") or 0.0)
    parts = ["RAG: ищу фрагменты в подключённых базах знаний проекта"]
    if rerank:
        engine = "cross-encoder" if str(data.get("rerank_backend") or "") == "cross-encoder" \
            else ("признаки" if str(data.get("rerank_backend") or "") == "features"
                  else rag_rerank.backend_name(rag_rerank.backend()))
        parts.append("ДВА ЭТАПА: сначала широкий пул кандидатов (до %d с базы) "
                     "векторным поиском, затем реранкинг пула (%s)"
                     % (rag_search.top_k_before(), engine))
    else:
        parts.append("реранкинг выключен, поэтому пула нет: беру %d лучших "
                     "фрагментов по вектору и словам" % rag_search.top_k())
    # ДВА ПОРОГА, ДВЕ ШКАЛЫ — и в дебаге они названы раздельно. Порог уверенности
    # модели принадлежит ВТОРОМУ этапу: без реранкинга он не применяется, и
    # подменять его первичной релевантностью нельзя (шкалы разные).
    if filter_on and min_score > 0:
        parts.append("фрагменты с первичной релевантностью ниже %.2f (косинус + "
                     "слова запроса — то число, что видно в карточке источника) "
                     "отсею до реранкинга" % min_score)
    if filter_on and min_ce > 0:
        if rerank:
            parts.append("фрагменты с уверенностью модели ниже %.2f отсею после "
                         "реранкинга" % min_ce)
        else:
            parts.append("порог уверенности модели %.2f не применяется: он "
                         "относится к реранкингу, а реранкинг выключен" % min_ce)
    if bool(data.get("rewrite")):
        parts.append("запрос перед поиском переформулирую служебным вызовом модели "
                     "(Query Rewrite: в базу идут ключевые слова, а не разговорный "
                     "вопрос; отключается галочкой в «Поиск и ответы»)")
    return ("; ".join(parts) + ". Поиск локальный — эмбеддинги, перебор индекса"
            + (" и реранкер" if rerank else "")
            + "; обращений к LLM за фрагментами нет.")


def _mcp_files_text(files: List[Dict[str, Any]]) -> str:
    """Подпись узла чата с файлами, полученными от MCP-инструментов.

    Сам файл показывается карточкой со ссылкой на скачивание (её рисует
    интерфейс по полю `files`); текст — короткое пояснение, что это и откуда.
    """
    if not files:
        return ""
    if len(files) == 1:
        item = files[0]
        origin = f" ({item['origin']})" if item.get("origin") else ""
        return (f"📎 Файл готов: {item['name']} — {item['size_text']}{origin}. "
                "Скачать можно по ссылке в карточке ниже.")
    return ("📎 Готовы файлы: "
            + "; ".join(f"{item['name']} ({item['size_text']})" for item in files)
            + ". Скачать можно по ссылкам в карточках ниже.")


async def _stop_mcp_started(session: Optional[Dict[str, Any]], reason: str) -> List[str]:
    """Останавливает ВНЕШНИЕ СБОРЫ, запущенные задачей (наблюдения, подписки).

    Задача отменена или удалена — значит и работа, которую она завела на серверах
    (например, сбор погоды каждые 15 минут), должна прекратиться: иначе сервер
    будет собирать и хранить данные для задачи, которой больше нет. Какие именно
    инструменты отменяют сбор, ЗНАЕТ СЕРВЕР: пары «запускающий ↔ отменяющий»
    найдены по объявленному списку, а id наблюдения сохранён вместе с задачей
    (см. app/ai/mcp.py, started_calls).

    Возвращает строки диагностики для журнала чата (пусто — отменять было нечего).
    Сбой отмены НЕ скрывается: пользователь должен знать, что сбор остался.
    """
    if not session:
        return []
    dialog = session.get("dialog")
    if not isinstance(dialog, dict):
        return []
    entries = workspace_store.mcp_started(dialog)
    if not entries:
        return []
    reports = await mcp_store.async_cancel_started(entries)
    # Снимаем только УСПЕШНО отменённые: не отменённые остаются в памяти задачи —
    # если задачу удаляют, попытка повторится у следующей задачи с теми же сборами.
    done = [entry for entry, report in zip(entries, reports) if report.get("ok")]
    workspace_store.clear_mcp_started(dialog, done)
    note = mcp_store.cancel_note(reports)
    if not note:
        return []
    line = f"{periodic_store.AUTO_MARK} Внешние сборы задачи: {note} ({reason})."
    workspace_store.add_log(dialog, workspace_store.LOG_DEBUG, line)
    logger.info("MCP: отмена внешних сборов задачи (%s): %s", reason, note)
    return [line]


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


async def _preflight_rag(task: Dict[str, Any], session: Dict[str, Any], text: str,
                         state: "task_state.TaskState", reuse: bool = False,
                         machine_step: bool = False, fresh: bool = False,
                         agent: Optional[Agent] = None
                         ) -> Tuple[Dict[str, Any], List[str], bool, Dict[str, Any]]:
    """Фрагменты баз знаний (RAG) по запросу — ДО этапа планирования.

    Возвращает (данные RAG, строки диагностики, «поиск выполнялся сейчас»,
    расход служебного вызова переформулировки).

    Подключённых баз у проекта нет — ни поиска, ни данных: запрос идёт как
    раньше, и в модель НЕ уходит блок с фрагментами (см. app/ai/rag_search.py).

    Поиск — ДВА ЭТАПА и, если включено, ПЕРЕФОРМУЛИРОВКА ЗАПРОСА. Настройки
    берутся у ПРОЕКТА (task["rag"], панель «Поиск и ответы»): реранкинг, порог,
    топ-K до и после. Переформулировка — служебный вызов модели (app/ai/rag_query.py);
    без модели работает тот же модуль локально, поэтому поиск не остаётся без
    запроса. Строка дебага «запрос после rewriting: «…»» говорит, по какому
    запросу искали и чем он получен.

    Фрагменты кладутся в dialog["rag"] вместе с подписью «включённые базы + их
    отпечаток + запрос задачи»: шаги плана выполняются отдельными HTTP-запросами,
    и каждый из них должен видеть те же документы, а векторный поиск стоит
    времени (вектор запроса + перебор индекса). `reuse` (служебная реплика шага,
    подтверждение плана, «работай автономно», «перезапусти») означает «это НЕ
    новый запрос»: искать по служебной фразе нельзя — по ней нашлось бы что
    угодно. Новый содержательный текст пользователя — наоборот, новый запрос:
    документы подбираются по НЕМУ, иначе ответ опирался бы на прежние фрагменты.

    `fresh` — АВТОЗАПУСК периодической задачи: запрос тот же, но документы могли
    обновиться (их переиндексировали), поэтому поиск идёт заново.

    Отпечаток индексов (`corpus_stamp`) важен отдельно: без него сохранённые
    фрагменты пережили бы переиндексацию базы, и агент отвечал бы по прежней
    версии документа.

    Поиск идёт В ПОТОКЕ: считается вектор запроса (модель эмбеддингов) и
    перебираются векторы индекса — цикл событий на это время блокировать нельзя,
    параллельные задачи ждали бы.
    """
    dialog = session["dialog"]
    enabled = workspace_store.rag_enabled(task)
    if not enabled:
        return {}, [], False, {}
    # Служебные фразы описывают ПРЕЖНИЙ запрос задачи (state.request), новый текст
    # пользователя — сам является запросом.
    request_text = ((state.request or text or "").strip() if (reuse or machine_step)
                    else (text or "").strip())
    profile = _current_profile_id()
    # Отпечаток индексов читается с диска (паспорта баз) — это дешёвая проверка
    # «документы те же?», и только по ней решается, нужен ли поиск вообще.
    stamp = await asyncio.to_thread(rag_search.corpus_stamp, enabled, profile)
    signature = rag_search.signature(enabled, request_text, stamp)
    stored = workspace_store.dialog_rag(dialog)
    # Данные по ЭТОМУ запросу уже собраны (шаг плана, проверка результата или
    # повтор того же запроса): искать заново нечего.
    if stored.get("signature") == signature and not fresh:
        return stored, [], False, {}
    if (reuse or machine_step) and not request_text:
        # Служебная реплика шага, а запроса задачи в состоянии НЕТ (задача пришла
        # из файла до первого шага): искать по служебной фразе нельзя — по ней
        # нашлось бы что угодно. Обычно же искать ЕСТЬ по чему: подпись могла не
        # совпасть из-за переиндексации базы или смены набора баз, а запрос задачи
        # (`state.request`) от этого не меняется — по нему поиск и идёт.
        return {}, [], False, {}
    settings = workspace_store.rag_settings(task)
    lines: List[str] = []
    usage: Dict[str, Any] = {}
    # ПЕРЕФОРМУЛИРОВКА ЗАПРОСА (Query Rewrite) — ДО поиска: в базу идёт строка
    # ключевых слов, а не разговорный вопрос. Выключена у проекта — ищем по
    # исходному тексту, и вызова модели нет вовсе.
    query_text = request_text
    rewrite_info: Dict[str, Any] = {}
    if settings.get("rewrite") and agent is not None:
        rewrite_info = await agent.rewrite_query(request_text)
        usage = merge_usage(usage, dict(agent.last_usage or {}))
        if rewrite_info.get("query"):
            query_text = str(rewrite_info["query"])
        line = rag_query.debug_line(rewrite_info)
        if line:
            lines.append(line)
            if str(rewrite_info.get("by") or "") and query_text != request_text:
                lines.append("ищу по обоим запросам: переформулированному «%s» "
                             "и исходному «%s» — так неточная переформулировка "
                             "ничего не теряет" % (query_text, request_text))
    # ПОИСК — В ПОТОКЕ: считается вектор запроса (модель эмбеддингов) и
    # перебираются векторы индекса; цикл событий на это время блокировать нельзя,
    # иначе параллельные задачи ждали бы чужой поиск.
    # Ищем по ОБОИМ запросам: переформулированному (если он был) и исходному тексту
    # пользователя. Переформулировка тогда может только добавить кандидатов, но не
    # отнять найденное по исходному запросу — живой случай: модель переписала
    # опечатку «ьестия» как есть, и поиск по одной такой строке нужного чанка не
    # находил.
    result = await asyncio.to_thread(rag_search.search, enabled, query_text,
                                     profile=profile, settings=settings,
                                     rewrite=rewrite_info,
                                     also=(request_text if rewrite_info else ""))
    # ЧЕМ ПРОДОЛЖИТЬ, если ответа в документах нет: список собирается ДО
    # сохранения, потому что его читает и блок для модели, и проверка результата.
    result["alternatives"] = _rag_alternatives(task)
    # Можно ли отвечать по общим знаниям по этому запросу — решает ПОЛЬЗОВАТЕЛЬ
    # (см. `_rag_general_state`): без его решения агент остановится и спросит.
    result["general"] = _rag_general_state(task, request_text, reuse=reuse,
                                           machine_step=machine_step)
    workspace_store.set_dialog_rag(dialog, signature, request_text, result)
    data = workspace_store.dialog_rag(dialog)
    note = rag_search.results_note(data)
    if note:
        lines.append(note)
    return data, lines, True, usage


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


def _gate_cache_key(steps: List[str], snapshot: Dict[str, Any]) -> tuple:
    """Ключ проверки шагов: сами шаги + действующие правила."""
    return (tuple(str(step) for step in steps or []),
            invariants_store.rules_signature(snapshot))


def _data_basis(results: Any, rag: Any = None) -> str:
    """ОСНОВА ДАННЫХ плана: какие внешние данные по запросу получены.

    План строится по фактическим данным (см. блок «ДАННЫЕ MCP»), поэтому смена
    основы — это другой план: в живой задаче первый выбор диспетчера был «запусти
    сбор», план вышел про наблюдение, а после исправления выбор стал «прочитай
    текущую погоду» — и прежний план к такой основе уже не подходит. Считаем только
    УСПЕШНЫЕ вызовы: план, построенный на отказе инструмента, — это план без данных,
    и его переиспользовать нельзя. Пустая строка — данных MCP нет (у проекта MCP
    выключен, запрос их не требует или все вызовы отказали).

    `rag` — фрагменты баз знаний по тому же запросу: они такой же вход плана, как
    и данные инструментов. База, из которой фрагменты НАШЛИСЬ, входит в основу по
    идентификатору: переиндексация базы (или её включение у проекта) меняет
    основу, и план не переиспользуется «по старой памяти».
    """
    pairs = sorted({f"{item.get('server')}·{item.get('tool')}"
                    for item in (results if isinstance(results, list) else [])
                    if isinstance(item, dict) and item.get("tool") and item.get("ok")})
    bases = sorted({str(hit.get("base_id") or "")
                    for hit in rag_search.hits_of(rag) if hit.get("base_id")})
    if bases:
        pairs.append("rag:" + ",".join(bases))
    return ", ".join(pairs)[:300]


def _plan_signature(text: str, snapshot: Dict[str, Any],
                    basis: str = "") -> Dict[str, str]:
    """Подпись плана: для какого ЗАПРОСА, при каких ПРАВИЛАХ и на КАКИХ данных.

    По ней видно, можно ли переиспользовать уже построенный план: если запрос
    тот же, правила не менялись и данные те же, новый вызов планировщика и
    код-гейт — это потраченные впустую токены (например, при перезапуске задачи
    после ошибки шага: `start_planning` шаги сохраняет, а гейт и план
    оплачивались заново).
    """
    return {
        "request": str(text or "").strip()[:task_state.REQUEST_LIMIT],
        "rules": invariants_store.rules_signature(snapshot),
        "basis": str(basis or "")[:300],
    }


def _plan_is_fresh(dialog: Optional[Dict[str, Any]], signature: Dict[str, str]) -> bool:
    """True, если план в диалоге построен ровно по этой подписи."""
    saved = dict((dialog or {}).get("plan_signature") or {})
    return bool(signature.get("request")) and saved == signature


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
    # Кэш ЖИВЁТ ВНУТРИ одного вызова: перепланирование иногда возвращает ровно
    # те же шаги, и повторная проверка того же текста — оплата за тот же
    # результат. За пределы запроса вердикт не выходит (там он мог бы устареть).
    checked: Dict[tuple, Any] = {}
    while True:
        cache_key = _gate_cache_key(steps, snapshot)
        if cache_key in checked:
            keep = checked[cache_key]
            lines.append("эти шаги уже проверялись в этом же запросе — беру "
                         "прежний вердикт (повторный вызов LLM не нужен).")
        else:
            keep = await analyzer.check_plan(steps, snapshot)
            usage = merge_usage(usage, dict(analyzer.last_usage or {}))
            if keep is not None:
                checked[cache_key] = keep
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
# MCP: внешние инструменты агента (кнопка «MCP» рядом с шестерёнкой проекта)
# ---------------------------------------------------------------------------
@router.get("/agent/mcp")
async def mcp_get(force: int = 0) -> dict:
    """Серверы MCP проекта: название, описание, инструменты, состояние галочек.

    Отдаёт {"servers": [{"id", "name", "description", "source", "enabled",
    "available", "error", "tools": [{"name", "title", "description"}, ...]}, ...],
    "enabled": [...], "counts": {"servers", "enabled", "tools", "available"},
    "project_id": ...} — этим снимком живёт диалог «MCP» и подпись кнопки.

    Серверы ОПРАШИВАЮТСЯ по-настоящему (initialize + tools/list): модалка
    показывает не только галочки, но и то, что за ними стоит, — сколько
    инструментов у сервера и работает ли он вообще. force=1 заставляет опросить
    заново (кнопка «обновить» в диалоге), иначе работает кэш в памяти процесса.
    """
    task = _current_task()
    return await _mcp_view_async(task, force=bool(force))


@router.post("/agent/mcp")
async def mcp_apply(payload: McpApply) -> dict:
    """Применяет набор включённых серверов MCP проекта (кнопка «применить»).

    Приходит ПОЛНЫЙ список галочек: сервер, которого в нём нет, выключается.
    Неизвестные id отбрасываются (см. workspace_store.set_mcp_enabled) — включить
    то, чего нет в реестре, нельзя. Обращений к модели и к серверам при записи
    НЕТ: настройка — это данные; серверы опрашиваются только чтобы показать
    состояние галочек в ответе.

    Данные прежнего запроса при этом сбрасываются (dialog["mcp"]): набор
    инструментов изменился, и старые данные могли быть получены сервером,
    который пользователь только что выключил.
    """
    task = _current_task()
    if task is None:
        raise HTTPException(
            status_code=400,
            detail="Сначала создайте проект — MCP-инструменты привязаны к проекту")
    async with _workspace_lock:
        workspace_store.set_mcp_enabled(task, payload.enabled)
        for session in task.get("sessions") or []:
            dialog = session.get("dialog")
            if isinstance(dialog, dict):
                workspace_store.set_dialog_mcp(dialog, "", "", [])
        await _persist()
    # Снимок собираем ПОСЛЕ записи и вне блокировки: опрос серверов запускает
    # процессы и занимает время — держать на нём блокировку всего workspace
    # нельзя (её ждали бы удаление задач, память, профили).
    return await _mcp_view_async(task)


# ---------------------------------------------------------------------------
# RAG: базы знаний проекта (кнопка «RAG» рядом с «MCP»)
#
# Пользователь загружает документы, они разбиваются на чанки, считается
# эмбеддинг каждого чанка, и всё это ложится локальным индексом (SQLite —
# рабочее, JSON — выгрузка) с метаданными. Перед каждым запросом агент ищет во
# ВКЛЮЧЁННЫХ у проекта базах фрагменты по вопросу, отвечает по ним и показывает
# источники под ответом (см. app/ai/rag_search.py и `_preflight_rag` ниже).
# Поиск идёт ЛОКАЛЬНО и обращений к LLM не делает.
# ---------------------------------------------------------------------------
def _rag_view(task: Optional[Dict[str, Any]], force: bool = False) -> Dict[str, Any]:
    """Снимок баз знаний для интерфейса (диалог «База знаний»).

    Отдаёт базы ТЕКУЩЕГО профиля с метриками и галочками проекта, доступные
    стратегии разбиения, состояние бэкенда эмбеддингов и хранилищ, пределы
    размеров. Ни модели, ни сети здесь нет: список баз читается с диска, поэтому
    диалог открывается мгновенно. `force` — кнопка «обновить»: перепроверить
    доступность модели эмбеддингов (см. rag.snapshot).
    """
    profile = _current_profile_id()
    settings = workspace_store.rag_settings(task) if task else {}
    data = rag.snapshot(profile=profile,
                        enabled_ids=workspace_store.rag_enabled(task),
                        settings=settings, force=force)
    data["project_id"] = (task or {}).get("id")
    return data


@router.get("/agent/rag")
async def rag_get(force: int = 0) -> dict:
    """Базы знаний профиля: метрики, галочки проекта, стратегии и пределы.

    Снимок живёт в диалоге «База знаний» и в подписи кнопки: сколько баз
    включено у проекта и сколько в них чанков. force=1 (кнопка «обновить»)
    заставляет перепроверить доступность модели эмбеддингов — если она
    появилась после запуска приложения, это видно без перезапуска.
    """
    return _rag_view(_current_task(), force=bool(force))


@router.post("/agent/rag")
async def rag_apply(payload: RagApply) -> dict:
    """Применяет набор включённых баз знаний проекта, параметры разбиения и настройки поиска.

    Приходит ПОЛНЫЙ список галочек: база, которой в нём нет, выключается.
    Несуществующие и ЧУЖИЕ (другого профиля) базы отбрасываются — включить то,
    чего у профиля нет, нельзя. Здесь же запоминаются настройки ПОИСКА (панель
    «Поиск и ответы»): переформулировка запроса, реранкинг, фильтрация по порогу,
    топ-K до и после второго этапа, порог релевантности. Обращений к модели нет:
    настройка — это данные.
    """
    task = _current_task()
    if task is None:
        raise HTTPException(
            status_code=400,
            detail="Сначала создайте проект — базы знаний привязаны к проекту")
    async with _workspace_lock:
        enabled = rag.filter_enabled(payload.enabled, profile=_current_profile_id())
        workspace_store.set_rag_enabled(task, enabled)
        workspace_store.set_rag_chunking(task, payload.strategy,
                                         payload.chunk_size, payload.overlap)
        settings_now = workspace_store.rag_settings(task)
        # Движок реранкинга выбирает ОКРУЖЕНИЕ (RAG_RERANK_BACKEND), а не проект:
        # движок один — модель, нужная фильтрации. Здесь только проверяем, что она
        # есть, прежде чем включать фильтр.
        backend_now = settings_now.get("rerank_backend") or rag_rerank.backend()
        # ЧТО ТРЕБУЕТ МОДЕЛИ, А ЧТО НЕТ. Порог ПЕРВИЧНОЙ РЕЛЕВАНТНОСТИ считается
        # без модели вообще — ему реранкер не нужен. А вот порог УВЕРЕННОСТИ
        # МОДЕЛИ без модели-реранкера применить нечем: вероятностей у признакового
        # бэкенда нет. Настроить такое «на будущее» можно, но включать реранкинг с
        # неработающим порогом — нет: это ровно та настройка, которая выглядит
        # действующей, а не действует.
        next_ce = payload.min_ce if payload.min_ce is not None \
            else settings_now.get("min_ce")
        next_rerank = payload.rerank if payload.rerank is not None \
            else settings_now.get("rerank")
        if payload.filter and next_rerank and float(next_ce or 0.0) > 0 \
                and not rag_rerank.filter_available(backend_now):
            raise HTTPException(
                status_code=400,
                detail="Порог уверенности модели требует модель-реранкер: "
                       + rag_rerank.filter_reason(backend_now)
                       + ". Порог первичной релевантности работает и без неё — "
                         "им и можно отсекать фрагменты.")
        workspace_store.set_rag_search(
            task,
            rewrite=payload.rewrite,
            rerank=payload.rerank,
            filter=payload.filter,
            ask_when_empty=payload.ask_when_empty,
            top_k_before=payload.top_k_before,
            top_k_after=payload.top_k_after,
            min_score=payload.min_score,
            min_ce=payload.min_ce)
        # Фрагменты прежнего запроса сбрасываются (dialog["rag"]): набор баз
        # изменился, и найденное ранее могло быть подобрано по базе, которую
        # пользователь только что выключил. Подпись данных всё равно не совпала бы
        # (в неё входят включённые базы), но чистить — надёжнее: иначе выключенная
        # база оставалась бы видна в карточках источников до первого нового поиска.
        for session in task.get("sessions") or []:
            dialog = session.get("dialog")
            if isinstance(dialog, dict):
                workspace_store.set_dialog_rag(dialog, "", "", {})
        await _persist()
    return _rag_view(task)


@router.post("/agent/rag/relax")
async def rag_relax(payload: RagRelax) -> dict:
    """СНИЖАЕТ ПОРОГ по выбору пользователя и разрешает новый поиск.

    Порогов ДВА, и маршрут принимает любой из них (`min_score` — первичная
    релевантность, `min_ce` — уверенность модели): у каждого своя шкала, и снижать
    надо именно тот, который отсёк (см. `_rag_choice_view`).

    Зачем отдельный маршрут. Порог отсёк ВСЁ найденное — агент останавливается и
    предлагает варианты, один из них — «снизить порог до … и повторить поиск». Клик по варианту обязан что-то делать: здесь сервер
    запоминает новое значение порога и СБРАСЫВАЕТ сохранённые фрагменты запроса
    (`dialog["rag"]`) — без сброса следующий запрос с тем же текстом нашёл бы в
    диалоге прежнюю пустую запись по ТОЙ ЖЕ подписи (базы + отпечаток + запрос) и
    поиск не повторился бы: пользователь получил бы тот же вопрос заново, то есть
    цикл.

    Меняется ТОЛЬКО переданный порог: второй порог, реранкинг, выборки, галочки
    баз и правила проекта остаются как были. Обращений к модели здесь нет — это настройка.
    """
    task = _current_task()
    if task is None:
        raise HTTPException(status_code=400,
                            detail="Сначала создайте проект — настройка живёт на проекте")
    if payload.min_ce is None and payload.min_score is None:
        raise HTTPException(status_code=400,
                            detail="Не передан новый порог (ни первичной "
                                   "релевантности, ни уверенности модели)")
    async with _workspace_lock:
        settings_before = workspace_store.rag_settings(task)
        workspace_store.set_rag_search(task, min_ce=payload.min_ce,
                                       min_score=payload.min_score)
        settings_after = workspace_store.rag_settings(task)
        for session in task.get("sessions") or []:
            dialog = session.get("dialog")
            if isinstance(dialog, dict):
                workspace_store.set_dialog_rag(dialog, "", "", {})
        await _persist()
    logger.info("RAG: порог снижен по выбору пользователя: первичная "
                "релевантность %s → %s, уверенность модели %s → %s",
                settings_before.get("min_score"), settings_after.get("min_score"),
                settings_before.get("min_ce"), settings_after.get("min_ce"))
    return _rag_view(task)


@router.post("/agent/rag/upload")
async def rag_upload(payload: RagUpload) -> dict:
    """Загружает свою базу знаний: файлы → индекс (чанки + эмбеддинги + метаданные).

    Индексация синхронная и может занять время (разбор PDF, работа модели), но
    она НЕ держит блокировку workspace и идёт в отдельном потоке: цикл событий
    остаётся свободным, и параллельные запросы других профилей не ждут.

    Файл, который не прочитался (скан без текста, битый формат), НЕ роняет
    загрузку — его причина попадает в метаданные базы и видна в диалоге; если не
    прочитался ни один файл, приходит 400 с причинами.
    """
    profile = _current_profile_id()
    if not payload.files:
        raise HTTPException(status_code=400, detail="Не передан ни один файл")
    files = _decode_upload_files(payload.files)

    task = _current_task()
    settings = workspace_store.rag_settings(task) if task else {}
    try:
        rag.ensure_capacity(profile=profile)
    except rag.RagError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:                       # pragma: no cover - защита
        logger.warning("RAG: проверка вместимости не удалась — %s", str(exc)[:150])
    strategy = payload.strategy if payload.strategy is not None else settings.get("strategy")
    chunk_size = payload.chunk_size if payload.chunk_size is not None else settings.get("chunk_size")
    overlap = payload.overlap if payload.overlap is not None else settings.get("overlap")

    try:
        meta = await asyncio.to_thread(
            rag.index_files, files, name=payload.name, profile=profile,
            strategy=strategy, chunk_size=chunk_size, overlap=overlap)
    except rag.RagError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("RAG: индексация не удалась")
        raise HTTPException(status_code=500, detail="Индексация не удалась: %s" % str(exc)[:200])

    # Новая база сразу включается у проекта (её для того и загружали) — и
    # параметры разбиения запоминаются, чтобы следующая загрузка шла с ними.
    if task is not None and payload.enabled:
        async with _workspace_lock:
            enabled = workspace_store.rag_enabled(task)
            if meta["id"] not in enabled:
                enabled.append(meta["id"])
            workspace_store.set_rag_enabled(task, enabled)
            workspace_store.set_rag_chunking(task, meta.get("strategy"),
                                             meta.get("chunk_size"), meta.get("overlap"))
            await _persist()
    return {"base": rag.base_view(meta, enabled=bool(payload.enabled and task)),
            "view": _rag_view(task)}


@router.delete("/agent/rag/{base_id}")
async def rag_delete(base_id: str) -> dict:
    """Удаляет базу знаний вместе с её индексом (кнопка 🗑 в диалоге).

    Удалить можно только СВОЮ базу: чужую профиль не видит вовсе, поэтому для
    него её не существует (404). Из настроек проектов профиля база убирается
    сразу — иначе в них осталась бы галочка несуществующей базы.
    """
    profile = _current_profile_id()
    if not rag.delete_base(base_id, profile=profile):
        raise HTTPException(status_code=404, detail="База знаний не найдена")
    # БРОШЕННЫЕ ВРЕМЕННЫЕ ФАЙЛЫ ЗАГРУЗОК убираем ВМЕСТЕ с базой: потоковая
    # загрузка пишет копию документа в `.incoming`, работник задачи удаляет её в
    # `finally`, но переживший перезапуск файл остаётся — и это копия ТОГО ЖЕ
    # документа, то есть «удалённая база» продолжала бы лежать на диске (живой
    # случай: 241 МБ копии PDF после удаления базы). Штатная уборка
    # (`prune_incoming`) вызывается на следующей загрузке, которой может не быть
    # вовсе, поэтому зовём её здесь. Файлы свежее часа не трогаются: их может
    # писать идущая индексация.
    removed_files = rag_jobs.prune_incoming()
    if removed_files:
        logger.info("RAG: вместе с базой %s убрано брошенных файлов загрузки: %d",
                    base_id, removed_files)
    async with _workspace_lock:
        for task in workspace_store.profile_tasks(_workspace, profile):
            settings = workspace_store.rag_settings(task)
            if base_id in settings.get("enabled", []):
                workspace_store.set_rag_enabled(
                    task, [item for item in settings["enabled"] if item != base_id])
        await _persist()
    return _rag_view(_current_task())


@router.post("/agent/rag/upload/stream")
async def rag_upload_stream(request: Request, filename: str = "", name: str = "",
                            strategy: str = "", chunk_size: Optional[int] = None,
                            overlap: Optional[int] = None, base_id: str = "",
                            enabled: int = 1) -> dict:
    """ПОТОКОВАЯ загрузка одного файла: тело запроса — сам файл, без base64.

    Зачем отдельный маршрут: через base64 в JSON крупный файл проходит плохо.
    На 250 МБ браузер собирает строку в сотни мегабайт, она же уезжает в теле
    запроса, сервер разбирает её целиком — и всё это ради того, чтобы получить
    те же байты. Здесь тело пишется на диск КУСКАМИ (`request.stream()`): ни
    браузер, ни сервер не держат документ в памяти целиком, а разбор PDF потом
    идёт постранично прямо из файла (см. rag_documents.extract_path).

    `base_id` (необязательный) — ДОПИСАТЬ файл в существующую базу: так
    интерфейс грузит несколько крупных файлов по одному, не заводя базу на
    каждый. Параметры разбиения при добавлении берутся из паспорта базы.

    Временный файл удаляется ВСЕГДА (в finally): незавершённая загрузка не
    должна оставлять мусор в каталоге баз.
    """
    profile = _current_profile_id()
    task = _current_task()
    source = rag.sanitize_source(filename)
    incoming = os.path.join(rag_store.directory(), _INCOMING_DIR)
    try:
        os.makedirs(incoming, exist_ok=True)
    except OSError as exc:
        raise HTTPException(status_code=500,
                            detail="Не удалось подготовить каталог загрузки: %s"
                                   % str(exc)[:120])
    handle, temp_path = tempfile.mkstemp(dir=incoming, suffix=".part")
    size = 0
    try:
        with os.fdopen(handle, "wb") as target:
            async for piece in request.stream():
                size += len(piece)
                if size > rag_documents.MAX_FILE_BYTES:
                    raise HTTPException(
                        status_code=400,
                        detail="Файл больше %s — такой документ не индексируется. "
                               "Предел меняется переменной RAG_MAX_FILE_BYTES."
                               % rag_documents.human_bytes(rag_documents.MAX_FILE_BYTES))
                target.write(piece)
        if not size:
            raise HTTPException(status_code=400,
                                detail="Файл «%s» передан пустым" % source)

        settings = workspace_store.rag_settings(task) if task else {}
        # ИНДЕКСАЦИЯ УХОДИТ В ФОН, а не выполняется прямо в этом запросе: у
        # крупного документа это минуты, и держать на них HTTP-соединение нельзя
        # (обрыв браузера или посредника убил бы работу, а пользователь всё это
        # время не видел бы ничего, кроме «индексирую…»). Запрос отдаёт задачу,
        # интерфейс опрашивает её состояние и рисует прогресс.
        if not base_id:
            rag.ensure_capacity(profile=profile)
        try:
            job = await rag_jobs.start(
                profile=profile, files=[{"filename": source, "path": temp_path}],
                name=name, strategy=strategy or settings.get("strategy"),
                chunk_size=chunk_size if chunk_size is not None else settings.get("chunk_size"),
                overlap=overlap if overlap is not None else settings.get("overlap"),
                base_id=base_id)
        except rag.RagError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.exception("RAG: не удалось завести задачу индексации")
            raise HTTPException(status_code=500,
                                detail="Индексация не запущена: %s" % str(exc)[:200])
    except HTTPException:
        # Задачу не завели — временный файл остаётся за нами.
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise

    # Настройки проекта запоминаются СРАЗУ (они не зависят от исхода индексации),
    # а включение новой базы произойдёт по завершении задачи: её id появляется
    # только после записи индекса.
    if task is not None and enabled:
        async with _workspace_lock:
            workspace_store.set_rag_chunking(task, strategy or settings.get("strategy"),
                                             chunk_size, overlap)
            await _persist()
    return {"job": job, "view": _rag_view(task)}


@router.get("/agent/rag/jobs")
async def rag_jobs_list(active: int = 0) -> dict:
    """Состояние фоновых индексаций: что идёт, как далеко, чем закончилось.

    Опрос раз в секунду — этим живёт полоса прогресса в диалоге. active=1 отдаёт
    только идущие задачи (интерфейсу в опросе нужны они одни).
    """
    profile = _current_profile_id()
    return {"jobs": rag_jobs.listing(profile=profile, active_only=bool(active)),
            "active": rag_jobs.active_count(),
            "summary": rag_jobs.summary(profile=profile)}


@router.get("/agent/rag/jobs/{job_id}")
async def rag_job_get(job_id: str) -> dict:
    """Одна задача индексации (чужая или неизвестная — 404)."""
    job = rag_jobs.get(job_id, profile=_current_profile_id())
    if job is None:
        raise HTTPException(status_code=404, detail="Задача индексации не найдена")
    return {"job": job}


@router.post("/agent/rag/jobs/{job_id}/cancel")
async def rag_job_cancel(job_id: str) -> dict:
    """Просит остановить индексацию.

    Остановка происходит между страницами и батчами (работник проверяет флаг в
    отчёте о ходе), и до записи индекса она безопасна: прежний индекс остаётся
    целым, а временный файл удаляется самой задачей.
    """
    job = rag_jobs.cancel(job_id, profile=_current_profile_id())
    if job is None:
        raise HTTPException(status_code=404, detail="Задача индексации не найдена")
    return {"job": job}


@router.post("/agent/rag/jobs/finish")
async def rag_job_finish(payload: RagJobDone) -> dict:
    """Применяет ИТОГ завершённой индексации к настройкам проекта.

    Новая база включается у проекта ТОЛЬКО здесь: её идентификатор появляется
    после записи индекса, и раньше включать было нечего. Интерфейс зовёт маршрут
    один раз, когда опрос показал завершение, — тогда включение не может
    потеряться, даже если пользователь закрыл диалог.
    """
    job = rag_jobs.get(payload.job_id, profile=_current_profile_id())
    if job is None:
        raise HTTPException(status_code=404, detail="Задача индексации не найдена")
    task = _current_task()
    if task is None or job["state"] != "done" or not job["base_id"]:
        return {"applied": False, "view": _rag_view(task)}
    async with _workspace_lock:
        ids = workspace_store.rag_enabled(task)
        if payload.enabled and job["base_id"] not in ids:
            ids.append(job["base_id"])
            workspace_store.set_rag_enabled(task, ids)
        result = (job.get("result") or {})
        workspace_store.set_rag_chunking(task, None,
                                         result.get("chunk_size"), result.get("overlap"))
        await _persist()
    return {"applied": True, "base_id": job["base_id"], "view": _rag_view(task)}


@router.get("/agent/rag/{base_id}/chunks")
async def rag_chunks(base_id: str, offset: int = 0, limit: int = 10,
                     source: str = "", q: str = "", chunk: int = 0) -> dict:
    """Страница чанков базы: посмотреть своими глазами, как нарезан документ.

    Отдаёт текст чанков с их адресом (источник, раздел, номер, границы в
    документе) и общее число под фильтром. Фильтры: по документу (`source`) и
    по тексту (`q`). Постранично — база на 20 000 чанков в диалог не поместится.

    `chunk` — номер чанка, к которому надо перейти (клик по источнику под ответом
    агента): страница сдвигается так, чтобы он был первым. Так вариант «открыть
    фрагмент, по которому получен ответ» действительно открывает ЕГО, а не первую
    страницу базы.

    Чужую базу профиль не видит (404): просмотр — такая же работа с базой, как
    и удаление.
    """
    if not rag_store.valid_id(base_id):
        raise HTTPException(status_code=404, detail="База знаний не найдена")
    try:
        return rag.chunks_view(base_id, profile=_current_profile_id(),
                               offset=offset, limit=limit, source=source, query=q,
                               chunk=chunk)
    except rag.RagError:
        raise HTTPException(status_code=404, detail="База знаний не найдена")


# ---------------------------------------------------------------------------
# ТЕСТОВЫЙ ПРОГОН RAG: команда /test_rag в чате (app/ai/rag_suite.py)
#
# ИСКЛЮЧЕНИЕ ИЗ ОБЫЧНОЙ РАБОТЫ, и это намеренно. Проверяется ПОИСК И ОТВЕТ ПО
# ДОКУМЕНТАМ, а не умение агента планировать, поэтому прогон идёт МИМО пайплайна
# задачи: без планировщика, подтверждения плана, шагов, автомата и приёмщика. В
# память диалога (messages) и в автомат (state) он НЕ пишет ничего — иначе
# контрольные вопросы перемешались бы с настоящей перепиской и ухудшили бы
# следующие ответы задачи; в журнал чата (log) пишет: пользователь должен видеть
# результат, в том числе после переключения задачи. Расход токенов прогона в
# замер задачи (dialog["usage"]) не попадает — он показывается итоговой строкой.
# ---------------------------------------------------------------------------
def _rag_test_context() -> Tuple[List[str], str, str]:
    """(включённые базы, готовый блок запроса, причина отказа) для прогона.

    Базы берутся у ТЕКУЩЕГО проекта профиля — тот же набор, по которому отвечает
    агент. Базы не включены — прогонять нечего: тест проверяет RAG, а не модель.
    """
    task = _current_task()
    enabled = workspace_store.rag_enabled(task) if task else []
    if not enabled:
        return [], "", ("к проекту не подключено ни одной базы знаний — включите базу "
                        "кнопкой «RAG», иначе проверять нечего")
    return enabled, "", ""


@router.post("/agent/rag/test")
async def rag_test(request: Request) -> StreamingResponse:
    """Прогон контрольных вопросов по базам знаний — команда `/test_rag` в чате.

    Поток событий (NDJSON, как у чата агента):

      * `{"type": "test_start", "total": N, "bases": [...]}` — начали, столько-то
        вопросов по таким-то базам;
      * `{"type": "test_question", "n": i, "total": N, "text": "…"}` — вопрос;
      * `{"type": "bot", "text": "…", "test": i, "sources": [...]}` — ответ модели
        по фрагментам ВМЕСТЕ С ИСТОЧНИКАМИ (`rag_search.sources`): интерфейс рисует
        под ним те же карточки, что под ответом агента — с цитатой и переходом к
        чанку (иначе тест проверял бы поиск, но не то, что видит человек);
      * `{"type": "test_error", "n": i, "text": "…"}` — вопрос остался без ответа
        (прогон НЕ обрывается: остальные вопросы всё равно проверяются);
      * `{"type": "test_verdict", "text": …, "items": [...]}` — вердикт судьи;
      * `{"type": "done", "usage": {...}}` — итог прогона и его расход.

    Вопросы, эталоны и промпты живут в `app/ai/rag_suite.py`; здесь только ход
    прогона. Поиск по базе идёт тем же кодом, что у агента (`rag_search.search`),
    и в потоке — цикл событий на нём не стоит. Настройки поиска тоже берутся у
    ПРОЕКТА (панель «Поиск и ответы»): реранкинг, порог, топ-K до и после, а если
    включена переформулировка запроса — вопрос перед поиском превращается в
    поисковый запрос (тем же модулем, см. app/ai/rag_query.py). Иначе прогон
    проверял бы не то, что делает агент.
    """
    enabled, _, reason = _rag_test_context()
    if reason:
        raise HTTPException(status_code=400, detail=reason)
    task = _current_task()
    if task is None:
        raise HTTPException(status_code=400,
                            detail="Сначала создайте проект — базы знаний привязаны к проекту")
    # Место для вывода: активный диалог задачи. Его нет (задача без диалогов) —
    # заводим, как это делает обычный запрос агента: результаты теста должны быть
    # видны в чате, а не потеряться.
    session = workspace_store.active_session(_workspace, task)
    if session is None:
        session = workspace_store.create_session(task)
    dialog = session.get("dialog")
    profile = _current_profile_id()
    test_settings = workspace_store.rag_settings(task)
    base_names = []
    for base_id in enabled:
        meta = rag_store.get_base(base_id, profile=profile) or {}
        base_names.append(str(meta.get("name") or base_id))

    def log(kind: str, text: str, sources: Any = None) -> None:
        """Запись в журнал ЧАТА (не в память диалога): тест должен быть виден.

        `sources` — фрагменты, по которым модель ответила на вопрос теста: узел
        журнала хранит их тем же полем, что и ответ агента, поэтому карточки
        источников (с цитатой и переходом к чанку) видны и после переключения
        задачи или перезагрузки страницы.
        """
        if dialog is not None:
            workspace_store.add_log(dialog, kind, text, sources=sources)

    async def event_stream():
        # Прогон НЕ берёт блокировку задачи: он ничего не меняет в её состоянии, а
        # держать на нём замок значило бы блокировать настоящую работу.
        #
        # 0. ИСТОЧНИК ОТВЕТА. Прогон зовёт модель на КАЖДОМ вопросе и ещё раз на
        #    вердикте судьи — при выбранной локальной модели сервер обязан
        #    отвечать ДО первого вызова: иначе все вопросы вернулись бы ошибкой
        #    связи, а причина («сервер не запущен») осталась бы не названной:
        #    человек прочитал бы «вопрос остался без ответа» и не понял, почему.
        notes: List[str] = []
        reason = await local_source_guard(notes.append)
        for note in notes:
            log(workspace_store.LOG_DEBUG, note)
            yield json.dumps({"type": "debug", "text": note},
                             ensure_ascii=False) + "\n"
        if reason:
            text = LOCAL_SOURCE_NOT_READY % reason
            log(workspace_store.LOG_ERROR, text)
            yield json.dumps({"type": "error", "text": text},
                             ensure_ascii=False) + "\n"
            yield json.dumps({"type": "done", "usage": {}},
                             ensure_ascii=False) + "\n"
            return
        rows: List[Dict[str, Any]] = []
        usage: Dict[str, Any] = {}
        start = {"type": "test_start", "total": rag_suite.total(),
                 "bases": base_names}
        log(workspace_store.LOG_DEBUG, (
            "🧪 Тест RAG: %d контрольных вопросов по базам: %s. Прогон идёт мимо "
            "плана задачи — состояние и память задачи не меняются. Настройки "
            "поиска — как у проекта: реранкинг %s, порог %s, топ-K %s→%s, "
            "переформулировка запроса %s."
            % (rag_suite.total(), ", ".join(base_names),
               "вкл" if test_settings.get("rerank") else "выкл",
               ("%.2f" % float(test_settings.get("min_score") or 0.0))
               if test_settings.get("filter") else "выкл",
               test_settings.get("top_k_before"), test_settings.get("top_k_after"),
               "вкл" if test_settings.get("rewrite") else "выкл")))
        yield json.dumps(start, ensure_ascii=False) + "\n"
        for number, case_data in enumerate(rag_suite.CASES, 1):
            if await request.is_disconnected():
                return
            question = case_data["question"]
            log(workspace_store.LOG_USER, "🧪 Вопрос %d/%d: %s"
                % (number, rag_suite.total(), question))
            event = {"type": "test_question", "n": number,
                     "total": rag_suite.total(), "text": question}
            yield json.dumps(event, ensure_ascii=False) + "\n"
            # ЧТО ПРОИСХОДИТ СЕЙЧАС: строка уходит СРАЗУ после вопроса, а не
            # только вместе с ответом. Иначе в чате между вопросом и ответом
            # висела тишина (поиск + вызов модели), и прогон читался как
            # «всё появилось в конце» (жалоба 03.10 — см. §7, «потоки без gzip»).
            yield json.dumps({"type": "debug", "text": (
                "🔎 Ищу фрагменты по вопросу в базах проекта: «%s»"
                % question[:200])}, ensure_ascii=False) + "\n"
            started = time.monotonic()
            try:
                # 0. ПЕРЕФОРМУЛИРОВКА ЗАПРОСА — если она включена у проекта: тот
                #    же модуль, что у агента. Расход этого вызова входит в итог
                #    прогона (метрики забирает обёртка): он настоящий.
                query_text = question
                rewrite_line = ""
                info: Dict[str, Any] = {}
                if test_settings.get("rewrite"):
                    spent: Dict[str, Any] = {}

                    async def call(**kwargs: Any) -> Any:
                        content, metrics = await llm_client.call_llm_async(**kwargs)
                        spent["metrics"] = metrics
                        return content, metrics

                    info = await rag_query.rewrite(question, call)
                    # Расход считаем только по ФАКТИЧЕСКОМУ ответу модели: сбой
                    # вызова (сеть, ключ) не должен добавлять прогону «вызов без
                    # токенов» — в замере это выглядело бы как лишняя оплата.
                    if spent.get("metrics"):
                        usage = _merge_test_usage(usage, spent["metrics"])
                    if info.get("query"):
                        query_text = str(info["query"])
                    rewrite_line = rag_query.debug_line(info)
                # 1. ПОИСК по базам проекта — тем же кодом, что у агента, и с теми
                #    же настройками проекта (два этапа: пул → реранкинг → порог).
                result = await asyncio.to_thread(rag_search.search, enabled, query_text,
                                                 profile=profile,
                                                 settings=test_settings,
                                                 rewrite=info if rewrite_line else None)
                hits = rag_search.hits_of(result)
                block = rag_search.block(result)
                # 2. ОТВЕТ модели по найденным фрагментам — ОДИН вызов, без плана.
                content, metrics = await llm_client.call_llm_async(
                    user_text=rag_suite.answer_payload(case_data, block),
                    model=config.active_model(),
                    disable_thinking=True,
                    max_tokens=rag_suite.ANSWER_MAX_TOKENS,
                    messages=[
                        {"role": "system", "content": rag_suite.ANSWER_PROMPT},
                        {"role": "user",
                         "content": rag_suite.answer_payload(case_data, block)},
                    ],
                    timeout=_rag_test_timeout())
                answer = str(content or "").strip()
                usage = _merge_test_usage(usage, metrics)
                source_line = ("%d. Найдено фрагментов: %d%s"
                               % (number, len(hits),
                                  ("; лучший — " + rag_search.address_of(hits[0]))
                                  if hits else ""))
                row = {"n": number, "question": question, "answer": answer,
                       "hits": len(hits), "error": "", "metrics": metrics,
                       "seconds": round(time.monotonic() - started, 1)}
            except Exception as exc:                    # сбой одного вопроса
                logger.warning("RAG-тест: вопрос %d не отработан — %s",
                               number, str(exc)[:200])
                row = {"n": number, "question": question, "answer": "",
                       "hits": 0,
                       "error": llm_client.redact_secrets(str(exc))[:300],
                       "metrics": {}, "seconds": round(time.monotonic() - started, 1)}
            rows.append(row)
            if rewrite_line:
                # Строка о переформулировке идёт ПЕРЕД строкой о находках: видно,
                # по какому запросу искали, а уже потом — что нашлось.
                yield json.dumps({"type": "debug", "text": rewrite_line},
                                 ensure_ascii=False) + "\n"
                log(workspace_store.LOG_DEBUG, rewrite_line)
            yield json.dumps({"type": "debug", "text": source_line},
                             ensure_ascii=False) + "\n"
            log(workspace_store.LOG_DEBUG, source_line)
            if row["answer"]:
                # ИСТОЧНИКИ — вместе с ответом, как у агента: по ним интерфейс
                # рисует карточки фрагментов (файл, раздел, номер чанка, цитата) и
                # делает кликабельными ссылки «[N]» в тексте ответа. В журнал они
                # идут тем же полем `sources`, поэтому видны и после переключения
                # задачи (см. workspace.add_log).
                answer_sources = rag_search.sources(result)
                log(workspace_store.LOG_ASSISTANT, row["answer"],
                    sources=answer_sources)
                yield json.dumps({"type": "bot", "text": row["answer"],
                                  "test": number, "sources": answer_sources},
                                 ensure_ascii=False) + "\n"
            else:
                text = ("⚠ Вопрос %d остался без ответа: %s"
                        % (number, row["error"] or "причина неизвестна"))
                log(workspace_store.LOG_ERROR, text)
                yield json.dumps({"type": "test_error", "n": number, "text": text},
                                 ensure_ascii=False) + "\n"

        # 3. ОЦЕНКА: вопросы, ЭТАЛОНЫ и ответы уходят судье — отдельным вызовом.
        verdict = {"items": [], "summary": ""}
        if any(row["answer"] for row in rows):
            yield json.dumps({"type": "debug", "text": (
                "🧪 Отдаю ответы на оценку модели: сравнит факты с эталонами, "
                "выписанными из самой базы.")}, ensure_ascii=False) + "\n"
            payload = rag_suite.judge_payload(rows)
            try:
                content, metrics = await llm_client.call_llm_async(
                    user_text=payload,
                    model=config.active_model(),
                    disable_thinking=True,
                    max_tokens=rag_suite.JUDGE_MAX_TOKENS,
                    messages=[
                        {"role": "system", "content": rag_suite.JUDGE_PROMPT},
                        {"role": "user", "content": payload},
                    ],
                    timeout=_rag_test_timeout())
                usage = _merge_test_usage(usage, metrics)
                verdict = rag_suite.parse_verdict(content)
            except Exception as exc:
                logger.warning("RAG-тест: оценка не получена — %s", str(exc)[:200])
                verdict["summary"] = ("Оценку получить не удалось: %s"
                                      % llm_client.redact_secrets(str(exc))[:200])
        else:
            verdict["summary"] = "Ни один вопрос не получил ответа — оценивать нечего."

        lines = rag_suite.verdict_lines(verdict)
        text = "🧪 Оценка ответов\n" + rag_suite.summary_text(verdict)
        # Вопросы без найденных фрагментов — та же подсказка про порог: прогон
        # проверяет поиск, и «в документах нет» может быть следствием настройки.
        rag_stats = {"turns": len(rows),
                     "with_sources": sum(1 for row in rows if row.get("hits"))}
        rag_hint = rag_dialog.yield_hint(rag_stats, test_settings.get("min_score"),
                                         test_settings.get("min_ce"),
                                         bool(test_settings.get("filter")))
        if rag_hint:
            text += "\n" + rag_hint
        if lines:
            text += "\n" + "\n".join(lines)
        if verdict.get("summary"):
            text += "\n\n" + str(verdict["summary"])
        text += "\n\n" + _rag_test_usage_line(usage, rows)
        log(workspace_store.LOG_ASSISTANT, text)
        yield json.dumps({"type": "test_verdict", "text": text,
                          "items": verdict.get("items") or [],
                          "summary": verdict.get("summary") or ""},
                         ensure_ascii=False) + "\n"
        yield json.dumps({"type": "done", "usage": usage}, ensure_ascii=False) + "\n"

    async def stream():
        """Поток прогона + запись журнала на диск (результаты видны и после
        переключения задачи). Сбой записи прогон не отменяет."""
        try:
            async for chunk in event_stream():
                yield chunk
        finally:
            if dialog is not None:
                try:
                    async with _workspace_lock:
                        await _persist()
                except Exception as exc:            # pragma: no cover - защита
                    logger.warning("RAG-тест: журнал не сохранён — %s", str(exc)[:150])

    return StreamingResponse(stream(), media_type="application/x-ndjson")


def _rag_test_timeout() -> float:
    """Таймаут одного вызова теста: ответы короткие, ждать дольше незачем."""
    return float(llm_client.HTTP_TIMEOUT)


# ---------------------------------------------------------------------------
# МИНИ-ЧАТ ПО БАЗАМ ЗНАНИЙ (app/ai/rag_dialog.py)
#
# ЧТО ЭТО. Диалог с опорой на документы проекта: история разговора + поиск по
# включённым базам на КАЖДЫЙ новый вопрос + ответ по найденным фрагментам +
# источники под ответом + память задачи (цель, уточнения, ограничения, термины).
#
# МИМО ПАЙПЛАЙНА ЗАДАЧИ — так же, как контрольный прогон `/test_rag`: ни
# планировщика, ни подтверждения плана, ни шагов, ни приёмщика, ни переходов
# автомата. Причина та же: короткий разговор «вопрос — ответ по документам» не
# нуждается в шагах, а каждый шаг стоил бы отдельного вызова модели. Состояние
# автомата мини-чат НЕ трогает: вернувшись в режим «AI-агент», задача продолжает
# свой этап ровно так, как его оставили.
#
# ЧТО МИНИ-ЧАТ ЗАПИСЫВАЕТ. Реплики — в память диалога (messages: следующий ход
# мини-чата и обычный запрос агента видят разговор), ответ и служебные строки — в
# журнал чата (log: окно восстанавливается при переключении задачи), расход — в
# dialog["usage"] (панель «Токены задачи»), память задачи — в dialog["task_memory"].
# НЕ записывает: состояние автомата, подпись плана. Фрагменты поиска тоже НЕ
# кладутся в dialog["rag"]: это запись «к какому запросу ЗАДАЧИ относятся
# данные» — мини-чат ищет по каждому своему вопросу заново, и его поиск не должен
# подменять собой данные шага плана (см. §5.13 в SESSION_PROMPT).
# ---------------------------------------------------------------------------
def _rag_dialog_target(session_id: str = "") -> Tuple[Optional[Dict[str, Any]],
                                                     Optional[Dict[str, Any]]]:
    """(проект, диалог) для мини-чата: своя задача — по id, иначе текущая.

    Диалога ещё нет (задача без диалогов) — он заводится: мини-чат хранит
    ИСТОРИЮ, а хранить её негде без сессии.
    """
    explicit = str(session_id or "").strip()
    if explicit:
        task, session = _find_session_anywhere(explicit)
        return task, session
    task = _current_task()
    if task is None:
        return None, None
    session = workspace_store.active_session(_workspace, task)
    if session is None:
        session = workspace_store.create_session(task)
    return task, session


def _rag_dialog_bases(task: Dict[str, Any]) -> List[str]:
    """Включённые базы проекта — по ним отвечает мини-чат (как агент)."""
    return workspace_store.rag_enabled(task)


def _rag_hits_line(hits: List[Dict[str, Any]], number: int = 0) -> str:
    """Строка диагностики «сколько нашлось и что лучший» (как у `/test_rag`)."""
    prefix = ("%d. " % number) if number else ""
    if not hits:
        return prefix + "Найдено фрагментов: 0"
    return ("%sНайдено фрагментов: %d; лучший — %s"
            % (prefix, len(hits), rag_search.address_of(hits[0])))


def _rag_cut_note(result: Dict[str, Any]) -> str:
    """Чем отсеклись фрагменты («score»/«ce») — для честной строки источников.

    Это тот же разбор, что у диагностики RAG: «порог отсёк всё» и «в документах
    нет» — РАЗНЫЕ вещи, и путать их в ответе мини-чата нельзя.
    """
    kind = rag_search.cut_kind(result)
    if kind == "score":
        stages = (result or {}).get("stages") or {}
        return "порог первичной релевантности %.2f" % float(
            stages.get("min_score") or 0.0)
    if kind == "ce":
        stages = (result or {}).get("stages") or {}
        return "порог уверенности модели %.2f" % float(stages.get("min_ce") or 0.0)
    return ""


async def _rag_dialog_turn(question: str, *, memory: Dict[str, Any],
                           history: str, profile: str,
                           profile_id: Optional[str],
                           settings: Dict[str, Any], enabled: List[str],
                           tracker: Agent,
                           number: int = 0) -> AsyncIterator[Dict[str, Any]]:
    """ОДИН ход мини-чата: поиск по базам → ответ по фрагментам → память задачи.

    Отдаёт события по мере хода (асинхронный генератор): диагностику поиска,
    ответ (`bot` вместе с ИСТОЧНИКАМИ), снимок памяти задачи и расход хода.
    Запись в диалог — дело вызывающего кода: этим же ходом пользуются и обычный
    мини-чат (пишет в диалог), и контрольный прогон (ничего не пишет).

    ПОИСК ИДЁТ НА КАЖДЫЙ ВОПРОС, без исключений: в этом и смысл мини-чата.
    Запрос переформулируется тем же модулем, что у агента (если включено у
    проекта), ищется тем же `rag_search.search` с настройками ПРОЕКТА — иначе
    мини-чат мерил бы не то, что видит человек в панели «Поиск и ответы».
    """
    # 1. ПЕРЕФОРМУЛИРОВКА ЗАПРОСА (Query Rewrite) — если включена у проекта:
    #    тот же модуль и тот же вид служебного вызова, что у агента.
    query_text = question
    rewrite_info: Dict[str, Any] = {}
    if settings.get("rewrite"):
        async def call(**kwargs: Any) -> Any:
            content, metrics = await llm_client.call_llm_async(**kwargs)
            tracker.note_usage(metrics, Agent.SERVICE_REWRITE)
            return content, metrics

        rewrite_info = await rag_query.rewrite(question, call)
        if rewrite_info.get("query"):
            query_text = str(rewrite_info["query"])
        line = rag_query.debug_line(rewrite_info)
        if line:
            yield {"type": "debug", "text": line}
    # 2. ПОИСК ПО БАЗАМ — в потоке: считается вектор запроса и перебирается
    #    индекс, цикл событий на это время блокировать нельзя.
    #    ВТОРОЙ ЗАПРОС: переформулировка (если была) и — у «сводной» просьбы
    #    («собери памятку», «сведи чек-лист») — ЦЕЛЬ задачи. У такой просьбы нет
    #    своих слов о теме, и поиск по ней приносил случайные разделы: живой
    #    прогон 03.10 показал, как на «собери памятку из того, что нашли» модель
    #    получила чужие фрагменты и честно ответила «в этих фрагментах этого
    #    нет» — свод разваливался, хотя все факты были найдены раньше.
    goal_query = rag_dialog.search_query(question, memory)[1]
    yield {"type": "debug", "text": "🔎 Ищу в базах знаний по запросу: «%s»"
           % query_text[:200]}
    # Собираем второй запрос из частей: исходный текст (если была
    # переформулировка — она ничего не должна терять) и цель задачи.
    also_parts: List[str] = []
    if rewrite_info:
        also_parts.append(question)
    if goal_query:
        also_parts.append(goal_query)
        yield {"type": "debug", "text": (
            "🔎 Это просьба свести уже найденное — ищу ещё и по ЦЕЛИ задачи: «%s»"
            % goal_query[:200])}
    result = await asyncio.to_thread(
        rag_search.search, enabled, query_text, profile=profile_id,
        settings=settings, rewrite=rewrite_info or None,
        also=" ".join(also_parts))
    hits = rag_search.hits_of(result)
    note = rag_search.results_note(result)
    if note:
        yield {"type": "debug", "text": note}
    yield {"type": "debug", "text": _rag_hits_line(hits, number)}
    # 3. ОТВЕТ ПО ФРАГМЕНТАМ: один вызов модели, без плана и шагов.
    messages = rag_dialog.answer_messages(
        memory_store.block(memory), rag_search.block(result), history, question,
        profile)
    content, metrics = await llm_client.call_llm_async(
        user_text=question, model=config.active_model(), disable_thinking=True,
        max_tokens=rag_dialog.ANSWER_MAX_TOKENS, messages=messages,
        timeout=_rag_test_timeout())
    tracker.note_usage(metrics, None)
    if not str(content or "").strip():
        # Модель не ответила (сбой вызова, пустой ответ): ВЫДАВАТЬ строку
        # источников за ответ нельзя — это выглядело бы как ответ по документам.
        # Сбой помечается ошибкой, ход считается несостоявшимся, а расход вызова
        # (если он был) остаётся в замере как сбойный.
        yield {"type": "error", "text": (
            "⚠ Мини-чат не получил ответа от модели — повторите вопрос.")}
        return
    # ИСТОЧНИКИ — ГАРАНТИЕЙ КОДА: строка под ответом есть ВСЕГДА (либо номера
    # фрагментов, либо честное «фрагментов не найдено»).
    answer = rag_dialog.ensure_sources(content, hits, searched=len(enabled),
                                       cut=_rag_cut_note(result))
    yield {"type": "bot", "text": answer,
           "sources": rag_search.sources(result), "hits": len(hits),
           "cited": rag_dialog.has_citation(answer),
           # ССЫЛКИ НА НЕСУЩЕСТВУЮЩИЕ ФРАГМЕНТЫ: номера обновляются на каждом
           # вопросе, и перенесённый из прошлого ответа номер указывает на чужой
           # документ. Это проверяет КОД (ссылка — это число), а судья прогона
           # проверяет то, что кодом не проверить: подтверждает ли цитата слова.
           "bad_cites": rag_dialog.bad_citations(answer, len(hits)),
           "honest": rag_dialog.looks_like_no_data(answer)}
    # 4. ПАМЯТЬ ЗАДАЧИ: что пользователь уточнил, что зафиксировано. Сбой этого
    #    служебного вызова ответ не отменяет — память дополнит локальный разбор.
    #    Перед ним — «ответ показан, ввод свободен» (см. тот же приём в прямом
    #    ответе агента): мини-чат тоже не должен держать поле ввода, пока
    #    досчитывается память.
    yield {"type": "answer_ready"}
    async for event in _rag_memory_update(question, answer, memory, tracker):
        yield event


async def _rag_memory_update(question: str, answer: str, memory: Dict[str, Any],
                             tracker: Agent,
                             turn: int = 0) -> AsyncIterator[Dict[str, Any]]:
    """Обновление ПАМЯТИ ЗАДАЧИ после реплики (служебный вызов + страховка).

    Модель получает прежнюю память и новую пару реплик и возвращает память
    целиком; результат СЛИВАЕТСЯ с прежней (см. app/ai/task_memory.py): ни одна
    договорённость не теряется даже при сбое или мусорном ответе. Цель и явные
    ограничения дополнительно достаются КОДОМ из самой реплики — если модель не
    ответила, разговор всё равно помнит, чего от него хотят.
    """
    payload = memory_store.extract_payload(memory, question, answer, turn)
    content = ""
    try:
        content, metrics = await llm_client.call_llm_async(
            user_text=payload, model=config.active_model(), disable_thinking=True,
            max_tokens=rag_dialog.MEMORY_MAX_TOKENS,
            messages=[{"role": "system", "content": memory_store.EXTRACT_PROMPT},
                      {"role": "user", "content": payload}],
            timeout=_rag_test_timeout())
        tracker.note_usage(metrics, Agent.SERVICE_MEMORY)
    except Exception as exc:                    # память не роняем никогда
        logger.warning("Мини-чат: память задачи обновлена локально — %s",
                       str(exc)[:200])
    update = memory_store.parse(content, turn=turn)
    local = memory_store.local_update(question, memory, turn=turn)
    if not update.get("goal"):
        update["goal"] = local.get("goal") or ""
    for key, _, _ in memory_store.LISTS:
        if not update.get(key):
            update[key] = local.get(key) or []
    merged = memory_store.merge(memory, update)
    # ИЗМЕНИЛАСЬ ли память: строка в чате нужна, когда есть новость. Повторный
    # тот же вопрос ничего не добавляет — и «🧠 Память задачи — цель: …» второй раз
    # подряд выглядела как поломка (жалоба 05.10).
    before = memory_store.snapshot(memory)
    after = memory_store.snapshot(merged)
    changed = after != before
    yield {"type": "task_memory", "memory": after, "changed": changed,
           "text": "🧠 Память задачи — " + memory_store.summary_line(merged)}


@router.post("/agent/rag/dialog")
async def rag_dialog_chat(msg: ChatMessage) -> StreamingResponse:
    """РАЗГОВОР ПО ДОКУМЕНТАМ — движок прямого ответа, без правил проекта и MCP.

    ПОСЛЕ ОБЪЕДИНЕНИЯ ПУТЕЙ (03.10) интерфейс этим маршрутом НЕ пользуется: в
    режиме «AI-агент» путь выбирает гейт (`_plan_needed`), и прямой ответ идёт
    через `agent_chat` — в нём те же правила ответа плюс гейт инвариантов, данные
    внешних инструментов и останов «в документах ничего нет — решает
    пользователь». Маршрут остался как ЧИСТЫЙ разговор по документам: его
    используют проверки, из него же растёт движок (`_rag_dialog_turn`), и он
    годится для внешних вызовов, которым правила проекта и MCP не нужны.

    Поток событий (NDJSON, как у чата агента):

      * `{"type": "debug", "text": …}` — что делает мини-чат: переформулировка
        запроса, что нашлось в базах, порог;
      * `{"type": "bot", "text": …, "sources": [...]}` — ответ ВМЕСТЕ С
        ИСТОЧНИКАМИ (в тексте всегда есть строка «📄 Источники: …», в `sources` —
        те же фрагменты, что у агента: интерфейс рисует по ним карточки с
        цитатой и переходом к чанку);
      * `{"type": "task_memory", "memory": {...}, "text": …}` — память задачи
        после этой реплики (цель, уточнения, ограничения, термины);
      * `{"type": "usage", "usage": {...}}` — расход хода (ответ +
        переформулировка + обновление памяти) одной записью, как у агента;
      * `{"type": "done", "usage": {...}}` — конец потока.

    Базы не включены — отказ СРАЗУ (400) без обращений к модели: мини-чат
    отвечает ПО ДОКУМЕНТАМ, и без документов он не имеет смысла.
    """
    task, session = _rag_dialog_target(str(getattr(msg, "session_id", "") or ""))
    if task is None or session is None:
        return JSONResponse(
            {"detail": "Сначала создайте проект — мини-чат отвечает по базам знаний проекта"},
            status_code=400)
    question = (msg.content or "").strip()
    if not question:
        raise HTTPException(status_code=400,
                            detail="Введите вопрос — мини-чат отвечает по документам")
    enabled = _rag_dialog_bases(task)
    if not enabled:
        raise HTTPException(status_code=400, detail=(
            "к проекту не подключено ни одной базы знаний — мини-чат отвечает по "
            "документам, включите базу кнопкой «RAG»"))
    settings = workspace_store.rag_settings(task)
    profile_id = _current_profile_id()
    profile = _profile_block()
    log_target: Dict[str, Any] = {}

    def encode(event: dict) -> str:
        _log_event(log_target.get("dialog"), event)
        return json.dumps(event, ensure_ascii=False) + "\n"

    async def event_stream():
        async with _session_lock(str(session["id"])):
            # Диалог берём ПОД блокировкой СВОЕЙ задачи: за время подготовки
            # ответа задачу могли переключить.
            task_now, session_now = _rag_dialog_target(
                str(getattr(msg, "session_id", "") or ""))
            if task_now is None or session_now is None:
                yield encode({"type": "error", "text": "Диалог не найден."})
                yield encode({"type": "done", "usage": {}})
                return
            dialog_now: Dict[str, Any] = session_now["dialog"]
            log_target["dialog"] = dialog_now
            workspace_store.add_log(dialog_now, workspace_store.LOG_USER, question)
            # Пометку «задача выполняется» мини-чат ставит и ОБЯЗАН снять:
            # ход держит диалог задачи, и пока он идёт, «Пауза» откладывается, а
            # периодический повтор считает задачу занятой. Снимается в finally —
            # иначе после первой же реплики задача осталась бы «занятой» навсегда
            # (как у чата агента, см. хвост `agent_chat`).
            _running_sessions.add(str(session_now["id"]))
            try:
                # 0. ИСТОЧНИК ОТВЕТА: мини-чат ходит к модели сам (ответ, память
                #    задачи), поэтому готовность источника проверяет здесь, а не
                #    только в чате агента. Неготовый источник — понятная причина в
                #    потоке, а не «не получил ответа от модели» (см. §5.14).
                notes: List[str] = []
                reason = await local_source_guard(notes.append)
                for note in notes:
                    yield encode({"type": "debug", "text": note})
                if reason:
                    yield encode({"type": "error",
                                  "text": LOCAL_SOURCE_NOT_READY % reason})
                    yield encode({"type": "done", "usage": {},
                                  "memory": memory_store.snapshot(
                                      workspace_store.task_memory(dialog_now))})
                    return
                yield encode({"type": "debug", "text": (
                    "💬 Мини-чат: отвечаю по документам проекта (план задачи не "
                    "участвует). Базы: %d, настройки поиска — как у проекта."
                    % len(enabled))})
                usage: Dict[str, Any] = {}
                answer = ""
                sources: List[Dict[str, Any]] = []
                memory = workspace_store.task_memory(dialog_now)
                turn = int(len([m for m in dialog_now["messages"]
                                if m.get("role") == "user"]) or 0) + 1
                tracker = Agent(AgentConfig(max_tokens=rag_dialog.ANSWER_MAX_TOKENS))
                try:
                    async for event in _rag_dialog_turn(
                            question, memory=memory,
                            history=rag_dialog.history_text(dialog_now["messages"]),
                            profile=profile, profile_id=profile_id, settings=settings,
                            enabled=enabled, tracker=tracker):
                        kind = event.get("type")
                        if kind == "bot":
                            answer = str(event.get("text") or "")
                            sources = list(event.get("sources") or [])
                        elif kind == "task_memory":
                            # Память задачи — в диалог: следующий ход (и обычный
                            # запрос агента) обязан её видеть.
                            memory = workspace_store.set_task_memory_value(
                                dialog_now, event.get("memory") or {})
                            yield encode({"type": "debug", "text": event.get("text")})
                            continue
                        yield encode(event)
                except Exception as exc:        # сбой хода — ответа нет, поток жив
                    logger.warning("Мини-чат: ход не выполнен — %s", str(exc)[:200])
                    yield encode({"type": "error", "text": (
                        "⚠ Мини-чат не смог ответить: %s"
                        % llm_client.redact_secrets(str(exc))[:300])})
                # ЗАПИСЬ ХОДА: реплики — в память диалога, расход — одной записью,
                # память задачи — в диалог. Состояние автомата НЕ трогаем.
                delta = tracker.usage_snapshot()
                if answer:
                    dialog_now["messages"].append({"role": "user",
                                                   "content": question})
                    dialog_now["messages"].append({"role": "assistant",
                                                   "content": answer})
                    usage = merge_usage(usage, delta)
                    if usage:
                        dialog_now["usage"].append(dict(usage))
                        _usage_matches_history(dialog_now)
                elif delta.get("requests") or delta.get("failed_requests"):
                    # Ответа нет (сбой модели), но вызовы были — расход не теряем.
                    dialog_now.setdefault("usage", []).append(
                        dict(delta, kind="service"))
                await _persist()
                yield encode({"type": "done", "usage": dict(usage),
                              "memory": memory_store.snapshot(memory)})
            finally:
                # Пометка «задача выполняется» снимается ВСЕГДА: и сбой хода, и
                # обрыв потока не должны оставлять задачу занятой навсегда.
                _running_sessions.discard(str(session_now["id"]))

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


# ---------------------------------------------------------------------------
# КОНТРОЛЬНЫЕ ДИАЛОГИ RAG: команды /test_rag_dialog_1 и /test_rag_dialog_2
#
# ЧТО ПРОВЕРЯЕТСЯ. Два РАЗНЫХ разговора по 10 реплик (app/ai/rag_dialog.py,
# SCENARIOS): старт задан хардкодом, дальше реплики пишет имитация пользователя
# (служебный вызов LLM), а отвечает тот же мини-чат, что и в жизни. Проверяется
# главное свойство разговора: агент НЕ ТЕРЯЕТ ЦЕЛЬ задачи (память задачи) и
# КАЖДЫЙ ответ приходит с источниками.
#
# МИМО ПАЙПЛАЙНА — как `/test_rag`: ни планировщика, ни подтверждения, ни шагов,
# ни автомата, ни приёмщика. В память диалога (messages), замер (usage) и автомат
# (state) прогон НЕ пишет: синтетический разговор не должен становиться историей
# рабочей задачи. В журнал чата (`log`) пишет — результат виден пользователю.
# Память задачи прогон ведёт ЛОКАЛЬНО: она показана в событиях и в вердикте, но
# в диалог не попадает (по той же причине).
#
# ЧАСТЬ ИТОГА СЧИТАЕТ КОД, А НЕ СУДЬЯ: сколько ответов пришло с источниками и
# сказал ли агент «в документах этого нет» на вопрос-ловушку — это факты потока
# (см. rag_dialog.turn_stats). Судья отвечает только за то, чего код знать не
# может: служит ли ответ цели разговора и нет ли в нём выдуманных фактов.
# ---------------------------------------------------------------------------
@router.post("/agent/rag/dialog/test")
async def rag_dialog_test(payload: RagDialogTest) -> StreamingResponse:
    """Контрольный ДИАЛОГ по базам знаний — команды `/test_rag_dialog_1|2`.

    Поток событий (NDJSON):

      * `{"type": "dialog_start", "scenario", "title", "turns", "bases", "goal",
        "plan": [...]}` — что за разговор начинается и из каких реплик он состоит;
      * `{"type": "dialog_turn", "n", "total", "text", "source"}` — реплика
        пользователя (`source`: "fixed" — из сценария, "model" — имитация);
      * `{"type": "debug", …}` — ход поиска и работа с памятью задачи;
      * `{"type": "bot", "text", "sources", "turn", "hits"}` — ответ агента
        ВМЕСТЕ С ИСТОЧНИКАМИ (как в чате: карточки фрагментов и цитаты);
      * `{"type": "task_memory", "turn", "memory", "text"}` — память задачи после
        реплики: её и проверяет тест на «не потерял ли агент цель»;
      * `{"type": "dialog_error", "n", "text"}` — шаг остался без ответа (прогон
        НЕ обрывается: остальные реплики всё равно проверяются);
      * `{"type": "dialog_verdict", "text", "items", "summary", "stats",
        "memory"}` — вердикт судьи и ФАКТЫ прогона (сколько ответов с источниками);
      * `{"type": "done", "usage": {...}}` — итог прогона и его расход.
    """
    sc = rag_dialog.scenario(payload.scenario)
    if sc is None:
        raise HTTPException(status_code=400, detail=(
            "неизвестный сценарий диалога: есть 1 и 2 (команды "
            "/test_rag_dialog_1 и /test_rag_dialog_2)"))
    task, session = _rag_dialog_target(payload.session_id)
    if task is None or session is None:
        raise HTTPException(status_code=400, detail=(
            "Сначала создайте проект — базы знаний привязаны к проекту"))
    enabled = _rag_dialog_bases(task)
    if not enabled:
        raise HTTPException(status_code=400, detail=(
            "к проекту не подключено ни одной базы знаний — включите базу "
            "кнопкой «RAG», иначе проверять нечего"))
    settings = workspace_store.rag_settings(task)
    profile_id = _current_profile_id()
    profile = _profile_block()
    dialog = session.get("dialog")
    base_names = []
    for base_id in enabled:
        meta = rag_store.get_base(base_id, profile=profile_id) or {}
        base_names.append(str(meta.get("name") or base_id))

    def log(kind: str, text: str, sources: Any = None) -> None:
        """Запись в журнал ЧАТА (не в память диалога): прогон должен быть виден."""
        if dialog is not None:
            workspace_store.add_log(dialog, kind, text, sources=sources)

    async def event_stream():
        # 0. ИСТОЧНИК ОТВЕТА. Контрольный разговор зовёт модель особенно много:
        #    имитация пользователя, каждый ответ по документам, оценка хода и
        #    память задачи. При выбранной локальной модели сервер обязан отвечать
        #    ДО первой реплики: иначе весь прогон превратился бы в десять ошибок
        #    связи, а причина осталась бы не названной (см. §5.14).
        notes: List[str] = []
        reason = await local_source_guard(notes.append)
        for note in notes:
            log(workspace_store.LOG_DEBUG, note)
            yield json.dumps({"type": "debug", "text": note},
                             ensure_ascii=False) + "\n"
        if reason:
            text = LOCAL_SOURCE_NOT_READY % reason
            log(workspace_store.LOG_ERROR, text)
            yield json.dumps({"type": "error", "text": text},
                             ensure_ascii=False) + "\n"
            yield json.dumps({"type": "done", "usage": {}},
                             ensure_ascii=False) + "\n"
            return
        rows: List[Dict[str, Any]] = []
        # ПАМЯТЬ ЗАДАЧИ прогона — ЛОКАЛЬНАЯ: синтетический разговор не должен
        # подменять память рабочей задачи. Начинается с пустой: тест проверяет,
        # что агент САМ её построит по первой реплике и не потеряет к концу.
        memory: Dict[str, Any] = memory_store.empty()
        history: List[Dict[str, str]] = []
        tracker = Agent(AgentConfig())
        plan = rag_dialog.turn_plan(sc)
        first_goal = ""
        log(workspace_store.LOG_DEBUG, (
            "🧪 Диалог «%s»: %d реплик, базы: %s. Прогон идёт мимо плана задачи — "
            "состояние и память задачи не меняются. Цель сценария: %s"
            % (sc.get("title"), rag_dialog.TURNS, ", ".join(base_names),
               sc.get("goal"))))
        yield json.dumps({"type": "dialog_start", "scenario": sc["id"],
                          "title": sc.get("title") or "",
                          "turns": rag_dialog.TURNS, "bases": base_names,
                          "goal": sc.get("goal") or "", "plan": plan},
                         ensure_ascii=False) + "\n"
        for step in plan:
            number = int(step["n"])
            source = step["source"]
            question = ""
            if source == "fixed":
                question = str(step["text"])
            else:
                # Имитация пользователя — тоже вызов модели (несколько секунд):
                # строку про неё показываем ДО вызова, иначе между репликами
                # висела тишина и прогон выглядел «молчащим» до конца.
                yield json.dumps({"type": "debug", "text": (
                    "✍️ Готовлю реплику %d/%d (имитация пользователя)…"
                    % (number, rag_dialog.TURNS))}, ensure_ascii=False) + "\n"
                # ИМИТАЦИЯ ПОЛЬЗОВАТЕЛЯ: одна реплика на служебный вызов. Сбой
                # вызова НЕ пропускает шаг — реплика берётся из плана сценария
                # (пункты сформулированы как вопросы), и разговор продолжается.
                topic = str(step.get("topic") or "")
                payload_text = rag_dialog.simulator_payload(
                    sc, number, rag_dialog.history_text(history), memory)
                try:
                    content, metrics = await llm_client.call_llm_async(
                        user_text=payload_text, model=config.active_model(),
                        disable_thinking=True,
                        max_tokens=rag_dialog.ANSWER_MAX_TOKENS,
                        messages=[{"role": "system",
                                   "content": rag_dialog.USER_SIM_PROMPT},
                                  {"role": "user", "content": payload_text}],
                        timeout=_rag_test_timeout())
                    tracker.note_usage(metrics, None)
                    question = rag_dialog.parse_user(content)
                except Exception as exc:
                    logger.warning("Диалог RAG: имитация пользователя не ответила — %s",
                                   str(exc)[:200])
                if not question:
                    question = rag_dialog.fallback_question(topic)
                    source = "script"
            log(workspace_store.LOG_USER, "🧪 Реплика %d/%d: %s"
                % (number, rag_dialog.TURNS, question))
            yield json.dumps({"type": "dialog_turn", "n": number,
                              "total": rag_dialog.TURNS, "text": question,
                              "source": source},
                             ensure_ascii=False) + "\n"
            row: Dict[str, Any] = {
                "n": number, "question": question, "answer": "", "hits": 0,
                "sources": [], "cited": False, "error": "", "metrics": {},
                "bad_cites": [], "trap": rag_dialog.is_trap(sc, number),
                "honest": False,
            }
            try:
                async for event in _rag_dialog_turn(
                        question, memory=memory,
                        history=rag_dialog.history_text(history), profile=profile,
                        profile_id=profile_id, settings=settings,
                        enabled=enabled, tracker=tracker, number=number):
                    kind = event.get("type")
                    if kind == "bot":
                        row["answer"] = str(event.get("text") or "")
                        row["sources"] = list(event.get("sources") or [])
                        row["hits"] = int(event.get("hits") or 0)
                        row["cited"] = bool(event.get("cited"))
                        row["honest"] = bool(event.get("honest"))
                        row["bad_cites"] = list(event.get("bad_cites") or [])
                        event = dict(event, turn=number)
                    elif kind == "task_memory":
                        memory = memory_store.normalize(event.get("memory") or {})
                        event = dict(event, turn=number)
                    if kind == "debug":
                        log(workspace_store.LOG_DEBUG, str(event.get("text") or ""))
                    elif kind == "bot" and row["answer"]:
                        log(workspace_store.LOG_ASSISTANT, row["answer"],
                            sources=row["sources"])
                    elif kind == "task_memory":
                        log(workspace_store.LOG_DEBUG, str(event.get("text") or ""))
                    yield json.dumps(event, ensure_ascii=False) + "\n"
            except Exception as exc:                # сбой шага прогон не обрывает
                logger.warning("Диалог RAG: реплика %d не отработана — %s",
                               number, str(exc)[:200])
                row["error"] = llm_client.redact_secrets(str(exc))[:300]
                text = ("⚠ Реплика %d осталась без ответа: %s"
                        % (number, row["error"]))
                log(workspace_store.LOG_ERROR, text)
                yield json.dumps({"type": "dialog_error", "n": number,
                                  "text": text}, ensure_ascii=False) + "\n"
            rows.append(row)
            # История разговора для следующих шагов: реплика и ответ. Хвост
            # истории обрезает сам модуль (rag_dialog.history_text).
            history.append({"role": "user", "content": question})
            if row["answer"]:
                history.append({"role": "assistant", "content": row["answer"]})
            if number == 1 and memory_store.goal_of(memory):
                first_goal = memory_store.goal_of(memory)

        # ОЦЕНКА РАЗГОВОРА: судья видит цель сценария, каждый ответ и фрагменты,
        # из которых ответ взят. Сбоя оценки прогон не отменяет.
        verdict: Dict[str, Any] = {"items": [], "summary": "", "goal_kept": None}
        stats = rag_dialog.turn_stats(rows)
        if any(row["answer"] for row in rows):
            # Судье уходит и ПАМЯТЬ ЗАДАЧИ разговора: без неё он принимал
            # названное пользователем за выдумку агента (см. JUDGE_PROMPT).
            judge_payload = rag_dialog.judge_payload(sc, rows, memory)
            try:
                content, metrics = await llm_client.call_llm_async(
                    user_text=judge_payload, model=config.active_model(),
                    disable_thinking=True, max_tokens=rag_suite.JUDGE_MAX_TOKENS,
                    messages=[{"role": "system",
                               "content": rag_dialog.JUDGE_PROMPT},
                              {"role": "user", "content": judge_payload}],
                    timeout=_rag_test_timeout())
                tracker.note_usage(metrics, None)
                verdict = rag_dialog.parse_verdict(content)
            except Exception as exc:
                logger.warning("Диалог RAG: оценка не получена — %s",
                               str(exc)[:200])
                verdict["summary"] = ("Оценку получить не удалось: %s"
                                      % llm_client.redact_secrets(str(exc))[:200])
        else:
            verdict["summary"] = "Ни одна реплика не получила ответа — оценивать нечего."
        # ЦЕЛЬ НЕ ПОТЕРЯНА — проверка КОДОМ по памяти задачи: в начале разговора
        # она построена по первой реплике, в конце обязана быть на месте. Смена
        # формулировки — не потеря (модель уточняет её по ходу), поэтому
        # сравниваем не текст, а наличие цели; потеря ловится судьёй по шагам.
        goal_now = memory_store.goal_of(memory)
        goal_line = ("Память задачи: цель %s; уточнений %d, ограничений %d, "
                     "терминов %d."
                     % ("удержана ✅" if goal_now else "⚠ ПУСТА",
                        len(memory_store.texts(memory, "clarified")),
                        len(memory_store.texts(memory, "constraints")),
                        len(memory_store.texts(memory, "terms"))))
        if first_goal and goal_now and first_goal != goal_now:
            goal_line += " Цель уточнялась по ходу: «%s» → «%s»." % (
                first_goal[:80], goal_now[:80])
        lines = rag_dialog.verdict_lines(verdict)
        text = ("🧪 Оценка диалога «%s»\n%s"
                % (sc.get("title"), rag_dialog.summary_text(verdict, stats)))
        if stats.get("bad_cites"):
            text += ("\n⚠ Ссылки на несуществующие фрагменты (номер вне выдачи, "
                     "то есть ссылка врёт) — шагов: %d, номера: %s"
                     % (stats["bad_cites"],
                        ", ".join(str(number) for number
                                  in (stats.get("bad_cite_numbers") or []))))
        # ФРАГМЕНТОВ ПОЧТИ НЕТ — это про НАСТРОЙКУ ПОИСКА, а не про агента:
        # говорим об этом прямо, называя порог проекта (живой случай 03.10:
        # порог 0,96 давал 0 ответов с источниками, и прогон выглядел провалом).
        hint = rag_dialog.yield_hint(stats, settings.get("min_score"),
                                     settings.get("min_ce"),
                                     bool(settings.get("filter")))
        if hint:
            text += "\n" + hint
        trap = next((row for row in rows if row["trap"]), None)
        if trap is not None:
            text += ("\nЛовушка (реплика %d, ответа в базе нет): %s"
                     % (trap["n"], "честно сказано «в документах этого нет» ✅"
                     if trap["honest"] else "⚠ агент не признал отсутствие данных"))
        if lines:
            text += "\n" + "\n".join(lines)
        text += "\n" + goal_line
        if verdict.get("summary"):
            text += "\n\n" + str(verdict["summary"])
        text += "\n\n" + rag_dialog.usage_line(tracker.usage_snapshot(), stats)
        log(workspace_store.LOG_ASSISTANT, text)
        yield json.dumps({"type": "dialog_verdict", "text": text,
                          "items": verdict.get("items") or [],
                          "summary": verdict.get("summary") or "",
                          "stats": stats,
                          "memory": memory_store.snapshot(memory)},
                         ensure_ascii=False) + "\n"
        yield json.dumps({"type": "done", "usage": tracker.usage_snapshot()},
                         ensure_ascii=False) + "\n"

    async def stream():
        """Поток прогона + запись журнала на диск (сбой записи прогон не отменяет)."""
        try:
            async for chunk in event_stream():
                yield chunk
        finally:
            if dialog is not None:
                try:
                    async with _workspace_lock:
                        await _persist()
                except Exception as exc:            # pragma: no cover - защита
                    logger.warning("Диалог RAG: журнал не сохранён — %s",
                                   str(exc)[:150])

    return StreamingResponse(stream(), media_type="application/x-ndjson")


def _merge_test_usage(total: Dict[str, Any], metrics: Any) -> Dict[str, Any]:
    """Складывает расход прогона (вопросы + судья) в ОТДЕЛЬНЫЙ замер теста."""
    if not isinstance(metrics, dict):
        return total
    result = dict(total or {})
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        try:
            result[key] = int(result.get(key) or 0) + int(metrics.get(key) or 0)
        except (TypeError, ValueError):
            continue
    result["calls"] = int(result.get("calls") or 0) + 1
    result["model"] = str(metrics.get("model") or result.get("model") or "")
    return result


def _rag_test_usage_line(usage: Dict[str, Any], rows: List[Dict[str, Any]]) -> str:
    """Строка расхода прогона: отдельным замером, в замер задачи не входит.

    Токены теста считаются и показываются, но НЕ складываются с расходом задачи
    (dialog["usage"]): иначе панель «Токены задачи» показывала бы работу, которой
    по запросу задачи не было.
    """
    calls = int(usage.get("calls") or 0)
    spent = 0.0
    for row in rows:
        metrics = row.get("metrics") or {}
        spent += float(metrics.get("cost_rub") or 0.0)
    line = ("Расход теста (в замер задачи не входит): вызовов %d, вход %s / выход %s "
            "токенов" % (calls, int(usage.get("prompt_tokens") or 0),
                         int(usage.get("completion_tokens") or 0)))
    if spent:
        line += ", стоимость %.4f руб." % spent
    return line


def _too_big_detail(name: str, size: int, limit: int) -> str:
    """Понятный отказ по размеру: сколько весит, каков предел и что делать.

    «Размер превышен» без чисел и без выхода из положения бесполезен: у
    пользователя остаётся только догадываться, где предел и можно ли его
    поднять. Поэтому в отказе — ФАКТИЧЕСКИЙ вес файла, предел, имя переменной
    окружения и подсказка про потоковую загрузку.
    """
    return ("Файл «%s» весит %s — это больше предела %s. Предел поднимается "
            "переменной %s; крупные файлы интерфейс отправляет потоком "
            "(до %s), поэтому обычно достаточно выбрать файл заново."
            % (name, rag_documents.human_bytes(size), rag_documents.human_bytes(limit),
               "RAG_MAX_JSON_FILE_BYTES", rag_documents.human_bytes(
                   rag_documents.MAX_FILE_BYTES)))


def _decode_upload_files(items: List[Any]) -> List[Dict[str, Any]]:
    """Разбирает загруженные файлы: base64 → байты с проверкой размера.

    Предел здесь свой и МЕНЬШИЙ, чем у потоковой загрузки (`MAX_JSON_FILE_BYTES`):
    этот путь держит файл в памяти ТРИЖДЫ — тело запроса, строка base64 и
    декодированные байты, — поэтому большие документы идут потоком
    (`/agent/rag/upload/stream`), а тут остаются небольшие файлы.

    Размер проверяется ДО декодирования (по длине base64): принимать в память
    десятки мегабайт, чтобы потом отказать, незачем. Ошибка формата у одного
    файла не отменяет остальные — как и ошибка разбора на сервере.
    """
    limit = rag_documents.MAX_JSON_FILE_BYTES
    files: List[Dict[str, Any]] = []
    for item in items:
        name = rag.sanitize_source(getattr(item, "filename", ""))
        raw = str(getattr(item, "content_base64", "") or "").strip()
        if not raw:
            raise HTTPException(status_code=400,
                                detail="Файл «%s» передан без содержимого" % name)
        # Оценка размера до декодирования: 4 символа base64 ≈ 3 байта.
        estimated = len(raw) // 4 * 3
        if estimated > limit:
            raise HTTPException(status_code=400,
                                detail=_too_big_detail(name, estimated, limit))
        try:
            data = base64.b64decode(raw, validate=False)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400,
                                detail="Содержимое файла «%s» не разобрано: %s"
                                       % (name, str(exc)[:120]))
        if len(data) > limit:
            raise HTTPException(status_code=400,
                                detail=_too_big_detail(name, len(data), limit))
        files.append({"filename": name, "data": data})
    return files


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
    с ним: его задачи со всеми диалогами, рабочая и долговременная память, а
    также его БАЗЫ ЗНАНИЙ (индексы RAG на диске).
    """
    async with _workspace_lock:
        if not profile_store.delete_profile(_profiles, profile_id):
            raise HTTPException(status_code=404, detail="Профиль не найден")
        profile_store.ensure_profile(_profiles)
        await _persist_profiles()
        # ВНЕШНИЕ СБОРЫ задач профиля останавливаются ДО удаления данных: задачи
        # (вместе с записями о своих сборах) исчезнут, и отменять их будет уже
        # нечем — а сбор на сервере остался бы висеть навсегда.
        for task in workspace_store.profile_tasks(_workspace, profile_id):
            for session in list(task.get("sessions", [])):
                await _stop_mcp_started(session, "профиль удалён")
        # БАЗЫ ЗНАНИЙ профиля удаляются вместе с ним: после удаления профиля они
        # не видны никому (список читается по владельцу), то есть остались бы
        # навсегда лежать на диске мусором, который нельзя удалить через
        # интерфейс, — а «удаление профиля» и означает «удаление его данных».
        for meta in rag_store.list_bases(profile=profile_id, with_meta=False):
            rag.delete_base(meta.get("id"), profile=profile_id)
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
    # ПАРАМЕТРЫ МОДЕЛИ ПРОЕКТА — temperature и предел длины ответа (шестерёнка ⚙️
    # проекта, app/ai/workspace.py: llm_settings). Действуют во всех задачах
    # проекта и уходят в модель ДЕЙСТВУЮЩЕГО источника: и в локальную, и в
    # удалённую — адрес и ключ выбирает клиент, а параметры генерации передаются
    # как есть. Поле «Длина» в шапке чата — лимит РАЗОВОГО запроса и главнее
    # настройки проекта: раз пусто, решение отдаётся проекту (пусто и там —
    # параметр в API не уходит вовсе, работает предел провайдера).
    project_llm = workspace_store.llm_settings(task)
    agent_max_tokens = (msg.max_tokens if msg.max_tokens is not None
                        else project_llm["max_tokens"])
    # Настройки генерации общие для ответа и для служебного плана.
    agent_settings = dict(
        max_tokens=agent_max_tokens,
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
    # Температура проекта — ТОЛЬКО ответу шага (и прямому ответу по источникам):
    # у служебных вызовов свои задачи. План, проверка результата, разбор
    # инвариантов и память задачи возвращают JSON, и высокая температура ломала бы
    # его разбор — такие вызовы температуру не передают вовсе (значение
    # провайдера), а предел их ответа задан их собственными константами
    # (PLAN_MAX_TOKENS, REVIEW_MAX_TOKENS, …), а не «Длиной» и не проектом.
    agent = Agent(AgentConfig(**agent_settings,
                              temperature=project_llm["temperature"]))
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

    async def run_direct_answer(*, task_now: Dict[str, Any],
                                session_now: Dict[str, Any],
                                dialog_now: Dict[str, Any],
                                state: "task_state.TaskState", text: str,
                                reason: str, offer_plan: bool,
                                analyzer: Agent,
                                invariants_now: Dict[str, Any], profile: str,
                                memory_now: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """ПРЯМОЙ ОТВЕТ ПО ИСТОЧНИКАМ — второй путь ОДНОГО режима «AI-агент».

        ЧЕМ ОТЛИЧАЕТСЯ ОТ ПЛАНА: только отсутствием шагов. Источники те же:
        правила проекта (гейт инвариантов), внешние инструменты (MCP — до
        подтверждения плана только ЧТЕНИЯ) и фрагменты баз знаний тем же поиском
        и с теми же настройками. Ответ строится ОДНИМ вызовом модели, под ним
        ВСЕГДА строка источников (`rag_dialog.ensure_sources` — гарантия кода),
        память задачи обновляется как в мини-чате.

        АВТОМАТ ЗАДАЧИ НЕ ТРОГАЕТСЯ: план, если он был или будет построен,
        продолжается с того же места. Именно поэтому «разговор по документам» и
        «работа по плану» могут жить в одной задаче и в одном диалоге.

        Отдаёт события чата; последним — служебное `{"type": "usage"}` с расходом
        хода. Запись в память диалога и замер делает вызывающий код: он же решает,
        считать ли ход состоявшимся (был ли ответ).
        """
        # Замер — свой агент: он копит дельту ЭТОГО хода, не смешиваясь с
        # замерами автомата (у того свои служебные вызовы).
        tracker = Agent(AgentConfig())
        total: Dict[str, Any] = {}
        # ПАРАМЕТРЫ МОДЕЛИ ПРОЕКТА действуют и на этом пути: это тот же режим
        # «AI-агент», и ответ строится тем же источником. Предел «Длины» из шапки
        # главнее проекта; пусто и там — предел ответа берётся из константы
        # движка разговора (rag_dialog.ANSWER_MAX_TOKENS). Температура — из
        # проекта; не задана — в запрос не уходит (значение провайдера).
        project_llm = workspace_store.llm_settings(task_now)
        answer_max_tokens = (msg.max_tokens if msg.max_tokens is not None
                             else project_llm["max_tokens"]
                             or rag_dialog.ANSWER_MAX_TOKENS)
        # Про источники и вариант «разложить на шаги» говорим ТОЛЬКО когда они
        # есть. Иначе строка обещала бы то, чего нет: «отвечаю по источникам» при
        # выключенном RAG и MCP — лишний шум, а вариант под ответом на простой
        # вопрос выглядит как навязанная работа. Причину выбора пути уже назвала
        # строка гейта выше — здесь она не повторяется.
        has_sources = bool(workspace_store.rag_enabled(task_now)
                           or workspace_store.mcp_enabled(task_now))
        note = (f"{_MACHINE}: отвечаю по источникам." if has_sources
                else f"{_MACHINE}: отвечаю сразу.")
        if project_llm["temperature"] is not None or project_llm["max_tokens"]:
            # Настройка проекта — из шестерёнки ⚙️: человек должен видеть, что она
            # ДЕЙСТВУЕТ, а не молча надеяться на неё (у шага с планом такую
            # строку пишет сам агент, см. Agent._merge_params).
            parts: List[str] = []
            if project_llm["temperature"] is not None:
                parts.append(f"temperature {project_llm['temperature']:g}")
            if project_llm["max_tokens"] is not None:
                parts.append(f"предел ответа {project_llm['max_tokens']} токенов")
            note += " Параметры модели проекта: " + ", ".join(parts) + "."
        if offer_plan:
            note += (" План и шаги не строю; если это была работа, под ответом "
                     "будет вариант «⚙ Разложить работу на шаги».")
        yield {"type": "debug", "text": note}
        # 1. ПРАВИЛА ПРОЕКТА: тот же гейт, что и перед планом. Запрос, нарушающий
        #    правило, не выполняется НИ ОДНИМ путём — иначе объединение путей
        #    стало бы лазейкой в правилах.
        analysis, pre_usage, preverified = await _preflight_invariants(
            task_now, session_now, text, analyzer, invariants_now)
        total = merge_usage(total, pre_usage)
        if invariants_store.blocks(analysis):
            view = _analysis_view(analysis)
            workspace_store.add_log_event(
                dialog_now, workspace_store.LOG_SUGGESTIONS,
                view["message"], view)
            yield {"type": "suggestions", "analysis": view,
                   "text": view["message"]}
            yield {"type": "usage", "usage": total}
            return
        # 2. ВНЕШНИЕ ИНСТРУМЕНТЫ: данные доступны и прямому ответу — в этом и
        #    смысл объединения (прежде «разговор по документам» их не спрашивал).
        #    `approved=False` — до подтверждения плана инструменты только читают:
        #    побочные действия откладываются, как и в пути с планом.
        mcp_data: List[Dict[str, Any]] = []
        if workspace_store.mcp_enabled(task_now):
            mcp_data, mcp_usage, mcp_lines, _ = await _preflight_mcp(
                task_now, session_now, text, analyzer, state, approved=False,
                direct=True)
            total = merge_usage(total, mcp_usage)
            for line in mcp_lines:
                yield {"type": "debug", "text": f"{_MACHINE}: {line}"}
        # 3. ФРАГМЕНТЫ БАЗ ЗНАНИЙ — тот же поиск, те же настройки проекта.
        rag_data, rag_lines, rag_searched, rag_usage = await _preflight_rag(
            task_now, session_now, text, state, agent=analyzer)
        total = merge_usage(total, rag_usage)
        for line in rag_lines:
            yield {"type": "debug", "text": f"{_MACHINE}: {line}"}
        # 3а. НИЧЕГО НЕ НАШЛОСЬ — РЕШАЕТ ПОЛЬЗОВАТЕЛЬ, как и перед планом:
        #     молчаливого перехода в общие знания нет ни на одном пути.
        if (rag_searched and not rag_search.has_hits(rag_data)
                and str(rag_data.get("general") or "") == "ask"):
            view = _rag_choice_view(rag_data, task_now)
            workspace_store.add_log_event(
                dialog_now, workspace_store.LOG_SUGGESTIONS,
                view["message"], view)
            yield {"type": "choices", "text": view["message"],
                   "options": view["options"], "analysis": view}
            yield {"type": "usage", "usage": total}
            return
        # 4. ОДИН ВЫЗОВ МОДЕЛИ: ответ по собранным источникам.
        enabled = workspace_store.rag_enabled(task_now)
        hits = rag_search.hits_of(rag_data)
        messages = rag_dialog.answer_messages(
            memory_store.block(memory_now), rag_search.block(rag_data),
            rag_dialog.history_text(dialog_now["messages"]), text, profile,
            extra_blocks=[invariants_store.block(invariants_now),
                          mcp_store.block(mcp_data)])
        content, metrics = await llm_client.call_llm_async(
            user_text=text, model=config.active_model(), disable_thinking=True,
            max_tokens=answer_max_tokens,
            temperature=project_llm["temperature"], messages=messages,
            timeout=_rag_test_timeout())
        tracker.note_usage(metrics)
        if not str(content or "").strip():
            hint = (" Повторите запрос." if not offer_plan else
                    (" Повторите запрос или отправьте его как задачу (вариант "
                     "«⚙ Разложить работу на шаги»)."))
            # ПРИЧИНА ВАЖНЕЕ ФОРМУЛИРОВКИ: у локальной модели сбой вызова имеет
            # конкретное объяснение (поток генерации умер, веса не нашлись,
            # сервер не поднялся), и «вызов не удался» без него отправляло бы
            # человека искать причину в запросе и документах (живой случай 09.10).
            reason = ""
            if config.llm_source() == "local":
                reason = local_llm.broken_reason() or local_llm.status()["hint"]
            reason_text = f" Причина: {reason}" if reason else ""
            yield {"type": "error",
                   "text": "⚠ Ответа от модели нет: вызов не удался."
                           + reason_text + hint}
            yield {"type": "usage",
                   "usage": merge_usage(total, tracker.usage_snapshot())}
            return
        answer = rag_dialog.ensure_sources(content, hits, searched=len(enabled),
                                           cut=_rag_cut_note(rag_data))
        yield {"type": "bot", "text": answer,
               "sources": rag_search.sources(rag_data), "hits": len(hits),
               "cited": rag_dialog.has_citation(answer),
               "bad_cites": rag_dialog.bad_citations(answer, len(hits))}
        # ОТВЕТ ПОКАЗАН — ВВОД СВОБОДЕН. Дальше идёт СЛУЖЕБНАЯ работа (память
        # задачи, варианты продолжения) своим вызовом модели, и на локальной
        # модели она занимает секунды: поток держал поле ввода занятым до самого
        # конца, и человек, уже читая ответ, ещё несколько секунд видел «AI-агент
        # думает…» и не мог писать (живая жалоба 10.10). Интерфейс освобождает
        # ввод по этому событию, а кнопки памяти и варианты дорисуются, когда
        # подоспеют.
        yield {"type": "answer_ready"}
        # 5. ПАМЯТЬ ЗАДАЧИ — как в мини-чате: цель, уточнения, ограничения.
        async for event in _rag_memory_update(text, answer, memory_now, tracker):
            yield event
        # 6. ВОССТАНОВЛЕНИЕ ПОСЛЕ ОШИБКИ ГЕЙТА: та же работа, но планом. Вариант
        #    отправляет ГОТОВУЮ ФРАЗУ (см. rag_dialog.plan_choice) — сервер узнаёт
        #    по ней решение, и кнопка работает даже после перезагрузки страницы.
        #    Показывается он ТОЛЬКО когда путь выбран не «просто вопросом»
        #    (пустая причина гейта): под ответом на обычный вопрос это выглядело
        #    навязанной работой, а не подсказкой. Кому нужно — у того есть тип
        #    задачи «всегда сразу ответ» или просьба «ответь прямо»: там причина
        #    названа, и вариант на месте.
        if offer_plan:
            option = rag_dialog.plan_choice(text)
            offer = ("Если это была работа, а не вопрос, её можно разложить на шаги: "
                     "план с подтверждением, шаги и проверка результата.")
            yield {"type": "choices", "text": offer, "options": [option],
                   "analysis": {"message": offer, "options": [option],
                                "kind": "plan_offer"}}
        yield {"type": "usage",
               "usage": merge_usage(total, tracker.usage_snapshot())}

    async def event_stream():
        # Переменные объявлены заранее: в аварийной ветке (except) нужно
        # сохранить состояние и закрыть поток событием "done".
        state: Optional[task_state.TaskState] = None
        session_now: Optional[Dict[str, Any]] = None
        # Данные внешних инструментов MCP по текущему запросу задачи: заполняются
        # ДО планирования (см. _preflight_mcp) и уходят в план, ответ и проверку.
        mcp_data: List[Dict[str, Any]] = []
        # Фрагменты баз знаний (RAG) по текущему запросу задачи: заполняются
        # сразу после данных MCP (см. _preflight_rag) и уходят в план, ответ и
        # проверку тем же системным блоком.
        rag_data: Dict[str, Any] = {}
        # Источники под финальным ответом: фрагменты, которые были у модели
        # (карточки «файл · раздел · близость» в интерфейсе).
        rag_sources: List[Dict[str, Any]] = []
        # Файлы, полученные инструментами в ЭТОМ запросе (карточки в чате), и
        # признак «готовый результат уже есть» (по нему не спрашиваем
        # подтверждение плана: работа сделана, ждать пользователя нечего).
        mcp_files: List[Dict[str, Any]] = []
        mcp_delivered = False
        # Карточка файла, которую показываем ПОСЛЕ финального отчёта (см. ниже).
        mcp_files_event: Optional[Dict[str, Any]] = None
        mcp_files_shown = False
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
                # ПАМЯТЬ ЗАДАЧИ (см. app/ai/task_memory.py): выжимку разговора
                # ведёт мини-чат по базам знаний, а читают ВСЕ вызовы задачи —
                # план, ответ шага и проверка результата. Ставим её каждому
                # агенту один раз на запрос: иначе, вернувшись из мини-чата в
                # режим «AI-агент», пользователь потерял бы зафиксированные в
                # разговоре цель, уточнения и ограничения — они бы просто не
                # попали в контекст. Пустая память блока не даёт вовсе, поэтому
                # там, где мини-чат не работал, поведение прежнее.
                task_memory_now = workspace_store.task_memory(dialog_now)
                for worker in (agent, planner, reviewer, analyzer):
                    worker.task_memory = task_memory_now
                # «Пауза»/«Отменить», нажатые в предыдущем шаге, могли не успеть
                # примениться (блокировку держал поток) — применяем до работы.
                _apply_pending_stop(session_now, state)
                text = (msg.content or "").strip()
                # Запрос от АВТОМАТА (continue_step): пользователь ничего не
                # писал — выполняется текущий шаг плана. Так интерфейс ведёт
                # задачу сам (после «Подтвердить план» и между шагами), и
                # пользователю не нужно придумывать сообщение ради шага.
                machine_step = bool(getattr(msg, "continue_step", False))
                # Запрос от АВТОЗАПУСКА ПЕРИОДИЧЕСКОЙ задачи (см.
                # app/periodic_runner.py): пользователя за клавиатурой нет —
                # прогон идёт автономно, данные внешних инструментов берутся
                # заново, а реплика помечается в журнале как автозапуск.
                periodic_run = bool(getattr(msg, "periodic", False)) and not machine_step
                # ПЕРИОДИЧЕСКАЯ задача (расписание есть у САМОЙ задачи): у неё свой
                # ход работы — план строится ОДИН раз на запрос и дальше
                # переиспользуется, итоговой проверки результата нет, а «готово»
                # не ставится вовсе: задача повторяется, пока её не остановит
                # пользователь (см. app/ai/periodic.py, §5.12).
                periodic_task = bool(workspace_store.periodic_meta(session_now))
                # Запрос периодической задачи ИЗМЕНИЛСЯ (пользователь написал новый):
                # прежний план к нему не подходит — план строится заново, тоже один
                # раз. Заполняется в блоке 2а ниже.
                request_changed = False
                # Команда остановки, полученная во время текущего шага (см. ниже).
                stopped: Optional[str] = None
                # Проверку отложили в ЭТОМ ЖЕ запросе («Пауза» на последнем шаге):
                # выполнять её сразу нельзя — задача должна остаться на этапе
                # validation и ждать «Продолжить».
                validation_deferred_here = False
                # Задача перезапущена ПОСЛЕ ОШИБКИ: только в этом случае прежний
                # план (если запрос и правила те же) переиспользуется — при
                # обычном ответе пользователя план по-прежнему строится заново,
                # чтобы правки к плану не потерялись.
                restarted_from_failure = False
                if text and not machine_step:
                    # Реплика пользователя — в журнал чата: в память диалога она
                    # попадает только вместе с ответом, а при построении плана
                    # ответа нет, и текст запроса терялся (в восстановленном
                    # диалоге его не было видно).
                    if periodic_run:
                        # Автозапуск — не реплика пользователя, а служебная
                        # строка: видно, что задача повторяется сама и с каким
                        # периодом (тот же текст запроса пользователь уже писал).
                        meta = workspace_store.periodic_meta(session_now)
                        workspace_store.add_log(dialog_now, workspace_store.LOG_PERIODIC, (
                            f"{periodic_store.AUTO_MARK} Автозапуск "
                            f"({periodic_store.label(meta.get('interval'))}): {text}"
                        ))
                    else:
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

                # 0. ИСТОЧНИК ОТВЕТА. Выбрана локальная модель — сервер MLX
                #    должен отвечать ДО первого вызова: иначе запрос ушёл бы в
                #    закрытую дверь, а ответ вернулся пустым (и выглядел бы как
                #    «модель промолчала»). Если сервер не запущен, он поднимается
                #    сам, а ожидание загрузки весов идёт в отдельном потоке:
                #    человек видит строку прогресса, а не зависший чат.
                notes: List[str] = []
                reason = await local_source_guard(notes.append)
                for note in notes:
                    yield encode({"type": "debug", "text": note})
                if reason:
                    yield encode(_state_event(session_now, state))
                    yield encode({"type": "error",
                                  "text": LOCAL_SOURCE_NOT_READY % reason})
                    yield encode({"type": "done", "usage": {},
                                  "state": _state_snapshot(session_now, state)})
                    return

                # 1. ПАУЗА. Автомат остановлен кнопкой «Пауза»: шаг не
                #    выполняем, просим нажать «Продолжить».
                if state.paused:
                    yield encode(_state_event(session_now, state))
                    yield encode({"type": "error", "text": (
                        "Задача на паузе: нажмите «Продолжить» в полосе состояния, "
                        "чтобы продолжить работу."
                    )})
                    yield encode({"type": "done", "usage": {}, "state": _state_snapshot(session_now, state)})
                    return

                if not text and not machine_step:
                    yield encode({"type": "bot", "text": "Пожалуйста, введите сообщение."})
                    yield encode({"type": "done", "usage": {}, "state": _state_snapshot(session_now, state)})
                    return
                if machine_step and state.stage not in ("execution", "validation"):
                    # Автомат сам приходит только за шагом (execution) или за
                    # ОТЛОЖЕННОЙ проверкой результата (validation). В остальных
                    # этапах он ждёт пользователя: план, ошибка, завершение.
                    yield encode(_state_event(session_now, state))
                    yield encode({"type": "error", "text": (
                        "Работать нечего: задача на этапе "
                        f"«{task_state.STAGE_LABELS.get(state.stage, state.stage)}». "
                        "Отправьте сообщение или подтвердите план."
                    )})
                    yield encode({"type": "done", "usage": {}, "state": _state_snapshot(session_now, state)})
                    return

                # 1а. ИНВАРИАНТЫ. Правило проекта и правило задачи могут
                #     противоречить друг другу: выбор («главнее проект» или
                #     «главнее задача») делает пользователь, и пока решения нет,
                #     агент не работает — иначе он молча нарушил бы одно из
                #     правил. Проверка и решение — в модалке «Инварианты».
                # 2. ЭТАП: смотрим, где задача, и что означает это сообщение.
                confirmed = _is_plan_confirmation(text)
                # Автозапуск периодической задачи равносилен режиму «работай
                # автономно»: подтвердить план пользователю негде — его нет за
                # клавиатурой, а задача должна повторяться сама.
                autonomous = _wants_autonomous(text) or periodic_run
                restart = _wants_restart(text)

                # 2а. РАСПИСАНИЕ ПЕРИОДИЧЕСКОЙ ЗАДАЧИ. Сообщение ПОЛЬЗОВАТЕЛЯ в
                #     такой задаче — её новый запрос: он и повторяется по
                #     расписанию. Период берём из текста («Сводка погоды в Москве
                #     за последние сутки, раз в час»), а если он не назван —
                #     оставляем прежний (у новой задачи это сутки по умолчанию).
                #     Служебные фразы («ок», «работай автономно», «перезапусти»)
                #     запросом НЕ являются: они не меняют ни текст задачи, ни её
                #     расписание.
                if periodic_run:
                    yield encode({"type": "debug", "text": (
                        f"{_MACHINE}: автозапуск периодической задачи "
                        f"({periodic_store.label(workspace_store.periodic_meta(session_now).get('interval'))}) — "
                        "выполняю её запрос по сохранённому плану, данные внешних "
                        "инструментов беру свежими."
                    )})
                elif text and not machine_step and not confirmed and not autonomous and not restart:
                    meta = workspace_store.periodic_meta(session_now)
                    if meta:
                        previous_request = str(meta.get("request") or "").strip()
                        request_changed = bool(previous_request) and text != previous_request
                        raw_interval, interval, phrase = \
                            periodic_store.parse_request_detailed(text)
                        moment = periodic_store.now()
                        workspace_store.set_periodic(session_now, periodic_store.reschedule(
                            meta, interval=interval, request=text, moment=moment))
                        if interval is not None:
                            # Период назван меньше допустимого — говорим об этом
                            # прямо: «раз в 5 секунд» станет «раз в минуту», и
                            # пользователь должен видеть, что это НАШ предел
                            # (повтор — целый цикл задачи, то есть вызовы LLM), а
                            # не интервал сбора внешнего инструмента: его задаёт
                            # сам сервер инструмента.
                            raise_note = ""
                            if raw_interval and interval and raw_interval != interval:
                                raise_note = (
                                    f" (в запросе «{phrase}», но чаще "
                                    f"«{periodic_store.label(interval)}» повторять "
                                    f"нельзя: повтор — это целый цикл задачи)")
                            yield encode({"type": "debug", "text": (
                                f"{_MACHINE}: задача периодическая — «{phrase}»: "
                                f"повторю её запрос {periodic_store.label(interval)}"
                                + raise_note
                                + (", первый повтор — через указанный период."
                                   if not meta.get("runs") else ".")
                                + " Интервал сбора внешнего инструмента задаёт его "
                                  "сервер, период повтора на него не влияет."
                            )})
                        elif not meta.get("enabled"):
                            yield encode({"type": "debug", "text": (
                                f"{_MACHINE}: повтор этой задачи остановлен "
                                "(🔁 в списке задач) — запрос выполню один раз, "
                                "автозапуск останется выключенным."
                            )})

                # ЖДЁТ ЛИ РЕШЕНИЯ УЖЕ ПОКАЗАННЫЙ ПЛАН. Считаем ЭТО до переходов
                # этапов: этап awaiting_user ниже превратится в planning, а
                # разница между «плана нет» и «план ждёт «ок»» для гейта важна —
                # в ожидании плана текст пользователя обычно ЕГО ПРАВКА, а не
                # новый разговор (см. условие гейта ниже).
                plan_pending = bool(state.steps) or state.stage == "awaiting_user"
                if state.stage in ("done", "cancelled"):
                    # Предыдущая задача завершена — это НОВАЯ задача: автомат
                    # рождается заново (planning) с записью о сбросе в истории.
                    state = task_state.reset(
                        session_now["id"], state,
                        "предыдущая задача завершена — начинаю новую по новому запросу",
                        autonomous=autonomous,
                    )
                if state.stage == "failed":
                    restarted_from_failure = True
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
                if periodic_task and request_changed and state.stage == "execution" \
                        and state.steps:
                    # Периодическая задача с уже готовым планом: пользователь
                    # написал ДРУГОЙ запрос — прежний план к нему не подходит.
                    # Строим план заново (и тоже один раз — дальше он живёт, пока
                    # запрос не изменится снова).
                    task_state.replan(
                        state,
                        "запрос периодической задачи изменился — план строю заново",
                    )
                if autonomous:
                    state.autonomous = True
                yield encode(_state_event(session_now, state))

                # 1б-0. ОТВЕТ ИЛИ ПЛАН — ГЕЙТ (см. `_plan_needed`). ОДИН РЕЖИМ
                #       умеет и отвечать по источникам, и работать по плану:
                #       источники у обоих путей ОДНИ (правила проекта, внешние
                #       инструменты, фрагменты баз знаний), разница — нужна ли
                #       ПРОЦЕДУРА. Решает КОД по признакам запроса, человек
                #       переопределяет типом задачи или вариантом в ответе.
                #       Служебные реплики («ок», «работай автономно», шаг плана,
                #       автозапуск) гейт не проходят вовсе: это не новый запрос.
                #       ГЕЙТ РАБОТАЕТ ТОЛЬКО НА ЭТАПЕ ПЛАНИРОВАНИЯ — и это не
                #       мелочь: в execution текст пользователя означает «выполни
                #       текущий шаг», в validation — отложенную проверку. Прямой
                #       ответ там подменил бы работу шага (так и вышло в первой
                #       редакции: шаг «выполняй» превращался в ответ по базе).
                #       Завершённая/отменённая/ошибочная задача к этому месту уже
                #       сброшена в planning (см. переходы этапов выше), поэтому
                #       новый вопрос после «готово» отвечается прямо, а не планом.
                #       ЕСЛИ ПЛАН УЖЕ ПОКАЗАН и ждёт «ок», прямым ответом уходит
                #       только ВОПРОС (человек спрашивает, а не правит план), и
                #       показанный план при этом остаётся ждать подтверждения.
                #       Остальной текст — правка плана, ей занимается автомат.
                direct_allowed = (not plan_pending) or task_state.is_question(text)
                if text and not machine_step and not confirmed and not autonomous \
                        and not restart and state.stage == "planning" \
                        and direct_allowed:
                    plan_needed, plan_reason = _plan_needed(
                        text, session_now,
                        force_plan=bool(getattr(msg, "force_plan", False)),
                        periodic=bool(periodic_task))
                    if plan_needed:
                        yield encode({"type": "debug", "text": (
                            f"{_MACHINE}: строю план — {plan_reason}."
                        )})
                    else:
                        yield encode({"type": "debug", "text": (
                            f"{_MACHINE}: плана не будет — {plan_reason or 'работа в один шаг'}."
                        )})
                        direct_answer = ""
                        direct_answered = False
                        async for event in run_direct_answer(
                                task_now=task_now, session_now=session_now,
                                dialog_now=dialog_now, state=state, text=text,
                                reason=plan_reason or "работа в один шаг",
                                # Пустая причина гейта = «просто вопрос» (ни
                                # признаков работы, ни просьбы ответить без плана):
                                # только тогда вариант «разложить на шаги» под
                                # ответом не показывается.
                                offer_plan=bool(str(plan_reason or "").strip()),
                                analyzer=analyzer, invariants_now=invariants_now,
                                profile=profile, memory_now=task_memory_now):
                            kind = event.get("type")
                            if kind == "usage":
                                # Расход хода (ответ + поиск + память) — одной
                                # записью в замер задачи, как у шага плана.
                                usage = merge_usage(usage, event.get("usage") or {})
                                continue
                            if kind == "bot" and str(event.get("text") or "").strip():
                                direct_answered = True
                                direct_answer = str(event["text"])
                            elif kind == "task_memory":
                                # Память задачи — в диалог: следующий ход (и план,
                                # если пользователь его попросит) её видит.
                                workspace_store.set_task_memory_value(
                                    dialog_now, event.get("memory") or {})
                                # Строку в чат пишем, ТОЛЬКО когда память
                                # изменилась: иначе повторный вопрос давал бы ту
                                # же строку второй раз подряд.
                                if event.get("changed") is not False:
                                    yield encode({"type": "debug",
                                                  "text": event.get("text")})
                                continue
                            elif kind == "choices":
                                # Варианты («разложить на шаги») — в журнал: по нему
                                # окно восстанавливается вместе с кнопками.
                                workspace_store.add_log_event(
                                    dialog_now, workspace_store.LOG_SUGGESTIONS,
                                    event.get("text"), event.get("analysis"))
                            yield encode(event)
                        # ЗАПИСЬ ХОДА: реплики — в память диалога, расход — одной
                        # записью (служебной, если ответа не было: токены всё
                        # равно потрачены). Состояние автомата НЕ трогаем: план,
                        # если он был, продолжается с того же места.
                        if direct_answered:
                            dialog_now["messages"].append(
                                {"role": "user", "content": text})
                            dialog_now["messages"].append(
                                {"role": "assistant", "content": direct_answer})
                        if usage:
                            dialog_now["usage"].append(
                                dict(usage) if direct_answered
                                else dict(usage, kind="service"))
                            _usage_matches_history(dialog_now)
                        # «Пауза»/«Отменить», нажатые во время ответа: шагов тут
                        # нет, но команда должна примениться — полоса состояния
                        # обязана отреагировать.
                        stopped_now = _pending_stops.pop(str(session_now["id"]), None)
                        if stopped_now is not None:
                            _apply_pending_stop(session_now, state, force=stopped_now)
                        await _persist()
                        yield encode(_state_event(session_now, state))
                        yield encode({"type": "done", "usage": dict(usage),
                                      "state": _state_snapshot(session_now, state)})
                        return

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
                        # Расход этого запроса (разбор запроса + проверка
                        # вариантов) сохраняем служебной записью: ответа
                        # пользователю не было, но токены потрачены — иначе
                        # замер исчезал бы после перезагрузки страницы.
                        if usage:
                            dialog_now.setdefault("usage", []).append(
                                dict(usage, kind="service"))
                        yield encode(_state_event(session_now, state))
                        yield encode({"type": "done", "usage": dict(usage),
                                      "state": _state_snapshot(session_now, state)})
                        return

                # 1в. ВНЕШНИЕ ИНСТРУМЕНТЫ (MCP). Если у проекта включены
                #     MCP-серверы (погода, курсы валют, цены), агент сам решает,
                #     какие из них нужны для запроса, и вызывает их ДО
                #     планирования: план и ответ должны строиться по фактическим
                #     данным, а не по догадке (см. app/ai/mcp.py). Данные по
                #     запросу задачи сохраняются в диалоге — шаги плана и
                #     проверка результата приходят отдельными запросами и
                #     берут их оттуда, не выбирая инструменты заново.
                #     Ответственный расход этого запроса — отдельная служебная
                #     строка («из них служебные вызовы», вид "mcp").
                #     ПОСЛЕДНИЙ ЛИ ЭТО ШАГ ПЛАНА: результат работы (сохранение
                #     данных и файл) выдаётся именно на нём — перед финальным
                #     отчётом. До последнего шага и до подтверждения плана внешние
                #     инструменты только читают.
                _steps_total = int(getattr(state, "steps_total", 0) or 0)
                last_step = bool(_steps_total) and int(state.step_number or 0) >= _steps_total
                if workspace_store.mcp_enabled(task_now):
                    if not machine_step:
                        yield encode({"type": "debug", "text": (
                            f"{_MACHINE}: " + _mcp_debug()
                        )})
                    mcp_data, mcp_usage, mcp_lines, mcp_fresh = await _preflight_mcp(
                        task_now, session_now, text, analyzer, state,
                        machine_step=machine_step,
                        # Подтверждение плана и фразы управления («работай
                        # автономно», «перезапусти») — не новый запрос: данные
                        # уже собраны по запросу задачи. Любой другой текст —
                        # новый запрос (в том числе правка запроса до подтверждения
                        # плана): инструменты выбираются по НЕМУ.
                        reuse=(confirmed or autonomous or restart),
                        # АВТОЗАПУСК периодической задачи: запрос тот же, но
                        # данные нужны СВЕЖИЕ — иначе повтор отдавал бы в чат
                        # числа прошлого повтора (подпись «серверы + запрос» у
                        # них та же).
                        fresh=periodic_run,
                        # ГИБРИДНАЯ СХЕМА: до подтверждения плана внешние
                        # инструменты только ЧИТАЮТ. Вызовы, которые создают
                        # РЕЗУЛЬТАТ (сохранение набора, выгрузка файла), выполняются
                        # на ПОСЛЕДНЕМ шаге плана — перед финальным отчётом: так
                        # пользователь видит сначала готовый ответ, а под ним
                        # таблицу, а не наоборот. До последнего шага и до «ок»
                        # результат откладывается (см. `_resume_mcp_chain`).
                        approved=bool((confirmed or machine_step or autonomous
                                       or periodic_run) and last_step),
                        # А ЦЕПОЧКУ продолжаем сразу после подтверждения — на любом
                        # шаге: шаг «получить данные» обязан идти с данными, иначе
                        # он отвечает «данных нет», и этот ответ уходит в контекст
                        # следующего шага (живая задача: прогноз добывался только на
                        # последнем шаге, шаг 1 ответил без погоды, шаг 2 повторил
                        # его вывод, и проверка отклонила оба шага).
                        resume_chain=bool(confirmed or machine_step or autonomous
                                          or periodic_run))
                    if mcp_usage:
                        usage = merge_usage(usage, mcp_usage)
                    for line in mcp_lines:
                        yield encode({"type": "debug", "text": f"{_MACHINE}: {line}"})
                    # ФАЙЛЫ от MCP-инструментов (xlsx и т. п.): их отдаём
                    # пользователю КАРТОЧКОЙ в чате со ссылкой на скачивание
                    # (см. GET /api/agent/files/{id}). Узел пишется в журнал
                    # сессии, поэтому карточки видны и после переключения задачи.
                    # Показываем только файлы, полученные СЕЙЧАС: шаги плана и
                    # проверка берут данные из диалога, и без этой проверки одна и
                    # та же карточка появлялась бы в чате на каждом шаге.
                    mcp_files = mcp_store.attachments_of(mcp_data) if mcp_fresh else []
                    # Готовый результат (файл или сохранённый набор), полученный в
                    # ЭТОМ запросе: по нему решается, нужно ли вообще спрашивать
                    # подтверждение плана (см. ниже).
                    mcp_delivered = bool(mcp_fresh) and mcp_store.produced_result(mcp_data)
                    if mcp_files:
                        # КАРТОЧКА ФАЙЛА — ПОСЛЕ ФИНАЛЬНОГО ОТЧЁТА, а не до него:
                        # пользователь видит сначала ответ, а под ним — таблицу
                        # (см. `mcp_files_event` ниже). В журнал узел попадёт САМ:
                        # encode() пишет каждое событие через _log_event, а ручная
                        # запись здесь давала ДВЕ одинаковые карточки в чате.
                        mcp_files_event = {
                            "type": "bot",
                            "text": _mcp_files_text(mcp_files),
                            "files": mcp_files,
                        }

                # 1г. БАЗЫ ЗНАНИЙ (RAG). Если у проекта включены базы знаний,
                #     агент ищет в них фрагменты по этому запросу ДО планирования:
                #     план и ответ должны строиться по документам пользователя, а
                #     не по догадке (см. app/ai/rag_search.py). Найденные
                #     фрагменты уходят в модель отдельным системным блоком,
                #     сохраняются в диалоге под подписью запроса (шаги плана и
                #     проверка берут их оттуда, а не ищут заново) и показываются
                #     пользователю карточками источников под финальным ответом.
                #     Поиск идёт ЛОКАЛЬНО (эмбеддинги + перебор индекса) и в ДВА
                #     ЭТАПА: широкий пул кандидатов, затем реранкинг и порог —
                #     настройки у проекта (панель «Поиск и ответы»). Токены тратит
                #     только переформулировка запроса, и лишь когда она включена
                #     (служебный вызов, вид "rewrite").
                if workspace_store.rag_enabled(task_now):
                    if not machine_step:
                        yield encode({"type": "debug", "text": (
                            f"{_MACHINE}: "
                            + _rag_debug(workspace_store.rag_settings(task_now))
                        )})
                    rag_data, rag_lines, rag_searched, rag_usage = await _preflight_rag(
                        task_now, session_now, text, state,
                        # Служебные фразы и подтверждение плана — не новый запрос:
                        # фрагменты по запросу задачи уже найдены.
                        reuse=(confirmed or autonomous or restart),
                        machine_step=machine_step,
                        # АВТОЗАПУСК периодической задачи: запрос тот же, но
                        # документы могли переиндексировать — ищем заново.
                        fresh=periodic_run,
                        # Переформулировка запроса — служебный вызов модели
                        # (см. app/ai/rag_query.py): он делается через агента,
                        # чтобы токены попали в замер запроса с видом "rewrite".
                        agent=analyzer)
                    if rag_usage:
                        usage = merge_usage(usage, rag_usage)
                    for line in rag_lines:
                        yield encode({"type": "debug", "text": f"{_MACHINE}: {line}"})
                    # 1г-2. В ДОКУМЕНТАХ НИЧЕГО НЕТ — РЕШАЕТ ПОЛЬЗОВАТЕЛЬ.
                    #       Базы включены, поиск по НОВОМУ запросу прошёл, а
                    #       фрагментов нет (или порог отсёк всё): агент НЕ уходит
                    #       сам в общие знания, а останавливается и предлагает
                    #       варианты. Ответ строится только после его решения —
                    #       тогда в тексте запроса будет разрешение, и поиск
                    #       вернётся с состоянием "allowed" (см. _rag_general_state).
                    if (rag_searched and not rag_search.has_hits(rag_data)
                            and str(rag_data.get("general") or "") == "ask"):
                        view = _rag_choice_view(rag_data, task_now)
                        # Узел журнала: без него сообщение исчезло бы при
                        # переключении задачи (журнал — единственное место, где
                        # живёт переписка чата).
                        workspace_store.add_log_event(
                            dialog_now, workspace_store.LOG_SUGGESTIONS,
                            view["message"], view)
                        yield encode({"type": "choices", "text": view["message"],
                                      "options": view["options"], "analysis": view})
                        if state.stage == "planning" and not state.steps:
                            task_state.await_confirmation(
                                state, [],
                                "в документах проекта ответа нет — жду решения пользователя")
                        # Расход этого запроса (сверка инвариантов, MCP, поиск)
                        # сохраняем служебной записью: ответа не было, но токены
                        # потрачены — иначе замер исчез бы после перезагрузки.
                        if usage:
                            dialog_now.setdefault("usage", []).append(
                                dict(usage, kind="service"))
                        yield encode(_state_event(session_now, state))
                        yield encode({"type": "done", "usage": dict(usage),
                                      "state": _state_snapshot(session_now, state)})
                        return
                    # ИСТОЧНИКИ под ответом: показываем фрагменты, которые были у
                    # модели. Решаем это здесь, а подпись к карточкам — в интерфейсе:
                    # «подобрано по запросу», а не «использовано в ответе» (что
                    # именно попало в текст, решает модель).
                    if rag_searched or rag_data:
                        rag_sources = rag_search.sources(rag_data)

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
                        yield encode(_state_event(session_now, state))
                    # Новая попытка проверки: признак «проверку выполнить не
                    # удалось» снимается — иначе интерфейс показывал бы «Принять
                    # вручную» и во время самой проверки.
                    was_blocked = bool(state.check_blocked)
                    task_state.validation_checking(
                        state, "повторная попытка проверки результата")
                    if was_blocked:
                        yield encode(_state_event(session_now, state))
                    yield encode({"type": "debug", "text": (
                        f"{_MACHINE}: этап validation — самопроверка полученного ответа "
                        f"({state.steps_total or 1} "
                        f"{task_state.steps_word(state.steps_total or 1)}, "
                        + ("ответ и обмен взяты из диалога)." if resumed
                           else "ответ получен, обмен сохранён).")
                    )})
                    self_ok, note = _self_check(
                        answered=answered_step,
                        errors=step_errors,
                        steps_total=state.steps_total,
                        steps_done=state.step_index + 1,
                        stored=stored_exchange,
                    )
                    # ok — итог проверки: самопроверка, а затем вердикт модели.
                    ok = self_ok
                    redo_step = state.step_index
                    # Вердикт содержательной проверки: None — проверку выполнить
                    # НЕ удалось (модель не ответила / ответ не разобран) либо она
                    # пропущена по команде остановки. Это НЕ «не принято».
                    review: Optional[Dict[str, Any]] = None
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
                            # Проверка видит те же данные MCP, что и ответ: без них
                            # «сходил за погодой» выглядело бы выдуманным числом.
                            mcp=mcp_data,
                            # И те же фрагменты баз знаний: иначе ссылка на документ
                            # пользователя выглядела бы выдуманным источником.
                            rag=rag_data,
                        )
                        if reviewer.last_usage:
                            usage = merge_usage(usage, reviewer.last_usage)
                            # Расход запроса вырос (проверка — служебный вызов):
                            # сообщаем интерфейсу ОБНОВЛЁННЫЙ замер.
                            yield encode({"type": "usage", "usage": dict(usage)})
                        if review is None:
                            yield encode({"type": "debug", "text": (
                                f"{_MACHINE}: содержательная проверка не получена "
                                "(модель не ответила) — задачу готовой не объявляю: "
                                "нужно повторить проверку или принять результат вручную."
                            )})
                        else:
                            if review.get("steps"):
                                # Проверка идёт по КАЖДОМУ шагу: показываем разбор,
                                # чтобы возврат на доработку был объяснён.
                                yield encode({"type": "debug", "text": (
                                    f"{_MACHINE}: разбор по шагам — " + "; ".join(
                                        f"шаг {item['n']}: "
                                        + ("принят" if item["ok"] else "НЕ принят")
                                        + (f" ({item['comment']})" if item["comment"] else "")
                                        for item in review["steps"])[:400]
                                )})
                            if review["ok"]:
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
                    if review is None and self_ok:
                        # Проверку выполнить НЕ удалось: ни «принято», ни «не
                        # принято». Задачу готовой не объявляем и на доработку не
                        # возвращаем (недоступная проверка — не отказ проверки,
                        # доработки не тратятся): решение за пользователем —
                        # «▶ повторить проверку» или «Принять вручную».
                        # Ветка стоит ДО «проверка пройдена»: иначе задача
                        # закрывалась бы по одной самопроверке, которая про
                        # соответствие результата задаче ничего не знает.
                        task_state.validation_blocked(
                            state,
                            "проверку результата выполнить не удалось: "
                            + ("команда остановки" if stopped is not None
                               else "содержательная проверка не получена"),
                        )
                        if stopped is not None:
                            yield encode({"type": "error", "text": (
                                "⚠️ Проверка результата пропущена (поступила команда "
                                "остановки) — задачу готовой не объявляю."
                            )})
                        else:
                            yield encode({"type": "error", "text": (
                                "⚠️ Проверку результата выполнить не удалось: модель не "
                                "ответила. Задачу готовой не объявляю — «▶ повторить "
                                "проверку» в полосе состояния запустит проверку снова, "
                                "«Принять вручную» завершит задачу без проверки."
                            )})
                    elif ok:
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
                    # План, который уже построен для запроса ЭТОЙ задачи и при тех
                    # же правилах, не строим заново: повторные вызовы планировщика
                    # и код-гейта — оплата за тот же результат (задача перезапущена
                    # после ошибки: `start_planning` шаги сохраняет).
                    #
                    # Переиспользование включается ТОЛЬКО в автономном режиме, где
                    # подтверждение плана не требуется: в интерактивной задаче план
                    # по-прежнему показывается заново и ждёт «ок» — иначе
                    # пользователь потерял бы шаг подтверждения.
                    # ПЕРИОДИЧЕСКАЯ задача — второй такой случай: её план строится
                    # ОДИН раз на запрос (в том числе когда задача пришла на
                    # автозапуск с показанным, но не подтверждённым планом) —
                    # планировщик и гейт плана за тот же запрос больше не платятся.
                    _snapshot_for_plan = _invariants_snapshot(task_now, session_now)
                    # ЗАПРОС, для которого строится или переиспользуется план.
                    # Служебные фразы («ок», «работай автономно», «перезапусти») и
                    # реплика шага описывают ПРЕЖНИЙ запрос задачи (state.request);
                    # новый содержательный текст пользователя — сам является
                    # запросом. Без этого разделения повтор периодической задачи
                    # или правка запроса переиспользовали бы ЧУЖОЙ план: подпись
                    # совпадала бы со старой, и шаги прежнего запроса выполнялись бы
                    # по новому.
                    plan_request = (state.request or text) \
                        if (machine_step or confirmed or restart) \
                        else (text or state.request)
                    plan_signature = _plan_signature(
                        plan_request, _snapshot_for_plan, _data_basis(mcp_data, rag_data))
                    reuse_plan = bool(
                        (restarted_from_failure or periodic_task)
                        and state.autonomous and not confirmed
                        and state.steps
                        and _plan_is_fresh(dialog_now, plan_signature))
                    if confirmed and state.steps:
                        task_state.plan_ready(state, state.steps, "план подтверждён пользователем")
                        yield encode({"type": "debug", "text": (
                            f"{_MACHINE}: план подтверждён — этап planning → execution, "
                            f"{state.step_label()}."
                        )})
                    elif reuse_plan:
                        task_state.plan_ready(
                            state, state.steps,
                            "запрос не изменился — беру прежний план без нового вызова LLM")
                        yield encode({"type": "debug", "text": (
                            f"{_MACHINE}: план по этому запросу уже построен — беру прежний "
                            f"({len(state.steps)} {task_state.steps_word(len(state.steps))}), "
                            "вызов модели не нужен."
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
                            # Данные внешних инструментов MCP: планировщик должен
                            # знать, что погода/курс уже получены, — иначе он
                            # поставит в план шаг «узнать погоду» вместо работы по
                            # фактическим данным.
                            mcp=mcp_data,
                            # Фрагменты баз знаний — по той же причине: шаг «найти
                            # требования в регламенте» не нужен, если они уже
                            # найдены, а шаг «сделать по регламенту» без них был бы
                            # выдумкой.
                            rag=rag_data,
                        )
                        # Замер вызова плана — ДЕЛЬТА (агент обнуляет счётчик в
                        # начале вызова), поэтому повторное перепланирование не
                        # удваивает уже учтённые токены.
                        plan_call_usage = dict(planner.last_usage or {})
                        plan_usage = dict(plan_call_usage)

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
                                mcp=mcp_data,
                                rag=rag_data,
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
                        # Общий расход запроса: уже накопленный (разбор запроса)
                        # + план + проверка плана. Без этой строки расход разбора
                        # инвариантов терялся: дальше ветки отдают plan_usage.
                        usage = merge_usage(usage, plan_usage)
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
                                dict(usage, kind="plan"))
                            yield encode(_state_event(session_now, state))
                            yield encode({"type": "done", "usage": dict(usage),
                                          "state": _state_snapshot(session_now, state)})
                            return
                        steps = gate["steps"]
                        # ФОРМА ПЛАНА: шаги-«оформление» («извлечь данные» →
                        # «сгруппировать» → «сформировать таблицу» → «вывести
                        # таблицу») новогo результата не дают, а каждый — отдельный
                        # вызов LLM. Правило в промпте планировщика — просьба,
                        # поэтому хвостовые такие шаги схлопывает КОД
                        # (task_state.compact_steps); действия с последствиями
                        # («отправить», «создать заявку», «в файл») он не трогает.
                        # Правку плана пользователем (✎ «План») не трогаем — там
                        # решение человека.
                        steps, dropped = task_state.compact_steps(steps, plan_request)
                        if dropped:
                            yield encode({"type": "debug", "text": (
                                f"{_MACHINE}: план сокращён на {dropped} "
                                f"{task_state.steps_word(dropped)} — это было оформление "
                                "уже полученного результата, а не отдельная работа "
                                "(каждый шаг стоил отдельного вызова модели)."
                            )})
                        # ПОИСК ПО БАЗЕ ЗНАНИЙ УЖЕ СДЕЛАН (см. _preflight_rag): шаг
                        # «найти что-то в базе знаний» ищет уже найденное. Правило в
                        # промпте планировщика и в блоке фрагментов — просьба,
                        # поэтому такой шаг убирает КОД (task_state.drop_kb_steps);
                        # применяется только когда базы включены и поиск выполнен —
                        # без поиска «посмотреть в базе» законная работа. Шаги с
                        # последствиями и другой работой в том же шаге не трогаются.
                        if rag_data:
                            steps, dropped_kb = task_state.drop_kb_steps(steps,
                                                                        plan_request)
                            if dropped_kb:
                                yield encode({"type": "debug", "text": (
                                    f"{_MACHINE}: из плана убран {dropped_kb} "
                                    f"{task_state.steps_word(dropped_kb)} поиска в базе "
                                    "знаний — фрагменты по этому запросу уже найдены и "
                                    "переданы модели (поиск идёт ДО планирования)."
                                )})
                        state.steps = steps
                        # Запоминаем, для какого запроса и правил план построен:
                        # ровно за такой же план больше не платим (см. reuse_plan).
                        # Подпись — по ТОМУ ЖЕ запросу, для которого план строился
                        # (plan_request), иначе она записала бы прежний запрос: при
                        # новом запросе это выглядело бы как «план уже есть» и
                        # вернуло бы шаги чужого запроса.
                        dialog_now["plan_signature"] = dict(_plan_signature(
                            plan_request,
                            _invariants_snapshot(task_now, session_now),
                            _data_basis(mcp_data, rag_data)))
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
                        plan_text = _plan_message(state, autonomous=state.autonomous,
                                                  delivered=mcp_delivered)
                        # Запрос пользователя и показанный план — в память диалога
                        # сессии: иначе после переключения сессии/перезагрузки их
                        # не было бы видно в окне чата (память хранила только
                        # обмены «шаг → ответ»). Текст запроса нужен и модели как
                        # контекст, поэтому он идёт обычной репликой пользователя.
                        if text and not machine_step:
                            dialog_now.setdefault("messages", []).append(
                                {"role": "user", "content": text,
                                 # Автозапуск — не реплика пользователя: пометка
                                 # переживает запись в файл (см.
                                 # _restore_message_sources) и в старом диалоге
                                 # без журнала такая реплика рисуется служебной.
                                 **({"source": periodic_store.SOURCE_AUTO}
                                    if periodic_run else {})})
                        dialog_now.setdefault("messages", []).append(
                            {"role": "assistant", "content": plan_text})
                        if state.autonomous or mcp_delivered:
                            # РАБОТА УЖЕ ВЫПОЛНЕНА ИНСТРУМЕНТАМИ (цепочка MCP
                            # собрала данные, сохранила набор и выгрузила файл):
                            # подтверждение плана спрашивать не о чем — пользователь
                            # не может «поправить» уже сделанное, а задача висела бы
                            # в ожидании при готовом результате (живой случай: файл
                            # в чате есть, а полоса этапов просит «ок»). План
                            # подтверждаем сами и идём выполнять ответ.
                            reason = ("режим «работай автономно» — подтверждение "
                                      "плана не требуется" if state.autonomous else
                                      "результат уже получен внешними инструментами "
                                      "(файл приложен) — подтверждение плана не требуется")
                            if mcp_delivered and not state.autonomous:
                                yield encode({"type": "debug", "text": (
                                    f"{_MACHINE}: результат по запросу уже получен "
                                    "внешними инструментами (вложение/сохранённый "
                                    "набор) — план подтверждаю сам, задача не ждёт "
                                    "пользователя."
                                )})
                            task_state.plan_ready(state, steps, reason)
                            yield encode({"type": "bot", "text": plan_text})
                        else:
                            task_state.await_confirmation(
                                state, steps, "план показан пользователю — жду «ок» или правок")
                            yield encode({"type": "bot", "text": plan_text})
                            yield encode(_state_event(session_now, state))
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
                                yield encode(_state_event(session_now, state))
                            if usage:
                                # Расход этого запроса — отдельной записью
                                # (kind="plan"): ответа пользователю не было, но
                                # токены потрачены (включая разбор запроса).
                                dialog_now.setdefault("usage", []).append(
                                    dict(usage, kind="plan"))
                            workspace_store.set_dialog_state(session_now, state)
                            await _persist()
                            yield encode({"type": "done", "usage": dict(usage),
                                          "state": _state_snapshot(session_now, state)})
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
                        # Данные внешних инструментов MCP по этому запросу: без них
                        # ответ шага не знал бы о том, что агент уже получил.
                        mcp=mcp_data,
                        # Фрагменты баз знаний по этому запросу: ответ строится по
                        # документам пользователя и ссылается на них.
                        rag=rag_data,
                    ):
                        # Событие "done" несёт расход токенов текущего запроса.
                        kind = event.get("type")
                        if kind == "done" and isinstance(event.get("usage"), dict):
                            # Расход ответа шага добавляется к уже накопленному
                            # (разбор запроса + план + проверка плана): замер
                            # агента — дельта одного вызова.
                            usage = merge_usage(usage, event["usage"])
                        elif kind == "bot" and str(event.get("text") or "").strip():
                            answered = True
                            # Запасной ответ (модель не ответила) — шаг НЕ
                            # выполнен: execution → failed (см. ниже).
                            if event.get("fallback"):
                                fallback_answer = True
                            # ВЫДУМАННЫЕ ДАННЫЕ: данных нет вовсе, а в ответе
                            # таблица с числами — показываем отказ инструмента.
                            replacement = _mcp_fabrication_guard(
                                str(event.get("text") or ""), mcp_data)
                            if replacement:
                                event = dict(event, text=replacement)
                                yield encode({"type": "debug", "text": (
                                    f"{_MACHINE}: ответ содержал таблицу, но внешние "
                                    "данные не получены — показываю причину отказа "
                                    "вместо выдуманных значений."
                                )})
                        elif kind == "error":
                            note = str(event.get("text") or "")
                            # Предупреждение о лимите токенов — не провал шага.
                            if _LIMIT_WARNING_MARK not in note:
                                errors.append(note)
                        if kind == "bot" and not last_step \
                                and str(event.get("text") or "").strip():
                            # ПРОМЕЖУТОЧНЫЙ ШАГ: его ответ не показываем репликой
                            # в чате — работа идёт по плану, а ход виден в журнале
                            # (debug). Сам текст остаётся в ПАМЯТИ диалога: он нужен
                            # следующему шагу и проверке результата (см. `memory`
                            # ниже), поэтому «прячем» только показ, а не данные.
                            yield encode({"type": "debug", "text": (
                                f"{_MACHINE}: {state.step_label()} выполнен — ответ "
                                "ушёл в следующий шаг и в проверку (в чате не "
                                "показываю: смотрите журнал работы выше)."
                            )})
                            continue
                        if kind == "bot" and last_step and rag_sources \
                                and not event.get("fallback"):
                            # ИСТОЧНИКИ — К САМОМУ ОТВЕТУ, а не отдельным узлом:
                            # карточки «файл · раздел · близость» рисуются под
                            # финальным ответом, попадают в журнал вместе с ним и
                            # восстанавливаются при переключении задачи. Только у
                            # ПОКАЗЫВАЕМОГО ответа (последний шаг): у промежуточных
                            # ответов карточки были бы мусором в чате, а сами
                            # фрагменты и так видны в контексте каждого шага.
                            # Запасной ответ (модель не ответила, `fallback`) карточек
                            # не получает: фрагментами он не пользовался, и список
                            # источников под отказом читался бы как «ответ по
                            # документам».
                            event = dict(event, sources=rag_sources)
                        yield encode(event)
                        # ФИНАЛЬНЫЙ ОТЧЁТ + ТАБЛИЦА: карточка файла идёт ПОСЛЕ
                        # ответа последнего шага, чтобы пользователь видел сначала
                        # готовый отчёт, а под ним — файл.
                        if kind == "bot" and mcp_files_event and not mcp_files_shown:
                            yield encode(mcp_files_event)
                            mcp_files_shown = True
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
                    elif stored and periodic_run:
                        # Реплика отправлена АВТОЗАПУСКОМ периодической задачи (план
                        # был взят готовым, и в ветку планирования запрос не
                        # попадал): в памяти диалога пометка нужна так же, как в
                        # журнале чата — иначе в старом диалоге без журнала повтор
                        # выглядел бы как сообщение пользователя.
                        exchange[-2]["source"] = periodic_store.SOURCE_AUTO
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
                    elif stopped == PENDING_PAUSE and not periodic_task:
                        # Последний шаг выполнен, НО пользователь просил паузу:
                        # проверку не запускаем — переводим задачу на этап
                        # validation и останавливаемся. Проверка выполнится после
                        # «Продолжить» (см. ветку validation ниже).
                        task_state.to_validation(
                            state,
                            "шаг выполнен, но пользователь нажал «Пауза» — проверку отложил",
                        )
                        validation_deferred_here = True
                        yield encode(_state_event(session_now, state))
                        yield encode({"type": "debug", "text": (
                            f"{_MACHINE}: «Пауза» на последнем шаге — проверку результата "
                            "отложил: задача остановится НА ЭТАПЕ validation, проверка "
                            "выполнится после «Продолжить»."
                        )})
                    elif periodic_task:
                        # ПЕРИОДИЧЕСКАЯ задача: план пройден до конца — ЦИКЛ
                        # завершён. Итоговой проверки нет (сверять «соответствие
                        # задаче» у повторяющегося ответа не с чем, а «готово» у
                        # такой задачи не бывает), план СОХРАНЯЕТСЯ, место — снова
                        # первый шаг: следующий повтор выполнит план с начала, а
                        # пока задача ждёт расписания (см. task_state.cycle_done).
                        _cycle_reason = (
                            "повтор завершён («Пауза» на последнем шаге) — жду "
                            "«Продолжить» и следующего повтора"
                            if stopped == PENDING_PAUSE else
                            "повтор завершён: все шаги плана выполнены — жду "
                            "следующего повтора по расписанию"
                        )
                        task_state.cycle_done(state, _cycle_reason)
                        yield encode({"type": "debug", "text": (
                            f"{_MACHINE}: план пройден — повтор завершён "
                            f"({state.steps_total} "
                            f"{task_state.steps_word(state.steps_total)}). План сохранён, "
                            "задача остаётся периодической: следующий повтор выполнит "
                            "его с первого шага. Проверки результата и статуса «готово» "
                            "у периодической задачи нет."
                        )})
                    else:
                        # Последний шаг выполнен — проверяем результат (общий
                        # помощник: тот же путь используется для отложенной
                        # проверки после «Продолжить»).
                        async for chunk in run_validation(
                            answered and not fallback_answer, errors, stored, resumed=False
                        ):
                            yield chunk

                if state.stage == "validation" and not validation_deferred_here \
                        and periodic_task:
                    # Периодическая задача: этап validation ей не положен (проверки
                    # нет). Такое состояние остаётся от прежней версии или от
                    # отложенной проверки: закрываем цикл, а не гоняем приёмщика.
                    task_state.cycle_done(
                        state,
                        "повтор завершён: проверка результата у периодической задачи "
                        "не выполняется — жду следующего повтора",
                    )
                    yield encode({"type": "debug", "text": (
                        f"{_MACHINE}: у периодической задачи этапа «проверка» нет — "
                        "считаю повтор завершённым, план сохранён."
                    )})
                    yield encode(_state_event(session_now, state))

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
                    yield encode(_state_event(session_now, state))

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
                yield encode(_state_event(session_now, state))

                # Карточка файла, если финального ответа в этом запросе не было
                # (шаг не выполнен, пауза, отмена): файл всё равно получен и его
                # надо показать — но ВСЕГДА после ответа, а не до него.
                if mcp_files_event and not mcp_files_shown:
                    yield encode(mcp_files_event)
                    mcp_files_shown = True

                # Финальный замер запроса одним событием "done": в этой ветке
                # (обычный ответ шага) раньше событие не отправлялось вовсе, и
                # панель токенов показывала только внутренний замер агента — без
                # разбора запроса, плана и проверки плана. Теперь расход приходит
                # ОДИН раз и полностью, как он же записан в диалоге.
                if usage:
                    yield encode({"type": "done", "usage": dict(usage),
                                  "state": _state_snapshot(session_now, state)})

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
                    yield encode(_state_event(session_now, state))
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
                              "state": _state_snapshot(session_now, state)})
        finally:
            # Пометку «задача выполняется» снимаем ВСЕГДА: иначе сбой в потоке
            # оставил бы её висеть и «Пауза» вечно откладывалась бы.
            if session_now is not None:
                _running_sessions.discard(str(session_now["id"]))

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@router.get("/agent/files/{file_id}")
async def agent_file(file_id: str) -> FileResponse:
    """СКАЧИВАНИЕ файла, который вернул MCP-инструмент (карточка файла в чате).

    Файлы инструментов сохраняются на диск при получении (см.
    app/ai/attachments.py), а в диалоге и журнале остаётся ссылка на этот
    маршрут. Каталог вложений — единственное место, откуда отдаются файлы: id
    проверяется шаблоном (никаких путей и `..`), поэтому скачать что-то другое
    через этот маршрут нельзя.
    """
    path = attach_store.resolve(file_id)
    if not path:
        raise HTTPException(status_code=404, detail="Файл не найден")
    return FileResponse(path, filename=os.path.basename(path),
                        media_type="application/octet-stream")


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
    log = [dict(item) for item in dialog.get("log", [])]
    # Журнал чата и память диалога ДУБЛИРУЮТ друг друга: интерфейс рисует окно
    # по журналу, а messages берёт только когда журнала ещё нет (старые
    # диалоги). Отдавать оба — лишняя четверть ответа на каждое переключение.
    messages = [] if log else [dict(m) for m in dialog["messages"]]
    return {
            "messages": messages,
            "log": log,
            "usage": [dict(item) for item in dialog["usage"]],
            "summary": list(dialog["summary"]),
            "facts": dict(dialog["facts"]),
            "branches": {bid: dict(branch) for bid, branch in dialog["branches"].items()},
            "active_branch": dialog["active_branch"],
            "session": _session_payload(session),
            "state": _state_snapshot(session, workspace_store.dialog_state(session)),
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
