"""Маршруты чата — приём сообщений и возврат ответа бота.

Классический POST /api/chat (обычный/экспертный/температура/тест моделей) и
потоковый POST /api/agent/chat для режима «AI-агент»: сервер отдаёт NDJSON,
каждая строка — событие агента (debug/bot/error), которое фронтенд выводит
в чат отдельным сообщением по мере появления.

Диалог режима «AI-агент» живёт в «рабочем пространстве» (workspace): задачи →
сессии (диалоги) → состояние диалога (память сообщений, замеры токенов, резюме
стратегии «summary» с границей covered, блок фактов «sticky facts», ветви плана
и активная ветка «branching»). Диалог нельзя начать, пока не создана задача; у
каждой задачи своя история диалогов, а внутри задачи можно переключаться между
сессиями — у каждой своё независимое состояние. Всё это переживает перезапуск
приложения: workspace загружается из JSON-файла при старте процесса и
сохраняется после каждого изменения (см. app/ai/workspace.py).
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from app.ai import service
from app.ai import workspace as workspace_store
from app.ai.agent import Agent, AgentConfig, DEFAULT_SUMMARY_SIZE, DEFAULT_WINDOW_SIZE
from app.schemas import ChatMessage, MemoryEntryCreate, NameUpdate, TaskCreate

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# Рабочее пространство режима «AI-агент»: задачи с их сессиями-диалогами.
# При старте процесса восстанавливается из JSON-файла (data/agent_workspace.json),
# поэтому после перезапуска задачи, диалоги, панель токенов и ветви плана
# остаются на месте. Изменения сериализуются блокировкой: агент работает
# асинхронно, а workspace у нас один на приложение.
_workspace: Dict[str, Any] = workspace_store.load_workspace()
_agent_lock = asyncio.Lock()


def _current_task() -> Optional[Dict[str, Any]]:
    """Текущая задача (None — пользователь ещё не создал ни одной)."""
    return workspace_store.active_task(_workspace)


def _current_session() -> Optional[Dict[str, Any]]:
    """Текущая сессия-диалог текущей задачи (None — диалогов ещё нет)."""
    return workspace_store.active_session(_workspace)


def _find_session_anywhere(session_id: str) -> Tuple[Optional[Dict[str, Any]],
                                                     Optional[Dict[str, Any]]]:
    """Ищет сессию по id во всех задачах: (задача, сессия)."""
    for task in _workspace.get("tasks", []):
        session = workspace_store.find_session(task, session_id)
        if session is not None:
            return task, session
    return None, None


async def _persist() -> None:
    """Сохраняет workspace в файл (сбой записи не рвёт диалог)."""
    try:
        await asyncio.to_thread(workspace_store.save_workspace, _workspace)
    except Exception:  # noqa: BLE001 — сбой записи логируем, работу продолжаем
        logger.warning("Не удалось сохранить workspace AI-агента", exc_info=True)


def _snapshot() -> dict:
    """Снимок workspace для фронтенда: задачи, текущая задача, её диалоги."""
    return workspace_store.snapshot(_workspace)


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
    while len(usage) > requests:
        usage.pop(0)


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

    Рабочая память привязана к ТЕКУЩЕЙ ЗАДАЧЕ (её нет — добавлять некуда,
    вернётся None), долговременная — к workspace целиком: она глобальная.
    """
    normalized = str(layer or "").strip().lower()
    if normalized in ("long", "long_term", "long-term", "долговременная"):
        return _workspace, workspace_store.MEMORY_LONG_TERM
    if normalized in ("work", "working", "рабочая"):
        return _current_task(), workspace_store.MEMORY_WORKING
    raise HTTPException(status_code=400, detail="Неизвестный слой памяти")


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
    async with _agent_lock:
        return _snapshot()


@router.post("/agent/tasks")
async def task_create(payload: TaskCreate) -> dict:
    """Создаёт задачу (кнопка «Новая задача») и делает её текущей.

    Задача создаётся без диалогов: первый запрос пользователя заведёт сессию
    сам. Пока задачи нет, диалог в режиме агента начать нельзя.
    """
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Введите название задачи")
    async with _agent_lock:
        workspace_store.create_task(_workspace, name)
        await _persist()
        return _snapshot()


@router.put("/agent/tasks/{task_id}")
async def task_rename(task_id: str, payload: NameUpdate) -> dict:
    """Переименовывает задачу (карандаш рядом со списком задач)."""
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Введите название задачи")
    async with _agent_lock:
        task = workspace_store.find_task(_workspace, task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Задача не найдена")
        task["name"] = name[:120]
        await _persist()
        return _snapshot()


@router.delete("/agent/tasks/{task_id}")
async def task_delete(task_id: str) -> dict:
    """Удаляет задачу вместе со всеми её диалогами (корзина у списка задач)."""
    async with _agent_lock:
        if not workspace_store.delete_task(_workspace, task_id):
            raise HTTPException(status_code=404, detail="Задача не найдена")
        await _persist()
        return _snapshot()


@router.post("/agent/tasks/{task_id}/select")
async def task_select(task_id: str) -> dict:
    """Переключает текущую задачу (выпадающий список «Текущая задача»).

    У каждой задачи своя история диалогов: фронт после переключения заново
    запрашивает диалог текущей сессии (GET /api/agent/history).
    """
    async with _agent_lock:
        task = workspace_store.find_task(_workspace, task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Задача не найдена")
        _workspace["active_task"] = task["id"]
        await _persist()
        return _snapshot()


@router.post("/agent/sessions")
async def session_create() -> dict:
    """Создаёт в текущей задаче новый пустой диалог (кнопка «Новая сессия»).

    Сессия сразу становится текущей — её диалог пуст, а заголовок в истории
    появится по первому запросу пользователя.
    """
    async with _agent_lock:
        task = _current_task()
        if task is None:
            raise HTTPException(status_code=400, detail="Сначала создайте задачу")
        workspace_store.create_session(task)
        await _persist()
        return _snapshot()


@router.put("/agent/sessions/{session_id}")
async def session_rename(session_id: str, payload: NameUpdate) -> dict:
    """Переименовывает диалог (карандаш в элементе истории сессий)."""
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Введите название диалога")
    async with _agent_lock:
        _, session = _find_session_anywhere(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Диалог не найден")
        session["title"] = name[:200]
        await _persist()
        return _snapshot()


@router.delete("/agent/sessions/{session_id}")
async def session_delete(session_id: str) -> dict:
    """Удаляет диалог (корзина в элементе истории сессий).

    Если удалён текущий диалог, текущим становится соседний — или ни одного,
    и тогда следующий запрос заведёт новый.
    """
    async with _agent_lock:
        task, session = _find_session_anywhere(session_id)
        if task is None or session is None:
            raise HTTPException(status_code=404, detail="Диалог не найден")
        workspace_store.delete_session(task, session_id)
        await _persist()
        return _snapshot()


@router.post("/agent/sessions/{session_id}/select")
async def session_select(session_id: str) -> dict:
    """Переключает текущий диалог внутри его задачи (клик по элементу истории)."""
    async with _agent_lock:
        task, session = _find_session_anywhere(session_id)
        if task is None or session is None:
            raise HTTPException(status_code=404, detail="Диалог не найден")
        _workspace["active_task"] = task["id"]
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
    # окончания ответа агента (он держит `_agent_lock` всё время стрима).
    return workspace_store.memory_snapshot(_workspace)


@router.post("/agent/memory")
async def memory_add(payload: MemoryEntryCreate) -> dict:
    """Добавляет запись в слой памяти (кнопки под сообщениями в чате)."""
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Нечего добавлять в память")
    async with _agent_lock:
        container, key = _memory_target(payload.layer)
        if container is None:
            raise HTTPException(
                status_code=400,
                detail="Сначала создайте задачу — рабочая память привязана к задаче",
            )
        workspace_store.add_memory(container, key, text, payload.source)
        await _persist()
        return workspace_store.memory_snapshot(_workspace)


@router.delete("/agent/memory/{layer}/{entry_id}")
async def memory_delete(layer: str, entry_id: str) -> dict:
    """Удаляет запись из слоя памяти (корзина в панели «Состояние памяти»)."""
    async with _agent_lock:
        container, key = _memory_target(layer)
        if container is None or not workspace_store.delete_memory(container, key, entry_id):
            raise HTTPException(status_code=404, detail="Запись не найдена")
        await _persist()
        return workspace_store.memory_snapshot(_workspace)


@router.post("/agent/chat")
async def agent_chat(msg: ChatMessage) -> StreamingResponse:
    """Режим «AI-агент»: потоковый ответ NDJSON.

    Каждая строка ответа — JSON-событие агента:
      {"type": "debug", "text": "чем агент сейчас занят"} — отдельное
        сообщение в чате (выводится по мере появления);
      {"type": "branches", "analysis": {...}, "active": "A"} — план ветвления
        (стратегия «branching»): из него фронтенд рисует блок плана и список
        веток для переключения;
      {"type": "bot", "text": "..."} — финальный ответ;
      {"type": "error", "text": "..."} — понятное сообщение об ошибке;
      {"type": "done"} — служебный конец потока.

    Запрос обрабатывается в диалоге ТЕКУЩЕЙ сессии текущей задачи: её память и
    состояние стратегий уходят агенту на вход и обновляются по завершении
    обмена. Без созданной задачи диалог начать нельзя — 400 (фронт показывает
    подсказку «сначала создайте задачу»). Если у задачи ещё нет ни одного
    диалога, сессия создаётся автоматически под первый запрос.

    Агент берёт из сообщения длину (max_tokens) и условие завершения (stop),
    стратегию работы с контекстом (agent_strategy) с её полями (summary/window)
    и выбранную ветку плана (branch).
    """
    # Формат ответа агенту не передаём: системных промптов у него нет,
    # запрос уходит в модель как есть (см. app/ai/agent.py).
    task = _current_task()
    if task is None:
        return JSONResponse({"detail": "Сначала создайте задачу"}, status_code=400)
    session = workspace_store.active_session(_workspace, task)
    if session is None:
        session = workspace_store.create_session(task)
    dialog: Dict[str, Any] = session["dialog"]
    # Слои памяти пользователя (memory layers): рабочая память задачи и
    # долговременная (глобальная) база знаний. Это снимки — агент получает их
    # на каждый запрос и кладёт в контекст ВСЕГДА, независимо от стратегии.
    working_memory = workspace_store.memory_texts(task, workspace_store.MEMORY_WORKING)
    long_term_memory = workspace_store.memory_texts(_workspace, workspace_store.MEMORY_LONG_TERM)

    agent = Agent(AgentConfig(
        max_tokens=msg.max_tokens,
        stop=msg.stop,
        strategy=msg.agent_strategy,
        summary_size=msg.summary or DEFAULT_SUMMARY_SIZE,
        window_size=msg.window or DEFAULT_WINDOW_SIZE,
    ))

    def encode(event: dict) -> str:
        return json.dumps(event, ensure_ascii=False) + "\n"

    async def event_stream():
        try:
            async with _agent_lock:
                usage: Dict[str, Any] = {}
                async for event in agent.stream_generate(
                    msg.content,
                    history=dialog["messages"],
                    summary=dialog["summary"],
                    covered=dialog["covered"],
                    facts=dialog["facts"],
                    branches=dialog["branches"],
                    branch=msg.branch,
                    working_memory=working_memory,
                    long_term_memory=long_term_memory,
                ):
                    # Событие "done" несёт расход токенов текущего запроса.
                    if event.get("type") == "done" and isinstance(event.get("usage"), dict):
                        usage = event["usage"]
                    yield encode(event)
                # Обработка завершена — запоминаем обмен репликами и состояние
                # стратегии (резюме и его границу, факты, ветви плана, активную
                # ветку) В ДИАЛОГЕ ЭТОЙ СЕССИИ и сохраняем workspace в файл,
                # чтобы всё это пережило перезапуск приложения.
                memory = list(agent.memory)
                stored = _exchange_stored(_exchange_memory(agent, msg.branch, memory), msg.content)
                dialog["messages"] = memory
                dialog["summary"] = list(agent.summary)
                dialog["covered"] = agent.covered
                dialog["facts"] = dict(agent.facts)
                dialog["branches"] = {bid: dict(branch) for bid, branch in agent.branches.items()}
                dialog["active_branch"] = agent.active_branch
                if usage and stored:
                    dialog["usage"].append(dict(usage))
                    _usage_matches_history(dialog)
                await _persist()
        except asyncio.CancelledError:
            raise  # клиент отключился — просто останавливаем поток
        except Exception:  # noqa: BLE001
            logger.exception("Режим AI-агента: ошибка обработки запроса")
            yield encode({"type": "error", "text": "Внутренняя ошибка агента. Попробуйте ещё раз."})

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
    "session": {"id", "title"}|None} —
    сообщения агент использует как память диалога, usage — расход токенов по
    запросам пользователя (панель «Токены диалога»), summary — резюме прежней
    переписки (стратегия «summary»), facts — блок фактов (стратегия «sticky
    facts»), branches/active_branch — ветви плана и активная ветка (стратегия
    «branching»; у каждой ветки своя история сообщений). Фронтенд вызывает этот
    маршрут при включении режима агента и при каждом переключении задачи или
    сессии, чтобы нарисовать реплики, панель токенов и список веток — диалог
    выглядит так, будто агент не выключался. Задачи нет / диалогов нет — пустой
    диалог.
    """
    async with _agent_lock:
        session = _current_session()
        dialog = session["dialog"] if session else workspace_store.empty_dialog()
        return {
            "messages": list(dialog["messages"]),
            "usage": [dict(item) for item in dialog["usage"]],
            "summary": list(dialog["summary"]),
            "facts": dict(dialog["facts"]),
            "branches": {bid: dict(branch) for bid, branch in dialog["branches"].items()},
            "active_branch": dialog["active_branch"],
            "session": ({
                "id": session["id"],
                "title": workspace_store.session_title(session),
            } if session else None),
        }


@router.delete("/agent/history")
async def agent_history_clear() -> dict:
    """Очищает диалог ТЕКУЩЕЙ сессии режима «AI-агент».

    Стирает память диалога, замеры токенов, резюме (стратегия «summary») с его
    границей, блок фактов (стратегия «sticky facts») и ветви плана со всеми
    диалогами внутри них (стратегия «branching») — сама сессия (и её название)
    остаётся в истории задач. Вызывается по кнопке «Очистить историю» рядом с
    «Отправить» в режиме агента. Диалогов нет — ничего не меняется.
    """
    async with _agent_lock:
        session = _current_session()
        if session is not None:
            session["dialog"] = workspace_store.empty_dialog()
            await _persist()
    return {"ok": True}
