"""Планировщик периодических задач режима «AI-агент».

Периодическая задача — обычная задача-диалог (сессия workspace) с расписанием
(поле `session["periodic"]`, см. app/ai/periodic.py). Этот модуль — её мотор:
раз в `config.PERIODIC_TICK_SECONDS` он смотрит расписания всех задач и у тех, у
которых наступил срок, запускает ПОВТОР.

Повтор — это ТОТ ЖЕ путь, что и обычный запрос агента: планировщик вызывает
маршрут `POST /api/agent/chat` напрямую (как это делал бы интерфейс) и доводит
прогон до конца. Никакого своего конвейера у повтора нет — поэтому в нём
работают и конечный автомат задачи, и правила проекта, и внешние инструменты
MCP, и проверка результата, а всё увиденное ложится в ЖУРНАЛ ЧАТА задачи: придя в
неё, пользователь видит, что задача повторялась и что она ответила.

Что повтор делает иначе, чем обычный запрос (см. ChatMessage.periodic):

* идёт АВТОНОМНО — подтверждать план некому, пользователя за клавиатурой нет;
* берёт СВЕЖИЕ данные MCP — запрос у повтора тот же, а данные за прошедший
  период уже изменились (иначе в чат попадали бы числа прошлого повтора);
* помечает свою реплику в журнале как автозапуск (`⏱ Автозапуск (раз в час): …`).

Шаги плана выполняются по одному на запрос (так же, как их ведёт интерфейс),
поэтому повтор — это цепочка запросов: план и первый шаг, затем остальные шаги и
проверка результата. Предел цепочки — `config.PERIODIC_MAX_TURNS`.

Запуск: `start()` при старте приложения (main.py) и `stop()` при остановке.
Планировщик живёт в том же процессе и цикле событий, что и веб-приложение:
расписания берутся из уже загруженного workspace, а не читаются с диска заново.
"""

import asyncio
import json
import logging
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

from fastapi.responses import StreamingResponse

from app import config
from app.ai import periodic as periodic_store
from app.ai import workspace as workspace_store
from app.routers import chat
from app.schemas import ChatMessage

logger = logging.getLogger(__name__)

# Задача планировщика (одна на процесс). None — планировщик не запущен.
_task: Optional[asyncio.Task] = None


def start() -> Optional[asyncio.Task]:
    """Запускает планировщик (вызывается при старте приложения).

    Планировщик выключен настройкой (`PERIODIC_ENABLED=0`) — повторов не будет,
    и об этом пишется в лог: расписания задач остаются в файле, но никто их не
    исполняет (полезно для отладки и тестов).
    """
    global _task
    if not config.PERIODIC_ENABLED:
        logger.info("Планировщик периодических задач выключен (PERIODIC_ENABLED=0)")
        return None
    if _task is None or _task.done():
        _task = asyncio.get_running_loop().create_task(_loop(), name="periodic-runner")
        logger.info(
            "Планировщик периодических задач запущен: проверка расписаний каждые %d с",
            config.PERIODIC_TICK_SECONDS)
    return _task


async def stop() -> None:
    """Останавливает планировщик (при остановке приложения).

    Незавершённый повтор прерывается: он продолжит работу при следующем запуске
    приложения (срок повтора уже сдвинут, см. periodic.started), а состояние
    задачи сохраняется после каждого запроса цепочки.
    """
    global _task
    task, _task = _task, None
    if task is None:
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    logger.info("Планировщик периодических задач остановлен")


async def _loop() -> None:
    """Цикл планировщика: тик — пауза — тик..."""
    if config.PERIODIC_START_DELAY:
        await asyncio.sleep(config.PERIODIC_START_DELAY)
    while True:
        try:
            await tick()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — сбой тика не должен убивать цикл
            logger.exception("Планировщик периодических задач: сбой проверки расписаний")
        await asyncio.sleep(config.PERIODIC_TICK_SECONDS)


async def tick(moment: Optional[Any] = None) -> List[str]:
    """Одна проверка расписаний: запускает повторы, у которых наступил срок.

    Возвращает id задач, повтор которых запущен (для проверок и лога). Повтор
    идёт ОТДЕЛЬНОЙ задачей: длинная цепочка шагов не задерживает следующие тики
    и не мешает другим периодическим задачам. Задача, у которой уже идёт прогон
    (свой шаг пользователя или прежний повтор), пропускается — она не брошена,
    срок остался в прошлом и будет взят на следующем тике.
    """
    moment = moment or periodic_store.now()
    started: List[str] = []
    for task, session in workspace_store.due_periodic_sessions(chat._workspace, moment):
        session_id = str(session.get("id"))
        if session_id in chat._periodic_running or session_id in chat._running_sessions:
            continue
        state = workspace_store.dialog_state(session)
        if state.stage == "cancelled":
            # Задача ОТМЕНЕНА пользователем — это и есть остановка периодической
            # задачи: повторять отменённое нельзя. Автозапуск выключаем один раз,
            # чтобы он не «воскрешал» задачу каждым тиком (включить снова — 🔁).
            meta = workspace_store.periodic_meta(session)
            if meta.get("enabled"):
                workspace_store.set_periodic(session, periodic_store.set_enabled(meta, False))
                dialog = session.get("dialog")
                if isinstance(dialog, dict):
                    workspace_store.add_log(dialog, workspace_store.LOG_DEBUG, (
                        f"{periodic_store.AUTO_MARK} Задача отменена — повторять её "
                        "нельзя, автозапуск остановлен. Чтобы вернуть задачу к "
                        "работе, отправьте в неё новое сообщение (план будет "
                        "построен заново), а повтор включите кнопкой 🔁."
                    ))
                await chat._persist()
                logger.info("Периодическая задача %s отменена — автозапуск выключен",
                            session_id)
            # ВНЕШНИЕ СБОРЫ задачи (наблюдения MCP) останавливаются и здесь: задача
            # остановлена — работа на сервере ей больше не нужна. Кнопка «Отменить»
            # делает то же самое сразу (см. state_cancel), а этот путь страхует
            # задачи, отменённые до появления уборки.
            await chat._stop_mcp_started(session, "периодическая задача отменена")
            await chat._persist()
            continue
        if state.paused:
            # «Пауза» — команда пользователя: пока её не сняли, повтор не идём.
            logger.info("Периодическая задача %s на паузе — повтор отложен", session_id)
            continue
        chat._periodic_running.add(session_id)
        runner = asyncio.get_running_loop().create_task(
            _run(task, session), name=f"periodic-{session_id}")
        runner.add_done_callback(partial(_settle, session_id=session_id))
        started.append(session_id)
    return started


def _settle(runner: asyncio.Task, session_id: str) -> None:
    """Снимает признак «идёт повтор» и забирает исключение задачи (если было)."""
    chat._periodic_running.discard(session_id)
    if runner.cancelled():
        return
    error = runner.exception()
    if error is not None:
        logger.error("Периодическая задача %s: сбой повтора: %s", session_id, error,
                     exc_info=error)


async def _run(task: Dict[str, Any], session: Dict[str, Any]) -> None:
    """Один ПОВТОР периодической задачи: цепочка запросов до конца + учёт.

    Срок следующего повтора сдвигается СРАЗУ (см. periodic.started): пока идёт
    прогон, следующий тик не запустит эту же задачу второй раз. По завершении
    расписание получает отметку о повторе (сколько раз, когда, с какой ошибкой) и
    записывается в файл — расписания переживают перезапуск приложения.
    """
    session_id = str(session.get("id"))
    meta = workspace_store.periodic_meta(session)
    started_at = periodic_store.now()
    request_text = str(meta.get("request") or "").strip()
    error = ""
    interrupted = False
    skipped = False
    try:
        periodic_store.started(meta, started_at)
        workspace_store.set_periodic(session, meta)
        await chat._persist()
        if not request_text:
            # Запрос ещё не написан (задачу создали, но сообщение не отправили):
            # повторять нечего — только сдвигаем срок, ничего не выполняя и не
            # засчитывая это за повтор.
            skipped = True
            logger.info("Периодическая задача %s: запроса ещё нет — повтор пропущен",
                        session_id)
            return
        logger.info("Периодическая задача %s: повтор (%s)",
                    session_id, periodic_store.label(meta.get("interval")))
        error, interrupted = await _drive(task, session, request_text)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 — сбой повтора не должен ломать планировщик
        error = f"сбой автозапуска: {exc}"
        logger.exception("Периодическая задача %s: сбой повтора", session_id)
    finally:
        # Пропущенный повтор (запроса ещё нет) в счётчики не идёт — срок уже
        # сдвинут, файл просто сохраняется.
        if not skipped:
            meta = workspace_store.periodic_meta(session)
            dialog = session.get("dialog")
            if interrupted:
                # Повтор прервал ПОЛЬЗОВАТЕЛЬ (нажал «Паузу» во время прогона):
                # это не сбой и не выполненный повтор — в счётчики не идём, а в
                # чате видно, почему ответа не будет до «Продолжить».
                if isinstance(dialog, dict):
                    workspace_store.add_log(dialog, workspace_store.LOG_DEBUG, (
                        f"{periodic_store.AUTO_MARK} Автозапуск прерван: задача на "
                        "паузе. Повтор не засчитан, следующий — по расписанию "
                        "после «Продолжить»."
                    ))
            else:
                periodic_store.finished(meta, started_at, not error, error)
                workspace_store.set_periodic(session, meta)
                if error:
                    # Сбой автозапуска пользователь должен ВИДЕТЬ в самой задаче:
                    # молчащий повтор выглядел бы как «задача перестала работать».
                    if isinstance(dialog, dict):
                        workspace_store.add_log(dialog, workspace_store.LOG_ERROR, (
                            f"{periodic_store.AUTO_MARK} Автозапуск не удался: {error}. "
                            "Следующий повтор — по расписанию (🔁 в списке задач)."
                        ))
        await chat._persist()


def _log_mark(session: Dict[str, Any]) -> tuple:
    """Отпечаток конца журнала чата: по нему видно, сделал ли запрос что-нибудь.

    Именно отпечаток, а не длина: журнал ограничен (_MAX_LOG), и на длинном
    диалоге новые записи вытесняют старые — длина при этом не растёт, а записи
    добавляются. Берём длину и тексты двух последних записей: совпадение всех
    трёх означает, что запрос не добавил в чат ничего.
    """
    dialog = session.get("dialog")
    log = dialog.get("log") if isinstance(dialog, dict) else None
    log = log or []
    tail = tuple(str(item.get("text") or "") for item in log[-2:])
    return (len(log), tail)


def _last_error(events: List[Dict[str, Any]]) -> str:
    """Последнее сообщение об ошибке из потока событий (пусто — ошибок не было)."""
    texts = [str(event.get("text") or "").strip() for event in events
             if event.get("type") == "error" and str(event.get("text") or "").strip()]
    return texts[-1] if texts else ""


async def _drive(task: Dict[str, Any], session: Dict[str, Any],
                 request_text: str) -> Tuple[str, bool]:
    """Ведёт повтор до конца: запрос → шаги плана → проверка результата.

    Возвращает (описание ошибки, прерван ли повтор пользователем). Пустая ошибка
    и False — повтор выполнен. Цепочка идёт В ПРОФИЛЕ ВЛАДЕЛЬЦА задачи (см.
    chat.profile_override): у профиля свои системный блок и долговременная
    память, и повтор задачи неоткрытого профиля обязан выполняться в своём.

    Решение о конце цепочки принимает СОСТОЯНИЕ АВТОМАТА. Повтор идёт по
    СОХРАНЁННОМУ плану: планировщик и проверка результата в нём не участвуют
    (план строится один раз, у периодической задачи нет ни «готово», ни итоговой
    проверки). Признак конца повтора — возврат на ПЕРВЫЙ шаг (`cycle_done`:
    `step_index` снова 0): цикл пройден, план сохранён, задача ждёт следующего
    расписания. Внутри одного цикла `step_index` только растёт.
    «Пауза» (своя или нажатая во время прогона) обрывает цепочку сразу: команда
    пользователя важнее расписания, а повтор считается прерванным, а не сбоем.
    """
    error = ""
    interrupted = False
    with chat.profile_override(workspace_store.task_owner(task)):
        for turn in range(config.PERIODIC_MAX_TURNS):
            state = workspace_store.dialog_state(session)
            if state.paused:
                interrupted = True
                break
            # Первый запрос — первый шаг плана, дальше автомат просит выполнить
            # текущий шаг: ровно так же его ведёт интерфейс (continue_step),
            # поэтому шаги идут по одному.
            continuous = turn > 0
            mark_before = _log_mark(session)
            events, route_error = await _turn(session, request_text, continuous)
            if route_error:
                error = route_error
                break
            state = workspace_store.dialog_state(session)
            if state.paused:
                interrupted = True
                break
            if _log_mark(session) == mark_before:
                # Запрос не оставил в чате ни следа (например, задача только что
                # стала на паузу): повторять его в этом же повторе нельзя — иначе
                # цепочка крутилась бы до предела запросов впустую.
                error = _last_error(events) or "повтор не выполнен"
                break
            if state.stage == "execution" and state.step_index == 0:
                # ЦИКЛ ПРОЙДЕН: автомат вернулся на первый шаг (cycle_done).
                break
            if state.stage == "execution":
                continue
            if state.stage == "failed":
                error = state.reason or "шаг задачи не выполнен"
            elif state.stage == "awaiting_user":
                error = "задача ждёт пользователя (план не подтверждён)"
            elif state.stage == "cancelled":
                error = "задача отменена"
            break
        else:
            error = (f"повтор не уложился в {config.PERIODIC_MAX_TURNS} запросов — "
                     "продолжу при следующем повторе")
    return error, interrupted


async def _turn(session: Dict[str, Any], request_text: str,
                continuous: bool) -> Tuple[List[Dict[str, Any]], str]:
    """Один запрос автомата в рамках повтора.

    Возвращает (события потока, отказ маршрута). Отказ — только технический:
    задача не найдена (её удалили во время повтора) или не создана. Сообщения об
    ошибках самой задачи возвращаются в событиях — их разбирает _drive.
    """
    message = ChatMessage(
        content="" if continuous else request_text,
        session_id=str(session.get("id")),
        continue_step=continuous,
        periodic=True,
    )
    response = await chat.agent_chat(message)
    if not isinstance(response, StreamingResponse):
        detail = ""
        try:
            detail = str(json.loads(bytes(response.body).decode("utf-8")).get("detail") or "")
        except Exception:  # noqa: BLE001
            detail = ""
        status = getattr(response, "status_code", 500)
        return [], f"запрос автомата отклонён ({status}{': ' + detail if detail else ''})"
    events: List[Dict[str, Any]] = []
    async for chunk in response.body_iterator:
        for line in str(chunk).splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except ValueError:
                continue
    return events, ""
