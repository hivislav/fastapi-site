"""Самопроверка Task State Machine (конечного автомата задачи «AI-агента»).

Запуск (сеть и API-ключ НЕ нужны — вызовы LLM подменяются заглушкой):

    ./venv/bin/python tools/check_task_machine.py

Скрипт проверяет ядро (app/ai/task_state.py), хранение состояния в сессии
(app/ai/workspace.py) и контроллер переходов вместе с маршрутами
(app/routers/chat.py): этапы planning → execution → validation → done, запрет
перескоков, пауза/«Продолжить», правку и подтверждение плана, ошибку шага
(execution → failed) и сохранение состояния в файле workspace.

Рабочие данные не трогаются: workspace, история агента и профили пишутся в
временный каталог (переменные AGENT_*_FILE выставляются ДО импорта chat).
"""

import asyncio
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Изоляция данных: всё пишем во временный каталог ------------------------
_TMP = tempfile.mkdtemp(prefix="tsm-check-")
os.environ["AGENT_WORKSPACE_FILE"] = os.path.join(_TMP, "workspace.json")
os.environ["AGENT_MEMORY_FILE"] = os.path.join(_TMP, "agent_memory.json")
os.environ["AGENT_PROFILES_FILE"] = os.path.join(_TMP, "profiles.json")

from app.ai import client, task_state, workspace as workspace_store  # noqa: E402
from app.routers import chat  # noqa: E402
from app.schemas import ChatMessage, PlanUpdate  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    """Одна проверка: печатает результат и копит провалы."""
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# Заглушка LLM: план — JSON из PLAN_STEPS, ответ — ANSWER
# ---------------------------------------------------------------------------
PLAN_STEPS = ["Собрать данные", "Написать код"]
ANSWER = "Ответ модели по текущему шагу."
# Вердикт содержательной проверки (этап validation): {"verdict", "step", "comment"}.
# None — «модель не ответила»: проверка недоступна, задача завершается по
# локальной самопроверке.
REVIEW = {"verdict": "ok", "step": 0, "comment": "результат соответствует плану"}
CALLS = []
# «Медленный» ответ модели: шаг «в полёте», пока тест не отпустит событие
# (проверяем мгновенную реакцию «Паузы»/«Отмены» и переключение сессии).
SLOW_ANSWER = None
# Очередь «медленных» ответов: i-й вызов ждёт i-е событие. Нужна для проверки
# ПАРАЛЛЕЛЬНОСТИ: две задачи держим «в полёте» одновременно и отпускаем по одной.
SLOW_QUEUE = []
# Медленное ПОСТРОЕНИЕ ПЛАНА (служебный вызов планировщика): нужен, чтобы
# проверить паузу, нажатую во время планирования.
SLOW_PLAN = None


def _metrics(prompt=20, completion=10):
    return {"model": "stub", "elapsed_seconds": 0.01, "prompt_tokens": prompt,
            "completion_tokens": completion, "total_tokens": prompt + completion}


async def fake_call_llm_async(*args, **kwargs):
    """Подмена client.call_llm_async: план — JSON, ответ — текст, всё локально."""
    messages = kwargs.get("messages") or []
    system = str(messages[0].get("content") or "") if messages else ""
    CALLS.append({"system": system[:40], "messages": len(messages),
                  "user_text": kwargs.get("user_text")})
    if system.startswith("Ты — планировщик"):
        if SLOW_PLAN is not None:
            await asyncio.wait_for(SLOW_PLAN.wait(), timeout=10)
        if not PLAN_STEPS:
            return "", _metrics()
        return json.dumps({"steps": list(PLAN_STEPS)}, ensure_ascii=False), _metrics(30, 15)
    if system.startswith("Ты — приёмщик"):
        if REVIEW is None:
            return "", _metrics()
        return json.dumps(REVIEW, ensure_ascii=False), _metrics(40, 8)
    if SLOW_QUEUE:
        await asyncio.wait_for(SLOW_QUEUE.pop(0).wait(), timeout=10)
    elif SLOW_ANSWER is not None:
        await asyncio.wait_for(SLOW_ANSWER.wait(), timeout=10)
    return ANSWER, _metrics()


client.call_llm_async = fake_call_llm_async


async def run_chat(text, **kwargs):
    """Прогон POST /api/agent/chat без сети: собирает события NDJSON-потока."""
    response = await chat.agent_chat(ChatMessage(content=text, **kwargs))
    events = []
    async for chunk in response.body_iterator:
        for line in str(chunk).splitlines():
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def sources_of(events):
    """Реплики пользователя, помеченные как сгенерированные автоматом."""
    return [m.get("source") for m in events if m.get("type") == "user_source"]


def stage_of(events):
    """Последнее состояние автомата из потока событий."""
    states = [e["state"] for e in events if e.get("type") == "state"]
    return states[-1] if states else None


def texts(events, kind):
    return [e.get("text", "") for e in events if e.get("type") == kind]


# ---------------------------------------------------------------------------
# 1. Ядро автомата
# ---------------------------------------------------------------------------
def test_core():
    print("\n[1] Ядро: этапы, переходы, история, пауза")
    state = task_state.new_state("s-test")
    check("новое состояние — planning", state.stage == "planning")
    check("три обязательных поля заполнены",
          state.stage == "planning" and state.current_step == "" and bool(state.expected_action))

    # Запрещённые переходы: перескоки этапов.
    for target in ("done", "validation", "cancelled"):
        try:
            state.transition(target, "перескок")
            check(f"запрет planning → {target}", False, "(переход прошёл)")
        except task_state.IllegalTransition:
            check(f"запрет planning → {target}", True)
    check("состояние не изменилось после запрета", state.stage == "planning")

    steps = ["Первый шаг", "Второй шаг", "Третий шаг"]
    task_state.await_confirmation(state, steps, "план показан")
    check("planning → awaiting_user", state.stage == "awaiting_user")
    check("план сохранён", state.steps == steps)
    # Место в базовой цепочке: расширение awaiting_user не сдвигает полосу этапов.
    check("awaiting_user не сдвигает базовый этап", state.base_stage == "planning")
    check("текущий шаг подсвечен до подтверждения плана",
          all(step["active"] for step in task_state.snapshot(state)["steps"][:1]))
    task_state.start_planning(state, "пользователь ответил «ок»")
    check("awaiting_user → planning", state.stage == "planning")
    task_state.plan_ready(state, steps, "план подтверждён")
    check("planning → execution, шаг 1", state.stage == "execution" and state.current_step == "step_1")
    check("expected_action шага", "Первый шаг" in state.expected_action)

    check("следующий шаг", task_state.next_step(state, "шаг 1 выполнен"))
    check("current_step обновился", state.current_step == "step_2")
    check("шаг 2 в ожидании", "Второй шаг" in state.expected_action)
    check("шагов больше нет", task_state.next_step(state, "шаг 2 выполнен")
          and not task_state.next_step(state, "шаг 3 выполнен"))

    task_state.to_validation(state, "шаги выполнены")
    check("execution → validation, шаг check", state.stage == "validation" and state.current_step == "check")
    check("validation — в базовой цепочке", state.base_stage == "validation")
    check("ожидание проверки шага", "проверить результат шага" in state.expected_action)
    task_state.validation_failed(state, "проверка не пройдена", 1)
    check("validation → execution на нужный шаг",
          state.stage == "execution" and state.current_step == "step_2")
    task_state.to_validation(state, "шаги выполнены повторно")
    task_state.validation_ok(state, "проверка пройдена")
    check("validation → done, поля очищены",
          state.stage == "done" and state.current_step == "" and state.expected_action == "")
    check("терминальный этап: переходов нет", not state.can_transition("execution"))
    check("done — конец базовой цепочки", state.base_stage == "done")
    check("снимок состояния: три обязательных поля",
          all(key in task_state.snapshot(state) for key in ("stage", "current_step", "expected_action")))

    # История переходов — формат {"from", "to", "step", "at", "reason"}.
    required = {"from", "to", "step", "at", "reason"}
    check("history: все записи в формате спецификации",
          all(required <= set(record) for record in state.history))
    check("history: переходы записаны",
          any(r["from"] == "planning" and r["to"] == "execution" for r in state.history)
          and any(r["from"] == "validation" and r["to"] == "done" for r in state.history))
    check("history: причина у каждого перехода",
          all(str(r["reason"]).strip() for r in state.history))

    # Пауза: этап и шаг не меняются.
    paused = task_state.new_state("s-pause")
    task_state.plan_ready(paused, steps, "план подтверждён")
    task_state.next_step(paused, "шаг 1 выполнен")
    before = (paused.stage, paused.current_step, paused.expected_action)
    task_state.pause(paused, "пауза пользователем")
    check("пауза не меняет этап и шаг",
          (paused.stage, paused.current_step) == (before[0], before[1]) and paused.paused)
    check("ожидание на паузе", paused.expected_action == task_state.ACTION_PAUSED)
    task_state.resume(paused, "продолжить")
    check("продолжение восстанавливает ожидание",
          not paused.paused and paused.expected_action == before[2])
    try:
        task_state.pause(state, "поздно")
        check("пауза после done запрещена", False)
    except task_state.IllegalTransition:
        check("пауза после done запрещена", True)

    # Ошибка: current_step сохраняется.
    failed = task_state.new_state("s-fail")
    task_state.plan_ready(failed, steps, "план подтверждён")
    task_state.next_step(failed, "шаг 1 выполнен")
    task_state.fail(failed, "модель не ответила")
    check("execution → failed с сохранением шага",
          failed.stage == "failed" and failed.current_step == "step_2")
    check("failed помнит, где задача остановилась", failed.base_stage == "execution")
    task_state.start_planning(failed, "перезапуск")
    check("failed → planning", failed.stage == "planning")

    # Отмена — остановка из незавершённого этапа.
    cancel_state = task_state.new_state("s-cancel")
    task_state.plan_ready(cancel_state, steps, "план подтверждён")
    task_state.cancel(cancel_state, "пользователь отменил задачу")
    check("отмена переводит в cancelled", cancel_state.stage == "cancelled")
    try:
        task_state.pause(cancel_state, "уже отменена")
        check("пауза после отмены запрещена", False)
    except task_state.IllegalTransition:
        check("пауза после отмены запрещена", True)

    # Восстановление из файла.
    raw = task_state.to_dict(state)
    restored = task_state.from_dict(json.loads(json.dumps(raw)), "s-test")
    check("round-trip состояния",
          restored.stage == state.stage and restored.steps == state.steps)
    check("round-trip истории", len(restored.history) == len(state.history))
    garbage = task_state.from_dict({"stage": "мусор", "step_index": "ой", "history": 5,
                                    "steps": [{"text": "  шаг  "}, "", None, 42]})
    check("битые данные → planning", garbage.stage == "planning")
    check("шаги нормализованы", garbage.steps == ["шаг", "42"])

    # ASCII-блок состояния.
    block = task_state.ascii_block(state)
    widths = {len(line) for line in block.splitlines()}
    check("ASCII-блок: одинаковая ширина строк", len(widths) == 1, f"(ширины: {widths})")
    check("ASCII-блок: этапы и кнопка на месте",
          all(name in block for name in task_state.BASE_STAGES) and "done" in block)

    # Локальный план (запасной вариант).
    local = task_state.fallback_steps("1. Собрать требования\n2. Написать код\n3. Прогнать тесты")
    check("локальный план: 3 шага", local == ["Собрать требования", "Написать код", "Прогнать тесты"])
    check("локальный план: один шаг для простого запроса",
          task_state.fallback_steps("скажи привет") == ["скажи привет"])


# ---------------------------------------------------------------------------
# 2. Хранение состояния в сессии (workspace)
# ---------------------------------------------------------------------------
def test_workspace():
    print("\n[2] Хранение: состояние живёт в диалоге сессии и в файле workspace")
    dialog = workspace_store.empty_dialog("s-1")
    check("empty_dialog содержит state", isinstance(dialog.get("state"), dict))
    check("state начинается с planning", dialog["state"]["stage"] == "planning")

    task = {"id": "t-1", "name": "Проверка", "sessions": [], "profile": None}
    session = workspace_store.create_session(task)
    check("create_session задаёт task_id = id сессии",
          session["dialog"]["state"]["task_id"] == session["id"])

    state = workspace_store.dialog_state(session)
    task_state.plan_ready(state, ["Шаг"], "план подтверждён")
    workspace_store.set_dialog_state(session, state)
    check("состояние записалось в диалог", session["dialog"]["state"]["stage"] == "execution")

    broken = workspace_store.normalize_dialog({"state": {"stage": "неизвестно"}}, "s-2")
    check("битый state при чтении → planning", broken["state"]["stage"] == "planning")

    workspace = workspace_store.normalize_workspace({})
    workspace["tasks"] = [task]
    workspace["active_tasks"] = {"": "t-1"}
    path = os.path.join(_TMP, "ws-roundtrip.json")
    workspace_store.save_workspace(workspace, path)
    loaded = workspace_store.load_workspace(path)
    loaded_state = loaded["tasks"][0]["sessions"][0]["dialog"]["state"]
    check("этап пережил запись/чтение", loaded_state["stage"] == "execution")
    check("история переходов пережила запись/чтение", len(loaded_state["history"]) >= 2)


# ---------------------------------------------------------------------------
# 3. Контроллер и маршруты
# ---------------------------------------------------------------------------
async def test_routes():
    print("\n[3] Маршруты: полный цикл автомата, пауза, план, ошибка")
    global PLAN_STEPS, ANSWER
    PLAN_STEPS = ["Собрать данные", "Написать код"]
    ANSWER = "Ответ модели по текущему шагу."
    snapshot = await chat.task_create(chat.TaskCreate(name="Задача автомата"))
    check("задача создана", bool(snapshot.get("active_task")))
    session_id = None

    # 3.1 Первый запрос: план и ожидание подтверждения.
    events = await run_chat("Сделай отчёт по продажам")
    state = stage_of(events)
    check("первый запрос → awaiting_user", state and state["stage"] == "awaiting_user")
    check("план показан в чате", any("План задачи" in t for t in texts(events, "bot")))
    check("шаги плана в состоянии",
          state and [s["text"] for s in state["steps"]] == PLAN_STEPS)
    check("в режиме планирования ответ модели не запрашивался",
          all("Ты — планировщик" in c["system"] for c in CALLS))
    check("есть событие state", bool(stage_of(events)))

    # 3.2 «ок» — подтверждение плана: выполняется ПЕРВЫЙ шаг (один шаг на
    #     запрос пользователя), задача идёт к следующему шагу.
    plan_calls = sum(1 for c in CALLS if c["system"].startswith("Ты — планировщик"))
    events = await run_chat("ок")
    state = stage_of(events)
    check("«ок» → execution, шаг 1 выполнен, текущий шаг 2",
          state and state["stage"] == "execution" and state["current_step"] == "step_2",
          f"({state and state['stage']} / {state and state['current_step']})")
    check("ответ модели получен", "Ответ модели по текущему шагу." in texts(events, "bot"))
    check("шаг 1 ушёл в модель с блоком состояния",
          any("СОСТОЯНИЕ ЗАДАЧИ" in t for t in texts(events, "debug"))
          or CALLS[-1]["messages"] >= 2)
    check("подтверждённый план заново не строился",
          sum(1 for c in CALLS if c["system"].startswith("Ты — планировщик")) == plan_calls)
    step_lines = texts(events, "debug")
    check("движение по плану описано в debug",
          any("шаг 1 из 2" in t for t in step_lines)
          and any("шаг 1 из 2 выполнен" in t for t in step_lines),
          str([t for t in step_lines if "шаг" in t][:4]))

    # 3.3 Последний шаг: любой запрос выполняет текущий шаг, затем проверка.
    events = await run_chat("продолжай")
    state = stage_of(events)
    check("последний шаг → validation → done", state and state["stage"] == "done")
    check("самопроверка описана в debug",
          any("проверка пройдена" in t for t in texts(events, "debug")))
    check("в истории есть переход validation → done",
          any(r["from"] == "validation" and r["to"] == "done" for r in state["history"]))

    # 3.3а Порядок событий: проверка результата видна в потоке (этап validation
    #      отдельным событием состояния) и стоит ПОСЛЕ ответа модели; в дебаге
    #      нет двух одинаковых строк про один и тот же шаг.
    kinds = [(e.get("type"), (e.get("state") or {}).get("stage", "")) for e in events]
    check("этап validation виден в потоке событий",
          ("state", "validation") in kinds, str(kinds[-6:]))
    check("событие validation идёт до финального done",
          [i for i, k in enumerate(kinds) if k == ("state", "validation")]
          and max(i for i, k in enumerate(kinds) if k == ("state", "validation"))
          < max(i for i, k in enumerate(kinds) if k == ("state", "done")))
    check("сообщение модели стоит раньше проверки",
          max(i for i, e in enumerate(events) if e.get("type") == "bot")
          < max(i for i, k in enumerate(kinds) if k == ("state", "validation")))
    # Строка «что делаем сейчас и что ожидается» должна быть РОВНО одна на шаг:
    # раньше её дублировали контроллер (после перехода) и сам агент.
    check("в дебаге одна строка про текущий шаг",
          len([t for t in texts(events, "debug") if "Ожидается:" in t]) == 1,
          str([t for t in texts(events, "debug") if "Ожидается:" in t]))

    # 3.4 Новый запрос после done — новая задача (автомат с нуля).
    events = await run_chat("Теперь сделай презентацию")
    state = stage_of(events)
    check("запрос после done → новая задача (awaiting_user)",
          state and state["stage"] == "awaiting_user")
    check("сброс задачи записан в историю",
          any(r.get("reset") for r in state["history"]))

    # 3.4 Правка плана и подтверждение кнопкой.
    fixed = await chat.state_plan(PlanUpdate(steps=["Только один шаг"]))
    check("правка плана сохранена", [s["text"] for s in fixed["state"]["steps"]] == ["Только один шаг"])
    confirmed = await chat.state_confirm()
    check("кнопка «Подтвердить план» → execution",
          confirmed["state"]["stage"] == "execution"
          and confirmed["state"]["current_step"] == "step_1")

    events = await run_chat("выполняй")
    state = stage_of(events)
    check("после подтверждения задача завершается за один шаг",
          state and state["stage"] == "done")

    # 3.5 Пауза и «Продолжить» (план из одного шага — цикл короче).
    PLAN_STEPS = ["Единственный шаг"]
    await chat.agent_history_clear()
    events = await run_chat("Подготовь план встречи")
    state = stage_of(events)
    check("новая задача ждёт подтверждения плана",
          state and state["stage"] == "awaiting_user" and len(state["steps"]) == 1)
    paused = await chat.state_pause()
    check("пауза: состояние помечено", paused["state"]["paused"] is True)
    check("пауза: этап не изменился", paused["state"]["stage"] == "awaiting_user")
    check("пауза: ожидание — «Продолжить»",
          paused["state"]["expected_action"] == task_state.ACTION_PAUSED)
    events = await run_chat("ок")
    state = stage_of(events)
    check("на паузе запрос не выполняется",
          state and state["paused"] and state["stage"] == "awaiting_user")
    check("на паузе пользователю сказано нажать «Продолжить»",
          any("Продолжить" in t for t in texts(events, "error")))
    resumed = await chat.state_resume()
    check("«Продолжить» снимает паузу", resumed["state"]["paused"] is False)
    events = await run_chat("ок")
    state = stage_of(events)
    check("после «Продолжить» задача доходит до done", state and state["stage"] == "done")

    # 3.6 Ошибка шага: пустой ответ модели → failed, затем перезапуск.
    ANSWER = ""
    await chat.agent_history_clear()
    events = await run_chat("Сделай то, на что модель не ответит")
    state = stage_of(events)
    check("новая задача ждёт подтверждения плана", state and state["stage"] == "awaiting_user")
    events = await run_chat("ок")
    state = stage_of(events)
    check("пустой ответ модели → failed", state and state["stage"] == "failed",
          f"({state and state['stage']})")
    check("при ошибке шаг сохранён", state and state["current_step"] == "step_1")
    check("причина ошибки — в истории",
          any(r["to"] == "failed" and "не выполнен" in r["reason"]
              for r in (state or {}).get("history", [])))
    ANSWER = "Ответ модели по текущему шагу."
    PLAN_STEPS = ["Один шаг"]
    events = await run_chat("перезапусти")
    state = stage_of(events)
    check("«перезапусти» → failed → planning (новый план)",
          state and state["stage"] == "awaiting_user" and len(state["steps"]) == 1)
    check("в истории есть переход failed → planning",
          any(r["from"] == "failed" and r["to"] == "planning"
              for r in (state or {}).get("history", [])))

    # 3.7 Автономный режим: подтверждение не требуется.
    events = await run_chat("Сделай всё сам, работай автономно")
    state = stage_of(events)
    check("«работай автономно» → выполнение без подтверждения",
          state and state["stage"] == "done" and state["autonomous"] is True)

    # 3.8 Состояние видно в истории диалога и в отдельном маршруте.
    history = await chat.agent_history()
    check("GET /api/agent/history отдаёт состояние",
          history["state"]["stage"] == "done" and history["state"]["steps"])
    single = await chat.state_get()
    check("GET /api/agent/state отдаёт состояние", single["state"]["stage"] == "done")
    check("состояние задачи сохранено в файле",
          os.path.exists(os.environ["AGENT_WORKSPACE_FILE"]))

    # 3.9 Очистка истории начинает автомат заново.
    await chat.agent_history_clear()
    fresh = await chat.state_get()
    check("очистка истории → автомат с нуля",
          fresh["state"]["stage"] == "planning" and not fresh["state"]["steps"])

    # 3.10 Одновременность/согласованность: подтверждение без плана — 400.
    try:
        await chat.state_confirm()
        check("подтверждение без плана запрещено", False)
    except Exception as exc:  # HTTPException
        check("подтверждение без плана запрещено",
              getattr(exc, "status_code", None) == 400, f"({exc})")

    # 3.11 Запрос от автомата (continue_step): шаг выполняется без сообщения
    #      пользователя, а реплика помечается source="machine".
    ANSWER = "Ответ модели по текущему шагу."
    PLAN_STEPS = ["Первый шаг авто-прогона", "Второй шаг авто-прогона"]
    await chat.agent_history_clear()
    await run_chat("Сделай что-нибудь по шагам")
    events = await run_chat("ок")
    state = stage_of(events)
    check("«ок» уже выполняет первый шаг",
          state and state["stage"] == "execution" and state["current_step"] == "step_2")
    calls_before = len(CALLS)
    events = await run_chat("", continue_step=True)
    state = stage_of(events)
    check("continue_step выполняет шаг без сообщения пользователя",
          len(CALLS) > calls_before and state and state["stage"] == "done",
          f"({state and state['stage']})")
    check("шаг выполнен моделью", "Ответ модели по текущему шагу." in texts(events, "bot"))
    session = chat._current_session()
    machine = [m for m in session["dialog"]["messages"] if m.get("source") == "machine"]
    check("реплика автомата помечена source=machine", len(machine) == 1,
          f"(размечено: {len(machine)})")
    check("реплика автомата содержит текст шага",
          machine and "Второй шаг авто-прогона" in machine[0]["content"],
          machine and machine[0]["content"])
    check("пометка переживает запись/чтение workspace",
          all(m.get("source") == "machine"
              for m in workspace_store.normalize_dialog(session["dialog"])["messages"]
              if "Продолжай по плану" in m["content"]))
    calls_before = len(CALLS)
    events = await run_chat("", continue_step=True)
    state = stage_of(events)
    check("continue_step на завершённой задаче ничего не выполняет",
          len(CALLS) == calls_before and state and state["stage"] == "done")
    check("и объясняет, что делать",
          any("Работать нечего" in t for t in texts(events, "error")))

    # 3.12 Содержательная проверка результата моделью (этап validation).
    global REVIEW
    PLAN_STEPS = ["Единственный шаг"]
    ANSWER = "Ответ модели по текущему шагу."
    REVIEW = {"verdict": "ok", "step": 0, "comment": "план закрыт"}
    await chat.agent_history_clear()
    await run_chat("Сделай отчёт")
    events = await run_chat("ок")
    state = stage_of(events)
    check("проверка моделью пройдена → done", state and state["stage"] == "done")
    check("проверка моделью видна в дебаге",
          any("содержательная проверка пройдена" in t for t in texts(events, "debug")))
    check("вердикт модели записан в историю перехода",
          any("модель: план закрыт" in r["reason"] for r in (state or {}).get("history", [])))
    saved = chat._current_session()["dialog"]["usage"][-1]
    check("токены проверки вошли в замер запроса",
          saved.get("summary_requests", 0) >= 1 and saved.get("requests", 0) >= 2, str(saved))
    check("в потоке есть уточнённый замер (событие usage)",
          any(e.get("type") == "usage" for e in events))
    plan_records = [u for u in chat._current_session()["dialog"]["usage"] if u.get("kind") == "plan"]
    check("замер запроса планирования сохранён отдельной записью", len(plan_records) == 1,
          str(plan_records))

    # Доработка по требованию модели: validation → execution на указанный шаг.
    REVIEW = {"verdict": "redo", "step": 0, "comment": "шаг не раскрыт"}
    await chat.agent_history_clear()
    await run_chat("Проверь работу")
    events = await run_chat("ок")
    state = stage_of(events)
    check("проверка вернула задачу в execution",
          state and state["stage"] == "execution" and state["redo_count"] == 1,
          f"({state and state['stage']}, redo={state and state['redo_count']})")
    check("переход validation → execution с причиной модели",
          any(r["from"] == "validation" and "модель не приняла" in r["reason"]
              for r in state["history"]))
    events = await run_chat("", continue_step=True)
    state = stage_of(events)
    check("вторая неудача проверки — ещё доработка (redo_count 2)",
          state and state["stage"] == "execution" and state["redo_count"] == 2,
          f"({state and state['stage']}, redo={state and state['redo_count']})")
    events = await run_chat("", continue_step=True)
    state = stage_of(events)
    check("после MAX_REDO доработок задача уходит в failed",
          state and state["stage"] == "failed", state and state["stage"])
    check("в причине ошибки — число доработок и вердикт модели",
          any(r["to"] == "failed" and "после 2" in r["reason"] and "не раскрыт" in r["reason"]
              for r in (state or {}).get("history", [])))
    events = await run_chat("продолжай")
    state = stage_of(events)
    check("из failed задача перезапускается сообщением",
          state and any(r["from"] == "failed" and r["to"] == "planning"
                        for r in state["history"]))

    # Проверка недоступна: задача завершается по локальной самопроверке.
    REVIEW = None
    await chat.agent_history_clear()
    await run_chat("Сделай отчёт")
    events = await run_chat("ок")
    state = stage_of(events)
    check("недоступная проверка не мешает завершить задачу",
          state and state["stage"] == "done", state and state["stage"])
    check("об этом сказано в дебаге",
          any("содержательная проверка не получена" in t for t in texts(events, "debug")))
    REVIEW = {"verdict": "ok", "step": 0, "comment": "ок"}

    # 3.13 Отмена задачи кнопкой «Отменить»: cancelled — терминальный этап.
    REVIEW = {"verdict": "ok", "step": 0, "comment": "ок"}
    await chat.agent_history_clear()
    await run_chat("Сделай отчёт")
    state = (await chat.state_get())["state"]
    check("нетерминальную задачу можно отменить", state["can_cancel"] is True)
    cancelled = (await chat.state_cancel())["state"]
    check("отмена переводит задачу в cancelled",
          cancelled["stage"] == "cancelled" and cancelled["terminal"] is True)
    check("после отмены отменять нечего", cancelled["can_cancel"] is False)
    check("причина отмены в истории",
          any(r["to"] == "cancelled" and "отменил" in r["reason"] for r in cancelled["history"]))
    try:
        await chat.state_cancel()
        check("повторная отмена запрещена", False)
    except Exception as exc:   # HTTPException
        check("повторная отмена запрещена", getattr(exc, "status_code", None) == 400, str(exc))
    events = await run_chat("Новая задача: посчитай смету")
    state = stage_of(events)
    check("сообщение после отмены начинает новую задачу",
          state and state["stage"] == "awaiting_user" and state["redo_count"] == 0)
    check("сброс после отмены записан в историю",
          any(r.get("reset") for r in state["history"]))

    # 3.14 Счётчик доработок в снимке состояния (для полосы этапов).
    REVIEW = {"verdict": "redo", "step": 0, "comment": "не раскрыто"}
    await chat.agent_history_clear()
    await run_chat("Проверь работу")
    events = await run_chat("ок")
    state = stage_of(events)
    check("в снимке есть счётчик и лимит доработок",
          state and state["redo_count"] == 1 and state["max_redo"] == task_state.MAX_REDO
          and state["can_redo"] is True,
          str({k: (state or {}).get(k) for k in ("redo_count", "max_redo", "can_redo")}))
    events = await run_chat("", continue_step=True)
    state = stage_of(events)
    check("после лимита can_redo выключается",
          state and state["redo_count"] == task_state.MAX_REDO and state["can_redo"] is False,
          f"({state and state['redo_count']}, can_redo={state and state['can_redo']})")
    events = await run_chat("", continue_step=True)
    state = stage_of(events)
    check("доработки не бесконечны: задача уходит в failed",
          state and state["stage"] == "failed", state and state["stage"])
    REVIEW = {"verdict": "ok", "step": 0, "comment": "ок"}

    # 3.15 Журнал чата: запрос, план, ответы и debug видны после переключения.
    global SLOW_ANSWER
    REVIEW = {"verdict": "ok", "step": 0, "comment": "ок"}
    PLAN_STEPS = ["Первый шаг плана", "Второй шаг плана", "Третий шаг плана"]
    await chat.agent_history_clear()
    await run_chat("Дай рецепт борща")
    history = await chat.agent_history()
    log = history.get("log") or []
    kinds = [item["kind"] for item in log]
    check("журнал чата ведётся", bool(log), str(kinds))
    check("в журнале есть запрос пользователя",
          any(i["kind"] == "user" and "рецепт борща" in i["text"] for i in log), str(log[:3]))
    check("в журнале есть показанный план",
          any(i["kind"] == "assistant" and "План задачи" in i["text"] for i in log))
    check("в журнале есть debug-строки агента", "debug" in kinds)
    messages = history["messages"]
    check("запрос пользователя сохранён и в памяти диалога",
          any(m["role"] == "user" and "рецепт борща" in m["content"] for m in messages),
          str([m["content"][:40] for m in messages]))
    check("план сохранён и в памяти диалога",
          any(m["role"] == "assistant" and "План задачи" in m["content"] for m in messages))
    title = workspace_store.session_title(chat._current_session())
    check("заголовок сессии — по запросу, а не по служебной фразе",
          "борща" in title and "Продолжай по плану" not in title, title)

    await run_chat("ок")                                 # шаг 1 из 3
    await run_chat("", continue_step=True)                # шаг 2 из 3
    events = await run_chat("", continue_step=True)       # шаг 3 из 3 → проверка → done
    history = await chat.agent_history()
    check("журнал не теряется между шагами и растёт",
          len(history["log"]) > len(log), f"({len(history['log'])} против {len(log)})")
    machine = [m for m in history["messages"] if m.get("source") == "machine"]
    check("пометка source=machine переживает следующие запросы",
          len(machine) >= 2, f"(размечено: {len(machine)})")
    check("в состоянии сохранён исходный запрос задачи",
          (history["state"].get("request") or "").startswith("Дай рецепт борща"),
          history["state"].get("request"))
    review_payloads = [c["user_text"] for c in CALLS
                       if (c.get("user_text") or "").startswith("Пользователь")]
    check("проверка результата получает исходный запрос, а не «ок»",
          any("Исходный запрос пользователя:\nДай рецепт борща" in (c["user_text"] or "")
              for c in CALLS),
          str([(c["user_text"] or "")[:60] for c in CALLS[-2:]]))

    # 3.16 Мгновенные «Пауза»/«Отменить» во время шага + переключение сессии.
    await chat.agent_history_clear()
    await run_chat("Задача для проверки остановки")       # план из 3 шагов
    await run_chat("ок")                                  # шаг 1 → execution step_2
    first_session = chat._current_session()["id"]
    created = await chat.session_create()                 # вторая сессия
    second_session = created["active_session"]
    check("создана вторая сессия", second_session != first_session)
    await chat.session_select(first_session)

    SLOW_ANSWER = asyncio.Event()                         # шаг 2 «в полёте»
    slow = asyncio.ensure_future(run_chat("", continue_step=True))
    await asyncio.sleep(0.05)
    started = time.perf_counter()
    paused = await chat.state_pause()
    pause_ms = (time.perf_counter() - started) * 1000
    check("«Пауза» во время шага отвечает мгновенно",
          pause_ms < 200 and paused["state"].get("pending") == "pause",
          f"({pause_ms:.0f} мс, pending={paused['state'].get('pending')})")
    check("снимок сразу показывает паузу", paused["state"]["paused"] is True)

    started = time.perf_counter()
    selected = await chat.session_select(second_session)
    select_ms = (time.perf_counter() - started) * 1000
    check("переключение сессии во время шага мгновенное", select_ms < 200,
          f"({select_ms:.0f} мс)")
    check("переключение не сбило активную сессию", selected["active_session"] == second_session)

    SLOW_ANSWER.set()                                     # отпускаем шаг
    await slow
    SLOW_ANSWER = None
    await chat.session_select(first_session)
    state = (await chat.state_get())["state"]
    check("после ответа модели пауза применена", state["paused"] is True, str(state["paused"]))
    check("пауза записана в историю с причиной",
          any("во время выполнения шага" in r["reason"] for r in state["history"]),
          str([r["reason"] for r in state["history"][-2:]]))
    check("ответ шага при этом не потерян",
          any("Ответ модели" in item["text"] for item in (await chat.agent_history())["log"]))
    await chat.state_resume()

    # «Пауза» целится в ВЫПОЛНЯЕМУЮ задачу, даже если открыта другая: иначе
    # намерение «залипало» бы в очереди чужой задачи и срабатывало позже.
    # Свежая задача с планом из 3 шагов: шаг 1 выполнен, шаг 2 будет «в полёте».
    # Именно три шага: у ПОСЛЕДНЕГО шага своя семантика паузы (перед проверкой).
    PLAN_STEPS = ["Первый шаг", "Второй шаг", "Третий шаг"]
    await chat.agent_history_clear()
    await run_chat("Задача для адресной паузы")
    await run_chat("ок")                                   # шаг 1 → execution step_2
    running_session = chat._current_session()["id"]
    created = await chat.session_create()                  # открываем ВТОРУЮ задачу
    opened_session = created["active_session"]
    await chat.session_select(running_session)

    # (1) «Пауза» в САМОЙ выполняющейся задаче: ответ мгновенный (pending), а
    #     переход применяется, когда модель закончит шаг.
    SLOW_ANSWER = asyncio.Event()
    slow = asyncio.ensure_future(run_chat("", continue_step=True))   # шаг 2 из 3
    await asyncio.sleep(0.05)
    check("шаг действительно выполняется (блокировка задачи занята)",
          chat._session_lock(running_session).locked())
    check("сервер знает, ЧЬЯ задача выполняется",
          running_session in chat._running_sessions,
          str(chat._running_sessions))
    started = time.perf_counter()
    paused_running = await chat.state_pause()
    running_ms = (time.perf_counter() - started) * 1000
    check("«Пауза» в выполняющейся задаче отвечает мгновенно",
          running_ms < 200 and paused_running["state"].get("pending") == "pause",
          f"({running_ms:.0f} мс, pending={paused_running['state'].get('pending')})")
    check("снимок сразу показывает паузу", paused_running["state"]["paused"] is True)
    SLOW_ANSWER.set()
    await slow
    SLOW_ANSWER = None
    state_running = (await chat.state_get())["state"]
    check("после ответа модели пауза применена",
          state_running["paused"] is True and state_running["current_step"] == "step_3",
          f"(paused={state_running['paused']}, шаг={state_running['current_step']})")
    check("причина паузы — в истории",
          any("во время выполнения шага" in r["reason"] for r in state_running["history"]))
    await chat.state_resume()                              # продолжаем задачу

    # (2) Открыта ДРУГАЯ задача: её «Пауза» применяется МГНОВЕННО (диалог этой
    #     задачи никто не пишет) и не ждёт шага выполняющейся — раньше кнопка
    #     выглядела «зависшей».
    await chat.session_select(running_session)
    SLOW_ANSWER = asyncio.Event()
    slow = asyncio.ensure_future(run_chat("", continue_step=True))   # шаг 3 (последний)
    await asyncio.sleep(0.05)
    await chat.session_select(opened_session)
    started = time.perf_counter()
    paused_opened = await chat.state_pause()
    opened_ms = (time.perf_counter() - started) * 1000
    check("«Пауза» в открытой задаче не ждёт чужой шаг",
          opened_ms < 200 and paused_opened["state"]["paused"] is True,
          f"({opened_ms:.0f} мс, paused={paused_opened['state']['paused']})")
    check("пауза применена к открытой задаче",
          (paused_opened["session"] or {}).get("id") == opened_session,
          str((paused_opened["session"] or {}).get("id")))
    SLOW_ANSWER.set()
    await slow
    SLOW_ANSWER = None
    await chat.session_select(running_session)
    state_running = (await chat.state_get())["state"]
    check("выполнявшаяся задача этой паузой не тронута и дошла до конца",
          state_running["paused"] is False and state_running["stage"] == "done",
          f"(paused={state_running['paused']}, stage={state_running['stage']})")
    await chat.session_select(opened_session)
    await chat.state_resume()

    # Пауза на ПОСЛЕДНЕМ шаге: задача должна остановиться ПЕРЕД проверкой (на
    # этапе validation), а сама проверка — выполниться после «Продолжить».
    await chat.agent_history_clear()
    PLAN_STEPS = ["Первый шаг", "Последний шаг"]
    REVIEW = {"verdict": "ok", "step": 0, "comment": "ок"}
    await run_chat("Задача из двух шагов")
    await run_chat("ок")                                  # шаг 1 → execution, шаг 2 из 2
    reviews_before = sum(1 for c in CALLS if c["system"].startswith("Ты — приёмщик"))
    SLOW_ANSWER = asyncio.Event()
    slow = asyncio.ensure_future(run_chat("", continue_step=True))   # последний шаг
    await asyncio.sleep(0.05)
    pending = await chat.state_pause()                    # пауза во время шага
    check("пауза на последнем шаге принята (pending)",
          pending["state"].get("pending") == "pause", str(pending["state"].get("pending")))
    SLOW_ANSWER.set()
    events = await slow
    SLOW_ANSWER = None
    reviews_after = sum(1 for c in CALLS if c["system"].startswith("Ты — приёмщик"))
    state = stage_of(events) or (await chat.state_get())["state"]
    check("задача остановилась НА ЭТАПЕ ПРОВЕРКИ, а не ушла в done",
          state["stage"] == "validation", state["stage"])
    check("пауза при этом применена", state["paused"] is True, str(state["paused"]))
    check("при команде остановки содержательная проверка не вызывается",
          reviews_after == reviews_before, f"({reviews_after} против {reviews_before})")
    check("в дебаге сказано, что проверка отложена",
          any("проверку результата отложил" in t for t in texts(events, "debug")),
          str([t for t in texts(events, "debug") if "Пауза" in t][-2:]))

    # «Продолжить» → отложенная проверка выполняется и задача завершается.
    await chat.state_resume()
    reviews_before_resume = sum(1 for c in CALLS if c["system"].startswith("Ты — приёмщик"))
    events = await run_chat("", continue_step=True)
    state = stage_of(events) or (await chat.state_get())["state"]
    check("после «Продолжить» отложенная проверка выполнена",
          sum(1 for c in CALLS if c["system"].startswith("Ты — приёмщик")) > reviews_before_resume,
          "проверка не вызывалась")
    check("задача завершилась после проверки", state["stage"] == "done", state["stage"])
    check("в дебаге видно отложенную проверку",
          any("ОТЛОЖЕННУЮ проверку" in t for t in texts(events, "debug")),
          str([t for t in texts(events, "debug") if "validation" in t][:2]))

    # Отмена во время шага — на свежей задаче (шаг не последний).
    PLAN_STEPS = ["Первый шаг", "Второй шаг", "Третий шаг"]
    await chat.agent_history_clear()
    await run_chat("Задача для отмены на ходу")
    await run_chat("ок")                                  # шаг 1 → execution step_2
    SLOW_ANSWER = asyncio.Event()
    slow = asyncio.ensure_future(run_chat("", continue_step=True))
    await asyncio.sleep(0.05)
    started = time.perf_counter()
    cancelled = await chat.state_cancel()
    cancel_ms = (time.perf_counter() - started) * 1000
    check("«Отменить» во время шага отвечает мгновенно",
          cancel_ms < 200 and cancelled["state"].get("pending") == "cancel",
          f"({cancel_ms:.0f} мс, pending={cancelled['state'].get('pending')})")
    SLOW_ANSWER.set()
    await slow
    SLOW_ANSWER = None
    state = (await chat.state_get())["state"]
    check("после ответа модели задача отменена", state["stage"] == "cancelled", state["stage"])

    # 3.18 Фоновое выполнение: запрос шага называет СВОЮ задачу (session_id).
    PLAN_STEPS = ["Первый шаг", "Второй шаг", "Третий шаг"]
    REVIEW = {"verdict": "ok", "step": 0, "comment": "ок"}
    await chat.agent_history_clear()
    await run_chat("Фоновая задача")
    await run_chat("ок")                                  # шаг 1 → execution, шаг 2 из 3
    background = chat._current_session()["id"]
    created = await chat.session_create()                 # открываем ДРУГУЮ задачу
    opened = created["active_session"]
    check("открыта другая задача", opened != background)

    events = await run_chat("", continue_step=True, session_id=background)
    state_bg = stage_of(events) or {}
    check("шаг выполнен в НАЗВАННОЙ задаче, а не в открытой",
          state_bg.get("stage") == "execution" and state_bg.get("current_step") == "step_3",
          f"({state_bg.get('stage')}, {state_bg.get('current_step')})")
    state_opened = (await chat.state_get())["state"]
    check("открытая задача не тронута",
          state_opened["stage"] == "planning" and not state_opened["steps"],
          f"({state_opened['stage']}, шагов: {len(state_opened['steps'])})")
    history_opened = await chat.agent_history()
    check("в журнал открытой задачи чужая работа не попала",
          len(history_opened.get("log") or []) == 0,
          str([i["text"][:30] for i in (history_opened.get("log") or [])]))
    await chat.session_select(background)
    history_bg = await chat.agent_history()
    check("ответ записан в журнал СВОЕЙ задачи",
          any("Ответ модели" in i["text"] for i in history_bg["log"]),
          str([i["kind"] for i in history_bg["log"]][-3:]))
    check("шаг помечен как реплика автомата",
          any(m.get("source") == "machine" for m in history_bg["messages"]))

    # Пауза в задаче, к которой обратились по id, соблюдается.
    await chat.state_pause()
    events = await run_chat("", continue_step=True, session_id=background)
    check("у паузы приоритет над фоновым шагом",
          any("паузе" in t for t in texts(events, "error")), str(texts(events, "error")[:1]))
    await chat.state_resume()

    # Неизвестная задача — понятная ошибка, а не выполнение в открытой.
    response = await chat.agent_chat(ChatMessage(content="", continue_step=True,
                                                 session_id="s-нет-такой"))
    check("неизвестная задача → 404", getattr(response, "status_code", None) == 404,
          str(getattr(response, "status_code", None)))

    # 3.19 Задачи работают ПАРАЛЛЕЛЬНО (раньше их сериализовала одна блокировка).
    PLAN_STEPS = ["Первый шаг", "Второй шаг", "Третий шаг"]
    await chat.agent_history_clear()
    await run_chat("Задача A")
    await run_chat("ок")                                   # A: шаг 1 → execution step_2
    session_a = chat._current_session()["id"]
    await chat.session_create()
    await run_chat("Задача B")
    await run_chat("ок")                                   # B: шаг 1 → execution step_2
    session_b = chat._current_session()["id"]
    check("две разные задачи подготовлены", session_a != session_b)

    event_a, event_b = asyncio.Event(), asyncio.Event()
    SLOW_QUEUE.extend([event_a, event_b])
    step_a = asyncio.ensure_future(run_chat("", continue_step=True, session_id=session_a))
    await asyncio.sleep(0.05)
    step_b = asyncio.ensure_future(run_chat("", continue_step=True, session_id=session_b))
    await asyncio.sleep(0.05)
    check("обе задачи выполняются ОДНОВРЕМЕННО",
          {session_a, session_b} <= chat._running_sessions,
          str(chat._running_sessions))
    event_b.set()                                          # отпускаем только B
    await asyncio.wait_for(step_b, timeout=5)
    check("вторая задача завершила шаг, не дожидаясь первой",
          session_a in chat._running_sessions and session_b not in chat._running_sessions,
          str(chat._running_sessions))
    state_b = (await chat.state_get())["state"]
    check("шаг второй задачи зафиксирован",
          state_b["stage"] == "execution" and state_b["current_step"] == "step_3",
          f"({state_b['stage']}, {state_b['current_step']})")
    event_a.set()
    await asyncio.wait_for(step_a, timeout=5)
    check("первая задача тоже завершила свой шаг", not chat._running_sessions,
          str(chat._running_sessions))

    # 3.20 Пауза, нажатая во время ПОСТРОЕНИЯ ПЛАНА: применяется сразу после
    #      показа плана (иначе намерение висело, а кнопки были несогласованы).
    global SLOW_PLAN
    PLAN_STEPS = ["Первый шаг", "Второй шаг"]
    await chat.agent_history_clear()
    SLOW_PLAN = asyncio.Event()
    planning = asyncio.ensure_future(run_chat("Задача с долгим планированием"))
    await asyncio.sleep(0.05)
    pending = await chat.state_pause()                    # пауза во время плана
    check("пауза во время планирования принята (pending)",
          pending["state"].get("pending") == "pause", str(pending["state"].get("pending")))
    SLOW_PLAN.set()
    events = await planning
    SLOW_PLAN = None
    state = stage_of(events) or (await chat.state_get())["state"]
    check("план показан и задача СРАЗУ на паузе",
          state["stage"] == "awaiting_user" and state["paused"] is True,
          f"({state['stage']}, paused={state['paused']})")
    check("намерение остановки не осталось висеть", not chat._pending_stops,
          str(chat._pending_stops))
    check("в дебаге сказано, что остановили после плана",
          any("во время построения плана" in t for t in texts(events, "debug")),
          str([t for t in texts(events, "debug") if "план" in t][-2:]))
    # Подтверждение плана на паузе не проходит (нужно «Продолжить»), а после
    # «Продолжить» — работает и прогон идёт.
    try:
        await chat.state_confirm()
        check("подтверждение плана на паузе запрещено", False)
    except Exception as exc:   # HTTPException
        check("подтверждение плана на паузе запрещено",
              getattr(exc, "status_code", None) == 400, str(exc))
    await chat.state_resume()
    confirmed = await chat.state_confirm()
    check("после «Продолжить» план подтверждается",
          confirmed["state"]["stage"] == "execution", confirmed["state"]["stage"])
    events = await run_chat("", continue_step=True)
    state = stage_of(events) or {}
    check("шаги после этого выполняются", state.get("stage") == "execution", str(state.get("stage")))

    # 3.17 Тексты намерений пользователя.
    for text in ("ок", "Ок!", "да", "поехали", "подтверждаю план", "работай автономно"):
        check(f"подтверждение распознано: «{text}»", chat._is_plan_confirmation(text))
    for text in ("ок, но шаг 2 переставь", "сделай иначе — сначала тесты", "почему так?"):
        check(f"правки не считаются подтверждением: «{text}»",
              not chat._is_plan_confirmation(text))
    check("перезапуск распознан", chat._wants_restart("перезапусти задачу"))


def main():
    print("Проверка Task State Machine (без сети и LLM)")
    test_core()
    test_workspace()
    asyncio.get_event_loop().run_until_complete(test_routes())
    print("\nИтог: " + (f"ПРОВАЛЕНО проверок: {len(FAILURES)} → {FAILURES}"
                       if FAILURES else "все проверки пройдены"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
