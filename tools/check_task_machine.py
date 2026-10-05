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
import re
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Изоляция данных: всё пишем во временный каталог ------------------------
_TMP = tempfile.mkdtemp(prefix="tsm-check-")
os.environ["AGENT_WORKSPACE_FILE"] = os.path.join(_TMP, "workspace.json")
os.environ["AGENT_MEMORY_FILE"] = os.path.join(_TMP, "agent_memory.json")
os.environ["AGENT_PROFILES_FILE"] = os.path.join(_TMP, "profiles.json")

from app.ai import client, invariants as invariants_store  # noqa: E402
from app.ai import task_state, workspace as workspace_store  # noqa: E402
from app.routers import chat  # noqa: E402
from app.schemas import (  # noqa: E402
    ChatMessage, InvariantCreate, InvariantPick, InvariantResolve, PlanUpdate,
)

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
# Вердикт содержательной проверки (этап validation):
# {"verdict", "steps": [{"n", "ok", "comment"}], "step", "comment"}.
# None — «модель не ответила»: проверку выполнить НЕ удалось, задача остаётся на
# этапе validation с признаком check_blocked и ждёт решения пользователя.
# Строка — «сырой» ответ модели (например обрезанный лимитом токенов JSON).
REVIEW = {"verdict": "ok", "step": 0, "comment": "результат соответствует плану"}
# Последняя user-часть вызова проверки: по ней видно, что уходит приёмщику
# (исходный запрос + план + решение модели по шагам).
REVIEW_PAYLOAD = []
CALLS = []
# Сами сообщения последнего вызова: по ним проверяем, что в контекст агента
# уходит системный блок инвариантов (а в диалог они не пишутся).
LAST_MESSAGES = []
# «Медленный» ответ модели: шаг «в полёте», пока тест не отпустит событие
# (проверяем мгновенную реакцию «Паузы»/«Отмены» и переключение сессии).
SLOW_ANSWER = None
# Очередь «медленных» ответов: i-й вызов ждёт i-е событие. Нужна для проверки
# ПАРАЛЛЕЛЬНОСТИ: две задачи держим «в полёте» одновременно и отпускаем по одной.
SLOW_QUEUE = []
# Медленное ПОСТРОЕНИЕ ПЛАНА (служебный вызов планировщика): нужен, чтобы
# проверить паузу, нажатую во время планирования.
SLOW_PLAN = None
# Очередь планов: пока непуста, i-й вызов планировщика берёт i-й список шагов
# (нужно код-гейту плана: первый план нарушает правила, второй — нет).
PLAN_QUEUE = []
# Вердикты проверки инвариантов (служебный вызов «Ты проверяешь ИНВАРИАНТЫ»):
# номер пары (строкой) -> вердикт. Пустой словарь — «проверка не удалась»:
# так проверяем, что пары остаются НЕпроверенными, а не «совместимыми».
INVARIANTS_VERDICTS = {"1": {"вердикт": "conflict", "причина": "СУБД разная"}}
# Разбор ЗАПРОСА на соответствие инвариантам (служебный вызов «Ты — арбитр
# инвариантов», до планирования): вердикт + варианты решения. По умолчанию —
# "clear" (нарушений нет), чтобы прочие проверки работали как раньше.
INVARIANTS_ANALYSIS = {"вердикт": "clear", "объяснение": "", "варианты": []}
ANALYSIS_CALLS = 0
# Очередь разборов: пока непуста, i-й вызов арбитра берёт i-й элемент (дальше —
# снова INVARIANTS_ANALYSIS). Нужна, чтобы проверить ПОВТОРНЫЙ запрос вариантов:
# первый разбор даёт мало пригодных вариантов, второй — другие.
ANALYSIS_QUEUE = []
# Вердикты ПРОВЕРКИ ВАРИАНТОВ (служебный вызов «Ты — арбитр инвариантов:
# проверка ВАРИАНТОВ»): номер варианта (строкой) -> вердикт; не указан — берётся
# SUGGESTIONS_DEFAULT. None — «проверка не удалась» (пустой ответ): тогда
# варианты показывать нельзя.
SUGGESTIONS_VERDICTS = {}
SUGGESTIONS_DEFAULT = "clear"
# Очередь вердиктов проверки: пока непуста, i-я проверка берёт i-й словарь.
SUGGESTIONS_QUEUE = []
SUGGESTION_CALLS = 0
# Вердикты КОД-ГЕЙТА ПЛАНА (служебный вызов «Ты — арбитр инвариантов: проверка
# ШАГОВ ПЛАНА»): номер шага (строкой) -> вердикт; не указан — PLAN_DEFAULT.
# None — «проверка не удалась»: тогда план не принимается.
PLAN_VERDICTS = {}
PLAN_DEFAULT = "clear"
# Слова шага, по которым заглушка считает шаг нарушающим (если вердикт не задан
# явно): так проверяется гейт по СМЫСЛУ шага, а не по его номеру.
PLAN_VIOLATION_WORDS = ("kmp", "ios", "мультиплатформ", "обе платформ", "веб")
PLAN_CALLS = 0
# Последние user-части ВСЕХ вызовов: по ним видно, что уходит модели (в CALLS
# лежат только системные префиксы).
LAST_USER_TEXTS = []


def _metrics(prompt=20, completion=10):
    return {"model": "stub", "elapsed_seconds": 0.01, "prompt_tokens": prompt,
            "completion_tokens": completion, "total_tokens": prompt + completion}


async def fake_call_llm_async(*args, **kwargs):
    """Подмена client.call_llm_async: план — JSON, ответ — текст, всё локально."""
    messages = kwargs.get("messages") or []
    system = str(messages[0].get("content") or "") if messages else ""
    CALLS.append({"system": system[:40], "messages": len(messages),
                  "user_text": kwargs.get("user_text")})
    if messages:
        LAST_USER_TEXTS.append(str(messages[-1].get("content") or ""))
    if system.startswith("Ты — планировщик"):
        LAST_MESSAGES.clear()
        LAST_MESSAGES.extend(messages)
    if system.startswith("Ты — планировщик"):
        if SLOW_PLAN is not None:
            await asyncio.wait_for(SLOW_PLAN.wait(), timeout=10)
        preset = PLAN_QUEUE.pop(0) if PLAN_QUEUE else PLAN_STEPS
        if not preset:
            return "", _metrics()
        return json.dumps({"steps": list(preset)}, ensure_ascii=False), _metrics(30, 15)
    if system.startswith("Ты — приёмщик"):
        REVIEW_PAYLOAD.clear()
        if messages:
            REVIEW_PAYLOAD.append(str(messages[-1].get("content") or ""))
        if REVIEW is None:
            return "", _metrics()
        if isinstance(REVIEW, str):
            # «Сырой» ответ: так проверяется обрезанный лимитом токенов JSON.
            return REVIEW, _metrics(40, 8)
        return json.dumps(REVIEW, ensure_ascii=False), _metrics(40, 8)
    if system.startswith("Ты — арбитр инвариантов: проверка ШАГОВ ПЛАНА"):
        # КОД-ГЕЙТ ПЛАНА: префикс проверяем ДО общего «Ты — арбитр инвариантов».
        global PLAN_CALLS
        PLAN_CALLS += 1
        if PLAN_VERDICTS is None:
            return "", _metrics()   # «проверка не удалась» — план не принимается
        user = str(messages[-1].get("content") or "") if messages else ""
        tail = user.split("ШАГИ ПЛАНА (проверь")[-1]
        steps = re.findall(r"^\d+\) (.*)$", tail, re.M) or ["?"]
        verdicts = {}
        for number, step in enumerate(steps, 1):
            preset = PLAN_VERDICTS.get(str(number)) if PLAN_VERDICTS else None
            if preset is not None:
                verdicts[str(number)] = dict(preset)
                continue
            # Без явного вердикта — по СОДЕРЖАНИЮ шага: запрещённые технологии
            # в тексте шага = нарушение (как это делает живая модель).
            bad = any(word in step.lower() for word in PLAN_VIOLATION_WORDS)
            verdicts[str(number)] = {"вердикт": "violation" if bad else PLAN_DEFAULT,
                                     "причина": "запрещённая технология" if bad else ""}
        return json.dumps(verdicts, ensure_ascii=False), _metrics(30, 12)
    if system.startswith("Ты — арбитр инвариантов: проверка ВАРИАНТОВ"):
        # ПРОВЕРКА вариантов-альтернатив: этот префикс проверяем ДО общего
        # «Ты — арбитр инвариантов», иначе проверка получила бы вердикт запроса.
        global SUGGESTION_CALLS
        SUGGESTION_CALLS += 1
        preset = SUGGESTIONS_QUEUE.pop(0) if SUGGESTIONS_QUEUE else SUGGESTIONS_VERDICTS
        if preset is None:
            return "", _metrics()   # «проверка не удалась»
        user = str(messages[-1].get("content") or "") if messages else ""
        count = len(re.findall(r"^\d+\) ", user.split("ВАРИАНТЫ (проверь")[-1], re.M)) or 1
        verdicts = {}
        for number in range(1, count + 1):
            verdicts[str(number)] = dict(
                preset.get(str(number))
                or {"вердикт": SUGGESTIONS_DEFAULT, "причина": ""})
        return json.dumps(verdicts, ensure_ascii=False), _metrics(25, 10)
    if system.startswith("Ты — арбитр инвариантов"):
        global ANALYSIS_CALLS
        ANALYSIS_CALLS += 1
        preset = ANALYSIS_QUEUE.pop(0) if ANALYSIS_QUEUE else INVARIANTS_ANALYSIS
        if preset is None:
            return "", _metrics()   # «разбор не удался»
        return json.dumps(preset, ensure_ascii=False), _metrics(35, 20)
    if system.startswith("Ты проверяешь ИНВАРИАНТЫ"):
        if not INVARIANTS_VERDICTS:
            return "", _metrics()   # «проверка не удалась»
        # Пар в запросе может быть больше, чем заготовленных вердиктов: считаем
        # их по тексту запроса (строки «N) Инвариант проекта: ...») и отвечаем на
        # все, иначе часть пар осталась бы «не проверенной».
        user = str(messages[-1].get("content") or "") if messages else ""
        count = len(re.findall(r"^\d+\) Инвариант проекта:", user, re.M)) or 1
        verdicts = {}
        for number in range(1, count + 1):
            verdicts[str(number)] = dict(INVARIANTS_VERDICTS.get(str(number))
                                         or {"вердикт": "clear", "причина": ""})
        return json.dumps(verdicts, ensure_ascii=False), _metrics(20, 6)
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

    # ПРЕДОХРАНИТЕЛЬ ОТ ПОТЕРИ ИСТОРИИ. Файл workspace — единственное место, где
    # живут задачи и диалоги пользователя (не в git, копий нет), а запись может
    # «похудеть» сразу в разы — например, если битый файл прочитан как пустой и
    # пустое состояние тут же записано поверх. Резкое уменьшение обязано оставить
    # копию прежнего состояния рядом.
    guard_path = os.path.join(_TMP, "ws-guard.json")
    fat = workspace_store.normalize_workspace({"tasks": [
        {"id": "t-fat%d" % index, "name": "Проект %d" % index,
         "sessions": [{"id": "s-fat%d" % index,
                       # Формат реплики диалога — {"role", "content"}:
                       # «text» здесь не сохраняется (см. _clean_messages).
                       "dialog": {"messages": [{"role": "user",
                                                "content": "длинная реплика " * 100}] * 30}}]}
        for index in range(6)]})
    workspace_store.save_workspace(fat, guard_path)
    fat_size = os.path.getsize(guard_path)
    if os.path.exists(guard_path + ".bak"):
        os.unlink(guard_path + ".bak")
    workspace_store.save_workspace(
        workspace_store.normalize_workspace({"tasks": [{"id": "t-one", "name": "Одна"}]}),
        guard_path)
    check("резкое уменьшение записи сохраняет копию прежнего файла",
          os.path.isfile(guard_path + ".bak")
          and os.path.getsize(guard_path + ".bak") == fat_size,
          "копия: %s" % os.path.isfile(guard_path + ".bak"))
    recovered = workspace_store.load_workspace(guard_path + ".bak")
    check("в копии лежит ПРЕЖНЕЕ состояние (задачи и их диалоги)",
          len(recovered["tasks"]) == 6
          and len(recovered["tasks"][0]["sessions"][0]["dialog"]["messages"]) == 30,
          "задач в копии: %d" % len(recovered["tasks"]))
    check("копия — валидный workspace (её можно просто вернуть на место)",
          recovered["tasks"][0]["name"] == "Проект 0")
    # Обычная правка копий не плодит: файл небольшой и меняется не в разы.
    if os.path.exists(guard_path + ".bak"):
        os.unlink(guard_path + ".bak")
    workspace_store.save_workspace(
        workspace_store.normalize_workspace({"tasks": [{"id": "t-one", "name": "Одна"},
                                                       {"id": "t-two", "name": "Две"}]}),
        guard_path)
    check("небольшая правка копию не создаёт", not os.path.isfile(guard_path + ".bak"))
    check("маленький файл копий не плодит (терять нечего)",
          not os.path.isfile(os.path.join(_TMP, "ws-roundtrip.json") + ".bak"))


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
    # ГИБРИД UX: ответ ПРОМЕЖУТОЧНОГО шага не показывается репликой в чате
    # (ход работы виден в журнале), но ответ обязан остаться в ПАМЯТИ диалога:
    # по нему работает следующий шаг и проверка результата.
    check("ответ модели получен и сохранён в память диалога",
          any("Ответ модели по текущему шагу." in str(m.get("content") or "")
              for m in chat._current_session()["dialog"]["messages"]),
          str(texts(events, "bot"))[:150])
    check("вместо ответа промежуточного шага — строка в журнале",
          any("в чате не показываю" in t for t in texts(events, "debug")),
          str(texts(events, "debug"))[-200:])
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

    # Проверка НЕДОСТУПНА: задача готовой НЕ объявляется (ни «принято», ни
    # «не принято»), стоит на этапе validation с признаком check_blocked и ждёт
    # решения пользователя: повторить проверку или принять результат вручную.
    REVIEW = None
    await chat.agent_history_clear()
    await run_chat("Сделай отчёт")
    events = await run_chat("ок")
    state = stage_of(events)
    check("недоступная проверка НЕ завершает задачу",
          state and state["stage"] == "validation", state and state["stage"])
    check("в снимке выставлен признак «проверка не выполнена»",
          state and state["check_blocked"] is True and state["can_accept"] is True,
          str({k: (state or {}).get(k) for k in ("check_blocked", "can_accept")}))
    check("о сбое проверки сказано в дебаге",
          any("содержательная проверка не получена" in t for t in texts(events, "debug")))
    check("пользователю предложено повторить проверку или принять вручную",
          any("Принять вручную" in t for t in texts(events, "error")),
          str(texts(events, "error")))
    saved = (await chat.state_get())["state"]
    check("признак переживает перезагрузку (сохранён в состоянии)",
          saved["check_blocked"] is True and saved["stage"] == "validation", str(saved["stage"]))
    check("доработки сбоем проверки НЕ тратятся",
          not saved["redo_count"], str(saved["redo_count"]))
    # «▶ повторить проверку»: признак снимается и проверка выполняется заново.
    REVIEW = {"verdict": "ok", "step": 0, "comment": "теперь всё закрыто"}
    events = await run_chat("", continue_step=True)
    state = stage_of(events)
    check("повторная проверка снимает признак и завершает задачу",
          state and state["stage"] == "done" and state["check_blocked"] is False,
          f"({state and state['stage']}, blocked={state and state['check_blocked']})")
    # Принимать вручную нечего, когда задача закрыта проверкой.
    try:
        await chat.state_accept()
        refused = False
    except Exception as exc:                      # HTTPException
        refused = getattr(exc, "status_code", None) == 400
    check("принимать вручную нечего — маршрут отвечает 400", refused)
    # Сбой проверки снова: задача ждёт решения, и его можно принять вручную.
    REVIEW = None
    await chat.agent_history_clear()
    await run_chat("Ещё отчёт")
    await run_chat("ок")
    accepted = await chat.state_accept()
    check("«Принять вручную» завершает задачу",
          accepted["state"]["stage"] == "done" and accepted["state"]["check_blocked"] is False,
          str(accepted["state"]["stage"]))
    check("в истории перехода сказано, что принято вручную",
          any("принят пользователем вручную" in r["reason"]
              for r in accepted["state"]["history"]),
          str([r["reason"] for r in accepted["state"]["history"]][-2:]))
    REVIEW = {"verdict": "ok", "step": 0, "comment": "ок"}

    # Разбор ПО КАЖДОМУ шагу: несоответствие отдельного шага возвращает задачу на
    # ВЫПОЛНЕНИЕ, начиная с ПЕРВОГО непринятого шага — даже если общий verdict «ok».
    PLAN_QUEUE.append(["Собрать данные", "Посчитать итоги", "Написать отчёт"])
    REVIEW = {
        "verdict": "ok", "step": 0, "comment": "в целом похоже на правду",
        "steps": [
            {"n": 1, "ok": True, "comment": "данные собраны"},
            {"n": 2, "ok": False, "comment": "итоги не посчитаны"},
            {"n": 3, "ok": True, "comment": "отчёт написан"},
        ],
    }
    await chat.agent_history_clear()
    await run_chat("Сделай отчёт по продажам")        # план из трёх шагов
    await run_chat("ок")                              # подтверждение + шаг 1
    await run_chat("", continue_step=True)            # шаг 2
    events = await run_chat("", continue_step=True)   # шаг 3 → validation → проверка
    state = stage_of(events)
    check("непринятый шаг возвращает задачу в execution",
          state and state["stage"] == "execution" and state["redo_count"] == 1,
          f"({state and state['stage']}, redo={state and state['redo_count']})")
    check("возврат идёт на ПЕРВЫЙ непринятый шаг, а не на последний",
          state and state["step_number"] == 2 and state["steps_total"] == 3,
          f"(шаг {state and state['step_number']} из {state and state['steps_total']})")
    check("в причине назван непринятый шаг и объяснение модели",
          any("не приняты — шаг 2: итоги не посчитаны" in r["reason"]
              for r in state["history"]),
          str([r["reason"] for r in state["history"]][-1:]))
    check("разбор по шагам показан пользователю",
          any("шаг 2: НЕ принят" in t for t in texts(events, "debug")),
          str([t for t in texts(events, "debug") if "разбор по шагам" in t]))
    # Что именно уходит приёмщику: изначальная задача + план + решение модели.
    payload = REVIEW_PAYLOAD[0] if REVIEW_PAYLOAD else ""
    check("в проверку уходит ИЗНАЧАЛЬНАЯ задача пользователя",
          "Исходный запрос пользователя:" in payload
          and "Сделай отчёт по продажам" in payload,
          payload[:200])
    check("в проверку уходит план модели",
          "План работы (шаги):" in payload and "Посчитать итоги" in payload)
    check("в проверку уходит решение модели ПО ШАГАМ (а не хвост диалога)",
          "Решение модели (ответы по шагам" in payload
          and "Шаг 1 (Собрать данные)" in payload
          and "Шаг 3 (Написать отчёт)" in payload,
          payload[-400:])
    # ОБРЕЗАННЫЙ ответ проверки (лимит токенов): вердикт и целые записи шагов всё
    # равно учитываются — сбой разбора JSON не должен выглядеть как «проверку
    # выполнить не удалось» и не должен превращаться в accepted-по-умолчанию.
    REVIEW = ('{"verdict": "ok", "steps": [{"n": 1, "ok": true, "comment": "ок"}, '
              '{"n": 2, "ok": false, "comment": "итоги не посч')
    events = await run_chat("", continue_step=True)   # доработка шага 2
    events = await run_chat("", continue_step=True)   # шаг 3 → снова проверка
    state = stage_of(events)
    check("обрезанный ответ проверки разобран, а не потерян",
          state and state["stage"] == "execution" and state["redo_count"] == 2,
          f"({state and state['stage']}, redo={state and state['redo_count']})")
    check("возврат и по обрезанному ответу — на непринятый шаг",
          state and state["step_number"] == 2
          and any("не приняты — шаг 2" in r["reason"] for r in state["history"]),
          f"(шаг {state and state['step_number']})")
    REVIEW = {"verdict": "ok", "step": 0, "comment": "ок"}

    # 3.13 Отмена задачи кнопкой «Отменить»: cancelled — терминальный этап.
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
    # force_plan: раздел проверяет ЖУРНАЛ И ШАГИ пути с планом (показанный план,
    # пометка source=machine, заголовок по запросу). Сам запрос — просьба ответа
    # («дай рецепт»), и гейт отправил бы его прямым ответом: см. `_plan_needed`.
    await run_chat("Дай рецепт борща", force_plan=True)
    history = await chat.agent_history()
    log = history.get("log") or []
    kinds = [item["kind"] for item in log]
    check("журнал чата ведётся", bool(log), str(kinds))
    check("в журнале есть запрос пользователя",
          any(i["kind"] == "user" and "рецепт борща" in i["text"] for i in log), str(log[:3]))
    check("в журнале есть показанный план",
          any(i["kind"] == "assistant" and "План задачи" in i["text"] for i in log))
    check("в журнале есть debug-строки агента", "debug" in kinds)
    # Память диалога: маршрут отдаёт messages только для диалогов без журнала
    # чата (иначе ответ дублировал бы одно и то же), поэтому смотрим её в сессии.
    messages = chat._current_session()["dialog"]["messages"]
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
    machine = [m for m in chat._current_session()["dialog"]["messages"]
               if m.get("source") == "machine"]
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
    # force_plan: строки вида «Задача …» — искусственные (в них нет ни объекта
    # результата, ни действий), а раздел проверяет ШАГИ И ПАУЗЫ пути с планом;
    # гейт «ответ или план» (chat._plan_needed) отправил бы их прямым ответом.
    await run_chat("Задача для проверки остановки", force_plan=True)   # план из 3 шагов
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
    # Ответ промежуточного шага в журнал чата репликой не пишется (показывается
    # ход работы), но он обязан остаться в памяти диалога — иначе «Пауза» после
    # шага потеряла бы результат шага.
    check("ответ шага при этом не потерян (остался в памяти диалога)",
          any("Ответ модели" in str(m.get("content") or "")
              for m in chat._current_session()["dialog"]["messages"]),
          str([item["kind"] for item in (await chat.agent_history())["log"]][-3:]))
    await chat.state_resume()

    # «Пауза» целится в ВЫПОЛНЯЕМУЮ задачу, даже если открыта другая: иначе
    # намерение «залипало» бы в очереди чужой задачи и срабатывало позже.
    # Свежая задача с планом из 3 шагов: шаг 1 выполнен, шаг 2 будет «в полёте».
    # Именно три шага: у ПОСЛЕДНЕГО шага своя семантика паузы (перед проверкой).
    PLAN_STEPS = ["Первый шаг", "Второй шаг", "Третий шаг"]
    await chat.agent_history_clear()
    await run_chat("Задача для адресной паузы", force_plan=True)
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
    await run_chat("Задача из двух шагов", force_plan=True)
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
    await run_chat("Задача для отмены на ходу", force_plan=True)
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
    await run_chat("Фоновая задача", force_plan=True)
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
    # Ответ промежуточного шага хранится в ПАМЯТИ своей задачи (в журнале — ход
    # работы): проверяем, что работа легла именно в СВОЮ задачу.
    check("ответ записан в память СВОЕЙ задачи",
          any("Ответ модели" in str(m.get("content") or "")
              for m in chat._current_session()["dialog"]["messages"]),
          str([i["kind"] for i in history_bg["log"]][-3:]))
    check("ход работы своей задачи виден в её журнале",
          any("выполнен" in str(i.get("text") or "") for i in history_bg["log"]),
          str([i["kind"] for i in history_bg["log"]][-3:]))
    check("шаг помечен как реплика автомата",
          any(m.get("source") == "machine"
              for m in chat._current_session()["dialog"]["messages"]))

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
    await run_chat("Задача A", force_plan=True)
    await run_chat("ок")                                   # A: шаг 1 → execution step_2
    session_a = chat._current_session()["id"]
    await chat.session_create()
    await run_chat("Задача B", force_plan=True)
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


# ---------------------------------------------------------------------------
# 4. Инварианты: правила, которые агент не имеет права нарушить
# ---------------------------------------------------------------------------
def _invariants_of(container):
    return workspace_store.invariants(container)


def _inv_texts(items):
    return [entry["text"] for entry in items]


async def test_invariants():
    print("\n[4] Инварианты: хранение отдельно от диалога, контекст, противоречие")
    global PLAN_STEPS, INVARIANTS_VERDICTS, ANSWER, INVARIANTS_ANALYSIS
    PLAN_STEPS = ["Собрать данные"]
    ANSWER = "Ответ модели."
    INVARIANTS_VERDICTS = {"1": {"вердикт": "conflict", "причина": "СУБД разная"}}

    await chat.task_create(chat.TaskCreate(name="Проект с инвариантами"))
    task = chat._current_task()
    await chat.session_create()
    session = chat._current_session()
    # Чистим правила «прошлых» проверок этого файла (задача новая, но диалог
    # мог остаться от прежних шагов).
    task["invariants"] = []
    session["dialog"]["invariants"] = []
    session["dialog"]["conflicts"] = []
    session["dialog"]["unchecked"] = []

    # 4.1 Инвариант проекта: хранится в проекте, в диалоге его нет.
    view = await chat.invariant_create(InvariantCreate(text="Только PostgreSQL", scope="project"))
    check("инвариант проекта сохранён", _inv_texts(_invariants_of(task)) == ["Только PostgreSQL"])
    check("инвариант проекта НЕ попал в диалог", _invariants_of(session["dialog"]) == [])
    check("в диалог (messages) инвариант не пишется", session["dialog"]["messages"] == [])
    check("вызова LLM на проверку нет, пока правила только с одной стороны",
          not any(c["system"].startswith("Ты проверяешь ИНВАРИАНТЫ") for c in CALLS))

    # 4.2 Инварианты уходят агенту системным блоком (план — тоже контекст агента).
    await run_chat("Сделай API")
    # У служебного вызова плана контекст-блоки идут в user-части (system там —
    # сам промпт планировщика), у ответа агента — отдельным system-сообщением.
    planner_text = "\n\n".join(m["content"] for m in LAST_MESSAGES
                               if m.get("role") == "user")
    # Блок инвариантов идёт ПЕРВЫМ системным блоком (перед ним может быть только
    # профиль пользователя, если он заполнен), поэтому ищем его в тексте контекста.
    block = planner_text[planner_text.index("ИНВАРИАНТЫ"):] if "ИНВАРИАНТЫ" in planner_text else ""
    check("блок инвариантов ушёл в контекст агента", bool(block))
    check("в блоке есть правило проекта", "Только PostgreSQL" in block)
    check("в блоке есть требование не нарушать правила", "нарушать нельзя" in block)
    check("в блоке есть отказ от нарушающих решений", "не предлагай решений" in block)
    check("в блоке сказано, что правило проекта главнее",
          "правило ПРОЕКТА всегда главнее" in block)
    check("в блоке сказано, что противоречащее правило задачи не действует",
          "НЕ ДЕЙСТВУЕТ" in block and "не применяй" in block)

    # 4.3 Инвариант задачи: пишется БЕЗ обращения к модели (правило — это данные).
    calls_before = len(CALLS)
    view = await chat.invariant_create(InvariantCreate(text="Только MongoDB", scope="task"))
    check("инвариант задачи сохранён (в диалоге, не в проекте)",
          _inv_texts(_invariants_of(session["dialog"])) == ["Только MongoDB"]
          and _inv_texts(_invariants_of(task)) == ["Только PostgreSQL"])
    check("при записи правил обращений к модели НЕТ",
          len(CALLS) == calls_before,
          f"вызовов LLM при записи: {len(CALLS) - calls_before}")
    check("проверок пар в снимке модалки нет",
          "checks" not in view and view["has_conflict"] is False, str(sorted(view))[:120])

    # 4.4 Запрос, нарушающий правило, разбирается в диалоге (до планирования).
    # Диалог начинаем с чистого листа: в 4.2 задача уже получила план, а проверяем
    # именно то, что нарушающий запрос НЕ доводит дело до плана.
    session = chat._current_session()
    session["dialog"] = workspace_store.empty_dialog(session["id"])
    task = chat._current_task()
    calls_before = len(CALLS)
    INVARIANTS_ANALYSIS = {
        "вердикт": "violation",
        "объяснение": "Веб-приложение нарушает инвариант: разрешён только Kotlin.",
        "варианты": [{"заголовок": "Нативное Android-приложение",
                      "пояснение": "В стеке", "запрос": "Сделай Android-приложение"},
                     {"заголовок": "План экранов Android на Compose",
                      "пояснение": "Только Android, MVVM + Compose",
                      "запрос": "Спланируй экраны Android-приложения на Compose"}],
    }
    events = await run_chat("сделай веб-приложение")
    check("разбор запроса вызван в диалоге",
          len(CALLS) > calls_before
          and any(c["system"].startswith("Ты — арбитр инвариантов") for c in CALLS[calls_before:]))
    blocked = [e["analysis"] for e in events if e.get("type") == "suggestions"]
    check("отказ с вариантами показан", bool(blocked))
    check("план при нарушении не строится",
          not (stage_of(events) or {}).get("steps"),
          str((stage_of(events) or {}).get("steps")))

    # 4.5 КОНФЛИКТ правила задачи с правилом ПРОЕКТА: приоритет всегда у проекта.
    #     Агент поступает так же, как с запрещённым запросом: отказывается и даёт
    #     альтернативы. Выбора «какое правило главнее» пользователю не даём.
    session["dialog"]["analysis"] = None   # свежий вердикт (кэш не переиспользуем)
    INVARIANTS_ANALYSIS = {
        "вердикт": "violation",
        "объяснение": "Правило проекта требует нативную платформу Android (Kotlin), "
                      "а правило задачи просит веб-сайт — действует правило проекта.",
        "варианты": [
            {"заголовок": "Нативное Android-приложение",
             "пояснение": "Укладывается в стек проекта: Kotlin + Compose.",
             "запрос": "Сделай нативное Android-приложение погоды на Kotlin"},
            {"заголовок": "Экраны Android-приложения",
             "пояснение": "Только Android, MVVM + Compose.",
             "запрос": "Спланируй экраны Android-приложения на Compose"},
        ],
    }
    events = await run_chat("сделай веб-сайт погоды")
    blocked = [e["analysis"] for e in events if e.get("type") == "suggestions"]
    check("конфликт правил показан как нарушение с альтернативами", bool(blocked))
    analysis = blocked[0] if blocked else {}
    check("вердикт — нарушение (не «выбор приоритета»)",
          analysis.get("verdict") == "violation", str(analysis.get("verdict")))
    check("выбора «главнее проект/задача» больше не предлагается",
          not analysis.get("resolutions") and "resolution" not in analysis,
          str(sorted(analysis.keys())))
    check("предложены альтернативы, ни одна не нарушает правила",
          len(analysis.get("suggestions") or []) >= 2
          and all("веб" not in (item["send"] or "").lower()
                  for item in analysis.get("suggestions") or []),
          str(analysis.get("suggestions")))
    check("в объяснении сказано про приоритет правила проекта",
          "проекта" in str(analysis.get("explanation") or "").lower(),
          str(analysis.get("explanation"))[:120])
    check("план при конфликте правил не строится",
          not any("План задачи" in text for text in texts(events, "bot")))

    # 4.6 Клик по альтернативе: сервер отдаёт её текст, страница отправляет запрос.
    picked = await chat.invariant_choose(InvariantPick(index=0))
    check("выбор альтернативы — действие «отправить»", picked.get("action") == "send",
          str(picked.get("action")))
    check("текст альтернативы пришёл с сервера",
          "Kotlin" in (picked.get("text") or ""), str(picked.get("text"))[:60])
    check("альтернатива не нарушает правила проекта",
          "веб" not in (picked.get("text") or "").lower())
    check("«главнее задача» больше не принимается",
          await _raises(lambda: chat.invariant_resolve(
              InvariantResolve(key="any", winner="task")), status=409))

    # 4.7 Запрос по альтернативе: правила соблюдены — план строится.
    INVARIANTS_ANALYSIS = {"вердикт": "clear", "объяснение": "", "варианты": []}
    events = await run_chat(picked["text"])
    check("по альтернативе план строится",
          any("План задачи" in text for text in texts(events, "bot")),
          str(texts(events, "bot"))[:140])
    check("шаги плана в состоянии",
          [s["text"] for s in (stage_of(events) or {}).get("steps", [])] == PLAN_STEPS)

    # 4.8 Удаление правила: решения по его парам больше не хранятся.
    project_id = chat._invariants_view(task, session)["project"][0]["id"]
    view = await chat.invariant_delete("project", project_id)
    check("правило проекта удалено", _inv_texts(_invariants_of(task)) == [])
    check("решения по удалённому правилу не остаются",
          view["exceptions"] == [], str(view["exceptions"]))
    check("повторный разбор не показывает противоречий",
          view["has_conflict"] is False)

    # 4.9 Профильная изоляция: чужие правила не видны (новый профиль — свои данные).
    task2 = await chat.task_create(chat.TaskCreate(name="Другой проект"))
    session2 = chat._current_session()
    check("у нового проекта своих правил нет",
          chat._invariants_view(chat._current_task(), session2)["counts"]["project"] == 0)
    check("снимок инвариантов без проекта пуст",
          chat._invariants_view(None, None)["counts"]
          == {"project": 0, "task": 0, "conflict": 0, "exceptions": 0},
          str(chat._invariants_view(None, None)["counts"]))


# ---------------------------------------------------------------------------
# 5. Разбор запроса на соответствие инвариантам (ДО планирования)
# ---------------------------------------------------------------------------
def _suggestions_events(events):
    return [e["analysis"] for e in events if e.get("type") == "suggestions"]


async def test_request_compliance():
    print("\n[5] Разбор запроса: отказ при нарушении и варианты решения")
    global PLAN_STEPS, INVARIANTS_VERDICTS, INVARIANTS_ANALYSIS, ANALYSIS_CALLS
    PLAN_STEPS = ["Собрать требования", "Написать код"]
    INVARIANTS_VERDICTS = {"1": {"вердикт": "clear", "причина": ""}}
    INVARIANTS_ANALYSIS = {"вердикт": "clear", "объяснение": "", "варианты": []}

    await chat.task_create(chat.TaskCreate(name="Проект погоды"))
    await chat.session_create()          # диалог нужен для журнала и разбора
    task = chat._current_task()
    session = chat._current_session()
    task["invariants"] = []
    session["dialog"]["invariants"] = []
    session["dialog"]["conflicts"] = []
    session["dialog"]["analysis"] = None
    await chat.invariant_create(InvariantCreate(
        text="Только Kotlin под Android: никакого веба и мультиплатформы", scope="project"))

    # 5.1 Запрос нарушает инвариант: агент ОТКАЗЫВАЕТСЯ и предлагает варианты.
    ANALYSIS_CALLS = 0
    INVARIANTS_ANALYSIS = {
        "вердикт": "violation",
        "объяснение": "Веб-приложение нарушает инвариант: разрешён только Kotlin/Android.",
        "варианты": [
            {"заголовок": "Нативное Android-приложение на Kotlin",
             "пояснение": "Укладывается в стек: Kotlin + Compose.",
             "запрос": "Сделай нативное Android-приложение погоды на Kotlin и Compose"},
            {"заголовок": "Экраны Android-приложения",
             "пояснение": "Только Android, MVVM + Compose.",
             "запрос": "Спланируй экраны Android-приложения на Compose"},
        ],
    }
    plan_calls_before = len(CALLS)
    events = await run_chat("нужно веб-приложение погоды, открывается в браузере",
                           force_plan=True)
    analyses = _suggestions_events(events)
    check("разбор запроса вызван ДО планирования", ANALYSIS_CALLS == 1,
          f"вызовов разбора: {ANALYSIS_CALLS}")
    check("событие с вариантами отправлено", len(analyses) == 1)
    analysis = analyses[0] if analyses else {}
    check("вердикт — нарушение требования инварианта",
          analysis.get("verdict") == "violation", str(analysis.get("verdict")))
    check("предложено не меньше двух вариантов",
          len(analysis.get("suggestions") or []) >= 2,
          f"вариантов: {len(analysis.get('suggestions') or [])}")
    check("у каждого варианта есть заголовок и текст запроса",
          all(item["title"] and item["send"] for item in analysis.get("suggestions") or []))
    check("среди вариантов нет нарушающего правила (веб-приложения)",
          not any("веб" in (item["send"] or "").lower()
                  for item in analysis.get("suggestions") or []))
    check("объяснение, почему требование невозможно, показано",
          "нарушает инвариант" in str(analysis.get("explanation") or ""))
    check("план НЕ построен — шагов нет",
          not (stage_of(events) or {}).get("steps"),
          str((stage_of(events) or {}).get("steps")))
    check("планировщик не вызывался (запрос отсечён до планирования)",
          not any(c["system"].startswith("Ты — планировщик") for c in CALLS[plan_calls_before:]),
          str([c["system"][:20] for c in CALLS[plan_calls_before:]]))
    check("задача ждёт решения пользователя",
          (stage_of(events) or {}).get("stage") == "awaiting_user",
          str((stage_of(events) or {}).get("stage")))
    check("отказ виден пользователю (текст сообщения с вариантами)",
          any("нарушает инвариант" in t for t in texts(events, "suggestions")),
          str(texts(events, "suggestions"))[:120])
    check("текст отказа сохранён в журнале чата",
          any("нарушает инвариант" in item["text"] for item in
              (await chat.agent_history())["log"]))

    # 5.2 Варианты и объяснение переживают переключение задачи (журнал чата).
    history = await chat.agent_history()
    logged = [item for item in history["log"] if item.get("kind") == "suggestions"]
    check("варианты сохранены в журнале чата", len(logged) == 1)
    check("в журнале есть сами варианты и объяснение",
          len(logged[0].get("analysis", {}).get("suggestions") or []) >= 2
          and "нарушает инвариант" in logged[0]["text"])

    # 5.3 Повторный разбор того же запроса не тратит вызов LLM (кэш по подписи).
    ANALYSIS_CALLS = 0
    events = await run_chat("нужно веб-приложение погоды, открывается в браузере",
                           force_plan=True)
    check("тот же запрос при тех же правилах не разбирается повторно",
          ANALYSIS_CALLS == 0, f"вызовов разбора: {ANALYSIS_CALLS}")
    check("отказ повторяется из кэша", bool(_suggestions_events(events)))

    # 5.4 Клик по варианту-запросу: сервер отдаёт текст варианта (не фронт).
    picked = await chat.invariant_choose(InvariantPick(index=0))
    check("выбор варианта — действие «отправить запрос»",
          picked.get("action") == "send", str(picked.get("action")))
    check("текст варианта пришёл с сервера",
          "Kotlin" in (picked.get("text") or ""), str(picked.get("text"))[:60])
    check("варианты с фронта не подменяются (номер вне списка — 404)",
          await _raises(lambda: chat.invariant_choose(InvariantPick(index=9))))

    # 5.5 Пользователь отправил предложенный вариант: он УЖЕ проверен по правилам,
    #     поэтому повторного разбора нет (иначе выбор варианта снова упирался бы в
    #     отказ) — агент сразу строит план.
    INVARIANTS_ANALYSIS = {"вердикт": "clear", "объяснение": "", "варианты": []}
    ANALYSIS_CALLS = 0
    events = await run_chat(picked["text"])
    check("проверенный вариант разбора НЕ требует",
          ANALYSIS_CALLS == 0, f"вызовов разбора: {ANALYSIS_CALLS}")
    check("нарушений нет — отказа не было",
          not _suggestions_events(events))
    check("план построен по совместимому запросу",
          any("План задачи" in t for t in texts(events, "bot")),
          str(texts(events, "bot"))[:120])
    check("шаги плана в состоянии",
          [s["text"] for s in (stage_of(events) or {}).get("steps", [])] == PLAN_STEPS)
    check("после ответа задача ждёт подтверждения плана",
          (stage_of(events) or {}).get("stage") == "awaiting_user")
    # 5.5б Новый (не из вариантов) запрос по-прежнему разбирается моделью.
    session["dialog"]["analysis"] = None
    ANALYSIS_CALLS = 0
    events = await run_chat("теперь опиши тестирование")
    check("новый запрос разбирается заново",
          ANALYSIS_CALLS == 1, f"вызовов разбора: {ANALYSIS_CALLS}")

    # 5.6 Конфликт правила задачи с правилом ПРОЕКТА: приоритет у проекта,
    #     агент отказывается и даёт альтернативы (как при запрещённом запросе).
    await chat.invariant_create(InvariantCreate(
        text="Только нативная платформа Android (Kotlin)", scope="project"))
    await chat.invariant_create(InvariantCreate(
        text="Разрешить веб-приложение для этой задачи", scope="task"))
    view_rules = chat._invariants_view(chat._current_task(), chat._current_session())
    project_texts = [item["text"] for item in view_rules["project"]]
    task_texts = [item["text"] for item in view_rules["task"]]
    check("правила обеих областей на месте (есть что сравнивать)",
          any("Android" in text for text in project_texts)
          and any("веб" in text for text in task_texts),
          f"проект: {project_texts}, задача: {task_texts}")
    INVARIANTS_ANALYSIS = {
        "вердикт": "violation",
        "объяснение": "Правило проекта запрещает веб и мультиплатформу, а правило "
                      "задачи просит веб — действует правило проекта.",
        "варианты": [
            {"заголовок": "Нативное Android-приложение",
             "пояснение": "Укладывается в правило проекта.", "запрос": "Сделай Android-приложение на Kotlin"},
            {"заголовок": "План экранов Android",
             "пояснение": "Только Android, MVVM.", "запрос": "Спланируй экраны Android-приложения"},
        ],
    }
    chat._current_session()["dialog"]["analysis"] = None   # свежий вердикт
    events = await run_chat("сделай теперь веб-версию")
    analyses = _suggestions_events(events)
    check("конфликт правил показан вариантами-альтернативами", bool(analyses))
    analysis = analyses[0] if analyses else {}
    check("вердикт — нарушение правила проекта, а не «выбор приоритета»",
          analysis.get("verdict") == "violation", str(analysis.get("verdict")))
    check("вариантов-альтернатив не меньше двух",
          len(analysis.get("suggestions") or []) >= 2, str(analysis.get("suggestions")))
    check("ни один вариант не нарушает правило проекта (нет веба)",
          all("веб" not in (item.get("send") or "").lower()
              for item in analysis.get("suggestions") or []))
    check("при конфликте план не строится",
          not any("План задачи" in text for text in texts(events, "bot")),
          str(texts(events, "bot"))[:120])

    # 5.7 Альтернатива уходит обычным запросом и правила не нарушает.
    picked = await chat.invariant_choose(InvariantPick(index=0))
    check("выбор альтернативы — действие «send»", picked.get("action") == "send")
    INVARIANTS_ANALYSIS = {"вердикт": "clear", "объяснение": "", "варианты": []}
    events = await run_chat(picked["text"])
    check("по альтернативе план строится",
          any("План задачи" in text for text in texts(events, "bot")),
          str(texts(events, "bot"))[:140])

    # 5.8 Сбой разбора не выдумывает нарушение: агент работает как раньше.
    INVARIANTS_ANALYSIS = None
    chat._current_session()["dialog"]["analysis"] = None
    events = await run_chat("сделай что-нибудь по погоде")
    check("без вердикта отказа нет", not _suggestions_events(events))
    check("без вердикта план строится",
          any("План задачи" in t for t in texts(events, "bot")),
          str(texts(events, "bot"))[:100])
    INVARIANTS_ANALYSIS = {"вердикт": "clear", "объяснение": "", "варианты": []}


# ---------------------------------------------------------------------------
# 6. Варианты-альтернативы: проверка по правилам, повтор, расход
# ---------------------------------------------------------------------------
def _sends(analysis):
    return [str(item.get("send") or "").lower()
            for item in (analysis or {}).get("suggestions") or []]


async def test_suggestion_verification():
    print("\n[6] Варианты-альтернативы: проверка по правилам, повтор, расход")
    global INVARIANTS_ANALYSIS, SUGGESTIONS_VERDICTS, ANALYSIS_CALLS
    global SUGGESTION_CALLS, ANALYSIS_QUEUE, PLAN_STEPS
    global SUGGESTIONS_QUEUE, SUGGESTIONS_DEFAULT

    # Чистые функции проверки вариантов: вердикты по номерам.
    check("неразобранный ответ проверки — СБОЙ, а не «совместимо»",
          invariants_store.parse_suggestion_verdicts("не json", 2) is None)
    check("нет вердикта по варианту — вариант НЕ подтверждён",
          invariants_store.parse_suggestion_verdicts(
              '{"1": {"вердикт": "clear"}}', 2) == [True, False])
    check("лишние ключи ответа проверке не мешают",
          invariants_store.parse_suggestion_verdicts(
              '{"1": {"вердикт": "violation"}, "2": {"вердикт": "clear"}, "3": {}}',
              2) == [False, True])

    PLAN_STEPS = ["Собрать требования", "Написать код"]
    ANALYSIS_QUEUE = []
    await chat.task_create(chat.TaskCreate(name="Проект с проверкой вариантов"))
    await chat.session_create()
    task = chat._current_task()
    session = chat._current_session()
    task["invariants"] = []
    session["dialog"]["invariants"] = []
    session["dialog"]["conflicts"] = []
    session["dialog"]["analysis"] = None
    await chat.invariant_create(InvariantCreate(text="язык только kotlin", scope="project"))
    await chat.invariant_create(InvariantCreate(
        text="только нативная платформа, никакой мультиплатформы", scope="project"))

    # 6.1 Модель предложила нарушающий вариант (iOS + Swift): он НЕ показывается.
    INVARIANTS_ANALYSIS = {
        "вердикт": "violation",
        "объяснение": "Веб-сайт нарушает правило проекта о нативной платформе.",
        "варианты": [
            {"заголовок": "Нативное приложение для Android",
             "пояснение": "Только Kotlin + Compose.",
             "запрос": "Составь план нативного Android-приложения на Kotlin с MVVM и Compose"},
            {"заголовок": "Нативное приложение для обеих платформ отдельно",
             "пояснение": "Kotlin для Android, Swift для iOS.",
             "запрос": "Составь план двух приложений: Android (Kotlin + Compose) "
                       "и iOS (Swift + MVVM)"},
            {"заголовок": "План экранов Android-приложения",
             "пояснение": "Только Android, MVVM + Compose.",
             "запрос": "Спланируй экраны Android-приложения на Compose"},
        ],
    }
    SUGGESTIONS_VERDICTS = {
        "1": {"вердикт": "clear", "причина": ""},
        "2": {"вердикт": "violation", "причина": "язык только kotlin"},
        "3": {"вердикт": "clear", "причина": ""},
    }
    SUGGESTION_CALLS = 0
    ANALYSIS_CALLS = 0
    events = await run_chat("нужен план разработки веб сайта для просмотра прогноза погоды")
    analyses = _suggestions_events(events)
    check("отказ с вариантами показан", bool(analyses))
    analysis = analyses[0] if analyses else {}
    sends = _sends(analysis)
    check("варианты проверены отдельным служебным вызовом",
          SUGGESTION_CALLS == 1, f"вызовов проверки: {SUGGESTION_CALLS}")
    check("нарушающий вариант отброшен (нет Swift/iOS)",
          len(sends) == 2 and not any("swift" in s or "ios" in s for s in sends),
          str(sends))
    check("совместимые варианты остались",
          all(item.get("title") and item.get("send")
              for item in analysis.get("suggestions") or []))
    check("в разборе отмечено, что варианты проверены",
          analysis.get("suggestions_checked") is True,
          str(analysis.get("suggestions_checked")))
    check("повторный запрос вариантов не понадобился",
          ANALYSIS_CALLS == 1, f"вызовов разбора: {ANALYSIS_CALLS}")
    check("сообщение обещает проверенные варианты",
          "проверен" in str(analysis.get("message") or "").lower(),
          str(analysis.get("message"))[-120:])
    check("план не построен", not (stage_of(events) or {}).get("steps"))
    check("в дебаге видно, что варианты проверены",
          any("варианты проверены" in t for t in texts(events, "debug")),
          str(texts(events, "debug"))[:160])
    check("расход проверки учтён как служебный",
          any((e.get("usage") or {}).get("summary_requests", 0) >= 2
              for e in events if e.get("type") == "done"),
          str([e.get("usage") for e in events if e.get("type") == "done"])[:200])
    history = await chat.agent_history()
    logged = [item for item in history["log"] if item.get("kind") == "suggestions"]
    check("отметка проверки переживает журнал и перезагрузку",
          bool(logged) and logged[-1]["analysis"].get("suggestions_checked") is True
          and all("swift" not in (item.get("send") or "").lower()
                  for item in logged[-1]["analysis"].get("suggestions") or []),
          str(logged[-1]["analysis"].get("suggestions_checked")) if logged else "нет узла")

    # 6.2 Пригодных вариантов меньше нормы — одна попытка попросить ДРУГИЕ,
    #     и только ПРОВЕРЕННЫЕ из них попадают в сообщение.
    INVARIANTS_ANALYSIS = None      # очередь ниже подменяет ответ по порядку
    ANALYSIS_QUEUE[:] = [
        {"вердикт": "violation", "объяснение": "Правило проекта: только Kotlin/Android.",
         "варианты": [
             {"заголовок": "Нативное Android-приложение",
              "пояснение": "В стеке.", "запрос": "Сделай Android-приложение на Kotlin"},
             {"заголовок": "iOS на Swift",
              "пояснение": "Другая платформа.", "запрос": "Сделай iOS-приложение на Swift"},
         ]},
        {"вердикт": "violation", "объяснение": "Правило проекта: только Kotlin/Android.",
         "варианты": [
             {"заголовок": "Экраны Android на Compose",
              "пояснение": "Только Android.", "запрос": "Спланируй экраны Android на Compose"},
             {"заголовок": "Данные и репозиторий Android",
              "пояснение": "Только Android.", "запрос": "Спланируй слой данных Android на Kotlin"},
         ]},
    ]
    SUGGESTIONS_VERDICTS = {"2": {"вердикт": "violation", "причина": "язык только kotlin"}}
    SUGGESTIONS_QUEUE[:] = [SUGGESTIONS_VERDICTS, {}]
    SUGGESTION_CALLS = 0
    ANALYSIS_CALLS = 0
    LAST_USER_TEXTS.clear()
    chat._current_session()["dialog"]["analysis"] = None
    events = await run_chat("сделай веб-версию приложения")
    analysis = (_suggestions_events(events) or [{}])[0]
    sends = _sends(analysis)
    check("повторный запрос вариантов сделан (их было меньше нормы)",
          ANALYSIS_CALLS == 2, f"вызовов разбора: {ANALYSIS_CALLS}")
    check("проверены оба набора вариантов",
          SUGGESTION_CALLS == 2, f"вызовов проверки: {SUGGESTION_CALLS}")
    check("в сообщении только проверенные варианты (3 штуки)",
          len(sends) == 3 and "swift" not in " ".join(sends), str(sends))
    check("в повторном запросе перечислены отклонённые варианты",
          any("ОТКЛОНЕНО" in text for text in LAST_USER_TEXTS),
          "отклонённый вариант должен уйти модели в повторном запросе")

    # 6.3 Повтор не бесконечный: снова ничего пригодного — вариантов нет, но
    #     сообщение честно говорит, что подходящих вариантов не нашлось.
    ANALYSIS_QUEUE[:] = [
        {"вердикт": "violation", "объяснение": "Только Kotlin/Android.",
         "варианты": [{"заголовок": "iOS на Swift", "пояснение": "Другая ОС.",
                       "запрос": "Сделай iOS-приложение на Swift"}]},
        {"вердикт": "violation", "объяснение": "Только Kotlin/Android.",
         "варианты": [{"заголовок": "Веб на React", "пояснение": "Веб запрещён.",
                       "запрос": "Сделай веб-сайт на React"}]},
    ]
    SUGGESTIONS_QUEUE[:] = []
    SUGGESTIONS_VERDICTS = {}
    SUGGESTIONS_DEFAULT = "violation"   # проверка работает, но ВСЕ варианты нарушают
    SUGGESTION_CALLS = 0
    ANALYSIS_CALLS = 0
    chat._current_session()["dialog"]["analysis"] = None
    events = await run_chat("сделай веб-сайт на React")
    analysis = (_suggestions_events(events) or [{}])[0]
    check("попытка ровно одна (без цикла)",
          ANALYSIS_CALLS == 2 and SUGGESTION_CALLS == 2,
          f"разбор: {ANALYSIS_CALLS}, проверка: {SUGGESTION_CALLS}")
    check("нарушающие варианты в сообщение не попали",
          _sends(analysis) == [], str(_sends(analysis)))
    check("сообщение не обещает вариантов",
          "не нашлось" in str(analysis.get("message") or "").lower(),
          str(analysis.get("message"))[-140:])

    # 6.4 Сбой проверки вариантов не выглядит как «варианты совместимы».
    ANALYSIS_QUEUE[:] = []
    SUGGESTIONS_DEFAULT = "clear"
    INVARIANTS_ANALYSIS = {
        "вердикт": "violation", "объяснение": "Только Kotlin/Android.",
        "варианты": [{"заголовок": "Нативное Android-приложение", "пояснение": "В стеке.",
                      "запрос": "Сделай Android-приложение на Kotlin"}],
    }
    SUGGESTIONS_VERDICTS = None      # «проверка не удалась»
    chat._current_session()["dialog"]["analysis"] = None
    events = await run_chat("сделай веб-сайт погоды")
    analysis = (_suggestions_events(events) or [{}])[0]
    check("при сбое проверки варианты не показываются",
          _sends(analysis) == [] and analysis.get("suggestions_checked") is False,
          str(analysis.get("suggestions")))
    check("сообщение объясняет, что проверка не удалась",
          "не удалась" in str(analysis.get("message") or "").lower(),
          str(analysis.get("message"))[-160:])
    check("в дебаге видно сбой проверки вариантов",
          any("варианты не проверены" in t for t in texts(events, "debug")),
          str(texts(events, "debug"))[:160])

    # 6.5 Запись журнала БЕЗ отметки проверки (старые данные): клик возможен, но
    #     текст уходит ОБЫЧНЫМ разбором — «не проверено» ≠ «разрешено». А
    #     проверенный вариант разбора не требует вовсе (см. раздел [7]).
    workspace_store.add_log_event(
        chat._current_session()["dialog"], workspace_store.LOG_SUGGESTIONS,
        "⛔ Запрос нарушает инвариант", {
            "verdict": "violation", "kind": "violation", "message": "старая запись",
            "suggestions": [{"title": "iOS на Swift", "details": "Другая ОС.",
                             "send": "Сделай iOS-приложение на Swift", "kind": "suggestion"}],
        })
    picked = await chat.invariant_choose(InvariantPick(index=0))
    check("старый вариант отправляется как обычный запрос",
          picked.get("action") == "send" and "Swift" in picked.get("text", ""),
          str(picked)[:80])
    SUGGESTIONS_VERDICTS = {}
    SUGGESTIONS_DEFAULT = "clear"
    INVARIANTS_ANALYSIS = {
        "вердикт": "violation", "объяснение": "Правило проекта: только Kotlin/Android.",
        "варианты": [{"заголовок": "Нативное Android-приложение", "пояснение": "В стеке.",
                      "запрос": "Сделай Android-приложение на Kotlin"},
                     {"заголовок": "Экраны Android на Compose", "пояснение": "В стеке.",
                      "запрос": "Спланируй экраны Android на Compose"}]}
    ANALYSIS_CALLS = 0
    events = await run_chat(picked["text"])
    check("текст из старой записи разбирается разбором",
          ANALYSIS_CALLS == 1, f"вызовов разбора: {ANALYSIS_CALLS}")
    check("нарушающий текст получает отказ",
          bool(_suggestions_events(events)), str(texts(events, "suggestions"))[:80])
    workspace_store.add_log_event(
        chat._current_session()["dialog"], workspace_store.LOG_SUGGESTIONS,
        "⛔ Запрос нарушает инвариант", {
            "verdict": "violation", "kind": "violation", "message": "проверенная запись",
            "suggestions_checked": True,
            "suggestions": [{"title": "Нативное Android-приложение", "details": "В стеке.",
                             "send": "Сделай Android-приложение на Kotlin", "kind": "suggestion"}],
        })
    picked = await chat.invariant_choose(InvariantPick(index=0))
    check("проверенный вариант отправляется",
          picked.get("action") == "send" and "Kotlin" in picked.get("text", ""),
          str(picked)[:80])

    ANALYSIS_QUEUE[:] = []
    SUGGESTIONS_QUEUE[:] = []
    SUGGESTIONS_VERDICTS = {}
    SUGGESTIONS_DEFAULT = "clear"
    INVARIANTS_ANALYSIS = {"вердикт": "clear", "объяснение": "", "варианты": []}


# ---------------------------------------------------------------------------
# 7. Выбранный вариант: повторного отказа быть не может
# ---------------------------------------------------------------------------
async def test_chosen_alternative():
    print("\n[7] Выбранный вариант проходит без повторного разбора")
    global INVARIANTS_ANALYSIS, SUGGESTIONS_VERDICTS, SUGGESTIONS_DEFAULT
    global ANALYSIS_CALLS, SUGGESTION_CALLS, ANALYSIS_QUEUE, SUGGESTIONS_QUEUE, PLAN_STEPS

    check("в промпте разбора противоречие правила задачи — НЕ нарушение запроса",
          "тоже нарушение правила проекта" not in invariants_store.ANALYSIS_PROMPT
          and "НЕ ДЕЙСТВУЕТ" in invariants_store.ANALYSIS_PROMPT)
    check("проверка вариантов смотрит на технологии, а не на заверения варианта",
          "НА ВЕРУ не принимай" in invariants_store.SUGGESTIONS_PROMPT)

    PLAN_STEPS = ["Собрать требования", "Написать код"]
    ANALYSIS_QUEUE = []
    SUGGESTIONS_QUEUE[:] = []
    SUGGESTIONS_VERDICTS = {}
    SUGGESTIONS_DEFAULT = "clear"
    await chat.task_create(chat.TaskCreate(name="Проект: только Android"))
    await chat.session_create()
    task = chat._current_task()
    session = chat._current_session()
    task["invariants"] = []
    session["dialog"]["invariants"] = []
    session["dialog"]["conflicts"] = []
    session["dialog"]["analysis"] = None
    await chat.invariant_create(InvariantCreate(text="язык только kotlin", scope="project"))
    await chat.invariant_create(InvariantCreate(
        text="только нативная платформа, никакой мультиплатформы", scope="project"))
    # Правило задачи, ПРОТИВОРЕЧАЩЕЕ правилу проекта: из-за него агент раньше
    # отказывался от ЛЮБОГО запроса, включая свой же проверенный вариант.
    await chat.invariant_create(InvariantCreate(
        text="пишем только под мультиплатформу (iOS + Android)", scope="task"))

    REFUSAL = {
        "вердикт": "violation",
        "объяснение": "Запрос требует мультиплатформу, а правило проекта — только "
                      "нативную платформу: действует правило проекта.",
        "варианты": [
            {"заголовок": "Нативное Android-приложение",
             "пояснение": "Только Android, Kotlin + Compose.",
             "запрос": "Нужен план разработки нативного Android-приложения для "
                       "просмотра прогноза погоды на Kotlin с MVVM и Compose."},
            {"заголовок": "Экраны Android-приложения",
             "пояснение": "Только Android, MVVM + Compose.",
             "запрос": "Спланируй экраны Android-приложения на Kotlin и Compose."},
        ],
    }

    # 7.1 Отказ по конфликту правил → выбор варианта → план, БЕЗ нового отказа.
    INVARIANTS_ANALYSIS = REFUSAL
    ANALYSIS_CALLS = 0
    SUGGESTION_CALLS = 0
    events = await run_chat("нужен план разработки приложения просмотра прогноза погоды")
    analyses = _suggestions_events(events)
    check("отказ при конфликте правил показан", bool(analyses))
    check("варианты проверены", (analyses or [{}])[0].get("suggestions_checked") is True,
          str((analyses or [{}])[0].get("suggestions_checked")))
    picked = await chat.invariant_choose(InvariantPick(index=0))
    check("сервер отдал текст выбранного варианта",
          picked.get("action") == "send" and "Android" in picked.get("text", ""),
          str(picked)[:80])
    ANALYSIS_CALLS = 0
    events = await run_chat(picked["text"])
    check("выбранный вариант больше НЕ разбирается (нет вызова LLM)",
          ANALYSIS_CALLS == 0, f"вызовов разбора: {ANALYSIS_CALLS}")
    check("повторного отказа нет", not _suggestions_events(events))
    check("по выбранному варианту строится план",
          any("План задачи" in text for text in texts(events, "bot")),
          str(texts(events, "bot"))[:140])
    check("в дебаге объяснено, почему разбора не было",
          any("проверенный вариант-альтернатива" in text for text in texts(events, "debug")),
          str(texts(events, "debug"))[:160])

    # 7.2 Правила изменились после проверки — тот же текст разбирается заново
    #     (отметка rules_signature не даёт считать старую проверку актуальной).
    session["dialog"]["analysis"] = None
    INVARIANTS_ANALYSIS = REFUSAL
    events = await run_chat("сделай веб-сайт прогноза погоды")
    check("отказ получен снова (есть узел с вариантами)", bool(_suggestions_events(events)))
    picked = await chat.invariant_choose(InvariantPick(index=0))
    await chat.invariant_create(InvariantCreate(text="тесты только на JUnit", scope="project"))
    INVARIANTS_ANALYSIS = {"вердикт": "clear", "объяснение": "", "варианты": []}
    ANALYSIS_CALLS = 0
    events = await run_chat(picked["text"])
    check("после правки правил вариант разбирается заново",
          ANALYSIS_CALLS == 1, f"вызовов разбора: {ANALYSIS_CALLS}")
    check("план по нему строится",
          any("План задачи" in text for text in texts(events, "bot")),
          str(texts(events, "bot"))[:120])

    # 7.3 Посторонний текст рядом с узлом вариантов проверку НЕ отменяет.
    session["dialog"]["analysis"] = None
    INVARIANTS_ANALYSIS = REFUSAL
    events = await run_chat("сделай веб-версию")
    check("отказ получен (узел с вариантами есть)", bool(_suggestions_events(events)))
    ANALYSIS_CALLS = 0
    events = await run_chat("придумай что-нибудь про погоду")
    check("посторонний текст разбирается разбором",
          ANALYSIS_CALLS == 1, f"вызовов разбора: {ANALYSIS_CALLS}")

    SUGGESTIONS_QUEUE[:] = []
    SUGGESTIONS_VERDICTS = {}
    SUGGESTIONS_DEFAULT = "clear"
    INVARIANTS_ANALYSIS = {"вердикт": "clear", "объяснение": "", "варианты": []}


# ---------------------------------------------------------------------------
# 8. Запрет проекта не обходится требованием задачи
# ---------------------------------------------------------------------------
async def test_project_ban_wins():
    print("\n[8] Правило задачи, противоречащее проекту, помечается НЕДЕЙСТВУЮЩИМ")
    global INVARIANTS_ANALYSIS, PLAN_STEPS, SUGGESTIONS_VERDICTS, SUGGESTIONS_DEFAULT

    # Чистые функции: номера правил задачи из разбора -> тексты для блока.
    task_rules = [{"id": "t1", "text": "пишем под мультиплатформу (android + iOS)"},
                  {"id": "t2", "text": "отвечает только на русском"}]
    check("номера недействующих правил разворачиваются в тексты",
          invariants_store.overridden_texts(task_rules, [1])
          == ["пишем под мультиплатформу (android + iOS)"])
    check("номер вне списка правил игнорируется",
          invariants_store.overridden_texts(task_rules, [7]) == [])
    check("номера-строки из ответа модели принимаются",
          invariants_store.overridden_texts(task_rules, ["1", "2"])
          == ["пишем под мультиплатформу (android + iOS)", "отвечает только на русском"])
    check("разбор понимает поле «недействующие»",
          invariants_store.parse_analysis(
              '{"вердикт": "clear", "объяснение": "", "недействующие": ["1"], '
              '"варианты": []}')["overridden"] == [1])
    check("без анализа блок всё равно называет правило проекта главным",
          "правило ПРОЕКТА всегда главнее"
          in invariants_store.invariants_block([{"id": "p1", "text": "только Android"}],
                                               [{"id": "t1", "text": "мультиплатформа"}]))

    PLAN_STEPS = ["Спроектируй архитектуру MVVM", "Разработай UI на Compose"]
    SUGGESTIONS_VERDICTS = {}
    SUGGESTIONS_DEFAULT = "clear"
    check("промпт разбора требует сообщать о противоречии правил",
          "вердикт — violation" in invariants_store.ANALYSIS_PROMPT
          and "СКРЫВАТЬ НЕЛЬЗЯ" in invariants_store.ANALYSIS_PROMPT,
          "иначе правило задачи снова будет молча проигнорировано")
    await chat.task_create(chat.TaskCreate(name="Проект: только Android"))
    await chat.session_create()
    task = chat._current_task()
    session = chat._current_session()
    task["invariants"] = []
    session["dialog"]["invariants"] = []
    session["dialog"]["conflicts"] = []
    session["dialog"]["analysis"] = None
    await chat.invariant_create(InvariantCreate(
        text="только нативная платформа android, никакой мультиплатформы", scope="project"))
    await chat.invariant_create(InvariantCreate(text="язык только kotlin", scope="project"))
    await chat.invariant_create(InvariantCreate(
        text="пишем только под iOS", scope="task"))
    await chat.invariant_create(InvariantCreate(
        text="отвечает только на русском", scope="task"))

    # Нейтральный запрос в задаче, чьё правило противоречит правилам проекта:
    # агент ОБЯЗАН предупредить о противоречии и дать варианты по правилам
    # проекта (молчаливое «сделаю по-своему» и есть тот самый баг).
    INVARIANTS_ANALYSIS = {
        "вердикт": "violation",
        "объяснение": "Правило задачи №1 «пишем только под iOS» противоречит "
                      "правилам проекта «только нативная платформа android, никакой "
                      "мультиплатформы» и «язык только kotlin»: действует правило "
                      "проекта, поэтому iOS-приложение сделать нельзя.",
        "недействующие": [1],
        "варианты": [
            {"заголовок": "Нативное Android-приложение",
             "пояснение": "Только Android, Kotlin + Compose.",
             "запрос": "Нужен план нативного Android-приложения погоды на Kotlin "
                       "с MVVM и Compose."},
            {"заголовок": "Экраны Android-приложения",
             "пояснение": "Только Android, MVVM + Compose.",
             "запрос": "Спланируй экраны Android-приложения погоды на Compose."},
        ],
    }
    ANALYSIS_CALLS = 0
    SUGGESTION_CALLS = 0
    events = await run_chat("нужен план разработки приложения для просмотра погоды")
    analyses = _suggestions_events(events)
    check("о противоречии правил сообщено (отказ с вариантами)", bool(analyses))
    analysis = analyses[0] if analyses else {}
    check("в объяснении названы оба правила",
          "iOS" in str(analysis.get("explanation") or "")
          and "проекта" in str(analysis.get("explanation") or "").lower(),
          str(analysis.get("explanation"))[:160])
    check("варианты проверены по правилам и не нарушают их",
          analysis.get("suggestions_checked") is True
          and len(_sends(analysis)) >= 2
          and not any("ios" in s for s in _sends(analysis)),
          str(_sends(analysis)))
    check("в сообщении сказано, что правило задачи не действует и где его поправить",
          "не действуют" in str(analysis.get("message") or "")
          and "Инварианты" in str(analysis.get("message") or ""),
          str(analysis.get("message"))[-200:])
    check("план не построен, пока пользователь не выбрал вариант",
          not (stage_of(events) or {}).get("steps"),
          str((stage_of(events) or {}).get("steps")))

    # Выбранный вариант: работа идёт по правилам ПРОЕКТА, а правило задачи
    # остаётся помеченным НЕДЕЙСТВУЮЩИМ в блоке, который видит планировщик.
    picked = await chat.invariant_choose(InvariantPick(index=0))
    LAST_MESSAGES.clear()
    events = await run_chat(picked["text"])
    planner_text = "\n\n".join(m["content"] for m in LAST_MESSAGES
                               if m.get("role") == "user")
    block = planner_text[planner_text.index("ИНВАРИАНТЫ"):] \
        if "ИНВАРИАНТЫ" in planner_text else ""
    check("по проверенному варианту план строится",
          bool((stage_of(events) or {}).get("steps")),
          str((stage_of(events) or {}).get("steps")))
    check("в блоке правило проекта названо главным",
          "правило ПРОЕКТА всегда главнее" in block)
    check("недействующее правило задачи ушло в блок отдельным разделом",
          "НЕ ДЕЙСТВУЮТ" in block
          and block.index("НЕ ДЕЙСТВУЮТ") < block.index("пишем только под iOS"),
          block[:240])
    check("действующее правило задачи осталось в списке «нарушать нельзя»",
          "отвечает только на русском" in block
          and block.index("отвечает только на русском")
          < block.index("пишем только под iOS"),
          block[:240])
    check("блок запрещает переносить требование задачи в план",
          "НЕ переноси требование" in block or "НЕ выполняй" in block,
          block[:240])

    INVARIANTS_ANALYSIS = {"вердикт": "clear", "объяснение": "", "варианты": []}


# ---------------------------------------------------------------------------
# 9. Код-гейт плана: шаги проверяются по правилам
# ---------------------------------------------------------------------------
async def test_plan_gate():
    print("\n[9] Код-гейт плана: нарушающие шаги в работу не уходят")
    global INVARIANTS_ANALYSIS, PLAN_STEPS, PLAN_QUEUE, PLAN_VERDICTS, PLAN_DEFAULT
    global PLAN_CALLS, ANALYSIS_CALLS, SUGGESTIONS_VERDICTS, SUGGESTIONS_DEFAULT

    BAD = ["Реализуй сетевой слой в KMP", "Собери приложение под Android и iOS"]
    GOOD = ["Спроектируй архитектуру MVVM", "Собери и протестируй под Android"]
    INVARIANTS_ANALYSIS = {"вердикт": "clear", "объяснение": "", "варианты": []}
    SUGGESTIONS_VERDICTS = {}
    SUGGESTIONS_DEFAULT = "clear"
    PLAN_STEPS = list(GOOD)
    PLAN_QUEUE[:] = []
    PLAN_VERDICTS = {}
    PLAN_DEFAULT = "clear"

    async def fresh_task() -> None:
        """Новая задача с правилом проекта: под каждый случай планирования.

        Переиспользовать одну задачу нельзя: после принятого плана следующее
        сообщение — уже правка/подтверждение плана, а не новое планирование.
        """
        await chat.task_create(chat.TaskCreate(name="Проект: только Android"))
        await chat.session_create()
        task_now = chat._current_task()
        session_now = chat._current_session()
        task_now["invariants"] = []
        session_now["dialog"]["invariants"] = []
        session_now["dialog"]["conflicts"] = []
        session_now["dialog"]["analysis"] = None
        await chat.invariant_create(InvariantCreate(
            text="только нативная платформа android, никакой мультиплатформы",
            scope="project"))

    # 9.1 Чистый план принимается, проверка видна в дебаге.
    await fresh_task()
    PLAN_QUEUE[:] = [GOOD]
    PLAN_CALLS = 0
    events = await run_chat("нужен план приложения погоды")
    check("план проверен код-гейтом", PLAN_CALLS == 1, f"вызовов проверки: {PLAN_CALLS}")
    check("чистый план принят",
          [s["text"] for s in (stage_of(events) or {}).get("steps", [])] == GOOD,
          str((stage_of(events) or {}).get("steps")))
    check("в дебаге видно, что шаги проверены",
          any("шаги плана проверены по инвариантам" in t for t in texts(events, "debug")),
          str(texts(events, "debug"))[-160:])
    check("расход проверки учтён как служебный",
          any((e.get("usage") or {}).get("summary_requests", 0) >= 2
              for e in events if e.get("type") == "done"),
          str([e.get("usage") for e in events if e.get("type") == "done"])[:160])

    # 9.2 План нарушает правило — перепланирование, в пометке запрещённые шаги.
    await fresh_task()
    PLAN_QUEUE[:] = [BAD, GOOD]
    PLAN_VERDICTS = {}
    PLAN_CALLS = 0
    LAST_MESSAGES.clear()
    events = await run_chat("сделай план мультиплатформенного приложения")
    check("нарушающий план отправлен на перепланирование",
          PLAN_CALLS == 2, f"вызовов проверки: {PLAN_CALLS}")
    planner_text = "\n\n".join(m["content"] for m in LAST_MESSAGES
                               if m.get("role") == "user")
    check("в перепланировании перечислены запрещённые шаги",
          "ОТКЛОНИЛА" in planner_text and BAD[0] in planner_text,
          planner_text[-200:])
    check("принят план после перепланирования",
          [s["text"] for s in (stage_of(events) or {}).get("steps", [])] == GOOD,
          str((stage_of(events) or {}).get("steps")))
    check("в чат не ушло ошибки",
          not texts(events, "error"), str(texts(events, "error"))[:120])

    # 9.3 Нарушения остались после попытки — план НЕ принят, задача остановлена.
    await fresh_task()
    PLAN_QUEUE[:] = [BAD, BAD]
    PLAN_VERDICTS = {}
    PLAN_CALLS = 0
    events = await run_chat("сделай план под обе платформы")
    # Перепланирование вызвано ровно один раз, но ПОВТОРНАЯ проверка тех же самых
    # шагов не оплачивается: вердикт по ним уже получен в этом же запросе
    # (кэш живёт внутри одного _plan_gate). Поэтому проверка вызвана один раз.
    check("перепланирование вызвано ровно один раз, повторная проверка тех же шагов "
          "не оплачивается",
          PLAN_CALLS == 1 and any("ОТКЛОНИЛА" in text for text in LAST_USER_TEXTS),
          f"вызовов проверки: {PLAN_CALLS}")
    check("план не принят — шагов нет",
          not (stage_of(events) or {}).get("steps"),
          str((stage_of(events) or {}).get("steps")))
    check("в чате объяснено, какие шаги нарушают правила",
          any("нарушает инварианты" in t and BAD[0] in t for t in texts(events, "error")),
          str(texts(events, "error"))[:200])
    check("плана в диалоге нет",
          not any("План задачи" in t for t in texts(events, "bot")),
          str(texts(events, "bot"))[:120])
    check("задача ждёт пользователя (не выполняется)",
          (stage_of(events) or {}).get("stage") == "awaiting_user",
          str((stage_of(events) or {}).get("stage")))

    # 9.4 Сбой проверки плана не выглядит как «нарушений нет».
    await fresh_task()
    PLAN_QUEUE[:] = [GOOD]
    PLAN_VERDICTS = None      # «проверка не удалась»
    PLAN_CALLS = 0
    events = await run_chat("сделай план приложения погоды ещё раз")
    check("проверка плана вызвана", PLAN_CALLS == 1, f"вызовов проверки: {PLAN_CALLS}")
    check("при сбое проверки план не принимается",
          not (stage_of(events) or {}).get("steps"),
          str((stage_of(events) or {}).get("steps")))
    check("сообщение говорит, что проверить план не удалось",
          any("не удалось проверить шаги" in t for t in texts(events, "error")),
          str(texts(events, "error"))[:200])

    PLAN_QUEUE[:] = []
    PLAN_VERDICTS = {}
    PLAN_DEFAULT = "clear"
    PLAN_STEPS = ["Собрать данные", "Написать код"]
    INVARIANTS_ANALYSIS = {"вердикт": "clear", "объяснение": "", "варианты": []}


async def _raises(coro_fn, status=404):
    """True, если вызов бросил HTTPException с ожидаемым кодом."""
    try:
        await coro_fn()
    except Exception as exc:  # noqa: BLE001
        return getattr(exc, "status_code", None) == status
    return False


def main():
    print("Проверка Task State Machine (без сети и LLM)")
    test_core()
    test_workspace()
    # ОДИН цикл событий на все прогоны: блокировки задач создаются при первом
    # обращении и привязаны к циклу, поэтому отдельный цикл на каждый прогон дал
    # бы «Lock is bound to a different event loop». Создаётся он ЯВНО:
    # asyncio.get_event_loop() устарел в Python 3.12 и будет удалён.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(test_routes())
        loop.run_until_complete(test_invariants())
        loop.run_until_complete(test_request_compliance())
        loop.run_until_complete(test_suggestion_verification())
        loop.run_until_complete(test_chosen_alternative())
        loop.run_until_complete(test_project_ban_wins())
        loop.run_until_complete(test_plan_gate())
    finally:
        loop.close()
    print("\nИтог: " + (f"ПРОВАЛЕНО проверок: {len(FAILURES)} → {FAILURES}"
                       if FAILURES else "все проверки пройдены"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
