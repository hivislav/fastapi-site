"""Самопроверка ПЕРИОДИЧЕСКИХ задач режима «AI-агент» (без сети и без ключа).

Запуск:

    ./venv/bin/python tools/check_periodic.py

Проверяется всё, что делает задачу периодической:

* разбор периода из текста запроса («Сводка погоды в Москве за последние сутки,
  раз в час» → час; «каждые 30 минут» → 30 минут; период не назван → сутки) и
  защита от ложных срабатываний («отчёт за последние сутки» — НЕ «раз в сутки»);
* хранение расписания в задаче-диалоге (сессии) и в файле workspace;
* маршруты создания периодической задачи и правки её расписания
  (POST /api/agent/sessions, GET|POST /api/agent/periodic);
* ПОВТОР: планировщик запускает задачу сам, она идёт АВТОНОМНО (план не ждёт
  подтверждения), выполняется цепочкой шагов до проверки результата и пишет
  ответ в ЧАТ задачи (журнал) с пометкой автозапуска;
* СВЕЖИЕ данные внешних инструментов MCP: повтор берёт их заново, а не
  переиспользует прошлый набор (иначе в чат попадали бы вчерашние числа);
* пропуски повтора: задача на паузе и задача, в которой уже идёт прогон;
* ход периодической задачи: план строится ОДИН раз на запрос (дальше повторы
  берут его из состояния — ни планировщик, ни гейт плана не вызываются), итоговой
  проверки результата нет, «готово» не ставится (цикл заканчивается возвратом на
  первый шаг), а «Пауза» обрывает цепочку сразу и повтор не засчитывается;
* смена запроса: новый текст пользователя строит новый план (тоже один раз),
  отмена задачи выключает автозапуск.

Вызовы LLM и обращения к серверам MCP подменяются заглушками, данные пишутся в
временный каталог: рабочие data/*.json не трогаются.
"""

import asyncio
import json
import os
import sys
import tempfile
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Изоляция данных: всё пишем во временный каталог ------------------------
_TMP = tempfile.mkdtemp(prefix="periodic-check-")
os.environ["AGENT_WORKSPACE_FILE"] = os.path.join(_TMP, "workspace.json")
os.environ["AGENT_MEMORY_FILE"] = os.path.join(_TMP, "agent_memory.json")
os.environ["AGENT_PROFILES_FILE"] = os.path.join(_TMP, "profiles.json")
# Планировщик в проверке не запускается сам: тики вызываются вручную.
os.environ["PERIODIC_ENABLED"] = "0"

from app import config  # noqa: E402
from app import periodic_runner  # noqa: E402
from app.ai import agent as agent_module  # noqa: E402
from app.ai import client, mcp as mcp_store  # noqa: E402
from app.ai import periodic as periodic_store  # noqa: E402
from app.ai import task_state, workspace as workspace_store  # noqa: E402
from app.routers import chat  # noqa: E402
from app.schemas import ChatMessage, PeriodicUpdate, SessionCreate  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    """Одна проверка: печатает результат и копит провалы."""
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# Заглушка LLM: план — JSON из PLAN_STEPS, ответ — ANSWER, проверка — REVIEW
# ---------------------------------------------------------------------------
PLAN_STEPS = ["Получить данные", "Сформировать сводку"]
ANSWER = "Сводка за прошедший период: +14, облачно."
REVIEW = {"verdict": "ok", "step": 0, "comment": "результат соответствует плану"}
# Очередь вердиктов проверки: непустая — i-я проверка берёт свой вердикт (нужна,
# чтобы проверить ДОРАБОТКУ по требованию проверки внутри повтора).
REVIEW_QUEUE = []
# ПРИМЕР внешнего инструмента для проверок: нарочно НЕ погодный и не из реестра
# проекта — повтор не должен зависеть ни от какого конкретного инструмента.
EXAMPLE_SERVER = "currency"
EXAMPLE_TOOL = "get_data"
# id внешнего сбора: его возвращает запускающий инструмент, по нему сбор
# отменяется (см. заглушку сервера выше).
COLLECTION_ID = "moscow"
# Признак «запуск сбора не удался» (заглушка сервера отвечает ошибкой).
START_FAILS = False
# Признак «чтение данных не удалось» (отчёт по несуществующему id).
FAIL_READS = False
# Признак «модель выдумывает таблицу» (ответ шага — таблица с числами).
FABRICATE = False
MCP_CALLS = []
MCP_CHOICE_CALLS = 0
DISCOVER_CALLS = 0
TOOL_CALLS = []
SYSTEM_PREFIXES = []
# Сколько раз вызывались служебные агенты: планировщик и приёмщик. По ним видно,
# что у периодической задачи план строится ОДИН раз, а итоговой проверки нет.
PLANNER_CALLS = 0
REVIEW_CALLS = 0
# user-части служебного выбора инструментов MCP: по ним видно, что уходит модели
# (в том числе блок об уже идущих внешних сборах задачи).
PAYLOADS = []
# Очередь ответов диспетчера MCP: пустая — отвечает MCP_CALLS.
DISPATCHER_QUEUE = []


def _metrics(prompt=20, completion=10):
    return {"model": "stub", "elapsed_seconds": 0.01, "prompt_tokens": prompt,
            "completion_tokens": completion, "total_tokens": prompt + completion}


async def fake_call_llm_async(*args, **kwargs):
    """Подмена client.call_llm_async: всё локально, без сети."""
    messages = kwargs.get("messages") or []
    system = str(messages[0].get("content") or "") if messages else ""
    SYSTEM_PREFIXES.append(system[:40])
    if system.startswith("Ты — планировщик"):
        global PLANNER_CALLS
        PLANNER_CALLS += 1
        return json.dumps({"steps": list(PLAN_STEPS)}, ensure_ascii=False), _metrics(30, 15)
    if system.startswith("Ты — приёмщик"):
        global REVIEW_CALLS
        REVIEW_CALLS += 1
        preset = REVIEW_QUEUE.pop(0) if REVIEW_QUEUE else REVIEW
        if preset is None:
            return "", _metrics()
        return json.dumps(preset, ensure_ascii=False), _metrics(40, 8)
    if system.startswith("Ты — арбитр инвариантов"):
        # Проверка шагов плана отдаёт вердикты по шагам, разбор запроса — «чисто».
        if "ШАГОВ ПЛАНА" in system:
            return json.dumps({"1": {"вердикт": "clear", "причина": ""},
                               "2": {"вердикт": "clear", "причина": ""}},
                              ensure_ascii=False), _metrics(30, 12)
        return json.dumps({"вердикт": "clear", "объяснение": "", "варианты": []},
                          ensure_ascii=False), _metrics(35, 20)
    if system.startswith("Ты — диспетчер внешних инструментов"):
        global MCP_CHOICE_CALLS
        MCP_CHOICE_CALLS += 1
        PAYLOADS.append(str(messages[-1].get("content") or "") if messages else "")
        # Очередь ответов диспетчера: непустая — i-й выбор берёт свой список
        # (нужна, чтобы проверить ПОВТОРНЫЙ вопрос «нужны фактические данные»).
        calls = DISPATCHER_QUEUE.pop(0) if DISPATCHER_QUEUE else list(MCP_CALLS)
        return json.dumps({"calls": list(calls)}, ensure_ascii=False), _metrics(45, 18)
    if FABRICATE:
        # Модель «дорисовывает» таблицу, хотя данных нет (регресс живой задачи).
        return ("| Время (МСК) | Температура |\n|---|---|\n| 14:00 | +18 |\n"
                "| 15:00 | +19 |"), _metrics()
    return ANSWER, _metrics()


client.call_llm_async = fake_call_llm_async


def install_mcp_stubs() -> None:
    """Подменяет обращения к серверам MCP: процессы в проверке не запускаются.

    Инструменты нарочно ОБЕЗЛИЧЕНЫ («get_data» у любого сервера): повтор
    периодической задачи не должен зависеть ни от конкретного сервера, ни от
    конкретного инструмента — набор инструментов проекта меняется, а свежесть
    данных нужна всегда (см. app/periodic_runner.py, `fresh` в _preflight_mcp).
    """

    async def fake_discover(ids=None, force=False):
        global DISCOVER_CALLS
        DISCOVER_CALLS += 1
        # Каждый включённый сервер объявляет ОДИН инструмент данных и пару
        # «запускающий ↔ отменяющий»: так проверяется, что уборка внешних сборов
        # ищет пару по ОБЪЯВЛЕННОМУ списку, а не по зашитым именам.
        return [{
            "id": server_id, "ok": True, "error": "",
            "server_name": server_id, "server_version": "0.1",
            "tools": [
                {"name": "get_data", "title": "Данные",
                 "description": "Данные внешнего источника",
                 "schema": {"type": "object",
                            "properties": {"place": {"type": "string"}}}},
                {"name": "start_collection", "title": "Начать сбор",
                 "description": "Начать сбор данных по месту",
                 "schema": {"type": "object",
                            "properties": {"place": {"type": "string"}}}},
                {"name": "stop_collection", "title": "Остановить сбор",
                 "description": "Остановить сбор по его id",
                 "schema": {"type": "object", "required": ["collection_id"],
                            "properties": {"collection_id": {"type": "string"}}}},
            ],
        } for server_id in (ids or [mcp_store.OPEN_METEO])]

    async def fake_run_calls(calls):
        TOOL_CALLS.append(list(calls))
        results = []
        for call in calls:
            tool = str(call.get("tool") or "")
            if tool == "start_collection" and START_FAILS:
                results.append({
                    "server": call.get("server"), "server_name": str(call.get("server")),
                    "source": "тест", "tool": tool,
                    "arguments": call.get("arguments") or {},
                    "ok": False, "text": "",
                    "error": 'No place matched "Тула"',
                })
                continue
            if tool == "start_collection":
                # Запускающий инструмент возвращает id — по нему потом читается
                # сводка и по нему же сбор отменяется.
                text = (f'Сбор начат (id: "{COLLECTION_ID}"). '
                        "Сводку читай инструментом отчёта по этому id.")
            elif tool == "stop_collection":
                text = f'Сбор "{call.get("arguments", {}).get("collection_id")}" остановлен.'
            elif tool == "get_data" and FAIL_READS:
                results.append({
                    "server": call.get("server"), "server_name": str(call.get("server")),
                    "source": "тест", "tool": tool,
                    "arguments": call.get("arguments") or {},
                    "ok": False, "text": "",
                    "error": ('No watch with id "%s"'
                              % (call.get("arguments") or {}).get("id", "")),
                })
                continue
            elif tool == "get_data":
                # Данные «меняются»: по номеру обращения видно, что повтор взял СВЕЖИЕ.
                text = f"данные get_data #{len(TOOL_CALLS)}"
            else:
                text = f"данные {tool} #{len(TOOL_CALLS)}"
            results.append({
                "server": call.get("server"), "server_name": str(call.get("server")),
                "source": "тест", "tool": tool,
                "arguments": call.get("arguments") or {},
                "ok": True, "text": text, "error": "",
            })
        return results

    def fake_call_tool(server_id, tool, arguments=None, timeout=30.0):
        """Одиночный вызов инструмента — им пользуется отмена внешних сборов."""
        TOOL_CALLS.append([{"server": server_id, "tool": tool,
                            "arguments": arguments or {}}])
        return {"ok": True,
                "text": f'{tool} для {arguments or {}}: остановлено', "error": ""}

    mcp_store.async_discover = fake_discover
    mcp_store.async_run_calls = fake_run_calls
    mcp_store.call_tool = fake_call_tool
    chat.mcp_store.async_discover = fake_discover
    chat.mcp_store.async_run_calls = fake_run_calls
    chat.mcp_store.call_tool = fake_call_tool


def reset_mcp():
    """Готовит диспетчеру ответ: один вызов внешнего инструмента за запрос."""
    global MCP_CHOICE_CALLS, DISCOVER_CALLS
    MCP_CHOICE_CALLS = 0
    DISCOVER_CALLS = 0
    TOOL_CALLS.clear()
    MCP_CALLS[:] = [{"server": EXAMPLE_SERVER, "tool": EXAMPLE_TOOL,
                     "arguments": {"place": "Москва"}}]


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


async def new_project(name="Проект погоды"):
    """Новый проект агента (в нём заводятся задачи-диалоги)."""
    await chat.task_create(chat.TaskCreate(name=name))


def session_by_id(session_id):
    """Сессия по id среди задач (без привязки к текущему профилю)."""
    for task in chat._workspace.get("tasks", []):
        session = workspace_store.find_session(task, session_id)
        if session is not None:
            return task, session
    return None, None


def log_texts(session, kind=None):
    return [item.get("text", "") for item in (session.get("dialog", {}).get("log") or [])
            if kind is None or item.get("kind") == kind]


# ---------------------------------------------------------------------------
# 1. Разбор периода из текста
# ---------------------------------------------------------------------------
def test_parse():
    print("\n[1] Разбор периода из текста запроса")
    cases = [
        ("Сводка погоды в Москве за последние сутки, раз в час", 3600),
        ("Качество воздуха в Москве каждые 30 минут", 1800),
        ("Присылай прогноз ежедневно", 86400),
        ("каждый час", 3600),
        ("каждую минуту", 60),
        ("раз в 15 минут", 900),
        ("раз в 2 часа", 7200),
        ("каждые 3 дня", 3 * 86400),
        ("раз в неделю", 7 * 86400),
    ]
    for text, expected in cases:
        seconds, phrase = periodic_store.parse_request(text)
        check(f"«{text[:44]}» → {periodic_store.label(expected)}",
              seconds == expected, f"получилось {seconds} ({phrase})")
    # Ложные срабатывания: периодом считается только фраза с триггером.
    for text in ("Отчёт за последние сутки", "дай рецепт борща", "ок",
                 "Сводка за прошлую неделю", "цена за час работы"):
        seconds, _ = periodic_store.parse_request(text)
        check(f"«{text}» — период не назван", seconds is None, f"получилось {seconds}")
    # Границы: чаще минуты и реже 30 суток периодов нет.
    check("«раз в 30 секунд» поднимается до минуты",
          periodic_store.parse_request("раз в 30 секунд")[0]
          == periodic_store.MIN_INTERVAL)
    check("«раз в 3 месяца» ограничен 30 сутками",
          periodic_store.parse_request("раз в 3 месяца")[0]
          == periodic_store.MAX_INTERVAL,
          str(periodic_store.parse_request("раз в 3 месяца")))
    check("период по умолчанию — сутки",
          periodic_store.DEFAULT_INTERVAL == 86400)
    check("мусор в периоде — сутки по умолчанию",
          periodic_store.normalize({"enabled": True, "interval": "вчера"})["interval"]
          == 86400)
    check("метки периода читаются человеком",
          periodic_store.label(3600) == "раз в час"
          and periodic_store.label(86400) == "раз в сутки"
          and periodic_store.label(1800) == "каждые 30 минут"
          and periodic_store.label(300) == "каждые 5 минут",
          " / ".join(periodic_store.label(s) for s in (3600, 86400, 1800, 300)))


# ---------------------------------------------------------------------------
# 2. Расписание сессии: хранение, снимок, сроки
# ---------------------------------------------------------------------------
def test_log_times():
    print("\n[2b] Журнал чата помнит время сообщений")
    dialog = workspace_store.empty_dialog("s-times")
    workspace_store.add_log(dialog, workspace_store.LOG_USER, "реплика")
    workspace_store.add_log(dialog, workspace_store.LOG_ASSISTANT, "ответ")
    workspace_store.add_log(dialog, workspace_store.LOG_DEBUG, "служебная строка")
    stamps = [item.get("at") for item in dialog["log"]]
    check("у каждой записи журнала есть время", all(stamps), str(stamps))
    check("время в том же формате, что метки задач (ISO до секунд)",
          all(len(value) == 19 and value[10] == "T" for value in stamps), str(stamps))
    restored = workspace_store._clean_log(dialog["log"])
    check("время переживает запись и чтение файла",
          [item.get("at") for item in restored] == stamps, str(restored))
    old_entry = workspace_store._clean_log(
        [{"kind": "user", "text": "старая запись без времени"}])
    check("у записи без времени подпись пустая, а не выдуманная",
          old_entry and old_entry[0].get("at") == "", str(old_entry))
    # Время узел получает СВОЁ: две записи подряд не делят одну метку.
    workspace_store.add_log(dialog, workspace_store.LOG_USER, "вторая",
                            at="2026-01-02T09:07:00")
    check("явное время узла сохраняется как есть",
          dialog["log"][-1]["at"] == "2026-01-02T09:07:00",
          str(dialog["log"][-1]))


def test_storage():
    print("\n[2] Расписание задачи: хранение, снимок, сроки")
    task = {"id": "t-1", "name": "Проект", "sessions": [], "working": [], "invariants": [],
            "mcp": {"enabled": []}}
    plain = workspace_store.create_session(task, title="Обычная")
    periodic = workspace_store.create_session(task, title="Сводка погоды", periodic=3600)
    check("обычная задача не периодическая",
          not workspace_store.periodic_meta(plain), str(plain.get("periodic")))
    meta = workspace_store.periodic_meta(periodic)
    check("периодическая задача получила расписание",
          meta.get("enabled") and meta.get("interval") == 3600, str(meta))
    check("первый повтор — через период",
          abs((periodic_store.parse_time(meta["next_run"])
               - periodic_store.now()).total_seconds() - 3600) < 5, meta["next_run"])

    # Нормализация: расписание переживает запись в файл и чтение обратно.
    raw = json.loads(json.dumps(task))
    normalized = workspace_store._normalize_task(raw)
    session = workspace_store.find_session(normalized, periodic["id"])
    check("расписание переживает нормализацию",
          workspace_store.periodic_meta(session).get("interval") == 3600,
          str(session.get("periodic")))
    check("расписание обычной задачи в файл не пишется",
          not workspace_store.find_session(normalized, plain["id"]).get("periodic"))

    # Выключенный повтор: задача ОСТАЁТСЯ периодической.
    off = {"enabled": False, "interval": 1800, "request": "текст"}
    normalized_off = periodic_store.normalize(off)
    check("выключенный повтор сохраняется как периодическая задача",
          normalized_off.get("enabled") is False and normalized_off["interval"] == 1800,
          str(normalized_off))
    check("выключенный повтор не срабатывает по сроку",
          not periodic_store.is_due(normalized_off))

    # Сроки: просроченное расписание — пора повторять.
    due = periodic_store.make(3600)
    due["next_run"] = periodic_store.to_iso(periodic_store.now() - timedelta(minutes=1))
    check("просроченный повтор — пора", periodic_store.is_due(due))
    check("будущий повтор — не пора", not periodic_store.is_due(periodic_store.make(3600)))
    # Начало и конец повтора: счётчик, отметки и следующий срок.
    started_at = periodic_store.now()
    periodic_store.started(due, started_at)
    check("на старте повтора срок уже сдвинут",
          not periodic_store.is_due(due, started_at + timedelta(seconds=30)))
    periodic_store.finished(due, started_at, True)
    check("после повтора: счётчик, отметка времени и новый срок",
          due["runs"] == 1 and due["last_run"]
          and periodic_store.parse_time(due["next_run"]) > started_at,
          str(due))
    periodic_store.finished(due, started_at, False, "модель не ответила")
    check("сбой повтора записан причиной", due["error"] == "модель не ответила"
          and due["runs"] == 2, str(due))


# ---------------------------------------------------------------------------
# 3. Маршруты: создание периодической задачи и правка расписания
# ---------------------------------------------------------------------------
async def test_routes():
    print("\n[3] Маршруты: создание и правка расписания")
    await new_project()

    plain = await chat.session_create(SessionCreate())
    check("«Новая задача» создаёт обычную задачу",
          not workspace_store.periodic_meta(chat._current_session()),
          str(chat._current_session().get("periodic")))

    created = await chat.session_create(SessionCreate(periodic=True))
    session = chat._current_session()
    check("«Новая периодическая задача» создаёт задачу с расписанием",
          workspace_store.periodic_meta(session).get("interval")
          == periodic_store.DEFAULT_INTERVAL, str(session.get("periodic")))
    brief = [item for item in created["sessions"] if item["id"] == session["id"]]
    check("в снимке задачи есть расписание (для метки в списке)",
          brief and brief[0].get("periodic", {}).get("label") == "раз в сутки",
          str(brief))
    check("расписание обычной задачи в снимке не появляется",
          not [item for item in created["sessions"]
                if item["id"] == plain.get("active_session") and item.get("periodic")],
          str([item for item in created["sessions"] if item.get("periodic")]))

    # Период задаётся числом секунд и текстом («как в запросе»).
    session_id = session["id"]
    await chat.periodic_update(session_id, PeriodicUpdate(enabled=True, interval=1800))
    check("период меняется числом секунд",
          workspace_store.periodic_meta(session)["interval"] == 1800,
          str(session.get("periodic")))
    await chat.periodic_update(session_id, PeriodicUpdate(enabled=True, period="раз в час"))
    check("период понимается из текста",
          workspace_store.periodic_meta(session)["interval"] == 3600,
          str(session.get("periodic")))
    try:
        await chat.periodic_update(session_id, PeriodicUpdate(enabled=True, period="когда-нибудь"))
        check("непонятный период — 400", False, "(ошибки не было)")
    except Exception as exc:  # noqa: BLE001
        check("непонятный период — 400", getattr(exc, "status_code", None) == 400,
              str(getattr(exc, "detail", exc)))

    # Остановка и включение повтора.
    await chat.periodic_update(session_id, PeriodicUpdate(enabled=False))
    check("автозапуск выключается",
          workspace_store.periodic_meta(session)["enabled"] is False,
          str(session.get("periodic")))
    check("выключенная задача остаётся периодической и не срабатывает",
          bool(workspace_store.periodic_meta(session))
          and not periodic_store.is_due(workspace_store.periodic_meta(session)))
    await chat.periodic_update(session_id, PeriodicUpdate(enabled=True))
    meta = workspace_store.periodic_meta(session)
    check("включение ставит срок от текущего момента",
          meta["enabled"] and periodic_store.parse_time(meta["next_run"])
          > periodic_store.now(), str(meta))

    # Обычную задачу периодической этим маршрутом не сделать (409), чужую задачу
    # не найти (404).
    plain_id = plain["active_session"]
    try:
        await chat.periodic_update(plain_id, PeriodicUpdate(enabled=True))
        check("обычная задача — 409", False, "(ошибки не было)")
    except Exception as exc:  # noqa: BLE001
        check("обычная задача — 409", getattr(exc, "status_code", None) == 409,
              str(getattr(exc, "detail", exc)))
    try:
        await chat.periodic_update("s-нет-такой", PeriodicUpdate(enabled=True))
        check("неизвестная задача — 404", False, "(ошибки не было)")
    except Exception as exc:  # noqa: BLE001
        check("неизвестная задача — 404", getattr(exc, "status_code", None) == 404,
              str(getattr(exc, "detail", exc)))

    # Снимок расписаний для опроса интерфейсом.
    payload = await chat.periodic_get()
    item = [row for row in payload["tasks"] if row["session_id"] == session_id]
    check("снимок расписаний отдаёт задачу с периодом и сроком",
          item and item[0]["periodic"]["label"] == "раз в час"
          and item[0]["periodic"]["next_run"], str(item)[:200])
    check("в снимке расписаний есть размер журнала (для опроса)", 
          item and "log_len" in item[0]["periodic"], str(item)[:200])


# ---------------------------------------------------------------------------
# 4. Период, названный в тексте запроса
# ---------------------------------------------------------------------------
async def test_text_period():
    print("\n[4] Период в тексте запроса")
    await new_project("Проект: период в тексте")
    await chat.session_create(SessionCreate(periodic=True))
    session = chat._current_session()
    session_id = session["id"]

    events = await run_chat("Сводка погоды в Москве за последние сутки, раз в час",
                            session_id=session_id)
    meta = workspace_store.periodic_meta(session)
    check("период из запроса применён к расписанию",
          meta["interval"] == 3600, str(meta))
    check("запрос задачи сохранён как повторяющийся",
          meta["request"].startswith("Сводка погоды в Москве"), meta["request"])
    check("в чате объяснено, что задача периодическая",
          any("раз в час" in text for text in log_texts(session, "debug")),
          str(log_texts(session, "debug"))[:200])
    check("первый повтор — через названный период",
          abs((periodic_store.parse_time(meta["next_run"]) - periodic_store.now())
              .total_seconds() - 3600) < 10, meta["next_run"])

    # Служебные фразы запросом не считаются: «ок» не меняет ни текст, ни период.
    before = dict(meta)
    await run_chat("ок", session_id=session_id)
    # Период ниже предела: в чате сказано, что он ПОДНЯТ, и что интервал сбора
    # внешнего инструмента задаёт его сервер (период повтора на него не влияет).
    await chat.session_create(SessionCreate(periodic=True))
    clamped = chat._current_session()
    await run_chat("сводка погоды в Екатеринбурге, раз в 5 секунд", session_id=clamped["id"])
    debug_lines = log_texts(clamped, "debug")
    check("период ниже предела поднят до минуты",
          workspace_store.periodic_meta(clamped)["interval"] == periodic_store.MIN_INTERVAL,
          str(workspace_store.periodic_meta(clamped)["interval"]))
    check("в чате сказано, что период поднят и почему",
          any("чаще" in text and "раз в минуту" in text for text in debug_lines),
          str(debug_lines)[:300])
    check("в чате сказано, что интервал сбора задаёт сервер инструмента",
          any("Интервал сбора" in text for text in debug_lines),
          str(debug_lines)[:300])
    await chat.session_delete(clamped["id"])

    check("подтверждение плана не подменяет запрос задачи",
          workspace_store.periodic_meta(session)["request"] == before["request"],
          workspace_store.periodic_meta(session)["request"])

    # Новый запрос без периода — период остаётся прежним, а текст обновляется.
    await run_chat("Теперь присылай качество воздуха в Москве", session_id=session_id)
    meta = workspace_store.periodic_meta(session)
    check("новый запрос без периода сохраняет прежний период",
          meta["interval"] == 3600 and meta["request"].startswith("Теперь присылай"),
          str(meta))

    # Остановленный повтор не включается сам от нового сообщения.
    await chat.periodic_update(session_id, PeriodicUpdate(enabled=False))
    stopped_events = await run_chat("И ещё добавь уровень шума", session_id=session_id)
    check("остановленный повтор от сообщения не включается",
          workspace_store.periodic_meta(session)["enabled"] is False,
          str(session.get("periodic")))
    check("в чате сказано, что автозапуск остановлен",
          any("остановлен" in text for text in
              [e.get("text", "") for e in stopped_events if e.get("type") == "debug"]),
          str([e.get("text") for e in stopped_events if e.get("type") == "debug"])[:200])


# ---------------------------------------------------------------------------
# 5. ПОВТОР: планировщик сам выполняет задачу и пишет в чат
# ---------------------------------------------------------------------------
async def make_due(session_id):
    """Делает расписание задачи просроченным (как будто срок наступил)."""
    _, session = session_by_id(session_id)
    meta = workspace_store.periodic_meta(session)
    meta["next_run"] = periodic_store.to_iso(periodic_store.now() - timedelta(seconds=1))
    workspace_store.set_periodic(session, meta)
    return session


async def test_repeat():
    print("\n[5] Автозапуск: повтор проходит цикл по сохранённому плану")
    install_mcp_stubs()
    await new_project("Проект: автозапуск")
    await chat.session_create(SessionCreate(periodic=True))
    session = chat._current_session()
    session_id = session["id"]
    # Задача с уже известным запросом: так расписание появляется у задачи, которую
    # пользователь завёл и описал сообщением (см. раздел [4]).
    plan_calls_start = PLANNER_CALLS
    await run_chat("Сводка погоды в Москве за последние сутки, раз в час",
                   session_id=session_id)
    # Первый пользовательский запрос оставил задачу ждать подтверждения плана:
    # автозапуск обязан пройти его САМ (без «ок» от пользователя) — и по УЖЕ
    # построенному плану: планировщик за тот же запрос больше не платится.
    check("задача пользователя ждёт подтверждения плана",
          workspace_store.dialog_state(session).stage == "awaiting_user",
          workspace_store.dialog_state(session).stage)
    plan_calls_before = PLANNER_CALLS
    check("первый запрос построил план (один раз)",
          plan_calls_before == plan_calls_start + 1
          and len(workspace_store.dialog_state(session).steps) == len(PLAN_STEPS),
          f"вызовов планировщика: {plan_calls_before - plan_calls_start}")

    await make_due(session_id)
    log_before = len(log_texts(session))
    started = await periodic_runner.tick()
    check("тик запустил повтор задачи", session_id in started, str(started))
    # Тик создаёт отдельную задачу — ждём её завершения.
    for _ in range(200):
        if not chat._periodic_running:
            break
        await asyncio.sleep(0.02)
    check("повтор завершился (задача не висит в работе)", not chat._periodic_running,
          str(chat._periodic_running))

    state = workspace_store.dialog_state(session)
    check("повтор прошёл АВТОНОМНО по сохранённому плану",
          state.autonomous is True and len(state.steps) == len(PLAN_STEPS),
          f"{state.stage}, шагов {len(state.steps)}")
    check("у периодической задачи НЕТ статуса «готово»",
          state.stage == "execution"
          and state.stage not in task_state.TERMINAL_STAGES,
          f"{state.stage}: {state.reason}")
    check("цикл пройден: место снова первый шаг (следующий повтор — с начала)",
          state.step_index == 0 and state.current_step == "step_1",
          f"step_index={state.step_index}, {state.current_step}")
    check("повтор не строил план заново", PLANNER_CALLS == plan_calls_before,
          f"вызовов планировщика: {PLANNER_CALLS}")
    check("повтор не выполнял итоговой проверки", REVIEW_CALLS == 0,
          f"вызовов приёмщика: {REVIEW_CALLS}")
    check("в чате нет этапа «проверка»",
          not any("этап validation" in text for text in log_texts(session, "debug")),
          str(log_texts(session, "debug"))[-300:])
    check("в чате сказано, что повтор завершён и план сохранён",
          any("повтор завершён" in text for text in log_texts(session, "debug")),
          str(log_texts(session, "debug"))[-300:])
    # Признак «ждём следующего повтора» — ЯВНЫЙ (по нему интерфейс понимает, что
    # прогон шагов продолжать нечем: этап у периодической задачи не «Готово»).
    check("состояние помечено «цикл пройден, ждём повтора»", state.repeat_ready is True,
          str(state.repeat_ready))
    check("признак уходит интерфейсу в снимке",
          (await chat.state_get())["state"]["repeat_ready"] is True)
    check("после перезагрузки страницы признак не теряется",
          task_state.from_dict(task_state.to_dict(state)).repeat_ready is True)

    markers = [text for text in log_texts(session, "periodic")]
    check("в чате есть пометка автозапуска с периодом",
          markers and markers[0].startswith(periodic_store.AUTO_MARK)
          and "раз в час" in markers[0], str(markers)[:200])
    check("в чате появились ответы всех шагов повтора",
          sum(1 for text in log_texts(session, "assistant") if ANSWER in text) >= len(PLAN_STEPS),
          str(log_texts(session, "assistant"))[-200:])
    check("журнал задачи вырос (интерфейсу есть что подхватить)",
          len(log_texts(session)) > log_before,
          f"было {log_before}, стало {len(log_texts(session))}")

    meta = workspace_store.periodic_meta(session)
    check("повтор учтён: счётчик, отметка времени, новый срок",
          meta["runs"] == 1 and meta["last_run"]
          and not periodic_store.is_due(meta) and not meta["error"], str(meta))
    check("запрос задачи не изменился от автозапуска",
          meta["request"].startswith("Сводка погоды в Москве"), meta["request"])
    # Реплика автозапуска в памяти диалога помечена: старый диалог без журнала
    # нарисует её служебной строкой, а не репликой пользователя.
    sources = [m.get("source") for m in session["dialog"]["messages"]]
    check("реплика автозапуска помечена в памяти диалога",
          periodic_store.SOURCE_AUTO in sources, str(sources))
    # Расход повтора тоже записан: повторы стоят токенов, и это видно в панели.
    check("расход повтора записан в замеры задачи",
          bool(session["dialog"]["usage"]), str(session["dialog"]["usage"])[:200])

    # ПОЛОСА ЭТАПОВ: у периодической задачи нет блоков «проверка» и «готово» —
    # этих этапов она не проходит (см. task_state.PERIODIC_BASE_STAGES).
    snapshot = (await chat.state_get())["state"]
    ids = [item["id"] for item in snapshot["base_stages"]]
    check("в полосе периодической задачи только планирование и выполнение",
          ids == ["planning", "execution"], str(ids))
    check("у ОБЫЧНОЙ задачи полоса прежняя (четыре этапа)",
          [item["id"] for item in task_state.snapshot(task_state.new_state("s-x"))["base_stages"]]
          == ["planning", "execution", "validation", "done"])

    # ВТОРОЙ повтор: тот же порядок — план не перестраивается, проверки нет.
    await make_due(session_id)
    await periodic_runner.tick()
    for _ in range(200):
        if not chat._periodic_running:
            break
        await asyncio.sleep(0.02)
    state = workspace_store.dialog_state(session)
    check("второй повтор тоже идёт по сохранённому плану",
          PLANNER_CALLS == plan_calls_before and REVIEW_CALLS == 0
          and state.stage == "execution" and state.step_index == 0,
          f"планировщик: {PLANNER_CALLS}, приёмщик: {REVIEW_CALLS}, {state.stage}")
    check("второй повтор засчитан отдельно",
          workspace_store.periodic_meta(session)["runs"] == 2,
          str(workspace_store.periodic_meta(session)["runs"]))


async def test_repeat_fresh_mcp():
    print("\n[6] Повтор берёт СВЕЖИЕ данные внешних инструментов")
    install_mcp_stubs()
    reset_mcp()
    await new_project("Проект: свежие данные")
    await chat.session_create(SessionCreate(periodic=True))
    await chat.mcp_apply(chat.McpApply(enabled=[EXAMPLE_SERVER]))
    session = chat._current_session()
    session_id = session["id"]
    await run_chat("Присылай данные по Москве, раз в час", session_id=session_id)
    await run_chat("ок", session_id=session_id)          # подтверждаем план
    await run_chat("", continue_step=True, session_id=session_id)
    stored = workspace_store.dialog_mcp(session["dialog"])
    check("данные MCP собраны по запросу задачи",
          stored.get("signature") and stored["results"], str(stored)[:200])

    # Шаг плана данные НЕ переспрашивает: они уже есть по этому запросу.
    reset_mcp()
    await run_chat("", continue_step=True, session_id=session_id)
    check("шаг плана не выбирает инструменты заново", MCP_CHOICE_CALLS == 0,
          f"выборов: {MCP_CHOICE_CALLS}")

    # ПОВТОР: тот же запрос — те же ЧИТАЮЩИЕ вызовы, только данные свежие.
    # Диспетчера повтор НЕ спрашивает: выбор уже сделан, а заново он мог решить
    # иначе (в живой задаче повтор решил, что данные не нужны, и ответ ушёл без
    # погоды).
    reset_mcp()
    await make_due(session_id)
    await periodic_runner.tick()
    for _ in range(200):
        if not chat._periodic_running:
            break
        await asyncio.sleep(0.02)
    check("повтор повторяет те же вызовы, а не спрашивает диспетчера заново",
          MCP_CHOICE_CALLS == 0 and len(TOOL_CALLS) == 1,
          f"выборов: {MCP_CHOICE_CALLS}, вызовов: {TOOL_CALLS}")
    check("повтор повторил именно ЧИТАЮЩИЙ вызов",
          bool(TOOL_CALLS) and TOOL_CALLS[0][0]["tool"] == EXAMPLE_TOOL,
          str(TOOL_CALLS)[:200])
    fresh = workspace_store.dialog_mcp(session["dialog"])
    text = (fresh["results"][0]["text"] if fresh.get("results") else "")
    check("в задаче лежат СВЕЖИЕ данные (а не прошлый набор)",
          text.endswith(f"#{len(TOOL_CALLS)}") and EXAMPLE_TOOL in text, text)
    check("повтор не зависит от конкретного инструмента: данные взяты у "
          "произвольного сервера",
          fresh.get("results") and fresh["results"][0]["server"] == EXAMPLE_SERVER,
          str(fresh.get("results"))[:200])


# ---------------------------------------------------------------------------
# 6b. Выбор инструментов: чтение вместо запуска сбора
# ---------------------------------------------------------------------------
async def test_read_instead_of_start():
    global FAIL_READS, FABRICATE
    print("\n[6b] «Проверяй погоду раз в минуту» — это ЧТЕНИЕ, а не запуск сбора")
    install_mcp_stubs()
    reset_mcp()
    await new_project("Проект: чтение или сбор")
    await chat.session_create(SessionCreate(periodic=True))
    await chat.mcp_apply(chat.McpApply(enabled=[EXAMPLE_SERVER]))
    session = chat._current_session()
    session_id = session["id"]

    # Диспетчер «запускает сбор» на запрос, которому нужны ТЕКУЩИЕ данные: агент
    # обязан переспросить и получить читающий вызов — иначе ответ уходит без фактов
    # («наблюдение зарегистрировано» вместо погоды).
    DISPATCHER_QUEUE[:] = [
        [{"server": EXAMPLE_SERVER, "tool": "start_collection",
          "arguments": {"place": "Екатеринбург"}}],
        [{"server": EXAMPLE_SERVER, "tool": "get_data",
          "arguments": {"place": "Екатеринбург"}}],
    ]
    PAYLOADS.clear()
    await run_chat("проверяй погоду в Екатеринбурге раз в минуту", session_id=session_id)
    check("диспетчера переспросили про данные", MCP_CHOICE_CALLS == 2,
          f"выборов: {MCP_CHOICE_CALLS}")
    check("во втором вопросе сказано, что нужны ФАКТИЧЕСКИЕ данные",
          len(PAYLOADS) > 1 and "НЕ читают данные" in PAYLOADS[1], str(PAYLOADS[1])[:200])
    check("выполнен ЧИТАЮЩИЙ вызов",
          bool(TOOL_CALLS) and TOOL_CALLS[0][0]["tool"] == EXAMPLE_TOOL,
          str(TOOL_CALLS)[:200])
    check("запуск сбора НЕ выполнялся",
          all(call["tool"] != "start_collection" for call in TOOL_CALLS[0])
          if TOOL_CALLS else True, str(TOOL_CALLS)[:200])
    check("в чате объяснено, что первые вызовы только запускали сбор",
          any("переспросил" in text for text in log_texts(session, "debug")),
          str(log_texts(session, "debug"))[:300])
    stored = workspace_store.dialog_mcp(session["dialog"])
    check("выбранные вызовы сохранены для повторов",
          bool(stored.get("calls")) and stored["calls"][0]["tool"] == EXAMPLE_TOOL,
          str(stored.get("calls")))
    check("вызов-запуск помечен как действие (повтор его не повторит)",
          mcp_store.mark_call_kinds(
              [{"server": EXAMPLE_SERVER, "tool": "start_collection"}], [{
                  "server": EXAMPLE_SERVER, "tool": "start_collection", "schema": {}},
                  {"server": EXAMPLE_SERVER, "tool": "stop_collection", "schema": {}}])
          and mcp_store.read_calls([{"server": EXAMPLE_SERVER,
                                     "tool": "start_collection", "action": True}]) == [])
    # ОСНОВА ДАННЫХ входит в подпись плана: план, построенный на одних данных, не
    # переиспользуется, когда читать стали другие (иначе задача с планом «проверить
    # наблюдение» осталась бы с ним навсегда, хотя читает уже текущую погоду).
    signature = session["dialog"].get("plan_signature") or {}
    check("подпись плана помнит основу данных",
          signature.get("basis", "") == f"{EXAMPLE_SERVER}·{EXAMPLE_TOOL}",
          str(signature))
    check("подпись плана с основой переживает перезагрузку",
          workspace_store._clean_plan_signature(signature) == signature,
          str(workspace_store._clean_plan_signature(signature)))

    # ПОВТОР: запуск сбора не повторяется (иначе на сервере копились бы дубли), а
    # читающий вызов повторяется со свежими данными и без диспетчера.
    reset_mcp()
    await make_due(session_id)
    await periodic_runner.tick()
    for _ in range(300):
        if not chat._periodic_running:
            break
        await asyncio.sleep(0.02)
    check("повтор повторил только ЧТЕНИЕ",
          bool(TOOL_CALLS) and [call["tool"] for call in TOOL_CALLS[0]] == [EXAMPLE_TOOL],
          str(TOOL_CALLS)[:200])
    check("повтор не спрашивал диспетчера", MCP_CHOICE_CALLS == 0,
          f"выборов: {MCP_CHOICE_CALLS}")
    check("повтор не запустил сбор заново",
          not workspace_store.mcp_started(session["dialog"]),
          str(workspace_store.mcp_started(session["dialog"])))

    # ПОСЛЕ ЗАПУСКА СБОРА ДАННЫЕ ЧИТАЮТСЯ СРАЗУ. Id известен только из ответа на
    # запуск, и без второго вопроса ответ уходит вообще без данных, хотя на сервере
    # они есть (живая задача: запуск возобновил наблюдение с 7 образцами, а отчёт
    # так и не был запрошен — модель пересказала ответ запуска как «отчёт»).
    await chat.session_create(SessionCreate(periodic=True))
    starter = chat._current_session()
    DISPATCHER_QUEUE[:] = [
        [{"server": EXAMPLE_SERVER, "tool": "start_collection",
          "arguments": {"place": "Екатеринбург"}}],
        [{"server": EXAMPLE_SERVER, "tool": "get_data",
          "arguments": {"collection_id": COLLECTION_ID, "details": True}}],
    ]
    reset_mcp()
    await run_chat("сводка погоды за сутки, раз в час", session_id=starter["id"])
    check("после запуска сбора диспетчера спросили про ДАННЫЕ",
          MCP_CHOICE_CALLS == 2, f"выборов: {MCP_CHOICE_CALLS}")
    flat_calls = [call for batch in TOOL_CALLS for call in batch]
    check("данные прочитаны в ЭТОМ ЖЕ запросе, а не со следующего повтора",
          [(call["tool"], call["arguments"].get("collection_id")) for call in flat_calls]
          == [("start_collection", None), ("get_data", COLLECTION_ID)],
          str(flat_calls)[:250])
    check("в чате сказано, что данные читаются сразу по запущенному сбору",
          any("сбор запущен — читаю данные по нему" in text
              for text in log_texts(starter, "debug")),
          str(log_texts(starter, "debug"))[-300:])
    stored = workspace_store.dialog_mcp(starter["dialog"])
    check("сохранены и запуск, и чтение (повтор повторит только чтение)",
          [call["tool"] for call in (stored.get("calls") or [])]
          == ["start_collection", "get_data"],
          str(stored.get("calls")))
    check("повтору остаётся ЧИТАЮЩИЙ вызов с настоящим id",
          [call["tool"] for call in mcp_store.read_calls(stored.get("calls"))]
          == ["get_data"],
          str(mcp_store.read_calls(stored.get("calls"))))
    check("пометка про подробные записи есть в подсказке диспетчера",
          "include_samples" in mcp_store.TOOLS_PROMPT)

    # СЛОМАННЫЙ ВЫЗОВ НЕ ПОВТОРЯЕТСЯ: если прошлый запрос данных отказал, повтор
    # выбирает инструменты заново (иначе он снова получит отказ, а модель начнёт
    # выдумывать данные — так и случилось в живой задаче с отчётом наблюдения).
    await chat.session_create(SessionCreate(periodic=True))
    broken = chat._current_session()
    DISPATCHER_QUEUE[:] = [
        [{"server": EXAMPLE_SERVER, "tool": "get_data",
          "arguments": {"place": "Екатеринбург", "id": "екатеринбург"}}],
    ]
    FAIL_READS = True
    try:
        await run_chat("сводка погоды за сутки, раз в час", session_id=broken["id"])
    finally:
        FAIL_READS = False
    stored = workspace_store.dialog_mcp(broken["dialog"])
    check("отказавший вызов сохранён, но помечен отказом",
          bool(stored.get("results")) and stored["results"][0]["ok"] is False,
          str(stored.get("results"))[:200])
    check("в блоке модели стоит пометка «данных нет»",
          "ДАННЫХ НЕТ" in mcp_store.block(stored["results"]),
          mcp_store.block(stored["results"])[:300])
    # Повтор: сломанный вызов не повторяем — выбираем инструменты заново.
    reset_mcp()
    DISPATCHER_QUEUE[:] = [
        [{"server": EXAMPLE_SERVER, "tool": "get_data",
          "arguments": {"place": "Екатеринбург", "id": "ekaterinburg"}}],
    ]
    await make_due(broken["id"])
    await periodic_runner.tick()
    for _ in range(300):
        if not chat._periodic_running:
            break
        await asyncio.sleep(0.02)
    check("повтор НЕ повторяет отказавший вызов, а выбирает заново",
          MCP_CHOICE_CALLS == 1 and bool(TOOL_CALLS)
          and TOOL_CALLS[0][0]["arguments"].get("id") == "ekaterinburg",
          f"выборов: {MCP_CHOICE_CALLS}, вызовы: {str(TOOL_CALLS)[:200]}")
    check("в чате сказано, что прошлый повтор данных не дал",
          any("не дал данных для чтения" in text
              for text in log_texts(broken, "debug")),
          str(log_texts(broken, "debug"))[-300:])

    # ВЫДУМАННАЯ ТАБЛИЦА НЕ ПОКАЗЫВАЕТСЯ: если ни один вызов не дал данных, а
    # модель всё равно нарисовала таблицу с числами — в чат уходит причина отказа.
    await chat.session_create(SessionCreate(periodic=True))
    fab = chat._current_session()
    FABRICATE = True
    FAIL_READS = True
    DISPATCHER_QUEUE[:] = [
        [{"server": EXAMPLE_SERVER, "tool": "get_data",
          "arguments": {"place": "Екатеринбург"}}],
    ]
    try:
        await run_chat("сводка погоды за последний час, раз в минуту",
                       session_id=fab["id"])
        await run_chat("ок", session_id=fab["id"])            # подтверждаем план
        events = await run_chat("", continue_step=True, session_id=fab["id"])
    finally:
        FAIL_READS = False
        FABRICATE = False
    bot_texts = [event.get("text") for event in events if event.get("type") == "bot"]
    check("вместо выдуманной таблицы показана причина отказа",
          bool(bot_texts) and "Данные от внешних инструментов не получены" in bot_texts[-1],
          str(bot_texts)[-200:])
    check("выдуманных чисел в ответе нет",
          bool(bot_texts) and "| 14:00 |" not in bot_texts[-1], str(bot_texts)[-200:])
    check("в чате объяснено, почему таблица не показана",
          any("показываю причину отказа вместо выдуманных значений" in text
              for text in log_texts(fab, "debug")),
          str(log_texts(fab, "debug"))[-200:])
    check("отказавшие вызовы заставили переспросить диспетчера",
          MCP_CHOICE_CALLS >= 2, f"выборов: {MCP_CHOICE_CALLS}")

    # А запрос ПРО СВОДКУ за период по-прежнему может начать сбор: переспрашивать
    # не нужно — вызов-действие здесь и есть то, что требуется.
    await chat.session_create(SessionCreate(periodic=True))
    summary = chat._current_session()
    DISPATCHER_QUEUE[:] = [
        [{"server": EXAMPLE_SERVER, "tool": "start_collection",
          "arguments": {"place": "Москва"}}],
    ]
    reset_mcp()
    await run_chat("присылай сводку погоды за сутки, раз в час",
                   session_id=summary["id"])
    flat_summary = [call for batch in TOOL_CALLS for call in batch]
    check("для сводки за период сбор запускается и данные читаются сразу",
          MCP_CHOICE_CALLS == 2
          and [call["tool"] for call in flat_summary]
          == ["start_collection", "get_data"],
          f"выборов: {MCP_CHOICE_CALLS}, вызовы: {str(flat_summary)[:200]}")
    check("и задача помнит этот сбор",
          bool(workspace_store.mcp_started(summary["dialog"])),
          str(workspace_store.mcp_started(summary["dialog"])))


# ---------------------------------------------------------------------------
# 6c. Форма плана: оформление не дробится на шаги
# ---------------------------------------------------------------------------
def test_plan_shape():
    print("\n[6c] План не дробится на оформление (код-гарантия)")
    # Живой план пользователя: одна таблица была разбита на четыре вызова модели.
    user_plan = [
        "Извлечь данные о погоде за час из отчёта MCP",
        "Сгруппировать значения температуры, осадков и ветра по времени",
        "Сформировать таблицу из 10 сэмплов с метками времени",
        "Вывести таблицу без дополнительной информации",
    ]
    steps, dropped = task_state.compact_steps(user_plan, "сводка погоды за час, в виде таблицы")
    check("план из одного оформления схлопывается в один шаг",
          dropped == 4 and steps == ["сводка погоды за час, в виде таблицы"],
          f"убрано {dropped}: {steps}")
    # Хвостовое оформление убирается, работа остаётся.
    steps, dropped = task_state.compact_steps(
        ["Собрать данные о продажах", "Сгруппировать по регионам", "Вывести таблицу"],
        "отчёт по продажам")
    check("хвостовое оформление убрано, работа осталась",
          dropped == 2 and steps == ["Собрать данные о продажах"], f"{steps}")
    # ДЕЙСТВИЯ с последствиями оформлением НЕ считаются — их терять нельзя.
    for plan, why in (
        (["Собрать данные", "Отправить отчёт клиенту"], "отправка"),
        (["Собрать данные", "Вывести таблицу в файл"], "запись в файл"),
        (["Собрать данные", "Проверить наличие товара и создать заявку"], "создание заявки"),
        (["Посчитать метрики", "Сформировать счёт и передать в бухгалтерию"], "счёт"),
    ):
        steps, dropped = task_state.compact_steps(plan, "запрос")
        check(f"шаг-действие ({why}) не выбрасывается", dropped == 0, f"убрано: {steps}")
    # В промпте планировщика правило тоже есть (просьба к модели).
    prompt = agent_module.PLAN_PROMPT
    check("планировщику сказано не дробить оформление",
          "НЕ ДРОБИ ОФОРМЛЕНИЕ НА ШАГИ" in prompt and "ОДИН шаг" in prompt)
    check("планировщику сказано считать шаги по числу действий",
          "по числу РАЗНЫХ действий" in prompt)


# ---------------------------------------------------------------------------
# 7. Планировщик: пропуски и сбои
# ---------------------------------------------------------------------------
async def test_tick_guards():
    print("\n[7] Планировщик: пропуски повтора")
    await new_project("Проект: пропуски")
    await chat.session_create(SessionCreate(periodic=True))
    session = chat._current_session()
    session_id = session["id"]
    await run_chat("Присылай сводку, раз в час", session_id=session_id)
    await make_due(session_id)

    # Задача на паузе: повтор не начинаем (команда пользователя важнее).
    state = workspace_store.dialog_state(session)
    task_state.pause(state, "проверка: задача на паузе")
    workspace_store.set_dialog_state(session, state)
    started = await periodic_runner.tick()
    check("задача на паузе не повторяется", session_id not in started, str(started))
    check("срок повтора остался просроченным (ждём «Продолжить»)",
          periodic_store.is_due(workspace_store.periodic_meta(session)))
    briefs = {row["session_id"]: row["periodic"] for row in (await chat.periodic_get())["tasks"]}
    check("снимок расписаний объясняет, почему повторов нет (hold=paused)",
          briefs.get(session_id, {}).get("hold") == "paused",
          str(briefs.get(session_id)))

    task_state.resume(state, "проверка: продолжили задачу")
    workspace_store.set_dialog_state(session, state)
    # Задача уже выполняется: повтор не должен лезть в неё второй раз.
    chat._running_sessions.add(session_id)
    started = await periodic_runner.tick()
    check("занятая задача не повторяется параллельно", session_id not in started,
          str(started))
    chat._running_sessions.discard(session_id)

    # Задача удалена, пока повтор шёл: повтор не падает, а завершается ошибкой
    # В САМОЙ задаче (в чате видно, что автозапуск не удался).
    started = await periodic_runner.tick()
    check("свободная задача с наступившим сроком повторяется", session_id in started,
          str(started))
    for _ in range(200):
        if not chat._periodic_running:
            break
        await asyncio.sleep(0.02)
    meta = workspace_store.periodic_meta(session)
    check("успешный повтор ошибки не оставил", not meta["error"], str(meta))

    # Сбой модели: повтор не удался — причина видна и в расписании, и в чате.
    async def failing_llm(*args, **kwargs):
        return "", _metrics()

    original = client.call_llm_async
    client.call_llm_async = failing_llm
    try:
        await make_due(session_id)
        await periodic_runner.tick()
        for _ in range(200):
            if not chat._periodic_running:
                break
            await asyncio.sleep(0.02)
    finally:
        client.call_llm_async = original
    meta = workspace_store.periodic_meta(session)
    check("сбой повтора записан в расписание", bool(meta["error"]), str(meta))
    check("сбой повтора виден в чате задачи",
          any("Автозапуск не удался" in text for text in log_texts(session, "error")),
          str(log_texts(session, "error"))[:200])
    check("после сбоя повтор запланирован снова",
          bool(meta["next_run"]) and not periodic_store.is_due(meta), str(meta))

    # Задача без запроса: повтор пропускается, счётчик не растёт.
    await chat.session_create(SessionCreate(periodic=True))
    empty = chat._current_session()
    await make_due(empty["id"])
    runs_before = workspace_store.periodic_meta(empty)["runs"]
    await periodic_runner.tick()
    for _ in range(200):
        if not chat._periodic_running:
            break
        await asyncio.sleep(0.02)
    check("задача без запроса не засчитывается за повтор",
          workspace_store.periodic_meta(empty)["runs"] == runs_before,
          str(workspace_store.periodic_meta(empty)))
    check("срок повтора у задачи без запроса сдвинут",
          not periodic_store.is_due(workspace_store.periodic_meta(empty)),
          str(workspace_store.periodic_meta(empty)))


# ---------------------------------------------------------------------------
# 8. Цикл повтора: план один раз, «Пауза», новый запрос, отмена
# ---------------------------------------------------------------------------
async def test_chain_behaviour():
    print("\n[8] Цикл повтора: план один раз, «Пауза» и новый запрос")
    await new_project("Проект: цикл повтора")
    await chat.session_create(SessionCreate(periodic=True))
    session = chat._current_session()
    session_id = session["id"]
    plan_calls_start = PLANNER_CALLS
    await run_chat("Сводка погоды в Москве за последние сутки, раз в час",
                   session_id=session_id)
    check("первый запрос построил план один раз",
          PLANNER_CALLS == plan_calls_start + 1
          and len(workspace_store.dialog_state(session).steps) == len(PLAN_STEPS),
          f"вызовов планировщика: {PLANNER_CALLS - plan_calls_start}")

    # Три повтора подряд: план НЕ перестраивается, приёмщик не вызывается, «готово»
    # не ставится — цикл заканчивается признаком «ждём следующего повтора».
    plan_calls = PLANNER_CALLS
    for index in range(3):
        await make_due(session_id)
        await periodic_runner.tick()
        for _ in range(300):
            if not chat._periodic_running:
                break
            await asyncio.sleep(0.02)
        state = workspace_store.dialog_state(session)
        check(f"повтор {index + 1}: цикл пройден без «готово» и без нового плана",
              state.stage == "execution" and state.repeat_ready
              and state.step_index == 0
              and PLANNER_CALLS == plan_calls and REVIEW_CALLS == 0,
              f"{state.stage}/шаг {state.step_index}, планировщик: {PLANNER_CALLS}, "
              f"приёмщик: {REVIEW_CALLS}")
        check(f"повтор {index + 1} засчитан",
              workspace_store.periodic_meta(session)["runs"] == index + 1
              and not workspace_store.periodic_meta(session)["error"],
              str(workspace_store.periodic_meta(session)))

    # «Пауза» во время повтора: цепочка обрывается сразу, повтор НЕ засчитывается
    # (это не сбой) и в чате видно, почему ответа не будет.
    task, _ = session_by_id(session_id)
    state = workspace_store.dialog_state(session)
    task_state.pause(state, "проверка: пользователь поставил паузу")
    workspace_store.set_dialog_state(session, state)
    runs_before = workspace_store.periodic_meta(session)["runs"]
    log_before = len(log_texts(session))
    error, interrupted = await periodic_runner._drive(
        task, session, "Сводка погоды в Москве за последние сутки, раз в час")
    check("повтор на паузе считается прерванным, а не сбоем",
          interrupted and not error, f"error={error!r}, interrupted={interrupted}")
    check("прерванный повтор не крутит цепочку впустую",
          len(log_texts(session)) - log_before <= 2,
          f"записей добавилось: {len(log_texts(session)) - log_before}")
    check("прерванный повтор не засчитан",
          workspace_store.periodic_meta(session)["runs"] == runs_before,
          str(workspace_store.periodic_meta(session)))
    task_state.resume(state, "проверка: продолжили задачу")
    workspace_store.set_dialog_state(session, state)

    # Полный повтор через планировщик после «Продолжить»: прерывание не сбило
    # расписание, повтор выполняется и засчитывается.
    await make_due(session_id)
    started = await periodic_runner.tick()
    check("после «Продолжить» повтор снова запускается", session_id in started,
          str(started))
    for _ in range(300):
        if not chat._periodic_running:
            break
        await asyncio.sleep(0.02)
    check("повтор после прерывания выполнен и засчитан",
          workspace_store.periodic_meta(session)["runs"] == runs_before + 1
          and not workspace_store.periodic_meta(session)["error"],
          str(workspace_store.periodic_meta(session)))

    # НОВЫЙ запрос пользователя: прежний план к нему не подходит — план строится
    # заново, и тоже ОДИН раз (дальше повторы идут по нему).
    ask_calls = PLANNER_CALLS
    await run_chat("Теперь присылай качество воздуха в Москве, раз в час",
                   session_id=session_id)
    state = workspace_store.dialog_state(session)
    check("новый запрос — новый план (один раз)",
          PLANNER_CALLS == ask_calls + 1,
          f"вызовов планировщика: {PLANNER_CALLS - ask_calls}")
    check("подпись плана — от НОВОГО запроса (прежний план больше не подходит)",
          str(session["dialog"].get("plan_signature", {}).get("request", ""))
          .startswith("Теперь присылай"),
          str(session["dialog"].get("plan_signature")))
    # Задача уже в автономном режиме (его включил автозапуск): подтверждения плана
    # не ждём — новый запрос выполняется сразу, а план виден в чате и в полосе.
    check("новый запрос выполняется без ожидания подтверждения",
          state.autonomous is True and state.stage == "execution",
          f"{state.stage}, autonomous={state.autonomous}")
    # Автозапуск подтверждает план сам и работает по сохранённому плану.
    await make_due(session_id)
    await periodic_runner.tick()
    for _ in range(300):
        if not chat._periodic_running:
            break
        await asyncio.sleep(0.02)
    state = workspace_store.dialog_state(session)
    check("повтор по новому запросу идёт без перестройки плана",
          PLANNER_CALLS == ask_calls + 1 and state.stage == "execution"
          and state.step_index == 0,
          f"планировщик: {PLANNER_CALLS}, {state.stage}/шаг {state.step_index}")
    check("запрос задачи обновился",
          workspace_store.periodic_meta(session)["request"].startswith("Теперь присылай"),
          workspace_store.periodic_meta(session)["request"])

    # ОТМЕНА задачи — это остановка периодической задачи: повторять её нельзя, а
    # автозапуск выключается (включить снова можно кнопкой 🔁).
    state = workspace_store.dialog_state(session)
    task_state.cancel(state, "проверка: пользователь отменил задачу")
    workspace_store.set_dialog_state(session, state)
    await make_due(session_id)
    started = await periodic_runner.tick()
    check("отменённая периодическая задача не повторяется", session_id not in started,
          str(started))
    check("автозапуск отменённой задачи выключен",
          workspace_store.periodic_meta(session)["enabled"] is False,
          str(workspace_store.periodic_meta(session)))
    check("в чате сказано, что задача отменена и автозапуск остановлен",
          any("Задача отменена" in text for text in log_texts(session, "debug")),
          str(log_texts(session, "debug"))[-200:])
    briefs = {row["session_id"]: row["periodic"] for row in (await chat.periodic_get())["tasks"]}
    check("снимок расписаний помечает отменённую задачу (hold=cancelled)",
          briefs.get(session_id, {}).get("hold") == "cancelled",
          str(briefs.get(session_id)))

# ---------------------------------------------------------------------------
# 9. Профиль владельца задачи
# ---------------------------------------------------------------------------
async def test_profile_isolation():
    print("\n[9] Повтор идёт в профиле ВЛАДЕЛЬЦА задачи")
    from app.ai import profiles as profile_store

    await new_project("Проект: профили")
    await chat.session_create(SessionCreate(periodic=True))
    session = chat._current_session()
    session_id = session["id"]
    await run_chat("Присылай сводку погоды, раз в час", session_id=session_id)

    current = chat._current_profile_id()
    other = profile_store.create_profile(chat._profiles, {"profile_name": "Второй"})
    check("второй профиль создан и активен", other["id"] == chat._current_profile_id()
          and other["id"] != current, str(other.get("id")))
    task, _ = session_by_id(session_id)
    check("задача осталась за прежним профилем",
          workspace_store.task_owner(task) == current, workspace_store.task_owner(task))

    # Пока открыт ДРУГОЙ профиль, повтор задачи первого всё равно находит её:
    # профиль берётся у владельца задачи (PROFILE_OVERRIDE), а не «активный».
    await make_due(session_id)
    started = await periodic_runner.tick()
    check("повтор задачи неоткрытого профиля запускается", session_id in started,
          str(started))
    for _ in range(200):
        if not chat._periodic_running:
            break
        await asyncio.sleep(0.02)
    state = workspace_store.dialog_state(session)
    check("повтор задачи неоткрытого профиля выполнен (цикл пройден, без «готово»)",
          state.stage == "execution" and state.step_index == 0,
          f"{state.stage}/шаг {state.step_index}: {state.reason}")
    check("активный профиль не подменился",
          chat._current_profile_id() == other["id"], str(chat._current_profile_id()))
    # Возвращаем прежний профиль: проверка не должна оставлять чужой активный
    # профиль (дальше проверок нет, но состояние обязано быть согласованным).
    chat._profiles["active"] = current


# ---------------------------------------------------------------------------
# 9b. Внешние сборы: задача их помнит, а отмена/удаление — останавливают
# ---------------------------------------------------------------------------
async def test_external_collections():
    global START_FAILS
    print("\n[9b] Внешние сборы на сервере: память, отмена, удаление")
    install_mcp_stubs()
    await new_project("Проект: внешние сборы")
    await chat.session_create(SessionCreate(periodic=True))
    await chat.mcp_apply(chat.McpApply(enabled=[EXAMPLE_SERVER]))
    session = chat._current_session()
    session_id = session["id"]
    # Повтор начинает ВНЕШНИЙ СБОР на сервере (start_collection → id).
    reset_mcp()
    MCP_CALLS[:] = [{"server": EXAMPLE_SERVER, "tool": "start_collection",
                     "arguments": {"place": "Москва"}}]
    await run_chat("Присылай сводку по Москве за сутки, раз в час", session_id=session_id)
    started = workspace_store.mcp_started(session["dialog"])
    check("задача запомнила запущенный внешний сбор (с id для отмены)",
          started and started[0]["tool"] == "start_collection"
          and started[0]["stop_tool"] == "stop_collection"
          and started[0]["arguments"] == {"collection_id": COLLECTION_ID},
          str(started))
    check("в чате сказано, что сбор остановится вместе с задачей",
          any("запустила внешний сбор" in text for text in log_texts(session, "debug")),
          str(log_texts(session, "debug"))[-200:])
    # Модель видит ФАКТ действий отдельной строкой блока: без вызова запуска она
    # не может «зарегистрировать наблюдение» словами.
    block = mcp_store.block([dict(item, action=item.get("action")) for item
                             in workspace_store.dialog_mcp(session["dialog"])["results"]])
    check("в блоке данных есть строка про действия (ВЫПОЛНЕНО)",
          "ДЕЙСТВИЯ НА СЕРВЕРАХ В ЭТОМ ЗАПРОСЕ" in block
          and "ВЫПОЛНЕНО: start_collection" in block, block[:400])

    # Повтор: сбор УЖЕ идёт — модель видит это и не начинает его заново.
    reset_mcp()
    MCP_CALLS[:] = [{"server": EXAMPLE_SERVER, "tool": "get_data",
                     "arguments": {"place": "Москва"}}]
    PAYLOADS.clear()
    await make_due(session_id)
    await periodic_runner.tick()
    for _ in range(300):
        if not chat._periodic_running:
            break
        await asyncio.sleep(0.02)
    payload = PAYLOADS[0] if PAYLOADS else ""
    check("повтор видит уже идущий сбор (не начинает его заново)",
          "УЖЕ ИДЁТ" in payload and COLLECTION_ID in payload, payload[:200])
    check("внешний сбор не запускался повторно",
          all(call.get("tool") != "start_collection" for call in TOOL_CALLS[0])
          if TOOL_CALLS else True, str(TOOL_CALLS[:1]))

    # ОТМЕНА задачи: внешний сбор на сервере останавливается.
    reset_mcp()
    await chat.state_cancel()
    check("отмена задачи остановила внешний сбор",
          (EXAMPLE_SERVER, "stop_collection") in _calls_of(TOOL_CALLS),
          str(TOOL_CALLS))
    check("после отмены задача больше не помнит сбор",
          not workspace_store.mcp_started(session["dialog"]),
          str(workspace_store.mcp_started(session["dialog"])))
    check("в чате видно, что внешние сборы остановлены",
          any("Внешние сборы задачи" in text for text in log_texts(session, "debug")),
          str(log_texts(session, "debug"))[-200:])

    # УДАЛЕНИЕ задачи: то же самое (сбор останавливается ДО удаления данных).
    await chat.session_create(SessionCreate(periodic=True))
    second = chat._current_session()
    reset_mcp()
    MCP_CALLS[:] = [{"server": EXAMPLE_SERVER, "tool": "start_collection",
                     "arguments": {"place": "Тула"}}]
    await run_chat("Присылай сводку по Туле, раз в час", session_id=second["id"])
    check("вторая задача тоже помнит свой сбор",
          bool(workspace_store.mcp_started(second["dialog"])),
          str(workspace_store.mcp_started(second["dialog"])))
    reset_mcp()
    await chat.session_delete(second["id"])
    check("удаление задачи остановило её внешний сбор",
          (EXAMPLE_SERVER, "stop_collection") in _calls_of(TOOL_CALLS),
          str(TOOL_CALLS))

    # УДАЛЕНИЕ ПРОЕКТА: сборы всех его задач останавливаются.
    await chat.session_create(SessionCreate(periodic=True))
    third = chat._current_session()
    reset_mcp()
    MCP_CALLS[:] = [{"server": EXAMPLE_SERVER, "tool": "start_collection",
                     "arguments": {"place": "Сочи"}}]
    await run_chat("Присылай сводку по Сочи, раз в час", session_id=third["id"])
    current_task = chat._current_task()
    reset_mcp()
    await chat.task_delete(current_task["id"])
    check("удаление ПРОЕКТА остановило внешние сборы его задач",
          (EXAMPLE_SERVER, "stop_collection") in _calls_of(TOOL_CALLS),
          str(TOOL_CALLS))

    # Обычная (одноразовая) задача: «Отмена» останавливает заход, но НЕ удаляет
    # накопленные на сервере данные — это было бы потерей без спроса.
    await new_project("Проект: обычная задача со сбором")
    await chat.session_create(SessionCreate())
    await chat.mcp_apply(chat.McpApply(enabled=[EXAMPLE_SERVER]))
    plain = chat._current_session()
    reset_mcp()
    MCP_CALLS[:] = [{"server": EXAMPLE_SERVER, "tool": "start_collection",
                     "arguments": {"place": "Пермь"}}]
    await run_chat("Собери данные по Перми", session_id=plain["id"])
    check("обычная задача тоже помнит запущенный сбор",
          bool(workspace_store.mcp_started(plain["dialog"])),
          str(workspace_store.mcp_started(plain["dialog"])))
    reset_mcp()
    await chat.state_cancel()
    check("отмена ОБЫЧНОЙ задачи внешний сбор НЕ трогает",
          not _calls_of(TOOL_CALLS)
          and bool(workspace_store.mcp_started(plain["dialog"])),
          f"вызовы: {TOOL_CALLS}")


    # ДЕЙСТВИЯ В БЛОКЕ ДАННЫХ: модель не должна «регистрировать» словами то,
    # чего вызов не делал. Проверяем на отдельном проекте, чтобы не сбивать
    # текущие задачи предыдущих проверок.
    await new_project("Проект: только чтение")
    await chat.session_create(SessionCreate(periodic=True))
    await chat.mcp_apply(chat.McpApply(enabled=[EXAMPLE_SERVER]))
    read_only = chat._current_session()
    reset_mcp()
    MCP_CALLS[:] = [{"server": EXAMPLE_SERVER, "tool": "get_data",
                     "arguments": {"place": "Москва"}}]
    await run_chat("Присылай данные по Москве, раз в час", session_id=read_only["id"])
    block = mcp_store.block(workspace_store.dialog_mcp(read_only["dialog"])["results"])
    check("блок данных прямо говорит, что действий не было",
          "не выполнялись" in block, block[:400])
    check("задача не помнит никаких сборов",
          not workspace_store.mcp_started(read_only["dialog"]))
    # НЕУДАЧНЫЙ запуск сбора: действие считается НЕ выполненным, и задача его не
    # помнит (иначе «отмена» вызывала бы остановку несуществующего сбора).
    reset_mcp()
    MCP_CALLS[:] = [{"server": EXAMPLE_SERVER, "tool": "start_collection",
                     "arguments": {"place": "Тула"}}]
    START_FAILS = True
    try:
        await run_chat("Собери данные по Туле", session_id=read_only["id"])
    finally:
        START_FAILS = False
    block = mcp_store.block(workspace_store.dialog_mcp(read_only["dialog"])["results"])
    check("неудачный запуск показан как НЕ ВЫПОЛНЕННЫЙ",
          "НЕ ВЫПОЛНЕНО" in block, block[:400])
    check("неудачный запуск не запоминается как сбор",
          not workspace_store.mcp_started(read_only["dialog"]))
    await chat.session_delete(read_only["id"])
    await chat.task_delete(chat._current_task()["id"])


def _calls_of(tool_calls):
    """Пары (сервер, инструмент) из всех выполненных вызовов."""
    pairs = []
    for batch in tool_calls:
        for call in batch:
            pairs.append((call.get("server"), call.get("tool")))
    return pairs


# ---------------------------------------------------------------------------
# 10. Связь с внешними инструментами: только через MCP, без имён инструментов
# ---------------------------------------------------------------------------
def test_no_tool_coupling():
    print("\n[10] Повтор не завязан на конкретные инструменты")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Имена инструментов и серверов — данные РЕЕСТРА и ОБЪЯВЛЕНИЯ СЕРВЕРА: в коде
    # повторов их быть не должно, иначе замена или отключение инструмента (погода
    # сегодня есть, завтра нет) ломала бы периодические задачи.
    forbidden = ("get_air_quality", "get_current_weather", "get_weather_forecast",
                 "start_weather_watch", "stop_weather_watch", "list_weather_watches",
                 "get_weather_watch_report", "geocode_location",
                 "open_meteo", "open-meteo", " weather")
    for name in ("app/ai/periodic.py", "app/periodic_runner.py"):
        with open(os.path.join(root, name), encoding="utf-8") as fh:
            text = fh.read().lower()
        found = [word for word in forbidden if word in text]
        check(f"{name}: без имён конкретных инструментов и серверов",
              not found, f"найдено: {found}")
    # Диспетчер внешних инструментов тоже не должен знать имён по памяти: он
    # выбирает из того, что объявил сервер (правило 6 — про данные ЗА ПЕРИОД).
    prompt = mcp_store.TOOLS_PROMPT.lower()
    found = [word for word in forbidden if word in prompt]
    check("подсказка диспетчера не называет конкретных инструментов",
          not found, f"найдено: {found}")
    check("подсказка диспетчера требует брать имена из списка сервера",
          "только из" in prompt and "меняется" in prompt,
          prompt[-260:])


# ---------------------------------------------------------------------------
# 11. Приложение: планировщик поднимается вместе с приложением
# ---------------------------------------------------------------------------
def test_wiring():
    print("\n[11] Планировщик подключён к приложению")
    source = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "main.py"), encoding="utf-8").read()
    check("main.py запускает планировщик", "periodic_runner.start()" in source)
    check("main.py останавливает планировщик", "periodic_runner.stop()" in source)
    check("планировщик выключается настройкой",
          hasattr(config, "PERIODIC_ENABLED")
          and hasattr(config, "PERIODIC_TICK_SECONDS"))
    check("периодическая задача знает свой профиль-владельца",
          "profile_override" in open(
              os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "app", "periodic_runner.py"), encoding="utf-8").read())


async def main():
    test_parse()
    test_storage()
    test_log_times()
    await test_routes()
    await test_text_period()
    await test_repeat()
    await test_repeat_fresh_mcp()
    await test_read_instead_of_start()
    test_plan_shape()
    await test_tick_guards()
    await test_chain_behaviour()
    await test_profile_isolation()
    await test_external_collections()
    test_no_tool_coupling()
    test_wiring()
    print("\nИтог: " + ("ПРОВАЛЕНО проверок: " + str(len(FAILURES))
                        if FAILURES else "все проверки пройдены"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(asyncio.get_event_loop().run_until_complete(main()))
