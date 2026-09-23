"""Самопроверка MCP — внешних инструментов агента (режим «AI-агент»).

Запуск (сеть, Node.js и API-ключ НЕ нужны):

    ./venv/bin/python tools/check_mcp.py

Скрипт проверяет:

  [1] клиент MCP по stdio — против НАСТОЯЩЕГО сервера, но написанного на Python
      во временном каталоге: рукопожатие initialize, tools/list, tools/call,
      текст результата, ошибка инструмента (isError), сбой сервера (процесс
      завершился, таймаут) и понятная причина недоступности;
  [2] разбор ответа модели о вызовах (parse_calls): русские ключи, отбрасывание
      придуманного инструмента, предел вызовов на запрос;
  [3] хранение настройки ПРОЕКТА: включённые серверы живут в task["mcp"] и
      ПЕРЕЖИВАЮТ запись файла (иначе ключ молча терялся бы при первом _persist);
  [4] маршруты GET/POST /api/agent/mcp: снимок серверов с инструментами,
      400 без проекта, фильтр неизвестных id, сброс данных прежнего запроса;
  [5] работу в диалоге: служебный выбор инструментов, вызов, системный блок
      «ДАННЫЕ MCP» в контексте планировщика и ответа, повторное использование
      данных на ШАГАХ плана (без нового служебного вызова) и выключенный MCP
      (ни вызовов, ни расхода).

Рабочие данные не трогаются: workspace, история агента и профили пишутся во
временный каталог (переменные AGENT_*_FILE выставляются ДО импорта chat).
"""

import asyncio
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Изоляция данных: всё пишем во временный каталог ------------------------
_TMP = tempfile.mkdtemp(prefix="mcp-check-")
os.environ["AGENT_WORKSPACE_FILE"] = os.path.join(_TMP, "workspace.json")
os.environ["AGENT_MEMORY_FILE"] = os.path.join(_TMP, "agent_memory.json")
os.environ["AGENT_PROFILES_FILE"] = os.path.join(_TMP, "profiles.json")
# Каталог серверов MCP — тоже временный: настоящие серверы (Node) здесь не
# запускаются, их место занимает тестовый сервер на Python.
os.environ["MCP_SERVERS_DIR"] = os.path.join(_TMP, "mcp_servers")

from app.ai import client, mcp as mcp_store  # noqa: E402
from app.ai import workspace as workspace_store  # noqa: E402
from app.routers import chat  # noqa: E402
from app.schemas import ChatMessage, McpApply  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    """Одна проверка: печатает результат и копит провалы."""
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# Тестовый MCP-сервер: НАСТОЯЩИЙ протокол (JSON-RPC 2.0 по stdio), но на Python
# ---------------------------------------------------------------------------
# Так проверяется именно клиент: рукопожатие, список инструментов, вызов,
# ошибки. Ни Node.js, ни сеть для этого не нужны.
FAKE_SERVER = r'''
import json
import sys

TOOLS = [
    {
        "name": "get_weather",
        "title": "Погода сейчас",
        "description": "Текущая погода в городе",
        "inputSchema": {"type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"]},
    },
    {
        "name": "boom",
        "title": "Сломанный инструмент",
        "description": "Всегда возвращает ошибку",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def send(payload):
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except ValueError:
            continue
        method = message.get("method")
        request_id = message.get("id")
        if method == "initialize":
            if request_id is None:
                continue
            send({"jsonrpc": "2.0", "id": request_id, "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake-mcp", "version": "0.1"}}})
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": request_id,
                  "result": {"tools": TOOLS}})
        elif method == "tools/call":
            params = message.get("params") or {}
            name = params.get("name")
            args = params.get("arguments") or {}
            if name == "get_weather":
                sys.stderr.write("[fake] get_weather called\n")
                send({"jsonrpc": "2.0", "id": request_id, "result": {
                    "content": [{"type": "text",
                                 "text": "Погода в " + str(args.get("city", "?"))
                                         + ": +14, облачно"}]}})
            elif name == "boom":
                send({"jsonrpc": "2.0", "id": request_id, "result": {
                    "isError": True,
                    "content": [{"type": "text", "text": "источник недоступен"}]}})
            else:
                send({"jsonrpc": "2.0", "id": request_id,
                      "error": {"code": -32602, "message": "unknown tool"}})
        elif method == "notifications/initialized":
            continue
        elif request_id is not None:
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32601, "message": "unknown method"}})


main()
'''

# Сервер, который падает сразу: проверяем понятную ошибку вместо зависания.
DEAD_SERVER = "import sys\nsys.exit(3)\n"
# Сервер, который молчит: проверяем таймаут (в тесте он выставлен в 1 секунду).
SILENT_SERVER = "import time\ntime.sleep(30)\n"

# Подмена реестра серверов: id "fake" — тестовый сервер на Python.
FAKE_ID = "fake"


def _write_fake_servers() -> None:
    """Кладёт тестовые серверы в каталог MCP (он у нас временный)."""
    directory = mcp_store.servers_dir()
    os.makedirs(directory, exist_ok=True)
    for name, body in (("fake_server.py", FAKE_SERVER),
                       ("dead_server.py", DEAD_SERVER),
                       ("silent_server.py", SILENT_SERVER)):
        with open(os.path.join(directory, name), "w", encoding="utf-8") as fh:
            fh.write(body)


def use_fake_registry(script: str = "fake_server.py") -> None:
    """Переводит клиент MCP на тестовый сервер (настоящий реестр отключается)."""
    entry = {
        "id": FAKE_ID,
        "name": "Тестовый MCP",
        "description": "Сервер проверки",
        "source": "локальный тест",
        "command": [sys.executable, os.path.join(mcp_store.servers_dir(), script)],
    }
    mcp_store.SERVERS = [entry]
    mcp_store.SERVER_IDS = [FAKE_ID]
    mcp_store.forget()


def restore_registry() -> None:
    """Возвращает настоящий реестр серверов проекта (погода, курсы, крипта)."""
    import importlib

    reloaded = importlib.reload(mcp_store)
    mcp_store.SERVERS = reloaded.SERVERS
    mcp_store.SERVER_IDS = reloaded.SERVER_IDS


# ---------------------------------------------------------------------------
# Заглушка LLM: план, ответ, проверка, арбитр инвариантов и выбор инструментов
# ---------------------------------------------------------------------------
PLAN_STEPS = ["Узнать погоду", "Написать ответ"]
ANSWER = "Ответ модели по текущему шагу."
REVIEW = {"verdict": "ok", "step": 0, "comment": "результат соответствует плану"}
INVARIANTS_ANALYSIS = {"вердикт": "clear", "объяснение": "", "варианты": []}
# Вызовы внешних инструментов, которые «выбирает» модель: список словарей.
MCP_CALLS = [{"server": FAKE_ID, "tool": "get_weather",
              "arguments": {"city": "Москва"}}]
MCP_CHOICE_CALLS = 0
# Все сообщения всех вызовов: по ним видно, что уходит модели (в том числе
# системный блок «ДАННЫЕ MCP»).
CALL_MESSAGES = []
# user-части служебного выбора инструментов и вызова планировщика.
MCP_PAYLOADS = []
PLANNER_PAYLOADS = []
ANSWER_PAYLOADS = []
# Счётчики обращений к серверам MCP (реальные процессы не запускаем).
DISCOVER_CALLS = 0
TOOL_CALLS = []


def _metrics(prompt=20, completion=10):
    return {"model": "stub", "elapsed_seconds": 0.01, "prompt_tokens": prompt,
            "completion_tokens": completion, "total_tokens": prompt + completion}


async def fake_call_llm_async(*args, **kwargs):
    """Подмена client.call_llm_async: всё локально, без сети."""
    messages = kwargs.get("messages") or []
    system = str(messages[0].get("content") or "") if messages else ""
    user = str(messages[-1].get("content") or "") if messages else ""
    CALL_MESSAGES.append(list(messages))
    if system.startswith("Ты — планировщик"):
        PLANNER_PAYLOADS.append(user)
        return json.dumps({"steps": list(PLAN_STEPS)}, ensure_ascii=False), _metrics(30, 15)
    if system.startswith("Ты — приёмщик"):
        return json.dumps(REVIEW, ensure_ascii=False), _metrics(40, 8)
    if system.startswith("Ты — арбитр инвариантов"):
        return json.dumps(INVARIANTS_ANALYSIS, ensure_ascii=False), _metrics(35, 20)
    if system.startswith("Ты — диспетчер внешних инструментов"):
        global MCP_CHOICE_CALLS
        MCP_CHOICE_CALLS += 1
        MCP_PAYLOADS.append(user)
        return (json.dumps({"calls": list(MCP_CALLS)}, ensure_ascii=False),
                _metrics(45, 18))
    ANSWER_PAYLOADS.append(user)
    return ANSWER, _metrics()


client.call_llm_async = fake_call_llm_async


def install_mcp_stubs() -> None:
    """Подменяет обращения к серверам: процессы MCP в проверке не запускаются."""

    async def fake_discover(ids=None, force=False):
        global DISCOVER_CALLS
        DISCOVER_CALLS += 1
        return [{
            "id": FAKE_ID, "ok": True, "error": "",
            "server_name": "fake-mcp", "server_version": "0.1",
            "tools": [
                {"name": "get_weather", "title": "Погода сейчас",
                 "description": "Текущая погода в городе",
                 "schema": {"type": "object",
                            "properties": {"city": {"type": "string"}}}},
            ],
        }]

    async def fake_run_calls(calls):
        TOOL_CALLS.append(list(calls))
        return [{
            "server": FAKE_ID, "server_name": "Тестовый MCP", "source": "локальный тест",
            "tool": "get_weather", "arguments": {"city": "Москва"},
            "ok": True, "text": "Погода в Москва: +14, облачно", "error": "",
        }]

    mcp_store.async_discover = fake_discover
    mcp_store.async_run_calls = fake_run_calls
    chat.mcp_store.async_discover = fake_discover
    chat.mcp_store.async_run_calls = fake_run_calls


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


def texts(events, kind):
    return [e.get("text", "") for e in events if e.get("type") == kind]


def stage_of(events):
    states = [e["state"] for e in events if e.get("type") == "state"]
    return states[-1] if states else None


def system_texts():
    """Содержимое системных сообщений всех вызовов модели (одной строкой)."""
    parts = []
    for messages in CALL_MESSAGES:
        for message in messages:
            if message.get("role") == "system":
                parts.append(str(message.get("content") or ""))
    return "\n".join(parts)


def all_context():
    """ВСЁ, что уходило модели (любая роль): блок MCP может идти и в user-части
    служебного вызова (планировщик получает слои и данные текстом)."""
    return "\n".join(str(message.get("content") or "")
                     for messages in CALL_MESSAGES for message in messages)


def reset_calls():
    CALL_MESSAGES.clear()
    MCP_PAYLOADS.clear()
    PLANNER_PAYLOADS.clear()
    ANSWER_PAYLOADS.clear()
    TOOL_CALLS.clear()
    global MCP_CHOICE_CALLS, DISCOVER_CALLS
    MCP_CHOICE_CALLS = 0
    DISCOVER_CALLS = 0


# ---------------------------------------------------------------------------
# 1. Клиент MCP по stdio: настоящий сервер, настоящий протокол
# ---------------------------------------------------------------------------
def test_client():
    print("\n[1] Клиент MCP по stdio (тестовый сервер на Python)")
    use_fake_registry()
    found = mcp_store.discover(FAKE_ID, force=True)
    check("сервер запустился и объявил инструменты", found.get("ok"), str(found)[:200])
    check("имя сервера из рукопожатия", found.get("server_name") == "fake-mcp",
          str(found.get("server_name")))
    names = [tool["name"] for tool in found.get("tools") or []]
    check("инструменты прочитаны", names == ["get_weather", "boom"], str(names))
    check("описание инструмента сохранено",
          "погода" in (found["tools"][0]["description"] or "").lower(),
          str(found["tools"][0]))
    check("схема аргументов сохранена",
          isinstance(found["tools"][0]["schema"], dict)
          and "city" in json.dumps(found["tools"][0]["schema"]),
          str(found["tools"][0]["schema"])[:120])

    result = mcp_store.call_tool(FAKE_ID, "get_weather", {"city": "Москва"})
    check("вызов инструмента вернул текст", result["ok"] and "Москва" in result["text"],
          str(result)[:200])

    broken = mcp_store.call_tool(FAKE_ID, "boom", {})
    check("ошибка инструмента (isError) не выдаёт себя за данные",
          not broken["ok"] and "источник недоступен" in broken["error"],
          str(broken)[:200])

    unknown = mcp_store.call_tool(FAKE_ID, "нет_такого", {})
    check("ошибка протокола возвращается понятным текстом",
          not unknown["ok"] and "unknown tool" in unknown["error"],
          str(unknown)[:200])

    # Кэш: второй discover не поднимает процесс заново (иначе на каждый запрос
    # агента запускался бы лишний сервер).
    again = mcp_store.discover(FAKE_ID)
    check("повторное обнаружение берётся из кэша",
          again.get("tools") and again["tools"][0]["name"] == "get_weather")
    mcp_store.forget(FAKE_ID)
    check("сброс кэша работает", mcp_store._cached(FAKE_ID) is None)

    # Сбой сервера: понятная причина, а не исключение наружу.
    use_fake_registry("dead_server.py")
    dead = mcp_store.discover(FAKE_ID, force=True)
    check("упавший сервер не бросает исключение",
          dead.get("ok") is False and bool(dead.get("error")), str(dead)[:200])
    check("причина сбоя содержит код выхода",
          "кодом 3" in str(dead.get("error")), str(dead.get("error"))[:200])

    # Молчащий сервер: таймаут (в проверке — 1 секунда вместо 25).
    use_fake_registry("silent_server.py")
    saved_init, saved_list = mcp_store.INIT_TIMEOUT, mcp_store.LIST_TIMEOUT
    mcp_store.INIT_TIMEOUT = mcp_store.LIST_TIMEOUT = 1.0
    silent = mcp_store.discover(FAKE_ID, force=True)
    mcp_store.INIT_TIMEOUT, mcp_store.LIST_TIMEOUT = saved_init, saved_list
    check("молчащий сервер отсекается таймаутом",
          silent.get("ok") is False and "не ответил" in str(silent.get("error")),
          str(silent)[:200])

    # Причина недоступности видна БЕЗ запуска процесса: так модалка объясняет
    # «почему не работает», не поднимая сервер.
    missing = {"id": "no", "name": "Нет", "description": "",
               "command": ["/nonexistent/node-binary", "x.mjs"]}
    check("отсутствующий запускаемый файл виден заранее",
          "не найден запускаемый файл" in mcp_store.availability_error(missing),
          mcp_store.availability_error(missing))
    restore_registry()


# ---------------------------------------------------------------------------
# 2. Разбор ответа модели о вызовах
# ---------------------------------------------------------------------------
def test_parse_calls():
    print("\n[2] Разбор выбора инструментов (ответ модели)")
    tools = [{"server": "weather", "tool": "get_weather"}]
    parsed = mcp_store.parse_calls(
        '{"calls": [{"server": "weather", "tool": "get_weather", '
        '"arguments": {"city": "Казань"}}]}', tools)
    check("обычный ответ разобран",
          parsed == [{"server": "weather", "tool": "get_weather",
                      "arguments": {"city": "Казань"}}], str(parsed))
    check("ответ в ```-ограждении разобран",
          mcp_store.parse_calls('```json\n{"calls": [{"server": "weather", '
                                '"tool": "get_weather", "arguments": {}}]}\n```',
                                tools) != [])
    russian = mcp_store.parse_calls(
        '{"вызовы": [{"сервер": "weather", "инструмент": "get_weather", '
        '"аргументы": {"city": "Сочи"}}]}', tools)
    check("русские ключи приняты", russian and russian[0]["arguments"] == {"city": "Сочи"},
          str(russian))
    invented = mcp_store.parse_calls(
        '{"calls": [{"server": "weather", "tool": "сделай_всё_сам", "arguments": {}}]}',
        tools)
    check("придуманный инструмент отброшен", invented == [], str(invented))
    many = mcp_store.parse_calls(json.dumps({"calls": [
        {"server": "weather", "tool": "get_weather", "arguments": {"city": str(i)}}
        for i in range(10)]}), tools)
    check("вызовов не больше предела",
          len(many) == mcp_store.MAX_CALLS_PER_REQUEST, str(len(many)))
    check("пустой ответ — «данные не нужны»",
          mcp_store.parse_calls('{"calls": []}', tools) == []
          and mcp_store.parse_calls("мусор", tools) == [])


# ---------------------------------------------------------------------------
# 3. Хранение: настройка проекта и данные запроса
# ---------------------------------------------------------------------------
def test_storage():
    print("\n[3] Хранение настройки MCP проекта")
    task = {"id": "t-1", "name": "Проект", "sessions": [], "active_session": None,
            "working": []}
    check("по умолчанию MCP выключен", workspace_store.mcp_enabled(task) == [])
    check("включить можно только известный сервер",
          workspace_store.set_mcp_enabled(task, ["weather", "чужой"]) == ["weather"],
          str(task.get("mcp")))
    check("повторы id не дублируются",
          workspace_store.set_mcp_enabled(task, ["crypto", "crypto", "weather"])
          == ["crypto", "weather"], str(task.get("mcp")))

    # ЛОВУШКА: ключ, не добавленный в нормализацию, молча теряется при записи.
    normalized = workspace_store._normalize_task(dict(task))
    check("настройка переживает нормализацию задачи (не теряется при записи)",
          normalized.get("mcp") == {"enabled": ["crypto", "weather"]},
          str(normalized.get("mcp")))
    check("битая настройка не ломает задачу",
          workspace_store._normalize_mcp("мусор") == {"enabled": []}
          and workspace_store._normalize_mcp(["weather"]) == {"enabled": ["weather"]})

    dialog = workspace_store.empty_dialog("s-1")
    check("данных MCP у нового диалога нет", workspace_store.dialog_mcp(dialog) == {})
    workspace_store.set_dialog_mcp(dialog, "sig-1", "погода в Москве", [{
        "server": "weather", "tool": "get_weather", "arguments": {"city": "Москва"},
        "ok": True, "text": "Погода в Москва: +14", "server_name": "Погода",
    }])
    check("данные запроса сохранены",
          workspace_store.dialog_mcp(dialog)["signature"] == "sig-1")
    restored = workspace_store.normalize_dialog(dialog, "s-1")
    kept = workspace_store.dialog_mcp(restored)
    check("данные MCP диалога переживают нормализацию (и запись в файл)",
          kept.get("signature") == "sig-1"
          and kept["results"][0]["text"].startswith("Погода"), str(kept)[:200])
    check("пустая подпись не сохраняется (данные сброшены)",
          workspace_store._normalize_dialog_mcp({"signature": "", "results": []}) == {})


# ---------------------------------------------------------------------------
# 4. Маршруты /api/agent/mcp
# ---------------------------------------------------------------------------
async def test_routes():
    print("\n[4] Маршруты MCP: снимок серверов и применение набора")
    install_mcp_stubs()
    use_fake_registry()
    # Реестр серверов подменён на тестовый — маршруты читают его же.
    await chat.task_create(chat.TaskCreate(name="MCP-проект"))
    await chat.session_create()

    view = await chat.mcp_get()
    check("снимок содержит сервер проекта",
          [item["id"] for item in view["servers"]] == [FAKE_ID], str(view)[:200])
    server = view["servers"][0]
    check("в снимке есть название и краткое описание",
          bool(server["name"]) and bool(server["description"]), str(server)[:200])
    check("в снимке есть инструменты сервера (все, что объявил сервер)",
          [tool["name"] for tool in server["tools"]] == ["get_weather", "boom"],
          str(server["tools"]))
    check("сервер показан доступным", server["available"] is True)
    check("пока ничего не включено", view["enabled"] == [])
    check("счётчики снимка заполнены",
          view["counts"] == {"servers": 1, "enabled": 0, "tools": 2, "available": 1},
          str(view["counts"]))
    check("снимок привязан к проекту", view["project_id"] == chat._current_task()["id"])

    applied = await chat.mcp_apply(McpApply(enabled=[FAKE_ID, "чужой-сервер"]))
    check("неизвестный id не включается", applied["enabled"] == [FAKE_ID],
          str(applied["enabled"]))
    check("настройка записалась в проект",
          workspace_store.mcp_enabled(chat._current_task()) == [FAKE_ID])
    check("счётчик включённых в снимке", applied["counts"]["enabled"] == 1,
          str(applied["counts"]))

    task = chat._current_task()
    task["sessions"][0]["dialog"]["mcp"] = {"signature": "старый", "request": "старое",
                                            "results": []}
    await chat.mcp_apply(McpApply(enabled=[]))
    check("выключение сбрасывает данные прежнего запроса",
          workspace_store.dialog_mcp(task["sessions"][0]["dialog"]) == {})
    check("выключенный набор сохранён", workspace_store.mcp_enabled(task) == [])

    # Проекта нет — включать нечего.
    saved_tasks = chat._workspace["tasks"]
    saved_active = dict(chat._workspace.get("active_tasks") or {})
    chat._workspace["tasks"] = []
    chat._workspace["active_tasks"] = {}
    try:
        await chat.mcp_apply(McpApply(enabled=[FAKE_ID]))
        check("без проекта — 400", False, "(ошибки не было)")
    except Exception as exc:  # noqa: BLE001
        check("без проекта — 400", getattr(exc, "status_code", None) == 400,
              str(getattr(exc, "detail", exc)))
    chat._workspace["tasks"] = saved_tasks
    chat._workspace["active_tasks"] = saved_active


# ---------------------------------------------------------------------------
# 5. Работа в диалоге агента
# ---------------------------------------------------------------------------
async def fresh_project():
    """Новый проект с включённым тестовым MCP и одним диалогом."""
    await chat.task_create(chat.TaskCreate(name="Проект с MCP"))
    await chat.session_create()
    await chat.mcp_apply(McpApply(enabled=[FAKE_ID]))


async def test_dialog():
    print("\n[5] Диалог агента: выбор инструментов, данные в контексте, шаги")
    install_mcp_stubs()
    use_fake_registry()
    await fresh_project()

    # 5.1 Запрос с включённым MCP: служебный выбор + вызов + блок данных.
    reset_calls()
    MCP_CALLS[:] = [{"server": FAKE_ID, "tool": "get_weather",
                     "arguments": {"city": "Москва"}}]
    events = await run_chat("Какая сейчас погода в Москве?")
    check("служебный выбор инструментов вызван один раз", MCP_CHOICE_CALLS == 1,
          f"вызовов: {MCP_CHOICE_CALLS}")
    check("модели показан список инструментов сервера",
          "get_weather" in (MCP_PAYLOADS[0] if MCP_PAYLOADS else ""),
          (MCP_PAYLOADS[0] if MCP_PAYLOADS else "")[:200])
    check("вызов инструмента выполнен", len(TOOL_CALLS) == 1
          and TOOL_CALLS[0][0]["tool"] == "get_weather", str(TOOL_CALLS)[:200])
    check("в чате видно, что сделал MCP",
          any("MCP" in t for t in texts(events, "debug")),
          str(texts(events, "debug"))[-200:])
    context = all_context()
    check("данные MCP уходят в модель (блоком системного промпта и в служебные вызовы)",
          "ДАННЫЕ MCP" in system_texts() or "ДАННЫЕ MCP" in context,
          context[-300:])
    check("планировщик видит данные MCP",
          any("ДАННЫЕ MCP" in payload for payload in PLANNER_PAYLOADS),
          str(PLANNER_PAYLOADS)[:200])
    dialog = chat._current_session()["dialog"]
    stored = workspace_store.dialog_mcp(dialog)
    check("данные сохранены в диалоге под подписью запроса",
          stored.get("signature") and stored["results"][0]["tool"] == "get_weather",
          str(stored)[:200])
    check("расход выбора учтён как служебный (вид mcp)",
          any((e.get("usage") or {}).get("service", {}).get("mcp")
              for e in events if e.get("type") == "done"),
          str([e.get("usage") for e in events if e.get("type") == "done"])[:300])

    # 5.2 Подтверждение плана и шаг: данные берутся из диалога, БЕЗ нового выбора.
    reset_calls()
    await run_chat("ок")
    check("подтверждение плана не выбирает инструменты заново", MCP_CHOICE_CALLS == 0,
          f"вызовов выбора: {MCP_CHOICE_CALLS}")
    check("шаг выполняется по сохранённым данным",
          "Погода в Москва" in system_texts(), system_texts()[-200:])

    reset_calls()
    step_events = await run_chat("", continue_step=True)
    check("шаг плана не оплачивает выбор инструментов повторно",
          MCP_CHOICE_CALLS == 0 and not TOOL_CALLS,
          f"выбор: {MCP_CHOICE_CALLS}, вызовы: {len(TOOL_CALLS)}")
    check("шаг плана видит те же данные MCP",
          "Погода в Москва" in system_texts(), system_texts()[-200:])
    check("шаг плана выполнен", bool(texts(step_events, "bot")),
          str(texts(step_events, "bot"))[:120])

    # 5.3 Новый запрос — новая подпись: инструменты выбираются заново.
    reset_calls()
    MCP_CALLS[:] = []
    events = await run_chat("А теперь просто поздоровайся")
    check("новый запрос снова спрашивает модель о вызовах", MCP_CHOICE_CALLS == 1,
          f"вызовов: {MCP_CHOICE_CALLS}")
    check("«данные не нужны» — вызовов инструментов нет", not TOOL_CALLS,
          str(TOOL_CALLS)[:120])
    check("прежние данные не подставляются новому запросу",
          "Погода в Москва" not in system_texts(), system_texts()[-200:])

    # 5.3а Правка запроса до подтверждения плана — тоже НОВЫЙ запрос: агент не
    # должен отвечать по данным прежнего (иначе «лучше курс доллара» получил бы
    # погоду).
    await fresh_project()
    reset_calls()
    MCP_CALLS[:] = [{"server": FAKE_ID, "tool": "get_weather",
                     "arguments": {"city": "Москва"}}]
    await run_chat("Какая сейчас погода в Москве?")
    check("первый запрос собрал данные", MCP_CHOICE_CALLS == 1 and len(TOOL_CALLS) == 1,
          f"выбор: {MCP_CHOICE_CALLS}, вызовы: {len(TOOL_CALLS)}")
    reset_calls()
    await run_chat("нет, лучше курс доллара")
    check("правка запроса до подтверждения плана выбирает инструменты заново",
          MCP_CHOICE_CALLS == 1, f"выборов: {MCP_CHOICE_CALLS}")
    check("правка запроса не тянет данные прежнего запроса",
          "Погода в Москва" not in system_texts(), system_texts()[-200:])

    # 5.3б Фраза управления («работай автономно») запросом НЕ является: данные
    # прежнего запроса задачи должны сохраниться для шагов плана.
    await fresh_project()
    reset_calls()
    MCP_CALLS[:] = [{"server": FAKE_ID, "tool": "get_weather",
                     "arguments": {"city": "Москва"}}]
    await run_chat("Какая сейчас погода в Москве?")
    reset_calls()
    await run_chat("работай автономно")
    check("фраза управления не оплачивает выбор инструментов заново",
          MCP_CHOICE_CALLS == 0, f"выборов: {MCP_CHOICE_CALLS}")
    check("фраза управления сохраняет данные прежнего запроса",
          "Погода в Москва" in all_context(), all_context()[-200:])

    # 5.4 MCP выключен у проекта: ни вызовов, ни расхода, ни блока.
    reset_calls()
    await chat.mcp_apply(McpApply(enabled=[]))
    events = await run_chat("Какая погода в Москве?")
    check("выключенный MCP не вызывает модель для выбора", MCP_CHOICE_CALLS == 0,
          f"вызовов: {MCP_CHOICE_CALLS}")
    check("выключенный MCP не обращается к серверам", not TOOL_CALLS and DISCOVER_CALLS == 0,
          f"вызовы: {len(TOOL_CALLS)}, обнаружений: {DISCOVER_CALLS}")
    check("блока данных MCP в контексте нет", "ДАННЫЕ MCP" not in system_texts(),
          system_texts()[-200:])
    check("ответ получен как обычно", bool(texts(events, "bot")),
          str(texts(events, "bot"))[:120])

    # 5.5 Сбой выбора инструментов не ломает запрос (ответ как раньше).
    reset_calls()
    await chat.mcp_apply(McpApply(enabled=[FAKE_ID]))
    saved = client.call_llm_async

    async def broken_choice(*args, **kwargs):
        messages = kwargs.get("messages") or []
        system = str(messages[0].get("content") or "") if messages else ""
        if system.startswith("Ты — диспетчер внешних инструментов"):
            raise RuntimeError("сеть недоступна")
        return await saved(*args, **kwargs)

    client.call_llm_async = broken_choice
    try:
        events = await run_chat("Какая погода в Москве?")
    finally:
        client.call_llm_async = saved
    check("сбой выбора не выдумывает вызовы", not TOOL_CALLS, str(TOOL_CALLS)[:120])
    check("сбой выбора не ломает ответ", bool(texts(events, "bot")),
          str(texts(events, "bot"))[:120])
    check("сбой выбора не оставляет блок данных",
          "ДАННЫЕ MCP" not in system_texts(), system_texts()[-200:])

    # 5.6 Реестр проекта: три бесплатных сервера без ключей.
    restore_registry()
    registry = mcp_store.servers()
    check("к проекту подключены три MCP-сервера", len(registry) == 3,
          str([entry["id"] for entry in registry]))
    check("у каждого есть название и краткое описание",
          all(entry["name"] and entry["description"] for entry in registry),
          str(registry)[:200])
    check("id серверов — погода, курсы валют, криптовалюты",
          [entry["id"] for entry in registry] == ["weather", "currency", "crypto"],
          str([entry["id"] for entry in registry]))
    check("серверы проекта — локальные файлы MCP SDK",
          all(str(entry.get("script", "")).endswith(".mjs") for entry in registry),
          str([entry.get("script") for entry in registry]))


def main():
    print("Проверка MCP (внешних инструментов агента) — без сети и ключей")
    _write_fake_servers()
    test_client()
    test_parse_calls()
    test_storage()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_routes())
    loop.run_until_complete(test_dialog())
    print("\nИтог: " + (f"ПРОВАЛЕНО проверок: {len(FAILURES)} → {FAILURES}"
                       if FAILURES else "все проверки пройдены"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
