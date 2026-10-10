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
  [6] удалённый сервер проекта (свой open-meteo-mcp на VPS) — транспорт
      Streamable HTTP против НАСТОЯЩЕГО HTTP-сервера на stdlib во временном
      каталоге: рукопожатие, tools/list, tools/call, ответ потоком SSE, токен
      заголовком Authorization, отказ по неверному токену, причина без токена и
      недоступный сервер (туннель не поднят).
  [7] ЦЕПОЧКА вызовов (agent loop): маршрутизация «разовый запрос ↔ многошаговый»,
      раунды по РЕЗУЛЬТАТАМ (прогноз → сохранить → выгрузить файл), защита
      (выдуманный идентификатор, повтор, разрушительный вызов без просьбы) и
      ДОСТАВКА ФАЙЛА в чат: карточка со ссылкой и скачивание по маршруту.

Рабочие данные не трогаются: workspace, история агента и профили пишутся во
временный каталог (переменные AGENT_*_FILE выставляются ДО импорта chat).
"""

import asyncio
import json
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from fastapi import HTTPException

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Изоляция данных: всё пишем во временный каталог ------------------------
_TMP = tempfile.mkdtemp(prefix="mcp-check-")
os.environ["AGENT_WORKSPACE_FILE"] = os.path.join(_TMP, "workspace.json")
os.environ["AGENT_MEMORY_FILE"] = os.path.join(_TMP, "agent_memory.json")
os.environ["AGENT_PROFILES_FILE"] = os.path.join(_TMP, "profiles.json")
# Каталог серверов MCP — тоже временный: настоящие серверы (Node) здесь не
# запускаются, их место занимает тестовый сервер на Python.
os.environ["MCP_SERVERS_DIR"] = os.path.join(_TMP, "mcp_servers")
# Вложения MCP (файлы, которые вернули инструменты) — во временный каталог:
# проверка не должна оставлять файлы в рабочих данных проекта.
os.environ["MCP_ATTACH_DIR"] = os.path.join(_TMP, "mcp_files")

from app.ai import attachments as attach_store  # noqa: E402
from app.ai import client, mcp as mcp_store  # noqa: E402
from app.ai import workspace as workspace_store  # noqa: E402
from app.routers import chat  # noqa: E402
from app.schemas import ChatMessage, McpApply, SessionMode  # noqa: E402

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
# Тестовый MCP-сервер по Streamable HTTP: тот же протокол, но в теле POST
# ---------------------------------------------------------------------------
# Так проверяется УДАЛЁННЫЙ сервер проекта (свой open-meteo-mcp на VPS): сеть
# здесь только локальная (127.0.0.1), внешние источники не участвуют. Сервер
# требует токен заголовком Authorization и умеет отвечать и телом JSON, и
# потоком SSE — клиент обязан понимать оба вида ответа.
HTTP_TOKEN = "test-token-42"
HTTP_ID = "remote"

HTTP_TOOLS = [
    {
        "name": "get_current_weather",
        "title": "Погода сейчас",
        "description": "Текущая погода по названию места",
        "inputSchema": {"type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"]},
    },
    {
        "name": "boom",
        "title": "Сломанный инструмент",
        "description": "Всегда возвращает ошибку",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def http_answer(message):
    """Ответ тестового HTTP-сервера на сообщение JSON-RPC (None — уведомление)."""
    method = message.get("method")
    request_id = message.get("id")
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": request_id, "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "fake-http-mcp", "version": "0.2"}}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": HTTP_TOOLS}}
    if method == "tools/call":
        params = message.get("params") or {}
        args = params.get("arguments") or {}
        if params.get("name") == "get_current_weather":
            return {"jsonrpc": "2.0", "id": request_id, "result": {"content": [
                {"type": "text",
                 "text": "Погода в " + str(args.get("location", "?"))
                         + ": +12,6, слабая морось"}]}}
        if params.get("name") == "boom":
            return {"jsonrpc": "2.0", "id": request_id, "result": {
                "isError": True,
                "content": [{"type": "text", "text": "источник недоступен"}]}}
        return {"jsonrpc": "2.0", "id": request_id,
                "error": {"code": -32602, "message": "unknown tool"}}
    if method == "notifications/initialized":
        return None
    return {"jsonrpc": "2.0", "id": request_id,
            "error": {"code": -32601, "message": "unknown method"}}


class FakeMcpHttp:
    """MCP-сервер по HTTP на stdlib: токен обязателен, ответ — JSON или SSE."""

    def __init__(self, token: str, stream: bool = False) -> None:
        self.token = token
        self.stream = stream
        self.headers = []      # заголовки запросов: видно, что ушёл токен
        self.messages = []     # тела запросов: видно, что просил клиент
        self._server = None

    def start(self) -> str:
        """Поднимает сервер на свободном порту, возвращает адрес эндпоинта."""
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):  # тишина в выводе проверки
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length)
                try:
                    message = json.loads(raw.decode("utf-8"))
                except ValueError:
                    message = {}
                outer.messages.append(message)
                outer.headers.append(
                    {key.lower(): value for key, value in self.headers.items()})
                if self.headers.get("Authorization") != "Bearer " + outer.token:
                    self.reply(401, {"error": "unauthorized"}, "application/json")
                    return
                answer = http_answer(message)
                kind = "text/event-stream" if outer.stream else "application/json"
                self.reply(202 if answer is None else 200, answer, kind)

            def reply(self, status, payload, kind):
                if payload is None:
                    data = b""
                elif kind == "text/event-stream":
                    body = json.dumps(payload, ensure_ascii=False)
                    data = ("event: message\ndata: " + body + "\n\n").encode("utf-8")
                else:
                    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                if data:
                    self.wfile.write(data)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        host, port = self._server.server_address[:2]
        return "http://%s:%d/mcp" % (host, port)

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None


def use_http_registry(url: str) -> None:
    """Переводит клиент на тестовый HTTP-сервер (как на удалённый сервер VPS)."""
    mcp_store.SERVERS = [{
        "id": HTTP_ID,
        "name": "Тестовый HTTP",
        "description": "Сервер проверки транспорта",
        "source": "локальный тест",
        "transport": "http",
        "url": url,
        "url_env": "OPEN_METEO_MCP_URL",
        "token_env": "OPEN_METEO_MCP_TOKEN",
    }]
    mcp_store.SERVER_IDS = [HTTP_ID]
    mcp_store.forget()


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
# ОЧЕРЕДЬ РЕШЕНИЙ ДИСПЕТЧЕРА для проверки ЦЕПОЧКИ (см. test_chain): каждое
# обращение к диспетчеру берёт следующий элемент — так воспроизводится цикл
# «решение → вызов → результат → решение». Пустая очередь — прежнее поведение
# (один ответ MCP_CALLS на любой вызов).
MCP_DECISIONS = []


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
        if MCP_DECISIONS:
            decision = MCP_DECISIONS.pop(0)
            return json.dumps(decision, ensure_ascii=False), _metrics(45, 18)
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

    async def fake_run_calls(calls, limit=None):
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
    # force_plan: раздел 5 проверяет MCP В ПУТИ С ПЛАНОМ (планировщик видит данные,
    # шаги берут их из диалога без нового выбора). Вопрос «какая погода» гейт
    # `_plan_needed` отправил бы ПРЯМЫМ ответом — это тоже верно (см. 5.6), но
    # проверяет другое.
    events = await run_chat("Какая сейчас погода в Москве?", force_plan=True)
    check("служебный выбор инструментов вызван один раз", MCP_CHOICE_CALLS == 1,
          f"вызовов: {MCP_CHOICE_CALLS}")
    check("модели показан список инструментов сервера",
          "get_weather" in (MCP_PAYLOADS[0] if MCP_PAYLOADS else ""),
          (MCP_PAYLOADS[0] if MCP_PAYLOADS else "")[:200])
    check("в подсказке диспетчера названы аргументы поиска (язык и страна)",
          "language" in mcp_store.TOOLS_PROMPT
          and "countryCode" in mcp_store.TOOLS_PROMPT,
          mcp_store.TOOLS_PROMPT[-200:])
    check("вызов инструмента выполнен", len(TOOL_CALLS) == 1
          and TOOL_CALLS[0][0]["tool"] == "get_weather", str(TOOL_CALLS)[:200])
    check("в чате видно, что сделал MCP",
          any("MCP" in t for t in texts(events, "debug")),
          str(texts(events, "debug"))[-200:])
    context = all_context()
    # Ищем именно БЛОК данных: само словосочетание «ДАННЫЕ MCP» встречается ещё и
    # в промпте планировщика (правило про готовые данные), и по нему нельзя понять,
    # пришли ли данные на самом деле.
    block_head = mcp_store.BLOCK_HEADER[:40]
    check("данные MCP уходят в модель (блоком системного промпта и в служебные вызовы)",
          block_head in system_texts() or block_head in context,
          context[-300:])
    check("планировщик видит данные MCP",
          any(block_head in payload for payload in PLANNER_PAYLOADS),
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
    events = await run_chat("А теперь просто поздоровайся", force_plan=True)
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
    await run_chat("Какая сейчас погода в Москве?", force_plan=True)
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
    await run_chat("Какая сейчас погода в Москве?", force_plan=True)
    reset_calls()
    await run_chat("работай автономно")
    check("фраза управления не оплачивает выбор инструментов заново",
          MCP_CHOICE_CALLS == 0, f"выборов: {MCP_CHOICE_CALLS}")
    check("фраза управления сохраняет данные прежнего запроса",
          "Погода в Москва" in all_context(), all_context()[-200:])

    # 5.6 ОДИН РЕЖИМ, ДВА ПУТИ: вопрос отвечается ПРЯМО (без плана и шагов), но
    #     ДАННЫЕ ВНЕШНИХ ИНСТРУМЕНТОВ ему доступны — в этом и смысл объединения
    #     «разговора» и «задачи». Прежде мини-чат не спрашивал MCP вовсе.
    await fresh_project()
    reset_calls()
    MCP_CALLS[:] = [{"server": FAKE_ID, "tool": "get_weather",
                     "arguments": {"city": "Москва"}}]
    await chat.mcp_apply(McpApply(enabled=[FAKE_ID]))
    reset_calls()
    PLANNER_PAYLOADS.clear()
    events = await run_chat("Какая сейчас погода в Москве?")
    check("вопрос отвечается ПРЯМО: планировщик не вызывается",
          not PLANNER_PAYLOADS, str(PLANNER_PAYLOADS)[:120])
    check("прямой ответ видит данные внешних инструментов",
          "Погода в Москва" in system_texts(), system_texts()[-200:])
    check("прямой ответ строится по блоку данных MCP и даёт ответ",
          mcp_store.BLOCK_HEADER[:40] in system_texts()
          and bool(texts(events, "bot")), str(texts(events, "bot"))[:120])
    check("прямой ответ сохраняет данные в диалоге (шаги возьмут их же)",
          workspace_store.dialog_mcp(chat._current_session()["dialog"]).get("signature")
          is not None,
          str(workspace_store.dialog_mcp(chat._current_session()["dialog"]))[:160])
    check("в чате сказано, почему плана не будет",
          any("плана не будет" in t for t in texts(events, "debug")),
          str(texts(events, "debug"))[-200:])

    # 5.4 MCP выключен у проекта: ни вызовов, ни расхода, ни блока.
    reset_calls()
    await chat.mcp_apply(McpApply(enabled=[]))
    events = await run_chat("Какая погода в Москве?")
    check("выключенный MCP не вызывает модель для выбора", MCP_CHOICE_CALLS == 0,
          f"вызовов: {MCP_CHOICE_CALLS}")
    check("выключенный MCP не обращается к серверам", not TOOL_CALLS and DISCOVER_CALLS == 0,
          f"вызовы: {len(TOOL_CALLS)}, обнаружений: {DISCOVER_CALLS}")
    check("блока данных MCP в контексте нет",
          mcp_store.BLOCK_HEADER[:40] not in system_texts(),
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
          mcp_store.BLOCK_HEADER[:40] not in system_texts(), system_texts()[-200:])

    # 5.6 Реестр проекта: три локальных сервера без ключей и ДВА своих на VPS.
    restore_registry()
    registry = mcp_store.servers()
    check("к проекту подключены пять MCP-серверов", len(registry) == 5,
          str([entry["id"] for entry in registry]))
    check("у каждого есть название и краткое описание",
          all(entry["name"] and entry["description"] for entry in registry),
          str(registry)[:200])
    check("id серверов — погода, курсы валют, криптовалюты, свой Open-Meteo и реестр городов",
          [entry["id"] for entry in registry]
          == ["weather", "currency", "crypto", "open_meteo", "city_registry"],
          str([entry["id"] for entry in registry]))
    local = [entry for entry in registry
             if mcp_store.transport_of(entry) == mcp_store.STDIO_TRANSPORT]
    check("три сервера проекта — локальные файлы MCP SDK на stdio",
          len(local) == 3
          and all(str(entry.get("script", "")).endswith(".mjs") for entry in local),
          str([entry.get("script") for entry in local]))
    remote = mcp_store.find_server("open_meteo") or {}
    check("четвёртый сервер — удалённый, транспорт http",
          mcp_store.transport_of(remote) == mcp_store.HTTP_TRANSPORT,
          str(remote)[:200])
    check("по умолчанию адрес — локальный туннель, а не адрес VPS",
          str(remote.get("url") or "").startswith("http://127.0.0.1:3000"),
          str(remote.get("url")))
    check("секрет в реестре не хранится: только ИМЯ переменной окружения",
          remote.get("token_env") == "OPEN_METEO_MCP_TOKEN" and "token" not in remote,
          str(sorted(remote))[:200])
    cities = mcp_store.find_server("city_registry") or {}
    check("пятый сервер — удалённый, транспорт http",
          mcp_store.transport_of(cities) == mcp_store.HTTP_TRANSPORT,
          str(cities)[:200])
    check("адрес реестра городов — свой туннель на 3001, не адрес VPS",
          str(cities.get("url") or "").startswith("http://127.0.0.1:3001"),
          str(cities.get("url")))
    check("секрет реестра городов в реестре тоже не хранится",
          cities.get("token_env") == "CITY_REGISTRY_MCP_TOKEN"
          and "token" not in cities,
          str(sorted(cities))[:200])
    # Ловушка: оба .env.vps называют свой токен MCP_AUTH_TOKEN. Общая переменная
    # в приложении отправила бы одному из серверов чужой секрет — и это выглядело
    # бы как сломанный туннель (401), а не как ошибка настройки.
    check("у двух серверов на VPS РАЗНЫЕ переменные с токеном",
          bool(remote.get("token_env")) and remote.get("token_env") != cities.get("token_env"),
          "%s vs %s" % (remote.get("token_env"), cities.get("token_env")))
    saved_token = os.environ.pop("OPEN_METEO_MCP_TOKEN", "")
    try:
        gap = mcp_store.availability_error(remote)
    finally:
        if saved_token:
            os.environ["OPEN_METEO_MCP_TOKEN"] = saved_token
    check("без переменной с токеном причина видна заранее",
          "OPEN_METEO_MCP_TOKEN" in gap, gap)


# ---------------------------------------------------------------------------
# 6. Удалённый сервер проекта: транспорт Streamable HTTP
# ---------------------------------------------------------------------------
def test_http():
    print("\n[6] Свой сервер на VPS: транспорт Streamable HTTP")
    server = FakeMcpHttp(HTTP_TOKEN)
    url = server.start()
    os.environ["OPEN_METEO_MCP_TOKEN"] = HTTP_TOKEN
    os.environ["OPEN_METEO_MCP_URL"] = url
    try:
        use_http_registry(url)
        found = mcp_store.discover(HTTP_ID, force=True)
        check("HTTP-сервер объявил инструменты", found.get("ok"), str(found)[:200])
        check("имя сервера из рукопожатия (HTTP)",
              found.get("server_name") == "fake-http-mcp",
              str(found.get("server_name")))
        names = [tool["name"] for tool in found.get("tools") or []]
        check("инструменты прочитаны по HTTP",
              names == ["get_current_weather", "boom"], str(names))
        check("токен уходит заголовком Authorization",
              bool(server.headers)
              and all(str(item.get("authorization") or "")
                      == "Bearer " + HTTP_TOKEN for item in server.headers),
              str(server.headers[:1])[:200])
        check("клиент соглашается и на JSON, и на поток SSE",
              bool(server.headers)
              and all("text/event-stream" in str(item.get("accept") or "")
                      for item in server.headers),
              str(server.headers[:1])[:200])
        check("адрес берётся из переменной окружения",
              mcp_store.server_url(mcp_store.find_server(HTTP_ID) or {}) == url,
              mcp_store.server_url(mcp_store.find_server(HTTP_ID) or {}))

        result = mcp_store.call_tool(HTTP_ID, "get_current_weather",
                                     {"location": "Москва"})
        check("вызов инструмента по HTTP вернул текст",
              result["ok"] and "Москва" in result["text"], str(result)[:200])

        broken = mcp_store.call_tool(HTTP_ID, "boom", {})
        check("ошибка инструмента (isError) по HTTP не выдаёт себя за данные",
              not broken["ok"] and "источник недоступен" in broken["error"],
              str(broken)[:200])

        unknown = mcp_store.call_tool(HTTP_ID, "нет_такого", {})
        check("ошибка протокола по HTTP возвращается понятным текстом",
              not unknown["ok"] and "unknown tool" in unknown["error"],
              str(unknown)[:200])

        # Тот же транспорт, но ответ приходит ПОТОКОМ (SSE): сервер вправе
        # выбрать вид ответа, и клиент обязан понять оба.
        stream_server = FakeMcpHttp(HTTP_TOKEN, stream=True)
        stream_url = stream_server.start()
        try:
            use_http_registry(stream_url)
            streamed = mcp_store.discover(HTTP_ID, force=True)
            check("ответ потоком SSE разобран",
                  streamed.get("ok")
                  and streamed.get("server_name") == "fake-http-mcp",
                  str(streamed)[:200])
            streamed_call = mcp_store.call_tool(HTTP_ID, "get_current_weather",
                                                {"location": "Казань"})
            check("вызов по SSE вернул текст",
                  streamed_call["ok"] and "Казань" in streamed_call["text"],
                  str(streamed_call)[:200])
        finally:
            stream_server.stop()

        # Неверный токен: сервер отвечает 401 — причина понятная, секрет не течёт.
        os.environ["OPEN_METEO_MCP_TOKEN"] = "wrong-token"
        use_http_registry(url)
        denied = mcp_store.discover(HTTP_ID, force=True)
        check("неверный токен — понятная причина, а не исключение",
              denied.get("ok") is False and "401" in str(denied.get("error")),
              str(denied)[:200])
        check("токен не попадает в текст ошибки",
              HTTP_TOKEN not in str(denied.get("error")), str(denied)[:200])

        # Токен с не-ASCII символами: заголовок HTTP их не примет — говорим об
        # этом словами, а не ошибкой кодека.
        os.environ["OPEN_METEO_MCP_TOKEN"] = "чужой-токен"
        use_http_registry(url)
        broken_header = mcp_store.discover(HTTP_ID, force=True)
        check("не-ASCII токен объясняется понятно",
              broken_header.get("ok") is False
              and "HTTP-заголовке" in str(broken_header.get("error")),
              str(broken_header)[:200])
        os.environ["OPEN_METEO_MCP_TOKEN"] = HTTP_TOKEN

        # Токена нет вовсе: причина видна ЗАРАНЕЕ, без обращения к серверу.
        saved = os.environ.pop("OPEN_METEO_MCP_TOKEN", "")
        problem = mcp_store.availability_error(mcp_store.find_server(HTTP_ID) or {})
        check("без токена причина названа заранее",
              "OPEN_METEO_MCP_TOKEN" in problem, problem)
        os.environ["OPEN_METEO_MCP_TOKEN"] = saved

        # Сервер (или туннель) недоступен: подсказка вместо трассировки.
        server.stop()
        down = mcp_store.discover(HTTP_ID, force=True)
        check("недоступный сервер объясняется понятно",
              down.get("ok") is False and "недоступен" in str(down.get("error")),
              str(down)[:200])
        check("в причине есть подсказка про туннель",
              "туннел" in str(down.get("error")), str(down)[:200])
    finally:
        server.stop()
        os.environ.pop("OPEN_METEO_MCP_TOKEN", None)
        os.environ.pop("OPEN_METEO_MCP_URL", None)
        restore_registry()


# ---------------------------------------------------------------------------
# 7. Цепочка вызовов (agent loop): зависимости по данным и файлы в чат
# ---------------------------------------------------------------------------
# Тестовый сервер «как open-meteo на VPS»: чтение (прогноз), запись (сохранение
# набора) и выгрузка файла, которой нужен идентификатор от сохранения. Именно
# такая зависимость по данным и проверяется — ни имена, ни предметная область в
# механике цепочки не зашиты.
CHAIN_ID = "chain-mcp"
CHAIN_CALLS = []
# «Файл» из инструмента: настоящий xlsx здесь не нужен, проверяется доставка
# байтов (сигнатура PK — как у xlsx, чтобы имя и тип были осмысленными).
XLSX_BYTES = b"PK\x03\x04check-xlsx-payload"
DELIVER_REQUEST = ("получи прогноз погоды на завтра в Екатеринбурге, прогноз погоды "
                   "на завтра в Казани, сохрани их и отдай в виде эксель таблицы")
SIMPLE_REQUEST = "какая сейчас погода в Казани"
# Живой случай (09.10), обобщённый до синтетических данных: город человека НЕ
# НАЗВАН, его надо сначала найти, а потом взять по нему данные. Ни «сохрани»,
# ни «затем» в запросе нет.
LOOKUP_REQUEST = "погода в городе, где живёт Егор"
CHAIN_TOOLS = [
    {"name": "get_forecast", "title": "Прогноз",
     "description": "Прогноз погоды по городу на несколько дней",
     "schema": {"type": "object", "properties": {"location": {"type": "string"}},
                "required": ["location"]}},
    {"name": "save_weather_summary", "title": "Сохранить таблицу",
     "description": "Сохраняет таблицу значений на сервере под идентификатором",
     "schema": {"type": "object",
                "properties": {"dataset_id": {"type": "string"},
                               "entries": {"type": "array"},
                               # Как у настоящего сервера: без флага перезаписи
                               # сохранение под существующим id отклоняется.
                               "replace": {"type": "boolean"}},
                "required": ["entries"]}},
    {"name": "export_weather_summary_excel", "title": "Выгрузить Excel",
     "description": "Строит файл .xlsx из сохранённого набора",
     "schema": {"type": "object", "properties": {"dataset_id": {"type": "string"}},
                "required": ["dataset_id"]}},
    {"name": "delete_weather_watch", "title": "Удалить наблюдение",
     "description": "Удаляет наблюдение вместе с данными",
     "schema": {"type": "object", "properties": {"id": {"type": "string"}},
                "required": ["id"]}},
]
DATASET_ID = "ekb-kzn-27-09"


def use_chain_registry() -> None:
    """Реестр из одного «цепочного» сервера (без процессов и сети)."""
    mcp_store.SERVER_IDS = [CHAIN_ID]
    mcp_store.forget()


def install_chain_stubs() -> None:
    """Подмена серверов цепочки: инструменты отвечают как настоящие, но локально."""
    async def fake_discover(ids=None, force=False):
        return [{"id": CHAIN_ID, "ok": True, "error": "", "server_name": "chain-mcp",
                 "server_version": "0.1",
                 "tools": [dict(tool) for tool in CHAIN_TOOLS]}]

    def result(call, text, attachments=None):
        return {"server": CHAIN_ID, "server_name": "chain-mcp",
                "source": "локальный тест", "tool": str(call.get("tool") or ""),
                "arguments": dict(call.get("arguments") or {}),
                "ok": True, "text": text, "error": "",
                "attachments": list(attachments or [])}

    def fake_discover_sync(server_id=None, force=False):
        """Синхронное обнаружение (нужно там, где схемы читаются из кэша)."""
        found = {"id": CHAIN_ID, "ok": True, "error": "", "server_name": "chain-mcp",
                 "server_version": "0.1",
                 "tools": [dict(tool) for tool in CHAIN_TOOLS]}
        return found if server_id in (None, CHAIN_ID) else {
            "id": str(server_id), "ok": False, "error": "нет такого сервера",
            "server_name": "", "server_version": "", "tools": []}

    async def fake_run_calls(calls, limit=None):
        # Лимит соблюдаем как настоящий run_calls: иначе проверка не заметила бы
        # потерю вызова цепочки (умолчание — предел ОДНОГО раунда).
        cap = mcp_store.MAX_CALLS_PER_REQUEST if limit is None else max(0, int(limit))
        calls = list(calls)[:cap]
        CHAIN_CALLS.append(list(calls))
        out = []
        for call in calls:
            tool = str(call.get("tool") or "")
            args = call.get("arguments") or {}
            if tool == "list_owners":
                # Инструмент-ПОИСК (см. раздел 7.11): отдаёт пары «человек — город»,
                # то есть значения, которых в запросе нет.
                out.append(result(call, "Владельцы: Егор — Самара; Нина — Тула"))
            elif tool == "get_forecast":
                out.append(result(call, f"Прогноз для {args.get('location')}: "
                                        "2026-09-27, минимум 6.9, максимум 17.5"))
            elif tool == "save_weather_summary":
                out.append(result(call, "Таблица сохранена. dataset_id: "
                                        f"\"{args.get('dataset_id')}\", строк: "
                                        f"{len(args.get('entries') or [])}"))
            elif tool == "export_weather_summary_excel":
                ref = attach_store.store(
                    XLSX_BYTES, name="weather-ekb-kzn.xlsx",
                    mime="application/vnd.openxmlformats-officedocument."
                         "spreadsheetml.sheet",
                    origin="chain-mcp · export_weather_summary_excel")
                out.append(result(call, "Файл выгружен на сервере: "
                                        f"/app/data/{ref['name']}", [ref]))
        return out

    mcp_store.async_discover = fake_discover
    mcp_store.async_run_calls = fake_run_calls
    mcp_store.discover = fake_discover_sync
    chat.mcp_store.async_discover = fake_discover
    chat.mcp_store.async_run_calls = fake_run_calls
    chat.mcp_store.discover = fake_discover_sync


def flat_chain_tools():
    """Инструменты цепочки в том виде, в каком их видит диспетчер."""
    return [{"server": CHAIN_ID, "server_name": "chain-mcp", "tool": tool["name"],
             "description": tool.get("description") or "",
             "schema": tool.get("schema") or {}}
            for tool in CHAIN_TOOLS]


async def test_chain():
    print("\n[7] Цепочка вызовов: зависимости по данным и файл в чате")
    flat = flat_chain_tools()

    # 7.1 Маршрутизация: разовый запрос — дешёвый путь, «сохрани и отдай Excel» — цепочка.
    simple = mcp_store.chain_signals(flat, SIMPLE_REQUEST, expected=1)
    deliver = mcp_store.chain_signals(flat, DELIVER_REQUEST, expected=2)
    check("разовый запрос не уходит в цепочку", not simple["needed"], str(simple))
    check("запрос «сохрани и отдай Excel» уходит в цепочку", deliver["needed"], str(deliver))
    check("назван инструмент, которому нужен идентификатор из результата",
          "export_weather_summary_excel" in [item["tool"] for item in deliver["id_tools"]],
          str(deliver["id_tools"]))

    # 7.2 Разбор решения диспетчера: режим работы и признак «готово».
    parsed = mcp_store.parse_decision(json.dumps({
        "mode": "chain", "reason": "нужно сохранить",
        "calls": [{"server": CHAIN_ID, "tool": "get_forecast",
                   "arguments": {"location": "Казань"}}]}), flat)
    check("режим цепочки разобран из ответа диспетчера",
          parsed["mode"] == "chain" and len(parsed["calls"]) == 1, str(parsed))
    old = mcp_store.parse_decision(
        json.dumps({"calls": [{"server": CHAIN_ID, "tool": "get_forecast",
                               "arguments": {"location": "Казань"}}]}), flat)
    check("прежний формат ответа читается как разовый режим",
          old["mode"] == "single" and len(old["calls"]) == 1, str(old))

    # 7.3 Защита вызовов: выдуманный идентификатор, повтор, разрушительное.
    forecast_results = [{"server": CHAIN_ID, "tool": "get_forecast", "arguments": {},
                         "ok": True, "text": "Прогноз для Казани: 2026-09-27", "error": ""}]
    export_call = {"server": CHAIN_ID, "tool": "export_weather_summary_excel",
                   "arguments": {"dataset_id": DATASET_ID}}
    invented = [{"server": CHAIN_ID, "tool": "export_weather_summary_excel",
                 "arguments": {"dataset_id": "я-придумал-этот-id"}}]
    allowed, rejected = mcp_store.guard_calls(invented, forecast_results, flat,
                                              user_text=DELIVER_REQUEST)
    check("выдуманный dataset_id отброшен до вызова", not allowed and bool(rejected),
          str(rejected))
    saved_results = forecast_results + [
        {"server": CHAIN_ID, "tool": "save_weather_summary", "arguments": {},
         "ok": True, "text": f"Таблица сохранена. dataset_id: \"{DATASET_ID}\"", "error": ""},
        {"server": CHAIN_ID, "tool": "get_forecast", "arguments": {},
         "ok": True, "text": "Наблюдение активно. watch_id: \"w-1\"", "error": ""}]
    allowed, rejected = mcp_store.guard_calls([export_call], saved_results, flat,
                                              user_text=DELIVER_REQUEST)
    check("идентификатор из результатов пропускается", bool(allowed) and not rejected,
          str(rejected))
    allowed, rejected = mcp_store.guard_calls([export_call], forecast_results, flat,
                                              user_text=DELIVER_REQUEST,
                                              issued_ids={"dataset_id": DATASET_ID})
    check("идентификатор, названный сохранением, тоже годится",
          bool(allowed) and not rejected, str(rejected))
    allowed, rejected = mcp_store.guard_calls([export_call], saved_results, flat,
                                              user_text=DELIVER_REQUEST,
                                              done_keys={mcp_store.call_key(export_call)})
    check("повтор уже выполненного вызова отброшен", not allowed and bool(rejected),
          str(rejected))
    kill = [{"server": CHAIN_ID, "tool": "delete_weather_watch",
             "arguments": {"id": "w-1"}}]
    allowed, rejected = mcp_store.guard_calls(kill, saved_results, flat,
                                              user_text=SIMPLE_REQUEST)
    check("разрушительный вызов без просьбы пользователя отброшен",
          not allowed and bool(rejected), str(rejected))
    allowed, rejected = mcp_store.guard_calls(
        kill, saved_results, flat, user_text="удали наблюдение w-1")
    check("разрушительный вызов по прямой просьбе проходит", bool(allowed), str(rejected))

    # 7.4 ГИБРИД: до подтверждения плана — ТОЛЬКО ЧТЕНИЯ. Данные для плана есть,
    # а побочных эффектов нет: ни набора на сервере, ни файла в чате.
    install_chain_stubs()
    use_chain_registry()
    await chat.task_create(chat.TaskCreate(name="Проект с цепочкой"))
    await chat.session_create()
    await chat.mcp_apply(McpApply(enabled=[CHAIN_ID]))
    reset_calls()
    CHAIN_CALLS.clear()
    save_call = {"server": CHAIN_ID, "tool": "save_weather_summary",
                 "arguments": {"dataset_id": DATASET_ID,
                               "entries": [{"location": "Екатеринбург",
                                            "date": "2026-09-27"},
                                           {"location": "Казань",
                                            "date": "2026-09-27"}]}}
    export_call = {"server": CHAIN_ID, "tool": "export_weather_summary_excel",
                   "arguments": {"dataset_id": DATASET_ID}}
    MCP_DECISIONS[:] = [
        {"mode": "chain", "reason": "нужны прогнозы по двум городам",
         "calls": [{"server": CHAIN_ID, "tool": "get_forecast",
                    "arguments": {"location": "Екатеринбург", "days": 2}},
                   {"server": CHAIN_ID, "tool": "get_forecast",
                    "arguments": {"location": "Казань", "days": 2}}]},
        # Раунд цепочки ДО подтверждения: сохранение — оно откладывается.
        {"done": False, "reason": "сохраняю прогнозы одним набором",
         "calls": [save_call]},
    ]
    events = await run_chat(DELIVER_REQUEST)
    executed = [call["tool"] for batch in CHAIN_CALLS for call in batch]
    check("до подтверждения плана выполнены только чтения",
          executed == ["get_forecast", "get_forecast"], str(executed))
    check("цепочка отложена до «ок» и это сказано в чате",
          any("цепочка продолжится после подтверждения плана" in text
              for text in texts(events, "debug")),
          str(texts(events, "debug"))[-300:])
    check("файла в чате до подтверждения плана нет",
          not any(e.get("type") == "bot" and e.get("files") for e in events),
          str([e.get("type") for e in events])[:200])
    final_states = [e["state"] for e in events
                    if e.get("type") == "done" and e.get("state")]
    stage_after = (final_states[-1].get("stage") if final_states else None)
    check("задача честно ждёт подтверждения плана", stage_after == "awaiting_user",
          str(stage_after))
    check("текст плана просит подтверждение (работа ещё не сделана)",
          any("Подтвердите план — кнопка" in text for text in texts(events, "bot")),
          str(texts(events, "bot"))[:250])
    stored = workspace_store.dialog_mcp(chat._current_session()["dialog"])
    check("цепочка помечена как НЕ доигранная (ждёт подтверждения)",
          (stored.get("chain") or {}).get("pending") is True,
          str(stored.get("chain"))[:200])
    # ДО ПОДТВЕРЖДЕНИЯ ПЛАНА цепочка больше НЕ спрашивает диспетчера: его
    # предложения всё равно откладывались до «ок», то есть вызов LLM уходил
    # впустую. Остаётся ровно один вызов — первичный выбор инструментов.
    check("до подтверждения плана диспетчер спрашивается РОВНО один раз",
          MCP_CHOICE_CALLS == 1, f"вызовов диспетчера: {MCP_CHOICE_CALLS}")

    # 7.4б «ОК» — ЭТО ШАГ 1 ИЗ 2: до последнего шага результат не выдаётся.
    reset_calls()
    CHAIN_CALLS.clear()
    MCP_DECISIONS[:] = [
        {"done": False, "reason": "сохраняю прогнозы одним набором",
         "calls": [save_call]},
        {"done": False, "reason": "выгружаю файл по идентификатору набора",
         "calls": [export_call]},
        {"done": True, "reason": "данные получены, набор сохранён, файл выгружен",
         "calls": []},
    ]
    ok_events = await run_chat("ок")
    check("на непоследнем шаге результат ещё не выдаётся",
          not [call["tool"] for batch in CHAIN_CALLS for call in batch],
          str(CHAIN_CALLS))
    check("ответ промежуточного шага не показывается репликой в чате",
          not any(e.get("type") == "bot" and ANSWER in str(e.get("text") or "")
                  for e in ok_events),
          str([(e.get("type"), str(e.get("text"))[:40]) for e in ok_events])[:250])
    check("вместо ответа промежуточного шага — строка в журнале",
          any("в чате не показываю" in text for text in texts(ok_events, "debug")),
          str(texts(ok_events, "debug"))[-250:])
    stored_mid = workspace_store.dialog_mcp(chat._current_session()["dialog"])
    check("цепочка всё ещё ждёт последнего шага",
          (stored_mid.get("chain") or {}).get("pending") is True,
          str(stored_mid.get("chain"))[:200])

    # 7.4в ПОСЛЕДНИЙ ШАГ: отложенная часть доигрывается — сохранение и выгрузка,
    # затем ФИНАЛЬНЫЙ ОТЧЁТ, и только под ним — карточка файла.
    reset_calls()
    CHAIN_CALLS.clear()
    MCP_DECISIONS[:] = [
        {"done": False, "reason": "сохраняю прогнозы одним набором",
         "calls": [save_call]},
        {"done": False, "reason": "выгружаю файл по идентификатору набора",
         "calls": [export_call]},
        {"done": True, "reason": "данные получены, набор сохранён, файл выгружен",
         "calls": []},
    ]
    events = await run_chat("", continue_step=True)
    executed = [call["tool"] for batch in CHAIN_CALLS for call in batch]
    check("на последнем шаге выполнены сохранение и выгрузка",
          executed == ["save_weather_summary", "export_weather_summary_excel"],
          str(executed))
    check("идентификатор сохранения передан в выгрузку",
          (CHAIN_CALLS[1][0].get("arguments") or {}).get("dataset_id") == DATASET_ID,
          str(CHAIN_CALLS[1:]))
    check("в чате сказано, что цепочка доиграна после подтверждения",
          any("доигрываю цепочку" in text for text in texts(events, "debug")),
          str(texts(events, "debug"))[-300:])
    # ОЧЕВИДНАЯ ДОСТРОЙКА: после сохранения остался ровно один инструмент
    # результата (выгрузка) и его обязательный dataset_id уже известен — он
    # вызывается БЕЗ диспетчера (экономия целого вызова LLM).
    check("очевидная выгрузка выполнена без вызова диспетчера",
          MCP_CHOICE_CALLS == 1, f"вызовов диспетчера в этой фазе: {MCP_CHOICE_CALLS}")
    check("в чате сказано, что шаг выполнен без диспетчера",
          any("остался один очевидный шаг результата" in text
              for text in texts(events, "debug")),
          str(texts(events, "debug"))[-250:])
    final_answer_at = [index for index, e in enumerate(events)
                       if e.get("type") == "bot" and ANSWER in str(e.get("text") or "")]
    file_card_at = [index for index, e in enumerate(events)
                    if e.get("type") == "bot" and e.get("files")]
    check("финальный отчёт показан репликой в чате", bool(final_answer_at),
          str([(e.get("type"), str(e.get("text"))[:40]) for e in events])[:250])
    check("карточка файла идёт ПОСЛЕ финального отчёта",
          bool(final_answer_at) and bool(file_card_at)
          and min(file_card_at) > max(final_answer_at),
          f"отчёт: {final_answer_at}, файл: {file_card_at}")
    # После выдачи файла цепочка больше НЕ спрашивает диспетчера «всё ли готово»
    # (это ещё один полный вызов LLM), и выгрузку тоже сделала сама: в очереди
    # остались неиспользованными решения «выгрузка» и «готово».
    check("после файла цепочка не тратит вызов диспетчера на «всё ли готово»",
          len(MCP_DECISIONS) == 2, f"неиспользованных решений: {len(MCP_DECISIONS)}")

    # 7.5 Файл: карточка в чате, запись на диск, скачивание по ссылке.
    file_events = [e for e in events if e.get("type") == "bot" and e.get("files")]
    check("файл показан в чате карточкой", len(file_events) == 1
          and file_events[0]["files"][0]["name"] == "weather-ekb-kzn.xlsx",
          str(file_events)[:200])
    ref = (file_events[0]["files"][0] if file_events else {})
    check("у файла есть ссылка на скачивание",
          str(ref.get("url") or "").startswith("/api/agent/files/"), str(ref))
    path = attach_store.resolve(ref.get("id"))
    check("файл сохранён на диск",
          bool(path) and os.path.getsize(path) == len(XLSX_BYTES), str(path))
    if path:
        with open(path, "rb") as handle:
            check("на диск записаны те же байты", handle.read() == XLSX_BYTES, str(path))
    response = await chat.agent_file(str(ref.get("id")))
    check("маршрут скачивания отдаёт файл",
          os.path.basename(str(getattr(response, "path", ""))) == ref.get("id"),
          str(getattr(response, "path", "")))
    try:
        await chat.agent_file("../../etc/passwd")
        blocked = False
    except HTTPException:
        blocked = True
    check("через маршрут нельзя выйти за каталог вложений", blocked)

    dialog = chat._current_session()["dialog"]
    stored = workspace_store.dialog_mcp(dialog)
    check("все вызовы цепочки сохранены в диалоге",
          [call["tool"] for call in stored.get("calls") or []]
          == ["get_forecast", "get_forecast", "save_weather_summary",
              "export_weather_summary_excel"], str(stored.get("calls"))[:200])
    check("состояние цепочки запомнено (для повторов задачи)",
          (stored.get("chain") or {}).get("ids", {}).get("dataset_id") == DATASET_ID,
          str(stored.get("chain"))[:200])
    check("файл в журнале чата (карточка переживёт перезагрузку)",
          any(item.get("files") for item in dialog.get("log") or []),
          str([item.get("kind") for item in dialog.get("log") or []])[:200])
    cards = [item for item in dialog.get("log") or [] if item.get("files")]
    check("в журнале РОВНО одна карточка файла (дубля в чате нет)",
          len(cards) == 1, f"карточек: {len(cards)}")
    check("модель видит, что файл уже у пользователя",
          "ПОЛУЧЕННЫЕ ФАЙЛЫ" in all_context(), all_context()[-300:])

    # 7.6 ПОВТОР периодической задачи: цепочка обязана повториться ЦЕЛИКОМ и
    # идемпотентно — тот же набор данных на сервере (тот же dataset_id), свежие
    # значения, новый файл. Читающие вызовы повторяются без выбора инструментов
    # (диспетчер не оплачивается), но вызовы ЗАПИСИ и ВЫГРУЗКИ в цепочке — тоже
    # часть работы задачи, и пропустить их нельзя: файл остался бы прошлым.
    CHAIN_CALLS.clear()
    reset_calls()
    # Автозапуск повторяет ЗАПРОС задачи (см. periodic_runner._turn): тот же
    # текст, флаг periodic — прогон автономный, данные берутся заново.
    repeat_events = await run_chat(DELIVER_REQUEST, periodic=True)
    replayed = [call["tool"] for batch in CHAIN_CALLS for call in batch]
    check("повтор задачи выполняет ВСЮ цепочку, а не только чтение",
          replayed == ["get_forecast", "get_forecast", "save_weather_summary",
                       "export_weather_summary_excel"], str(replayed))
    repeat_save = next((call for batch in CHAIN_CALLS for call in batch
                        if call["tool"] == "save_weather_summary"), {})
    check("повтор сохраняет под ТЕМ ЖЕ идентификатором (идемпотентность)",
          (repeat_save.get("arguments") or {}).get("dataset_id") == DATASET_ID,
          str(repeat_save))
    check("повтор САМ ставит флаг перезаписи (сервер иначе отказывает)",
          (repeat_save.get("arguments") or {}).get("replace") is True,
          str(repeat_save))
    check("в чате сказано, что флаг перезаписи поставлен агентом",
          any("флаг перезаписи" in text for text in texts(repeat_events, "debug")),
          str(texts(repeat_events, "debug"))[-200:])
    check("повтор не оплачивает выбор инструментов заново", MCP_CHOICE_CALLS == 0,
          f"вызовов диспетчера: {MCP_CHOICE_CALLS}")
    stored_after = workspace_store.dialog_mcp(chat._current_session()["dialog"])
    check("после повтора в диалоге по-прежнему все вызовы цепочки",
          [call["tool"] for call in stored_after.get("calls") or []]
          == ["get_forecast", "get_forecast", "save_weather_summary",
              "export_weather_summary_excel"], str(stored_after.get("calls"))[:200])
    check("идентификатор набора после повтора не изменился",
          (stored_after.get("chain") or {}).get("ids", {}).get("dataset_id") == DATASET_ID,
          str(stored_after.get("chain"))[:200])

    # 7.7 Разовый запрос при том же наборе инструментов цикл НЕ оплачивает.
    reset_calls()
    CHAIN_CALLS.clear()
    MCP_CALLS[:] = [{"server": CHAIN_ID, "tool": "get_forecast",
                     "arguments": {"location": "Казань"}}]
    MCP_DECISIONS.clear()
    simple_events = await run_chat(SIMPLE_REQUEST)
    check("разовый запрос обошёлся одним вызовом диспетчера",
          MCP_CHOICE_CALLS == 1, f"вызовов диспетчера: {MCP_CHOICE_CALLS}")
    check("разовый запрос сделал один вызов инструмента",
          len([call for batch in CHAIN_CALLS for call in batch]) == 1, str(CHAIN_CALLS))
    # КОНТРАКТ С 09.10: разовый запрос теперь ОБЪЯВЛЯЕТ цепочку (вход в неё стал
    # структурным: остались невызванные инструменты) — и всё равно делает РОВНО
    # один вызов инструмента: лишние закрывает сам диспетчер ответом «готово».
    check("разовый запрос объявляет цепочку, но лишних вызовов не делает",
          any("цепочкой" in text for text in texts(simple_events, "debug")),
          str(texts(simple_events, "debug"))[-200:])

    # 7.8 Чистка каталога вложений: срок хранения и предел по числу файлов.
    old = attach_store.store(b"old-attachment", name="old.txt", mime="text/plain")
    fresh = attach_store.store(b"fresh-attachment", name="fresh.txt", mime="text/plain")
    old_path = attach_store.resolve((old or {}).get("id"))
    ancient = time.time() - 90 * 86400
    if old_path:
        os.utime(old_path, (ancient, ancient))
    removed = attach_store.prune(force=True)
    check("старое вложение удалено по сроку хранения",
          (old or {}).get("id") in removed and not attach_store.resolve((old or {}).get("id")),
          str(removed))
    check("свежее вложение после чистки на месте",
          bool(attach_store.resolve((fresh or {}).get("id"))), "файл свежего вложения пропал")
    os.environ["MCP_ATTACH_MAX_FILES"] = "2"
    try:
        for index in range(4):
            attach_store.store(f"bulk-{index}".encode(), name=f"bulk{index}.txt",
                               mime="text/plain")
            time.sleep(0.01)
        attach_store.prune(force=True)
        left = sorted(name for name in os.listdir(attach_store.directory()))
        check("каталог не превышает предел по числу файлов", len(left) <= 2, str(left))
    finally:
        os.environ.pop("MCP_ATTACH_MAX_FILES", None)
    check("пределы хранения читаются из переменных окружения",
          attach_store.limits() == (attach_store.DEFAULT_TTL_DAYS,
                                    attach_store.DEFAULT_MAX_FILES,
                                    attach_store.DEFAULT_MAX_DIR_BYTES),
          str(attach_store.limits()))

    # 7.9 ДИСПЕТЧЕР РЕШИЛ «ДАННЫХ ХВАТАЕТ». Живой случай: на тот же запрос модель
    # то строит цепочку, то отвечает «ничего больше не нужно» — и задача остаётся
    # без сохранения и файла. После «ок» агент обязан уточнить задачу, назвав
    # невызванные инструменты результата, и довести работу до файла.
    nudge_request = ("прогноз на завтра для Екатеринбурга и Казани, сохрани его "
                     "и приложи файлом Excel")
    CHAIN_CALLS.clear()
    reset_calls()
    MCP_DECISIONS[:] = [
        {"mode": "chain", "reason": "нужны прогнозы",
         "calls": [{"server": CHAIN_ID, "tool": "get_forecast",
                    "arguments": {"location": "Екатеринбург"}},
                   {"server": CHAIN_ID, "tool": "get_forecast",
                    "arguments": {"location": "Казань"}}]},
    ]
    nudge_events = await run_chat(nudge_request)
    nudge_executed = [call["tool"] for batch in CHAIN_CALLS for call in batch]
    check("до подтверждения плана прочитаны данные и цепочка отложена",
          nudge_executed == ["get_forecast", "get_forecast"], str(nudge_executed))
    # ПОДТВЕРЖДЕНИЕ: доигрывание цепочки. Первый ответ диспетчера — «данных
    # хватает» (отказ), значит агент обязан переспросить.
    reset_calls()
    CHAIN_CALLS.clear()
    MCP_DECISIONS[:] = [
        # ШАГ 1 (не последний): цепочка продолжается и на нём — но в режиме
        # чтений. Новых чтений диспетчер не предлагает, цепочка на этом шаге
        # закрывается, а сохранение и выгрузка ждут последнего шага.
        {"done": True, "reason": "данных для шага хватает", "calls": []},
        # ПОСЛЕДНИЙ шаг: первый ответ диспетчера — «данных достаточно» (отказ),
        # значит агент обязан переспросить (уточнение бывает только там, где
        # разрешены эффекты, — на промежуточном шаге оно просило бы невозможное).
        {"done": True, "reason": "данных достаточно для ответа", "calls": []},
        {"done": False, "reason": "сохраняю прогнозы",
         "calls": [{"server": CHAIN_ID, "tool": "save_weather_summary",
                    "arguments": {"dataset_id": DATASET_ID, "replace": True,
                                  "entries": [{"location": "Екатеринбург"},
                                              {"location": "Казань"}]}}]},
        {"done": False, "reason": "выгружаю файл",
         "calls": [{"server": CHAIN_ID, "tool": "export_weather_summary_excel",
                    "arguments": {"dataset_id": DATASET_ID}}]},
    ]
    await run_chat("ок")
    approve_events = await run_chat("", continue_step=True)
    nudge_payloads = [p for p in MCP_PAYLOADS if "ЗАПРОС ВЫПОЛНЕН НЕ ПОЛНОСТЬЮ" in p]
    check("в уточнении названы невызванные инструменты результата",
          len(nudge_payloads) == 1
          and "save_weather_summary" in nudge_payloads[0]
          and "export_weather_summary_excel" in nudge_payloads[0],
          str(nudge_payloads)[:200])
    check("уточнение запрашивается ОДИН раз (без цикла уговоров)",
          len(nudge_payloads) == 1, f"уточнений: {len(nudge_payloads)}")
    check("уточнение видно в чате",
          any("уточняю задачу диспетчеру" in text for text in texts(approve_events, "debug")),
          str(texts(approve_events, "debug"))[-200:])
    check("файл получен после уточнения и подтверждения",
          any(e.get("type") == "bot" and e.get("files") for e in approve_events),
          str([e.get("type") for e in approve_events])[:200])

    # 7.10 ДИСПЕТЧЕР ОТВЕТИЛ «ДАННЫЕ НЕ НУЖНЫ» на запрос, который явно просит
    # сохранить и выдать файл (живой случай: модель вернула пустой список, и
    # цепочка даже не начиналась). Агент обязан переспросить и выполнить работу.
    initial_request = ("прогноз на завтра по Екатеринбургу и Казани, сохрани его "
                       "и отдай файлом xlsx")
    # Свежий проект и диалог: проверка не должна зависеть от состояния задачи,
    # оставшегося от предыдущих разделов.
    await chat.task_create(chat.TaskCreate(name="Проект: «данные не нужны»"))
    await chat.session_create()
    await chat.mcp_apply(McpApply(enabled=[CHAIN_ID]))
    CHAIN_CALLS.clear()
    reset_calls()
    MCP_DECISIONS[:] = [
        {"mode": "single", "reason": "для ответа хватит знаний модели", "calls": []},
        {"mode": "chain", "reason": "нужны прогнозы",
         "calls": [{"server": CHAIN_ID, "tool": "get_forecast",
                    "arguments": {"location": "Екатеринбург"}},
                   {"server": CHAIN_ID, "tool": "get_forecast",
                    "arguments": {"location": "Казань"}}]},
        {"done": False, "reason": "сохраняю",
         "calls": [{"server": CHAIN_ID, "tool": "save_weather_summary",
                    "arguments": {"dataset_id": DATASET_ID, "replace": True,
                                  "entries": [{"location": "Екатеринбург"},
                                              {"location": "Казань"}]}}]},
    ]
    initial_events = await run_chat(initial_request)
    initial_executed = [call["tool"] for batch in CHAIN_CALLS for call in batch]
    check("после «данные не нужны» агент переспросил и собрал данные",
          initial_executed == ["get_forecast", "get_forecast"], str(initial_executed))
    check("первичное уточнение требует чтение и режим chain",
          any('mode="chain"' in payload for payload in MCP_PAYLOADS),
          str([p[:80] for p in MCP_PAYLOADS])[:200])
    check("первичное уточнение видно в чате",
          any("хотя запрос просит" in text for text in texts(initial_events, "debug")),
          str(texts(initial_events, "debug"))[-200:])
    check("повторных уговоров нет (уточнение одно)",
          sum(1 for payload in MCP_PAYLOADS
              if "ЗАПРОС ВЫПОЛНЕН НЕ ПОЛНОСТЬЮ" in payload) == 1,
          f"уточнений: {sum(1 for p in MCP_PAYLOADS if 'НЕ ПОЛНОСТЬЮ' in p)}")
    stored_initial = workspace_store.dialog_mcp(chat._current_session()["dialog"])
    check("цепочка после уточнения помечена как не доигранная",
          (stored_initial.get("chain") or {}).get("pending") is True,
          str(stored_initial.get("chain"))[:300])
    initial_states = [e["state"] for e in initial_events
                      if e.get("type") == "done" and e.get("state")]
    check("в 7.10 задача ждёт подтверждения плана",
          (initial_states[-1].get("stage") if initial_states else None) == "awaiting_user",
          str(initial_states[-1] if initial_states else None)[:200])
    reset_calls()
    CHAIN_CALLS.clear()
    MCP_DECISIONS[:] = [
        # ШАГ 1 (не последний): цепочка продолжается здесь, но только чтениями —
        # эффекты ждут последнего шага, поэтому решений на нём не спрашивается.
        {"done": True, "reason": "данных для шага хватает", "calls": []},
        # ПОСЛЕДНИЙ шаг: сохранение, а выгрузку агент достраивает сам
        # (auto_followup — единственный оставшийся вид результата).
        {"done": False, "reason": "сохраняю",
         "calls": [{"server": CHAIN_ID, "tool": "save_weather_summary",
                    "arguments": {"dataset_id": DATASET_ID, "replace": True,
                                  "entries": [{"location": "Екатеринбург"},
                                              {"location": "Казань"}]}}]},
        {"done": False, "reason": "выгружаю файл",
         "calls": [{"server": CHAIN_ID, "tool": "export_weather_summary_excel",
                    "arguments": {"dataset_id": DATASET_ID}}]},
    ]
    await run_chat("ок")
    initial_approve = await run_chat("", continue_step=True)
    check("после подтверждения получен файл (работа доведена до конца)",
          any(e.get("type") == "bot" and e.get("files") for e in initial_approve),
          str([e.get("type") for e in initial_approve])[:200])

    # 7.11 Единичные признаки: что считается РЕЗУЛЬТАТОМ, а что чтением.
    check("сохранение набора считается результатом работы",
          mcp_store.produced_result([{"server": CHAIN_ID, "tool": "save_weather_summary",
                                      "arguments": {}, "ok": True, "text": "saved",
                                      "error": ""}]) is True, "")
    check("чтение отчёта результатом НЕ считается",
          mcp_store.produced_result([{"server": CHAIN_ID,
                                      "tool": "get_weather_watch_report",
                                      "arguments": {}, "ok": True, "text": "report",
                                      "error": ""}]) is False, "")
    check("чтения и изменения делятся по имени инструмента",
          [call["tool"] for call in mcp_store.split_calls(
              [{"server": CHAIN_ID, "tool": "get_forecast", "arguments": {}},
               {"server": CHAIN_ID, "tool": "save_weather_summary", "arguments": {}},
               {"server": CHAIN_ID, "tool": "export_weather_summary_excel",
                "arguments": {}}], flat_chain_tools())[1]]
          == ["save_weather_summary", "export_weather_summary_excel"], "")

    # 7.12 ЭКОНОМИЯ ТОКЕНОВ: список инструментов и данные для приёмщика сжаты.
    # Полный дамп схем занимал ~24 000 символов (~8 000 токенов) на КАЖДЫЙ вызов
    # диспетчера, а полный блок данных уходил ещё и в каждый акт проверки.
    rich_tool = {
        "server": CHAIN_ID, "server_name": "chain-mcp", "tool": "get_forecast",
        "description": "Прогноз. " + "Подробное описание. " * 40,
        "schema": {"type": "object",
                   "properties": {
                       "location": {"type": "string",
                                    "description": "Название города. " + "Ещё текст. " * 30},
                       "days": {"type": "integer", "description": "Сколько дней."},
                       "countryCode": {"type": "string", "description": "ISO-код страны."}},
                   "required": ["location"]}}
    compact = mcp_store.tools_text([rich_tool])
    full_dump = json.dumps(rich_tool["schema"], ensure_ascii=False)
    check("список инструментов сжат, а не полный дамп схем",
          len(compact) * 2 < len(full_dump),
          f"сжато: {len(compact)}, схема: {len(full_dump)}")
    check("в сжатом списке есть имена, типы и обязательность аргументов",
          "location!" in compact and "days: integer" in compact
          and "countryCode: string" in compact, compact[:200])
    check("подсказка по аргументу сохранена (язык/страна важны диспетчеру)",
          "ISO-код страны" in compact and "Название города" in compact, compact[:250])

    long_result = [{"server": CHAIN_ID, "tool": "get_forecast", "arguments": {},
                    "ok": True, "text": "данные: " + "7.6 18.3 5.8 0 5.5 3.2 " * 120,
                    "error": ""},
                   {"server": CHAIN_ID, "tool": "export_weather_summary_excel",
                    "arguments": {}, "ok": True, "text": "файл выгружен", "error": "",
                    "attachments": [{"id": "f1", "name": "t.xlsx", "size": 10,
                                     "size_text": "10 Б", "url": "/api/agent/files/f1",
                                     "mime": "application/vnd", "sha256": "x",
                                     "origin": "тест", "previewable": False}]}]
    full_block = mcp_store.block(long_result)
    digest = mcp_store.review_digest(long_result)
    check("сводка для приёмщика короче полного блока данных",
          len(digest) < len(full_block), f"{len(digest)} против {len(full_block)}")
    check("в сводке приёмщика есть вызовы, их исход и файлы",
          f"get_forecast — OK" in digest and "t.xlsx" in digest, digest[:250])
    check("значения источника в сводку приёмщика не попадают",
          "7.6 18.3 5.8" not in digest, digest[:250])
    # Список вызовов обязан дойти ЦЕЛИКОМ: длинная инструкция-заголовок полного
    # блока (про «источник истины») занимала весь предел сводки, список
    # обрезался, и приёмщик писал «сохранение не подтверждено вызовом» при
    # живых вызовах в блоке.
    many = [{"server": CHAIN_ID, "tool": f"save_set_{index}", "arguments": {},
             "ok": True, "text": "saved " + "x" * 400, "error": ""}
            for index in range(6)]
    many.append({"server": CHAIN_ID, "tool": "export_weather_summary_excel",
                 "arguments": {"dataset_id": "ds-1"}, "ok": True,
                 "text": "файл", "error": ""})
    many_digest = mcp_store.review_digest(many)
    check("сводка приёмщика перечисляет ВСЕ вызовы, не обрезая список",
          all(f"save_set_{index}" in many_digest for index in range(6))
          and "export_weather_summary_excel" in many_digest
          and "dataset_id=ds-1" in many_digest,
          many_digest[-300:])
    check("полный блок данных при этом значения сохраняет",
          "7.6 18.3 5.8" in full_block, full_block[:120])
    today = time.strftime("%Y-%m-%d")
    check("в блоке данных есть якорь даты (относительные сроки считаются от него)",
          "СЕГОДНЯ" in full_block and today in full_block,
          full_block[full_block.find("СЕГОДНЯ"):full_block.find("СЕГОДНЯ") + 120])


# ---------------------------------------------------------------------------
# 8. НЕСКОЛЬКО СЕРВЕРОВ: чужие инструменты не подмешиваются, а цепочки
#    остаются КРОСС-СЕРВЕРНЫМИ (запрос может читать одним сервером, а
#    записывать другим).
    # 7.11 ЦЕПОЧКА БЕЗ СЛОВ-МАРКЕРОВ И БЕЗ ОБЪЯВЛЕНИЯ РЕЖИМА МОДЕЛЬЮ — регрессия
    # живого случая 09.10 («погода в городе человека»): модель объявила single,
    # маркеров «сохрани/затем» в запросе нет, а зависимость по данным есть (в
    # списке людей лежит город, а прогноз берётся ПО ГОРОДУ). До правки цикл на
    # этом и заканчивался: агент отвечал по списку («погоды в данных нет») и
    # второго вызова не делал.
    await chat.task_create(chat.TaskCreate(name="Проект: цепочка без маркеров"))
    await chat.session_create()
    # Инструмент-ПОИСК без обязательных аргументов: он возвращает значения (имя
    # человека и его город), которых в запросе НЕТ, — это и есть зависимость по
    # данным в общем виде. Добавляем его НА ВРЕМЯ своего раздела: общий набор
    # инструментов делят другие сценарии (они считают невызванные инструменты).
    lookup_tool = {"name": "list_owners", "title": "Список владельцев",
                   "description": "Список людей с их городами",
                   "schema": {"type": "object", "properties": {}}}
    CHAIN_TOOLS.append(lookup_tool)
    use_chain_registry()
    install_chain_stubs()
    await chat.mcp_apply(McpApply(enabled=[CHAIN_ID]))
    # Тип задачи «всегда сразу ответ»: живой случай шёл именно этим путём (автомат
    # сказал «плана не будет — работа в один шаг»), и цепочка в нём идёт СРАЗУ.
    await chat.session_mode_set(chat._current_session()["id"],
                                SessionMode(mode="answer"))
    reset_calls()
    CHAIN_CALLS.clear()
    MCP_DECISIONS[:] = [
        {"mode": "single", "reason": "сначала список владельцев",
         "calls": [{"server": CHAIN_ID, "tool": "list_owners", "arguments": {}}]},
        {"mode": "chain", "reason": "город найден в списке — беру прогноз по нему",
         "calls": [{"server": CHAIN_ID, "tool": "get_forecast",
                    "arguments": {"location": "Самара"}}]},
        {"done": True, "calls": []},
    ]
    lookup_events = await run_chat(LOOKUP_REQUEST)
    lookup_executed = [call["tool"] for batch in CHAIN_CALLS for call in batch]
    check("цепочка идёт и без слов-маркеров, и без объявления режима моделью",
          lookup_executed == ["list_owners", "get_forecast"], str(lookup_executed))
    check("второй вызов получил значение ИЗ РЕЗУЛЬТАТА первого (город из списка)",
          (CHAIN_CALLS[1][0].get("arguments") or {}).get("location") == "Самара",
          str(CHAIN_CALLS[1:]))
    check("диспетчер в раунде цепочки видел результат первого вызова",
          "Самара" in MCP_PAYLOADS[-1] and "Егор" in MCP_PAYLOADS[-1],
          str(MCP_PAYLOADS[-1])[-200:])
    check("в чате названа причина «остались невызванные инструменты»",
          any("цепочкой" in text and "невызванные инструменты" in text
              for text in texts(lookup_events, "debug")),
          str(texts(lookup_events, "debug"))[-200:])
    if lookup_tool in CHAIN_TOOLS:
        CHAIN_TOOLS.remove(lookup_tool)
    install_chain_stubs()


# ---------------------------------------------------------------------------
REGISTRY_ID = "city_registry"
REGISTRY_REQUEST = "запиши данные о пользователе Иван, живёт в Москве"
CROSS_REQUEST = ("получи прогноз погоды в Москве и запиши в реестр городов, "
                 "что я живу в Москве")
REGISTRY_TOOLS = [
    {"name": "save_city", "title": "Записать жителя",
     "description": "Записывает, что человек живёт в городе",
     "schema": {"type": "object",
                "properties": {"city": {"type": "string"},
                               "resident": {"type": "string"}},
                "required": ["city", "resident"]}},
    {"name": "list_cities", "title": "Показать записи",
     "description": "Возвращает все записи реестра",
     "schema": {"type": "object", "properties": {}, "required": []}},
]


def flat_registry_tools():
    return [{"server": REGISTRY_ID, "server_name": "city-registry-mcp",
             "tool": tool["name"], "description": tool.get("description") or "",
             "schema": tool.get("schema") or {}}
            for tool in REGISTRY_TOOLS]


def test_multi_server():
    print("\n[8] Несколько серверов: чужие инструменты не подмешиваются")
    both = flat_chain_tools() + flat_registry_tools()
    save_city = {"server": REGISTRY_ID, "tool": "save_city",
                 "arguments": {"city": "Москва", "resident": "Иван"}}

    # 8.1 Запрос «запиши в реестр» закрывается ОДНИМ вызовом, даже когда рядом
    # включён погодный сервер, у которого есть инструменты с обязательным id.
    save_only = mcp_store.chain_signals(both, REGISTRY_REQUEST, expected=1)
    check("запись в реестр не уходит в цепочку из-за чужого сервера",
          not save_only["needed"], str(save_only.get("id_tools_relevant")))
    check("виды результата берутся из запроса",
          mcp_store.requested_kinds(REGISTRY_REQUEST) == ["save"],
          str(mcp_store.requested_kinds(REGISTRY_REQUEST)))

    # 8.2 Просьба уже закрыта — уточнять про чужие save-инструменты нечего.
    left = mcp_store.unsatisfied_kinds(REGISTRY_REQUEST, [save_city])
    check("выполненная запись закрывает просьбу «запиши»", left == [], str(left))
    check("уточнение не называет инструменты чужого сервера",
          mcp_store.delivery_candidates(both, {(REGISTRY_ID, "save_city")},
                                        kinds=left) == [],
          str(mcp_store.delivery_candidates(both, {(REGISTRY_ID, "save_city")},
                                            kinds=left)))

    # 8.3 Кросс-серверная цепочка НЕ сужена: после чтения погоды диспетчеру
    # видны создающие инструменты обоих серверов.
    read_only = [{"server": CHAIN_ID, "tool": "get_forecast",
                  "arguments": {"location": "Москва"}}]
    cross_left = mcp_store.unsatisfied_kinds(CROSS_REQUEST, read_only)
    cross = mcp_store.delivery_candidates(both, {(CHAIN_ID, "get_forecast")},
                                          kinds=cross_left)
    pairs = {(item["server"], item["tool"]) for item in cross}
    check("в кросс-серверном запросе видны создающие инструменты обоих серверов",
          (CHAIN_ID, "save_weather_summary") in pairs
          and (REGISTRY_ID, "save_city") in pairs,
          str(sorted(pairs)))
    check("чужая выгрузка в кросс-серверный запрос не подмешивается",
          all(item["kind"] == "save" for item in cross),
          str([(item["tool"], item["kind"]) for item in cross]))

    # 8.4 Очевидная достройка снова срабатывает при двух включённых серверах:
    # ожидаемый вид остаётся один, и «ровно один кандидат» не ломается.
    after_save = [{"server": CHAIN_ID, "tool": "get_forecast", "arguments": {}},
                  {"server": CHAIN_ID, "tool": "save_weather_summary",
                   "arguments": {"dataset_id": DATASET_ID}}]
    export_left = mcp_store.unsatisfied_kinds(DELIVER_REQUEST, after_save)
    done = {(CHAIN_ID, "get_forecast"), (CHAIN_ID, "save_weather_summary")}
    auto = mcp_store.auto_followup(both, done, {"dataset_id": DATASET_ID},
                                   kinds=export_left)
    check("очевидная выгрузка выполняется без диспетчера и при втором сервере",
          bool(auto) and auto["tool"] == "export_weather_summary_excel"
          and auto["server"] == CHAIN_ID, str(auto))
    check("после сохранения ожидается ровно выгрузка", export_left == ["export"],
          str(export_left))

    # 8.5 Погодная цепочка не потеряла ни маршрут, ни правдивую причину.
    weather = mcp_store.chain_signals(both, DELIVER_REQUEST, expected=2)
    check("погодная цепочка при двух серверах по-прежнему нужна",
          weather["needed"], str(weather.get("id_tools_relevant")))
    check("в причине цепочки нет инструментов чужого сервера",
          [item["tool"] for item in weather["id_tools_relevant"]]
          == ["export_weather_summary_excel"],
          mcp_store.chain_reason(weather))

    # 8.6 Глобальная проверка «остались ли создающие инструменты» (после выдачи
    # файла) по-прежнему видит весь набор: вид её не сужает.
    names = {item["tool"] for item in mcp_store.delivery_candidates(both, set())}
    check("без ограничения по виду видны создающие инструменты всех серверов",
          names >= {"save_weather_summary", "export_weather_summary_excel",
                    "save_city"},
          str(sorted(names)))


def use_two_server_registry() -> None:
    """Реестр проекта из ДВУХ серверов: погодный и реестр городов."""
    mcp_store.SERVER_IDS = [CHAIN_ID, REGISTRY_ID]
    mcp_store.forget()


def install_two_server_stubs() -> None:
    """Заглушки обоих серверов: инструменты отвечают локально, без процессов."""
    table = {CHAIN_ID: ("chain-mcp", CHAIN_TOOLS),
             REGISTRY_ID: ("city-registry-mcp", REGISTRY_TOOLS)}

    def entry(server_id):
        name, tools = table.get(str(server_id), ("", []))
        if not name:
            return {"id": str(server_id), "ok": False, "error": "нет такого сервера",
                    "server_name": "", "server_version": "", "tools": []}
        return {"id": str(server_id), "ok": True, "error": "", "server_name": name,
                "server_version": "0.1", "tools": [dict(tool) for tool in tools]}

    async def fake_discover(ids=None, force=False):
        return [entry(server_id) for server_id in (CHAIN_ID, REGISTRY_ID)]

    def fake_discover_sync(server_id=None, force=False):
        return entry(server_id)

    async def fake_run_calls(calls, limit=None):
        cap = mcp_store.MAX_CALLS_PER_REQUEST if limit is None else max(0, int(limit))
        calls = list(calls)[:cap]
        CHAIN_CALLS.append(list(calls))
        out = []
        for call in calls:
            tool = str(call.get("tool") or "")
            server_id = str(call.get("server") or "")
            arguments = dict(call.get("arguments") or {})
            name = table.get(server_id, ("", []))[0]
            if tool == "save_city":
                text = (f"Сохранено: {arguments.get('resident')} — "
                        f"{arguments.get('city')}.")
            elif tool == "list_cities":
                text = "Записей нет."
            elif tool == "get_forecast":
                text = (f"Прогноз для города {arguments.get('location')}: "
                        "2026-09-27, минимум 6.9, максимум 17.5")
            else:
                text = f"{tool}: готово."
            out.append({"server": server_id, "server_name": name,
                        "source": "локальный тест", "tool": tool,
                        "arguments": arguments, "ok": True, "text": text,
                        "error": "", "attachments": []})
        return out

    mcp_store.async_discover = fake_discover
    mcp_store.async_run_calls = fake_run_calls
    mcp_store.discover = fake_discover_sync
    chat.mcp_store.async_discover = fake_discover
    chat.mcp_store.async_run_calls = fake_run_calls
    chat.mcp_store.discover = fake_discover_sync


async def test_two_servers_flow():
    """Живой случай: запрос-запись при ДВУХ включённых серверах.

    Проверяется не эвристика, а весь ход диалога: запись выполняется после
    подтверждения плана, лишнего уточнения про инструменты чужого сервера нет, а
    в отладке не появляются погодные имена (раньше запрос про реестр городов
    уходил в цепочку из-за погодных наблюдений, а после save_city агент уточнял
    про save_weather_summary).
    """
    print("\n[9] Живой случай: запись в реестр при двух включённых серверах")
    await chat.task_create(chat.TaskCreate(name="Проект: погода и реестр городов"))
    await chat.session_create()
    use_two_server_registry()
    await chat.mcp_apply(McpApply(enabled=[CHAIN_ID, REGISTRY_ID]))
    install_two_server_stubs()
    reset_calls()
    CHAIN_CALLS.clear()

    save_call = {"server": REGISTRY_ID, "tool": "save_city",
                 "arguments": {"city": "Москва", "resident": "Иван"}}
    MCP_DECISIONS[:] = [
        # ЗАПРОС: диспетчер выбирает запись — до «ок» она откладывается.
        {"mode": "single", "reason": "нужно записать", "calls": [dict(save_call)]},
        # ШАГ 1 (не последний): цепочка продолжается и здесь, но только чтениями —
        # запись снова откладывается до последнего шага.
        {"done": True, "reason": "для этого шага данных хватает", "calls": []},
        # ПОСЛЕДНИЙ шаг: запись выполняется именно здесь.
        {"mode": "single", "reason": "записываю", "calls": [dict(save_call)]},
        # ...и цепочка закрывается: без этого решения стенд уходит в ЗАПАСНОЙ
        # ответ диспетчера (список MCP_CALLS), где лежат вызовы прежних проверок.
        {"done": True, "reason": "запись выполнена", "calls": []},
    ]
    request_events = await run_chat(REGISTRY_REQUEST)
    await run_chat("ок")
    step_events = await run_chat("", continue_step=True)

    executed = [call["tool"] for batch in CHAIN_CALLS for call in batch]
    check("запись в реестр выполнена после подтверждения плана",
          executed == ["save_city"], str(executed))
    nudges = [payload for payload in MCP_PAYLOADS
              if "ЗАПРОС ВЫПОЛНЕН НЕ ПОЛНОСТЬЮ" in payload]
    check("уточнения про чужие инструменты нет (запись уже сделана)",
          nudges == [], str(nudges)[:200])
    debug = texts(request_events, "debug") + texts(step_events, "debug")
    noisy = [line for line in debug
             if "weather" in line.lower() or "наблюдени" in line.lower()]
    check("в отладке нет погодных инструментов", not noisy, str(noisy)[:200])
    check("запись в реестр не объявлена многошаговой цепочкой",
          not any("многошаговая" in line for line in debug),
          str([line for line in debug if "многошаговая" in line])[:200])
    MCP_DECISIONS.clear()


async def test_chain_data_on_step():
    """Зависимые ЧТЕНИЯ добываются на ТОМ шаге, который их требует.

    Живой случай: «проверь прогноз погоды в городах Славы и Ивана» — план из двух
    шагов («получить прогноз» → «сообщить кратко»). До «ок» цепочка успела только
    прочитать реестр городов, а прогноз по найденным городам (зависимый вызов)
    добывался лишь на ПОСЛЕДНЕМ шаге. Шаг 1 отвечал «данных о погоде нет», этот
    ответ уходил в контекст шага 2, и модель противоречила уже полученным данным:
    проверка отклоняла оба шага, задача уходила на доработку и выполнялась только
    со второго раза. Здесь проверяется, что данные приходят на ПЕРВОМ шаге.
    """
    print("\n[10] Цепочка из разных MCP: данные приходят на нужный шаг")
    await chat.task_create(chat.TaskCreate(name="Проект: прогноз по городам"))
    await chat.session_create()
    use_two_server_registry()
    await chat.mcp_apply(McpApply(enabled=[CHAIN_ID, REGISTRY_ID]))
    install_two_server_stubs()
    reset_calls()
    CHAIN_CALLS.clear()

    MCP_DECISIONS[:] = [
        # ЗАПРОС: до «ок» читается реестр городов — из него находятся города Славы
        # и Ивана. Цепочка остаётся не доигранной: зависимые чтения ждут «ок».
        {"mode": "chain", "reason": "нужен реестр городов",
         "calls": [{"server": REGISTRY_ID, "tool": "list_cities",
                    "arguments": {}}]},
        # ШАГ 1: цепочка продолжается ЗДЕСЬ и добирает прогнозы по обоим городам.
        {"mode": "chain", "reason": "нужны прогнозы по обоим городам",
         "calls": [{"server": CHAIN_ID, "tool": "get_forecast",
                    "arguments": {"location": "Москва"}},
                   {"server": CHAIN_ID, "tool": "get_forecast",
                    "arguments": {"location": "Екатеринбург"}}]},
        # ШАГ 2 (последний): данных уже хватает, к серверам не ходим.
        {"done": True, "reason": "данные получены", "calls": []},
    ]
    # force_plan: раздел проверяет, что ЦЕПОЧКА ВЫЗОВОВ доигрывается НА ШАГЕ
    # выполнения (план, подтверждение, шаги). Гейт «ответ или план» отправил бы
    # такую просьбу прямым ответом — тоже рабочий путь (см. 5.6), но другой.
    await run_chat("прогноз для городов Славы и Ивана", force_plan=True)
    reset_calls()
    CHAIN_CALLS.clear()

    ok_events = await run_chat("ок")
    ok_calls = [call["tool"] for batch in CHAIN_CALLS for call in batch]
    check("прогноз добыт на ПЕРВОМ шаге, а не только на последнем",
          ok_calls == ["get_forecast", "get_forecast"], str(ok_calls))
    check("цепочка продолжилась именно на шаге выполнения",
          any("раунд 1 цепочки" in line for line in texts(ok_events, "debug")),
          str(texts(ok_events, "debug"))[-200:])
    check("шаг, которому нужны данные, получает их в контекст модели",
          "Прогноз для города Москва" in all_context()
          and "Прогноз для города Екатеринбург" in all_context(),
          all_context()[-300:])

    reset_calls()
    CHAIN_CALLS.clear()
    await run_chat("", continue_step=True)
    step2_calls = [call["tool"] for batch in CHAIN_CALLS for call in batch]
    check("на последнем шаге новых вызовов не требуется", not step2_calls,
          str(step2_calls))
    check("ответ последнего шага строится по полученным данным",
          "Прогноз для города Москва" in all_context(), all_context()[-200:])
    MCP_DECISIONS.clear()


def main():
    print("Проверка MCP (внешних инструментов агента) — без сети и ключей")
    _write_fake_servers()
    test_client()
    test_parse_calls()
    test_storage()
    # Цикл создаётся ЯВНО: asyncio.get_event_loop() устарел в Python 3.12.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(test_routes())
    loop.run_until_complete(test_dialog())
    test_http()
    loop.run_until_complete(test_chain())
    test_multi_server()
    loop.run_until_complete(test_two_servers_flow())
    loop.run_until_complete(test_chain_data_on_step())
    print("\nИтог: " + (f"ПРОВАЛЕНО проверок: {len(FAILURES)} → {FAILURES}"
                       if FAILURES else "все проверки пройдены"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
