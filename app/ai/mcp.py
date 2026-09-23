"""MCP (Model Context Protocol) для режима «AI-агент».

MCP — открытый протокол, по которому модель получает ВНЕШНИЕ ИНСТРУМЕНТЫ: сервер
объявляет список инструментов (`tools/list`) с описанием и схемой аргументов, а
клиент вызывает их (`tools/call`) и получает текстовый результат. Этот модуль —
клиент MCP и реестр серверов проекта.

Зачем он нужен в этом приложении: у модели нет доступа к сети, поэтому «какая
сейчас погода в Москве» или «курс доллара» она может только выдумать. Включённые
MCP-серверы проекта уходят в КАЖДЫЙ запрос агента: сначала служебный вызов
выбирает нужные инструменты и их аргументы по запросу пользователя
(`choose`), затем инструменты выполняются (`run_calls`), а полученные данные
уходят в модель отдельным системным блоком (`block`) — вместе с планом задачи,
ответом и проверкой результата.

Серверы — ЛОКАЛЬНЫЕ процессы на официальном MCP SDK (@modelcontextprotocol/sdk,
каталог mcp_servers/), общение по stdio: JSON-RPC 2.0 построчно. Все три сервера
проекта бесплатные и работают БЕЗ ключей и регистрации:

    weather   — погода (7timer.info + геокодер Open-Meteo);
    currency  — курсы валют Банка России (cbr.ru);
    crypto    — курсы криптовалют (CoinGecko).

Клиент сам по себе не зависит от SDK: он говорит на протоколе, поэтому к проекту
можно подключить любой MCP-сервер (в том числе сторонний) — достаточно добавить
запись в `SERVERS` (или переопределить каталог переменной окружения
`MCP_SERVERS_FILE`).

Модуль без состояния в файлах: на вход приходят включённые id серверов и запрос
пользователя, на выход — вызовы и текст блока. Работа с моделью идёт через
готовую функцию вызова LLM (`choose`, как в app/ai/invariants.py), поэтому
модуль проверяется без сети (см. tools/check_mcp.py).
"""

import asyncio
import json
import logging
import os
import queue
import shutil
import subprocess
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from app import config
from app.ai import json_utils

logger = logging.getLogger(__name__)

# Формат вызова LLM: async (messages, **kwargs) -> (текст, метрики|None).
LlmCall = Callable[..., Any]

# Версия протокола MCP, которую объявляет клиент. Серверы на официальном SDK
# отвечают своей версией — клиент принимает ответ как есть (совместимость в
# пределах 2024-11-05/2025-06-18 для инструментов не отличается).
MCP_PROTOCOL_VERSION = "2024-11-05"
CLIENT_NAME = "fastapi-site-agent"
CLIENT_VERSION = "1.0.0"

# Таймауты (секунды). Запуск node и импорт SDK занимают доли секунды, но первый
# запуск может быть медленнее; вызов инструмента ходит во внешний источник.
INIT_TIMEOUT = 25.0
LIST_TIMEOUT = 25.0
CALL_TIMEOUT = 30.0
CLOSE_TIMEOUT = 3.0

# Сколько времени держим в памяти список инструментов сервера (успешный ответ) и
# сколько — сбой (сбой кэшируем ненадолго, чтобы не дёргать сервер на каждый
# запрос, но и не показывать «недоступен» после починки сети).
TOOLS_TTL = 600.0
ERROR_TTL = 60.0

# Сколько инструментов агент может вызвать за ОДИН запрос (каждый вызов — это
# запуск сервера и обращение к внешнему источнику) и сколько символов результата
# уходит в контекст модели.
MAX_CALLS_PER_REQUEST = 3
RESULT_CHARS = 4000
BLOCK_CHARS = 12000
# Сколько символов описания инструмента показываем модели и интерфейсу.
DESCRIPTION_CHARS = 400
TOOLS_MAX_TOKENS = 700
TOOLS_TIMEOUT = 45.0

# ---------------------------------------------------------------------------
# Реестр серверов проекта.
#
# Каждая запись — описание сервера MCP: имя для интерфейса, краткое пояснение
# (его видит пользователь в диалоге «MCP») и команда запуска. `command` — либо
# готовая команда (["node", "script.mjs"]), либо имя файла в каталоге серверов
# (тогда запускается через node). Каталог по умолчанию — mcp_servers/ в корне
# проекта; путь можно переопределить переменной окружения MCP_SERVERS_DIR.
# ---------------------------------------------------------------------------
SERVERS_DIR_ENV = "MCP_SERVERS_DIR"
NODE_ENV = "MCP_NODE_BIN"

WEATHER = "weather"
CURRENCY = "currency"
CRYPTO = "crypto"

SERVERS: List[Dict[str, Any]] = [
    {
        "id": WEATHER,
        "name": "Погода",
        "description": (
            "Текущая погода и прогноз по дням для любого города. "
            "Данные 7timer.info (модель NOAA GFS) и геокодера Open-Meteo."
        ),
        "source": "7timer.info · Open-Meteo Geocoding",
        "script": "weather.mjs",
    },
    {
        "id": CURRENCY,
        "name": "Курсы валют",
        "description": (
            "Официальные курсы Банка России к рублю: курс валюты на дату, "
            "перевод суммы из валюты в валюту, список действующих курсов."
        ),
        "source": "cbr.ru (Банк России)",
        "script": "currency.mjs",
    },
    {
        "id": CRYPTO,
        "name": "Криптовалюты",
        "description": (
            "Цены криптовалют в обычных валютах и обзор рынка монеты: "
            "капитализация, объём, изменение за сутки и за неделю."
        ),
        "source": "CoinGecko",
        "script": "crypto.mjs",
    },
]

# Пометка в диагностике: серверов нет вовсе (реестр пуст).
SERVER_IDS = [entry["id"] for entry in SERVERS]

# Порядок и вид данных в блоке, который уходит модели.
BLOCK_HEADER = (
    "ДАННЫЕ MCP (внешние инструменты) — это УЖЕ ПОЛУЧЕННЫЕ фактические данные по "
    "текущему запросу пользователя. Опирайся на них как на источник истины: "
    "используй эти числа и формулировки в плане, шагах и ответе, не выдумывай "
    "других значений и не пересчитывай их по памяти. Если нужных данных в блоке "
    "нет — честно скажи, что их нет, и не подменяй их догадкой."
)

# Инструкция служебного вызова: какие инструменты вызвать по запросу.
TOOLS_PROMPT = (
    "Ты — диспетчер внешних инструментов (MCP). Тебе дают ЗАПРОС ПОЛЬЗОВАТЕЛЯ и "
    "СПИСОК ДОСТУПНЫХ ИНСТРУМЕНТОВ (сервер, имя, назначение, схема аргументов в "
    "формате JSON Schema).\n"
    "Реши, каких данных не хватает для ответа, и вызови ровно те инструменты, "
    "которые эти данные дают.\n"
    "ПРАВИЛА:\n"
    "1) Вызывай инструмент ТОЛЬКО если без него ответить нельзя: погода, курс, "
    "цена, прогноз. Если запрос можно выполнить и без внешних данных (написать "
    "код, объяснить, посчитать по условию) — верни пустой список.\n"
    "2) Не больше " + str(MAX_CALLS_PER_REQUEST) + " вызовов за запрос. Не "
    "вызывай один и тот же инструмент дважды.\n"
    "3) Бери инструменты ТОЛЬКО из списка: имя сервера и имя инструмента должны "
    "совпадать с ним точно. Придумывать инструменты нельзя.\n"
    "4) Аргументы — строго по схеме: обязательные поля заполнены, лишних нет. "
    "Названия городов и валют бери из запроса пользователя; город пиши словами "
    "(«Москва»), валюту — кодом (USD, EUR, RUB).\n"
    "5) Если в запросе не хватает обязательного для вызова сведения (например, "
    "не назван город) — НЕ вызывай инструмент: пусть агент сначала уточнит "
    "запрос у пользователя.\n"
    "Ответ — ТОЛЬКО JSON без пояснений:\n"
    '{"calls": [{"server": "weather", "tool": "get_weather", '
    '"arguments": {"city": "Москва"}}]}\n'
    "Если внешние данные не нужны — {\"calls\": []}."
)

# Служебные маркеры ошибок сервера.
_EOF = object()


class McpError(Exception):
    """Сбой MCP: сервер не запустился, не ответил, отказал или вернул ошибку."""


# ---------------------------------------------------------------------------
# Каталог серверов и путь к их файлам
# ---------------------------------------------------------------------------
def servers_dir() -> str:
    """Каталог локальных MCP-серверов (по умолчанию mcp_servers/ в корне)."""
    return os.getenv(SERVERS_DIR_ENV) or os.path.join(config.PROJECT_ROOT, "mcp_servers")


def node_bin() -> str:
    """Путь к node (переопределяется переменной окружения MCP_NODE_BIN)."""
    return os.getenv(NODE_ENV) or "node"


def servers() -> List[Dict[str, Any]]:
    """Реестр серверов (копии записей — вызывающий их не портит)."""
    return [dict(entry) for entry in SERVERS]


def find_server(server_id: Any) -> Optional[Dict[str, Any]]:
    """Запись реестра по id (None — такого сервера нет)."""
    key = str(server_id or "").strip().lower()
    for entry in SERVERS:
        if entry["id"] == key:
            return dict(entry)
    return None


def command_for(entry: Dict[str, Any]) -> List[str]:
    """Команда запуска сервера (node + файл сервера в каталоге серверов)."""
    command = entry.get("command")
    if isinstance(command, list) and command:
        return [str(part) for part in command]
    script = str(entry.get("script") or "").strip()
    return [node_bin(), os.path.join(servers_dir(), script)]


def availability_error(entry: Dict[str, Any]) -> str:
    """Почему сервер заведомо не запустится (пусто — предпосылок к сбою нет).

    Проверяем то, что видно без запуска: есть ли запускаемая программа, файл
    сервера и установленный MCP SDK (для серверов на Node — а ими и являются
    серверы проекта). Так пользователь в диалоге «MCP» видит понятную причину
    («Node.js не найден», «не выполнен npm install»), а не молчаливое
    «недоступен».

    Сервер с собственной командой (`command` в записи реестра) проверяется только
    на существование программы и файла: SDK нужен лишь серверам на Node.
    """
    command = command_for(entry)
    binary = command[0]
    if os.path.isabs(binary) or os.sep in binary:
        if not os.path.isfile(binary):
            return "не найден запускаемый файл: " + binary
    elif shutil.which(binary) is None:
        return "не найден " + binary + " (нужен Node.js для локальных MCP-серверов)"
    script = command[-1]
    if script.endswith((".mjs", ".js", ".cjs")) and not os.path.isfile(script):
        return "не найден файл сервера: " + script
    if os.path.basename(binary).startswith("node"):
        modules = os.path.join(servers_dir(), "node_modules",
                               "@modelcontextprotocol", "sdk")
        if not os.path.isdir(modules):
            return ("не установлен MCP SDK: выполните npm install в каталоге "
                    + servers_dir())
    return ""


# ---------------------------------------------------------------------------
# Клиент MCP по stdio: JSON-RPC 2.0 построчно
# ---------------------------------------------------------------------------
class _StdioSession:
    """Одно соединение с MCP-сервером: запуск процесса, запросы, закрытие.

    Протокол — JSON-RPC 2.0, по одному объекту в строке. stdout сервера занят
    протоколом, поэтому диагностика сервера идёт в stderr и попадает в текст
    ошибки клиента: без него сбой выглядел бы как «сервер не ответил».
    """

    def __init__(self, command: List[str], cwd: str) -> None:
        self.command = list(command)
        self.cwd = cwd
        self._queue: "queue.Queue" = queue.Queue()
        self._stderr: List[str] = []
        self._send_lock = threading.Lock()
        self._next_id = 0
        self._closed = False
        self.started = time.monotonic()
        # Имя и версия сервера из рукопожатия (initialize): tools/list их не
        # возвращает, а интерфейсу нужно показать, ЧТО именно ответило.
        self.server_name = ""
        self.server_version = ""
        try:
            self._proc = subprocess.Popen(
                self.command,
                cwd=cwd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise McpError("не удалось запустить сервер: " + str(exc))
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

    # -- чтение потоков процесса -------------------------------------------
    def _read_stdout(self) -> None:
        stream = self._proc.stdout
        try:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except ValueError:
                    # Не JSON в stdout — не наш протокол: пропускаем строку,
                    # чтобы одна диагностическая печать не ломала соединение.
                    continue
                if isinstance(message, dict):
                    self._queue.put(message)
        except (OSError, ValueError):
            pass
        finally:
            self._queue.put(_EOF)

    def _read_stderr(self) -> None:
        stream = self._proc.stderr
        try:
            for line in stream:
                text = line.strip()
                if text:
                    self._stderr.append(text)
                    del self._stderr[:-10]
        except (OSError, ValueError):
            pass

    def _diagnostics(self, reason: str) -> str:
        code = self._proc.poll()
        parts = [reason]
        if code is not None:
            parts.append(f"процесс завершился с кодом {code}")
        if self._stderr:
            parts.append("вывод сервера: " + " | ".join(self._stderr[-3:]))
        return "; ".join(parts)

    # -- протокол ----------------------------------------------------------
    def notify(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        """Уведомление серверу (ответа не ждём): notifications/initialized."""
        payload: Dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        try:
            self._proc.stdin.write(json.dumps(payload) + "\n")
            self._proc.stdin.flush()
        except (OSError, ValueError) as exc:
            raise McpError(self._diagnostics("сервер закрыл ввод: " + str(exc)))

    def request(self, method: str, params: Optional[Dict[str, Any]] = None,
                timeout: float = LIST_TIMEOUT) -> Dict[str, Any]:
        """Запрос к серверу: возвращает result (сбой — McpError)."""
        with self._send_lock:
            self._next_id += 1
            request_id = self._next_id
            payload: Dict[str, Any] = {"jsonrpc": "2.0", "id": request_id,
                                       "method": method}
            if params is not None:
                payload["params"] = params
            try:
                self._proc.stdin.write(json.dumps(payload) + "\n")
                self._proc.stdin.flush()
            except (OSError, ValueError) as exc:
                raise McpError(self._diagnostics("сервер закрыл ввод: " + str(exc)))
            deadline = time.monotonic() + max(1.0, float(timeout))
            while True:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise McpError(self._diagnostics(
                        f"сервер не ответил на {method} за {timeout:.0f} с"))
                try:
                    message = self._queue.get(timeout=left)
                except queue.Empty:
                    raise McpError(self._diagnostics(
                        f"сервер не ответил на {method} за {timeout:.0f} с"))
                if message is _EOF:
                    raise McpError(self._diagnostics(
                        f"сервер завершился, не ответив на {method}"))
                if message.get("id") != request_id:
                    # Чужой ответ или уведомление сервера — ждём свой.
                    continue
                if message.get("error"):
                    error = message.get("error")
                    text = error.get("message") if isinstance(error, dict) else str(error)
                    raise McpError(f"{method}: сервер вернул ошибку: {text}")
                result = message.get("result")
                return result if isinstance(result, dict) else {}

    def close(self) -> None:
        """Закрывает соединение (сначала вежливо, потом силой)."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._proc.stdin and not self._proc.stdin.closed:
                self._proc.stdin.close()
        except OSError:
            pass
        try:
            self._proc.terminate()
            self._proc.wait(timeout=CLOSE_TIMEOUT)
        except (OSError, subprocess.TimeoutExpired):
            try:
                self._proc.kill()
                self._proc.wait(timeout=CLOSE_TIMEOUT)
            except (OSError, subprocess.TimeoutExpired):
                logger.warning("MCP: не удалось остановить процесс %s", self.command)


def _open_session(entry: Dict[str, Any]) -> _StdioSession:
    """Запускает сервер и выполняет рукопожатие MCP (initialize)."""
    problem = availability_error(entry)
    if problem:
        raise McpError(problem)
    session = _StdioSession(command_for(entry), servers_dir())
    try:
        result = session.request("initialize", {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": CLIENT_NAME, "version": CLIENT_VERSION},
        }, timeout=INIT_TIMEOUT)
        # Обязательное уведомление протокола: без него сервер не считает
        # соединение готовым (SDK ждёт его до работы с инструментами).
        session.notify("notifications/initialized")
        info = result.get("serverInfo") if isinstance(result.get("serverInfo"), dict) else {}
        session.server_name = str(info.get("name") or "").strip()
        session.server_version = str(info.get("version") or "").strip()
    except McpError:
        session.close()
        raise
    return session


def _tool_entry(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Описание инструмента сервера (для модели и для интерфейса)."""
    name = str(raw.get("name") or "").strip()
    description = str(raw.get("description") or "").strip()
    title = str(raw.get("title") or "").strip()
    schema = raw.get("inputSchema")
    return {
        "name": name,
        "title": title or name,
        "description": description[:DESCRIPTION_CHARS],
        "schema": schema if isinstance(schema, dict) else {},
    }


# ---------------------------------------------------------------------------
# Обнаружение инструментов (tools/list) с кэшем в памяти процесса
# ---------------------------------------------------------------------------
_TOOLS_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_CACHE_LOCK = threading.Lock()


def _cached(server_id: str) -> Optional[Dict[str, Any]]:
    with _CACHE_LOCK:
        item = _TOOLS_CACHE.get(server_id)
    if item is None:
        return None
    stamp, payload = item
    ttl = TOOLS_TTL if payload.get("ok") else ERROR_TTL
    if time.monotonic() - stamp > ttl:
        return None
    return payload


def _remember(server_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    with _CACHE_LOCK:
        _TOOLS_CACHE[server_id] = (time.monotonic(), payload)
    return payload


def forget(server_id: str = "") -> None:
    """Сбрасывает кэш инструментов (пусто — весь): нужен проверкам и перезапуску."""
    with _CACHE_LOCK:
        if server_id:
            _TOOLS_CACHE.pop(str(server_id), None)
        else:
            _TOOLS_CACHE.clear()


def discover(server_id: str, force: bool = False) -> Dict[str, Any]:
    """Список инструментов сервера (с кэшем): {"ok", "tools", "error", ...}.

    Сбой не бросает исключение: интерфейсу нужен ответ «сервер недоступен и вот
    почему», а агенту — просто отсутствие инструментов (запрос выполняется без
    внешних данных, как раньше).
    """
    entry = find_server(server_id)
    if entry is None:
        return {"id": str(server_id), "ok": False, "tools": [],
                "error": "сервер не подключён к проекту"}
    if not force:
        cached = _cached(entry["id"])
        if cached is not None:
            return dict(cached)
    session = None
    try:
        session = _open_session(entry)
        result = session.request("tools/list", {}, timeout=LIST_TIMEOUT)
        tools = [_tool_entry(item) for item in (result.get("tools") or [])
                 if isinstance(item, dict) and str(item.get("name") or "").strip()]
        payload = {
            "id": entry["id"],
            "ok": True,
            "tools": tools,
            # Имя сервера — из рукопожатия (initialize), а не из tools/list.
            "server_name": session.server_name or entry["id"],
            "server_version": session.server_version,
            "error": "",
        }
    except McpError as exc:
        logger.warning("MCP %s: список инструментов не получен: %s", entry["id"], exc)
        payload = {"id": entry["id"], "ok": False, "tools": [],
                   "server_name": entry["id"], "server_version": "",
                   "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 — сбой сервера не должен ломать запрос
        logger.warning("MCP %s: неожиданный сбой обнаружения: %s", entry["id"], exc)
        payload = {"id": entry["id"], "ok": False, "tools": [],
                   "server_name": entry["id"], "server_version": "",
                   "error": str(exc)}
    finally:
        if session is not None:
            session.close()
    return _remember(entry["id"], payload)


def discover_many(ids: Optional[List[str]] = None,
                  force: bool = False) -> List[Dict[str, Any]]:
    """Инструменты нескольких серверов (последовательно, в порядке id)."""
    wanted = [str(item) for item in (ids if ids is not None else SERVER_IDS)]
    return [discover(server_id, force=force) for server_id in wanted]


# ---------------------------------------------------------------------------
# Вызов инструмента (tools/call)
# ---------------------------------------------------------------------------
def _content_text(result: Dict[str, Any]) -> str:
    """Текст результата инструмента (MCP content -> строка)."""
    parts: List[str] = []
    for item in (result.get("content") or []):
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            parts.append(str(item.get("text") or ""))
        elif item.get("type") == "resource":
            resource = item.get("resource")
            if isinstance(resource, dict) and resource.get("text"):
                parts.append(str(resource.get("text")))
    return "\n".join(part for part in parts if part.strip()).strip()


def call_tool(server_id: str, tool: str, arguments: Optional[Dict[str, Any]] = None,
              timeout: float = CALL_TIMEOUT) -> Dict[str, Any]:
    """Вызывает инструмент сервера: {"ok", "text", "error"}.

    Каждый вызов — отдельный процесс сервера: соединение не переиспользуется,
    поэтому «залипший» сервер не портит следующие запросы. Ошибку источника
    инструмент возвращает сам (isError), и её текст уходит модели как есть.
    """
    entry = find_server(server_id)
    if entry is None:
        return {"ok": False, "text": "", "error": "сервер не подключён к проекту"}
    name = str(tool or "").strip()
    if not name:
        return {"ok": False, "text": "", "error": "не указан инструмент"}
    args = arguments if isinstance(arguments, dict) else {}
    session = None
    try:
        session = _open_session(entry)
        result = session.request("tools/call", {"name": name, "arguments": args},
                                 timeout=timeout)
    except McpError as exc:
        logger.warning("MCP %s.%s: вызов не удался: %s", entry["id"], name, exc)
        return {"ok": False, "text": "", "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        logger.warning("MCP %s.%s: неожиданный сбой вызова: %s", entry["id"], name, exc)
        return {"ok": False, "text": "", "error": str(exc)}
    finally:
        if session is not None:
            session.close()
    text = _content_text(result)
    # isError — инструмент отработал, но источник данных отказал: текст ошибки
    # уходит модели, чтобы она не выдумывала значения вместо недоступных.
    failed = bool(result.get("isError"))
    return {
        "ok": not failed,
        "text": text[:RESULT_CHARS],
        "error": "" if not failed else (text[:400] or "инструмент вернул ошибку"),
    }


def run_calls(calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Выполняет вызовы инструментов и возвращает результаты (последовательно).

    Последовательно — намеренно: вызовы ходят в один и тот же локальный node, а
    параллельный запуск нескольких процессов только добавил бы нагрузку; за
    запрос их и так не больше MAX_CALLS_PER_REQUEST.
    """
    results: List[Dict[str, Any]] = []
    for call in (calls or [])[:MAX_CALLS_PER_REQUEST]:
        if not isinstance(call, dict):
            continue
        server_id = str(call.get("server") or "").strip()
        tool = str(call.get("tool") or "").strip()
        arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        outcome = call_tool(server_id, tool, arguments)
        entry = find_server(server_id) or {}
        results.append({
            "server": server_id,
            "server_name": str(entry.get("name") or server_id),
            "source": str(entry.get("source") or ""),
            "tool": tool,
            "arguments": dict(arguments),
            "ok": bool(outcome.get("ok")),
            "text": str(outcome.get("text") or ""),
            "error": str(outcome.get("error") or ""),
        })
    return results


# ---------------------------------------------------------------------------
# Асинхронные обёртки: запуск процессов не должен блокировать цикл событий
# ---------------------------------------------------------------------------
async def async_discover(ids: Optional[List[str]] = None,
                         force: bool = False) -> List[Dict[str, Any]]:
    """Асинхронный список инструментов серверов (процессы — в отдельном потоке)."""
    return await asyncio.to_thread(discover_many, ids, force)


async def async_run_calls(calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Асинхронное выполнение вызовов (процессы — в отдельном потоке)."""
    return await asyncio.to_thread(run_calls, calls)


# ---------------------------------------------------------------------------
# Выбор инструментов служебным вызовом модели
# ---------------------------------------------------------------------------
def tools_text(tools: List[Dict[str, Any]]) -> str:
    """Список инструментов текстом для модели: сервер, имя, описание, схема."""
    lines: List[str] = []
    for tool in tools:
        lines.append(
            f"- сервер: {tool.get('server')} ({tool.get('server_name') or ''}); "
            f"инструмент: {tool.get('tool')}; назначение: {tool.get('description') or ''}")
        schema = tool.get("schema")
        if isinstance(schema, dict) and schema:
            lines.append("  аргументы: " + json.dumps(schema, ensure_ascii=False))
    return "\n".join(lines) if lines else "(инструментов нет)"


def build_query(user_message: str, tools: List[Dict[str, Any]]) -> str:
    """Текст запроса к модели: запрос пользователя + доступные инструменты."""
    return (
        "ЗАПРОС ПОЛЬЗОВАТЕЛЯ:\n" + (str(user_message or "").strip() or "(пусто)") +
        "\n\nДОСТУПНЫЕ ИНСТРУМЕНТЫ:\n" + tools_text(tools)
    )


def _load_object(content: str) -> Any:
    """JSON из ответа модели (снимаем ```-ограждение и чиним обрыв)."""
    text = str(content or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = text.strip("`")
        newline = text.find("\n")
        if newline != -1 and text[:newline].strip().lower() in ("json", "javascript"):
            text = text[newline + 1:]
    try:
        return json.loads(text)
    except ValueError:
        pass
    repaired = json_utils.repair_json(text)
    if not repaired:
        return None
    try:
        return json.loads(repaired)
    except ValueError:
        return None


def parse_calls(content: str,
                tools: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Разбирает ответ модели в вызовы инструментов (пусто — разобрать не удалось).

    Принимает и русские ключи («вызовы», «сервер», «инструмент», «аргументы»).
    Вызовы с неизвестным сервером или инструментом ОТБРАСЫВАЮТСЯ: модель может
    придумать инструмент, которого нет, а «вызвать» его нельзя — иначе агент
    показал бы пользователю данные из ниоткуда.
    """
    payload = _load_object(content)
    if isinstance(payload, list):
        payload = {"calls": payload}
    if not isinstance(payload, dict):
        return []
    raw = payload.get("calls")
    if raw is None:
        raw = payload.get("вызовы")
    if raw is None:
        # Модель ответила одним вызовом без обёртки — принимаем и так.
        raw = [payload] if (payload.get("tool") or payload.get("инструмент")) else []
    allowed = None
    if tools is not None:
        allowed = {(str(item.get("server")), str(item.get("tool"))) for item in tools}
    out: List[Dict[str, Any]] = []
    for item in (raw if isinstance(raw, list) else []):
        if not isinstance(item, dict):
            continue
        server_id = str(item.get("server") or item.get("сервер") or "").strip().lower()
        tool = str(item.get("tool") or item.get("инструмент") or "").strip()
        arguments = item.get("arguments")
        if arguments is None:
            arguments = item.get("аргументы")
        if not server_id or not tool:
            continue
        if allowed is not None and (server_id, tool) not in allowed:
            logger.info("MCP: инструмент %s.%s не объявлен сервером — пропускаю",
                        server_id, tool)
            continue
        if not isinstance(arguments, dict):
            arguments = {}
        call = {"server": server_id, "tool": tool, "arguments": arguments}
        if call not in out:
            out.append(call)
        if len(out) >= MAX_CALLS_PER_REQUEST:
            break
    return out


async def choose(user_message: str, tools: List[Dict[str, Any]],
                 call: LlmCall) -> List[Dict[str, Any]]:
    """Служебный вызов: какие инструменты вызвать по запросу пользователя.

    Пустой список означает «внешние данные не нужны» ИЛИ «вызов не удался»: в
    обоих случаях агент работает как раньше — без данных MCP. Сбой не выдумывает
    вызовы (как и сбой разбора инвариантов не выдумывает нарушение).
    """
    if not tools:
        return []
    messages = [
        {"role": "system", "content": TOOLS_PROMPT},
        {"role": "user", "content": build_query(user_message, tools)},
    ]
    try:
        content, _metrics = await call(
            user_text="",
            messages=messages,
            response_format="free",
            max_tokens=TOOLS_MAX_TOKENS,
            stop=None,
            system_prompt=None,
            temperature=None,
            model=None,
            disable_thinking=True,
            timeout=TOOLS_TIMEOUT,
        )
    except Exception:  # noqa: BLE001 — сбой выбора не должен ломать запрос
        logger.warning("MCP: выбор инструментов не удался", exc_info=True)
        return []
    if not content:
        return []
    return parse_calls(content, tools)


# ---------------------------------------------------------------------------
# Блок системного промпта с полученными данными
# ---------------------------------------------------------------------------
def normalize_results(raw: Any) -> List[Dict[str, Any]]:
    """Приводит результаты вызовов к безопасному виду (для блока и интерфейса)."""
    out: List[Dict[str, Any]] = []
    for item in (raw if isinstance(raw, list) else []):
        if not isinstance(item, dict):
            continue
        server_id = str(item.get("server") or "").strip()
        tool = str(item.get("tool") or "").strip()
        if not server_id or not tool:
            continue
        entry = find_server(server_id) or {}
        out.append({
            "server": server_id,
            "server_name": str(item.get("server_name") or entry.get("name") or server_id),
            "source": str(item.get("source") or entry.get("source") or ""),
            "tool": tool,
            "arguments": item.get("arguments") if isinstance(item.get("arguments"), dict) else {},
            "ok": bool(item.get("ok")),
            "text": str(item.get("text") or "")[:RESULT_CHARS],
            "error": str(item.get("error") or "")[:400],
        })
    return out[:MAX_CALLS_PER_REQUEST]


def arguments_text(arguments: Any) -> str:
    """Аргументы вызова одной строкой (для блока и диагностики)."""
    if not isinstance(arguments, dict) or not arguments:
        return ""
    return json.dumps(arguments, ensure_ascii=False, sort_keys=True)


def block(results: Any) -> str:
    """Системный блок с данными MCP (пусто — данных нет, блока тоже нет)."""
    data = normalize_results(results)
    if not data:
        return ""
    lines: List[str] = [BLOCK_HEADER, ""]
    for index, item in enumerate(data, 1):
        args = arguments_text(item["arguments"])
        head = (f"{index}) {item['server_name']} — {item['tool']}"
                + (f"({args})" if args else ""))
        if item["source"]:
            head += f" · источник: {item['source']}"
        lines.append(head)
        if item["ok"]:
            lines.append(item["text"] or "(пустой ответ инструмента)")
        else:
            lines.append("ОШИБКА ИНСТРУМЕНТА: " + (item["error"] or "данных нет"))
        lines.append("")
    text = "\n".join(lines).strip()
    return text[:BLOCK_CHARS]


def has_data(results: Any) -> bool:
    """Есть ли данные MCP (в том числе неудачные вызовы: их тоже видит модель)."""
    return bool(normalize_results(results))


def results_note(results: Any) -> str:
    """Строка debug-чата: что именно вызвал агент и что получил."""
    data = normalize_results(results)
    if not data:
        return ""
    parts = []
    for item in data:
        args = arguments_text(item["arguments"])
        status = "данные получены" if item["ok"] else "источник отказал"
        parts.append(f"{item['server_name']}.{item['tool']}"
                     + (f"({args})" if args else "") + f" — {status}")
    body = "; ".join(parts)
    return (f"MCP: {body}. Данные уходят в модель отдельным системным блоком: "
            "план и ответ строятся по ним, выдумывать значения вместо них нельзя.")


# ---------------------------------------------------------------------------
# Сводка для интерфейса: серверы + их инструменты + состояние переключателей
# ---------------------------------------------------------------------------
def view(enabled: Optional[List[str]] = None,
         ids: Optional[List[str]] = None,
         force: bool = False) -> Dict[str, Any]:
    """Снимок каталога MCP для интерфейса (модалка «MCP»).

    Каждая запись: id, название, краткое описание, источник, признак «включён»,
    доступность (запустился ли сервер) и его инструменты. Сбой сервера не
    скрывается: пользователь должен видеть, что инструмент не подключился и
    почему, — иначе «включено, но не работает» выглядит как ошибка агента.
    """
    active = {str(item) for item in (enabled or [])}
    wanted = [str(item) for item in (ids if ids is not None else SERVER_IDS)]
    items: List[Dict[str, Any]] = []
    for entry in SERVERS:
        if entry["id"] not in wanted:
            continue
        found = discover(entry["id"], force=force)
        items.append({
            "id": entry["id"],
            "name": entry["name"],
            "description": entry["description"],
            "source": entry["source"],
            "enabled": entry["id"] in active,
            "available": bool(found.get("ok")),
            "error": str(found.get("error") or ""),
            "server_name": str(found.get("server_name") or entry["id"]),
            "server_version": str(found.get("server_version") or ""),
            "tools": [
                {"name": tool["name"], "title": tool["title"],
                 "description": tool["description"]}
                for tool in (found.get("tools") or [])
            ],
        })
    enabled_count = len([item for item in items if item["enabled"]])
    return {
        "servers": items,
        "enabled": [item["id"] for item in items if item["enabled"]],
        "counts": {
            "servers": len(items),
            "enabled": enabled_count,
            "tools": sum(len(item["tools"]) for item in items),
            "available": len([item for item in items if item["available"]]),
        },
    }


async def async_view(enabled: Optional[List[str]] = None,
                     ids: Optional[List[str]] = None,
                     force: bool = False) -> Dict[str, Any]:
    """Асинхронный снимок каталога (опрошенные серверы — в отдельном потоке)."""
    return await asyncio.to_thread(view, enabled, ids, force)
