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

Серверы проекта бывают двух видов, но говорят на одном протоколе:

    weather    — погода (7timer.info + геокодер Open-Meteo)      stdio, локальный
    currency   — курсы валют Банка России (cbr.ru)               stdio, локальный
    crypto     — курсы криптовалют (CoinGecko)                   stdio, локальный
    open_meteo — погода, прогноз, качество воздуха, координаты    http, свой сервер
                 (ensemble-api, geocoding-api, air-quality-api)  на VPS

Три первых — ЛОКАЛЬНЫЕ процессы на официальном MCP SDK
(@modelcontextprotocol/sdk, каталог mcp_servers/), общение по stdio: JSON-RPC 2.0
построчно; бесплатные, без ключей и регистрации. Четвёртый — СВОЙ сервер
`open-meteo-mcp` (тот же SDK), развёрнутый на VPS: он слушает только loopback
сервера, поэтому агент ходит к нему через SSH-туннель по Streamable HTTP
(JSON-RPC 2.0 в теле POST) и предъявляет токен доступа. Токен в реестре НЕ
хранится — в записи указано только ИМЯ переменной окружения
(`OPEN_METEO_MCP_TOKEN`), а сам секрет лежит в `.env` (в git не попадает).

Клиент сам по себе не зависит от SDK: он говорит на протоколе, поэтому к проекту
можно подключить любой MCP-сервер (в том числе сторонний) — достаточно добавить
запись в `SERVERS` (или переопределить каталог локальных серверов переменной
окружения `MCP_SERVERS_DIR`).

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
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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
# (его видит пользователь в диалоге «MCP») и способ подключения.
#
# `transport` — как клиент говорит с сервером:
#   "stdio" (по умолчанию) — ЛОКАЛЬНЫЙ процесс: `command` — либо готовая команда
#       (["node", "script.mjs"]), либо имя файла в каталоге серверов (тогда
#       запускается через node). Каталог по умолчанию — mcp_servers/ в корне
#       проекта; путь переопределяет переменная окружения MCP_SERVERS_DIR.
#   "http" — УДАЛЁННЫЙ сервер по Streamable HTTP: `url` (адрес эндпоинта, его
#       переопределяет переменная из `url_env`) и `token_env` — имя переменной
#       окружения с токеном доступа. Секрет в записи НЕ хранится.
# ---------------------------------------------------------------------------
SERVERS_DIR_ENV = "MCP_SERVERS_DIR"
NODE_ENV = "MCP_NODE_BIN"

# Транспорты клиента.
STDIO_TRANSPORT = "stdio"
HTTP_TRANSPORT = "http"

WEATHER = "weather"
CURRENCY = "currency"
CRYPTO = "crypto"
OPEN_METEO = "open_meteo"

# Свой сервер на VPS: адрес туннеля и ИМЯ переменной с токеном (не сам токен).
# Туннель (launchd на Mac) поднимает 127.0.0.1:3000 -> loopback VPS; сам сервер
# снаружи недоступен, поэтому в записи стоит адрес туннеля, а не адрес VPS.
OPEN_METEO_URL_ENV = "OPEN_METEO_MCP_URL"
OPEN_METEO_TOKEN_ENV = "OPEN_METEO_MCP_TOKEN"
OPEN_METEO_DEFAULT_URL = "http://127.0.0.1:3000/mcp"

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
    {
        "id": OPEN_METEO,
        "name": "Погода Open-Meteo (свой сервер на VPS)",
        "description": (
            "Текущая погода по названию города или координатам, прогноз по "
            "дням, качество воздуха и определение координат места. Данные "
            "отдаёт свой сервер open-meteo-mcp на VPS (через SSH-туннель)."
        ),
        "source": "Open-Meteo: ensemble-api · geocoding-api · air-quality-api "
                  "(свой сервер на VPS)",
        "transport": HTTP_TRANSPORT,
        "url": OPEN_METEO_DEFAULT_URL,
        "url_env": OPEN_METEO_URL_ENV,
        "token_env": OPEN_METEO_TOKEN_ENV,
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
    "нет — честно скажи, что их нет, и не подменяй их догадкой.\n"
    "СТРОГО: любые числа, таблицы, ряды значений, времена и даты можно брать "
    "ТОЛЬКО из этого блока — целиком, как они там написаны. Нельзя: достраивать "
    "ряд по одному значению, повторять одно значение как разные, подписывать "
    "часы или даты от себя (в том числе часовым поясом, которого в данных нет) и "
    "выдавать правдоподобный пример за измерение. Если вызов вернул ОШИБКУ или "
    "данных за нужный период не хватает — так и скажи и покажи причину; таблицу "
    "или сводку в этом случае НЕ строй.\n"
    "ДЕЙСТВИЯ ВНЕШНИХ ИНСТРУМЕНТОВ (запуск наблюдения, подписка, задание) "
    "подтверждаются ТОЛЬКО вызовом из этого блока: нет вызова или он вернул "
    "ошибку — значит действие НЕ выполнено, и утверждать обратное нельзя. "
    "Параметры работы внешнего инструмента (интервал сбора, окно, лимиты) задаёт "
    "САМ сервер — они указаны в его описании и в ответе вызова; период повтора "
    "задачи к ним отношения не имеет, поэтому не переноси его в интервал сбора."
)

# Машинная пометка «данных нет вовсе»: ставится в блок, когда НИ ОДИН вызов не
# дал данных. Правило в заголовке модель может проигнорировать (в живой задаче
# проигнорировала: отчёт отказал, а суточная таблица всё равно была нарисована),
# поэтому факт «данных нет» идёт отдельной строкой ПЕРЕД ошибками.
NO_DATA_NOTE = (
    "⚠ ДАННЫХ НЕТ: ни один вызов не дал данных — ниже причины отказов. Таблицу, "
    "ряд значений, числа и времена формировать НЕЛЬЗЯ: сообщи пользователю, что "
    "данные не получены, и назови причину отказа (это факт из блока, а не догадка)."
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
    "(«Москва»), валюту — кодом (USD, EUR, RUB). Заполняй и необязательные "
    "аргументы ПОИСКА, если они у инструмента есть: язык (`language`) и страну "
    "(`countryCode`) — по языку запроса (для русского названия `language` «ru», "
    "для российского города `countryCode` «RU»): у геокодеров язык влияет на "
    "поиск, и без него русское название может не найтись.\n"
    "5) Если в запросе не хватает обязательного для вызова сведения (например, "
    "не назван город) — НЕ вызывай инструмент: пусть агент сначала уточнит "
    "запрос у пользователя.\n"
    "6) ПЕРИОД В ЗАПРОСЕ — это НЕ интервал сбора и НЕ повод что-то запускать. "
    "Повтор запроса ведёт сам агент по своему расписанию («раз в минуту», «раз в "
    "час»), а внешний инструмент при каждом повторе просто вызывается заново. "
    "Запуск сбора на сервере (инструмент, который НАЧИНАЕТ наблюдение/подписку) "
    "нужен ТОЛЬКО когда запрос просит НАКОПЛЕННЫЕ данные за прошедший промежуток: "
    "сводку, итог, минимум/максимум/среднее, динамику, историю («сводка за "
    "сутки», «как менялась погода за неделю»). Запрос «проверяй/сообщай/следи за "
    "чем-то» — это ПОВТОРЯЮЩЕЕСЯ ЧТЕНИЕ, а не сбор: собирать ничего не нужно.\n"
    "7) ЧТО ЧИТАТЬ ПОД ЗАПРОС. Есть два разных вида данных: (а) значение СЕЙЧАС "
    "или прогноз — их отдают читающие инструменты («текущая погода», «прогноз», "
    "«координаты», «курс на дату»); (б) сводка ЗА ПРОШЕДШИЙ период — её отдаёт "
    "инструмент накопленной истории, а копит данные сбор. Выбирай по смыслу "
    "запроса: «проверяй погоду раз в минуту» → читающий инструмент текущей "
    "погоды (каждый повтор агент вызовет его снова и получит свежее значение); "
    "«сводка за сутки» → сначала сбор (если он ещё не идёт), затем отчёт по нему.\n"
    "   ЕСЛИ НУЖНА ТАБЛИЦА ИЛИ РЯД ЗНАЧЕНИЙ ПО ВРЕМЕНИ — проси ПОДРОБНЫЕ записи: "
    "у инструментов накопленной истории часто есть флаг (в схеме: include_samples, "
    "details, raw, verbose и подобные) — без него вернутся только агрегаты "
    "(минимум/максимум/среднее), и построить таблицу по часам будет НЕ из чего. "
    "Передавай такой флаг, когда запрос просит таблицу, ряд значений или «всё, что "
    "есть»; и НИКОГДА не достраивай ряд из агрегатов.\n"
    "   ИДЕНТИФИКАТОР СБОРА НЕ УГАДЫВАЙ: инструмент, которому нужен id, в одном "
    "ответе с ЗАПУСКОМ сбора вызывать бессмысленно — id станет известен только из "
    "ответа на запуск (и попадёт в блок про уже идущие сборы этой задачи). В таком "
    "запросе верни ТОЛЬКО запуск, а отчёт прочитаешь следующим запросом: "
    "придуманный id — это отказ инструмента и пустой ответ вместо данных.\n"
    "8) ЗАПУСК СБОРА — ЭТО ДЕЙСТВИЕ НА СЕРВЕРЕ: он живёт, пока его не "
    "остановят, и останавливается вместе с задачей (отмена или удаление). "
    "Поэтому: не запускай сбор «на всякий случай», при сомнении — ЧИТАЙ; не "
    "запускай повторно сбор, который уже идёт (см. блок про уже идущие сборы "
    "ниже); наличие идущего сбора НЕ отменяет чтение данных — если запросу нужны "
    "фактические значения, читающий вызов всё равно нужен. Имена и назначение "
    "инструментов бери ТОЛЬКО из списка ниже: набор у проекта МЕНЯЕТСЯ "
    "(инструменты добавляются и отключаются), ничего не придумывай по памяти. "
    "Параметры сбора (интервал, срок хранения) задаёт СЕРВЕР инструмента: не "
    "передавай интервал из текста запроса, если такого аргумента нет в схеме.\n"
    "Ответ — ТОЛЬКО JSON без пояснений:\n"
    '{"calls": [{"server": "weather", "tool": "get_weather", '
    '"arguments": {"city": "Москва"}}]}\n'
    "Если внешние данные не нужны — {\"calls\": []}."
)

# Что говорим диспетчеру ПОВТОРНО: первый ответ состоял из одних ЗАПУСКОВ сбора,
# а данных для ответа не принёс. Так запрос «проверяй погоду раз в минуту» не
# превращается в «зарегистрировать наблюдение»: агенту нужны ФАКТЫ.
NEED_READS_NOTE = (
    "Твои вызовы только ЗАПУСКАЮТ работу на сервере и НЕ читают данные — "
    "ответить по ним нельзя. Верни вызов ЧИТАЮЩЕГО инструмента, который отдаёт "
    "нужные запросу фактические данные ПРЯМО СЕЙЧАС (текущее значение, прогноз, "
    "координаты, курс, сводку по уже идущему сбору). Запускать новый сбор не "
    "нужно, если только запрос прямо не просит копить данные за период."
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


def transport_of(entry: Dict[str, Any]) -> str:
    """Транспорт записи реестра ("stdio" по умолчанию, "http" — удалённый сервер)."""
    value = str(entry.get("transport") or STDIO_TRANSPORT).strip().lower()
    return HTTP_TRANSPORT if value == HTTP_TRANSPORT else STDIO_TRANSPORT


def server_url(entry: Dict[str, Any]) -> str:
    """Адрес удалённого сервера: переменная окружения важнее значения в записи."""
    name = str(entry.get("url_env") or "").strip()
    value = os.getenv(name) if name else None
    return str(value or entry.get("url") or "").strip()


def server_token(entry: Dict[str, Any]) -> str:
    """Токен доступа удалённого сервера (пусто — переменная не задана).\n"

    Значение читается ТОЛЬКО из окружения: в реестре лежит имя переменной,
    поэтому секрет не попадает ни в код, ни в интерфейс, ни в диагностику.
    """
    name = str(entry.get("token_env") or "").strip()
    return str(os.getenv(name) or "").strip() if name else ""


def availability_error(entry: Dict[str, Any]) -> str:
    """Почему сервер заведомо не подключится (пусто — предпосылок к сбою нет).\n"

    Проверяем то, что видно без обращения к серверу. Для локальных серверов это
    наличие запускаемой программы, файла сервера и установленного MCP SDK (иначе
    пользователь в диалоге «MCP» видел бы молчаливое «недоступен» вместо причины
    «Node.js не найден», «не выполнен npm install»). Для удалённого — заданы ли
    адрес и токен доступа: без токена сервер ответит 401, и об этом лучше сказать
    заранее и словами, а не кодом ошибки.

    Сервер с собственной командой (`command` в записи реестра) проверяется только
    на существование программы и файла: SDK нужен лишь серверам на Node.
    """
    if transport_of(entry) == HTTP_TRANSPORT:
        if not server_url(entry):
            name = str(entry.get("url_env") or "").strip()
            return ("не задан адрес сервера MCP"
                    + (": переменная окружения " + name if name else ""))
        token_env = str(entry.get("token_env") or "").strip()
        if token_env and not server_token(entry):
            return "не задан токен доступа: переменная окружения " + token_env
        return ""
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
# Клиент MCP по stdio: JSON-RPC 2.0 построчно (локальный сервер)
# ---------------------------------------------------------------------------
class _StdioSession:
    """Одно соединение с MCP-сервером: запуск процесса, запросы, закрытие.\n"

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


# ---------------------------------------------------------------------------
# Клиент MCP по Streamable HTTP: JSON-RPC 2.0 в теле POST (удалённый сервер)
# ---------------------------------------------------------------------------
class _HttpSession:
    """Соединение с УДАЛЁННЫМ MCP-сервером по Streamable HTTP.\n"

    Каждый запрос — отдельный HTTP POST на адрес сервера: так работает
    Streamable HTTP из спецификации MCP. Сервер без состояния соединение не
    держит, а SSH-туннель живёт отдельно от агента, поэтому закрывать нечего.
    Ответ приходит либо телом JSON, либо потоком SSE (`text/event-stream`):
    это один транспорт, и вид ответа выбирает сервер — понимаем оба.

    Токен доступа уходит ТОЛЬКО заголовком `Authorization` и никогда не попадает
    в текст ошибки: этот текст видит пользователь в диалоге «MCP».
    """

    def __init__(self, url: str, token: str = "", token_env: str = "") -> None:
        self.url = url
        self._token = token
        self._token_env = token_env
        self._send_lock = threading.Lock()
        self._next_id = 0
        self._session_id = ""
        self._closed = False
        self.started = time.monotonic()
        # Имя и версия сервера из рукопожатия (initialize) — как у stdio-сессии.
        self.server_name = ""
        self.server_version = ""

    # -- HTTP ---------------------------------------------------------------
    def _headers(self) -> Dict[str, str]:
        """Заголовки запроса: протокол, авторизация и идентификатор сессии."""
        headers = {
            "content-type": "application/json",
            # Оба типа ответа перечислены явно: сервер вправе ответить потоком,
            # и без согласия на `text/event-stream` он вправе отказать.
            "accept": "application/json, text/event-stream",
            "user-agent": CLIENT_NAME + "/" + CLIENT_VERSION,
        }
        if self._token:
            headers["authorization"] = "Bearer " + self._token
        if self._session_id:
            # Сервер с состоянием выдаёт идентификатор сессии в ответе на
            # initialize и ждёт его обратно; сервер без состояния его не шлёт,
            # и тогда заголовка просто нет.
            headers["mcp-session-id"] = self._session_id
        return headers

    def _post(self, payload: Dict[str, Any],
              timeout: float) -> Optional[Dict[str, Any]]:
        """Один POST: сообщение JSON-RPC -> ответ (None — ответа нет)."""
        body = json.dumps(payload).encode("utf-8")
        limit = max(1.0, float(timeout))
        try:
            request = urllib.request.Request(self.url, data=body,
                                             headers=self._headers(),
                                             method="POST")
            with urllib.request.urlopen(request, timeout=limit) as response:
                kind = str(response.headers.get("Content-Type") or "")
                session_id = response.headers.get("Mcp-Session-Id")
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raise McpError(self._http_error(exc))
        except UnicodeEncodeError:
            # Заголовки HTTP — только ASCII: не-ASCII токен (например, скопированный
            # с лишними символами) иначе падал бы ошибкой кодека.
            raise McpError("токен доступа содержит символы, недопустимые в "
                           "HTTP-заголовке (нужны только ASCII-символы)")
        except socket.timeout:
            raise McpError(f"сервер не ответил за {limit:.0f} с: {self.url}")
        except (urllib.error.URLError, OSError) as exc:
            # Сюда попадают и «туннель не поднят», и «сервер остановлен»:
            # пользователю нужна причина и подсказка, а не трассировка.
            reason = getattr(exc, "reason", None) or exc
            raise McpError("сервер недоступен: " + str(reason) + " (" + self.url
                           + " — проверьте, что MCP-сервер и SSH-туннель запущены)")
        if session_id:
            self._session_id = str(session_id).strip()
        return self._parse(raw, kind)

    def _http_error(self, exc: Any) -> str:
        """Текст ошибки HTTP: код, подсказка и короткая выдержка из тела."""
        detail = ""
        try:
            detail = exc.read()[:300].decode("utf-8", "replace").strip()
        except Exception:  # noqa: BLE001 — тело нужно лишь для пояснения
            detail = ""
        if exc.code in (401, 403):
            hint = "проверьте токен доступа"
            if self._token_env:
                hint += ": переменная окружения " + self._token_env
        elif exc.code == 404:
            hint = "проверьте адрес сервера: " + self.url
        else:
            hint = self.url
        parts = [f"сервер ответил ошибкой HTTP {exc.code}", hint]
        if detail:
            parts.append(detail.replace("\n", " ")[:200])
        return "; ".join(part for part in parts if part)

    def _parse(self, raw: bytes, kind: str) -> Optional[Dict[str, Any]]:
        """Ответ сервера: тело JSON или поток SSE -> сообщение JSON-RPC."""
        if "text/event-stream" in kind.lower():
            return self._from_stream(raw)
        text = raw.decode("utf-8", "replace").strip()
        if not text:
            return None
        try:
            message = json.loads(text)
        except ValueError:
            raise McpError("сервер вернул не JSON: " + text[:200])
        return message if isinstance(message, dict) else None

    @staticmethod
    def _from_stream(raw: bytes) -> Optional[Dict[str, Any]]:
        """Сообщение JSON-RPC из потока SSE (строки «data: {...}»).

        В потоке могут быть и уведомления сервера, поэтому берём ПОСЛЕДНЕЕ
        сообщение с ответом (`result` или `error`) — оно и есть ответ на запрос.
        """
        found: Optional[Dict[str, Any]] = None
        for line in raw.decode("utf-8", "replace").splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            chunk = line[len("data:"):].strip()
            if not chunk or chunk == "[DONE]":
                continue
            try:
                message = json.loads(chunk)
            except ValueError:
                continue
            if isinstance(message, dict) and ("result" in message
                                              or "error" in message):
                found = message
        return found

    # -- протокол ----------------------------------------------------------
    def notify(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        """Уведомление серверу (ответа не ждём).

        Сбой уведомления НЕ считается сбоем соединения: рукопожатие уже прошло,
        а уведомление нужно не всем серверам — сервер без состояния соединение
        не хранит и часто отвечает на него пустым 202.
        """
        payload: Dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        try:
            self._post(payload, INIT_TIMEOUT)
        except McpError as exc:
            logger.info("MCP %s: уведомление %s не принято: %s",
                        self.url, method, exc)

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
            message = self._post(payload, timeout)
        if not isinstance(message, dict):
            raise McpError(f"сервер не ответил на {method} (пустой ответ)")
        if message.get("error"):
            error = message.get("error")
            text = error.get("message") if isinstance(error, dict) else str(error)
            raise McpError(f"{method}: сервер вернул ошибку: {text}")
        result = message.get("result")
        return result if isinstance(result, dict) else {}

    def close(self) -> None:
        """Закрывать нечего: соединение живёт ровно один HTTP-запрос."""
        self._closed = True


def _open_session(entry: Dict[str, Any]) -> Union[_StdioSession, _HttpSession]:
    """Открывает соединение с сервером и выполняет рукопожатие MCP (initialize)."""
    problem = availability_error(entry)
    if problem:
        raise McpError(problem)
    session: Union[_StdioSession, _HttpSession]
    if transport_of(entry) == HTTP_TRANSPORT:
        session = _HttpSession(server_url(entry), server_token(entry),
                               str(entry.get("token_env") or "").strip())
    else:
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
    """Список инструментов сервера (с кэшем): {"ok", "tools", "error", ...}.\n"

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
    """Вызывает инструмент сервера: {"ok", "text", "error"}.\n"

    Каждый вызов — отдельное соединение с сервером (процесс для локального,
    HTTP-запрос для удалённого): соединение не переиспользуется, поэтому
    «залипший» сервер не портит следующие запросы. Ошибку источника инструмент
    возвращает сам (isError), и её текст уходит модели как есть.
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
    """Выполняет вызовы инструментов и возвращает результаты (последовательно).\n"

    Последовательно — намеренно: вызовы запускают локальные процессы и ходят во
    внешние источники, а параллельный запуск только добавил бы нагрузку; за
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
# Асинхронные обёртки: запуск процессов и HTTP-запросы не должны блокировать
# цикл событий
# ---------------------------------------------------------------------------
async def async_discover(ids: Optional[List[str]] = None,
                         force: bool = False) -> List[Dict[str, Any]]:
    """Асинхронный список инструментов серверов (серверы — в отдельном потоке)."""
    return await asyncio.to_thread(discover_many, ids, force)


async def async_run_calls(calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Асинхронное выполнение вызовов (серверы — в отдельном потоке)."""
    return await asyncio.to_thread(run_calls, calls)


# ---------------------------------------------------------------------------
# Внешние обязательства: что задача ЗАПУСТИЛА на стороне сервера
# ---------------------------------------------------------------------------
# Инструмент может не только прочитать данные, но и ЗАПУСТИТЬ что-то на сервере:
# сбор наблюдений, подписку, задание. Такое нельзя бросить — иначе на сервере
# останется висеть вечный сбор, о котором задача уже забыла (её отменили или
# удалили). Код НЕ знает имён инструментов: пара «запускающий ↔ отменяющий»
# определяется по ИМЕНИ (общий префикс start_/stop_) и проверяется по списку,
# который объявил сам сервер (`tools/list`). Аргументы отмены (id наблюдения)
# достаются из ТЕКСТА ответа запускающего инструмента — по именам обязательных
# полей схемы отменяющего.
START_PREFIX = "start_"
STOP_PREFIX = "stop_"
# Сколько обязательств помним на задачу (страховка от разрастания диалога).
MAX_STARTED = 20
# Ключи, по которым ищем значение в тексте ответа («id: "abc"», `"id": "abc"`,
# «id=abc», «id abc»). Регистр не важен, кавычки любые.
_VALUE_PATTERNS = (
    r'["\']?{key}["\']?\s*[:=]\s*["\']([^"\'\s,;)]+)["\']',
    r'["\']?{key}["\']?\s*[:=]\s*([^\s,;)]+)',
)


def undo_tool(tool: Any, tool_names: Any) -> str:
    """Имя ОТМЕНЯЮЩЕГО инструмента для запускающего ("" — пары нет).\n"

    Соглашение об именах + проверка по объявленному списку: `start_weather_watch`
    → `stop_weather_watch`. Если сервер такого инструмента не объявлял, отменять
    нечем — и выдумывать вызов нельзя.
    """
    name = str(tool or "").strip()
    if not name.startswith(START_PREFIX):
        return ""
    candidate = STOP_PREFIX + name[len(START_PREFIX):]
    names = {str(item).strip() for item in (tool_names or [])}
    return candidate if candidate in names else ""


def _schema_of(tools: List[Dict[str, Any]], server_id: str, tool: str) -> Dict[str, Any]:
    """Схема аргументов инструмента из объявленного сервером списка."""
    for item in (tools or []):
        if str(item.get("server") or "") != server_id:
            continue
        if str(item.get("tool") or "") != tool:
            continue
        schema = item.get("schema")
        return schema if isinstance(schema, dict) else {}
    return {}


def find_value(text: Any, key: str) -> str:
    """Значение поля `key` из текста ответа инструмента ("" — не нашлось).\n"

    Ответы инструментов — обычный текст (часто с JSON внутри), поэтому значение
    ищем по образцам «ключ: значение», «"ключ": "значение"», «ключ=значение».

    Поле-идентификатор у сервера может называться иначе, чем в схеме отменяющего
    инструмента: например, инструмент отмены требует `collection_id`, а ответ
    называет его просто `id`. Поэтому пробуем не только точное имя, но и его
    «хвосты» (последнее слово, `id`) — от точного к самому общему; первое
    найденное значение побеждает.
    """
    body = str(text or "")
    if not body or not str(key or "").strip():
        return ""
    for candidate in _key_candidates(key):
        for pattern in _VALUE_PATTERNS:
            match = re.search(pattern.format(key=re.escape(candidate)), body,
                              re.IGNORECASE)
            if match:
                value = match.group(1).strip().strip('"\'')
                if value:
                    return value[:200]
    return ""


def _key_candidates(key: Any) -> List[str]:
    """Имена, под которыми значение поля может стоять в тексте ответа.\n"

    «collection_id» → ["collection_id", "id", "collection"]: точное имя первым,
    дальше — «хвосты», которые сервер мог использовать вместо него.
    """
    name = str(key or "").strip()
    if not name:
        return []
    candidates = [name]
    parts = [part for part in re.split(r"[_\-\s]+", name) if part]
    if len(parts) > 1:
        # Последнее слово обычно и есть идентификатор («…_id»).
        if parts[-1].lower() != name.lower():
            candidates.append(parts[-1])
        if parts[0].lower() != name.lower():
            candidates.append(parts[0])
    if name.lower().endswith("id") and name.lower() != "id":
        candidates.append("id")
    out: List[str] = []
    for item in candidates:
        if len(item) >= 2 and item not in out:
            out.append(item)
    return out


def stop_arguments(result: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """Аргументы отмены из ответа запускающего инструмента.\n"

    Берём ОБЯЗАТЕЛЬНЫЕ строковые поля схемы отменяющего инструмента (у
    наблюдения это «id») и ищем их значения в тексте ответа. Хотя бы одно
    обязательное поле не нашлось — отменять нечем: пустой словарь.
    """
    required = [str(key) for key in ((schema or {}).get("required") or [])]
    properties = (schema or {}).get("properties") or {}
    if not required:
        # Схема без обязательных полей: отменяющий инструмент самодостаточен
        # (например, «остановить всё») — вызываем его без аргументов.
        return {}
    found: Dict[str, Any] = {}
    for key in required:
        kind = (properties.get(key) or {}).get("type")
        if kind not in (None, "string"):
            return {}
        value = find_value(result.get("text"), key)
        if not value:
            return {}
        found[key] = value
    return found


def started_calls(results: Any, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Что задача ЗАПУСТИЛА на серверах: список обязательств для отмены.\n"

    Смотрим результаты ВЫПОЛНЕННЫХ вызовов: если у вызванного инструмента есть
    отменяющая пара в объявленном списке и из ответа достаются её аргументы,
    обязательство запоминается — потом его отменит отмена/удаление задачи
    (см. cancel_started). Имена инструментов берутся из данных сервера.
    """
    out: List[Dict[str, Any]] = []
    for item in normalize_results(results):
        if not item.get("ok"):
            continue
        server_id = item.get("server") or ""
        tool = item.get("tool") or ""
        names = [entry.get("tool") for entry in (tools or [])
                 if str(entry.get("server") or "") == server_id]
        stop = undo_tool(tool, names)
        if not stop:
            continue
        schema = _schema_of(tools or [], server_id, stop)
        required = [str(key) for key in ((schema or {}).get("required") or [])]
        arguments = stop_arguments(item, schema)
        if required and not arguments:
            # Аргументы отмены (id наблюдения) в ответе не нашлись: запоминать
            # нечего — «отмена» вызвала бы инструмент без нужных аргументов и
            # сбор остался бы висеть, а пользователь думал бы, что он остановлен.
            logger.info(
                "MCP: %s/%s запустил внешний сбор, но аргументы отмены (%s) в "
                "ответе не найдены — задача о сборе не помнит",
                server_id, tool, ", ".join(required))
            continue
        entry = {
            "server": server_id,
            "server_name": item.get("server_name") or server_id,
            "tool": tool,
            "stop_tool": stop,
            "arguments": arguments,
        }
        if entry not in out:
            out.append(entry)
    return out[:MAX_STARTED]


def normalize_calls(raw: Any) -> List[Dict[str, Any]]:
    """Приводит список вызовов к безопасному виду: [{"server","tool","arguments"}]."""
    out: List[Dict[str, Any]] = []
    for item in (raw if isinstance(raw, list) else []):
        if not isinstance(item, dict):
            continue
        server_id = str(item.get("server") or "").strip()[:80]
        tool = str(item.get("tool") or "").strip()[:120]
        if not server_id or not tool:
            continue
        arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
        call: Dict[str, Any] = {
            "server": server_id, "tool": tool,
            "arguments": {str(key)[:60]: value
                          for key, value in list(arguments.items())[:10]},
        }
        # Признак «вызов что-то ЗАПУСКАЕТ на сервере» ставится при выборе (тогда
        # известен список инструментов) и хранится вместе с вызовом: повтору он
        # нужен, чтобы НЕ повторять запуск сбора и не ходить за списком сервера.
        if item.get("action") is True:
            call["action"] = True
        if call not in out:
            out.append(call)
    return out[:MAX_CALLS_PER_REQUEST]


def is_action_tool(tool: Any, tools: List[Dict[str, Any]],
                   server_id: str = "") -> bool:
    """True — инструмент ЗАПУСКАЕТ/ОСТАНАВЛИВАЕТ работу на сервере, а не читает.\n"

    Действием считается вызов, у которого есть отменяющая пара в объявленном
    списке (см. undo_tool): регистрация наблюдения, подписка, задание. Такие
    вызовы имеют побочный эффект на сервере, поэтому к ним относимся строже:
    чтение данных безопасно, а брошенный сбор придётся останавливать.
    """
    names = [entry.get("tool") for entry in (tools or [])
             if not server_id or str(entry.get("server") or "") == server_id]
    return bool(undo_tool(tool, names))


def read_calls(calls: Any, tools: Optional[List[Dict[str, Any]]] = None
               ) -> List[Dict[str, Any]]:
    """Только ЧИТАЮЩИЕ вызовы (без запуска/остановки работы на сервере).

    Признак берётся из самого вызова (`action`, ставится при выборе). Список
    инструментов нужен только для старых записей без признака — тогда действие
    определяется по паре start_/stop_ в объявленном списке.
    """
    out: List[Dict[str, Any]] = []
    for call in normalize_calls(calls):
        if call.get("action") is True:
            continue
        if call.get("action") is None and tools is not None \
                and is_action_tool(call.get("tool"), tools,
                                   str(call.get("server") or "")):
            continue
        out.append(call)
    return out


def has_reads(calls: Any, tools: Optional[List[Dict[str, Any]]] = None) -> bool:
    """Есть ли среди вызовов хотя бы один читающий (иначе данных не будет)."""
    return bool(read_calls(calls, tools))


# Просьба о НАКОПЛЕННЫХ данных за промежуток. Нужна ТОЛЬКО как предохранитель
# (см. looks_aggregate): по ней решается, можно ли переспрашивать диспетчера,
# когда он ответил одними запусками сбора. Инструменты по этим словам НЕ
# выбираются — выбор всегда за моделью.
AGGREGATE_MARKERS = (
    "сводк", "итог", "суммарн", "за сутки", "за день", "за недел", "за месяц",
    "за период", "за последн", "за прошл", "минимум", "максимум", "средн",
    "динамик", "тренд", "истори", "накопл", "как менял", "статистик", "за сутки",
)


def looks_aggregate(text: Any) -> bool:
    """Похож ли запрос на просьбу о СВОДКЕ за прошедший промежуток.

    Предохранитель: запрос «сводка за сутки» требует НАКОПЛЕННЫХ данных, поэтому
    вызовы, которые только запускают сбор, для него — правильный ответ, и
    переспрашивать диспетчера не нужно (см. _preflight_mcp). Запрос «проверяй
    погоду раз в минуту» — наоборот, про текущее значение: если диспетчер ответил
    одним запуском сбора, ответ нужно уточнить.
    """
    body = " ".join(str(text or "").lower().split())
    return any(marker in body for marker in AGGREGATE_MARKERS)


def mark_call_kinds(calls: Any, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Помечает вызовы признаком «действие на сервере» (запуск/остановка).

    Вызывается сразу после выбора инструментов: тогда список объявленных
    инструментов под рукой, и повтору достаточно посмотреть на сам вызов.
    """
    for call in (calls if isinstance(calls, list) else []):
        if not isinstance(call, dict):
            continue
        if is_action_tool(call.get("tool"), tools, str(call.get("server") or "")):
            call["action"] = True
        else:
            call.pop("action", None)
    return calls if isinstance(calls, list) else []


def mark_actions(results: Any, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Помечает в результатах ДЕЙСТВИЯ и возвращает удачные сборы для отмены.\n"

    Действие — вызов инструмента, у которого есть отменяющая пара в объявленном
    списке сервера (см. undo_tool): регистрация наблюдения, подписка и т. п.
    Помеченные результаты уходят модели с явной строкой «ДЕЙСТВИЯ В ЭТОМ
    ЗАПРОСЕ» (см. block), поэтому выдумать несостоявшуюся регистрацию нельзя,
    а неудачное действие считается НЕ выполненным.

    Возвращает то же, что started_calls: список обязательств для последующей
    отмены (только успешно запущенные).
    """
    marked: List[Dict[str, Any]] = []
    # Помечаем ИМЕННО те словари, что лежат в dialog["mcp"] (их читает block):
    # копия здесь была бы бесполезна — модель видела бы старые результаты.
    for item in (results if isinstance(results, list) else []):
        if not isinstance(item, dict):
            continue
        server_id = str(item.get("server") or "").strip()
        tool = str(item.get("tool") or "").strip()
        names = [entry.get("tool") for entry in (tools or [])
                 if str(entry.get("server") or "") == server_id]
        if not tool or not undo_tool(tool, names):
            continue
        item["action"] = "started" if item.get("ok") else "failed"
        marked.append(item)
    return marked


def cancel_started(entries: Any) -> List[Dict[str, Any]]:
    """ОТМЕНЯЕТ внешние сборы задачи: вызывает отменяющие инструменты.\n"

    Возвращает отчёты: [{"server", "tool", "ok", "text", "error"}]. Сбои не
    скрываются: их видит пользователь (в чате задачи), потому что незакрытый
    сбор на сервере — это работающий впустую источник и место на диске.
    """
    reports: List[Dict[str, Any]] = []
    for item in (entries if isinstance(entries, list) else []):
        if not isinstance(item, dict):
            continue
        server_id = str(item.get("server") or "").strip()
        tool = str(item.get("stop_tool") or "").strip()
        if not server_id or not tool:
            continue
        arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
        try:
            result = call_tool(server_id, tool, arguments)
        except Exception as exc:  # noqa: BLE001 — сбой отмены не должен ломать удаление
            logger.warning("MCP: отмена %s/%s не удалась", server_id, tool, exc_info=True)
            reports.append({"server": server_id, "tool": tool, "ok": False,
                            "text": "", "error": str(exc)[:300]})
            continue
        reports.append({
            "server": server_id, "tool": tool,
            "ok": bool(result.get("ok")),
            "text": str(result.get("text") or "")[:400],
            "error": str(result.get("error") or "")[:300],
        })
    return reports


async def async_cancel_started(entries: Any) -> List[Dict[str, Any]]:
    """То же в отдельном потоке: отмена — сетевые вызовы, цикл событий не держим."""
    if not entries:
        return []
    return await asyncio.to_thread(cancel_started, entries)


def cancel_note(reports: List[Dict[str, Any]]) -> str:
    """Строка для чата: что удалось (и не удалось) отменить на серверах."""
    if not reports:
        return ""
    done = [item for item in reports if item.get("ok")]
    failed = [item for item in reports if not item.get("ok")]
    parts: List[str] = []
    if done:
        parts.append("остановлены внешние сборы: " + "; ".join(
            f"{item['server']} · {item['tool']} "
            f"({(item.get('text') or '').splitlines()[0][:80]})" for item in done))
    for item in failed:
        parts.append(f"НЕ удалось остановить {item['server']} · {item['tool']}"
                     + (f": {item['error']}" if item.get("error") else ""))
    return "; ".join(parts)


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


def build_query(user_message: str, tools: List[Dict[str, Any]],
                started: Optional[List[Dict[str, Any]]] = None,
                extra: str = "") -> str:
    """Текст запроса к модели: запрос пользователя + доступные инструменты.\n"

    `started` — ВНЕШНИЕ СБОРЫ, которые эта задача уже начала (наблюдения,
    подписки: см. started_calls). Они идут у сервера прямо сейчас, и начинать их
    повторно нельзя: модель получает их отдельным блоком вместе с аргументами
    отмены (в них — id, по которому читается накопленная сводка). Блок строится
    из ДАННЫХ сервера, а не из имён в коде: имена инструментов код не знает.
    """
    text = (
        "ЗАПРОС ПОЛЬЗОВАТЕЛЯ:\n" + (str(user_message or "").strip() or "(пусто)")
    )
    if extra:
        # Дополнительное условие (например, «нужны ФАКТИЧЕСКИЕ данные, а не запуск
        # сбора»): так диспетчер спрашивается повторно, если первый ответ состоял
        # из одних действий и данных для ответа не дал.
        text += "\n\nВАЖНОЕ УСЛОВИЕ:\n" + str(extra).strip()
    if started:
        lines = []
        for item in started:
            args = item.get("arguments") or {}
            args_text = ", ".join(f"{key}={value}" for key, value in args.items())
            lines.append(
                f"- сервер: {item.get('server')}; сбор УЖЕ ИДЁТ (запущен инструментом "
                f"{item.get('tool')}); его идентификатор и параметры: "
                f"{args_text or '(без параметров)'} — сводку по нему читай "
                "инструментом отчёта (тем, который принимает эти параметры), "
                "а сбор заново НЕ начинай"
            )
        text += (
            "\n\nУЖЕ ИДУЩИЕ ВНЕШНИЕ СБОРЫ ЭТОЙ ЗАДАЧИ (их НЕ надо начинать заново; "
            "накопленные данные читай по их параметрам — например, сводку за "
            "прошедший период):\n" + "\n".join(lines)
        )
    return text + "\n\nДОСТУПНЫЕ ИНСТРУМЕНТЫ:\n" + tools_text(tools)


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
    """Разбирает ответ модели в вызовы инструментов (пусто — разобрать не удалось).\n"

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
                 call: LlmCall,
                 started: Optional[List[Dict[str, Any]]] = None,
                 extra: str = "") -> List[Dict[str, Any]]:
    """Служебный вызов: какие инструменты вызвать по запросу пользователя.\n"

    Пустой список означает «внешние данные не нужны» ИЛИ «вызов не удался»: в
    обоих случаях агент работает как раньше — без данных MCP. Сбой не выдумывает
    вызовы (как и сбой разбора инвариантов не выдумывает нарушение).

    `started` — уже идущие внешние сборы задачи (см. build_query): модель видит
    их и не начинает заново.
    """
    if not tools:
        return []
    messages = [
        {"role": "system", "content": TOOLS_PROMPT},
        {"role": "user", "content": build_query(user_message, tools, started, extra)},
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
            # ДЕЙСТВИЕ на сервере (запуск/остановка сбора, подписки): ставит
            # mark_actions. Нужно, чтобы модель не выдавала желаемое за сделанное.
            "action": str(item.get("action") or "")[:20],
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
    # ДЕЙСТВИЯ этого запроса — отдельной строкой: без неё модель «регистрирует»
    # наблюдения словами, когда никакого вызова не было (и пользователь получает
    # «✅ всё зарегистрировано» при пустом списке наблюдений).
    started = [item for item in data if item.get("action") == "started"]
    failed = [item for item in data if item.get("action") == "failed"]
    summary = []
    for item in started:
        summary.append(f"ВЫПОЛНЕНО: {item['tool']} "
                       f"({(item['text'] or '').splitlines()[0][:80]})")
    for item in failed:
        summary.append(f"НЕ ВЫПОЛНЕНО (ошибка): {item['tool']} "
                       f"({(item['error'] or 'без пояснения')[:80]})")
    if not any(item.get("ok") for item in data):
        lines.append(NO_DATA_NOTE)
        lines.append("")
    lines.append("ДЕЙСТВИЯ НА СЕРВЕРАХ В ЭТОМ ЗАПРОСЕ: "
                 + ("; ".join(summary) if summary
                    else "не выполнялись (были только запросы данных) — значит, "
                         "ничего не запускалось и не регистрировалось"))
    lines.append("")
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
    """Снимок каталога MCP для интерфейса (модалка «MCP»).\n"

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
