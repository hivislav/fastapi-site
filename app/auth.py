"""Вход по паролю: гейт доступа из внешней сети.

ЗАЧЕМ ЭТОТ МОДУЛЬ. Приложение стоит на маке и открыто наружу пробросом порта
на роутере (внешний адрес → адрес мака в сети дома, см. tools/serve.sh). Своей
защиты у приложения нет и не было: пятьдесят восемь маршрутов API отдают
диалоги и документы, читают и пишут рабочую историю (data/*.json), зовут
внешние инструменты (MCP, в том числе свои серверы на VPS) и модель за деньги
владельца. Поэтому вход по паролю — часть САМОГО приложения: правило роутера
защитой не является (его снимают одной галочкой, а порт сканируют за минуты).

ЧТО ЗДЕСЬ ЕСТЬ:

* `AuthGate` — ASGI-middleware перед всем приложением: пропускает внутрь только
  того, кто вошёл, и НЕ отдаёт наружу ничего, пока пароль не предъявлен;
* вход (`/api/auth/login`) сверяет логин и пароль из .env через
  `hmac.compare_digest` — сравнение не зависит от того, на каком символе
  строки разошлись, и не выдаёт пароль по времени ответа;
* доказательство входа — ПОДПИСАННАЯ кука: HMAC-SHA256 от логина, срока и
  случайного числа. Секрет берётся из `ACCESS_SECRET`, а если он не задан — из
  файла `data/auth_secret` (создаётся сам, права 600). Так подделать куку
  нельзя, а вход переживает перезапуск приложения;
* защита от перебора: после `ACCESS_MAX_FAILS` неудачных попыток с одного
  адреса за `ACCESS_FAIL_WINDOW` секунд адрес получает 429 с `Retry-After`;
* ЛОКАЛЬНЫЙ клиент (127.0.0.1) по умолчанию пропускается без пароля
  (`ACCESS_LOCAL_BYPASS`) — этого требуют живые проверки проекта, которые
  ходят по HTTP на 127.0.0.1, и работа владельца за своим маком;
* БЕЗ ПАРОЛЯ НАРУЖУ НЕ ПУСКАЕМ ВООБЩЕ: если пароль не задан, внешний клиент
  получает 403 с причиной. Гейт «выключен — значит пускаем всех» на публичном
  адресе означал бы, что один забытый пункт .env открывает всё.

ПОЧЕМУ MIDDLEWARE, А НЕ ЗАВИСИМОСТЬ В КАЖДОМ МАРШРУТЕ. Маршрутов пятьдесят
восемь, и любой новый по умолчанию оказался бы незащищённым — «забыли
поставить зависимость» не должно открывать доступ. Гейт стоит ОДИН и перед
всем приложением, включая /docs, /openapi.json и /health; исключение
перечислено явно (`OPEN_PATHS`) — это только страница входа и сами маршруты
входа/выхода.

ПОЧЕМУ ЧИСТЫЙ ASGI, А НЕ BaseHTTPMiddleware. Ответы агента — потоки (NDJSON):
`BaseHTTPMiddleware` оборачивает ответ в свою задачу и буферизует его, из-за
чего события перестали бы приходить по мере появления (ровно этот дефект уже
чинили в main.py для gzip). Здесь сообщения проходят насквозь как есть.
"""

import base64
import hmac
import os
import secrets
import time
from hashlib import sha256
from typing import Any, Dict, List, Optional, Tuple

from starlette.responses import JSONResponse, RedirectResponse

from app import config

# Имя куки. Не «session»: в приложении уже есть сессии задач (s-…), и одно
# слово на два разных понятия путало бы и в браузере, и в коде.
COOKIE_NAME = "site_access"

# Пути, доступные БЕЗ входа. Всё остальное закрыто гейтом, включая
# /docs, /openapi.json и /health: документация API для внешнего мира — это
# карта всех возможностей приложения, а health-маршрут сканеру сообщает, что
# здесь что-то живое.
LOGIN_PATH = "/login"
LOGIN_API = "/api/auth/login"
LOGOUT_API = "/api/auth/logout"
STATE_API = "/api/auth/state"
# `/api/auth/state` открыт намеренно: страница входа обязана узнать ДО входа,
# спрашивают ли пароль вообще (иначе «вход не настроен» выглядел бы как
# «неверный пароль»), а интерфейс чата — нужна ли ему кнопка «Выйти». Секретов
# ответ не содержит: только «гейт включён/выключен», «клиент локальный» и
# «кука верна» — и всё это про самого спрашивающего.
OPEN_PATHS = frozenset({LOGIN_PATH, LOGIN_API, LOGOUT_API, STATE_API, "/favicon.ico"})

# Заголовок ответа, объясняющий отказ (его видно и в браузере, и в curl).
DENY_HEADER = "X-Access-Denied"

_SECRET_CACHE: Optional[bytes] = None
# Неудачные попытки входа: адрес → список отметок времени (monotonic).
_FAILS: Dict[str, List[float]] = {}


def reset_cache() -> None:
    """Сбросить кэш секрета (нужен проверкам, которые меняют .env)."""
    global _SECRET_CACHE
    _SECRET_CACHE = None


# ---------------------------------------------------------------------------
# Настройки (значения — в app/config.py, здесь только чтение)
# ---------------------------------------------------------------------------

def enabled() -> bool:
    """Включён ли гейт (см. config.access_enabled)."""
    return config.access_enabled()


def local_bypass() -> bool:
    """Пропускать ли локального клиента без пароля."""
    return bool(config.ACCESS_LOCAL_BYPASS)


def ttl_seconds() -> int:
    """Срок жизни входа в секундах."""
    return int(config.ACCESS_TTL_HOURS) * 3600


def secret() -> bytes:
    """Секрет подписи куки: из .env или из файла (файл создаётся при первом входе).

    Файл, а не «случайный секрет в памяти», потому что иначе любой перезапуск
    приложения (а его перезапускают и правкой .env, и перезагрузкой мака)
    выкидывал бы владельца из его же сайта.
    """
    global _SECRET_CACHE
    if _SECRET_CACHE is not None:
        return _SECRET_CACHE
    value = (config.ACCESS_SECRET or "").strip()
    if value:
        _SECRET_CACHE = value.encode("utf-8")
        return _SECRET_CACHE
    path = config.ACCESS_SECRET_FILE
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as fh:
            stored = fh.read().strip()
        if stored:
            _SECRET_CACHE = stored.encode("utf-8")
            return _SECRET_CACHE
    fresh = secrets.token_hex(32)  # 64 знака — 256 бит энтропии
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(fresh + "\n")
    os.chmod(path, 0o600)
    _SECRET_CACHE = fresh.encode("utf-8")
    return _SECRET_CACHE


# ---------------------------------------------------------------------------
# Кука: подпись, проверка срока, разбор
# ---------------------------------------------------------------------------

def _b64(value: str) -> str:
    """Логин в виде, безопасном для куки (в логине может быть что угодно)."""
    raw = base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii")
    return raw.rstrip("=")


def _unb64(value: str) -> Optional[str]:
    padding = "=" * (-len(value) % 4)
    try:
        return base64.urlsafe_b64decode(value + padding).decode("utf-8")
    except Exception:  # noqa: BLE001 — битая кука это «не вошёл», а не сбой
        return None


def _sign(user: str, expires: int, nonce: str) -> str:
    payload = f"{user}|{expires}|{nonce}".encode("utf-8")
    return hmac.new(secret(), payload, sha256).hexdigest()


def make_token(user: str, now: Optional[float] = None) -> str:
    """Собрать куку входа: логин, срок, случайное число и подпись.

    Случайное число (nonce) делает каждую куку непохожей на прежнюю: две куки
    одного человека — разные строки, поэтому «украл вчерашнюю куку» не
    превращается в «подписи совпали».
    """
    moment = time.time() if now is None else now
    expires = int(moment) + ttl_seconds()
    nonce = secrets.token_hex(8)
    signature = _sign(user, expires, nonce)
    return f"{_b64(user)}.{expires}.{nonce}.{signature}"


def verify_token(token: Optional[str], now: Optional[float] = None) -> Optional[str]:
    """Проверить куку: вернуть логин, если подпись верна и срок не вышел."""
    if not token:
        return None
    parts = token.split(".")
    if len(parts) != 4:
        return None
    encoded_user, raw_expires, nonce, signature = parts
    user = _unb64(encoded_user)
    if user is None:
        return None
    try:
        expires = int(raw_expires)
    except ValueError:
        return None
    expected = _sign(user, expires, nonce)
    if not hmac.compare_digest(expected, signature):
        return None
    moment = time.time() if now is None else now
    if expires <= moment:
        return None
    return user


def normalize_secret(value: str) -> str:
    """Привести логин и пароль к сравнимому виду: убрать невидимое.

    Убираются пробелы по краям и невидимые символы (мягкий перенос, нулевой
    ширины), которые приносит КОПИРОВАНИЕ: телефон, копируя пароль из сообщения
    или заметки, добавляет пробел или мягкий перенос, человек видит «пароль
    верный», а сервер — другой (живой случай: вход с телефона отвечал 401,
    хотя пароль набирался из того же сообщения). Пароль, состоящий из пробелов
    и невидимых символов, после такой чистки становится пустым — и не проходит
    (пустой пароль не пускает никого).
    """
    text = (value or "").strip()
    for invisible in ("\u00ad", "\u200b", "\u200c", "\u200d", "\ufeff"):
        text = text.replace(invisible, "")
    return text


def check_credentials(user: str, password: str) -> bool:
    """Сверка логина и пароля с настройками (.env).

    Пароль хранится в .env (он и так закрыт от git) и сверяется
    `compare_digest` — посимвольное сравнение выдавало бы пароль по времени
    ответа. Логин сравнивается так же: это не секрет, но правило одно на оба
    поля, и «логин угадывается по времени» лишней подсказкой ни к чему.
    Сравниваются БАЙТЫ UTF-8: `compare_digest` со строками принимает только
    ASCII, а пароль на русском уронил бы маршрут входа пятисоткой.
    """
    expected_user = normalize_secret(config.ACCESS_USER or "")
    expected_password = normalize_secret(config.ACCESS_PASSWORD or "")
    if not expected_password:
        return False
    user_ok = hmac.compare_digest(
        normalize_secret(user).encode("utf-8"), expected_user.encode("utf-8"))
    password_ok = hmac.compare_digest(
        normalize_secret(password).encode("utf-8"), expected_password.encode("utf-8"))
    return bool(user_ok and password_ok)


# ---------------------------------------------------------------------------
# Защита от перебора пароля
# ---------------------------------------------------------------------------

def note_failure(ip: str) -> None:
    """Записать неудачную попытку входа с адреса."""
    now = time.monotonic()
    window = config.ACCESS_FAIL_WINDOW
    marks = [mark for mark in _FAILS.get(ip, []) if now - mark < window]
    marks.append(now)
    _FAILS[ip] = marks


def clear_failures(ip: str) -> None:
    """Забыть неудачные попытки адреса (вход удался)."""
    _FAILS.pop(ip, None)


def retry_after(ip: str) -> int:
    """Сколько секунд адрес ещё заблокирован (0 — не заблокирован)."""
    now = time.monotonic()
    window = config.ACCESS_FAIL_WINDOW
    marks = [mark for mark in _FAILS.get(ip, []) if now - mark < window]
    if marks:
        _FAILS[ip] = marks
    else:
        _FAILS.pop(ip, None)
    if len(marks) < config.ACCESS_MAX_FAILS:
        return 0
    oldest = min(marks)
    return max(1, int(window - (now - oldest)))


def reset_failures() -> None:
    """Забыть все неудачные попытки (нужен проверкам)."""
    _FAILS.clear()


# ---------------------------------------------------------------------------
# Адрес клиента и разбор запроса
# ---------------------------------------------------------------------------

def client_ip(scope: Dict[str, Any]) -> str:
    """Адрес клиента из ASGI-scope (пусто, если его нет)."""
    client: Tuple[str, int] | None = scope.get("client")
    return (client[0] if client else "") or ""


def is_local_ip(ip: str) -> bool:
    """Локальный ли адрес (тот же мак, а не сеть и не интернет).

    Только петля: 127.0.0.1 и ::1, в том числе в IPv4-обёртке (::ffff:127.0.0.1),
    как её отдаёт часть окружений. Адреса локальной сети (192.168.3.x) —
    НЕ локальные: с ноутбука соседа по Wi-Fi пароль спрашивается.
    """
    if not ip:
        return False
    value = ip.strip().lower()
    if value.startswith("::ffff:"):
        value = value[len("::ffff:"):]
    return value in ("127.0.0.1", "::1", "localhost") or value.startswith("127.")


def header(scope: Dict[str, Any], name: str) -> str:
    """Значение заголовка запроса (заголовки в ASGI — список байтовых пар)."""
    wanted = name.lower().encode("latin-1")
    for key, value in scope.get("headers") or []:
        if key.lower() == wanted:
            return value.decode("latin-1", errors="replace")
    return ""


def cookie_value(scope: Dict[str, Any]) -> Optional[str]:
    """Кука входа из заголовка Cookie (None — куки нет)."""
    raw = header(scope, "cookie")
    if not raw:
        return None
    for chunk in raw.split(";"):
        name, _, value = chunk.strip().partition("=")
        if name == COOKIE_NAME:
            return value or None
    return None


def wants_html(scope: Dict[str, Any]) -> bool:
    """Идёт ли запрос как переход браузера по странице (а не как запрос данных).

    Разница важна для отказа: браузеру надо показать страницу входа, а
    `fetch` из открытой страницы — получить 401 и уйти на неё сам (см. обёртку
    fetch в app/web/chat.html). Отсюда признак — не путь, а Accept.
    """
    if scope.get("method") != "GET":
        return False
    return "text/html" in header(scope, "accept").lower()


def set_cookie(response: Any, token: str) -> None:
    """Положить куку входа в ответ (флаги — в одном месте, а не по маршрутам)."""
    response.set_cookie(
        COOKIE_NAME,
        token,
        max_age=ttl_seconds(),
        httponly=True,      # скрипту страницы кука не нужна
        samesite="lax",     # чужой сайт не отправит её своей POST-формой
        secure=bool(config.ACCESS_COOKIE_SECURE),
        path="/",
    )


def clear_cookie(response: Any) -> None:
    """Снять куку входа (выход)."""
    response.delete_cookie(COOKIE_NAME, path="/")


# ---------------------------------------------------------------------------
# Сам гейт
# ---------------------------------------------------------------------------

class AuthGate:
    """ASGI-middleware: без входа внутрь не пускает (подробности — в докстринге модуля).

    Добавляется ПОСЛЕДНЕЙ в main.py: тогда она оказывается самым внешним слоем
    и отказ приходит раньше, чем запрос дойдёт до gzip и маршрутов.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            # WebSocket и время жизни приложения гейт не трогает.
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        if path in OPEN_PATHS:
            await self.app(scope, receive, send)
            return

        ip = client_ip(scope)
        if enabled():
            if local_bypass() and is_local_ip(ip):
                await self.app(scope, receive, send)
                return
            user = verify_token(cookie_value(scope))
            if user:
                await self.app(scope, receive, send)
                return
            await self._deny_locked(scope, receive, send)
            return

        # Гейт выключен (пароль не задан). Локально работаем как раньше, а
        # наружу не отдаём НИЧЕГО: одна незаполненная строка в .env не должна
        # открывать всему интернету диалоги, документы и ключи.
        if is_local_ip(ip):
            await self.app(scope, receive, send)
            return
        await self._deny_no_password(scope, receive, send)

    async def _deny_locked(self, scope: Dict[str, Any], receive: Any, send: Any) -> None:
        """Пароль задан, но клиент не вошёл."""
        if wants_html(scope):
            response = RedirectResponse(LOGIN_PATH, status_code=303)
        else:
            response = JSONResponse(
                {"detail": "Требуется вход: откройте страницу входа и введите пароль."},
                status_code=401,
            )
        response.headers[DENY_HEADER] = "login-required"
        response.headers["Cache-Control"] = "no-store"
        await response(scope, receive, send)

    async def _deny_no_password(self, scope: Dict[str, Any], receive: Any, send: Any) -> None:
        """Пароль не задан, а клиент пришёл снаружи: это отказ, а не «пусто»."""
        response = JSONResponse(
            {"detail": "Доступ из внешней сети закрыт: в .env не задан пароль "
                       "(ACCESS_PASSWORD). Смотрите tools/serve.sh и SESSION_PROMPT."},
            status_code=403,
        )
        response.headers[DENY_HEADER] = "no-password-configured"
        response.headers["Cache-Control"] = "no-store"
        await response(scope, receive, send)
