"""Самопроверка ВХОДА ПО ПАРОЛЮ — гейта доступа из внешней сети (app/auth.py).

Запуск:

    ./venv/bin/python tools/check_auth.py

Почему это проверяется отдельно и подробно. Сайт открыт наружу пробросом порта
на роутере, и гейт — единственное, что отделяет интернет от диалогов,
документов, рабочей истории (data/*.json), вызовов внешних инструментов и
модели за деньги владельца. Ошибка здесь не «мелкий дефект», а открытая дверь,
поэтому проверяется не только «правильный пароль пускает», но и все способы
обойти гейт: подделанная кука, просроченная кука, чужой секрет, адрес локальной
сети вместо петли, перебор пароля, выключенный гейт и выключенный обход петли.

Что проверяется (разделы):

* [1] кука входа: подпись, срок, подделка, чужой секрет, битые строки;
* [2] сверка логина и пароля (и что пустой пароль в настройках не пускает никого);
* [3] ГЕЙТ: что закрыто снаружи (страница, API, /docs, /openapi.json, несуществующие
  пути), что открыто (страница входа, вход, выход, состояние), что пускает кука;
* [4] локальный клиент: петля (127.0.0.1, ::1, ::ffff:127.0.0.1) проходит без
  пароля — этого требуют живые проверки проекта и работа за своим маком, — а
  адрес локальной сети и интернета пароль получает; ACCESS_LOCAL_BYPASS=0
  заставляет спрашивать пароль и локально;
* [5] вход и выход: флаги куки, 401 при неверной паре, 429 при переборе (закрыт
  даже верный пароль), сброс счётчика удачным входом, снятие куки выходом;
* [6] ГЕЙТ ВЫКЛЮЧЕН (пароль не задан): локально работа как раньше, а снаружи 403
  с причиной — не 200 ни при каком раскладе; ACCESS_ENABLED=1 без пароля закрывает
  вход, но наружу всё равно не пускает;
* [7] секрет подписи: файл создаётся с правами 600, вход переживает перезапуск
  приложения, секрет из настроек побеждает файл;
* [8] гейт НЕ буферизует потоки: события NDJSON проходят по мере появления;
* [9] проводка в приложении: гейт стоит во внешнем слое main.py, маршруты входа
  подключены, страница входа есть, а chat.html уходит на вход по 401 и умеет «Выйти».

Сеть не нужна: приложение собирается в памяти, запросы идут прямо в ASGI без
сокетов, рабочие data/*.json не трогаются (секрет пишется во временный каталог).
"""

import asyncio
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Изоляция: секрет подписи и настройки — свои, данные не трогаются --------
_TMP = tempfile.mkdtemp(prefix="auth-check-")
os.environ["ACCESS_SECRET_FILE"] = os.path.join(_TMP, "auth_secret")
os.environ["ACCESS_USER"] = "user"
os.environ["ACCESS_PASSWORD"] = "проверочный-пароль"
os.environ["ACCESS_TTL_HOURS"] = "168"
os.environ["ACCESS_MAX_FAILS"] = "3"
os.environ["ACCESS_FAIL_WINDOW"] = "600"
# Каждая настройка доступа называется ЯВНО, даже когда совпадает со значением
# по умолчанию: config читает .env, а там рабочие значения (например
# ACCESS_COOKIE_SECURE=1 для https), и проверка, «унаследовавшая» их, проверяла
# бы не то, что написано в её же названии (на этом она один раз и споткнулась).
os.environ["ACCESS_LOCAL_BYPASS"] = "1"
os.environ["ACCESS_COOKIE_SECURE"] = "0"
os.environ["ACCESS_SECRET"] = ""
os.environ["ACCESS_ENABLED"] = ""

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402

from app import auth, config  # noqa: E402
from app.routers import auth as auth_routes  # noqa: E402
from app.routers import pages  # noqa: E402
from app.web import frontend  # noqa: E402

FAILURES = []
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASSWORD = "проверочный-пароль"


def check(name, condition, detail=""):
    """Одна проверка: печатает результат и копит провалы."""
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(name)


def body_json(payload) -> bytes:
    """Тело запроса на вход — как его отправляет страница входа."""
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


# ---------------------------------------------------------------------------
# Настройки: config читает их один раз при импорте, поэтому проверки меняют
# значения прямо в модуле (и возвращают прежние).
# ---------------------------------------------------------------------------
class settings:
    """Временная подмена настроек доступа."""

    def __init__(self, **values):
        self.values = values
        self.saved = {}

    def __enter__(self):
        for key, value in self.values.items():
            self.saved[key] = getattr(config, key)
            setattr(config, key, value)
        auth.reset_cache()
        return self

    def __exit__(self, *exc):
        for key, value in self.saved.items():
            setattr(config, key, value)
        auth.reset_cache()
        return False


# ---------------------------------------------------------------------------
# Приложение для проверок: гейт + маршруты входа + страницы + потоковый маршрут
# (последний — как /api/agent/chat: NDJSON, который гейт обязан пропускать
# по мере появления, а не копить).
# ---------------------------------------------------------------------------
STREAM_CHUNKS = [b'{"type":"state"}\n', b'{"type":"bot"}\n', b'{"type":"done"}\n']


def build_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(auth.AuthGate)
    app.include_router(pages.router)
    app.include_router(auth_routes.router)

    @app.post("/api/agent/chat")
    async def stream_route():  # noqa: ANN202 — маршрут-заглушка
        async def body():
            for chunk in STREAM_CHUNKS:
                yield chunk
        return StreamingResponse(body(), media_type="application/x-ndjson")

    @app.get("/api/agent/workspace")
    def workspace_route() -> dict:
        return {"tasks": []}

    return app


async def call(app, path, *, method="GET", headers=None, client=("203.0.113.5", 41234),
               body=b""):
    """Запрос прямо в ASGI, без сокета: (код, заголовки, тело, все сообщения).

    Свой вызов, а не TestClient, потому что проверяется именно АДРЕС КЛИЕНТА
    (`scope["client"]`): от него зависит и обход петли, и защита от перебора, —
    а TestClient подставляет один и тот же адрес и подменить его не даёт.
    """
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": b"",
        "root_path": "",
        "headers": [(k.lower().encode("latin-1"), v.encode("latin-1"))
                    for k, v in (headers or {}).items()],
        "client": client,
        "server": ("127.0.0.1", 8000),
    }

    # Тело запроса отдаётся ОДИН раз, дальше получатель «молчит и не рвёт
    # соединение» (событие, которое никто не взводит). Так устроен и настоящий
    # сервер: потоковые ответы Starlette слушают разрыв соединения и завершают
    # работу, когда весь ответ отправлен, — а приёмник, отвечающий бесконечно,
    # подвесил бы проверку навсегда.
    answered = False
    idle = asyncio.Event()

    async def receive():
        nonlocal answered
        if not answered:
            answered = True
            return {"type": "http.request", "body": body, "more_body": False}
        await idle.wait()
        return {"type": "http.disconnect"}

    messages = []

    async def send(message):
        messages.append(message)

    # Предел времени: проверка обязана ЗАВЕРШИТЬСЯ (пусть и провалом), а не
    # висеть, если маршрут решил ничего не отвечать.
    await asyncio.wait_for(app(scope, receive, send), timeout=15)
    start = next(m for m in messages if m["type"] == "http.response.start")
    out_headers = {k.decode("latin-1").lower(): v.decode("latin-1")
                   for k, v in start["headers"]}
    out_body = b"".join(m.get("body", b"") for m in messages
                        if m["type"] == "http.response.body")
    return start["status"], out_headers, out_body, messages


def cookie_header(token: str) -> dict:
    return {"Cookie": f"{auth.COOKIE_NAME}={token}"}


def login_headers(payload) -> dict:
    return {"Content-Type": "application/json"}


EXTERNAL = ("203.0.113.5", 41234)      # «интернет» (документационный диапазон)
HOME_LAN = ("192.168.3.77", 51234)     # другое устройство в сети дома
LOCAL = ("127.0.0.1", 51999)           # сам мак


# ---------------------------------------------------------------------------
# [1] Кука входа
# ---------------------------------------------------------------------------
def test_token():
    print("\n[1] Кука входа: подпись и срок")
    auth.reset_cache()
    token = auth.make_token("user")
    check("своя кука принимается", auth.verify_token(token) == "user", str(token[:24]))
    check("в куке четыре части (логин, срок, число, подпись)",
          len(token.split(".")) == 4, token)

    head, expires, nonce, signature = token.split(".")
    check("подделанная подпись не проходит",
          auth.verify_token(f"{head}.{expires}.{nonce}.{'0' * len(signature)}") is None)
    check("подменённый логин не проходит (подпись считается по логину)",
          auth.verify_token(
              f"{auth._b64('admin')}.{expires}.{nonce}.{signature}") is None)
    check("подменённый срок не проходит",
          auth.verify_token(
              f"{head}.{int(expires) + 99999}.{nonce}.{signature}") is None)
    check("подменённое случайное число не проходит",
          auth.verify_token(f"{head}.{expires}.{'f' * 16}.{signature}") is None)

    # Срок проверяется по времени из самой куки — так проверка не зависит от
    # того, сколько она шла.
    moment = int(expires)
    check("кука живёт заявленный срок (за секунду до конца — годна)",
          auth.verify_token(token, now=moment - 1) == "user")
    check("кука перестаёт годиться по истечении срока",
          auth.verify_token(token, now=moment + 1) is None)

    check("пустой куки нет — и входа нет",
          auth.verify_token(None) is None and auth.verify_token("") is None)
    check("битые строки не проходят (и не роняют проверку)",
          all(auth.verify_token(bad) is None for bad in
              ("мусор", "a.b.c", "a.b.c.d.e", "!!!.###.%%%.$$$", "user.not-a-number.x.y")))

    with settings(ACCESS_SECRET="другой-секрет"):
        check("кука, подписанная ДРУГИМ секретом, не проходит",
              auth.verify_token(token) is None)
    check("после возврата настройки прежняя кука снова годна",
          auth.verify_token(token) == "user")

    second = auth.make_token("user")
    check("две куки одного человека не совпадают (случайное число)",
          second != token and auth.verify_token(second) == "user")

    with settings(ACCESS_TTL_HOURS=1):
        short = auth.make_token("user")
        deadline = int(short.split(".")[1])
        check("срок куки следует настройке (1 час)",
              auth.verify_token(short, now=deadline - 1) == "user"
              and auth.verify_token(short, now=deadline + 1) is None)


# ---------------------------------------------------------------------------
# [2] Сверка логина и пароля
# ---------------------------------------------------------------------------
def test_credentials():
    print("\n[2] Сверка логина и пароля")
    check("верная пара проходит", auth.check_credentials("user", PASSWORD))
    check("неверный пароль не проходит",
          not auth.check_credentials("user", PASSWORD[:-1]))
    check("неверный логин не проходит",
          not auth.check_credentials("root", PASSWORD))
    check("пустой пароль не проходит", not auth.check_credentials("user", ""))
    with settings(ACCESS_PASSWORD=""):
        check("пустой пароль в настройках не пускает НИКОГО (в том числе пустым)",
              not auth.check_credentials("user", "")
              and not auth.check_credentials("", ""))
    check("регистр логина значим (User != user)",
          not auth.check_credentials("User", PASSWORD))

    # НЕВИДИМОЕ НЕ СЧИТАЕТСЯ ОШИБКОЙ В ПАРОЛЕ: телефон при копировании добавляет
    # пробел или мягкий перенос, и «верный» пароль уходил бы на сервер с лишним
    # символом (живой случай: вход с телефона отвечал 401 при верном пароле).
    check("пробел по краям не мешает входу",
          auth.check_credentials(" user ", " " + PASSWORD + " "))
    check("мягкий перенос и символ нулевой ширины не мешают входу",
          auth.check_credentials("user", PASSWORD[:3] + "\u00ad" + PASSWORD[3:])
          and auth.check_credentials("user", "\u200b" + PASSWORD))
    check("в середине пароля подмена символа по-прежнему не проходит",
          not auth.check_credentials("user", PASSWORD[:3] + "x" + PASSWORD[4:]))
    check("пароль из одних пробелов не проходит",
          not auth.check_credentials("user", "   "))
    with settings(ACCESS_PASSWORD="   "):
        check("пробельный пароль в настройках не пускает никого (он и есть пустой)",
              not auth.check_credentials("user", "   ")
              and not auth.check_credentials("user", ""))


# ---------------------------------------------------------------------------
# [3] Что гейт закрывает и что открывает
# ---------------------------------------------------------------------------
async def test_gate():
    print("\n[3] Гейт: закрытое и открытое")
    app = build_app()
    auth.reset_failures()

    status, headers, _, _ = await call(app, "/", client=EXTERNAL,
                                       headers={"Accept": "text/html"})
    check("переход браузера на главную снаружи — 303 на страницу входа",
          status == 303 and headers.get("location") == auth.LOGIN_PATH,
          f"{status} {headers.get('location')}")
    check("отказ помечен заголовком (видно и в curl)",
          headers.get(auth.DENY_HEADER.lower()) == "login-required", str(headers))
    check("отказ не кэшируется браузером",
          headers.get("cache-control") == "no-store", str(headers.get("cache-control")))

    for path, method in (("/api/agent/workspace", "GET"),
                         ("/api/agent/chat", "POST"),
                         ("/health", "GET"),
                         ("/docs", "GET"),
                         ("/openapi.json", "GET"),
                         ("/чего-угодно", "GET")):
        status, _, _, _ = await call(app, path, method=method, client=EXTERNAL)
        check(f"снаружи закрыт {method} {path} (401)", status == 401, str(status))

    status, _, body, _ = await call(app, "/health", client=EXTERNAL)
    check("в отказе API причина названа словами",
          "Требуется вход" in body.decode("utf-8", "replace"),
          body[:120].decode("utf-8", "replace"))

    status, _, body, _ = await call(app, "/login", client=EXTERNAL,
                                    headers={"Accept": "text/html"})
    check("страница входа доступна без входа (200)", status == 200, str(status))
    check("страница входа — это app/web/login.html",
          b"ACCESS_PASSWORD" in body and "Войти".encode("utf-8") in body,
          body[:80].decode("utf-8", "replace"))

    status, _, body, _ = await call(app, auth.STATE_API, client=EXTERNAL)
    check("состояние доступа доступно без входа (странице входа оно нужно)",
          status == 200 and b'"enabled":true' in body.replace(b" ", b""), str(status))

    status, _, _, _ = await call(app, auth.LOGIN_API, method="POST", client=EXTERNAL,
                                 headers=login_headers(None),
                                 body=body_json({"user": "user", "password": "мимо"}))
    check("страница входа и маршрут входа за гейтом не прячутся (дошли до маршрута)",
          status == 401, str(status))

    token = auth.make_token(config.ACCESS_USER)
    status, _, body, _ = await call(app, "/", client=EXTERNAL,
                                    headers={"Accept": "text/html", **cookie_header(token)})
    check("с верной кукой главная отдаётся снаружи (200)", status == 200, str(status))
    check("снаружи с кукой отдаётся именно чат", b"agent-toggle" in body, str(len(body)))
    status, _, _, _ = await call(app, "/api/agent/workspace", client=EXTERNAL,
                                 headers=cookie_header(token))
    check("с верной кукой API работает снаружи", status == 200, str(status))
    status, _, _, _ = await call(app, "/", client=EXTERNAL,
                                 headers={"Accept": "text/html",
                                          **cookie_header(token + "x")})
    check("испорченная кука снаружи не пускает", status == 303, str(status))

    # Запрос данных (fetch из открытой страницы приходит без Accept: text/html) —
    # получает 401, а не перенаправление: перенаправление fetch бы «проглотил».
    status, _, _, _ = await call(app, "/", client=EXTERNAL, headers={"Accept": "*/*"})
    check("запрос данных без входа — 401 (а не 303)", status == 401, str(status))


# ---------------------------------------------------------------------------
# [4] Локальный клиент
# ---------------------------------------------------------------------------
async def test_local_client():
    print("\n[4] Локальный клиент (петля) и сеть дома")
    app = build_app()

    for ip in ("127.0.0.1", "::1", "::ffff:127.0.0.1", "127.0.0.53"):
        status, _, _, _ = await call(app, "/api/agent/workspace", client=(ip, 5000))
        check(f"локальный адрес {ip} проходит без пароля", status == 200, str(status))

    for ip in ("192.168.3.77", "10.0.0.5", "203.0.113.5", "172.16.0.9"):
        status, _, _, _ = await call(app, "/api/agent/workspace", client=(ip, 5000))
        check(f"адрес {ip} пароль получает (401)", status == 401, str(status))

    with settings(ACCESS_LOCAL_BYPASS=0):
        status, _, _, _ = await call(app, "/", client=LOCAL,
                                     headers={"Accept": "text/html"})
        check("при ACCESS_LOCAL_BYPASS=0 пароль спрашивают и локально",
              status == 303, str(status))
        status, _, _, _ = await call(app, "/api/agent/workspace", client=LOCAL,
                                     headers=cookie_header(auth.make_token("user")))
        check("локально с кукой вход работает как снаружи (200)", status == 200, str(status))
    status, _, _, _ = await call(app, "/", client=LOCAL, headers={"Accept": "text/html"})
    check("после возврата ACCESS_LOCAL_BYPASS=1 локально снова без пароля",
          status == 200, str(status))


# ---------------------------------------------------------------------------
# [5] Вход и выход
# ---------------------------------------------------------------------------
async def test_login_logout():
    print("\n[5] Вход, отказ и перебор пароля")
    app = build_app()
    auth.reset_failures()
    good = body_json({"user": "user", "password": PASSWORD})
    good_spaced = body_json({"user": " user ", "password": PASSWORD + " "})
    bad = body_json({"user": "user", "password": "мимо"})

    status, headers, body, _ = await call(
        app, auth.LOGIN_API, method="POST", client=EXTERNAL,
        headers=login_headers(None), body=good)
    check("верная пара: вход выполнен (200)", status == 200, f"{status} {body[:120]}")
    raw_cookie = headers.get("set-cookie", "")
    check("кука выставлена с флагами HttpOnly, SameSite=Lax и путём /",
          "HttpOnly" in raw_cookie and "samesite=lax" in raw_cookie.lower()
          and "Path=/" in raw_cookie, raw_cookie)
    check("срок куки соответствует настройке (168 часов)",
          f"Max-Age={config.ACCESS_TTL_HOURS * 3600}" in raw_cookie, raw_cookie)
    check("флаг Secure по умолчанию НЕ ставится (сайт отдаётся по http)",
          "secure" not in raw_cookie.lower(), raw_cookie)

    # https: кука обязана уходить только по шифрованному каналу.
    with settings(ACCESS_COOKIE_SECURE=1):
        _, secure_headers, _, _ = await call(
            app, auth.LOGIN_API, method="POST", client=EXTERNAL,
            headers=login_headers(None), body=good)
        check("при ACCESS_COOKIE_SECURE=1 кука помечена Secure (только https)",
              "secure" in secure_headers.get("set-cookie", "").lower(),
              secure_headers.get("set-cookie", ""))
    check("ответ входа не кэшируется",
          headers.get("cache-control") == "no-store", str(headers))
    check("пароль в ответе не появляется", PASSWORD.encode() not in body, body[:120])

    status, _, _, _ = await call(app, auth.LOGIN_API, method="POST", client=EXTERNAL,
                                 headers=login_headers(None), body=good_spaced)
    check("вход с лишними пробелами по краям тоже проходит (так приходит копирование)",
          status == 200, str(status))

    token = raw_cookie.split(f"{auth.COOKIE_NAME}=")[1].split(";")[0]
    check("полученная кука действительно пускает (подпись верна)",
          auth.verify_token(token) == "user", token[:24])

    status, _, _, _ = await call(app, auth.LOGIN_API, method="POST", client=EXTERNAL,
                                 headers=login_headers(None), body=bad)
    check("неверный пароль: 401", status == 401, str(status))

    # Перебор: держим неудачные попытки до предела (в этой проверке он равен 3),
    # а следующий запрос с того же адреса обязан получить отказ — и с верным
    # паролем тоже.
    for attempt in range(2, config.ACCESS_MAX_FAILS + 1):
        status, _, _, _ = await call(app, auth.LOGIN_API, method="POST", client=EXTERNAL,
                                     headers=login_headers(None), body=bad)
        check(f"неудачная попытка {attempt} из {config.ACCESS_MAX_FAILS} — 401",
              status == 401, str(status))

    status, _, body, _ = await call(app, auth.LOGIN_API, method="POST", client=EXTERNAL,
                                    headers=login_headers(None), body=bad)
    check("перебор: запрос после предела неудач закрыт (429)",
          status == 429, f"{status} {body[:140]}")
    status, headers, _, _ = await call(app, auth.LOGIN_API, method="POST", client=EXTERNAL,
                                       headers=login_headers(None), body=good)
    check("закрытый адрес не пускает даже с ВЕРНЫМ паролем", status == 429, str(status))
    check("в отказе перебора сказано, сколько ждать (Retry-After)",
          headers.get("retry-after", "").isdigit(), str(headers.get("retry-after")))
    status, _, _, _ = await call(app, auth.LOGIN_API, method="POST", client=HOME_LAN,
                                 headers=login_headers(None), body=good)
    check("другой адрес перебором не задет (вход проходит)", status == 200, str(status))

    auth.reset_failures()
    for _ in range(config.ACCESS_MAX_FAILS - 1):
        await call(app, auth.LOGIN_API, method="POST", client=EXTERNAL,
                   headers=login_headers(None), body=bad)
    status, _, _, _ = await call(app, auth.LOGIN_API, method="POST", client=EXTERNAL,
                                 headers=login_headers(None), body=good)
    check("удачный вход счётчик неудач сбрасывает", status == 200, str(status))
    status, _, _, _ = await call(app, auth.LOGIN_API, method="POST", client=EXTERNAL,
                                 headers=login_headers(None), body=bad)
    check("после сброса отсчёт начинается заново (401, а не 429)", status == 401, str(status))

    status, _, body, _ = await call(app, auth.STATE_API, client=EXTERNAL,
                                    headers=cookie_header(auth.make_token("user")))
    check("состояние показывает вход для клиента с кукой",
          status == 200 and b'"authenticated":true' in body.replace(b" ", b""), str(body))

    status, headers, _, _ = await call(app, auth.LOGOUT_API, method="POST", client=EXTERNAL,
                                       headers=cookie_header(auth.make_token("user")))
    check("выход проходит без входа (маршрут открыт)", status == 200, str(status))
    cleared = headers.get("set-cookie", "")
    check("выход снимает куку (пустое значение)",
          f"{auth.COOKIE_NAME}=" in cleared
          and ("Max-Age=0" in cleared or "expires=" in cleared.lower()), cleared)

    # Тело запроса разбирает САМ маршрут, а не схема Pydantic: схема на отказе
    # ПРИВОДИТ значение поля в теле ошибки, и слишком длинный пароль уехал бы
    # обратно в ответ (и в инструменты разработчика, и в прокси).
    long_password = "п" * 300
    status, _, body, _ = await call(app, auth.LOGIN_API, method="POST", client=EXTERNAL,
                                    headers=login_headers(None),
                                    body=body_json({"user": "user", "password": long_password}))
    check("слишком длинный пароль отвергается (422), а не сравнивается",
          status == 422, str(status))
    check("в отказе НЕ повторяется присланный пароль",
          long_password.encode("utf-8") not in body, body[:200].decode("utf-8", "replace"))
    status, _, body, _ = await call(app, auth.LOGIN_API, method="POST", client=EXTERNAL,
                                    headers=login_headers(None),
                                    body=body_json({"user": "п" * 100, "password": "x"}))
    check("слишком длинный логин отвергается (422)", status == 422, str(status))
    status, _, body, _ = await call(app, auth.LOGIN_API, method="POST", client=EXTERNAL,
                                    headers=login_headers(None), body="не json".encode("utf-8"))
    check("битое тело — 400 с причиной, а не 500", status == 400, str(status))
    status, _, body, _ = await call(app, auth.LOGIN_API, method="POST", client=EXTERNAL,
                                    headers=login_headers(None),
                                    body=body_json({"user": 5, "password": ["x"]}))
    check("не-строки в полях — 400 с причиной", status == 400, str(status))
    status, _, _, _ = await call(app, auth.LOGIN_API, method="POST", client=EXTERNAL,
                                 headers=login_headers(None), body=b"[]")
    check("тело не объект — 400", status == 400, str(status))

    auth.reset_failures()


# ---------------------------------------------------------------------------
# [6] Гейт выключен
# ---------------------------------------------------------------------------
async def test_gate_off():
    print("\n[6] Гейт выключен (пароль не задан)")
    app = build_app()

    with settings(ACCESS_PASSWORD="", _ACCESS_ENABLED_RAW=""):
        check("без пароля гейт считает себя выключенным", not auth.enabled())
        status, _, _, _ = await call(app, "/", client=LOCAL, headers={"Accept": "text/html"})
        check("локально работа как раньше (страница 200)", status == 200, str(status))
        status, _, _, _ = await call(app, "/api/agent/workspace", client=LOCAL)
        check("локально API работает как раньше (200)", status == 200, str(status))
        status, headers, body, _ = await call(app, "/api/agent/workspace", client=EXTERNAL)
        check("снаружи — отказ (403), а не работа", status == 403, str(status))
        check("в отказе назван незаполненный ACCESS_PASSWORD",
              b"ACCESS_PASSWORD" in body, body[:160].decode("utf-8", "replace"))
        check("отказ помечен заголовком no-password-configured",
              headers.get(auth.DENY_HEADER.lower()) == "no-password-configured", str(headers))
        status, _, _, _ = await call(app, "/", client=HOME_LAN, headers={"Accept": "text/html"})
        check("из сети дома без пароля тоже отказ (403)", status == 403, str(status))
        status, _, body, _ = await call(app, auth.LOGIN_API, method="POST", client=EXTERNAL,
                                        headers=login_headers(None),
                                        body=body_json({"user": "user", "password": ""}))
        check("вход при незаданном пароле — 403 с причиной, а не 401",
              status == 403 and b"ACCESS_PASSWORD" in body, f"{status} {body[:140]}")

    with settings(ACCESS_PASSWORD="", _ACCESS_ENABLED_RAW="1"):
        check("ACCESS_ENABLED=1 при пустом пароле включает гейт (наружу не пускает)", auth.enabled())
        status, _, _, _ = await call(app, "/api/agent/workspace", client=EXTERNAL)
        check("тогда снаружи закрыто (401, а не 403: гейт включён)", status == 401, str(status))
        status, _, _, _ = await call(app, "/api/agent/workspace", client=LOCAL)
        check("локальная петля при этом всё равно проходит", status == 200, str(status))

    with settings(_ACCESS_ENABLED_RAW="0"):
        check("ACCESS_ENABLED=0 выключает гейт даже при заданном пароле",
              not auth.enabled())


# ---------------------------------------------------------------------------
# [7] Секрет подписи
# ---------------------------------------------------------------------------
def test_secret():
    print("\n[7] Секрет подписи куки")
    path = config.ACCESS_SECRET_FILE
    if os.path.isfile(path):
        os.remove(path)
    auth.reset_cache()
    first = auth.secret()
    check("секрет создаётся сам при первом входе (файл появился)",
          os.path.isfile(path), path)
    check("секрет длинный (256 бит энтропии — 64 знака)", len(first) == 64, str(len(first)))
    mode = os.stat(path).st_mode & 0o777
    check("файл секрета закрыт правами 600", mode == 0o600, oct(mode))
    auth.reset_cache()
    check("после «перезапуска» (сброса кэша) секрет тот же — вход переживает рестарт",
          auth.secret() == first)
    token = auth.make_token("user")
    auth.reset_cache()
    check("кука, выданная ДО перезапуска, после него годна",
          auth.verify_token(token) == "user")
    with settings(ACCESS_SECRET="секрет-из-настроек-длиннее-двадцати-четырёх"):
        check("настройка ACCESS_SECRET побеждает файл",
              auth.secret() == "секрет-из-настроек-длиннее-двадцати-четырёх".encode())
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("   \n")
    auth.reset_cache()
    check("пустой файл секрета не превращается в пустой секрет",
          len(auth.secret()) == 64, str(auth.secret()[:8]))


# ---------------------------------------------------------------------------
# [8] Потоки проходят насквозь
# ---------------------------------------------------------------------------
async def test_streaming():
    print("\n[8] Гейт не буферизует потоки (NDJSON агента)")
    app = build_app()
    token = auth.make_token("user")

    status, _, body, messages = await call(
        app, "/api/agent/chat", method="POST", client=EXTERNAL,
        headers=cookie_header(token), body=body_json({"content": "привет"}))
    bodies = [m["body"] for m in messages if m["type"] == "http.response.body"]
    # Первое сообщение тела у Starlette пустое (объявление потока) — считаем
    # только настоящие куски.
    chunks = [chunk for chunk in bodies if chunk]
    check("потоковый маршрут открывается с верной кукой (200)", status == 200, str(status))
    check("а без куки он закрыт 401, а не «молчит»",
          (await call(app, "/api/agent/chat", method="POST", client=EXTERNAL))[0] == 401)
    check("гейт пропустил поток КУСКАМИ, а не склеил его в один ответ",
          chunks == STREAM_CHUNKS, str(chunks))
    check("кусков столько же, сколько записал маршрут",
          len(chunks) == len(STREAM_CHUNKS), str(len(chunks)))
    check("первый кусок дошёл до получателя до последнего",
          len(chunks) > 1 and chunks[0] == STREAM_CHUNKS[0], str(bodies[:2]))


# ---------------------------------------------------------------------------
# [9] Проводка в приложении и в интерфейсе
# ---------------------------------------------------------------------------
def test_wiring():
    print("\n[9] Проводка: main.py, маршруты, страница входа, интерфейс")
    main_src = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    check("main.py импортирует гейт из app/auth.py",
          "from app.auth import AuthGate" in main_src)
    check("main.py подключает маршруты входа", "auth_routes.router" in main_src)
    check("гейт добавляется ПОСЛЕДНИМ (то есть внешним слоем)",
          main_src.index("add_middleware(AuthGate)")
          > main_src.index("add_middleware(GZipExceptStreams"), "порядок в main.py")
    check("в main.py сказано, что запуск наружу — tools/serve.sh",
          "tools/serve.sh" in main_src)

    app = build_app()
    paths = {getattr(route, "path", "") for route in app.routes}
    check("маршрут страницы входа есть в приложении", auth.LOGIN_PATH in paths, str(paths))
    check("маршрут входа есть в приложении", auth.LOGIN_API in paths)
    check("маршрут выхода есть в приложении", auth.LOGOUT_API in paths)
    check("маршрут состояния есть в приложении", auth.STATE_API in paths)
    check("страница входа существует файлом", os.path.isfile(frontend.LOGIN_HTML_PATH))
    login_html = open(frontend.LOGIN_HTML_PATH, encoding="utf-8").read()
    check("страница входа не тянет ресурсы со стороны (самодостаточна)",
          "src=" not in login_html.split("<body")[1], "внешние ресурсы у страницы входа")
    check("страница входа говорит про .env, когда пароль не задан",
          "ACCESS_PASSWORD" in login_html and "не настроен" in login_html)
    check("на странице входа есть кнопка «показать пароль» (проверить, что набрано)",
          'id="reveal"' in login_html and "passwordInput.type" in login_html)
    check("у полей входа выключены автозаглавные буквы и автозамена",
          login_html.count('autocapitalize="none"') >= 2
          and login_html.count('autocorrect="off"') >= 2, "атрибуты полей входа")
    check("на странице входа сказано, что пароль — латиница",
          "латинские буквы" in login_html)
    check("страница входа срезает пробелы до отправки (и сервер тоже)",
          ".trim()" in login_html and "normalize_secret" in
          open(os.path.join(ROOT, "app", "auth.py"), encoding="utf-8").read())

    chat = open(frontend.CHAT_HTML_PATH, encoding="utf-8").read()
    check("chat.html уходит на вход при 401 (обёртка fetch)",
          "response.status === 401" in chat and "location.replace('/login')" in chat)
    check("chat.html спрашивает состояние доступа при открытии",
          "loadAccessState()" in chat)
    check("кнопка «Выйти» есть в разметке и скрыта по умолчанию",
          'id="logout-btn"' in chat and 'logout-btn" type="button" hidden' in chat)
    check("выход снимает вход на сервере и уводит на страницу входа",
          "/api/auth/logout" in chat and "logoutBtn.addEventListener" in chat)

    serve = os.path.join(ROOT, "tools", "serve.sh")
    check("скрипт запуска наружу существует", os.path.isfile(serve))
    if os.path.isfile(serve):
        text = open(serve, encoding="utf-8").read()
        check("скрипт запуска слушает 0.0.0.0 (иначе снаружи «connection refused»)",
              "0.0.0.0" in text)
        check("скрипт запуска не поднимает сервер без пароля",
              "require_password" in text and "ACCESS_PASSWORD" in text)
        check("запуск переживает закрытие терминала (nohup и pid-файл)",
              "nohup" in text and "serve.pid" in text)
        check("serve.sh предупреждает, что правки .env без перезапуска НЕ применены",
              "env_changed_since_start" in text and "НЕ применены" in text
              and "ENV_STAMP" in text)

    env_example = open(os.path.join(ROOT, ".env.example"), encoding="utf-8").read()
    check("в .env.example описаны настройки доступа",
          all(key in env_example for key in
              ("ACCESS_USER", "ACCESS_PASSWORD", "ACCESS_LOCAL_BYPASS",
               "ACCESS_TTL_HOURS", "ACCESS_MAX_FAILS")))


async def main():
    test_token()
    test_credentials()
    await test_gate()
    await test_local_client()
    await test_login_logout()
    await test_gate_off()
    test_secret()
    await test_streaming()
    test_wiring()
    print("\nИтог: " + ("ПРОВАЛЕНО проверок: " + str(len(FAILURES))
                        if FAILURES else "все проверки пройдены"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    # Цикл создаётся ЯВНО: asyncio.get_event_loop() устарел в Python 3.12.
    sys.exit(asyncio.run(main()))
