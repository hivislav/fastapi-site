"""Маршруты входа, выхода и состояния доступа.

Сама защита живёт в app/auth.py (ASGI-гейт перед всем приложением); здесь
только то, что обязано быть доступно ДО входа: страница входа и два маршрута —
`/api/auth/login` и `/api/auth/logout`. Эти три пути перечислены в
`auth.OPEN_PATHS` — единственное исключение из гейта.
"""

import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from app import auth, config
from app.web import frontend

router = APIRouter()

# Пределы длины пары. Не косметика: пара приходит ИЗ ИНТЕРНЕТА, и без предела
# поле стало бы способом прислать мегабайт и занять процесс сравнением.
MAX_USER_CHARS = 64
MAX_PASSWORD_CHARS = 256


async def _credentials(request: Request) -> tuple:
    """Логин и пароль из тела запроса — БЕЗ ЭХА значений в ответе.

    Почему не схема Pydantic, хотя в проекте тела запросов описаны схемами. На
    нарушении предела FastAPI отвечает 422 и в теле ошибки ПРИВОДИТ ЗНАЧЕНИЕ
    поля («input»): слишком длинный пароль уехал бы обратно в ответ и остался бы
    в инструментах разработчика, в прокси и на скриншоте. Правило «пароль не
    попадает никуда, кроме сверки» важнее единообразия, поэтому разбор здесь
    свой: причина отказа называется словами и без значений.
    """
    try:
        raw = await request.body()
        data = json.loads(raw.decode("utf-8") or "{}")
    except Exception:  # noqa: BLE001 — битое тело это отказ, а не сбой
        raise HTTPException(status_code=400, detail="Неверный формат запроса входа: ожидается JSON.")
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Неверный формат запроса входа: ожидается объект.")
    user = data.get("user", "")
    password = data.get("password", "")
    if not isinstance(user, str) or not isinstance(password, str):
        raise HTTPException(status_code=400, detail="Неверный формат запроса входа: логин и пароль — строки.")
    if len(user) > MAX_USER_CHARS or len(password) > MAX_PASSWORD_CHARS:
        raise HTTPException(
            status_code=422,
            detail=f"Слишком длинная пара: логин до {MAX_USER_CHARS} символов, "
                   f"пароль до {MAX_PASSWORD_CHARS}.",
        )
    return user, password


@router.get("/login", response_class=FileResponse)
def login_page() -> str:
    """Страница входа (отдаётся и когда вход уже выполнен: так честнее — видно,
    что доступ закрыт паролем, и можно войти другим логином)."""
    return frontend.LOGIN_HTML_PATH


@router.post("/api/auth/login")
async def login(request: Request) -> JSONResponse:
    """Вход по логину и паролю: ставит подписанную куку.

    Отказы разные и с причиной: 429 — перебор с этого адреса, 403 — вход не
    настроен (не задан пароль), 401 — неверная пара, 400/422 — битое или слишком
    длинное тело. Одинаковый ответ «не получилось» на все случаи заставил бы
    владельца гадать, что сломалось: пароль, .env или защита от перебора.
    """
    ip = auth.client_ip(request.scope)
    wait = auth.retry_after(ip)
    if wait:
        raise HTTPException(
            status_code=429,
            detail=f"Слишком много неудачных попыток входа. Повторите через {wait} с.",
            headers={"Retry-After": str(wait)},
        )
    user, password = await _credentials(request)
    if not auth.enabled():
        raise HTTPException(
            status_code=403,
            detail="Вход не настроен: в .env не задан ACCESS_PASSWORD.",
        )
    if not auth.check_credentials(user, password):
        auth.note_failure(ip)
        raise HTTPException(status_code=401, detail="Неверный логин или пароль.")

    auth.clear_failures(ip)
    response = JSONResponse({"ok": True, "user": config.ACCESS_USER})
    auth.set_cookie(response, auth.make_token(config.ACCESS_USER))
    response.headers["Cache-Control"] = "no-store"
    return response


@router.post("/api/auth/logout")
def logout() -> JSONResponse:
    """Выход: кука снимается. Сама сессия на сервере не хранится, поэтому
    «выйти» — это ровно «забыть куку» (и она же перестанет годиться, когда
    истечёт её срок)."""
    response = JSONResponse({"ok": True})
    auth.clear_cookie(response)
    response.headers["Cache-Control"] = "no-store"
    return response


@router.get("/api/auth/state")
def state(request: Request) -> dict:
    """Состояние доступа для интерфейса.

    `required` — спрашивают ли пароль у ЭТОГО клиента: только тогда в шапке
    чата показывается кнопка «Выйти» (локально, где пароль не спрашивается,
    кнопка была бы мёртвой).
    """
    scope = request.scope
    ip = auth.client_ip(scope)
    local = auth.is_local_ip(ip)
    authenticated = bool(auth.verify_token(auth.cookie_value(scope)))
    gate = auth.enabled()
    required = gate and not (local and auth.local_bypass())
    return {
        "enabled": gate,
        "authenticated": authenticated,
        "user": config.ACCESS_USER if authenticated else "",
        "local": local,
        "required": required,
    }
