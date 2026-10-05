"""Клиент реальной LLM.

Отправляет запросы к OpenAI-совместимому endпоинту выбранного ПРОВАЙДЕРА:
официальный API DeepSeek (по умолчанию, модель DeepSeek-V4-Flash) или
Yandex Cloud AI Studio (модели настройки «Тест моделей»). У провайдеров свои
адрес и ключ — они выбираются по имени провайдера или по модели
(config.provider_spec / provider_for_model), а не берутся «одни на всё».
Не содержит маршрутов и демо-логики — только HTTP-вызов модели.

Особенности слоя (важны для расхода токенов и устойчивости):

* **keep-alive.** Соединение переиспользуется (одно на поток): раньше каждый
  вызов делал новый TCP+TLS-handshake, а служебных вызовов у одного запроса
  пользователя бывает 3–5.
* **Повторы с backoff.** 429/5xx/сетевые сбои повторяются (см. RETRY_ATTEMPTS),
  потому что один 429 на шаге уничтожал всю задачу, а токены предыдущих шагов
  уже оплачены. Ошибки клиента (400/401/403) не повторяются: это не сбой сети.
* **Отменяемость.** `call_llm_async` выполняется в отдельном потоке; при отмене
  (клиент закрыл страницу) соединение рвётся из event loop, а не ждёт таймаут —
  иначе провайдер продолжал генерировать (и тарифицировать) ответ.
* **Учёт сбоя.** Неудачный вызов возвращает не `None`, а метрики с пометкой
  `failed`: вызывающий код видит, что обращение к модели БЫЛО (и сколько их
  было), а не «модель ничего не ответила».
"""

import asyncio
import json
import logging
import random
import re
import socket
import ssl
import threading
import time
import urllib.parse
from http.client import HTTPConnection, HTTPSConnection
from typing import Any, Dict, List, Optional, Tuple

from app import config

logger = logging.getLogger(__name__)

# Таймаут HTTP-запроса к модели по умолчанию (секунды). Ответ на 2000 токенов
# при ~25 ток/с занимает около 80 с, поэтому прежние 60 с обрезали длинные
# ответы и тяжёлые служебные вызовы агента (план ветвления) — запрос падал по
# таймауту, а клиент молча возвращал пустую строку.
HTTP_TIMEOUT = 120

# Сколько ДОПОЛНИТЕЛЬНЫХ попыток делается при сбое (0 — одна попытка).
RETRY_ATTEMPTS = int(getattr(config, "LLM_RETRIES", 2) or 0)
# База экспоненциальной паузы между попытками (секунды) + случайный разброс.
RETRY_BACKOFF = 0.6
# HTTP-коды, при которых повтор осмыслен (сервер занят/временно недоступен).
RETRY_STATUSES = (408, 409, 425, 429, 500, 502, 503, 504)
# Повторять ли запрос, оборвавшийся по таймауту: провайдер мог уже начать
# генерировать ответ, поэтому такой повтор дороже — одна попытка.
RETRY_ON_TIMEOUT = bool(getattr(config, "LLM_RETRY_TIMEOUT", 0))

# Базовая системная инструкция ассистента.
SYSTEM_PROMPT = (
    "Ты — дружелюбный ассистент на сайте. Отвечай кратко, "
    "по делу и на том же языке, на котором задан вопрос."
)

# Дополнение к системной инструкции для JSON-режима.
SYSTEM_PROMPT_JSON = (
    " Отвечай строго в формате одного валидного JSON-объекта "
    "(без пояснений, markdown-обёрток ```json и лишнего текста)."
)

# Модели, которые НЕ принимают поле thinking (отключение reasoning): они
# отвечают HTTP 400 на этот параметр (см. app/ai/service.py). Список —
# подстроки идентификатора модели.
THINKING_UNSUPPORTED = ("alice",)


def _split_stop_sequences(stop: Optional[str]) -> Optional[list]:
    """Разбирает условие завершения в список stop-последовательностей."""
    if not stop or not stop.strip():
        return None
    sequences = [s.strip() for s in stop.split(",") if s.strip()]
    return sequences or None


def _redact_secrets(text: str, *secrets: str) -> str:
    """Убирает ключи из текста — тело ошибки провайдера может их содержать.

    Провайдер в ответе 401 нередко печатает присланный ключ целиком или его
    хвост («Your api key: ****pD7C is invalid»). Тело ошибки уходит и в лог, и
    в метрики сбоя (`error`), которые показываются в интерфейсе, — ключ не
    должен попадать ни туда, ни туда. Поэтому текст ошибки чистится СРАЗУ, на
    входе в клиент.
    """
    out = str(text or "")
    for secret in secrets:
        if secret and len(secret) >= 8:
            out = out.replace(secret, "***")
    # Заодно любые «sk-…»/«Bearer …» — на случай, если провайдер напечатал не
    # тот ключ, который отправляли мы, а свой/чужой.
    out = re.sub(r"(?i)\b(sk-[A-Za-z0-9_\-]{4,}|Bearer\s+[A-Za-z0-9_\-\.]{6,})",
                 "***", out)
    # И «хвост» ключа, который провайдер печатает сам: «Your api key: ****pD7C
    # is invalid» — четыре последних символа ключа тоже не наше дело.
    return re.sub(r"(?i)(api[\s_-]*key[^:\n]{0,40}:\s*)\S+", r"\1***", out)


def redact_secrets(text: str) -> str:
    """Текст ошибки без ключей — для тех, кто показывает её пользователю.

    Обёртка над `_redact_secrets`: чистить приходится и ВНЕ клиента (например,
    тестовый прогон RAG показывает причину сбоя вызова прямо в чате). Правило
    «что считать секретом» здесь не повторяется — оно одно на проект.
    """
    return _redact_secrets(text)


def _supports_thinking(model: str) -> bool:
    """False, если модель отклоняет поле thinking (тогда его не отправляем)."""
    lowered = str(model or "").lower()
    return not any(marker in lowered for marker in THINKING_UNSUPPORTED)


# ---------------------------------------------------------------------------
# Транспорт: пул keep-alive соединений и прерывание вызова
# ---------------------------------------------------------------------------
class _ConnectionPool:
    """Пул соединений с провайдером: по одному на рабочий поток.

    `http.client` не потокобезопасен, поэтому соединение живёт в thread-local:
    вызовы LLM идут из пула потоков (`asyncio.to_thread`), и каждое соединение
    переиспользуется своим потоком — это и даёт keep-alive без блокировок.
    """

    def __init__(self) -> None:
        self._local = threading.local()

    def connection(self, url: str, timeout: float):
        parts = urllib.parse.urlsplit(url)
        key = (parts.scheme, parts.hostname, parts.port)
        conn = getattr(self._local, "conn", None)
        if conn is not None and getattr(self._local, "key", None) == key:
            conn.timeout = timeout
            return conn
        self.close()
        if parts.scheme == "https":
            conn = HTTPSConnection(
                parts.hostname, parts.port or 443, timeout=timeout,
                context=ssl.create_default_context(),
            )
        else:
            conn = HTTPConnection(parts.hostname, parts.port or 80, timeout=timeout)
        self._local.conn = conn
        self._local.key = key
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001 — закрытие не должно мешать
                pass
        self._local.conn = None
        self._local.key = None


_POOL = _ConnectionPool()


class _AbortBox:
    """Дескриптор вызова: позволяет оборвать HTTP из другого потока.

    Нужен для отмены: `call_llm_async` запускает синхронный запрос в потоке, и
    отменить поток нельзя — но можно закрыть его сокет. Тогда `getresponse()`
    или чтение тела падают сразу, а не через таймаут (120 с), и провайдер
    прекращает генерацию.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._conn = None
        self._aborted = False

    def attach(self, conn) -> None:
        with self._lock:
            self._conn = conn
            aborted = self._aborted
        if aborted:
            _hard_close(conn)

    def detach(self) -> None:
        with self._lock:
            self._conn = None

    def abort(self) -> None:
        with self._lock:
            self._aborted = True
            conn = self._conn
        if conn is not None:
            _hard_close(conn)


def _hard_close(conn) -> None:
    """Обрывает соединение (shutdown + close), чтобы чтение упало сразу."""
    sock = getattr(conn, "sock", None)
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
    try:
        conn.close()
    except Exception:  # noqa: BLE001
        pass


def _read_stream(response) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Читает SSE-поток chat/completions: (текст ответа, usage|None).

    Каждая строка читается с таймаутом соединения: висящий поток обрывается
    раньше, чем «общий» таймаут всего ответа. `usage` приходит последним
    блоком, если запрошен `stream_options.include_usage`.
    """
    parts: List[str] = []
    usage: Optional[Dict[str, Any]] = None
    for raw in response:
        line = raw.decode("utf-8", "replace").strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except ValueError:
            continue
        if isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            piece = delta.get("content")
            if isinstance(piece, str):
                parts.append(piece)
    return "".join(parts), usage


def _post_json(url: str, payload: Dict[str, Any], timeout: float,
               abort: Optional[_AbortBox], api_key: str) -> Tuple[int, str, Optional[Tuple[str, Optional[Dict[str, Any]]]], Dict[str, str]]:
    """Один HTTP-запрос: (статус, тело ошибки, поток|None, заголовки).

    Успешный нестриминговый ответ отдаётся телом (строка), стриминговый —
    парой (содержимое, usage). Тело ошибки читается всегда: без него сбой
    выглядел как «модель не ответила».

    api_key — ключ ТОГО провайдера, к которому идёт запрос (у официального
    DeepSeek и Yandex Cloud ключи разные), поэтому он не берётся из config здесь.
    """
    parts = urllib.parse.urlsplit(url)
    path = parts.path or "/"
    if parts.query:
        path = path + "?" + parts.query
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if payload.get("stream") else "application/json",
        "Connection": "keep-alive",
    }
    conn = _POOL.connection(url, timeout)
    if abort is not None:
        abort.attach(conn)
    try:
        conn.request("POST", path, body=body, headers=headers)
        response = conn.getresponse()
        status = int(response.status)
        head = {k.lower(): v for k, v in response.getheaders()}
        if status >= 400:
            # Тело ошибки чистим от ключей: оно уходит в лог и в метрики сбоя.
            text = _redact_secrets(
                response.read().decode("utf-8", "replace")[:500], api_key)
            return status, text, None, head
        if payload.get("stream"):
            return status, "", _read_stream(response), head
        return status, response.read().decode("utf-8", "replace"), None, head
    finally:
        if abort is not None:
            abort.detach()


def _parse_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _failed_metrics(model: str, elapsed: float, reason: str) -> Dict[str, Any]:
    """Метрики НЕУДАЧНОГО вызова: обращение к модели было, расхода нет.

    Такой замер не портит суммы токенов (они неизвестны), но позволяет
    показать, что вызов состоялся и не удался, — раньше сбой был неотличим от
    «модель ничего не ответила».
    """
    return {
        "model": model,
        "elapsed_seconds": round(elapsed, 3),
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_rub": 0.0,
        "failed": True,
        "error": str(reason)[:200],
    }


def _retry_delay(attempt: int, headers: Dict[str, str]) -> float:
    """Пауза перед повтором: Retry-After провайдера либо экспоненциальный backoff."""
    raw = str(headers.get("retry-after") or "").strip()
    if raw:
        try:
            return max(0.0, min(30.0, float(raw)))
        except ValueError:
            pass
    return RETRY_BACKOFF * (2 ** attempt) * (1 + random.random() * 0.25)


def call_llm(
    user_text: str,
    response_format: str = "free",
    max_tokens: Optional[int] = None,
    stop: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    model: Optional[str] = None,
    disable_thinking: bool = False,
    messages: Optional[list] = None,
    timeout: float = HTTP_TIMEOUT,
    stream: Optional[bool] = None,
    provider: Optional[str] = None,
) -> str:
    """Отправляет запрос к реальной модели и возвращает текст ответа.

    Использует OpenAI-совместимый формат (chat/completions). Если API-ключ
    не задан или возникла ошибка, возвращает пустую строку — тогда вызывающий
    код может использовать демо-ответ.

    Нас интересует только content: reasoning пользователю не показывается.

    max_tokens задаёт лимит длины ответа и уходит провайдеру ровно как есть.
    Когда лимит задан, reasoning принудительно отключается (thinking disabled):
    иначе модель тратит бюджет токенов на «размышления» и content не появляется.
    stop принимает одну или несколько (через запятую) stop-последовательностей.
    temperature — необязательное значение «температуры» модели (0..n), уходит
    провайдеру как есть. Его задание также отключает reasoning.

    model — необязательный идентификатор модели (URI). По умолчанию
    config.LLM_MODEL; поддержка выбора модели используется настройкой
    «Тест моделей». Модели, не поддерживающие thinking (alice), получают запрос
    без этого поля — иначе провайдер отвечает ошибкой 400.

    Для JSON-режима используется только системная инструкция (без нативного
    response_format=json_object), т.к. он заставляет модель дописывать лишний
    служебный вид и сжирает токены — важно для малых max_tokens.

    system_prompt — необязательная полностью своя системная инструкция
    (используется экспертными режимами вместо стандартной). Передача своего
    системного промпта также отключает reasoning, как и задание max_tokens/stop.

    messages — необязательный ПОЛНЫЙ список сообщений [{"role", "content"}, ...]
    (например, system + история диалога + текущий запрос, как у AI-агента).
    Если задан, используется как есть: response_format/system_prompt не влияют
    на построение запроса.

    stream — читать ответ потоком (SSE). По умолчанию — настройка окружения
    LLM_STREAM: поток снимает «общий» таймаут на весь ответ, но требует, чтобы
    провайдер отдавал usage в потоке (иначе расход токенов неизвестен).

    provider — имя провайдера ("deepseek-official" — по умолчанию, "yandex" —
    модели «Теста моделей»); из него берутся адрес и ключ. Не задан — провайдер
    определяется по модели (config.provider_for_model).
    """
    content, _ = _perform_call(
        user_text,
        response_format=response_format,
        max_tokens=max_tokens,
        stop=stop,
        system_prompt=system_prompt,
        temperature=temperature,
        model=model,
        disable_thinking=disable_thinking,
        messages=messages,
        timeout=timeout,
        stream=stream,
        provider=provider,
    )
    return content


def call_llm_with_metrics(
    user_text: str,
    response_format: str = "free",
    max_tokens: Optional[int] = None,
    stop: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    model: Optional[str] = None,
    disable_thinking: bool = False,
    messages: Optional[list] = None,
    timeout: float = HTTP_TIMEOUT,
    stream: Optional[bool] = None,
    provider: Optional[str] = None,
) -> tuple:
    """Как call_llm, но дополнительно возвращает метрики запроса.

    Возвращает (content, metrics|None). metrics — словарь
    {"model", "elapsed_seconds", "prompt_tokens", "completion_tokens",
    "total_tokens"}; равен None, если ключ не задан (запроса не было). Если
    запрос ушёл, но не удался, метрики приходят с пометкой "failed": расход
    неизвестен, но факт обращения к модели виден (см. _failed_metrics).
    Используется настройкой «Тест моделей» для аналитики судьи.

    disable_thinking — принудительно отключает reasoning (thinking: disabled)
    независимо от других параметров. Нужно для честного/быстрого сравнения
    моделей: иначе reasoning-модель (deepseek) тратит время на цепочку
    размышлений и отвечает в разы дольше остальных.

    messages — см. call_llm: полный список сообщений вместо схемы
    [system, user], используется AI-агентом.
    """
    return _perform_call(
        user_text,
        response_format=response_format,
        max_tokens=max_tokens,
        stop=stop,
        system_prompt=system_prompt,
        temperature=temperature,
        model=model,
        disable_thinking=disable_thinking,
        messages=messages,
        timeout=timeout,
        stream=stream,
        provider=provider,
    )


async def call_llm_async(
    user_text: str,
    response_format: str = "free",
    max_tokens: Optional[int] = None,
    stop: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    model: Optional[str] = None,
    disable_thinking: bool = False,
    messages: Optional[list] = None,
    omit_default_max_tokens: bool = False,
    timeout: float = HTTP_TIMEOUT,
    stream: Optional[bool] = None,
    provider: Optional[str] = None,
) -> tuple:
    """Асинхронная версия вызова LLM: (content, metrics|None).

    Синхронный HTTP-запрос выполняется в отдельном потоке (asyncio.to_thread),
    поэтому агента можно использовать из async-кода FastAPI/Flask, не блокируя
    event loop. Параметры — как у call_llm_with_metrics.

    omit_default_max_tokens=True — если max_tokens не задан, параметр вообще
    НЕ отправляется в API (действует предел провайдера), а не подставляется
    лимит приложения config.LLM_MAX_TOKENS. Использует AI-агент.

    timeout — таймаут HTTP-запроса: у длинных служебных вызовов агента
    (план ветвления) он больше, чем у обычного ответа.

    Отмена (клиент закрыл страницу) рвёт соединение: поток не отменить, но его
    сокет — можно, иначе запрос жил бы до таймаута и тарифицировался.
    """
    abort = _AbortBox()
    task = asyncio.ensure_future(asyncio.to_thread(
        _perform_call,
        user_text=user_text,
        response_format=response_format,
        max_tokens=max_tokens,
        stop=stop,
        system_prompt=system_prompt,
        temperature=temperature,
        model=model,
        disable_thinking=disable_thinking,
        messages=messages,
        omit_default_max_tokens=omit_default_max_tokens,
        timeout=timeout,
        stream=stream,
        abort=abort,
        provider=provider,
    ))
    try:
        return await task
    except asyncio.CancelledError:
        abort.abort()
        raise


def _perform_call(
    user_text: str,
    response_format: str = "free",
    max_tokens: Optional[int] = None,
    stop: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    model: Optional[str] = None,
    disable_thinking: bool = False,
    messages: Optional[list] = None,
    omit_default_max_tokens: bool = False,
    timeout: float = HTTP_TIMEOUT,
    stream: Optional[bool] = None,
    abort: Optional[_AbortBox] = None,
    provider: Optional[str] = None,
) -> tuple:
    """Низкоуровневый вызов: возвращает (content, metrics|None).

    messages — необязательный полный список сообщений (AI-агент), иначе
    строится стандартная схема [system, user] из остальных параметров.
    timeout — таймаут HTTP-запроса в секундах (по умолчанию HTTP_TIMEOUT);
    тяжёлые служебные вызовы агента (план ветвления) задают больше.
    Сбой (ошибка провайдера, сеть, битый JSON) возвращает пустой ответ и
    метрики с пометкой "failed"; причина пишется в лог. Повторы при 429/5xx и
    сетевых сбоях — см. RETRY_ATTEMPTS.

    provider — имя провайдера ("deepseek-official" по умолчанию или "yandex");
    из него берутся адрес endpoint и КЛЮЧ (у провайдеров они разные). Не задан
    — провайдер определяется по модели (config.provider_for_model): URI «gpt://…»
    обслуживает Yandex, остальные модели — провайдер по умолчанию.
    """
    spec = config.provider_spec(provider or config.provider_for_model(model))
    api_key = spec["api_key"]
    if not api_key:
        # Ключа нет — обращения к модели не было (метрик тоже нет).
        return "", None

    if messages is None:
        # Стандартная схема запроса: системная инструкция (своя или базовая +
        # дополнение для JSON) и один user-запрос.
        if system_prompt is not None:
            system_prompt_content = system_prompt
        else:
            system_prompt_content = SYSTEM_PROMPT
            if response_format == "json":
                system_prompt_content += SYSTEM_PROMPT_JSON
        # Свой системный промпт считается экспертной инструкцией: reasoning
        # отключаем, чтобы весь бюджет ушёл на сам ответ.
        expert_mode = system_prompt is not None
        payload_messages = [
            {"role": "system", "content": system_prompt_content},
            {"role": "user", "content": user_text},
        ]
    else:
        # Полный список сообщений задан вызывающим кодом (AI-агент сам собрал
        # системный промпт + историю + запрос) — используем как есть.
        expert_mode = False
        payload_messages = messages

    used_model = model or spec["model"] or config.active_model()
    payload: Dict[str, Any] = {"model": used_model, "messages": payload_messages}

    # max_tokens уходит ровно тем значением, которое задал пользователь.
    # Если его нет — либо подставляем лимит приложения, либо (при
    # omit_default_max_tokens) не отправляем параметр вообще: тогда ответ
    # ограничивает только сам провайдер. Значение в max_tokens отправляем
    # ровно как ввёл пользователь.
    if max_tokens:
        payload["max_tokens"] = max_tokens
    elif not omit_default_max_tokens:
        payload["max_tokens"] = config.LLM_MAX_TOKENS

    # Условие завершения — OpenAI принимает его как список stop-последовательностей.
    stop_sequences = _split_stop_sequences(stop)
    if stop_sequences:
        payload["stop"] = stop_sequences

    # Настройка «Температура» легла в параметр temperature API. Если задано —
    # уходит провайдеру ровно как есть (так же, как max_tokens).
    if temperature is not None:
        payload["temperature"] = temperature

    # Если задано ЛЮБОЕ управление генерацией (лимит токенов, стоп, температура
    # или экспертный системный промпт) — отключаем reasoning. Иначе reasoning-модель
    # может потратить бюджет на «размышления» и оставить content пустым.
    # disable_thinking принудительно отключает reasoning (используется в
    # «Тест моделей»), чтобы все модели отвечали сопоставимо и быстро.
    # У модели по умолчанию (провайдер deepseek-official) reasoning выключен
    # ВСЕГДА — thinking: disabled уходит в каждом запросе к ней (config.LLM_DISABLE_THINKING).
    # Моделям, которые поля не принимают (alice), оно не отправляется — иначе 400.
    # Провайдер с thinking="ignore" (локальный сервер MLX) поля НЕ знает вовсе:
    # рассуждения у локальной модели выключаются настройкой её шаблона чата
    # (см. app/ai/local_llm.py), а не полем запроса.
    if (
        spec["thinking"] != "ignore"
        and (
            spec["thinking"] == "disabled"
            or max_tokens or stop_sequences or expert_mode or temperature is not None
            or disable_thinking
        )
        and _supports_thinking(used_model)
    ):
        payload["thinking"] = {"type": "disabled"}

    # Потоковый режим: ответ читается по частям, а usage (если провайдер его
    # отдаёт) приходит последним блоком — см. stream_options.
    use_stream = config.LLM_STREAM if stream is None else bool(stream)
    if use_stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}

    # Адрес endpoint — от ВЫБРАННОГО провайдера (у Yandex и DeepSeek они разные).
    url = str(spec["base_url"]).rstrip("/") + "/chat/completions"
    start = time.perf_counter()
    attempts = RETRY_ATTEMPTS + 1
    timeout_retried = False
    data: Optional[Dict[str, Any]] = None
    streamed: Optional[Tuple[str, Optional[Dict[str, Any]]]] = None

    for attempt in range(attempts):
        status = 0
        error_text = ""
        headers: Dict[str, str] = {}
        try:
            status, error_text, streamed, headers = _post_json(
                url, payload, timeout, abort, api_key)
        except Exception as exc:  # noqa: BLE001 — таймаут, сеть, обрыв соединения
            name = exc.__class__.__name__
            is_timeout = isinstance(exc, (socket.timeout, TimeoutError))
            # Сброшенное соединение в пуле не оставляем: следующий вызов
            # должен поднять новое, а не получить ту же мёртвую трубу.
            _POOL.close()
            retry = attempt + 1 < attempts and (
                not is_timeout or (RETRY_ON_TIMEOUT and not timeout_retried))
            if is_timeout:
                timeout_retried = True
            logger.warning(
                "LLM: %s за %.1f с (timeout=%.0f с, попытка %d из %d)%s: %s",
                name, time.perf_counter() - start, timeout, attempt + 1, attempts,
                " — повторяю" if retry else "", exc,
            )
            if retry:
                time.sleep(_retry_delay(attempt, {}))
                continue
            return "", _failed_metrics(used_model, time.perf_counter() - start, name)

        if status >= 400:
            logger.warning(
                "LLM: HTTP %s за %.1f с (попытка %d из %d): %s",
                status, time.perf_counter() - start, attempt + 1, attempts, error_text,
            )
            retry = status in RETRY_STATUSES and attempt + 1 < attempts
            if retry:
                time.sleep(_retry_delay(attempt, headers))
                continue
            return "", _failed_metrics(
                used_model, time.perf_counter() - start, f"HTTP {status}: {error_text}")

        if streamed is not None:
            content, usage = streamed
            elapsed = time.perf_counter() - start
            if not usage:
                # Провайдер не отдал usage в потоке: расход неизвестен.
                logger.warning(
                    "LLM: потоковый ответ без usage — расход токенов неизвестен "
                    "(нужен stream_options.include_usage)")
                metrics = _failed_metrics(used_model, elapsed, "usage missing in stream")
                metrics["failed"] = False
                metrics["usage_missing"] = True
                metrics["cost_rub"] = 0.0
                return _clean_content(content), metrics
            return _clean_content(content), _usage_metrics(
                used_model, elapsed, usage, provider=spec["provider"])

        data = _parse_json(error_text)
        if data is None:
            logger.warning("LLM: ответ не разобран за %.1f с",
                           time.perf_counter() - start)
            if attempt + 1 < attempts:
                time.sleep(_retry_delay(attempt, {}))
                continue
            return "", _failed_metrics(
                used_model, time.perf_counter() - start, "нечитаемый JSON ответа")
        break

    elapsed = time.perf_counter() - start
    if data is None:
        return "", _failed_metrics(used_model, elapsed, "пустой ответ")

    try:
        msg = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        logger.warning("LLM: неожиданная структура ответа за %.1f с", elapsed)
        return "", _failed_metrics(used_model, elapsed, "неожиданная структура ответа")

    return _clean_content(msg.get("content")), _usage_metrics(
        used_model, elapsed, data.get("usage") or {}, provider=spec["provider"])


def _clean_content(content: Any) -> str:
    """Ответ модели без внешних пробелов; не-строка и пустое — пустая строка."""
    if not (isinstance(content, str) and content.strip()):
        return ""
    return content.strip()


def _cache_hit_tokens(usage: Dict[str, Any]) -> int:
    """Сколько токенов входа провайдер взял ИЗ КЭША (0 — не сообщил).

    Официальный DeepSeek отдаёт это двумя способами: полем
    `prompt_cache_hit_tokens` и вложенным `prompt_tokens_details.cached_tokens`.
    Значение важно для стоимости: вход из кэша стоит в разы дешевле, чем вход
    мимо кэша.
    """
    details = usage.get("prompt_tokens_details")
    cached = details.get("cached_tokens") if isinstance(details, dict) else 0
    hit = usage.get("prompt_cache_hit_tokens")
    if hit is None:
        hit = cached
    try:
        return max(0, int(hit or 0))
    except (TypeError, ValueError):
        return 0


def _usage_metrics(model: str, elapsed: float, usage: Dict[str, Any],
                   provider: Optional[str] = None) -> Dict[str, Any]:
    """Метрики успешного вызова: время, токены (промпт, ответ, сумма), стоимость.

    Стоимость — ОЦЕНКА по тарифу ПРОВАЙДЕРА (config.usage_cost): она нужна
    панели токенов и таблице аналитики, чтобы расход был виден в рублях, а не
    только в токенах. У официального DeepSeek тариф зависит от пиковых часов и
    от того, какая часть входа пришла из кэша, поэтому кэш-токены попадают в
    метрики отдельными полями (cache_hit_tokens / cache_miss_tokens) — по ним
    видно, почему стоимость именно такая.
    """
    prompt = int(usage.get("prompt_tokens", 0) or 0)
    completion = int(usage.get("completion_tokens", 0) or 0)
    hit = min(_cache_hit_tokens(usage), prompt)
    cost = config.usage_cost(
        model, prompt_tokens=prompt, completion_tokens=completion,
        cache_hit_tokens=hit, provider=provider,
    )
    return {
        "model": model,
        "elapsed_seconds": round(elapsed, 3),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
        "cache_hit_tokens": hit,
        "cache_miss_tokens": prompt - hit,
        "cost_rub": cost,
    }
