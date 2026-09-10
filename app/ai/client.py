"""Клиент реальной LLM.

Отправляет запросы к OpenAI-совместимому endпоинту (Yandex Cloud).
Не содержит маршрутов и демо-логики — только HTTP-вызов модели.
"""

import asyncio
import json
import time
import urllib.request
from typing import Optional

from app import config

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


def _split_stop_sequences(stop: Optional[str]) -> Optional[list]:
    """Разбирает условие завершения в список stop-последовательностей."""
    if not stop or not stop.strip():
        return None
    sequences = [s.strip() for s in stop.split(",") if s.strip()]
    return sequences or None


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
    «Тест моделей».

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
) -> tuple:
    """Как call_llm, но дополнительно возвращает метрики запроса.

    Возвращает (content, metrics|None). metrics — словарь
    {"model", "elapsed_seconds", "prompt_tokens", "completion_tokens",
    "total_tokens"}; равен None, если ключ не задан или запрос не выполнен
    (тогда content='' — вызывающий использует фолбэк). Используется настройкой
    «Тест моделей» для аналитики судьи.

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
) -> tuple:
    """Асинхронная версия вызова LLM: (content, metrics|None).

    Синхронный HTTP-запрос выполняется в отдельном потоке (asyncio.to_thread),
    поэтому агента можно использовать из async-кода FastAPI/Flask, не блокируя
    event loop. Параметры — как у call_llm_with_metrics.

    omit_default_max_tokens=True — если max_tokens не задан, параметр вообще
    НЕ отправляется в API (действует предел провайдера), а не подставляется
    лимит приложения config.LLM_MAX_TOKENS. Использует AI-агент.
    """
    return await asyncio.to_thread(
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
    )


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
) -> tuple:
    """Низкоуровневый вызов: возвращает (content, metrics|None).

    messages — необязательный полный список сообщений (AI-агент), иначе
    строится стандартная схема [system, user] из остальных параметров.
    """
    if not config.LLM_API_KEY:
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

    used_model = model or config.LLM_MODEL
    payload = {"model": used_model, "messages": payload_messages}

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
    if (
        max_tokens or stop_sequences or expert_mode or temperature is not None
        or disable_thinking
    ):
        payload["thinking"] = {"type": "disabled"}

    url = config.LLM_BASE_URL.rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.LLM_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception:
        return "", None
    elapsed = time.perf_counter() - start

    try:
        msg = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return "", None

    content = msg.get("content")
    if not (isinstance(content, str) and content.strip()):
        content = ""
    else:
        content = content.strip()

    usage = data.get("usage") or {}
    metrics = {
        "model": used_model,
        "elapsed_seconds": round(elapsed, 3),
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
    }
    return content, metrics