"""Клиент реальной LLM.

Отправляет запросы к OpenAI-совместимому endпоинту (Yandex Cloud).
Не содержит маршрутов и демо-логики — только HTTP-вызов модели.
"""

import json
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

    Для JSON-режима используется только системная инструкция (без нативного
    response_format=json_object), т.к. он заставляет модель дописывать лишний
    служебный вид и сжирает токены — важно для малых max_tokens.

    system_prompt — необязательная полностью своя системная инструкция
    (используется экспертными режимами вместо стандартной). Передача своего
    системного промпта также отключает reasoning, как и задание max_tokens/stop.
    """
    if not config.LLM_API_KEY:
        return ""

    if system_prompt is not None:
        system_prompt_content = system_prompt
    else:
        system_prompt_content = SYSTEM_PROMPT
        if response_format == "json":
            system_prompt_content += SYSTEM_PROMPT_JSON
    # Свой системный промпт считается экспертной инструкцией: reasoning
    # отключаем, чтобы весь бюджет ушёл на сам ответ.
    expert_mode = system_prompt is not None

    payload = {
        "model": config.LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt_content},
            {"role": "user", "content": user_text},
        ],
        "max_tokens": max_tokens or config.LLM_MAX_TOKENS,
    }

    # Если задан лимит токенов — отключаем reasoning, чтобы весь бюджет
    # ушёл именно на ответ, а не на «размышления». Значение в max_tokens
    # отправляем ровно как ввёл пользователь.
    if max_tokens:
        payload["max_tokens"] = max_tokens

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
    if max_tokens or stop_sequences or expert_mode or temperature is not None:
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

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
        msg = data["choices"][0]["message"]

        # Возвращаем только content; reasoning пользователю не показываем.
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        return ""
    except Exception:
        return ""