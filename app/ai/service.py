"""Сервисный слой генерации ответа.

Оркестрирует выбор источника ответа: сначала пробует реальную LLM, при её
недоступности откатывается на демо-режим. Маршруты и HTTP-слой не знают о
том, какая модель сработает.
"""

import json
from typing import Optional

from app import config
from app.ai import client, demo
from app.ai.json_utils import is_valid_json, repair_json, wrap_as_json


def generate_response(
    user_text: str,
    response_format: str = "free",
    max_tokens: Optional[int] = None,
    stop: Optional[str] = None,
) -> str:
    """Возвращает ответ бота на заданный текст.

    Сначала пытается получить ответ от реальной LLM. Если API-ключ не задан —
    отвечает через демо-правила.

    response_format="json" всегда возвращает валидный JSON: если LLM обрезала
    ответ из-за малого max_tokens, обрезанный фрагмент чинится (закрываются
    скобки/строки), а при невозможности — оборачивается в валидный {"reply": ...}.
    max_tokens ограничивает длину ответа LLM; None — значение по умолчанию.
    stop задаёт stop-последовательности завершения генерации.
    """
    answer = client.call_llm(
        user_text,
        response_format=response_format,
        max_tokens=max_tokens,
        stop=stop,
    )

    # JSON-режим: никогда не показываем «ошибку» вместо ответа. Если ответ не
    # парсится (обрезан лимитом), дочиняем или оборачиваем в валидный JSON.
    if response_format == "json":
        if answer:
            if not is_valid_json(answer):
                repaired = repair_json(answer)
                answer = repaired if repaired is not None else wrap_as_json(answer)
            return answer

        # Ответ пуст — различаем офлайн-режим и реальный сбой.
        if not config.LLM_API_KEY:
            return json.dumps(
                {"reply": demo.demo_ai(user_text)}, ensure_ascii=False
            )
        return wrap_as_json(
            "Ответ не влез в заданный лимит токенов — попробуйте увеличить «Длину»."
        )

    # Свободный режим.
    if answer:
        return answer
    if not config.LLM_API_KEY:
        return demo.demo_ai(user_text)
    return "Извините, не удалось получить ответ от модели. Попробуйте ещё раз."