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
    expert_mode: bool = False,
    expert_mode_type: str = "direct",
    expert_roles: Optional[list] = None,
) -> tuple:
    """Возвращает кортеж (ответ, вердикт).

    Вердикт (correct) — True/False, если модель сама оценила свой ответ в
    экспертном режиме; None — вердикт не определялся (обычный режим) или
    не удалось его получить. В обычном режиме используется только первый
    элемент кортежа.

    Сначала пытается получить ответ от реальной LLM. Если API-ключ не задан —
    отвечает через демо-правила.

    response_format="json" всегда возвращает валидный JSON: если LLM обрезала
    ответ из-за малого max_tokens, обрезанный фрагмент чинится (закрываются
    скобки/строки), а при невозможности — оборачивается в валидный {"reply": ...}.
    max_tokens ограничивает длину ответа LLM; None — значение по умолчанию.
    stop задаёт stop-последовательности завершения генерации.

    expert_mode включает один из экспертных режимов (expert_mode_type:
    direct/stepwise/prompt/group). В экспертном режиме format/max_tokens/stop
    не учитываются — используется только собственная системная инструкция.
    """
    if expert_mode:
        return _expert_response(user_text, expert_mode_type, expert_roles)

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
            return answer, None

        # Ответ пуст — различаем офлайн-режим и реальный сбой.
        if not config.LLM_API_KEY:
            return json.dumps(
                {"reply": demo.demo_ai(user_text)}, ensure_ascii=False
            ), None
        return wrap_as_json(
            "Ответ не влез в заданный лимит токенов — попробуйте увеличить «Длину»."
        ), None

    # Свободный режим.
    if answer:
        return answer, None
    if not config.LLM_API_KEY:
        return demo.demo_ai(user_text), None
    return "Извините, не удалось получить ответ от модели. Попробуйте ещё раз.", None


def _build_expert_system_prompt(mode: str, roles: list) -> str:
    """Собирает системную инструкцию для выбранного экспертного режима."""
    if mode == "direct":
        return (
            "Ты — строгий ИИ-ассистент. Не размышляй, не давай рекомендаций и "
            "никаких дополнительных пояснений. Отвечай на вопрос максимально сухо, "
            "по существу и сразу. Если уместен однозначный ответ — приведи его в "
            "формате «Ответ: …»."
        )
    if mode == "stepwise":
        return (
            "Ты — ИИ-ассистент, который решает задачи по шагам. Опиши ход решения "
            "в виде нумерованного списка шагов (1., 2., 3., …). В самом конце "
            "обязательно приведи итоговый ответ в формате «Ответ: …»."
        )
    if mode == "prompt":
        return (
            "Ты — ИИ-ассистент, умеющий составлять грамотные промпты для других "
            "ИИ-моделей. Сначала составь максимально полный и корректный промпт, "
            "который можно отправить другой модели, и выведи его в чат в формате "
            "«Промпт: …». Затем возьми этот промпт и ответь на него сам, как будто "
            "ты и есть та модель. Итоговый ответ выведи в формате «Ответ: …»."
        )
    if mode == "group":
        listed = "\n".join(f"- {r}" for r in roles if r and r.strip())
        return (
            "Ты — ИИ-ассистент, организующий работу экспертной группы. В твоей "
            "команде следующие эксперты:\n" + listed + "\n"
            "Каждый эксперт обязан предоставить собственное решение полученного "
            "вопроса/задачи. Выведи решения всех экспертов подряд, указывая перед "
            "каждым реальное наименование роли в формате «Роль 1: …», «Роль 2: …» "
            "и т.д."
        )
    # Неизвестный режим — консервативный прямой ответ.
    return (
        "Ты — строгий ИИ-ассистент. Отвечай по существу и без лишнего текста, "
        "в конце приведи ответ в формате «Ответ: …»."
    )


def _judge_correctness(user_text: str, answer: str, mode: str) -> Optional[bool]:
    """Просит модель оценить свой ответ как верный/неверный.

    Оценку даёт сама модель «как в свободном режиме». Для группы экспертов
    réponse считается верным только при полном консенсусе состава в одном
    правильном ответе. Возвращает True/False либо None, если вердикт получить
    не удалось.
    """
    if mode == "group":
        system = (
            "Ты — объективный судья. Оцени результат работы экспертной группы. "
            "Результат верен ТОЛЬКО если все эксперты пришли к одному и тому же "
            "правильному ответу. Если ответы экспертов расходятся или допущена "
            "ошибка — результат неверен. Ответь строго одним словом: ВЕРНО или НЕВЕРНО."
        )
    else:
        system = (
            "Ты — объективный судья, отвечай как если бы сам решал эту задачу в "
            "свободном режиме. Определи, верно ли задача решена в предложенном "
            "ответе ИИ. Ответь строго одним словом: ВЕРНО или НЕВЕРНО."
        )
    user = (
        f"Вопрос/задача: {user_text}\n\n"
        f"Ответ ИИ:\n{answer}\n\n"
        "Верно или неверно решена задача? Ответь одним словом: ВЕРНО или НЕВЕРНО."
    )
    verdict = client.call_llm(user, system_prompt=system)
    if not verdict:
        return None
    v = verdict.strip().upper()
    if "НЕВЕРНО" in v:
        return False
    if "ВЕРНО" in v:
        return True
    return None


def _expert_response(
    user_text: str, mode: str, roles: Optional[list]
) -> tuple:
    """Генерирует ответ в одном из экспертных режимов + вердикт о верности."""
    roles = [r.strip() for r in (roles or []) if r and r.strip()]

    # Группа экспертов требует хотя бы одной заполненной роли.
    if mode == "group" and not roles:
        return "Список ролей пуст.", None

    system_prompt = _build_expert_system_prompt(mode, roles)
    answer = client.call_llm(user_text, system_prompt=system_prompt)
    if answer:
        correct = _judge_correctness(user_text, answer, mode)
        return answer, correct

    # Фолбэк (нет API-ключа / сбой): демо-правила, без вердикта.
    if not config.LLM_API_KEY:
        return demo.demo_ai(user_text), None
    return (
        "Извините, не удалось получить ответ от модели. Попробуйте ещё раз.",
        None,
    )