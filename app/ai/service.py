"""Сервисный слой генерации ответа.

Оркестрирует выбор источника ответа: сначала пробует реальную LLM, при её
недоступности откатывается на демо-режим. Маршруты и HTTP-слой не знают о
том, какая модель сработает.
"""

import json
import re
from typing import Optional

from app import config
from app.ai import client, demo
from app.ai.json_utils import is_valid_json, repair_json, wrap_as_json

# Формулировка «верный ответ: …» в тексте пользователя. Она НЕ должна попадать
# в обычные запросы LLM — нужна только судье-аналитику для оценки точности.
CORRECT_ANSWER_RE = re.compile(r"верный\s+ответ\s*:\s*(.+)$", re.IGNORECASE)

# Модели для настройки «Тест моделей»: ключ настройки -> URI модели.
# deepseek — LLM по умолчанию (config.LLM_MODEL).
MODEL_URIS = {
    "deepseek": None,  # подставляется config.LLM_MODEL
    "alice": "gpt://b1gkm5u908if6dc0focb/aliceai-llm/latest",
    "alice-flash": "gpt://b1gkm5u908if6dc0focb/aliceai-llm-flash/latest",
}
# Человекочитаемые названия моделей (для вывода в ответах).
MODEL_NAMES = {
    "deepseek": "DeepSeek 4 Flash",
    "alice": "Alice AI LLM",
    "alice-flash": "Alice AI LLM Flash",
}
# Цена за 1000 токенов (вход/выход), руб., по тарифам Yandex AI Studio.
# Используется для расчёта стоимости в аналитике судьи.
MODEL_PRICING = {
    "deepseek": {"input": 0.3, "output": 0.5},
    "alice": {"input": 0.5, "output": 1.2},
    "alice-flash": {"input": 0.1, "output": 0.2},
}
# Какие модели поддерживают поле thinking (отключение reasoning).
# DeepSeek — reasoning-модель, поддерживает "thinking": {"type": "disabled"}.
# Alice-модели это поле НЕ принимают (HTTP 400), потому им его не отправляем.
MODEL_SUPPORTS_THINKING = {
    "deepseek": True,
    "alice": False,
    "alice-flash": False,
}


def generate_response(
    user_text: str,
    response_format: str = "free",
    max_tokens: Optional[int] = None,
    stop: Optional[str] = None,
    expert_mode: bool = False,
    expert_mode_type: str = "direct",
    expert_roles: Optional[list] = None,
    temperatures: Optional[list] = None,
    models: Optional[list] = None,
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

    temperatures — непустой список чисел означает настройку «Температура»:
    выполняется отдельный запрос к LLM для каждого значения, и возвращается
    список ответов {"temperature", "text"} как первый элемент кортежа.
    Пустой/None — настройка не используется, отдаётся один ответ.
    """
    if expert_mode:
        return _expert_response(user_text, expert_mode_type, expert_roles)

    # Настройка «Температура»: сколько заполненных значений — столько отдельных
    # ответов. После вывода всех ответов выполняется дополнительный запрос
    # судьи-аналитика. Возвращается словарь {"responses", "judge"} как первый
    # элемент кортежа — маршрут передаёт его клиенту.
    if temperatures:
        # Фраза «верный ответ: …» вырезается из текста для обычных запросов,
        # но передаётся судье для оценки точности.
        prompt_text, correct_answer = _extract_correct_answer(user_text)
        responses = _temperature_responses(
            prompt_text, temperatures, response_format, max_tokens, stop
        )
        judge = _judge_analyst(user_text, correct_answer, responses) if responses else None
        return {"responses": responses, "judge": judge}, None

    # Настройка «Тест моделей»: запрос отправляется в каждую выбранную модель.
    # Возвращается словарь {"model_responses": [...]} как первый элемент кортежа.
    if models:
        return _model_responses(
            user_text, models, response_format, max_tokens, stop
        ), None

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


def _extract_correct_answer(user_text: str) -> tuple:
    """Выделяет формулировку «верный ответ: …» из текста пользователя.

    Возвращает (текст_без_фразы, верный_ответ_или_None). Фраза не должна
    попадать в обычные запросы LLM — только судье-аналитику. Если весь текст
    состоит из этой фразы, очищенный текст оставляем исходным (чтобы не уйти
    в пустоту).
    """
    match = CORRECT_ANSWER_RE.search(user_text)
    if not match:
        return user_text, None
    cleaned = CORRECT_ANSWER_RE.sub("", user_text).strip()
    if not cleaned:
        cleaned = user_text
    answer = match.group(1).strip()
    return cleaned, answer or None


def _judge_analyst(
    user_text: str, correct_answer: Optional[str], responses: list
):
    """Запрашивает у модели резюме-аналитику по готовым ответам.

    Судья оценивает КАЖДЫЙ ответ отдельно по шкале 1–10 по пяти параметрам:
    точность, креативность, вариативность словарного запаса, лаконичность,
    оптимальность затраченных токенов. Параметр «точность» оценивается против
    известного верного ответа (если он был в запросе юзера).

    Возвращает список строк таблицы
    [{"temperature", "accuracy", "creativity", "vocabulary", "conciseness",
      "tokens", "summary"}, …] — по одному на каждую температуру. Если ответ
    модели не распарсился как JSON — возвращает сырой текст; если ответ пуст —
    None.
    """
    system = (
        "Ты — строгий судья-аналитик. Перед тобой вопрос пользователя и ответы "
        "ИИ, сгенерированные при разных значениях температуры (temperature). "
        "Оцени КАЖДЫЙ ответ отдельно по пяти параметрам по шкале от 1 до 10:\n"
        "- accuracy — точность (если указан верный ответ — по соответствию ему);\n"
        "- creativity — креативность;\n"
        "- vocabulary — вариативность словарного запаса;\n"
        "- conciseness — лаконичность;\n"
        "- tokens — оптимальность затраченных токенов.\n"
        "Для каждого ответа добавь поле summary — КРАТКОЕ резюме (1–2 предложения).\n"
        "Отвечай строго одним валидным JSON — массивом объектов, по одному на "
        "каждый ответ:\n"
        '[{"temperature": <число>, "accuracy": <1-10>, "creativity": <1-10>, '
        '"vocabulary": <1-10>, "conciseness": <1-10>, "tokens": <1-10>, '
        '"summary": "…"}]\n'
        "Без пояснений, без markdown-обёрток ```json и без лишнего текста."
    )
    parts = [f"Вопрос пользователя:\n{user_text}"]
    if correct_answer:
        parts.append(f"Верный ответ (эталон): {correct_answer}")
    parts.append(
        "Ответы ИИ:\n"
        + "\n\n".join(
            f"Ответ {i + 1} (temperature={r['temperature']}):\n{r['text']}"
            for i, r in enumerate(responses)
        )
    )
    verdict = client.call_llm("\n\n".join(parts), system_prompt=system)
    if not verdict:
        return None
    parsed = _parse_judge_json(verdict)
    return parsed if parsed is not None else verdict.strip()


def _parse_judge_json(text: str):
    """Разбирает JSON-ответ судьи в список строк таблицы (или None).

    Толерантно к markdown-обёрткам. Значения оценок ограничиваются диапазоном
    1–10. Если разобрать не удалось — возвращает None.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z\s]*\n?|\n?```$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
    except Exception:
        return None
    if not isinstance(data, list):
        return None

    def clamp(v):
        try:
            n = int(round(float(v)))
        except (TypeError, ValueError):
            return None
        return max(1, min(10, n))

    rows = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            temperature = round(float(item.get("temperature")), 6)
        except (TypeError, ValueError):
            continue
        ratings = {}
        ok = True
        for key in ("accuracy", "creativity", "vocabulary", "conciseness", "tokens"):
            value = clamp(item.get(key))
            if value is None:
                ok = False
                break
            ratings[key] = value
        if not ok:
            continue
        ratings["temperature"] = temperature
        ratings["summary"] = str(item.get("summary") or "").strip()
        rows.append(ratings)
    return rows or None


def _judge_model_test(answers: list, analytics: list) -> list:
    """Краткое резюме судьи по каждой модели настройки «Тест моделей».

    По ответу и метрикам каждой модели (время, вход/выход токены, стоимость)
    модель формирует резюме: скорость, стоимость/ресурсоёмкость и качество
    ответа. Возвращает список [{"model", "summary"}, …]; пустой список, если
    резюме получить не удалось.
    """
    system = (
        "Ты — строгий судья-аналитик. Перед тобой ответы нескольких моделей на "
        "один вопрос и их метрики: время обработки (секунды), входные и выходные "
        "токены, стоимость в рублях. Для КАЖДОЙ модели дай КРАТКОЕ резюме "
        "(2–3 предложения), оценив скорость, стоимость/ресурсоёмкость и качество "
        "ответа. Верни строго один валидный JSON — массив объектов, по одному на "
        'каждую модель, в формате: [{"model": "<название>", "summary": "…"}]. '
        "Без пояснений и без markdown-обёрток ```json."
    )
    blocks = []
    for answer, row in zip(answers, analytics):
        blocks.append(
            f"Модель {row['model']} ({row['seconds']:.2f}с, вход {row['input_tokens']} / "
            f"выход {row['output_tokens']} токенов, стоимость {row['cost_rub']:.2f} руб.):\n"
            f"{answer['text']}"
        )
    verdict = client.call_llm(
        "Ответы и метрики моделей:\n\n" + "\n\n".join(blocks), system_prompt=system
    )
    if not verdict:
        return []
    return _parse_model_summaries(verdict)


def _parse_model_summaries(text: str) -> list:
    """Разбирает JSON-ответ судьи по «Тесту моделей» в [{"model", "summary"}, …].

    Толерантен к markdown-обёрткам; невалидные/пустые записи отбрасываются.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z\s]*\n?|\n?```$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        model = str(item.get("model") or "").strip()
        summary = str(item.get("summary") or "").strip()
        if model and summary:
            out.append({"model": model, "summary": summary})
    return out


def _model_responses(
    user_text: str,
    models: list,
    response_format: str,
    max_tokens: Optional[int],
    stop: Optional[str],
) -> dict:
    """Отправляет запрос в каждую выбранную модель («Тест моделей»).

    Возвращает
    {"model_responses": [{"model", "text"}, …],
     "analytics": [{"model", "seconds", "input_tokens", "output_tokens",
                    "cost_rub"}, …]}.
    Каждая модель обрабатывается независимо; заодно собираются метрики
    (время, входной/выходной токены) и считается стоимость по тарифам
    MODEL_PRICING — это аналитика судьи-аналитика для вывода таблицей.
    """
    answers = []
    analytics = []
    for key in models:
        uri = MODEL_URIS.get(key)
        if uri is None:
            uri = config.LLM_MODEL
        name = MODEL_NAMES.get(key, key)
        text, metrics = client.call_llm_with_metrics(
            user_text,
            response_format=response_format,
            max_tokens=max_tokens,
            stop=stop,
            model=uri,
            # Отключаем reasoning только у моделей, которые это поддерживают
            # (deepseek) — остальные (alice) отклоняют поле thinking (HTTP 400).
            disable_thinking=MODEL_SUPPORTS_THINKING.get(key, False),
        )
        answers.append({"model": name, "text": _normalize_answer(user_text, text, response_format)})

        if metrics:
            in_tokens = metrics["prompt_tokens"]
            out_tokens = metrics["completion_tokens"]
            price = MODEL_PRICING.get(key) or {"input": 0, "output": 0}
            cost = (in_tokens / 1000) * price["input"] + (out_tokens / 1000) * price["output"]
            analytics.append(
                {
                    "model": name,
                    "seconds": metrics["elapsed_seconds"],
                    "input_tokens": in_tokens,
                    "output_tokens": out_tokens,
                    "cost_rub": round(cost, 2),
                }
            )
        else:
            analytics.append(
                {
                    "model": name,
                    "seconds": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_rub": 0,
                }
            )
    # Резюме судьи-аналитика: краткий разбор каждой модели (скорость, стоимость,
    # ресурсоёмкость, качество). Прикрепляем summary к строкам таблицы.
    if answers:
        summaries = _judge_model_test(answers, analytics)
        by_model = {s["model"]: s["summary"] for s in summaries}
        for row in analytics:
            row["summary"] = by_model.get(row["model"], "")
    return {"model_responses": answers, "analytics": analytics}


def _normalize_answer(user_text: str, answer: str, response_format: str) -> str:
    """Приводит сырой ответ LLM к финальному виду (JSON-дочистка / фолбэк).

    Та же логика, что в обычном (неэкспертном) пути generate_response, но
    изолирована, чтобы каждый ответ при нескольких «температурах» обрабатывался
    одинаково, не трогая основной путь.
    """
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


def _temperature_responses(
    user_text: str,
    temperatures: list,
    response_format: str,
    max_tokens: Optional[int],
    stop: Optional[str],
) -> list:
    """Выполняет отдельный запрос к LLM для каждого значения «Температуры».

    Каждое заполненное поле = отдельный запрос с temperature=<значение>.
    Возвращает список словарей {"temperature": t, "text": ответ} — ровно
    столько ответов, сколько заполненных полей (минимум один).
    """
    results = []
    for raw in temperatures:
        try:
            t = float(raw)
        except (TypeError, ValueError):
            continue
        answer = client.call_llm(
            user_text,
            response_format=response_format,
            max_tokens=max_tokens,
            stop=stop,
            temperature=t,
        )
        results.append(
            {"temperature": t, "text": _normalize_answer(user_text, answer, response_format)}
        )
    # Если после фильтра значений не осталось — ведём себя как обычный режим.
    if not results:
        answer = client.call_llm(
            user_text,
            response_format=response_format,
            max_tokens=max_tokens,
            stop=stop,
        )
        return [
            {
                "temperature": None,
                "text": _normalize_answer(user_text, answer, response_format),
            }
        ]
    return results


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