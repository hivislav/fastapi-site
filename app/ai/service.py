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

# Модели настройки «Тест моделей» (значения галочек в #param-models): у каждой
# свой ПРОВАЙДЕР, идентификатор модели и человекочитаемое имя.
#
# "deepseek" — модель по умолчанию: провайдер deepseek-official
# (agent-default-model), модель провайдера берётся из config.LLM_MODEL.
# Остальные ключи — СТАРЫЕ модели (Yandex Cloud AI Studio): они остаются
# рабочими, но запрос уходит в них только при РУЧНОМ выборе в этой настройке:
# обычные запросы, «Температура», судья и AI-агент идут в модель по умолчанию.
DEFAULT_MODEL_KEY = "deepseek"
MODEL_SPECS = {
    "deepseek": {
        "provider": "deepseek-official",
        "model": None,  # None — модель провайдера по умолчанию (config.LLM_MODEL)
        "name": "DeepSeek-V4-Flash",
        "supports_thinking": True,
    },
    "deepseek-yandex": {
        "provider": "yandex",
        "model": config.YANDEX_MODEL,  # URI старого провайдера (был моделью по умолчанию)
        "name": "DeepSeek 4 Flash (Yandex)",
        "supports_thinking": True,
    },
    "alice": {
        "provider": "yandex",
        "model": "gpt://b1gkm5u908if6dc0focb/aliceai-llm/latest",
        "name": "Alice AI LLM",
        "supports_thinking": False,
    },
    "alice-flash": {
        "provider": "yandex",
        "model": "gpt://b1gkm5u908if6dc0focb/aliceai-llm-flash/latest",
        "name": "Alice AI LLM Flash",
        "supports_thinking": False,
    },
}
# Человекочитаемые названия моделей (для вывода в ответах).
MODEL_NAMES = {key: spec["name"] for key, spec in MODEL_SPECS.items()}
# Тарифы: у модели по умолчанию (официальный DeepSeek) — долларовый прайс с
# поправкой на пиковые часы и кэш; у старых моделей — рублёвый тариф Yandex
# (config.YANDEX_MODEL_PRICING). Стоимость считает ровно ОДНА функция —
# config.usage_cost — а тариф для показа даёт config.pricing_info: своих цен
# этот модуль не считает, иначе числа в таблице и в панели разошлись бы.
MODEL_PRICING = config.YANDEX_MODEL_PRICING
# Какие модели ПРИНИМАЮТ поле thinking (отключение reasoning).
# DeepSeek — reasoning-модель, поддерживает "thinking": {"type": "disabled"}.
# Alice-модели это поле НЕ принимают (HTTP 400), потому им его не отправляем.
MODEL_SUPPORTS_THINKING = {
    key: spec["supports_thinking"] for key, spec in MODEL_SPECS.items()
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
    """Возвращает кортеж (ответ, вердикт, аналитика).

    Вердикт (correct) — True/False, если модель сама оценила свой ответ в
    экспертном режиме; None — вердикт не определялся (обычный режим) или
    не удалось его получить.

    Аналитика — список строк таблицы метрик по КАЖДОМУ обращению к модели в
    этом запросе: время, токены, стоимость (включая служебный вызов судьи,
    который оценивает верность ответа). Раньше метрики выбрасывались, и на
    странице статистики не было видно ни расхода, ни цены запроса.

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
        responses, analytics = _temperature_responses(
            prompt_text, temperatures, response_format, max_tokens, stop
        )
        judge, judge_row = (None, None)
        if responses:
            judge, judge_row = _judge_analyst(user_text, correct_answer, responses)
        if judge_row:
            analytics.append(judge_row)
        return {"responses": responses, "analytics": analytics,
                "judge": judge}, None, analytics

    # Настройка «Тест моделей»: запрос отправляется в каждую выбранную модель.
    # Возвращается словарь {"model_responses": [...]} как первый элемент кортежа.
    if models:
        payload = _model_responses(user_text, models, response_format, max_tokens, stop)
        return payload, None, payload.get("analytics") or []

    answer, metrics = client.call_llm_with_metrics(
        user_text,
        response_format=response_format,
        max_tokens=max_tokens,
        stop=stop,
    )
    analytics = [_analytics_row("ответ", config.LLM_MODEL, metrics)]

    # JSON-режим: никогда не показываем «ошибку» вместо ответа. Если ответ не
    # парсится (обрезан лимитом), дочиняем или оборачиваем в валидный JSON.
    if response_format == "json":
        if answer:
            if not is_valid_json(answer):
                repaired = repair_json(answer)
                answer = repaired if repaired is not None else wrap_as_json(answer)
            return answer, None, analytics

        # Ответ пуст — различаем офлайн-режим и реальный сбой.
        if not config.LLM_API_KEY:
            return json.dumps(
                {"reply": demo.demo_ai(user_text)}, ensure_ascii=False
            ), None, analytics
        return wrap_as_json(
            "Ответ не влез в заданный лимит токенов — попробуйте увеличить «Длину»."
        ), None, analytics

    # Свободный режим.
    if answer:
        return answer, None, analytics
    if not config.LLM_API_KEY:
        return demo.demo_ai(user_text), None, analytics
    return "Извините, не удалось получить ответ от модели. Попробуйте ещё раз.", None, analytics


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


def _fmt_rub(value) -> str:
    """Рубли с точностью по величине: мелкие суммы не превращаются в «0.00».

    Официальный DeepSeek берёт за вызов тысячные доли рубля, поэтому для сумм
    меньше копейки показываем четыре знака — иначе расход выглядел бы нулевым.
    """
    amount = float(value or 0.0)
    return f"{amount:.2f}" if abs(amount) >= 0.01 else f"{amount:.4f}"


def _analytics_row(label: str, model: str, metrics: Optional[dict],
                   provider: Optional[str] = None) -> dict:
    """Строка таблицы метрик: время, токены и стоимость одного вызова LLM.

    Аналитика собирается для КАЖДОГО обращения к модели (включая служебные —
    судью), поэтому в таблице видно не только ответы, но и цену их оценки.

    Стоимость НЕ считается здесь заново: её уже посчитал клиент по тарифу
    провайдера (включая кэш-токены, пиковые часы и курс) и положил в метрики —
    второй расчёт тех же денег неизбежно разошёлся бы с первым. Пересчёт
    остаётся только страховкой для метрик без стоимости.

    В строку добавляется `pricing` — тариф, по которому посчитана эта цифра
    (пиковый/непиковый, ставки, курс): интерфейс показывает его под таблицей,
    чтобы стоимость была объяснимой.
    """
    if not metrics or metrics.get("failed"):
        return {"label": label, "seconds": 0, "input_tokens": 0,
                "output_tokens": 0, "cost_rub": 0, "summary": "",
                "pricing": config.pricing_info(model, provider)}
    in_tokens = int(metrics.get("prompt_tokens") or 0)
    out_tokens = int(metrics.get("completion_tokens") or 0)
    cost = metrics.get("cost_rub")
    if cost is None:
        cost = config.usage_cost(
            model, prompt_tokens=in_tokens, completion_tokens=out_tokens,
            cache_hit_tokens=metrics.get("cache_hit_tokens") or 0,
            provider=provider,
        )
    return {
        "label": label,
        "seconds": float(metrics.get("elapsed_seconds") or 0),
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "cost_rub": round(float(cost or 0.0), 5),
        "summary": "",
        "pricing": config.pricing_info(model, provider),
    }


def _judge_analyst(
    user_text: str, correct_answer: Optional[str], responses: list
):
    """Запрашивает у модели резюме-аналитику по готовым ответам.

    Судья оценивает КАЖДЫЙ ответ отдельно по шкале 1–10 по пяти параметрам:
    точность, креативность, вариативность словарного запаса, лаконичность,
    оптимальность затраченных токенов. Параметр «точность» оценивается против
    известного верного ответа (если он был в запросе юзера).

    Возвращает ПАРУ (резюме, метрики): резюме — список строк таблицы
    [{"temperature", "accuracy", "creativity", "vocabulary", "conciseness",
      "tokens", "summary"}, …] (по одному на каждую температуру) либо сырой текст,
    если ответ модели не разобрался как JSON, либо None, если ответа нет;
    метрики — строка таблицы аналитики по САМОМУ вызову судьи (время, токены,
    стоимость): без неё расход на оценку ответов оставался невидимым.
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
    verdict, metrics = client.call_llm_with_metrics(
        "\n\n".join(parts), system_prompt=system)
    row = _analytics_row("судья-аналитик", config.LLM_MODEL, metrics)
    if not verdict:
        return None, row
    parsed = _parse_judge_json(verdict)
    return (parsed if parsed is not None else verdict.strip()), row


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
    ответа. Возвращает пару (резюме, метрики): резюме — список
    [{"model", "summary"}, …] (пустой, если получить не удалось), метрики —
    строка таблицы аналитики по САМОМУ вызову судьи.
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
        # Стоимость показываем с достаточной точностью: у официального DeepSeek
        # вызов стоит тысячные доли рубля, и «0.00» судье ничего не говорит.
        blocks.append(
            f"Модель {row['model']} ({row['seconds']:.2f}с, вход {row['input_tokens']} / "
            f"выход {row['output_tokens']} токенов, стоимость "
            f"{_fmt_rub(row['cost_rub'])} руб.):\n"
            f"{answer['text']}"
        )
    verdict, metrics = client.call_llm_with_metrics(
        "Ответы и метрики моделей:\n\n" + "\n\n".join(blocks), system_prompt=system
    )
    row = _analytics_row("судья-аналитик", config.LLM_MODEL, metrics)
    if not verdict:
        return [], row
    return _parse_model_summaries(verdict), row


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
    (время, входной/выходной токены) и берётся стоимость, посчитанная клиентом
    по тарифу ТОГО провайдера, которому ушёл запрос, — это аналитика
    судьи-аналитика для вывода таблицей. В строку кладётся и сам тариф
    (`pricing`): модели живут у разных провайдеров, и одна цифра «по тарифам»
    на всю таблицу была бы неправдой.

    Модель может жить у РАЗНЫХ провайдеров (по умолчанию — deepseek-official,
    старые — Yandex): провайдер берётся из описания модели (MODEL_SPECS) и
    уходит в клиент, потому что адрес и ключ у провайдеров свои. Неизвестный
    ключ трактуется как модель по умолчанию — запрос не уходит «в никуда».
    """
    answers = []
    analytics = []
    for key in models:
        spec = MODEL_SPECS.get(key) or MODEL_SPECS[DEFAULT_MODEL_KEY]
        uri = spec["model"] or config.LLM_MODEL
        name = spec["name"]
        text, metrics = client.call_llm_with_metrics(
            user_text,
            response_format=response_format,
            max_tokens=max_tokens,
            stop=stop,
            model=uri,
            provider=spec["provider"],
            # Отключаем reasoning только у моделей, которые это поддерживают
            # (deepseek) — остальные (alice) отклоняют поле thinking (HTTP 400).
            disable_thinking=spec["supports_thinking"],
        )
        answers.append({"model": name, "text": _normalize_answer(user_text, text, response_format)})

        analytics.append(
            {
                "label": name,
                "model": name,
                "seconds": float((metrics or {}).get("elapsed_seconds") or 0),
                "input_tokens": int((metrics or {}).get("prompt_tokens") or 0),
                "output_tokens": int((metrics or {}).get("completion_tokens") or 0),
                # Стоимость уже посчитана клиентом (тариф провайдера, кэш,
                # пиковые часы, курс) — здесь она только переносится.
                "cost_rub": round(float((metrics or {}).get("cost_rub") or 0.0), 5),
                "pricing": config.pricing_info(uri, spec["provider"]),
            }
        )
    # Резюме судьи-аналитика: краткий разбор каждой модели (скорость, стоимость,
    # ресурсоёмкость, качество). Прикрепляем summary к строкам таблицы.
    if answers:
        summaries, judge_row = _judge_model_test(answers, analytics)
        by_model = {s["model"]: s["summary"] for s in summaries}
        for row in analytics:
            row["summary"] = by_model.get(row["model"], "")
        # Вызов судьи — тоже обращение к модели: показываем его цену отдельной
        # строкой, иначе расход на оценку в таблице не виден.
        if judge_row:
            judge_row["summary"] = ""
            analytics.append(judge_row)
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
) -> tuple:
    """Выполняет отдельный запрос к LLM для каждого значения «Температуры».

    Каждое заполненное поле = отдельный запрос с temperature=<значение>.
    Возвращает пару (ответы, аналитика): ответы — список словарей
    {"temperature": t, "text": ответ} (ровно столько, сколько заполненных полей,
    минимум один), аналитика — строки таблицы метрик (время, токены, стоимость)
    по каждому вызову. Раньше метрики выбрасывались, и на странице статистики
    вместо реальных чисел стояли только субъективные оценки судьи 1–10.
    """
    results = []
    analytics = []
    for raw in temperatures:
        try:
            t = float(raw)
        except (TypeError, ValueError):
            continue
        answer, metrics = client.call_llm_with_metrics(
            user_text,
            response_format=response_format,
            max_tokens=max_tokens,
            stop=stop,
            temperature=t,
        )
        results.append(
            {"temperature": t, "text": _normalize_answer(user_text, answer, response_format)}
        )
        analytics.append(_analytics_row(f"temperature {t}", config.LLM_MODEL, metrics))
    # Если после фильтра значений не осталось — ведём себя как обычный режим.
    if not results:
        answer, metrics = client.call_llm_with_metrics(
            user_text,
            response_format=response_format,
            max_tokens=max_tokens,
            stop=stop,
        )
        results = [
            {
                "temperature": None,
                "text": _normalize_answer(user_text, answer, response_format),
            }
        ]
        analytics = [_analytics_row("обычный ответ", config.LLM_MODEL, metrics)]
    return results, analytics


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
    return _verdict_of(*client.call_llm_with_metrics(user, system_prompt=system))


def _judge_correctness_metrics(user_text: str, answer: str,
                               mode: str) -> tuple:
    """Как _judge_correctness, но вместе с метриками своего вызова.

    Возвращает (вердикт|None, метрики|None): страница статистики показывает
    расход и на сам вердикт — это служебный вызов, о котором иначе не видно.
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
    verdict, metrics = client.call_llm_with_metrics(user, system_prompt=system)
    return _verdict_of(verdict, metrics), metrics


def _verdict_of(verdict: str, _metrics=None):
    """Разбирает ответ судьи: True — ВЕРНО, False — НЕВЕРНО, None — не понял."""
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
    answer, metrics = client.call_llm_with_metrics(user_text, system_prompt=system_prompt)
    analytics = [_analytics_row("ответ эксперта", config.LLM_MODEL, metrics)]
    if answer:
        # Вердикт о верности даёт отдельный служебный вызов судьи — его расход
        # тоже показываем (и токены, и стоимость).
        correct, judge_metrics = _judge_correctness_metrics(user_text, answer, mode)
        analytics.append(_analytics_row("судья (верность)", config.LLM_MODEL,
                                        judge_metrics))
        return answer, correct, analytics

    # Фолбэк (нет API-ключа / сбой): демо-правила, без вердикта.
    if not config.LLM_API_KEY:
        return demo.demo_ai(user_text), None, analytics
    return (
        "Извините, не удалось получить ответ от модели. Попробуйте ещё раз.",
        None, analytics,
    )