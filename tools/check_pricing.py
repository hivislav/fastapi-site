"""Самопроверка тарифов и подсчёта стоимости (app/config.py, app/ai/client.py).

Проверяет ровно то, от чего зависит цифра «стоимость» в интерфейсе:

  * тарифы официального DeepSeek — долларовый прайс Flash/Pro с поправкой на
    пиковые часы (непик = половина) и курсом рубля;
  * ПИКОВЫЕ ЧАСЫ: окна UTC 01:00–04:00 и 06:00–10:00 по будням, выходные
    целиком непиковые, соответствие времени Екатеринбурга (UTC+5);
  * кэш входа: часть prompt_tokens, пришедшая из кэша, считается по цене
    cache hit (в разы дешевле) — в обоих форматах, которыми её отдаёт провайдер;
  * модели старого провайдера (Yandex) считаются по своему РУБЛЁВОМУ тарифу, а
    не по долларовому;
  * стоимость считается ОДНИМ способом: метрики клиента и строка таблицы
    аналитики дают одну и ту же сумму (раньше это были два разных расчёта);
  * тариф для показа (pricing_info) совпадает с тем, по которому посчитана
    стоимость.

Запуск (внешняя сеть и API-ключ не нужны):

    ./venv/bin/python tools/check_pricing.py
"""

import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
os.environ.setdefault("YANDEX_API_KEY", "test-key")

from app import config  # noqa: E402
from app.ai import client, service  # noqa: E402

FAILURES = []

# Момент времени ПИКОВЫХ часов по прайсу DeepSeek: среда, 02:00 UTC
# (= 07:00 по Екатеринбургу). Непиковый — та же среда, 05:00 UTC.
PEAK_AT = datetime(2026, 9, 23, 2, 0, tzinfo=timezone.utc)
OFFPEAK_AT = datetime(2026, 9, 23, 5, 0, tzinfo=timezone.utc)
SATURDAY = datetime(2026, 9, 26, 2, 0, tzinfo=timezone.utc)
SUNDAY = datetime(2026, 9, 27, 7, 0, tzinfo=timezone.utc)

FLASH = "deepseek-v4-flash"
PRO = "deepseek-v4-pro"


def check(name, condition, detail=""):
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(name)


def close(a, b, eps=1e-9):
    return abs(float(a) - float(b)) <= eps


def usd_cost_manual(model, prompt, completion, hit, when):
    """Ручной расчёт стоимости в рублях — по прайсу, независимо от кода."""
    rates = config.deepseek_tariff(model, when)
    usd = ((prompt - hit) * rates["cache_miss"]
           + hit * rates["cache_hit"]
           + completion * rates["output"]) / 1_000_000.0
    return usd * config.USD_RUB


def main():
    print("\n[1] Прайс официального DeepSeek ($ за 1M токенов)")
    flash_peak = config.deepseek_tariff(FLASH, PEAK_AT)
    check("Flash в пик: вход из кэша 0,006 $", close(flash_peak["cache_hit"], 0.006),
          str(flash_peak))
    check("Flash в пик: вход мимо кэша 0,3 $", close(flash_peak["cache_miss"], 0.3),
          str(flash_peak))
    check("Flash в пик: выход 1,2 $", close(flash_peak["output"], 1.2), str(flash_peak))
    pro_peak = config.deepseek_tariff(PRO, PEAK_AT)
    check("Pro в пик: 0,044 / 1,32 / 3,96 $",
          close(pro_peak["cache_hit"], 0.044) and close(pro_peak["cache_miss"], 1.32)
          and close(pro_peak["output"], 3.96), str(pro_peak))
    check("легаси-имя deepseek-v4-flash считается по тарифу Flash",
          config.deepseek_tariff_key(FLASH) == "flash"
          and config.deepseek_tariff_key("deepseek-flash") == "flash"
          and config.deepseek_tariff_key("deepseek-v4-pro") == "v4-pro")

    print("\n[2] Непиковые часы: ровно половина цены")
    flash_off = config.deepseek_tariff(FLASH, OFFPEAK_AT)
    check("непик = половина пика (все три ставки)",
          all(close(flash_off[key], flash_peak[key] / 2) for key in flash_peak),
          f"{flash_peak} → {flash_off}")

    print("\n[3] Пиковые часы по прайсу (UTC, Пн–Пт)")
    # Окна: 01:00–04:00 и 06:00–10:00 UTC; границы — начало включительно,
    # конец — исключительно.
    cases = [
        ("00:59 UTC", datetime(2026, 9, 23, 0, 59), False),
        ("01:00 UTC (начало окна)", datetime(2026, 9, 23, 1, 0), True),
        ("03:59 UTC (конец окна)", datetime(2026, 9, 23, 3, 59), True),
        ("04:00 UTC (окно кончилось)", datetime(2026, 9, 23, 4, 0), False),
        ("05:59 UTC", datetime(2026, 9, 23, 5, 59), False),
        ("06:00 UTC (начало окна)", datetime(2026, 9, 23, 6, 0), True),
        ("09:59 UTC (конец окна)", datetime(2026, 9, 23, 9, 59), True),
        ("10:00 UTC (окно кончилось)", datetime(2026, 9, 23, 10, 0), False),
        ("23:30 UTC", datetime(2026, 9, 23, 23, 30), False),
    ]
    for label, moment, expected in cases:
        got = config.deepseek_is_peak(moment.replace(tzinfo=timezone.utc))
        check(f"{label} → {'пик' if expected else 'непик'}", got is expected, f"получено {got}")
    check("суббота — всегда непик", config.deepseek_is_peak(SATURDAY) is False)
    check("воскресенье — всегда непик", config.deepseek_is_peak(SUNDAY) is False)
    check("наивное время трактуется как UTC",
          config.deepseek_is_peak(datetime(2026, 9, 23, 2, 0)) is True)
    check("часовой пояс сервера не влияет (тот же момент в +5)",
          config.deepseek_is_peak(PEAK_AT.astimezone(timezone(timedelta(hours=5)))) is True)

    print("\n[4] Пиковые окна во времени Екатеринбурга (UTC+5)")
    yekt = timezone(timedelta(hours=config.SITE_UTC_OFFSET_HOURS))
    check("сайт считает себя по Екатеринбургу (UTC+5)",
          config.SITE_UTC_OFFSET_HOURS == 5, str(config.SITE_UTC_OFFSET_HOURS))
    windows = [
        (datetime(2026, 9, 23, 1, 0, tzinfo=timezone.utc), "06:00"),
        (datetime(2026, 9, 23, 4, 0, tzinfo=timezone.utc), "09:00"),
        (datetime(2026, 9, 23, 6, 0, tzinfo=timezone.utc), "11:00"),
        (datetime(2026, 9, 23, 10, 0, tzinfo=timezone.utc), "15:00"),
    ]
    for moment, expected in windows:
        local = moment.astimezone(yekt).strftime("%H:%M")
        check(f"{moment.strftime('%H:%M')} UTC = {expected} по Екатеринбургу",
              local == expected, f"получено {local}")

    print("\n[5] Стоимость вызова: ручной расчёт против usage_cost")
    cases = [
        ("1M входа мимо кэша, пик", PEAK_AT, 1_000_000, 0, 0),
        ("1M выхода, пик", PEAK_AT, 0, 1_000_000, 0),
        ("1M входа из кэша, пик", PEAK_AT, 1_000_000, 0, 1_000_000),
        ("1000 вход / 500 выход, пик", PEAK_AT, 1_000, 500, 0),
        ("4000 из кэша + 6000 мимо, 2500 выход, непик", OFFPEAK_AT, 10_000, 2_500, 4_000),
        ("Pro: 2000 вход / 1000 выход, пик", PEAK_AT, 2_000, 1_000, 0),
    ]
    for label, when, prompt, completion, hit in cases:
        model = PRO if label.startswith("Pro") else FLASH
        expected = round(usd_cost_manual(model, prompt, completion, hit, when), 5)
        got = config.usage_cost(model, prompt, completion, hit, when=when)
        check(f"{label} = {expected} руб.", close(got, expected), f"получено {got}")
    # Стоимость округляется до 5 знаков (0,00001 руб.), поэтому «ровно вдвое»
    # сходится с точностью до этого округления (у каждой из двух величин своя
    # погрешность), а не до последнего бита.
    check("пик ровно в 2 раза дороже непика на тех же токенах",
          close(config.usage_cost(FLASH, 100_000, 50_000, when=PEAK_AT),
                2 * config.usage_cost(FLASH, 100_000, 50_000, when=OFFPEAK_AT), 2e-5))

    print("\n[6] Кэш входа: разбор usage провайдера и выравнивание")
    check("prompt_cache_hit_tokens", client._cache_hit_tokens(
        {"prompt_tokens": 100, "prompt_cache_hit_tokens": 64}) == 64)
    check("prompt_tokens_details.cached_tokens",
          client._cache_hit_tokens(
              {"prompt_tokens": 100,
               "prompt_tokens_details": {"cached_tokens": 32}}) == 32)
    check("провайдер не сообщил кэш — 0",
          client._cache_hit_tokens({"prompt_tokens": 100}) == 0)
    check("мусор в поле кэша не ломает метрики",
          client._cache_hit_tokens({"prompt_cache_hit_tokens": "нет"}) == 0)
    # Кэш больше входа (провайдер посчитал как-то иначе) не должен уменьшать
    # вход: цена берётся только за реально пришедшие токены.
    metrics = client._usage_metrics(FLASH, 0.5, {
        "prompt_tokens": 100, "completion_tokens": 10,
        "prompt_cache_hit_tokens": 500,
        "prompt_tokens_details": {"cached_tokens": 500},
    })
    check("кэш обрезан по входу (100 токенов)",
          metrics["cache_hit_tokens"] == 100 and metrics["cache_miss_tokens"] == 0,
          str(metrics))

    print("\n[7] Метрики клиента: кэш, стоимость и единый расчёт")
    usage = {"prompt_tokens": 10_000, "completion_tokens": 2_000,
             "total_tokens": 12_000, "prompt_cache_hit_tokens": 4_000}
    metrics = client._usage_metrics(FLASH, 1.25, usage, provider="deepseek-official")
    expected = config.usage_cost(FLASH, 10_000, 2_000, 4_000, provider="deepseek-official")
    check("стоимость метрик = usage_cost", close(metrics["cost_rub"], expected),
          f"{metrics['cost_rub']} != {expected}")
    check("кэш-токены в метриках", metrics["cache_hit_tokens"] == 4_000
          and metrics["cache_miss_tokens"] == 6_000, str(metrics))
    check("токены и время не пострадали",
          metrics["prompt_tokens"] == 10_000 and metrics["completion_tokens"] == 2_000
          and metrics["elapsed_seconds"] == 1.25, str(metrics))
    row = service._analytics_row("ответ", FLASH, metrics)
    check("строка таблицы берёт ту же стоимость",
          close(row["cost_rub"], expected), f"{row['cost_rub']} != {expected}")
    check("строка таблицы несёт тариф для показа", bool(row.get("pricing")))
    # Метрики без посчитанной стоимости (старые записи) считаются заново.
    legacy = dict(metrics)
    legacy.pop("cost_rub")
    row_legacy = service._analytics_row("ответ", FLASH, legacy)
    check("метрики без cost_rub считаются снова",
          close(row_legacy["cost_rub"], expected, 1e-6),
          f"{row_legacy['cost_rub']} != {expected}")

    print("\n[8] Старый провайдер (Yandex): рублёвый тариф, не долларовый")
    # Тарифы Yandex: alice-flash 0,1/0,2; alice 0,5/1,2; deepseek 0,3/0,5 руб.
    # за 1000 токенов. Проверяем и Flash, и старшую модель: поиск тарифа идёт
    # по URI, и «aliceai-llm-flash» не должен попадать в тариф «aliceai-llm».
    yandex_cases = [
        ("alice-flash", "gpt://b1gkm5u908if6dc0focb/aliceai-llm-flash/latest", 0.3),
        ("alice", "gpt://b1gkm5u908if6dc0focb/aliceai-llm/latest", 1.7),
        ("deepseek", "gpt://b1gkm5u908if6dc0focb/deepseek-v4-flash/latest", 0.8),
    ]
    for label, uri, expected in yandex_cases:
        got = config.usage_cost(uri, 1_000, 1_000, provider="yandex")
        check(f"{label}: 1000 вход + 1000 выход = {expected} руб.",
              close(got, expected), f"получено {got}")
    check("Yandex-модель распознаётся по URI",
          config.provider_for_model("gpt://b1gkm5u908if6dc0focb/aliceai-llm/latest")
          == "yandex")
    check("модель по умолчанию считается по долларовому тарифу",
          not close(config.usage_cost(FLASH, 1_000, 1_000), 0.8),
          str(config.usage_cost(FLASH, 1_000, 1_000)))
    check("кэш у Yandex не выдумывается (весь вход по одной цене)",
          close(config.usage_cost("gpt://x/aliceai-llm-flash/latest", 1_000, 0,
                                  cache_hit_tokens=1_000, provider="yandex"), 0.1),
          str(config.usage_cost("gpt://x/aliceai-llm-flash/latest", 1_000, 0,
                               cache_hit_tokens=1_000, provider="yandex")))

    print("\n[9] Тариф для показа (pricing_info) совпадает с расчётом")
    info = config.pricing_info(FLASH, when=PEAK_AT)
    check("провайдер и модель", info["provider"] == "deepseek-official"
          and info["model"] == FLASH, str(info))
    check("в пик тариф подписан как пиковый",
          info["peak"] is True and "пиковый" in info["tariff"], str(info))
    check("ставки за 1M в рублях = прайс × курс",
          close(info["rub_per_mtok"]["cache_miss"], 0.3 * config.USD_RUB)
          and close(info["rub_per_mtok"]["output"], 1.2 * config.USD_RUB), str(info))
    check("в тарифе есть курс и часы пика",
          info["usd_rub"] == config.USD_RUB and "Екатеринбург" in info["peak_note"],
          str(info))
    check("окна пика в подписи посчитаны из тех же констант",
          config.deepseek_peak_windows_local() == ["06:00–09:00", "11:00–15:00"]
          and all(window in info["peak_note"]
                  for window in config.deepseek_peak_windows_local()),
          f"{config.deepseek_peak_windows_local()} / {info['peak_note']}")
    check("непиковый тариф подписан и вдвое дешевле",
          config.pricing_info(FLASH, when=OFFPEAK_AT)["peak"] is False
          and close(config.pricing_info(FLASH, when=OFFPEAK_AT)["rub_per_mtok"]["output"],
                    1.2 * config.USD_RUB / 2), str(info))
    yandex_info = config.pricing_info(
        "gpt://b1gkm5u908if6dc0focb/aliceai-llm/latest", provider="yandex")
    check("Yandex-модель показывает рублёвый тариф и не врёт про пик",
          yandex_info["provider"] == "yandex" and yandex_info["peak"] is False
          and not yandex_info["usd_rub"], str(yandex_info))
    check("цена за 1000 токенов согласована с расчётом",
          close(config.model_price("gpt://x/aliceai-llm/latest", provider="yandex")["input"],
                0.5), str(config.model_price("gpt://x/aliceai-llm/latest", provider="yandex")))

    print("\n[10] Замер агента несёт тариф (панель «Токены задачи»)")
    from app.ai import agent as agent_mod  # noqa: E402
    fresh = agent_mod.Agent._new_usage()
    check("в замере есть pricing", isinstance(fresh.get("pricing"), dict),
          str(list(fresh.keys())))
    check("пик/непик и ставки на месте",
          "tariff" in fresh["pricing"] and "rub_per_mtok" in fresh["pricing"],
          str(fresh.get("pricing")))
    check("стоимость замера начинается с нуля", fresh["cost_rub"] == 0.0)
    # Замер копит стоимость ИЗ метрик вызова — то есть из того же расчёта, что
    # показывает таблица. Проверяем на синтетических метриках.
    agent = agent_mod.Agent.__new__(agent_mod.Agent)
    agent.last_usage = agent_mod.Agent._new_usage()
    agent_mod.Agent._track_usage(agent, {
        "model": FLASH, "elapsed_seconds": 0.4, "prompt_tokens": 10_000,
        "completion_tokens": 2_000, "total_tokens": 12_000,
        "cache_hit_tokens": 4_000, "cache_miss_tokens": 6_000,
        "cost_rub": config.usage_cost(FLASH, 10_000, 2_000, 4_000),
    })
    check("стоимость шага = стоимость вызова по тарифу",
          close(agent.last_usage["cost_rub"],
                config.usage_cost(FLASH, 10_000, 2_000, 4_000)),
          str(agent.last_usage["cost_rub"]))
    check("токены шага посчитаны", agent.last_usage["input"] == 10_000
          and agent.last_usage["output"] == 2_000, str(agent.last_usage))

    print()
    if FAILURES:
        print(f"Итог: ПРОВАЛЕНО проверок — {len(FAILURES)}: {', '.join(FAILURES)}")
        sys.exit(1)
    print("Итог: все проверки пройдены")


if __name__ == "__main__":
    main()
