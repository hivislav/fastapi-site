"""Конфигурация приложения.

Централизованно загружает настройки из переменных окружения (.env) и
предоставляет их остальным модулям. Здесь нет бизнес-логики и зависимостей
от FastAPI, поэтому модуль можно переиспользовать где угодно.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Корень проекта: каталог, где лежат main.py, .env, data/ и т.п.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Простая загрузка .env (без внешней зависимости): читает KEY=VALUE строки
# из файла .env рядом с корнем проекта, не перезаписывая уже заданные
# переменные окружения.
# ---------------------------------------------------------------------------
def load_dotenv() -> None:
    env_path = os.path.join(PROJECT_ROOT, ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_dotenv()

# ---------------------------------------------------------------------------
# Провайдеры LLM: endpoint, ключ и модель по умолчанию.
#
# У КАЖДОГО провайдера свой адрес и СВОЙ ключ: официальный API DeepSeek и
# Yandex Cloud AI Studio — разные сервисы, один ключ к другому не подходит,
# поэтому провайдер выбирается до запроса, а не «одним адресом на всё».
#
# Модель по умолчанию — DeepSeek-V4-Flash у провайдера deepseek-official
# (`agent-default-model`): её обслуживают обычные запросы, экспертные режимы,
# «Температура», судья-аналитик и AI-агент. Модели старого провайдера (Yandex)
# остаются РАБОЧИМИ, но выбираются только вручную — в настройке «Тест моделей»
# (см. app/ai/service.py). Общий выбор делает provider_spec().
# ---------------------------------------------------------------------------

# Провайдер обычных запросов (не «Тест моделей») — фиксирован: официальный API
# DeepSeek. Отдельной переменной окружения для него нет намеренно: старые модели
# должны выбираться РУЧНО в «Тесте моделей», а не «переключателем по умолчанию»,
# который к тому же увёл бы к Yandex идентификатор чужой модели.
DEFAULT_PROVIDER = "deepseek-official"

# --- Провайдер по умолчанию: официальный API DeepSeek (llm-deepseek) ---
# Адрес, ключ и модель — свои переменные окружения (DEEPSEEK_*), а НЕ
# исторические LLM_BASE_URL/LLM_MODEL: их значения в старых .env указывают на
# Yandex, и модель по умолчанию не должна из-за этого «переезжать» на чужой
# endpoint с чужим ключом.
LLM_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# Reasoning (thinking) у модели по умолчанию выключен ВСЕГДА: «размышления»
# тратят время и бюджет токенов, а на видимый ответ не влияют. 0 — не
# выключать принудительно (поле thinking уходит только по общим правилам).
LLM_DISABLE_THINKING = int(os.getenv("LLM_DISABLE_THINKING", "1"))

# --- Старый провайдер: Yandex Cloud AI Studio (только «Тест моделей») ---
# Имена переменных — YANDEX_*: прежние LLM_BASE_URL/LLM_MODEL (тогда Yandex был
# единственным провайдером) больше НЕ читаются — иначе старый .env молча увёл бы
# запросы к старым моделям на чужой endpoint с чужим ключом.
YANDEX_BASE_URL = os.getenv("YANDEX_BASE_URL", "https://ai.api.cloud.yandex.net/v1")
YANDEX_MODEL = os.getenv(
    "YANDEX_MODEL", "gpt://b1gkm5u908if6dc0focb/deepseek-v4-flash/latest"
)
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY", "")

# --- ЛОКАЛЬНАЯ модель (MLX) — третий источник ответа ---
# Отдельный провайдер "local": сервер mlx_lm.server (пакет mlx-lm) работает на
# ЭТОМ ЖЕ Mac по адресу LOCAL_LLM_BASE_URL и отдаёт OpenAI-совместимый
# /chat/completions. Ни ключа, ни сети не нужно — но клиент требует непустой
# ключ, поэтому у локального провайдера он условный ("local").
#
# Всё окружение и веса живут ВНУТРИ проекта (data/local_llm, каталог в
# .gitignore): свой venv с mlx-lm (его ставит tools/local_llm.sh — интерпретатор
# берётся системный, Python 3.11+), веса в HF_HOME — наружу (в ~/.cache) не
# пишется ничего.
LOCAL_LLM_HOME = os.getenv(
    "LOCAL_LLM_HOME", os.path.join(PROJECT_ROOT, "data", "local_llm"),
)
LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8080/v1")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "mlx-community/Qwen3-8B-4bit")
LOCAL_LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "local")
# Человекочитаемое имя модели — для подписи в интерфейсе (в промпт не идёт).
LOCAL_LLM_TITLE = os.getenv("LOCAL_LLM_TITLE", "Qwen3-8B-4bit (MLX)")
LOCAL_LLM_HOST = os.getenv("LOCAL_LLM_HOST", "127.0.0.1")
LOCAL_LLM_PORT = int(os.getenv("LOCAL_LLM_PORT", "8080"))
# Файл выбранного источника ответа (кнопка «локальная/удалённая» в панели
# workspace): выбор переживает перезапуск приложения.
LLM_SOURCE_FILE = os.getenv(
    "LLM_SOURCE_FILE", os.path.join(LOCAL_LLM_HOME, "source.json"),
)
# Источник ДО первого переключения (дальше решает файл выше): "remote" —
# как было всегда, "local" — сразу локальная модель.
LLM_SOURCE_DEFAULT = os.getenv("LLM_SOURCE", "remote")

# Ключа модели по умолчанию нет (например, .env не обновлён после перехода на
# официальный DeepSeek): обычные запросы уйдут в демо-режим, а старые модели
# «Теста моделей» продолжат работать, если задан YANDEX_API_KEY. Пишем об этом
# в лог при запуске — иначе подмена живого ответа демо-текстом выглядит как
# «модель не ответила».
if not LLM_API_KEY:
    logging.getLogger(__name__).warning(
        "LLM: не задан DEEPSEEK_API_KEY — модель по умолчанию (%s, %s) недоступна, "
        "обычные запросы отвечает демо-режим; старым моделям «Теста моделей» "
        "нужен YANDEX_API_KEY", LLM_MODEL, LLM_BASE_URL,
    )

LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2000"))

# Сколько ДОПОЛНИТЕЛЬНЫХ попыток делает клиент модели при сбое (429/5xx/сеть).
# Один 429 на шаге раньше уничтожал всю задачу, хотя токены предыдущих шагов
# уже оплачены, поэтому по умолчанию две дополнительные попытки.
LLM_RETRIES = int(os.getenv("LLM_RETRIES", "2"))

# Повторять ли вызов, оборвавшийся по таймауту (0 — нет). Провайдер мог уже
# начать генерировать ответ, поэтому такой повтор дороже: по умолчанию выключен.
LLM_RETRY_TIMEOUT = int(os.getenv("LLM_RETRY_TIMEOUT", "0"))

# Читать ответ модели потоком (SSE). Снимает «общий» таймаут на весь ответ, но
# требует от провайдера usage в потоке (stream_options.include_usage): без него
# расход токенов неизвестен, а панель статистики считает именно его.
LLM_STREAM = int(os.getenv("LLM_STREAM", "0"))

# ---------------------------------------------------------------------------
# Тарифы: во что обходится запрос (оценка)
#
# Провайдеры считают по-разному, поэтому тариф берётся у ТОГО провайдера,
# которому ушёл запрос:
#   * официальный DeepSeek — в ДОЛЛАРАХ за 1M токенов, отдельно за вход с
#     попаданием в кэш (cache hit), вход без попадания (cache miss) и выход,
#     и ВДВОЕ дешевле в непиковые часы (см. deepseek_is_peak);
#   * Yandex Cloud AI Studio — в рублях за 1000 токенов, без разбивки по кэшу.
# Считает ОДНА функция — usage_cost(): и клиент LLM (метрики вызова), и таблица
# аналитики, и панель токенов агента берут стоимость оттуда, поэтому числа не
# расходятся. Тариф показывается в интерфейсе через pricing_info().
# ---------------------------------------------------------------------------

# --- Официальный DeepSeek: $ за 1M токенов В ПИКОВЫЕ ЧАСЫ ---
# Прайс: https://api-docs.deepseek.com/quick_start/pricing/
# В непиковые часы — половина (DEEPSEEK_OFFPEAK_FACTOR).
DEEPSEEK_USD_PER_MTOK = {
    # ключ тарифа: (вход из кэша, вход мимо кэша, выход), $ за 1M токенов
    "flash": {"cache_hit": 0.006, "cache_miss": 0.3, "output": 1.2},
    "v4-pro": {"cache_hit": 0.044, "cache_miss": 1.32, "output": 3.96},
}
DEEPSEEK_OFFPEAK_FACTOR = float(os.getenv("DEEPSEEK_OFFPEAK_FACTOR", "0.5"))

# Пиковые часы DeepSeek (по прайсу) — в UTC: 01:00–04:00 и 06:00–10:00, Пн–Пт,
# кроме праздников КНР. Всё остальное время, включая выходные целиком, —
# непиковое. Сайт живёт по Екатеринбургу (UTC+5), поэтому для него пик — это
# 06:00–09:00 и 11:00–15:00 по местному времени.
# Праздники КНР не учитываются: в эти дни оценка может быть ЗАВЫШЕНА вдвое
# (реально действует непиковый тариф), но никогда не занижена.
DEEPSEEK_PEAK_UTC = ((1, 4), (6, 10))
SITE_UTC_OFFSET_HOURS = int(os.getenv("SITE_UTC_OFFSET_HOURS", "5"))  # Екатеринбург

# Курс доллара: провайдер выставляет счёт в долларах, а сайт показывает рубли.
# ОЦЕНКА, обновляется вручную (курс ЦБ РФ на 23.09.2026 — 84,0657 ₽ за $1).
USD_RUB = float(os.getenv("USD_RUB", "84.0657"))

# --- Старый провайдер (Yandex Cloud AI Studio): руб. за 1000 токенов ---
# Ключ — ФРАГМЕНТ идентификатора модели (URI). Совпадение ищется по самой
# ДЛИННОЙ подходящей подстроке — иначе «aliceai-llm-flash» подходил бы под
# «aliceai-llm» и Flash считался бы по тарифу старшей модели, а URI DeepSeek
# («gpt://…/deepseek-v4-flash/latest») вообще не находил тарифа и стоил 0.
YANDEX_MODEL_PRICING = {
    "aliceai-llm-flash": {"input": 0.1, "output": 0.2},
    "aliceai-llm": {"input": 0.5, "output": 1.2},
    "deepseek": {"input": 0.3, "output": 0.5},
}


def _yandex_price(model: str) -> dict:
    """Тариф старого провайдера по идентификатору модели (неизвестная — нули).

    Побеждает САМЫЙ ДЛИННЫЙ подходящий фрагмент: он точнее описывает модель
    (flash против обычной), поэтому тариф не зависит от порядка ключей.
    """
    key = str(model or "").lower()
    matched = None
    for fragment, price in YANDEX_MODEL_PRICING.items():
        if fragment in key and (matched is None or len(fragment) > len(matched)):
            matched = fragment
    if matched is None:
        return {"input": 0, "output": 0}
    return dict(YANDEX_MODEL_PRICING[matched])


def deepseek_tariff_key(model: str) -> str:
    """Ключ тарифа официального DeepSeek по идентификатору модели.

    Модель по умолчанию — Flash. `deepseek-v4-flash` — прежнее (легаси) имя той
    же Flash-модели: провайдер принимает его и считает по тарифу Flash
    (https://api-docs.deepseek.com/quick_start/pricing/).
    """
    return "v4-pro" if "pro" in str(model or "").lower() else "flash"


def deepseek_is_peak(when: Optional[Any] = None) -> bool:
    """Идут ли ПИКОВЫЕ часы DeepSeek прямо сейчас.

    Время считается в UTC (провайдер биллингует по UTC), поэтому результат не
    зависит от часового пояса сервера. `when` — момент времени (по умолчанию
    «сейчас»); принимается и наивное время — оно трактуется как UTC.
    """
    moment = when or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    moment = moment.astimezone(timezone.utc)
    if moment.weekday() >= 5:  # суббота и воскресенье — всегда непиковые
        return False
    hour = moment.hour + moment.minute / 60.0
    return any(start <= hour < end for start, end in DEEPSEEK_PEAK_UTC)


def deepseek_peak_windows_local() -> list:
    """Пиковые окна во времени САЙТА (Екатеринбург, UTC+5) как «06:00–09:00».

    Считается из тех же констант, что и `deepseek_is_peak`, поэтому подпись
    тарифа в интерфейсе не может разойтись с реальной проверкой часов.
    """
    windows = []
    for start, end in DEEPSEEK_PEAK_UTC:
        first = (start + SITE_UTC_OFFSET_HOURS) % 24
        last = (end + SITE_UTC_OFFSET_HOURS) % 24
        windows.append(f"{first:02d}:00–{last:02d}:00")
    return windows


def deepseek_tariff(model: str, when: Optional[Any] = None) -> Dict[str, float]:
    """Тариф официального DeepSeek в $ за 1M токенов на момент `when`."""
    rates = dict(DEEPSEEK_USD_PER_MTOK[deepseek_tariff_key(model)])
    if not deepseek_is_peak(when):
        rates = {name: rate * DEEPSEEK_OFFPEAK_FACTOR for name, rate in rates.items()}
    return rates


def usage_cost(
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cache_hit_tokens: int = 0,
    provider: Optional[str] = None,
    when: Optional[Any] = None,
) -> float:
    """Стоимость одного вызова LLM в рублях (оценка по тарифу провайдера).

    ЕДИНСТВЕННОЕ место, где считается стоимость: им пользуются метрики клиента,
    таблица аналитики и панель токенов агента, поэтому числа совпадают.

    prompt_tokens — весь вход запроса; cache_hit_tokens — его часть, пришедшая
    ИЗ КЭША провайдера (DeepSeek отдаёт её в usage и берёт за неё в разы
    меньше). Остаток входа считается по цене cache miss. Если провайдер кэш не
    сообщает (Yandex), весь вход считается обычным.

    provider — имя провайдера; не задан — определяется по модели
    (provider_for_model). Модели Yandex считаются по рублёвому тарифу
    MODEL_PRICING, модели официального DeepSeek — по долларовому прайсу с
    поправкой на пиковые часы и курсом USD_RUB.
    """
    prompt = max(0, int(prompt_tokens or 0))
    completion = max(0, int(completion_tokens or 0))
    hit = max(0, min(int(cache_hit_tokens or 0), prompt))
    miss = prompt - hit
    name = provider or provider_for_model(model)
    if name == "local":
        # Локальная модель считает на своём железе: счёт за токены не
        # выставляется ни в какой валюте. Ноль здесь — не «тариф неизвестен»,
        # а именно отсутствие оплаты (в панели это видно по подписи тарифа).
        return 0.0
    if name == "yandex":
        # Yandex кэш отдельно не тарифицирует: весь вход стоит одинаково,
        # поэтому переданные кэш-токены здесь игнорируются (иначе вход из
        # «кэша» оказался бы бесплатным).
        price = _yandex_price(model)
        cost = (prompt / 1000.0) * price["input"] + (completion / 1000.0) * price["output"]
        return round(cost, 5)
    rates = deepseek_tariff(model, when)
    usd = (
        hit * rates["cache_hit"]
        + miss * rates["cache_miss"]
        + completion * rates["output"]
    ) / 1_000_000.0
    return round(usd * USD_RUB, 5)


def model_price(model: str, provider: Optional[str] = None,
                when: Optional[Any] = None) -> dict:
    """Тариф модели в рублях за 1000 токенов ({"input", "output"}).

    Для показа «сколько стоит тысяча токенов»: у официального DeepSeek это
    цена входа мимо кэша и цена выхода на текущий момент (пик/непик), у
    моделей Yandex — их рублёвый тариф, у локальной модели — нули (счёт не
    выставляется). Неизвестная модель — нули, без выдумок.
    """
    name = provider or provider_for_model(model)
    if name == "local":
        return {"input": 0.0, "output": 0.0}
    if name == "yandex":
        return _yandex_price(model)
    rates = deepseek_tariff(model, when)
    return {
        "input": rates["cache_miss"] * USD_RUB / 1000.0,
        "output": rates["output"] * USD_RUB / 1000.0,
    }


def pricing_info(model: Optional[str] = None, provider: Optional[str] = None,
                 when: Optional[Any] = None) -> Dict[str, Any]:
    """Тариф, по которому посчитана стоимость вызова, — для показа в интерфейсе.

    Возвращает {"provider", "model", "peak", "tariff", "usd_rub",
    "usd_per_mtok"|None, "rub_per_mtok", "peak_note"}: интерфейс показывает
    строку тарифа под таблицей расхода, чтобы цифра стоимости была объяснимой
    (какой тариф, пиковый ли, по какому курсу).
    """
    used = model or active_model()
    name = provider or provider_for_model(used)
    if name == "local":
        # Локальная модель: тарифа нет вовсе — подпись говорит об этом прямо,
        # иначе «0,00 ₽» читалось бы как «тариф неизвестен» или «ошибка расчёта».
        return {
            "provider": "local",
            "model": used,
            "peak": False,
            "tariff": "локальная модель — платить не за что",
            "usd_rub": None,
            "usd_per_mtok": None,
            "rub_per_mtok": {"cache_hit": 0.0, "cache_miss": 0.0, "output": 0.0},
            "peak_note": (
                "Модель работает на этом же компьютере (MLX, Apple Silicon): "
                "запросы не уходят в сеть, счёт за токены не выставляется."
            ),
        }
    if name == "yandex":
        price = _yandex_price(used)
        return {
            "provider": "yandex",
            "model": used,
            "peak": False,
            "tariff": "тариф Yandex, руб. за 1000 токенов",
            "usd_rub": None,
            "usd_per_mtok": None,
            "rub_per_mtok": {
                "cache_hit": price["input"] * 1000.0,
                "cache_miss": price["input"] * 1000.0,
                "output": price["output"] * 1000.0,
            },
            "peak_note": "",
        }
    peak = deepseek_is_peak(when)
    rates = deepseek_tariff(used, when)
    return {
        "provider": "deepseek-official",
        "model": used,
        "peak": peak,
        "tariff": "пиковый тариф" if peak else "непиковый тариф (×0,5)",
        "usd_rub": USD_RUB,
        "usd_per_mtok": dict(rates),
        "rub_per_mtok": {
            key: rate * USD_RUB for key, rate in rates.items()
        },
        "peak_note": (
            "Пиковые часы DeepSeek: "
            + " и ".join(deepseek_peak_windows_local())
            + " по Екатеринбургу (Пн–Пт, UTC "
            + " и ".join(f"{start:02d}:00–{end:02d}:00" for start, end in DEEPSEEK_PEAK_UTC)
            + "); в остальное время тариф вдвое ниже."
        ),
    }


# ---------------------------------------------------------------------------
# Источник ответа: локальная модель или удалённая (переключатель в интерфейсе)
# ---------------------------------------------------------------------------
# У приложения ОДИН действующий источник ответа на весь процесс: переключатель
# «локальная / удалённая» в панели workspace меняет его для ВСЕХ обращений к
# модели (обычный режим, эксперт, «Температура», судья, AI-агент, RAG), потому
# что источник — это адрес и модель по умолчанию, а не настройка одного режима.
#
# Различие с «Тестом моделей» намеренное: там модель выбирается ВРУЧНУЮ и её
# провайдер приходит в клиент явно (MODEL_SPECS), поэтому сравнить локальный и
# удалённый ответы можно в любой момент, независимо от переключателя.
SOURCES = ("remote", "local")

_llm_source = str(LLM_SOURCE_DEFAULT or "").strip().lower()
if _llm_source not in SOURCES:
    _llm_source = "remote"


def llm_source() -> str:
    """Действующий источник ответа: "remote" (провайдер по умолчанию) или "local"."""
    return _llm_source


def set_llm_source(name: Optional[str]) -> str:
    """Переключает источник ответа. Неизвестное имя — ошибка, а не «тихий remote».

    Опечатка в имени не должна молча оставлять человека на удалённой модели:
    он думает, что запросы больше не уходят в сеть, а они уходят.
    """
    key = str(name or "").strip().lower()
    if key not in SOURCES:
        raise ValueError(f"Неизвестный источник ответа: {name!r}")
    global _llm_source
    _llm_source = key
    return _llm_source


def active_provider() -> str:
    """Провайдер по умолчанию: локальный сервер MLX или официальный DeepSeek."""
    return "local" if _llm_source == "local" else DEFAULT_PROVIDER


def active_model() -> str:
    """Модель по умолчанию у действующего источника.

    ЕЮ обслуживаются все обращения к модели, где модель не задана явно
    (обычный запрос, судья, шаги агента, панель токенов): раньше на этих местах
    стоял config.LLM_MODEL — удалённая модель.
    """
    return LOCAL_LLM_MODEL if _llm_source == "local" else LLM_MODEL


def source_info() -> Dict[str, Any]:
    """Источник ответа для интерфейса: что за модель и куда уходит запрос."""
    if _llm_source == "local":
        return {
            "source": "local",
            "provider": "local",
            "title": LOCAL_LLM_TITLE,
            "model": LOCAL_LLM_MODEL,
            "base_url": LOCAL_LLM_BASE_URL,
            "remote": False,
        }
    return {
        "source": "remote",
        "provider": DEFAULT_PROVIDER,
        "title": "DeepSeek (облако)",
        "model": LLM_MODEL,
        "base_url": LLM_BASE_URL,
        "remote": True,
    }


# ---------------------------------------------------------------------------
# Выбор провайдера
# ---------------------------------------------------------------------------
def provider_spec(name: Optional[str] = None) -> Dict[str, Any]:
    """Параметры провайдера: адрес, ключ, модель по умолчанию и режим thinking.

    `name` — "deepseek-official" (официальный API DeepSeek, провайдер по
    умолчанию), "local" (локальный сервер MLX на этом же Mac) или "yandex"
    (Yandex Cloud AI Studio — модели «Теста моделей»). Не задано имя —
    берётся ДЕЙСТВУЮЩИЙ источник (active_provider); неизвестное имя трактуется
    так же: опечатка в ключе модели не должна отправлять запрос «в никуда» с
    чужим ключом.

    Возвращает словарь с полями provider / title / base_url / api_key / model /
    thinking, где thinking = "disabled" — reasoning выключать в КАЖДОМ запросе
    (так работает модель по умолчанию), "auto" — по общим правилам клиента,
    "ignore" — поля thinking у провайдера НЕТ и отправлять его нельзя
    (локальный сервер MLX его не знает).
    """
    key = str(name or "").strip() or active_provider()
    if key == "yandex":
        return {
            "provider": "yandex",
            "title": "Yandex Cloud AI Studio",
            "base_url": YANDEX_BASE_URL,
            "api_key": YANDEX_API_KEY,
            "model": YANDEX_MODEL,
            "thinking": "auto",
        }
    if key == "local":
        return {
            "provider": "local",
            "title": LOCAL_LLM_TITLE,
            "base_url": LOCAL_LLM_BASE_URL,
            "api_key": LOCAL_LLM_API_KEY,
            "model": LOCAL_LLM_MODEL,
            "thinking": "ignore",
        }
    return {
        "provider": "deepseek-official",
        "title": "DeepSeek",
        "base_url": LLM_BASE_URL,
        "api_key": LLM_API_KEY,
        "model": LLM_MODEL,
        "thinking": "disabled" if LLM_DISABLE_THINKING else "auto",
    }


def provider_for_model(model: Optional[str]) -> str:
    """Имя провайдера по идентификатору модели.

    Модели старого провайдера — URI вида «gpt://…» (Yandex); идентификатор
    локальной модели обслуживает локальный сервер MLX; идентификатор удалённой
    модели по умолчанию — официальный DeepSeek. Всё остальное (модель не
    названа, имя неизвестно) обслуживает ДЕЙСТВУЮЩИЙ источник: так обычные
    запросы и шаги агента уходят туда, куда переключён переключатель.
    """
    key = str(model or "").strip().lower()
    if key.startswith("gpt://"):
        return "yandex"
    if key and key in ("local", str(LOCAL_LLM_MODEL).strip().lower()):
        return "local"
    if key and key == str(LLM_MODEL).strip().lower():
        return DEFAULT_PROVIDER
    return active_provider()


# Предел длины ответа ШАГА агента, если пользователь не задал «Длину» (0 — нет
# предела: действует ограничение провайдера, как было по умолчанию). Длинный
# ответ шага оплачивается ещё раз — он попадает в контекст следующих шагов.
LLM_AGENT_MAX_TOKENS = int(os.getenv("LLM_AGENT_MAX_TOKENS", "0"))

# Файл, в котором хранится история диалога режима «AI-агент» (JSON).
# По умолчанию — data/agent_memory.json в корне проекта; путь можно
# переопределить переменной окружения AGENT_MEMORY_FILE (например, в тестах).
# Файл остаётся только для обратной совместимости: при первом запуске с
# задачами (workspace) прежняя единая история переносится в задачу «Задача 1»
# (см. app/ai/workspace.py, _migrate_legacy).
AGENT_MEMORY_FILE = os.getenv(
    "AGENT_MEMORY_FILE",
    os.path.join(PROJECT_ROOT, "data", "agent_memory.json"),
)

# Файл «рабочего пространства» режима «AI-агент» (JSON): задачи и их сессии
# (диалоги). У каждой сессии своё состояние агентского диалога — память,
# замеры токенов, резюме, факты, ветви плана. По умолчанию —
# data/agent_workspace.json; путь переопределяется env AGENT_WORKSPACE_FILE.
AGENT_WORKSPACE_FILE = os.getenv(
    "AGENT_WORKSPACE_FILE",
    os.path.join(PROJECT_ROOT, "data", "agent_workspace.json"),
)

# Файл профилей пользователя (JSON): сведения о пользователе (имя, род
# деятельности, стиль общения, формат ответа, ограничения), которые уходят в
# системный промпт сессии режима «AI-агент». Профилей может быть несколько —
# пользователь создаёт, удаляет и переключает их в меню профиля. Пока профиля
# нет, он создаётся автоматически с идентификатором user_<цифры>. По умолчанию —
# data/profiles.json; путь переопределяется env AGENT_PROFILES_FILE.
AGENT_PROFILES_FILE = os.getenv(
    "AGENT_PROFILES_FILE",
    os.path.join(PROJECT_ROOT, "data", "profiles.json"),
)

# ---------------------------------------------------------------------------
# Периодические задачи режима «AI-агент» (планировщик, см. app/periodic_runner.py)
# ---------------------------------------------------------------------------
# Работает ли планировщик повторов: он раз в PERIODIC_TICK_SECONDS проверяет
# расписания периодических задач и запускает те, у которых наступил срок. Каждый
# повтор — обычный прогон задачи (план, шаги, проверка) со свежими данными
# внешних инструментов, поэтому в выключенном виде приложение просто не повторяет
# задачи (сами они и их расписания остаются в файле).
PERIODIC_ENABLED = int(os.getenv("PERIODIC_ENABLED", "1"))
# Как часто планировщик заглядывает в расписания (секунды). Это НЕ период
# задачи: период задаётся у каждой задачи свой, а тик определяет, насколько
# поздно повтор заметит наступивший срок.
PERIODIC_TICK_SECONDS = max(5, int(os.getenv("PERIODIC_TICK_SECONDS", "20")))
# Задержка первого тика после запуска приложения: даём приложению подняться и
# не начинаем повторы в ту же секунду, что и старт.
PERIODIC_START_DELAY = max(0, int(os.getenv("PERIODIC_START_DELAY", "10")))
# Предел запросов автомата на ОДИН повтор: первый запрос строит план и выполняет
# первый шаг, дальше каждый шаг и отложенная проверка — отдельный запрос. Предел
# страхует от зацикливания (в интерфейсе авто-прогон шагов ограничен так же);
# обычный повтор до него не доходит — шагов в плане не больше task_state.MAX_STEPS.
PERIODIC_MAX_TURNS = max(1, int(os.getenv("PERIODIC_MAX_TURNS", "24")))

# ---------------------------------------------------------------------------
# Доступ из внешней сети: вход по паролю (см. app/auth.py)
#
# Приложение стоит на маке и теперь доступно из интернета по внешнему адресу
# (проброс порта на роутере, см. tools/serve.sh). Своей защиты у него не было
# никакой: любой, кто дотянулся до порта, получал диалоги, документы, историю,
# вызовы внешних инструментов и модель за деньги владельца. Поэтому вход —
# часть самого приложения, а не «настройка роутера».
# ---------------------------------------------------------------------------
# Логин и пароль. Пустой пароль = гейт выключен (в этом случае наружу
# приложение всё равно НЕ пускает: внешний клиент получает 403 с причиной —
# молчаливое «не настроено, значит пускаем всех» было бы худшим поведением).
ACCESS_USER = os.getenv("ACCESS_USER", "user")
ACCESS_PASSWORD = os.getenv("ACCESS_PASSWORD", "")
# Явный выключатель гейта: пустая строка — «включён тогда и только тогда,
# когда задан пароль», "0" — выключен, "1" — включён (и без пароля тогда
# наружу не пускает никого).
_ACCESS_ENABLED_RAW = os.getenv("ACCESS_ENABLED", "").strip()
# Секрет подписи куки. Если не задан — берётся из файла ACCESS_SECRET_FILE
# (создаётся при первом входе со случайным содержимым): иначе подпись
# обнулялась бы при каждом перезапуске и вход слетал бы после рестарта.
ACCESS_SECRET = os.getenv("ACCESS_SECRET", "")
ACCESS_SECRET_FILE = os.getenv(
    "ACCESS_SECRET_FILE", os.path.join(PROJECT_ROOT, "data", "auth_secret"),
)
# Локального клиента (127.0.0.1) пропускать без пароля. Включено по умолчанию:
# иначе живые проверки проекта, которые ходят по HTTP на 127.0.0.1
# (tools/check_local_llm_live.py, check_rag_dialog_live.py,
# check_mcp_cleanup_live.py), и сам владелец за своим маком упирались бы в
# страницу входа. Это не дыра: локальный клиент уже на этом маке.
ACCESS_LOCAL_BYPASS = int(os.getenv("ACCESS_LOCAL_BYPASS", "1"))
# Сколько живёт вход (часы). Это срок куки, а не срок пароля.
ACCESS_TTL_HOURS = max(1, int(os.getenv("ACCESS_TTL_HOURS", "168")))
# Флаг Secure у куки. По умолчанию выключен: сайт отдаётся по http (домена и
# сертификата нет), а с Secure браузер не отправил бы куку по http и вход
# выглядел бы сломанным. Появится https — поставить 1.
ACCESS_COOKIE_SECURE = int(os.getenv("ACCESS_COOKIE_SECURE", "0"))
# Защита от перебора: столько неудачных попыток входа с одного адреса за окно
# ACCESS_FAIL_WINDOW секунд — и адрес получает 429, пока окно не истечёт.
ACCESS_MAX_FAILS = max(1, int(os.getenv("ACCESS_MAX_FAILS", "10")))
ACCESS_FAIL_WINDOW = max(30, int(os.getenv("ACCESS_FAIL_WINDOW", "600")))


def access_enabled() -> bool:
    """Включён ли гейт доступа: явная настройка, а без неё — «задан пароль»."""
    if _ACCESS_ENABLED_RAW == "0":
        return False
    if _ACCESS_ENABLED_RAW:
        return True
    return bool(ACCESS_PASSWORD)