"""ВТОРОЙ ЭТАП ПОИСКА: реранкинг пула кандидатов — признаки или cross-encoder.

ЧТО ЭТО. Первый этап (`rag_store.search`) ищет по ВСЕЙ базе и потому обязан быть
дешёвым: косинус плюс доля слов запроса. Он слеп к порядку слов и к адресу чанка.
Второй этап работает уже с небольшим пулом (десятки кандидатов) и может позволить
себе более дорогие признаки — здесь их два набора, и оба дают ОДНУ И ТУ ЖЕ
величину: добавку к оценке первого этапа в тех же единицах.

  * **ПРИЗНАКИ** (`features`, работают всегда, без сети и без модели):
      - ФРАЗА — доля биграмм запроса, найденных в тексте ПОДРЯД: «резервное
        копирование» как фраза весит больше, чем те же слова в разных абзацах;
      - АДРЕС — доля значимых слов запроса в СОБСТВЕННОЙ первой строке чанка
        (его заголовке), в разделе, заголовке и имени файла: чанк, который сам
        называется так же, как вопрос, — это «про это»;
      - ШТРАФ короткому чанку БЕЗ слов запроса — лечение hubness: короткий
        чанк-заголовок лежит у центра облака векторов и «похож на любой запрос».
    Добавки считаются ОТ СЕРЕДИНЫ ПУЛА (см. `features`): признак, одинаковый у
    всего пула, фрагменты не различает, а прибавленный ко всем — перетасовывает
    выдачу первого этапа (это измерено, см. `_median`).

  * **CROSS-ENCODER** (`cross-encoder`): модель читает пару (запрос, фрагмент)
    ЦЕЛИКОМ и сама решает, насколько фрагмент релевантен. Это заметно сильнее
    признаков и вектора: модель видит порядок слов, отрицания и смысл. Цена —
    модель на диске (скачивается один раз, `RAG_RERANK_MODEL`, кэш в проекте) и
    время на каждую пару (десятки миллисекунд на CPU), поэтому её считают ТОЛЬКО
    по пулу, уже отобранному первым этапом.

ВЫБОР БЭКЕНДА (`RAG_RERANK_BACKEND` / настройка проекта):
  * `auto` (по умолчанию) — cross-encoder, ЕСЛИ его модель уже лежит в кэше
    проекта, иначе признаки с честной причиной. Сети `auto` НЕ касается: поиск
    не должен внезапно скачивать сотни мегабайт;
  * `features` — только признаки;
  * `cross-encoder` — модель, и если её нет в кэше, она СКАЧИВАЕТСЯ (это
    осознанный выбор человека); сбой загрузки или счёта НЕ ломает запрос: поиск
    честно откатывается на признаки, а причина уходит в диагностику чата.

ЧЕГО ЗДЕСЬ НЕТ: обращений к LLM. Cross-encoder — локальная модель эмбеддингового
класса (`sentence_transformers.CrossEncoder`), сеть нужна только на загрузку
модели один раз, и только по явному выбору бэкенда.
"""

import logging
import math
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

from app.ai import rag_store

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Выбор бэкенда и модель
# ---------------------------------------------------------------------------
BACKEND_ENV = "RAG_RERANK_BACKEND"
DEFAULT_BACKEND = "auto"
BACKENDS = ("auto", "features", "cross-encoder")
BACKEND_NAMES = {
    "auto": "авто (cross-encoder, если модель уже скачана)",
    "features": "признаки (без модели, работает всегда)",
    "cross-encoder": "cross-encoder (модель, точнее и медленнее)",
}

MODEL_ENV = "RAG_RERANK_MODEL"
# Мультиязычная (в том числе русская) модель-реранкер: 12 слоёв, ~120 МБ,
# считается на CPU за десятки миллисекунд на пару. Модели крупнее
# (bge-reranker-base, jina-reranker-v2) точнее, но в разы медленнее — их можно
# указать через RAG_RERANK_MODEL.
DEFAULT_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
CACHE_ENV = "RAG_MODEL_CACHE"
DEFAULT_CACHE = os.path.join("data", "rag", "models")
MAX_LENGTH_ENV = "RAG_RERANK_MAX_LENGTH"
DEFAULT_MAX_LENGTH = 512
BATCH_SIZE = 16
# Предохранитель: сколько пар (запрос, фрагмент) разрешено оценить моделью за
# один поиск. Пул на базу ограничен 50 фрагментами, а баз у проекта бывает
# несколько: без предела один запрос мог бы занять моделью десятки секунд.
PAIRS_ENV = "RAG_RERANK_MAX_PAIRS"
DEFAULT_MAX_PAIRS = 64
# ПРЕДОХРАНИТЕЛЬ ПО ВРЕМЕНИ. Счёт модели — единственное место поиска, которое
# может «зависнуть» (torch, память, конкуренция потоков), а поиск идёт в потоке
# ОБСЛУЖИВАНИЯ ЗАПРОСА: без предела зависший счёт означал бы зависший ответ
# агента. Поэтому счёт идёт в ОДНОМ рабочем потоке (torch не любит конкурентные
# вызовы) и с пределом ожидания: не уложились — работают признаки, а причина
# называется в диагностике.
TIMEOUT_ENV = "RAG_RERANK_TIMEOUT"
DEFAULT_TIMEOUT = 20.0

# Вес вероятности cross-encoder в итоговой оценке (в тех же единицах, что лексика).
CE_WEIGHT_ENV = "RAG_RERANK_CE_WEIGHT"
DEFAULT_CE_WEIGHT = 1.0

# ---------------------------------------------------------------------------
# Признаки (бэкенд без модели)
# ---------------------------------------------------------------------------
PHRASE_WEIGHT_ENV = "RAG_RERANK_PHRASE_WEIGHT"
DEFAULT_PHRASE_WEIGHT = 0.6
ADDRESS_WEIGHT_ENV = "RAG_RERANK_ADDRESS_WEIGHT"
DEFAULT_ADDRESS_WEIGHT = 0.25
# ШТРАФ КОРОТКОМУ ЧАНКУ БЕЗ СЛОВ ЗАПРОСА — лечение hubness: вектор короткого
# чанка лежит близко к центру облака и «похож на любой запрос». Штраф даётся
# только тем коротким чанкам, в которых НЕТ ни одного слова запроса: короткий
# фрагмент с нужными словами — законная находка (строка таблицы, правило).
SHORT_PENALTY_ENV = "RAG_RERANK_SHORT_PENALTY"
DEFAULT_SHORT_PENALTY = 0.35
SHORT_CHUNK_CHARS = 160
# Ниже этой доли слов считать, что слов запроса в чанке нет вовсе.
SHORT_PENALTY_LEXICAL = 0.05
# Сколько символов первой строки чанка считается его ЗАГОЛОВКОМ (см. _head_of).
HEAD_CHARS = 120

_MODEL_LOCK = threading.Lock()
# Один рабочий поток на все счёты модели: конкурентные вызовы torch — известный
# источник зависаний, а поиск к нам приходит из разных запросов одновременно.
_PREDICT_POOL: Any = None
_PREDICT_LOCK = threading.Lock()
# Состояние загрузки модели: имя, сама модель, была ли попытка и чем кончилась.
_MODEL_STATE: Dict[str, Any] = {"name": "", "model": None, "tried": False,
                                "download": False, "error": ""}


def _env_float(name: str, default: float, low: float, high: float) -> float:
    """Число из настроек с зажимом в границы (пусто/битое — по умолчанию)."""
    raw = (os.getenv(name) or "").strip().replace(",", ".")
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(low, min(high, value))


def _env_int(name: str, default: int, low: int, high: int) -> int:
    """Целое из настроек с зажимом в границы (битое — по умолчанию)."""
    raw = (os.getenv(name) or "").strip()
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = default
    return max(low, min(high, value))


def backend() -> str:
    """Запрошенный бэкенд реранкинга (RAG_RERANK_BACKEND)."""
    raw = (os.getenv(BACKEND_ENV) or "").strip().lower()
    return raw if raw in BACKENDS else DEFAULT_BACKEND


def backend_name(value: Any) -> str:
    """Человеческое название бэкенда (для панели и диагностики)."""
    return BACKEND_NAMES.get(str(value or "").strip().lower(), str(value or ""))


def model_name() -> str:
    """Имя модели cross-encoder (RAG_RERANK_MODEL)."""
    return (os.getenv(MODEL_ENV) or "").strip() or DEFAULT_MODEL


def model_cache_dir() -> str:
    """Каталог кэша моделей — ТОТ ЖЕ, что у эмбеддингов (внутри проекта)."""
    raw = (os.getenv(CACHE_ENV) or "").strip()
    return raw or DEFAULT_CACHE


def max_length() -> int:
    """Предел длины пары (запрос + фрагмент) в токенах для cross-encoder."""
    return _env_int(MAX_LENGTH_ENV, DEFAULT_MAX_LENGTH, 128, 2048)


def max_pairs() -> int:
    """Сколько пар разрешено оценить моделью за один поиск (предохранитель)."""
    return _env_int(PAIRS_ENV, DEFAULT_MAX_PAIRS, 1, 500)


def timeout() -> float:
    """Сколько секунд ждать счёт модели, прежде чем откатиться на признаки."""
    return _env_float(TIMEOUT_ENV, DEFAULT_TIMEOUT, 1.0, 600.0)


def _predict_pool() -> Any:
    """Единственный рабочий поток для счёта модели (создаётся при первом счёте)."""
    global _PREDICT_POOL
    with _PREDICT_LOCK:
        if _PREDICT_POOL is None:
            import concurrent.futures
            _PREDICT_POOL = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="rag-rerank")
        return _PREDICT_POOL


def _predict(model: Any, pairs: List[Any]) -> Any:
    """Счёт модели с ПРЕДЕЛОМ ОЖИДАНИЯ (сбой по времени — не зависший запрос).

    Возвращает список оценок либо строку с причиной отказа: вызывающий в этом
    случае работает признаками. Поток, который не уложился, убить нельзя (Python
    не умеет), но ответ агента от него больше не зависит: он получит признаки и
    честную причину, а счёт помечается сбойным — следующий запрос не встанет в
    очередь за тем же зависанием.
    """
    import concurrent.futures
    future = _predict_pool().submit(model.predict, pairs, batch_size=BATCH_SIZE,
                                    show_progress_bar=False)
    try:
        return future.result(timeout=timeout())
    except concurrent.futures.TimeoutError:
        return "счёт модели не уложился в %.0f с — работают признаки" % timeout()
    except Exception as exc:          # сбой счёта не должен ломать поиск
        return "%s: %s" % (type(exc).__name__, str(exc)[:300])


def ce_weight() -> float:
    """Вес вероятности cross-encoder в итоговой оценке фрагмента."""
    return _env_float(CE_WEIGHT_ENV, DEFAULT_CE_WEIGHT, 0.0, 5.0)


def phrase_weight() -> float:
    """Вес совпадения ФРАЗЫ (порядка слов) в оценке признакового реранкинга."""
    return _env_float(PHRASE_WEIGHT_ENV, DEFAULT_PHRASE_WEIGHT, 0.0, 5.0)


def address_weight() -> float:
    """Вес совпадения слов запроса с ЗАГОЛОВКОМ и РАЗДЕЛОМ фрагмента."""
    return _env_float(ADDRESS_WEIGHT_ENV, DEFAULT_ADDRESS_WEIGHT, 0.0, 5.0)


def short_penalty() -> float:
    """Штраф короткому чанку БЕЗ слов запроса (лечение hubness)."""
    return _env_float(SHORT_PENALTY_ENV, DEFAULT_SHORT_PENALTY, 0.0, 5.0)


def settings(raw: Any = None) -> Dict[str, Any]:
    """Настройки РЕРАНКИНГА к рабочему виду (настройка проекта поверх окружения)."""
    data = raw if isinstance(raw, dict) else {}
    value = str(data.get("rerank_backend") or "").strip().lower()
    return {"rerank_backend": value if value in BACKENDS else backend(),
            "ce_weight": ce_weight()}


# ---------------------------------------------------------------------------
# Бэкенд признаков
# ---------------------------------------------------------------------------
def rerank_features(hits: List[Dict[str, Any]], query_text: str) -> List[Dict[str, Any]]:
    """Пересчёт пула ДОБАВКАМИ ОТ СЕРЕДИНЫ ПУЛА — признаки без модели.

    Первый этап считает косинус и долю слов запроса — это дёшево и работает по
    ВСЕЙ базе, но слепо к порядку слов и к адресу чанка. Здесь пул уже небольшой
    (десятки чанков), поэтому можно позволить себе фразу, адрес и штраф за
    hubness (см. описание в шапке модуля).

    ДОБАВКА СЧИТАЕТСЯ ОТ СЕРЕДИНЫ ПУЛА, А НЕ ОТ НУЛЯ. Это не украшение, а
    лечение измеренного перекоса: живой замер (tools/check_rag_quality.py, вопрос
    про «Жизненный путь» и братьев/сестёр) показал, что у ПОЧТИ ВСЕХ кандидатов
    пула фраза совпадала одинаково (1 биграмма из 6) и адрес тоже (2 слова из 7).
    Признак, общий для всего пула, фрагменты НЕ различает, но, будучи прибавленным
    ко всем, перетасовывал порядок первого этапа: нужный чанк уезжал с третьего
    места на шестое и выпадал из выдачи. Поэтому в оценку идёт только ПРЕВЫШЕНИЕ
    над серединой пула: признак, одинаковый у всех, не меняет ничего, а необычный —
    поднимает фрагмент. Направление только вверх: штрафовать за «средний» адрес
    не за что.

    Итоговая оценка остаётся в тех же единицах, что и раньше (косинус + лексика +
    добавки − штраф), поэтому порог, карточки источников и разбор оценки в
    интерфейсе продолжают работать без изменений; слагаемые видны в полях
    `phrase` и `address`, а снятый штраф — в `penalty`.
    """
    p_weight = phrase_weight()
    a_weight = address_weight()
    penalty = short_penalty()
    measured: List[Dict[str, Any]] = []
    for hit in hits:
        item = dict(hit)
        text = str(item.get("text") or "")
        lexical = float(item.get("lexical") or 0.0)
        # РЕЛЕВАНТНОСТЬ ПЕРВИЧНОГО ПОИСКА (косинус + доля слов запроса) — то, с
        # чем фрагмент пришёл из первого этапа. Её видно в интерфейсе рядом с
        # оценкой модели: живой замер показал, что по ней одной мусор (0,85) не
        # отличить от нужного (0,73), а оценка модели их разводит (0,003 и 0,78).
        item["base_score"] = round(float(item.get("score") or 0.0), 6)
        item["phrase"] = round(_phrase_score(text, query_text), 6) if p_weight else 0.0
        item["address"] = round(_address_score(item, query_text), 6) if a_weight else 0.0
        item["penalty"] = round(
            penalty if (penalty and len(text) < SHORT_CHUNK_CHARS
                        and lexical <= SHORT_PENALTY_LEXICAL) else 0.0, 6)
        item["ce"] = 0.0
        measured.append(item)
    phrase_base = _median([item["phrase"] for item in measured])
    address_base = _median([item["address"] for item in measured])
    for item in measured:
        bonus = (p_weight * max(0.0, float(item["phrase"]) - phrase_base)
                 + a_weight * max(0.0, float(item["address"]) - address_base))
        item["score"] = round(float(item.get("score") or 0.0) + bonus
                              - float(item["penalty"] or 0.0), 6)
    measured.sort(key=lambda entry: float(entry.get("score") or 0.0), reverse=True)
    return measured


def _median(values: List[float]) -> float:
    """Середина набора (пустой — ноль): опора для добавок признакового бэкенда."""
    numbers = sorted(float(value or 0.0) for value in values)
    if not numbers:
        return 0.0
    middle = len(numbers) // 2
    if len(numbers) % 2:
        return numbers[middle]
    return (numbers[middle - 1] + numbers[middle]) / 2.0


def _phrase_score(text: Any, query_text: Any) -> float:
    """Доля БИГРАММ запроса, найденных в тексте ПОДРЯД (0…1).

    Слова запроса, стоящие в чанке рядом и в том же порядке, — сильный признак
    «здесь про это»: «резервное копирование базы» отвечает на вопрос про
    резервное копирование, а чанк, где «копирование» и «база» встретились в
    разных абзацах, — скорее нет. Первый этап этого не видит: доля слов считается
    без порядка.

    Сравнение основ — тем же правилом, что у лексической части (`rag_store.same_stem`),
    иначе оценки двух этапов нельзя было бы складывать.
    """
    words = rag_store.content_stems(query_text)
    if len(words) < 2:
        return 0.0
    pairs: List[Any] = []
    for index in range(len(words) - 1):
        pair = (words[index], words[index + 1])
        if pair not in pairs:
            pairs.append(pair)
    body = rag_store.content_stems(text)
    if len(body) < 2:
        return 0.0
    found = 0
    for left, right in pairs:
        for index in range(len(body) - 1):
            if rag_store.same_stem(left, body[index]) \
                    and rag_store.same_stem(right, body[index + 1]):
                found += 1
                break
    return found / len(pairs)


def _address_score(hit: Dict[str, Any], query_text: Any) -> float:
    """Доля слов запроса, найденных в ЗАГОЛОВКЕ, РАЗДЕЛЕ и имени файла (0…1).

    Адрес фрагмента — это то, чем он называет САМ СЕБЯ. Кроме раздела и имени
    документа сюда входит СОБСТВЕННАЯ первая строка чанка: разбиение по структуре
    ставит заголовок раздела в начало текста, и у чанка «БРАТЬЯ И СЁСТРЫ» раздел в
    метаданных — родительский («ПРОВЕДЁННОЕ ДЕТСТВО»), а своё название он носит в
    тексте. Без первой строки признак терял именно этот случай (живой замер: чанк
    получал адрес 0, хотя назывался почти так же, как заданный вопрос).
    """
    words = list(dict.fromkeys(rag_store.content_stems(query_text)))
    if not words:
        return 0.0
    place = rag_store.content_stems("%s %s %s %s" % (
        _head_of(hit.get("text")), hit.get("title") or "",
        hit.get("section") or "", hit.get("source") or ""))
    if not place:
        return 0.0
    matched = 0
    for stem in words:
        if any(rag_store.same_stem(stem, other) for other in place):
            matched += 1
    return matched / len(words)


def _head_of(text: Any, limit: int = HEAD_CHARS) -> str:
    """Первая непустая строка текста чанка — его собственный заголовок."""
    for line in str(text or "").splitlines():
        line = line.strip()
        if line:
            return line[:limit]
    return ""


# ---------------------------------------------------------------------------
# Бэкенд cross-encoder
# ---------------------------------------------------------------------------
def sbert_installed() -> bool:
    """Есть ли пакет sentence-transformers (без него cross-encoder недоступен)."""
    import importlib.util
    try:
        return importlib.util.find_spec("sentence_transformers") is not None
    except (ImportError, ValueError):
        return False


def _configure_hf_env(cache: str) -> None:
    """Уводит кэш HuggingFace В КАТАЛОГ ПРОЕКТА (как у эмбеддингов).

    Библиотека по умолчанию пишет в `~/.cache/huggingface`, а часть файлов качает
    через «xet»-клиент со своим каталогом вне дерева проекта: в ограниченном
    окружении такая запись запрещена, и загрузка модели падала бы с «Operation not
    permitted». Явно заданные человеком HF_* НЕ перезаписываются.
    """
    values = {
        "HF_HOME": cache,
        "HF_HUB_CACHE": os.path.join(cache, "hub"),
        "HF_XET_CACHE": os.path.join(cache, "xet"),
        "HF_HUB_DISABLE_XET": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "SENTENCE_TRANSFORMERS_HOME": cache,
    }
    for key, value in values.items():
        os.environ.setdefault(key, value)


def _model_is_cached(cache: str, name: str) -> bool:
    """Лежит ли В КЭШЕ именно ЭТА модель (по каталогу-снапшоту HuggingFace).

    Проверяется ИМЯ модели, а не «есть ли в кэше хоть что-то»: в проекте уже
    лежит модель эмбеддингов, и общая проверка сказала бы «модель есть» про
    реранкер, которого нет. Смотрим два места: снапшоты кладутся и прямо в
    каталог кэша, и в подкаталог `hub`.
    """
    slug = "models--" + str(name).replace("/", "--")
    for root in (cache, os.path.join(cache, "hub")):
        snapshots = os.path.join(root, slug, "snapshots")
        try:
            if os.path.isdir(snapshots) and os.listdir(snapshots):
                return True
        except OSError:
            continue
    return False


def model_cached() -> bool:
    """Скачана ли модель реранкера в кэш проекта (без загрузки в память)."""
    return _model_is_cached(model_cache_dir(), model_name())


def model_ready() -> bool:
    """Готова ли модель к работе ПРЯМО СЕЙЧАС: пакет есть и модель лежит в кэше.

    Именно этим условием стережётся ФИЛЬТРАЦИЯ по уверенности: без модели
    вероятностей нет, и «фильтровать по порогу» было бы нечем — поэтому включение
    галочки без модели обязано быть ОШИБКОЙ, а не тихой подменой шкалы.
    """
    return sbert_installed() and model_cached()


def scoring_available() -> bool:
    """Может ли МОДЕЛЬ оценить пул прямо сейчас — БЕЗ выхода в сеть.

    Отдельно от `filter_available`: та отвечает на вопрос «чем фильтровать при
    таком выборе бэкенда» (признаковый бэкенд вероятностей не даёт — фильтровать
    нечем), а эта — «есть ли вообще модель, которой можно посчитать уверенность».
    Нужна там, где вероятности обязаны быть НЕ по выбору человека, а по
    требованию порога: порог уверенности модели задан — значит пул обязан
    посчитать именно модель (см. `rag_search.search`). Условие ровно то же, что у
    фильтрации: пакет установлен и модель лежит в кэше проекта. Скачивать модель
    «на всякий случай» нельзя — это делается только явным выбором бэкенда.
    """
    return model_ready()


def filter_available(backend_value: Any = None) -> bool:
    """Можно ли включать фильтрацию по уверенности модели (и почему нет)."""
    requested = str(backend_value or backend()).strip().lower()
    if requested == "features":
        return False
    return model_ready()


def filter_reason(backend_value: Any = None) -> str:
    """Почему фильтрацию включать нельзя (пусто — можно)."""
    requested = str(backend_value or backend()).strip().lower()
    if not sbert_installed():
        return "пакет sentence-transformers не установлен — модель-реранкер недоступна"
    if not model_cached():
        return ("модель %s не скачана в кэш (%s): выберите бэкенд «cross-encoder» "
                "и выполните поиск — модель загрузится один раз"
                % (model_name(), model_cache_dir()))
    if requested == "features":
        return ("выбран признаковый реранкинг («признаки»): вероятностей модели нет — "
                "выберите «авто» или «cross-encoder»")
    return ""


def _load_model(allow_download: bool) -> Tuple[Any, str]:
    """Загружает модель реранкера ОДИН РАЗ за процесс. Возвращает (модель, причина).

    Причина не бросается наружу: решение «работать признаками» принимает
    вызывающий код, а здесь важно честно назвать, почему модели нет (пакет не
    установлен, модель не скачана, сеть недоступна, не хватило памяти).

    `allow_download=False` (режим `auto`) НЕ ходит в сеть: поиск не должен
    внезапно скачивать сотни мегабайт. Скачивание — только по явному выбору
    бэкенда `cross-encoder` в панели проекта.
    """
    name = model_name()
    cache = model_cache_dir()
    with _MODEL_LOCK:
        state = _MODEL_STATE
        if state["model"] is not None and state["name"] == name:
            return state["model"], ""
        if state["tried"] and state["name"] == name and state["error"] \
                and (state["download"] or not allow_download):
            return None, state["error"]
        if not sbert_installed():
            reason = ("пакет sentence-transformers не установлен — cross-encoder "
                      "недоступен, работают признаки")
            state.update({"tried": True, "name": name, "model": None,
                          "download": allow_download, "error": reason})
            return None, reason
        cached = _model_is_cached(cache, name)
        if not cached and not allow_download:
            # Причина короткая: она попадает в СТРОКУ ДЕБАГА каждого поиска, а
            # длинное объяснение с путями живёт в панели (см. status).
            reason = "модель кросс-энкодера не скачана — работают признаки"
            state.update({"tried": True, "name": name, "model": None,
                          "download": False, "error": reason})
            return None, reason
        state.update({"tried": True, "name": name, "error": ""})
        try:
            try:
                os.makedirs(cache, exist_ok=True)
            except OSError:
                pass
            _configure_hf_env(cache)
            if cached:
                # Модель уже на диске: офлайн-режим избавляет от обращения к
                # HuggingFace (сеть может быть недоступна, и запрос «повис» бы
                # на таймауте там, где всё нужное лежит рядом).
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from sentence_transformers import CrossEncoder
            model = CrossEncoder(name, cache_folder=cache,
                                 max_length=max_length())
            state.update({"model": model, "download": allow_download, "error": ""})
            logger.info("RAG: реранкер cross-encoder загружен (%s)", name)
            return model, ""
        except Exception as exc:      # сеть, память, несовместимость версий
            reason = "%s: %s" % (type(exc).__name__, str(exc)[:300])
            state.update({"model": None, "download": allow_download, "error": reason})
            logger.warning("RAG: модель реранкера %s недоступна — %s", name, reason)
            return None, reason


def reset_state() -> None:
    """Забыть состояние загрузки модели (кнопка «обновить» в диалоге).

    Нужна после того, как человек скачал модель или починил окружение: иначе
    запомненный сбой держался бы до перезапуска приложения.
    """
    with _MODEL_LOCK:
        _MODEL_STATE.update({"tried": False, "error": "", "model": None})


def _probabilities(values: Any) -> List[float]:
    """Оценки модели к вероятностям (0…1).

    `CrossEncoder.predict` применяет функцию активации из конфигурации модели,
    поэтому одни модели отдают уже вероятность, а другие — «сырые» логиты
    (примерно −10…+10). Различить их можно по диапазону: значения вне [0, 1]
    однозначно логиты, и тогда применяется сигмоида. Иначе оценка реранкера
    складывалась бы с лексикой в разных единицах.
    """
    numbers = [float(value) for value in values]
    if numbers and (min(numbers) < 0.0 or max(numbers) > 1.0):
        return [1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))
                for value in numbers]
    return [max(0.0, min(1.0, value)) for value in numbers]


def rerank_cross_encoder(hits: List[Dict[str, Any]], query_text: str,
                         allow_download: bool) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """Пересчёт пула моделью. Возвращает (фрагменты, причина отказа).

    Причина непустая — модель не посчитала, и вызывающий обязан откатиться на
    признаки: поиск не может остаться без второго этапа из-за недоступной модели.

    Оценка фрагмента = первый этап + вес × вероятность модели − штраф за hubness.
    Штраф остаётся и здесь: короткий чанк без слов запроса врёт вектором, и
    модель это ловит, но пусть оценка об этом говорит прямо.
    """
    model, reason = _load_model(allow_download)
    if model is None:
        return None, reason
    texts = [str(hit.get("text") or "")[:max_length() * 4] for hit in hits]
    values = _predict(model, [(query_text, text) for text in texts])
    if isinstance(values, str):
        reason = values
        with _MODEL_LOCK:
            # Запоминаем сбой, чтобы СЛЕДУЮЩИЙ запрос не ждал то же зависание:
            # причина честно уходит в диагностику, поиск работает признаками.
            _MODEL_STATE.update({"tried": True, "name": model_name(),
                                 "model": None, "download": True, "error": reason})
        logger.warning("RAG: cross-encoder не посчитал пул — %s", reason)
        return None, reason
    probabilities = _probabilities(values)
    if len(probabilities) != len(hits):
        return None, "модель вернула не столько оценок, сколько было пар"
    penalty = short_penalty()
    weight = ce_weight()
    ranked: List[Dict[str, Any]] = []
    for hit, probability in zip(hits, probabilities):
        item = dict(hit)
        item["base_score"] = round(float(item.get("score") or 0.0), 6)
        text = str(item.get("text") or "")
        lexical = float(item.get("lexical") or 0.0)
        cut = penalty if (penalty and len(text) < SHORT_CHUNK_CHARS
                          and lexical <= SHORT_PENALTY_LEXICAL) else 0.0
        item["phrase"] = 0.0
        item["address"] = 0.0
        item["ce"] = round(float(probability), 6)
        item["penalty"] = round(cut, 6)
        item["score"] = round(float(item.get("score") or 0.0)
                              + weight * float(probability) - cut, 6)
        ranked.append(item)
    ranked.sort(key=lambda entry: float(entry.get("score") or 0.0), reverse=True)
    return ranked, ""


# ---------------------------------------------------------------------------
# Единая точка входа: чем реранкить
# ---------------------------------------------------------------------------
def rerank(hits: List[Dict[str, Any]], query_text: str, options: Any = None
           ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """ВТОРОЙ ЭТАП: переставляет пул кандидатов выбранным бэкендом.

    Возвращает (фрагменты, сведения): `{"backend", "model", "reason", "pairs"}`.
    Сведения уходят в диагностику чата: человек должен видеть, ЧЕМ реранкили и
    почему вышло не то, что он выбрал (модель не скачана, сбой счёта).
    """
    requested = str((options or {}).get("rerank_backend") or backend()).strip().lower()
    if requested not in BACKENDS:
        requested = backend()
    if requested == "features":
        return rerank_features(hits, query_text), {
            "backend": "features", "model": "", "reason": "",
            "pairs": len(hits)}
    # Явный выбор cross-encoder разрешает загрузку модели; `auto` — только кэш.
    allow_download = requested == "cross-encoder"
    pool = hits[:max_pairs()]
    ranked, reason = rerank_cross_encoder(pool, query_text, allow_download)
    if ranked is not None:
        info = {"backend": "cross-encoder", "model": model_name(), "reason": "",
                "pairs": len(pool)}
        if len(pool) < len(hits):
            info["reason"] = ("моделью оценено %d фрагментов из %d (предел "
                              "RAG_RERANK_PAIRS)" % (len(pool), len(hits)))
        return ranked, info
    # ОТКАТ: модель недоступна — работают признаки, а причина честно называется.
    return rerank_features(hits, query_text), {
        "backend": "features", "model": model_name(), "reason": reason,
        "pairs": 0}


def status(force: bool = False) -> Dict[str, Any]:
    """Состояние реранкинга для диалога «База знаний» (модель НЕ загружается).

    `force` — кнопка «обновить»: забыть запомненный сбой загрузки и проверить
    кэш заново (человек мог скачать модель только что).
    """
    if force:
        reset_state()
    requested = backend()
    name = model_name()
    cached = model_cached()
    installed = sbert_installed()
    available = installed and (cached or requested == "cross-encoder")
    reason = ""
    if not installed:
        reason = "пакет sentence-transformers не установлен — работает признаковый реранкинг"
    elif not cached and requested == "auto":
        reason = ("модель %s не скачана: работает признаковый реранкинг (выберите "
                  "«cross-encoder», чтобы загрузить её один раз)" % name)
    elif not cached and requested == "cross-encoder":
        reason = "модель %s будет скачана при первом поиске" % name
    effective = "cross-encoder" if (available and requested != "features") else "features"
    return {
        "requested": requested,
        "requested_name": backend_name(requested),
        "backend": effective,
        "backend_name": backend_name(effective),
        "model": name,
        "cached": cached,
        "installed": installed,
        "available": available,
        "reason": reason,
        "cache_dir": model_cache_dir(),
        "ce_weight": ce_weight(),
        "max_pairs": max_pairs(),
        "timeout": timeout(),
        "max_length": max_length(),
        "backends": [{"id": key, "name": BACKEND_NAMES[key]} for key in BACKENDS],
    }
