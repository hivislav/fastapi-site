"""Эмбеддинги для RAG: подключаемые бэкенды и честный запасной путь.

Задача модуля — превратить текст чанка в вектор чисел. КАКОЙ моделью это
делается, решает бэкенд; у проекта их два, и они взаимозаменяемы:

- **sentence-transformers** — настоящие семантические эмбеддинги (модель по
  умолчанию `paraphrase-multilingual-MiniLM-L12-v2`, 384 числа: она понимает
  русский, поэтому «автомобиль» и «машина» оказываются рядом). Модель
  скачивается один раз в ЛОКАЛЬНЫЙ кэш (`RAG_MODEL_CACHE`, по умолчанию
  `data/rag/models`) и дальше работает без сети.
- **hashing** — встроенный запасной бэкенд БЕЗ зависимостей и без сети:
  подписанное хеширование слов, биграмм и символьных 4-грамм в вектор заданной
  размерности (`RAG_EMBED_DIM`, по умолчанию 512) с весами tf и L2-нормой.
  Это лексическая близость (общие слова и куски слов), а не семантика; но она
  детерминирована, считается на stdlib и работает всегда — поэтому пайплайн
  индексации не может «не запуститься» из-за отсутствия модели.

ПОЧЕМУ БЭКЕНД НУЖЕН ВЫБИРАЕМЫЙ: venv проекта — Python 3.9, а свежие версии
sentence-transformers требуют 3.10+. Когда модель недоступна (нет пакета, нет
сети для первой загрузки, не хватило памяти), индексация не должна падать:
режим `auto` откатывается на `hashing` и ПИШЕТ ПРИЧИНУ в результат — в
интерфейсе и в метаданных базы видно, чем именно она посчитана.

СОВМЕСТИМОСТЬ ВАЖНА: у каждой базы знаний в метаданных лежат `backend`,
`model` и `dim`. Вектора разных бэкендов и размерностей несравнимы, поэтому
`check_compatible()` отказывается считать близость по чужой базе, а не выдаёт
мусорные результаты.

Сеть здесь только на скачивание модели; сам расчёт — локальный.
"""

import hashlib
import importlib.util
import logging
import math
import os
import re
import struct
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Бэкенды
# ---------------------------------------------------------------------------
BACKEND_SBERT = "sentence-transformers"
BACKEND_HASHING = "hashing"
BACKEND_AUTO = "auto"

BACKEND_IDS = [BACKEND_AUTO, BACKEND_SBERT, BACKEND_HASHING]

BACKEND_NAMES = {
    BACKEND_SBERT: "sentence-transformers (семантические эмбеддинги)",
    BACKEND_HASHING: "встроенный офлайн (хеширование слов и n-грамм)",
    BACKEND_AUTO: "автоматически",
}

# Переменные окружения — настройка ПРОЕКТА, а не пользователя: модель одна на
# всё приложение, потому что каждая база помнит, чем посчитана.
BACKEND_ENV = "RAG_EMBED_BACKEND"
MODEL_ENV = "RAG_EMBED_MODEL"
DIM_ENV = "RAG_EMBED_DIM"
CACHE_ENV = "RAG_MODEL_CACHE"

# Мультиязычная модель: русский для проекта — основной язык документов.
DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_HASH_DIM = 512

# Размерность офлайн-бэкенда: границы, чтобы вектор не был ни вырожденным, ни
# бессмысленно огромным (память базы считается этим числом).
MIN_HASH_DIM = 64
MAX_HASH_DIM = 4096

# Сколько текстов уходит в модель за один вызов encode(): батч нужен, чтобы
# большая база не превращалась в тысячи одиночных проходов по сети-модели.
EMBED_BATCH = 32
# Тексты длиннее этого обрезаются по словам: у модели окно ~512 токенов, и
# «хвост» всё равно не попадёт в вектор, а память съест.
MAX_EMBED_CHARS = 4000

_TOKEN_RE = re.compile(r"[0-9a-zа-яё]+", re.IGNORECASE)

# Состояние ленивой загрузки модели. Модель грузится ОДИН раз на процесс и
# переиспользуется всеми базами; повторная загрузка в параллельных запросах
# гасится замком (инициализация тяжёлая — секунды и сотни мегабайт).
_MODEL_LOCK = threading.Lock()
_MODEL_STATE: Dict[str, Any] = {"model": None, "name": "", "error": "", "tried": False}


def _env(name: str, default: str = "") -> str:
    """Значение переменной окружения (пустое — как отсутствующее)."""
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def requested_backend() -> str:
    """Запрошенный бэкенд из настроек: auto, sentence-transformers или hashing."""
    key = _env(BACKEND_ENV, BACKEND_AUTO).lower()
    if key in BACKEND_IDS:
        return key
    logger.info("RAG: неизвестный бэкенд эмбеддингов %r — работаю в режиме auto", key)
    return BACKEND_AUTO


def model_name() -> str:
    """Имя модели sentence-transformers (по умолчанию — мультиязычная MiniLM)."""
    return _env(MODEL_ENV, DEFAULT_MODEL)


def hash_dim() -> int:
    """Размерность встроенного офлайн-вектора (RAG_EMBED_DIM)."""
    raw = _env(DIM_ENV, str(DEFAULT_HASH_DIM))
    try:
        value = int(raw)
    except ValueError:
        value = DEFAULT_HASH_DIM
    return max(MIN_HASH_DIM, min(MAX_HASH_DIM, value))


def model_cache_dir() -> str:
    """Каталог локального кэша моделей (по умолчанию data/rag/models)."""
    default = os.path.join(_project_data_dir(), "rag", "models")
    return _env(CACHE_ENV, default)


def _project_data_dir() -> str:
    """Каталог данных проекта. Импорт config — ленивый, чтобы модуль оставался чистым."""
    try:
        from app import config  # локальный импорт: избегаем цикла на старте
        return os.path.join(config.PROJECT_ROOT, "data")
    except Exception:  # pragma: no cover - запасной путь для отдельного запуска
        return os.path.join(os.getcwd(), "data")


def sbert_installed() -> bool:
    """Установлен ли пакет sentence-transformers (без его импорта — это долго)."""
    try:
        return importlib.util.find_spec("sentence_transformers") is not None
    except (ImportError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Бэкенд sentence-transformers
# ---------------------------------------------------------------------------
def _configure_hf_env(cache: str) -> None:
    """Уводит кэш HuggingFace В КАТАЛОГ ПРОЕКТА и глушит лишние каталоги.

    Библиотека по умолчанию пишет в `~/.cache/huggingface`, а часть файлов
    качает через «xet»-клиент со СВОИМ каталогом вне дерева проекта: в
    ограниченном окружении такая запись запрещена, и загрузка модели падала с
    «Operation not permitted (os error 1)» — то есть база знаний не
    индексировалась бы вовсе. Поэтому кэш модели, кэш загрузчика и кэш xet живут
    внутри проекта, а сам xet выключен: обычный HTTPS надёжнее и не заводит
    каталогов на стороне.

    Явно заданные пользователем HF_HOME/HF_HUB_CACHE НЕ перезаписываются:
    машину настраивает человек, а не модуль.
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


def _model_is_cached(cache: str) -> bool:
    """Лежит ли модель в локальном кэше (по каталогам-снапшотам HuggingFace).

    Смотрим ДВА места: `SentenceTransformer(cache_folder=…)` кладёт снапшоты прямо
    в указанный каталог, а `HF_HUB_CACHE` (когда каталог задан через HF_HOME) —
    в подкаталог `hub`. Пропустить хоть одно значило бы решить, что модели нет,
    и полезть за ней в сеть при живом локальном кэше.
    """
    for root in (cache, os.path.join(cache, "hub")):
        if not os.path.isdir(root):
            continue
        try:
            entries = os.listdir(root)
        except OSError:
            continue
        for entry in entries:
            if not entry.startswith("models--"):
                continue
            snapshots = os.path.join(root, entry, "snapshots")
            try:
                if os.path.isdir(snapshots) and os.listdir(snapshots):
                    return True
            except OSError:
                continue
    return False


def _load_model() -> Tuple[Any, str]:
    """Загружает модель один раз за процесс. Возвращает (модель, ошибка).

    Ошибка НЕ бросается наружу: решение «откатиться на офлайн-бэкенд» принимает
    вызывающий код, а здесь важно лишь честно назвать причину (нет пакета,
    не скачалась модель, не хватило памяти).
    """
    name = model_name()
    with _MODEL_LOCK:
        if _MODEL_STATE["model"] is not None and _MODEL_STATE["name"] == name:
            return _MODEL_STATE["model"], ""
        if _MODEL_STATE["tried"] and _MODEL_STATE["name"] == name and _MODEL_STATE["error"]:
            return None, _MODEL_STATE["error"]
        _MODEL_STATE.update({"tried": True, "name": name, "error": ""})
        try:
            cache = model_cache_dir()
            try:
                os.makedirs(cache, exist_ok=True)
            except OSError:
                pass
            _configure_hf_env(cache)
            if _model_is_cached(cache):
                # Модель уже скачана: офлайн-режим избавляет от обращения к
                # HuggingFace — сеть может быть недоступна, и запрос «повиснет»
                # на таймауте там, где всё нужное лежит на диске.
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(name, cache_folder=cache)
            _MODEL_STATE["model"] = model
            return model, ""
        except Exception as exc:  # сеть, память, несовместимость версий
            reason = "%s: %s" % (type(exc).__name__, str(exc)[:300])
            _MODEL_STATE["model"] = None
            _MODEL_STATE["error"] = reason
            logger.warning("RAG: модель %s недоступна — %s", name, reason)
            return None, reason


def embedded_dim() -> Optional[int]:
    """Размерность модели, ЕСЛИ она уже загружена (иначе None).

    Снимок для интерфейса не должен тянуть загрузку модели на 500 МБ ради
    одной цифры: до первой индексации размерность просто неизвестна.
    """
    model = _MODEL_STATE.get("model")
    if model is None:
        return None
    try:
        return int(model.get_sentence_embedding_dimension())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Встроенный офлайн-бэкенд
# ---------------------------------------------------------------------------
def _features(text: str) -> Dict[str, float]:
    """Признаки текста с весами: слова, биграммы слов и символьные 4-граммы.

    Символьные n-граммы добавлены намеренно: они ловят русскую морфологию
    («установка»/«установки»/«установить») там, где точного совпадения слова нет.
    """
    tokens = _TOKEN_RE.findall(str(text or "").lower())
    feats: Dict[str, float] = {}
    for token in tokens:
        feats[token] = feats.get(token, 0.0) + 1.0
    for first, second in zip(tokens, tokens[1:]):
        key = first + " " + second
        feats[key] = feats.get(key, 0.0) + 1.0
    for token in tokens:
        if len(token) >= 4:
            for index in range(len(token) - 3):
                key = "#" + token[index:index + 4]
                feats[key] = feats.get(key, 0.0) + 0.5
    return feats


def _hash_embed_one(text: str, dim: int) -> List[float]:
    """Вектор одного текста: подписанное хеширование признаков + L2-норма.

    Хеш считается blake2b, а НЕ встроенным `hash()`: у строк он солится при
    каждом запуске процесса, и векторы перестали бы совпадать между запусками —
    то есть индекс, записанный сегодня, нельзя было бы искать завтра.
    """
    vector = [0.0] * dim
    for feature, count in _features(text).items():
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=16).digest()
        index = struct.unpack_from("<Q", digest, 0)[0] % dim
        sign = 1.0 if digest[8] & 1 else -1.0
        weight = 1.0 + math.log(count)          # сглаженная частота: не «перевес» повторов
        vector[index] += sign * weight
    norm = math.sqrt(sum(value * value for value in vector))
    if norm > 0.0:
        vector = [value / norm for value in vector]
    return vector


# ---------------------------------------------------------------------------
# Публичный интерфейс
# ---------------------------------------------------------------------------
def backend_status(force: bool = False) -> Dict[str, Any]:
    """Что за бэкенд доступен и чем он посчитает векторы (снимок для интерфейса).

    Модель здесь НЕ загружается: проверяется только наличие пакета — иначе
    открытие диалога «База знаний» тянуло бы сотни мегабайт. `force` оставлен
    для симметрии с MCP-снимком и сбрасывает кэш неудачной загрузки.
    """
    if force:
        with _MODEL_LOCK:
            _MODEL_STATE.update({"tried": False, "error": ""})
    requested = requested_backend()
    installed = sbert_installed()
    if requested == BACKEND_HASHING:
        active, reason = BACKEND_HASHING, "выбран вручную (RAG_EMBED_BACKEND=hashing)"
    elif installed:
        active, reason = BACKEND_SBERT, ""
    else:
        active = BACKEND_HASHING
        reason = ("пакет sentence-transformers не установлен — работаю встроенным "
                  "офлайн-бэкендом")
    if requested == BACKEND_AUTO and _MODEL_STATE.get("error") and installed:
        # Модель уже пытались загрузить и не смогли: интерфейс должен показать
        # это ДО индексации, а не удивлять после.
        active, reason = BACKEND_HASHING, _MODEL_STATE["error"]
    return {
        "requested": requested,
        "backend": active,
        "backend_name": BACKEND_NAMES.get(active, active),
        "model": model_name() if active == BACKEND_SBERT else "",
        "dim": embedded_dim() if active == BACKEND_SBERT else hash_dim(),
        "sbert_installed": installed,
        "reason": reason,
        "cache_dir": model_cache_dir(),
        "backends": [
            {"id": BACKEND_SBERT, "name": BACKEND_NAMES[BACKEND_SBERT],
             "available": installed,
             "description": "Семантическая близость: находит синонимы и перефраз. "
                            "Модель скачивается один раз (кэш на диске)."},
            {"id": BACKEND_HASHING, "name": BACKEND_NAMES[BACKEND_HASHING],
             "available": True,
             "description": "Лексическая близость по словам и кускам слов. "
                            "Работает всегда и без сети — запасной путь."},
        ],
    }


def embed_texts(texts: List[str], backend: Optional[str] = None,
                on_progress: Optional[Callable[[int, int], None]] = None
                ) -> Tuple[List[List[float]], Dict[str, Any]]:
    """Считает векторы для списка текстов. Возвращает (векторы, сведения).

    `backend` — принудительный бэкенд (иначе из настроек). В режиме `auto`
    недоступная модель НЕ роняет индексацию: сведения `info` содержат поле
    `fallback` с причиной, и эта причина уходит в метаданные базы — по ней потом
    видно, почему база посчитана лексически.

    Порядок векторов совпадает с порядком текстов: пайплайн кладёт их в индекс
    рядом с чанками по индексу, а не по идентификатору.
    """
    items = [str(item or "") for item in (texts or [])]
    info: Dict[str, Any] = {"fallback": "", "batches": 0}
    if not items:
        status = backend_status()
        info.update({"backend": status["backend"], "model": status["model"],
                     "dim": int(status["dim"] or 0)})
        return [], info

    chosen = (backend or "").strip().lower() or requested_backend()
    if chosen == BACKEND_AUTO:
        chosen = BACKEND_SBERT if sbert_installed() else BACKEND_HASHING
    # «Вручную» — это и настройка бэкенда, и ЯВНО переданный аргумент: молча
    # подменить модель, которую просили именно эту, нельзя (пользователь ждал
    # семантику, а получил бы лексику и не узнал об этом).
    manual = chosen == BACKEND_SBERT and (
        bool((backend or "").strip()) and (backend or "").strip().lower() != BACKEND_AUTO
        or (not (backend or "").strip() and requested_backend() == BACKEND_SBERT))

    if chosen == BACKEND_SBERT:
        model, error = _load_model()
        if model is not None:
            vectors = _encode_sbert(model, items, on_progress)
            if vectors is not None:
                info.update({
                    "backend": BACKEND_SBERT,
                    "model": model_name(),
                    "dim": len(vectors[0]) if vectors else 0,
                    "batches": _batch_count(len(items)),
                })
                return vectors, info
            info["fallback"] = "модель вернула неожиданный результат"
        else:
            info["fallback"] = error or "модель недоступна"
        if manual:
            raise RuntimeError("Бэкенд sentence-transformers недоступен: %s" % info["fallback"])
        logger.warning("RAG: откат на встроенный офлайн-бэкенд (%s)", info["fallback"])

    dim = hash_dim()
    vectors = []
    total = len(items)
    for number, text in enumerate(items):
        vectors.append(_hash_embed_one(text[:MAX_EMBED_CHARS], dim))
        if on_progress and (number % 50 == 0 or number + 1 == total):
            on_progress(number + 1, total)
    info.update({"backend": BACKEND_HASHING, "model": "", "dim": dim,
                 "batches": _batch_count(total)})
    return vectors, info


def _batch_count(total: int) -> int:
    """Сколько вызовов модели потребуется: размер батча — EMBED_BATCH."""
    return (int(total) + EMBED_BATCH - 1) // EMBED_BATCH if total else 0


def _encode_sbert(model: Any, items: List[str],
                  on_progress: Optional[Callable[[int, int], None]]) -> Optional[List[List[float]]]:
    """Прогон текстов через модель батчами. None — модель не справилась.

    Тексты режутся по MAX_EMBED_CHARS: у модели ограничено окно, и длинный
    «хвост» в вектор всё равно не попадёт, зато память он занимает.
    """
    trimmed = [item[:MAX_EMBED_CHARS] for item in items]
    total = len(trimmed)
    vectors: List[List[float]] = []
    try:
        for start in range(0, total, EMBED_BATCH):
            batch = trimmed[start:start + EMBED_BATCH]
            encoded = model.encode(batch, batch_size=EMBED_BATCH,
                                   normalize_embeddings=True, show_progress_bar=False)
            for row in encoded:
                vectors.append([float(value) for value in row])
            if on_progress:
                on_progress(min(total, start + len(batch)), total)
    except Exception as exc:
        logger.warning("RAG: сбой кодирования моделью — %s: %s",
                       type(exc).__name__, str(exc)[:200])
        return None
    return vectors


def embed_query(text: str, backend: Optional[str] = None) -> Tuple[List[float], Dict[str, Any]]:
    """Вектор одного запроса тем же бэкендом, что и база (для поиска).

    Отдельная функция нужна потому, что запрос обязан считаться ТЕМ ЖЕ
    бэкендом и той же моделью, что и чанки: вектор запроса от другой модели
    геометрически несравним с индексом. Так его и считает поиск по базам
    (app/ai/rag_search.py), беря бэкенд из паспорта базы.
    """
    vectors, info = embed_texts([text], backend=backend)
    return (vectors[0] if vectors else []), info


def embedding_fields(meta: Dict[str, Any]) -> Tuple[str, int]:
    """Бэкенд и размерность из ПАСПОРТА базы (или из плоского словаря).

    Паспорт базы хранит их вложенно (`embedding.backend` / `embedding.dim`), а
    вызовы «на лету» передают плоский словарь. Поддерживаются обе формы —
    иначе проверка совместимости читала бы пустое место и отказывала живой базе
    («в метаданных не записан бэкенд»), хотя всё записано.
    """
    data = meta if isinstance(meta, dict) else {}
    nested = data.get("embedding") if isinstance(data.get("embedding"), dict) else None
    source = nested if nested else data
    try:
        dim = int(source.get("dim") or 0)
    except (TypeError, ValueError):
        dim = 0
    return str(source.get("backend") or ""), dim


def check_compatible(meta: Dict[str, Any], backend: Optional[str] = None) -> str:
    """Проверяет, что база посчитана тем же бэкендом, что доступен сейчас.

    Возвращает "" (совместимо) или причину несовместимости. Без этой проверки
    поиск по базе, посчитанной другой моделью, выдавал бы случайные чанки —
    «нашлось, но не то», что хуже честного отказа.
    """
    stored, dim = embedding_fields(meta)
    if not stored:
        return "в метаданных базы не записан бэкенд эмбеддингов"
    status = backend_status()
    active = (backend or "").strip().lower() or status["backend"]
    if not active or active == BACKEND_AUTO:
        active = status["backend"]
    if stored != active:
        return ("база посчитана бэкендом «%s», а сейчас доступен «%s»" % (stored, active))
    current = int(status["dim"] or 0)
    if dim and current and dim != current:
        return "размерность базы %d не совпадает с текущей %d" % (dim, current)
    return ""
