"""Хранилище индекса базы знаний (RAG): SQLite (рабочее) и JSON (выгрузка).

Каталог базы знаний (`RAG_DIR`, по умолчанию `data/rag`):

    data/rag/kb-1a2b3c4d/
        meta.json      — паспорт базы: название, владелец, стратегия и её
                         параметры, чем посчитаны эмбеддинги, документы, метрики
        index.sqlite   — РАБОЧЕЕ хранилище: чанки, их метаданные и векторы
        index.json     — выгрузка индекса (то же самое в JSON)

ПОЧЕМУ ДВА ХРАНИЛИЩА, А НЕ ОДНО:

- **SQLite — источник истины для работы.** Чанков в базе бывают тысячи, и
  выбирать их надо по метаданным (`source`, `section`) и по номеру, а не
  вычитывать весь файл: индексы по колонкам даёт именно SQLite. Он в стандартной
  библиотеке, то есть не добавляет зависимостей и работает всегда.
- **JSON — переносимость и ГЛАЗА.** Индекс базы можно открыть, посмотреть
  метаданные любого чанка и унести в другой инструмент; `meta.json` вообще
  читается человеком и служит паспортом базы.

ПОЧЕМУ ЗДЕСЬ НЕТ FAISS (и когда он понадобится). Замер на этой машине: база из
400 чанков ищется перебором за 14 мс, из 5 000 — за 1,2 с, и узкое место здесь
НЕ алгоритм, а счёт близости на чистом Python (O(n·dim) умножений). Настоящий
перебор на numpy считает 100 000 чанков за 11 мс — то есть до сотен тысяч
чанков упираться не во что, и отдельная векторная библиотека ничего не добавляет.
К тому же `faiss.IndexFlatIP` — это ТОЧНЫЙ перебор (просто на C++), а не
приближённый поиск: выигрыш FAISS появляется только с IVF/HNSW, где полнота
обменивается на скорость, и это нужно на корпусах в 10⁵–10⁶ чанков. Тогда его и
стоит добавлять — осознанно, вместе с настройкой `nlist`/`nprobe` и метрикой
полноты. Ветка «если библиотека есть» такого выигрыша не даёт, зато добавляет
путь, который на машине без библиотеки НИКОГДА не исполняется и не проверяется.

ВЕКТОРЫ лежат как float32 (`array('f')`), а не как текст: 384 числа в JSON —
это несколько килобайт на чанк, в двоичном виде — те же числа вчетверо дешевле.
Порядок байт проверяется по метке в паспорте (`vector_encoding`), поэтому база,
собранная на другой машине, читается верно.

ИДЕНТИФИКАТОР БАЗЫ проверяется шаблоном, а путь собирается только из него:
имена каталогов приходят из интерфейса, и `../../etc` не должен стать путём
(та же защита, что у раздачи вложений MCP, см. app/ai/attachments.py).
"""

import array
import base64
import heapq
import importlib.util
import json
import logging
import os
import re
import shutil
import sqlite3
import sys
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Расположение и пределы
# ---------------------------------------------------------------------------
DIR_ENV = "RAG_DIR"
META_FILE = "meta.json"
SQLITE_FILE = "index.sqlite"
JSON_FILE = "index.json"

# Идентификатор базы: «kb-» + 8 шестнадцатеричных знаков. Шаблон — не украшение,
# а защита пути: всё, что ему не соответствует, до файловой системы не доходит.
ID_PREFIX = "kb-"
ID_RE = re.compile(r"^kb-[0-9a-f]{8}$")

MAX_BASES = 50
MAX_DOCS_PER_BASE = 200
MAX_CHUNKS_PER_BASE = 200_000

# Версия формата индекса. Меняется вместе со схемой таблиц: файл старой версии
# не должен читаться «как получится».
SCHEMA_VERSION = 1
VECTOR_ENCODING = "float32le"


def directory() -> str:
    """Каталог баз знаний (RAG_DIR или data/rag рядом с проектом)."""
    raw = (os.getenv(DIR_ENV) or "").strip()
    if raw:
        return raw
    try:
        from app import config
        root = config.PROJECT_ROOT
    except Exception:  # pragma: no cover - отдельный запуск модуля
        root = os.getcwd()
    return os.path.join(root, "data", "rag")


def new_id() -> str:
    """Новый идентификатор базы знаний."""
    return ID_PREFIX + uuid.uuid4().hex[:8]


def valid_id(raw: Any) -> bool:
    """Похож ли идентификатор на идентификатор базы (защита пути)."""
    return bool(ID_RE.match(str(raw or "").strip().lower()))


def _normalize_id(raw: Any) -> str:
    """Идентификатор базы в каноническом виде (пустая строка — невалидный)."""
    value = str(raw or "").strip().lower()
    return value if valid_id(value) else ""


def base_path(base_id: Any) -> str:
    """Путь каталога базы. Невалидный идентификатор → пустая строка.

    Никакой склейки путей из пользовательской строки: путь строится ТОЛЬКО из
    проверенного идентификатора, поэтому выйти за каталог баз знаний нельзя.
    """
    key = _normalize_id(base_id)
    if not key:
        return ""
    return os.path.join(directory(), key)


def _now() -> str:
    """Метка времени (ISO, до секунд) — как у задач и сессий workspace."""
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Векторы
# ---------------------------------------------------------------------------
def pack_vector(vector: Iterable[float]) -> bytes:
    """Вектор → двоичные float32 (порядок байт — little-endian, как в паспорте)."""
    values = array.array("f", [float(value) for value in vector])
    if sys.byteorder != "little":
        values.byteswap()
    return values.tobytes()


def unpack_vector(blob: bytes, dim: int = 0) -> List[float]:
    """Двоичные float32 → вектор (обратная к pack_vector операция)."""
    if not blob:
        return []
    values = array.array("f")
    values.frombytes(bytes(blob))
    if sys.byteorder != "little":
        values.byteswap()
    vector = [float(value) for value in values]
    if dim and len(vector) != int(dim):
        # Размерность не та — вектор испорчен: лучше пустой, чем «почти верный».
        logger.warning("RAG: вектор размерности %d вместо %d", len(vector), dim)
        return []
    return vector


def human_size(size: Any) -> str:
    """Размер в человеческом виде (для диалога «База знаний»)."""
    try:
        value = float(size)
    except (TypeError, ValueError):
        return "0 Б"
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if value < 1024 or unit == "ГБ":
            return ("%d %s" % (int(value), unit) if unit == "Б"
                    else "%.1f %s" % (value, unit))
        value /= 1024.0
    return "%.1f ГБ" % value


def estimate_tokens(chars: Any) -> int:
    """Оценка числа токенов по символам (для метрик базы).

    Это ОЦЕНКА (примерно 4 символа на токен для русского и английского текста) —
    в интерфейсе она так и подписана, потому что точный счёт даёт только
    токенизатор конкретной модели.
    """
    try:
        return int(round(float(chars) / 4.0))
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Чтение паспорта и списка баз
# ---------------------------------------------------------------------------
def read_meta(base_id: Any) -> Optional[Dict[str, Any]]:
    """Паспорт базы (meta.json) или None, если базы нет / файл битый."""
    folder = base_path(base_id)
    if not folder:
        return None
    path = os.path.join(folder, META_FILE)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        logger.warning("RAG: паспорт базы %s не прочитан — %s", base_id, str(exc)[:150])
        return None
    if not isinstance(data, dict):
        return None
    data["id"] = _normalize_id(data.get("id")) or _normalize_id(base_id)
    return data


def list_bases(profile: Optional[str] = None, with_meta: bool = True) -> List[Dict[str, Any]]:
    """Все базы знаний (при необходимости — только одного профиля).

    Список читается с ДИСКА, а не из отдельного реестра: паспорт базы лежит
    рядом с её индексом, и второй источник истины разошёлся бы с первым при
    первом же сбое записи. Каталоги, не похожие на базы (`models` — кэш
    моделей), пропускаются.
    """
    root = directory()
    if not os.path.isdir(root):
        return []
    bases: List[Dict[str, Any]] = []
    try:
        entries = sorted(os.listdir(root))
    except OSError:
        return []
    for entry in entries:
        if not valid_id(entry):
            continue
        meta = read_meta(entry) if with_meta else {"id": entry}
        if meta is None:
            continue
        if profile and str(meta.get("profile") or "") not in ("", str(profile)):
            continue
        bases.append(meta)
    bases.sort(key=lambda item: str(item.get("created") or ""), reverse=True)
    return bases


def get_base(base_id: Any, profile: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Одна база по идентификатору. `profile` — проверить владельца.

    Профили изолированы (как задачи и память): чужая база для профиля не
    существует, поэтому `None`, а не «есть, но не ваша».
    """
    meta = read_meta(base_id)
    if meta is None:
        return None
    if profile and str(meta.get("profile") or "") not in ("", str(profile)):
        return None
    return meta


def existing_ids(profile: Optional[str] = None) -> List[str]:
    """Идентификаторы существующих баз (по ним проверяются галочки проекта)."""
    return [str(item["id"]) for item in list_bases(profile=profile)]


# ---------------------------------------------------------------------------
# Запись индекса
# ---------------------------------------------------------------------------
def save_index(meta: Dict[str, Any], documents: List[Dict[str, Any]],
               chunks: List[Dict[str, Any]],
               vectors: List[List[float]]) -> Dict[str, Any]:
    """Записывает индекс базы целиком: SQLite (рабочее) + JSON (выгрузка).

    Векторы обязаны соответствовать чанкам ПО НОМЕРУ: пайплайн кладёт их в том
    же порядке, в каком отдал тексты на кодирование. Несовпадение длин — ошибка
    вызывающего кода, и она ловится здесь, а не превращается в индекс, где
    половина чанков ищется, а половина нет.
    """
    base_id = _normalize_id(meta.get("id"))
    if not base_id:
        raise ValueError("некорректный идентификатор базы знаний")
    if vectors and len(vectors) != len(chunks):
        raise ValueError("векторов %d, а чанков %d — индекс записан не будет"
                         % (len(vectors), len(chunks)))
    if len(chunks) > MAX_CHUNKS_PER_BASE:
        raise ValueError("чанков больше предела индексации (%d)" % MAX_CHUNKS_PER_BASE)
    folder = base_path(base_id)
    os.makedirs(folder, exist_ok=True)

    dim = len(vectors[0]) if vectors else int(
        (meta.get("embedding") or {}).get("dim") or 0)
    storage: Dict[str, Any] = {"sqlite": False, "json": False}

    _write_sqlite(folder, base_id, documents, chunks, vectors, dim)
    storage["sqlite"] = True
    _write_json(folder, meta, documents, chunks, vectors)
    storage["json"] = True

    meta = dict(meta)
    meta.update({
        "version": SCHEMA_VERSION,
        "vector_encoding": VECTOR_ENCODING,
        "storage": storage,
        "updated": _now(),
        "size_bytes": folder_size(folder),
    })
    meta["size_human"] = human_size(meta["size_bytes"])
    _atomic_json(os.path.join(folder, META_FILE), meta)
    return meta


def _write_sqlite(folder: str, base_id: str, documents: List[Dict[str, Any]],
                  chunks: List[Dict[str, Any]], vectors: List[List[float]],
                  dim: int) -> None:
    """Пишет чанки, метаданные и векторы в SQLite (перезаписывая старый индекс).

    Запись идёт в ВРЕМЕННЫЙ файл, который затем подменяет рабочий (`os.replace`):
    иначе обрыв на середине оставил бы базу с половиной чанков — и «нашлось бы»
    только то, что успело записаться.
    """
    target = os.path.join(folder, SQLITE_FILE)
    temp = target + ".tmp"
    if os.path.exists(temp):
        os.remove(temp)
    connection = sqlite3.connect(temp)
    try:
        connection.executescript(
            "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);"
            "CREATE TABLE documents ("
            "  id INTEGER PRIMARY KEY, doc_index INTEGER, source TEXT, title TEXT,"
            "  kind TEXT, format TEXT, chars INTEGER, chunks INTEGER, pages INTEGER,"
            "  warning TEXT);"
            "CREATE TABLE chunks ("
            "  id INTEGER PRIMARY KEY, chunk_id TEXT, chunk_index INTEGER,"
            "  doc_index INTEGER, position INTEGER,"
            "  source TEXT, title TEXT, section TEXT, kind TEXT, start INTEGER,"
            "  end INTEGER, chars INTEGER, text TEXT, dim INTEGER, vector BLOB);"
            "CREATE INDEX idx_chunks_chunk_id ON chunks(chunk_id);"
            "CREATE INDEX idx_chunks_source ON chunks(source);"
            "CREATE INDEX idx_chunks_section ON chunks(section);"
        )
        connection.executemany(
            "INSERT INTO meta (key, value) VALUES (?, ?)",
            [("base_id", base_id), ("dim", str(dim)),
             ("vector_encoding", VECTOR_ENCODING),
             ("schema", str(SCHEMA_VERSION)), ("updated", _now())])
        connection.executemany(
            "INSERT INTO documents (doc_index, source, title, kind, format, chars,"
            " chunks, pages, warning) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(int(item.get("doc_index") or 0), str(item.get("source") or ""),
              str(item.get("title") or ""), str(item.get("kind") or ""),
              str(item.get("format") or ""), int(item.get("chars") or 0),
              int(item.get("chunks") or 0), int(item.get("pages") or 0),
              str(item.get("warning") or "")) for item in documents])
        rows = []
        for index, chunk in enumerate(chunks):
            vector = vectors[index] if index < len(vectors) else []
            rows.append((
                str(chunk.get("chunk_id") or ""),
                # Сквозной номер чанка в базе: без него после чтения нельзя
                # понять, в каком порядке чанки лежат в индексе, а порядок нужен
                # и поиску, и показу места чанка в базе.
                int(chunk.get("index", index)),
                int(chunk.get("doc_index") or 0),
                int(chunk.get("position") or 0), str(chunk.get("source") or ""),
                str(chunk.get("title") or ""), str(chunk.get("section") or ""),
                str(chunk.get("kind") or ""), int(chunk.get("start") or 0),
                int(chunk.get("end") or 0), int(chunk.get("chars") or 0),
                str(chunk.get("text") or ""), len(vector) or dim,
                pack_vector(vector) if vector else None))
        connection.executemany(
            "INSERT INTO chunks (chunk_id, chunk_index, doc_index, position, source,"
            " title, section, kind, start, end, chars, text, dim, vector)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
        connection.commit()
    finally:
        connection.close()
    os.replace(temp, target)


def _write_json(folder: str, meta: Dict[str, Any], documents: List[Dict[str, Any]],
                chunks: List[Dict[str, Any]], vectors: List[List[float]]) -> None:
    """Пишет выгрузку индекса в JSON (человекочитаемый вид базы).

    Векторы округляются до 6 знаков: полная точность float32 в текстовом виде
    раздувает файл в разы, а для просмотра и переноса шести знаков достаточно.
    """
    payload = {
        "version": SCHEMA_VERSION,
        "vector_encoding": VECTOR_ENCODING,
        "meta": {key: value for key, value in meta.items() if key != "storage"},
        "documents": documents,
        "chunks": [],
    }
    for index, chunk in enumerate(chunks):
        item = {key: value for key, value in chunk.items() if key != "text"}
        item["text"] = chunk.get("text") or ""
        vector = vectors[index] if index < len(vectors) else []
        item["vector"] = [round(float(value), 6) for value in vector]
        payload["chunks"].append(item)
    path = os.path.join(folder, JSON_FILE)
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
    except OSError as exc:
        logger.warning("RAG: выгрузка JSON не записана — %s", str(exc)[:150])


def _atomic_json(path: str, payload: Dict[str, Any]) -> None:
    """Атомарная запись JSON (как workspace: через временный файл и os.replace)."""
    temp = path + ".tmp"
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temp, path)


def folder_size(folder: str) -> int:
    """Суммарный размер файлов базы («вес» базы знаний на диске)."""
    total = 0
    try:
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if os.path.isfile(path):
                total += os.path.getsize(path)
    except OSError:
        return 0
    return total


# ---------------------------------------------------------------------------
# Чтение индекса
# ---------------------------------------------------------------------------
def load_chunks(base_id: Any, with_vectors: bool = False,
                limit: int = 0) -> List[Dict[str, Any]]:
    """Чанки базы из SQLite (с векторами или без).

    Читаем из SQLite, а не из JSON: у больших баз JSON-выгрузка в разы больше и
    разбирается целиком в память, тогда как здесь можно взять срез.
    """
    folder = base_path(base_id)
    path = os.path.join(folder, SQLITE_FILE) if folder else ""
    if not path or not os.path.isfile(path):
        return []
    try:
        connection = sqlite3.connect("file:%s?mode=ro" % path, uri=True)
    except sqlite3.Error as exc:
        logger.warning("RAG: индекс %s не открыт — %s", base_id, str(exc)[:150])
        return []
    try:
        columns = ("chunk_id, chunk_index, doc_index, position, source, title,"
                   " section, kind, start, end, chars, text, dim")
        if with_vectors:
            columns += ", vector"
        sql = "SELECT %s FROM chunks ORDER BY chunk_index, id" % columns
        if limit:
            sql += " LIMIT %d" % int(limit)
        rows = connection.execute(sql).fetchall()
    except sqlite3.Error as exc:
        logger.warning("RAG: чанки %s не прочитаны — %s", base_id, str(exc)[:150])
        return []
    finally:
        connection.close()
    chunks: List[Dict[str, Any]] = []
    for row in rows:
        item = {
            "chunk_id": row[0], "index": row[1], "doc_index": row[2],
            "position": row[3], "source": row[4], "title": row[5], "section": row[6],
            "kind": row[7], "start": row[8], "end": row[9], "chars": row[10],
            "text": row[11], "dim": row[12],
        }
        if with_vectors:
            item["vector"] = unpack_vector(row[13], int(row[12] or 0))
        chunks.append(item)
    return chunks


def chunks_page(base_id: Any, offset: int = 0, limit: int = 10,
                source: str = "", query: str = "") -> Tuple[List[Dict[str, Any]], int]:
    """Страница чанков базы для просмотра: (чанки, всего подходящих).

    Просмотр чанков — это то, чем пользователь проверяет, годится ли выбранная
    стратегия и размер: глазами видно, что раздел не разорван и что чанк не
    состоит из одного заголовка. Поэтому нужны и срез по документу, и поиск по
    тексту, и постраничная выдача — база на 20 000 чанков целиком в диалог не
    поместится.

    Поиск по тексту идёт В SQLite, а не перебором в Python: сравнение регистра
    делает ЗАРЕГИСТРИРОВАННАЯ питоновская функция `py_lower`, потому что
    встроенные `LIKE`/`lower()` в SQLite понимают только латиницу — на русском
    поиск «насос» не нашёл бы «Насос».
    """
    folder = base_path(base_id)
    path = os.path.join(folder, SQLITE_FILE) if folder else ""
    if not path or not os.path.isfile(path):
        return [], 0
    where: List[str] = []
    params: List[Any] = []
    source = str(source or "").strip()
    query = str(query or "").strip()
    if source:
        where.append("source = ?")
        params.append(source)
    if query:
        where.append("py_lower(text) LIKE ?")
        params.append("%" + query.lower() + "%")
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    size = max(1, min(100, int(limit or 1)))
    start = max(0, int(offset or 0))
    try:
        connection = sqlite3.connect("file:%s?mode=ro" % path, uri=True)
        try:
            connection.create_function("py_lower", 1,
                                       lambda value: str(value or "").lower(),
                                       deterministic=True)
            total = int(connection.execute(
                "SELECT COUNT(*) FROM chunks" + clause, params).fetchone()[0])
            rows = connection.execute(
                "SELECT chunk_id, chunk_index, doc_index, position, source, title,"
                " section, kind, start, end, chars, text FROM chunks" + clause
                + " ORDER BY chunk_index, id LIMIT ? OFFSET ?",
                params + [size, start]).fetchall()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        logger.warning("RAG: страница чанков %s не прочитана — %s", base_id, str(exc)[:150])
        return [], 0
    chunks = [{
        "chunk_id": row[0], "index": row[1], "doc_index": row[2], "position": row[3],
        "source": row[4], "title": row[5], "section": row[6], "kind": row[7],
        "start": row[8], "end": row[9], "chars": row[10], "text": row[11],
    } for row in rows]
    return chunks, total


def load_meta_from_sqlite(base_id: Any) -> Dict[str, str]:
    """Служебные записи из SQLite (id, размерность, версия схемы)."""
    folder = base_path(base_id)
    path = os.path.join(folder, SQLITE_FILE) if folder else ""
    if not path or not os.path.isfile(path):
        return {}
    try:
        connection = sqlite3.connect("file:%s?mode=ro" % path, uri=True)
        try:
            rows = connection.execute("SELECT key, value FROM meta").fetchall()
        finally:
            connection.close()
    except sqlite3.Error:
        return {}
    return {str(key): str(value) for key, value in rows}


def present_files(base_id: Any) -> Dict[str, bool]:
    """Какие файлы индекса реально есть на диске.

    Имя отличается от `rag.index_files` (сборка базы) намеренно: здесь —
    ПРОВЕРКА, что лежит в каталоге базы, а не индексация документов.
    """
    folder = base_path(base_id)
    if not folder:
        return {}
    return {
        "meta": os.path.isfile(os.path.join(folder, META_FILE)),
        "sqlite": os.path.isfile(os.path.join(folder, SQLITE_FILE)),
        "json": os.path.isfile(os.path.join(folder, JSON_FILE)),
    }


# ---------------------------------------------------------------------------
# Поиск (задел под подключение RAG к агенту)
# ---------------------------------------------------------------------------
def search(base_id: Any, query_vector: List[float], top_k: int = 5,
           profile: Optional[str] = None) -> List[Dict[str, Any]]:
    """Ближайшие чанки базы по вектору запроса. Возвращает список попаданий.

    Вектор запроса обязан быть посчитан ТЕМ ЖЕ бэкендом, что и база (это
    проверяет `rag_embedding.check_compatible`): иначе близость бессмысленна.

    Считается косинусная близость — обычное скалярное произведение: векторы
    нормированные, поэтому угол между ними и есть их произведение. Агент этим
    пока НЕ пользуется: задача дня — индексация, а поиск здесь как готовая
    точка подключения.
    """
    if not valid_id(base_id):
        return []
    meta = get_base(base_id, profile=profile)
    if meta is None:
        return []
    dim = int((meta.get("embedding") or {}).get("dim") or 0) or len(query_vector)
    if not query_vector:
        return []
    vector = [float(value) for value in query_vector]
    limit = max(1, min(50, int(top_k or 5)))
    return _search_scan(base_id, vector, limit, dim)


def _search_scan(base_id: Any, vector: List[float], limit: int,
                 dim: int) -> List[Dict[str, Any]]:
    """Перебор векторов базы: близость каждого чанка к запросу.

    Сложность линейная (O(n·dim)), и на этом масштабе она и есть правильная: у
    базы знаний пользователя сотни–десятки тысяч чанков, а точный перебор
    считается миллисекунды (замер: 400 чанков — 5 мс, 100 000 — 11 мс). Узкое
    место здесь не алгоритм, а СЧЁТ на чистом Python, поэтому он вынесен в
    `_score_all`: с numpy это одно умножение матрицы на вектор, без него —
    цикл (работает всегда, но на больших базах медленнее).

    Чанк с испорченным вектором (не та размерность, пусто) пропускается: выдать
    его за найденный значило бы показать пользователю случайный фрагмент.
    """
    chunks = load_chunks(base_id, with_vectors=True)
    if not chunks:
        return []
    rows: List[List[float]] = []
    positions: List[int] = []
    for position, item in enumerate(chunks):
        stored = item.pop("vector", None) or []
        if not stored or (dim and len(stored) != dim):
            continue
        positions.append(position)
        rows.append(stored)
    if not rows:
        return []
    scores = _score_all(rows, vector)
    best = heapq.nlargest(min(limit, len(scores)), range(len(scores)),
                          key=scores.__getitem__)
    hits: List[Dict[str, Any]] = []
    for index in best:
        entry = dict(chunks[positions[index]])
        entry["score"] = round(float(scores[index]), 6)
        hits.append(entry)
    return hits


def _score_all(rows: List[List[float]], vector: List[float]) -> List[float]:
    """Близость каждой строки к запросу: numpy, если он есть, иначе перебор.

    Оба пути считают ОДНО И ТО ЖЕ (скалярное произведение float32), поэтому
    результат от наличия numpy не зависит — меняется только скорость. Если
    numpy вдруг не смог (нет, сломан, не сошлась форма), тихо откатываемся на
    перебор: поиск не должен падать из-за ускорителя.
    """
    if numpy_available():
        try:
            import numpy as np
            matrix = np.asarray(rows, dtype="float32")
            query = np.asarray(vector, dtype="float32")
            return [float(value) for value in matrix @ query]
        except Exception as exc:      # pragma: no cover - защита от сбоя numpy
            logger.warning("RAG: счёт на numpy не удался, считаю перебором — %s",
                           str(exc)[:150])
    return [sum(left * right for left, right in zip(row, vector)) for row in rows]


def numpy_available() -> bool:
    """Есть ли numpy (ускоряет счёт близости; без него поиск работает, но медленнее)."""
    try:
        return importlib.util.find_spec("numpy") is not None
    except (ImportError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Удаление и уборка
# ---------------------------------------------------------------------------
def delete_base(base_id: Any, profile: Optional[str] = None) -> bool:
    """Удаляет базу знаний вместе с индексом. False — базы нет или она чужая.

    Удаляется ИМЕННО каталог базы по проверенному идентификатору: ни путей, ни
    шаблонов здесь быть не может.
    """
    meta = get_base(base_id, profile=profile)
    folder = base_path(base_id)
    if meta is None or not folder:
        return False
    try:
        shutil.rmtree(folder)
    except OSError as exc:
        logger.warning("RAG: база %s не удалена — %s", base_id, str(exc)[:150])
        return False
    logger.info("RAG: база знаний %s удалена", base_id)
    return True


def prune_temp(profile: Optional[str] = None) -> int:
    """Убирает временные файлы незавершённых сборок индекса.

    Сборка пишет `index.sqlite.tmp` и `meta.json.tmp`: если процесс упал на
    середине, файл остаётся и копится. Убираем только старые (старше 10 минут) —
    свежий может принадлежать идущей прямо сейчас индексации.
    """
    root = directory()
    if not os.path.isdir(root):
        return 0
    removed = 0
    threshold = time.time() - 600
    for entry in os.listdir(root):
        if not valid_id(entry):
            continue
        folder = os.path.join(root, entry)
        try:
            names = os.listdir(folder)
        except OSError:
            continue
        for name in names:
            if not name.endswith(".tmp"):
                continue
            path = os.path.join(folder, name)
            try:
                if os.path.isfile(path) and os.path.getmtime(path) < threshold:
                    os.remove(path)
                    removed += 1
            except OSError:
                continue
    return removed


# ---------------------------------------------------------------------------
# Метрики
# ---------------------------------------------------------------------------
def storage_report() -> Dict[str, Any]:
    """Чем живёт база: где лежит индекс и чем считается близость при поиске.

    Снимок для диалога: хранилища у нас всегда одни и те же (SQLite для работы,
    JSON для выгрузки), а вот СЧЁТ близости зависит от машины — с numpy поиск
    быстрее на порядки. Пользователь должен видеть, каким путём идёт поиск, а не
    догадываться, почему база на 20 000 чанков отвечает медленнее.
    """
    fast = numpy_available()
    return {
        "sqlite": True,
        "json": True,
        "vectors": "numpy" if fast else "python",
        "vectors_name": ("numpy — счёт близости матрицей" if fast
                         else "перебор на Python"),
        "vectors_reason": ("" if fast else
                           "numpy не установлен — поиск идёт перебором на Python "
                           "(на больших базах заметно медленнее; numpy приходит "
                           "вместе с sentence-transformers)"),
        "dir": directory(),
    }


def aggregate_stats(bases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Сводка по базам знаний: чанки, документы, объём — для диалога и подписи кнопки."""
    chunks = 0
    documents = 0
    chars = 0
    size = 0
    for item in bases:
        stats = item.get("stats") or {}
        chunks += int(stats.get("chunks") or 0)
        documents += int(stats.get("documents") or 0)
        chars += int(stats.get("chars_total") or 0)
        size += int(item.get("size_bytes") or 0)
    return {
        "bases": len(bases),
        "chunks": chunks,
        "documents": documents,
        "chars_total": chars,
        "est_tokens": estimate_tokens(chars),
        "size_bytes": size,
        "size_human": human_size(size),
    }


def share_of(base: Dict[str, Any], total_bytes: int) -> float:
    """Доля базы в общем объёме баз профиля (0…1) — «вес» базы в диалоге."""
    try:
        total = float(total_bytes)
        if total <= 0:
            return 0.0
        return max(0.0, min(1.0, float(base.get("size_bytes") or 0) / total))
    except (TypeError, ValueError):
        return 0.0


def encode_b64(data: bytes) -> str:
    """Байты → base64 (для выгрузки/переноса индекса во внешние инструменты)."""
    return base64.b64encode(bytes(data)).decode("ascii")
