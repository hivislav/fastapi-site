"""Пайплайн индексации базы знаний (RAG) — от файлов до готового индекса.

ЧТО ЗДЕСЬ ПРОИСХОДИТ, ШАГ ЗА ШАГОМ:

    1. ИЗВЛЕЧЕНИЕ   файл (PDF, DOCX, текст…) → текст            rag_documents
    2. РАЗБИЕНИЕ    текст → чанки с метаданными                rag_chunking
    3. ЭМБЕДДИНГИ   чанки → векторы                            rag_embedding
    4. СОХРАНЕНИЕ   чанки + метаданные + векторы → индекс      rag_store
                    (SQLite — рабочее, JSON — выгрузка)

Модуль — ОРКЕСТРАТОР: сам он не режет текст, не считает векторы и не пишет
файлы, а связывает четыре части и следит за тем, чтобы индекс получился
ЦЕЛЫМ: векторы идут ровно в том порядке, в каком посчитаны; документ, который
не прочитался, не роняет загрузку, но и не исчезает молча — его причина
остаётся в метаданных базы; чанк не смешивает два файла.

МЕТАДАННЫЕ БАЗЫ (паспорт, `meta.json`) содержат всё, о чём спрашивает диалог
«База знаний»: название, владельца-профиль, стратегию разбиения и её параметры,
количество чанков и документов, средний/минимальный/максимальный размер чанка,
число разделов, объём («вес» базы на диске), оценку токенов, чем посчитаны
эмбеддинги (бэкенд, модель, размерность) и какие хранилища задействованы.

ПРОФИЛИ ИЗОЛИРОВАНЫ: у базы есть владелец (`profile`), и список баз, галочки
проекта и удаление проверяют владельца — как задачи, диалоги и память.
"""

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.ai import rag_chunking, rag_documents, rag_embedding, rag_ocr
from app.ai import rag_rerank
from app.ai import rag_search
from app.ai import rag_store

logger = logging.getLogger(__name__)

# Название базы знаний: если пользователь его не задал, берём имя первого файла.
MAX_NAME_CHARS = 120

# Имя файла в метаданных и в интерфейсе: длину ограничиваем, путь оставляем
# только «хвостом» — имя приходит от браузера и может содержать полный путь.
MAX_SOURCE_CHARS = 160

ProgressFn = Optional[Callable[[str, int, int], None]]


class RagError(Exception):
    """Индексацию выполнить нельзя (причина — в сообщении, для пользователя)."""


def sanitize_name(raw: Any, fallback: str = "") -> str:
    """Название базы знаний: одна строка, без управляющих символов и лишней длины."""
    value = str(raw or "").replace("\n", " ").replace("\r", " ").strip()
    value = " ".join(value.split())
    if not value:
        value = str(fallback or "").strip()
    return value[:MAX_NAME_CHARS] or "База знаний"


def sanitize_source(raw: Any) -> str:
    """Имя файла-источника: только последний сегмент пути (браузер шлёт и путь)."""
    name = str(raw or "").strip().replace("\\", "/").rsplit("/", 1)[-1]
    name = name.strip() or "документ"
    return name[:MAX_SOURCE_CHARS]


# ---------------------------------------------------------------------------
# Индексация
# ---------------------------------------------------------------------------
def index_files(files: List[Dict[str, Any]], *, name: Any = "", profile: Optional[str] = None,
                strategy: Any = None, chunk_size: Any = None, overlap: Any = None,
                backend: Optional[str] = None, on_progress: ProgressFn = None) -> Dict[str, Any]:
    """Собирает НОВУЮ базу знаний из загруженных файлов и записывает индекс.

    `files` — список {"filename": ...} и ОДНИМ из трёх источников содержимого:
    "text" (готовый текст), "path" (файл на диске — так приходит потоковая
    загрузка крупных документов) или "data" (байты, base64 из JSON-запроса).

    Возвращает паспорт базы (`meta`). Всё, что пошло не так с ОТДЕЛЬНЫМ файлом,
    попадает в `documents[].warning` — база собирается из того, что прочиталось;
    если не прочиталось НИЧЕГО, бросается RagError с причинами: пустая база,
    о которой пользователю не сказали, — худший результат из возможных.
    """
    settings = rag_chunking.chunk_settings(strategy, chunk_size, overlap)
    base_id = rag_store.new_id()
    documents, chunks, failures, first_source = _read_documents(
        files, base_id, settings, first_doc_index=0, on_progress=on_progress)
    _require_content(documents, chunks, failures)
    _number_chunks(chunks)
    vectors, embed_info = _embed_chunks(chunks, base_id, backend, on_progress)

    meta: Dict[str, Any] = {
        "id": base_id,
        "name": sanitize_name(name, fallback=first_source),
        "profile": str(profile or ""),
        "created": rag_store._now(),
        "strategy": settings["strategy"],
        "strategy_name": rag_chunking.strategy_name(settings["strategy"]),
        "chunk_size": settings["chunk_size"],
        "overlap": settings["overlap"],
        "embedding": embed_info,
        "documents": documents,
        "failures": failures,
        "stats": {},
    }
    meta["stats"] = build_stats(documents, chunks, meta["embedding"])

    _progress(on_progress, "save", 0, 1, "пишу индекс: %s чанков" % len(chunks))
    saved = rag_store.save_index(meta, documents, chunks, vectors)
    _progress(on_progress, "save", 1, 1, "индекс записан")
    if embed_info.get("fallback"):
        logger.warning("RAG: база %s посчитана запасным бэкендом — %s",
                       saved.get("id"), embed_info["fallback"])
    logger.info("RAG: база знаний %s собрана (%d чанков, %d документов, %s)",
                saved.get("id"), len(chunks), len(documents), saved.get("size_human"))
    return saved


def append_files(base_id: Any, files: List[Dict[str, Any]], *,
                 profile: Optional[str] = None,
                 on_progress: ProgressFn = None) -> Dict[str, Any]:
    """ДОБАВЛЯЕТ документы в существующую базу и перезаписывает её индекс.

    Нужно для потоковой загрузки нескольких файлов: первый создаёт базу, а
    каждый следующий дописывается к ней — иначе крупные файлы, которые нельзя
    отправить одним запросом, заводили бы по отдельной базе на файл.

    Два правила, из-за которых добавление не портит базу:

    - **параметры разбиения берутся из ПАСПОРТА базы**, а не из запроса: в
      паспорте записаны ОДНА стратегия и один размер чанка, и если добавлять
      документы с другими, паспорт перестанет соответствовать содержимому;
    - **эмбеддинги считаются тем же бэкендом и той же моделью**, что у базы.
      Векторы разных моделей несравнимы, поэтому «посчитать как получится» здесь
      означало бы испортить индекс (см. rag_embedding.check_compatible).

    Возвращает обновлённый паспорт базы.
    """
    meta = rag_store.get_base(base_id, profile=profile)
    if meta is None:
        raise RagError("база знаний не найдена")
    if not files:
        raise RagError("не передан ни один файл")
    if len(files) > rag_documents.MAX_FILES_PER_UPLOAD:
        raise RagError("за одну загрузку принимается не больше %d файлов"
                       % rag_documents.MAX_FILES_PER_UPLOAD)

    stored = meta.get("documents") or []
    if len(stored) >= rag_store.MAX_DOCS_PER_BASE:
        raise RagError("в базе уже %d документов — больше не принимается"
                       % rag_store.MAX_DOCS_PER_BASE)
    settings = rag_chunking.chunk_settings(
        meta.get("strategy"), meta.get("chunk_size"), meta.get("overlap"))
    documents, chunks, failures, _ = _read_documents(
        files, base_id, settings, first_doc_index=len(stored), on_progress=on_progress)
    _require_content(documents, chunks, failures)

    old_chunks = rag_store.load_chunks(base_id, with_vectors=True)
    old_vectors = [chunk.pop("vector", None) or [] for chunk in old_chunks]
    # Разбор старых векторов мог дать пустые (испорченная запись) — такую базу
    # дописывать нельзя: чанк без вектора в поиск не попадает, и добавление
    # молча «потеряло» бы прежние документы.
    if any(not vector for vector in old_vectors):
        raise RagError("в индексе базы есть чанки без векторов — пересоберите базу")

    backend = str((meta.get("embedding") or {}).get("backend") or "") or None
    reason = rag_embedding.check_compatible(meta, backend=backend)
    if reason:
        raise RagError("индекс базы посчитан иначе, чем сейчас: %s" % reason)
    vectors, embed_info = _embed_chunks(chunks, base_id, backend, on_progress)

    all_documents = list(stored) + documents
    all_chunks = old_chunks + chunks
    _number_chunks(all_chunks)
    meta = dict(meta)
    meta.update({
        "documents": all_documents,
        "failures": list(meta.get("failures") or []) + failures,
        "stats": {},
    })
    meta["stats"] = build_stats(all_documents, all_chunks, meta.get("embedding") or embed_info)
    saved = rag_store.save_index(meta, all_documents, all_chunks,
                                 old_vectors + vectors)
    logger.info("RAG: в базу %s добавлено %d документ(ов), всего чанков %d",
                saved.get("id"), len(documents), len(all_chunks))
    return saved


def _read_documents(files: List[Dict[str, Any]], base_id: str, settings: Dict[str, Any],
                    first_doc_index: int = 0,
                    on_progress: ProgressFn = None) -> Tuple[List[Dict[str, Any]],
                                                             List[Dict[str, Any]],
                                                             List[str], str]:
    """Файлы → документы и чанки (без эмбеддингов).

    Отдельная функция, потому что этим шагом пользуются ОБА пути: сборка новой
    базы и добавление в существующую. `first_doc_index` — с какого номера
    нумеровать документы: при добавлении нумерация продолжается, иначе чанки
    новых документов получили бы идентификаторы, уже занятые прежними.
    """
    if not files:
        raise RagError("не передан ни один файл")
    if len(files) > rag_documents.MAX_FILES_PER_UPLOAD:
        raise RagError("за одну загрузку принимается не больше %d файлов"
                       % rag_documents.MAX_FILES_PER_UPLOAD)
    documents: List[Dict[str, Any]] = []
    chunks: List[Dict[str, Any]] = []
    failures: List[str] = []
    first_source = ""

    _progress(on_progress, "extract", 0, len(files))
    for number, item in enumerate(files):
        source = sanitize_source(item.get("filename"))
        if not first_source:
            first_source = source
        def page_progress(page: int, pages: int, phase: str = "read",
                          _name: str = source) -> None:
            """Ход разбора ОДНОГО файла: страница читается или РАСПОЗНАЁТСЯ.

            Фазы разные по смыслу и по времени: чтение текстового слоя — мгновения,
            распознавание скана — секунды на страницу. Без пометки пользователь
            видел бы два одинаковых «страница 1 из 1» подряд и не понял бы, что
            происходит и почему это долго.

            ВЕС ЕДИНИЦЫ ПРОГРЕССА — ДОЛЯ ФАЙЛА, а не номер файла: раньше сюда
            уходил номер файла в списке, и у ОДНОГО большого скана доля всегда
            была нулевой — полоса замирала на 1% до конца распознавания (256
            страниц это десятки минут), хотя работа шла. Теперь «страница 28 из
            256» даёт свою долю, и полоса двигается.
            """
            detail = ("«%s»: распознаю (OCR) страницу %s из %s" if phase == "ocr"
                      else "«%s»: страница %s из %s")
            share = (float(page) / float(pages)) if pages else 0.0
            _progress(on_progress, "extract", number + min(1.0, max(0.0, share)),
                      len(files), detail % (_name, page, pages),
                      heavy=(phase == "ocr"))

        try:
            extracted = _extract_one(item, source, on_page=page_progress)
        except rag_documents.DocumentError as exc:
            failures.append("%s — %s" % (source, exc))
            logger.info("RAG: файл %s не прочитан — %s", source, exc)
            continue
        doc_index = first_doc_index + len(documents)
        doc_chunks = rag_chunking.chunk_document(
            extracted["text"], source=source,
            title=rag_chunking.strip_extension(source),
            strategy=settings["strategy"], chunk_size=settings["chunk_size"],
            overlap=settings["overlap"], doc_index=doc_index, id_prefix=base_id)
        documents.append({
            "doc_index": doc_index,
            "source": source,
            "title": rag_chunking.strip_extension(source),
            "kind": extracted["kind"],
            "format": extracted["format"],
            "chars": extracted["chars"],
            "pages": extracted["pages"],
            "chunks": len(doc_chunks),
            "warning": extracted["warning"],
        })
        chunks.extend(doc_chunks)
        _progress(on_progress, "extract", number + 1, len(files),
                  "«%s»: чанков %s" % (source, len(doc_chunks)))
    return documents, chunks, failures, first_source


def _require_content(documents: List[Dict[str, Any]], chunks: List[Dict[str, Any]],
                     failures: List[str]) -> None:
    """Не даёт записать пустую базу и объясняет, что именно не прочиталось.

    Формулировки здесь ВАЖНЫ: по ним пользователь решает, что делать с файлом.
    Поэтому «текст прочитан, но чанков не получилось» вместе с «текстового слоя
    нет» больше не встречается — это противоречие сбивало с толку (так и было в
    живом случае со сканом). Причина берётся от документа как есть, а общая фраза
    говорит ровно то, что произошло: текста для индексации не нашлось.
    """
    if not documents:
        # Причины отказов важнее общего текста: пользователь должен знать, ЧТО
        # именно не так с его файлами.
        if failures:
            raise RagError("ни один файл не удалось прочитать: " + "; ".join(failures))
        raise RagError("в загруженных файлах нет текста")
    if not chunks:
        reasons = [str(item["warning"]) for item in documents if item.get("warning")]
        if not reasons:
            raise RagError("текста для индексации не нашлось: документы пусты")
        raise RagError("текста для индексации не нашлось — %s" % "; ".join(reasons))


def _number_chunks(chunks: List[Dict[str, Any]]) -> None:
    """Сквозная нумерация чанков в базе (по ней строится порядок в индексе)."""
    for position, chunk in enumerate(chunks):
        chunk["index"] = position


def _embed_chunks(chunks: List[Dict[str, Any]], base_id: str, backend: Optional[str],
                  on_progress: ProgressFn) -> Tuple[List[List[float]], Dict[str, Any]]:
    """Считает векторы для чанков и собирает сведения о бэкенде."""
    texts = [chunk["text"] for chunk in chunks]
    _progress(on_progress, "embed", 0, len(texts),
              "векторы: 0 из %s чанков" % len(texts))
    try:
        vectors, info = rag_embedding.embed_texts(
            texts, backend=backend,
            on_progress=lambda done, total: _progress(
                on_progress, "embed", done, total,
                "векторы: %s из %s чанков" % (done, total)))
    except RuntimeError as exc:
        # Выбранный вручную бэкенд недоступен: индексировать «чем получится»
        # нельзя — векторы разных моделей несравнимы.
        raise RagError(str(exc))
    if len(vectors) != len(chunks):
        raise RagError("векторов %d, а чанков %d — индекс собран не будет"
                       % (len(vectors), len(chunks)))
    return vectors, {
        "backend": info.get("backend") or "",
        "model": info.get("model") or "",
        "dim": int(info.get("dim") or 0),
        "fallback": info.get("fallback") or "",
    }


def _extract_one(item: Dict[str, Any], source: str,
                 on_page: Any = None) -> Dict[str, Any]:
    """Текст одного файла: готовый `text`, файл на диске `path` или байты `data`.

    Три источника — это три транспорта загрузки: `text` (проверки и сборка из
    готового текста), `path` (потоковая загрузка крупных файлов: содержимое
    лежит на диске и читается постранично/почастям) и `data` (небольшие файлы,
    пришедшие в base64 одним JSON-запросом).
    """
    text = item.get("text")
    if isinstance(text, str):
        body = rag_documents._clean(text)
        if not body:
            raise rag_documents.DocumentError("файл пуст")
        return {"text": body, "kind": rag_documents.kind_of(source),
                "format": rag_documents.describe_format(source),
                "chars": len(body), "pages": 0, "warning": ""}
    path = item.get("path")
    if isinstance(path, str) and path:
        return rag_documents.extract_path(path, source, on_progress=on_page)
    data = item.get("data")
    if data is None:
        raise rag_documents.DocumentError("содержимое файла не передано")
    return rag_documents.extract(data, source)


def _progress(callback: ProgressFn, stage: str, done: float, total: float,
              detail: str = "", heavy: bool = False) -> None:
    """Сообщает о ходе индексации, если вызывающий код этого хочет.

    `detail` — человеческая строка для интерфейса («страница 123 из 500»,
    «векторы: 1200 из 4445 чанков»): проценты показывают, СКОЛЬКО сделано, а
    подробность — ЧТО именно происходит, и без неё долгая индексация выглядит
    зависанием.
    """
    if callback is None:
        return
    try:
        callback(stage, float(done), float(total), str(detail or ""), bool(heavy))
    except TypeError:
        # Наблюдатель старой формы (три аргумента) — не повод ронять работу.
        try:
            callback(stage, float(done), float(total))
        except Exception:
            logger.info("RAG: наблюдатель хода индексации упал", exc_info=False)
    except Exception:  # сбой наблюдателя не должен ломать индексацию
        # Ловим именно Exception: ОТМЕНА задачи (rag_jobs) наследуется от
        # BaseException и обязана пройти насквозь, иначе кнопка «остановить»
        # не останавливала бы ничего.
        logger.info("RAG: наблюдатель хода индексации упал", exc_info=False)


def build_stats(documents: List[Dict[str, Any]], chunks: List[Dict[str, Any]],
                embedding: Dict[str, Any]) -> Dict[str, Any]:
    """Метрики базы для диалога «База знаний» и подписи кнопки.

    Считается по ФАКТИЧЕСКИМ чанкам, а не по настройкам: пользователь должен
    видеть, что получилось, а не что заказывали (у структурной стратегии размер
    чанка плавает — раздел режется по заголовку, а не по числу символов).
    """
    summary = rag_chunking.describe(chunks)
    chars_total = int(summary["chars_total"])
    dim = int(embedding.get("dim") or 0)
    return {
        "chunks": int(summary["chunks"]),
        "documents": len(documents),
        "chars_total": chars_total,
        "chars_avg": int(summary["chars_avg"]),
        "chars_min": int(summary["chars_min"]),
        "chars_max": int(summary["chars_max"]),
        "sections": int(summary["sections"]),
        "kinds": summary["kinds"],
        "est_tokens": rag_store.estimate_tokens(chars_total),
        "dim": dim,
        # Объём самих векторов — «сколько база занимает из-за эмбеддингов»:
        # 4 байта на число float32.
        "vectors_bytes": int(summary["chunks"]) * dim * 4,
        "backend": embedding.get("backend") or "",
        "model": embedding.get("model") or "",
    }


# ---------------------------------------------------------------------------
# Снимок для интерфейса
# ---------------------------------------------------------------------------
def base_view(meta: Dict[str, Any], enabled: bool = False,
              total_bytes: int = 0) -> Dict[str, Any]:
    """Одна база в виде, пригодном для диалога: галочка + метрики + документы."""
    stats = meta.get("stats") or {}
    embedding = meta.get("embedding") or {}
    storage = meta.get("storage") or {}
    size_bytes = int(meta.get("size_bytes") or 0)
    return {
        "id": meta.get("id") or "",
        "name": meta.get("name") or "База знаний",
        "enabled": bool(enabled),
        "strategy": meta.get("strategy") or "",
        "strategy_name": meta.get("strategy_name")
                         or rag_chunking.strategy_name(meta.get("strategy")),
        "chunk_size": int(meta.get("chunk_size") or 0),
        "overlap": int(meta.get("overlap") or 0),
        "chunks": int(stats.get("chunks") or 0),
        "documents": int(stats.get("documents") or 0),
        "chars_total": int(stats.get("chars_total") or 0),
        "chars_avg": int(stats.get("chars_avg") or 0),
        "chars_min": int(stats.get("chars_min") or 0),
        "chars_max": int(stats.get("chars_max") or 0),
        "sections": int(stats.get("sections") or 0),
        "est_tokens": int(stats.get("est_tokens") or 0),
        "vectors_bytes": int(stats.get("vectors_bytes") or 0),
        "dim": int(embedding.get("dim") or 0),
        "backend": embedding.get("backend") or "",
        "model": embedding.get("model") or "",
        "fallback": embedding.get("fallback") or "",
        "size_bytes": size_bytes,
        "size_human": meta.get("size_human") or rag_store.human_size(size_bytes),
        "share": rag_store.share_of(meta, total_bytes),
        "created": meta.get("created") or "",
        "updated": meta.get("updated") or "",
        # Где лежит индекс этой базы (SQLite — рабочее, JSON — выгрузка). Про
        # FAISS здесь ничего нет намеренно: индекс у нас ОДИН и он в SQLite, а
        # чем считается близость при поиске — свойство машины, а не базы (см.
        # storage_report в снимке).
        "storage": {
            "sqlite": bool(storage.get("sqlite")),
            "json": bool(storage.get("json")),
        },
        "sources": [{
            "source": item.get("source") or "",
            "format": item.get("format") or "",
            "chunks": int(item.get("chunks") or 0),
            "chars": int(item.get("chars") or 0),
            "pages": int(item.get("pages") or 0),
            "warning": item.get("warning") or "",
        } for item in (meta.get("documents") or [])],
        "failures": list(meta.get("failures") or []),
    }


def snapshot(profile: Optional[str] = None, enabled_ids: Optional[List[str]] = None,
             settings: Optional[Dict[str, Any]] = None,
             force: bool = False) -> Dict[str, Any]:
    """Полный снимок «Базы знаний» для интерфейса (диалог и подпись кнопки).

    Содержит: список баз профиля с метриками и галочками, настройки разбиения
    проекта, доступные стратегии, состояние бэкенда эмбеддингов, состояние
    хранилищ (SQLite/JSON и чем считается близость), пределы и итоговые
    счётчики. Модель здесь НЕ
    загружается и к серверам обращений нет — диалог открывается мгновенно.

    `force` — кнопка «обновить» в диалоге: она заново проверяет ДОСТУПНОСТЬ
    модели эмбеддингов (сбрасывает кэш неудачной загрузки), поэтому после
    появления модели список баз и подпись бэкенда обновляются без перезапуска
    приложения.
    """
    bases = rag_store.list_bases(profile=profile)
    total_bytes = sum(int(item.get("size_bytes") or 0) for item in bases)
    wanted = [str(item).strip().lower() for item in (enabled_ids or [])]
    views = [base_view(meta, enabled=str(meta.get("id")) in wanted, total_bytes=total_bytes)
             for meta in bases]
    counts = rag_store.aggregate_stats(bases)
    counts["enabled"] = sum(1 for item in views if item["enabled"])
    counts["size_human"] = rag_store.human_size(counts["size_bytes"])
    return {
        "bases": views,
        "enabled": [item["id"] for item in views if item["enabled"]],
        "counts": counts,
        "strategies": [dict(item) for item in rag_chunking.CHUNKING_STRATEGIES],
        "settings": rag_chunking.chunk_settings(
            (settings or {}).get("strategy"),
            (settings or {}).get("chunk_size"),
            (settings or {}).get("overlap")),
        "defaults": {
            "strategy": rag_chunking.DEFAULT_STRATEGY,
            "chunk_size": rag_chunking.DEFAULT_CHUNK_SIZE,
            "overlap": rag_chunking.DEFAULT_CHUNK_OVERLAP,
        },
        "limits": {
            "chunk_size": {"min": rag_chunking.MIN_CHUNK_SIZE,
                           "max": rag_chunking.MAX_CHUNK_SIZE},
            "overlap": {"min": rag_chunking.MIN_CHUNK_OVERLAP,
                        "max": int(rag_chunking.MAX_CHUNK_SIZE
                                   * rag_chunking.MAX_OVERLAP_RATIO)},
            # Два предела, и интерфейс выбирает по ним ТРАНСПОРТ: до
            # json_file_bytes файл уходит в base64 одним запросом, крупнее —
            # потоком (см. rag_upload_stream).
            "file_bytes": rag_documents.MAX_FILE_BYTES,
            "file_size_human": rag_store.human_size(rag_documents.MAX_FILE_BYTES),
            "json_file_bytes": rag_documents.MAX_JSON_FILE_BYTES,
            "json_file_size_human": rag_store.human_size(
                rag_documents.MAX_JSON_FILE_BYTES),
            "files_per_upload": rag_documents.MAX_FILES_PER_UPLOAD,
            "bases": rag_store.MAX_BASES,
            "chunks_page": CHUNKS_PAGE,
            "chunks_page_max": CHUNKS_PAGE_MAX,
        },
        "formats": list(rag_documents.SUPPORTED_EXTENSIONS),
        "embedding": rag_embedding.backend_status(force=force),
        # Настройки ПОИСКА (сколько фрагментов уходит в ответ и с какой
        # релевантностью, см. app/ai/rag_search.py): интерфейс подписывает ими
        # строку «поиск подключён к ответам» и заполняет панель «Поиск и ответы»
        # (переформулировка запроса, реранкинг, порог, топ-K до и после второго
        # этапа). Не заданные у проекта значения берутся из окружения — снимок
        # показывает то, чем поиск работает НА САМОМ ДЕЛЕ.
        "search": rag_search.settings(settings),
        # НАСТРОЙКИ ПО УМОЛЧАНИЮ для кнопки «по умолчанию»: интерфейс заполняет
        # ими поля панели, а сохраняет их обычным «применить». Значения считает
        # СЕРВЕР (rag_search.defaults) — иначе кнопка проставляла бы числа,
        # придуманные страницей, и панель показывала бы не то, чем поиск работает.
        "search_defaults": rag_search.defaults(),
        # Границы полей этой панели: интерфейс не выдумывает их сам.
        "search_limits": rag_search.limits(),
        # СОСТОЯНИЕ РЕРАНКЕРА: чем реранкить сейчас и почему именно так (модель
        # cross-encoder скачана или нет). Модель при снимке НЕ загружается —
        # диалог открывается мгновенно, как и с эмбеддингами.
        "rerank": rag_rerank.status(force=force),
        "storage": rag_store.storage_report(),
        # Распознавание сканов: доступно ли и чем (см. app/ai/rag_ocr.py).
        "ocr": rag_ocr.status(),
        "dir": rag_store.directory(),
    }


# Страница просмотра чанков: сколько показывать за раз по умолчанию и сколько
# добавлять кнопкой «показать ещё». Меньше десяти — не видно структуры, больше
# тридцати — страница превращается в простыню, в которой ничего не найти.
CHUNKS_PAGE = 10
CHUNKS_PAGE_MAX = 50


def chunks_view(base_id: Any, profile: Optional[str] = None, offset: int = 0,
                limit: int = CHUNKS_PAGE, source: Any = "",
                query: Any = "", chunk: int = 0) -> Dict[str, Any]:
    """Страница чанков базы для диалога просмотра.

    Отдаёт сам текст чанков вместе с их адресом (источник, раздел, номер,
    границы в документе) — это и есть «посмотреть, как стратегия порезала
    документ». Плюс список документов базы (для фильтра) и общее число чанков
    под фильтром, чтобы интерфейс мог показать «показано 10 из 42».

    `chunk` — номер чанка, к которому надо ПЕРЕЙТИ (клик по источнику под
    ответом агента): страница сдвигается так, чтобы этот фрагмент был первым.
    Номер чанка — тот же, что в карточке источника («№ 1081»), поэтому по нему
    фрагмент находится без поиска глазами.
    """
    meta = rag_store.get_base(base_id, profile=profile)
    if meta is None:
        raise RagError("база знаний не найдена")
    size = max(1, min(CHUNKS_PAGE_MAX, int(limit or CHUNKS_PAGE)))
    start = max(0, int(offset or 0))
    # Имя документа для фильтра нормализуем ТАК ЖЕ, как при индексации (хвост
    # пути, ограничение длины), но БЕЗ запасного «документ»: пустой фильтр
    # означает «все документы», а не поиск файла с таким именем.
    filter_source = (str(source or "").strip().replace("\\", "/")
                     .rsplit("/", 1)[-1][:MAX_SOURCE_CHARS])
    chunks, total, start = rag_store.chunks_page(
        base_id, offset=start, limit=size, source=filter_source,
        query=str(query or "").strip()[:200], chunk=max(0, int(chunk or 0)))
    return {
        "base": {
            "id": meta.get("id") or "",
            "name": meta.get("name") or "База знаний",
            "strategy": meta.get("strategy") or "",
            "strategy_name": meta.get("strategy_name")
                             or rag_chunking.strategy_name(meta.get("strategy")),
            "chunk_size": int(meta.get("chunk_size") or 0),
            "overlap": int(meta.get("overlap") or 0),
            "chunks": int((meta.get("stats") or {}).get("chunks") or 0),
            "chars_avg": int((meta.get("stats") or {}).get("chars_avg") or 0),
            "dim": int((meta.get("embedding") or {}).get("dim") or 0),
            "backend": (meta.get("embedding") or {}).get("backend") or "",
            "model": (meta.get("embedding") or {}).get("model") or "",
            "sources": [item.get("source") or ""
                        for item in (meta.get("documents") or [])],
        },
        "chunks": chunks,
        "total": total,
        "offset": start,
        "limit": size,
        "has_more": start + len(chunks) < total,
        "filter": {"source": filter_source, "query": str(query or "").strip()[:200]},
    }


def filter_enabled(ids: Any, profile: Optional[str] = None) -> List[str]:
    """Оставляет только существующие базы ЭТОГО профиля, без повторов.

    Галочка, оставшаяся от удалённой базы или от чужого профиля, в настройку не
    попадает: включённым не может быть то, чего у проекта нет (как с серверами
    MCP — см. app/ai/workspace.py).
    """
    known = set(rag_store.existing_ids(profile=profile))
    result: List[str] = []
    for item in (ids if isinstance(ids, list) else []):
        key = str(item or "").strip().lower()
        if key and key in known and key not in result:
            result.append(key)
    return result[:rag_store.MAX_BASES]


def delete_base(base_id: Any, profile: Optional[str] = None) -> bool:
    """Удаляет базу знаний профиля (индекс и метаданные)."""
    return rag_store.delete_base(base_id, profile=profile)


def ensure_capacity(profile: Optional[str] = None) -> None:
    """Проверяет, что у профиля есть место под новую базу знаний."""
    if len(rag_store.list_bases(profile=profile, with_meta=False)) >= rag_store.MAX_BASES:
        raise RagError("у профиля уже %d баз знаний — удалите ненужные"
                       % rag_store.MAX_BASES)


def base_dir_for(base_id: Any) -> str:
    """Каталог базы (для подписи в интерфейсе: где лежит индекс)."""
    return rag_store.base_path(base_id)


def relative_dir() -> str:
    """Каталог баз знаний относительно корня проекта (для подписи в диалоге)."""
    path = rag_store.directory()
    try:
        from app import config
        root = config.PROJECT_ROOT
        if path.startswith(root + os.sep):
            return os.path.relpath(path, root)
    except Exception:  # pragma: no cover
        pass
    return path
