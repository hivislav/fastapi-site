"""RAG ДЛЯ AI-РЕВЬЮ: индекс кода и документации проекта — и поиск НУЖНЫХ чанков.

ЗАЧЕМ ЭТО ОТДЕЛЬНЫЙ МОДУЛЬ. Ревью нужно читать правку ВМЕСТЕ с кодовой базой, но
отправлять в модель весь код нельзя: в проекте десятки файлов и сотни тысяч
символов, и «весь код целиком» — это и дорого, и хуже по качеству (модель тонет в
постороннем). Поэтому кодовая база ЗАГРУЖАЕТСЯ В RAG — в обычную базу знаний
проекта (`app/ai/rag.py`: те же чанки, те же эмбеддинги, тот же поиск
`rag_search.search`), — а в модель уходят только фрагменты, найденные по самой
правке: изменённые имена, модули, пожелание человека.

ЧТО ЗДЕСЬ ЕСТЬ, А ЧЕГО НЕТ. Модуль НЕ читает diff и НЕ зовёт модель: изменения
приходят от MCP git (`ai_review`), рассуждение — от модели. Его дело — индекс:

* `documents()` — какие файлы репозитория вообще индексируются (те же правила,
  что у отбора контекста: текстовые расширения, закрытые пути и слишком большие
  файлы пропускаются — и это НАЗЫВАЕТСЯ, а не молчит);
* `ensure()` — «индекс есть и он про ЭТОТ коммит?» Если нет — собирает; если
  да — не тратит минуты на пересборку (ключ свежести — хеш HEAD, который назвал
  сервер git);
* `search()` — поиск по базе НЕСКОЛЬКИМИ запросами (имена правки, модули,
  пожелание), с удалением повторов: один запрос по большому проекту даёт
  заметно худшую выборку;
* `index_in_background()` — сборка индекса ФОНОВОЙ задачей (штатный механизм
  `app/ai/rag_jobs.py`: прогресс виден в диалоге «RAG»). Этим пользуется команда
  `/ai_review_enable`: режим включили — база кода поехала в индекс сама, и первое
  ревью не ждёт индексации.

ПОЧЕМУ ИНДЕКС — ЭТО БАЗА ЗНАНИЙ, А НЕ СВОЙ ФОРМАТ. Так ревью пользуется тем же
поиском, что и весь проект (два этапа, пороги, реранкинг, карточки источников), а
человек ВИДИТ эту базу в диалоге «RAG» и может её удалить: индекс ревью — не
тайник, а обычная база с понятным именем. Свежесть базы помечается ФАЙЛОМ в её
каталоге (`ai_review.json`: какой репозиторий и какой коммит проиндексированы) —
паспорт базы для этого не годится, он общий для всех баз и его формат трогать
нельзя.

ЧЕГО СТОИТ ОПАСАТЬСЯ. Индексация — работа на минуты: с моделью эмбеддингов
(`sentence-transformers`, есть на своей машине) на большом проекте это долго, а со
встроенным офлайн-бэкендом (`hashing`) — секунды. Поэтому индекс кэшируется по
коммиту, а причина «почему индекс не собрался» называется словами: ревью всё
равно состоится — на отборе контекста кодом (`ai_review.context_block`), и об
этом будет сказано и в отчёте, и в диагностике.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.ai import ai_review
from app.ai import rag
from app.ai import rag_documents
from app.ai import rag_search
from app.ai import rag_store

logger = logging.getLogger(__name__)

# Имя базы в диалоге «RAG»: человек должен видеть, что это за индекс и откуда он.
BASE_NAME = "Код и документация (AI-ревью)"
# Файл-маркер в каталоге базы: какой репозиторий и какой коммит проиндексированы.
MARKER_FILE = "ai_review.json"
# Поля маркера: по ним проверяется свежесть индекса.
MARKER_KEYS = ("root", "sha", "fingerprint", "files", "documents", "chunks",
               "backend", "chars", "at")

# Пределы индексации. Файлы больше `ai_review.SCAN_MAX_BYTES` не индексируются
# вовсе (это решает `ai_review.repo_files`), а здесь — пределы ВСЕГО индекса:
# мегабайты текста превратились бы в десятки тысяч чанков, и такая база стоила бы
# дороже, чем помогает. Предел числа документов взят ИЗ ХРАНИЛИЩА
# (`rag_store.MAX_DOCS_PER_BASE`), а не своим числом: база всё равно не примет
# больше, и «свой предел» обещал бы то, чего хранилище не сделает.
MAX_FILES = min(200, rag_store.MAX_DOCS_PER_BASE)
MAX_CHARS = 4_000_000
# За одну загрузку RAG принимает не больше `rag_documents.MAX_FILES_PER_UPLOAD`
# файлов (двадцать): индекс ревью поэтому собирается ПАРТИЯМИ — первый файл
# заводит базу, остальные дописываются в неё. Число берётся у самого RAG, чтобы
# правка его предела не превратила сборку индекса в отказ.
UPLOAD_BATCH = max(1, int(rag_documents.MAX_FILES_PER_UPLOAD))
# Меньше этого текст не документ: пустые файлы и огрызки только засоряют поиск.
MIN_CHARS = 40

# Сколько запросов делается по одной правке. Первый — имена и модули (что
# изменилось), второй — пожелание человека (если оно есть), третий —
# архитектурный («как это устроено в проекте»). Больше — уже шум и время.
QUERIES = 3

# НАСТРОЙКИ ПОИСКА ПО ИНДЕКСУ — ОДНИ НА ДВА ВХОДА (чат и пайплайн). Порог
# первичной релевантности здесь выключен, а выборка шире обычной: ревью не
# отвечает на вопрос «есть ли это в документах», ему нужен БЛИЖАЙШИЙ код к правке.
# Порог, подобранный для ответов по документам, отсекал бы фрагменты кода и
# оставлял ревью без контекста — молча.
REVIEW_SEARCH = {"filter": False, "min_score": 0.0, "top_k_before": 40,
                 "top_k_after": 10}


def search_settings(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Настройки поиска по индексу ревью (общие для чата и пайплайна).

    Считаются ТЕМ ЖЕ кодом, что и сам поиск (`rag_search.search_settings`),
    поэтому «настройка ревью» не может дать невозможных чисел, а значения
    окружения (`RAG_RERANK_BACKEND` и прочие) действуют и здесь. Второй набор
    чисел в другом модуле разошёлся бы с этим при первой правке.
    """
    raw = dict(REVIEW_SEARCH)
    raw.update(extra or {})
    return rag_search.search_settings(raw)


# Состояния фоновой задачи индексации (app/ai/rag_jobs.py: поле `state`): по ним
# видно, чем кончилась сборка, и только после успеха пишется маркер. Поле именно
# `state` — в задаче есть ещё `stage` (этап работы), и путать их нельзя.
JOB_DONE = "done"
JOB_FAILED = "failed"
JOB_CANCELLED = "cancelled"
# Как часто спрашивать состояние фоновой задачи и сколько ждать её окончания.
JOB_POLL_S = 2.0
JOB_WAIT_S = 3600.0


def file_list(root: str, max_files: int = MAX_FILES) -> Dict[str, Any]:
    """Файлы-кандидаты для индекса: (список [(путь, размер)], пропуски, причина).

    Читается ТОЛЬКО СТАТИСТИКА файлов (обход и размеры, без содержимого): по ней
    считается отпечаток набора, а он — ключ свежести индекса. Полное чтение
    содержимого нужно один раз — при самой сборке (`documents`), а проверка
    «индекс про этот набор файлов?» обязана быть дешёвой: она идёт на КАЖДОМ
    ревью, а обход каталога — миллисекунды.

    Документация идёт ПЕРВОЙ: она отвечает на «как принято в проекте», и её
    фрагменты должны попасть в базу, даже если предел числа файлов сработает на
    коде.
    """
    base = str(root or "").strip()
    if not base or not os.path.isdir(base):
        return {"files": [], "skipped": [], "reason":
                "каталог репозитория не найден: %s" % (base or "не задан")}
    paths = ai_review.repo_files(base, max_files=max_files * 3)
    docs = [path for path in paths if path.lower().endswith(ai_review.DOC_EXTS)]
    code = [path for path in paths if path not in docs]
    pairs: List[Tuple[str, int]] = []
    skipped: List[str] = []
    for rel in docs + code:
        if len(pairs) >= max(1, int(max_files)):
            skipped.append("файлов больше предела индексации (%d)" % max_files)
            break
        try:
            size = os.path.getsize(os.path.join(base, rel))
        except OSError:
            skipped.append("%s — не прочитался размер" % rel)
            continue
        if size < MIN_CHARS:
            skipped.append("%s — пусто или почти пусто" % rel)
            continue
        pairs.append((rel, int(size)))
    if not pairs:
        return {"files": [], "skipped": skipped[:5],
                "reason": "в репозитории не нашлось ни одного текстового файла"}
    return {"files": pairs, "skipped": skipped[:20], "reason": ""}


def fingerprint(pairs: Any) -> str:
    """Отпечаток набора файлов по именам и размерам (без чтения содержимого).

    Служит ключом свежести вместе с хешем коммита: хеш называет сервер git, а
    отпечаток работает и без него (например в команде включения режима, которой
    незачем дёргать MCP ради кеша). Правка файла меняет размер — отпечаток
    меняется; правка «в тот же размер» ловится хешем коммита.
    """
    hasher = hashlib.sha256()
    for rel, size in (pairs or []):
        hasher.update(str(rel).encode("utf-8"))
        hasher.update(str(size).encode("utf-8"))
    return "fp:" + hasher.hexdigest()[:16]


def documents(root: str, max_files: int = MAX_FILES,
              max_chars: int = MAX_CHARS) -> Dict[str, Any]:
    """Файлы репозитория для индекса: (документы, пропуски, причина).

    Берутся ТЕ ЖЕ файлы, что и для отбора контекста (`ai_review.repo_files`):
    текстовые расширения, без закрытых путей (секреты, `data/`, `node_modules/`)
    и без слишком больших файлов, а сверху — пределы ВСЕГО индекса (число файлов
    и объём текста): сотни файлов и мегабайты текста превратились бы в десятки
    тысяч чанков, и такая база стоила бы дороже, чем помогает.

    Пропуски не молчат: причина называется и уходит в диагностику ревью —
    «индекс без половины проекта» человек обязан увидеть.
    """
    listing = file_list(root, max_files=max_files)
    if listing["reason"]:
        return {"documents": [], "skipped": listing["skipped"], "chars": 0,
                "files": 0, "fingerprint": "fp:", "reason": listing["reason"]}
    base = str(root or "").strip()
    out: List[Dict[str, str]] = []
    skipped = list(listing["skipped"])
    chars = 0
    for rel, _size in listing["files"]:
        text = ai_review.read_text(base, rel)
        if len(text.strip()) < MIN_CHARS:
            skipped.append("%s — пусто или почти пусто" % rel)
            continue
        if chars + len(text) > max_chars:
            skipped.append("текста больше предела индексации (%d символов)" % max_chars)
            break
        out.append({"filename": rel, "text": text})
        chars += len(text)
    if not out:
        return {"documents": [], "skipped": skipped[:5], "chars": 0, "files": 0,
                "fingerprint": fingerprint(listing["files"]),
                "reason": "в репозитории не нашлось ни одного текстового файла"}
    return {"documents": out, "skipped": skipped[:20], "chars": chars,
            "files": len(out), "fingerprint": fingerprint(listing["files"]),
            "reason": ""}


def running_job(profile: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Идёт ли СЕЙЧАС сборка индекса ревью (иначе None).

    Нужно, чтобы ревью не запускало ВТОРУЮ индексацию поверх фоновой (команда
    включения режима уже поставила её в очередь): вторая сборка — это двойная
    работа на минуты и две базы-близнеца. Идёт — ревью честно скажет, что индекс
    ещё собирается, и соберёт контекст отбором кодом.
    """
    from app.ai import rag_jobs

    for job in rag_jobs.listing(profile=profile, active_only=True):
        if str(job.get("name") or "") == BASE_NAME:
            return job
    return None


def marker_path(base_id: Any) -> str:
    """Путь файла-маркера внутри каталога базы."""
    return os.path.join(rag.base_dir_for(base_id), MARKER_FILE)


def read_marker(base_id: Any) -> Dict[str, Any]:
    """Маркер базы (пусто — это НЕ наша база или маркер испорчен)."""
    try:
        with open(marker_path(base_id), "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {key: data.get(key) for key in MARKER_KEYS}


def write_marker(base_id: Any, marker: Dict[str, Any]) -> bool:
    """Записывает маркер (сбой записи — не сбой индекса: свежесть просто
    перепроверится в следующий раз, поэтому причина только логируется)."""
    path = marker_path(base_id)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({key: marker.get(key) for key in MARKER_KEYS}, fh,
                      ensure_ascii=False, indent=2)
        return True
    except OSError as exc:
        logger.warning("AI-ревью: маркер индекса не записан — %s", exc)
        return False


def review_bases(profile: Optional[str] = None) -> List[Dict[str, Any]]:
    """Базы ревью этого профиля (по маркеру — чужие базы не трогаем).

    Именно по МАРКЕРУ, а не по имени: имя человек может изменить или создать
    свою базу с таким же именем, и удалять её ревью не имеет права.
    """
    out: List[Dict[str, Any]] = []
    for meta in rag_store.list_bases(profile=profile, with_meta=True):
        base_id = str(meta.get("id") or "")
        if not base_id:
            continue
        marker = read_marker(base_id)
        if marker:
            out.append({"id": base_id, "name": str(meta.get("name") or ""),
                        "chunks": int((meta.get("stats") or {}).get("chunks") or 0),
                        "marker": marker})
    return out


def state(root: str, profile: Optional[str] = None, sha: str = "") -> Dict[str, Any]:
    """Что с индексом ревью: свежий, устаревший или его нет.

    Возвращает {"base_id", "fresh", "marker", "reason", "stale", "fingerprint"}:
    `stale` — базы, оставшиеся от прежних сборок (их убирает успешная сборка).

    ПРИЗНАК СВЕЖЕСТИ ОДИН, И ВЫБИРАЕТСЯ ОН ПО ДАННЫМ: коммит известен (`sha`
    назвал сервер git) — сверяем КОММИТ и только его; коммит неизвестен (команда
    включения режима, которой незачем дёргать MCP ради кеша) — сверяем отпечаток
    набора файлов (имена и размеры). Считать по отпечатку, когда коммит известен,
    нельзя: правка «в тот же размер» прошла бы за свежий индекс, и ревью увидело
    бы СТАРЫЙ код, не сказав об этом.
    """
    wanted_root = os.path.abspath(str(root or ""))
    current_fp = fingerprint(file_list(root)["files"])
    fresh: Dict[str, Any] = {}
    stale: List[Dict[str, Any]] = []
    for item in review_bases(profile=profile):
        marker = item["marker"]
        same_root = os.path.abspath(str(marker.get("root") or "")) == wanted_root
        if sha:
            same = str(marker.get("sha") or "") == str(sha)
        else:
            same = str(marker.get("fingerprint") or "") == current_fp
        if same_root and same and not fresh:
            fresh = item
        else:
            stale.append(item)
    if fresh:
        return {"base_id": fresh["id"], "fresh": True, "marker": fresh["marker"],
                "stale": stale, "reason": "", "fingerprint": current_fp}
    reason = ""
    if stale:
        marker = stale[0]["marker"]
        reason = ("индекс кода есть, но он про другой коммит или другой набор "
                  "файлов (%s), а сейчас %s"
                  % (str(marker.get("sha") or marker.get("fingerprint") or "без ключа")[:24],
                     str(sha or current_fp)[:24]))
    else:
        reason = "индекса кода ещё нет"
    return {"base_id": "", "fresh": False, "marker": {}, "stale": stale,
            "reason": reason, "fingerprint": current_fp}


def build(root: str, profile: Optional[str] = None, sha: str = "",
          on_progress: Callable[..., None] = None,
          settings: Optional[Dict[str, Any]] = None,
          backend: Optional[str] = None) -> Dict[str, Any]:
    """Собирает индекс ревью: код и документация репозитория → новая база.

    Возвращает {"ok", "base_id", "documents", "chunks", "chars", "backend",
    "seconds", "reason"}. Отказ НЕ бросается наружу: «индекс не собрался» — это
    причина, а не сбой ревью (оно состоится на отборе контекста кодом).
    """
    started = time.time()
    info = documents(root)
    if info["reason"]:
        return {"ok": False, "base_id": "", "documents": 0, "chunks": 0,
                "chars": 0, "backend": "", "seconds": 0.0,
                "reason": info["reason"]}
    chunk_settings = settings or {}
    batch = UPLOAD_BATCH
    parts = [info["documents"][index:index + batch]
             for index in range(0, len(info["documents"]), batch)]
    try:
        rag.ensure_capacity(profile=profile)
        # ИНДЕКС СОБИРАЕТСЯ ПАРТИЯМИ: RAG принимает не больше двадцати файлов за
        # одну загрузку (у потоковой загрузки тот же предел), а в репозитории их
        # десятки. Первая партия заводит базу, остальные дописываются в неё —
        # параметры разбиения и бэкенд при этом берутся из паспорта базы (это
        # делает сам `rag.append_files`), поэтому индекс остаётся однородным.
        meta = rag.index_files(
            parts[0], name=BASE_NAME, profile=profile,
            strategy=chunk_settings.get("strategy"),
            chunk_size=chunk_settings.get("chunk_size"),
            overlap=chunk_settings.get("overlap"),
            backend=backend, on_progress=on_progress)
        for piece in parts[1:]:
            meta = rag.append_files(meta.get("id"), piece, profile=profile,
                                    on_progress=on_progress)
    except Exception as exc:                     # pragma: no cover - защита
        logger.exception("AI-ревью: индекс кода не собран")
        return {"ok": False, "base_id": "", "documents": info["files"], "chunks": 0,
                "chars": info["chars"], "backend": "", "seconds": 0.0,
                "reason": "индекс кода не собран: %s" % str(exc)[:200]}
    base_id = str(meta.get("id") or "")
    # СЧЁТЧИКИ БЕРУТСЯ ИЗ `stats`, А НЕ ИЗ ВЕРХНИХ ПОЛЕЙ: в паспорте базы
    # `documents` — это СПИСОК документов, а числа (чанки, документы, размер)
    # лежат в `stats` (см. app/ai/rag.py: build_stats).
    stats = meta.get("stats") or {}
    marker = {
        "root": os.path.abspath(str(root or "")),
        "sha": str(sha or ""),
        "documents": int(stats.get("documents") or info["files"]),
        "chunks": int(stats.get("chunks") or 0),
        "chars": info["chars"],
        "backend": str((meta.get("embedding") or {}).get("backend")
                       or meta.get("backend") or ""),
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "fingerprint": info["fingerprint"],
        "files": info["files"],
    }
    write_marker(base_id, marker)
    return {"ok": True, "base_id": base_id, "documents": marker["documents"],
            "chunks": marker["chunks"], "chars": info["chars"],
            "backend": marker["backend"], "seconds": round(time.time() - started, 1),
            "skipped": info["skipped"], "reason": ""}


def drop_bases(bases: List[Dict[str, Any]], profile: Optional[str] = None,
               keep: str = "") -> int:
    """Убирает прежние индексы ревью (кроме `keep`) и возвращает, сколько убрал.

    Убираются ТОЛЬКО базы с маркером (`review_bases`): своя база человека с таким
    же именем остаётся на месте. Уборка идёт ПОСЛЕ успешной сборки: удалить
    прежний индекс раньше значило бы остаться без обоих.
    """
    removed = 0
    for item in bases:
        if str(item.get("id")) == str(keep or ""):
            continue
        if rag.delete_base(item["id"], profile=profile):
            removed += 1
    return removed


def ensure(root: str, profile: Optional[str] = None, sha: str = "",
           force: bool = False, on_progress: Callable[..., None] = None,
           settings: Optional[Dict[str, Any]] = None,
           backend: Optional[str] = None) -> Dict[str, Any]:
    """Индекс ревью для ЭТОГО коммита: свежий отдаётся, устаревший пересобирается.

    Возвращает {"ok", "base_id", "built", "chunks", "documents", "backend",
    "seconds", "reason"}. `built` = False означает «взяли готовый индекс».
    """
    if not force:
        current = state(root, profile=profile, sha=sha)
        if current["fresh"]:
            marker = current["marker"]
            return {"ok": True, "base_id": current["base_id"], "built": False,
                    "chunks": int(marker.get("chunks") or 0),
                    "documents": int(marker.get("documents") or 0),
                    "chars": int(marker.get("chars") or 0),
                    "backend": str(marker.get("backend") or ""), "seconds": 0.0,
                    "stale": current["stale"], "reason": ""}
    built = build(root, profile=profile, sha=sha, on_progress=on_progress,
                  settings=settings, backend=backend)
    if not built["ok"]:
        return {"ok": False, "base_id": "", "built": False, "chunks": 0,
                "documents": built["documents"], "chars": built["chars"],
                "backend": "", "seconds": built["seconds"], "stale": [],
                "reason": built["reason"]}
    # Прежние индексы убираются ПОСЛЕ успешной сборки: маркер новой базы уже
    # записан, поэтому `state` теперь считает устаревшими ровно их.
    stale = state(root, profile=profile, sha=sha)["stale"]
    built["stale_removed"] = drop_bases(stale, profile=profile,
                                        keep=built["base_id"])
    built["built"] = True
    return built


# ---------------------------------------------------------------------------
# Фоновая сборка: режим включили — индекс кода поехал в базу сам
# ---------------------------------------------------------------------------
class _Stopped(BaseException):
    """Индексацию отменили из интерфейса (`rag_jobs.cancel`).

    Наследуется от BaseException НАМЕРЕННО, как и отмена у самих задач RAG
    (`rag_jobs._Cancelled`): наблюдатель хода индексации ловит обычные исключения
    (`rag._progress`), и «отмена» не должна выглядеть как сбой наблюдателя —
    иначе кнопка «остановить» не останавливала бы ничего.
    """


async def index_in_background(root: str, profile: Optional[str] = None,
                              sha: str = "",
                              settings: Optional[Dict[str, Any]] = None,
                              ) -> Dict[str, Any]:
    """Запускает сборку индекса ФОНОВОЙ задачей RAG (прогресс виден в «RAG»).

    Так работает команда `/ai_review_enable`: индексация крупного проекта — это
    минуты, и держать на них HTTP-соединение нельзя (штатный механизм проекта:
    задача + прогресс, см. app/ai/rag_jobs.py).

    ЗАДАЧА ВЕДЁТСЯ ЗДЕСЬ, а не через `rag_jobs.start`, ПО ДВУМ причинам:

      * **партии.** RAG принимает не больше `MAX_FILES_PER_UPLOAD` файлов за одну
        загрузку, а в репозитории их десятки: `rag_jobs.start` отказал бы («за одну
        загрузку принимается не больше 20 файлов») — на этом и споткнулся живой
        прогон 14.10. Партии собирает `build`, а задача лишь показывает прогресс;
      * **один индекс и один маркер.** Несколько задач подряд дали бы несколько баз
        и «какая из них настоящая» решалось бы по времени, а не по коммиту.

    Возвращает {"started", "job_id", "documents", "task", "reason"}: `task` —
    заведённая задача (её видно в проверках), `reason` — почему не запущено
    («индекс уже свежий», «нечего индексировать», «нет места»).
    """
    from app.ai import rag_jobs

    current = state(root, profile=profile, sha=sha)
    if current["fresh"]:
        return {"started": False, "job_id": "", "documents": 0, "task": None,
                "reason": "индекс кода уже свежий (коммит %s)"
                          % (str(sha)[:12] or "—")}
    # Чтение файлов — в отдельном потоке: их десятки мегабайт, и цикл событий на
    # этом стоять не должен.
    info = await asyncio.to_thread(documents, root)
    if info["reason"]:
        return {"started": False, "job_id": "", "documents": 0, "task": None,
                "reason": info["reason"]}
    try:
        rag.ensure_capacity(profile=profile)
        job = rag_jobs.create(profile=profile, name=BASE_NAME)
    except rag.RagError as exc:
        return {"started": False, "job_id": "", "documents": info["files"],
                "task": None, "reason": "индекс кода не запущен: %s" % str(exc)[:200]}
    except Exception as exc:                     # pragma: no cover - защита
        logger.exception("AI-ревью: фоновая индексация не запущена")
        return {"started": False, "job_id": "", "documents": info["files"],
                "task": None, "reason": "индекс кода не запущен: %s" % str(exc)[:200]}
    job_id = str(job.get("id") or "")
    task = asyncio.ensure_future(_background_worker(job_id, root, profile, sha,
                                                    settings))
    return {"started": True, "job_id": job_id, "documents": info["files"],
            "skipped": info["skipped"], "task": task, "reason": ""}


async def _background_worker(job_id: str, root: str, profile: Optional[str],
                             sha: str, settings: Optional[Dict[str, Any]],
                             ) -> Dict[str, Any]:
    """Собирает индекс в фоне и ведёт задачу RAG: прогресс, отмена, итог, маркер.

    Возвращает итог сборки (тот же словарь, что `build`), чтобы вызывающий мог
    дождаться задачи и увидеть результат: «сколько файлов, сколько чанков, чем
    считались векторы». Прогресс и отмена идут через `rag_jobs`: `progress`
    возвращает False, когда задачу отменили, — это и есть проверка отмены.
    """
    from app.ai import rag_jobs

    def report(stage: str = "", done: float = 0, total: float = 0,
               detail: str = "", heavy: bool = False) -> None:
        if not rag_jobs.progress(job_id, str(stage or "index"), done, total,
                                 str(detail or ""), bool(heavy)):
            raise _Stopped()

    try:
        built = await asyncio.to_thread(build, root, profile=profile, sha=sha,
                                       on_progress=report, settings=settings)
    except _Stopped:
        rag_jobs.cancelled(job_id)
        return {"ok": False, "base_id": "", "chunks": 0, "documents": 0,
                "reason": "индексация отменена"}
    except Exception as exc:                     # pragma: no cover - защита
        logger.exception("AI-ревью: фоновая индексация упала")
        rag_jobs.fail(job_id, "%s: %s" % (type(exc).__name__, str(exc)[:200]))
        return {"ok": False, "base_id": "", "chunks": 0, "documents": 0,
                "reason": "индексация не удалась: %s" % str(exc)[:200]}
    if not built.get("ok"):
        rag_jobs.fail(job_id, str(built.get("reason") or "индексация не удалась"))
        return built
    # Маркер новой базы уже записан `build`; убираем прежние индексы ревью и
    # закрываем задачу ИТОГОМ БАЗЫ (по нему интерфейс показывает чанки).
    built["stale_removed"] = drop_bases(
        [item for item in review_bases(profile=profile) if item["id"] != built["base_id"]],
        profile=profile)
    meta = rag_store.get_base(built["base_id"], profile=profile) or {}
    rag_jobs.finish(job_id, meta)
    return built


async def start_background(root: str, profile: Optional[str] = None, sha: str = "",
                           settings: Optional[Dict[str, Any]] = None,
                           ) -> Dict[str, Any]:
    """Запускает фоновую сборку индекса (её задачу возвращает в `task`).

    Отдельное имя — потому что этим пользуется команда включения режима: она не
    ждёт индексацию (она в фоне), но держит ссылку на задачу, чтобы её нельзя было
    снять сборщиком мусора и чтобы проверки могли дождаться окончания.
    """
    return await index_in_background(root, profile=profile, sha=sha,
                                     settings=settings)


# ---------------------------------------------------------------------------
# Поиск нужных чанков по правке
# ---------------------------------------------------------------------------
def queries(files: List[Dict[str, Any]], names: List[str], note: str = "",
            limit: int = QUERIES) -> List[str]:
    """Запросы к индексу — из САМОЙ ПРАВКИ: имена, модули, пожелание человека.

    Почему не один запрос: правка обычно трогает несколько несвязанных мест, и
    один общий запрос усредняет их в нечто среднее — поиск отдаёт фрагменты «про
    проект вообще». Три точечных запроса (что менялось, о чём просили, как это
    устроено) дают выборку по делу.
    """
    paths = [str(item.get("path") or "") for item in (files or [])
             if isinstance(item, dict)]
    packs = sorted({"/".join(path.split("/")[:-1]) or path for path in paths if path})
    words = [str(name) for name in (names or []) if str(name).strip()]
    out: List[str] = []
    first = " ".join(words[:8] + packs[:4]).strip()
    if first:
        out.append(first[:300])
    if str(note or "").strip():
        out.append(str(note).strip()[:300] + " " + " ".join(packs[:2]))
    second = ("как устроено в проекте: " + " ".join(packs[:3])).strip()
    if second and second not in out:
        out.append(second[:300])
    if not out:
        out.append(ai_review.query_for(files, names, note)[:300])
    return out[:max(1, int(limit or 1))]


def search(base_ids: Any, query_list: Any, *, profile: Optional[str] = None,
           settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Поиск по индексу НЕСКОЛЬКИМИ запросами с удалением повторов.

    Возвращает данные RAG в том же виде, что `rag_search.search` (один словарь):
    блок для модели, карточки источников и диагностика собираются ИМ ЖЕ, поэтому
    ревью не обрастает своим форматом фрагментов. Повторы (один и тот же чанк из
    разных запросов) убираются, порядок — по релевантности.
    """
    ids = [str(item) for item in (base_ids or []) if str(item).strip()]
    queries_list = [str(item).strip() for item in (query_list or [])
                    if str(item).strip()]
    if not ids or not queries_list:
        return {}
    merged: Dict[str, Any] = {}
    seen: Dict[str, bool] = {}
    hits: List[Dict[str, Any]] = []
    bases: List[Dict[str, Any]] = []
    stages: Dict[str, Any] = {}
    for query in queries_list:
        data = rag_search.search(ids, query, profile=profile, settings=settings)
        if not merged:
            merged = data
            stages = dict((data or {}).get("stages") or {})
        for item in rag_search.hits_of(data):
            key = "%s|%s|%s" % (item.get("base_id"), item.get("chunk_id"),
                                item.get("number"))
            if seen.get(key):
                continue
            seen[key] = True
            hits.append(item)
        for item in (data or {}).get("bases") or []:
            if item not in bases:
                bases.append(item)
    hits.sort(key=lambda item: -float(item.get("score") or 0.0))
    limit = int(stages.get("top_k_after") or len(hits) or 1)
    out = dict(merged)
    out.update({"query": " | ".join(queries_list), "hits": hits[:max(1, limit)],
                "bases": bases, "stages": stages,
                "query_count": len(queries_list),
                # Тот же запрос — те же фрагменты: подпись считается по всем базам
                # и всем запросам, иначе кэш поиска склеил бы разные выборки.
                "queries": list(queries_list)})
    return out


def context_info(index: Optional[Dict[str, Any]], data: Dict[str, Any],
                 *, base_name: str = BASE_NAME) -> Dict[str, Any]:
    """Сведения о контексте для ОТЧЁТА: чем он собран и сколько в нём фрагментов.

    Форма та же, что у отбора контекста кодом (`ai_review.context_block`), плюс
    `chunks` и `base`: шапка отчёта должна называть ИСТОЧНИК контекста — «собран
    поиском по индексу кода» или «отобран кодом», — а не молчать об этом.
    """
    hits = rag_search.hits_of(data) if data else []
    index = index if isinstance(index, dict) else {}
    return {"block": rag_search.block(data) if data else "", "whole": [],
            "related": [], "docs": [], "chars": len(rag_search.block(data)) if data else 0,
            "chunks": len(hits), "base": base_name if hits else "",
            "index": {"built": bool(index.get("built")),
                      "chunks": int(index.get("chunks") or 0),
                      "documents": int(index.get("documents") or 0),
                      "backend": str(index.get("backend") or ""),
                      "seconds": float(index.get("seconds") or 0.0)}}
