"""Фоновые задачи индексации RAG: состояние, прогресс, отмена.

Зачем отдельный модуль. Индексация крупного документа — это минуты: разбор PDF
идёт по страницам, потом модель считает векторы для каждого чанка. Держать на
этом HTTP-запрос нельзя: браузер (и любой посредник) может оборвать соединение,
а пользователь всё это время не видит ничего, кроме «индексирую…». Поэтому
индексация уходит в ФОН, а интерфейс опрашивает её состояние и рисует прогресс:
этап, счётчик и проценты.

    POST /api/agent/rag/upload/stream   → тело принято, заведена ЗАДАЧА
    GET  /api/agent/rag/jobs            → что сейчас происходит (опрос раз в секунду)
    POST /api/agent/rag/jobs/{id}/cancel → отменить

СОСТОЯНИЕ ЖИВЁТ В ПАМЯТИ ПРОЦЕССА. Это осознанно: задача — это «прямо сейчас
идёт работа над базой», а не долговременные данные. Перезапуск приложения
индексацию всё равно прерывает (поток умирает), поэтому хранить её в файле
незачем — а недоделанный временный файл убирает `prune_incoming`.

ЭТАПЫ И ВЕСА. Проценты считаются по этапам с весами, а не «сколько сделано
вообще»: у разбора и у эмбеддингов разная цена, и без весов полоса прогресса
замирала бы на 50% на две трети времени. Внутри этапа процент — от счётчика
(страницы PDF, батчи эмбеддингов), а если общее число ещё неизвестно, полоса
показывается неопределённой, но текстовое описание всё равно есть.

ОТМЕНА — кооперативная: `cancel()` ставит флаг, а работник проверяет его в
обработчике прогресса (между страницами и батчами) и прекращает работу. До
`save_index` индекс НЕ пишется, поэтому отмена не оставляет ни половины базы, ни
мусора: старый индекс остаётся целым.
"""

import asyncio
import logging
import os
import re
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.ai import rag
from app.ai import rag_store

logger = logging.getLogger(__name__)

# Идентификатор задачи: «job-» + 8 шестнадцатеричных знаков — как у баз, чтобы
# проверка шаблоном защищала и здесь (идентификатор приходит из интерфейса).
ID_PREFIX = "job-"
ID_RE = re.compile(r"^job-[0-9a-f]{8}$")

# Сколько задач помним и сколько считаем одновременно. Помним и завершённые:
# интерфейс должен успеть показать итог (и ошибку), даже если опрос чуть отстал.
MAX_JOBS = 20
MAX_RUNNING = 4
KEEP_FINISHED_SECONDS = 900

# Этапы: (ключ, человеческое имя, начало %, конец %). Веса подобраны по замерам:
# у текстового PDF разбор занимает ~40% времени, эмбеддинги ~55%, запись ~5%.
STAGES: Tuple[Tuple[str, str, int, int], ...] = (
    ("queued", "в очереди", 0, 1),
    ("extract", "разбор документов", 1, 40),
    ("embed", "эмбеддинги", 40, 95),
    ("save", "запись индекса", 95, 100),
)
# А у СКАНА всё наоборот: распознавание страниц — это и есть почти вся работа
# (на плотной странице A4 около секунды, у 256-страничного скана — десятки
# минут), а эмбеддинги и запись рядом с этим мгновенны. С общими весами полоса
# замирала бы на 1% до самого конца распознавания — ровно то, на что жалуются.
STAGES_OCR: Tuple[Tuple[str, str, int, int], ...] = (
    ("queued", "в очереди", 0, 1),
    ("extract", "распознавание и разбор", 1, 92),
    ("embed", "эмбеддинги", 92, 99),
    ("save", "запись индекса", 99, 100),
)
_STAGE_MAPS = {
    False: {item[0]: item for item in STAGES},
    True: {item[0]: item for item in STAGES_OCR},
}
_STAGE_MAP = _STAGE_MAPS[False]

_LOCK = threading.RLock()
_JOBS: Dict[str, Dict[str, Any]] = {}
_TASKS: Dict[str, Any] = {}


def _now() -> str:
    """Метка времени (ISO, до секунд) — как у задач и баз."""
    return datetime.now().isoformat(timespec="seconds")


def valid_id(raw: Any) -> bool:
    """Похож ли идентификатор на идентификатор задачи."""
    return bool(ID_RE.match(str(raw or "").strip().lower()))


def _percent(stage: str, done: float, total: float, heavy: bool = False) -> int:
    """Проценты по этапу и счётчику (начало этапа — если общее число неизвестно).

    `done` и `total` — ДРОБНЫЕ: внутри этапа это может быть «страница 28 из 256»,
    то есть доля страницы, а не целое число файлов. Целочисленное деление здесь
    и давало застывшую на 1% полосу у одного большого скана.
    """
    item = _STAGE_MAPS[bool(heavy)].get(stage)
    if item is None:
        return 0
    start, end = item[2], item[3]
    if total <= 0:
        # Общее число ещё не знаем: показываем начало этапа, а не выдуманную
        # середину — «полоса на 40%» честнее «полосы на 70%».
        return start
    share = max(0.0, min(1.0, float(done) / float(total)))
    return int(round(start + (end - start) * share))


def _eta(job: Dict[str, Any]) -> float:
    """Сколько ещё ждать ТЕКУЩИЙ этап, секунды (0 — посчитать нельзя).

    Оценивается по уже наблюдённой скорости внутри этапа: «осталось столько же,
    сколько заняло сделанное». Это честная оценка этапа, а не всего задания:
    у следующих этапов своя скорость, и обещать общее время было бы гаданием.
    """
    total = float(job.get("total") or 0)
    done = float(job.get("done") or 0)
    started = job.get("_stage_at")
    if total <= 0 or done <= 0 or not started:
        return 0.0
    elapsed = time.time() - float(started)
    if elapsed <= 0.5:                 # рано судить: скорость ещё не видна
        return 0.0
    rate = done / elapsed
    if rate <= 0:
        return 0.0
    return max(0.0, (total - done) / rate)


def _payload(job: Dict[str, Any]) -> Dict[str, Any]:
    """Снимок задачи для интерфейса (без внутренних полей)."""
    started = job.get("started") or _now()
    elapsed = job.get("elapsed")
    if job.get("state") == "running":
        elapsed = time.time() - job.get("_started_at", time.time())
    return {
        "id": job.get("id") or "",
        "state": job.get("state") or "running",
        "stage": job.get("stage") or "queued",
        "stage_name": _STAGE_MAPS[bool(job.get("heavy"))].get(
            job.get("stage") or "queued", ("", "", 0, 0))[1],
        "done": int(float(job.get("done") or 0)),
        "total": int(float(job.get("total") or 0)),
        "percent": int(job.get("percent") or 0),
        # Оценка «сколько ещё ждать» — по текущему этапу (см. _eta).
        "eta": round(_eta(job) if job.get("state") == "running" else 0.0, 1),
        "heavy": bool(job.get("heavy")),
        "detail": job.get("detail") or "",
        "base_id": job.get("base_id") or "",
        "name": job.get("name") or "",
        "append": bool(job.get("append")),
        "chunks": int(job.get("chunks") or 0),
        "documents": int(job.get("documents") or 0),
        "error": job.get("error") or "",
        "cancel_requested": bool(job.get("cancel")),
        "started": started,
        "finished": job.get("finished") or "",
        "elapsed": round(float(elapsed or 0.0), 1),
        "result": job.get("result") or None,
    }


def _prune_locked() -> None:
    """Убирает старые завершённые задачи (память процесса не бесконечна)."""
    now = time.time()
    finished = [(job.get("_finished_at", 0.0), key) for key, job in _JOBS.items()
                if job.get("state") != "running"]
    for stamp, key in sorted(finished):
        if now - stamp > KEEP_FINISHED_SECONDS or len(_JOBS) > MAX_JOBS:
            _JOBS.pop(key, None)
    while len(_JOBS) > MAX_JOBS:
        oldest = min(_JOBS.items(), key=lambda pair: pair[1].get("_started_at", 0.0))
        if oldest[1].get("state") == "running":
            break
        _JOBS.pop(oldest[0], None)


def create(*, profile: Optional[str], name: str = "", base_id: str = "",
           append: bool = False) -> Dict[str, Any]:
    """Заводит задачу индексации в состоянии «в очереди».

    Задача создаётся ДО работы, а не после: пользователь должен получить
    идентификатор сразу, чтобы опрашивать прогресс, даже если сам разбор начнётся
    через мгновение.
    """
    job_id = ID_PREFIX + uuid.uuid4().hex[:8]
    with _LOCK:
        running = sum(1 for item in _JOBS.values() if item.get("state") == "running")
        if running >= MAX_RUNNING:
            raise rag.RagError("одновременно индексируется уже %d баз — "
                               "дождитесь окончания" % running)
        _prune_locked()
        _JOBS[job_id] = {
            "id": job_id,
            "state": "running",
            "stage": "queued",
            "percent": _percent("queued", 0, 0),
            "heavy": False,
            "_stage_at": time.time(),
            "done": 0, "total": 0,
            "detail": "задача заведена",
            "profile": str(profile or ""),
            "base_id": str(base_id or ""),
            "append": bool(append),
            "name": str(name or ""),
            "started": _now(),
            "_started_at": time.time(),
            "cancel": False,
        }
        return _payload(_JOBS[job_id])


def progress(job_id: str, stage: str, done: float = 0, total: float = 0,
             detail: str = "", heavy: bool = False) -> bool:
    """Обновляет ход задачи. False — задача отменена (работнику пора выходить).

    Вызывается из РАБОЧЕГО ПОТОКА, а читается из цикла событий, поэтому доступ
    под замком. Возвращаемое значение используется как «проверка отмены»: одна
    функция на два дела, потому что отменять работу имеет смысл только между
    шагами, где прогресс и так обновляется.

    `heavy=True` — разбор оказался распознаванием скана: этап занимает почти всё
    время, и веса этапов берутся из STAGES_OCR, иначе полоса не двигалась бы.
    """
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return False
        changed = job.get("stage") != stage
        if stage in _STAGE_MAP:
            job["stage"] = stage
        if heavy:
            job["heavy"] = True
        if changed or job.get("_stage_at") is None:
            # Скорость считается ОТ НАЧАЛА ЭТАПА: у этапов она разная, и общая
            # оценка от старта задачи врала бы в разы.
            job["_stage_at"] = time.time()
            job["_stage_base"] = done
        job["done"] = float(done or 0)
        job["total"] = float(total or 0)
        job["percent"] = _percent(job["stage"], job["done"], job["total"],
                                  bool(job.get("heavy")))
        job["detail"] = str(detail or "")
        return not job.get("cancel")


def set_fields(job_id: str, **fields: Any) -> None:
    """Дописывает поля задачи (например, id базы после первой записи)."""
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is not None:
            job.update(fields)


def finish(job_id: str, meta: Dict[str, Any]) -> None:
    """Задача завершена успешно: сохраняем ИТОГ для интерфейса и чата."""
    stats = (meta or {}).get("stats") or {}
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return
        job.update({
            "state": "done",
            "stage": "save",
            "percent": 100,
            "detail": "готово",
            "base_id": str((meta or {}).get("id") or job.get("base_id") or ""),
            "name": str((meta or {}).get("name") or job.get("name") or ""),
            "chunks": int(stats.get("chunks") or 0),
            "documents": int(stats.get("documents") or 0),
            "finished": _now(),
            "_finished_at": time.time(),
            "elapsed": time.time() - job.get("_started_at", time.time()),
            "error": "",
            # Итог — только то, что нужно интерфейсу для сообщения: сам паспорт
            # базы в задаче не держим (в нём чанки и документы).
            "result": {
                "id": (meta or {}).get("id") or "",
                "name": (meta or {}).get("name") or "",
                "strategy_name": (meta or {}).get("strategy_name") or "",
                "chunk_size": int((meta or {}).get("chunk_size") or 0),
                "overlap": int((meta or {}).get("overlap") or 0),
                "chunks": int(stats.get("chunks") or 0),
                "documents": int(stats.get("documents") or 0),
                "chars_avg": int(stats.get("chars_avg") or 0),
                "size_human": (meta or {}).get("size_human") or "",
                "failures": [str(item)[:200] for item in ((meta or {}).get("failures") or [])],
            },
        })
        logger.info("RAG: задача %s завершена (%s, %.1f с)", job_id, job.get("name"),
                    job["elapsed"])


def fail(job_id: str, error: str) -> None:
    """Задача не удалась: причина остаётся в задаче и видна в интерфейсе."""
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return
        job.update({
            "state": "failed",
            "detail": "не удалось",
            "error": str(error or "индексация не удалась")[:400],
            "finished": _now(),
            "_finished_at": time.time(),
            "elapsed": time.time() - job.get("_started_at", time.time()),
        })
        logger.warning("RAG: задача %s не удалась — %s", job_id, str(error)[:200])


def cancelled(job_id: str) -> None:
    """Задача остановлена пользователем (индекс НЕ записан)."""
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return
        job.update({
            "state": "cancelled",
            "detail": "отменено",
            "error": "",
            "finished": _now(),
            "_finished_at": time.time(),
            "elapsed": time.time() - job.get("_started_at", time.time()),
        })
        logger.info("RAG: задача %s отменена пользователем", job_id)


def cancel(job_id: Any, profile: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Просит остановить задачу. None — задачи нет или она чужая.

    Сама работа прекращается не мгновенно: работник проверяет флаг между
    страницами и батчами (см. progress). До записи индекса отмена безопасна —
    прежний индекс остаётся целым.
    """
    with _LOCK:
        job = _JOBS.get(str(job_id or "").strip().lower())
        if job is None or not _visible(job, profile):
            return None
        if job.get("state") == "running":
            job["cancel"] = True
            job["detail"] = "останавливаю…"
        return _payload(job)


def _visible(job: Dict[str, Any], profile: Optional[str]) -> bool:
    """Своя ли задача профилю (профили изолированы, как задачи и базы)."""
    if not profile:
        return True
    return str(job.get("profile") or "") in ("", str(profile))


def get(job_id: Any, profile: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Снимок одной задачи (None — нет такой или она чужого профиля)."""
    key = str(job_id or "").strip().lower()
    if not valid_id(key):
        return None
    with _LOCK:
        job = _JOBS.get(key)
        if job is None or not _visible(job, profile):
            return None
        return _payload(job)


def listing(profile: Optional[str] = None, active_only: bool = False) -> List[Dict[str, Any]]:
    """Задачи профиля: сначала идущие, потом завершённые (свежие вперёд)."""
    with _LOCK:
        _prune_locked()
        items = [_payload(job) for job in _JOBS.values() if _visible(job, profile)]
    if active_only:
        items = [item for item in items if item["state"] == "running"]
    items.sort(key=lambda item: (item["state"] != "running", item["started"]), reverse=False)
    return items


def running_for_base(base_id: Any) -> Optional[Dict[str, Any]]:
    """Идущая задача по этой базе (чтобы не запустить вторую на неё же)."""
    key = str(base_id or "").strip().lower()
    if not key:
        return None
    with _LOCK:
        for job in _JOBS.values():
            if job.get("state") == "running" and str(job.get("base_id") or "") == key:
                return _payload(job)
    return None


def active_count() -> int:
    """Сколько индексаций идёт прямо сейчас (для подписи кнопки и проверок)."""
    with _LOCK:
        return sum(1 for job in _JOBS.values() if job.get("state") == "running")


def _run_sync(job_id: str, files: List[Dict[str, Any]], *, profile: Optional[str],
              name: str, strategy: Any, chunk_size: Any, overlap: Any,
              base_id: str) -> Dict[str, Any]:
    """Работа задачи: индексация с отчётом о ходе. Выполняется В ПОТОКЕ."""

    def report(stage: str, done: float, total: float, detail: str = "",
               heavy: bool = False) -> None:
        """Отчёт о ходе. Отмена приходит сюда же (см. progress)."""
        if not progress(job_id, stage, done, total, detail, heavy=heavy):
            raise _Cancelled()

    if base_id:
        return rag.append_files(base_id, files, profile=profile,
                                on_progress=_adapter(report, "extract"))
    return rag.index_files(files, name=name, profile=profile, strategy=strategy,
                           chunk_size=chunk_size, overlap=overlap,
                           on_progress=_adapter(report, "extract"))


class _Cancelled(BaseException):
    """Работа остановлена пользователем (внутренний сигнал задачи).

    Наследуется от BaseException, а НЕ от Exception, и это принципиально: отчёт о
    ходе идёт через наблюдателей, которые намеренно глотают любые СВОИ сбои
    (`except Exception`), чтобы ошибка наблюдателя не ломала индексацию. Отмена,
    брошенная из такого наблюдателя, через это «глотание» не проходит — и работа
    продолжалась бы до конца, хотя пользователь нажал «остановить».
    """


def _adapter(report: Callable[..., None], default_stage: str) -> Callable[..., None]:
    """Приводит вызовы пайплайна к единому виду отчёта.

    Пайплайн зовёт `on_progress(stage, done, total)` (см. rag._progress), а
    разбор PDF внутри сообщает о страницах отдельным колбэком. Обе формы должны
    попасть в состояние задачи, поэтому здесь они и сводятся вместе.
    """
    def callback(stage: str, done: float, total: float, detail: str = "",
                 heavy: bool = False) -> None:
        report(stage or default_stage, done, total, detail, heavy)
    return callback


async def start(*, profile: Optional[str], files: List[Dict[str, Any]], name: str = "",
                strategy: Any = None, chunk_size: Any = None, overlap: Any = None,
                base_id: str = "") -> Dict[str, Any]:
    """Заводит задачу и запускает её В ФОНЕ. Возвращает снимок задачи.

    Файлы уже на диске (потоковая загрузка) или в памяти (`data`): задача только
    читает их. Временные файлы удаляются по окончании работы — и при успехе, и
    при ошибке, и при отмене (см. `_finish_job`).
    """
    base_id = str(base_id or "").strip().lower()
    if base_id and running_for_base(base_id):
        raise rag.RagError("по этой базе уже идёт индексация — дождитесь окончания")
    job = create(profile=profile, name=name, base_id=base_id, append=bool(base_id))
    job_id = job["id"]
    task = asyncio.ensure_future(_run(job_id, files, profile=profile, name=name,
                                      strategy=strategy, chunk_size=chunk_size,
                                      overlap=overlap, base_id=base_id))
    _TASKS[job_id] = task
    task.add_done_callback(lambda _task: _TASKS.pop(job_id, None))
    return job


async def _run(job_id: str, files: List[Dict[str, Any]], **params: Any) -> None:
    """Выполняет работу в отдельном потоке и записывает итог в состояние задачи."""
    try:
        meta = await asyncio.to_thread(_run_sync, job_id, files, **params)
    except _Cancelled:
        cancelled(job_id)
    except rag.RagError as exc:
        fail(job_id, str(exc))
    except Exception as exc:               # сбой индексации не должен теряться
        logger.exception("RAG: задача %s упала", job_id)
        fail(job_id, "%s: %s" % (type(exc).__name__, str(exc)[:200]))
    else:
        finish(job_id, meta)
    finally:
        _cleanup_files(files)


def _cleanup_files(files: List[Dict[str, Any]]) -> None:
    """Убирает временные файлы задачи (потоковая загрузка пишет их на диск)."""
    for item in files or []:
        path = item.get("path")
        if isinstance(path, str) and path:
            try:
                os.unlink(path)
            except OSError:
                pass


def prune_incoming(older_than: float = 3600.0) -> int:
    """Убирает брошенные временные файлы загрузок (старше часа).

    Задача, пережившая перезапуск приложения, своей работы не закончит, а её
    файл остался бы в `.incoming` навсегда. Час — потому что идущая индексация
    законного 250-МБ документа может быть долгой.
    """
    folder = os.path.join(rag_store.directory(), ".incoming")
    if not os.path.isdir(folder):
        return 0
    removed = 0
    threshold = time.time() - older_than
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        try:
            if os.path.isfile(path) and os.path.getmtime(path) < threshold:
                os.unlink(path)
                removed += 1
        except OSError:
            continue
    if removed:
        logger.info("RAG: убрано брошенных файлов загрузки: %d", removed)
    return removed


def summary(profile: Optional[str] = None) -> Dict[str, Any]:
    """Сводка для снимка баз знаний: идёт ли что-то и сколько."""
    items = listing(profile=profile)
    active = [item for item in items if item["state"] == "running"]
    return {
        "jobs": items[:5],
        "active": len(active),
        "running": bool(active),
        "percent": max([item["percent"] for item in active], default=0),
    }
