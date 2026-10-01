"""OCR сканов через встроенный в macOS Vision (распознавание текста в PDF и картинках).

ЗАЧЕМ ЭТО НУЖНО. Скан (документ, отсканированный как изображения) текстового
слоя не имеет — индексировать в нём нечего, и раньше загрузка честно падала с
«нужен OCR». Теперь OCR ЕСТЬ, и он запасной путь: обычный PDF с текстовым слоем
индексируется как раньше, распознавание включается только тогда, когда текста в
файле не нашлось.

ПОЧЕМУ СВОЙ ХЕЛПЕР НА SWIFT. `pyobjc-framework-Vision` требует Python ≥3.10, а
venv проекта — 3.9 (та же стена, что у sentence-transformers). Зато Vision есть в
самой macOS: работает ОФЛАЙН, понимает русский, не требует ни brew, ни tesseract,
ни интернета. Поэтому здесь: исходник `tools/vision_ocr.swift`, сборка через уже
установленный `swiftc` в кэш проекта (`data/rag/bin`) и запуск как процесса.

НАСТРОЙКИ (переменные окружения):

    RAG_OCR        auto | off | always   (по умолчанию auto)
                   auto   — распознавать, когда текстового слоя в файле нет;
                   off    — не распознавать вовсе (остаётся прежнее «нужен OCR»);
                   always — распознавать ВСЕГДА и брать результат, если он длиннее
                            того, что дал текстовый слой (для сканов с испорченным
                            или частичным слоем);
    RAG_OCR_LANGS  языки распознавания, по умолчанию ru-RU,en-US;
    RAG_OCR_HELPER путь к уже собранному хелперу (если задан — сборки не будет).

РИСКИ, О КОТОРЫХ ЧЕСТНО. Распознанный текст несовершенен: опечатки, склейка
строк, потеря таблиц. Поэтому факт распознавания записывается в предупреждение
документа и виден в паспорте базы, а не выдаётся за исходный текст.
"""

import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Исходник хелпера лежит рядом с проверками (tools/), собранный бинарь — в
# каталоге проекта под data/ (в git не хранится).
SOURCE_NAME = "vision_ocr.swift"
HELPER_NAME = "vision_ocr"

MODE_ENV = "RAG_OCR"
LANGS_ENV = "RAG_OCR_LANGS"
HELPER_ENV = "RAG_OCR_HELPER"

MODE_AUTO = "auto"
MODE_OFF = "off"
MODE_ALWAYS = "always"
MODES = (MODE_AUTO, MODE_OFF, MODE_ALWAYS)

DEFAULT_LANGS = "ru-RU,en-US"

# Сборка хелпера — секунды (на этой машине ~40 с), поэтому она делается один раз
# и под замком: параллельные загрузки не должны компилировать его одновременно.
_BUILD_LOCK = threading.Lock()
_HELPER_STATE: Dict[str, Any] = {"path": "", "error": "", "tried": False}

# Сколько ждать распознавание. Одна страница — около секунды, поэтому предел
# считается от числа страниц, а не берётся фиксированным.
SECONDS_PER_PAGE = 30.0
MIN_TIMEOUT = 120.0


def _env(name: str) -> str:
    return (os.getenv(name) or "").strip()


def mode() -> str:
    """Режим распознавания из настроек (неизвестное значение — auto)."""
    value = _env(MODE_ENV).lower()
    return value if value in MODES else MODE_AUTO


def languages() -> List[str]:
    """Языки распознавания (по умолчанию русский и английский)."""
    raw = _env(LANGS_ENV) or DEFAULT_LANGS
    return [item.strip() for item in raw.split(",") if item.strip()]


def supported_platform() -> bool:
    """Vision есть только в macOS — на других системах распознавания нет."""
    return sys.platform == "darwin"


def helper_dir() -> str:
    """Каталог собранного хелпера.

    Именно `data/ocr/bin` В КОРНЕ ПРОЕКТА, а не внутри `RAG_DIR`: хелпер — это
    ПРОГРАММА, а не данные базы знаний. Сборка занимает десятки секунд, и если
    привязывать её к каталогу баз, то любая смена `RAG_DIR` (или временный каталог
    в проверках) заставляла бы собирать заново.
    """
    try:
        from app import config
        root = config.PROJECT_ROOT
    except Exception:                      # pragma: no cover - отдельный запуск
        root = os.getcwd()
    return os.path.join(root, "data", "ocr", "bin")


def source_path() -> str:
    """Путь к исходнику хелпера (лежит рядом с проверками проекта)."""
    try:
        from app import config
        root = config.PROJECT_ROOT
    except Exception:                      # pragma: no cover
        root = os.getcwd()
    return os.path.join(root, "tools", SOURCE_NAME)


def helper_path() -> str:
    """Путь к собранному хелперу (сборка — по требованию, см. ensure_helper)."""
    explicit = _env(HELPER_ENV)
    if explicit:
        return explicit
    return os.path.join(helper_dir(), HELPER_NAME)


def status() -> Dict[str, Any]:
    """Состояние OCR для снимка интерфейса: доступен ли и чем распознаёт.

    Хелпер здесь НЕ собирается: диалог баз знаний должен открываться мгновенно,
    а сборка занимает секунды. Сообщается только то, что видно сразу.
    """
    mode_value = mode()
    if mode_value == MODE_OFF:
        return {"available": False, "mode": mode_value, "reason": "выключено настройкой",
                "languages": languages(), "helper": "", "compiler": ""}
    if not supported_platform():
        return {"available": False, "mode": mode_value, "languages": languages(),
                "reason": "распознавание через Vision работает только в macOS",
                "helper": "", "compiler": ""}
    helper = helper_path()
    compiler = shutil.which("swiftc") or ""
    if os.path.isfile(helper):
        return {"available": True, "mode": mode_value, "reason": "",
                "languages": languages(), "helper": helper, "compiler": compiler}
    if not _env(HELPER_ENV) and not os.path.isfile(source_path()):
        return {"available": False, "mode": mode_value, "languages": languages(),
                "reason": "не найден исходник хелпера %s" % SOURCE_NAME,
                "helper": helper, "compiler": compiler}
    if not compiler:
        return {"available": False, "mode": mode_value, "languages": languages(),
                "reason": "нет swiftc (нужен Xcode Command Line Tools) — хелпер не собрать",
                "helper": helper, "compiler": ""}
    return {"available": True, "mode": mode_value, "reason": "",
            "languages": languages(), "helper": helper, "compiler": compiler,
            "build_required": True}


def ensure_helper(force: bool = False) -> Tuple[str, str]:
    """Собирает хелпер при необходимости. Возвращает (путь, причина сбоя).

    Сборка идёт в каталог проекта (`data/rag/bin`), а кэш модулей компилятора —
    туда же: `swiftc` по умолчанию пишет его в системный каталог, а в
    ограниченном окружении такая запись запрещена, и сборка падает с
    «Operation not permitted».
    """
    explicit = _env(HELPER_ENV)
    if explicit:
        if os.path.isfile(explicit) and os.access(explicit, os.X_OK):
            return explicit, ""
        return "", "хелпер по пути RAG_OCR_HELPER не найден или не исполняемый"
    helper = helper_path()
    with _BUILD_LOCK:
        if os.path.isfile(helper) and not force:
            return helper, ""
        if _HELPER_STATE["tried"] and _HELPER_STATE["error"] and not force:
            return "", _HELPER_STATE["error"]
        _HELPER_STATE.update({"tried": True, "error": ""})
        if not supported_platform():
            _HELPER_STATE["error"] = "распознавание доступно только в macOS"
            return "", _HELPER_STATE["error"]
        compiler = shutil.which("swiftc")
        if not compiler:
            _HELPER_STATE["error"] = ("нет swiftc — установите Xcode Command Line Tools "
                                      "(xcode-select --install)")
            return "", _HELPER_STATE["error"]
        source = source_path()
        if not os.path.isfile(source):
            _HELPER_STATE["error"] = "не найден исходник %s" % source
            return "", _HELPER_STATE["error"]
        try:
            os.makedirs(helper_dir(), exist_ok=True)
        except OSError as exc:
            _HELPER_STATE["error"] = "каталог для хелпера недоступен: %s" % str(exc)[:120]
            return "", _HELPER_STATE["error"]
        cache = os.path.join(helper_dir(), "modulecache")
        try:
            os.makedirs(cache, exist_ok=True)
        except OSError:
            cache = helper_dir()
        target = helper + ".tmp"
        command = [compiler, "-O", "-module-cache-path", cache, source, "-o", target,
                   "-framework", "Vision", "-framework", "PDFKit",
                   "-framework", "ImageIO"]
        started = time.time()
        try:
            done = subprocess.run(command, capture_output=True, timeout=600)
        except (OSError, subprocess.TimeoutExpired) as exc:
            _HELPER_STATE["error"] = "сборка хелпера не удалась: %s" % str(exc)[:160]
            logger.warning("RAG OCR: %s", _HELPER_STATE["error"])
            return "", _HELPER_STATE["error"]
        if done.returncode != 0 or not os.path.isfile(target):
            detail = (done.stderr or b"").decode("utf-8", "replace").strip().split("\n")[-1:]
            _HELPER_STATE["error"] = ("сборка хелпера не удалась: %s"
                                      % (detail[0][:200] if detail else "swiftc вернул ошибку"))
            logger.warning("RAG OCR: %s", _HELPER_STATE["error"])
            return "", _HELPER_STATE["error"]
        os.replace(target, helper)
        os.chmod(helper, 0o755)
        logger.info("RAG OCR: хелпер собран за %.1f с (%s)", time.time() - started, helper)
        return helper, ""


def available() -> bool:
    """Доступно ли распознавание (по настройке и по среде)."""
    state = status()
    return bool(state["available"])


def should_recognize(text_chars: int, has_images: bool = False) -> bool:
    """Нужно ли распознавать документ.

    `auto` — только когда текста нет вовсе (обычный текстовый PDF не трогаем:
    распознавание медленнее и менее точно, чем готовый слой). `always` —
    независимо от слоя, для сканов с испорченным или частичным слоем.
    """
    if mode() == MODE_OFF or not available():
        return False
    if mode() == MODE_ALWAYS:
        return True
    return int(text_chars or 0) == 0


def recognize(path: str, on_progress: Optional[Callable[[int, int], None]] = None,
              pages: int = 0) -> Tuple[str, str]:
    """Распознаёт текст файла. Возвращает (текст, предупреждение/причина).

    Прогресс идёт ПО СТРАНИЦАМ: хелпер печатает «PROGRESS <готово> <всего>» в
    stderr, и эти строки сразу уходят наблюдателю — на 200-страничном скане
    пользователь видит «распознаю: страница 40 из 200», а не молчание.

    Отмена поддерживается: если наблюдатель бросает исключение (это делает задача
    индексации), процесс хелпера УБИВАЕТСЯ, а исключение уходит дальше — иначе
    распознавание продолжалось бы после нажатия «остановить».
    """
    helper, error = ensure_helper()
    if not helper:
        return "", error or "распознавание недоступно"
    # Текст распознавания идёт в ФАЙЛ, а канал остаётся для прогресса.
    # Причина: читая stderr до конца, мы не вычерпываем stdout, и как только
    # буфер канала (64 КБ ≈ 20–30 страниц скана) заполняется, хелпер ЗАВИСАЕТ —
    # прогресс останавливается, а индексация не заканчивается никогда. Живой
    # случай: «уже полторы минуты 1%» на скане в 256 страниц.
    handle, text_path = tempfile.mkstemp(prefix="rag-ocr-text-", suffix=".txt")
    os.close(handle)
    command = [helper, path, ",".join(languages())]
    command.append(str(int(pages)) if pages else "0")
    command.append(str(int(_env_int("RAG_OCR_MAX_PX", 2400))))
    command.append(text_path)
    timeout = max(MIN_TIMEOUT, SECONDS_PER_PAGE * (pages or 100))
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True,
                                   encoding="utf-8", errors="replace")
    except OSError as exc:
        return "", "не удалось запустить распознавание: %s" % str(exc)[:150]

    warnings: List[str] = []
    try:
        assert process.stderr is not None
        # Читаем ТОЛЬКО канал прогресса (текст идёт в файл) — дедлока быть не
        # может: объём stderr крохотный, он не заполнит буфер.
        for line in process.stderr:
            line = line.strip()
            if line.startswith("PROGRESS"):
                parts = line.split()
                if len(parts) == 3 and on_progress is not None:
                    try:
                        on_progress(int(parts[1]), int(parts[2]))
                    except Exception:
                        # Отмена (или сбой наблюдателя): хелпер больше не нужен.
                        process.kill()
                        raise
            elif line.startswith("WARN"):
                warnings.append(line[4:].strip())
            elif line.startswith("ERROR"):
                warnings.append(line[5:].strip())
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            return "", "распознавание не уложилось в отведённое время"
        # Текст забираем ИЗ ФАЙЛА: он может быть в десятки мегабайт, и держать
        # его в канале незачем.
        text = _read_text(text_path)
    finally:
        if process.poll() is None:
            process.kill()
        for stream in (process.stdout, process.stderr):
            try:
                if stream:
                    stream.close()
            except OSError:
                pass
        try:
            os.unlink(text_path)
        except OSError:
            pass

    text = _clean(text)
    if process.returncode not in (0,) and not text.strip():
        reason = warnings[-1] if warnings else "распознавание не дало текста"
        return "", reason
    if not text.strip():
        return "", (warnings[-1] if warnings else "текст не распознан")
    return text, "; ".join(warnings)


def _env_int(name: str, default: int) -> int:
    """Целое из переменной окружения (нечисло — значение по умолчанию)."""
    try:
        return int(float((os.getenv(name) or "").strip()))
    except ValueError:
        return default


def _read_text(path: str) -> str:
    """Читает распознанный текст из файла (потоком, чтобы не держать копии)."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError as exc:
        logger.warning("RAG OCR: текст распознавания не прочитан — %s", str(exc)[:150])
        return ""


# Перенос слова на границе строки: «резерв-» + «ное копирование». В книге правил
# и в газетной вёрстке переносы часты, и склейка через пробел оставляла бы
# «резерв- ное» — слово, которого нет ни в документе, ни в языке.
_HYPHEN_TAIL = re.compile(r"[^\W\d_]-$", re.UNICODE)


def _join_lines(lines: List[str]) -> str:
    """Склеивает строки одного абзаца: перенос в конце строки — без дефиса.

    Обычные строки соединяются пробелом (в вёрстке абзац рвётся по ширине
    колонки, а не по смыслу), а строка, кончающаяся дефисом между буквами,
    стыкуется со следующей напрямую — так восстанавливается целое слово.
    """
    result = ""
    for line in lines:
        if not result:
            result = line
            continue
        if _HYPHEN_TAIL.search(result):
            result = result[:-1] + line        # дефис убираем, пробел не ставим
        else:
            result = result + " " + line
    return result


def _clean(text: str) -> str:
    """Приводит распознанный текст к виду, пригодному для разбиения на чанки.

    Хелпер отдаёт текст БЛОКАМИ: строки одного абзаца — своими строками, между
    блоками пустая строка (порядок блоков — порядок чтения с учётом колонок и
    врезок, см. vision_ocr.swift). Здесь строки абзаца склеиваются в один текст
    (с учётом переносов), а блоки остаются отдельными абзацами через пустую
    строку.
    """
    pages: List[str] = []
    current: List[str] = []
    for raw in str(text or "").split("\n"):
        line = raw.strip()
        if not line:
            if current:
                pages.append(_join_lines(current))
                current = []
            continue
        current.append(line)
    if current:
        pages.append(_join_lines(current))
    return "\n\n".join(page for page in pages if page.strip()).strip()
