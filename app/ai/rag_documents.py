"""Извлечение текста из загруженных документов базы знаний (RAG).

Пользователь приносит ФАЙЛ, а индексировать можно только текст. Модуль
превращает байты в текст и честно называет результат: сколько знаков вышло,
сколько страниц в PDF, каким форматом он опознан и что не получилось.

Форматы:

- **PDF** — через `pypdf` (чистый Python, ставится в venv проекта). Скан без
  текстового слоя распознать нечем (OCR — отдельная тяжёлая вещь), поэтому такой
  файл НЕ выдаётся за пустую базу: в причине прямо сказано, что текстового слоя
  нет. Шифрованный PDF пробуем открыть с пустым паролем.
- **DOCX / ODT** — это zip с XML внутри, поэтому читаются БЕЗ внешних библиотек
  (`zipfile` + разбор разметки): абзацы становятся строками, служебная разметка
  не попадает в текст.
- **TXT/MD/RST/CSV/TSV/JSON/YAML/код** — текст как есть; CSV/TSV строится в
  строки «колонка: значение», JSON — в строки путей («настройки.порт: 8080»):
  так кусок таблицы или настроек остаётся осмысленным чанком.
- **HTML/XML** — текст без тегов и без `<script>/<style>`.
- Незнакомый формат: если байты похожи на текст — читаем как текст, если нет —
  отказываем с понятной причиной (лучше отказ, чем база из «мусорных» чанков).

Кодировки: UTF-8 (с BOM и без), затем cp1251 (частая для русских выгрузок),
затем koi8-r, затем latin-1 как последний шанс — с пометкой в предупреждении.

Модуль ничего не пишет на диск и не ходит в сеть: только разбор байтов.
"""

import csv
import io
import json
import logging
import os
import re
import tempfile
import zipfile
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple

from app.ai import rag_ocr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Пределы. Их ДВА, и это не случайность: транспорты разные.
#
# MAX_FILE_BYTES — общий предел на документ (по умолчанию 256 МБ, настраивается
# переменной RAG_MAX_FILE_BYTES). Столько принимает ПОТОКОВАЯ загрузка: файл
# льётся на диск кусками, в памяти целиком не лежит, поэтому размер ограничен
# только здравым смыслом и местом на диске.
#
# MAX_JSON_FILE_BYTES — предел для старого пути через base64 в JSON (по
# умолчанию 25 МБ, переменная RAG_MAX_JSON_FILE_BYTES). Здесь потолок задаёт
# ПАМЯТЬ: base64 раздувает файл на треть, а в браузере из 250 МБ получается
# строка в сотни мегабайт — и она же уезжает в тело запроса. Поэтому крупные
# файлы интерфейс шлёт потоком, а этим путём остаются небольшие.
# ---------------------------------------------------------------------------
def _env_bytes(name: str, default: int) -> int:
    """Размер из переменной окружения (МБ), с запасным значением по умолчанию."""
    raw = (os.getenv(name) or "").strip()
    try:
        value = int(float(raw) * 1024 * 1024)
    except ValueError:
        return default
    return value if value > 0 else default


MAX_FILE_BYTES = _env_bytes("RAG_MAX_FILE_BYTES", 256 * 1024 * 1024)
MAX_JSON_FILE_BYTES = _env_bytes("RAG_MAX_JSON_FILE_BYTES", 25 * 1024 * 1024)
MAX_FILES_PER_UPLOAD = 20
# Предохранитель на извлечённый текст: гигабайт текста из одного файла положим
# только вместе с оперативной памятью, а индексация — не место для этого. Он же
# ограничивает разбор ОГРОМНЫХ pdf: страницы читаются по одной и накопление
# останавливается на пределе, а не после всего файла.
MAX_TEXT_CHARS = 8_000_000
# Сколько байт текстового файла читаем с диска: 8 млн символов кириллицы в UTF-8
# это ~16 МБ, поэтому дальше читать нечего — всё равно обрежется по MAX_TEXT_CHARS.
MAX_TEXT_BYTES = 64 * 1024 * 1024

TEXT_EXTENSIONS = {
    ".txt", ".text", ".md", ".markdown", ".rst", ".log", ".me", ".org",
    ".py", ".js", ".mjs", ".ts", ".tsx", ".jsx", ".java", ".kt", ".go", ".rs",
    ".c", ".h", ".cpp", ".hpp", ".cs", ".rb", ".php", ".sh", ".bash", ".zsh",
    ".sql", ".ini", ".cfg", ".conf", ".toml", ".env", ".properties", ".srt",
    ".tex", ".vue", ".swift", ".scala", ".pl", ".lua", ".r", ".m",
}
HTML_EXTENSIONS = {".html", ".htm", ".xhtml", ".xml", ".svg"}
CSV_EXTENSIONS = {".csv", ".tsv"}
JSON_EXTENSIONS = {".json", ".jsonl", ".ndjson", ".geojson"}
PDF_EXTENSIONS = {".pdf"}
ZIP_XML_EXTENSIONS = {".docx", ".odt", ".pptx", ".xlsx"}

# Расширения, которые заведомо НЕ текст: отказ понятнее, чем «прочитали как
# latin-1 и получили кашу из символов».
BINARY_EXTENSIONS = {
    ".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz", ".exe", ".dll", ".so",
    ".dylib", ".bin", ".iso", ".dmg", ".gif", ".ico",
    ".mp3", ".mp4", ".avi", ".mov", ".mkv", ".wav",
    ".flac", ".ogg", ".woff", ".woff2", ".ttf", ".otf", ".eot", ".db", ".sqlite",
}
# Картинки: текста в них нет по определению, но их МОЖНО распознать (OCR).
# Поэтому это не «неподдерживаемый формат», а отдельная ветка — распознавание
# доступно только там, где есть Vision (macOS, см. app/ai/rag_ocr.py).
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".heic"}

# Какие расширения диалог показывает пользователем: список — подсказка, а не
# запрет (незнакомый текстовый файл всё равно прочитается).
SUPPORTED_EXTENSIONS = sorted(
    TEXT_EXTENSIONS | HTML_EXTENSIONS | CSV_EXTENSIONS | JSON_EXTENSIONS
    | PDF_EXTENSIONS | ZIP_XML_EXTENSIONS | IMAGE_EXTENSIONS)

_EXT_NAMES = {
    ".pdf": "PDF",
    ".docx": "Word (DOCX)",
    ".odt": "OpenDocument (ODT)",
    ".pptx": "PowerPoint (PPTX)",
    ".xlsx": "Excel (XLSX)",
    ".txt": "текст",
    ".md": "Markdown",
    ".csv": "CSV",
    ".tsv": "TSV",
    ".json": "JSON",
    ".html": "HTML",
    ".htm": "HTML",
    ".xml": "XML",
    ".rst": "reStructuredText",
    ".png": "изображение (PNG)",
    ".jpg": "изображение (JPEG)",
    ".jpeg": "изображение (JPEG)",
    ".tif": "изображение (TIFF)",
    ".tiff": "изображение (TIFF)",
    ".bmp": "изображение (BMP)",
    ".webp": "изображение (WebP)",
    ".heic": "изображение (HEIC)",
}


class DocumentError(Exception):
    """Документ не удалось превратить в текст (причина — в сообщении)."""


def extension_of(filename: str) -> str:
    """Расширение файла в нижнем регистре ("" — расширения нет)."""
    name = str(filename or "").strip()
    if "." not in name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]:
        return ""
    return "." + name.rsplit(".", 1)[-1].lower()


def kind_of(filename: str) -> str:
    """Вид документа для метаданных: pdf, docx, text, csv, json, html, unknown."""
    ext = extension_of(filename)
    if ext in PDF_EXTENSIONS:
        return "pdf"
    if ext in ZIP_XML_EXTENSIONS:
        return ext.lstrip(".")
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in CSV_EXTENSIONS:
        return "csv"
    if ext in JSON_EXTENSIONS:
        return "json"
    if ext in HTML_EXTENSIONS:
        return "html"
    if ext in TEXT_EXTENSIONS:
        return "text"
    return "unknown"


def describe_format(filename: str) -> str:
    """Человеческое название формата файла (для диалога и метаданных)."""
    ext = extension_of(filename)
    return _EXT_NAMES.get(ext, (ext.lstrip(".").upper() if ext else "без расширения"))


# ---------------------------------------------------------------------------
# Кодировки и «текстовость»
# ---------------------------------------------------------------------------
def looks_binary(data: bytes) -> bool:
    """Похожи ли байты на двоичные: NUL или высокая доля управляющих символов."""
    sample = data[:4096]
    if b"\x00" in sample:
        return True
    if not sample:
        return False
    control = sum(1 for byte in sample
                  if byte < 9 or (13 < byte < 32))
    return control / float(len(sample)) > 0.05


def decode_text(data: bytes, filename: str = "") -> Tuple[str, str]:
    """Декодирует байты в текст. Возвращает (текст, предупреждение).

    Порядок кодировок — от самой вероятной к запасной. Если ни одна не подошла
    без потерь, берём cp1251 с заменой и говорим об этом прямо: молча
    «прочитать» файл в latin-1 значило бы отдать в индекс нечитаемую кашу.
    """
    if data.startswith(b"\xef\xbb\xbf"):
        try:
            return data.decode("utf-8-sig"), ""
        except UnicodeDecodeError:
            pass
    for codec in ("utf-8", "cp1251", "koi8-r"):
        try:
            return data.decode(codec), ("" if codec == "utf-8"
                                        else "кодировка определена как %s" % codec)
        except UnicodeDecodeError:
            continue
    text = data.decode("cp1251", errors="replace")
    warning = "кодировка не распознана: нечитаемые символы заменены"
    logger.info("RAG: %s — %s", filename or "файл", warning)
    return text, warning


# ---------------------------------------------------------------------------
# HTML/XML
# ---------------------------------------------------------------------------
_BLOCK_TAGS = {
    "p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section",
    "article", "header", "footer", "table", "ul", "ol", "pre", "blockquote",
    "title", "td", "th", "dd", "dt",
}
_SKIP_TAGS = {"script", "style", "noscript", "head", "nav", "svg"}


class _TextHTMLParser(HTMLParser):
    """Собирает текст страницы: теги отбрасываются, блоки становятся строками."""

    def __init__(self) -> None:
        HTMLParser.__init__(self, convert_charrefs=True)
        self.parts: List[str] = []
        self.skip = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in _SKIP_TAGS:
            self.skip += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self.skip:
            self.skip -= 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skip and data.strip():
            self.parts.append(data)

    def text(self) -> str:
        """Текст с нормализованными пробелами и пустыми строками."""
        raw = "".join(self.parts)
        raw = re.sub(r"[ \t]{2,}", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return "\n".join(line.strip() for line in raw.split("\n")).strip()


def html_to_text(data: bytes) -> Tuple[str, str]:
    """Текст из HTML/XML без тегов (заголовки сохраняются как строки)."""
    text, warning = decode_text(data)
    parser = _TextHTMLParser()
    try:
        parser.feed(text)
    except Exception as exc:          # битая разметка — берём что успели
        logger.info("RAG: разбор разметки неполный — %s", str(exc)[:120])
        warning = warning or "разметка разобрана частично"
    return parser.text(), warning


# ---------------------------------------------------------------------------
# JSON / CSV
# ---------------------------------------------------------------------------
def json_to_text(data: bytes) -> Tuple[str, str]:
    """JSON → строки «путь: значение».

    Плоские пары читаются человеком и моделью, а «сырой» JSON с отступами
    тратит чанк на скобки. Не разобрался JSON (JSONL, битый файл) — читаем как
    текст: отказываться от файла из-за одной ошибки разбора незачем.
    """
    text, warning = decode_text(data)
    stripped = text.strip()
    if not stripped:
        return "", warning
    lines: List[str] = []
    try:
        if stripped.startswith("["):
            payload = json.loads(stripped)
            for index, item in enumerate(payload if isinstance(payload, list) else []):
                lines.extend(_flatten(item, "[%d]" % index))
        else:
            # JSONL/NDJSON: по объекту на строку.
            objects = []
            for line in stripped.split("\n"):
                if line.strip():
                    objects.append(json.loads(line))
            if len(objects) == 1:
                lines.extend(_flatten(objects[0], ""))
            else:
                for index, item in enumerate(objects):
                    lines.extend(_flatten(item, "[%d]" % index))
    except (ValueError, TypeError):
        return text, (warning or "JSON не разобран — прочитан как текст")
    return "\n".join(lines), warning


def _flatten(value: Any, prefix: str) -> List[str]:
    """Разворачивает JSON в строки «путь: значение» (вложенность — через точку)."""
    if isinstance(value, dict):
        result: List[str] = []
        for key, item in value.items():
            path = "%s.%s" % (prefix, key) if prefix else str(key)
            result.extend(_flatten(item, path))
        return result
    if isinstance(value, list):
        result = []
        for index, item in enumerate(value):
            result.extend(_flatten(item, "%s[%d]" % (prefix, index)))
        return result
    if value is None:
        return []
    return ["%s: %s" % (prefix or "значение", value)]


def csv_to_text(data: bytes, delimiter: Optional[str] = None) -> Tuple[str, str]:
    """CSV/TSV → строки «колонка: значение» (заголовок переносится в каждую строку).

    Так строка таблицы остаётся САМОДОСТАТОЧНОЙ: попав в чанк отдельно от
    заголовка, «Москва: 12» без подписи колонки ничего не значит.
    """
    text, warning = decode_text(data)
    if not text.strip():
        return "", warning
    sample = text[:4096]
    if delimiter is None:
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            delimiter = "\t" if sample.count("\t") > sample.count(",") else ","
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows = [row for row in reader if any(cell.strip() for cell in row)]
    if not rows:
        return "", warning
    header = [cell.strip() for cell in rows[0]]
    lines: List[str] = []
    if len(rows) == 1:
        lines.append(" | ".join(header))
        return "\n".join(lines), warning
    lines.append("колонки: " + " | ".join(header))
    for row in rows[1:]:
        cells = []
        for index, cell in enumerate(row):
            name = header[index] if index < len(header) and header[index] else "колонка %d" % (index + 1)
            cells.append("%s: %s" % (name, cell.strip()))
        if cells:
            lines.append(" | ".join(cells))
    return "\n".join(lines), warning


# ---------------------------------------------------------------------------
# PDF (pypdf)
# ---------------------------------------------------------------------------
def _pdf_assets(page: Any) -> Tuple[int, bool]:
    """Что на странице: сколько изображений и объявлены ли шрифты.

    Смотрим РЕСУРСЫ страницы, а не разбираем картинки: у скана на 500 страниц
    декодирование изображений заняло бы минуты, а нам нужен только ответ на
    вопрос «это картинки или текст».
    """
    images = 0
    fonts = False
    try:
        resources = page.get("/Resources")
        if resources is None:
            return 0, False
        resources = resources.get_object()
        fonts = bool(resources.get("/Font"))
        xobjects = resources.get("/XObject")
        if xobjects is not None:
            for name in xobjects.get_object():
                item = xobjects.get_object()[name].get_object()
                if str(item.get("/Subtype") or "") == "/Image":
                    images += 1
    except Exception:              # битые ресурсы — не повод падать
        return images, fonts
    return images, fonts


def _pdf_no_text_reason(reader: Any, sample: int = 8) -> str:
    """Объясняет, ПОЧЕМУ в PDF не нашлось текста — по содержимому страниц.

    Разница принципиальная для пользователя: скан («нужен OCR») и PDF, где текст
    есть, но не извлекается («нестандартная кодировка»), требуют РАЗНЫХ действий.
    Общее «текста нет» заставляло бы гадать, что делать с файлом.
    """
    pages = list(getattr(reader, "pages", []) or [])
    if not pages:
        return "в PDF не найдено ни одной страницы"
    images = 0
    fonts = 0
    checked = 0
    for page in pages[:sample]:
        found_images, found_fonts = _pdf_assets(page)
        images += found_images
        fonts += 1 if found_fonts else 0
        checked += 1
    per_page = images / float(max(1, checked))
    # ФАКТ без совета: совет добавляет вызывающий код, потому что он зависит от
    # того, доступно ли распознавание. Иначе в предупреждении оказывалось бы и
    # «без OCR индексировать нечего», и «текст получен распознаванием» — прямое
    # противоречие в одной строке.
    if images and not fonts:
        return ("текстового слоя нет: страницы — изображения (в среднем %.1f на "
                "страницу), шрифтов в файле нет" % per_page)
    if images and fonts:
        return ("текстовый слой не извлекается, хотя в файле есть и изображения, и "
                "шрифты: похоже на скан с испорченным или частичным слоем")
    if fonts:
        return ("шрифты в PDF объявлены, но текст не извлекается: вероятно, "
                "нестандартная кодировка шрифта (нет таблицы соответствия символов)")
    return ("на страницах нет ни текста, ни изображений: файл пуст или содержит "
            "только разметку без содержимого")


def _pdf_retry_layout(reader: Any, on_progress: Any, limit: int) -> Tuple[str, int]:
    """Повторная попытка разбора другим режимом pypdf.

    Обычный режим иногда не берёт текст, который берёт режим разметки (другая
    обработка позиций и кодировок). Пробуем ОДИН раз и только когда обычный
    разбор не дал вообще ничего: это запасной путь, а не второй проход всегда.
    """
    parts: List[str] = []
    total = 0
    pages = list(getattr(reader, "pages", []) or [])
    for number, page in enumerate(pages, start=1):
        try:
            piece = page.extract_text(extraction_mode="layout") or ""
        except Exception:
            piece = ""
        total += len(piece)
        parts.append(piece)
        if on_progress is not None and number % 5 == 0:
            try:
                on_progress(number, len(pages), "read")
            except Exception:
                pass
        if total >= limit:
            break
    return "\n\n".join(part.strip() for part in parts if part.strip()), total


def pdf_to_text(source: Any, on_progress: Any = None,
                filename: str = "") -> Tuple[str, str, int]:
    """Текст PDF. Возвращает (текст, предупреждение, страниц).

    `source` — путь к файлу ИЛИ байты. Путь предпочтительнее: pypdf читает файл
    постранично, не поднимая в память весь документ, — поэтому PDF на 250 МБ
    разбирается, а не падает по памяти. Накопление текста всё равно
    останавливается на MAX_TEXT_CHARS: за пределом чанки всё равно не построить,
    а память он занимает настоящую.

    Без `pypdf` (или при сбое разбора) — понятная причина, а не пустая база:
    пользователь должен знать, что файл не прочитан, и почему.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        raise DocumentError(
            "для PDF нужен пакет pypdf — установите его командой "
            "`./venv/bin/pip install pypdf` (в проекте он уже стоит)")
    try:
        # Строка — это путь; байты оборачиваем, чтобы не дублировать код.
        handle = source if isinstance(source, str) else io.BytesIO(source)
        reader = PdfReader(handle)
        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")
            except Exception:
                raise DocumentError("PDF защищён паролем — текст не извлечён")
        pages: List[str] = []
        total_chars = 0
        truncated = False
        # Число страниц известно ДО разбора — по нему интерфейс показывает
        # настоящий прогресс («страница 123 из 500»), а не крутящийся значок.
        try:
            total_pages = len(reader.pages)
        except Exception:
            total_pages = 0
        for number, page in enumerate(reader.pages, start=1):
            try:
                piece = page.extract_text() or ""
            except Exception as exc:      # одна битая страница не рушит документ
                logger.info("RAG: страница PDF не разобрана — %s", str(exc)[:120])
                piece = ""
            total_chars += len(piece)
            pages.append(piece)
            if on_progress is not None and (number % 5 == 0 or number == total_pages):
                # Каждую страницу дёргать незачем: у PDF на 500 страниц это сотни
                # обновлений состояния без пользы для глаза.
                try:
                    on_progress(number, total_pages, "read")
                except Exception:      # наблюдатель не должен ломать разбор
                    logger.info("RAG: наблюдатель разбора PDF упал", exc_info=False)
            if total_chars >= MAX_TEXT_CHARS:
                # Дальше читать незачем: текст всё равно обрежется по пределу,
                # а страницы в сканах бывают тяжёлыми.
                truncated = True
                break
    except DocumentError:
        raise
    except Exception as exc:
        raise DocumentError("PDF не разобран: %s" % str(exc)[:200])
    total = len(pages)
    text = "\n\n".join(part.strip() for part in pages if part.strip())
    warning = ""
    if truncated:
        warning = ("PDF больше предела индексации: прочитаны первые %d страниц"
                   % total)
    if not text.strip():
        # Обычный разбор не дал ничего: пробуем режим разметки, прежде чем
        # объявлять файл нечитаемым (иногда он берёт текст, который не взял
        # обычный режим), и только потом объясняем причину по СОДЕРЖИМОМУ файла.
        retry, retry_chars = _pdf_retry_layout(reader, on_progress, MAX_TEXT_CHARS)
        if retry.strip():
            text = retry
            warning = _join_warning(
                warning, "текст извлечён вторым способом (режим разметки): "
                         "%d символов" % retry_chars)
        else:
            # Текста нет — сюда приходит скан. Если распознавание доступно, оно и
            # есть решение; иначе объясняем причину по содержимому файла.
            ocr_text, ocr_note = _recognize_fallback(
                source, on_progress, _pdf_no_text_reason(reader),
                page=_ocr_page_reporter(on_progress), filename=filename)
            if ocr_text.strip():
                text = ocr_text
                warning = _join_warning(warning, ocr_note)
            else:
                warning = _join_warning(warning, ocr_note)
    elif rag_ocr.mode() == rag_ocr.MODE_ALWAYS and rag_ocr.available():
        # Настройка «always»: слой есть, но он может быть испорченным. Берём
        # распознанный текст, если он ДЛИННЕЕ — иначе смысла менять нет.
        ocr_text, ocr_note = _recognize_fallback(source, on_progress, "",
                                                 page=_ocr_page_reporter(on_progress),
                                                 filename=filename)
        if ocr_text.strip() and len(ocr_text) > len(text):
            warning = _join_warning(warning or ocr_note,
                                    "текст взят распознаванием (OCR): он полнее "
                                    "текстового слоя")
            text = ocr_text
    return text, warning, total


# ---------------------------------------------------------------------------
# DOCX / ODT / PPTX / XLSX (zip + XML, без внешних библиотек)
# ---------------------------------------------------------------------------
_ZIP_PARTS = {
    ".docx": ("word/document.xml",),
    ".odt": ("content.xml",),
    ".pptx": ("ppt/slides/slide",),
    ".xlsx": ("xl/sharedStrings.xml",),
}


def _ocr_advice() -> str:
    """Совет, что делать с файлом без текста — с учётом доступности OCR.

    Это ЕДИНСТВЕННОЕ место, где такой совет пишется: держать его рядом с фактом
    («текстового слоя нет») нельзя, иначе одна и та же строка утверждала бы и
    «без распознавания индексировать нечего», и «текст получен распознаванием».
    """
    state = rag_ocr.status()
    if state["available"]:
        return ("текст можно получить распознаванием (OCR) — оно включено, "
                "но не сработало")
    return ("без распознавания (OCR) такой файл индексировать нечем; "
            "распознавание недоступно: %s" % (state.get("reason") or "причина неизвестна"))


def _ocr_source_path(source: Any, reason: str,
                     filename: str = "") -> Tuple[str, str]:
    """Путь к файлу для распознавания. Возвращает (путь, пояснение/флаг).

    Vision работает с ФАЙЛОМ, а содержимое приходит двумя путями: потоковая
    загрузка кладёт его на диск (путь), а JSON-загрузка передаёт байты. Во втором
    случае байты временно пишутся на диск — иначе сканы через API молча не
    распознавались бы («распознавание доступно только для файлов на диске»).
    Временный файл удаляет вызывающий код.
    """
    if isinstance(source, str) and source:
        return source, ""
    if not isinstance(source, (bytes, bytearray)):
        return "", "содержимое файла недоступно для распознавания"
    suffix = extension_of(filename) or ".bin"
    handle, path = tempfile.mkstemp(prefix="rag-ocr-", suffix=suffix)
    try:
        with os.fdopen(handle, "wb") as target:
            target.write(bytes(source))
    except OSError as exc:
        try:
            os.unlink(path)
        except OSError:
            pass
        return "", "не удалось подготовить файл к распознаванию: %s" % str(exc)[:120]
    return path, "temporary"


def _ocr_page_reporter(on_progress: Any) -> Any:
    """Наблюдатель страниц РАСПОЗНАВАНИЯ: та же форма, но с пометкой «OCR».

    Без пометки пользователь видел бы подряд два одинаковых «страница 1 из 1» (сначала
    чтение PDF, потом распознавание) и не понимал бы, что происходит.
    """
    if on_progress is None:
        return None
    return lambda done, total: on_progress(done, total, "ocr")


def _recognize_fallback(source: Any, on_progress: Any, reason: str,
                        page: Any = None, filename: str = "") -> Tuple[str, str]:
    """Пробует распознать файл. Возвращает (текст, пояснение для пользователя).

    Причина отказа передаётся сюда, чтобы в пояснении осталось ВИДНО, почему
    понадобилось распознавание («текстового слоя нет: страницы — изображения»),
    и чем оно закончилось. Молча подменять причину на «распознано» нельзя: у
    распознанного текста своя точность, и пользователь должен это знать.
    """
    if not rag_ocr.should_recognize(0):
        return "", _join_warning(reason, _ocr_advice())
    path, temporary = _ocr_source_path(source, reason, filename)
    if not path:
        return "", _join_warning(reason, temporary)
    try:
        text, problem = rag_ocr.recognize(path, on_progress=page)
    finally:
        if temporary:
            try:
                os.unlink(path)
            except OSError:
                pass
    if not text.strip():
        return "", _join_warning(reason, "распознать не удалось: %s" % problem,
                                 _ocr_advice())
    note = ("текст получен РАСПОЗНАВАНИЕМ (OCR, macOS Vision): %d символов — "
            "возможны опечатки" % len(text))
    if problem:
        note = _join_warning(note, problem)
    return text, _join_warning(reason, note)


def image_to_text(source: Any, filename: str,
                  on_progress: Any = None) -> Tuple[str, str]:
    """Текст из картинки (PNG/JPG/TIFF…): другого способа, кроме OCR, нет.

    Одиночный скан в JPG встречается не реже многостраничного PDF, поэтому
    картинки принимаются — но ЧЕСТНО: без распознавания текста в них взять
    нечего, и отказ говорит именно об этом.
    """
    state = rag_ocr.status()
    if not state["available"]:
        raise DocumentError(
            "в изображении нет текстового слоя, а распознавание недоступно: %s. "
            "Формат %s индексируется только через OCR"
            % (state.get("reason") or "причина неизвестна",
               describe_format(filename)))
    path, temporary = _ocr_source_path(source, "", filename)
    if not path:
        raise DocumentError("изображение не удалось подготовить к распознаванию: %s"
                            % (temporary or "причина неизвестна"))
    try:
        text, problem = rag_ocr.recognize(path,
                                          on_progress=_ocr_page_reporter(on_progress))
    finally:
        if temporary:
            try:
                os.unlink(path)
            except OSError:
                pass
    text = _clean(text)
    if not text.strip():
        raise DocumentError("текст в изображении не распознан: %s"
                            % (problem or "причина неизвестна"))
    warning = "текст получен РАСПОЗНАВАНИЕМ (OCR, macOS Vision): %d символов" % len(text)
    return text, (warning if not problem else _join_warning(warning, problem))


def zip_xml_to_text(data: Any, filename: str) -> Tuple[str, str]:
    """Текст из офисного файла: это zip с XML, поэтому хватает stdlib.

    Извлекаются только текстовые части (абзацы, строки таблицы), служебная
    разметка и стили в индекс не идут.
    """
    ext = extension_of(filename)
    try:
        archive = zipfile.ZipFile(data if isinstance(data, str) else io.BytesIO(data))
    except zipfile.BadZipFile:
        raise DocumentError("файл повреждён: %s — это не читаемый архив"
                            % describe_format(filename))
    names = archive.namelist()
    wanted: List[str] = []
    for part in _ZIP_PARTS.get(ext, ()):
        if part.endswith("slide"):
            wanted.extend(sorted(name for name in names
                                 if name.startswith(part) and name.endswith(".xml")))
        elif part in names:
            wanted.append(part)
    if not wanted and ext == ".odt" and "content.xml" in names:
        wanted = ["content.xml"]
    if not wanted:
        raise DocumentError("в файле %s не найдено текстовых частей"
                            % describe_format(filename))
    parts: List[str] = []
    for name in wanted:
        try:
            # Читаем ЧАСТЯМИ: у офисного файла текстовые части небольшие, но
            # вложения внутри архива могут быть любого размера — в память их
            # поднимать незачем.
            with archive.open(name) as handle:
                raw = handle.read(MAX_TEXT_BYTES)
        except (KeyError, zipfile.BadZipFile, OSError) as exc:
            logger.info("RAG: часть %s не прочитана — %s", name, str(exc)[:120])
            continue
        parts.append(_xml_text(raw))
    archive.close()
    text = "\n\n".join(part for part in parts if part.strip())
    warning = ""
    if not text.strip():
        warning = "в документе не найдено текста"
    return text, warning


_DOCX_PARA_RE = re.compile(r"</w:p>|</a:p>|<w:br\s*/>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_ENTITIES = {"&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"',
             "&apos;": "'", "&nbsp;": " "}


def _xml_text(raw: bytes) -> str:
    """Текст из XML офисного документа: абзацы — строками, разметка — прочь."""
    try:
        body = raw.decode("utf-8")
    except UnicodeDecodeError:
        body = raw.decode("cp1251", errors="replace")
    body = _DOCX_PARA_RE.sub("\n", body)
    body = _TAG_RE.sub("", body)
    for entity, char in _ENTITIES.items():
        body = body.replace(entity, char)
    lines = [line.strip() for line in body.split("\n")]
    return "\n".join(line for line in lines if line)


# ---------------------------------------------------------------------------
# Публичная точка входа
# ---------------------------------------------------------------------------
def human_bytes(size: Any) -> str:
    """Размер по-человечески (для сообщений об отказе)."""
    try:
        value = float(size)
    except (TypeError, ValueError):
        return "0 Б"
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if value < 1024 or unit == "ГБ":
            return ("%d %s" % (int(value), unit) if unit == "Б"
                    else "%.0f %s" % (value, unit))
        value /= 1024.0
    return "%.0f ГБ" % value


def _head_bytes(source: Any) -> bytes:
    """Первые килобайты файла — по ним решаем, текст это или двоичные данные."""
    if not isinstance(source, str):
        return bytes(source)[:4096]
    try:
        with open(source, "rb") as handle:
            return handle.read(4096)
    except OSError as exc:
        raise DocumentError("файл не читается: %s" % str(exc)[:120])


def _read_capped(source: Any, limit: int = MAX_TEXT_BYTES) -> bytes:
    """Читает файл (или отдаёт байты), обрезая по пределу.

    Текстовым форматам (CSV, JSON, HTML) нужно содержимое целиком — их нельзя
    разбирать постранично. Поэтому читаем не больше `limit`: 64 МБ текста это
    ~8 млн символов кириллицы, то есть ровно предел индексации; дальше байты
    всё равно были бы выброшены, но память бы заняли.
    """
    if not isinstance(source, str):
        return bytes(source)[:limit]
    try:
        with open(source, "rb") as handle:
            return handle.read(limit)
    except OSError as exc:
        raise DocumentError("файл не читается: %s" % str(exc)[:120])


def extract(data: bytes, filename: str = "",
            on_progress: Any = None) -> Dict[str, Any]:
    """Извлекает текст из загруженного файла (содержимое — уже в памяти).

    Возвращает {"text", "kind", "format", "chars", "pages", "warning"}.
    Не получилось — бросает DocumentError с причиной, понятной пользователю:
    «не читается» без объяснения заставило бы гадать, что не так с файлом.
    """
    if data is None:
        raise DocumentError("файл пуст")
    if not isinstance(data, (bytes, bytearray)):
        raise DocumentError("файл передан в неожиданном виде")
    blob = bytes(data)
    if not blob:
        raise DocumentError("файл пуст (0 байт)")
    if len(blob) > MAX_FILE_BYTES:
        raise DocumentError("файл больше %s — такой документ не индексируется"
                            % human_bytes(MAX_FILE_BYTES))
    return _extract_source(blob, filename, on_progress)


def extract_path(path: str, filename: str = "",
                 on_progress: Any = None) -> Dict[str, Any]:
    """Извлекает текст из файла НА ДИСКЕ (для загрузки потоком).

    Отличие от `extract` не только в источнике: крупные форматы читаются
    ПОСТРАНИЧНО И ЧАСТЯМИ (PDF — `pypdf` по страницам, офисные архивы — нужные
    части по отдельности, текстовые — с ограничением на прочитанное), поэтому
    документ на сотни мегабайт разбирается, не занимая память целиком. Именно
    ради этого потоковая загрузка пишет файл на диск, а не держит его в теле
    запроса.
    """
    if not path or not os.path.isfile(path):
        raise DocumentError("файл не найден на диске")
    size = os.path.getsize(path)
    if not size:
        raise DocumentError("файл пуст (0 байт)")
    if size > MAX_FILE_BYTES:
        raise DocumentError("файл больше %s — такой документ не индексируется"
                            % human_bytes(MAX_FILE_BYTES))
    return _extract_source(path, filename or os.path.basename(path), on_progress)


def _extract_source(source: Any, filename: str,
                    on_progress: Any = None) -> Dict[str, Any]:
    """Общий разбор: `source` — путь к файлу или байты.

    Логика форматов живёт ЗДЕСЬ одна: добавить формат — значит дописать ветку
    тут, а не в двух местах (байты и путь), которые разошлись бы.
    """
    path_mode = isinstance(source, str)
    name = str(filename or "").strip() or "документ.txt"
    ext = extension_of(name)
    kind = kind_of(name)
    pages = 0
    warning = ""
    head = _head_bytes(source) if path_mode else bytes(source)

    if ext in PDF_EXTENSIONS:
        text, warning, pages = pdf_to_text(source, on_progress, name)
    elif ext in ZIP_XML_EXTENSIONS:
        text, warning = zip_xml_to_text(source, name)
    elif ext in HTML_EXTENSIONS:
        text, warning = html_to_text(_read_capped(source))
    elif ext in JSON_EXTENSIONS:
        text, warning = json_to_text(_read_capped(source))
    elif ext in IMAGE_EXTENSIONS:
        text, warning = image_to_text(source, name, on_progress)
    elif ext in CSV_EXTENSIONS:
        text, warning = csv_to_text(_read_capped(source))
    elif ext in BINARY_EXTENSIONS:
        raise DocumentError("формат %s не поддерживается: это не текстовый документ"
                            % describe_format(name))
    elif looks_binary(head):
        raise DocumentError("файл похож на двоичный — текст из него не извлечён "
                            "(поддерживаются PDF, DOCX/ODT, текстовые форматы)")
    else:
        # Незнакомое расширение, но содержимое текстовое — читаем как текст.
        text, warning = decode_text(_read_capped(source), name)
        kind = kind if kind != "unknown" else "text"

    text = _clean(text)
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]
        warning = _join_warning(warning, "текст обрезан по пределу индексации (%d симв.)"
                                % MAX_TEXT_CHARS)
    if not text.strip() and not warning:
        warning = "текст не извлечён"
    return {
        "text": text,
        "kind": kind,
        "format": describe_format(name),
        "chars": len(text),
        "pages": pages,
        "warning": warning,
    }


def _clean(text: str) -> str:
    """Приводит извлечённый текст к виду, пригодному для разбиения на чанки."""
    value = str(text or "")
    value = value.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    value = re.sub(r"[ \t]{3,}", "  ", value)
    value = re.sub(r"\n{4,}", "\n\n\n", value)
    return value.strip()


def _join_warning(*parts: str) -> str:
    """Склеивает предупреждения через точку с запятой (пустые пропускает)."""
    return "; ".join(part.strip().rstrip(".") for part in parts if part and part.strip())
