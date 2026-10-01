"""Разбиение документов на чанки — подготовка базы знаний для RAG.

Модуль ЧИСТЫЙ: ни файлов, ни сети, ни модели. На входе текст документа, на
выходе список чанков с метаданными (source, title, section, chunk_id и служебные
смещения). Поэтому его одинаково используют и пайплайн индексации
(app/ai/rag.py), и проверки без сети (tools/check_rag.py).

ДВЕ СТРАТЕГИИ (выбор пользователя в диалоге «База знаний»):

- **fixed** — фиксированный размер: окно в `chunk_size` символов со сдвигом на
  `chunk_size - overlap`. Границы окна по возможности подтягиваются к концу
  абзаца или предложения (в пределах последней пятой части окна), но размер
  чанка НИКОГДА не превышает `chunk_size`; если естественной границы рядом нет,
  остаётся жёсткий разрез. Никакой структуры документа стратегия не знает:
  это честный «нарезать ровно», с которым сравнивают структурную.

- **structure** — по структуре: документ режется по заголовкам (markdown `#`,
  подчёркивание `===`/`---`, нумерация `1.2.`, русские «Глава/Раздел/Часть/
  Статья/Параграф», строка КАПСОМ), каждый раздел — отдельный чанк, а раздел
  длиннее `chunk_size` дорезается тем же оконным алгоритмом. В метаданных
  чанка остаётся ПУТЬ раздела (`section`: «Глава 1 › 1.2 Установка»), поэтому
  найденный фрагмент можно показать человеку со ссылкой на место в документе.
  Мелкие разделы (короче `min_section`) склеиваются со следующим — иначе база
  состояла бы из чанков-оглавлений; склеенные заголовки видны в поле `merged`.
  Структуры в документе нет вовсе — стратегия честно вырождается в разбиение
  по абзацам (`fallback: "paragraphs"`), а не молча отдаёт один чанк на весь
  файл.

ФАЙЛЫ — граница разбиения: `chunk_document` вызывается НА КАЖДЫЙ документ
отдельно (см. app/ai/rag.py), поэтому чанк никогда не смешивает два файла и
`source` у чанка всегда один.

МЕТАДАННЫЕ ЧАНКА (обязательный минимум, названный в задаче, — source, title,
section, chunk_id):

    {
      "chunk_id": "kb-1a2b3c4d-0007",   # уникален внутри базы, стабилен
      "source": "ГОСТ 34.602-2020.pdf",  # файл-источник (имя, как у пользователя)
      "title": "ГОСТ 34.602-2020",       # заголовок документа (имя без расширения)
      "section": "Глава 1 › 1.2 Установка",  # путь раздела ("" у стратегии fixed)
      "doc_index": 0,                    # номер документа внутри базы (0-based)
      "position": 7,                     # номер чанка внутри документа (0-based)
      "index": 7,                        # номер чанка внутри базы (ставит пайплайн)
      "start": 1200, "end": 2200,        # смещения в тексте документа
      "chars": 1000,                     # длина чанка в символах
      "strategy": "structure",           # какой стратегией получен
      "path": "Глава 1 › 1.2 Установка", # синоним section (плоская копия)
      "merged": ["Введение"],            # заголовки, склеенные в этот чанк
      "kind": "section" | "window" | "paragraph",  # как именно нарезан
    }

Смещения (`start`/`end`) — это позиции в ИСХОДНОМ тексте документа (после
нормализации переводов строк), поэтому по ним можно вырезать контекст вокруг
найденного фрагмента.
"""

import hashlib
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Стратегии разбиения: идентификатор → человеческое имя. Список уходит в
# интерфейс (диалог «База знаний»), поэтому он же — единственный источник
# названий стратегий и в коде, и на экране.
# ---------------------------------------------------------------------------
STRATEGY_FIXED = "fixed"
STRATEGY_STRUCTURE = "structure"

CHUNKING_STRATEGIES: List[Dict[str, str]] = [
    {
        "id": STRATEGY_FIXED,
        "name": "Фиксированный размер",
        "description": ("Режет текст окнами по N символов с перекрытием. Предсказуемый "
                        "размер чанка, но раздел может разорваться посередине."),
    },
    {
        "id": STRATEGY_STRUCTURE,
        "name": "По структуре (заголовки/разделы/файлы)",
        "description": ("Режет по заголовкам и разделам: каждый чанк — осмысленный "
                        "раздел со своим путём в документе. Длинный раздел дорезается "
                        "окнами, мелкие склеиваются. Границы файлов не смешиваются."),
    },
]

STRATEGY_IDS = [item["id"] for item in CHUNKING_STRATEGIES]
DEFAULT_STRATEGY = STRATEGY_STRUCTURE

# ---------------------------------------------------------------------------
# Пределы размеров. Значения — в СИМВОЛАХ (не токенах): пользователь задаёт их
# числом в интерфейсе, и символы он может посчитать глазами, а токены — нет.
# ---------------------------------------------------------------------------
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150

MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 8000
MIN_CHUNK_OVERLAP = 0

# Перекрытие не может быть «почти размером чанка»: при overlap >= chunk_size
# окно не двигалось бы вперёд и разбиение зациклилось бы. Верхняя граница —
# половина окна: этого хватает, чтобы фраза не терялась на стыке.
MAX_OVERLAP_RATIO = 0.5

# Доля окна, в пределах которой ищется естественная граница (конец абзаца или
# предложения). Меньше — режем жёстко, больше — чанки становятся заметно короче
# заданного размера.
_BOUNDARY_TAIL_RATIO = 0.2

# Раздел короче этого порога склеивается с соседними, пока накопленный чанк не
# дойдёт до порога (см. _merge_small_sections). Порог намеренно НЕ зависит от
# размера чанка: он нужен против вырожденных разделов-заголовков, а не для
# «оптимизации» структуры.
_MIN_SECTION_FLOOR = 120


def strategy_name(strategy: Any) -> str:
    """Человеческое имя стратегии по её идентификатору (для интерфейса)."""
    key = normalize_strategy(strategy)
    for item in CHUNKING_STRATEGIES:
        if item["id"] == key:
            return item["name"]
    return key


def normalize_strategy(raw: Any) -> str:
    """Приводит название стратегии к известному идентификатору.

    Неизвестное (старый файл, опечатка, пустое значение) — стратегия по
    умолчанию: молча падать из-за одной строки в настройках проект не должен.
    """
    key = str(raw or "").strip().lower()
    return key if key in STRATEGY_IDS else DEFAULT_STRATEGY


def normalize_chunk_size(raw: Any, default: int = DEFAULT_CHUNK_SIZE) -> int:
    """Размер чанка: целое в границах MIN_CHUNK_SIZE…MAX_CHUNK_SIZE."""
    return _as_int(raw, default, MIN_CHUNK_SIZE, MAX_CHUNK_SIZE)


def normalize_overlap(raw: Any, size: Any = None,
                      default: int = DEFAULT_CHUNK_OVERLAP) -> int:
    """Перекрытие чанков: целое от 0 до половины размера чанка.

    Верхняя граница зависит от РАЗМЕРА, поэтому проверяется вместе с ним:
    перекрытие в 900 символов при чанке в 200 — это не «плотнее», это цикл.
    """
    limit = int(normalize_chunk_size(size) * MAX_OVERLAP_RATIO)
    return _as_int(raw, default, MIN_CHUNK_OVERLAP, max(MIN_CHUNK_OVERLAP, limit))


def _as_int(raw: Any, default: int, low: int, high: int) -> int:
    """Целое из чего угодно с зажимом в границы (нечисло → значение по умолчанию)."""
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        value = int(default)
    return max(low, min(high, value))


def chunk_settings(strategy: Any = None, chunk_size: Any = None,
                   overlap: Any = None) -> Dict[str, Any]:
    """Нормализованные настройки разбиения одной парой (стратегия, размер, перекрытие).

    Единая точка приведения: и настройки проекта (workspace), и запрос на
    загрузку базы, и пайплайн индексации считают границы одинаково.
    """
    size = normalize_chunk_size(chunk_size)
    return {
        "strategy": normalize_strategy(strategy),
        "chunk_size": size,
        "overlap": normalize_overlap(overlap, size),
    }


# ---------------------------------------------------------------------------
# Текст: нормализация и определение «заголовочности» строк
# ---------------------------------------------------------------------------
_ATX_RE = re.compile(r"^ {0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_SETEXT_RE = re.compile(r"^ {0,3}(=+|-{2,})\s*$")
# Нумерация вида «1.2. Название», «3) Название» — но НЕ список («1. купить хлеб»):
# требуем, чтобы после номера шла заглавная буква и строка была короткой.
_NUMBERED_RE = re.compile(r"^ {0,3}(\d+(?:\.\d+)*)[.)]?\s+([A-ZА-ЯЁ][^\n]{0,79})$")
# Русские заголовки без разметки: «Глава 2. Установка», «Раздел IV».
_RU_HEADING_RE = re.compile(
    r"^ {0,3}(Глава|Раздел|Часть|Статья|Параграф|Приложение)\s+"
    r"([IVXLCХ\d]+)[.)]?\s*(.{0,79})$", re.IGNORECASE)
# КАПС-заголовок: короткая строка из заглавных букв без точки в конце.
_CAPS_RE = re.compile(r"^ {0,3}([^a-zа-яё\n]{3,80})$")
_BLANK_RE = re.compile(r"\n[ \t]*\n")
# Конец предложения — для подтягивания границы окна к естественному стыку.
_SENTENCE_END_RE = re.compile(r"[.!?…][\"'»)\]]?\s")


def normalize_text(raw: Any) -> str:
    """Нормализует текст документа: переводы строк, табы, хвостовые пробелы.

    Смещения чанков считаются уже по нормализованному тексту — иначе `\\r\\n`
    в PDF-выгрузке сдвигал бы все позиции на единицу.
    """
    text = str(raw or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    text = "\u00a0".join(text.split("\u00a0"))  # неразрывный пробел → обычный
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    # Три и более пустых строк подряд — это одна граница абзаца, не три.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip("\n")


def _is_heading(line: str, next_line: str = "") -> Optional[Tuple[int, str]]:
    """Распознаёт заголовок в строке. Возвращает (уровень, текст) или None.

    Уровень — глубина вложенности (1 — самый верхний): по нему строится путь
    раздела. У markdown он берётся из числа `#`, у остальных видов выводится из
    вида заголовка, а не из отступов: PDF и Word отступов не сохраняют.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return None
    match = _ATX_RE.match(line)
    if match:
        return len(match.group(1)), match.group(2).strip()
    if _SETEXT_RE.match(next_line or ""):
        level = 1 if (next_line or "").strip().startswith("=") else 2
        return level, stripped
    match = _RU_HEADING_RE.match(line)
    if match:
        level = 1 if match.group(1).lower() in ("глава", "раздел", "часть") else 2
        title = (match.group(1) + " " + match.group(2) + " " + match.group(3)).strip()
        return level, re.sub(r"\s{2,}", " ", title)
    match = _NUMBERED_RE.match(line)
    if match:
        # Уровень — по числу частей номера: «1.2.3» глубже, чем «1.2».
        return min(6, match.group(1).count(".") + 1), stripped
    match = _CAPS_RE.match(stripped)
    if match and not stripped.endswith((".", ",", ";", ":")):
        letters = [ch for ch in stripped if ch.isalpha()]
        if letters and len(letters) >= 3:
            return 2, stripped
    return None


def _heading_sections(text: str) -> List[Dict[str, Any]]:
    """Режет текст на разделы по заголовкам.

    Возвращает список разделов: {"section": путь, "title": заголовок,
    "text": текст раздела ВМЕСТЕ с заголовком, "start": смещение, "level": …}.
    Текст перед первым заголовком становится разделом с пустым путём.
    """
    sections: List[Dict[str, Any]] = []
    stack: List[Tuple[int, str]] = []      # путь заголовков: (уровень, текст)
    current: Optional[Dict[str, Any]] = None
    offset = 0
    lines = text.split("\n")
    for number, line in enumerate(lines):
        next_line = lines[number + 1] if number + 1 < len(lines) else ""
        found = _is_heading(line, next_line)
        if found:
            level, title = found
            # Заголовок закрывает предыдущий раздел и открывает новый.
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            current = {
                "section": " › ".join(item[1] for item in stack),
                "title": title,
                "level": level,
                "start": offset,
                "lines": [line],
            }
            sections.append(current)
        elif current is None:
            current = {"section": "", "title": "", "level": 0, "start": offset, "lines": []}
            sections.append(current)
            current["lines"].append(line)
        else:
            current["lines"].append(line)
        offset += len(line) + 1               # +1 — сам перевод строки
    for item in sections:
        item["text"] = "\n".join(item.pop("lines")).strip("\n")
    # Пустые разделы (заголовок без текста) и пустой «хвост» до первого заголовка
    # в чанки не идут: чанк из одного заголовка ничего не находит.
    return [item for item in sections if item["text"].strip()]


def _paragraph_sections(text: str) -> List[Dict[str, Any]]:
    """Запасной путь структурной стратегии: разделы — абзацы.

    Вызывается, когда заголовков в документе нет вовсе (сплошной текст, выгрузка
    из чата, распознанный скан). Разделы здесь — абзацы, поэтому `section`
    остаётся пустым: выдумывать заголовок, которого в документе нет, нельзя.
    """
    sections: List[Dict[str, Any]] = []
    offset = 0
    for block in _BLANK_RE.split(text):
        block_text = block.strip("\n")
        if block_text.strip():
            sections.append({
                "section": "", "title": "", "level": 0,
                "start": offset, "text": block_text,
            })
        offset += len(block) + 2              # разделитель абзацев — "\n\n"
    return sections


# ---------------------------------------------------------------------------
# Оконное разбиение (используется стратегией fixed и дорезкой длинных разделов)
# ---------------------------------------------------------------------------
def _cut_point(text: str, start: int, limit: int) -> int:
    """Ищет естественную границу в конце окна: конец абзаца, затем предложения.

    Окно — [start, start + limit). Возвращает позицию разреза так, чтобы чанк не
    превысил limit. Границы ищутся только в последней пятой части окна: если
    подтягивать разрез к любому стыку, чанк окажется вдвое короче заданного.
    """
    end = min(len(text), start + limit)
    if end >= len(text):
        return len(text)
    floor = end - max(1, int(limit * _BOUNDARY_TAIL_RATIO))
    window = text[floor:end]
    cut = window.rfind("\n\n")
    if cut >= 0:
        return floor + cut + 2                # сам разделитель остаётся в чанке
    cut = window.rfind("\n")
    if cut >= 0:
        return floor + cut + 1
    best = -1
    for match in _SENTENCE_END_RE.finditer(window):
        best = match.end()
    if best > 0:
        return floor + best
    return end


def _window_chunks(text: str, size: int, overlap: int, *,
                   section: str = "", start_at: int = 0,
                   kind: str = "window") -> List[Dict[str, Any]]:
    """Режет текст на окна с перекрытием. Возвращает заготовки чанков.

    Шаг окна = size - overlap (гарантированно положительный: перекрытие
    нормализовано до половины размера). Каждое следующее окно начинается на
    overlap символов раньше конца предыдущего — фраза на стыке попадает в оба
    чанка и не теряется при поиске.
    """
    chunks: List[Dict[str, Any]] = []
    length = len(text)
    step = max(1, size - overlap)
    position = 0
    cursor = 0
    while cursor < length:
        end = _cut_point(text, cursor, size)
        if end <= cursor:                      # защита от нулевого шага
            end = min(length, cursor + size)
        piece = text[cursor:end]
        if piece.strip():
            chunks.append({
                "section": section,
                "start": start_at + cursor,
                "end": start_at + end,
                "chars": len(piece),
                "text": piece.strip("\n"),
                "kind": kind,
                "position": position,
                "merged": [],
            })
            position += 1
        if end >= length:
            break
        cursor = max(cursor + step, end - overlap)
        if cursor >= length:
            break
    return chunks


def _common_section(left: str, right: str) -> str:
    """Общий путь двух разделов: «Глава 1 › 1.2» + «Глава 1 › 1.3» → «Глава 1».

    Склеенный чанк лежит сразу в двух разделах, и подписывать его одним из них
    значило бы соврать: найденный фрагмент показали бы не на том месте
    документа. Общий родитель — честный и всё ещё полезный адрес.
    """
    left_parts = [part for part in str(left or "").split(" › ") if part]
    right_parts = [part for part in str(right or "").split(" › ") if part]
    common: List[str] = []
    for first, second in zip(left_parts, right_parts):
        if first != second:
            break
        common.append(first)
    return " › ".join(common)


# Тело хвостового раздела короче этого — раздел считается вырожденным (заголовок
# и одна строка) и присоединяется к предыдущему. Раздел с настоящим текстом
# остаётся своим чанком, даже если он короче общего порога склейки.
_TAIL_BODY_MIN = 40


def _tail_is_degenerate(item: Dict[str, Any]) -> bool:
    """Заголовок без тела, а не раздел с коротким текстом."""
    text = str(item.get("text") or "")
    title = str(item.get("title") or "").strip()
    lines = text.split("\n", 1)
    first = lines[0].strip() if lines else ""
    # Тело — всё, кроме строки заголовка (если текст начинается именно с него).
    body = lines[1] if (len(lines) > 1 and (not title or first == title)) else ""
    if not body:
        body = "" if (title and first == title) else text
    return len(body.strip()) < _TAIL_BODY_MIN


def _merge_small_sections(sections: List[Dict[str, Any]],
                          min_chars: int, max_chars: int) -> List[Dict[str, Any]]:
    """Склеивает МЕЛКИЕ разделы, пока накопленный чанк не станет осмысленным.

    Заголовок без текста («Введение» из двух строк) — плохой чанк: он находится
    по любому запросу про документ и не несёт ответа. Поэтому подряд идущие
    разделы копятся, пока накопленное не дойдёт до `min_chars`; как только
    дошло — чанк отдаётся и копление начинается заново.

    Два важных ограничения:

    - порог `min_chars` НЕ зависит от размера чанка: он ловит вырожденные
      разделы (один заголовок), а не «оптимизирует» структуру. При пороге,
      растущем вместе с `chunk_size`, документ из коротких разделов схлопывался
      бы в ОДИН чанк, и структурная стратегия переставала бы что-либо значить;
    - склейка не выводит чанк за `max_chars`. Если накопленный мелкий раздел
      вместе со следующим в предел не влезает, мелкий НЕ отдаётся отдельным
      чанком (это снова был бы чанк-заголовок): его текст входит в НАЧАЛО
      следующего раздела, а сам следующий сохраняет свой путь — он конкретнее, и
      длинный текст дальше дорежется окнами. Два ПОЛНОЦЕННЫХ раздела при этом не
      склеиваются: у каждого свой адрес в документе, и дорезка только размыла бы
      подписи.

    Заголовки склеенных разделов перечисляются в `merged`, а общий адрес чанка
    считается как их ОБЩИЙ путь (см. _common_section): показать фрагмент по
    адресу одного из двух разделов значило бы соврать.
    """
    if len(sections) < 2:
        return sections
    merged: List[Dict[str, Any]] = []
    carry: Optional[Dict[str, Any]] = None
    for item in sections:
        current = dict(item)
        if carry is not None:
            combined = carry["text"] + "\n\n" + current["text"]
            if len(combined) <= max_chars:
                current = {
                    "section": _common_section(carry["section"], current["section"]),
                    "title": current["title"] or carry["title"],
                    "level": current["level"],
                    "start": carry["start"],
                    "text": combined,
                    "merged": list(carry.get("merged") or [])
                              + ([carry["title"]] if carry.get("title") else []),
                }
                carry = None
            else:
                # В один чанк не влезает, но и отдавать мелкий раздел отдельным
                # чанком нельзя: чанк из одного заголовка находится по любому
                # запросу и ответа не несёт. Заголовок мелкого раздела входит в
                # НАЧАЛО следующего, а сам следующий сохраняет свой путь (он
                # конкретнее — фрагмент показывается по нему); длинный текст
                # дальше дорежется окнами.
                current = {
                    "section": current["section"],
                    "title": current["title"] or carry["title"],
                    "level": current["level"],
                    "start": carry["start"],
                    "text": combined,
                    "merged": list(carry.get("merged") or [])
                              + ([carry["title"]] if carry.get("title") else []),
                }
                carry = None
        if len(current["text"]) < min_chars:
            carry = current
            continue
        merged.append(current)
    if carry is not None:
        if merged and _tail_is_degenerate(carry):
            # Хвостовой раздел присоединяем к предыдущему ТОЛЬКО если он
            # вырожденный: заголовок, под которым почти ничего нет. Иначе — своя
            # часть: склейка НАЗАД соединяет два РАЗНЫХ раздела, и подпись места
            # теряется вовсе (общий путь у них пустой), а найденный фрагмент
            # оказывается «нигде». Именно так терялись разделы распознанного скана.
            last = merged[-1]
            last["section"] = _common_section(last["section"], carry["section"])
            last["text"] = last["text"] + "\n\n" + carry["text"]
            last["merged"] = list(last.get("merged") or []) + list(carry.get("merged") or [])
            if carry.get("title"):
                last["merged"].append(carry["title"])
        else:
            # Документ целиком меньше порога: единственный чанк и есть документ.
            merged.append(carry)
    return merged


# ---------------------------------------------------------------------------
# Публичная точка входа
# ---------------------------------------------------------------------------
def chunk_document(text: Any, *, source: str, title: str = "",
                   strategy: Any = DEFAULT_STRATEGY,
                   chunk_size: Any = DEFAULT_CHUNK_SIZE,
                   overlap: Any = DEFAULT_CHUNK_OVERLAP,
                   doc_index: int = 0, id_prefix: str = "") -> List[Dict[str, Any]]:
    """Разбивает ОДИН документ на чанки с метаданными.

    `source` — имя файла, `title` — заголовок документа (по умолчанию имя файла
    без расширения). `id_prefix` — префикс идентификатора чанка (пайплайн ставит
    сюда id базы знаний), чтобы chunk_id был читаемым: «kb-1a2b3c4d-0-0007».
    Без префикса идентификатор выводится из имени файла.

    В идентификатор входит НОМЕР ДОКУМЕНТА в базе: у базы из нескольких файлов
    нумерация чанков начинается заново в каждом документе, и без номера документа
    два чанка из разных файлов получили бы ОДИН chunk_id — а он обязан быть
    уникальным внутри базы (по нему чанк находят и на него ссылаются).

    Функция вызывается на каждый документ отдельно — границы файлов не
    смешиваются (см. заголовок модуля).
    """
    settings = chunk_settings(strategy, chunk_size, overlap)
    body = normalize_text(text)
    name = str(source or "").strip() or "документ"
    doc_title = str(title or "").strip() or strip_extension(name)
    prefix = str(id_prefix or "").strip() or ("doc-" + _digest(name)[:8])
    number = int(doc_index)

    if not body:
        return []

    if settings["strategy"] == STRATEGY_FIXED:
        raw_chunks = _window_chunks(body, settings["chunk_size"], settings["overlap"])
    else:
        raw_chunks = _structure_chunks(body, settings)

    chunks: List[Dict[str, Any]] = []
    for position, item in enumerate(raw_chunks):
        chunk = dict(item)
        chunk.update({
            "chunk_id": "%s-%d-%04d" % (prefix, number, position),
            "source": name,
            "title": doc_title,
            "doc_index": number,
            "position": position,
            "strategy": settings["strategy"],
            "path": chunk.get("section") or "",
            "chars": len(chunk["text"]),
        })
        chunks.append(chunk)
    return chunks


def _structure_chunks(body: str, settings: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Структурная стратегия: разделы по заголовкам, дорезка оконным алгоритмом."""
    size = settings["chunk_size"]
    overlap = settings["overlap"]
    sections = _heading_sections(body)
    # Заголовков нет — определяем это по ПУСТЫМ путям разделов, а не по пустому
    # списку: _heading_sections всегда отдаёт хотя бы один раздел (текст перед
    # первым заголовком), и «список не пуст» ничего не говорит о структуре.
    fallback = "" if any(item.get("section") for item in sections) else "paragraphs"
    if fallback:
        sections = _paragraph_sections(body)
    if not sections:
        return _window_chunks(body, size, overlap, kind="fixed-empty")

    min_section = _MIN_SECTION_FLOOR
    sections = _merge_small_sections(sections, min_section, size)

    chunks: List[Dict[str, Any]] = []
    for item in sections:
        section = item.get("section") or ""
        merged = list(item.get("merged") or [])
        text = item["text"]
        if len(text) <= size:
            chunks.append({
                "section": section, "start": item["start"],
                "end": item["start"] + len(text), "chars": len(text),
                "text": text.strip("\n"),
                "kind": "paragraph" if fallback else ("section" if section else "preface"),
                "position": 0, "merged": merged,
            })
            continue
        # Длинный раздел дорезается окнами; путь раздела у всех частей ОДИН —
        # найденный фрагмент должен ссылаться на настоящий раздел документа.
        # Вид чанка в документе без заголовков — «абзац»: там окна режут абзацы,
        # и называть их «частью раздела» было бы неправдой.
        pieces = _window_chunks(text, size, overlap, section=section,
                                start_at=item["start"],
                                kind="paragraph" if fallback else "section-window")
        if merged and pieces:
            pieces[0]["merged"] = merged
        for piece in pieces:
            piece["section"] = section
        chunks.extend(pieces)
    for position, item in enumerate(chunks):
        item["position"] = position
    return chunks


def strip_extension(name: str) -> str:
    """Имя файла без расширения — заголовок документа по умолчанию."""
    base = str(name or "").strip()
    for separator in ("/", "\\"):
        base = base.rsplit(separator, 1)[-1]
    if "." in base[1:]:
        base = base.rsplit(".", 1)[0]
    return base.strip() or "документ"


def _digest(value: str) -> str:
    """Короткий устойчивый хеш строки (blake2b: не зависит от запуска процесса).

    Встроенный `hash()` для строк солится при каждом старте интерпретатора —
    идентификаторы чанков менялись бы от запуска к запуску.
    """
    return hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).hexdigest()


def describe(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Сводка по чанкам документа: сколько их и какого они размера.

    Нужна и пайплайну (метрики базы), и проверкам: «стратегия сработала» —
    это не только «чанки есть», но и «размеры в заданных границах».
    """
    sizes = [int(item.get("chars") or 0) for item in chunks]
    if not sizes:
        return {"chunks": 0, "chars_total": 0, "chars_avg": 0,
                "chars_min": 0, "chars_max": 0, "sections": 0,
                "kinds": {}}
    kinds: Dict[str, int] = {}
    for item in chunks:
        key = str(item.get("kind") or "chunk")
        kinds[key] = kinds.get(key, 0) + 1
    sections = {str(item.get("section") or "") for item in chunks}
    sections.discard("")
    return {
        "chunks": len(sizes),
        "chars_total": sum(sizes),
        "chars_avg": int(round(sum(sizes) / float(len(sizes)))),
        "chars_min": min(sizes),
        "chars_max": max(sizes),
        "sections": len(sections),
        "kinds": kinds,
    }
