"""Периодические задачи режима «AI-агент»: расписание и разбор периода.

Периодическая задача — это обычная задача-диалог (сессия workspace), у которой
есть РАСПИСАНИЕ: сервер сам повторяет её запрос через заданный промежуток и
кладает результат в чат (см. app/periodic_runner.py). Ничего своего в диалоге
такая задача не имеет — весь конвейер (конечный автомат, MCP-данные, проверка
результата) остаётся общим; расписание живёт РЯДОМ с сессией, в поле
`session["periodic"]`, и в переписку (`messages`) не попадает.

Правила периода (что видит пользователь):

* по умолчанию задача повторяется РАЗ В СУТКИ;
* если в тексте запроса назван другой период («Сводка погоды в Москве за
  последние сутки, раз в час») — берётся он; назвать период можно и позже,
  новым сообщением в этой же задаче;
* период ограничен снизу (`MIN_INTERVAL` — минута) и сверху (`MAX_INTERVAL`);
* повтор включает и выключает пользователь (кнопка 🔁 у задачи в списке), и
  выключенный повтор сам не возвращается — даже если в задаче написать ещё
  сообщение.

Модуль чистый: ни файлов, ни сети, ни модели. Время — НАИВНОЕ ЛОКАЛЬНОЕ, той же
природы, что метки `_now()` в workspace (ISO, до секунд), поэтому расписание
переживает перезапуск приложения и читается без возни с зонами.
"""

import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

# Период по умолчанию — сутки: задача без названного периода повторяется раз в
# день (см. правило в заголовке модуля).
DEFAULT_INTERVAL = 24 * 60 * 60
# Нижняя граница: чаще раза в минуту повторять нельзя — каждый повтор это
# полный цикл задачи (план, шаги, проверка), то есть несколько вызовов LLM.
MIN_INTERVAL = 60
# Верхняя граница — 30 суток (примерно месяц): дальше «периодическая» задача
# перестаёт отличаться от обычной.
MAX_INTERVAL = 30 * 24 * 60 * 60
# Сколько символов запроса задачи храним в расписании: этим текстом планировщик
# повторяет задачу, поэтому его нельзя терять (как state.request автомата).
REQUEST_LIMIT = 2000

# Пометка автоматического запуска в журнале чата и в памяти диалога.
AUTO_MARK = "⏱"
# Признак реплики в памяти диалога: сообщение отправлено АВТОЗАПУСКОМ, а не
# пользователем (интерфейс рисует его служебной строкой, см. roleOf в chat.html).
SOURCE_AUTO = "periodic"

# Слова-триггеры периода: без них «час»/«сутки» в обычной фразе периодом не
# считаются («отчёт за последние сутки» — это НЕ «раз в сутки»).
_TRIGGER = r"(?:раз\s+в|кажды[ехйо]|каждую|каждое|каждый)"
# Единицы периода: (слово, секунды). Порядок важен — от меньшей единицы к
# большей: «раз в 30 минут» должно стать минутами, а не «минутами в часе».
_UNITS: List[Tuple[str, int]] = [
    (r"(?:сек\w*|с\b)", 1),
    (r"(?:мин\w*|минут\w*)", 60),
    (r"(?:час\w*|ч\b)", 60 * 60),
    (r"(?:сут\w*|дн\w*|день|дня|дней)", 24 * 60 * 60),
    (r"(?:недел\w*|нед\b)", 7 * 24 * 60 * 60),
    (r"(?:месяц\w*|мес\b)", 30 * 24 * 60 * 60),
]
# «Односложные» наречия: период назван одним словом, без числа.
_ADVERBS: List[Tuple[str, int]] = [
    (r"ежеминутно|поминутно", 60),
    (r"ежечасно|почасово", 60 * 60),
    (r"ежедневно|ежесуточно|каждый\s+день|каждые\s+сутки", 24 * 60 * 60),
    (r"еженедельно|каждую\s+неделю|раз\s+в\s+неделю", 7 * 24 * 60 * 60),
    (r"ежемесячно|каждый\s+месяц|раз\s+в\s+месяц", 30 * 24 * 60 * 60),
]

_NUMBER = r"(\d{1,4})"


def _compile() -> List[Tuple[Any, int]]:
    """Шаблоны поиска периода: сначала «раз в N <единица>», потом наречия."""
    patterns: List[Tuple[Any, int]] = []
    for unit, seconds in _UNITS:
        # «раз в час» / «каждые 30 минут» / «каждую минуту»: число необязательно.
        pattern = re.compile(
            r"\b" + _TRIGGER + r"\s+(?:" + _NUMBER + r"\s*)?" + unit + r"\b")
        patterns.append((pattern, seconds))
    for adverb, seconds in _ADVERBS:
        patterns.append((re.compile(r"\b(?:" + adverb + r")"), seconds))
    return patterns


_PATTERNS = _compile()


def now() -> datetime:
    """Текущее наивное локальное время (та же шкала, что метки workspace)."""
    return datetime.now()


def to_iso(moment: datetime) -> str:
    """Метка времени для файла workspace (ISO, до секунд)."""
    return moment.replace(microsecond=0).isoformat(timespec="seconds")


def parse_time(value: Any) -> Optional[datetime]:
    """Читает метку времени расписания. Битое значение — None (не падаем)."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def clamp(interval: Any) -> int:
    """Приводит период к допустимому: минута … 30 суток (мусор → сутки)."""
    try:
        seconds = int(interval)
    except (TypeError, ValueError):
        return DEFAULT_INTERVAL
    if seconds <= 0:
        return DEFAULT_INTERVAL
    return max(MIN_INTERVAL, min(MAX_INTERVAL, seconds))


def plural(count: int, one: str, few: str, many: str) -> str:
    """Русская форма слова по числу: 1 минута, 2 минуты, 5 минут."""
    value = abs(int(count)) % 100
    if 11 <= value <= 14:
        return many
    value %= 10
    if value == 1:
        return one
    if 2 <= value <= 4:
        return few
    return many


def label(interval: Any) -> str:
    """Человеческая запись периода для интерфейса: «раз в час», «каждые 5 минут»."""
    seconds = clamp(interval)
    if seconds < 60 * 60:
        minutes = max(1, seconds // 60)
        return "раз в минуту" if minutes == 1 else (
            f"каждые {minutes} {plural(minutes, 'минуту', 'минуты', 'минут')}")
    if seconds < 24 * 60 * 60:
        hours = max(1, seconds // (60 * 60))
        return "раз в час" if hours == 1 else (
            f"каждые {hours} {plural(hours, 'час', 'часа', 'часов')}")
    week = 7 * 24 * 60 * 60
    if seconds % week == 0:
        weeks = max(1, seconds // week)
        return "раз в неделю" if weeks == 1 else (
            f"каждые {weeks} {plural(weeks, 'неделю', 'недели', 'недель')}")
    days = max(1, seconds // (24 * 60 * 60))
    return "раз в сутки" if days == 1 else (
        f"каждые {days} {plural(days, 'день', 'дня', 'дней')}")


def parse_request_detailed(text: Any) -> Tuple[Optional[int], Optional[int], str]:
    """Ищет период в тексте: (названные секунды, приведённые к границам, фраза).

    «Названные секунды» нужны, чтобы честно сказать пользователю «в запросе было
    раз в 5 секунд, чаще минуты повторять нельзя» (см. chat.py): приведённое
    значение одно и то же и для ответа, и для расписания.
    Ничего не нашлось — (None, None, "").
    """
    normalized = " ".join(str(text or "").lower().split())
    if not normalized:
        return None, None, ""
    for pattern, unit_seconds in _PATTERNS:
        match = pattern.search(normalized)
        if not match:
            continue
        phrase = match.group(0).strip()
        number = None
        for group in match.groups():
            if group:
                number = int(group)
                break
        count = number if number else 1
        raw = count * unit_seconds
        return raw, clamp(raw), phrase
    return None, None, ""


def parse_request(text: Any) -> Tuple[Optional[int], str]:
    """Ищет период в тексте запроса: (секунды, найденная фраза).

    Ничего не нашлось — (None, ""): период остаётся прежним (для новой
    периодической задачи это сутки по умолчанию). Число берётся из фразы
    («каждые 30 минут» → 1800), «раз в час» → 3600, «ежедневно» → 86400.
    Значение уже приведено к допустимым границам (см. parse_request_detailed).
    """
    _raw, seconds, phrase = parse_request_detailed(text)
    return seconds, phrase


def normalize(raw: Any) -> Dict[str, Any]:
    """Приводит расписание сессии к безопасному виду ({} — задача не периодическая).

    Расписание есть только там, где в файле лежит признак «enabled»: выключенный
    повтор (enabled: False) — это по-прежнему ПЕРИОДИЧЕСКАЯ задача, которую
    пользователь остановил, поэтому расписание сохраняется и задача остаётся
    помеченной в списке.
    """
    if not isinstance(raw, dict) or "enabled" not in raw:
        return {}
    meta: Dict[str, Any] = {
        "enabled": bool(raw.get("enabled")),
        "interval": clamp(raw.get("interval")),
        "request": str(raw.get("request") or "").strip()[:REQUEST_LIMIT],
        "next_run": str(raw.get("next_run") or "").strip()[:40],
        "last_run": str(raw.get("last_run") or "").strip()[:40],
        "runs": max(0, _as_int(raw.get("runs"))),
        "error": str(raw.get("error") or "").strip()[:400],
    }
    if meta["enabled"]:
        due = parse_time(meta["next_run"])
        if due is None:
            # Срок потерян (битая метка) — ставим его заново от текущего момента:
            # из-за мусора в файле повтор не должен начаться «прямо сейчас».
            meta["next_run"] = to_iso(now() + timedelta(seconds=int(meta["interval"])))
        else:
            meta["next_run"] = to_iso(due)
    return meta


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def make(interval: Any = DEFAULT_INTERVAL, request: str = "",
         enabled: bool = True, moment: Optional[datetime] = None) -> Dict[str, Any]:
    """Новое расписание: период + первый срок следующего повтора."""
    moment = moment or now()
    seconds = clamp(interval)
    return {
        "enabled": bool(enabled),
        "interval": seconds,
        "request": str(request or "").strip()[:REQUEST_LIMIT],
        "next_run": to_iso(moment + timedelta(seconds=seconds)) if enabled else "",
        "last_run": "",
        "runs": 0,
        "error": "",
    }


def reschedule(meta: Dict[str, Any], interval: Any = None,
               request: Optional[str] = None,
               moment: Optional[datetime] = None) -> Dict[str, Any]:
    """Пересчитывает срок по (новому) периоду — правка из интерфейса или запроса.

    Период не назван — остаётся прежний; запрос не назван — остаётся прежний.
    Срок считается ОТ ПРАВКИ: пользователь, поставивший «раз в час», ждёт первый
    повтор через час, а не «когда-нибудь по прежнему расписанию».
    """
    moment = moment or now()
    if interval is not None:
        meta["interval"] = clamp(interval)
    if request is not None:
        meta["request"] = str(request or "").strip()[:REQUEST_LIMIT]
    meta["enabled"] = bool(meta.get("enabled", True))
    if meta["enabled"]:
        meta["next_run"] = to_iso(moment + timedelta(seconds=int(meta["interval"])))
    return meta


def set_enabled(meta: Dict[str, Any], enabled: bool,
                moment: Optional[datetime] = None) -> Dict[str, Any]:
    """Включает/выключает повтор. Включение сразу ставит срок от текущего момента."""
    moment = moment or now()
    meta["enabled"] = bool(enabled)
    if meta["enabled"]:
        meta["error"] = ""
        meta["next_run"] = to_iso(moment + timedelta(seconds=int(meta["interval"])))
    return meta


def started(meta: Dict[str, Any], moment: Optional[datetime] = None) -> None:
    """Повтор НАЧАЛСЯ: срок сдвигается сразу.

    Так один и тот же повтор не запустится дважды (пока идёт прогон, следующий
    тик планировщика видит уже будущий срок), а сам прогон может идти дольше
    периода — тогда срок поправит `finished` по фактическому времени.
    """
    moment = moment or now()
    meta["next_run"] = to_iso(moment + timedelta(seconds=int(meta["interval"])))


def finished(meta: Dict[str, Any], started_at: datetime, ok: bool,
             error: str = "", moment: Optional[datetime] = None) -> None:
    """Повтор ЗАКОНЧИЛСЯ: счётчик, отметка времени, причина сбоя и новый срок.

    Срок считается от НАЧАЛА повтора (расписание не «плывёт» от длительности
    прогона). Если прогон занял больше периода, следующий повтор — через период
    от текущего момента: иначе повторы шли бы подряд без пауз.
    """
    moment = moment or now()
    interval = int(meta.get("interval") or DEFAULT_INTERVAL)
    meta["last_run"] = to_iso(moment)
    meta["runs"] = _as_int(meta.get("runs")) + 1
    meta["error"] = "" if ok else str(error or "повтор не выполнен")[:400]
    due = started_at + timedelta(seconds=interval)
    if due <= moment:
        due = moment + timedelta(seconds=interval)
    meta["next_run"] = to_iso(due)


def is_due(meta: Dict[str, Any], moment: Optional[datetime] = None) -> bool:
    """True — повтор пора запускать (включён и срок наступил/прошёл).

    Срок не читается (пустой или битый) — False: повторы не начинаются «сами»
    от испорченной метки, срок починит правка расписания.
    """
    if not isinstance(meta, dict) or not meta.get("enabled"):
        return False
    due = parse_time(meta.get("next_run"))
    if due is None:
        return False
    return due <= (moment or now())


def seconds_left(meta: Dict[str, Any], moment: Optional[datetime] = None) -> Optional[int]:
    """Сколько секунд до следующего повтора (None — повтора нет/срок неизвестен)."""
    if not isinstance(meta, dict) or not meta.get("enabled"):
        return None
    due = parse_time(meta.get("next_run"))
    if due is None:
        return None
    return int((due - (moment or now())).total_seconds())


def when_label(meta: Dict[str, Any], moment: Optional[datetime] = None) -> str:
    """Строка «когда следующий повтор» для интерфейса."""
    if not isinstance(meta, dict) or not meta.get("enabled"):
        return "повтор остановлен"
    left = seconds_left(meta, moment)
    if left is None:
        return "срок неизвестен"
    if left <= 0:
        return "повтор вот-вот начнётся"
    if left < 60:
        return f"повтор через {left} с"
    if left < 60 * 60:
        minutes = left // 60
        return f"повтор через {minutes} {plural(minutes, 'минуту', 'минуты', 'минут')}"
    if left < 24 * 60 * 60:
        hours = left // (60 * 60)
        return f"повтор через {hours} {plural(hours, 'час', 'часа', 'часов')}"
    days = left // (24 * 60 * 60)
    return f"повтор через {days} {plural(days, 'день', 'дня', 'дней')}"


def brief(meta: Optional[Dict[str, Any]], running: bool = False, hold: str = "",
          log_len: int = 0, moment: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
    """Краткая запись расписания для снимка workspace и опроса интерфейсом.

    `log_len` — размер журнала чата задачи: по его изменению интерфейс понимает,
    что автозапуск дописал в диалог новое (см. pollPeriodic в chat.html).

    `hold` — почему повторы СЕЙЧАС не идут, хотя расписание включено: "paused"
    (задача на паузе) или "cancelled" (задача отменена — остановка периодической
    задачи, см. app/periodic_runner.py). Пусто — повторы идут по расписанию.
    """
    if not isinstance(meta, dict) or not meta:
        return None
    moment = moment or now()
    return {
        "enabled": bool(meta.get("enabled")),
        "interval": int(meta.get("interval") or DEFAULT_INTERVAL),
        "label": label(meta.get("interval")),
        "next_run": str(meta.get("next_run") or ""),
        "last_run": str(meta.get("last_run") or ""),
        "left": seconds_left(meta, moment),
        "when": when_label(meta, moment),
        "runs": _as_int(meta.get("runs")),
        "error": str(meta.get("error") or ""),
        "request": str(meta.get("request") or ""),
        "running": bool(running),
        "hold": str(hold or "")[:20],
        "log_len": max(0, _as_int(log_len)),
    }
