"""Инварианты режима «AI-агент»: что агент НЕ имеет права нарушить.

Инвариант — короткое текстовое правило, которое агент обязан соблюдать при
рассуждениях и не имеет права предлагать решения, его нарушающие: выбранная
архитектура, принятые технические решения, ограничения по стеку, бизнес-правила.

Инварианты живут ОТДЕЛЬНО ОТ ДИАЛОГА (в диалог они не пишутся и стратегиям
контекста не подчиняются — как слои памяти):

    область «Задача»  — dialog["invariants"] сессии: правила ЭТОГО диалога;
    область «Проект»  — task["invariants"] задачи-workspace: правила всего
                        проекта (в терминах интерфейса «проект» — это task).

Хранение и CRUD — в app/ai/workspace.py (там же живут и диалоги); этот модуль
отвечает за два других обязательных свойства:

1) блок системного промпта (`invariants_block`), который уходит модели в КАЖДОМ
   запросе агента: «учитывай явно, отказывайся нарушать, сообщай о противоречии»;
2) РАЗБОР ЗАПРОСА до планирования (`analyze`): нарушает ли запрос правило
   (правила проекта всегда главнее правил задачи) — если да, агент отказывается
   работать по такому запросу, а плана не строит. Варианты-альтернативы, которые
   показываются пользователю, ПРОВЕРЯЮТСЯ отдельным служебным вызовом
   (`check_suggestions`): обещание «эти варианты правил не нарушают» должно
   опираться на проверку, а не на просьбу в промпте.

Модуль без состояния и без файлов: на вход приходят списки инвариантов, на
выход — строки и словари. Работа с моделью идёт через готовую функцию вызова LLM
(`analyze`, `check_suggestions`; как agent.py получает `client.call_llm_async`),
поэтому модуль проверяется без сети.

Проверка пар «правило проекта × правило задачи» (`pairs`, `detect`,
`resolve_pair`) осталась только для совместимости со старыми записями: маршруты
её НЕ вызывают — правила пишутся без обращений к модели, а противоречие правил
выясняется разбором запроса (правило проекта всегда главнее, выбора нет).
"""

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from app.ai import json_utils

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Области (scope) инвариантов: где правило действует.
# ---------------------------------------------------------------------------
SCOPE_PROJECT = "project"  # правила всего проекта (task workspace)
SCOPE_TASK = "task"        # правила конкретной задачи-диалога (сессия)
SCOPES = (SCOPE_PROJECT, SCOPE_TASK)

# Вердикты проверки пары «инвариант проекта × инвариант задачи».
VERDICT_CONFLICT = "conflict"   # правила противоречат друг другу
VERDICT_CLEAR = "clear"         # правила совместимы
VERDICTS = (VERDICT_CONFLICT, VERDICT_CLEAR)

# Чьё правило главнее после решения пользователя.
WINNER_PROJECT = SCOPE_PROJECT
WINNER_TASK = SCOPE_TASK
WINNERS = (WINNER_PROJECT, WINNER_TASK)

REASON_LIMIT = 400
# Формат вызова LLM: async (messages, **kwargs) -> (текст, метрики|None).
LlmCall = Callable[..., Any]

# ---------------------------------------------------------------------------
# Разбор ЗАПРОСА на соответствие инвариантам (до этапа планирования).
# ---------------------------------------------------------------------------
# Вердикты разбора запроса: запрос можно выполнять / он нарушает инвариант /
# противоречат сами правила (задачи и проекта) — тогда решает пользователь.
COMPLIANCE_CLEAR = "clear"
COMPLIANCE_VIOLATION = "violation"
# Совместимость: прежний вердикт «противоречие правил» приводим к нарушению —
# правило проекта всегда главнее, выбора между правилами пользователю не даём.
COMPLIANCE_CONFLICT = COMPLIANCE_VIOLATION
COMPLIANCE_VERDICTS = (COMPLIANCE_CLEAR, COMPLIANCE_VIOLATION)
# Виды разбора (для интерфейса): нарушение запроса правилом.
ANALYSIS_VIOLATION = COMPLIANCE_VIOLATION
ANALYSIS_CONFLICT = COMPLIANCE_VIOLATION
ANALYSIS_EXCEPTION = "exception"
# Сколько вариантов решения показываем и сколько ждём от модели.
MIN_SUGGESTIONS = 2
MAX_SUGGESTIONS = 4
# Сколько раз просим модель ПЕРЕДЕЛАТЬ варианты, если после проверки их меньше
# нормы. Каждая попытка — служебный вызов LLM, поэтому попытка ровно одна.
SUGGESTION_RETRIES = 1
# Сколько раз просим планировщика ПЕРЕДЕЛАТЬ ПЛАН, если проверка нашла шаги,
# нарушающие правила (код-гейт плана, см. `check_steps`). Одна попытка: каждая —
# отдельный служебный вызов планировщика.
PLAN_RETRIES = 1
# Сколько номеров правил принимаем в поле «недействующие» (правил задачи не
# может быть больше, чем MAX_INVARIANTS в workspace.py).
MAX_OVERRIDDEN = 50
TITLE_LIMIT = 120
DETAIL_LIMIT = 600
TEXT_LIMIT = 600
EXPLANATION_LIMIT = 1500
# Кэш разбора в диалоге: ключ (запрос + правила) и подпись — по ним видно, что
# сохранённый разбор ещё актуален.
_REQUEST_LIMIT = 4000

ANALYSIS_PROMPT = (
    "Ты — арбитр инвариантов. Тебе дают ЗАПРОС ПОЛЬЗОВАТЕЛЯ и ИНВАРИАНТЫ — "
    "правила, которые исполнитель НАРУШАТЬ НЕ ИМЕЕТ ПРАВА (архитектура, "
    "технические решения, ограничения стека, бизнес-правила): правила ПРОЕКТА "
    "(действуют во всех задачах) и правила ТЕКУЩЕЙ ЗАДАЧИ.\n"
    "ГЛАВНОЕ ПРАВИЛО ПРИОРИТЕТА: инвариант ПРОЕКТА всегда главнее. Правило "
    "задачи, противоречащее правилу проекта (проект требует только Android, а "
    "задача просит только iOS), для этой задачи НЕ ДЕЙСТВУЕТ — проект его "
    "перебивает.\n"
    "НО САМО ПРОТИВОРЕЧИЕ СКРЫВАТЬ НЕЛЬЗЯ: если правило ТЕКУЩЕЙ ЗАДАЧИ "
    "противоречит правилу проекта, вердикт — violation, даже когда запрос "
    "сформулирован нейтрально («нужен план приложения погоды» без платформы): "
    "задача в целом требует запрещённого проектом, и пользователь должен об этом "
    "узнать и выбрать вариант — работать по правилам проекта или поправить свои "
    "правила. Молча выполнить работу «по-своему» вместо требования задачи — "
    "ошибка: правило задачи выглядит для пользователя проигнорированным.\n"
    "В «объяснении» для такого случая назови ОБА правила: что просит правило "
    "задачи, что запрещает правило проекта и что поэтому действует правило "
    "проекта; в «недействующие» укажи номер правила задачи. Варианты — как "
    "выполнить работу по правилам ПРОЕКТА.\n"
    "ЕСЛИ правила задачи правилам проекта НЕ противоречат, суди ЗАПРОС: violation "
    "— только когда ЗАПРОС сам требует запрещённого (веб, iOS, мультиплатформа, "
    "чужой язык или стек), иначе clear.\n"
    "Верни строго JSON без пояснений и markdown:\n"
    '{"вердикт": "clear"|"violation", "объяснение": "...", '
    '"недействующие": [номера правил задачи, противоречащих правилам проекта], '
    '"варианты": [{"заголовок": "...", "пояснение": "...", "запрос": "..."}]}\n'
    "ПРАВИЛА ВЕРДИКТА:\n"
    '- "clear" — запрос не нарушает ни одного ДЕЙСТВУЮЩЕГО правила и правила '
    'задачи правилам проекта НЕ противоречат; "варианты" пустые.\n'
    '- "violation" — либо ЗАПРОС требует нарушить действующее правило (сам просит '
    "веб, iOS, мультиплатформу, чужой язык или стек, другое требование проекта; "
    "либо нарушает правило задачи, не противоречащее правилам проекта), либо "
    "правило ТЕКУЩЕЙ ЗАДАЧИ противоречит правилу проекта (см. выше — об этом "
    "обязательно сообщаем). В «объяснении» назови КОНКРЕТНОЕ правило и объясни, "
    "почему требование невозможно выполнить: для противоречия правил отдельно "
    "скажи, что запрещает правило проекта и что просит правило задачи, и что "
    "поэтому действует правило проекта.\n"
    "ПРАВИЛА ВАРИАНТОВ (для violation):\n"
    "- от 2 до 4 РАЗУМНЫХ альтернатив; НИ ОДИН вариант не нарушает НИ ОДНО "
    "правило (в том числе правило проекта);\n"
    "- «заголовок» — короткое название варианта (до 8 слов);\n"
    "- «пояснение» — почему вариант укладывается в правила и чем отличается от "
    "исходного требования (1–2 предложения);\n"
    "- «запрос» — готовый текст запроса от лица пользователя, который он может "
    "отправить как есть (конкретный, с выбранной технологией/подходом).\n"
    "ПРАВИЛО ПОЛЯ «недействующие» (заполняется ВСЕГДА, при любом вердикте):\n"
    "в «недействующие» перечисли НОМЕРА правил ТЕКУЩЕЙ ЗАДАЧИ, которые "
    "противоречат правилам проекта (нумерация — как в списке «ИНВАРИАНТЫ ТЕКУЩЕЙ "
    "ЗАДАЧИ»). Такие правила для этой задачи НЕ ДЕЙСТВУЮТ: исполнитель обязан "
    "работать по правилу проекта и не переносить их требование в план и ответы, а "
    "сами они в проверке вариантов и шагов плана не участвуют. Правило задачи, "
    "которое правилам проекта НЕ противоречит (даже если оно дополняет их), в этот "
    "список НЕ попадает. Если противоречащих правил нет — пустой список []."
)


ANALYSIS_MAX_TOKENS = 1400
ANALYSIS_TIMEOUT = 60.0

# Проверка ВАРИАНТОВ-альтернатив (после вердикта violation). Промпт разбора лишь
# ПРОСИТ модель не нарушать правила — обещать это пользователю можно только
# после проверки, поэтому тексты вариантов уходят в модель ещё раз: по каждому
# варианту нужен вердикт «не нарушает». Проверка — служебный вызов LLM.
SUGGESTIONS_PROMPT = (
    "Ты — арбитр инвариантов: проверка ВАРИАНТОВ решения. Тебе дают ИНВАРИАНТЫ — "
    "правила, которые исполнитель НАРУШАТЬ НЕ ИМЕЕТ ПРАВА (архитектура, "
    "технические решения, ограничения стека, бизнес-правила): правила ПРОЕКТА "
    "(действуют во всех задачах) и правила ТЕКУЩЕЙ ЗАДАЧИ.\n"
    "ГЛАВНОЕ ПРАВИЛО ПРИОРИТЕТА: инвариант ПРОЕКТА всегда главнее.\n"
    "ВАЖНО про правила задачи: если правило ЗАДАЧИ противоречит правилу ПРОЕКТА, "
    "оно для вариантов НЕ действует (проект его перебивает). Вариант, который "
    "укладывается в правила проекта, считается совместимым, даже если он не "
    "выполняет такое противоречащее правило задачи (например проект требует "
    "только Kotlin/Android, а задача просит мультиплатформу: вариант «нативное "
    "Android-приложение» совместим). Несовместим вариант, который нарушает правило "
    "ПРОЕКТА или правило задачи, НЕ противоречащее правилам проекта.\n"
    "Тебе дают НУМЕРОВАННЫЕ ВАРИАНТЫ — готовые тексты запросов, которые "
    "предлагают пользователю вместо требования, нарушающего правила. Для КАЖДОГО "
    "варианта реши, требует ли он нарушить ХОТЬ ОДНО действующее правило. Проверяй "
    "СМЫСЛ И НАЗВАНИЯ ТЕХНОЛОГИЙ, а не слова-заверения: другой язык "
    "программирования, другая платформа/операционная система, запрещённый стек "
    "или подход («Compose Multiplatform», «Kotlin Multiplatform», iOS при проекте "
    "только под Android) — это нарушение, даже если запрет назван не дословно. "
    "Заверения самого варианта («это соответствует правилам», «без "
    "мультиплатформы») НА ВЕРУ не принимай: смотри, что в варианте реально "
    "предлагается делать.\n"
    "Верни строго JSON без пояснений и markdown:\n"
    '{"1": {"вердикт": "clear"|"violation", "причина": "..."}, "2": {...}}\n'
    "ПРАВИЛА ВЕРДИКТА:\n"
    '- "clear" — вариант не нарушает ни одного ДЕЙСТВУЮЩЕГО правила (правила '
    "проекта и те правила задачи, что не противоречат правилам проекта);\n"
    '- "violation" — нарушает: в «причине» назови КОНКРЕТНОЕ правило;\n'
    "- если вариант непонятен или проверить его нельзя — это violation "
    "(непроверенный вариант нарушающим не считается, но и совместимым тоже)."
)

SUGGESTIONS_MAX_TOKENS = 900
SUGGESTIONS_TIMEOUT = 60.0

# Проверка ШАГОВ ПЛАНА (код-гейт плана). Тот же арбитр, но про план: шаг плана
# может требовать запрещённого («реализуй сетевой слой в KMP»), даже если сам
# запрос ничего запрещённого не просил, — промпт-блок правил об этом лишь просит.
PLAN_PROMPT = (
    "Ты — арбитр инвариантов: проверка ШАГОВ ПЛАНА. Тебе дают ИНВАРИАНТЫ — "
    "правила, которые исполнитель НАРУШАТЬ НЕ ИМЕЕТ ПРАВА (архитектура, "
    "технические решения, ограничения стека, бизнес-правила): правила ПРОЕКТА "
    "(действуют во всех задачах) и правила ТЕКУЩЕЙ ЗАДАЧИ.\n"
    "ГЛАВНОЕ ПРАВИЛО ПРИОРИТЕТА: инвариант ПРОЕКТА всегда главнее.\n"
    "ВАЖНО про правила задачи: если правило ЗАДАЧИ противоречит правилу ПРОЕКТА, "
    "оно НЕ действует (проект его перебивает) — в списке правил задачи его может "
    "уже не быть. Шаг, который укладывается в правила проекта, считается "
    "совместимым, даже если он не выполняет такое противоречащее правило задачи.\n"
    "Тебе дают НУМЕРОВАННЫЕ ШАГИ ПЛАНА. Для КАЖДОГО шага реши, требует ли он "
    "нарушить ХОТЬ ОДНО действующее правило — то есть ведёт ли выполнение шага к "
    "запрещённой технологии, платформе, языку, стеку или подходу. Проверяй СМЫСЛ "
    "И НАЗВАНИЯ ТЕХНОЛОГИЙ, а не формулировку: «общий код для обеих платформ», "
    "KMP, Compose Multiplatform, iOS, веб — это нарушение правила проекта «только "
    "нативная платформа android, никакой мультиплатформы», даже если слово "
    "«мультиплатформа» в шаге не написано.\n"
    "Верни строго JSON без пояснений и markdown:\n"
    '{"1": {"вердикт": "clear"|"violation", "причина": "..."}, "2": {...}}\n'
    "ПРАВИЛА ВЕРДИКТА:\n"
    '- "clear" — выполнение шага не нарушает ни одного действующего правила;\n'
    '- "violation" — нарушает: в «причине» назови КОНКРЕТНОЕ правило;\n'
    "- если шаг непонятен или проверить его нельзя — это violation "
    "(непроверенный шаг совместимым не считается)."
)

PLAN_MAX_TOKENS = 900
PLAN_TIMEOUT = 60.0


def _rules_text(snapshot_data: Any, exceptions: Any = None) -> str:
    """Текст действующих правил (вид правила + текст + решения) — основа подписей."""
    data = normalize(snapshot_data)
    parts: List[str] = []
    for scope in (SCOPE_PROJECT, SCOPE_TASK):
        for entry in data[scope]:
            parts.append(f"{scope}:{_normalize(entry['text'])}")
    # Решения приходят либо снимком (exceptions внутри), либо отдельным списком.
    for item in list(data["exceptions"]) + list(clean_exceptions(exceptions)):
        if isinstance(item, dict):
            parts.append(f"exc:{item.get('key') or _pair_key_of(item)}:{item.get('winner') or ''}")
    return "\n".join(parts)


def rules_signature(snapshot_data: Any, exceptions: Any = None) -> str:
    """Подпись ТОЛЬКО правил (без запроса): «проверка сделана при ЭТИХ правилах».

    Нужна вариантам-альтернативам: сервер проверил текст варианта по правилам, и
    если набор правил с тех пор не менялся, повторно судить этот же текст не надо
    (см. `_verified_choice` в chat.py) — иначе пользователь выбирает вариант и
    снова получает отказ.
    """
    return _short_hash(_rules_text(snapshot_data, exceptions))


def analysis_signature(request: str, snapshot_data: Any,
                       exceptions: Any = None) -> str:
    """Подпись разбора: запрос + действующие правила + принятые решения.

    Нужна, чтобы не звать модель повторно на тот же запрос (и чтобы сохранённый
    разбор пересчитывался, когда правила или решения изменились). Учитывается вид
    правила (проект/задача), его текст и решения по противоречиям: после решения
    «главнее задача» запрос нужно сверить заново — действующее правило изменилось.
    """
    return _short_hash("\n".join([_normalize(request),
                                  _rules_text(snapshot_data, exceptions)]))


def _short_hash(text: str) -> str:
    """Короткая устойчивая подпись строки (не криптография, только кэш)."""
    import hashlib

    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def empty_analysis() -> Dict[str, Any]:
    """Пустой разбор: запрос не проверялся / нарушений нет."""
    return {"verdict": "", "kind": "", "explanation": "", "suggestions": [],
            "suggestions_checked": True, "overridden": [], "signature": "",
            "rules_signature": "", "request": ""}


def unresolved_keys(records: Any, project_invariants: Any = None,
                    task_invariants: Any = None,
                    unchecked: Any = None,
                    exceptions: Any = None) -> List[Dict[str, str]]:
    """Пары правил, по которым решения пользователя ещё НЕТ.

    Разбор запроса показывает противоречие правил ДВУМЯ решениями («главнее
    проект» / «главнее задача»), и каждое решение применяется к СВОЕЙ паре — по
    этим ключам.

    Сюда входят и пары, которые штатная проверка пар не разобрала (`unchecked`,
    «не проверено»): именно такой случай разбор и должен ловить — правило задачи
    противоречит правилу проекта, а вердикта проверки нет.
    """
    resolved_keys = {item["key"] for item in merge_exceptions(records, exceptions)}
    keys = [record["key"] for record in unresolved(records)]
    if project_invariants is not None and task_invariants is not None:
        pairs_now = pairs(project_invariants, task_invariants)
        pending = needs_check(records, pairs_now, unchecked)
        keys.extend(pair["key"] for pair in pending)
        if not pairs_now:
            # Правил с одной из сторон не осталось — прежние конфликты неактуальны.
            keys = []
    out: List[Dict[str, str]] = []
    for key in keys:
        if not key or key in resolved_keys:
            continue
        if key not in [item["key"] for item in out]:
            out.append({"key": key})
    return out


def normalize_analysis(raw: Any) -> Dict[str, Any]:
    """Приводит разбор (из файла, от модели или снимок для фронта) к одному виду.

    Принимает и «сырой» ответ модели (explanation/suggestions), и снимок для
    интерфейса (message/explanation/suggestions), и русские ключи — журнал чата
    хранит снимок, который затем снова нормализуется при чтении.
    """
    if not isinstance(raw, dict):
        return empty_analysis()
    if "message" in raw or "варианты" in raw:
        raw = {
            "verdict": raw.get("verdict") or raw.get("вердикт"),
            "kind": raw.get("kind"),
            "explanation": raw.get("explanation") or raw.get("message"),
            "suggestions": raw.get("suggestions") or raw.get("варианты") or [],
            "suggestions_checked": raw.get("suggestions_checked"),
            "overridden": raw.get("overridden") or raw.get("недействующие"),
            "signature": raw.get("signature"),
            "rules_signature": raw.get("rules_signature"),
            "request": raw.get("request"),
        }
    verdict = str(raw.get("verdict") or "").strip().lower()
    if verdict == "conflict":
        # Прежний вердикт «противоречие правил» = нарушение правила проекта:
        # правило проекта всегда главнее, выбора между правилами нет.
        verdict = COMPLIANCE_VIOLATION
    if verdict not in COMPLIANCE_VERDICTS:
        verdict = ""
    kind = str(raw.get("kind") or "").strip().lower()
    if kind == "conflict" or kind not in COMPLIANCE_VERDICTS:
        # Прежний вид «противоречие правил» больше не существует: приоритет
        # всегда у правила проекта, поэтому это обычное нарушение.
        kind = verdict
    suggestions: List[Dict[str, Any]] = []
    for item in (raw.get("suggestions") or [])[:MAX_SUGGESTIONS]:
        if not isinstance(item, dict):
            continue
        title = _one_line(item.get("title"))[:TITLE_LIMIT]
        send = _one_line(item.get("send"))[:TEXT_LIMIT]
        if not title or not send:
            # Вариант без готового текста запроса бесполезен: пользователь не
            # сможет им воспользоваться (выбора «какое правило главнее» больше нет).
            continue
        suggestions.append({
            "title": title,
            "details": _one_line(item.get("details"))[:DETAIL_LIMIT],
            "send": send,
            "kind": "suggestion",
        })
    checked_raw = raw.get("suggestions_checked")
    if isinstance(checked_raw, bool):
        checked = checked_raw
    else:
        # Отметки проверки нет (старые записи журнала): пустой список проверять
        # нечего — он «проверен»; непустой без отметки совместимым НЕ считается.
        checked = not suggestions
    return {
        "verdict": verdict,
        "kind": kind,
        "explanation": _one_line(raw.get("explanation"))[:EXPLANATION_LIMIT],
        "suggestions": suggestions,
        "suggestions_checked": checked,
        "overridden": _clean_numbers(raw.get("overridden")),
        "signature": str(raw.get("signature") or "")[:64],
        "rules_signature": str(raw.get("rules_signature") or "")[:64],
        "request": str(raw.get("request") or "")[:_REQUEST_LIMIT],
    }


def _clean_numbers(raw: Any) -> List[int]:
    """Номера правил (1, 2, 3…) из ответа модели: строки тоже принимаем.

    Нужны для поля «недействующие»: номера правил ТЕКУЩЕЙ задачи, которые
    противоречат правилам проекта (см. ANALYSIS_PROMPT). Всё,
    что не похоже на номер, отбрасываем: по этим номерам строится блок правил,
    который уходит модели в каждом запросе.
    """
    out: List[int] = []
    for item in (raw if isinstance(raw, (list, tuple)) else []):
        try:
            number = int(str(item).strip())
        except (TypeError, ValueError):
            continue
        if number > 0 and number not in out:
            out.append(number)
    return out[:MAX_OVERRIDDEN]


def blocks(analysis: Any) -> bool:
    """True — разбор запрещает работать: запрос нарушает правило.

    Противоречие правила задачи правилу проекта — тоже нарушение (действует
    правило проекта), поэтому выбора «какое правило главнее» не предлагаем:
    агент отказывается от запроса и даёт альтернативы, укладывающиеся в правила.
    """
    return normalize_analysis(analysis)["verdict"] == COMPLIANCE_VIOLATION


def _one_line(value: Any) -> str:
    """Текст в одну строку (модель любит переносы и лишние пробелы)."""
    return re.sub(r"\s+", " ", str(value or "")).strip()


# ---------------------------------------------------------------------------
# Запрос и разбор ответа модели
# ---------------------------------------------------------------------------
def build_analysis_query(request: str, snapshot_data: Any) -> str:
    """Текст запроса к модели: запрос пользователя + правила обеих областей."""
    data = normalize(snapshot_data)
    lines: List[str] = [f"ЗАПРОС ПОЛЬЗОВАТЕЛЯ:\n{str(request or '').strip()}", ""]
    if data["project"]:
        lines.append("ИНВАРИАНТЫ ПРОЕКТА (нарушать нельзя):")
        lines.extend(f"{i}) {entry['text']}" for i, entry in enumerate(data["project"], 1))
        lines.append("")
    if data["task"]:
        lines.append("ИНВАРИАНТЫ ТЕКУЩЕЙ ЗАДАЧИ (нарушать нельзя):")
        lines.extend(f"{i}) {entry['text']}" for i, entry in enumerate(data["task"], 1))
        lines.append("")
    if data["exceptions"]:
        project_text = text_by_id(data["project"])
        task_text = text_by_id(data["task"])
        lines.append(
            "УТВЕРЖДЁННЫЕ ИСКЛЮЧЕНИЯ (пользователь решил, что в этой задаче "
            "действует правило задачи):")
        for i, item in enumerate(data["exceptions"], 1):
            lines.append(
                f"{i}) «{task_text.get(str(item.get('task_id')), '')}» ВМЕСТО "
                f"«{project_text.get(str(item.get('project_id')), '')}»")
        lines.append("")
    lines.append(
        "Верни JSON с вердиктом и вариантами решения (см. инструкцию). "
        "Если запрос нарушает инвариант — ни один вариант не должен его нарушать."
    )
    return "\n".join(lines)


def parse_analysis(content: str, request: str = "") -> Dict[str, Any]:
    """Разбирает ответ модели в разбор (см. normalize_analysis).

    Понимает и английские ключи (verdict/explanation/suggestions), и русские
    (вердикт/объяснение/варианты). Неразобранный ответ — пустой разбор: без
    вердикта агент просто продолжает работу как раньше (нарушение НЕ выдумываем).
    """
    payload = _load_object(content)
    if not isinstance(payload, dict):
        return empty_analysis()
    raw = payload
    if "вердикт" in payload or "варианты" in payload:
        raw = {
            "verdict": payload.get("вердикт"),
            "explanation": payload.get("объяснение") or payload.get("пояснение"),
            "suggestions": payload.get("варианты") or [],
            "overridden": payload.get("недействующие") or payload.get("overridden"),
        }
    verdict = _one_line(raw.get("verdict")).lower()
    if verdict == "conflict":
        # Модель может ответить старым вердиктом «противоречие правил»: это
        # нарушение правила проекта (правило проекта всегда главнее).
        verdict = COMPLIANCE_VIOLATION
    if verdict not in COMPLIANCE_VERDICTS:
        return empty_analysis()
    items = raw.get("suggestions")
    suggestions: List[Dict[str, Any]] = []
    for item in (items if isinstance(items, list) else []):
        if not isinstance(item, dict):
            continue
        suggestions.append({
            "title": item.get("title") or item.get("заголовок"),
            "details": item.get("details") or item.get("пояснение") or item.get("детали"),
            "send": item.get("send") or item.get("запрос"),
            "resolve": item.get("resolve") or item.get("приоритет"),
        })
    return normalize_analysis({
        "verdict": verdict,
        "kind": verdict,
        "explanation": raw.get("explanation") or raw.get("объяснение") or raw.get("why"),
        "suggestions": suggestions,
        "overridden": raw.get("overridden") or raw.get("недействующие"),
        "request": request,
    })


async def _ask_analysis(system_prompt: str, query: str,
                        call: LlmCall) -> Optional[Dict[str, Any]]:
    """Один служебный вызов арбитра: None — вызова/ответа нет (сбой не выдумывает вердикт).

    Разбор запроса и повторный запрос вариантов — один и тот же служебный вызов
    (разный текст запроса), поэтому собираем его в одном месте.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    try:
        content, _metrics = await call(
            user_text="",
            messages=messages,
            response_format="free",
            max_tokens=ANALYSIS_MAX_TOKENS,
            stop=None,
            system_prompt=None,
            temperature=None,
            model=None,
            disable_thinking=True,
            timeout=ANALYSIS_TIMEOUT,
        )
    except Exception:  # noqa: BLE001 — сбой разбора не должен ломать запрос
        logger.warning("Разбор инвариантов: вызов модели не удался", exc_info=True)
        return None
    if not content:
        return None
    return parse_analysis(content)


def _rules_lines(data: Dict[str, Any], with_exceptions: bool = False) -> List[str]:
    """Строки «правила проекта / действующие правила задачи» для запроса к модели.

    Недействующие правила задачи (противоречащие правилам проекта) в проверку НЕ
    уходят: они не действуют, и требовать их выполнения от варианта или шага
    плана нельзя (иначе совместимых решений не остаётся вовсе).
    """
    lines: List[str] = []
    if data["project"]:
        lines.append("ИНВАРИАНТЫ ПРОЕКТА (нарушать нельзя, всегда главнее):")
        lines.extend(f"{i}) {entry['text']}" for i, entry in enumerate(data["project"], 1))
        lines.append("")
    dead = {_one_line(text) for text in data.get("overridden") or []}
    task_live = [entry for entry in data["task"] if entry["text"] not in dead]
    if task_live:
        lines.append("ИНВАРИАНТЫ ТЕКУЩЕЙ ЗАДАЧИ (нарушать нельзя):")
        lines.extend(f"{i}) {entry['text']}" for i, entry in enumerate(task_live, 1))
        lines.append("")
    if with_exceptions and data["exceptions"]:
        project_text = text_by_id(data["project"])
        task_text = text_by_id(data["task"])
        lines.append("УТВЕРЖДЁННЫЕ ИСКЛЮЧЕНИЯ (в этой задаче действует правило задачи):")
        for i, item in enumerate(data["exceptions"], 1):
            lines.append(
                f"{i}) «{task_text.get(str(item.get('task_id')), '')}» ВМЕСТО "
                f"«{project_text.get(str(item.get('project_id')), '')}»")
        lines.append("")
    return lines


def build_suggestions_query(suggestions: Any, snapshot_data: Any) -> str:
    """Текст запроса на ПРОВЕРКУ вариантов: те же правила + нумерованные тексты."""
    lines = _rules_lines(normalize(snapshot_data), with_exceptions=True)
    lines.append("ВАРИАНТЫ (проверь КАЖДЫЙ по правилам выше):")
    items = [item for item in (suggestions or []) if isinstance(item, dict)]
    for i, item in enumerate(items, 1):
        lines.append(f"{i}) {_one_line(item.get('send'))}")
    lines.append("")
    lines.append("Верни JSON с вердиктом по каждому варианту (см. инструкцию).")
    return "\n".join(lines)


def build_steps_query(steps: Any, snapshot_data: Any) -> str:
    """Текст запроса на ПРОВЕРКУ шагов плана (код-гейт плана)."""
    lines = _rules_lines(normalize(snapshot_data), with_exceptions=True)
    lines.append("ШАГИ ПЛАНА (проверь КАЖДЫЙ по правилам выше):")
    for i, step in enumerate([str(item) for item in (steps or [])], 1):
        lines.append(f"{i}) {_one_line(step)}")
    lines.append("")
    lines.append("Верни JSON с вердиктом по каждому шагу (см. инструкцию).")
    return "\n".join(lines)


def build_retry_query(request: str, snapshot_data: Any, rejected: Any) -> str:
    """Повторный запрос вариантов: отклонённые проверкой повторять НЕЛЬЗЯ."""
    lines = [build_analysis_query(request, snapshot_data), ""]
    items = [item for item in (rejected or []) if isinstance(item, dict)]
    if items:
        lines.append(
            "ПРЕДЫДУЩИЕ ВАРИАНТЫ ПРОВЕРЕНО И ОТКЛОНЕНО — они нарушают правила, "
            "повторять их нельзя:")
        lines.extend(f"{i}) {_one_line(item.get('send'))}"
                     for i, item in enumerate(items, 1))
    else:
        lines.append("ПРЕДЫДУЩИЙ ОТВЕТ НЕ ДАЛ ПРИГОДНЫХ ВАРИАНТОВ.")
    lines.append(
        f"Дай ДРУГИЕ варианты (от {MIN_SUGGESTIONS} до {MAX_SUGGESTIONS}) — ни один "
        "из них не нарушает правила. Верни тот же JSON с вердиктом violation."
    )
    return "\n".join(lines)


def parse_suggestion_verdicts(content: str, count: int) -> Optional[List[bool]]:
    """Вердикты проверки вариантов: True — вариант правил НЕ нарушает.

    None — в ответе НЕТ ни одного вердикта по вариантам (нераспознанный ответ,
    болтовня вместо JSON): это СБОЙ проверки, а не «нарушений нет». Вердикт по
    конкретному варианту, которого в ответе нет, — False: «не подтверждено
    проверкой» ≠ «совместимо».
    """
    payload = _load_object(content)
    if not isinstance(payload, dict) or not payload:
        return None
    verdicts: List[bool] = []
    found_any = False
    for number in range(1, int(count) + 1):
        item = payload.get(str(number), payload.get(number))
        if item is not None:
            found_any = True
        if isinstance(item, str):
            item = {"вердикт": item}
        if not isinstance(item, dict):
            item = {}
        verdict = _one_line(item.get("вердикт") or item.get("verdict")).lower()
        verdicts.append(verdict == COMPLIANCE_CLEAR)
    if not found_any:
        return None
    return verdicts


async def _check_texts(texts: Sequence[str], query: str, system_prompt: str,
                       call: LlmCall, max_tokens: int, timeout: float,
                       what: str) -> Optional[List[int]]:
    """Общий шаг проверки: номера текстов, признанных НЕ нарушающими правила.

    None — проверка не удалась (вызова нет / ответ не разобран): и «нарушений
    нет», и «вариант совместим» — разные вещи, сбой проверки не должен выглядеть
    как совместимость (как у пар правил, §5.10).
    """
    if not texts:
        return []
    try:
        content, _metrics = await call(
            user_text="",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            response_format="free",
            max_tokens=max_tokens,
            stop=None,
            system_prompt=None,
            temperature=None,
            model=None,
            disable_thinking=True,
            timeout=timeout,
        )
    except Exception:  # noqa: BLE001 — сбой проверки не должен ломать запрос
        logger.warning("Проверка инвариантов (%s): вызов модели не удался",
                       what, exc_info=True)
        return None
    verdicts = parse_suggestion_verdicts(content or "", len(texts))
    if verdicts is None:
        return None
    return [index for index, ok in enumerate(verdicts) if ok]


async def check_suggestions(suggestions: Any, snapshot_data: Any,
                            call: LlmCall) -> Optional[List[int]]:
    """Номера вариантов, которые проверка признала НЕ нарушающими правила."""
    items = [item for item in (suggestions or []) if isinstance(item, dict)]
    return await _check_texts(
        [_one_line(item.get("send")) for item in items],
        build_suggestions_query(items, snapshot_data),
        SUGGESTIONS_PROMPT, call, SUGGESTIONS_MAX_TOKENS, SUGGESTIONS_TIMEOUT,
        what="варианты")


async def check_steps(steps: Any, snapshot_data: Any,
                      call: LlmCall) -> Optional[List[int]]:
    """Номера ШАГОВ ПЛАНА, которые проверка признала НЕ нарушающими правила.

    Код-гейт плана: промпт-блок правил лишь ПРОСИТ планировщика не нарушать
    правила, поэтому шаги уходят арбитру отдельным служебным вызовом. None —
    проверка не удалась (шаги не подтверждены, см. `_plan_gate` в chat.py).
    """
    texts = [_one_line(step) for step in (steps or [])]
    return await _check_texts(
        texts, build_steps_query(texts, snapshot_data),
        PLAN_PROMPT, call, PLAN_MAX_TOKENS, PLAN_TIMEOUT, what="шаги плана")


async def _verified_analysis(analysis: Dict[str, Any], request: str,
                             snapshot_data: Any, call: LlmCall) -> Dict[str, Any]:
    """Оставляет только ПРОВЕРЕННЫЕ варианты-альтернативы (все остальные отбрасывает).

    Варианты приходят от модели текстом, а правила нарушать нельзя: разбор лишь
    ПРОСИТ их не нарушать, поэтому каждый вариант уходит на проверку арбитру.
    Если после проверки вариантов меньше нормы — одна попытка попросить ДРУГИЕ
    (отклонённые перечисляются в запросе). Пока вариантов нет, пользователю
    обещать их нельзя: сообщение отказа скажет об этом прямо (см. chat.py).
    """
    data = normalize_analysis(analysis)
    items = list(data["suggestions"])
    keep = await check_suggestions(items, snapshot_data, call)
    checked = keep is not None
    if keep is None:
        variants, rejected = [], items
    else:
        kept = set(keep)
        variants = [item for index, item in enumerate(items) if index in kept]
        rejected = [item for index, item in enumerate(items) if index not in kept]
    seen = {item["send"] for item in variants}
    for _ in range(SUGGESTION_RETRIES):
        if len(variants) >= MIN_SUGGESTIONS:
            break
        retry = await _ask_analysis(
            ANALYSIS_PROMPT, build_retry_query(request, snapshot_data, rejected), call)
        if retry is None or retry["verdict"] != COMPLIANCE_VIOLATION:
            break
        candidates = list(retry["suggestions"])
        if not candidates:
            break
        keep_retry = await check_suggestions(candidates, snapshot_data, call)
        if keep_retry is None:
            break
        checked = True
        for index in keep_retry:
            item = candidates[index]
            if item["send"] in seen:
                continue
            seen.add(item["send"])
            variants.append(item)
    return {
        **data,
        "suggestions": variants[:MAX_SUGGESTIONS],
        "suggestions_checked": checked,
    }


async def analyze(request: str, snapshot_data: Any, call: LlmCall) -> Dict[str, Any]:
    """Разбирает запрос на соответствие инвариантам служебными вызовами LLM.

    Возвращает разбор (см. normalize_analysis). Сбой вызова или неразобранный
    ответ — ПУСТОЙ разбор: агент работает как раньше (нарушение не выдумываем,
    но и не подтверждаем; в интерфейсе видно, что проверка не дала вердикта).

    Вызовов может быть несколько: вердикт по запросу, а при нарушении — ещё
    ПРОВЕРКА вариантов и (если проверенных меньше нормы) повторный запрос
    вариантов. Все они служебные; расход считает вызывающая сторона
    (см. Agent.check_invariants).
    """
    signature = analysis_signature(request, snapshot_data)
    analysis = await _ask_analysis(
        ANALYSIS_PROMPT, build_analysis_query(request, snapshot_data), call)
    if analysis is None:
        return empty_analysis()
    analysis["signature"] = signature
    analysis["rules_signature"] = rules_signature(snapshot_data)
    analysis["request"] = str(request or "")[:_REQUEST_LIMIT]
    if analysis["verdict"] != COMPLIANCE_VIOLATION:
        # Нарушений нет — вариантов нет и проверять нечего.
        return analysis
    return await _verified_analysis(analysis, request, snapshot_data, call)

_REASON_KEYS = ("причина", "reason", "comment", "комментарий", "why")
_VERDICT_KEYS = ("вердикт", "verdict", "result", "результат", "status")


# ---------------------------------------------------------------------------
# Пары инвариантов и ключи конфликтов
# ---------------------------------------------------------------------------
def text_of(item: Any) -> str:
    """Текст инварианта из записи файла (словарь) или голой строки."""
    if isinstance(item, dict):
        return str(item.get("text") or "").strip()
    return str(item or "").strip()


def invariants_texts(items: Any) -> List[str]:
    """Тексты инвариантов по порядку (пустые отбрасываются)."""
    return [text for text in (text_of(item) for item in (items or [])) if text]


def id_text_pairs(items: Any) -> List[Tuple[str, str]]:
    """Пары «id, текст» — по ним считается ключ конфликта."""
    out: List[Tuple[str, str]] = []
    for item in (items or []):
        if not isinstance(item, dict):
            continue
        entry_id = str(item.get("id") or "").strip()
        text = text_of(item)
        if entry_id and text:
            out.append((entry_id, text))
    return out


def pair_key(project_id: str, task_id: str) -> str:
    """Ключ пары «инвариант проекта — инвариант задачи».

    Ключ упорядочен: проект всегда первый. Он же — id конфликта в интерфейсе
    (кнопки «главнее проект» / «главнее задача») и ключ исключения в контексте.
    """
    return f"{project_id}|{task_id}"


def split_key(key: str) -> Tuple[str, str]:
    """Обратная операция: ключ -> (id проекта, id задачи)."""
    project_id, _, task_id = str(key or "").partition("|")
    return project_id.strip(), task_id.strip()


def _pair_key_of(item: Any) -> str:
    """Ключ пары из записи проверки."""
    if not isinstance(item, dict):
        return ""
    project_id = str(item.get("project_id") or "").strip()
    task_id = str(item.get("task_id") or "").strip()
    if not project_id or not task_id:
        return ""
    return pair_key(project_id, task_id)


def pairs(project_invariants: Any, task_invariants: Any) -> List[Dict[str, str]]:
    """Все пары «инвариант проекта × инвариант задачи», которые надо проверить.

    Пустой любой из сторон — пар нет (проверять нечего, вызова LLM не будет).
    """
    out: List[Dict[str, str]] = []
    for project_id, project_text in id_text_pairs(project_invariants):
        for task_id, task_text in id_text_pairs(task_invariants):
            out.append({
                "key": pair_key(project_id, task_id),
                "project_id": project_id,
                "task_id": task_id,
                "project_text": project_text,
                "task_text": task_text,
            })
    return out


# ---------------------------------------------------------------------------
# Записи проверки: список конфликтов в диалоге
# ---------------------------------------------------------------------------
def clean_records(raw: Any) -> List[Dict[str, Any]]:
    """Приводит список проверок к безопасному виду (битые записи отбрасываются).

    Запись: {"key", "project_id", "task_id", "verdict": conflict|clear,
    "reason", "resolved": bool, "winner": project|task|"", "checked": метка}.
    """
    out: List[Dict[str, Any]] = []
    seen = set()
    for item in (raw if isinstance(raw, list) else []):
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip() or _pair_key_of(item)
        if not key or key in seen:
            continue
        verdict = str(item.get("verdict") or "").strip().lower()
        if verdict not in VERDICTS:
            continue
        winner = str(item.get("winner") or "").strip().lower()
        seen.add(key)
        out.append({
            "key": key,
            "project_id": str(item.get("project_id") or "").strip(),
            "task_id": str(item.get("task_id") or "").strip(),
            "verdict": verdict,
            "reason": str(item.get("reason") or "").strip()[:REASON_LIMIT],
            "resolved": item.get("resolved") is True,
            "winner": winner if winner in WINNERS else "",
            "checked": str(item.get("checked") or "").strip(),
        })
    return out


def apply_results(records: Any, pairs_now: Sequence[Dict[str, str]],
                  results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Накладывает свежие вердикты на прежние проверки.

    `results` — вердикты ТОЛЬКО по прошедшим проверку парам (ключ пары ->
    {"verdict", "reason"}). Прежние вердикты сохраняются: проверка идёт не по
    всем парам сразу, а по «непроверенным и устаревшим» (см. needs_check).
    Решение пользователя (`resolved`/`winner`) при перепроверке НЕ теряется.
    """
    current = {item["key"] for item in pairs_now}
    merged: Dict[str, Dict[str, Any]] = {}
    for record in clean_records(records):
        if record["key"] not in current:
            continue  # инвариант удалён — запись больше не нужна
        if record["key"] in results:
            verdict = results[record["key"]]
            record = dict(record)
            record["verdict"] = verdict["verdict"]
            record["reason"] = str(verdict.get("reason") or "")[:REASON_LIMIT]
            # Метка «какие тексты правил проверялись» — по ней видно, что вердикт
            # ещё актуален (правка инварианта делает его устаревшим).
            record["checked"] = str(verdict.get("checked") or "")
        merged[record["key"]] = record
    for pair in pairs_now:
        if pair["key"] in merged or pair["key"] not in results:
            continue
        verdict = results[pair["key"]]
        # Пары принимаем и в «лёгком» виде — только ключ (например, решение по
        # конфликту приходит из разбора): стороны достаются из самого ключа.
        project_id = str(pair.get("project_id") or "") or split_key(pair["key"])[0]
        task_id = str(pair.get("task_id") or "") or split_key(pair["key"])[1]
        merged[pair["key"]] = {
            "key": pair["key"],
            "project_id": project_id,
            "task_id": task_id,
            "verdict": verdict["verdict"],
            "reason": str(verdict.get("reason") or "")[:REASON_LIMIT],
            "resolved": False,
            "winner": "",
            "checked": str(verdict.get("checked") or ""),
        }
    return [merged[key] for key in sorted(merged)]


def exceptions_pairs(records: Any) -> List[Dict[str, str]]:
    """Пары, где решение пользователя оставило ГЛАВНЫМ инвариант задачи.

    Это исключения из правил проекта для конкретной задачи: они уходят модели
    отдельным списком, чтобы агент не считал их нарушением инварианта проекта.
    Решение пользователя приходит из диалога (разбор запроса показал
    противоречие, пользователь выбрал приоритет) и лежит либо записью проверки
    (`conflicts`), либо простой записью решения (`exceptions` в диалоге).
    """
    out: List[Dict[str, str]] = []
    for record in clean_exceptions(records):
        if record["winner"] != WINNER_TASK:
            continue
        out.append({"key": record["key"],
                    "project_id": record["project_id"],
                    "task_id": record["task_id"]})
    return out


def clean_exceptions(raw: Any) -> List[Dict[str, Any]]:
    """Решения пользователя по противоречиям правил (без записей проверки).

    Запись: {"key", "project_id", "task_id", "winner"}. Пишется, когда в диалоге
    пользователь выбрал, ЧЬЁ правило главнее. Проверок правил при добавлении
    инвариантов НЕТ — решение всегда приходит из диалога.
    """
    out: List[Dict[str, Any]] = []
    seen = set()
    for item in (raw if isinstance(raw, list) else []):
        if not isinstance(item, dict):
            continue
        winner = str(item.get("winner") or "").strip().lower()
        if winner not in WINNERS:
            continue
        key = str(item.get("key") or "").strip()
        if not key:
            project_id, task_id = (str(item.get("project_id") or "").strip(),
                                   str(item.get("task_id") or "").strip())
            if not project_id or not task_id:
                continue
            key = pair_key(project_id, task_id)
        if key in seen:
            continue
        seen.add(key)
        project_id, task_id = split_key(key)
        out.append({"key": key, "project_id": project_id, "task_id": task_id,
                    "winner": winner})
    return out


def merge_exceptions(records: Any, raw: Any) -> List[Dict[str, Any]]:
    """Сливает решения из записей проверки и из простых записей решений."""
    out = {item["key"]: item for item in clean_exceptions(raw)}
    for record in clean_records(records):
        if not record["resolved"] or record["winner"] not in WINNERS:
            continue
        out[record["key"]] = {"key": record["key"],
                              "project_id": record["project_id"],
                              "task_id": record["task_id"],
                              "winner": record["winner"]}
    return [out[key] for key in sorted(out)]


def unresolved(records: Any) -> List[Dict[str, Any]]:
    """Конфликты, по которым пользователь ещё не принял решение."""
    return [record for record in clean_records(records)
            if record["verdict"] == VERDICT_CONFLICT and not record["resolved"]]


def has_conflict(records: Any) -> bool:
    """Есть ли неразрешённое противоречие (агент ждёт решения пользователя)."""
    return bool(unresolved(records))


def needs_check(records: Any, pairs_now: Sequence[Dict[str, str]],
                unchecked: Any = None) -> List[Dict[str, str]]:
    """Пары, которые надо проверить: новые и изменившиеся.

    Проверенной считается пара, запись о которой есть и в ней лежит ТОТ ЖЕ текст
    правил (сравнение по нормализованному тексту): если пользователь отредактировал
    инвариант, вердикт для старого текста недействителен. Пары из `unchecked`
    (прошлая проверка не удалась) перепроверяются всегда.
    """
    by_key = {record["key"]: record for record in clean_records(records)}
    pending = {str(key) for key in (unchecked or [])}
    out: List[Dict[str, str]] = []
    for pair in pairs_now:
        if pair["key"] in pending:
            out.append(pair)
            continue
        record = by_key.get(pair["key"])
        if record is None or record["checked"] != _pair_signature(pair):
            out.append(pair)
    return out


def _pair_signature(pair: Dict[str, str]) -> str:
    """Метка «какие тексты правил проверялись» — хранится в записи проверки."""
    return _normalize(f"{pair['project_text']}\n{pair['task_text']}")[:200]


def _normalize(text: str) -> str:
    """Нормализация текста для сравнения (регистр и пробелы не важны)."""
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def mark(results: Dict[str, Dict[str, Any]],
         pairs_now: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
    """Добавляет к вердиктам метку «какие тексты правил проверялись».

    Метка нужна, чтобы правка текста инварианта делала вердикт устаревшим: без
    неё правило считалось бы проверенным по старому тексту (см. needs_check).
    """
    marks = {pair["key"]: _pair_signature(pair) for pair in pairs_now}
    for key, verdict in results.items():
        if key in marks:
            verdict["checked"] = marks[key]
    return results


def resolve_pair(records: Any, key: str, winner: str) -> List[Dict[str, Any]]:
    """Фиксирует решение пользователя по конфликту.

    winner="project" — главнее правило проекта (инвариант задачи ослабляется для
    этой задачи), winner="task" — главнее правило задачи (для этой задачи
    действует исключение из правила проекта). Возвращает новый список проверок;
    ключа нет или вердикт не «конфликт» — список не меняется.
    """
    winner = str(winner or "").strip().lower()
    if winner not in WINNERS:
        return clean_records(records)
    out: List[Dict[str, Any]] = []
    for record in clean_records(records):
        if record["key"] == key and record["verdict"] == VERDICT_CONFLICT:
            record = dict(record)
            record["resolved"] = True
            record["winner"] = winner
        out.append(record)
    return out


# ---------------------------------------------------------------------------
# Результат проверки для фронтенда
# ---------------------------------------------------------------------------
def payload(pairs_now: Sequence[Dict[str, str]], records: Any,
            unchecked: Any = None) -> List[Dict[str, Any]]:
    """Список проверок для интерфейса: конфликты, решения и «не проверено».

    В список попадают ТОЛЬКО конфликты и непроверенные пары: вердикт «clear»
    пользователю показывать нечего, а вот «проверка не выполнена» — важно (иначе
    отсутствие конфликта выглядело бы как подтверждённая совместимость).
    """
    by_key = {record["key"]: record for record in clean_records(records)}
    pending = {str(key) for key in (unchecked or [])}
    for pair in pairs_now:
        record = by_key.get(pair["key"])
        if pair["key"] in pending or record is None \
                or record["checked"] != _pair_signature(pair):
            pending.add(pair["key"])
    out: List[Dict[str, Any]] = []
    for pair in pairs_now:
        record = by_key.get(pair["key"])
        if record is not None and record["verdict"] == VERDICT_CONFLICT:
            out.append(dict(record, project_text=pair["project_text"],
                            task_text=pair["task_text"], checked_ok=True))
        elif pair["key"] in pending:
            out.append({
                "key": pair["key"],
                "project_id": pair["project_id"],
                "task_id": pair["task_id"],
                "verdict": "",
                "reason": "",
                "resolved": False,
                "winner": "",
                "checked": "",
                "checked_ok": False,
                "project_text": pair["project_text"],
                "task_text": pair["task_text"],
            })
    return out


# ---------------------------------------------------------------------------
# Блок системного промпта: правила уходят модели в каждом запросе
# ---------------------------------------------------------------------------
INVARIANTS_HEADER = (
    "ИНВАРИАНТЫ (правила, которые НАРУШАТЬ НЕЛЬЗЯ) — учитывай их явно в "
    "рассуждениях и проверяй по ним каждое предлагаемое решение. ГЛАВНОЕ ПРАВИЛО "
    "ПРИОРИТЕТА: правило ПРОЕКТА всегда главнее правила задачи. Если правило "
    "задачи противоречит правилу проекта, правило задачи НЕ ДЕЙСТВУЕТ: работай по "
    "правилу проекта и НЕ переноси требование такого правила задачи в план, шаги "
    "и ответы (в том числе когда оно сформулировано как «только так» или "
    "«обязательно»)."
)


def normalize(raw: Any) -> Dict[str, Any]:
    """Приводит снимок инвариантов (от веб-слоя) к безопасному виду.

    Снимок, который агент получает на каждый запрос:

        {"project": [{"id", "text", ...}, ...],   — правила ПРОЕКТА
         "task": [{"id", "text", ...}, ...],      — правила ЗАДАЧИ-диалога
         "exceptions": [{"project_id", "task_id"}, ...], — утверждённые исключения
         "overridden": ["текст правила задачи", ...]}
            — правила задачи, НЕ ДЕЙСТВУЮЩИЕ из-за противоречия правилам проекта
              (их номера вернул разбор запроса, см. `overridden_texts`)
    """
    if not isinstance(raw, dict):
        return snapshot()
    return {
        "project": _clean_items(raw.get("project")),
        "task": _clean_items(raw.get("task")),
        "exceptions": [item for item in (raw.get("exceptions") or [])
                       if isinstance(item, dict)],
        "overridden": [str(text) for text in (raw.get("overridden") or [])
                       if str(text).strip()],
    }


def overridden_texts(task_invariants: Any, numbers: Any) -> List[str]:
    """Тексты правил задачи по номерам из разбора (1 — первое правило задачи).

    Номера приходят от модели («недействующие»), а в блок правил уходят ТЕКСТЫ:
    модель должна видеть, какое именно правило задачи не действует, а не номер.
    Номер вне списка просто игнорируется — правило остаётся действующим.
    """
    pairs = id_text_pairs(task_invariants)
    out: List[str] = []
    for number in _clean_numbers(numbers):
        if 1 <= number <= len(pairs):
            text = pairs[number - 1][1]
            if text not in out:
                out.append(text)
    return out


def snapshot() -> Dict[str, Any]:
    """Пустой снимок инвариантов (правил нет — блока в контексте не будет)."""
    return {"project": [], "task": [], "exceptions": [], "overridden": []}


def _clean_items(raw: Any) -> List[Dict[str, str]]:
    """Инварианты снимка: словари с id/text (голые строки тоже принимаем)."""
    out: List[Dict[str, str]] = []
    for index, item in enumerate(raw if isinstance(raw, list) else [], 1):
        if isinstance(item, dict):
            entry_id = str(item.get("id") or "").strip() or f"i-{index}"
            text = str(item.get("text") or "").strip()
        else:
            entry_id, text = f"i-{index}", str(item or "").strip()
        if text:
            out.append({"id": entry_id, "text": text})
    return out


def block(snapshot_data: Any) -> str:
    """Системный блок инвариантов по снимку (пусто — блока нет)."""
    data = normalize(snapshot_data)
    return invariants_block(data["project"], data["task"], data["exceptions"],
                            data["overridden"])


def counts(snapshot_data: Any) -> Dict[str, int]:
    """Число правил по областям (для debug-строки и интерфейса)."""
    data = normalize(snapshot_data)
    return {
        "project": len(data["project"]),
        "task": len(data["task"]),
        "exceptions": len(data["exceptions"]),
    }


def has_rules(snapshot_data: Any) -> bool:
    """Есть ли хоть одно правило (или утверждённое исключение)."""
    return any(counts(snapshot_data).values())


def rules_note(snapshot_data: Any) -> str:
    """Строка debug-чата: сколько правил уходит в модель системным блоком."""
    numbers = counts(snapshot_data)
    parts = []
    if numbers["project"]:
        parts.append(f"проекта: {numbers['project']}")
    if numbers["task"]:
        parts.append(f"текущей задачи: {numbers['task']}")
    if numbers["exceptions"]:
        parts.append(f"утверждённых исключений: {numbers['exceptions']}")
    body = ", ".join(parts) if parts else "0"
    return (
        f"инварианты — {body}. Это правила, которые нарушать нельзя: они уходят "
        "в модель отдельным системным блоком в каждом запросе (стратегиям "
        "контекста не подчиняются), агент обязан учитывать их в рассуждениях, "
        "не предлагать нарушающих решений и сообщать о противоречии."
    )


def text_by_id(items: Any) -> Dict[str, str]:
    """Тексты инвариантов по их id (для расшифровки исключений в промпте)."""
    return {entry_id: text for entry_id, text in id_text_pairs(items)}


def invariants_block(project_invariants: Any, task_invariants: Any,
                     exceptions: Sequence[Dict[str, str]] = (),
                     overridden: Sequence[str] = ()) -> str:
    """Системный блок инвариантов для контекста агента.

    Порядок: правила проекта (действуют во всех задачах проекта), затем
    действующие исключения (решения пользователя «главнее инвариант задачи»),
    затем правила ТЕКУЩЕЙ задачи и отдельно — правила задачи, НЕ ДЕЙСТВУЮЩИЕ
    из-за противоречия правилам проекта (`overridden`, их требование выполнять
    нельзя: действует правило проекта). Блок идёт ОТДЕЛЬНО от диалога:
    инварианты не записываются в переписку и не подчиняются стратегиям
    контекста, поэтому модель видит их в каждом запросе.

    Пусто (нет ни правил, ни исключений) — пустая строка, блока в контексте нет.
    """
    project_pairs = id_text_pairs(project_invariants)
    task_pairs = id_text_pairs(task_invariants)
    exceptions = [item for item in (exceptions or []) if isinstance(item, dict)]
    dead = [_one_line(text) for text in (overridden or []) if _one_line(text)]
    if not project_pairs and not task_pairs and not exceptions:
        return ""
    project_text = text_by_id(project_invariants)
    task_text = text_by_id(task_invariants)
    # Правило задачи, признанное противоречащим правилу проекта, уходит в раздел
    # «НЕ ДЕЙСТВУЕТ»: держать его в списке «нарушать нельзя» нельзя — именно
    # из-за этого план строился по требованию задачи (KMP/iOS), а запрет проекта
    # не срабатывал.
    live_pairs = [pair for pair in task_pairs if pair[1] not in dead]
    parts: List[str] = [INVARIANTS_HEADER, ""]
    if project_pairs:
        parts.append("Инварианты проекта (нарушать нельзя, действуют во всех задачах):")
        parts.extend(f"{i}) {text}" for i, (_id, text) in enumerate(project_pairs, 1))
        parts.append("")
    if exceptions:
        parts.append(
            "Исключения, УТВЕРЖДЁННЫЕ пользователем (инвариант задачи главнее "
            "инварианта проекта — в этой задаче действует исключение):")
        for i, item in enumerate(exceptions, 1):
            project_label = project_text.get(str(item.get("project_id") or ""), "")
            task_label = task_text.get(str(item.get("task_id") or ""), "")
            if project_label and task_label:
                parts.append(f"{i}) «{task_label}» ВМЕСТО «{project_label}»")
            elif task_label:
                parts.append(f"{i}) действует инвариант задачи «{task_label}»")
            else:
                parts.append(f"{i}) действует инвариант задачи вместо правила проекта")
        parts.append("")
    if live_pairs:
        parts.append("Инварианты текущей задачи (нарушать нельзя):")
        parts.extend(f"{i}) {text}" for i, (_id, text) in enumerate(live_pairs, 1))
        parts.append("")
    if dead:
        parts.append(
            "Правила текущей задачи, которые НЕ ДЕЙСТВУЮТ (противоречат правилам "
            "проекта — их требование НЕ выполняй, действует правило проекта):")
        parts.extend(f"{i}) {text}" for i, text in enumerate(dead, 1))
        parts.append("")
    parts.append(
        "Правила поведения:\n"
        "1) В рассуждениях опирайся на эти инварианты и не предлагай решений, "
        "которые их нарушают.\n"
        "2) Если запрос или шаг плана требует нарушить инвариант — не выполняй "
        "его: назови конкретный инвариант и предложи вариант, который его "
        "соблюдает.\n"
        "3) Противоречие правил решает ПРАВИЛО ПРОЕКТА: правило задачи, "
        "противоречащее правилу проекта, не применяй и решение у пользователя не "
        "запрашивай — работай по правилу проекта."
    )
    return "\n".join(parts).strip()


def _load_object(content: str) -> Any:
    """Разбирает ответ модели в объект: снимает markdown, чинит обрезанный JSON.

    Пары «номер -> вердикт» приходят словарём, но модель иногда отдаёт их
    списком или оборачивает текст — поэтому пробуем оба варианта и общий репейр
    (json_utils), а не только строгий разбор.
    """
    text = str(content or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        lines = [line for line in text.splitlines()
                 if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    for candidate in (text, json_utils.repair_json(text)):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(data, (dict, list)):
            return data
    wrapped = json_utils.wrap_as_json(text)
    try:
        data = json.loads(wrapped)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, (dict, list)) else None


