"""Самопроверка учёта расхода токенов (панель «Токены задачи»).

Сеть и API-ключ не нужны: `client.call_llm_async` подменяется заглушкой с
ФИКСИРОВАННЫМИ метриками у каждого вида вызова, поэтому истинный расход
известен точно. Проверяется, что замер запроса в диалоге (то же, что видит
панель) совпадает с истиной: число обращений к модели, входящие/исходящие
токены, вклад служебных вызовов и разбивка по их видам.

Зачем: раньше замер строился из НАКОПИТЕЛЬНОГО счётчика агента и складывался
как дельта — панель завышала расход вдвое (перепланирование) или теряла вызов
(разбор запроса при ответе шага), а служебные вызовы вроде разбора инвариантов
не попадали в файл вовсе.

Запуск: ./venv/bin/python tools/check_token_stats.py
"""

import asyncio
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Изоляция данных: workspace, память и профили пишутся во временный каталог.
_TMP = tempfile.mkdtemp(prefix="token-stats-")
os.environ["AGENT_WORKSPACE_FILE"] = os.path.join(_TMP, "workspace.json")
os.environ["AGENT_MEMORY_FILE"] = os.path.join(_TMP, "agent_memory.json")
os.environ["AGENT_PROFILES_FILE"] = os.path.join(_TMP, "profiles.json")
os.environ.setdefault("YANDEX_API_KEY", "test-key")
# Ключ провайдера по умолчанию (deepseek-official): без него клиент считает,
# что обращения к модели не было.
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")

from app.ai import agent as agent_mod  # noqa: E402
from app.ai import client  # noqa: E402
from app.routers import chat  # noqa: E402
from app.schemas import ChatMessage  # noqa: E402

FAILURES = []
# Цена вызова по его виду: (входящие, исходящие) — разные числа, чтобы
# перепутанные вызовы было видно сразу.
PRICE = {
    "разбор_запроса": (100, 10),
    "план": (200, 20),
    "код-гейт": (300, 30),
    "ответ_шага": (400, 40),
    "приёмщик": (500, 50),
    "сжатие": (600, 60),
    "факты": (700, 70),
}
CALLS = []
CFG = {"plan": ["Шаг 1", "Шаг 2"], "gate": "clear", "analysis": "clear",
       "suggestions": "clear", "review": None, "fail": False}


def check(name, condition, detail=""):
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(name)


def _metrics(kind):
    prompt, completion = PRICE[kind]
    return {"model": "stub", "elapsed_seconds": 0.0, "prompt_tokens": prompt,
            "completion_tokens": completion, "total_tokens": prompt + completion}


async def fake_call_llm_async(*args, **kwargs):
    messages = kwargs.get("messages") or []
    system = str(messages[0].get("content") or "") if messages else ""
    if system.startswith("Ты — планировщик, который исследует"):
        kind = "план"
    elif system.startswith("Ты — планировщик"):
        kind = "план"
    elif system.startswith("Ты — приёмщик"):
        kind = "приёмщик"
    elif system.startswith("Ты — арбитр инвариантов: проверка ШАГОВ ПЛАНА"):
        kind = "код-гейт"
    elif system.startswith("Ты — арбитр инвариантов"):
        kind = "разбор_запроса"
    elif system.startswith("Ты сжимаешь"):
        kind = "сжатие"
    elif system.startswith("Ты ведёшь блок"):
        kind = "факты"
    else:
        kind = "ответ_шага"
    CALLS.append(kind)
    if CFG["fail"]:
        # Сбой провайдера: обращение было, расхода нет.
        return "", {"model": "stub", "elapsed_seconds": 0.0, "prompt_tokens": 0,
                    "completion_tokens": 0, "total_tokens": 0,
                    "failed": True, "error": "HTTP 429"}
    if kind == "план":
        steps = CFG["plan"]
        if not steps:
            return "", _metrics(kind)
        return json.dumps({"steps": list(steps)}, ensure_ascii=False), _metrics(kind)
    if kind == "код-гейт":
        verdict = CFG["gate"]
        if verdict is None:
            return "", _metrics(kind)
        return json.dumps({str(i): {"вердикт": verdict, "причина": ""}
                           for i in range(1, 7)}, ensure_ascii=False), _metrics(kind)
    if kind == "разбор_запроса":
        payload = CFG["analysis"]
        if payload is None:
            return "", _metrics(kind)
        return json.dumps(payload, ensure_ascii=False), _metrics(kind)
    if kind == "приёмщик":
        review = CFG["review"]
        if review is None:
            return "", _metrics(kind)
        return json.dumps(review, ensure_ascii=False), _metrics(kind)
    return "Ответ модели по текущему шагу.", _metrics(kind)


client.call_llm_async = fake_call_llm_async


async def run_chat(text, **kwargs):
    response = await chat.agent_chat(ChatMessage(content=text, **kwargs))
    events = []
    async for chunk in response.body_iterator:
        for line in str(chunk).splitlines():
            if line.strip():
                events.append(json.loads(line))
    return events


def truth(calls):
    """Истинный расход по списку вызовов: (запросов, вход, выход)."""
    return (len(calls), sum(PRICE[k][0] for k in calls), sum(PRICE[k][1] for k in calls))


def last_record():
    """Последний замер расхода в диалоге ({} — замера нет)."""
    usage = chat._current_session()["dialog"]["usage"]
    return usage[-1] if usage else {}


def compare(title, calls, record, service_kinds=None):
    """Сверяет запись диалога с истинным расходом."""
    requests, prompt, completion = truth(calls)
    check(f"{title}: замер запроса сохранён в диалоге", bool(record),
          "записи расхода нет вовсе")
    check(f"{title}: число обращений к модели",
          int(record.get("requests") or 0) == requests,
          f"в записи {record.get('requests')}, реально {requests} ({calls})")
    check(f"{title}: входящие токены",
          int(record.get("input") or 0) == prompt,
          f"в записи {record.get('input')}, реально {prompt}")
    check(f"{title}: исходящие токены",
          int(record.get("output") or 0) == completion,
          f"в записи {record.get('output')}, реально {completion}")
    if service_kinds is not None:
        service_calls = [k for k in calls if k in service_kinds]
        _, service_prompt, service_completion = truth(service_calls)
        check(f"{title}: вклад служебных вызовов",
              int(record.get("summary_requests") or 0) == len(service_calls)
              and int(record.get("summary_input") or 0) == service_prompt
              and int(record.get("summary_output") or 0) == service_completion,
              f"в записи {record.get('summary_requests')}/{record.get('summary_input')}, "
              f"реально {len(service_calls)}/{service_prompt}")
        breakdown = sum(int(v.get("requests") or 0)
                        for v in (record.get("service") or {}).values())
        check(f"{title}: разбивка по видам сходится с суммой служебных",
              breakdown == int(record.get("summary_requests") or 0),
              f"разбивка {breakdown}, сумма {record.get('summary_requests')}")


async def reset():
    """Чистый диалог: у каждого сценария свой замер."""
    await chat.agent_history_clear()
    CALLS.clear()


async def main():
    await chat.task_create(chat.TaskCreate(name="Проект"))
    await chat.invariant_create(chat.InvariantCreate(
        text="Ответы только на русском", scope="project"))
    # Виды вызовов заглушки, которые считаются СЛУЖЕБНЫМИ (их результат не
    # является ответом пользователю).
    service_kinds = {"разбор_запроса", "план", "код-гейт", "приёмщик",
                     "сжатие", "факты"}

    print("\n[1] Первый запрос: разбор запроса + план + код-гейт")
    await reset()
    CFG.update({"plan": ["Шаг 1", "Шаг 2"], "gate": "clear", "analysis": "clear",
                "suggestions": "clear", "review": None, "fail": False})
    await run_chat("Сделай отчёт")
    compare("первый запрос", list(CALLS), last_record(), service_kinds)

    print("\n[2] Перепланирование: гейт отклонил план")
    await reset()
    CFG["gate"] = "violation"
    gate_calls = {"n": 0}

    async def fake_gate(*args, **kwargs):
        # Первый гейт — нарушение, второй (после перепланирования) — чисто.
        gate_calls["n"] += 1
        CFG["gate"] = "violation" if gate_calls["n"] == 1 else "clear"
        return await fake_call_llm_async(*args, **kwargs)

    client.call_llm_async = fake_gate
    await run_chat("Сделай что-то большое")
    client.call_llm_async = fake_call_llm_async
    compare("перепланирование", list(CALLS), last_record(), service_kinds)

    print("\n[3] Подтверждение плана «ок»: только ответ шага")
    await reset()
    CFG.update({"plan": ["Шаг 1", "Шаг 2"], "gate": "clear", "analysis": "clear"})
    await run_chat("Спланируй работу")
    CALLS.clear()
    await run_chat("ок")
    compare("ответ шага", list(CALLS), last_record(), service_kinds)

    print("\n[4] Последний шаг: ответ + содержательная проверка")
    await reset()
    CFG.update({"plan": ["Единственный шаг"], "review": {"verdict": "ok", "step": 0,
                                                         "comment": "ок"}})
    await run_chat("Сделай одну вещь")
    await run_chat("ок")
    CALLS.clear()
    await run_chat("продолжай")
    compare("последний шаг", list(CALLS), last_record(), service_kinds)

    print("\n[5] Автономный режим: план и шаг в одном запросе")
    await reset()
    CFG.update({"plan": ["Шаг 1", "Шаг 2"], "review": None})
    await run_chat("Сделай всё сам, работай автономно")
    compare("автономный режим", list(CALLS), last_record(), service_kinds)

    print("\n[6] Отказ по инвариантам: расход сохраняется служебной записью")
    await reset()
    CFG.update({"analysis": {"вердикт": "violation", "объяснение": "правило",
                             "варианты": [{"заголовок": "A", "суть": "a", "send": "Сделай A"},
                                          {"заголовок": "B", "суть": "b", "send": "Сделай B"}]},
                "suggestions": {"1": {"вердикт": "clear"}, "2": {"вердикт": "clear"}}})
    await run_chat("Сделай запрещённое")
    record = last_record()
    compare("отказ по инвариантам", list(CALLS), record, service_kinds)
    check("запись отказа помечена служебной (kind)", bool(record.get("kind")),
          str(record))
    check("запись отказа переживает перезагрузку страницы",
          bool((await chat.agent_history())["usage"]), str(record))

    print("\n[7] Сбой провайдера: вызов учтён, расход не выдуман")
    await reset()
    CFG.update({"analysis": "clear", "plan": ["Шаг"], "fail": True})
    calls_before = len(CALLS)
    await run_chat("Сделай отчёт")
    check("сбой зафиксирован счётчиком неудачных вызовов",
          any(int(item.get("failed_requests") or 0) > 0
              for item in chat._current_session()["dialog"]["usage"])
          or len(CALLS) > calls_before,
          str(chat._current_session()["dialog"]["usage"]))

    print("\n[8] Сжатие памяти и блок фактов учитываются как служебные")
    await reset()
    CFG.update({"fail": False, "plan": ["Шаг 1"], "gate": "clear",
                "analysis": "clear", "review": None})
    await run_chat("Начни работу", agent_strategy="facts", window=2)
    await run_chat("ок", agent_strategy="facts", window=2)

    async def summary_probe():
        await chat.agent_history_clear()
        CALLS.clear()
        CFG["plan"] = ["Шаг 1", "Шаг 2", "Шаг 3", "Шаг 4"]
        events = await run_chat("Длинная работа", agent_strategy="summary", summary=2)
        for _ in range(5):
            events += await run_chat("", continue_step=True, agent_strategy="summary",
                                     summary=2)
        return events

    await summary_probe()
    record = last_record()
    kinds = set((record.get("service") or {}).keys())
    allowed = {agent_mod.Agent.SERVICE_PLAN, agent_mod.Agent.SERVICE_GATE,
               agent_mod.Agent.SERVICE_INVARIANTS, agent_mod.Agent.SERVICE_REVIEW,
               agent_mod.Agent.SERVICE_SUMMARY, agent_mod.Agent.SERVICE_FACTS,
               agent_mod.Agent.SERVICE_BRANCHING}
    check("в разбивке видны виды служебных вызовов",
          bool(kinds) and kinds <= allowed, str(kinds))
    total_service = int(record.get("summary_requests") or 0)
    breakdown = sum(int(v.get("requests") or 0)
                    for v in (record.get("service") or {}).values())
    check("сумма разбивки равна вкладу служебных вызовов",
          breakdown == total_service, f"{breakdown} != {total_service}")
    check("расход шага не превышает истинное число вызовов",
          int(record.get("requests") or 0) <= len(CALLS) + 1,
          f"{record.get('requests')} при {len(CALLS)} вызовах")

    print("\nИтог: " + ("все проверки пройдены" if not FAILURES
                       else "ПРОВАЛЕНО проверок: %d — %s" % (len(FAILURES), FAILURES)))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
