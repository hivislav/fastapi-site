"""ЖИВОЙ сценарий: агент сам работает с файлами проекта (MCP-сервер `files`).

Запуск (НУЖЕН рабочий источник ответа — ключ в `.env` или поднятая локальная
модель; скрипт делает настоящие вызовы модели и потому стоит денег):

    ./venv/bin/python tools/files_live.py [--goal "текст цели"] [--repo ПУТЬ] [--keep]

ЧТО ПРОВЕРЯЕТСЯ

Самый главный вопрос этой правки: агент получает задачу НА УРОВНЕ ЦЕЛИ («обнови
руководство по коду и заведи запись в журнале») и САМ решает, какие файлы
прочитать, где искать и что править — без подсказок вида «открой файл X».
Скрипт создаёт ВРЕМЕННУЮ рабочую копию из трёх файлов, проект с включённым
сервером «Файлы проекта» и прогоняет задачу целиком: запрос → план → «ок» →
шаги → ответ. В конце он проверяет ФАКТЫ, а не слова модели:

  * какие инструменты агент вызвал (и были ли среди них ЧТЕНИЯ файлов);
  * что изменения ЛЕЖАТ НА ДИСКЕ (содержимое читается напрямую, мимо агента);
  * что видно, ЧТО изменилось (diff по рабочей копии);
  * что записи не выполнялись до подтверждения плана (в гибридной схеме до «ок»
    идут только чтения) — это печатается по журналу вызовов;
  * сколько это стоило (расход вызовов за задачу).

`--repo ПУТЬ` берёт ГОТОВУЮ рабочую копию вместо временной (например копию самого
проекта, созданную `git worktree add /tmp/project HEAD`): тогда агент правит
настоящие файлы проекта, а рабочее дерево разработчика остаётся чистым.

Сценарий ВОСПРОИЗВОДИМ: копия каждый раз собирается заново из этого файла, а
`--keep` оставляет её на диске, чтобы посмотреть руками. Рабочие файлы проекта
скрипт не трогает: рабочая копия — временная (`FILES_REPO_DIR`), workspace и
профили — во временном каталоге.
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_TMP = tempfile.mkdtemp(prefix="files-live-")
os.environ["AGENT_WORKSPACE_FILE"] = os.path.join(_TMP, "workspace.json")
os.environ["AGENT_MEMORY_FILE"] = os.path.join(_TMP, "agent_memory.json")
os.environ["AGENT_PROFILES_FILE"] = os.path.join(_TMP, "profiles.json")
os.environ["RAG_DIR"] = os.path.join(_TMP, "rag")
os.environ["MCP_ATTACH_DIR"] = os.path.join(_TMP, "mcp_files")
os.environ["PERIODIC_ENABLED"] = "0"

from app import config  # noqa: E402,F401  (импорт читает .env)
from app.ai import mcp as mcp_store  # noqa: E402
from app.routers import chat  # noqa: E402
from app.schemas import ChatMessage, McpApply, TaskCreate  # noqa: E402

DEFAULT_GOAL = ("Обнови руководство docs/guide.md по коду: оно описывает расчёт "
                "итога, а функция теперь называется compute_total. И заведи файл "
                "docs/CHANGELOG.md с записью об этом изменении.")
MAX_STEPS = 6

MODULE_A = '''"""Модуль расчёта."""


def format_money(value):
    """Отформатировать сумму."""
    return f"{value:.2f} руб."


def compute_total(items):
    """Итог по позициям."""
    return sum(item["price"] for item in items)
'''

MODULE_B = '''"""Отчёт по заказу."""

from app.module_a import compute_total


def report(rows):
    return "Итого: " + str(compute_total(rows))
'''

GUIDE = """# Руководство проекта

## Расчёт итога

Считает функция `summary_total`. Оформляет сумму `format_money`.

## Запуск

Проект запускается командой `python -m app`.
"""


def write(rel, text):
    path = os.path.join(REPO, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def read(rel):
    path = os.path.join(REPO, rel)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def make_repo():
    """Временная рабочая копия: код + документ, который от него отстал."""
    write("app/__init__.py", "")
    write("app/module_a.py", MODULE_A)
    write("app/module_b.py", MODULE_B)
    write("docs/guide.md", GUIDE)
    env = dict(os.environ)
    env.update({"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull})
    run = lambda *args: subprocess.run(["git", "-C", REPO, *args], capture_output=True,
                                       text=True, env=env, check=False)
    run("init", "-q")
    run("add", "-A")
    run("-c", "user.name=Прогон", "-c", "user.email=run@example.invalid",
        "commit", "-q", "-m", "исходное состояние")


def diff_stat():
    """Короткая сводка изменений на диске (git diff --stat)."""
    env = dict(os.environ)
    env.update({"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull})
    out = subprocess.run(["git", "-C", REPO, "diff", "HEAD", "--stat"],
                         capture_output=True, text=True, env=env, check=False)
    status = subprocess.run(["git", "-C", REPO, "status", "--porcelain=v1"],
                            capture_output=True, text=True, env=env, check=False)
    return out.stdout.strip(), status.stdout.strip()


async def run_agent(text, **kwargs):
    """Прогон POST /api/agent/chat: собирает события NDJSON-потока."""
    response = await chat.agent_chat(ChatMessage(content=text, **kwargs))
    events = []
    async for chunk in response.body_iterator:
        for line in str(chunk).splitlines() or []:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return events


def show(events, keys=("debug", "bot", "choices", "error")):
    """Печать интересных событий прогона (без служебного шума)."""
    for event in events:
        kind = str(event.get("type") or "")
        if kind not in keys:
            continue
        text = str(event.get("text") or "").strip()
        if not text:
            continue
        lines = text.splitlines()
        head = lines[0][:180]
        print(f"    [{kind}] {head}" + (f"  …(+{len(lines) - 1} строк)" if len(lines) > 1 else ""))


def calls_of(session):
    """Вызовы MCP по текущему запросу задачи (что агент реально вызвал)."""
    dialog = session.get("dialog") or {}
    stored = dialog.get("mcp") or {}
    return list(stored.get("calls") or [])


async def main() -> int:
    global REPO
    goal = DEFAULT_GOAL
    keep = False
    given_repo = ""
    if "--goal" in sys.argv:
        goal = sys.argv[sys.argv.index("--goal") + 1]
    if "--repo" in sys.argv:
        given_repo = os.path.abspath(sys.argv[sys.argv.index("--repo") + 1])
    if "--keep" in sys.argv:
        keep = True
    print("Живой сценарий: агент работает с файлами проекта (MCP-сервер files).")
    print("Цель: " + goal)
    print("Временный каталог: " + _TMP)

    REPO = given_repo or os.path.join(_TMP, "repo")
    os.makedirs(REPO, exist_ok=True)
    os.environ["FILES_REPO_DIR"] = REPO
    if given_repo:
        print("Рабочая копия (готовая, НЕ временная): " + REPO + "\n")
    else:
        make_repo()
        print("Рабочая копия: " + REPO + " (файлов: 4, включая docs/guide.md)\n")

    source = config.llm_source()
    spec = config.provider_spec(config.active_provider())
    print(f"Источник ответа: {source} ({spec.get('base_url') or '—'}, "
          f"модель {config.active_model()})")
    if not spec.get("api_key") and source != "local":
        print("Нет ключа провайдера — живой прогон невозможен (см. .env).")
        return 1

    # Проект с включённым сервером «Файлы проекта»: без него инструментов у агента нет.
    await chat.task_create(TaskCreate(name="Файлы проекта (живой прогон)"))
    await chat.mcp_apply(McpApply(enabled=["files"]))
    tools = mcp_store.discover("files", force=True)
    print("Сервер files: ok=%s, инструментов %d" % (tools.get("ok"), len(tools.get("tools") or [])))

    print("\n1) Запрос цели (до подтверждения плана выполняются только ЧТЕНИЯ):")
    events = await run_agent(goal)
    show(events)
    session = chat._current_session()
    dialog = session["dialog"]
    state = dialog.get("state") or {}
    stage = str(state.get("stage") or "")
    steps = [str(step.get("text") if isinstance(step, dict) else step or "")
             for step in (state.get("steps") or [])]
    print(f"    этап: {stage}; шагов в плане: {len(steps)}")
    for index, step in enumerate(steps, 1):
        print(f"      {index}. {step[:120]}")
    before_confirm = calls_of(session)
    prints_before = [call for call in before_confirm
                     if mcp_store.is_deferred_call(call)]
    reads_before = [call for call in before_confirm
                    if not mcp_store.is_deferred_call(call)]
    print(f"    вызовов MCP до «ок»: {len(before_confirm)} "
          f"(чтения: {len(reads_before)}, записи: {len(prints_before)})")
    for call in before_confirm:
        print(f"      · {call.get('server')} · {call.get('tool')} "
              f"{json.dumps(call.get('arguments') or {}, ensure_ascii=False)[:110]}")

    if not steps:
        print("\nПлан не построен — задача пошла прямым ответом. "
              "Для правки файлов нужен путь с планом; прогон с force_plan:")
        events = await run_agent(goal, force_plan=True)
        show(events)
        session = chat._current_session()
        state = (session["dialog"].get("state") or {})
        steps = [str(step.get("text") if isinstance(step, dict) else step or "")
             for step in (state.get("steps") or [])]
        print(f"    шагов в плане: {len(steps)}")

    print("\n2) Подтверждение плана («ок») — здесь выполняются ОТЛОЖЕННЫЕ вызовы:")
    events = await run_agent("ок")
    show(events)

    print("\n3) Шаги задачи (пока автомат в работе, до %d шагов):" % MAX_STEPS)
    for index in range(MAX_STEPS):
        session = chat._current_session()
        state = session["dialog"].get("state") or {}
        stage = str(state.get("stage") or "")
        if stage in ("done", "failed", "cancelled"):
            break
        events = await run_agent("Продолжай по плану.", continue_step=True)
        show(events)
    session = chat._current_session()
    state = session["dialog"].get("state") or {}

    print("\n4) ЧТО ВЫЗЫВАЛ АГЕНТ (по журналу запроса):")
    calls = calls_of(session)
    for call in calls:
        deferred = mcp_store.is_deferred_call(call)
        print(f"    · {call.get('server')} · {call.get('tool')}"
              f"{'  [ЗАПИСЬ]' if deferred else ''} "
              f"{json.dumps(call.get('arguments') or {}, ensure_ascii=False)[:120]}")
    if not calls:
        print("    (агент не вызвал ни одного инструмента)")

    print("\n5) ФАКТЫ НА ДИСКЕ (читаем файлы напрямую, мимо агента):")
    guide = read("docs/guide.md") or ""
    changelog = read("docs/CHANGELOG.md")
    stat, status = diff_stat()
    print("    git status:")
    for line in status.splitlines() or ["(пусто)"]:
        print("      " + line)
    print("    git diff --stat:")
    for line in stat.splitlines() or ["(пусто)"]:
        print("      " + line)
    if not given_repo:
        print("    docs/guide.md — упоминает новое имя: "
              + ("ДА" if "compute_total" in guide else "НЕТ")
              + "; упоминает старое `summary_total`: "
              + ("да" if "summary_total" in guide else "НЕТ"))
        print("    docs/CHANGELOG.md: "
              + ("создан, " + str(len(changelog or "")) + " символов" if changelog
                 else "НЕ создан"))

    rows = session["dialog"].get("usage") or []
    rows = rows if isinstance(rows, list) else [rows]
    total = sum(float(row.get("cost_rub") or 0) for row in rows)
    calls_count = sum(int(row.get("calls") or 1) for row in rows)
    print(f"\n6) Расход на прогон: {round(total, 4)} ₽, записей о расходе: "
          f"{len(rows)} (вызовов модели ≈ {calls_count})")

    print("\nИТОГ:")
    # Изменённые и новые файлы — ИЗ GIT STATUS, а не по списку из сценария: при
    # прогоне на готовой копии (`--repo`) агент работает с настоящими файлами
    # проекта, и заранее известного перечня у проверки нет.
    files_changed = []
    for line in status.splitlines():
        name = line[2:].strip().strip(chr(34))
        if " -> " in name:
            name = name.split(" -> ")[-1]
        if name and name not in files_changed:
            files_changed.append(name)
    tools_used = sorted({str(call.get("tool")) for call in calls})
    print("    инструменты: " + (", ".join(tools_used) or "—"))
    print("    файлы изменены на диске: " + (", ".join(files_changed) or "—"))
    print("    до подтверждения плана записей не было: "
          + ("ДА" if not prints_before else "НЕТ"))
    ok = bool(files_changed) and bool(reads_before)
    print("    вердикт: " + ("агент выполнил задачу с файлами"
                             if ok else "НЕ выполнено — смотрите журнал выше"))
    if keep:
        print("\nКопия оставлена: " + REPO)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
