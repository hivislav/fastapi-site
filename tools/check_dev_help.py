"""Самопроверка АССИСТЕНТА РАЗРАБОТЧИКА — команды `/help` и MCP-сервера `git`.

Запуск (сеть, Node.js и API-ключ НЕ нужны):

    ./venv/bin/python tools/check_dev_help.py

ЧТО ПРОВЕРЯЕТСЯ

  [1] Реестр серверов: сервер `git` объявлен, доступен без предпосылок к сбою
      (есть node, файл сервера, MCP SDK) и объявляет ровно свои инструменты.
  [2] Инструменты git на НАСТОЯЩЕМ временном репозитории (`GIT_REPO_DIR`):
      ветка и последний коммит, список файлов, изменённые файлы, diff (пустой и
      непустой), история коммитов, содержимое файла по строкам — и ОХРАНА:
      секреты (`.env`), служебные каталоги (`data/`, `.git/`), выход за корень
      репозитория, аргумент-опция вместо пути. Отдельно проверяется, что ответ
      любого инструмента ВЛЕЗАЕТ в клиентский предел (`RESULT_CHARS`) — иначе
      хвост обрезался бы молча.
  [3] Разбор команды и набор вызовов контекста: команда узнаётся (а «/helper» —
      нет), вопрос берётся из строки, путь файла находится в тексте вопроса,
      вызовы выбирает КОД (файл, каталог, общий случай), история разговора
      собирается из журнала чата, сообщения модели собираются по данным.
  [4] Маршрут POST /api/agent/help (заглушка модели): поток событий, КОНТЕКСТ
      РЕПОЗИТОРИЯ в сообщениях модели, фрагменты документации и строка
      источников, запись в журнал чата и НЕзапись в память задачи и автомат.
  [5] Честность, когда источников нет: сервер MCP выключен у проекта, базы
      знаний не подключены, сервер отказал — команда отвечает, а причина
      названа словами (и не выдумывает ни ссылок, ни данных).

Рабочие данные не трогаются: workspace, профили, история агента и базы знаний
пишутся во временный каталог (переменные AGENT_*_FILE и RAG_DIR выставляются ДО
импорта приложения), а репозиторий для инструментов создаётся ВРЕМЕННЫЙ — сам
проект проверка не меняет.
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Изоляция данных: всё пишем во временный каталог ------------------------
_TMP = tempfile.mkdtemp(prefix="devhelp-check-")
os.environ["AGENT_WORKSPACE_FILE"] = os.path.join(_TMP, "workspace.json")
os.environ["AGENT_MEMORY_FILE"] = os.path.join(_TMP, "agent_memory.json")
os.environ["AGENT_PROFILES_FILE"] = os.path.join(_TMP, "profiles.json")
os.environ["RAG_DIR"] = os.path.join(_TMP, "rag")
os.environ["MCP_ATTACH_DIR"] = os.path.join(_TMP, "mcp_files")
# Эмбеддинги и реранкер — офлайн и детерминированно: проверка обязана идти без
# сети и давать один и тот же результат на любой машине.
os.environ["RAG_EMBED_BACKEND"] = "hashing"
os.environ["RAG_RERANK_BACKEND"] = "features"
# Планировщик периодических задач в проверке не участвует.
os.environ["PERIODIC_ENABLED"] = "0"

from fastapi import HTTPException  # noqa: E402

from app.ai import client as llm_client  # noqa: E402
from app.ai import dev_help  # noqa: E402
from app.ai import rag  # noqa: E402
from app.ai import rag_dialog  # noqa: E402
from app.ai import mcp as mcp_store  # noqa: E402
from app.ai import workspace as workspace_store  # noqa: E402
from app.routers import chat  # noqa: E402
from app.schemas import ChatMessage  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    """Одна проверка: печатает результат и копит провалы."""
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(name)


def _raises(exc_type, work):
    """Брошено ли ожидаемое исключение (для отказов маршрута)."""
    try:
        work()
    except exc_type:
        return True
    except Exception:                            # noqa: BLE001
        return False
    return False


async def _araises(exc_type, work):
    try:
        await work()
    except exc_type:
        return True
    except Exception:                            # noqa: BLE001
        return False
    return False


# ---------------------------------------------------------------------------
# Временный репозиторий: на нём проверяются инструменты git
# ---------------------------------------------------------------------------
GIT_ENV = {
    "GIT_AUTHOR_NAME": "Проверка",
    "GIT_AUTHOR_EMAIL": "check@example.invalid",
    "GIT_COMMITTER_NAME": "Проверка",
    "GIT_COMMITTER_EMAIL": "check@example.invalid",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
}
BRANCH = "work/check"
REPO = ""                       # путь временного репозитория (заполняет main)
MODULE_TEXT = "\n".join(
    "def step_%d():\n    return %d" % (number, number) for number in range(1, 41)
)
DOCS_TEXT = (
    "# Руководство проекта\n\n"
    "## Поиск по базам знаний\n\n"
    "Поиск идёт в два этапа: сначала широкий пул кандидатов, затем реранкинг и "
    "порог релевантности. Переформулировка запроса включается галочкой.\n"
)


def git_run(repo, *args, ok=True):
    """Команда git во временном репозитории (чужая настройка не читается)."""
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True,
                            text=True, env={**os.environ, **GIT_ENV})
    if ok and result.returncode != 0:
        raise RuntimeError("git %s: %s" % (" ".join(args), result.stderr.strip()))
    return result


def make_repo():
    """Создать временный репозиторий: ветка, коммит, правка, новый файл."""
    repo = os.path.join(_TMP, "repo")
    os.makedirs(os.path.join(repo, "app", "ai"))
    os.makedirs(os.path.join(repo, "docs"))
    os.makedirs(os.path.join(repo, "build"))
    with open(os.path.join(repo, "app", "ai", "module.py"), "w", encoding="utf-8") as fh:
        fh.write(MODULE_TEXT + "\n")
    with open(os.path.join(repo, "docs", "guide.md"), "w", encoding="utf-8") as fh:
        fh.write(DOCS_TEXT)
    with open(os.path.join(repo, "README.md"), "w", encoding="utf-8") as fh:
        fh.write("# Проект проверки\n")
    # Игнорируемые файлы — как в самом проекте: секреты и рабочие данные в git не
    # хранятся, и в списке файлов их быть не должно.
    with open(os.path.join(repo, ".gitignore"), "w", encoding="utf-8") as fh:
        fh.write(".env\ndata/\nbuild/\n")
    # Игнорируемый файл ВНЕ закрытых охраной каталогов: на нём проверяется ответ
    # «git его не отслеживает, diff по нему не бывает» (по .env и data/ сработала
    # бы охрана путей, и до этого текста дело не дошло бы).
    with open(os.path.join(repo, "build", "out.txt"), "w", encoding="utf-8") as fh:
        fh.write("сборка\n")
    # Секреты и служебные каталоги — чтобы охрана путей проверялась по-настоящему.
    with open(os.path.join(repo, ".env"), "w", encoding="utf-8") as fh:
        fh.write("SECRET_KEY=не-показывать\n")
    os.makedirs(os.path.join(repo, "data"))
    with open(os.path.join(repo, "data", "history.json"), "w", encoding="utf-8") as fh:
        fh.write("{}")
    git_run(repo, "init", "-b", BRANCH)
    git_run(repo, "add", ".")
    git_run(repo, "commit", "-m", "первый коммит: каркас проекта")
    # Рабочая копия «грязная»: правка отслеживаемого файла и новый файл.
    with open(os.path.join(repo, "app", "ai", "module.py"), "a", encoding="utf-8") as fh:
        fh.write("\n\ndef new_step():\n    return 'добавлено проверкой'\n")
    with open(os.path.join(repo, "notes.txt"), "w", encoding="utf-8") as fh:
        fh.write("черновик\n")
    return repo


def call_tool(tool, arguments=None):
    """Вызов инструмента через НАСТОЯЩИЙ клиент приложения (app/ai/mcp.py)."""
    results = mcp_store.run_calls([{"server": dev_help.SERVER_ID, "tool": tool,
                                    "arguments": arguments or {}}])
    return results[0]


# ---------------------------------------------------------------------------
# [1] Реестр серверов
# ---------------------------------------------------------------------------
def section_registry():
    print("\n[1] Сервер git в реестре MCP: объявлен, доступен, инструменты на месте")
    entry = mcp_store.find_server(dev_help.SERVER_ID)
    check("сервер объявлен в реестре", bool(entry), repr(entry))
    if not entry:
        return
    check("файл сервера — git.mjs в каталоге серверов",
          mcp_store.command_for(entry)[-1].endswith("git.mjs"),
          mcp_store.command_for(entry)[-1])
    check("предпосылок к сбою нет (node, файл, SDK)",
          mcp_store.availability_error(entry) == "",
          mcp_store.availability_error(entry))
    check("описание для интерфейса говорит про ветку и diff",
          "diff" in str(entry.get("description") or "").lower())
    info = mcp_store.discover(dev_help.SERVER_ID, force=True)
    check("сервер отвечает на tools/list", bool(info.get("ok")), str(info.get("error")))
    names = [str(tool.get("name")) for tool in info.get("tools") or []]
    check("объявлены все пять инструментов",
          names == [dev_help.STATUS_TOOL, dev_help.FILES_TOOL, dev_help.DIFF_TOOL,
                    dev_help.LOG_TOOL, dev_help.SHOW_TOOL], str(names))


# ---------------------------------------------------------------------------
# [2] Инструменты на настоящем временном репозитории
# ---------------------------------------------------------------------------
def section_tools():
    print("\n[2] Инструменты git на настоящем репозитории (временном) + охрана путей")
    # Клиент кэширует список инструментов сервера, а репозиторий у сервера теперь
    # ДРУГОЙ (временный): кэш надо сбросить, иначе проверялся бы прежний.
    mcp_store.forget(dev_help.SERVER_ID)

    status = call_tool(dev_help.STATUS_TOOL)
    text = status["text"]
    check("git_status отвечает", status["ok"], status["error"])
    check("названа текущая ветка", "Ветка: " + BRANCH in text, text[:200])
    check("назван последний коммит",
          "первый коммит: каркас проекта" in text and "98875c1" not in text)
    check("названы изменённые файлы и новый файл",
          "app/ai/module.py" in text and "notes.txt" in text)
    check("сказано, что изменений в рабочей копии два",
          "изменено файлов — 2" in text, text)
    check("ветка без upstream названа честно",
          "нет (у ветки нет upstream)" in text)

    files = call_tool(dev_help.FILES_TOOL, {"limit": 50})
    check("git_files отвечает и видит и отслеживаемые, и новые файлы",
          files["ok"] and "app/ai/module.py" in files["text"]
          and "docs/guide.md" in files["text"] and "notes.txt" in files["text"],
          files["text"][:200])
    check("игнорируемые файлы в список не попали (.env, data/)",
          ".env" not in files["text"] and "data/history.json" not in files["text"])

    changed = call_tool(dev_help.FILES_TOOL, {"changed_only": True})
    check("git_files changed_only показывает статусы изменений",
          changed["ok"] and "M app/ai/module.py" in changed["text"]
          and "?? notes.txt" in changed["text"], changed["text"][:200])

    diff = call_tool(dev_help.DIFF_TOOL, {"scope": "working"})
    check("git_diff показывает саму правку",
          diff["ok"] and "добавлено проверкой" in diff["text"]
          and "new_step" in diff["text"], diff["text"][:200])
    stat = call_tool(dev_help.DIFF_TOOL, {"scope": "head", "stat_only": True})
    check("сводка diff называет файл и правки",
          stat["ok"] and "app/ai/module.py" in stat["text"]
          and "1 file changed" in stat["text"], stat["text"][:200])
    empty = call_tool(dev_help.DIFF_TOOL, {"scope": "staged"})
    check("пустой diff — это ФАКТ, а не ошибка",
          empty["ok"] and "Изменений нет" in empty["text"], empty["text"][:200])
    # НОВЫЙ ФАЙЛ — САМЫЙ ЧАСТЫЙ ИСТОЧНИК ВРАНЬЯ: пустой diff у неотслеживаемого
    # файла означает не «изменений нет», а «сравнивать не с чем» (живой случай
    # 12.10: агент ответил «изменений в README.md нет», хотя файл был написан
    # только что и целиком был новым). Причину называет СЕРВЕР, а не модель.
    fresh_diff = call_tool(dev_help.DIFF_TOOL, {"path": "notes.txt", "scope": "head"})
    check("у НОВОГО файла diff не выдаётся за «изменений нет»",
          fresh_diff["ok"] and "НЕ ОТСЛЕЖИВАЕТСЯ" in fresh_diff["text"]
          and "git_show_file" in fresh_diff["text"]
          and "Изменений нет" not in fresh_diff["text"], fresh_diff["text"][:220])
    fresh_log = call_tool(dev_help.LOG_TOOL, {"path": "notes.txt"})
    check("у нового файла история объяснена, а не «коммитов не найдено»",
          fresh_log["ok"] and "НЕ ОТСЛЕЖИВАЕТСЯ" in fresh_log["text"],
          fresh_log["text"][:220])
    # ЧИСТАЯ РАБОЧАЯ КОПИЯ, НО ЕСТЬ НОВЫЙ ФАЙЛ: пустой diff обязан назвать новые
    # файлы, иначе «изменений нет» читается как «в проекте ничего не менялось».
    git_run(REPO, "commit", "-am", "правка модуля проверки")
    clean = call_tool(dev_help.DIFF_TOOL, {"scope": "head"})
    check("пустой diff называет новые (неотслеживаемые) файлы",
          clean["ok"] and "НОВЫЕ файлы" in clean["text"] and "notes.txt" in clean["text"],
          clean["text"][:240])
    # ПРАВКИ ТОЛЬКО В ИНДЕКСЕ: «пустой diff» по рабочей копии обязан называть
    # причину и подсказывать нужный scope.
    git_run(REPO, "add", "notes.txt")
    staged = call_tool(dev_help.DIFF_TOOL, {"path": "notes.txt", "scope": "working"})
    check("правки в индексе: сказано, где они, и каким scope их взять",
          staged["ok"] and "ИНДЕКСЕ" in staged["text"] and "scope=staged" in staged["text"],
          staged["text"][:220])
    staged_head = call_tool(dev_help.DIFF_TOOL, {"path": "notes.txt", "scope": "staged"})
    check("в индексе те же правки видны как diff",
          staged_head["ok"] and "черновик" in staged_head["text"], staged_head["text"][:200])
    # ИГНОРИРУЕМЫЙ файл: diff по нему не бывает — и это говорит инструмент.
    ignored = call_tool(dev_help.DIFF_TOOL, {"path": "build/out.txt", "scope": "head"})
    check("игнорируемый файл: сказано, что git его не отслеживает",
          ignored["ok"] and "gitignore" in ignored["text"], ignored["text"][:220])
    secret = call_tool(dev_help.DIFF_TOOL, {"path": ".env", "scope": "head"})
    check("охрана не пускает diff по секретам", not secret["ok"], secret.get("error", "")[:120])
    no_base = call_tool(dev_help.DIFF_TOOL, {"scope": "branch"})
    check("без базовой ветки git_diff честно отказывает и просит base",
          not no_base["ok"] and "базовую ветку" in no_base["error"],
          no_base["error"][:200])
    same = call_tool(dev_help.DIFF_TOOL, {"scope": "branch", "base": BRANCH})
    check("сравнение ветки с собой — пустой diff",
          same["ok"] and "Изменений нет" in same["text"], same["text"][:200])

    log = call_tool(dev_help.LOG_TOOL, {"limit": 5})
    check("git_log отдаёт коммиты ветки",
          log["ok"] and "первый коммит: каркас проекта" in log["text"]
          and "Проверка" in log["text"], log["text"][:200])

    show = call_tool(dev_help.SHOW_TOOL, {"path": "app/ai/module.py", "lines": 3})
    check("git_show_file показывает файл С НОМЕРАМИ строк",
          show["ok"] and "строки 1–3 из" in show["text"] and "1| def step_1" in show["text"],
          show["text"][:200])
    check("и говорит, что показан только отрывок",
          "файл продолжается" in show["text"] and "Следующий отрывок" in show["text"])
    later = call_tool(dev_help.SHOW_TOOL, {"path": "app/ai/module.py", "from": 40,
                                           "lines": 3})
    check("git_show_file отдаёт запрошенный отрывок",
          later["ok"] and "строки 40–42" in later["text"] and "40| " in later["text"],
          later["text"][:200])
    from_ref = call_tool(dev_help.SHOW_TOOL, {"path": "docs/guide.md", "ref": BRANCH,
                                              "lines": 2})
    check("файл можно взять из версии ветки",
          from_ref["ok"] and BRANCH + "), строки 1–2" in from_ref["text"],
          from_ref["text"][:200])
    past_end = call_tool(dev_help.SHOW_TOOL, {"path": "docs/guide.md", "from": 9999})
    check("начало за концом файла названо словами",
          past_end["ok"] and "только" in past_end["text"] and "строк" in past_end["text"],
          past_end["text"][:200])

    # ОХРАНА ПУТЕЙ: секреты, служебные каталоги, выход за корень, аргумент-опция.
    for tool, args, expected in (
        (dev_help.SHOW_TOOL, {"path": ".env"}, "секреты"),
        (dev_help.SHOW_TOOL, {"path": "data/history.json"}, "закрыт"),
        (dev_help.SHOW_TOOL, {"path": ".git/config"}, "закрыт"),
        (dev_help.SHOW_TOOL, {"path": "../outside.txt"}, "вне репозитория"),
        (dev_help.SHOW_TOOL, {"path": "--output=/tmp/x"}, "не может начинаться"),
        (dev_help.DIFF_TOOL, {"path": "data"}, "закрыт"),
        (dev_help.FILES_TOOL, {"path": "/etc"}, "вне репозитория"),
    ):
        outcome = call_tool(tool, args)
        check("охрана: %s %s" % (tool, json.dumps(args, ensure_ascii=False)),
              (not outcome["ok"]) and expected in (outcome["error"] or ""),
              outcome.get("error", "")[:160])

    # БЮДЖЕТ ОТВЕТА: клиент берёт у инструмента не больше RESULT_CHARS символов,
    # и всё, что длиннее, обрезалось бы МОЛЧА (выглядело бы как «данных больше
    # нет»). Поэтому сервер держит ответы в своём бюджете.
    biggest = [
        call_tool(dev_help.SHOW_TOOL, {"path": "app/ai/module.py", "lines": 400}),
        call_tool(dev_help.DIFF_TOOL, {"scope": "head"}),
        call_tool(dev_help.FILES_TOOL, {"limit": 1000}),
    ]
    check("ни один ответ не длиннее клиентского предела (%d)"
          % mcp_store.RESULT_CHARS,
          all(len(item["text"]) <= mcp_store.RESULT_CHARS for item in biggest),
          str([len(item["text"]) for item in biggest]))


# ---------------------------------------------------------------------------
# [3] Разбор команды и набор вызовов
# ---------------------------------------------------------------------------
def section_command():
    print("\n[3] Разбор команды /help и набор вызовов контекста")
    check("команда узнаётся", dev_help.is_command("/help")
          and dev_help.is_command("/help что делает модуль?"))
    check("«/helper» и «help» командой НЕ считаются",
          not dev_help.is_command("/helper") and not dev_help.is_command("help me")
          and not dev_help.is_command("/help-me"))
    check("вопрос берётся из строки команды",
          dev_help.parse("/help   что делает app/ai/mcp.py?  ")
          == (True, "что делает app/ai/mcp.py?"))
    check("вопрос по умолчанию — обзор проекта",
          dev_help.question("/help") == dev_help.DEFAULT_QUESTION
          and "проект" in dev_help.DEFAULT_QUESTION)
    check("путь файла находится в тексте вопроса",
          dev_help.mentions("что делает app/ai/mcp.py и chat.html?")
          == ["app/ai/mcp.py", "chat.html"],
          str(dev_help.mentions("что делает app/ai/mcp.py и chat.html?")))
    check("числа путями не считаются", dev_help.mentions("ставка 2.5 и версия 1.0") == [])

    general = [call["tool"] for call in dev_help.calls()]
    check("общий вопрос: состояние, сводка изменений, файлы, история",
          general == [dev_help.STATUS_TOOL, dev_help.DIFF_TOOL, dev_help.FILES_TOOL,
                      dev_help.LOG_TOOL], str(general))
    for_file = dev_help.calls("app/ai/mcp.py", dev_help.PATH_FILE)
    check("назван файл — спрашиваем ЕГО изменения и содержимое",
          [call["tool"] for call in for_file]
          == [dev_help.STATUS_TOOL, dev_help.DIFF_TOOL, dev_help.SHOW_TOOL]
          and for_file[1]["arguments"]["path"] == "app/ai/mcp.py"
          and for_file[2]["arguments"]["path"] == "app/ai/mcp.py", str(for_file))
    for_dir = dev_help.calls("app/ai", dev_help.PATH_DIR)
    check("назван каталог — спрашиваем его файлы",
          [call["tool"] for call in for_dir]
          == [dev_help.STATUS_TOOL, dev_help.DIFF_TOOL, dev_help.FILES_TOOL]
          and for_dir[2]["arguments"]["path"] == "app/ai", str(for_dir))
    check("вызовов не больше предела",
          all(len(dev_help.calls(*args)) <= dev_help.MAX_CONTEXT_CALLS
              for args in ((), ("app/ai", dev_help.PATH_DIR),
                           ("app/ai/mcp.py", dev_help.PATH_FILE))))

    log_nodes = [
        {"kind": "user", "text": "сделай отчёт"},
        {"kind": "debug", "text": "служебная строка — не разговор"},
        {"kind": "assistant", "text": "готово"},
        {"kind": "user", "text": "🧭 /help — что это за проект"},
    ]
    history = dev_help.history_from_log(log_nodes)
    check("история собирается из журнала и без служебных строк",
          "Пользователь: сделай отчёт" in history and "Ассистент: готово" in history
          and "служебная строка" not in history
          and history.index("сделай отчёт") < history.index("/help"), history)

    only_history = dev_help.messages("вопрос")
    check("без контекста и документации блоков нет",
          [m["role"] for m in only_history] == ["system", "user"], str(only_history))
    full = dev_help.messages("вопрос", profile="ПРОФИЛЬ", doc_block="ФРАГМЕНТЫ",
                             context_block="ДАННЫЕ MCP", history="ИСТОРИЯ")
    check("порядок блоков: профиль → контекст → документация → история → вопрос",
          [m["content"] for m in full[1:5]]
          == ["ПРОФИЛЬ", full[2]["content"], "ФРАГМЕНТЫ", full[4]["content"]]
          and "КОНТЕКСТ ПРОЕКТА" in full[2]["content"]
          and "ИСТОРИЯ РАЗГОВОРА" in full[4]["content"]
          and full[5] == {"role": "user", "content": "вопрос"}, str(full[2:5]))
    check("промпт выбирается по данным: с фрагментами и без",
          dev_help.prompt_for(True) == dev_help.PROMPT
          and dev_help.prompt_for(False) == dev_help.PROMPT_NO_DOCS
          and "НЕ ссылайся на документацию" in dev_help.PROMPT_NO_DOCS)
    check("подсказка про выключенный сервер называет кнопку и причину",
          "MCP" in dev_help.git_hint() and "не подключён" in dev_help.git_hint()
          and "потому что" in dev_help.git_hint("потому что нет node"))


# ---------------------------------------------------------------------------
# [4] Маршрут /api/agent/help (заглушка модели)
# ---------------------------------------------------------------------------
LLM_CALLS = []
PROMPTS = []                       # начала системных промптов всех вызовов
BASE_ID = ""
LLM_ANSWER = (
    "Проект — приложение на FastAPI с агентом и поиском по базам знаний [1]. "
    "Сейчас ветка work/check, изменён app/ai/module.py и добавлен notes.txt."
)


def _metrics(prompt=30, completion=20):
    return {"model": "stub", "elapsed_seconds": 0.01, "prompt_tokens": prompt,
            "completion_tokens": completion, "total_tokens": prompt + completion}


async def fake_call_llm_async(*args, **kwargs):
    """Подмена client.call_llm_async: ответ ассистента, диспетчер, переформулировка."""
    LLM_CALLS.append(kwargs)
    messages = kwargs.get("messages") or []
    system = str(messages[0].get("content") or "") if messages else ""
    user = str(kwargs.get("user_text") or "")
    PROMPTS.append(system[:60])
    if system.startswith("Ты — диспетчер внешних инструментов"):
        # Диспетчер MCP: на вопрос о состоянии работы выбирает git_status.
        return (json.dumps({"calls": [{"server": dev_help.SERVER_ID,
                                       "tool": dev_help.STATUS_TOOL,
                                       "arguments": {}}]}, ensure_ascii=False),
                _metrics(40, 12))
    if "ПОИСКОВЫЙ ЗАПРОС" in system:
        return json.dumps({"query": user[:120]}, ensure_ascii=False), _metrics(10, 5)
    return LLM_ANSWER, _metrics()


# Признак БЛОКА данных MCP (а не упоминания в правилах промпта): по нему видно,
# переданы ли модели сами данные репозитория.
MCP_BLOCK_MARK = mcp_store.BLOCK_HEADER[:60]


def _system_texts(call):
    return "\n".join(str(m.get("content") or "") for m in (call.get("messages") or [])
                     if m.get("role") == "system")


async def run_agent_chat(text, **kwargs):
    """Прогон POST /api/agent/chat без сети: собирает события NDJSON-потока."""
    response = await chat.agent_chat(ChatMessage(content=text, **kwargs))
    events = []
    async for chunk in response.body_iterator:
        for line in str(chunk).splitlines():
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


async def run_help(content, session_id=""):
    """Прогон POST /api/agent/help без сети: собирает события NDJSON-потока."""
    response = await chat.agent_help(ChatMessage(content=content,
                                                 session_id=session_id))
    events = []
    async for chunk in response.body_iterator:
        for line in str(chunk).splitlines():
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def texts(events, kind):
    return [e.get("text", "") for e in events if e.get("type") == kind]


async def section_route():
    print("\n[4] Маршрут /api/agent/help: поток, контекст проекта, источники")
    llm_client.call_llm_async = fake_call_llm_async

    await chat.task_create(chat.TaskCreate(name="Ассистент проекта"))
    task = chat._current_task()
    session = workspace_store.create_session(task)

    # Документация проекта: маленькая база с текстом про два этапа поиска. Порог и
    # фильтр сняты намеренно — проверка мерит ответ и контекст, а не настройку
    # поиска (у неё свои проверки в check_rag.py).
    base = rag.index_files([{"filename": "guide.md", "text": DOCS_TEXT}],
                           name="Документация проекта",
                           profile=chat._current_profile_id())
    global BASE_ID
    BASE_ID = str(base["id"])
    workspace_store.set_rag_enabled(task, [BASE_ID])
    workspace_store.set_rag_search(task, filter=False, min_score=0.0, rerank=False,
                                   rewrite=False)

    # Сервер контекста выключен у проекта — команда обязана сказать об этом.
    LLM_CALLS.clear()
    events = await run_help("/help как устроен поиск по базам знаний?",
                            session_id=str(session["id"]))
    start = [e for e in events if e.get("type") == "help_start"]
    check("без сервера git у проекта: заявлено, что контекста нет",
          bool(start) and start[0]["git"] is False and start[0]["bases"] == ["Документация проекта"],
          str(start))
    hint = " ".join(texts(events, "debug"))
    check("причина названа словами и с кнопкой",
          "MCP" in hint and "не включён у проекта" in hint, hint[:300])
    check("ответ всё равно пришёл (документация отвечает)",
          texts(events, "bot") and LLM_ANSWER in texts(events, "bot")[0])
    check("без контекста репозитория в промпте нет данных MCP",
          MCP_BLOCK_MARK not in _system_texts(LLM_CALLS[-1]))
    check("фрагменты документации в промпте есть",
          "ФРАГМЕНТЫ ИЗ БАЗ ЗНАНИЙ" in _system_texts(LLM_CALLS[-1]))
    check("под ответом строка источников от КОДА",
          "📄 Источники:" in texts(events, "bot")[0]
          and [e for e in events if e.get("type") == "bot"][0].get("sources"))
    check("ответ приходит со своими источниками для карточек",
          bool([e for e in events if e.get("type") == "bot"][0].get("sources")))

    # Теперь сервер git включён у проекта: контекст репозитория ДОЛЖЕН прийти.
    workspace_store.set_mcp_enabled(task, [dev_help.SERVER_ID])
    LLM_CALLS.clear()
    events = await run_help("/help что делает app/ai/module.py?", session_id=str(session["id"]))
    start = [e for e in events if e.get("type") == "help_start"]
    check("с включённым сервером контекст репозитория получен",
          bool(start) and start[0]["git"] is True
          and start[0]["path"] == "app/ai/module.py"
          and start[0]["kind"] == dev_help.PATH_FILE, str(start))
    system = _system_texts(LLM_CALLS[-1])
    check("в промпте ветка, изменения и файл (данные MCP)",
          MCP_BLOCK_MARK in system and BRANCH in system
          and "app/ai/module.py" in system and "new_step" in system, system[:400])
    check("названный файл показан содержимым с номерами строк",
          "1| def step_1" in system, system[-600:])
    check("документация и вопрос дошли до модели",
          ("ФРАГМЕНТЫ ИЗ БАЗ ЗНАНИЙ" in system
           or "БАЗЫ ЗНАНИЙ ПОЛЬЗОВАТЕЛЯ" in system)
          and any(m.get("role") == "user" and "app/ai/module.py" in m.get("content", "")
                  for m in LLM_CALLS[-1]["messages"]), system[:300])
    debug = " ".join(texts(events, "debug"))
    check("видно, какие инструменты вызваны",
          "git_status" in debug and "git_diff" in debug and "git_show_file" in debug,
          debug[:300])
    check("виден расход хода и конец потока",
          [e for e in events if e.get("type") == "usage"]
          and events[-1].get("type") == "done"
          and events[-1]["usage"].get("input") == 30,
          str(events[-1]))
    check("ответ показан (порядок bot → answer_ready → … → done)",
          [e.get("type") for e in events][-4:]
          == ["bot", "answer_ready", "usage", "done"],
          str([e.get("type") for e in events]))

    # Записи: в журнал — да, в память задачи и автомат — нет.
    dialog = session["dialog"]
    log = dialog.get("log") or []
    check("вопрос и ответ попали в ЖУРНАЛ чата",
          any(node.get("kind") == "user" and "/help" in node.get("text", "") for node in log)
          and any(node.get("kind") == "assistant" and LLM_ANSWER in node.get("text", "")
                  for node in log), str(log[-3:])[:300])
    check("узлы журнала хранят источники для карточек",
          any(node.get("kind") == "assistant" and node.get("sources") for node in log))
    check("в ПАМЯТЬ ЗАДАЧИ (messages) команда не пишет",
          not (dialog.get("messages") or []))
    check("автомат задачи не тронут: шагов и плана нет",
          not (dialog.get("state") or {}).get("steps")
          and not (dialog.get("state") or {}).get("plan_pending"))

    # История: следующий вопрос видит предыдущий ответ ассистента.
    LLM_CALLS.clear()
    await run_help("/help а что ещё изменено?", session_id=str(session["id"]))
    system = _system_texts(LLM_CALLS[-1])
    check("история разговора берётся из журнала (прошлый вопрос виден)",
          "ИСТОРИЯ РАЗГОВОРА" in system and "что делает app/ai/module.py?" in system,
          system[-500:])

    # Параметры модели: «Длина» из шапки главнее проекта, проект — главнее константы.
    workspace_store.set_llm_params(task, temperature=0.5, max_tokens=777)
    LLM_CALLS.clear()
    await run_help("/help что это за проект?", session_id=str(session["id"]))
    check("параметры модели проекта действуют (temperature и предел)",
          LLM_CALLS[-1].get("temperature") == 0.5
          and LLM_CALLS[-1].get("max_tokens") == 777,
          str({k: LLM_CALLS[-1].get(k) for k in ("temperature", "max_tokens")}))
    LLM_CALLS.clear()
    await run_help("/help что это за проект?", session_id=str(session["id"]))
    header = await chat.agent_help(ChatMessage(content="/help что это за проект?",
                                               max_tokens=1234,
                                               session_id=str(session["id"])))
    async for _ in header.body_iterator:
        pass
    check("«Длина» из шапки чата главнее проекта", LLM_CALLS[-1].get("max_tokens") == 1234,
          str(LLM_CALLS[-1].get("max_tokens")))
    workspace_store.set_llm_params(task, temperature=None, max_tokens=None)

    # Отказы маршрута: не команда и «команда без проекта».
    check("не команда — отказ 400 с объяснением",
          await _araises(HTTPException, lambda: chat.agent_help(
              ChatMessage(content="обычный вопрос"))))
    # Проекта нет вовсе: у профиля не остаётся ни одной задачи («текущей» тоже —
    # без активной задача выбирается последняя, поэтому мало очистить active).
    saved_tasks = chat._workspace.get("tasks")
    saved_active = dict(chat._workspace.get("active_tasks") or {})
    chat._workspace["tasks"] = []
    chat._workspace["active_tasks"] = {}
    try:
        refused = await _araises(HTTPException, lambda: chat.agent_help(
            ChatMessage(content="/help что это за проект?")))
        check("без проекта — отказ 400 (документация привязана к проекту)", refused)
    finally:
        chat._workspace["tasks"] = saved_tasks
        chat._workspace["active_tasks"] = saved_active


# ---------------------------------------------------------------------------
# [5] Честность при отказах
# ---------------------------------------------------------------------------
async def section_honesty():
    print("\n[5] Честность: базы не подключены, сервер отказал, модель молчит")
    task = chat._current_task()
    session = workspace_store.active_session(chat._workspace, task)

    # 1. Базы знаний выключены — команда работает по состоянию репозитория, и
    #    строки источников (как и ссылок [N]) быть НЕ ДОЛЖНО.
    workspace_store.set_rag_enabled(task, [])
    LLM_CALLS.clear()
    events = await run_help("/help что сейчас изменено?", session_id=str(session["id"]))
    start = [e for e in events if e.get("type") == "help_start"]
    check("без баз знаний заявлено, что документации нет",
          bool(start) and start[0]["bases"] == [] and start[0]["git"] is True,
          str(start))
    answer = texts(events, "bot")[0]
    check("строки источников нет вовсе (искать было негде)",
          "📄 Источники:" not in answer and not answer.count("["), answer)
    check("в промпте промпт БЕЗ правил про документацию",
          dev_help.PROMPT_NO_DOCS in _system_texts(LLM_CALLS[-1])
          and dev_help.PROMPT not in _system_texts(LLM_CALLS[-1]))
    check("контекст репозитория при этом в промпте есть",
          MCP_BLOCK_MARK in _system_texts(LLM_CALLS[-1]))

    # 2. Сервер включён, но работать не может (нет node): причина названа, ответ
    #    всё равно есть — документация остаётся.
    workspace_store.set_rag_enabled(task, [BASE_ID])
    saved_dir = os.environ.get("MCP_SERVERS_DIR")
    os.environ["MCP_SERVERS_DIR"] = os.path.join(_TMP, "no-servers")
    try:
        LLM_CALLS.clear()
        events = await run_help("/help что это за проект?", session_id=str(session["id"]))
        start = [e for e in events if e.get("type") == "help_start"]
        check("недоступный сервер: контекста нет, причина названа",
              bool(start) and start[0]["git"] is False
              and any("не установлен MCP SDK" in text or "не найден" in text
                      for text in texts(events, "debug")),
              " | ".join(texts(events, "debug"))[:300])
        check("ответ по документации всё равно пришёл", bool(texts(events, "bot")))
    finally:
        if saved_dir is None:
            os.environ.pop("MCP_SERVERS_DIR", None)
        else:
            os.environ["MCP_SERVERS_DIR"] = saved_dir
        mcp_store.forget(dev_help.SERVER_ID)

    # 3. Модель молчит: пустой ответ — это ОШИБКА с причиной, а не «ответ без
    #    источников» (иначе строка источников выдавалась бы за ответ).
    async def empty_call(*args, **kwargs):
        LLM_CALLS.append(kwargs)
        return "", _metrics()

    saved_call = llm_client.call_llm_async
    llm_client.call_llm_async = empty_call
    try:
        events = await run_help("/help что это за проект?", session_id=str(session["id"]))
        check("пустой ответ модели — событие error, а не ответ",
              any("не получил ответа от модели" in text for text in texts(events, "error"))
              and not texts(events, "bot"), str(events)[:300])
        check("поток закрыт событием done",
              [e.get("type") for e in events][-1] == "done")
    finally:
        llm_client.call_llm_async = saved_call


# ---------------------------------------------------------------------------
# [6] Прямой ответ, когда в контексте ЕСТЬ данные инструментов
# ---------------------------------------------------------------------------
async def section_direct_with_tools():
    """Вопрос про состояние работы, отвеченный ПРЯМЫМ ответом (не /help).

    Живой случай 12.10: на вопрос «какие изменения сейчас rag_search» агент пошёл
    прямым ответом, а промпт у того пути был «разговор по документам» — краткость
    2–5 предложений и ссылка в КАЖДОМ ответе. Данные git в контексте были, но
    ответ получился общим, а факты из git подписаны номерами фрагментов
    документации, к ним не относящихся. Промпт теперь выбирается ПО ДАННЫМ:
    есть данные инструментов — свой промпт, который требует приводить данные
    полно и запрещает подкреплять их номерами фрагментов.
    """
    print("\n[6] Прямой ответ с данными MCP: промпт выбирается по данным")
    task = chat._current_task()
    workspace_store.set_mcp_enabled(task, [dev_help.SERVER_ID])
    LLM_CALLS.clear()
    PROMPTS.clear()
    events = await run_agent_chat("На какой ветке идёт работа?")
    answer_prompts = [p for p in PROMPTS if p.startswith("Ты отвечаешь на запрос")
                      or p.startswith("Ты ведёшь РАЗГОВОР")]
    check("модель получила промпт ПО ДАННЫМ ИНСТРУМЕНТОВ",
          any(p.startswith("Ты отвечаешь на запрос") for p in answer_prompts),
          str(answer_prompts)[:160])
    check("промпт требует приводить данные полно, а не общей фразой",
          dev_help_prompt_has("ПОЛНО") and dev_help_prompt_has("НЕ подкрепляй"))
    # ВАЖНО: последний вызов хода — служебный (обновление памяти задачи), а не
    # ответ. Контекст ответа смотрим у ТОГО вызова, чей промпт — ответный.
    answer_calls = [call for call in LLM_CALLS
                    if str((call.get("messages") or [{}])[0].get("content") or "")
                    .startswith(("Ты отвечаешь на запрос", "Ты ведёшь РАЗГОВОР"))]
    system = _system_texts(answer_calls[-1]) if answer_calls else ""
    check("данные git в контексте есть (ветка из временного репозитория)",
          MCP_BLOCK_MARK in system and BRANCH in system, system[:200])
    check("в контексте есть и фрагменты документации",
          "ФРАГМЕНТЫ ИЗ БАЗ ЗНАНИЙ" in system or "БАЗЫ ЗНАНИЙ ПОЛЬЗОВАТЕЛЯ" in system)
    bot = ([e for e in events if e.get("type") == "bot"] or [{}])[-1]
    check("ответ пришёл", bool(texts(events, "bot")), str(events)[:200])
    # ГЛАВНОЕ ПРАВИЛО СВЯЗКИ (правка 12.10): ответ построен по данным инструментов и
    # на документацию не сослался — списка «📄 Источники:» под ним быть НЕ должно.
    # Раньше он прикладывался всегда, и ответ по git получал под собой адреса
    # разделов документации, которых в нём не было.
    check("к ответу по данным инструментов список источников НЕ приложен",
          "📄 Источники:" not in str(bot.get("text") or ""),
          str(bot.get("text"))[-200:])
    # Источников нет и в событии: ответ по данным инструментов, документация не
    # использована — карточек, которые выглядели бы источниками, тоже нет.
    check("карточек источников у такого ответа тоже нет",
          not bot.get("sources"), str(bot.get("sources"))[:120])
    # А ЕСЛИ МОДЕЛЬ СОСЛАЛАСЬ — строка перечисляет ТОЛЬКО сославшиеся номера.
    cited = rag_dialog.ensure_sources("Смотри правило [1].",
                                      [{"source": "guide.md", "number": 7,
                                        "base_score": 1.0}],
                                      searched=1, tools=True)
    check("со ссылкой в ответе строка источников появляется и называет её номер",
          "📄 Источники: [1] guide.md" in cited, cited)


def dev_help_prompt_has(needle):
    """Есть ли фраза в промпте «по данным инструментов» (тот, что взял ответ)."""
    from app.ai import rag_dialog as dialog
    return needle in dialog.ANSWER_PROMPT_WITH_TOOLS


# ---------------------------------------------------------------------------
def main():
    print("Проверка ассистента разработчика (команда /help) и MCP-сервера git.")
    print("Временный каталог: " + _TMP)

    global REPO
    REPO = make_repo()
    os.environ["GIT_REPO_DIR"] = REPO
    print("Временный репозиторий: " + REPO + " (ветка " + BRANCH + ")")

    section_registry()
    section_tools()
    section_command()
    asyncio.run(section_route())
    asyncio.run(section_direct_with_tools())
    asyncio.run(section_honesty())

    print("")
    if FAILURES:
        print("ПРОВАЛЫ (%d): %s" % (len(FAILURES), "; ".join(FAILURES)))
        return 1
    print("Все проверки пройдены.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
