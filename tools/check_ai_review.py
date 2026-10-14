"""Самопроверка AI-РЕВЬЮ КОДА — команды `/ai_review*` и пайплайн на pull request.

Запуск (сеть и ключ модели НЕ нужны; Node.js нужен — на нём MCP-сервер git):

    ./venv/bin/python tools/check_ai_review.py

ЧТО ПРОВЕРЯЕТСЯ

  [1] Разбор команд: `/ai_review` (с базовой веткой и пожеланием), отдельные
      команды включения и выключения, и то, что мусор («/ai_reviewer»,
      «/ai_review_enable лишнее», «/ai_review --help») командой НЕ считается.
  [2] Настройка ПРОЕКТА (`task["ai_review"]`): по умолчанию режим выключен,
      включение и выключение записываются, настройка ПЕРЕЖИВАЕТ запись файла
      (без строки в `_normalize_task` она терялась бы), битое значение из файла
      зажимается, а не ломает приложение.
  [3] Разбор ответов сервера `git`: список изменённых файлов (машинные строки),
      заголовок с веткой и базой, окна diff'а (начало, продолжение, конец) и
      честное «изменений нет».
  [4] Отбор контекста КОДОМ: имена из правки (только определения), целые
      небольшие файлы, связанные файлы (где встречаются изменённые имена),
      документация репозитория и охрана путей — та же, что у сервера (секреты,
      `data/`, выход за корень).
  [5] Промпты и сообщения модели: три раздела, правила «только по данным»,
      отдельный промпт без документации (и без правил про ссылки [N]), пустые
      блоки не добавляются, пожелание к ревью доходит.
  [6] Отчёт: шапку с фактами собирает КОД (ветка, база, файлы, строки), обрезка
      данных названа, недостающие разделы не замаскированы.
  [7] Прогон (заглушки инструментов, документации и модели): порядок работы,
      diff ОКНАМИ до конца файла, включение и выключение режима без вызова
      модели, причина при выключенном режиме и при недоступном сервере, честный
      отказ, когда модель молчит.
  [8] Маршрут POST /api/agent/review — на НАСТОЯЩЕМ временном репозитории и
      настоящем MCP-сервере git (заглушка только у модели): поток событий, отчёт
      в журнале чата и НЕзапись в память задачи, снимок GET /api/agent/review.
  [9] Вход пайплайна `tools/ai_review.py` отдельным процессом: отчёт файлом и в
      выводе, коды возврата (настройка против сбоя), «сухой» прогон без модели,
      режимы `--rag on|auto|off` (включая сломанный каталог индекса).
  [10] RAG-ИНДЕКС КОДА ДЛЯ РЕВЬЮ (`app/ai/ai_review_rag.py`): что индексируется,
      свежесть по коммиту и по отпечатку файлов, пересборка и уборка прежних
      индексов, своя база человека с тем же именем (её не трогают), поиск нужных
      чанков без повторов.
  [10б] ФОНОВАЯ ИНДЕКСАЦИЯ — то, что запускает команда включения режима: задача
      RAG, маркер по её окончании, повторный запуск не собирает второй индекс.
  [11] Проводка: потоковый маршрут в `main.py`, перехват команд в интерфейсе,
      файл пайплайна `.github/workflows/ai-review.yml` (в том числе self-hosted
      раннер) и скрипт `tools/review_runner.sh`.

Рабочие данные не трогаются: workspace, профили, история агента и базы знаний
пишутся во временный каталог (переменные AGENT_*_FILE и RAG_DIR выставляются ДО
импорта приложения), а репозиторий создаётся ВРЕМЕННЫЙ — сам проект проверка не
меняет.
"""

import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Изоляция данных: всё пишем во временный каталог ------------------------
_TMP = tempfile.mkdtemp(prefix="aireview-check-")
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

from app import config  # noqa: E402
from app.ai import ai_review  # noqa: E402
from app.ai import ai_review_rag  # noqa: E402
from app.ai import ai_review_run  # noqa: E402
from app.ai import client as llm_client  # noqa: E402
from app.ai import mcp as mcp_store  # noqa: E402
from app.ai import rag  # noqa: E402
from app.ai import rag_jobs  # noqa: E402
from app.ai import rag_store  # noqa: E402
from app.ai import rag_search  # noqa: E402
from app.ai import workspace as workspace_store  # noqa: E402
from app.routers import chat  # noqa: E402
import ai_review as ai_review_cli  # noqa: E402  (tools/ai_review.py — вход пайплайна)
from app.schemas import ChatMessage  # noqa: E402

FAILURES = []
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check(name, condition, detail=""):
    """Одна проверка: печатает результат и копит провалы."""
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# Временный репозиторий: на нём проверяются инструменты, контекст и маршрут
# ---------------------------------------------------------------------------
GIT_ENV = {
    "GIT_AUTHOR_NAME": "Проверка",
    "GIT_AUTHOR_EMAIL": "check@example.invalid",
    "GIT_COMMITTER_NAME": "Проверка",
    "GIT_COMMITTER_EMAIL": "check@example.invalid",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
}
BASE = "main"
BRANCH = "work/review"
REPO = ""

MODULE_TEXT = "\n".join(
    "def step_%d():\n    return %d" % (number, number) for number in range(1, 21)
)
# Файл, который ПОЛЬЗУЕТСЯ тем, что добавляет ветка: по нему проверяется
# лексический отбор связанных файлов («кто ещё это вызывает»).
OTHER_TEXT = (
    "from app.ai.module import step_1, review_target\n\n\n"
    "def run_all():\n"
    "    return step_1() + review_target({'result': 1})\n"
)
# Большой файл: он НЕ должен попадать в контекст целиком (в модель влезает не
# всё, и лучше показать пять файлов целиком, чем один наполовину).
BIG_TEXT = "\n".join("value_%d = %d" % (number, number) for number in range(1, 501))


def git_run(repo, *args, ok=True):
    """Команда git во временном репозитории (чужая настройка не читается)."""
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True,
                            text=True, env={**os.environ, **GIT_ENV})
    if ok and result.returncode != 0:
        raise RuntimeError("git %s: %s" % (" ".join(args), result.stderr.strip()))
    return result


def make_repo():
    """Временный репозиторий: ветка `main` и ветка работы `work/review`.

    Нужны ОБЕ ветки: ревью сравнивает ветку с базовой, а не с рабочей копией
    (у pull request рабочая копия чистая, и обычный diff показал бы пустоту).
    """
    repo = os.path.join(_TMP, "repo")
    os.makedirs(os.path.join(repo, "app", "ai"))
    os.makedirs(os.path.join(repo, "docs"))
    os.makedirs(os.path.join(repo, "data"))
    files = {
        "app/ai/module.py": MODULE_TEXT + "\n",
        "app/ai/other.py": OTHER_TEXT,
        "app/ai/big.py": BIG_TEXT + "\n",
        "docs/guide.md": "# Руководство проекта\n\nПоиск по базам идёт в два этапа.\n",
        "README.md": "# Проект проверки\n\nКарта модулей проекта.\n",
        ".gitignore": ".env\ndata/\n",
        ".env": "SECRET_KEY=не-показывать\n",
        "data/history.json": "{}",
    }
    for rel, text in files.items():
        with open(os.path.join(repo, rel), "w", encoding="utf-8") as fh:
            fh.write(text)
    git_run(repo, "init", "-b", BASE)
    git_run(repo, "add", ".")
    git_run(repo, "commit", "-m", "первый коммит: каркас проекта")
    # ВЕТКА РАБОТЫ: правка модуля, новый файл и правка документации.
    git_run(repo, "checkout", "-b", BRANCH)
    with open(os.path.join(repo, "app", "ai", "module.py"), "a", encoding="utf-8") as fh:
        fh.write("\n\ndef review_target(value):\n"
                 "    return value['result']\n")
    with open(os.path.join(repo, "app", "ai", "fresh.py"), "w", encoding="utf-8") as fh:
        fh.write("def fresh_step():\n    return 'новый файл ветки'\n")
    with open(os.path.join(repo, "docs", "guide.md"), "a", encoding="utf-8") as fh:
        fh.write("\n## Ревью\n\nФункция review_target возвращает поле result.\n")
    git_run(repo, "add", ".")
    git_run(repo, "commit", "-m", "ветка: разбор review_target и новый файл")
    return repo


def call_tool(tool, arguments=None):
    """Вызов инструмента через НАСТОЯЩИЙ клиент приложения (app/ai/mcp.py)."""
    results = mcp_store.run_calls([{"server": ai_review.SERVER_ID, "tool": tool,
                                    "arguments": arguments or {}}])
    return results[0]


# ---------------------------------------------------------------------------
# [1] Команды
# ---------------------------------------------------------------------------
def section_commands():
    print("\n[1] Разбор команд AI-ревью")
    check("«/ai_review» — команда разбора",
          ai_review.parse("/ai_review") == ("review", "", ""))
    check("базовая ветка берётся из команды",
          ai_review.parse("/ai_review main")[1] == "main")
    check("пожелание к ревью берётся из команды",
          ai_review.parse("/ai_review main проверь ошибки")[2] == "проверь ошибки")
    check("включение — ОТДЕЛЬНАЯ команда, а не разбор с аргументом",
          ai_review.parse("/ai_review_enable") == ("enable", "", "")
          and ai_review.parse("/ai_review_disable") == ("disable", "", ""))
    check("у включения и выключения аргументов быть не может",
          ai_review.parse("/ai_review_enable лишнее") == ("", "", ""))
    check("«/ai_reviewer» командой не считается",
          not ai_review.is_command("/ai_reviewer")
          and not ai_review.is_command("/ai_review_all"))
    check("опция вместо ветки («/ai_review --help») — не команда",
          not ai_review.is_command("/ai_review --help"))
    check("обычный текст командой не считается",
          not ai_review.is_command("сделай ревью моей ветки"))
    check("регистр не важен", ai_review.parse("/AI_REVIEW Main")[1] == "Main")
    check("состояние включённого режима называет команду разбора и выключение",
          ai_review.COMMAND in ai_review.state_text(True)
          and ai_review.COMMAND_DISABLE in ai_review.state_text(True))
    check("состояние выключенного режима называет команду включения",
          ai_review.COMMAND_ENABLE in ai_review.state_text(False))
    check("отказ выключенного режима называет причину и как включить",
          ai_review.COMMAND_ENABLE in ai_review.DISABLED_TEXT
          and "выключен" in ai_review.DISABLED_TEXT)
    check("база берётся из команды, а без неё — из настройки проекта",
          ai_review.base_of("/ai_review develop", "main") == "develop"
          and ai_review.base_of("/ai_review", "main") == "main")


# ---------------------------------------------------------------------------
# [2] Настройка проекта
# ---------------------------------------------------------------------------
def section_setting():
    print("\n[2] Настройка режима у ПРОЕКТА (task[\"ai_review\"])")
    task = {"id": "t-check", "name": "Проект проверки", "sessions": [],
            "active_session": None}
    check("по умолчанию режим ВЫКЛЮЧЕН",
          workspace_store.ai_review_enabled(task) is False)
    state = workspace_store.set_ai_review(task, enabled=True, base="main")
    check("включение записано вместе с базовой веткой",
          state == {"enabled": True, "base": "main"}, repr(state))
    check("база меняется, а состояние остаётся",
          workspace_store.set_ai_review(task, base="develop")
          == {"enabled": True, "base": "develop"})
    normalized = workspace_store.normalize_workspace({"tasks": [task]})
    saved = normalized["tasks"][0]
    check("настройка ПЕРЕЖИВАЕТ запись файла (строка в _normalize_task)",
          saved.get("ai_review") == {"enabled": True, "base": "develop"},
          repr(saved.get("ai_review")))
    check("выключение записано",
          workspace_store.set_ai_review(saved, enabled=False)["enabled"] is False)
    broken = workspace_store.normalize_workspace({"tasks": [
        {"id": "t-2", "name": "Битый", "ai_review": {"enabled": "как-нибудь",
                                                     "base": 42}}]})["tasks"][0]
    check("битое значение из файла зажимается, а не ломает приложение",
          broken["ai_review"] == {"enabled": True, "base": "42"},
          repr(broken["ai_review"]))
    plain = workspace_store.normalize_workspace({"tasks": [
        {"id": "t-3", "name": "Короткий", "ai_review": True}]})["tasks"][0]
    check("запасной формат (просто «включено») понимается",
          plain["ai_review"]["enabled"] is True)
    long_base = workspace_store.normalize_workspace({"tasks": [
        {"id": "t-4", "name": "Длинный",
         "ai_review": {"enabled": True, "base": "x" * 500}}]})["tasks"][0]
    check("имя базовой ветки ограничено по длине",
          len(long_base["ai_review"]["base"]) == workspace_store.AI_REVIEW_BASE_MAX)
    check("у другого проекта настройка СВОЯ",
          workspace_store.ai_review_enabled({"id": "t-5", "name": "Другой"}) is False)


# ---------------------------------------------------------------------------
# [3] Разбор ответов сервера
# ---------------------------------------------------------------------------
FILES_TEXT = (
    "Изменённые файлы ветки work/review против main (HEAD a1b2c3d): всего 3.\n"
    "M\t+14\t-6\tapp/ai/module.py\n"
    "A\t+3\t-0\tapp/ai/fresh.py\n"
    "M\t+2\t-1\t.env.example\n"
    "…показаны первые 3 из 3 (сузьте path или поднимите limit)")


def section_server_answers():
    print("\n[3] Разбор ответов сервера git (список файлов и окна diff)")
    files = ai_review.parse_files(FILES_TEXT)
    check("файлы разобраны из МАШИННЫХ строк ответа",
          [item["path"] for item in files]
          == ["app/ai/module.py", "app/ai/fresh.py", ".env.example"],
          repr(files))
    check("числа строк взяты из ответа",
          files[0]["added"] == 14 and files[0]["removed"] == 6
          and files[1]["status"] == "A")
    check("закрытый путь помечен (diff по нему не запрашивается)",
          files[2]["closed"] is True and files[0]["closed"] is False)
    check("заголовок ответа разобран: ветка, база, HEAD",
          ai_review.parse_files_header(FILES_TEXT)
          == {"branch": "work/review", "base": "main", "head": "a1b2c3d"})
    check("«изменений нет» распознано отдельно от пустого ответа",
          ai_review.changes_absent("Изменений нет: ветка не отличается от main.")
          and not ai_review.changes_absent(""))
    check("пустой список — это «изменений нет», а не файлы",
          ai_review.parse_files("Изменений нет: ветка не отличается.") == [])
    check("строка объёма считает строки и называет двоичные файлы",
          ai_review.files_line([{"path": "a.py", "added": 3, "removed": 1},
                                {"path": "b.png", "added": None, "removed": None}])
          == "Изменено файлов: 2 (+3 −1); без построчного счёта (двоичные): b.png.")

    window = ("Diff ветки против main, app/ai/module.py — строки 1–60 из 120; "
              "дальше from_line=61:\n"
              "diff --git a/app/ai/module.py b/app/ai/module.py\n+новая строка")
    parsed = ai_review.parse_window(window)
    check("окно diff'а разобрано: границы, всего строк, продолжение",
          parsed["start"] == 1 and parsed["end"] == 60 and parsed["total"] == 120
          and parsed["next_from"] == 61 and parsed["done"] is False, repr(parsed))
    check("тело окна — только сам diff, без заголовка",
          parsed["body"].startswith("diff --git")
          and "новая строка" in parsed["body"])
    last = ai_review.parse_window(
        "Diff ветки против main — строки 61–120 из 120; это КОНЕЦ diff:\n+хвост")
    check("последнее окно помечено концом (продолжения нет)",
          last["done"] is True and last["next_from"] == 0)
    check("ответ не про окно не выдаётся за окно",
          ai_review.parse_window("Изменений нет")["done"] is False
          and ai_review.parse_window("Изменений нет")["body"] == "")

    calls = ai_review.discovery_calls("main", "app/ai", limit=7)
    check("сначала состояние и список файлов, с базой, путём и пределом",
          [call["tool"] for call in calls]
          == [ai_review.STATUS_TOOL, ai_review.FILES_TOOL]
          and calls[1]["arguments"] == {"limit": 7, "base": "main", "path": "app/ai"},
          repr(calls))
    one = ai_review.diff_call("app/ai/module.py", "main", from_line=61)
    check("вызов diff'а — окном по строкам, с базой и путём",
          one["tool"] == ai_review.DIFF_TOOL
          and one["arguments"]["from_line"] == 61
          and one["arguments"]["base"] == "main"
          and one["arguments"]["path"] == "app/ai/module.py", repr(one))


# ---------------------------------------------------------------------------
# [4] Отбор контекста кодом
# ---------------------------------------------------------------------------
def section_context():
    print("\n[4] Отбор контекста: код и документация репозитория (правит КОД)")
    check("секреты закрыты так же, как у сервера",
          ai_review.is_closed(".env") and ai_review.is_closed(".env.example")
          and ai_review.is_closed("auth_secret") and ai_review.is_closed("keys.pem"))
    check("служебные каталоги закрыты",
          ai_review.is_closed("data/history.json")
          and ai_review.is_closed("node_modules/lib.js")
          and not ai_review.is_closed("app/ai/module.py"))
    check("выход за корень репозитория закрыт", ai_review.is_closed("../secret.py"))

    diff = ("+++ b/app/ai/module.py\n"
            "@@ -1,3 +1,4 @@\n"
            "-def old_target():\n"
            "+def review_target(value):\n"
            "+class ReviewThing:\n"
            "+    text = value\n"
            "+    data = value\n")
    names = ai_review.symbols(diff)
    check("имена взяты из ОПРЕДЕЛЕНИЙ, а не из любых строк правки",
          names == ["old_target", "review_target", "ReviewThing"], repr(names))
    check("служебные слова именами не считаются",
          "text" not in names and "data" not in names)

    whole, taken = ai_review.whole_files_block(
        REPO, [{"path": "app/ai/module.py"}, {"path": "app/ai/big.py"}])
    check("небольшой файл отдан целиком и с номерами строк",
          taken == ["app/ai/module.py"] and "1  def step_1" in whole, repr(taken))
    check("большой файл целиком в контекст не берётся",
          "app/ai/big.py" not in taken)

    related, related_names = ai_review.related_block(
        REPO, ["app/ai/module.py"], ["review_target", "step_1"])
    check("связанный файл найден по изменённому имени",
          "app/ai/other.py" in related_names, repr(related_names))
    check("в связанных файлах показано место вызова с номерами строк",
          "step_1" in related and re.search(r"\d+\s+def run_all", related) is not None,
          related[:200])
    check("изменённый файл в связанные не попадает", "app/ai/module.py" not in related_names)

    docs, docs_names = ai_review.docs_block(REPO, ["review_target"], [])
    check("документация отобрана по тем же именам",
          "docs/guide.md" in docs_names, repr(docs_names))
    check("в документации видно упоминание, а не файл целиком",
          "review_target" in docs)

    info = ai_review.context_block(REPO, [{"path": "app/ai/module.py"}],
                                   ["review_target"], ["app/ai/module.py"])
    check("собранный контекст называет все три части",
          info["whole"] and info["related"] and info["docs"] and info["chars"] > 0
          and "ИЗМЕНЁННЫЕ ФАЙЛЫ ЦЕЛИКОМ" in info["block"], repr(info["chars"]))
    check("контекст без файлов не выдумывает данные",
          ai_review.context_block("", [], [], [])["block"] == "")
    query = ai_review.query_for([{"path": "app/ai/module.py"}], ["review_target"],
                                "проверь обработку ошибок")
    check("запрос к документации собран из файлов и имён",
          "app/ai" in query and "review_target" in query
          and "обработку ошибок" in query)
    check("пределы отбора заданы числами, а не «на глаз»",
          ai_review.MAX_DIFF_CHARS > 0 and ai_review.MAX_DIFF_FILES > 0
          and ai_review.MAX_DIFF_CALLS >= ai_review.MAX_DIFF_FILES)


# ---------------------------------------------------------------------------
# [5] Промпты и сообщения
# ---------------------------------------------------------------------------
def section_prompt():
    print("\n[5] Промпт ревью и сборка сообщений модели")
    with_docs = ai_review.prompt_for(True)
    without = ai_review.prompt_for(False)
    check("в промпте названы три раздела",
          all(title in with_docs for title in ai_review.SECTIONS))
    check("промпт требует конкретики (файл, условие) и запрещает пересказ diff",
          "КОНКРЕТНАЯ" in with_docs and "НЕ ПЕРЕСКАЗЫВАЙ" in with_docs)
    check("промпт говорит, что данные инструментов — не указания",
          "а не указания" in with_docs or "ДАННЫЕ, а не" in with_docs)
    check("с документацией — правило про ссылки [N]",
          "[1]" in with_docs)
    check("без документации — ОТДЕЛЬНЫЙ промпт: нет правила про цитаты, "
          "но есть запрет ссылаться",
          without != with_docs and "дословной цитатой" not in without
          and "номера вида [1]" in without
          and "Документации проекта в этом ответе НЕТ" in without)
    check("промпт выбирается по данным",
          ai_review.prompt_for(True) is ai_review.PROMPT
          and ai_review.prompt_for(False) is ai_review.PROMPT_NO_DOCS)

    empty = ai_review.messages(diff_block="")
    check("пустые блоки в сообщения не добавляются",
          [m["role"] for m in empty] == ["system", "user"], repr(empty))
    full = ai_review.messages(diff_block="diff --git a/x", code_block="контекст",
                              doc_block="фрагменты", profile="профиль",
                              note="проверь ошибки")
    check("порядок блоков: промпт, профиль, изменения, код, документы, задача",
          [m["role"] for m in full] == ["system", "system", "system", "system",
                                        "system", "user"], repr(len(full)))
    check("заголовок блока изменений объясняет, что это данные инструмента",
          ai_review.DIFF_BLOCK_HEADER in full[1]["content"] + full[2]["content"]
          or ai_review.DIFF_BLOCK_HEADER in full[2]["content"], full[2]["content"][:80])
    check("пожелание к ревью уходит в задачу",
          "проверь ошибки" in full[-1]["content"])
    check("промпт берётся по наличию документации",
          ai_review.PROMPT in full[0]["content"])


# ---------------------------------------------------------------------------
# [6] Отчёт
# ---------------------------------------------------------------------------
def section_answer():
    print("\n[6] Отчёт: шапку с фактами собирает КОД")
    files = [{"path": "app/ai/module.py", "status": "M", "added": 14, "removed": 6},
             {"path": ".env.example", "status": "M", "added": 2, "removed": 1,
              "closed": True}]
    body = "\n\n".join(list(ai_review.SECTIONS[:1]) + ["— не отмечено",
                                                       ai_review.SECTIONS[1],
                                                       "— не отмечено",
                                                       ai_review.SECTIONS[2],
                                                       "— не отмечено"])
    text = ai_review.answer(body, branch="work/review", base="main", head="a1b2c3d",
                            files=files, reviewed=1, context_info={"whole": ["a"],
                                                                   "related": ["b"]},
                            model="stub-model", provider="remote")
    check("шапка называет ветку, базу и HEAD из ответов сервера",
          "ветка `work/review`" in text and "против `main`" in text
          and "HEAD `a1b2c3d`" in text)
    check("объём и что смотрел ревьюер — фактами",
          "Изменено файлов: 2 (+16 −7)" in text and "Что смотрел ревьюер" in text)
    check("закрытый файл помечен и не входит в знаменатель проверенных",
          "🔒" in text and "Проверено diff'ом: 1 из 1" in text, text[:400])
    check("подпись называет модель и источник",
          "stub-model" in text and "источник: remote" in text)
    check("все три раздела на месте — предупреждения нет",
          "не разложила ответ" not in text)
    partial = ai_review.answer("просто текст без разделов", files=files, reviewed=1,
                               truncated=ai_review.truncation_note(
                                   total_files=3, reviewed=1, diff_chars=100,
                                   budget=100, failed=2))
    check("недостающие разделы названы, а не замаскированы",
          "не разложила ответ по разделам" in partial
          and ai_review.SECTION_BUGS in partial)
    check("обрезка данных названа тремя разными причинами",
          "НЕ ВСЕ изменения" in partial and "diff оборван по объёму" in partial
          and "инструмент отказал 2 раз" in partial, partial[:400])
    check("без обрезки предупреждения нет",
          ai_review.truncation_note(total_files=1, reviewed=1, diff_chars=10,
                                    budget=100) == "")
    check("перечисление файлов не выдаёт часть за целое",
          "…и ещё 2 файла" in ai_review.files_block(
              files + [{"path": "c.py", "status": "A", "added": 1, "removed": 0}],
              reviewed=2, limit=1))


# ---------------------------------------------------------------------------
# [7] Прогон на заглушках
# ---------------------------------------------------------------------------
WINDOW_LINES = 60
DIFF_TOTAL = 200


def stub_window(path, start, end, total):
    """Ответ инструмента так, как его пишет сервер (заголовок + строки).

    В первом окне стоит ОПРЕДЕЛЕНИЕ функции: по нему ревью достаёт имена правки и
    подбирает связанные файлы — так проверяется весь путь отбора контекста, а не
    только его чтение.
    """
    more = ("; дальше from_line=%d" % (end + 1)) if end < total else "; это КОНЕЦ diff"
    lines = ["+строка %d" % number for number in range(start, end + 1)]
    if start == 1:
        lines[0] = "+def review_target(value):"
    body = "\n".join(lines)
    return ("Diff ветки против main, %s — строки %d–%d из %d%s:\n%s"
            % (path, start, end, total, more, body))


STUB_FILES = (
    "Изменённые файлы ветки work/review против main (HEAD a1b2c3d): всего 2.\n"
    "M\t+200\t-0\tapp/ai/module.py\n"
    "A\t+3\t-0\tapp/ai/fresh.py")
STUB_REVIEW = (
    "Изменён модуль разбора: правка добавляет функцию review_target.\n\n"
    + ai_review.SECTION_BUGS + "\n"
    "- `app/ai/module.py`: при пустом поле `result` вернётся ошибка ключа — "
    "проверьте вход.\n\n"
    + ai_review.SECTION_ARCH + "\n"
    "- Рядом с `step_1` появился второй путь разбора: правило проекта — один вход.\n\n"
    + ai_review.SECTION_TIPS + "\n"
    "- Дописать проверку поля `result` в `review_target`.")


class Stub:
    """Заглушки внешнего мира для прогона: инструменты, модель, документация."""

    def __init__(self, files_text=STUB_FILES, review=STUB_REVIEW,
                 fail_diff=False, empty_answer=False):
        self.files_text = files_text
        self.review = review
        self.fail_diff = fail_diff
        self.empty_answer = empty_answer
        self.calls = []
        self.messages = []
        self.queries = []
        self.enabled_calls = []

    async def call_tools(self, calls, limit=None):
        out = []
        for call in list(calls)[:limit if limit else len(calls)]:
            self.calls.append(dict(call))
            tool = call.get("tool")
            args = call.get("arguments") or {}
            if tool == ai_review.STATUS_TOOL:
                out.append({"tool": tool, "ok": True, "text": "Ветка: work/review",
                            "server_name": "Git"})
                continue
            if tool == ai_review.FILES_TOOL:
                out.append({"tool": tool, "ok": True, "text": self.files_text,
                            "server_name": "Git"})
                continue
            if self.fail_diff:
                out.append({"tool": tool, "ok": False, "server_name": "Git",
                            "error": "git не смог выполнить «git diff»"})
                continue
            start = int(args.get("from_line") or 1)
            end = min(DIFF_TOTAL, start + WINDOW_LINES - 1)
            out.append({"tool": tool, "ok": True, "server_name": "Git",
                        "text": stub_window(args.get("path") or "app/ai/module.py",
                                            start, end, DIFF_TOTAL)})
        return out

    async def ask(self, messages, max_tokens, temperature, timeout):
        self.messages = messages
        if self.empty_answer:
            return "", {}
        return self.review, {"prompt_tokens": 10, "completion_tokens": 5}

    async def documents(self, query):
        self.queries.append(query)
        return {"block": "ФРАГМЕНТЫ ИЗ БАЗ ЗНАНИЙ (1): guide.md",
                "hits": [{"address": "guide.md#1"}],
                "sources": [{"address": "guide.md#1"}], "searched": 1, "note": ""}

    async def set_enabled(self, value):
        self.enabled_calls.append(bool(value))
        return {"enabled": bool(value), "base": "main"}


def make_deps(stub, **kwargs):
    """Deps прогона на заглушках (без MCP, без модели, без сети)."""
    params = {"call_tools": stub.call_tools, "ask": stub.ask,
              "repo_dir": REPO, "enabled": True, "documents": stub.documents,
              "set_enabled": stub.set_enabled, "model": "stub-model",
              "provider": "remote"}
    params.update(kwargs)
    return ai_review_run.Deps(**params)


async def collect(deps, **kwargs):
    return [event async for event in ai_review_run.run(deps, **kwargs)]


def texts(events, kind):
    return [str(event.get("text") or "") for event in events
            if event.get("type") == kind]


async def section_run():
    print("\n[7] Прогон ревью на заглушках (инструменты, документация, модель)")

    # ВЫКЛЮЧЕННЫЙ РЕЖИМ: работа не начинается, причина названа.
    stub = Stub()
    events = await collect(make_deps(stub, enabled=False))
    check("выключенный режим: отказ с причиной и без обращений к инструментам",
          texts(events, "error") and ai_review.COMMAND_ENABLE in texts(events, "error")[0]
          and stub.calls == [] and not stub.messages)
    check("состояние режима видно и в событии разбора",
          [e for e in events if e.get("type") == "review_start"][0]["enabled"] is False)

    # НЕДОСТУПНЫЙ СЕРВЕР: причина называется словами, инструменты не зовутся.
    stub = Stub()
    events = await collect(make_deps(stub, server_reason="сервер не включён у проекта"))
    check("недоступный сервер git: причина названа, вызовов нет",
          "MCP" in texts(events, "error")[0] and stub.calls == [])

    # ВКЛЮЧЕНИЕ И ВЫКЛЮЧЕНИЕ: модель и инструменты НЕ участвуют.
    stub = Stub()
    events = await collect(make_deps(stub), mode=ai_review.MODE_ENABLE)
    check("включение режима записано и подтверждено репликой",
          stub.enabled_calls == [True] and "включён" in texts(events, "bot")[0]
          and stub.messages == [] and stub.calls == [])
    events = await collect(make_deps(stub), mode=ai_review.MODE_DISABLE)
    check("выключение режима записано и подтверждено репликой",
          stub.enabled_calls == [True, False] and "выключен" in texts(events, "bot")[0])
    events = await collect(ai_review_run.Deps(stub.call_tools, stub.ask),
                           mode=ai_review.MODE_ENABLE)
    check("без настройки проекта переключать нечем — причина названа",
          texts(events, "error") and "переменная" in texts(events, "error")[0])

    # ОБЫЧНЫЙ ПРОГОН.
    stub = Stub()
    events = await collect(make_deps(stub), base="main")
    start = [e for e in events if e.get("type") == "review_start"]
    check("разбор начат с объявлением режима и базы",
          start and start[0]["enabled"] is True and start[0]["base"] == "main")
    check("первый вызов — состояние и список файлов ветки, модель НЕ звалась до данных",
          [call["tool"] for call in stub.calls[:2]]
          == [ai_review.STATUS_TOOL, ai_review.FILES_TOOL], repr(stub.calls[:2]))
    windows = [call for call in stub.calls if call["tool"] == ai_review.DIFF_TOOL]
    check("diff читается ОКНАМИ до конца файла, а не одним куском",
          len(windows) >= 4 and windows[0]["arguments"]["from_line"] == 1
          and windows[1]["arguments"]["from_line"] == WINDOW_LINES + 1,
          repr([w["arguments"].get("from_line") for w in windows]))
    check("пределы вызовов соблюдены",
          len(stub.calls) <= 2 + ai_review.MAX_DIFF_CALLS + 2, str(len(stub.calls)))
    debug = " ".join(texts(events, "debug"))
    check("ход работы назван словами (инструменты, файлы, объём)",
          "MCP" in debug and "Diff прочитан" in debug and "Изменено файлов" in debug,
          debug[:300])
    check("документация ищется запросом, собранным из файлов и имён",
          stub.queries and "app/ai" in stub.queries[0], repr(stub.queries))

    system = "\n".join(str(m.get("content") or "") for m in stub.messages
                       if m.get("role") == "system")
    check("в модель ушёл diff ЦЕЛИКОМ (обе половины окна)",
          "+строка 1" in system and "+строка %d" % DIFF_TOTAL in system
          and "+строка 120" in system)
    check("в модель ушёл контекст кода репозитория (связанный файл)",
          "app/ai/other.py" in system and "run_all" in system)
    check("в модель ушли фрагменты документации",
          "ФРАГМЕНТЫ ИЗ БАЗ ЗНАНИЙ" in system)
    check("модель получила три раздела как требование",
          all(title in stub.messages[0]["content"] for title in ai_review.SECTIONS))
    bot = [e for e in events if e.get("type") == "bot"]
    check("отчёт собран: шапка кодом, разделы модели внутри",
          bot and all(title in bot[0]["text"] for title in ai_review.SECTIONS)
          and "ветка `work/review`" in bot[0]["text"])
    check("источники отданы интерфейсу для карточек",
          bot and bot[0].get("sources") and bot[0].get("hits") == 1)

    # ПРЕДЕЛ ФАЙЛОВ: обрезка названа и человеку, и модели.
    many = "".join("M\t+1\t-1\tapp/ai/file_%d.py\n" % number for number in range(1, 30))
    stub = Stub(files_text=(
        "Изменённые файлы ветки work/review против main (HEAD a1b2c3d): всего 29.\n"
        + many))
    events = await collect(make_deps(stub), base="main")
    bot = [e for e in events if e.get("type") == "bot"]
    check("при пределе файлов обрезка названа в отчёте",
          bot and "НЕ ВСЕ изменения" in bot[0]["text"], bot[0]["text"][:400] if bot else "")
    check("и модели сказано, что она видит не всё",
          any("ВНИМАНИЕ" in str(m.get("content") or "") for m in stub.messages),
          repr([m["role"] for m in stub.messages]))
    diff_calls = [c for c in stub.calls if c["tool"] == ai_review.DIFF_TOOL]
    files_read = {c["arguments"].get("path") for c in diff_calls}
    check("файлов прочитано не больше предела, вызовов — не больше бюджета",
          len(files_read) <= ai_review.MAX_DIFF_FILES
          and len(diff_calls) <= ai_review.MAX_DIFF_CALLS,
          "файлов %d, вызовов %d" % (len(files_read), len(diff_calls)))

    # ОТКАЗ ИНСТРУМЕНТА и МОЛЧАНИЕ МОДЕЛИ — честные ошибки, а не пустой отчёт.
    stub = Stub(fail_diff=True)
    events = await collect(make_deps(stub), base="main")
    check("отказ инструмента назван, отчёта нет",
          texts(events, "error") and not texts(events, "bot")
          and "отказал" in " ".join(texts(events, "debug")))
    stub = Stub(empty_answer=True)
    events = await collect(make_deps(stub), base="main")
    check("молчание модели — ошибка с причиной, а не «всё чисто»",
          texts(events, "error") and "модель не вернула ответ" in texts(events, "error")[0]
          and not texts(events, "bot"))

    # ИЗМЕНЕНИЙ НЕТ: честный ответ, а не пустой отчёт.
    stub = Stub(files_text="Изменений нет: ветка не отличается от main.")
    events = await collect(make_deps(stub), base="main")
    check("пустая ветка: «разбирать нечего» вместо выдуманного ревью",
          texts(events, "bot") and "изменений нет" in texts(events, "bot")[0]
          and not stub.messages)


# ---------------------------------------------------------------------------
# [8] Маршрут POST /api/agent/review (настоящий MCP-сервер на временном репозитории)
# ---------------------------------------------------------------------------
BASE_ID = ""


def _system_texts(call):
    return "\n".join(str(m.get("content") or "") for m in (call.get("messages") or [])
                     if m.get("role") == "system")


async def fake_call_llm_async(*args, **kwargs):
    """Подмена client.call_llm_async: ревью и переформулировка запроса."""
    LLM_CALLS.append(kwargs)
    messages = kwargs.get("messages") or []
    system = str(messages[0].get("content") or "") if messages else ""
    if "ПОИСКОВЫЙ ЗАПРОС" in system:
        return json.dumps({"query": "архитектура проекта"}, ensure_ascii=False), _metrics(5, 3)
    return STUB_REVIEW, _metrics(40, 30)


def _metrics(prompt=30, completion=20):
    return {"model": "stub", "elapsed_seconds": 0.01, "prompt_tokens": prompt,
            "completion_tokens": completion, "total_tokens": prompt + completion}


LLM_CALLS = []


async def run_review_route(content, session_id=""):
    """Прогон POST /api/agent/review без сети: собирает события NDJSON-потока."""
    response = await chat.agent_review(ChatMessage(content=content,
                                                   session_id=session_id))
    events = []
    async for chunk in response.body_iterator:
        for line in str(chunk).splitlines():
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


async def section_route():
    print("\n[8] Маршрут POST /api/agent/review (заглушка модели, настоящий MCP git)")
    mcp_store.forget(ai_review.SERVER_ID)
    llm_client.call_llm_async = fake_call_llm_async

    await chat.task_create(chat.TaskCreate(name="AI-ревью"))
    task = chat._current_task()
    session = workspace_store.create_session(task)

    # Не команда — отказ сразу, до всякой работы.
    check("не команда → 400",
          await _araises(lambda: chat.agent_review(ChatMessage(content="привет"))))

    # Режим выключен (по умолчанию) — команда разбора отвечает причиной.
    LLM_CALLS.clear()
    events = await run_review_route(ai_review.COMMAND, session_id=str(session["id"]))
    check("выключенный режим: отказ с причиной и командой включения",
          texts(events, "error") and ai_review.COMMAND_ENABLE in texts(events, "error")[0],
          str(texts(events, "error"))[:200])
    check("без обращения к модели и без вызова инструментов",
          LLM_CALLS == [])

    # ВКЛЮЧЕНИЕ РЕЖИМА: настройка записана, модель не звалась, и — главное — код
    # репозитория САМ ПОЕХАЛ В RAG фоновой задачей («загрузить кодовую базу
    # автоматически при включении режима»).
    profile_id = chat._current_profile_id()
    LLM_CALLS.clear()
    events = await run_review_route(ai_review.COMMAND_ENABLE,
                                    session_id=str(session["id"]))
    enable_note = texts(events, "bot")[0]
    check("включение: настройка проекта записана",
          workspace_store.ai_review_enabled(task) is True
          and "включён" in enable_note, enable_note[:200])
    check("включение модели не требует", LLM_CALLS == [])
    check("включение называет загрузку кода в RAG (задача или свежий индекс)",
          "Индекс кода" in enable_note, enable_note[:300])
    if chat._review_index_tasks:
        await asyncio.gather(*list(chat._review_index_tasks), return_exceptions=True)
    bases = ai_review_rag.review_bases(profile=profile_id)
    check("после включения база кода собрана и свежа по этому репозиторию",
          bool(bases) and ai_review_rag.state(REPO, profile=profile_id,
                                              sha="")["fresh"] is True, repr(bases))
    state = await chat.review_state()
    check("снимок настройки отдаёт состояние и команды",
          state["enabled"] is True and state["command"] == ai_review.COMMAND
          and state["enable_command"] == ai_review.COMMAND_ENABLE, repr(state))

    # СЕРВЕР GIT ЕЩЁ ВЫКЛЮЧЕН: разбор обязан назвать эту причину (а не молчать).
    LLM_CALLS.clear()
    events = await run_review_route("%s main" % ai_review.COMMAND,
                                    session_id=str(session["id"]))
    check("без сервера git: причина словами и подсказкой про кнопку «MCP»",
          texts(events, "error") and "MCP" in texts(events, "error")[0]
          and "не включён у проекта" in texts(events, "error")[0],
          str(texts(events, "error"))[:200])
    check("модель не звалась, пока данных нет", LLM_CALLS == [])

    # Сервер включён: полный прогон на НАСТОЯЩЕМ MCP-сервере и временном репозитории.
    workspace_store.set_mcp_enabled(task, [ai_review.SERVER_ID])
    LLM_CALLS.clear()
    events = await run_review_route("%s %s" % (ai_review.COMMAND, BASE),
                                    session_id=str(session["id"]))
    review_events = list(events)
    kinds = [event.get("type") for event in events]
    check("поток событий в правильном порядке",
          kinds[:1] == ["review_start"] and kinds[-2:] == ["usage", "done"]
          and "answer_ready" in kinds and kinds.index("bot") < kinds.index("answer_ready"),
          repr(kinds))
    start = [e for e in events if e.get("type") == "review_start"][0]
    check("состояние режима и база сравнения объявлены",
          start["enabled"] is True and start["base"] == BASE, repr(start))
    debug = " ".join(texts(events, "debug"))
    check("в диагностике названы инструменты MCP и объём прочитанного",
          "git_branch_files" in debug and "Diff прочитан" in debug, debug[:300])
    bot = [e for e in events if e.get("type") == "bot"][0]["text"]
    check("отчёт называет НАСТОЯЩУЮ ветку и базу из git",
          "ветка `%s`" % BRANCH in bot and "против `%s`" % BASE in bot, bot[:300])
    check("в отчёте есть все три раздела (ответ модели)",
          all(title in bot for title in ai_review.SECTIONS))
    check("в шапке настоящие имена файлов ветки",
          "app/ai/fresh.py" in bot and "app/ai/module.py" in bot, bot[:600])
    check("модель получила diff ветки, а не рабочей копии",
          "review_target" in _system_texts(LLM_CALLS[-1])
          and "fresh_step" in _system_texts(LLM_CALLS[-1]))
    system_now = _system_texts(LLM_CALLS[-1])
    check("контекст кода пришёл ПОИСКОМ по индексу (RAG), а не всем кодом",
          rag_search.BLOCK_HEADER[:40] in system_now
          and "review_target" in system_now and "fresh_step" in system_now,
          system_now[:300])
    check("в диагностике назван индекс кода (файлы, чанки, бэкенд)",
          "Индекс кода:" in debug and ai_review_rag.BASE_NAME in debug, debug[:400])
    check("шапка отчёта говорит, что контекст собран из индекса",
          "фрагментов кода и документации из индекса" in bot, bot[:400])
    check("база ревью заведена у профиля (видна в диалоге «RAG»)",
          bool(ai_review_rag.review_bases(profile=profile_id)))
    check("маркер индекса помнит коммит, который назвал сервер git",
          bool(ai_review_rag.review_bases(profile=profile_id)[0]["marker"].get("sha")),
          repr(ai_review_rag.review_bases(profile=profile_id)[0]["marker"]))

    # ПОВТОРНЫЙ РАЗБОР того же коммита: индекс берётся готовым, а не собирается
    # заново (иначе каждый прогон на PR стоил бы полной индексации проекта).
    events = await run_review_route("%s %s" % (ai_review.COMMAND, BASE),
                                    session_id=str(session["id"]))
    debug = " ".join(texts(events, "debug"))
    check("повторный разбор взял ГОТОВЫЙ индекс (кэш по коммиту)",
          "индекс был свежий" in debug, debug[:400])

    # ВЫКЛЮЧЕНИЕ — состояние меняется в обе стороны.
    events = await run_review_route(ai_review.COMMAND_DISABLE,
                                    session_id=str(session["id"]))
    check("выключение: настройка проекта снята",
          workspace_store.ai_review_enabled(task) is False
          and "выключен" in texts(events, "bot")[0])

    log_kinds = [node.get("kind") for node in session["dialog"].get("log") or []]
    check("отчёт и команда записаны в ЖУРНАЛ чата",
          "assistant" in log_kinds and "user" in log_kinds, repr(log_kinds[:6]))
    check("в ПАМЯТЬ ЗАДАЧИ (messages) ревью не пишет",
          not (session["dialog"].get("messages") or []),
          repr(session["dialog"].get("messages"))[:200])
    state = workspace_store.dialog_state(session)
    check("автомат задачи не тронут (этап не сдвинут, шагов нет)",
          not state.steps and state.stage == "planning",
          repr(state.stage))
    check("расход показан своим событием и в замер задачи не пишется",
          [e for e in review_events if e.get("type") == "usage"]
          and review_events[-1].get("type") == "done"
          and review_events[-1]["usage"].get("input") == 40
          and not (session["dialog"].get("usage") or {}).get("input"),
          str(review_events[-1]))


async def _araises(work):
    """Отказ маршрута (HTTPException) — без падения проверки."""
    from fastapi import HTTPException
    try:
        result = work()
        if asyncio.iscoroutine(result):
            await result
    except HTTPException:
        return True
    except Exception:                              # noqa: BLE001
        return False
    return False


# ---------------------------------------------------------------------------
# [9] Вход пайплайна — отдельным процессом
# ---------------------------------------------------------------------------
def run_cli(*args, env_extra=None):
    """Запуск tools/ai_review.py отдельным процессом (как это делает Action)."""
    env = dict(os.environ)
    env.update({
        "GIT_REPO_DIR": REPO,
        "AGENT_WORKSPACE_FILE": os.path.join(_TMP, "cli_workspace.json"),
        "AGENT_MEMORY_FILE": os.path.join(_TMP, "cli_memory.json"),
        "AGENT_PROFILES_FILE": os.path.join(_TMP, "cli_profiles.json"),
        "RAG_DIR": os.path.join(_TMP, "cli_rag"),
        "MCP_ATTACH_DIR": os.path.join(_TMP, "cli_mcp_files"),
        "PERIODIC_ENABLED": "0",
    })
    env.update(env_extra or {})
    return subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, "tools", "ai_review.py"),
                           *args], capture_output=True, text=True, env=env,
                          timeout=600)


def section_cli():
    print("\n[9] Вход пайплайна tools/ai_review.py (без модели: --dry-run)")
    # КАКОЙ ПРОЕКТ ЧИТАЕТ ПАЙПЛАЙН. Проекты принадлежат профилям, и пайплайну
    # неизвестно, кто из них сейчас работает: он обязан взять тот, который в
    # приложении держат АКТИВНЫМ (иначе `/ai_review_disable` в проекте человека не
    # выключал бы ревью — проверялось в живом вопросе).
    synthetic = {"version": 1, "active_tasks": {"p-1": "t-active"},
                 "tasks": [
                     {"id": "t-active", "name": "Активный проект", "profile": "p-1",
                      "sessions": [], "ai_review": {"enabled": False, "base": "main"}},
                     {"id": "t-later", "name": "Создан позже", "profile": "p-1",
                      "sessions": [], "ai_review": {"enabled": True, "base": ""}}]}
    chosen = ai_review_cli._project_of(workspace_store.normalize_workspace(synthetic))
    check("пайплайн читает настройку АКТИВНОГО проекта, а не последнего созданного",
          chosen is not None and chosen["id"] == "t-active"
          and workspace_store.ai_review_enabled(chosen) is False, repr(chosen))
    # Режим выключаем КОМАНДОЙ в чате: проверяется, что пайплайн уважает
    # настройку ПРОЕКТА, а она живёт в том же файле, что и у приложения.
    asyncio.run(run_review_route(ai_review.COMMAND_DISABLE))
    check("настройка проекта на диске: режим выключен",
          workspace_store.ai_review_enabled(
              workspace_store.active_task(
                  workspace_store.load_workspace())) is False)
    out_path = os.path.join(_TMP, "review.md")
    json_path = os.path.join(_TMP, "review.json")
    result = run_cli("--repo", REPO, "--base", BASE, "--out", out_path,
                     "--json", json_path, "--dry-run", "--print-prompt", "--quiet")
    check("«сухой» прогон завершился успешно", result.returncode == 0,
          result.stderr[-400:])
    check("ход работы назван в журнале прогона",
          "Беру изменения через MCP" in result.stderr
          and "Diff прочитан" in result.stderr, result.stderr[-400:])
    report = open(out_path, encoding="utf-8").read() if os.path.exists(out_path) else ""
    check("отчёт записан файлом с шапкой и разделами",
          "## 🤖 AI-ревью изменений" in report and "ветка `%s`" % BRANCH in report
          and "Что уйдёт в модель" in report, report[:300])
    check("в отчёт попал промпт с diff'ом ветки и фрагментами индекса",
          "ИЗМЕНЕНИЯ ВЕТКИ" in report and "def review_target" in report
          and rag_search.BLOCK_HEADER[:40] in report, report[:200])
    check("«сухой» прогон модель не звал (в отчёте заглушка, а не ревью)",
          "DRY-RUN" in report and "Что уйдёт в модель" in report)
    summary = json.loads(open(json_path, encoding="utf-8").read()) \
        if os.path.exists(json_path) else {}
    check("сводка для пайплайна: ok, режим и модель названы",
          summary.get("ok") is True and summary.get("enabled") is True
          and summary.get("dry_run") is True and summary.get("model"), repr(summary))

    # РУБИЛЬНИК: выключенный режим — это НЕ сбой (код 0) и НАЗВАННАЯ причина.
    result = run_cli("--repo", REPO, "--base", BASE, "--enabled", "off",
                     "--out", os.path.join(_TMP, "off.md"))
    check("выключенный режим: код 0 и причина с командой включения",
          result.returncode == 0 and "выключен" in result.stderr
          and ai_review.COMMAND_ENABLE in result.stderr, result.stderr[-300:])

    # `--enabled auto` уважает настройку ПРОЕКТА, когда файл проекта есть: режим
    # выключен командой в чате — значит и пайплайн (локальный запуск) молчит.
    result = run_cli("--repo", REPO, "--base", BASE, "--workspace",
                     os.environ["AGENT_WORKSPACE_FILE"],
                     "--out", os.path.join(_TMP, "auto.md"))
    check("auto: выключенный у ПРОЕКТА режим учтён и не считается сбоем",
          result.returncode == 0 and "выключен" in result.stderr
          and ai_review.COMMAND_ENABLE in result.stderr, result.stderr[-400:])

    # РЕЖИМЫ RAG: off — индекс не собирается вовсе (контекст отбором кодом),
    # auto — откат с названной причиной, on — требование индекса (сбой без него).
    index_dir = os.path.join(_TMP, "cli_index")
    result = run_cli("--repo", REPO, "--base", BASE, "--enabled", "on",
                     "--index-dir", index_dir, "--rag", "on",
                     "--out", os.path.join(_TMP, "rag_on.md"))
    check("--rag on: индекс собран, отчёт есть",
          result.returncode == 0 and "Индекс кода:" in result.stderr
          and "фрагментов" in result.stderr, result.stderr[-400:])
    report_on = open(os.path.join(_TMP, "rag_on.md"), encoding="utf-8").read()
    check("--rag on: в отчёте сказано, что контекст собран из индекса",
          "фрагментов кода и документации из индекса" in report_on, report_on[:300])
    check("--rag on: индекс кэширован между прогонами",
          run_cli("--repo", REPO, "--base", BASE, "--enabled", "on",
                  "--index-dir", index_dir, "--rag", "on",
                  "--quiet").stderr.count("индекс был свежий") == 1)
    result = run_cli("--repo", REPO, "--base", BASE, "--enabled", "on",
                     "--index-dir", index_dir, "--rag", "off",
                     "--out", os.path.join(_TMP, "rag_off.md"))
    report_off = open(os.path.join(_TMP, "rag_off.md"), encoding="utf-8").read()
    check("--rag off: индекс не собирается, контекст отобран кодом",
          result.returncode == 0 and "Индекс кода:" not in result.stderr
          and "Контекст кода: целиком" in result.stderr
          and "## 🤖 AI-ревью изменений" in report_off,
          result.stderr[-300:] + " || " + report_off[:200])
    # Сломанный каталог индекса (путь занят файлом): auto — откат с причиной,
    # on — сбой (режим обещает индекс, и скрывать его отсутствие нельзя).
    broken = os.path.join(_TMP, "not_a_dir")
    with open(broken, "w", encoding="utf-8") as fh:
        fh.write("это файл, а не каталог\n")
    result = run_cli("--repo", REPO, "--base", BASE, "--enabled", "on",
                     "--index-dir", broken, "--rag", "auto",
                     "--out", os.path.join(_TMP, "rag_broken.md"))
    check("--rag auto: индекс не собрался — ревью идёт, причина названа",
          result.returncode == 0 and "не собран" in result.stderr, result.stderr[-400:])
    result = run_cli("--repo", REPO, "--base", BASE, "--enabled", "on",
                     "--index-dir", broken, "--rag", "on", "--quiet")
    check("--rag on: без индекса — сбой с причиной, а не тихий откат",
          result.returncode == 1 and "--rag on" in result.stderr, result.stderr[-400:])

    # СБОЙ (а не настройка): чужой базовой ветки нет — код 1 с причиной.
    result = run_cli("--repo", REPO, "--base", "нет-такой-ветки", "--quiet")
    check("сбой отличается от настройки: код 1 и причина словами",
          result.returncode == 1 and "не сделано" in result.stderr,
          result.stderr[-300:])


# ---------------------------------------------------------------------------
# [10] RAG-индекс кода и документации для ревью
# ---------------------------------------------------------------------------
def make_user_base(name):
    """СВОЯ база человека с таким же именем (без маркера ревью).

    На ней проверяется главное обещание уборки: ревью убирает ТОЛЬКО свои индексы
    (по маркеру), а базу с тем же именем, собранную человеком, не трогает.
    """
    return rag.index_files([{"filename": "своё.md",
                             "text": "Мой документ про проект. " * 20}],
                           name=name, profile=chat._current_profile_id())


def section_rag_index():
    print("\n[10] RAG-индекс кода для ревью (индексация, свежесть, поиск чанков)")
    profile = chat._current_profile_id()
    # Начинаем с чистого листа: прежние индексы ревью (их мог оставить маршрут)
    # убираем, чтобы считались именно те, что появятся здесь.
    ai_review_rag.drop_bases(ai_review_rag.review_bases(profile=profile),
                             profile=profile)

    listing = ai_review_rag.file_list(REPO)
    paths = [rel for rel, _size in listing["files"]]
    check("в индекс идут текстовые файлы, документация — первой",
          paths and paths[0].lower().endswith(ai_review_rag.ai_review.DOC_EXTS)
          and "app/ai/module.py" in paths and "README.md" in paths, repr(paths[:5]))
    check("закрытые пути в индекс не попадают",
          not [rel for rel in paths if rel.startswith(".env")
               or rel.startswith("data/")], repr(paths))
    check("размеры файлов нужны для ключа свежести",
          all(isinstance(size, int) and size > 0 for _rel, size in listing["files"]))
    fp1 = ai_review_rag.fingerprint(listing["files"])
    check("отпечаток набора меняется от правки файла",
          fp1 != ai_review_rag.fingerprint(listing["files"][:-1] + [("x.py", 999)])
          and fp1.startswith("fp:"))

    docs = ai_review_rag.documents(REPO)
    check("документы репозитория прочитаны целиком (тексты, а не только имена)",
          docs["files"] and all(item.get("text") for item in docs["documents"])
          and docs["chars"] > 0 and docs["fingerprint"] == fp1, repr(docs["reason"]))
    empty_repo = os.path.join(_TMP, "empty-repo")
    os.makedirs(empty_repo, exist_ok=True)
    empty = ai_review_rag.documents(empty_repo)
    check("пустой репозиторий — причина словами, а не пустой индекс",
          not empty["documents"] and "не нашлось ни одного" in empty["reason"],
          repr(empty))
    check("пределы индексации названы числами",
          ai_review_rag.MAX_FILES >= 50 and ai_review_rag.MAX_CHARS > 100000)
    check("настройки поиска ревью: выборка шире, порог выключен",
          ai_review_rag.search_settings()["top_k_after"] >= 5
          and ai_review_rag.search_settings()["filter"] is False)

    built = ai_review_rag.build(REPO, profile=profile, sha="sha-1")
    check("индекс собран: база, чанки, бэкенд и время названы",
          built["ok"] and built["base_id"] and built["chunks"] > 0
          and built["documents"] > 0 and built["seconds"] >= 0, repr(built))
    marker = ai_review_rag.read_marker(built["base_id"])
    check("маркер базы помнит репозиторий, коммит и отпечаток",
          marker.get("root") == os.path.abspath(REPO) and marker.get("sha") == "sha-1"
          and marker.get("fingerprint") == fp1 and marker.get("chunks") == built["chunks"],
          repr(marker))
    check("свежесть по КОММИТУ: тот же коммит — свежий, другой — нет",
          ai_review_rag.state(REPO, profile=profile, sha="sha-1")["fresh"] is True
          and ai_review_rag.state(REPO, profile=profile, sha="sha-2")["fresh"] is False)
    check("свежесть БЕЗ коммита считается по отпечатку файлов",
          ai_review_rag.state(REPO, profile=profile, sha="")["fresh"] is True)
    check("второй раз индекс не пересобирается",
          ai_review_rag.ensure(REPO, profile=profile, sha="sha-1")["built"] is False)

    # ПРАВКА ФАЙЛА: индекс устаревает и пересобирается, прежний убирается.
    extra = os.path.join(REPO, "app", "ai", "late.py")
    with open(extra, "w", encoding="utf-8") as fh:
        fh.write("def late_change():\n    return 'добавлено после индексации'\n" * 3)
    stale = ai_review_rag.state(REPO, profile=profile, sha="")
    check("правка файла делает индекс устаревшим (отпечаток изменился)",
          stale["fresh"] is False and "индекс кода есть" in stale["reason"],
          repr(stale["reason"]))
    again = ai_review_rag.ensure(REPO, profile=profile, sha="")
    check("устаревший индекс пересобран, прежний убран",
          again["built"] is True and again.get("stale_removed") == 1
          and len(ai_review_rag.review_bases(profile=profile)) == 1, repr(again))
    os.unlink(extra)

    # ЧУЖАЯ БАЗА С ТЕМ ЖЕ ИМЕНЕМ: не наша (нет маркера) — не трогаем.
    user_base = make_user_base(ai_review_rag.BASE_NAME)
    ours = [item["id"] for item in ai_review_rag.review_bases(profile=profile)]
    check("своя база человека с тем же именем не считается индексом ревью",
          user_base["id"] not in ours, repr(ours))
    ai_review_rag.drop_bases(ai_review_rag.review_bases(profile=profile),
                             profile=profile)
    check("уборка индексов ревью не тронула базу человека",
          rag_store.get_base(user_base["id"], profile=profile) is not None)
    rag.delete_base(user_base["id"], profile=profile)

    # ПОИСК НУЖНЫХ ЧАНКОВ: несколько запросов, без повторов.
    built = ai_review_rag.build(REPO, profile=profile, sha="sha-3")
    query_list = ai_review_rag.queries(
        [{"path": "app/ai/module.py"}, {"path": "app/ai/fresh.py"}],
        ["review_target", "fresh_step"], "проверь обработку ошибок")
    check("запросы собраны из правки: имена, модули и пожелание",
          len(query_list) >= 2 and "review_target" in query_list[0]
          and "обработку ошибок" in " ".join(query_list), repr(query_list))
    data = ai_review_rag.search([built["base_id"]], query_list, profile=profile,
                                settings=ai_review_rag.search_settings())
    hits = rag_search.hits_of(data)
    keys = ["%s|%s|%s" % (item.get("base_id"), item.get("chunk_id"), item.get("number"))
            for item in hits]
    check("поиск по индексу вернул фрагменты без повторов",
          hits and len(keys) == len(set(keys)), "фрагментов: %d" % len(hits))
    check("выборка ограничена настройкой ревью",
          len(hits) <= ai_review_rag.search_settings()["top_k_after"], str(len(hits)))
    info = ai_review_rag.context_info(built, data)
    check("контекст для модели: блок с фрагментами и счётчики индекса",
          rag_search.BLOCK_HEADER[:40] in info["block"] and info["chunks"] == len(hits)
          and info["base"] == ai_review_rag.BASE_NAME and info["index"]["chunks"] > 0,
          repr(info["index"]))
    check("пустой поиск не выдаётся за контекст",
          ai_review_rag.context_info(built, {})["block"] == ""
          and ai_review_rag.search([], query_list, profile=profile) == {})

# ---------------------------------------------------------------------------
# [10б] Фоновая индексация: её запускает команда включения режима
# ---------------------------------------------------------------------------
async def section_async_rag():
    """Проверка ФОНОВОЙ сборки индекса — в одном цикле событий.

    Отдельной функцией, а не строкой в синхронной проверке: задача живёт в том же
    цикле событий, где заведена (`asyncio.ensure_future`), и дожидаться её из
    другого цикла нельзя — это и есть настоящий контракт `start_background`.
    """
    print("\n[10б] Фоновая индексация кода для ревью (как её запускает включение режима)")
    profile = chat._current_profile_id()
    ai_review_rag.drop_bases(ai_review_rag.review_bases(profile=profile),
                             profile=profile)
    started = await ai_review_rag.start_background(REPO, profile=profile,
                                                   sha="sha-live")
    check("фоновая индексация запущена как задача RAG (с прогрессом в «RAG»)",
          started.get("started") is True and started.get("job_id")
          and int(started.get("documents") or 0) > 0,
          repr({key: value for key, value in started.items() if key != "task"}))
    marked = await started["task"] if started.get("task") is not None else {}
    check("по завершении задачи база помечена и считается свежей",
          marked.get("ok") is True
          and ai_review_rag.state(REPO, profile=profile, sha="sha-live")["fresh"] is True
          and int(marked.get("chunks") or 0) > 0, repr(marked))
    again = await ai_review_rag.start_background(REPO, profile=profile, sha="sha-live")
    check("повторный запуск видит свежий индекс и второй раз не собирает",
          again.get("started") is False and "свежий" in str(again.get("reason") or ""),
          repr(again))
    # Команда включения идёт ТЕМ ЖЕ путём: она ждёт задачу и оставляет ровно одну
    # базу ревью — проверяется на маршруте в разделе [8].
    check("в каталоге индексов осталась одна база ревью",
          len(ai_review_rag.review_bases(profile=profile)) == 1,
          str(len(ai_review_rag.review_bases(profile=profile))))


#
# ---------------------------------------------------------------------------
# [11] Проводка
# ---------------------------------------------------------------------------
def section_wiring():
    print("\n[11] Проводка: потоковый маршрут, интерфейс, пайплайн, раннер")
    main_text = open(os.path.join(PROJECT_ROOT, "main.py"), encoding="utf-8").read()
    check("потоковый маршрут ревью в списке _STREAM_PATHS (иначе «молчит»)",
          "\"/api/agent/review\"" in main_text)
    page = open(os.path.join(PROJECT_ROOT, "app", "web", "chat.html"),
                encoding="utf-8").read()
    check("интерфейс перехватывает все три команды",
          "/^\\/ai_review(_enable|_disable)?(?:" in page
          and "runAiReview(" in page and "/api/agent/review" in page)
    check("у команд есть подсказка про неверную форму и состояние режима",
          "/^\\/ai_review[a-z_]*$/" in page and "review-mark" in page)
    workflow = os.path.join(PROJECT_ROOT, ".github", "workflows", "ai-review.yml")
    check("файл пайплайна на месте", os.path.isfile(workflow), workflow)
    if os.path.isfile(workflow):
        text = open(workflow, encoding="utf-8").read()
        check("пайплайн запускается на pull request",
              "pull_request:" in text and "pull-requests: write" in text)
        check("рубильник режима — переменная репозитория",
              "vars.AI_REVIEW_ENABLED" in text)
        check("в пайплайне есть шаги подготовки и запуска ревью",
              "actions/checkout" in text and "npm ci --prefix mcp_servers" in text
              and "tools/ai_review.py" in text and "github.base_ref" in text)
        check("отчёт публикуется комментарием и артефактом",
              "issues/$PR/comments" in text and "actions/upload-artifact" in text)
        check("красный статус только за сбой ревью, а не за находки",
              "steps.review.outcome == 'failure'" in text
              and "continue-on-error: true" in text)
        check("раннер выбирается переменной: облачный по умолчанию, свой — по метке",
              "runs-on: ${{ vars.AI_REVIEW_RUNNER || 'ubuntu-latest' }}" in text)
        check("для своей машины в пайплайне есть индекс вне рабочей копии, "
              "режим RAG и рабочий файл проекта (рубильник из чата)",
              "AI_REVIEW_INDEX_DIR" in text and "vars.AI_REVIEW_INDEX_DIR" in text
              and "--rag" in text and "AI_REVIEW_WORKSPACE" in text)
        check("пайплайн показывает, чем посчитаются эмбеддинги",
              "backend_status" in text and "sentence-transformers" in text)
    runner = os.path.join(PROJECT_ROOT, "tools", "review_runner.sh")
    check("скрипт своей машины-раннера на месте и запускаемый",
          os.path.isfile(runner) and os.access(runner, os.X_OK), runner)
    if os.path.isfile(runner):
        text = open(runner, encoding="utf-8").read()
        check("скрипт умеет проверить машину, поставить и запустить раннер",
              "check)" in text and "install)" in text and "config)" in text
              and "start)" in text and "status)" in text)
        check("скрипт называет метку раннера и не печатает секреты",
              "ai-review" in text and "--labels" in text
              and "token" in text and "не печатаются" in text)
        check("скрипт проверяет эмбеддинги и версию питона на машине",
              "rag_embedding" in text and "3.12" in text)


# ---------------------------------------------------------------------------
def main():
    print("Проверка AI-ревью кода (команды /ai_review*) и пайплайна на pull request.")
    print("Временный каталог: " + _TMP)

    global REPO
    REPO = make_repo()
    # Репозиторий для MCP-сервера — ВРЕМЕННЫЙ: иначе ревью разбирало бы сам
    # проект, и проверка зависела бы от того, что в нём сейчас изменено.
    os.environ[ai_review.REPO_ENV] = REPO
    print("Временный репозиторий: %s (ветки %s и %s)" % (REPO, BASE, BRANCH))

    section_commands()
    section_setting()
    section_server_answers()
    section_context()
    section_prompt()
    section_answer()
    asyncio.run(section_run())
    asyncio.run(section_route())
    section_cli()
    section_rag_index()
    asyncio.run(section_async_rag())
    section_wiring()

    print("")
    if FAILURES:
        print("ПРОВАЛЫ (%d): %s" % (len(FAILURES), "; ".join(FAILURES)))
        return 1
    print("Все проверки пройдены.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
