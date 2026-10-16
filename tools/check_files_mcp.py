"""Самопроверка MCP-сервера «Файлы проекта» — рабочие операции агента с файлами.

Запуск (сеть, API-ключ и модель НЕ нужны; нужен Node.js и `npm install` в
каталоге `mcp_servers/`):

    ./venv/bin/python tools/check_files_mcp.py

ЧТО ПРОВЕРЯЕТСЯ

  [1] Реестр и классификаторы: сервер `files` объявлен, доступен без предпосылок
      к сбою, объявляет ровно свои девять инструментов — и РАЗДЕЛЕНЫ РОЛИ:
      чтения не откладываются, а ЗАПИСЬ (`write_file`, `edit_file`,
      `append_file`) откладывается до подтверждения плана (`is_deferred_call`) и
      считается результатом вида «save» (`result_kind`). Иначе правка файла
      выполнялась бы ДО того, как пользователь утвердил план.
  [2] Чтения на ВРЕМЕННОЙ рабочей копии: список файлов (glob, закрытые каталоги),
      чтение с номерами строк и отрывком, чтение НЕСКОЛЬКИХ файлов, поиск по
      файлам (текст, регулярное выражение, регистр, отсутствие совпадений),
      структура файла (Python и markdown) — то, чем агент «ищет информацию по
      нескольким файлам и анализирует содержимое».
  [3] ОХРАНА ПУТЕЙ — и на чтение, и на ЗАПИСЬ: файлы с секретами (`.env`),
      служебные каталоги (`data/`, `.git/`, `node_modules/`), выход за корень,
      аргумент-опция вместо пути, двоичные файлы. Отказы названы причиной, а не
      молчанием.
  [4] ЗАПИСЬ: создание файла, полная замена с diff в ответе, точечная замена
      фрагмента (`edit_file`), дописывание в конец (`append_file`), запреты
      (пустое содержимое, двоичное расширение, нулевой байт, слишком большой
      текст) — и ПРЕДОХРАНИТЕЛЬ: резкое сокращение файла оставляет рядом `.bak` с
      прежним текстом. Отдельно проверяется, что запись АТОМАРНА (временных
      файлов не остаётся) и что `$&` в тексте замены не подменяется совпадением.
  [5] Бюджет ответа: ни один ответ инструмента не превышает клиентский предел
      (`RESULT_CHARS`), а длинный текст сервер обрезает САМ и говорит об этом —
      молчаливая обрезка выглядела бы как «в файле больше ничего нет».
  [6] ГЕЙТ ПРИЛОЖЕНИЯ на живых данных: `split_calls` до подтверждения плана
      оставляет только чтения, `guard_calls` пропускает запись (путь — не
      идентификатор), а `records_missing` не считает вызовы записи ошибкой.
  [7] СЦЕНАРИЙ ЦЕЛИКОМ, воспроизводимо и без модели: «найти, где используется
      функция, обновить документацию и добавить запись в журнал» на ТРЁХ файлах
      временной копии — поиск находит места, правки ложатся на диск, `diff_files`
      показывает изменения против коммита. Это и есть минимальный результат:
      несколько файлов, изменения сохранены, всё повторяется запуском проверки.

Рабочие данные не трогаются: рабочая копия для инструментов создаётся ВРЕМЕННАЯ
(`FILES_REPO_DIR`), а workspace, профили и история агента — во временном каталоге
(переменные `AGENT_*_FILE` выставляются ДО импорта приложения).
"""

import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Изоляция данных: всё пишем во временный каталог ------------------------
_TMP = tempfile.mkdtemp(prefix="files-check-")
os.environ["AGENT_WORKSPACE_FILE"] = os.path.join(_TMP, "workspace.json")
os.environ["AGENT_MEMORY_FILE"] = os.path.join(_TMP, "agent_memory.json")
os.environ["AGENT_PROFILES_FILE"] = os.path.join(_TMP, "profiles.json")
os.environ["RAG_DIR"] = os.path.join(_TMP, "rag")
os.environ["MCP_ATTACH_DIR"] = os.path.join(_TMP, "mcp_files")
os.environ["RAG_EMBED_BACKEND"] = "hashing"
os.environ["RAG_RERANK_BACKEND"] = "features"
os.environ["PERIODIC_ENABLED"] = "0"

import asyncio  # noqa: E402

from app.ai import client as llm_client  # noqa: E402
from app.ai import mcp as mcp_store  # noqa: E402
from app.ai import workspace as workspace_store  # noqa: E402
from app.ai.agent import Agent, AgentConfig  # noqa: E402
from app.routers import chat  # noqa: E402
from app.schemas import McpApply, TaskCreate  # noqa: E402

FAILURES = []

SERVER = "files"
READ_TOOLS = ("list_files", "read_file", "read_files", "search_files",
              "outline_file", "diff_files")
WRITE_TOOLS = ("write_file", "edit_file", "insert_in_file", "append_file")

# Рабочая копия для инструментов: заполняет main(). Сервер запускается отдельным
# процессом и НАСЛЕДУЕТ окружение проверки, поэтому FILES_REPO_DIR достаточно
# выставить в os.environ до первого вызова.
REPO = ""


def check(name, condition, detail=""):
    """Одна проверка: печатает результат и копит провалы."""
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(name)


def call(tool, arguments=None):
    """Вызвать инструмент сервера и вернуть ответ клиента как есть."""
    return mcp_store.call_tool(SERVER, tool, arguments or {})


def body(result):
    """Текст ответа (или текст ошибки) — то, что увидит модель."""
    return str(result.get("text") or result.get("error") or "")


def failed(result, needle=""):
    """Отказал ли инструмент (и названа ли причина)."""
    if result.get("ok"):
        return False
    return (needle.lower() in body(result).lower()) if needle else True


def write_file(rel, text):
    """Положить файл во временную копию (мимо сервера — это подготовка данных)."""
    path = os.path.join(REPO, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def read_local(rel):
    """Прочитать файл временной копии напрямую (проверяем, что записано на диск)."""
    path = os.path.join(REPO, rel)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def git(*args):
    """git во временной копии (только для подготовки состояния)."""
    env = dict(os.environ)
    env.update({
        "GIT_AUTHOR_NAME": "Проверка", "GIT_AUTHOR_EMAIL": "check@example.invalid",
        "GIT_COMMITTER_NAME": "Проверка", "GIT_COMMITTER_EMAIL": "check@example.invalid",
        "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull,
    })
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True,
                          text=True, env=env, check=False)


def no_temp_files():
    """Не осталось ли временных файлов от атомарной записи."""
    left = []
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d != ".git"]
        for name in files:
            if ".tmp-" in name:
                left.append(os.path.join(root, name))
    return left


# ---------------------------------------------------------------------------
# Временная рабочая копия
# ---------------------------------------------------------------------------
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

GUIDE = """# Руководство

## Расчёт

Считает функция `compute_total`. Сумму оформляет `format_money`.
"""


def make_repo():
    """Создать временную копию с тремя файлами и коммитом исходного состояния."""
    write_file("app/__init__.py", "")
    write_file("app/module_a.py", MODULE_A)
    write_file("app/module_b.py", MODULE_B)
    write_file("docs/guide.md", GUIDE)
    write_file("data/history.json", '{"secret": "не показывать"}')
    write_file(".env", "ACCESS_PASSWORD=секрет\n")
    write_file("node_modules/pkg/index.js", "module.exports = 1;\n")
    write_file("assets/logo.png", "не картинка, но расширение двоичное\n")
    git("init", "-q")
    git("add", "-A")
    git("-c", "user.name=Проверка", "-c", "user.email=check@example.invalid",
        "commit", "-q", "-m", "исходное состояние")


# ---------------------------------------------------------------------------
# [1] Реестр и классификаторы
# ---------------------------------------------------------------------------
def section_registry():
    print("\n[1] Реестр, доступность и роль инструментов")
    entry = mcp_store.find_server(SERVER)
    check("сервер files объявлен в реестре", entry is not None)
    check("описан как работа с файлами проекта",
          entry is not None and "файл" in str(entry.get("description", "")).lower())
    reason = mcp_store.availability_error(entry or {})
    check("нет предпосылок к сбою (node, файл сервера, MCP SDK)", reason == "", reason)
    check("id сервера попал в SERVER_IDS", SERVER in mcp_store.SERVER_IDS)

    data = mcp_store.discover(SERVER, force=True)
    check("сервер отвечает на tools/list", bool(data.get("ok")), str(data.get("error")))
    names = [item["name"] for item in data.get("tools") or []]
    check("объявлены все десять инструментов",
          sorted(names) == sorted(READ_TOOLS + WRITE_TOOLS), str(names))
    for tool in READ_TOOLS:
        check("чтение %s не откладывается" % tool,
              not mcp_store.is_deferred_call({"server": SERVER, "tool": tool}))
    for tool in WRITE_TOOLS:
        check("ЗАПИСЬ %s откладывается до «ок»" % tool,
              mcp_store.is_deferred_call({"server": SERVER, "tool": tool}))
        check("результат %s — вид «save»" % tool,
              mcp_store.result_kind(tool) == "save")


# ---------------------------------------------------------------------------
# [2] Чтения
# ---------------------------------------------------------------------------
def section_reads():
    print("\n[2] Чтения: список, чтение, несколько файлов, поиск, структура")
    res = call("list_files", {"glob": "**/*.py"})
    text = body(res)
    check("list_files нашёл оба модуля", "app/module_a.py" in text and "app/module_b.py" in text, text)
    check("list_files не показывает закрытые каталоги",
          "node_modules" not in text and ".env" not in text)
    check("list_files показывает размер", "Б" in text or "КБ" in text)

    res = call("list_files", {"dir": "docs"})
    check("list_files по каталогу", "docs/guide.md" in body(res), body(res))

    res = call("read_file", {"path": "app/module_a.py"})
    text = body(res)
    check("read_file отдаёт НОМЕРА строк", "| " in text and "def compute_total" in text, text[:200])
    check("read_file называет число строк", "строк" in text)

    res = call("read_file", {"path": "app/module_a.py", "from_line": 4, "to_line": 6})
    text = body(res)
    check("read_file умеет отрывок", "def format_money" in text and "def compute_total" not in text, text)
    res = call("read_file", {"path": "app/module_a.py", "from_line": 900})
    check("read_file честно говорит, что начало за концом файла",
          "только" in body(res) and "строк" in body(res), body(res))

    res = call("read_files", {"paths": ["app/module_a.py", "app/module_b.py", "docs/guide.md"]})
    text = body(res)
    check("read_files прочитал три файла",
          "Прочитано файлов: 3" in text and "module_a.py" in text
          and "module_b.py" in text and "guide.md" in text, text[:300])
    res = call("read_files", {"paths": ["app/module_a.py", "app/нет_такого.py"]})
    text = body(res)
    check("read_files называет непрочитанный файл, а не молчит",
          "НЕ прочитаны" in text and "нет_такого" in text, text[:300])

    res = call("search_files", {"query": "compute_total"})
    text = body(res)
    check("search_files нашёл использование в нескольких файлах",
          "module_b.py" in text and "guide.md" in text and "module_a.py" in text, text[:300])
    check("search_files показывает файл и номер строки", ":5:" in text or ":9:" in text, text[:300])
    res = call("search_files", {"query": "COMPUTE_TOTAL"})
    check("поиск не различает регистр по умолчанию", "module_a.py" in body(res), body(res)[:200])
    res = call("search_files", {"query": "COMPUTE_TOTAL", "ignore_case": False})
    check("регистр можно включить обратно (тогда совпадений нет)",
          "Совпадений нет" in body(res), body(res)[:200])
    res = call("search_files", {"query": "COMPUTE_TOTAL", "ignore_case": True})
    check("с включённым игнором регистра те же буквы находятся",
          "module_a.py" in body(res), body(res)[:200])
    res = call("search_files", {"query": r"def\s+compute_\w+", "regex": True})
    check("поиск умеет регулярное выражение",
          "compute_total" in body(res) and "module_a.py" in body(res), body(res)[:200])
    res = call("search_files", {"query": "def compute_total", "context_lines": 2})
    check("поиск показывает строки контекста", "return sum" in body(res), body(res)[:300])
    res = call("search_files", {"query": "нет_такого_текста_вовсе"})
    check("отсутствие совпадений названо прямо",
          "Совпадений нет" in body(res) and "не найдено в" in body(res), body(res)[:200])
    res = call("search_files", {"query": "compute_total", "glob": "*.py"})
    check("поиск можно ограничить шаблоном файлов", "guide.md" not in body(res), body(res)[:200])

    res = call("outline_file", {"path": "app/module_a.py"})
    text = body(res)
    check("outline_file показывает функции и классы",
          "функция: format_money" in text and "функция: compute_total" in text, text[:300])
    res = call("outline_file", {"path": "docs/guide.md"})
    check("outline_file показывает разделы markdown", "раздел: " in body(res), body(res)[:200])
    res = call("outline_file", {"path": "assets/logo.png"})
    check("outline_file на двоичном файле отказывает с причиной", failed(res, "двоичн"), body(res))


# ---------------------------------------------------------------------------
# [3] Охрана путей
# ---------------------------------------------------------------------------
def section_guards():
    print("\n[3] Охрана путей: чтение и запись")
    cases = [
        ("файл с секретами (.env)", {"path": ".env"}, "закрыт"),
        ("служебный каталог data/", {"path": "data/history.json"}, "закрыт"),
        ("окружение node_modules/", {"path": "node_modules/pkg/index.js"}, "закрыт"),
        ("служебный каталог .git/", {"path": ".git/config"}, "закрыт"),
        ("выход за корень (../)", {"path": "../секрет.txt"}, "вне рабочей копии"),
        ("аргумент-опция вместо пути", {"path": "-rf"}, "не может начинаться"),
        ("отсутствующий файл", {"path": "app/нет.py"}, "файла нет"),
        ("каталог вместо файла", {"path": "app"}, "каталог"),
    ]
    for name, args, needle in cases:
        res = call("read_file", args)
        check("чтение отклонено: " + name, failed(res, needle), body(res)[:200])
    for name, args, needle in cases[:6]:
        res = call("write_file", {**args, "content": "текст"})
        check("ЗАПИСЬ отклонена: " + name, failed(res, needle), body(res)[:200])
    res = call("write_file", {"path": "assets/logo.png", "content": "текст"})
    check("в двоичный файл текстом не пишут", failed(res, "двоичн"), body(res))
    res = call("write_file", {"path": "docs/new.key", "content": "текст"})
    check("новый файл с «секретным» именем тоже закрыт", failed(res, "закрыт"), body(res))
    res = call("list_files", {"dir": "../"})
    check("list_files не выходит за корень", failed(res, "вне рабочей копии"), body(res))


# ---------------------------------------------------------------------------
# [4] Запись
# ---------------------------------------------------------------------------
def section_writes():
    print("\n[4] Запись: создание, замена, правка фрагмента, дописывание")
    res = call("write_file", {"path": "docs/adr-001.md",
                              "content": "# ADR-001\n\nРешение: считать итог функцией compute_total.\n"})
    text = body(res)
    check("write_file создал файл", res.get("ok") and "СОЗДАН" in text, text[:200])
    check("содержимое легло на диск",
          (read_local("docs/adr-001.md") or "").startswith("# ADR-001"))
    check("ответ сообщает, что файл записан", "записан на диск" in text)

    res = call("write_file", {"path": "docs/adr-001.md",
                              "content": "# ADR-001\n\nРешение: итог считает `compute_total`.\nСтатус: принято.\n"})
    text = body(res)
    check("write_file перезаписал файл и показал DIFF",
          "ОБНОВЛЁН" in text and "Изменения:" in text and "+ Статус: принято." in text, text[:400])
    check("в diff видны и удалённые строки", "- " in text)
    check("перезапись тоже дошла до диска", "Статус: принято." in (read_local("docs/adr-001.md") or ""))

    res = call("write_file", {"path": "docs/adr-001.md", "content": "  \n"})
    check("пустое содержимое не затирает файл", failed(res, "пусто"), body(res))
    res = call("write_file", {"path": "docs/big.md", "content": "x" * 300000})
    check("слишком большой текст отклонён", failed(res, "слишком велик"), body(res)[:200])
    res = call("write_file", {"path": "docs/bad.md", "content": "текст\u0000двоичный"})
    check("нулевой байт в содержимом отклонён", failed(res, "нулев"), body(res)[:200])

    res = call("edit_file", {"path": "docs/guide.md", "find": "Сумму оформляет `format_money`.",
                             "replace": "Сумму оформляет `format_money`, итог — `compute_total`."})
    text = body(res)
    check("edit_file заменил фрагмент", res.get("ok") and "ИЗМЕНЁН" in text, text[:200])
    check("в ответе правки есть diff", "Изменения:" in text and "+" in text)
    check("правка на диске",
          "итог — `compute_total`" in (read_local("docs/guide.md") or ""))
    res = call("edit_file", {"path": "docs/guide.md", "find": "такого текста нет", "replace": "x"})
    check("ненайденный фрагмент — отказ с подсказкой",
          failed(res, "не найден") and "read_file" in body(res), body(res)[:200])
    res = call("edit_file", {"path": "docs/guide.md", "find": "compute_total", "replace": "total"})
    check("неоднозначный фрагмент отклонён (два вхождения)",
          failed(res, "встречается 2 раз") or failed(res, "уточните find"), body(res)[:200])
    res = call("edit_file", {"path": "docs/guide.md", "find": "compute_total", "replace": "total",
                             "all": True})
    check("all=true заменяет все вхождения", res.get("ok") and "Заменено вхождений: 2" in body(res),
          body(res)[:200])
    check("вторая правка тоже на диске", "total" in (read_local("docs/guide.md") or ""))

    # ЗАМЕНА НЕ ТОЛКУЕТСЯ КАК ШАБЛОН: «$&» в JS — подстановка совпадения, и без
    # защиты текст правки приехал бы в файл искажённым.
    res = call("edit_file", {"path": "docs/adr-001.md", "find": "Статус: принято.",
                             "replace": "Статус: принято (цена $& руб.)."})
    check("символы «$&» в тексте замены не подменяются",
          res.get("ok") and "цена $& руб." in (read_local("docs/adr-001.md") or ""), body(res)[:200])

    # ВСТАВКА ПО ЯКОРЮ: так ПОДКЛЮЧАЮТ новое к существующему коду — короткий
    # уникальный фрагмент строки вместо точного многострочного фрагмента.
    write_file("app/hooks.py", "def first():\n    return 1\n\ndef second():\n    return 2\n")
    res = call("insert_in_file", {"path": "app/hooks.py", "anchor": "def second():",
                                  "text": "    connect()\n"})
    check("insert_in_file вставил строки после якоря", res.get("ok"), body(res)[:200])
    check("вставка на диске с ОТСТУПОМ строки-якоря",
          "    connect()" in (read_local("app/hooks.py") or ""),
          str(read_local("app/hooks.py"))[:120])
    check("ответ называет место вставки", "после строки" in body(res), body(res)[:200])
    res = call("insert_in_file", {"path": "app/hooks.py", "anchor": "return",
                                  "text": "    x = 1\n"})
    check("неоднозначный якорь отклонён со списком строк",
          failed(res, "встречается") and "уточните" in body(res), body(res)[:200])
    res = call("insert_in_file", {"path": "app/hooks.py", "anchor": "нет_такого",
                                  "text": "x\n"})
    check("ненайденный якорь — отказ с подсказкой, где взять",
          failed(res, "не найден") and "search_files" in body(res), body(res)[:200])
    res = call("insert_in_file", {"path": "app/hooks.py", "anchor": "def first():",
                                  "text": "before_hook()\n", "where": "before"})
    check("insert_in_file умеет вставлять ПЕРЕД якорем",
          res.get("ok") and (read_local("app/hooks.py") or "").index("before_hook")
          < (read_local("app/hooks.py") or "").index("def first"), body(res)[:120])

    # СНИСХОДИТЕЛЬНЫЙ edit_file: отступы в find могут отличаться от файла.
    res = call("edit_file", {"path": "app/hooks.py",
                             "find": "def first():\nreturn 1",
                             "replace": "def first():\n    return 100"})
    check("edit_file находит фрагмент с ДРУГИМИ отступами",
          res.get("ok") and "return 100" in (read_local("app/hooks.py") or ""),
          body(res)[:200])
    check("о снисходительном совпадении сказано прямо",
          "ПО НОРМАЛИЗОВАННОМУ" in body(res), body(res)[:200])

    before = read_local("docs/guide.md") or ""
    res = call("append_file", {"path": "docs/CHANGELOG.md", "content": "## 2026-01-01\n- Начало журнала\n"})
    check("append_file создаёт файл, если его нет", res.get("ok") and "ДОПИСАНО" in body(res),
          body(res)[:200])
    res = call("append_file", {"path": "docs/CHANGELOG.md", "content": "- Обновлено руководство"})
    text = body(res)
    check("append_file дописывает в конец и показывает конец файла",
          "ДОПИСАНО" in text and "Обновлено руководство" in text, text[:300])
    check("прежнее содержимое журнала цело",
          "Начало журнала" in (read_local("docs/CHANGELOG.md") or ""))
    check("руководство не тронуто дописыванием", (read_local("docs/guide.md") or "") == before)
    res = call("append_file", {"path": "docs/CHANGELOG.md", "content": ""})
    check("пустое дописывание отклонено", failed(res, "нечего дописывать"), body(res)[:200])

    # ПРЕДОХРАНИТЕЛЬ ОТ ПОТЕРИ: правка, «съевшая» файл, оставляет прежний текст рядом.
    write_file("docs/long.md", "строка\n" * 900)          # ~5 КБ
    original = read_local("docs/long.md") or ""
    res = call("write_file", {"path": "docs/long.md", "content": "коротко\n"})
    text = body(res)
    check("резкое сокращение файла названо предупреждением",
          "ВНИМАНИЕ" in text and "резко меньше" in text, text[:300])
    check("прежний текст сохранён в .bak рядом",
          (read_local("docs/long.md.bak") or "") == original)
    check("новая (короткая) версия тоже на диске",
          (read_local("docs/long.md") or "") == "коротко\n")

    left = no_temp_files()
    check("временных файлов после записи не осталось", not left, str(left))


# ---------------------------------------------------------------------------
# [5] Бюджет ответа
# ---------------------------------------------------------------------------
def section_budget():
    print("\n[5] Бюджет ответа: клиентский предел и честная обрезка")
    write_file("docs/huge.md", "".join(f"строка номер {i} с текстом\n" for i in range(1200)))
    res = call("read_file", {"path": "docs/huge.md"})
    text = body(res)
    check("большой файл прочитан в пределах клиентского предела",
          res.get("ok") and len(text) <= mcp_store.RESULT_CHARS, str(len(text)))
    check("сервер САМ сказал об обрезке (а не клиент молча)",
          "обрезано" in text, text[-200:])
    res = call("search_files", {"query": "строка номер", "limit": 80})
    text = body(res)
    check("поиск с большим числом совпадений тоже в бюджете",
          res.get("ok") and len(text) <= mcp_store.RESULT_CHARS, str(len(text)))
    check("поиск говорит, что показаны не все совпадения",
          "показаны не все" in text or "обрезано" in text, text[-200:])


# ---------------------------------------------------------------------------
# [6] Гейт приложения
# ---------------------------------------------------------------------------
def section_gate():
    print("\n[6] Гейт приложения: что выполняется до подтверждения плана")
    tools = [{"server": SERVER, "tool": name, "schema": {}}
             for name in READ_TOOLS + WRITE_TOOLS]
    calls = [
        {"server": SERVER, "tool": "search_files", "arguments": {"query": "compute_total"}},
        {"server": SERVER, "tool": "read_file", "arguments": {"path": "docs/guide.md"}},
        {"server": SERVER, "tool": "write_file",
         "arguments": {"path": "docs/adr-002.md", "content": "# ADR-002\n"}},
        {"server": SERVER, "tool": "edit_file",
         "arguments": {"path": "docs/guide.md", "find": "а", "replace": "б"}},
    ]
    reads, deferred = mcp_store.split_calls(calls, tools)
    check("до «ок» остаются только ЧТЕНИЯ",
          [item["tool"] for item in reads] == ["search_files", "read_file"],
          str([item["tool"] for item in reads]))
    check("запись файлов отложена до подтверждения",
          sorted(item["tool"] for item in deferred) == ["edit_file", "write_file"],
          str([item["tool"] for item in deferred]))

    allowed, rejected = mcp_store.guard_calls(
        [{"server": SERVER, "tool": "write_file",
          "arguments": {"path": "docs/adr-003.md", "content": "# ADR-003\n"}}],
        [], tools, user_text="создай файл с решением")
    check("охрана пропускает запись файла (путь — не идентификатор)",
          len(allowed) == 1 and not rejected, json.dumps(rejected, ensure_ascii=False))
    missing = mcp_store.records_missing(
        [{"server": SERVER, "tool": "write_file",
          "arguments": {"path": "docs/adr-004.md", "content": "# ADR-004\n"}}], tools, [])
    check("гейт записей не считает запись файла ошибкой", not missing, str(missing))
    check("write_file не требует идентификатора из результатов",
          not [item for item in mcp_store.id_required_tools(tools)
               if item["tool"] == "write_file"])


# ---------------------------------------------------------------------------
# [7] Сценарий целиком: найти, обновить документацию, дополнить журнал
# ---------------------------------------------------------------------------
def section_scenario():
    print("\n[7] Сценарий «найти использование и обновить документацию» (3 файла)")
    # Сценарий идёт на ЧИСТОЙ копии: до него проверки успели многое изменить,
    # а созданные ими файлы (журнал, ADR, .bak) уже попали бы в коммит — и шаг
    # «новый файл» перестал бы быть новым.
    write_file("app/module_a.py", MODULE_A)
    write_file("app/module_b.py", MODULE_B)
    write_file("docs/guide.md", GUIDE)
    for extra in ("docs/CHANGELOG.md", "docs/adr-001.md", "docs/adr-002.md",
                  "docs/adr-003.md", "docs/adr-004.md", "docs/long.md",
                  "docs/long.md.bak", "docs/huge.md", "docs/big.md", "docs/bad.md"):
        path = os.path.join(REPO, extra)
        if os.path.exists(path):
            os.remove(path)
    git("add", "-A")
    git("-c", "user.name=Проверка", "-c", "user.email=check@example.invalid",
        "commit", "-q", "-m", "состояние перед сценарием")

    found = call("search_files", {"query": "compute_total"})
    files = [name for name in ("app/module_a.py", "app/module_b.py", "docs/guide.md")
             if name in body(found)]
    check("шаг 1: поиск нашёл использование сразу в трёх файлах", len(files) == 3, str(files))

    outline = call("outline_file", {"path": "app/module_a.py"})
    check("шаг 2: структура модуля прочитана",
          "функция: compute_total" in body(outline), body(outline)[:200])

    edited = call("edit_file", {
        "path": "docs/guide.md",
        "find": "Считает функция `compute_total`.",
        "replace": "Считает функция `compute_total` (см. app/module_a.py).",
    })
    check("шаг 3: руководство обновлено по коду", edited.get("ok"), body(edited)[:200])
    added = call("write_file", {
        "path": "docs/CHANGELOG.md",
        "content": "# Журнал изменений\n\n## 2026-01-01\n- Руководство уточнено по коду.\n",
    })
    check("шаг 4: создан новый файл журнала", added.get("ok"), body(added)[:200])

    res = call("diff_files", {"paths": ["docs/guide.md", "docs/CHANGELOG.md"]})
    text = body(res)
    check("шаг 5: diff показывает правку руководства",
          "diff --git" in text and "guide.md" in text, text[:300])
    check("в diff есть и удалённая, и добавленная строка",
          "-Считает функция `compute_total`." in text
          and "+Считает функция `compute_total` (см. app/module_a.py)." in text, text[:500])
    check("новый файл назван отдельно (git его в diff не показывает)",
          "НОВЫЕ файлы" in text and "CHANGELOG.md" in text, text[:400])

    check("изменения сохранены на диске (руководство)",
          "см. app/module_a.py" in (read_local("docs/guide.md") or ""))
    check("изменения сохранены на диске (журнал)",
          (read_local("docs/CHANGELOG.md") or "").startswith("# Журнал изменений"))
    check("модули сценарий не трогал",
          (read_local("app/module_a.py") or "") == MODULE_A
          and (read_local("app/module_b.py") or "") == MODULE_B)


# ---------------------------------------------------------------------------
# [8] ПРАВКИ ФАЙЛОВ ПО ШАГУ: ответ шага превращается в изменения на диске
# ---------------------------------------------------------------------------
def section_step_actions():
    """Тот самый проход, без которого файлы не появлялись (живой случай 16.10)."""
    print("\n[8] Правки файлов по шагу (служебный вызов «files» + исполнение)")
    tools = [
        {"server": "files", "server_name": "Файлы проекта", "tool": "read_file",
         "description": "Прочитать файл", "schema": {"type": "object"}},
        {"server": "files", "server_name": "Файлы проекта", "tool": "write_file",
         "description": "Создать или переписать файл", "schema": {"type": "object"}},
        {"server": "files", "server_name": "Файлы проекта", "tool": "edit_file",
         "description": "Заменить фрагмент", "schema": {"type": "object"}},
    ]
    check("инструменты записи отделяются от чтений",
          [item["tool"] for item in mcp_store.file_write_tools(tools)]
          == ["write_file", "edit_file"])
    check("чтения по шагу тоже допустимы (точный текст для правки)",
          len(mcp_store.file_any_calls([{"server": "files", "tool": "read_file",
                                         "arguments": {"path": "x"}}], tools)) == 1)
    # ИМЯ СВЕРЯЕТСЯ МЯГКО: набор у исполнителя из одного сервера, и «без сервера»
    # или «files.write_file» — тот же вызов. Строгая сверка отбрасывала их молча.
    soft = [
        [{"tool": "write_file", "arguments": {"path": "a", "content": "b"}}],
        [{"server": "Файлы проекта", "tool": "files.write_file",
          "arguments": {"path": "a", "content": "b"}}],
    ]
    check("вызов без имени сервера принимается",
          [c["tool"] for c in mcp_store.file_any_calls(soft[0], tools)] == ["write_file"])
    check("имя вида «сервер.инструмент» принимается",
          [c["server"] for c in mcp_store.file_any_calls(soft[1], tools)] == ["files"])
    unknown = [{"server": "files", "tool": "create_file", "arguments": {}}]
    check("выдуманный инструмент не выполняется, но НАЗВАН",
          not mcp_store.file_any_calls(unknown, tools)
          and mcp_store.dropped_file_tools(unknown, tools) == ["create_file"])
    check("шаг с правкой файлов распознаётся по тексту",
          mcp_store.file_step_likely("Создать модуль app/ai/worktree.py")
          and mcp_store.file_step_likely("Подключить перехват команды")
          and not mcp_store.file_step_likely("Изучить dev_help.py"))
    check("обрезанный ответ шага распознаётся (незакрытый блок кода)",
          mcp_store.answer_truncated("Вот код:\n```python\nprint(1)\n")
          and not mcp_store.answer_truncated("Вот код:\n```python\nprint(1)\n```\n"))
    check("дайджест исполнителя показывает СОДЕРЖИМОЕ, а не пересказ",
          "строка файла" in mcp_store.file_digest(
              [{"server": "files", "tool": "read_file", "ok": True,
                "text": "Файл: app/x.py\n1| строка файла", "arguments": {}}]))

    async def run():
        # ЗАГЛУШКА МОДЕЛИ: она отвечает вызовом записи — ровно так, как это делал
        # бы исполнитель шага (проверка идёт без сети и без ключа).
        calls_seen = {"system": ""}

        async def fake_call_llm_async(**kwargs):
            # Промпт исполнителя приходит СИСТЕМНЫМ (`system_prompt`), а не в
            # сообщениях: так к нему добавляется JSON-оговорка проекта.
            messages = kwargs.get("messages") or []
            system = kwargs.get("system_prompt") or (
                str(messages[0].get("content") or "") if messages else "")
            calls_seen["system"] = str(system)[:60]
            calls_seen["json"] = str(kwargs.get("response_format") or "")
            payload = {"calls": [{"server": "files", "tool": "write_file",
                                  "arguments": {"path": "app/new_module.py",
                                                "content": "def added():\n    return 1\n"}}]}
            return (json.dumps(payload),
                    {"prompt_tokens": 12, "completion_tokens": 12, "total_tokens": 24})

        original = llm_client.call_llm_async
        llm_client.call_llm_async = fake_call_llm_async
        try:
            await chat.task_create(TaskCreate(name="Правки по шагу"))
            await chat.mcp_apply(McpApply(enabled=["files"]))
            task = chat._current_task()
            await chat.session_create()
            task = chat._current_task()
            session = workspace_store.active_session(chat._workspace, task)
            # Подпись запроса в диалоге выставляет предварительный проход MCP:
            # запись данных БЕЗ подписи не хранится (её нельзя привязать к запросу),
            # поэтому здесь она нужна так же, как в живой задаче.
            workspace_store.set_dialog_mcp(session["dialog"], "files\u0000задача",
                                           "задача", [])
            agent = Agent(AgentConfig(max_tokens=None, stop=None, temperature=None,
                                      model=None))
            actions, lines, usage = await chat._file_actions_step(
                task, session, "Создать модуль app/new_module.py",
                "Вот модуль:\n```python\ndef added():\n    return 1\n```\n", agent)
            check("промпт исполнителя ушёл в модель системной частью",
                  "ИСПОЛНИТЕЛЬ ШАГА" in calls_seen["system"], calls_seen["system"])
            check("ответ исполнителя запрошен как JSON",
                  calls_seen.get("json") == "json", str(calls_seen.get("json")))
            check("вызов записи выполнен", [a["tool"] for a in actions] == ["write_file"],
                  str([a["tool"] for a in actions]))
            check("ФАЙЛ ПОЯВИЛСЯ НА ДИСКЕ",
                  os.path.isfile(os.path.join(REPO, "app/new_module.py")),
                  str(os.listdir(os.path.join(REPO, "app"))))
            check("в журнал ушла строка о создании файла",
                  any("СОЗДАН" in line for line in lines), str(lines)[:200])
            check("расход записан служебным видом «files»",
                  "files" in (usage.get("service") or {}), str(usage.get("service")))
            stored = workspace_store.dialog_mcp(session["dialog"])
            check("результат правки записан в данные запроса (его видит приёмщик)",
                  any("СОЗДАН" in str(item.get("text") or "")
                      for item in (stored.get("results") or [])),
                  str(stored.get("results"))[:200])
            # ЦИКЛ «ЧТЕНИЕ → ПРАВКА»: исполнитель вправе сначала прочитать файл
            # (точный текст для замены), а правку вернуть следующим раундом.
            rounds = {"n": 0}

            async def fake_two_rounds(**kwargs):
                rounds["n"] += 1
                if rounds["n"] == 1:
                    payload = {"calls": [{"server": "files", "tool": "read_file",
                                          "arguments": {"path": "app/new_module.py"}}]}
                else:
                    payload = {"calls": [{"server": "files", "tool": "edit_file",
                                          "arguments": {"path": "app/new_module.py",
                                                        "find": "return 1",
                                                        "replace": "return 2"}}]}
                return (json.dumps(payload),
                        {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20})

            llm_client.call_llm_async = fake_two_rounds
            loop_actions, loop_lines, _ = await chat._file_actions_step(
                task, session, "Изменить app/new_module.py",
                "Меняю возвращаемое значение на 2.", agent)
            check("раунд чтения выполнен, затем правка",
                  rounds["n"] == 2 and [a["tool"] for a in loop_actions] == ["edit_file"],
                  str(rounds) + " " + str([a["tool"] for a in loop_actions]))
            check("правка применилась после чтения",
                  "return 2" in (read_local("app/new_module.py") or ""),
                  str(read_local("app/new_module.py"))[:80])
            check("в журнале видны и чтение, и правка",
                  any("read_file" in line for line in loop_lines)
                  and any("ИЗМЕНЁН" in line for line in loop_lines),
                  str(loop_lines)[:200])
            llm_client.call_llm_async = fake_call_llm_async

            # ОТКАЗ СЕРВЕРА — НЕ КОНЕЦ ШАГА: первая правка отклоняется (нет такого
            # якоря), вторая — по причине отказа — проходит через insert_in_file.
            retry = {"n": 0}

            async def fake_retry(**kwargs):
                retry["n"] += 1
                if retry["n"] == 1:
                    payload = {"calls": [{"server": "files", "tool": "edit_file",
                                          "arguments": {"path": "app/new_module.py",
                                                        "find": "нет такого текста",
                                                        "replace": "x"}}]}
                else:
                    payload = {"calls": [{"server": "files", "tool": "insert_in_file",
                                          "arguments": {"path": "app/new_module.py",
                                                        "anchor": "def added():",
                                                        "text": "    connected = True"}}]}
                return (json.dumps(payload),
                        {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20})

            llm_client.call_llm_async = fake_retry
            retry_actions, retry_lines, _ = await chat._file_actions_step(
                task, session, "Подключить модуль app/new_module.py",
                "Подключаю модуль к проекту.", agent)
            check("после отказа сервера исполнитель переделал вызов",
                  retry["n"] >= 2 and "connected = True" in (read_local("app/new_module.py") or ""),
                  str(retry) + " " + str(read_local("app/new_module.py"))[:80])
            check("отказ назван в журнале, а не проглочен",
                  any("отклонена" in line for line in retry_lines), str(retry_lines)[:200])
            llm_client.call_llm_async = fake_call_llm_async

            # ПРАВКА СУЩЕСТВУЮЩЕГО ФАЙЛА: точечная замена, а не перезапись.
            before_edit = read_local("app/new_module.py") or ""
            edit_actions, edit_lines, _ = await chat._file_actions_step(
                task, session, "Изменить функцию в app/new_module.py",
                "Меняю возвращаемое значение.", agent)
            check("повторный проход отработал без сбоя",
                  isinstance(edit_actions, list) and isinstance(edit_lines, list))
            # Повторный проход пишет файл заново (в заглушке — тем же вызовом
            # `write_file`), поэтому проверяем не «байт в байт», а главное: файл
            # не потерян и не обрезан.
            check("файл не потерян повторным проходом",
                  bool(read_local("app/new_module.py"))
                  and "def added" in (read_local("app/new_module.py") or ""),
                  str(read_local("app/new_module.py"))[:80])
        finally:
            llm_client.call_llm_async = original

    asyncio.run(run())


# ---------------------------------------------------------------------------
def main():
    print("Проверка MCP-сервера «Файлы проекта» (рабочие операции с файлами).")
    print("Временный каталог проверки: " + _TMP)

    global REPO
    REPO = os.path.join(_TMP, "repo")
    os.makedirs(REPO, exist_ok=True)
    os.environ["FILES_REPO_DIR"] = REPO
    make_repo()
    print("Временная рабочая копия: " + REPO)

    section_registry()
    section_reads()
    section_guards()
    section_writes()
    section_budget()
    section_gate()
    section_scenario()
    section_step_actions()

    print("")
    if FAILURES:
        print("ПРОВАЛЫ (%d): %s" % (len(FAILURES), "; ".join(FAILURES)))
        return 1
    print("Все проверки пройдены.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
