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
    # АРГУМЕНТ path: модель называет его почти всегда («поищи в app/module_a.py»), и
    # без него она получала совпадения по ВСЕМУ проекту — живой прогон 16.10: поиск
    # «/help» с path в chat.py вернул 97 совпадений и 88 просмотренных файлов вместо 7
    # в одном файле, то есть адрес правки терялся в чужом шуме, а сервер молчал.
    res = call("search_files", {"query": "compute_total", "path": "app/module_a.py"})
    text = body(res)
    check("поиск по файлу (path) смотрит ТОЛЬКО этот файл",
          "просмотрено файлов: 1" in text and "module_b.py" not in text
          and "guide.md" not in text, text[:200])
    check("поиск по файлу называет область поиска",
          "область поиска: app/module_a.py" in text, text[:200])
    res = call("search_files", {"query": "compute_total", "path": "app"})
    check("поиск по каталогу (path) тоже работает",
          "module_a.py" in body(res) and "module_b.py" in body(res), body(res)[:200])
    res = call("search_files", {"query": "def compute_total", "path": "docs"})
    check("пустой результат честно говорит, где искали",
          "Совпадений нет" in body(res) and "docs" in body(res),
          body(res)[:200])

    # ПОТЕРЯ ЗАГОЛОВКА: правка заголовка без самого заголовка в замене названа
    # предупреждением (живой прогон 16.10: раздел README остался без имени).
    lost = call("edit_file", {"path": "docs/guide.md",
                              "find": "## Расчёт",
                              "replace": "Итог считается по позициям."})
    check("потеря заголовка названа предупреждением, а не проглочена",
          lost.get("ok") and "был ЗАГОЛОВКОМ" in body(lost), body(lost)[-200:])
    back = call("edit_file", {"path": "docs/guide.md",
                              "find": "Итог считается по позициям.",
                              "replace": "## Расчёт"})
    check("возврат заголовка предупреждения не даёт",
          back.get("ok") and "был ЗАГОЛОВКОМ" not in body(back), body(back)[-160:])
    renamed = call("edit_file", {"path": "docs/guide.md",
                                 "find": "## Расчёт",
                                 "replace": "## Расчёт итога"})
    check("переименование заголовка предупреждения не даёт",
          renamed.get("ok") and "был ЗАГОЛОВКОМ" not in body(renamed),
          body(renamed)[-160:])
    call("edit_file", {"path": "docs/guide.md", "find": "## Расчёт итога",
                       "replace": "## Расчёт"})

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
            # ОБРЕЗАННЫЙ ОТВЕТ — НЕ КОНЕЦ ШАГА: содержимое неполное, и правку просят
            # переделать частями (write_file + append_file), а не заканчивают проход
            # (живой прогон 16.10, пятнадцатый заход: так не записалось ничего).
            # Файл — СВОЙ (app/trunc_module.py): общий app/new_module.py нужен
            # следующим проверкам, и портить его нельзя.
            trunc = {"n": 0, "notes": []}

            async def fake_truncated(**kwargs):
                trunc["n"] += 1
                trunc["notes"].append(str(kwargs.get("user_text") or ""))
                if trunc["n"] == 1:
                    # Обрыв по пределу: JSON не закрыт, текста много.
                    raw = ('{"calls": [{"server": "files", "tool": "write_file", '
                           '"arguments": {"path": "app/trunc_module.py", "content": "'
                           + "строка содержимого" * 200)
                    return (raw, {"prompt_tokens": 10, "completion_tokens": 10,
                                  "total_tokens": 20})
                payload = {"calls": [{"server": "files", "tool": "write_file",
                                      "arguments": {
                                          "path": "app/trunc_module.py",
                                          "content": "def part() -> int:\n"
                                                     "    return 5\n"}}]}
                return (json.dumps(payload),
                        {"prompt_tokens": 10, "completion_tokens": 10,
                         "total_tokens": 20})

            llm_client.call_llm_async = fake_truncated
            try:
                task, session = await _fresh_task("Обрезанный ответ")
                actions, lines, _ = await chat._file_actions_step(
                    task, session, "Создать модуль app/trunc_module.py с частями",
                    "Вот модуль.", _agent())
                check("обрезанный ответ не применяется, но шаг не заканчивается",
                      trunc["n"] >= 2, str(trunc["n"]))
                check("в причине назван обрыв и способ писать частями",
                      any("обрезан" in line and "append_file" in line
                          for line in lines), str(lines)[:300])
                check("исполнителю сказано делить запись на части",
                      "ОБРЕЗАН" in trunc["notes"][1]
                      and "append_file" in trunc["notes"][1],
                      trunc["notes"][1][-260:])
                check("полноценная правка после обрыва применилась",
                      "def part()" in (read_local("app/trunc_module.py") or ""),
                      str(read_local("app/trunc_module.py"))[:120])
            finally:
                llm_client.call_llm_async = original

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
# [9] ЧТО ВИДИТ ПРИЁМЩИК И КАК ИСПОЛНИТЕЛЬ ИЩЕТ МЕСТО ПРАВКИ
# ---------------------------------------------------------------------------
async def _fresh_task(name: str):
    """Новая задача и сессия для стенда (как в живой задаче: подпись запроса)."""
    await chat.task_create(TaskCreate(name=name))
    await chat.mcp_apply(McpApply(enabled=["files"]))
    task = chat._current_task()
    await chat.session_create()
    task = chat._current_task()
    session = workspace_store.active_session(chat._workspace, task)
    workspace_store.set_dialog_mcp(session["dialog"], "files\u0000задача", "задача", [])
    return task, session


def _agent():
    """Агент со стандартной конфигурацией (заглушка модели подменена в тестах)."""
    return Agent(AgentConfig(max_tokens=None, stop=None, temperature=None, model=None))


def section_review_and_recon():
    """Живые отказы 16.10 (задача про команду /commands): проверка не видела
    созданного файла, а исполнитель не находил место правки в большом файле."""
    print("\n[9] Сводка для приёмщика и разведка места правки")

    # (а) СВОДКА ДЛЯ ПРИЁМЩИКА: одних имён вызовов мало — по ним не видно, КАКОЙ
    # файл создан, и проверка результата требовала «в данных MCP нет записи»,
    # хотя файл лежал на диске.
    written = [{
        "server": "files", "server_name": "Файлы проекта", "tool": "write_file",
        "ok": True, "arguments": {"path": "app/ai/commands.py",
                                  "content": "СЕКРЕТНОЕ СОДЕРЖИМОЕ МОДУЛЯ"},
        "text": "СОЗДАН app/ai/commands.py: строк 0 → 108; файл записан на диск",
        "error": "", "source": "", "action": "", "attachments": [],
    }]
    digest = mcp_store.review_digest(written)
    check("приёмщик видит ПУТЬ правки", "path=app/ai/commands.py" in digest, digest[:200])
    check("приёмщик видит ФАКТ записи («СОЗДАН»)",
          "СОЗДАН" in digest and "строк 0 → 108" in digest, digest[:200])
    check("содержимое файла в сводку НЕ попадает",
          "СЕКРЕТНОЕ СОДЕРЖИМОЕ" not in digest, digest[:200])
    refused = [{
        "server": "files", "server_name": "Файлы проекта", "tool": "edit_file",
        "ok": False, "arguments": {"path": "app/ai/commands.py", "find": "x"},
        "text": "", "error": "фрагмент не найден", "source": "", "action": "",
        "attachments": [],
    }]
    digest_fail = mcp_store.review_digest(refused)
    check("отказ правки назван приёмщику причиной",
          "path=app/ai/commands.py" in digest_fail and "фрагмент не найден" in digest_fail,
          digest_fail[:200])
    # ЛОЖНОЕ «ПРИНЯТО»: приёмщик читал отказ как «правка сделана» (живой прогон
    # 16.10, восемнадцатый заход: шаг с отказом был принят, файла на диске не
    # менялось). Строку-итог собирает КОД, и она однозначна.
    check("при одних отказах сводка говорит «ИЗМЕНЕНИЙ НЕТ»",
          "ИЗМЕНЕНИЙ НЕТ" in digest_fail and "НЕ применена" in digest_fail,
          digest_fail[:300])
    check("отказы перечислены отдельной строкой",
          "ПРАВКИ НЕ ПРИМЕНЕНЫ" in digest_fail, digest_fail[:300])
    mixed = mcp_store.review_digest(written + refused)
    check("при удачной и неудачной правке видны ОБА факта",
          "ИЗМЕНЕНИЯ ПРИМЕНЕНЫ" in mixed and "ПРАВКИ НЕ ПРИМЕНЕНЫ" in mixed
          and "ИЗМЕНЕНИЙ НЕТ" not in mixed, mixed[:400])

    # (б) ЗАПРОС ИСПОЛНИТЕЛЮ: цель и план задачи, а не только текст шага.
    query = mcp_store.build_file_query(
        "Подключить обработку команды /commands в app/routers/chat.py",
        "Ответ шага.", [{"server": "files", "tool": "insert_in_file",
                         "description": "Вставить рядом с якорем",
                         "schema": {"type": "object"}}],
        results=[{"server": "files", "tool": "write_file", "ok": True,
                  "arguments": {"path": "app/ai/commands.py", "content": "x"},
                  "text": "СОЗДАН app/ai/commands.py", "error": "",
                  "source": "", "action": "", "attachments": []}],
        request="создай команду /commands со справочником команд",
        plan="→ 1) Создать app/ai/commands.py\n  2) Подключить обработку в chat.py")
    check("исполнитель получает ЦЕЛЬ задачи",
          "ЦЕЛЬ ЗАДАЧИ" in query and "справочником команд" in query, query[:300])
    check("исполнитель получает ПЛАН задачи", "ПЛАН ЗАДАЧИ" in query, query[:300])
    check("файлы задачи названы (их текст читается инструментом)",
          "app/ai/commands.py" in query and "читай `read_file`" in query, query[:600])

    # (в) РЕЗУЛЬТАТ ПРЕДЫДУЩЕГО ШАГА: на шаге 2 агент обязан видеть, что сделал шаг 1.
    class _State:
        step_index = 1

    class _State0:
        step_index = 0

    dialog = {"messages": [
        {"role": "user", "content": "создай команду /commands"},
        {"role": "assistant", "content": "СОЗДАЛ app/ai/commands.py: def is_commands(text)"},
        {"role": "user", "content": "Продолжай по плану: шаг 2"},
    ]}
    check("результат прошлого шага берётся из памяти диалога",
          "def is_commands" in chat._previous_step_result(dialog, _State()))
    check("на первом шаге блока нет", chat._previous_step_result(dialog, _State0()) == "")
    agent = Agent(AgentConfig(max_tokens=None, stop=None, temperature=None, model=None))
    agent.step_result = "СОЗДАЛ app/ai/commands.py"
    blocks = "\n".join(item["content"] for item in agent._memory_blocks())
    check("блок «результат предыдущего шага» уходит в модель системной частью",
          "РЕЗУЛЬТАТ ПРЕДЫДУЩЕГО ШАГА" in blocks and "app/ai/commands.py" in blocks,
          blocks[:200])
    agent.step_result = ""
    check("без результата блока нет",
          "РЕЗУЛЬТАТ ПРЕДЫДУЩЕГО ШАГА"
          not in "\n".join(item["content"] for item in agent._memory_blocks()))

    # (г) РАЗВЕДКА В БОЛЬШОМ ФАЙЛЕ: двух раундов чтения хватает, чтобы найти место
    # правки (поиск → окно чтения), а третий раунд — только запись. Живой прогон
    # 16.10 (второй заход): исполнитель искал «/help» (96 совпадений на файл),
    # повторял поиск вместо чтения участка и до правки не дошёл.
    check("подсказка про повторный поиск говорит, что делать вместо него",
          "ПОИСК БЫЛ В РАУНДАХ РАЗВЕДКИ" in mcp_store.FILES_NO_SEARCH_NOTE
          and "read_file" in mcp_store.FILES_NO_SEARCH_NOTE,
          mcp_store.FILES_NO_SEARCH_NOTE[:160])
    target = "app/wire_target.py"
    with open(os.path.join(REPO, target), "w", encoding="utf-8") as handle:
        handle.write("def dispatch(text):\n"
                     + "".join(f"    step_{index} = {index}\n" for index in range(400))
                     + "    return text\n")
    tools = [
        {"server": "files", "server_name": "Файлы проекта", "tool": "search_files",
         "description": "Поиск по файлам", "schema": {"type": "object"}},
        {"server": "files", "server_name": "Файлы проекта", "tool": "read_file",
         "description": "Прочитать файл", "schema": {"type": "object"}},
        {"server": "files", "server_name": "Файлы проекта", "tool": "insert_in_file",
         "description": "Вставить рядом с якорем", "schema": {"type": "object"}},
    ]
    check("о последнем раунде чтения исполнителя предупреждают заранее",
          "ПОСЛЕДНИЙ РАУНД" in mcp_store.FILES_LAST_READ_NOTE
          and "ЗАПИСЬ" in mcp_store.FILES_LAST_READ_NOTE,
          mcp_store.FILES_LAST_READ_NOTE[:160])

    async def run():
        rounds = {"n": 0, "notes": []}

        async def fake_three(**kwargs):
            rounds["n"] += 1
            rounds["notes"].append(str(kwargs.get("user_text") or ""))
            payload = {"calls": []}
            if rounds["n"] == 1:
                payload = {"calls": [{"server": "files", "tool": "search_files",
                                      "arguments": {"query": "def dispatch",
                                                    "path": target}}]}
            elif rounds["n"] == 2:
                payload = {"calls": [{"server": "files", "tool": "read_file",
                                      "arguments": {"path": target, "from_line": 1,
                                                    "to_line": 3}}]}
            elif rounds["n"] == 3:
                # ТРЕТИЙ раунд — только запись: просьба прочитать больше не
                # исполняется, а исполнителю об этом говорят словами.
                payload = {"calls": [{"server": "files", "tool": "read_file",
                                      "arguments": {"path": target, "from_line": 5,
                                                    "to_line": 9}}]}
            else:
                payload = {"calls": [{"server": "files", "tool": "insert_in_file",
                                      "arguments": {"path": target,
                                                    "anchor": "def dispatch(text):",
                                                    "text": "    routed = True"}}]}
            return (json.dumps(payload),
                    {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20})

        original = llm_client.call_llm_async
        llm_client.call_llm_async = fake_three
        try:
            await chat.task_create(TaskCreate(name="Разведка места правки"))
            await chat.mcp_apply(McpApply(enabled=["files"]))
            task = chat._current_task()
            await chat.session_create()
            task = chat._current_task()
            session = workspace_store.active_session(chat._workspace, task)
            workspace_store.set_dialog_mcp(session["dialog"], "files\u0000задача",
                                           "задача", [])
            actions, lines, _ = await chat._file_actions_step(
                task, session, "Подключить обработку команды в " + target,
                "Ответ шага: подключаю команду по образцу соседней.",
                Agent(AgentConfig(max_tokens=None, stop=None, temperature=None, model=None)),
                request="создай команду /commands", plan="1) модуль  2) подключить")
            check("разведка двумя раундами, правка — третьим",
                  rounds["n"] == 4, str(rounds["n"]))
            check("правка после разведки легла на диск",
                  "routed = True" in (read_local(target) or ""),
                  str(read_local(target))[:120])
            check("в раунде правок исполняется чтение окна (одно)",
                  any("Файлы проекта.read_file" in line and "from_line=5" in line
                      for line in lines), str(lines)[:400])
            check("исполнителю сказано: прочитал — теперь верни правку",
                  "ТЕПЕРЬ ВЕРНИ ПРАВКУ" in rounds["notes"][3], rounds["notes"][3][-400:])
            offered_write = rounds["notes"][3].split("ДОСТУПНЫЕ ИНСТРУМЕНТЫ")[-1]
            check("в раунде правок объявлены запись и чтения ОКНА, но не поиск",
                  "write_file" in offered_write and "read_file" in offered_write
                  and "search_files" not in offered_write
                  and "outline_file" not in offered_write,
                  offered_write[:300])
            check("цель и план ушли в КАЖДЫЙ раунд исполнителя",
                  all("ЦЕЛЬ ЗАДАЧИ" in note and "ПЛАН ЗАДАЧИ" in note
                      for note in rounds["notes"]), str(rounds["notes"])[:200])
            check("правка названа в журнале", any(target in line for line in lines),
                  str(lines)[:200])

            # (д) ПОВТОРНЫЙ ПОИСК ВМЕСТО ЧТЕНИЯ: второй поиск не исполняется, а
            # причина названа — иначе исполнитель ищет по кругу, а правок нет.
            again = {"n": 0, "notes": []}

            async def fake_search_again(**kwargs):
                again["n"] += 1
                again["notes"].append(str(kwargs.get("user_text") or ""))
                # Раунды 1-2 — разведка (чтение), раунд 3 — ПОИСК В РАУНДЕ
                # ПРАВОК (не исполняется), раунд 4 — правка.
                tool = "search_files" if again["n"] == 3 else "insert_in_file"
                args = ({"query": "def dispatch", "path": target}
                        if tool == "search_files"
                        else {"path": target, "anchor": "def dispatch(text):",
                              "text": "    searched = True"})
                if again["n"] <= 2:
                    tool = "read_file"
                    args = {"path": target, "from_line": again["n"] * 20,
                            "to_line": again["n"] * 20 + 5}
                return (json.dumps({"calls": [{"server": "files", "tool": tool,
                                               "arguments": args}]}),
                        {"prompt_tokens": 10, "completion_tokens": 10,
                         "total_tokens": 20})

            llm_client.call_llm_async = fake_search_again
            try:
                task, session = await _fresh_task("Повторный поиск")
                _, again_lines, _ = await chat._file_actions_step(
                    task, session, "Подключить обработку команды в " + target,
                    "Ответ шага.", _agent())
                check("поиск в раунде правок не исполнен",
                      not any("Файлы проекта.search_files" in line
                              for line in again_lines),
                      str(again_lines)[:300])
                check("причина названа в журнале",
                      any("не исполняю" in line and "search_files" in line
                          for line in again_lines), str(again_lines)[:400])
                check("исполнителю сказано, что поиск был в раундах разведки",
                      "ПОИСК БЫЛ В РАУНДАХ РАЗВЕДКИ" in again["notes"][3],
                      again["notes"][3][-300:])
                check("правка после этого сделана",
                      "searched = True" in (read_local(target) or ""),
                      str(read_local(target))[:120])
            finally:
                llm_client.call_llm_async = original

            # (ж) В ПОСЛЕДНЕМ РАУНДЕ ЧТЕНИЕ НЕ ИСПОЛНЯЕТСЯ — и это НАЗВАНО, а не
            # проглочено: иначе шаг заканчивается без правок и без причины (живой
            # прогон 16.10, третий заход: исполнитель просил окно `chat.py`
            # раунд за раундом).
            late = {"n": 0, "notes": []}

            async def fake_late_reads(**kwargs):
                late["n"] += 1
                late["notes"].append(str(kwargs.get("user_text") or ""))
                payload = {"calls": [{"server": "files", "tool": "read_file",
                                      "arguments": {"path": target,
                                                    "from_line": late["n"] * 10,
                                                    "to_line": late["n"] * 10 + 5}}]}
                return (json.dumps(payload),
                        {"prompt_tokens": 10, "completion_tokens": 10,
                         "total_tokens": 20})

            llm_client.call_llm_async = fake_late_reads
            try:
                task, session = await _fresh_task("Чтение в раунде правок")
                late_actions, late_lines, _ = await chat._file_actions_step(
                    task, session, "Изменить " + target, "Ответ шага.", _agent())
                check("чтения окна исполняются и в раундах правок (не только в разведке)",
                      sum(1 for line in late_lines
                          if "Файлы проекта.read_file" in line)
                      > mcp_store.FILE_RECON_ROUNDS,
                      str(late_lines)[:400])
                check("шаг без правок назван словами, а не проглочен",
                      any("правок нет" in line or "не вернул ни одной правки" in line
                          for line in late_lines), str(late_lines)[-300:])
                check("о последнем раунде чтения сказано заранее",
                      "ПОСЛЕДНИЙ РАУНД" in late["notes"][
                          mcp_store.FILE_RECON_ROUNDS],
                      late["notes"][mcp_store.FILE_RECON_ROUNDS][-300:])
                check("правок не сделано — и это видно по данным",
                      late_actions == [] or all(not item.get("ok")
                                                for item in late_actions),
                      str(late_actions)[:200])
            finally:
                llm_client.call_llm_async = original

            # (е) ОТКАЗ СЕРВЕРА ОТКРЫВАЕТ ЧТЕНИЕ СНОВА: переделка требует взять
            # якорь из файла, поэтому раунд после отказа обязан уметь читать.
            reopen = {"n": 0}

            async def fake_reopen(**kwargs):
                reopen["n"] += 1
                step = reopen["n"]
                if step == 1:
                    payload = {"calls": [{"server": "files", "tool": "read_file",
                                          "arguments": {"path": target,
                                                        "from_line": 1, "to_line": 2}}]}
                elif step == 2:
                    payload = {"calls": [{"server": "files", "tool": "outline_file",
                                          "arguments": {"path": target}}]}
                elif step == 3:
                    payload = {"calls": [{"server": "files", "tool": "edit_file",
                                          "arguments": {"path": target,
                                                        "find": "нет такого текста",
                                                        "replace": "x"}}]}
                elif step == 4:
                    payload = {"calls": [{"server": "files", "tool": "read_file",
                                          "arguments": {"path": target,
                                                        "from_line": 1, "to_line": 3}}]}
                else:
                    payload = {"calls": [{"server": "files", "tool": "insert_in_file",
                                          "arguments": {"path": target,
                                                        "anchor": "def dispatch(text):",
                                                        "text": "    reopened = True"}}]}
                return (json.dumps(payload),
                        {"prompt_tokens": 10, "completion_tokens": 10,
                         "total_tokens": 20})

            llm_client.call_llm_async = fake_reopen
            try:
                task, session = await _fresh_task("Отказ открывает чтение")
                _, reopen_lines, _ = await chat._file_actions_step(
                    task, session, "Изменить " + target, "Ответ шага.", _agent())
                check("после отказа чтение снова разрешено (раунд 4 — чтение)",
                      reopen["n"] == 5 and any("read_file" in line
                                               for line in reopen_lines),
                      str(reopen["n"]) + " " + str(reopen_lines)[-200:])
                check("переделанная правка прошла, файл компилируется",
                      "reopened = True" in (read_local(target) or "")
                      and mcp_store.syntax_error(target) == "",
                      str(read_local(target))[:120])
            finally:
                llm_client.call_llm_async = original
        finally:
            llm_client.call_llm_async = original

    asyncio.run(run())


# ---------------------------------------------------------------------------
# [10] СЛОМАННЫЙ ФАЙЛ НЕ ВЫДАЁТСЯ ЗА РЕЗУЛЬТАТ
# ---------------------------------------------------------------------------
DECO_TARGET = '''"""Модуль с декоратором."""


def route(path):
    """Декоратор."""
    def wrap(func):
        return func
    return wrap


@route("/x")
def handler():
    return 1
'''


def section_step_structure():
    """Структуру целевых файлов шага читает ПРИЛОЖЕНИЕ, без вызова модели.

    Живые прогоны 16.10 (задача про /commands): шаг «дописать в
    app/routers/chat.py обработку команды» упирался в файл на 7 852 строки, и
    исполнитель искал место правки вслепую — четыре захода из шести закончились
    чтениями без правок.
    """
    print("\n[11] Структура целевых файлов шага — заранее, без модели")

    async def run():
        await chat.task_create(TaskCreate(name="Структура файлов шага"))
        await chat.mcp_apply(McpApply(enabled=["files"]))
        task = chat._current_task()
        data, lines = await chat._step_file_context(
            task, "дописать в app/module_a.py обработку команды и обновить docs/guide.md")
        check("структура файлов, названных в шаге, прочитана заранее",
              len(data) == 2 and all(item.get("ok") for item in data),
              str([item.get("tool") for item in data]))
        check("в структуре есть имена и НОМЕРА строк (по ним ищут место правки)",
              "compute_total" in str(data[0].get("text"))
              and "строка" not in str(data[0].get("text")).replace("строк", ""),
              str(data[0].get("text"))[:200])
        check("в структуре есть номер строки функции",
              any(line.strip().split()[0].isdigit()
                  for line in str(data[0].get("text")).splitlines()[1:4]),
              str(data[0].get("text"))[:200])
        check("журнал называет прочитанное по имени файла",
              any("app/module_a.py" in line for line in lines), str(lines)[:200])
        check("данные запроса их видят (это не только журнал)",
              all(item.get("server") == "files" for item in data))
        # ОКНО ВОКРУГ МЕСТА ПРАВКИ: поиск находит номера строк, но правку по ним
        # не собрать — живой прогон 16.10 (задача про /commands): исполнитель искал
        # «/help» (7 совпадений), окно не читал и угадывал текст (`edit_file` с
        # фрагментом, которого в файле нет). Участок читает КОД.
        with open(os.path.join(REPO, "app/dispatch_probe.py"), "w",
                  encoding="utf-8") as handle:
            handle.write("".join(f"line_{index} = {index}\n" for index in range(120)))
            handle.write('COMMAND = "/help"\n')
            handle.write("".join(f"tail_{index} = {index}\n" for index in range(200)))
        placed, place_lines = await chat._step_file_context(
            task, "Дописать в app/dispatch_probe.py разбор команды /help "
                  "в обработчике сообщений",
            "команда /help отвечает по документации")
        place_text = "\n".join(str(item.get("text") or "") for item in placed)
        check("место правки найдено поиском заранее и названо",
              any("место правки найдено поиском заранее" in line for line in place_lines),
              str(place_lines)[:260])
        check("окно вокруг места правки прочитано (в данных есть сам участок)",
              'COMMAND = "/help"' in place_text
              and any("окно вокруг места правки прочитано" in line
                      for line in place_lines), place_text[:200])
        check("окно читается вокруг НАЙДЕННОЙ строки, а не с начала файла",
              "line_0 = 0" not in place_text and "line_90 = 90" in place_text
              and "tail_199" not in place_text, place_text[:300])
        # Ни команды из шага, ни литералов команд в файле — окна нет, и это не ошибка.
        with open(os.path.join(REPO, "app/plain_probe.py"), "w",
                  encoding="utf-8") as handle:
            handle.write("def plain():\n    return 1\n")
        plain, plain_lines = await chat._step_file_context(
            task, "Дописать в app/plain_probe.py разбор команды", "")
        check("без находок окно не читается и ничего не выдумывается",
              not any("место правки найдено" in line for line in plain_lines),
              str(plain_lines)[:200])
        # КОМАНДЫ В ФАЙЛЕ ЕЩЁ НЕТ (обычный случай «создать команду»): догадок нет —
        # участок найдёт разведка исполнителя, а текст окна он прочитает в раунде
        # правок. Запасной поиск «по литералам команд» находил шапку модуля
        # (живой прогон: строку 3) и был убран — догадка хуже отсутствия данных.
        absent, absent_lines = await chat._step_file_context(
            task, "Дописать в app/plain_probe.py разбор команды /commands "
                  "в обработчике сообщений",
            "создай команду /commands со справочником команд")
        check("для новой команды окно НЕ угадывается",
              not any("место правки найдено" in line for line in absent_lines),
              str(absent_lines)[:200])

        # Файла нет — вызов НЕ выдумывает данные и называет причину.
        missing, missing_lines = await chat._step_file_context(
            task, "дописать в app/no_such_file.py обработку команды")
        check("отсутствующий файл назван причиной, а не пустой структурой",
              missing == [] and any("не удалось" in line for line in missing_lines),
              str(missing_lines)[:200])
        # Шаг без путей ничего не стоит.
        none_data, none_lines = await chat._step_file_context(
            task, "проверить, как устроен поиск по базам знаний")
        check("шаг без файлов ничего не читает",
              none_data == [] and none_lines == [])

    asyncio.run(run())


def section_author_second_pass():
    """ВТОРОЙ ЗАХОД АВТОРА: проход без правок получает прочитанное ещё раз.

    Живые прогоны 16.10 (задача про /commands, шесть заходов): четыре закончились
    чтениями без правок, хотя нужный участок уже был прочитан. Второй заход даёт
    тот же текст модели с заданием СОБРАТЬ правку и без инструментов чтения.
    """
    print("\n[12] Второй заход автора: правка по уже прочитанному")

    target = "app/draft_target.py"
    with open(os.path.join(REPO, target), "w", encoding="utf-8") as handle:
        handle.write("def dispatch(text):\n"
                     + "".join(f"    branch_{index} = {index}\n" for index in range(300))
                     + "    return text\n")
    tools = [
        {"server": "files", "server_name": "Файлы проекта", "tool": "read_file",
         "description": "Прочитать файл", "schema": {"type": "object"}},
        {"server": "files", "server_name": "Файлы проекта", "tool": "insert_in_file",
         "description": "Вставить рядом с якорем", "schema": {"type": "object"}},
    ]
    check("второй заход существует: один участок прочитать можно, дальше — правка",
          "АВТОР ПРАВКИ" in mcp_store.FILE_DRAFT_PROMPT
          and "ПРОЧИТАЙ УЧАСТОК" in mcp_store.FILE_DRAFT_PROMPT
          and "Второго чтения не будет" in mcp_store.FILE_DRAFT_PROMPT
          and "ПОДКЛЮЧЕНИЕ — ЭТО ВЫЗОВ" in mcp_store.FILE_DRAFT_PROMPT,
          mcp_store.FILE_DRAFT_PROMPT[:240])

    async def run():
        async def scenario(name, first_round, expect_draft):
            state = {"recon": 0, "draft": 0, "draft_query": "", "draft_system": ""}

            async def fake(**kwargs):
                system = str(kwargs.get("system_prompt") or "")
                if "АВТОР ПРАВКИ" in system:
                    state["draft"] += 1
                    state["draft_query"] = str(kwargs.get("user_text") or "")
                    state["draft_system"] = system[:80]
                    payload = {"calls": [{"server": "files", "tool": "insert_in_file",
                                          "arguments": {"path": target,
                                                        "anchor": "def dispatch(text):",
                                                        "text": "    drafted = True"}}]}
                else:
                    state["recon"] += 1
                    if state["recon"] == 1 and first_round:
                        payload = first_round
                    else:
                        payload = {"calls": []}
                return (json.dumps(payload),
                        {"prompt_tokens": 10, "completion_tokens": 10,
                         "total_tokens": 20})

            original = llm_client.call_llm_async
            llm_client.call_llm_async = fake
            try:
                task, session = await _fresh_task(name)
                actions, lines, usage = await chat._file_actions_step(
                    task, session, "Подключить обработку команды в " + target,
                    "Ответ шага: подключу по образцу соседней.", _agent())
                return state, actions, lines, usage
            finally:
                llm_client.call_llm_async = original

        read_first = {"calls": [{"server": "files", "tool": "read_file",
                                 "arguments": {"path": target, "from_line": 1,
                                               "to_line": 3}}]}
        state, actions, lines, usage = await scenario("Второй заход автора", read_first, True)
        check("второй заход вызван ровно один раз",
              state["draft"] == 1, str(state))
        check("правка второго захода легла на диск",
              "drafted = True" in (read_local(target) or ""),
              str(read_local(target))[:100])
        check("второму заходу дали ПРОЧИТАННОЕ (а не пустой контекст)",
              "ЧТО УЖЕ ПРОЧИТАНО" in state["draft_query"]
              and "branch_0 = 0" in state["draft_query"], state["draft_query"][:200])
        offered = state["draft_query"].split("ДОСТУПНЫЕ ИНСТРУМЕНТЫ")[-1]
        check("второму заходу объявлены запись и чтение окна, но не поиск",
              "insert_in_file" in offered and "read_file" in offered
              and "search_files" not in offered and "outline_file" not in offered,
              offered[:240])
        check("второй заход назван в журнале",
              any("второй заход" in line for line in lines), str(lines)[:300])
        check("расход второго захода записан видом «draft»",
              "draft" in (usage.get("service") or {}), str(usage.get("service")))
        check("результат второго захода попал в данные шага (его видит приёмщик)",
              any("drafted" in str(item.get("text") or "") or item.get("ok")
                  for item in actions), str(actions)[-200:])

        # ВТОРОЙ ЗАХОД ПРОСИТ УЧАСТОК — ЕМУ ЕГО ДАЮТ, И ОН ДОВОДИТ ДЕЛО ДО ПРАВКИ.
        # Живые прогоны 16.10: без права прочитать участок автор правки отвечал пустым
        # списком (`{"calls": []}`), и шаг оставался невыполненным.
        asked = {"draft": 0, "notes": []}

        async def fake_draft_reads(**kwargs):
            system = str(kwargs.get("system_prompt") or "")
            if "АВТОР ПРАВКИ" in system:
                asked["draft"] += 1
                asked["notes"].append(str(kwargs.get("user_text") or ""))
                if asked["draft"] == 1:
                    payload = {"calls": [{"server": "files", "tool": "read_file",
                                          "arguments": {"path": target,
                                                        "from_line": 1, "to_line": 3}}]}
                else:
                    payload = {"calls": [{"server": "files", "tool": "insert_in_file",
                                          "arguments": {
                                              "path": target,
                                              "anchor": "def dispatch(text):",
                                              "text": "    drafted_by_author = True"}}]}
            else:
                # Исполнитель делает разведку (одно чтение) и правок не возвращает —
                # тогда и включается второй заход автора.
                asked.setdefault("recon", 0)
                asked["recon"] += 1
                if asked["recon"] == 1:
                    payload = {"calls": [{"server": "files", "tool": "read_file",
                                          "arguments": {"path": target,
                                                        "from_line": 1, "to_line": 3}}]}
                else:
                    payload = {"calls": []}
            return (json.dumps(payload),
                    {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20})

        original_before = llm_client.call_llm_async
        llm_client.call_llm_async = fake_draft_reads
        try:
            task, session = await _fresh_task("Второй заход читает участок")
            actions, lines, _ = await chat._file_actions_step(
                task, session, "Подключить обработку команды в " + target,
                "Ответ шага.", _agent())
            check("второй заход прочитал участок и был спрошен снова",
                  asked["draft"] == 2 and any("прочитал участок" in line for line in lines),
                  str(asked["draft"]) + " " + str(lines)[:200])
            check("во втором раунде захода инструментов чтения уже нет",
                  "read_file" not in asked["notes"][1].split("ДОСТУПНЫЕ ИНСТРУМЕНТЫ")[-1],
                  asked["notes"][1].split("ДОСТУПНЫЕ ИНСТРУМЕНТЫ")[-1][:200])
            check("правка автора после чтения легла на диск",
                  "drafted_by_author = True" in (read_local(target) or ""),
                  str(read_local(target))[:160])
        finally:
            llm_client.call_llm_async = original_before

        # ЛИШНЕГО ВЫЗОВА НЕТ: если правка прошла сразу, второй заход не нужен.
        write_first = {"calls": [{"server": "files", "tool": "insert_in_file",
                                  "arguments": {"path": target,
                                                "anchor": "def dispatch(text):",
                                                "text": "    direct = True"}}]}
        state2, _, _, _ = await scenario("Без второго захода", write_first, False)
        check("после удачной правки второго захода НЕ бывает",
              state2["draft"] == 0 and "direct = True" in (read_local(target) or ""),
              str(state2))
        # И если разведка ничего не прочитала — повторять нечего.
        state3, _, lines3, _ = await scenario("Без чтения", None, False)
        check("без прочитанного второй заход не запускается",
              state3["draft"] == 0, str(state3))
        check("отсутствие правок всё равно названо",
              any("правок нет" in line or "не вернул ни одной правки" in line
                  for line in lines3), str(lines3)[:300])

    asyncio.run(run())


def section_step_files_only():
    """ПРАВКА ТОЛЬКО ПО ФАЙЛАМ ШАГА.

    Живой прогон 16.10 (девятый заход, задача про /commands): шаг «дописать в
    app/routers/chat.py обработку команды» закончился перезаписью
    app/ai/commands.py — модуля, созданного ПРЕДЫДУЩИМ шагом. Связки нет, а
    готовый файл испорчен.
    """
    print("\n[13] Правка идёт в файлы, названные в шаге")

    check("файлы шага читаются из текста шага",
          chat._step_paths("дописать в app/routers/chat.py обработку команды, "
                           "обновить README.md.") == ["app/routers/chat.py", "README.md"],
          str(chat._step_paths("дописать в app/routers/chat.py обработку, обновить README.md.")))
    check("шаг без файлов ничего не ограничивает",
          chat._step_paths("проверить, как устроен поиск") == [])

    module = {"server": "files", "tool": "write_file",
              "arguments": {"path": "app/ai/commands.py", "content": "x"}}
    target = {"server": "files", "tool": "edit_file",
              "arguments": {"path": "app/routers/chat.py", "find": "a", "replace": "b"}}
    keep, alien = chat._keep_step_writes([module, target], ["app/routers/chat.py"])
    check("чужая правка отсеивается, своя остаётся",
          [call["tool"] for call in keep] == ["edit_file"] and alien == ["app/ai/commands.py"],
          str(keep) + " " + str(alien))
    keep_all, alien_all = chat._keep_step_writes([module, target], [])
    check("шаг без имён файлов пропускает всё",
          len(keep_all) == 2 and alien_all == [], str(alien_all))
    check("причина отказа названа в подсказке",
          "НЕ В ТОТ ФАЙЛ" in mcp_store.FILES_WRONG_FILE_NOTE, "")

    async def run():
        target_path = "app/step_files_target.py"
        with open(os.path.join(REPO, target_path), "w", encoding="utf-8") as handle:
            handle.write("def entry():\n    return 1\n")
        other_path = "app/step_files_other.py"
        with open(os.path.join(REPO, other_path), "w", encoding="utf-8") as handle:
            handle.write("def other():\n    return 2\n")
        before_other = read_local(other_path)

        rounds = {"n": 0, "notes": []}

        async def fake_wrong_file(**kwargs):
            rounds["n"] += 1
            rounds["notes"].append(str(kwargs.get("user_text") or ""))
            # Первые два раунда — правка ЧУЖОГО файла (её отсекают), потом верная.
            tool_path = other_path if rounds["n"] <= 2 else target_path
            payload = {"calls": [{"server": "files", "tool": "insert_in_file",
                                  "arguments": {"path": tool_path,
                                                "anchor": ("def entry():"
                                                           if tool_path == target_path
                                                           else "def other():"),
                                                "text": "    hooked = True"}}]}
            return (json.dumps(payload),
                    {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20})

        original = llm_client.call_llm_async
        llm_client.call_llm_async = fake_wrong_file
        try:
            task, session = await _fresh_task("Правка не в тот файл")
            _, lines, _ = await chat._file_actions_step(
                task, session, "Подключить обработку команды в " + target_path,
                "Ответ шага.", _agent())
            check("чужая правка НЕ применена (файл не тронут)",
                  read_local(other_path) == before_other, str(read_local(other_path))[:120])
            check("отказ по чужому файлу назван в журнале",
                  any("их нет в шаге" in line for line in lines), str(lines)[:300])
            check("исполнителю сказано, куда править",
                  "НЕ В ТОТ ФАЙЛ" in rounds["notes"][2], rounds["notes"][2][-300:])
            check("верная правка после этого прошла",
                  "hooked = True" in (read_local(target_path) or ""),
                  str(read_local(target_path))[:120])
        finally:
            llm_client.call_llm_async = original

    asyncio.run(run())


def section_facts_survive():
    """ФАКТЫ ПРАВОК НЕ ВЫТЕСНЯЮТСЯ ИЗ ДАННЫХ ЗАПРОСА — и устаревший отказ снимается.

    Живой прогон 16.10 (девятнадцатый заход, задача про /commands): предварительный
    проход MCP занял все восемь мест в данных запроса, а результаты ПРАВОК ФАЙЛОВ,
    дописанные позже, обрезались. Из-за этого агент на следующем шаге видел старый
    отказ «файла нет» (чтение шло ДО создания файла) и отвечал «модуль не создан»,
    а приёмщик отклонял УЖЕ ВЫПОЛНЕННУЮ работу словами «в данных MCP нет записи».
    """
    print("\n[15] Факты правок доходят до модели и приёмщика")

    def read(rel, ok=True):
        return {"server": "files", "tool": "read_file", "ok": ok,
                "arguments": {"path": rel},
                "text": ("Файл: " + rel) if ok else "",
                "error": "" if ok else ("файла нет: " + rel), "source": "",
                "action": "", "attachments": []}

    def write(rel):
        return {"server": "files", "tool": "write_file", "ok": True,
                "arguments": {"path": rel, "content": "x"},
                "text": "СОЗДАН " + rel, "error": "", "source": "",
                "action": "", "attachments": []}

    preflight = [read("app/ai/commands.py", ok=False)]
    preflight += [read(f"docs/read_{index}.md") for index in range(7)]
    kept = mcp_store.normalize_results(preflight + [write("app/ai/commands.py")])
    check("запись файла не вытесняется пределом (данных было 9, предел 8)",
          any(item["tool"] == "write_file" for item in kept), str(len(kept)))
    check("устаревший отказ чтения снят: файл позже записан",
          not any(item["tool"] == "read_file" and not item["ok"] for item in kept),
          str([(i["tool"], i["ok"]) for i in kept]))
    only_reads = mcp_store.normalize_results(preflight)
    check("без записи отказы чтения НЕ пропадают (ничего не скрывается)",
          any(item["tool"] == "read_file" and not item["ok"] for item in only_reads),
          str(only_reads)[:200])
    digest = mcp_store.review_digest(preflight + [write("app/ai/commands.py")])
    check("приёмщик видит факт применения правки",
          "ИЗМЕНЕНИЯ ПРИМЕНЕНЫ" in digest and "СОЗДАН app/ai/commands.py" in digest,
          digest[-260:])
    check("проверки приложения тоже считаются фактами и не теряются",
          mcp_store.APP_FACT_TOOLS == ("syntax_check", "connect_check"),
          str(mcp_store.APP_FACT_TOOLS))
    checks = [{"server": "app", "server_name": "Проверка приложения", "tool": "syntax_check",
               "ok": False, "arguments": {"path": "app/x.py"}, "text": "",
               "error": "файл app/x.py не компилируется", "source": "", "action": "",
               "attachments": []}]
    kept_checks = mcp_store.normalize_results(preflight + [write("app/x.py")] + checks)
    check("запись проверки приложения сохраняется при переполнении",
          any(item["tool"] == "syntax_check" for item in kept_checks),
          str([i["tool"] for i in kept_checks]))


def section_connect_check():
    """ОБЪЯВЛЕНИЕ БЕЗ ВЫЗОВА — НЕ ПОДКЛЮЧЕНИЕ.

    Живой прогон 16.10 (одиннадцатый заход, задача про /commands): в chat.py
    добавили `_is_commands_command` и `_commands_help_text`, ни разу их не вызвали,
    а проверка приняла это за работу — «все три шага подтверждены записями файлов».
    """
    print("\n[14] Подключение = вызов: объявление без вызова не принимается")

    check("приватные объявления верхнего уровня видны",
          mcp_store.private_defs('def _a():\n    pass\n\n\nclass _B:\n    pass\n\n\ndef c():\n    pass\n')
          == ["_a", "_B"], str(mcp_store.private_defs('def _a():\n    pass\n')))
    check("вызванное имя не считается мёртвым",
          mcp_store.unreferenced_names('def _a():\n    pass\n\n\ndef b():\n    return _a()\n',
                                       ["_a"]) == [])
    check("объявленное и не вызванное — считается",
          mcp_store.unreferenced_names('def _a():\n    pass\n\n\ndef b():\n    pass\n',
                                       ["_a"]) == ["_a"])
    check("публичные имена так не проверяются (их может звать другой модуль)",
          mcp_store.private_defs('def public():\n    pass\n') == [])
    check("подсказка говорит, что делать",
          "ПОДКЛЮЧЕНИЕ НЕ СДЕЛАНО" in mcp_store.FILES_DEAD_CODE_NOTE
          and "ВЫЗОВ" in mcp_store.FILES_DEAD_CODE_NOTE, "")

    async def run():
        target = "app/connect_target.py"
        with open(os.path.join(REPO, target), "w", encoding="utf-8") as handle:
            handle.write("def entry(text):\n    return text\n")
        rounds = {"n": 0, "notes": []}

        async def fake_dead_then_call(**kwargs):
            rounds["n"] += 1
            rounds["notes"].append(str(kwargs.get("user_text") or ""))
            if rounds["n"] == 1:
                # Первая правка: объявление БЕЗ вызова (тот самый живой случай).
                content = ("def entry(text):\n    return text\n\n\n"
                           "def _is_commands(text):\n    return text == \"/commands\"\n")
                payload = {"calls": [{"server": "files", "tool": "write_file",
                                      "arguments": {"path": target, "content": content}}]}
            else:
                content = ("def entry(text):\n"
                           "    if _is_commands(text):\n        return \"help\"\n"
                           "    return text\n\n\n"
                           "def _is_commands(text):\n    return text == \"/commands\"\n")
                payload = {"calls": [{"server": "files", "tool": "write_file",
                                      "arguments": {"path": target, "content": content}}]}
            return (json.dumps(payload),
                    {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20})

        original = llm_client.call_llm_async
        llm_client.call_llm_async = fake_dead_then_call
        try:
            task, session = await _fresh_task("Подключение = вызов")
            actions, lines, _ = await chat._file_actions_step(
                task, session, "Подключить разбор команды /commands в " + target,
                "Ответ шага.", _agent())
            check("объявление без вызова распознано и названо",
                  any("без вызова" in line for line in lines), str(lines)[:300])
            check("исполнителю сказано дописать вызов",
                  "ПОДКЛЮЧЕНИЕ НЕ СДЕЛАНО" in rounds["notes"][1],
                  rounds["notes"][1][-300:])
            check("после доработки вызов появился",
                  "_is_commands(text)" in (read_local(target) or "")
                  and "if _is_commands" in (read_local(target) or ""),
                  str(read_local(target))[:200])
            check("в данных шага факта «без вызова» НЕТ (он исправлен)",
                  not any(str(item.get("tool")) == "connect_check"
                          for item in actions), str(actions)[:200])
        finally:
            llm_client.call_llm_async = original

        # НЕИСПРАВИМЫЙ СЛУЧАЙ: вызова не появляется — факт обязан дойти до приёмщика.
        stuck = {"n": 0}

        async def fake_always_dead(**kwargs):
            stuck["n"] += 1
            payload = {"calls": [{"server": "files", "tool": "write_file",
                                  "arguments": {
                                      "path": target,
                                      "content": ("def entry(text):\n    return text\n\n\n"
                                                  "def _dead_helper(text):\n"
                                                  "    return text\n")}}]}
            return (json.dumps(payload),
                    {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20})

        llm_client.call_llm_async = fake_always_dead
        try:
            task, session = await _fresh_task("Подключение не сделано")
            actions, lines, _ = await chat._file_actions_step(
                task, session, "Подключить разбор команды /commands в " + target,
                "Ответ шага.", _agent())
            facts = [item for item in actions
                     if str(item.get("tool")) == "connect_check"]
            check("факт «объявление без вызова» дошёл до данных шага",
                  len(facts) == 1 and "_dead_helper" in str(facts[0].get("error")),
                  str(facts)[:200])
            note = chat._file_actions_note(actions)
            check("в подписи шага это названо, а не спрятано",
                  "НЕ ПРИНЯТО" in note and "не сделано" in note, note[:300])
        finally:
            llm_client.call_llm_async = original

        # НЕ ПОДКЛЮЧАЮЩИЙ шаг так не проверяется: «создать модуль» — это создание.
        rounds2 = {"n": 0}

        async def fake_create(**kwargs):
            rounds2["n"] += 1
            payload = {"calls": [{"server": "files", "tool": "write_file",
                                  "arguments": {
                                      "path": target,
                                      "content": "def _helper(text):\n    return text\n"}}]}
            return (json.dumps(payload),
                    {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20})

        llm_client.call_llm_async = fake_create
        try:
            task, session = await _fresh_task("Создание модуля")
            actions, lines, _ = await chat._file_actions_step(
                task, session, "Создать файл " + target + " со справочником команд",
                "Ответ шага.", _agent())
            check("на шаге СОЗДАНИЯ модуля проверка подключения не мешает",
                  not any(str(item.get("tool")) == "connect_check" for item in actions),
                  str(actions)[:200])
        finally:
            llm_client.call_llm_async = original

    asyncio.run(run())


def section_broken_syntax():
    """Живой прогон 16.10 (задача про /commands): правка применилась, а файл
    перестал компилироваться — вставка легла между декоратором и функцией."""
    print("\n[10] Сломанный синтаксис: отказ сервера и проверка приложения")

    with open(os.path.join(REPO, "app/deco_target.py"), "w", encoding="utf-8") as handle:
        handle.write(DECO_TARGET)
    git("add", "-A")
    git("-c", "user.name=Проверка", "-c", "user.email=check@example.invalid",
        "commit", "-q", "-m", "цель с декоратором")

    bad = call("insert_in_file", {
        "path": "app/deco_target.py", "anchor": "def handler():",
        "text": "BROKEN = True", "where": "before"})
    check("вставка ПЕРЕД функцией под декоратором отклонена",
          not bad.get("ok") and "декоратор" in body(bad), body(bad)[:200])
    check("отказ называет выход (якорь — строка декоратора)",
          "декоратора" in body(bad) and "where=before" in body(bad), body(bad)[:300])
    check("файл после отказа не тронут",
          (read_local("app/deco_target.py") or "") == DECO_TARGET)
    good = call("insert_in_file", {
        "path": "app/deco_target.py", "anchor": '@route("/x")',
        "text": "GOOD = True", "where": "before"})
    check("вставка перед ДЕКОРАТОРОМ проходит",
          good.get("ok") and "GOOD = True" in (read_local("app/deco_target.py") or ""),
          body(good)[:200])
    compile(open(os.path.join(REPO, "app/deco_target.py"), encoding="utf-8").read(),
            "app/deco_target.py", "exec")
    check("файл после правильной вставки компилируется", True)

    check("проверка синтаксиса видит сломанный файл",
          "строка" in mcp_store.syntax_error("app/deco_target.py")
          or mcp_store.syntax_error("app/deco_target.py") == "",
          mcp_store.syntax_error("app/deco_target.py"))
    with open(os.path.join(REPO, "app/broken_probe.py"), "w",
              encoding="utf-8") as handle:
        handle.write("def probe(:\n    return 1\n")
    check("сломанный .py назван строкой и причиной",
          "строка" in mcp_store.syntax_error("app/broken_probe.py"),
          mcp_store.syntax_error("app/broken_probe.py"))
    check("целый файл ошибкой не считается",
          mcp_store.syntax_error("app/deco_target.py") == "",
          mcp_store.syntax_error("app/deco_target.py"))
    check("не-python файлы проверкой не трогаются",
          mcp_store.syntax_error("README.md") == "")

    async def run():
        target = "app/step_module.py"
        tools = [
            {"server": "files", "server_name": "Файлы проекта", "tool": "read_file",
             "description": "Прочитать файл", "schema": {"type": "object"}},
            {"server": "files", "server_name": "Файлы проекта", "tool": "write_file",
             "description": "Записать файл", "schema": {"type": "object"}},
        ]

        async def prepare():
            await chat.task_create(TaskCreate(name="Сломанный синтаксис"))
            await chat.mcp_apply(McpApply(enabled=["files"]))
            task = chat._current_task()
            await chat.session_create()
            task = chat._current_task()
            session = workspace_store.active_session(chat._workspace, task)
            workspace_store.set_dialog_mcp(session["dialog"], "files\u0000задача",
                                           "задача", [])
            return task, session

        # (1) ИСПРАВИМАЯ ОШИБКА: первый раунд пишет файл со сломанным синтаксисом,
        # второй — исправленный. Шаг закрывается, факта поломки в итоге нет.
        rounds = {"n": 0, "notes": []}

        async def fake_broken_then_fixed(**kwargs):
            rounds["n"] += 1
            rounds["notes"].append(str(kwargs.get("user_text") or ""))
            content = ("def broken(:\n    return 1\n" if rounds["n"] == 1
                       else "def fixed():\n    return 2\n")
            payload = {"calls": [{"server": "files", "tool": "write_file",
                                  "arguments": {"path": target, "content": content}}]}
            return (json.dumps(payload),
                    {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20})

        original = llm_client.call_llm_async
        llm_client.call_llm_async = fake_broken_then_fixed
        try:
            task, session = await prepare()
            agent = Agent(AgentConfig(max_tokens=None, stop=None, temperature=None,
                                      model=None))
            actions, lines, _ = await chat._file_actions_step(
                task, session, "Создать модуль " + target,
                "Вот модуль целиком.", agent)
            check("исполнитель получил причину и переделал правку",
                  rounds["n"] == 2, str(rounds["n"]))
            check("поломка названа в журнале по имени файла",
                  any("НЕ КОМПИЛИРУЕТСЯ" in line for line in lines), str(lines)[:300])
            check("просьба исправить синтаксис ушла исполнителю",
                  "СИНТАКСИС" in rounds["notes"][1], rounds["notes"][1][-300:])
            check("исправленный файл компилируется",
                  mcp_store.syntax_error(target) == ""
                  and "def fixed" in (read_local(target) or ""),
                  str(read_local(target))[:80])
            check("в итоговых данных поломки НЕТ (она исправлена)",
                  not any(str(item.get("tool")) == "syntax_check" for item in actions),
                  str(actions)[:200])
        finally:
            llm_client.call_llm_async = original

        # (2) НЕИСПРАВИМАЯ: исполнитель пишет сломанный файл каждый раз. Попытки
        # ограничены, а факт поломки обязан дойти до приёмщика.
        stuck = {"n": 0}

        async def fake_always_broken(**kwargs):
            stuck["n"] += 1
            payload = {"calls": [{"server": "files", "tool": "write_file",
                                  "arguments": {"path": target,
                                                "content": "def stuck(:\n    return 3\n"}}]}
            return (json.dumps(payload),
                    {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20})

        llm_client.call_llm_async = fake_always_broken
        try:
            task, session = await prepare()
            agent = Agent(AgentConfig(max_tokens=None, stop=None, temperature=None,
                                      model=None))
            actions, lines, _ = await chat._file_actions_step(
                task, session, "Создать модуль " + target,
                "Вот модуль целиком.", agent)
            check("попытки исправления ограничены",
                  stuck["n"] == 1 + mcp_store.FILES_SYNTAX_RETRIES, str(stuck["n"]))
            broken_facts = [item for item in actions
                            if str(item.get("tool")) == "syntax_check"]
            check("поломка дошла до данных запроса (её видит приёмщик)",
                  len(broken_facts) == 1 and not broken_facts[0].get("ok")
                  and "не компилируется" in str(broken_facts[0].get("error")),
                  str(broken_facts)[:200])
            note = chat._file_actions_note(actions)
            check("в подписи шага поломка названа, а не спрятана",
                  "НЕ ПРИНЯТО" in note and "не компилируется" in note, note[:300])
            stored = workspace_store.dialog_mcp(session["dialog"])
            check("поломка лежит и в данных диалога",
                  any(str(item.get("tool")) == "syntax_check"
                      for item in (stored.get("results") or [])),
                  str(stored.get("results"))[-200:])
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
    section_review_and_recon()
    section_broken_syntax()
    section_step_structure()
    section_author_second_pass()
    section_step_files_only()
    section_connect_check()
    section_facts_survive()

    print("")
    if FAILURES:
        print("ПРОВАЛЫ (%d): %s" % (len(FAILURES), "; ".join(FAILURES)))
        return 1
    print("Все проверки пройдены.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
