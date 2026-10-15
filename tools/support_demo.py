"""Демо-проект «Поддержка»: проект с включёнными сервером `support` и базой знаний.

ЧТО ДЕЛАЕТ. Создаёт (или дополняет уже существующий) проект для ассистента
поддержки: включает у него MCP-сервер «Поддержка: клиенты и обращения»
(mcp_servers/tickets.mjs) и базу знаний с документацией продукта. После этого
вопрос «почему не работает авторизация?» в этом проекте отвечается по данным
обращения и по документации, а команда `/help` в том же проекте берёт данные
обращения третьим источником (app/ai/dev_help.py).

Запуск:

    ./venv/bin/python tools/support_demo.py                 # создать/обновить
    ./venv/bin/python tools/support_demo.py --dry-run        # только показать план
    ./venv/bin/python tools/support_demo.py --name "Поддержка" --rag kb-3ba3f772

ПОЧЕМУ СКРИПТ, А НЕ КНОПКИ. Настройки проекта (галочки «MCP» и «RAG») ставятся
в интерфейсе, и для человека это два клика. Скрипт нужен там, где настройка
должна быть ПОВТОРЯЕМОЙ: проверка проекта, развёртывание на другом Mac, возврат
демонстрации в исходное состояние. Он идемпотентен — повторный запуск обновляет
настройки того же проекта, а не заводит второй.

ПРИЛОЖЕНИЕ ДОЛЖНО БЫТЬ ОСТАНОВЛЕНО. Работающее приложение держит workspace в
памяти и сохраняет его целиком при каждом изменении: правка файла «из-под него»
будет затёрта (и наоборот — правка приложения потерялась бы). Скрипт это
проверяет (живой pid из data/run/serve.pid и ответ HTTP на порту приложения) и
отказывается работать, пока приложение отвечает. Обойти проверку — `--force`,
но тогда сначала остановите приложение, иначе правку потеряет кто-то из двоих.

ТЕКУЩИЙ ПРОЕКТ НЕ ПЕРЕКЛЮЧАЕТСЯ: создание проекта делает его текущим у профиля,
а скрипт возвращает прежний активный проект на место — иначе человек после
запуска оказался бы в другом проекте и решил, что «всё пропало».
"""

import argparse
import json
import os
import ssl
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import config  # noqa: E402
from app.ai import mcp as mcp_store  # noqa: E402
from app.ai import rag as rag_store  # noqa: E402
from app.ai import workspace as workspace_store  # noqa: E402

DEFAULT_NAME = "Поддержка (демо)"
DEFAULT_BASE_NAME = "документация проекта"
PID_FILE = os.path.join(config.PROJECT_ROOT, "data", "run", "serve.pid")
# Приложение наружу поднимается по HTTPS с самоподписанным сертификатом
# (tools/serve.sh), поэтому проверяются ОБА адреса: по http живой https-сервер
# отвечает обрывом соединения, и «приложение не отвечает» было бы неверным
# выводом (живой случай этой правки: приложение сочли зависшим, а оно слушало
# https://0.0.0.0:8000).
APP_URLS = ("https://127.0.0.1:8000/api/agent/profiles",
            "http://127.0.0.1:8000/api/agent/profiles")


def app_is_running() -> str:
    """Работает ли приложение ("" — нет; иначе причина-подтверждение).

    Два признака: живой pid из файла запуска и ответ HTTP(S). Первого мало (pid
    мог остаться от прежнего запуска), второго тоже (человек мог поднять
    приложение другим способом и на другом порту).
    """
    notes = []
    try:
        with open(PID_FILE, encoding="utf-8") as handle:
            pid = int(str(handle.read()).strip() or 0)
    except (OSError, ValueError):
        pid = 0
    if pid:
        try:
            os.kill(pid, 0)
            notes.append("живой процесс %d из %s" % (pid, os.path.relpath(PID_FILE, config.PROJECT_ROOT)))
        except ProcessLookupError:
            pass
        except PermissionError:
            notes.append("процесс %d существует (нет прав на сигнал)" % pid)
    context = ssl._create_unverified_context()      # сертификат самоподписанный
    for url in APP_URLS:
        try:
            with urllib.request.urlopen(url, timeout=3, context=context) as resp:
                if resp.status == 200:
                    notes.append("приложение отвечает на %s" % url)
                    break
        except (urllib.error.URLError, OSError, ValueError):
            continue
    return "; ".join(notes)


def active_profile() -> str:
    """id текущего профиля: у профилей свои проекты и свои базы знаний."""
    try:
        with open(config.AGENT_PROFILES_FILE, encoding="utf-8") as handle:
            data = json.load(handle)
        active = str(data.get("active") or "").strip()
        if active:
            return active
        items = data.get("profiles") or []
        return str((items[0] or {}).get("id") or "") if items else ""
    except (OSError, ValueError):
        return ""


def find_task(workspace, name: str, profile: str):
    """Проект с таким названием у этого профиля (None — нет)."""
    for task in workspace.get("tasks") or []:
        if str(task.get("name") or "").strip() == name \
                and workspace_store.task_owner(task) == profile:
            return task
    return None


def resolve_base(base: str, profile: str):
    """База знаний по id или по названию: включить можно только СВОЮ базу.

    Проверка идёт по списку баз ПРОФИЛЯ (`rag_store.list_bases`), а не по файлам
    каталога: чужая база в настройки проекта не попадёт (её отбросил бы и
    веб-слой, `rag.filter_enabled`), и молча оставить проект без документации
    нельзя — причина называется в отказе.
    """
    known = rag_store.list_bases(profile=profile) or []
    for item in known:
        if str(item.get("id")) == base:
            return str(item.get("id")), str(item.get("name") or base)
    for item in known:
        if str(item.get("name") or "").strip().lower() == base.strip().lower():
            return str(item.get("id")), str(item.get("name") or "")
    return "", ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Демо-проект «Поддержка»: сервер обращений + база знаний продукта")
    parser.add_argument("--name", default=DEFAULT_NAME, help="название проекта")
    parser.add_argument("--mcp", default=mcp_store.SUPPORT,
                        help="id серверов MCP через запятую (по умолчанию support)")
    parser.add_argument("--rag", default=DEFAULT_BASE_NAME,
                        help="id или название базы знаний (по умолчанию «%s»)" % DEFAULT_BASE_NAME)
    parser.add_argument("--no-rag", action="store_true", help="не включать базы знаний")
    parser.add_argument("--dry-run", action="store_true", help="только показать план")
    parser.add_argument("--force", action="store_true",
                        help="работать, даже если приложение отвечает (см. предупреждение)")
    options = parser.parse_args()

    running = app_is_running()
    print("Демо-проект поддержки: %s" % options.name)
    print("  файл workspace: %s" % config.AGENT_WORKSPACE_FILE)
    if running and not options.force:
        print("\nОСТАНОВЛЕНО: приложение работает (%s)." % running)
        print("Работающее приложение держит workspace в памяти и сохраняет его целиком —")
        print("правка файла будет затёрта. Остановите приложение (tools/serve.sh stop),")
        print("запустите скрипт снова и поднимите приложение (заодно оно подхватит новый код).")
        return 2
    if running:
        print("  ⚠ приложение работает (%s) — правка может быть потеряна: --force" % running)

    profile = active_profile()
    if not profile:
        print("\nОСТАНОВЛЕНО: активный профиль не найден в data/profiles.json.")
        return 2
    print("  профиль: %s" % profile)

    servers = [item.strip() for item in str(options.mcp).split(",") if item.strip()]
    for server in servers:
        if mcp_store.find_server(server) is None:
            print("\nОСТАНОВЛЕНО: сервера «%s» нет в реестре (%s)."
                  % (server, ", ".join(mcp_store.SERVER_IDS)))
            return 2
    base_id, base_name = ("", "")
    if not options.no_rag:
        base_id, base_name = resolve_base(options.rag, profile)
        if not base_id:
            print("\nОСТАНОВЛЕНО: базы знаний «%s» у профиля нет. Загрузите документы"
                  " проекта (кнопка «RAG») или укажите --no-rag." % options.rag)
            return 2

    workspace = workspace_store.load_workspace()
    task = find_task(workspace, options.name, profile)
    previous = (workspace.get("active_tasks") or {}).get(profile)
    if task is None:
        print("  проект: будет создан")
    else:
        print("  проект: уже есть (%s) — настройки будут обновлены" % task.get("id"))
    print("  MCP: %s" % (", ".join(servers) or "(ничего)"))
    print("  RAG: %s" % (("%s (%s)" % (base_name, base_id)) if base_id else "(ничего)"))
    print("  текущий проект профиля: %s → останется прежним" % (previous or "(нет)"))

    if options.dry_run:
        print("\n--dry-run: ничего не записано.")
        return 0

    if task is None:
        task = workspace_store.create_task(workspace, options.name, profile)
    mcp_now = workspace_store.set_mcp_enabled(task, servers)
    rag_now = workspace_store.set_rag_enabled(task, [base_id] if base_id else [])
    # Возврат прежнего активного проекта: создание проекта делает текущим его.
    if previous and previous != task["id"]:
        workspace.setdefault("active_tasks", {})[profile] = previous
    workspace_store.save_workspace(workspace)

    print("\nГотово.")
    print("  проект: %s («%s»)" % (task["id"], task["name"]))
    print("  MCP включён: %s" % (", ".join(mcp_now) or "(ничего)"))
    print("  базы знаний: %s" % (", ".join(rag_now) or "(ничего)"))
    print("  текущий проект профиля: %s" % ((workspace.get("active_tasks") or {}).get(profile) or "(нет)"))
    print("\nДальше: поднимите приложение (tools/serve.sh start — оно подхватит новый код),")
    print("откройте проект «%s» и задайте вопрос «почему не работает авторизация?»." % options.name)
    print("В журнале чата будет видно вызовы MCP и найденные фрагменты документации.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
