"""ЖИВАЯ проверка уборки внешних сборов MCP (в offline-набор НЕ входит).

Запуск (нужны поднятый SSH-туннель к своему `open-meteo-mcp` и токен в `.env`):

    ./venv/bin/python tools/check_mcp_cleanup_live.py

Что проверяется — НАСТОЯЩИМИ вызовами настоящего сервера MCP:

    отмена периодической задачи   → на сервере уходит stop-инструмент сбора
    удаление задачи (сессии)      → на сервере уходит stop-инструмент сбора
    удаление ПРОЕКТА целиком      → сборы ВСЕХ его задач останавливаются
    пауза задачи                  → сбор НЕ останавливается (команда пользователя)
    отмена ОБЫЧНОЙ задачи         → сбор НЕ останавливается (задача жива, данные
                                    на сервере не удаляем без спроса)

Критерий «выполнение отменено» — задача сервера, а не имя инструмента: скрипт
смотрит, что наблюдение перестало СОБИРАТЬ (сервер помечает его иначе, чем
работающее), и что в чате задачи есть строка про остановленные сборы. Что именно
считать остановкой, решает СЕРВЕР: у `open-meteo-mcp` это `stop_weather_watch`
(пауза, история сохраняется), а есть и отдельный `delete_weather_watch`
(удаляет данные) — его агент сам НЕ вызывает: удалять историю без прямой просьбы
пользователя нельзя.

Как это делается: скрипт СОЗДАЁТ на сервере свои наблюдения (координатами, без
геокодера), кладёт их в задачи как «внешние сборы» (ровно в том виде, в каком их
пишет конвейер), поднимает РЕАЛЬНОЕ приложение (uvicorn) на временном порту и
дёргает его маршруты по HTTP — те же, что нажимает интерфейс. После проверки все
свои наблюдения удаляются (в том числе при сбое: блок finally).

Данные пользователя не трогаются: workspace и профили пишутся во временный
каталог. Чужие наблюдения на сервере скрипт не останавливает и не удаляет —
только созданные им самим (координатные, с префиксом `point-`).
"""

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_TMP = tempfile.mkdtemp(prefix="mcp-live-")
os.environ["AGENT_WORKSPACE_FILE"] = os.path.join(_TMP, "workspace.json")
os.environ["AGENT_MEMORY_FILE"] = os.path.join(_TMP, "agent_memory.json")
os.environ["AGENT_PROFILES_FILE"] = os.path.join(_TMP, "profiles.json")
# Планировщик повторов в проверке не нужен: он бы запускал задачи сам.
os.environ["PERIODIC_ENABLED"] = "0"

from app.ai import mcp as mcp_store  # noqa: E402
from app.ai import workspace as workspace_store  # noqa: E402

FAILURES = []
# Наблюдения, созданные ЭТИМ скриптом: их обязательно убрать в конце.
CREATED = []
START_TOOL = "start_weather_watch"


def retry(func, attempts=3, pause=1.0):
    """Повторяет обращение к серверу: туннель и сервер могут моргать.

    Проверка бьёт по ЖИВОМУ серверу, который пользователь в это время может
    перезапускать (в списке инструментов уже появлялись новые) — одиночный сбой
    связи не должен выглядеть как «сбор не остановился».
    """
    last = None
    for attempt in range(attempts):
        last = func()
        if last is not None:
            return last
        time.sleep(pause)
    return last


def check(name, condition, detail=""):
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# Инструменты сервера: какие имена он объявил (ничего не выдумываем)
# ---------------------------------------------------------------------------
def server_tools():
    found = mcp_store.discover_many([mcp_store.OPEN_METEO])
    for item in found:
        if item.get("id") == mcp_store.OPEN_METEO:
            if not item.get("ok"):
                return []
            return [{"server": item["id"], "tool": tool["name"],
                     "schema": tool.get("schema") or {}}
                    for tool in (item.get("tools") or [])]
    return []


def watches():
    """Состояние наблюдений по ответу САМОГО сервера: {id: "collecting"|"paused"}.

    Формат ответа — дело сервера, поэтому читаем аккуратно: блок на наблюдение
    (строка начинается с «- id:») и признак «state: …». Наблюдение, которого в
    ответе нет, — уже не существует (None вместо словаря).
    """
    result = retry(lambda: mcp_store.call_tool(
        mcp_store.OPEN_METEO, "list_weather_watches", {}))
    if not result or not result.get("ok"):
        return None
    text = result.get("text") or ""
    out = {}
    for block in re.split(r"^- ", text, flags=re.M)[1:]:
        name = block.split(":", 1)[0].strip()
        if not name or " " in name:
            continue
        state = "collecting" if "state: collecting" in block else "paused"
        out[name] = state
    return out


def start_watch(latitude, longitude, tools):
    """Заводит НАСТОЯЩЕЕ наблюдение и возвращает обязательство для задачи."""
    result = retry(lambda: mcp_store.call_tool(
        mcp_store.OPEN_METEO, START_TOOL,
        {"latitude": latitude, "longitude": longitude}))
    if not result or not result.get("ok"):
        print("  !! не удалось создать наблюдение:",
              (result or {}).get("error") or "нет ответа сервера")
        return None
    entry = dict(result, server=mcp_store.OPEN_METEO, tool=START_TOOL,
                 server_name="Погода Open-Meteo")
    started = mcp_store.started_calls([entry], tools)
    if not started:
        print("  !! сервер завёл наблюдение, но id в ответе не найден:",
              (result.get("text") or "")[:200])
        return None
    CREATED.append(started[0]["arguments"].get("id"))
    return started[0]


def drop_watch(watch_id):
    """Убирает СВОЁ тестовое наблюдение (данные тестовые — их не жаль)."""
    if not watch_id:
        return
    delete_tool = next((item["tool"] for item in (TOOLS or [])
                        if item["tool"].startswith("delete_")
                        and "watch" in item["tool"]), "")
    if not delete_tool:
        retry(lambda: mcp_store.call_tool(
            mcp_store.OPEN_METEO, "stop_weather_watch", {"id": watch_id}))
        return
    schema = next((item["schema"] for item in TOOLS if item["tool"] == delete_tool), {})
    arguments = {"id": watch_id}
    if "confirm" in (schema.get("properties") or {}):
        arguments["confirm"] = True
    result = retry(lambda: mcp_store.call_tool(
        mcp_store.OPEN_METEO, delete_tool, arguments))
    if not result or not result.get("ok"):
        print("  !! не удалось убрать своё наблюдение", watch_id,
              (result or {}).get("error") or "нет ответа сервера")
        return
    for _ in range(3):
        if watch_id not in (watches() or {}):
            return
        time.sleep(1)


# ---------------------------------------------------------------------------
# Временное приложение: workspace с задачами, в которых «висят» внешние сборы
# ---------------------------------------------------------------------------
def write_workspace(started_list, periodic=True):
    """Кладёт в файл проект с задачей-диалогом и её внешними сборами.

    Профиль у задачи пустой — приложение при старте само закрепит её за текущим
    профилем (см. adopt_orphan_tasks) и сделает текущей. `started_list` — список
    списков обязательств: первый идёт в первую задачу, остальные — в следующие
    (нужно для проверки «удаление проекта останавливает сборы ВСЕХ задач»).
    """
    workspace = {"version": 1, "tasks": [], "active_tasks": {},
                 "long_term": [], "long_term_by_profile": {}, "stats_by_profile": {}}
    task = workspace_store.create_task(workspace, "Проект проверки уборки", "")
    sessions = []
    for index, entries in enumerate(started_list):
        session = workspace_store.create_session(
            task, f"Задача проверки {index + 1}", periodic=3600 if periodic else None)
        workspace_store.add_mcp_started(session["dialog"], entries)
        sessions.append(session["id"])
    workspace["active_task"] = task["id"]
    with open(os.environ["AGENT_WORKSPACE_FILE"], "w", encoding="utf-8") as fh:
        json.dump(workspace, fh, ensure_ascii=False)
    return task["id"], sessions


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_ready(port, timeout=25.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                return True
        except Exception:  # noqa: BLE001 — сервер ещё поднимается
            time.sleep(0.3)
    return False


def request(port, method, path, payload=None):
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(payload or {}).encode("utf-8") if method != "DELETE" else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        return exc.code, {"detail": exc.read().decode("utf-8")[:200]}


def run_case(title, watch_ids, action, expect_stopped, periodic=True, second=None):
    """Поднимает приложение на этом workspace, выполняет действие, смотрит сервер.

    `watch_ids` — id наблюдений, которые должны быть в задаче(ах); `second` —
    обязательство для ВТОРОЙ задачи того же проекта (проверка удаления проекта).
    """
    entries = [[started] for started in watch_ids]
    if second is not None:
        entries.append([second])
    task_id, sessions = write_workspace(entries, periodic=periodic)
    port = free_port()
    env = dict(os.environ)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc = subprocess.Popen(
        [os.path.join(root, "venv", "bin", "python"), "-m", "uvicorn", "main:app",
         "--port", str(port), "--log-level", "warning"],
        cwd=root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if not wait_ready(port):
            check(f"{title}: приложение поднялось", False, "нет ответа /health")
            return
        print(f"\n[{title}]")
        before = watches()
        if before is None:
            check(f"{title}: сервер MCP отвечает (список наблюдений)", False,
                  "нет ответа — сбой связи, поведение агента не проверено")
            return
        collecting = [item for item in watch_ids
                      if before.get(item["arguments"]["id"]) == "collecting"]
        check("наблюдения собираются ДО действия",
              len(collecting) == len(watch_ids),
              f"собираются: {[item['arguments']['id'] for item in collecting]} "
              f"из {[item['arguments']['id'] for item in watch_ids]}")
        request(port, "POST", f"/api/agent/sessions/{sessions[0]}/select")
        if action == "cancel":
            status, data = request(port, "POST", "/api/agent/state/cancel")
        elif action == "pause":
            status, data = request(port, "POST", "/api/agent/state/pause")
        elif action == "delete_session":
            status, data = request(port, "DELETE", f"/api/agent/sessions/{sessions[0]}")
        else:
            status, data = request(port, "DELETE", f"/api/agent/tasks/{task_id}")
        check(f"{title}: действие выполнено (HTTP {status})", status == 200,
              str(data)[:200])
        after = watches()
        if after is None:
            check(f"{title}: сервер MCP отвечает после действия", False,
                  "нет ответа — сбой связи")
            return
        stopped = [item["arguments"]["id"] for item in watch_ids
                   if after.get(item["arguments"]["id"]) != "collecting"]
        if expect_stopped:
            check(f"{title}: сервер получил остановку сбора ({len(stopped)} из "
                  f"{len(watch_ids)})", len(stopped) == len(watch_ids),
                  f"состояния: {after}")
            if action == "cancel":
                # Журнал читаем только там, где задача ЖИВА после действия: при
                # удалении задачи её диалог исчезает вместе с записью журнала, и
                # проверять в нём нечего (в лог приложения строка всё равно уходит).
                status, history = request(port, "GET", "/api/agent/history")
                log = " ".join(item.get("text", "")
                               for item in (history.get("log") or []))
                check(f"{title}: в чате задачи есть строка про остановку сборов",
                      "Внешние сборы задачи" in log, log[-200:])
            check(f"{title}: история наблюдения НЕ удалена (stop ≠ delete)",
                  all(after.get(item["arguments"]["id"]) == "paused"
                      for item in watch_ids), f"состояния: {after}")
        else:
            check(f"{title}: сбор продолжается (так и задумано)",
                  all(after.get(item["arguments"]["id"]) == "collecting"
                      for item in watch_ids), f"состояния: {after}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:  # noqa: BLE001
            proc.kill()


TOOLS = []


def main():
    global TOOLS
    print("Живая проверка уборки внешних сборов MCP (реальный сервер)")
    TOOLS = server_tools()
    if not TOOLS:
        print("Сервер MCP недоступен: поднимите SSH-туннель и проверьте токен в .env")
        return 2
    print("инструменты сервера:", ", ".join(item["tool"] for item in TOOLS))
    if not any(item["tool"] == START_TOOL for item in TOOLS):
        print(f"на сервере нет инструмента {START_TOOL} — проверять нечего")
        return 2
    own_before = watches() or {}
    print("наблюдений на сервере до проверки:", len(own_before),
          f"(собирается: {sum(1 for v in own_before.values() if v == 'collecting')})")
    try:
        # 1. Отмена периодической задачи (кнопка «Отменить»).
        started = start_watch(10.1, 20.1, TOOLS)
        if not started:
            return 2
        run_case("отмена периодической задачи", [started], "cancel", True)

        # 2. Удаление задачи (корзина у задачи в списке).
        started = start_watch(10.2, 20.2, TOOLS)
        run_case("удаление задачи", [started], "delete_session", True)

        # 3. Удаление ПРОЕКТА: сборы ВСЕХ его задач (две задачи — две галочки).
        first = start_watch(10.3, 20.3, TOOLS)
        second = start_watch(10.4, 20.4, TOOLS)
        run_case("удаление проекта", [first], "delete_task", True, second=second)

        # 4. Обычная (не периодическая) задача: отмена сбор НЕ трогает.
        started = start_watch(10.5, 20.5, TOOLS)
        run_case("обычная задача", [started], "cancel", False, periodic=False)

        # 5. «Пауза» периодической задачи: сбор тоже НЕ останавливается — задача жива.
        started = start_watch(10.6, 20.6, TOOLS)
        run_case("пауза периодической задачи", [started], "pause", False)
    finally:
        # Убираем за собой: удаляем ВСЕ наблюдения, созданные проверкой.
        for watch_id in list(CREATED):
            drop_watch(watch_id)
        time.sleep(1)
        left = watches() or {}
        zombies = [item for item in CREATED if item in left]
        check("за проверкой не осталось висящих наблюдений", not zombies,
              f"осталось: {zombies}")
        shutil.rmtree(_TMP, ignore_errors=True)

    print("\nИтог: " + ("ПРОВАЛЕНО проверок: " + str(len(FAILURES))
                        if FAILURES else "все проверки пройдены"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
