"""Живая проверка СВОИХ MCP-серверов на VPS (через SSH-туннели).

Запуск (нужны поднятые туннели; токены и адреса берутся из `.env`, как у агента):

    ./venv/bin/python tools/check_mcp_live.py
    ./venv/bin/python tools/check_mcp_live.py --server city_registry
    ./venv/bin/python tools/check_mcp_live.py --call city_registry.list_cities

Чем отличается от `tools/check_mcp.py`: тот проверяет КОД агента на заглушках,
без сети и процессов. Этот скрипт проверяет САМИ СЕРВЕРЫ — тем же клиентом,
которым ходит агент (`app.ai.mcp`), и отвечает на вопрос «почему инструменты
молчат»: не поднят туннель, отсутствует/чужой токен, сервер ответил, но объявил
не тот набор инструментов.

  [1] туннель: кто слушает локальный порт из адреса сервера (`lsof`) — должен
      быть ssh; никто не слушает → туннель не поднят;
  [2] окружение: заданы ли адрес и токен (`availability_error` сообщает причину
      ДО обращения к серверу);
  [3] рукопожатие и `tools/list`: имя сервера из ответа, число инструментов и их
      имена (у city_registry ожидаются ровно save_city и list_cities);
  [4] `--call` — необязательная ЖИВАЯ проверка вызова. Разрешены ТОЛЬКО
      читающие инструменты: вызов, который создаёт результат или что-то удаляет
      (см. `is_deferred_call`), отклоняется — проверка не должна писать данные.

Секреты не печатаются: видно только «токен задан» и его длину. Код возврата:
0 — все проверенные серверы ответили, 1 — есть отказы.
"""

import argparse
import json
import os
import subprocess
import sys
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ai import mcp as mcp_store  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    """Печать результата проверки в стиле tools/check_mcp.py."""
    print(("  ok   " if ok else "  FAIL ") + str(name)
          + ("" if ok or not detail else f" — {detail}"))
    if not ok:
        FAILED.append(str(name))
    return bool(ok)


def remote_servers(only=None):
    """Серверы реестра с транспортом http (то есть свои, на VPS): id, запись, URL."""
    out = []
    for entry in mcp_store.servers():
        if mcp_store.transport_of(entry) != mcp_store.HTTP_TRANSPORT:
            continue
        server_id = str(entry.get("id") or "")
        if only and server_id != only:
            continue
        out.append((server_id, entry, mcp_store.server_url(entry)))
    return out


def port_of(url):
    """Порт из адреса сервера (у http-туннелей он всегда есть)."""
    parsed = urlparse(str(url or ""))
    if parsed.port:
        return str(parsed.port)
    return "443" if parsed.scheme == "https" else "80"


def listener(port):
    """Имя процесса, слушающего локальный порт, либо пустая строка.

    `lsof` может быть недоступен (или запрещён песочницей) — тогда проверка
    туннеля пропускается с пометкой, а не валит всю проверку сервера.
    """
    try:
        done = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                              timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    lines = done.stdout.decode("utf-8", "replace").splitlines()[1:]
    for row in lines:
        parts = row.split()
        if parts:
            return parts[0]
    return ""


def check_tunnel(server_id, url):
    """Проверка [1]: туннель до сервера. Возвращает True/False/None (неизвестно)."""
    port = port_of(url)
    who = listener(port)
    if who is None:
        print(f"  --   туннель {server_id}: порт {port} не проверен (нет lsof)")
        return None
    if who == "ssh":
        return check(f"туннель {server_id}: порт {port} слушает ssh", True)
    if who:
        return check(f"туннель {server_id}: порт {port} слушает {who}, а не ssh",
                     False, "порт занят другим процессом")
    label = "com.user." + server_id.replace("_", "-") + "-mcp-tunnel"
    return check(f"туннель {server_id}: порт {port} никто не слушает", False,
                 f"туннель не поднят: launchctl print gui/{os.getuid()}/{label}")


def check_call(server_id, spec, arguments):
    """Проверка [4]: живой вызов ЧИТАЮЩЕГО инструмента (запись отклоняется)."""
    tool = str(spec or "").split(".", 1)[1] if "." in str(spec or "") else ""
    if not tool:
        return check(f"вызов {server_id}: имя инструмента не указано", False,
                     "формат: --call сервер.инструмент")
    call = {"server": server_id, "tool": tool, "arguments": arguments}
    if mcp_store.is_deferred_call(call):
        return check(f"вызов {server_id} · {tool}", False,
                     "инструмент создаёт результат или удаляет данные — живая "
                     "проверка выполняет только чтения")
    result = mcp_store.call_tool(server_id, tool, arguments)
    text = str(result.get("text") or "").replace("\n", " ")
    ok = bool(result.get("ok"))
    return check(f"вызов {server_id} · {tool}", ok,
                 str(result.get("error") or "")[:200] or text[:200])


def check_server(server_id, entry, url, call=None, arguments=None):
    """Все проверки одного сервера. Возвращает True, если он ответил и объявил инструменты."""
    print(f"\n{server_id} — {entry.get('name')} ({url})")
    check_tunnel(server_id, url)

    token_name = str(entry.get("token_env") or "")
    token = mcp_store.server_token(entry)
    print(f"  --   токен: {token_name or '(не задан в записи)'} — "
          + (f"задан, длина {len(token)}" if token else "НЕ ЗАДАН"))
    gap = mcp_store.availability_error(entry)
    if not check(f"окружение {server_id} пригодно для обращения", not gap, gap):
        return False

    found = mcp_store.discover(server_id, force=True)
    if not check(f"рукопожатие и tools/list {server_id}", found.get("ok"),
                 str(found.get("error") or "")[:200]):
        return False
    tools = [str(item.get("name") or "") for item in (found.get("tools") or [])]
    print(f"  --   сервер ответил как «{found.get('server_name')}», "
          f"инструментов: {len(tools)}")
    for name in tools:
        print(f"       · {name}")
    check(f"у {server_id} есть хотя бы один инструмент", bool(tools))

    if call:
        # Вызов относится к ОДНОМУ серверу: имя инструмента у другого сервера
        # может совпасть случайно (жизнь: open_meteo · list_cities), и проверка
        # валилась бы на инструменте, которого у него нет.
        if str(call).split(".", 1)[0] == server_id:
            check_call(server_id, call, arguments or {})
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Живая проверка своих MCP-серверов на VPS (через туннели)")
    parser.add_argument("--server", default="",
                        help="проверить только этот сервер реестра")
    parser.add_argument("--call", default="",
                        help="живой вызов читающего инструмента: сервер.инструмент")
    parser.add_argument("--args", default="{}",
                        help="аргументы вызова --call в виде JSON (по умолчанию {})")
    options = parser.parse_args()

    entries = remote_servers(options.server or None)
    print("Живая проверка MCP-серверов проекта (через SSH-туннели)")
    if not entries:
        print("\nСвоих серверов на VPS в реестре нет — проверять нечего.")
        return 1

    arguments = {}
    if options.call:
        # Имя сервера в --call главнее --server: вызов относится ровно к одному
        # серверу, и проверяем тогда только его.
        call_server = options.call.split(".", 1)[0]
        if options.server and options.server != call_server:
            print(f"\n--server {options.server} не совпадает с --call "
                  f"{options.call} — вызов относится к серверу {call_server}.")
            return 1
        entries = remote_servers(call_server)
        try:
            arguments = json.loads(options.args)
        except ValueError as exc:
            print(f"\nАргументы --args не разобраны: {exc}")
            return 1
        if not isinstance(arguments, dict):
            print("\nАргументы --args должны быть объектом JSON, например "
                  '{"city": "Москва"}')
            return 1

    answered = 0
    for server_id, entry, url in entries:
        if check_server(server_id, entry, url, call=options.call,
                        arguments=arguments):
            answered += 1

    print("\nИтог: " + (f"ПРОВАЛЕНО проверок: {len(FAILED)} → {FAILED}"
                       if FAILED else
                       f"все проверки пройдены ({answered} сервер(ов) ответили)"))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
