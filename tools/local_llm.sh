#!/usr/bin/env bash
# Локальная модель (MLX) — установка, запуск и состояние.
#
# Локальный источник ответа чат-бота: OpenAI-совместимый сервер mlx_lm.server на
# этом же Mac (Apple Silicon). Всё окружение лежит ВНУТРИ проекта —
# data/local_llm (venv с mlx-lm, веса модели, журнал сервера); наружу, в
# домашний каталог или системные пути, не пишется ничего.
#
# Почему ОТДЕЛЬНЫЙ venv: mlx-lm требует Python 3.11+, а venv проекта — 3.12, но
# держать в нём тяжёлый ML-стек (mlx, ~350 МБ) незачем: веб-приложению он не
# нужен. Интерпретатор берётся СИСТЕМНЫЙ (`brew install python@3.12`), а если
# подходящего нет — `install` скачает uv и поставит Python сам (тогда всё
# останется внутри проекта).
#
# Использование:
#   tools/local_llm.sh install          окружение + веса (долго: ~5 ГБ)
#   tools/local_llm.sh pull             только веса модели
#   tools/local_llm.sh start            запустить сервер (в фоне, отдельным процессом)
#   tools/local_llm.sh stop             остановить сервер
#   tools/local_llm.sh restart          перезапустить
#   tools/local_llm.sh status           что установлено и отвечает ли сервер
#   tools/local_llm.sh logs             следить за журналом сервера
#   tools/local_llm.sh ask "вопрос"     один запрос к модели (проверка живьём)
#
# Выбор источника ответа («локальная / удалённая») делает приложение — кнопкой в
# панели workspace; этот скрипт только ставит и держит сервер.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLM_HOME="${LOCAL_LLM_HOME:-$ROOT/data/local_llm}"
UV="$LLM_HOME/bin/uv"
PY="$LLM_HOME/venv/bin/python"
MODEL="${LOCAL_LLM_MODEL:-mlx-community/Qwen3-8B-4bit}"
HOST="${LOCAL_LLM_HOST:-127.0.0.1}"
PORT="${LOCAL_LLM_PORT:-8080}"
WANT_PYTHON="${LOCAL_LLM_PYTHON:-3.12}"
LOG="$LLM_HOME/logs/server.log"

# Кэши — ВНУТРЬ проекта: ни пакеты, ни веса не должны появляться в ~/.cache
# (иначе «локальная модель» расползлась бы по машине, а установка перестала бы
# быть переносимой вместе с проектом).
export UV_PYTHON_INSTALL_DIR="$LLM_HOME/python"
export UV_CACHE_DIR="$LLM_HOME/uvcache"
export HF_HOME="$LLM_HOME/models"
export HF_HUB_DISABLE_TELEMETRY=1

# Python проекта — им запускаются команды управления: модуль app/ai/local_llm.py
# знает все пути и правила запуска, а тяжёлых зависимостей у него нет.
PROJECT_PY="$ROOT/venv/bin/python"
[ -x "$PROJECT_PY" ] || PROJECT_PY="$(command -v python3)"

say() { printf '%s\n' "$*"; }
die() { printf 'Ошибка: %s\n' "$*" >&2; exit 1; }

# Подходящий СИСТЕМНЫЙ интерпретатор (mlx-lm требует 3.11+): сначала точная
# версия, потом «какой есть» из Homebrew. Пустая строка — не нашли.
find_system_python() {
    local candidate
    for candidate in \
        "/opt/homebrew/opt/python@$WANT_PYTHON/bin/python$WANT_PYTHON" \
        "$(command -v python$WANT_PYTHON 2>/dev/null || true)" \
        "/opt/homebrew/bin/python3.13" "/opt/homebrew/bin/python3.12" \
        "/opt/homebrew/bin/python3.11"; do
        if [ -n "$candidate" ] && [ -x "$candidate" ] \
           && "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
            printf '%s' "$candidate"
            return 0
        fi
    done
    return 1
}

ensure_uv() {
    [ -x "$UV" ] && return 0
    mkdir -p "$LLM_HOME/bin"
    say "Скачиваю uv в $LLM_HOME/bin …"
    local tmp="$LLM_HOME/uv.tar.gz"
    curl -fsSL -o "$tmp" \
        "https://github.com/astral-sh/uv/releases/latest/download/uv-aarch64-apple-darwin.tar.gz" \
        || die "не удалось скачать uv (нужна сеть)"
    tar xzf "$tmp" -C "$LLM_HOME/bin" --strip-components=1 uv-aarch64-apple-darwin/uv
    rm -f "$tmp"
    [ -x "$UV" ] || die "uv не распаковался"
}

ensure_venv() {
    if [ ! -x "$PY" ]; then
        local system_python
        system_python="$(find_system_python || true)"
        if [ -n "$system_python" ]; then
            say "Создаю venv ($system_python) в $LLM_HOME/venv …"
            "$system_python" -m venv "$LLM_HOME/venv"
        else
            # Системного Python 3.11+ нет — ставим свой через uv (внутри проекта).
            say "Подходящего системного Python нет — ставлю его через uv …"
            ensure_uv
            "$UV" venv --python "$WANT_PYTHON" "$LLM_HOME/venv"
        fi
    fi
    if ! "$PY" -c "import mlx_lm" >/dev/null 2>&1; then
        say "Ставлю mlx-lm …"
        if [ -x "$UV" ]; then
            "$UV" pip install --python "$PY" mlx-lm
        else
            "$PY" -m pip install --quiet --upgrade pip
            "$PY" -m pip install mlx-lm
        fi
    fi
}

pull() {
    ensure_venv
    say "Качаю веса $MODEL в $HF_HOME (около 5 ГБ) …"
    "$LLM_HOME/venv/bin/hf" download "$MODEL"
}

cmd_install() {
    command -v curl >/dev/null || die "нужен curl"
    ensure_venv
    pull
    cmd_status
    say ""
    say "Готово. Запуск сервера: tools/local_llm.sh start"
    say "Переключение источника ответа — кнопкой в панели workspace приложения."
}

cmd_start() {
    "$PROJECT_PY" - "$ROOT" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from app.ai import local_llm

# Выбор источника читаем ИЗ ФАЙЛА: отдельный процесс не проходит через
# main.py, поэтому apply_saved_source зовём сами — иначе status показывал бы
# «удалённая» независимо от того, что выбрано в интерфейсе.
local_llm.apply_saved_source()

state = local_llm.status()
if state["server"]["running"]:
    print("Сервер уже отвечает:", ", ".join(state["server"]["models"]))
    raise SystemExit(0)
state = local_llm.start()
print("Сервер запускается (pid %s). Журнал: %s" % (
    state["server"]["pid"] or "?", state["server"]["log"]))
print("Готовность проверяйте командой: tools/local_llm.sh status")
PY
}

cmd_stop() {
    "$PROJECT_PY" - "$ROOT" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from app.ai import local_llm

# Выбор источника читаем ИЗ ФАЙЛА: отдельный процесс не проходит через
# main.py, поэтому apply_saved_source зовём сами — иначе status показывал бы
# «удалённая» независимо от того, что выбрано в интерфейсе.
local_llm.apply_saved_source()

state = local_llm.stop()
print("Сервер остановлен." if not state["server"]["running"] else "Сервер всё ещё отвечает.")
PY
}

cmd_status() {
    "$PROJECT_PY" - "$ROOT" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from app import config
from app.ai import local_llm

# Выбор источника читаем ИЗ ФАЙЛА: отдельный процесс не проходит через
# main.py, поэтому apply_saved_source зовём сами — иначе status показывал бы
# «удалённая» независимо от того, что выбрано в интерфейсе.
local_llm.apply_saved_source()

state = local_llm.status()
installed = state["installed"]
server = state["server"]
print("Источник ответа : %s (%s)" % (state["source"], state["title"]))
print("Модель          : %s" % state["model"])
print("Каталог         : %s" % installed["home"])
print("Окружение mlx-lm: %s" % ("есть" if installed["venv"] and installed["mlx"] else "НЕТ"))
print("Веса модели     : %s" % (
    "%.2f ГБ" % (installed["model_bytes"] / 1e9) if installed["model"] else "НЕТ"))
print("Сервер          : %s" % (
    "отвечает, pid %s, модель %s" % (server["pid"], ", ".join(server["models"]))
    if server["running"] else ("запускается" if server["starting"] else "не запущен")))
print("Адрес сервера    : %s" % config.LOCAL_LLM_BASE_URL)
print("Журнал          : %s" % server["log"])
print("Итог            : %s" % state["hint"])
PY
}

cmd_ask() {
    local prompt="${1:-Скажи одним предложением, кто ты.}"
    command -v curl >/dev/null || die "нужен curl"
    curl -fsS -m 300 -X POST "http://$HOST:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "$(printf '{"model":"%s","messages":[{"role":"user","content":%s}],"max_tokens":200}' \
              "$MODEL" "$(printf '%s' "$prompt" | "$PROJECT_PY" -c 'import json,sys; print(json.dumps(sys.stdin.read()))')")" \
        | "$PROJECT_PY" -c 'import json,sys; d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"]); print("--- токены:", d.get("usage"))' \
        || die "сервер не ответил (запущен ли он? tools/local_llm.sh status)"
}

case "${1:-status}" in
    install) cmd_install ;;
    pull)    pull ;;
    start)   cmd_start ;;
    stop)    cmd_stop ;;
    restart) cmd_stop; cmd_start ;;
    status)  cmd_status ;;
    logs)    mkdir -p "$(dirname "$LOG")"; tail -f "$LOG" ;;
    ask)     shift; cmd_ask "$@" ;;
    *)       sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' ; exit 1 ;;
esac
