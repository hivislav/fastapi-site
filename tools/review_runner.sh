#!/usr/bin/env bash
# СВОЯ МАШИНА КАК SELF-HOSTED РАННЕР ДЛЯ AI-РЕВЬЮ.
#
#   tools/review_runner.sh check                     # готова ли машина к ревью
#   tools/review_runner.sh install                   # скачать раннер GitHub
#   tools/review_runner.sh config <URL> <ТОКЕН> [ИМЯ]  # зарегистрировать машину
#   tools/review_runner.sh start|stop|status|logs    # жизнь раннера
#
# ЗАЧЕМ ЭТО. Облачный раннер GitHub не имеет ни пакета эмбеддингов, ни кэша
# моделей, ни постоянного каталога индекса: RAG по коду там либо считается
# встроенным офлайн-бэкендом, либо пересобирается с нуля на каждый прогон. На
# своей машине всё это есть, и ревью идёт по-настоящему: семантический поиск по
# коду и документации, индекс переживает прогоны, а команды `/ai_review_enable` и
# `/ai_review_disable` в приложении на этой же машине могут управлять пайплайном
# (переменная репозитория AI_REVIEW_WORKSPACE, см. .github/workflows/ai-review.yml).
#
# ЧТО СКРИПТ НЕ ДЕЛАЕТ. Не правит `.env`, не трогает рабочие данные приложения
# (`data/agent_workspace.json` только читается проверкой) и НЕ ЗАПУСКАЕТ ревью:
# его запускает GitHub, когда придёт pull request. Секреты не печатаются: токен
# регистрации передаётся аргументом и в журнал не пишется.
#
# ПЕРВАЯ РЕГИСТРАЦИЯ (порядок важен):
#   1) GitHub → репозиторий → Settings → Actions → Runners → New self-hosted
#      runner: там показан АДРЕС (URL) и выдаётся ТОКЕН (он живёт час);
#   2) tools/review_runner.sh check     — убедиться, что машина готова;
#   3) tools/review_runner.sh install   — скачать и распаковать раннер;
#   4) tools/review_runner.sh config <URL> <ТОКЕН> ai-review-mac;
#   5) tools/review_runner.sh start;
#   6) в репозитории завести переменную AI_REVIEW_RUNNER = метка раннера
#      (по умолчанию `ai-review`) — тогда `runs-on` уйдёт на эту машину.

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER_DIR="${REVIEW_RUNNER_DIR:-$PROJECT_ROOT/data/review_runner}"
RUN_DIR="$PROJECT_ROOT/data/run"
PID_FILE="$RUN_DIR/review-runner.pid"
LOG_FILE="$RUN_DIR/review-runner.log"
# Метки раннера: `ai-review` — та, которую ждёт workflow; имя машины добавляется
# второй меткой, чтобы можно было запускать и на конкретной машине.
DEFAULT_LABEL="ai-review"
RUNNER_VERSION="${REVIEW_RUNNER_VERSION:-2.338.0}"

say() { printf '%s\n' "$*"; }
fail() { printf '⚠ %s\n' "$*" >&2; }

# КЛЮЧИ МОДЕЛИ ИЗ .env ПРОЕКТА — В ОКРУЖЕНИЕ РАННЕРА. Своя машина уже умеет
# работать с моделью: ключ лежит в `.env` проекта (он в git не хранится и в
# выкладку раннера не попадает — каталог раннера получает ЧИСТУЮ копию кода).
# Раннер, запущенный этим скриптом, ПЕРЕДАЁТ свои переменные окружения задачам,
# поэтому секрет репозитория на своей машине не обязателен: достаточно `.env`.
#
# Разбор ФАЙЛА, а не `source`: значения из .env могут содержать кавычки, `$` и
# решётки, и выполнять их как команды оболочки нельзя — читаем построчно
# «КЛЮЧ=ЗНАЧЕНИЕ» и выставляем только то, что похоже на переменную.
load_env_file() {
    local file="${1:-$PROJECT_ROOT/.env}" count=0 key value
    [ -f "$file" ] || return 0
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in ''|'#'*) continue ;; esac
        case "$line" in *=*) ;; *) continue ;; esac
        key="${line%%=*}"
        value="${line#*=}"
        key="$(printf '%s' "$key" | tr -d '[:space:]')"
        case "$key" in
            ''|*[!A-Za-z0-9_]*) continue ;;
            [0-9]*) continue ;;
        esac
        value="${value%"${value##*[![:space:]]}"}"
        value="${value#"${value%%[![:space:]]*}"}"
        case "$value" in
            \"*\") value="${value#\"}"; value="${value%\"}" ;;
            \'*\') value="${value#\'}"; value="${value%\'}" ;;
        esac
        # Уже заданная переменная ГЛАВНЕЕ файла: у раннера может быть своё
        # окружение (например секрет из GitHub), и перекрывать его нельзя.
        if [ -z "$(eval "printf '%s' \"\${$key:-}\"")" ]; then
            export "$key=$value"
            count=$((count + 1))
        fi
    done < "$file"
    return 0
}

# Питон проекта: venv, если он есть (там пакеты эмбеддингов), иначе системный.
python_bin() {
    if [ -x "$PROJECT_ROOT/venv/bin/python" ]; then
        printf '%s' "$PROJECT_ROOT/venv/bin/python"
    else
        command -v python3 || true
    fi
}

# Архитектура машины в именовании GitHub (для скачивания раннера).
runner_arch() {
    case "$(uname -s)" in
        Darwin) printf 'osx-%s' "$( [ "$(uname -m)" = "arm64" ] && echo arm64 || echo x64 )" ;;
        Linux)  printf 'linux-%s' "$( [ "$(uname -m)" = "aarch64" ] && echo arm64 || echo x64 )" ;;
        *)      printf '' ;;
    esac
}

need_cmd() {
    if command -v "$1" >/dev/null 2>&1; then
        say "  ok   $1 — $(command -v "$1")"
        return 0
    fi
    fail "  НЕТ  $1 — $2"
    return 1
}

# ---------------------------------------------------------------------------
# check — готова ли машина: инструменты, эмбеддинги, каталог индекса
# ---------------------------------------------------------------------------
cmd_check() {
    local problems=0
    say "Проверка машины для self-hosted раннера AI-ревью."
    say "Каталог проекта: $PROJECT_ROOT"
    say "Каталог раннера:  $RUNNER_DIR"
    say ""
    say "Инструменты, без которых ревью не пойдёт:"
    need_cmd git "нужен сервер git и сама выкладка кода" || problems=$((problems + 1))
    need_cmd node "на Node работает MCP-сервер git (npm ci --prefix mcp_servers)" \
        || problems=$((problems + 1))
    need_cmd npm "ставит зависимости MCP-сервера" || problems=$((problems + 1))
    need_cmd curl "скачивает раннер и нужен самому GitHub Actions" \
        || problems=$((problems + 1))
    local py
    py="$(python_bin)"
    if [ -n "$py" ]; then
        # Версия важна: проект — Python 3.12 (в коде `X | None`, что 3.9 не умеет).
        local version
        version="$("$py" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)"
        if [ "$version" = "3.12" ] || [ "${version%%.*}" -ge 3 ] 2>/dev/null; then
            say "  ok   python — $py (версия $version)"
        else
            fail "  НЕТ  python 3.10+ — найден $version в $py"
            problems=$((problems + 1))
        fi
    else
        fail "  НЕТ  python3 — без него не запустить tools/ai_review.py"
        problems=$((problems + 1))
    fi
    say ""
    say "RAG по коду (можно и без него — будет встроенный офлайн-бэкенд):"
    if [ -n "$py" ]; then
        "$py" - "$PROJECT_ROOT" <<'PY'
import json
import sys
sys.path.insert(0, sys.argv[1])
try:
    from app.ai import rag_embedding
except Exception as exc:                       # noqa: BLE001
    print("  ⚠    не удалось прочитать состояние эмбеддингов: %s" % str(exc)[:200])
    raise SystemExit(0)
state = rag_embedding.backend_status()
print("  %s   бэкенд: %s%s" % ("ok  " if state["backend"] != "hashing" else "⚠   ",
                               state["backend"],
                               (" (" + state["model"] + ")") if state["model"] else ""))
if state.get("reason"):
    print("       причина: %s" % state["reason"])
print("       кэш моделей: %s" % state["cache_dir"])
if state["backend"] == "hashing":
    print("       подсказка: pip install sentence-transformers — тогда поиск по коду "
          "станет семантическим (модель скачается в кэш выше)")
PY
    fi
    say ""
    say "Связь с приложением (ею команды в чате управляют пайплайном):"
    if [ -f "$PROJECT_ROOT/data/agent_workspace.json" ]; then
        say "  ok   файл проекта есть — при запуске раннера он уйдёт задачам как"
        say "       AI_REVIEW_WORKSPACE: /ai_review_enable и /ai_review_disable"
        say "       будут включать и выключать ревью на pull request."
    else
        say "  ⚠    файла проекта нет ($PROJECT_ROOT/data/agent_workspace.json):"
        say "       команды в чате пайплайн не переключат — создайте проект в"
        say "       приложении (файл появится при первом сохранении)."
    fi
    say "       индекс кода (вне рабочей копии): ${AI_REVIEW_INDEX_DIR:-$RUN_DIR/review-index}"
    say ""
    say "Ключ модели (без него ревью не делается — но и это не сбой):"
    if [ -f "$PROJECT_ROOT/.env" ] && grep -qE '^[[:space:]]*DEEPSEEK_API_KEY=[^[:space:]]' \
            "$PROJECT_ROOT/.env"; then
        say "  ok   DEEPSEEK_API_KEY есть в $PROJECT_ROOT/.env — на СВОЕЙ машине"
        say "       секрет репозитория не нужен: скрипт передаст его раннеру"
        say "       (start читает .env проекта). В облачном раннере .env нет — там"
        say "       нужен секрет репозитория DEEPSEEK_API_KEY."
    else
        say "  ⚠    ключа в $PROJECT_ROOT/.env нет: заведите секрет репозитория"
        say "       DEEPSEEK_API_KEY (Settings → Secrets and variables → Actions)."
    fi
    say ""
    say "Каталог индекса (AI_REVIEW_INDEX_DIR у раннера):"
    local index_dir="${AI_REVIEW_INDEX_DIR:-$RUN_DIR/review-index}"
    if mkdir -p "$index_dir" 2>/dev/null; then
        say "  ok   $index_dir — запись есть ($(du -sh "$index_dir" 2>/dev/null | cut -f1) занято)"
    else
        fail "  НЕТ  $index_dir — нет прав на запись; задайте AI_REVIEW_INDEX_DIR"
        problems=$((problems + 1))
    fi
    say ""
    if [ "$problems" -eq 0 ]; then
        say "Машина готова: запускайте install → config → start."
        return 0
    fi
    fail "Не хватает $problems условий — см. строки «НЕТ» выше."
    return 1
}

# ---------------------------------------------------------------------------
# install — скачать раннер GitHub нужной архитектуры
# ---------------------------------------------------------------------------
cmd_install() {
    local arch tarball url
    arch="$(runner_arch)"
    if [ -z "$arch" ]; then
        fail "Не знаю эту систему: $(uname -s) $(uname -m). Скачайте раннер вручную."
        return 1
    fi
    tarball="actions-runner-${arch}-${RUNNER_VERSION}.tar.gz"
    url="https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/${tarball}"
    mkdir -p "$RUNNER_DIR" || { fail "Не создать $RUNNER_DIR"; return 1; }
    if [ -x "$RUNNER_DIR/config.sh" ]; then
        say "Раннер уже распакован: $RUNNER_DIR (обновление — удалите каталог и повторите)."
        return 0
    fi
    say "Скачиваю $url"
    if ! curl -fsSL "$url" -o "$RUNNER_DIR/$tarball"; then
        fail "Не скачалось: проверьте сеть и версию (REVIEW_RUNNER_VERSION=$RUNNER_VERSION)."
        return 1
    fi
    tar -xzf "$RUNNER_DIR/$tarball" -C "$RUNNER_DIR" || { fail "Не распаковалось"; return 1; }
    rm -f "$RUNNER_DIR/$tarball"
    say "Готово: $RUNNER_DIR"
    say "Дальше: tools/review_runner.sh config <URL> <ТОКЕН> [имя]"
    return 0
}

# ---------------------------------------------------------------------------
# config / start / stop / status / logs
# ---------------------------------------------------------------------------
cmd_config() {
    local url="${1:-}" token="${2:-}" name="${3:-review-runner-$(hostname -s)}"
    if [ -z "$url" ] || [ -z "$token" ]; then
        fail "Нужны адрес репозитория и токен регистрации: GitHub → Settings → Actions → Runners."
        say  "  tools/review_runner.sh config https://github.com/владелец/репозиторий ТОКЕН [имя]"
        return 1
    fi
    if [ ! -x "$RUNNER_DIR/config.sh" ]; then
        fail "Раннер не установлен ($RUNNER_DIR): сначала tools/review_runner.sh install"
        return 1
    fi
    ( cd "$RUNNER_DIR" && ./config.sh --url "$url" --token "$token" --name "$name" \
        --labels "$DEFAULT_LABEL,$(hostname -s)" --work _work --unattended ) \
        || { fail "Регистрация не прошла (токен живёт час — возьмите свежий)."; return 1; }
    say "Машина зарегистрирована с меткой «${DEFAULT_LABEL}»."
    say "Не забудьте переменную репозитория AI_REVIEW_RUNNER=$DEFAULT_LABEL."
    return 0
}

cmd_start() {
    if [ ! -x "$RUNNER_DIR/run.sh" ]; then
        fail "Раннер не установлен: tools/review_runner.sh install"
        return 1
    fi
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        say "Раннер уже работает (pid $(cat "$PID_FILE"))."
        return 0
    fi
    mkdir -p "$RUN_DIR"
    # Ключи из .env проекта — в окружение раннера: задачи self-hosted раннера
    # наследуют переменные процесса, поэтому ревью на своей машине работает БЕЗ
    # секрета в GitHub (см. load_env_file).
    load_env_file "$PROJECT_ROOT/.env"
    # ПУТИ ПРОЕКТА — ТОЖЕ В ОКРУЖЕНИЕ РАННЕРА. Это и есть связь «команды в чате
    # управляют пайплайном»: AI_REVIEW_WORKSPACE ведёт к файлу проекта, где
    # лежит настройка режима (её меняют `/ai_review_enable` и
    # `/ai_review_disable`), а AI_REVIEW_INDEX_DIR держит индекс кода ВНЕ рабочей
    # копии раннера — иначе он собирался бы заново на каждый PR и терялся при
    # каждой выкладке кода. Уже заданные значения не перекрываются: своё
    # окружение главнее.
    : "${AI_REVIEW_WORKSPACE:=$PROJECT_ROOT/data/agent_workspace.json}"
    : "${AI_REVIEW_INDEX_DIR:=$PROJECT_ROOT/data/run/review-index}"
    export AI_REVIEW_WORKSPACE AI_REVIEW_INDEX_DIR
    say "Файл проекта для пайплайна: ${AI_REVIEW_WORKSPACE}"
    say "Каталог индекса кода:       ${AI_REVIEW_INDEX_DIR}"
    if [ -n "${DEEPSEEK_API_KEY:-}" ]; then
        say "Ключ модели взят из ${PROJECT_ROOT}/.env — секрет в GitHub не нужен."
    else
        say "Ключа модели в .env нет: задайте секрет репозитория DEEPSEEK_API_KEY"
        say "(ревью без ключа честно не делается и не роняет pull request)."
    fi
    # ЗАПУСК В НОВОЙ СЕССИИ, а не просто `nohup … &`. Раннер — служба: он обязан
    # пережить закрытие терминала (и любого процесса, который его запустил), а
    # `nohup` оставляет его в той же группе процессов — сигнал группе (закрытие
    # окна, остановка запустившего скрипта) убил бы и раннер. Живой случай: раннер
    # поднялся, подключился к GitHub и умер вместе с запустившей его командой.
    # Питон нужен всё равно (разбор режима), поэтому отрыв делаем им: без
    # `setsid`, которого в macOS нет.
    if ! "$(python_bin)" - "$RUNNER_DIR/run.sh" "$LOG_FILE" "$PID_FILE" <<'PY'
import os
import subprocess
import sys

script, log_path, pid_path = sys.argv[1], sys.argv[2], sys.argv[3]
log = open(log_path, "a", buffering=1)
# start_new_session=True — новая СЕССИЯ (setsid своими руками): сигналы,
# посланные группе запустившего процесса, раннера не касаются. Окружение при
# этом наследуется — на этом и держится передача ключа и путей проекта.
proc = subprocess.Popen([script], cwd=os.path.dirname(script), stdin=subprocess.DEVNULL,
                        stdout=log, stderr=log, start_new_session=True)
with open(pid_path, "w") as fh:
    fh.write(str(proc.pid))
PY
    then
        fail "Не удалось запустить раннер (см. ${LOG_FILE})"
        return 1
    fi
    sleep 3
    if kill -0 "$(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null; then
        say "Раннер запущен (pid $(cat "$PID_FILE")). Журнал: ${LOG_FILE}"
        say "Статус в GitHub: ./tools/review_runner.sh status (или Settings → Actions → Runners)."
        return 0
    fi
    fail "Раннер не поднялся — смотрите ${LOG_FILE}"
    return 1
}

cmd_stop() {
    if [ ! -f "$PID_FILE" ]; then
        say "Раннер не запущен (pid-файла нет)."
        return 0
    fi
    local pid
    pid="$(cat "$PID_FILE" 2>/dev/null)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        # Гасим ВСЮ сессию раннера: он запускает дочерний Runner.Listener, и
        # сигнал одному процессу оставил бы второго висеть в фоне.
        "$(python_bin)" - "$pid" <<'PY'
import os
import signal
import sys
import time

pid = int(sys.argv[1])
try:
    os.killpg(os.getpgid(pid), signal.SIGTERM)
except (ProcessLookupError, PermissionError):
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
for _ in range(20):
    time.sleep(0.5)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        break
PY
        say "Раннер остановлен (pid $pid)."
    else
        say "Процесса раннера нет — только устаревший pid-файл."
    fi
    rm -f "$PID_FILE"
    return 0
}

cmd_status() {
    say "Каталог раннера: $RUNNER_DIR"
    if [ -x "$RUNNER_DIR/config.sh" ]; then
        say "Установлен: да"
        if [ -f "$RUNNER_DIR/.runner" ]; then
            say "Зарегистрирован: да (см. $RUNNER_DIR/.runner)"
        else
            say "Зарегистрирован: НЕТ — tools/review_runner.sh config <URL> <ТОКЕН>"
        fi
    else
        say "Установлен: НЕТ — tools/review_runner.sh install"
    fi
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        say "Работает: да (pid $(cat "$PID_FILE"))"
    else
        say "Работает: нет"
    fi
    say "Метка для AI_REVIEW_RUNNER: $DEFAULT_LABEL"
    return 0
}

# Что СЕЙЧАС думает пайплайн: тот же разбор, что делает сам прогон (команда
# читает настройку проекта и говорит, включено ли ревью и по какому проекту).
cmd_mode() {
    local py
    py="$(python_bin)"
    if [ -z "$py" ]; then
        fail "Нет python3 — разбор режима не выполнить."
        return 1
    fi
    AI_REVIEW_WORKSPACE="${AI_REVIEW_WORKSPACE:-$PROJECT_ROOT/data/agent_workspace.json}" \
    AI_REVIEW_INDEX_DIR="${AI_REVIEW_INDEX_DIR:-$RUN_DIR/review-index}" \
        "$py" "$PROJECT_ROOT/tools/ai_review.py" --mode-report \
        --repo "$PROJECT_ROOT" --workspace \
        "${AI_REVIEW_WORKSPACE:-$PROJECT_ROOT/data/agent_workspace.json}" \
        | sed 's/^/  /'
    return 0
}

cmd_logs() {
    if [ -f "$LOG_FILE" ]; then
        tail -n "${2:-40}" "$LOG_FILE"
    else
        say "Журнала ещё нет ($LOG_FILE) — раннер не запускался."
    fi
    return 0
}

case "${1:-}" in
    check)   cmd_check ;;
    install) cmd_install ;;
    config)  shift; cmd_config "$@" ;;
    start)   cmd_start ;;
    stop)    cmd_stop ;;
    mode)    cmd_mode ;;
    status)  cmd_status ;;
    logs)    cmd_logs "$@" ;;
    *)
        say "Своя машина как self-hosted раннер для AI-ревью."
        say ""
        say "  tools/review_runner.sh check                      готова ли машина"
        say "  tools/review_runner.sh install                    скачать раннер"
        say "  tools/review_runner.sh config <URL> <ТОКЕН> [имя] зарегистрировать"
        say "  tools/review_runner.sh start|stop|status|logs     жизнь раннера"
        say "  tools/review_runner.sh mode                      что сейчас думает пайплайн"
        say ""
        say "Подробности — в .github/workflows/ai-review.yml (шапка файла)."
        exit 0
        ;;
esac
