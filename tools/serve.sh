#!/usr/bin/env bash
#
# Запуск сайта для ДОСТУПА ИЗ ВНЕШНЕЙ СЕТИ.
#
# Зачем отдельный скрипт, если есть `uvicorn main:app`. Обычная команда слушает
# 127.0.0.1 — снаружи такой сервер недоступен, и проброс порта на роутере
# (внешний адрес → адрес мака в сети дома) упирался бы в «connection refused»,
# хотя приложение работает. Здесь uvicorn слушает 0.0.0.0, то есть принимает
# соединения и с петли, и из локальной сети, и из интернета (приходит уже
# проброшенный трафик).
#
# ПОЧЕМУ ЗДЕСЬ ЖЕ СТОИТ ПРОВЕРКА ПАРОЛЯ. Гейт доступа (app/auth.py) включён
# тогда и только тогда, когда в .env задан ACCESS_PASSWORD. Без пароля сервер
# поднялся бы и честно отвечал наружу 403 — «работает, но пускает только
# локальных». Это выглядело бы как поломка, а не как незаполненная строка, и
# владелец искал бы причину в роутере. Поэтому запуск «наружу» без пароля
# останавливается С ПРИЧИНОЙ и подсказкой, что именно дописать. Локально, как и
# раньше: `./venv/bin/python -m uvicorn main:app`.
#
# HTTPS БЕЗ ДОМЕНА (SITE_TLS=1, включён по умолчанию). Пароль входа и вся
# переписка идут через интернет, а по http их читает любой посредник (Wi-Fi
# кафе, провайдер). Домена нет, поэтому сертификат САМОПОДПИСАННЫЙ: он
# шифрует канал, но браузер один раз показывает предупреждение про
# неизвестного издателя — это ожидаемо, а не поломка (в сертификат вписаны
# имена всех адресов, по которым открывают сайт: петля, адрес в сети дома и
# внешний адрес). Появится домен — вместо самоподписанного ставится настоящий
# сертификат, и эта часть просто не нужна.
#
# Использование:
#   tools/serve.sh start              # поднять в фоне и напечатать адреса
#   tools/serve.sh start --foreground  # то же, но в текущем окне терминала
#   tools/serve.sh stop|restart|status|logs
#   tools/serve.sh cert                # перевыпустить самоподписанный сертификат
#
# Настройки (переменными окружения или строками в .env):
#   SITE_HOST   (по умолчанию 0.0.0.0) — адрес прослушивания
#   SITE_PORT   (по умолчанию 8000)    — порт
#   SITE_TLS    (по умолчанию 1)       — 1: https с самоподписанным сертификатом,
#                                        0: обычный http (пароль пойдёт открыто)
#   SITE_TLS_NAMES                     — имена для сертификата (через запятую),
#                                        по умолчанию: петля, адрес в сети, PUBLIC_URL
#   PUBLIC_URL  (необязательно)        — внешний адрес для подсказки в status
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/venv/bin/python"
RUN_DIR="$ROOT/data/run"
TLS_DIR="$ROOT/data/tls"
CERT_FILE="$TLS_DIR/site.crt"
KEY_FILE="$TLS_DIR/site.key"
PID_FILE="$RUN_DIR/serve.pid"
LOG_FILE="$RUN_DIR/serve.log"
ENV_FILE="$ROOT/.env"
# Отпечаток .env на момент запуска: настройки (и пароль входа в первую очередь)
# читаются ОДИН РАЗ при старте приложения, поэтому правка .env у работающего
# сервера ничего не меняет. Живой случай: пароль в .env поменяли, вход отвечал
# 401, и причина искалась в пароле, а не в неперезапущенном сервере.
ENV_STAMP="$RUN_DIR/env.stamp"

# Значение настройки: сначала окружение, потом .env (строки KEY=VALUE).
env_value() {
    local key="$1" value
    value="$(printenv "$key" 2>/dev/null || true)"
    if [ -n "$value" ]; then printf '%s' "$value"; return; fi
    if [ -f "$ROOT/.env" ]; then
        sed -n "s/^[[:space:]]*${key}[[:space:]]*=[[:space:]]*//p" "$ROOT/.env" \
            | tail -1 | sed 's/^"//; s/"$//; s/[[:space:]]*$//'
    fi
}

HOST="$(env_value SITE_HOST)"; HOST="${HOST:-0.0.0.0}"
PORT="$(env_value SITE_PORT)"; PORT="${PORT:-8000}"
PUBLIC_URL="$(env_value PUBLIC_URL)"
PASSWORD="$(env_value ACCESS_PASSWORD)"
TLS="$(env_value SITE_TLS)"; TLS="${TLS:-1}"
if [ "$TLS" = "1" ]; then SCHEME="https"; else SCHEME="http"; fi

lan_ip() {
    # Адрес мака в локальной сети (для проверки «с другого устройства дома»).
    local ip
    for iface in en0 en1 en2; do
        ip="$(ipconfig getifaddr "$iface" 2>/dev/null || true)"
        if [ -n "$ip" ]; then printf '%s' "$ip"; return; fi
    done
}

# Имена для сертификата: петля, адрес в сети дома и внешний адрес. Без них
# браузер ругается ДВУМЯ предупреждениями (неизвестный издатель И «имя не
# совпадает»), а со ними — только одним, про издателя.
cert_names() {
    local names; names="$(env_value SITE_TLS_NAMES)"
    if [ -n "$names" ]; then printf '%s' "$names"; return; fi
    local ip host out="DNS:localhost,IP:127.0.0.1,IP:::1"
    ip="$(lan_ip)"
    [ -n "$ip" ] && out="$out,IP:$ip"
    host="${PUBLIC_URL#*://}"; host="${host%%/*}"; host="${host%%:*}"
    case "$host" in
        ""|localhost|127.0.0.1) ;;
        *[!0-9.]*) out="$out,DNS:$host" ;;   # имя, а не адрес
        *) out="$out,IP:$host" ;;
    esac
    printf '%s' "$out"
}

ensure_cert() {
    [ "$TLS" = "1" ] || return 0
    command -v openssl >/dev/null 2>&1 || {
        echo "Отказ: нужен openssl, чтобы выпустить самоподписанный сертификат." >&2
        echo "Либо запустите без шифрования: SITE_TLS=0 tools/serve.sh start" >&2
        exit 1
    }
    if [ -s "$CERT_FILE" ] && [ -s "$KEY_FILE" ]; then return 0; fi
    mkdir -p "$TLS_DIR"
    # Ключ только для владельца: это не «сертификат сайта вообще», а ключ
    # именно этого мака.
    umask 077
    openssl req -x509 -newkey rsa:2048 -sha256 -days 825 -nodes \
        -keyout "$KEY_FILE" -out "$CERT_FILE" \
        -subj "/CN=fastapi-site" \
        -addext "subjectAltName=$(cert_names)" >/dev/null 2>&1
    echo "Выпущен самоподписанный сертификат:"
    echo "  сертификат: $CERT_FILE"
    echo "  ключ:       $KEY_FILE (права только у владельца)"
    echo "  имена:      $(cert_names)"
    echo "Браузер один раз покажет предупреждение про неизвестного издателя —"
    echo "это ожидаемо: сертификат выписан нами, а не удостоверяющим центром."
}

running_pid() {
    # pid живого сервера этого проекта (пусто — не запущен). Проверяем, что по
    # pid действительно наш uvicorn: pid-файл может остаться от прошлого запуска,
    # а номер — достаться чужому процессу. Если `ps` недоступен (бывает в
    # песочницах и урезанных окружениях), доверяем проверке живости pid:
    # «ps не сработал» — не повод объявить работающий сервер неработающим.
    [ -f "$PID_FILE" ] || return 0
    local pid command
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    [ -n "$pid" ] || return 0
    kill -0 "$pid" 2>/dev/null || return 0
    command="$(ps -o command= -p "$pid" 2>/dev/null || true)"
    if [ -n "$command" ] && ! printf '%s' "$command" | grep -q "uvicorn main:app"; then
        return 0
    fi
    printf '%s' "$pid"
}

env_fingerprint() {
    [ -f "$ENV_FILE" ] || return 0
    shasum -a 256 "$ENV_FILE" 2>/dev/null | awk '{print $1}'
}

env_changed_since_start() {
    # Истина, если .env правили ПОСЛЕ запуска работающего сервера.
    local now_fp env_time started pid
    now_fp="$(env_fingerprint)"
    [ -n "$now_fp" ] || return 1
    if [ -f "$ENV_STAMP" ]; then
        [ "$now_fp" != "$(cat "$ENV_STAMP" 2>/dev/null)" ]
        return
    fi
    # Отпечатка нет (сервер запущен до появления этой проверки): сравниваем время
    # правки .env со временем старта процесса. Не удалось узнать время (нет ps) —
    # молчим, а не пугаем наугад.
    pid="$(running_pid)"
    [ -n "$pid" ] || return 1
    started="$(ps -o lstart= -p "$pid" 2>/dev/null || true)"
    [ -n "$started" ] || return 1
    started="$(date -j -f "%a %b %d %T %Y" "$started" +%s 2>/dev/null || true)"
    [ -n "$started" ] || return 1
    env_time="$(stat -f %m "$ENV_FILE" 2>/dev/null || true)"
    [ -n "$env_time" ] || return 1
    [ "$env_time" -gt "$started" ]
}

warn_env_changed() {
    if env_changed_since_start; then
        echo "  ВНИМАНИЕ: .env правили ПОСЛЕ запуска — эти настройки (в том числе"
        echo "            пароль входа) НЕ применены: нужен tools/serve.sh restart."
    fi
}

ready() {
    # Сервер готов, когда отвечает СТРАНИЦА ВХОДА: это единственный маршрут,
    # доступный без входа всегда (при ACCESS_LOCAL_BYPASS=0 и /health за гейтом,
    # и проверка готовности по нему врала бы).
    curl -fsS -k -o /dev/null --max-time 2 "$SCHEME://127.0.0.1:$PORT/login" 2>/dev/null
}

# Аргументы uvicorn с шифрованием (пусто — обычный http).
uvicorn_tls_args() {
    [ "$TLS" = "1" ] || return 0
    printf '%s\n' --ssl-certfile "$CERT_FILE" --ssl-keyfile "$KEY_FILE"
}

print_urls() {
    local ip; ip="$(lan_ip)"
    echo "  локально:      $SCHEME://127.0.0.1:$PORT (пароль не спрашивается)"
    [ -n "$ip" ] && echo "  в сети дома:   $SCHEME://$ip:$PORT"
    if [ -n "$PUBLIC_URL" ]; then
        echo "  из интернета:  $PUBLIC_URL  (проброс порта на роутере)"
    else
        echo "  из интернета:  внешний адрес роутера, порт $PORT (проброс порта на роутере)"
    fi
    echo "  вход:          логин $(env_value ACCESS_USER || true)${PASSWORD:+ и пароль из .env (ACCESS_PASSWORD)}"
    if [ "$TLS" = "1" ]; then
        echo "  шифрование:    самоподписанный сертификат — браузер один раз спросит"
        echo "                 «Подробнее → Перейти на сайт» (это ожидаемо)"
    else
        echo "  ВНИМАНИЕ:      без шифрования пароль входа идёт открытым текстом"
    fi
}

require_password() {
    # Сочетание «http + кука только по https» иначе выглядело бы как поломка
    # входа: браузер молча не сохранит куку по открытому каналу, и верный пароль
    # «не срабатывал» бы. Лучше остановиться и сказать об этом.
    if [ "$TLS" != "1" ] && [ "$(env_value ACCESS_COOKIE_SECURE)" = "1" ]; then
        cat >&2 <<MSG
Отказ: ACCESS_COOKIE_SECURE=1, а запуск без шифрования (SITE_TLS=0).

Кука входа с флагом Secure по http браузером не сохраняется, поэтому вход
выглядел бы сломанным при верном пароле. Выберите одно:
    SITE_TLS=1 tools/serve.sh start      # https (по умолчанию) — тогда Secure уместен
    ACCESS_COOKIE_SECURE=0 в .env        # http — тогда кука без Secure
MSG
        exit 1
    fi
    if [ -z "$PASSWORD" ]; then
        cat >&2 <<MSG
Отказ: в .env не задан ACCESS_PASSWORD — наружу приложение не пустит никого.

Дело в том, что гейт доступа (app/auth.py) включён только при заданном пароле, и
без него внешний клиент получает 403 с причиной. Поднимать сервер в интернет в
таком виде бессмысленно, поэтому запуск остановлен здесь, а не «молча наружу».

Что сделать: дописать в .env две строки и повторить запуск
    ACCESS_USER=user
    ACCESS_PASSWORD=<ваш пароль>

Нужен только локальный доступ (как раньше, без пароля):
    ./venv/bin/python -m uvicorn main:app
MSG
        exit 1
    fi
}

cmd_start() {
    require_password
    local pid; pid="$(running_pid)"
    if [ -n "$pid" ]; then
        echo "Уже запущено (pid $pid). Адреса:"; print_urls; warn_env_changed; return 0
    fi
    mkdir -p "$RUN_DIR"
    ensure_cert
    local tls_args=()
    if [ "$TLS" = "1" ]; then
        while IFS= read -r line; do tls_args+=("$line"); done < <(uvicorn_tls_args)
    fi
    if [ "${1:-}" = "--foreground" ]; then
        echo "Сервер в этом окне. Адреса:"; print_urls
        exec "$PY" -m uvicorn main:app --host "$HOST" --port "$PORT" "${tls_args[@]}"
    fi
    # nohup + свой лог: сервер должен пережить закрытие терминала, иначе
    # «доступ из внешней сети» кончался бы вместе с окном терминала.
    nohup "$PY" -m uvicorn main:app --host "$HOST" --port "$PORT" "${tls_args[@]}" >>"$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    # Ждём не «секунду на всякий случай», а ГОТОВНОСТИ: приложение поднимает
    # планировщик периодических задач и применяет сохранённый источник ответа,
    # поэтому на слабом маке старт заметно длиннее секунды, и «не поднялось» на
    # живом сервере — это ложная тревога.
    for _ in $(seq 1 40); do
        if ready; then break; fi
        sleep 0.25
    done
    pid="$(running_pid)"
    if [ -z "$pid" ] || ! ready; then
        echo "Не поднялось (порт $PORT не отвечает). Хвост журнала ($LOG_FILE):" >&2
        tail -20 "$LOG_FILE" >&2 || true
        [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
        rm -f "$PID_FILE"
        exit 1
    fi
    mkdir -p "$RUN_DIR"; env_fingerprint > "$ENV_STAMP"
    echo "Запущено (pid $pid). Адреса:"; print_urls
    echo "  журнал:        $LOG_FILE (tools/serve.sh logs)"
}

cmd_stop() {
    local pid; pid="$(running_pid)"
    if [ -z "$pid" ]; then
        echo "Не запущено."; rm -f "$PID_FILE"; return 0
    fi
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.25
    done
    if kill -0 "$pid" 2>/dev/null; then
        echo "Не отвечает на мягкое завершение, посылаю SIGKILL (pid $pid)."
        kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE" "$ENV_STAMP"
    echo "Остановлено (pid $pid)."
}

cmd_status() {
    local pid; pid="$(running_pid)"
    if [ -z "$pid" ]; then
        # Отдельный случай: на порту кто-то есть, а нашего pid нет — обычно это
        # прежний запуск обычной командой `uvicorn main:app` (только петля),
        # и «не запущено» тут было бы неправдой.
        if curl -fsS -k -o /dev/null --max-time 2 "$SCHEME://127.0.0.1:$PORT/login" 2>/dev/null; then
            echo "На порту $PORT отвечает сервер БЕЗ pid-файла serve.sh —"
            echo "скорее всего это прежний запуск не наружу (только 127.0.0.1)."
            echo "Чтобы открыть доступ из внешней сети: tools/serve.sh stop не поможет —"
            echo "остановите тот процесс (Ctrl+C в его окне) и запустите tools/serve.sh start."
            return 0
        fi
        echo "Не запущено (pid-файл: $PID_FILE)."
        return 0
    fi
    if ready; then
        echo "Работает (pid $pid), слушает $HOST:$PORT, страница входа отвечает."
    else
        echo "Процесс жив (pid $pid), но порт $PORT не отвечает — смотрите журнал."
    fi
    print_urls
    warn_env_changed
    if [ -z "$PASSWORD" ]; then
        echo "  ВНИМАНИЕ: ACCESS_PASSWORD не задан — наружу отвечает 403, пускает только локальных."
    fi
}

cmd_cert() {
    # Перевыпуск: адреса могли смениться (например, появился PUBLIC_URL), а в
    # старом сертификате их нет — браузер тогда ругается ещё и на имя.
    rm -f "$CERT_FILE" "$KEY_FILE"
    ensure_cert
    local pid; pid="$(running_pid)"
    if [ -n "$pid" ]; then
        echo "Сертификат перевыпущен, но сервер (pid $pid) держит СТАРЫЙ в памяти:"
        echo "перезапустите его — tools/serve.sh restart."
    fi
}

case "${1:-start}" in
    start)    shift || true; cmd_start "${1:-}" ;;
    stop)     cmd_stop ;;
    restart)  cmd_stop; cmd_start ;;
    status)   cmd_status ;;
    logs)     tail -n "${2:-40}" "$LOG_FILE" ;;
    cert)     cmd_cert ;;
    *) echo "Использование: tools/serve.sh start|stop|restart|status|logs|cert [строк]" >&2; exit 2 ;;
esac
