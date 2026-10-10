"""Локальная модель (MLX): сервер, состояние и переключатель источника ответа.

Локальный источник ответа — это OpenAI-совместимый сервер `mlx_lm.server`
(пакет mlx-lm) на ЭТОМ же Mac: адрес `config.LOCAL_LLM_BASE_URL`, модель
`config.LOCAL_LLM_MODEL`. Приложение само модель не считает — оно только
выбирает, куда уходит запрос (app/ai/client.py берёт адрес и ключ у
`config.provider_spec`), а этот модуль отвечает за три вещи:

* **состояние** (`status`) — установлено ли окружение и веса, отвечает ли
  сервер, какая модель загружена, что показать человеку в интерфейсе;
* **жизнь сервера** (`start` / `stop`) — запуск ОТДЕЛЬНЫМ процессом, чтобы
  загрузка весов (десятки секунд и гигабайты памяти) не занимала ни поток
  приложения, ни его память; сервер переживает перезапуск приложения и
  останавливается только явной командой;
* **выбор источника** (`apply_saved_source` / `save_source`) — «локальная или
  удалённая модель» из панели workspace; выбор хранится в
  `config.LLM_SOURCE_FILE` и переживает перезапуск.

Всё окружение лежит ВНУТРИ проекта (`config.LOCAL_LLM_HOME`, каталог
`data/local_llm`, он в .gitignore): venv с mlx-lm, каталог весов (HF_HOME
уводится туда же) и журнал сервера. Наружу — ни в домашний каталог, ни в
системные пути — модуль не пишет ничего, и это осознанное правило: локальная
модель не должна «расползаться» по машине.

Модуль НЕ вызывает модель: ни в `status`, ни в переключателе нет обращений к
LLM (только короткий HTTP-опрос «жив ли сервер»). Проверки работают без сервера.
"""

import json
import logging
import os
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from app import config

logger = logging.getLogger(__name__)

# Сколько ждём ответа при опросе «жив ли сервер» (секунды). Коротко: это
# проверка на каждом открытии панели, и она не должна тормозить интерфейс.
PROBE_TIMEOUT = 1.5
# Сколько даём серверу на загрузку весов, прежде чем считать запуск неудачным.
# 4,6 ГБ весов на M4 читаются с диска за считанные секунды, но первый запуск
# после загрузки модели может быть дольше (проверка целостности файлов).
START_TIMEOUT = 180.0
# Сколько ждём остановки сервера перед SIGKILL (секунды).
STOP_TIMEOUT = 15.0
# Аргументы шаблона чата для сервера: Qwen3 — «размышляющая» модель, и без
# этой настройки она тратит время на цепочку рассуждений перед каждым ответом
# (у удалённой модели reasoning выключен так же — см. LLM_DISABLE_THINKING).
CHAT_TEMPLATE_ARGS = '{"enable_thinking": false}'

# ОТСРОЧКА АВТООСТАНОВКИ (секунды). Переключились на удалённую модель — сервер
# локальной ещё не нужен, но и гасить его мгновенно нельзя: человек часто
# переключается туда-обратно, чтобы СРАВНИТЬ ответы, а каждая загрузка весов
# стоит секунд. Через эту отсрочку сервер останавливается сам и освобождает
# ~6 ГБ памяти; вернулись на локальную раньше — отсрочка отменяется, и ничего
# перезагружать не приходится. 0 — гасить сразу.
STOP_GRACE_SECONDS = max(0.0, float(os.getenv("LOCAL_LLM_STOP_GRACE", "120")))

# Момент последнего запроса на запуск: по нему интерфейс видит «запускается»,
# пока сервер ещё не отвечает (загрузка весов идёт в ДРУГОМ процессе).
_started_at: Optional[float] = None
# Запланированная автоостановка: таймер и момент срабатывания. Таймер живёт в
# процессе приложения; если приложение закрыть раньше, сервер останется
# работать — его остановит кнопка в панели или tools/local_llm.sh stop.
_stop_timer: Optional[threading.Timer] = None
_stop_deadline: Optional[float] = None
_stop_lock = threading.Lock()


class LocalLlmError(RuntimeError):
    """Локальная модель не может ответить — с понятной человеку причиной."""


# ---------------------------------------------------------------------------
# Пути внутри проекта
# ---------------------------------------------------------------------------
def home() -> str:
    """Каталог локальной модели внутри проекта (venv, веса, журнал)."""
    return str(config.LOCAL_LLM_HOME)


def venv_python() -> str:
    """Интерпретатор venv с mlx-lm (его ставит tools/local_llm.sh)."""
    return os.path.join(home(), "venv", "bin", "python")


def models_dir() -> str:
    """Каталог весов: сюда смотрит HF_HOME и у сервера, и у загрузки модели."""
    return os.path.join(home(), "models")


def log_path() -> str:
    """Журнал сервера (stdout+stderr): по нему видна причина неудачного запуска."""
    return os.path.join(home(), "logs", "server.log")


def pid_path() -> str:
    """Файл с номером процесса сервера (им же проверяем, что он наш)."""
    return os.path.join(home(), "server.pid")


def source_file() -> str:
    """Файл выбранного источника ответа."""
    return str(config.LLM_SOURCE_FILE)


# ---------------------------------------------------------------------------
# Состояние
# ---------------------------------------------------------------------------
def _fetch_json(url: str, timeout: float = PROBE_TIMEOUT) -> Optional[dict]:
    """GET JSON без исключений: сервер может не отвечать — это нормальный случай."""
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", "replace") or "{}")
    except (urllib.error.URLError, OSError, ValueError):
        return None


# СЕРВЕР ЖИВ, НО ГЕНЕРИРОВАТЬ НЕ МОЖЕТ. Поток генерации у mlx_lm.server один, а
# память — видеопамять мака: живой случай 09.10 — две задачи пошли к локальной
# модели одновременно, Metal не смог выделить буфер команд
# ([METAL] Insufficient Memory), поток умер, и после этого КАЖДЫЙ вызов отвечал
# HTTP 404 «generation thread died», хотя /v1/models продолжал отвечать и модель
# была объявлена. Отличить это состояние от «готов» по HTTP нельзя, поэтому его
# отмечает клиент, увидев характерный ответ (см. app/ai/client.py), а здесь оно
# живёт до перезапуска сервера.
_BROKEN: Dict[str, Any] = {}
# Когда сервер в последний раз перезапускали ПОСЛЕ такого сбоя. Живёт отдельно от
# отметки: отметку снимает и перезапуск, и ручная остановка, а передышка нужна
# именно между ПЕРЕЗАПУСКАМИ — иначе повторный сбой через секунду вызвал бы новый
# перезапуск, и это превратилось бы в цикл по тридцать секунд загрузки весов.
_BROKEN_RESTART_AT = 0.0
# Как часто разрешено ПЕРЕЗАПУСКАТЬ сервер после такого сбоя (секунды): без
# предела повторные вызовы превратились бы в цикл «упал — перезапустили».
BROKEN_RESTART_COOLDOWN = float(os.getenv("LOCAL_LLM_BROKEN_COOLDOWN", "120"))

# Предел памяти кэша промптов (МБ) и число хранимых кэшей: см. комментарий у
# команды запуска. Значения подобраны по замеру: агентский запрос держит три-четыре
# разных промпта, каждому хватает ~250 МБ.
PROMPT_CACHE_MB = max(0, int(os.getenv("LOCAL_LLM_PROMPT_CACHE_MB", "1024")))
PROMPT_CACHE_SEQUENCES = max(1, int(os.getenv("LOCAL_LLM_PROMPT_CACHE_SEQUENCES", "4")))
KV_BITS = int(os.getenv("LOCAL_LLM_KV_BITS", "0"))


def prompt_cache_bytes() -> int:
    """Предел кэша промптов в байтах (0 — без предела)."""
    return PROMPT_CACHE_MB * 1024 * 1024


def mark_broken(reason: str) -> None:
    """Запомнить, что сервер отвечает, но генерировать не может."""
    if not _BROKEN:
        logger.warning("Локальная модель: %s", reason)
    _BROKEN.update({"reason": str(reason), "at": time.time()})


def clear_broken() -> None:
    """Снять отметку (сервер перезапущен или выбран другой источник)."""
    _BROKEN.clear()


def broken_reason() -> str:
    """Почему сервер считается неработоспособным (пусто — не считается)."""
    return str(_BROKEN.get("reason") or "")


def probe(timeout: float = PROBE_TIMEOUT) -> dict:
    """Отвечает ли локальный сервер и какие модели он объявляет.

    Возвращает {"running", "models": [id, …], "loaded", "error"}. Ошибка сети —
    не исключение: «сервер не запущен» это обычное состояние, а не сбой.

    `loaded` — ОБЪЯВИЛ ли сервер модель. «Отвечает» и «может сгенерировать» —
    не одно и то же: у сервера, которому не нашлись веса, умирает поток
    генерации, а HTTP-часть продолжает жить и отвечает на /v1/models пустым
    списком (mlx_lm пишет об этом только в свой журнал). Считать такой сервер
    готовым — значит пропустить вызов в пустоту и показать человеку «модель
    промолчала» вместо причины.
    """
    data = _fetch_json(_models_url(), timeout)
    broken = broken_reason()
    if data is None:
        return {"running": False, "models": [], "loaded": False, "error": None,
                "broken": broken}
    models: List[str] = []
    for item in data.get("data") or []:
        if isinstance(item, dict) and item.get("id"):
            models.append(str(item["id"]))
    return {"running": True, "models": models, "loaded": bool(models),
            "error": None, "broken": broken}


def _models_url() -> str:
    return str(config.LOCAL_LLM_BASE_URL).rstrip("/") + "/models"


def _model_bytes() -> int:
    """Сколько занимают веса на диске (байты). 0 — весов нет.

    Считается по каталогам снапшотов HF-кэша: у модели может быть несколько
    снапшотов, и «модель установлена» — это про наличие самих файлов весов,
    а не про запись в кэше.
    """
    for directory in _model_dirs():
        total = 0
        for name in os.listdir(directory):
            if name.endswith(".safetensors") or name.endswith(".npz"):
                try:
                    total += os.path.getsize(os.path.join(directory, name))
                except OSError:  # noqa: PERF203 — файл мог исчезнуть
                    continue
        if total:
            return total
    return 0


def _model_dirs() -> List[str]:
    """Каталоги, где лежат веса модели: путь как есть или снапшоты HF-кэша.

    LOCAL_LLM_MODEL — либо идентификатор модели хаба («org/name»), либо путь к
    каталогу с весами. Первый случай разворачивается в снапшоты каталога
    models/hub/models--org--name (так их кладёт `hf download` с HF_HOME внутри
    проекта), второй проверяется напрямую.
    """
    model = str(config.LOCAL_LLM_MODEL)
    direct = os.path.join(home(), model) if not os.path.isabs(model) else model
    if os.path.isdir(direct):
        return [direct]
    if "/" not in model or model.startswith("/"):
        return []
    repo = "models--" + model.replace("/", "--")
    snapshots = os.path.join(models_dir(), "hub", repo, "snapshots")
    if not os.path.isdir(snapshots):
        return []
    return [
        os.path.join(snapshots, name)
        for name in sorted(os.listdir(snapshots))
        if os.path.isdir(os.path.join(snapshots, name))
    ]


def _pid_alive(pid: int) -> bool:
    """Живой ли процесс (0 — не проверяем; процессы мы не свои не трогаем)."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read_pid() -> int:
    try:
        with open(pid_path(), "r", encoding="utf-8") as fh:
            return int(str(json.load(fh).get("pid") or 0))
    except (OSError, ValueError, TypeError):
        return 0


def _process_command(pid: int) -> Optional[str]:
    """Командная строка процесса. None — спросить систему не удалось.

    Неудача возможна и это не ошибка: в ограниченном окружении (песочница,
    запрет на запуск `ps`) команду узнать нельзя, а модуль обязан работать.
    """
    try:
        done = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return done.stdout


def _pid_state(pid: int) -> str:
    """Состояние процесса из pid-файла: "dead" | "foreign" | "alive".

    Нужно там, где важно отличить «процесс УМЕР» от «процесс жив, но сервер ещё
    не ответил» (идёт загрузка весов): `_own_process` для этого не годится —
    он отвечает «не наш», когда спросить систему нельзя (нет `ps`, песочница) и
    сервер пока молчит, а это ровно состояние ЗАПУСКА.

    "foreign" — процесс жив, но командная строка говорит, что это не наш сервер:
    значит, номер в pid-файле устарел (после перезагрузки он мог достаться
    другому процессу). Чужой процесс не наш — и убивать его нельзя.
    """
    if pid <= 0 or not _pid_alive(pid):
        return "dead"
    command = _process_command(pid)
    if command is None:
        # Спросить систему нельзя — верим номеру из СВОЕГО pid-файла.
        return "alive"
    ours = "mlx_lm.server" in command or "mlx_lm/server" in command
    return "alive" if ours else "foreign"


def _own_process(pid: int) -> bool:
    """Наш ли это процесс — проверка ПЕРЕД ОСТАНОВКОЙ (осторожная).

    Номер из файла мог достаться чужому процессу после перезагрузки, и убивать
    его нельзя. Если спросить систему нельзя (нет `ps`), остаётся косвенная
    проверка: процесс с записанным номером жив И на нашем адресе отвечает сервер
    модели. Такую пару «номер из своего файла + отвечающий по своему адресу
    сервер» считаем своим сервером — иначе в ограниченном окружении модуль не
    смог бы даже остановить модель, которую сам запустил.
    """
    state = _pid_state(pid)
    if state == "dead":
        return False
    if state == "alive" and _process_command(pid) is None:
        return bool(probe()["running"])
    return state == "alive"


def _log_tail(limit: int = 4000) -> str:
    """Хвост журнала сервера: причина падения запуска видна человеку."""
    try:
        with open(log_path(), "r", encoding="utf-8", errors="replace") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - limit))
            return fh.read()
    except OSError:
        return ""


def _failure_reason() -> Optional[str]:
    """Последняя строка ошибки из журнала (None — журнал о сбое не говорит)."""
    lines = [line.strip() for line in _log_tail().splitlines() if line.strip()]
    for line in reversed(lines):
        if "Error" in line or "error" in line or "Traceback" in line:
            return line[:300]
    return None


def _installed() -> Dict[str, Any]:
    """Что установлено: venv с mlx-lm и веса модели (без обращений к серверу)."""
    python = venv_python()
    weights = _model_bytes()
    packages = os.path.join(home(), "venv", "lib")
    mlx = False
    if os.path.isdir(packages):
        for name in os.listdir(packages):
            if os.path.isdir(os.path.join(packages, name, "site-packages", "mlx_lm")):
                mlx = True
                break
    return {
        "home": home(),
        "python": python,
        "venv": os.path.isfile(python),
        "mlx": mlx,
        "model": weights > 0,
        "model_bytes": weights,
        "model_dirs": _model_dirs(),
    }


def status(with_probe: bool = True) -> dict:
    """Полное состояние локальной модели для интерфейса и проверок.

    Возвращает {"source", "provider", "title", "model", "base_url", "remote",
    "installed": {...}, "server": {"running", "pid", "models", "loaded",
    "starting", "log", "error"}, "ready", "hint"}. Ни одного обращения к модели
    здесь нет: только короткий опрос «жив ли сервер».

    `ready` требует не только ответа, но и объявленной модели: сервер, у которого
    не нашлись веса, отвечает и при этом не может сгенерировать ни токена
    (см. `probe`) — «готов» про него сказано не будет.
    """
    global _started_at
    info = config.source_info()
    installed = _installed()
    server = probe() if with_probe else {"running": False, "models": [],
                                         "loaded": False, "error": None}
    pid = _read_pid()
    state = _pid_state(pid)
    # ЗАПУСК и ОШИБКА — разные вещи: пока процесс жив, идёт загрузка весов
    # (десятки секунд), и называть это «сервер остановился» нельзя — человек
    # видел бы ошибку ровно в тот момент, когда всё работает.
    starting = bool(_started_at) and not server["running"] and state != "dead"
    if server["running"]:
        _started_at = None
        starting = False
    error = None
    if not server["running"] and _started_at and state == "dead":
        # Процесс, которого мы запускали, действительно умер: причину берём из
        # его журнала.
        error = _failure_reason() or "сервер остановился сразу после запуска"
        _started_at = None
        starting = False
    server.update({
        "pid": pid if state != "dead" else 0,
        "starting": starting,
        "log": log_path(),
        "error": error,
        # Запланированная автоостановка: интерфейс по ней показывает «остановится
        # сам через ~N» и опрашивает состояние, пока сервер не погаснет.
        "stop_at": _stop_deadline,
        "stop_in": (max(0, int(_stop_deadline - time.time()))
                    if _stop_deadline else 0),
    })
    ready = (server["running"] and bool(server.get("loaded"))
             and not (server.get("broken") or broken_reason())) \
        or info["source"] == "remote"
    return {
        **info,
        "installed": installed,
        "server": server,
        # ready — можно ли СЕЙЧАС получить ответ от действующего источника:
        # удалённый готов всегда (сеть и ключ — его забота), локальный — когда
        # сервер отвечает И объявил модель (см. `probe`: отвечающий сервер без
        # модели — это сбой загрузки весов, а не готовность).
        "ready": ready,
        "hint": _hint(info, installed, server),
    }


def _left_text(seconds: int) -> str:
    """«2 мин» / «40 с» — для подписи отсрочки автоостановки."""
    if seconds >= 90:
        return f"{int(round(seconds / 60))} мин"
    return f"{max(0, int(seconds))} с"


def _hint(info: dict, installed: dict, server: dict) -> str:
    """Одна фраза для человека: что происходит и что делать."""
    if info["source"] == "remote":
        text = f"Запросы уходят в облако: {info['title']}."
        # Локальный сервер при удалённом источнике не используется: либо он
        # доживает отсрочку и погаснет сам, либо его подняли вручную — тогда об
        # этом надо сказать прямо, иначе «зачем он ест 6 ГБ» остаётся загадкой.
        if server.get("running") and server.get("stop_in"):
            return (text + " Локальный сервер ещё работает и остановится сам через ~"
                    + _left_text(server["stop_in"]) + " (раньше — кнопкой ниже).")
        if server.get("running"):
            return (text + " Локальный сервер работает, хотя выбран удалённый "
                    "источник: он не используется — остановите кнопкой ниже.")
        return text
    if server["running"] and (server.get("broken") or broken_reason()):
        return ("Локальный сервер отвечает, но ГЕНЕРИРОВАТЬ не может: поток "
                "генерации умер (обычно не хватило видеопамяти при параллельных "
                "задачах). Перезапускаю его сам при следующем запросе; вручную — "
                "кнопкой ниже или tools/local_llm.sh restart.")
    if server["running"] and server.get("loaded"):
        loaded = ", ".join(server["models"][:3]) or config.LOCAL_LLM_MODEL
        return f"Локальный сервер отвечает, модель: {loaded}."
    if server["running"]:
        # Отвечает, но модели не объявил: веса не нашлись (см. `probe`). Молчать
        # об этом нельзя — иначе каждый запрос возвращался бы пустым ответом, а
        # человек искал бы причину в документах и в промпте.
        return ("Локальный сервер отвечает, но модель НЕ загружена: веса не "
                "нашлись (причина — в журнале %s). Остановите сервер кнопкой "
                "ниже и запустите снова; если не поможет — "
                "tools/local_llm.sh restart." % log_path())
    if server["starting"]:
        return "Локальный сервер запускается — веса модели читаются с диска."
    if server["error"]:
        return f"Локальный сервер не поднялся: {server['error']}"
    if not installed["venv"] or not installed["mlx"]:
        return ("Окружение локальной модели не установлено: "
                "выполните tools/local_llm.sh install.")
    if not installed["model"]:
        return ("Веса модели не скачаны: выполните tools/local_llm.sh install "
                "(или tools/local_llm.sh pull).")
    return ("Локальный сервер не запущен — нажмите «🧠 Локальная» ещё раз "
            "(сервер поднимется) или выполните tools/local_llm.sh start.")


# ---------------------------------------------------------------------------
# Автоостановка с отсрочкой
#
# Зачем: переключение на удалённую модель освобождает память ТОЛЬКО если сервер
# погасить, но мгновенное гашение наказывает за сравнение ответов (каждое
# возвращение на локальную стоило бы загрузки весов). Отсрочка решает оба
# случая: вернулись раньше — ничего не перезагружается, не вернулись — память
# освободилась сама.
# ---------------------------------------------------------------------------
def stop_scheduled() -> Optional[float]:
    """Когда сервер остановится сам (None — автоостановка не запланирована)."""
    return _stop_deadline


def cancel_scheduled_stop() -> None:
    """Отменяет запланированную автоостановку (сервер снова нужен)."""
    global _stop_timer, _stop_deadline
    with _stop_lock:
        timer, _stop_timer, _stop_deadline = _stop_timer, None, None
    if timer is not None:
        timer.cancel()


def schedule_stop(delay: Optional[float] = None) -> None:
    """Планирует автоостановку сервера через `delay` секунд (по умолчанию — отсрочка).

    Повторный вызов переносит срок, а не заводит второй таймер.
    """
    global _stop_timer, _stop_deadline
    seconds = STOP_GRACE_SECONDS if delay is None else max(0.0, float(delay))
    cancel_scheduled_stop()
    with _stop_lock:
        _stop_deadline = time.time() + seconds
        timer = threading.Timer(seconds, _fire_scheduled_stop)
        # Демон: закрытие приложения не должно ждать таймера (сервер в этом
        # случае остаётся работать — его остановит кнопка или скрипт).
        timer.daemon = True
        _stop_timer = timer
    timer.start()


def _fire_scheduled_stop() -> None:
    """Сработала отсрочка: гасим сервер, ТОЛЬКО если он всё ещё не нужен.

    Проверка «источник всё ещё удалённый» обязательна: за время отсрочки
    человек мог вернуться на локальную модель, и гасить работающий сервер под
    ним нельзя.
    """
    cancel_scheduled_stop()
    if config.llm_source() == "local":
        return
    if not probe()["running"]:
        return
    logger.info("Локальная модель: автоостановка — источник удалённый, "
                "сервер больше не нужен")
    stop()


# ---------------------------------------------------------------------------
# Жизнь сервера
# ---------------------------------------------------------------------------
def _server_env() -> Dict[str, str]:
    """Окружение сервера: веса и кэш HuggingFace — ВНУТРИ проекта.

    HF_HOME уводит каталог моделей в data/local_llm/models (иначе mlx-lm полез
    бы в ~/.cache), а HF_HUB_OFFLINE запрещает серверу ходить в сеть за весами:
    локальная модель обязана работать без интернета — веса уже скачаны.

    HF_HUB_CACHE задаётся ЯВНО и перекрывает унаследованный, и это не мелочь:
    тот же процесс приложения настраивает кэш HuggingFace для моделей RAG
    (`HF_HUB_CACHE` = data/rag/models/hub, см. app/ai/rag_embedding.py), а
    HF_HUB_CACHE СИЛЬНЕЕ HF_HOME. Наследуй сервер этот каталог — он искал бы
    веса ЛОКАЛЬНОЙ МОДЕЛИ в кэше моделей RAG, не нашёл бы их (дозагрузку
    запрещает HF_HUB_OFFLINE=1) и остался бы жить процессом, который отвечает
    на /v1/models пустым списком и не выдаёт ни одного токена: поиск по базе
    работает, а ответа нет — «модель промолчала» вместо причины.
    """
    env = dict(os.environ)
    env["HF_HOME"] = models_dir()
    env["HF_HUB_CACHE"] = os.path.join(models_dir(), "hub")
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PATH"] = os.path.join(home(), "bin") + os.pathsep + env.get("PATH", "")
    return env


def start(wait: float = 0.0) -> dict:
    """Запускает локальный сервер ОТДЕЛЬНЫМ процессом и возвращает состояние.

    Ждать загрузки весов здесь не нужно и нельзя: это отдельный процесс, он
    живёт своей жизнью, а интерфейс следит за готовностью через `status`
    (поле server.starting). `wait` — сколько секунд подождать готовности перед
    возвратом (нужно скрипту tools/local_llm.sh start, интерфейсу — не нужно).
    """
    global _started_at
    # Сервер поднимают осознанно — запланированная автоостановка больше не нужна.
    cancel_scheduled_stop()
    if probe()["running"]:
        return status()
    # Поднимаем НОВЫЙ процесс — прежняя отметка «генерация сломана» к нему не
    # относится (её ставил клиент по ответу умершего сервера).
    clear_broken()
    installed = _installed()
    if not installed["venv"] or not installed["mlx"]:
        raise LocalLlmError(
            "окружение локальной модели не установлено — выполните "
            "tools/local_llm.sh install"
        )
    if not installed["model"]:
        raise LocalLlmError(
            f"веса модели {config.LOCAL_LLM_MODEL} не скачаны — выполните "
            "tools/local_llm.sh install"
        )
    os.makedirs(os.path.dirname(log_path()), exist_ok=True)
    command = [
        venv_python(), "-m", "mlx_lm.server",
        "--model", str(config.LOCAL_LLM_MODEL),
        "--host", str(config.LOCAL_LLM_HOST),
        "--port", str(int(config.LOCAL_LLM_PORT)),
        "--chat-template-args", CHAT_TEMPLATE_ARGS,
        # ОГРАНИЧЕНИЕ ПАМЯТИ КЭША ПРОМПТОВ. mlx_lm.server держит KV-кэши
        # промптов, и на агентских запросах (у каждого свой длинный промпт:
        # инструменты, правила проекта, фрагменты баз) кэш рос до 3,4 ГБ — при
        # шести гигабайтах весов на шестнадцати гигабайтах памяти мака это
        # кончалось срывом генерации: `Insufficient Memory`, «generation thread
        # died» и рвущееся соединение (живой случай 10.10: пользователь видел
        # «вызов не удался»). Предел задаётся в мегабайтах настройкой
        # LOCAL_LLM_PROMPT_CACHE_MB (0 — не ограничивать).
        "--prompt-cache-bytes", str(prompt_cache_bytes()),
        # Сколько РАЗНЫХ кэшей держать: агент ходит с несколькими повторяющимися
        # промптами (диспетчер, планировщик, ответ), и десяток вариантов ничего не
        # ускоряет, а память занимает.
        "--prompt-cache-size", str(PROMPT_CACHE_SEQUENCES),
    ]
    if KV_BITS:
        # Квантование KV-кэша (например, 8 бит) заметно уменьшает память на
        # длинных контекстах. Выключено по умолчанию: это влияет на качество
        # ответов, и включать его должен человек, а не приложение за него.
        command += ["--kv-bits", str(KV_BITS)]
    # start_new_session — сервер не должен умереть вместе с приложением и не
    # должен получать Ctrl+C, предназначенный ему: остановка только явная.
    with open(log_path(), "a", encoding="utf-8") as log:
        log.write(f"\n=== запуск {time.strftime('%Y-%m-%d %H:%M:%S')} "
                  f"{' '.join(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command, cwd=home(), env=_server_env(),
            stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
    _started_at = time.time()
    try:
        with open(pid_path(), "w", encoding="utf-8") as fh:
            json.dump({"pid": process.pid, "at": time.strftime("%Y-%m-%d %H:%M:%S")}, fh)
    except OSError as exc:  # noqa: BLE001 — без pid-файла сервер всё равно работает
        logger.warning("Локальная модель: не удалось записать %s: %s", pid_path(), exc)
    logger.info("Локальная модель: сервер запущен (pid %d)", process.pid)
    if wait > 0:
        wait_ready(wait)
    return status()


def wait_ready(timeout: float = START_TIMEOUT, step: float = 1.0) -> bool:
    """Ждёт, пока сервер начнёт отвечать И объявит модель. True — дождались.

    Ждём, ПОКА ПРОЦЕСС ЖИВ: если он умер, ждать нечего; а «спросить систему о
    процессе нельзя» — не повод бросить ожидание (см. `_pid_state`).

    Ответа мало: пока веса читаются, сервер молчит, а сервер, которому веса не
    нашлись, отвечает пустым списком моделей и генерировать не может — готовым
    считается только тот, что объявил модель (см. `probe`).
    """
    deadline = time.time() + max(0.0, timeout)
    while time.time() < deadline:
        info = probe()
        if info["running"] and info["loaded"]:
            return True
        pid = _read_pid()
        if pid and _pid_state(pid) == "dead":
            return False  # процесс умер — ждать нечего
        time.sleep(step)
    return probe()["running"]


def stop() -> dict:
    """Останавливает сервер: SIGTERM, при упорстве — SIGKILL.

    Чужой процесс не трогаем: перед сигналом проверяем, что в командной строке
    именно наш mlx_lm.server (номер в файле мог остаться от прошлой загрузки).
    """
    global _started_at
    cancel_scheduled_stop()   # остановка состоялась — ждать больше нечего
    clear_broken()            # отметка «генерация сломана» живёт до перезапуска
    pid = _read_pid()
    if pid and _own_process(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError as exc:  # noqa: BLE001 — процесс мог умереть сам
            logger.warning("Локальная модель: SIGTERM не ушёл (pid %d): %s", pid, exc)
        deadline = time.time() + STOP_TIMEOUT
        while time.time() < deadline and _pid_alive(pid):
            time.sleep(0.3)
        if _pid_alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:  # noqa: PERF203 — уже умер
                pass
    try:
        os.remove(pid_path())
    except OSError:
        pass
    _started_at = None
    return status()


def prepare(autostart: bool = True, wait: float = START_TIMEOUT) -> dict:
    """Доводит действующий источник до готовности и возвращает состояние.

    Нужен ПЕРЕД работой задачи: если выбрана локальная модель, а сервер ещё не
    отвечает, он поднимается, и ожидание загрузки весов (десятки секунд)
    происходит здесь — в отдельном потоке, а не в цикле событий приложения.
    Не дождались — понятная причина исключением (`LocalLlmError`), а не пустой
    ответ модели, который в обычном режиме подменился бы демо-ответом.

    Удалённый источник ожидания не требует: сеть и ключ — его забота (клиент
    сам повторяет вызовы, см. app/ai/client.py).
    """
    state = status()
    if state["source"] == "remote" or _serving(state):
        return state
    if state["server"]["running"] and (state["server"].get("broken")
                                       or broken_reason()):
        # СЕРВЕР ОТВЕЧАЕТ, НО ГЕНЕРИРОВАТЬ НЕ МОЖЕТ (см. mark_broken). Поднимать
        # нечего — порт занят своим же процессом, — а нужен ПЕРЕЗАПУСК: он и есть
        # лечение. Делаем его сами, но не чаще, чем раз в BROKEN_RESTART_COOLDOWN
        # секунд: иначе повторные запросы превратились бы в цикл «упал —
        # перезапустили», и каждый стоил бы тридцати секунд загрузки весов.
        global _BROKEN_RESTART_AT
        waited = time.time() - _BROKEN_RESTART_AT
        if waited >= BROKEN_RESTART_COOLDOWN:
            _BROKEN_RESTART_AT = time.time()
            logger.warning("Локальная модель: перезапускаю сервер после сбоя генерации")
            try:
                stop()
                state = start()
            except LocalLlmError as exc:  # noqa: PERF203 — причину назовём ниже
                logger.warning("Локальная модель: перезапуск не удался: %s", exc)
            if wait > 0:
                wait_ready(wait)
            state = status()
            if _serving(state):
                return state
        raise LocalLlmError(state["hint"])
    if state["server"]["running"]:
        # Сервер ОТВЕЧАЕТ, но модель не объявлена: веса не загрузились (см.
        # `probe`). Это не «запускается» — поднимать нечего, порт занят своим же
        # процессом, и ждать бессмысленно. Причина называется сразу: пустой ответ
        # модели выглядел бы как «модель промолчала».
        raise LocalLlmError(state["hint"])
    if autostart and not state["server"]["starting"]:
        state = start()
        if _serving(state):
            return state
    if _serving(state):
        return state
    if wait > 0 and wait_ready(wait):
        return status()
    state = status()
    if _serving(state):
        return state
    raise LocalLlmError(state["hint"])


def _serving(state: dict) -> bool:
    """Может ли сервер СГЕНЕРИРОВАТЬ ответ: отвечает И объявил модель.

    «Сервер отвечает» — ещё не «сервер готов»: у сервера, которому не нашлись
    веса, HTTP-часть живёт, а поток генерации умирает, и /v1/models отдаёт пустой
    список (см. `probe`). Такой источник — НЕГОТОВЫЙ, и вызывающий код обязан
    назвать причину, а не пропустить вызов в пустоту.
    """
    server = state.get("server") or {}
    if server.get("broken") or broken_reason():
        # Отвечает и модель объявлена, но генерация сломана (см. mark_broken):
        # пропустить в него вызов — значит показать «модель промолчала».
        return False
    return bool(server.get("running")) and bool(server.get("loaded", True))


# ---------------------------------------------------------------------------
# Выбор источника ответа
# ---------------------------------------------------------------------------
def apply_saved_source() -> str:
    """Применяет сохранённый выбор источника при запуске приложения.

    Вызывается при старте (main.py). Битый или отсутствующий файл — не ошибка:
    остаётся источник по умолчанию (config.LLM_SOURCE_DEFAULT, обычно remote).

    Заодно закрывает след прошлого запуска: сервер мог остаться работать, а
    источник — быть удалённым. Тогда он не нужен никому, и планируется та же
    автоостановка, что при переключении.
    """
    source = _read_saved_source()
    if source is not None:
        try:
            config.set_llm_source(source)
        except ValueError:
            logger.warning("Локальная модель: в %s неизвестный источник %r",
                           source_file(), source)
    source = config.llm_source()
    if source != "local" and probe()["running"]:
        schedule_stop()
    return source


def _read_saved_source() -> Optional[str]:
    """Источник из файла выбора (None — файла нет, он пуст или испорчен)."""
    try:
        with open(source_file(), "r", encoding="utf-8") as fh:
            return json.load(fh).get("source") or None
    except (OSError, ValueError, AttributeError):
        return None


def save_source(name: str) -> str:
    """Меняет источник ответа и сохраняет выбор на диск (атомарно).

    Возвращает имя действующего источника. Сначала пишем файл, потом меняем
    состояние процесса: если запись не удалась, приложение не должно оказаться
    на источнике, который не переживёт перезапуск.
    """
    key = config.set_llm_source(name)  # проверка имени — до любой записи
    directory = os.path.dirname(source_file())
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = source_file() + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"source": key, "at": time.strftime("%Y-%m-%d %H:%M:%S")}, fh,
                  ensure_ascii=False, indent=2)
    os.replace(tmp, source_file())
    return key


def switch(name: str, autostart: bool = True) -> dict:
    """Переключает источник ответа: сохраняет выбор, поднимает или гасит сервер.

    Возвращает состояние (`status`). Запуск сервера НЕ ждёт загрузки весов:
    интерфейс видит поле server.starting и опрашивает готовность сам. Если
    поднять сервер не удалось (нет окружения или весов), причина возвращается
    в поле `error` — переключение состоялось, а отвечать пока нечем.

    ПЕРЕХОД НА УДАЛЁННУЮ ГАСИТ ЛОКАЛЬНЫЙ СЕРВЕР — с отсрочкой (§ «Автоостановка»):
    иначе переключатель выглядел бы как выключатель, а шесть гигабайт памяти
    оставались бы занятыми непонятно чем. Отсрочка нужна для сравнения ответов:
    вернулись на локальную раньше срока — таймер отменяется и веса не читаются
    заново.
    """
    save_source(name)
    state = status()
    if config.llm_source() != "local":
        if state["server"]["running"]:
            schedule_stop()
        return status()
    cancel_scheduled_stop()   # вернулись на локальную — гасить нечего
    state = status()
    if state["server"]["running"] or state["server"]["starting"]:
        return state
    if not autostart:
        return state
    try:
        return start()
    except LocalLlmError as exc:
        state["error"] = str(exc)
        return state
