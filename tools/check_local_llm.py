"""Самопроверка ЛОКАЛЬНОЙ модели (провайдер "local", app/ai/local_llm.py).

Запуск (сеть, API-ключ и настоящий сервер модели НЕ нужны):

    ./venv/bin/python tools/check_local_llm.py

Проверяется ровно то, что важно для переключателя «локальная / удалённая
модель»:

  [1] конфигурация и тарифы: провайдер local, активная модель по источнику,
      выбор провайдера по модели, стоимость равна нулю (у локальной модели
      счёта за токены нет) и НЕ равна нулю у удалённой;
  [2] HTTP-слой: запрос к локальному серверу уходит с его моделью и БЕЗ поля
      thinking (локальный сервер его не знает), метрики и цена нулевые;
  [3] состояние: что установлено (venv, веса) и понятная подсказка, когда
      отвечать нечем;
  [4] выбор источника: файл настроек, «перезапуск приложения», неизвестное имя
      и битый файл — без падений и без «тихого» перехода на чужой источник;
  [5] защита маршрутов: при выбранной локальной модели и недоступном сервере
      задача и обычный запрос получают ПОНЯТНУЮ ПРИЧИНУ, а не пустой ответ
      (в обычном режиме он подменился бы демо-ответом);
  [6] маршруты источника ответа и безопасность остановки сервера (чужой
      процесс по записи в pid-файле не убивается);
  [7] автоостановка сервера при переходе на удалённый источник (с отсрочкой);
  [9] окружение сервера модели: кэш моделей RAG не уводит веса локальной
      модели (HF_HUB_CACHE сильнее HF_HOME, и унаследованный каталог ломал
      загрузку весов: сервер отвечал, но генерировать не мог);
  [8] RAG при локальном источнике: поиск по базам локальный, а генерация идёт
      через локальную модель — маршруты RAG (мини-чат, `/test_rag`, контрольный
      диалог) ждут готовности сервера и называют причину, если он не поднялся,
      а пометка «задача выполняется» после хода снимается.

Настоящий сервер модели не поднимается: вместо него — локальная заглушка на
http.server или заведомо закрытый адрес. Рабочие данные не трогаются: каталог
локальной модели, workspace и профили пишутся во временный каталог.
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Изоляция данных: ДО импорта config (он читает окружение при импорте) ----
_TMP = tempfile.mkdtemp(prefix="local-llm-check-")
# Каталог «всё установлено» (фальшивые venv и веса) и каталог «ничего нет».
HOME_INSTALLED = os.path.join(_TMP, "installed")
HOME_EMPTY = os.path.join(_TMP, "empty")
os.environ["LOCAL_LLM_HOME"] = HOME_EMPTY
os.environ["LLM_SOURCE_FILE"] = os.path.join(_TMP, "source.json")
os.environ["AGENT_WORKSPACE_FILE"] = os.path.join(_TMP, "workspace.json")
os.environ["AGENT_MEMORY_FILE"] = os.path.join(_TMP, "agent_memory.json")
os.environ["AGENT_PROFILES_FILE"] = os.path.join(_TMP, "profiles.json")
os.environ["RAG_DIR"] = os.path.join(_TMP, "rag")
# Заведомо закрытый адрес вместо настоящего сервера модели: порт 1 не слушает
# никто, и «сервер не запущен» проверяется без ожидания таймаута.
CLOSED_URL = "http://127.0.0.1:1/v1"
os.environ["LOCAL_LLM_BASE_URL"] = CLOSED_URL
# Поиск RAG в проверке детерминирован и без сети: эмбеддинги считает встроенный
# офлайн-бэкенд, второй этап — признаки (в рабочем проекте движок выбирает
# окружение, см. app/ai/rag_embedding.py и rag_rerank.py).
os.environ["RAG_EMBED_BACKEND"] = "hashing"
os.environ["RAG_RERANK_BACKEND"] = "features"

from app import config  # noqa: E402
from app.ai import client, local_llm, rag, rag_embedding  # noqa: E402
from app.routers import chat  # noqa: E402
from app.schemas import (ChatMessage, LlmServerAction, LlmSourceUpdate,  # noqa: E402
                         RagApply, RagDialogTest, TaskCreate)
from starlette.requests import Request  # noqa: E402

FAILURES = []
# Зовы модели: ни один маршрут не должен дойти до модели, пока локальный
# источник не готов (иначе «сбой» выглядел бы как пустой ответ модели).
CALLS = {"count": 0}
# Ответ заглушки «локального сервера» (проверяется, что до него дело дошло).
ANSWER = "Ответ локального сервера."


def check(name, condition, detail=""):
    """Одна проверка: печатает результат и копит провалы."""
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# Заглушка локального сервера модели (OpenAI-совместимый минимум)
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            if _STUB.get("empty_models"):
                # Сервер ОТВЕЧАЕТ, но модель не объявил — так выглядит сервер, у
                # которого не нашлись веса: поток генерации умирает, а HTTP-часть
                # продолжает отвечать пустым списком моделей (см. probe).
                self._json({"object": "list", "data": []})
                return
            self._json({"object": "list", "data": [
                {"id": config.LOCAL_LLM_MODEL, "object": "model"}]})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            payload = json.loads(body)
        except ValueError:
            payload = {}
        CALLS.setdefault("payloads", []).append(payload)
        CALLS["count"] += 1
        self._json({
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": ANSWER}}],
            "usage": {"prompt_tokens": 21, "completion_tokens": 5, "total_tokens": 26},
        })

    def _json(self, data, status=200):
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def start_stub():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# ---------------------------------------------------------------------------
# Фальшивая установка: venv с mlx-lm и веса модели
# ---------------------------------------------------------------------------
def make_installed_home():
    """Создаёт каталог, который выглядит как настоящая установка.

    Файлы пустые: проверяются ПРАВИЛА (найдены ли venv, веса, сколько они
    занимают), а не сами веса — качать 4,6 ГБ ради проверки нельзя.
    """
    python = os.path.join(HOME_INSTALLED, "venv", "bin", "python")
    os.makedirs(os.path.dirname(python), exist_ok=True)
    with open(python, "w", encoding="utf-8") as fh:
        fh.write("#!/bin/sh\nexit 0\n")
    os.chmod(python, 0o755)
    os.makedirs(os.path.join(HOME_INSTALLED, "venv", "lib", "python3.12",
                             "site-packages", "mlx_lm"), exist_ok=True)
    model_dir = os.path.join(
        HOME_INSTALLED, "models", "hub",
        "models--" + config.LOCAL_LLM_MODEL.replace("/", "--"),
        "snapshots", "deadbeef",
    )
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "model.safetensors"), "wb") as fh:
        fh.write(b"0" * 4096)
    return model_dir


# ---------------------------------------------------------------------------
# 1. Провайдер и тарифы
# ---------------------------------------------------------------------------
def test_provider_and_pricing():
    print("\n[1] Провайдер local, активный источник и тарифы (без сети)")
    config.set_llm_source("remote")
    spec = config.provider_spec("local")
    check("провайдер local: адрес, ключ и модель — из настроек локальной модели",
          spec["provider"] == "local"
          and spec["base_url"] == config.LOCAL_LLM_BASE_URL
          and spec["model"] == config.LOCAL_LLM_MODEL
          and bool(spec["api_key"]),
          str(spec))
    check("локальному провайдеру поле thinking не отправляется (thinking=ignore)",
          spec["thinking"] == "ignore", spec["thinking"])
    check("при удалённом источнике активная модель — модель провайдера по умолчанию",
          config.active_provider() == config.DEFAULT_PROVIDER
          and config.active_model() == config.LLM_MODEL,
          f"{config.active_provider()} / {config.active_model()}")
    config.set_llm_source("local")
    check("при локальном источнике активная модель — локальная",
          config.active_provider() == "local"
          and config.active_model() == config.LOCAL_LLM_MODEL,
          f"{config.active_provider()} / {config.active_model()}")
    check("provider_spec() без имени отдаёт действующий источник",
          config.provider_spec()["provider"] == "local",
          str(config.provider_spec()))
    check("провайдер по модели: локальная модель → local",
          config.provider_for_model(config.LOCAL_LLM_MODEL) == "local")
    check("провайдер по модели: URI «gpt://…» остаётся у Yandex",
          config.provider_for_model("gpt://folder/alice-llm/latest") == "yandex")
    check("провайдер по модели: удалённая модель по умолчанию — не локальный сервер",
          config.provider_for_model(config.LLM_MODEL) == config.DEFAULT_PROVIDER,
          config.provider_for_model(config.LLM_MODEL))
    check("провайдер по модели: пустая модель → действующий источник",
          config.provider_for_model(None) == "local"
          and config.provider_for_model("") == "local")

    local_cost = config.usage_cost(config.LOCAL_LLM_MODEL, 5000, 2000)
    check("стоимость локального вызова — ноль (счёта за токены нет)",
          local_cost == 0.0, str(local_cost))
    check("стоимость удалённого вызова — не ноль (сравнение осмысленно)",
          config.usage_cost(config.LLM_MODEL, 5000, 2000) > 0.0)
    check("ставки локальной модели — нули",
          config.model_price(config.LOCAL_LLM_MODEL) == {"input": 0.0, "output": 0.0},
          str(config.model_price(config.LOCAL_LLM_MODEL)))
    info = config.pricing_info(config.LOCAL_LLM_MODEL)
    check("тариф локальной модели назван словами, а не нулями без объяснения",
          info["provider"] == "local" and "локальная модель" in info["tariff"]
          and info["peak_note"], str(info))
    check("pricing_info() при локальном источнике не показывает удалённый тариф",
          config.pricing_info()["provider"] == "local",
          str(config.pricing_info()))
    config.set_llm_source("remote")
    check("pricing_info() при удалённом источнике снова удалённый",
          config.pricing_info()["provider"] != "local",
          str(config.pricing_info()))


# ---------------------------------------------------------------------------
# 2. HTTP-слой: куда и с чем уходит запрос
# ---------------------------------------------------------------------------
def test_client_payload(server):
    print("\n[2] Клиент: локальная модель, без поля thinking, цена нулевая")
    port = server.server_port
    config.LOCAL_LLM_BASE_URL = f"http://127.0.0.1:{port}/v1"
    CALLS["count"] = 0
    CALLS["payloads"] = []
    config.set_llm_source("local")
    content, metrics = client.call_llm_with_metrics(
        "Привет", max_tokens=100, temperature=0.5)
    payload = (CALLS.get("payloads") or [{}])[-1]
    check("запрос ушёл на локальный сервер с локальной моделью",
          payload.get("model") == config.LOCAL_LLM_MODEL, str(payload.get("model")))
    check("поля thinking в запросе НЕТ (локальный сервер его не знает)",
          "thinking" not in payload, str(sorted(payload.keys())))
    check("ответ и метрики получены",
          content == ANSWER and metrics and metrics["prompt_tokens"] == 21,
          f"{content!r} / {metrics}")
    check("в метриках — локальная модель и нулевая стоимость",
          metrics["model"] == config.LOCAL_LLM_MODEL and metrics["cost_rub"] == 0.0,
          str(metrics))

    # Для сравнения: у удалённого провайдера поле thinking остаётся как было.
    CALLS["payloads"] = []
    config.set_llm_source("remote")
    config.LLM_BASE_URL = f"http://127.0.0.1:{port}/v1"
    client.call_llm_with_metrics("Привет", max_tokens=100)
    remote_payload = (CALLS.get("payloads") or [{}])[-1]
    check("у удалённого провайдера thinking по-прежнему отправляется",
          remote_payload.get("thinking") == {"type": "disabled"},
          str(remote_payload.get("thinking")))
    check("удалённый вызов считается по тарифу (не ноль)",
          client.call_llm_with_metrics("Привет")[1]["cost_rub"] > 0.0)
    config.LOCAL_LLM_BASE_URL = CLOSED_URL


# ---------------------------------------------------------------------------
# 3. Состояние и подсказки
# ---------------------------------------------------------------------------
def test_status(server):
    print("\n[3] Состояние: что установлено и что делать, если отвечать нечем")
    config.set_llm_source("remote")
    config.LOCAL_LLM_HOME = HOME_EMPTY
    state = local_llm.status()
    check("пустой каталог: ни venv, ни весов",
          not state["installed"]["venv"] and not state["installed"]["model"], 
          str(state["installed"]))
    check("удалённый источник готов всегда, подсказка называет облако",
          state["ready"] and "облако" in state["hint"], state["hint"])
    config.set_llm_source("local")
    state = local_llm.status()
    check("локальный источник без установки НЕ готов",
          not state["ready"], str(state["ready"]))
    check("подсказка называет команду установки",
          "local_llm.sh install" in state["hint"], state["hint"])

    config.LOCAL_LLM_HOME = HOME_INSTALLED
    state = local_llm.status()
    check("установка найдена: venv с mlx-lm и веса",
          state["installed"]["venv"] and state["installed"]["mlx"]
          and state["installed"]["model"], str(state["installed"]))
    check("размер весов посчитан", state["installed"]["model_bytes"] == 4096,
          str(state["installed"]["model_bytes"]))
    check("сервер не отвечает → источник не готов и подсказка говорит о запуске",
          not state["server"]["running"] and not state["ready"]
          and ("не запущен" in state["hint"] or "localhost" in state["hint"]),
          state["hint"])
    config.LOCAL_LLM_BASE_URL = f"http://127.0.0.1:{server.server_port}/v1"
    state = local_llm.status()
    check("отвечающий сервер: running и ready, модель показана",
          state["server"]["running"] and state["ready"]
          and config.LOCAL_LLM_MODEL in state["server"]["models"], str(state["server"]))
    check("подсказка говорит, что сервер отвечает", "отвечает" in state["hint"],
          state["hint"])

    # ОТВЕЧАЕТ — НЕ ЗНАЧИТ ГОТОВ. Сервер, которому не нашлись веса, живёт и
    # отвечает на /v1/models пустым списком, но не выдаёт ни одного токена:
    # живой прогон 07.10 показал, как такой сервер выглядел «готовым», а ответы
    # возвращались пустыми («модель промолчала») — вместо причины.
    _STUB["empty_models"] = True
    try:
        state = local_llm.status()
        check("сервер отвечает без модели: источник НЕ готов",
              state["server"]["running"] and not state["server"]["loaded"]
              and not state["ready"], str(state["server"]))
        check("подсказка называет причину: модель не загружена",
              "НЕ загружена" in state["hint"] and "restart" in state["hint"],
              state["hint"])
        config.set_llm_source("local")
        try:
            local_llm.prepare()
            prepared_error = None
        except local_llm.LocalLlmError as exc:
            prepared_error = str(exc)
        check("prepare() отказывает с причиной, а не пропускает вызов в пустоту",
              bool(prepared_error) and "НЕ загружена" in prepared_error,
              str(prepared_error))
        config.set_llm_source("remote")
    finally:
        _STUB["empty_models"] = False
    state = local_llm.status()
    check("объявивший модель сервер снова готов",
          state["ready"] and state["server"]["loaded"], str(state["server"]))
    config.LOCAL_LLM_BASE_URL = CLOSED_URL
    check("probe() не бросает исключений на закрытом адресе",
          local_llm.probe()["running"] is False)

    # «Процесс жив, а сервер ещё молчит» — это ЗАПУСК, а не «сервер остановился»:
    # живой прогон 05.10 показал ложную ошибку ровно в этот момент (веса читаются
    # десятки секунд, и путать загрузку со сбоем нельзя).
    child = subprocess.Popen(["sleep", "30"])
    try:
        with open(local_llm.pid_path(), "w", encoding="utf-8") as fh:
            json.dump({"pid": child.pid}, fh)
        local_llm._started_at = time.time()
        state = local_llm.status()
        check("живой процесс и молчащий сервер — это запуск, а не ошибка",
              state["server"]["starting"] and not state["server"]["error"],
              str(state["server"]))
        child.terminate()
        child.wait(timeout=5)
        state = local_llm.status()
        check("умерший процесс из pid-файла — ошибка с причиной",
              not state["server"]["starting"] and bool(state["server"]["error"]),
              str(state["server"]))
    finally:
        local_llm._started_at = None
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=5)


# ---------------------------------------------------------------------------
# 4. Выбор источника
# ---------------------------------------------------------------------------
def test_source_switch():
    print("\n[4] Выбор источника: файл, «перезапуск», ошибки ввода")
    config.LOCAL_LLM_HOME = HOME_EMPTY
    saved = local_llm.save_source("remote")
    check("выбор «удалённая» сохранён", saved == "remote"
          and config.llm_source() == "remote")
    check("файл выбора записан", os.path.isfile(local_llm.source_file()))
    local_llm.save_source("local")
    check("выбор «локальная» сохранён", config.llm_source() == "local")
    # «Перезапуск приложения»: состояние процесса сбрасываем и читаем файл заново.
    config.set_llm_source("remote")
    applied = local_llm.apply_saved_source()
    check("после «перезапуска» выбор восстанавливается из файла",
          applied == "local" and config.llm_source() == "local", applied)
    before = open(local_llm.source_file(), encoding="utf-8").read()
    try:
        local_llm.save_source("openai")
        raised = False
    except ValueError:
        raised = True
    check("неизвестное имя источника — ошибка, а не тихий переход",
          raised and config.llm_source() == "local")
    check("при ошибке файл не переписан",
          open(local_llm.source_file(), encoding="utf-8").read() == before)
    with open(local_llm.source_file(), "w", encoding="utf-8") as fh:
        fh.write("{ это не JSON")
    check("битый файл выбора не роняет приложение",
          local_llm.apply_saved_source() == config.llm_source())
    local_llm.save_source("remote")

    # Переключение на локальную модель без установки: переключение СОСТОЯЛОСЬ,
    # а причина «отвечать нечем» вернулась текстом (интерфейс её покажет).
    state = local_llm.switch("local", autostart=True)
    check("переключение на локальную без установки: выбор сделан",
          state["source"] == "local", str(state["source"]))
    check("причина названа (команда установки), а не проглочена",
          bool(state.get("error")) and "local_llm.sh install" in state["error"],
          str(state.get("error")))
    local_llm.save_source("remote")


# ---------------------------------------------------------------------------
# 5. Защита маршрутов
# ---------------------------------------------------------------------------
async def run_chat(text, **kwargs):
    """Прогон POST /api/agent/chat без сети: собирает события NDJSON-потока."""
    response = await chat.agent_chat(ChatMessage(content=text, **kwargs))
    events = []
    async for chunk in response.body_iterator:
        for line in str(chunk).splitlines():
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def install_llm_stub():
    """Подменяет вызовы модели заглушкой: считаем, дошло ли до модели дело."""
    async def fake_async(*args, **kwargs):
        CALLS["count"] += 1
        return "Ответ модели.", {"model": "stub", "elapsed_seconds": 0.01,
                                 "prompt_tokens": 10, "completion_tokens": 5,
                                 "total_tokens": 15, "cost_rub": 0.0}

    client.call_llm_async = fake_async


async def test_routes_guard():
    print("\n[5] Защита: локальная модель не готова — причина, а не пустой ответ")
    install_llm_stub()
    config.LOCAL_LLM_HOME = HOME_EMPTY
    await chat.task_create(TaskCreate(name="Проверка источника"))
    LLM_HOME_SAVED = config.LOCAL_LLM_HOME

    config.set_llm_source("remote")
    CALLS["count"] = 0
    events = await run_chat("Скажи привет")
    check("удалённый источник: запрос доходит до модели и есть ответ",
          CALLS["count"] > 0 and any(e.get("type") == "bot" for e in events),
          f"вызовов: {CALLS['count']}")

    config.set_llm_source("local")
    CALLS["count"] = 0
    events = await run_chat("Скажи привет")
    errors = [e.get("text", "") for e in events if e.get("type") == "error"]
    check("локальный источник без сервера: модель НЕ звалась",
          CALLS["count"] == 0, f"вызовов: {CALLS['count']}")
    check("поток объясняет причину и называет команду установки",
          bool(errors) and "Локальная модель не готова" in errors[0]
          and "local_llm.sh install" in errors[0], str(errors)[:200])
    check("поток корректно закрыт событием done",
          bool(events) and events[-1].get("type") == "done",
          str(events[-1])[:120])

    CALLS["count"] = 0
    result = await chat.chat(ChatMessage(content="Скажи привет"))
    check("обычный режим: вместо демо-ответа — понятная причина",
          CALLS["count"] == 0 and str(result.get("bot", "")).startswith("⚠"),
          str(result.get("bot"))[:160])

    # Готовый локальный источник: маршрут больше не мешает, запрос уходит.
    config.LOCAL_LLM_HOME = HOME_INSTALLED
    config.LOCAL_LLM_BASE_URL = f"http://127.0.0.1:{_STUB['server'].server_port}/v1"
    CALLS["count"] = 0
    result = await chat.chat(ChatMessage(content="Скажи привет"))
    check("готовый локальный сервер: запрос уходит в модель",
          CALLS["count"] > 0 and "Ответ" in str(result.get("bot")), str(result)[:160])
    config.LOCAL_LLM_BASE_URL = CLOSED_URL
    config.LOCAL_LLM_HOME = LLM_HOME_SAVED
    config.set_llm_source("remote")


# ---------------------------------------------------------------------------
# 6. Маршруты источника ответа и безопасность остановки
# ---------------------------------------------------------------------------
async def test_source_routes():
    print("\n[6] Маршруты источника ответа и остановка сервера")
    state = await chat.llm_get()
    check("GET /api/agent/llm отдаёт состояние источника",
          state.get("source") in ("remote", "local") and "installed" in state
          and "server" in state and bool(state.get("hint")), str(state)[:160])
    state = await chat.llm_source_set(
        LlmSourceUpdate(source="local", autostart=False))
    check("POST /api/agent/llm/source переключает на локальную",
          state["source"] == "local" and config.llm_source() == "local")
    check("выбор записан в файл (переживёт перезапуск)",
          json.load(open(local_llm.source_file(), encoding="utf-8"))["source"] == "local")
    state = await chat.llm_source_set(
        LlmSourceUpdate(source="remote", autostart=False))
    check("POST /api/agent/llm/source возвращает удалённую",
          state["source"] == "remote" and config.llm_source() == "remote")

    # Остановка сервера, которого нет: без падений и без выдуманного «остановлен».
    state = await chat.llm_server_action(LlmServerAction(action="stop"))
    check("остановка при незапущенном сервере не падает",
          state["server"]["running"] is False)
    try:
        await chat.llm_server_action(LlmServerAction(action="start"))
        started_error = None
    except Exception as exc:  # noqa: BLE001 — ждём HTTPException 409
        started_error = exc
    check("запуск без установки — 409 с причиной, а не падение",
          started_error is not None
          and getattr(started_error, "status_code", None) == 409
          and "local_llm.sh install" in str(getattr(started_error, "detail", "")),
          str(started_error))

    # ЧУЖОЙ процесс по записи в pid-файле не убивается: под нашим присмотром
    # живёт собственный «сон», его командная строка — не mlx_lm.server.
    child = subprocess.Popen(["sleep", "30"])
    try:
        with open(local_llm.pid_path(), "w", encoding="utf-8") as fh:
            json.dump({"pid": child.pid}, fh)
        local_llm.stop()
        time.sleep(0.3)
        check("чужой процесс из pid-файла не остановлен",
              child.poll() is None, str(child.poll()))
    finally:
        child.terminate()
        child.wait(timeout=5)


def test_auto_stop(server):
    """Автоостановка: переход на удалённую гасит сервер — но с отсрочкой.

    Проверяется РЕШЕНИЕ, а не убийство процесса: `stop` подменяется счётчиком,
    поэтому ни один настоящий сервер в проверке не запускается и не гасится.
    """
    print("\n[7] Автоостановка: переход на удалённую гасит сервер с отсрочкой")
    config.LOCAL_LLM_HOME = HOME_INSTALLED
    config.LOCAL_LLM_BASE_URL = f"http://127.0.0.1:{server.server_port}/v1"
    stopped = {"calls": 0}
    real_stop = local_llm.stop

    def fake_stop():
        stopped["calls"] += 1
        local_llm.cancel_scheduled_stop()
        return {"source": config.llm_source(), "server": {"running": False}}

    local_llm.stop = fake_stop
    try:
        local_llm.save_source("local")
        state = local_llm.switch("remote", autostart=False)
        check("переход на удалённую планирует автоостановку",
              bool(state["server"]["stop_at"]) and state["server"]["stop_in"] > 0,
              str(state["server"]))
        check("подсказка говорит, что сервер остановится сам",
              "остановится сам" in state["hint"], state["hint"])
        check("до срока сервер ещё работает (отсрочка, а не мгновенное гашение)",
              state["server"]["running"] and stopped["calls"] == 0, str(stopped))

        local_llm._fire_scheduled_stop()
        check("отсрочка сработала — сервер остановлен", stopped["calls"] == 1,
              str(stopped))
        check("после автоостановки срок снят", not local_llm.stop_scheduled())

        # Вернулись на локальную до срока: таймер отменён, и сработавшая позже
        # отсрочка НЕ гасит сервер под работающим источником.
        stopped["calls"] = 0
        local_llm.switch("local", autostart=False)
        check("возврат на локальную отменяет отсрочку",
              not local_llm.stop_scheduled() and stopped["calls"] == 0, str(stopped))
        local_llm.schedule_stop(0.05)
        local_llm._fire_scheduled_stop()
        check("сработавшая отсрочка не гасит сервер при локальном источнике",
              stopped["calls"] == 0, str(stopped))

        # Явный запуск сервера отсрочку тоже снимает: сервер только что подняли.
        local_llm.schedule_stop(60)
        local_llm.start()
        check("явный запуск сервера снимает отсрочку",
              not local_llm.stop_scheduled())
    finally:
        local_llm.stop = real_stop
        local_llm.cancel_scheduled_stop()
        config.set_llm_source("remote")
        config.LOCAL_LLM_BASE_URL = CLOSED_URL


# ---------------------------------------------------------------------------
# 8. RAG при локальном источнике: поиск локальный, генерация — локальная модель
#
# ЧТО ПРОВЕРЯЕТСЯ. Маршруты RAG (мини-чат по документам, `/test_rag`, контрольный
# диалог) зовут модель САМИ, своими вызовами — мимо чата агента. Значит, и
# готовность источника они обязаны проверять сами: при выбранной локальной модели
# и остановленном сервере человек должен прочитать ПРИЧИНУ («сервер не запущен»,
# чем поднять), а не «мини-чат не получил ответа от модели». Проверка ловит ровно
# этот разрыв: до неё мини-чат при остановленном сервере отвечал ошибкой связи,
# хотя сервер достаточно было поднять.
#
# Поиск при этом остаётся ЛОКАЛЬНЫМ и без модели: его проверяет `check_rag.py`
# (там же — качество ранжирования). Здесь важно другое: до модели дело не доходит
# раньше, чем источник готов, а после состоявшегося хода задача не остаётся
# «выполняющейся» навсегда.
# ---------------------------------------------------------------------------
RAG_DOC = (
    "Ночные Клинки — банда из Найт-Сити. Основана в 2006 году. "
    "Лидер — Морган Блэкхенд. Специализация: охрана и наёмная работа."
)


def _test_request() -> Request:
    """POST-запрос без тела: маршруту `/test_rag` тело не нужно."""
    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request({"type": "http", "method": "POST",
                    "path": "/api/agent/rag/test", "headers": []}, receive)


async def run_stream(target):
    """Собирает события NDJSON-потока маршрута (как клиент чата)."""
    response = await target if asyncio.iscoroutine(target) else target
    events = []
    async for chunk in response.body_iterator:
        for line in str(chunk).splitlines():
            if line.strip():
                events.append(json.loads(line))
    return events


async def test_rag_local_source():
    print("\n[8] RAG при локальном источнике: мини-чат, /test_rag, диалог")
    config.set_llm_source("remote")
    install_llm_stub()
    profile = chat._current_profile_id()
    base = rag.index_files([{"filename": "заметка.txt", "text": RAG_DOC}],
                           name="Проверка локального RAG", profile=profile)
    await chat.task_create(TaskCreate(name="RAG и локальная модель"))
    await chat.session_create()
    session = chat._current_session()
    await chat.rag_apply(RagApply(enabled=[base["id"]], rewrite=False))
    check("проект с включённой базой готов к прогону",
          bool(chat._current_task()) and bool((base.get("stats") or {}).get("chunks")),
          f"чанков: {(base.get('stats') or {}).get('chunks')}")

    question = "Кто такие Ночные Клинки?"
    # ЛОКАЛЬНЫЙ ИСТОЧНИК, СЕРВЕР МОДЕЛИ НЕ ОТВЕЧАЕТ: причина, а не пустой ответ.
    config.LOCAL_LLM_HOME = HOME_EMPTY
    config.LOCAL_LLM_BASE_URL = CLOSED_URL
    config.set_llm_source("local")

    CALLS["count"] = 0
    events = await run_stream(chat.rag_dialog_chat(ChatMessage(content=question)))
    errors = [str(e.get("text") or "") for e in events if e.get("type") == "error"]
    check("мини-чат: модель НЕ звалась, пока локальный сервер не отвечает",
          CALLS["count"] == 0, f"вызовов: {CALLS['count']}")
    check("мини-чат: названа причина и что делать",
          bool(errors) and "Локальная модель не готова" in errors[0]
          and "local_llm.sh install" in errors[0], str(errors)[:200])
    check("мини-чат: поток закрыт событием done",
          bool(events) and events[-1].get("type") == "done",
          str(events[-1])[:120])

    CALLS["count"] = 0
    events = await run_stream(chat.rag_test(_test_request()))
    errors = [str(e.get("text") or "") for e in events if e.get("type") == "error"]
    check("`/test_rag`: модель НЕ звалась, причина названа",
          CALLS["count"] == 0 and bool(errors)
          and "Локальная модель не готова" in errors[0], str(errors)[:200])
    check("`/test_rag`: поток закрыт событием done",
          bool(events) and events[-1].get("type") == "done",
          str(events[-1])[:120])

    CALLS["count"] = 0
    events = await run_stream(chat.rag_dialog_test(RagDialogTest(scenario=1)))
    errors = [str(e.get("text") or "") for e in events if e.get("type") == "error"]
    check("контрольный диалог: модель НЕ звалась, причина названа",
          CALLS["count"] == 0 and bool(errors)
          and "Локальная модель не готова" in errors[0], str(errors)[:200])
    check("контрольный диалог: поток закрыт событием done",
          bool(events) and events[-1].get("type") == "done",
          str(events[-1])[:120])

    # СЕРВЕР ОТВЕЧАЕТ: тот же вопрос отвечается по документам, а пометка
    # «задача выполняется» после хода СНИМАЕТСЯ (иначе задача осталась бы занятой
    # навсегда: «Пауза» откладывалась бы, а периодический повтор считал бы её
    # идущей).
    config.LOCAL_LLM_HOME = HOME_INSTALLED
    config.LOCAL_LLM_BASE_URL = f"http://127.0.0.1:{_STUB['server'].server_port}/v1"
    CALLS["count"] = 0
    events = await run_stream(chat.rag_dialog_chat(ChatMessage(content=question)))
    bot = next((e for e in events if e.get("type") == "bot"), None)
    check("готовый локальный сервер: ответ получен, модель звалась",
          bot is not None and CALLS["count"] > 0,
          f"вызовов: {CALLS['count']}, события: {[e.get('type') for e in events]}")
    check("ответ по документам пришёл СО СТРОКОЙ ИСТОЧНИКОВ",
          bool(bot) and "📄 Источники:" in str(bot.get("text") or ""),
          str((bot or {}).get("text"))[:200])
    check("пометка «задача выполняется» снята после хода",
          str(session["id"]) not in chat._running_sessions,
          str(chat._running_sessions))

    config.LOCAL_LLM_BASE_URL = CLOSED_URL
    config.LOCAL_LLM_HOME = HOME_EMPTY
    config.set_llm_source("remote")


_STUB = {}


# ---------------------------------------------------------------------------
# 9. Окружение сервера: кэш моделей RAG не уводит веса локальной модели
#
# ЧТО ПРОВЕРЯЕТСЯ. Поиск по базам знаний настраивает кэш HuggingFace ВНУТРИ
# проекта (`HF_HUB_CACHE` = data/rag/models/hub, см. app/ai/rag_embedding.py), и
# это настройка ПРОЦЕССА. Сервер локальной модели запускается из того же процесса
# отдельным процессом, а HF_HUB_CACHE СИЛЬНЕЕ HF_HOME: наследуй он каталог
# моделей RAG — веса искались бы не там, где лежат, и при HF_HUB_OFFLINE=1 не
# нашлись бы вовсе. Сервер при этом остаётся жить и отвечает пустым списком
# моделей: поиск по документам работает, а генерации нет — ровно тот случай,
# который проверяет [3] и который в живом прогоне 07.10 выглядел как «модель
# промолчала».
# ---------------------------------------------------------------------------
def test_server_env():
    print("\n[9] Окружение сервера: кэш RAG не уводит веса локальной модели")
    config.LOCAL_LLM_HOME = HOME_INSTALLED
    saved = {key: os.environ.get(key)
             for key in ("HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE")}
    try:
        for key in saved:
            os.environ.pop(key, None)
        rag_cache = os.path.join(_TMP, "rag-models")
        rag_embedding._configure_hf_env(rag_cache)
        check("поиск RAG уводит кэш HuggingFace в свой каталог (настройка процесса)",
              os.environ.get("HF_HUB_CACHE") == os.path.join(rag_cache, "hub"),
              str(os.environ.get("HF_HUB_CACHE")))
        env = local_llm._server_env()
        check("сервер модели получает СВОЙ кэш, а не кэш моделей RAG",
              env["HF_HUB_CACHE"] == os.path.join(local_llm.models_dir(), "hub")
              and env["HF_HOME"] == local_llm.models_dir(),
              f"HF_HUB_CACHE={env.get('HF_HUB_CACHE')}")
        check("веса модели ищутся там, где они лежат",
              os.path.isdir(os.path.join(
                  env["HF_HUB_CACHE"],
                  "models--" + config.LOCAL_LLM_MODEL.replace("/", "--"))),
              str(os.listdir(env["HF_HUB_CACHE"]))
              if os.path.isdir(env["HF_HUB_CACHE"]) else "каталога нет")
        check("сервер модели остаётся офлайн (веса уже скачаны)",
              env["HF_HUB_OFFLINE"] == "1", str(env.get("HF_HUB_OFFLINE")))
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main():
    # Оба каталога создаются заранее: «пустой» — чтобы был куда писать pid-файл,
    # «установленный» — чтобы проверки состояния видели venv и веса.
    os.makedirs(HOME_EMPTY, exist_ok=True)
    make_installed_home()
    config.LOCAL_LLM_HOME = HOME_EMPTY
    _STUB["server"] = start_stub()
    config.LLM_BASE_URL = f"http://127.0.0.1:{_STUB['server'].server_port}/v1"
    try:
        test_provider_and_pricing()
        test_client_payload(_STUB["server"])
        test_status(_STUB["server"])
        test_source_switch()
        test_auto_stop(_STUB["server"])
        test_server_env()
        # Цикл событий — ОДИН на оба прогона (блокировки задач привязаны к циклу);
        # создаётся явно: asyncio.get_event_loop() устарел в Python 3.12.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(test_routes_guard())
            loop.run_until_complete(test_source_routes())
            loop.run_until_complete(test_rag_local_source())
        finally:
            loop.close()
    finally:
        _STUB["server"].shutdown()
    print("\nИтог: " + ("все проверки пройдены" if not FAILURES
                        else f"ПРОВАЛОВ: {len(FAILURES)} — {FAILURES}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
