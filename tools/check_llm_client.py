"""Самопроверка HTTP-слоя клиента LLM (app/ai/client.py) без внешней сети.

Поднимает локальный «провайдер» (http.server) и проверяет ровно то, что важно
для расхода токенов и устойчивости:

  * keep-alive — повторные вызовы идут по одному соединению;
  * повторы при 429/5xx с учётом Retry-After;
  * отсутствие повторов при 400 и понятные метрики сбоя (failed);
  * поле thinking не уходит моделям, которые его не принимают (alice);
  * потоковый режим (SSE): текст + usage из последнего блока;
  * отмена асинхронного вызова рвёт соединение, а не ждёт таймаут.

Запуск (внешняя сеть и API-ключ не нужны):

    ./venv/bin/python tools/check_llm_client.py
"""

import asyncio
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ключ и адрес подставляем ДО импорта клиента: клиент читает config при импорте.
# Ключи нужны ОБА: модель по умолчанию живёт у провайдера deepseek-official,
# а модели-URI «gpt://…» — у Yandex (проверки ниже ходят в оба провайдера, но
# обоим подставлен локальный «провайдер»-заглушка).
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
os.environ.setdefault("YANDEX_API_KEY", "test-key")
os.environ["DEEPSEEK_BASE_URL"] = "http://127.0.0.1:0/v1"
os.environ["YANDEX_BASE_URL"] = "http://127.0.0.1:0/v1"

from app import config  # noqa: E402
from app.ai import client  # noqa: E402

FAILURES = []
SCRIPT = {"mode": "ok", "payloads": [], "connections": 0, "requests": 0}


def check(name, condition, detail=""):
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(name)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"   # keep-alive: без него соединение закрывается

    def log_message(self, *args):   # тишина в выводе
        pass

    def do_POST(self):
        SCRIPT["requests"] += 1
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            SCRIPT["payloads"].append(json.loads(body))
        except ValueError:
            SCRIPT["payloads"].append({})
        mode = SCRIPT["mode"]
        if mode == "429":
            self._send(429, json.dumps({"error": "too many requests"}),
                       {"Retry-After": "0"})
        elif mode == "429x2":
            if SCRIPT["requests"] <= 2:
                self._send(429, json.dumps({"error": "too many requests"}),
                           {"Retry-After": "0"})
            else:
                self._send(200, json.dumps({
                    "choices": [{"message": {"content": "Ответ модели"}}],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 7,
                              "total_tokens": 18},
                }, ensure_ascii=False))
        elif mode == "400":
            self._send(400, json.dumps({"error": "bad request"}))
        elif mode == "500":
            self._send(500, json.dumps({"error": "server error"}))
        elif mode == "500x1":
            if SCRIPT["requests"] <= 1:
                self._send(500, json.dumps({"error": "server error"}))
            else:
                self._send(200, json.dumps({
                    "choices": [{"message": {"content": "Ответ модели"}}],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 7,
                              "total_tokens": 18},
                }, ensure_ascii=False))
        elif mode == "hang":
            time.sleep(30)
            self._send(200, json.dumps({"choices": [{"message": {"content": "поздно"}}]}))
        elif mode == "stream":
            self._send_stream()
        elif mode == "stream_no_usage":
            self._send_stream(with_usage=False)
        else:
            self._send(200, json.dumps({
                "choices": [{"message": {"content": "Ответ модели"}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7,
                          "total_tokens": 18},
            }, ensure_ascii=False))

    def _send(self, status, text, extra=None):
        payload = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(payload)

    def _send_stream(self, with_usage=True):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def chunk(data):
            raw = data.encode("utf-8")
            self.wfile.write(f"{len(raw):X}\r\n".encode())
            self.wfile.write(raw + b"\r\n")
            self.wfile.flush()

        for piece in ("От", "вет ", "потоком"):
            chunk("data: " + json.dumps(
                {"choices": [{"delta": {"content": piece}}]}, ensure_ascii=False) + "\n\n")
        if with_usage:
            chunk("data: " + json.dumps({
                "choices": [],
                "usage": {"prompt_tokens": 20, "completion_tokens": 5,
                          "total_tokens": 25},
            }) + "\n\n")
        chunk("data: [DONE]\n\n")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


def start_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)

    # Считаем УСТАНОВЛЕННЫЕ соединения: так проверяется keep-alive (иначе
    # каждый вызов открывал бы новое TCP-соединение).
    original_setup = Handler.setup

    def setup(self):
        original_setup(self)
        SCRIPT["connections"] += 1

    Handler.setup = setup
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def flags(**kwargs):
    """Список флагов payload (для проверки отсутствия thinking)."""
    return sorted(kwargs.get("payload", {}).keys())


def main():
    server = start_server()
    # Оба провайдера смотрят на локальную заглушку: проверяется HTTP-слой, а не
    # настоящие адреса (внешняя сеть тесту не нужна).
    config.LLM_BASE_URL = f"http://127.0.0.1:{server.server_port}/v1"
    config.YANDEX_BASE_URL = f"http://127.0.0.1:{server.server_port}/v1"
    print("\n[1] Успешный вызов, метрики и keep-alive")
    SCRIPT["mode"] = "ok"
    SCRIPT["connections"] = 0
    SCRIPT["requests"] = 0
    content, metrics = client.call_llm_with_metrics("Привет")
    check("ответ получен", content == "Ответ модели", repr(content))
    check("метрики токенов верны",
          metrics and metrics["prompt_tokens"] == 11 and metrics["completion_tokens"] == 7,
          str(metrics))
    check("сбой не помечен", metrics and not metrics.get("failed"))
    client.call_llm("Привет")
    client.call_llm("Привет")
    check("три вызова — одно соединение (keep-alive)",
          SCRIPT["connections"] == 1, f"соединений: {SCRIPT['connections']}")

    print("\n[2] Повтор при 429 (Retry-After) и при 500")
    SCRIPT["mode"] = "429x2"
    SCRIPT["requests"] = 0
    content, metrics = client.call_llm_with_metrics("Привет")
    check("после 429 вызов повторён и удался",
          content == "Ответ модели" and not (metrics or {}).get("failed"),
          f"{content!r} / {metrics}")
    check("попыток было больше одной", SCRIPT["requests"] >= 3,
          f"запросов: {SCRIPT['requests']}")

    SCRIPT["mode"] = "500x1"
    SCRIPT["requests"] = 0
    content, metrics = client.call_llm_with_metrics("Привет")
    check("после 500 вызов повторён и удался",
          content == "Ответ модели" and not (metrics or {}).get("failed"),
          f"{content!r} / {metrics}")
    check("при 500 была повторная попытка", SCRIPT["requests"] == 2,
          f"запросов: {SCRIPT['requests']}")

    print("\n[3] Ошибка клиента: без повторов и с метриками сбоя")
    SCRIPT["mode"] = "400"
    SCRIPT["requests"] = 0
    content, metrics = client.call_llm_with_metrics("Привет")
    check("400 — пустой ответ", content == "")
    check("одна попытка (400 не повторяется)", SCRIPT["requests"] == 1,
          f"запросов: {SCRIPT['requests']}")
    check("метрики сбоя с причиной",
          metrics and metrics.get("failed") and "400" in str(metrics.get("error")),
          str(metrics))

    print("\n[4] thinking: у модели по умолчанию выключен, alice поля не получает")
    SCRIPT["mode"] = "ok"
    SCRIPT["payloads"] = []
    # Модель по умолчанию (провайдер deepseek-official): reasoning выключен
    # ВСЕГДА — поле thinking уходит даже без max_tokens/stop/temperature.
    client.call_llm("Привет")
    check("модель по умолчанию всегда получает thinking: disabled",
          bool(SCRIPT["payloads"])
          and SCRIPT["payloads"][0].get("thinking") == {"type": "disabled"},
          str(SCRIPT["payloads"]))
    check("модель по умолчанию — deepseek-v4-flash",
          bool(SCRIPT["payloads"])
          and SCRIPT["payloads"][0].get("model") == config.LLM_MODEL,
          str(SCRIPT["payloads"]))
    SCRIPT["payloads"] = []
    client.call_llm("Привет", model="gpt://folder/deepseek-v4-flash/latest",
                    disable_thinking=True)
    check("deepseek получает thinking",
          any("thinking" in payload for payload in SCRIPT["payloads"]),
          str(SCRIPT["payloads"]))
    SCRIPT["payloads"] = []
    client.call_llm("Привет", model="gpt://folder/alice-llm/latest",
                    disable_thinking=True)
    check("alice НЕ получает thinking",
          all("thinking" not in payload for payload in SCRIPT["payloads"]),
          str(SCRIPT["payloads"]))

    print("\n[5] Потоковый режим (SSE)")
    SCRIPT["mode"] = "stream"
    content, metrics = client.call_llm_with_metrics("Привет", stream=True)
    check("текст собран из потока", content == "Ответ потоком", repr(content))
    check("usage взят из потока",
          metrics and metrics["total_tokens"] == 25 and not metrics.get("failed"),
          str(metrics))
    SCRIPT["mode"] = "stream_no_usage"
    content, metrics = client.call_llm_with_metrics("Привет", stream=True)
    check("поток без usage помечается", content == "Ответ потоком"
          and metrics and metrics.get("usage_missing") is True,
          f"{content!r} / {metrics}")

    print("\n[6] Отмена асинхронного вызова обрывает соединение")

    async def cancel_check():
        SCRIPT["mode"] = "hang"
        started = time.perf_counter()
        task = asyncio.ensure_future(client.call_llm_async("Привет", timeout=60))
        await asyncio.sleep(0.3)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return time.perf_counter() - started

    elapsed = asyncio.run(cancel_check())
    check("отмена завершилась быстро (без ожидания таймаута)", elapsed < 5,
          f"{elapsed:.1f} с")

    print("\nИтог: " + ("все проверки пройдены" if not FAILURES
                       else "ПРОВАЛЕНО проверок: %d — %s" % (len(FAILURES), FAILURES)))
    server.shutdown()
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
