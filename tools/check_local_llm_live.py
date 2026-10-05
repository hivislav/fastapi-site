"""ЖИВАЯ проверка МАРШРУТА: действительно ли запрос обслуживает локальная модель.

Запуск (нужны ЗАПУЩЕННОЕ приложение и установленная локальная модель — §5.14):

    ./venv/bin/python tools/check_local_llm_live.py [--url http://127.0.0.1:8000]

Проверка по HTTP, а не «в процессе»: рабочее приложение держит workspace в
памяти, и второй экземпляр затёр бы файл своим снимком. Ничего в данных не
меняется: запросы идут в обычный режим `POST /api/chat` (он workspace не
трогает), а выбранный источник в конце возвращается на прежний.

ЧТО ДОКАЗЫВАЕТСЯ (и чем именно, а не «на слово»):

  [1] источники ответа, который видит приложение (`GET /api/agent/llm`);
  [2] запрос ОБСЛУЖИЛ ЛОКАЛЬНЫЙ СЕРВЕР: в его журнале появляются строки
      HTTP-доступа на `/v1/chat/completions` — считаются ДО и ПОСЛЕ запроса,
      поэтому «ответ пришёл» и «сервер принял запрос» проверяются одним фактом;
  [3] расход запроса: `cost_rub` = 0 и тариф «локальная модель». Стоимость
      считает САМ КЛИЕНТ по провайдеру, чей адрес принял запрос
      (`client._usage_metrics(provider=spec["provider"])`), — то есть ноль в
      этой строке означает «вызов ушёл локальному провайдеру», а не «мы так
      настроили»;
  [4] ОБРАТНЫЙ ОПЫТ: при источнике «удалённая» тот же запрос даёт ненулевой
      расход по облачному тарифу И НЕ ОСТАВЛЯЕТ СЛЕДОВ в журнале локального
      сервера — маршруты действительно разные. Опыт стоит одну реплику облачной
      модели (единицы токенов); `--skip-remote` его выключает;
  [5] выбранный источник возвращается в исходное состояние.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAILURES = []
ACCESS_MARK = "/v1/chat/completions"


def check(name, condition, detail=""):
    """Одна проверка: печатает результат и копит провалы."""
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(name)


def http_json(url, payload=None, timeout=180):
    """GET/POST JSON через urllib (httpx в проекте не установлен)."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, method="POST" if data is not None else "GET",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", "replace") or "{}")


def access_lines(path):
    """Сколько запросов к модели принял локальный сервер (по его журналу).

    Журнал — ЗЕМЛЯНАЯ ПРАВДА маршрута: строку `"POST /v1/chat/completions"` в
    него пишет сам mlx_lm.server, и никакая настройка приложения её не создаёт.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return sum(1 for line in fh if ACCESS_MARK in line)
    except OSError:
        return 0


def wait_ready(base, timeout=180.0):
    """Ждёт, пока действующий источник станет готов отвечать."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = http_json(base + "/api/agent/llm")
        server = state.get("server") or {}
        if state.get("ready"):
            return state
        if server.get("error"):
            return state
        time.sleep(2)
    return http_json(base + "/api/agent/llm")


def ask(base, text, max_tokens):
    """Один обычный запрос; возвращает (ответ, строка аналитики)."""
    result = http_json(base + "/api/chat",
                       {"content": text, "max_tokens": max_tokens}, timeout=300)
    rows = result.get("analytics") or [{}]
    return str(result.get("bot") or ""), (rows[0] if rows else {})


def main():
    parser = argparse.ArgumentParser(description="Живая проверка маршрута запроса")
    parser.add_argument("--url", default="http://127.0.0.1:8000",
                        help="адрес запущенного приложения")
    parser.add_argument("--skip-remote", action="store_true",
                        help="не делать обратный опыт (не тратить облачные токены)")
    args = parser.parse_args()
    base = args.url.rstrip("/")

    print(f"\n[0] Приложение {base}")
    try:
        state = http_json(base + "/api/agent/llm", timeout=30)
    except (urllib.error.URLError, OSError) as exc:
        print(f"  FAIL приложение не отвечает: {exc}")
        return 1
    original = state.get("source")
    installed = state.get("installed") or {}
    print(f"  источник: {original} · модель: {state.get('model')}")
    print(f"  локальная установка: venv={installed.get('venv')} "
          f"веса={installed.get('model')} "
          f"({(installed.get('model_bytes') or 0) / 1e9:.2f} ГБ)")
    if not (installed.get("venv") and installed.get("model")):
        print("  FAIL локальная модель не установлена — tools/local_llm.sh install")
        return 1

    print("\n[1] Готовлю локальную модель (переключаю и/или поднимаю сервер)")
    if state.get("source") != "local":
        state = http_json(base + "/api/agent/llm/source",
                          {"source": "local", "autostart": True}, timeout=60)
    if not state.get("ready"):
        server = state.get("server") or {}
        if not server.get("running") and not server.get("starting"):
            # Источник УЖЕ локальный, а сервер остановлен: переключение в этом
            # случае ничего не поднимает (выбор не менялся) — просим сервер явно,
            # как это делает кнопка «🧠 Локальная» повторным нажатием.
            state = http_json(base + "/api/agent/llm/server",
                              {"action": "start"}, timeout=60)
        state = wait_ready(base)
    check("источник ответа — локальная модель",
          state.get("source") == "local", str(state.get("source")))
    check("локальный сервер отвечает",
          bool((state.get("server") or {}).get("running")),
          str(state.get("server")))
    if FAILURES:
        print("\nИтог: локальную модель поднять не удалось — " + str(state.get("hint")))
        return 1
    log_file = (state.get("server") or {}).get("log") or ""
    print(f"  журнал сервера: {log_file}")
    print(f"  моделей объявлено: {(state.get('server') or {}).get('models')}")

    print("\n[2] Запрос и след в журнале ЛОКАЛЬНОГО сервера")
    before = access_lines(log_file)
    text, row = ask(base, "Ответь одним коротким предложением: сколько будет два плюс два?",
                    max_tokens=40)
    after = access_lines(log_file)
    print(f"  ответ: {text[:120]!r}")
    print(f"  строк доступа в журнале: было {before}, стало {after} "
          f"(+{after - before})")
    check("ответ получен", bool(text.strip()), repr(text[:80]))
    check("локальный сервер ПРИНЯЛ запрос (в журнале +1 строка доступа)",
          after > before, f"{before} → {after}")

    print("\n[3] Чем этот вызов посчитан (по провайдеру, принявшему запрос)")
    print(f"  аналитика: {json.dumps(row, ensure_ascii=False)[:220]}")
    check("стоимость вызова — ноль рублей",
          float(row.get("cost_rub") or 0) == 0.0, str(row.get("cost_rub")))
    check("тариф назван локальным (не облачным)",
          (row.get("pricing") or {}).get("provider") == "local",
          str((row.get("pricing") or {}).get("provider")))
    check("в строке аналитики — локальная модель",
          str(state.get("model")) in json.dumps(row, ensure_ascii=False),
          json.dumps(row, ensure_ascii=False)[:160])

    remote_note = "пропущен"
    if not args.skip_remote:
        print("\n[4] Обратный опыт: тот же запрос при источнике «удалённая»")
        http_json(base + "/api/agent/llm/source",
                  {"source": "remote", "autostart": False}, timeout=60)
        before_r = access_lines(log_file)
        text_r, row_r = ask(base, "Ответь одним коротким предложением: сколько будет два плюс два?",
                            max_tokens=24)
        after_r = access_lines(log_file)
        print(f"  ответ: {text_r[:120]!r}")
        print(f"  аналитика: {json.dumps(row_r, ensure_ascii=False)[:220]}")
        check("облачный вызов стоит денег (маршрут отличается от локального)",
              float(row_r.get("cost_rub") or 0) > 0.0, str(row_r.get("cost_rub")))
        check("облачный вызов локальный сервер НЕ обслуживал",
              after_r == before_r, f"{before_r} → {after_r}")
        remote_note = f"{row_r.get('cost_rub')} руб."
    else:
        print("\n[4] Обратный опыт пропущен (--skip-remote)")

    print("\n[5] Возвращаю прежний источник")
    state = http_json(base + "/api/agent/llm/source",
                      {"source": original, "autostart": False}, timeout=60)
    check("источник вернулся в исходное состояние",
          state.get("source") == original,
          f"{state.get('source')} вместо {original}")

    print(f"\nСТОИМОСТЬ ПРОГОНА: локальный запрос — 0 руб., облачный — {remote_note}")
    print("Итог: " + ("запрос идёт через ЛОКАЛЬНУЮ модель — доказано"
                      if not FAILURES else f"ПРОВАЛОВ: {len(FAILURES)} — {FAILURES}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
