"""ЖИВОЙ прогон контрольных ДИАЛОГОВ RAG — команды `/test_rag_dialog_1|2`.

ЧТО ДЕЛАЕТ. Дёргает маршрут контрольного диалога у ЗАПУЩЕННОГО приложения (по
HTTP) и печатает стенограмму: 10 реплик на сценарий, где пользователя имитирует
отдельный вызов LLM, отвечает тот же мини-чат, что и в интерфейсе, а в конце
разговор оценивает судья.

ЧЕМ ОТЛИЧАЕТСЯ ОТ КОМАНД В ЧАТЕ. Ничем по сути: это тот же маршрут
(`POST /api/agent/rag/dialog/test`), только запускается из терминала. Нужен,
когда прогон надо повторить на другой базе или с другими настройками поиска, не
переключая проект руками в интерфейсе.

ПОЧЕМУ ПО HTTP, А НЕ «В ПРОЦЕССЕ». Рабочее приложение держит workspace в
памяти: прямой прогон в процессе записал бы файл, а запущенный сервер затёр бы
эту запись своим снимком (у файла один писатель). Поэтому скрипт разговаривает с
приложением по HTTP — ровно так же, как это делает интерфейс.

ГДЕ ПИШЕТ. Прогон идёт в ОТДЕЛЬНОМ проекте (по умолчанию «RAG-диалоги (живой
прогон)»): приложение создаёт его, включает в нём базу и прогоняет диалоги.
Стенограммы видны в интерфейсе — в журнале задачи этого проекта; прежний
активный проект возвращается на место в конце.

Запуск (приложение должно быть запущено):
  ./venv/bin/python tools/check_rag_dialog_live.py                  # оба сценария
  ./venv/bin/python tools/check_rag_dialog_live.py --scenario 1
  ./venv/bin/python tools/check_rag_dialog_live.py --min-score 0.5 --top-k 8
  ./venv/bin/python tools/check_rag_dialog_live.py --url http://127.0.0.1:8000
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any, Dict, Iterator, List

DEFAULT_PROJECT = "RAG-диалоги (живой прогон)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(
        "Живой прогон контрольных диалогов RAG (/test_rag_dialog_1|2) "
        "у запущенного приложения"))
    parser.add_argument("--url", default="http://127.0.0.1:8000",
                        help="адрес запущенного приложения")
    parser.add_argument("--scenario", default="all",
                        help="какой сценарий гнать: 1, 2 или all (по умолчанию all)")
    parser.add_argument("--project", default=DEFAULT_PROJECT,
                        help="название проекта для прогона (создаётся при отсутствии)")
    parser.add_argument("--base", default="",
                        help="id базы знаний (по умолчанию — включённая у профиля)")
    parser.add_argument("--min-score", type=float, default=0.3,
                        help="порог первичной релевантности (рабочий диапазон 0,3–0,5)")
    parser.add_argument("--min-ce", type=float, default=0.0,
                        help="порог уверенности модели (действует только с реранкингом)")
    parser.add_argument("--top-k", type=int, default=8,
                        help="сколько фрагментов идёт в ответ")
    parser.add_argument("--rewrite", action="store_true",
                        help="включить переформулировку запроса (по умолчанию выкл)")
    parser.add_argument("--rerank", action="store_true",
                        help="включить реранкинг пула (по умолчанию выкл)")
    parser.add_argument("--report", default="",
                        help="файл отчёта (по умолчанию — во временном каталоге)")
    parser.add_argument("--timestamps", action="store_true",
                        help="печатать время прихода каждого события — видно, что "
                             "прогон приходит по мере появления, а не в конце")
    parser.add_argument("--reuse-session", action="store_true",
                        help="писать в текущую задачу проекта, а не в новую "
                             "(по умолчанию каждый прогон идёт в свою задачу)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# HTTP: маленький клиент на stdlib (httpx в проекте нет намеренно)
# ---------------------------------------------------------------------------
def request(url: str, body: Any = None, timeout: float = 120.0) -> Any:
    """GET или POST JSON и разбор ответа (сбой — понятная причина, не трассировка)."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="GET" if body is None else "POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        try:
            detail = str(json.loads(detail).get("detail") or detail)
        except ValueError:
            pass
        raise RuntimeError("HTTP %s: %s" % (exc.code, detail[:300]))
    except urllib.error.URLError as exc:
        raise RuntimeError("приложение недоступно (%s): запустите его или укажите "
                           "--url" % exc.reason)
    return json.loads(payload) if payload.strip() else {}


def stream(url: str, body: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """POST и чтение NDJSON-потока построчно (одно событие — одна строка)."""
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json"})
    response = urllib.request.urlopen(req, timeout=900)
    try:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if line:
                yield json.loads(line)
    finally:
        response.close()


def _short(text: str, limit: int) -> str:
    """Строка для терминала: без переводов и с обрезкой."""
    value = " ".join(str(text or "").split())
    return value if len(value) <= limit else value[:limit] + "…"


def _address(item: Dict[str, Any]) -> str:
    """Короткий адрес фрагмента: файл · раздел · чанк · релевантность.

    Соседний фрагмент (кусок той же таблицы, разрезанный границей чанка) оценки
    не имеет и подписывается СЛОВОМ: «релевантность 0,00» читалось бы как
    «фрагмент никуда не годится» (так же подписывает карточки интерфейс).
    """
    parts = [str(item.get("source") or "?")]
    if item.get("section"):
        parts.append(str(item["section"])[:60])
    if item.get("neighbour"):
        parts.append("соседний фрагмент № %s" % (item.get("parent_chunk") or "?"))
        return " · ".join(parts)
    if item.get("number"):
        parts.append("№%s" % item["number"])
    try:
        parts.append("релевантность %.2f" % float(item.get("base_score")
                                                 or item.get("score") or 0.0))
    except (TypeError, ValueError):
        pass
    return " · ".join(parts)


def run_scenario(base: str, number: int, report: List[str],
                 timestamps: bool = False) -> Dict[str, Any]:
    """Один контрольный диалог: печатает стенограмму, копит строки отчёта."""
    print("\n" + "=" * 78)
    print("СЦЕНАРИЙ %d" % number)
    print("=" * 78)
    report.append("\n## Сценарий %d\n" % number)
    started = time.monotonic()
    result: Dict[str, Any] = {"scenario": number, "stats": {}, "usage": {},
                              "seconds": 0.0}
    for event in stream(base + "/api/agent/rag/dialog/test", {"scenario": number}):
        kind = event.get("type")
        if timestamps:
            print("[+%6.2f с] %s" % (time.monotonic() - started, kind))
        if kind == "dialog_start":
            print("Диалог «%s»: реплик %d, базы: %s"
                  % (event.get("title"), event.get("turns"),
                     ", ".join(event.get("bases") or [])))
            print("Цель задачи: %s" % event.get("goal"))
            report.append("\n**Диалог «%s»** — реплик %d, базы: %s\n"
                          % (event.get("title"), event.get("turns"),
                             ", ".join(event.get("bases") or [])))
            report.append("\n**Цель задачи:** %s\n" % event.get("goal"))
        elif kind == "dialog_turn":
            source = {"fixed": "сценарий", "model": "имитация",
                      "script": "план (сбой имитации)"}.get(
                          str(event.get("source")), str(event.get("source")))
            print("\n👤 [%d/%d, %s] %s"
                  % (event["n"], event["total"], source, event["text"]))
            report.append("\n### Реплика %d/%d (%s)\n\n> %s\n"
                          % (event["n"], event["total"], source, event["text"]))
        elif kind == "bot":
            answer = str(event.get("text") or "")
            found = len(event.get("sources") or [])
            print("🤖 [источников: %d, фрагментов: %s] %s"
                  % (found, event.get("hits"), _short(answer, 700)))
            report.append("\n**Ответ** (источников: %d, фрагментов: %s)\n\n%s\n"
                          % (found, event.get("hits"), answer))
            addresses = [_address(item) for item in (event.get("sources") or [])]
            if addresses:
                report.append("\nФрагменты: %s\n" % "; ".join(addresses))
        elif kind == "task_memory":
            print("🧠 %s" % event.get("text"))
            report.append("\n<sub>%s</sub>\n" % str(event.get("text") or ""))
        elif kind == "dialog_error":
            print("⚠ %s" % event.get("text"))
            report.append("\n**⚠ %s**\n" % event.get("text"))
        elif kind == "debug":
            print("   · %s" % event.get("text"))
        elif kind == "dialog_verdict":
            print("\n" + "-" * 78)
            print(event.get("text"))
            report.append("\n### Оценка разговора\n\n```\n%s\n```\n"
                          % event.get("text"))
            result["stats"] = event.get("stats") or {}
            report.append("\n**Факты прогона:** `%s`\n"
                          % json.dumps(result["stats"], ensure_ascii=False))
            report.append("\n**Память задачи в конце:** `%s`\n"
                          % json.dumps(event.get("memory") or {}, ensure_ascii=False))
        elif kind == "done":
            result["usage"] = event.get("usage") or {}
    result["seconds"] = time.monotonic() - started
    stats = result["stats"]
    turns = int(stats.get("turns") or 0)
    with_sources = int(stats.get("with_sources") or 0)
    print("\nИтог сценария %d за %.1f с: ответов с источниками %d из %d%s"
          % (number, result["seconds"], with_sources, turns,
             "" if with_sources == turns else " — ⚠ ЕСТЬ ОТВЕТЫ БЕЗ ИСТОЧНИКОВ"))
    report.append("\n**Длительность:** %.1f с\n" % result["seconds"])
    return result


def main() -> int:
    args = parse_args()
    base = args.url.rstrip("/")
    try:
        workspace = request(base + "/api/agent/workspace")
    except RuntimeError as exc:
        print(exc)
        return 1
    previous = str(workspace.get("active_task") or "")
    rag_view = request(base + "/api/agent/rag")
    bases = rag_view.get("bases") or []
    if not bases:
        print("У профиля нет ни одной базы знаний — прогонять нечего: соберите "
              "базу кнопкой «RAG» в интерфейсе.")
        return 1
    chosen = next((item for item in bases if item.get("id") == args.base), None) \
        if args.base else (next((item for item in bases if item.get("enabled")),
                                bases[0]))
    if chosen is None:
        print("База %s не найдена у профиля." % args.base)
        return 1
    if str(args.scenario).lower() == "all":
        numbers = [1, 2]
    else:
        try:
            numbers = [int(args.scenario)]
        except ValueError:
            print("Не понял сценарий %r: нужно 1, 2 или all." % args.scenario)
            return 2
    if any(number not in (1, 2) for number in numbers):
        print("Сценария с таким номером нет: есть 1 и 2.")
        return 2

    project = next((item for item in (workspace.get("tasks") or [])
                    if str(item.get("name") or "") == args.project), None)
    if project is None:
        print("Проект «%s» создаю (в нём будут видны стенограммы)." % args.project)
        created = request(base + "/api/agent/tasks", {"name": args.project})
        project = next((item for item in (created.get("tasks") or [])
                        if str(item.get("name") or "") == args.project), None)
    else:
        request(base + "/api/agent/tasks/%s/select" % project["id"], {})
    if project is None:
        print("Не удалось создать проект прогона.")
        return 1
    # СВОЯ ЗАДАЧА на прогон: стенограмма не смешивается с прошлыми прогонами —
    # в журнале задачи видно ровно этот разговор.
    if not args.reuse_session:
        request(base + "/api/agent/sessions", {})
    # Настройки поиска ПРОЕКТА прогона: рабочий диапазон порога (0,3–0,5) и
    # выключенные реранкинг с переформулировкой — так прогон мерит то, что видит
    # человек в панели «Поиск и ответы», и не тратит лишние вызовы модели.
    request(base + "/api/agent/rag", {
        "enabled": [chosen["id"]], "rewrite": bool(args.rewrite),
        "rerank": bool(args.rerank), "filter": True,
        "min_score": float(args.min_score), "min_ce": float(args.min_ce),
        "top_k_after": int(args.top_k), "ask_when_empty": True})
    settings = (request(base + "/api/agent/rag").get("search") or {})
    print("ЖИВОЙ ПРОГОН ДИАЛОГОВ RAG — %s" % base)
    print("Проект: «%s» | база: %s (%s чанков) | порог %.2f | реранкинг %s | "
          "переформулировка %s | фрагментов в ответ: %s"
          % (args.project, chosen.get("name"), chosen.get("chunks"),
             float(settings.get("min_score") or 0.0),
             "вкл" if settings.get("rerank") else "выкл",
             "вкл" if settings.get("rewrite") else "выкл",
             settings.get("top_k_after")))
    report: List[str] = [
        "# Живой прогон контрольных диалогов RAG",
        "",
        "Дата: %s" % datetime.now().strftime("%d.%m.%Y %H:%M"),
        "Приложение: %s" % base,
        "Проект: «%s» | база: %s (%s чанков)"
        % (args.project, chosen.get("name"), chosen.get("chunks")),
        "Порог первичной релевантности: %.2f | реранкинг: %s | "
        "переформулировка: %s | фрагментов в ответ: %s"
        % (float(settings.get("min_score") or 0.0),
           "вкл" if settings.get("rerank") else "выкл",
           "вкл" if settings.get("rewrite") else "выкл",
           settings.get("top_k_after")),
    ]
    results = []
    try:
        for number in numbers:
            results.append(run_scenario(base, number, report, args.timestamps))
    finally:
        # Прежний активный проект возвращаем на место: прогон — не «переезд».
        if previous and previous != project.get("id"):
            try:
                request(base + "/api/agent/tasks/%s/select" % previous, {})
            except RuntimeError as exc:
                print("⚠ не удалось вернуть прежний проект: %s" % exc)

    print("\n" + "=" * 78)
    print("ИТОГИ")
    report.append("\n## Итоги\n")
    for item in results:
        stats = item["stats"]
        line = ("сценарий %d: ответов с источниками %s из %s, ошибок %s, "
                "фрагментов %s, %.1f с"
                % (item["scenario"], stats.get("with_sources"), stats.get("turns"),
                   stats.get("errors"), stats.get("fragments"), item["seconds"]))
        print("  " + line)
        report.append("- " + line + "\n")
    path = args.report or os.path.join(
        "/tmp", "rag_dialog_live_%s.md" % datetime.now().strftime("%Y%m%d_%H%M%S"))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report) + "\n")
    print("Отчёт: %s" % path)
    print("Стенограмма видна и в интерфейсе: проект «%s», журнал его задачи."
          % args.project)
    return 0


if __name__ == "__main__":
    sys.exit(main())
