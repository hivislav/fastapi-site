"""СРАВНЕНИЕ КАЧЕСТВА ПОИСКА RAG: без фильтра и rewriting — и с ними.

ЧТО ИЗМЕРЯЕТСЯ. У контрольного набора `/test_rag` (`app/ai/rag_suite.py`) к каждому
вопросу выписан НОМЕР ЧАНКА базы, из которого взят эталон («источник»). Это и есть
золотая разметка: вопрос + чанк, который обязан быть найден. Проверка прогоняет
поиск (`rag_search.search` — тем же кодом, что у агента) в нескольких
конфигурациях и считает:

    * ПОПАДАНИЕ — нужный чанк попал в то, что уходит модели (топ-K после отбора);
    * МЕСТО — на каком месте нужный чанк стоит в полном ранжировании (по нему
      видно, помог ли реранкинг подняться, а не просто «повезло попасть»);
    * ФРАГМЕНТОВ — сколько фрагментов ушло в ответ (бюджет контекста);
    * БЕЗ СЛОВ — у скольких из них нет ни одного слова запроса: это дешёвый
      признак шума (фрагмент попал по одной векторной близости).

Зачем отдельная проверка. Настройки второго этапа (порог, топ-K до и после,
реранкинг, переформулировка) НЕЛЬЗЯ выбирать на глаз: слишком высокий порог
оставляет модель без документов (она честно ответит «в документах этого нет» на
вопрос, ответ на который в базе есть), слишком низкий — пускает шум. Спор решается
замером на настоящих базах.

РЕРАНКЕР ВЫБИРАЕТСЯ ФЛАГОМ `--reranker` (признаки или cross-encoder): так на
одних и тех же вопросах видно, что даёт модель, а что — признаки без модели. В
режиме `auto` (по умолчанию) модель НЕ скачивается: работают признаки.

ЧЕГО ЗДЕСЬ НЕТ: обращений к LLM. Базы только ЧИТАЮТСЯ (`mode=ro`), ничего не
меняется; данных для оценки не нужно — разметка уже в чанках. Переформулировка
запроса моделью тоже не проверяется (для неё нужна сеть): проверяется ЛОКАЛЬНАЯ
переформулировка (`rag_query.local_rewrite`) — она работает всегда и на ней же
основан запасной путь, когда модель недоступна.

Запуск:
    ./venv/bin/python tools/check_rag_quality.py
    ./venv/bin/python tools/check_rag_quality.py --sweep          # подбор порога
    ./venv/bin/python tools/check_rag_quality.py --base kb-… --top 5
    ./venv/bin/python tools/check_rag_quality.py --reranker cross-encoder   # сравнить реранкеры
"""

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import config                                     # noqa: E402
from app.ai import rag_embedding, rag_query, rag_rerank  # noqa: E402
from app.ai import rag_search                              # noqa: E402
from app.ai import rag_store                               # noqa: E402
from app.ai import rag_suite                               # noqa: E402

NUMBER_RE = re.compile(r"№\s*(\d+)")


def enabled_bases(profile: str = "") -> List[str]:
    """Базы, включённые в проектах пользователя (читается файл, не приложение)."""
    try:
        with open(config.AGENT_WORKSPACE_FILE, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return []
    ids: List[str] = []
    for task in (data.get("tasks") or []):
        if not isinstance(task, dict):
            continue
        if profile and str(task.get("profile") or "") != profile:
            continue
        for base_id in ((task.get("rag") or {}).get("enabled") or []):
            key = str(base_id or "").strip().lower()
            if key and key not in ids:
                ids.append(key)
    return ids


def gold_of(case_data: Dict[str, Any]) -> List[int]:
    """Номера чанков, которые обязаны быть найдены по этому вопросу.

    Пусто — ЛОВУШКА: ответа в базе нет вовсе, и правильное поведение поиска —
    не отдать ничего (или отдать заведомо не то, что ответ).
    """
    return [int(value) for value in NUMBER_RE.findall(str(case_data.get("source") or ""))]


def probe(base_ids: List[str], query: str, settings: Dict[str, Any],
          profile: Optional[str]) -> List[Tuple[str, int]]:
    """ПОЛНОЕ ранжирование ТЕМ ЖЕ поиском: (база, номер чанка) от лучшего к худшему.

    Считается тем же кодом, что у агента, но БЕЗ обрезки по Top-K и без порога:
    иначе «место нужного чанка» считалось бы по другой выборке, чем та, из которой
    выбирает поиск, и сравнение конфигураций было бы нечестным. Именно на такой
    ошибке замер уже спотыкался: пул кандидатов 50 против рабочего 20 давал другой
    порядок у одного и того же реранкинга (добавки считаются от середины пула), и
    «место» расходилось с фактической выдачей.
    """
    span = max(1, int(settings.get("top_k_before") or 20))
    rerank = bool(settings.get("rerank"))
    out: List[Tuple[str, int]] = []
    for base_id in base_ids:
        meta = rag_store.get_base(base_id, profile=profile)
        if meta is None:
            continue
        backend = str((meta.get("embedding") or {}).get("backend") or "")
        try:
            vector, _info = rag_embedding.embed_query(query, backend=backend)
        except Exception:                       # база несовместима — пропускаем
            continue
        candidates = rag_store.search(base_id, vector, top_k=span, profile=profile,
                                      query_text=query)
        if rerank:
            # Тем же кодом, что у агента: признаки или cross-encoder (если модель
            # скачана — она даёт совсем другой порядок, и это видно в таблице).
            ranked, _info = rag_rerank.rerank(candidates, query, settings)
        else:
            ranked = candidates
        out.extend((base_id, int(hit.get("index") or 0) + 1) for hit in ranked)
    return out


def run_config(base_ids: List[str], settings: Dict[str, Any], profile: Optional[str],
               rewrite: str, top: int) -> Dict[str, Any]:
    """Прогоняет контрольный набор в одной конфигурации и считает показатели."""
    rows: List[Dict[str, Any]] = []
    for case_data in rag_suite.CASES:
        question = case_data["question"]
        query = rag_query.local_rewrite(question) if rewrite == "local" else question
        result = rag_search.search(base_ids, query, profile=profile, settings=settings)
        hits = [hit for hit in rag_search.hits_of(result) if not hit.get("neighbour")]
        gold = gold_of(case_data)
        numbered = [(str(hit.get("base_id") or ""), int(hit.get("number") or 0))
                    for hit in hits]
        ordered = probe(base_ids, query, settings, profile)
        place = 0
        for index, pair in enumerate(ordered, 1):
            if pair[1] in gold:
                place = index
                break
        rows.append({
            "n": len(rows) + 1,
            "gold": gold,
            "hits": len(hits),
            "found": bool(gold) and any(number in gold for _base, number in numbered),
            "place": place,
            "span": len(ordered),
            "trap": not gold,
            "noisy": len([hit for hit in hits
                          if not float(hit.get("lexical") or 0.0)
                          and not float(hit.get("phrase") or 0.0)]),
            "query": query,
        })
    scored = [row for row in rows if not row["trap"]]
    traps = [row for row in rows if row["trap"]]
    return {
        "rows": rows,
        "hit": sum(1 for row in scored if row["found"]),
        "total": len(scored),
        "span": max([row["span"] for row in rows] or [0]),
        "place": (sum(row["place"] for row in scored if row["place"]) /
                  max(1, len([row for row in scored if row["place"]]))),
        "place_missing": len([row for row in scored if not row["place"]]),
        "hits": sum(row["hits"] for row in rows) / max(1, len(rows)),
        "noisy": sum(row["noisy"] for row in rows) / max(1, len(rows)),
        "trap_hits": sum(row["hits"] for row in traps) / max(1, len(traps)) if traps else 0.0,
    }


def ce_report(base_ids: List[str], settings: Dict[str, Any], profile: Optional[str],
              top: int) -> None:
    """ОТЧЁТ ПО CROSS-ENCODER: что даёт модель и где проходит порог по её оценке.

    Здесь меряется то, на что реранкер реально влияет, — ПОРЯДОК и УВЕРЕННОСТЬ
    модели по каждому фрагменту (попадания в топ-K ограничены первым этапом, и
    никакой реранкер не вернёт то, чего в пуле нет). Отдельно видно, что модель
    говорит про ЛОВУШКУ: вопрос, ответа на который в базе нет. У модели там
    вероятность около нуля у ВСЕХ фрагментов, тогда как итоговая оценка первого
    этапа (косинус + слова) у них 0,7–0,9 — то есть резать шум по смешанной
    оценке нельзя, а по вероятности модели можно. Цена: VERDICT-порог по CE
    выбрасывает и часть нужных фрагментов — таблица порогов показывает, сколько.
    """
    span = max(1, int(settings.get("top_k_before") or 20))
    rows: List[Dict[str, Any]] = []
    trap_ce: List[float] = []
    for case_data in rag_suite.CASES:
        question = case_data["question"]
        gold = gold_of(case_data)
        candidates: List[Tuple[str, Dict[str, Any]]] = []
        for base_id in base_ids:
            meta = rag_store.get_base(base_id, profile=profile)
            if meta is None:
                continue
            backend = str((meta.get("embedding") or {}).get("backend") or "")
            try:
                vector, _info = rag_embedding.embed_query(question, backend=backend)
            except Exception:
                continue
            for item in rag_store.search(base_id, vector, top_k=span, profile=profile,
                                         query_text=question):
                candidates.append((base_id, item))
        if not candidates:
            continue
        pool = [item for _base, item in candidates]
        features = rag_rerank.rerank_features([dict(item) for item in pool], question)
        by_model, info = rag_rerank.rerank([dict(item) for item in pool], question,
                                           {"rerank_backend": "cross-encoder"})
        def place(ranked: List[Dict[str, Any]]) -> int:
            for index, item in enumerate(ranked, 1):
                if int(item.get("index") or 0) + 1 in gold:
                    return index
            return 0
        numbers = [int(item.get("index") or 0) + 1 for item in by_model]
        entries = [float(item.get("ce") or 0.0) for item in by_model]
        gold_ce = [value for number, value in zip(numbers, entries) if number in gold]
        noise_ce = [value for number, value in zip(numbers, entries) if number not in gold]
        if not gold:
            # ЛОВУШКА: правильного ответа нет — интересно, что модель говорит про
            # всё, что первый этап всё равно принёс в пул.
            trap_ce = list(entries)
            print("  %2d. ЛОВУШКА (ответа в базе нет): вероятность модели у %d "
                  "фрагментов пула — %.3f…%.3f (лучшая %.3f)"
                  % (len(rows) + 1, len(entries), min(entries), max(entries),
                     max(entries)))
            continue
        rows.append({"n": len(rows) + 1, "place_features": place(features),
                     "place_model": place(by_model),
                     "gold_ce": max(gold_ce) if gold_ce else 0.0,
                     "noise_ce": max(noise_ce) if noise_ce else 0.0})
        print("  %2d. золото %-10s место: признаки %-5s модель %-5s · "
              "вероятность нужного %.3f, лучшего из прочих %.3f"
              % (rows[-1]["n"], str(gold),
                 rows[-1]["place_features"] or "нет", rows[-1]["place_model"] or "нет",
                 rows[-1]["gold_ce"], rows[-1]["noise_ce"]))
    if not rows:
        print("  нет вопросов с золотой разметкой — отчёт не построить")
        return
    mrr = lambda key: sum(1.0 / row[key] for row in rows if row[key]) / len(rows)
    print("  MRR (средний обратный ранг нужного): признаки %.3f · cross-encoder %.3f"
          % (mrr("place_features"), mrr("place_model")))
    # ПОРОГ ПО СМЕШАННОЙ ОЦЕНКЕ: что реально уходит модели при его росте. Долю
    # релевантных считает САМА модель-реранкер (ce ≥ 0,5): независимого судьи тут
    # нет, поэтому цифра говорит о согласованности порога с моделью, а не об
    # истинной релевантности. Зато видно главное: порог только убирает (меньше
    # фрагментов), а не заменяет их более релевантными, и вместе с шумом уходят
    # продолжения таблиц — нужный чанк теряется.
    print("\n  ПОРОГ ПО СМЕШАННОЙ ОЦЕНКЕ: что уходит модели (судья — сама модель, ce ≥ 0,5):")
    print("    порог  фрагментов  доля релевантных  нужный найден  на ловушке")
    for cut in (0.0, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5):
        sent = good = 0
        hits = total = trap = 0
        for case_data in rag_suite.CASES:
            gold = gold_of(case_data)
            result = rag_search.search(
                base_ids, case_data["question"], profile=profile,
                settings=dict(settings, rewrite=False, rerank=True, filter=True,
                              rerank_backend="cross-encoder", top_k_after=5,
                              min_score=cut))
            delivered = [hit for hit in rag_search.hits_of(result)
                         if not hit.get("neighbour")]
            sent += len(delivered)
            good += len([hit for hit in delivered if float(hit.get("ce") or 0.0) >= 0.5])
            if gold:
                total += 1
                hits += 1 if any(hit["number"] in gold for hit in delivered) else 0
            else:
                trap = len(delivered)
        print("    %-6.2f %-11.1f %-17s %-14s %d"
              % (cut, sent / max(1, len(rag_suite.CASES)),
                 "%.0f%%" % (100.0 * good / max(1, sent)),
                 "%d из %d" % (hits, total), trap))

    print("\n  ПОРОГ ПО ВЕРОЯТНОСТИ CROSS-ENCODER (какие фрагменты оставить):")
    for cut in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7):
        kept = [row for row in rows if row["gold_ce"] >= cut]
        noise = len([value for value in trap_ce if value >= cut])
        print("    от %.2f: нужный фрагмент остаётся у %d из %d вопросов · "
              "с ловушки проходит %d фрагментов из %d"
              % (cut, len(kept), len(rows), noise, len(trap_ce)))


def summary_line(title: str, report: Dict[str, Any]) -> str:
    """Одна строка отчёта: попадания, место, объём выдачи, шум."""
    return ("%-46s попаданий %2d/%2d · место нужного %4.1f из %d · фрагментов %.1f"
            " · без слов запроса %.1f · нет в выдаче %d · на ловушку %.1f"
            % (title, report["hit"], report["total"], report["place"],
               report["span"], report["hits"], report["noisy"],
               report["place_missing"], report["trap_hits"]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Сравнение качества поиска RAG")
    parser.add_argument("--query", default="", help="свой вопрос вместо набора")
    parser.add_argument("--base", action="append", default=[],
                        help="идентификатор базы (можно несколько раз)")
    parser.add_argument("--top", type=int, default=50,
                        help="глубина полного ранжирования (по умолчанию 50)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="порог уверенности модели (0…1; по умолчанию — из настроек)")
    parser.add_argument("--sweep", action="store_true",
                        help="показать, что даёт каждый порог (подбор значения)")
    parser.add_argument("--verbose", action="store_true",
                        help="печатать по каждому вопросу: что нашлось и каким запросом")
    parser.add_argument("--ce-report", action="store_true",
                        help="отчёт по cross-encoder: место нужного чанка и порог "
                             "по вероятности модели")
    parser.add_argument("--reranker", default="", choices=["", "auto", "features",
                                                           "cross-encoder"],
                        help="чем реранкить: признаки или cross-encoder "
                             "(по умолчанию — настройка проекта)")
    args = parser.parse_args()

    profile = os.getenv("RAG_PROFILE", "") or None
    bases = [str(item).strip().lower() for item in args.base] or enabled_bases(profile or "")
    if not bases:
        bases = [str(item.get("id")) for item in rag_store.list_bases(profile=profile)]
        print("Включённых баз не нашлось — беру все базы на диске (%d)." % len(bases))
    if not bases:
        print("Баз знаний нет вовсе: каталог %s" % rag_store.directory())
        return 1

    print("Каталог баз: %s" % rag_store.directory())
    print("Базы: %s" % ", ".join(bases))
    print("Вопросов в наборе: %d (разметка — номера чанков из источников эталонов)"
          % rag_suite.total())
    # Порог теперь ОДИН — уверенность модели (0…1).
    threshold = rag_search.min_ce() if args.threshold is None else args.threshold
    override = {"rerank_backend": args.reranker} if args.reranker else {}
    if args.reranker:
        print("Реранкер задан ключом: %s (состояние: %s)"
              % (args.reranker, rag_rerank.status()["reason"] or "модель доступна"))

    # Конфигурации сравниваются ВМЕСТЕ С ПРЕЖНЕЙ РЕАЛИЗАЦИЕЙ: первая строка —
    # именно она (все три галочки сняты), поэтому видно, что даёт доработка, а не
    # «стало ли лучше вообще».
    defaults = rag_search.settings()
    legacy = {"rewrite": False, "rerank": False, "filter": False,
              "top_k_before": rag_search.top_k(), "top_k_after": rag_search.top_k()}
    configs: List[Tuple[str, Dict[str, Any], str]] = [
        ("все галочки сняты — прежняя реализация (топ-%d)" % rag_search.top_k(),
         legacy, "off"),
        ("по умолчанию проекта: пул %d → ответ %d, порог уверенности %.2f"
         % (defaults["top_k_before"], defaults["top_k_after"], defaults["min_ce"]),
         {"rewrite": defaults["rewrite"], "rerank": defaults["rerank"],
          "filter": defaults["filter"], "top_k_before": defaults["top_k_before"],
          "top_k_after": defaults["top_k_after"], "min_ce": defaults["min_ce"]},
         "off"),
        ("компактнее: пул 20 → ответ 4, порог уверенности %.2f" % threshold,
         {"rewrite": False, "rerank": True, "filter": True,
          "top_k_before": 20, "top_k_after": 4, "min_ce": threshold}, "off"),
        ("то же + ЛОКАЛЬНЫЙ rewriting запроса",
         {"rewrite": True, "rerank": True, "filter": True,
          "top_k_before": 20, "top_k_after": 4, "min_ce": threshold}, "local"),
    ]
    print("\nСРАВНЕНИЕ КОНФИГУРАЦИЙ (порог %.2f)" % threshold)
    reports: List[Tuple[str, Dict[str, Any]]] = []
    for title, settings, rewrite in configs:
        merged = dict(settings)
        merged.update(override)
        report = run_config(bases, merged, profile, rewrite, args.top)
        reports.append((title, report))
        print("  " + summary_line(title, report))
    if args.verbose:
        for title, report in reports:
            print("\n[%s]" % title)
            for row in report["rows"]:
                mark = "ЛОВУШКА" if row["trap"] else (
                    ("НАЙДЕН на месте %d" % row["place"]) if row["found"]
                    else "не найден")
                print("  %2d. %s · %s · фрагментов %d · запрос: %s"
                      % (row["n"], mark, row["gold"] or "—", row["hits"],
                         row["query"][:90]))

    if args.ce_report:
        print("\nОТЧЁТ ПО CROSS-ENCODER (пул %d, порог по вероятности модели)"
              % (rag_search.top_k_before()))
        state = rag_rerank.status()
        print("  модель: %s · %s" % (state["model"],
                                     "в кэше" if state["cached"] else "не скачана"))
        if state["cached"] or args.reranker == "cross-encoder":
            ce_report(bases, {"top_k_before": rag_search.top_k_before()},
                      profile, args.top)
        else:
            print("  модель не скачана — отчёт построить нечем (выберите "
                  "«cross-encoder» в панели или задайте --reranker cross-encoder)")

    if args.sweep:
        print("\nПОДБОР ПОРОГА (реранкинг включён, rewriting выключен)")
        best = None
        for value in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
            report = run_config(bases, dict({"rewrite": False, "rerank": True,
                                             "filter": True, "top_k_before": 20,
                                             "top_k_after": 4, "min_ce": value},
                                            **override), profile, "off", args.top)
            print("  " + summary_line("порог %.2f" % value, report))
            if best is None or (report["hit"], -report["hits"]) > (best[1]["hit"],
                                                                   -best[1]["hits"]):
                best = (value, report)
        if best:
            print("  ЛУЧШИЙ ПО ЗАМЕРУ: порог %.2f — попаданий %d из %d"
                  % (best[0], best[1]["hit"], best[1]["total"]))
    print("\nПроверка ничего не меняла: базы только читались, к модели обращений нет.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
