"""ЖИВАЯ проверка поиска по НАСТОЯЩИМ базам знаний (без сети).

Отвечает на вопрос «почему агент не нашёл нужный фрагмент»: ищет по тем же базам,
что и агент, тем же кодом (app/ai/rag_search.py) и показывает ПОЛНУЮ картину —
итоговую релевантность, её слагаемые (вектор и текст), место нужного чанка и
УВЕРЕННОСТЬ МОДЕЛИ-РЕРАНКЕРА, по которой работает порог.

Зачем отдельная проверка. Живой отказ выглядел так: в базе из скана есть чанк с
заголовком «БАНДИТСКОЕ НАСИЛИЕ ПОЛЫХАЕТ НА УЛИЦАХ НАЙТ-СИТИ» и подписью «Автор
Исида Бес», но агент отвечал «автора в документах нет». Причина оказалась не в
индексе и не в обвязке: короткие чанки-заголовки лежат близко к центру облака
векторов и обгоняли нужный чанк по косинусу (0,84 против 0,33), и нужный фрагмент
стоял 211-м из 1188. Поиск это не показывает, а `--chunk` показывает: где нужный
чанк в ранжировании и сколько ему дала лексическая часть.

ПОРОГОВ ДВА, И ПРОВЕРКА ПОКАЗЫВАЕТ ОБА, потому что шкалы разные: первичная
релевантность (RAG_MIN_SCORE, 0…2 — косинус + слова запроса, то число, что видно в
карточке источника) и уверенность модели (RAG_MIN_CE, 0…1 — вероятность
cross-encoder, настройка ВТОРОГО этапа). Именно на их смешении строился живой случай
03.10: человек выставил «0,85», глядя на числа 0,63…0,73 в диалоге (это первичная
релевантность), а порог относился к реранкингу, которого в той конфигурации не было.

Что делает:
  * `--query "…"` — вопрос (обязателен): печатает лучшие фрагменты с разбором оценки;
  * `--find "Исида"` — слово, которое ДОЛЖНО быть в ответе: печатает, на каком месте
    стоит первый чанк с этим словом (и входит ли он в топ выдачи);
  * `--base kb-…` — конкретная база (по умолчанию — включённые в проектах профиля);
  * `--top N` — сколько фрагментов показывать (по умолчанию 10);
  * `--min-ce X` — порог уверенности модели для отчёта (по умолчанию — RAG_MIN_CE);
  * `--min-score X` — порог первичной релевантности (по умолчанию — RAG_MIN_SCORE).

Ничего не меняет: базы читаются только на чтение (`mode=ro`), workspace, профили и
файлы данных не переписываются. Секретов не печатает, к LLM не обращается (модель
реранкера — локальная).

Запуск:  ./venv/bin/python tools/check_rag_live.py --query "кто автор статьи" --find "Исида"
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import config                                   # noqa: E402
from app.ai import rag_embedding, rag_rerank, rag_search, rag_store  # noqa: E402


def enabled_bases(profile: str = "") -> List[str]:
    """Базы, включённые в проектах пользователя (из настоящего workspace).

    Читается ФАЙЛ, а не объект приложения: проверка не должна поднимать сервер и
    что-либо менять. Если файла нет или он битый — берём все базы на диске.
    """
    path = config.AGENT_WORKSPACE_FILE
    ids: List[str] = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return []
    for task in (data.get("tasks") or []):
        if not isinstance(task, dict):
            continue
        if profile and str(task.get("profile") or "") != profile:
            continue
        enabled = ((task.get("rag") or {}).get("enabled") or [])
        for base_id in enabled:
            key = str(base_id or "").strip().lower()
            if key and key not in ids:
                ids.append(key)
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Живой поиск по базам знаний")
    parser.add_argument("--query", required=True, help="вопрос для поиска")
    parser.add_argument("--base", action="append", default=[],
                        help="идентификатор базы (можно несколько раз)")
    parser.add_argument("--top", type=int, default=10, help="сколько фрагментов показать")
    parser.add_argument("--find", default="",
                        help="слово, которое должно быть в ответе: показать его место")
    parser.add_argument("--min-ce", type=float, default=None,
                        help="порог уверенности модели для отчёта (0…1)")
    parser.add_argument("--min-score", type=float, default=None,
                        help="порог первичной релевантности для отчёта (0…2)")
    args = parser.parse_args()

    profile = os.getenv("RAG_PROFILE", "")
    bases = [str(item).strip().lower() for item in args.base] or enabled_bases(profile)
    if not bases:
        bases = [str(item.get("id")) for item in rag_store.list_bases(profile=profile or None)]
        print("Включённых баз не нашлось — беру все базы на диске (%d)." % len(bases))
    if not bases:
        print("Баз знаний нет вовсе: каталог %s" % rag_store.directory())
        return 1

    print("Каталог баз: %s" % rag_store.directory())
    print("Запрос: %s" % args.query)
    for base_id in bases:
        meta = rag_store.get_base(base_id, profile=profile or None)
        if meta is None:
            print("\n[%s] базы нет (или она чужая для профиля %r)" % (base_id, profile))
            continue
        stats = meta.get("stats") or {}
        print("\n[%s] «%s»: чанков %s, документов %s, эмбеддинги %s (dim %s)"
              % (base_id, meta.get("name"), stats.get("chunks"), stats.get("documents"),
                 (meta.get("embedding") or {}).get("backend"),
                 (meta.get("embedding") or {}).get("dim")))
        reason = rag_embedding.check_compatible(meta)
        if reason:
            print("  база пропущена: %s" % reason)
            continue
        # Полное ранжирование базы: поиск агента берёт верхушку, а здесь нужно
        # видеть место нужного чанка, даже если он далеко.
        vector, _info = rag_embedding.embed_query(
            args.query, backend=(meta.get("embedding") or {}).get("backend") or "")
        hits = rag_store.search(base_id, vector, top_k=50, profile=profile or None,
                                query_text=args.query)
        if not hits:
            print("  попаданий нет вовсе (поиск вернул пустой список)")
            continue
        for number, hit in enumerate(hits[:max(1, args.top)], 1):
            print("  %2d) %.3f (вектор %.3f + текст %.3f) №%s %s · %s"
                  % (number, hit["score"], hit["vector_score"], hit["lexical"],
                     int(hit.get("index") or 0) + 1, hit["source"],
                     (hit["section"] or "без раздела")[:60]))
        # ДВА ПОРОГА — ДВЕ ШКАЛЫ, и показываем оба: человек, выставивший «0,85»,
        # должен видеть, к какой шкале он его отнёс.
        score_threshold = (rag_search.min_score() if args.min_score is None
                           else float(args.min_score))
        score_passed = [hit for hit in hits[:max(1, args.top)]
                        if float(hit.get("score") or 0.0) >= score_threshold]
        print("  порог первичной релевантности %.2f (RAG_MIN_SCORE): проходит %d из %d "
              "показанных" % (score_threshold, len(score_passed),
                              min(len(hits), max(1, args.top))))
        if not score_passed:
            print("  ПО ЭТОЙ ШКАЛЕ В МОДЕЛЬ НЕ УЙДЁТ НИ ОДНОГО ФРАГМЕНТА: порог выше "
                  "первичной релевантности найденного — агент остановится и "
                  "предложит варианты.")
        # УВЕРЕННОСТЬ МОДЕЛИ — шкала ВТОРОГО этапа (реранкинга). Модель локальная,
        # сети не требует; если её нет — честно говорим, что порог второго этапа
        # применить нечем.
        threshold = rag_search.min_ce() if args.min_ce is None else float(args.min_ce)
        if rag_rerank.scoring_available():
            pool = hits[:max(1, args.top)]
            ranked, info = rag_rerank.rerank(pool, args.query,
                                             {"rerank_backend": "auto"})
            if str(info.get("backend")) == "cross-encoder":
                probabilities = [float(item.get("ce") or 0.0) for item in ranked]
                print("  уверенность модели (%s): %s"
                      % (info.get("model") or "cross-encoder",
                         ", ".join("%.3f" % value for value in probabilities)))
                passed = [value for value in probabilities if value >= threshold]
                print("  порог %.2f (RAG_MIN_CE): проходит %d из %d показанных"
                      % (threshold, len(passed), len(probabilities)))
                if not passed:
                    print("  ПО ЭТОЙ ШКАЛЕ В МОДЕЛЬ НЕ УЙДЁТ НИ ОДНОГО ФРАГМЕНТА: порог "
                          "выше уверенности модели (снизьте его в панели «Поиск и "
                          "ответы»). Агент остановится и предложит варианты.")
            else:
                print("  вероятностей реранкера нет (%s): порог уверенности %.2f "
                      "применить нечем" % (info.get("reason") or "модель недоступна",
                                           threshold))
        else:
            print("  модель-реранкер недоступна (%s): порог уверенности %.2f "
                  "применить нечем — это настройка ВТОРОГО этапа, и без модели она "
                  "не действует" % (rag_rerank.filter_reason("auto"), threshold))
        if args.find:
            needle = args.find.lower()
            for place, hit in enumerate(hits, 1):
                if needle in str(hit.get("text") or "").lower():
                    inside = "в топ-выдачу входит" if place <= rag_search.top_k() \
                        else "в топ-%d НЕ входит" % rag_search.top_k()
                    print("  «%s» найден в №%d на месте %d из %d (%s)"
                          % (args.find, int(hit.get("index") or 0) + 1, place,
                             len(hits), inside))
                    break
            else:
                print("  «%s» не найден ни в одном из %d лучших фрагментов"
                      % (args.find, len(hits)))
    print("\nПроверка ничего не меняла: базы только читались.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
