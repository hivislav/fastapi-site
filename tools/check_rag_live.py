"""ЖИВАЯ проверка поиска по НАСТОЯЩИМ базам знаний (без сети и без модели).

Отвечает на вопрос «почему агент не нашёл нужный фрагмент»: ищет по тем же базам,
что и агент, тем же кодом (app/ai/rag_search.py) и показывает ПОЛНУЮ картину —
итоговую релевантность, её слагаемые (вектор и текст) и место нужного чанка.

Зачем отдельная проверка. Живой отказ выглядел так: в базе из скана есть чанк с
заголовком «БАНДИТСКОЕ НАСИЛИЕ ПОЛЫХАЕТ НА УЛИЦАХ НАЙТ-СИТИ» и подписью «Автор
Исида Бес», но агент отвечал «автора в документах нет». Причина оказалась не в
индексе и не в обвязке: короткие чанки-заголовки лежат близко к центру облака
векторов и обгоняли нужный чанк по косинусу (0,84 против 0,33), и нужный фрагмент
стоял 211-м из 1188. Поиск это не показывает, а `--chunk` показывает: где нужный
чанк в ранжировании и сколько ему дала лексическая часть.

Что делает:
  * `--query "…"` — вопрос (обязателен): печатает лучшие фрагменты с разбором оценки;
  * `--find "Исида"` — слово, которое ДОЛЖНО быть в ответе: печатает, на каком месте
    стоит первый чанк с этим словом (и входит ли он в топ выдачи);
  * `--base kb-…` — конкретная база (по умолчанию — включённые в проектах профиля);
  * `--top N` — сколько фрагментов показывать (по умолчанию 10).

Ничего не меняет: базы читаются только на чтение (`mode=ro`), workspace, профили и
файлы данных не переписываются. Секретов не печатает, к модели не обращается.

Запуск:  ./venv/bin/python tools/check_rag_live.py --query "кто автор статьи" --find "Исида"
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import config                                   # noqa: E402
from app.ai import rag_embedding, rag_search, rag_store  # noqa: E402


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
            print("  попаданий нет вовсе (порог релевантности %.2f)"
                  % rag_search.min_score())
            continue
        for number, hit in enumerate(hits[:max(1, args.top)], 1):
            print("  %2d) %.3f (вектор %.3f + текст %.3f) №%s %s · %s"
                  % (number, hit["score"], hit["vector_score"], hit["lexical"],
                     int(hit.get("index") or 0) + 1, hit["source"],
                     (hit["section"] or "без раздела")[:60]))
        threshold = rag_search.min_score()
        passed = [hit for hit in hits if float(hit["score"]) >= threshold]
        print("  порог %.2f: прошло %d из %d показанных"
              % (threshold, len(passed), len(hits)))
        if not passed:
            print("  В МОДЕЛЬ НЕ УЙДЁТ НИ ОДНОГО ФРАГМЕНТА: всё ниже порога "
                  "(снизьте RAG_MIN_SCORE, если это неверно)")
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
