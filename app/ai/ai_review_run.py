"""ПРОГОН AI-РЕВЬЮ: изменения через MCP git, контекст, вызов модели, события.

ЧТО ЭТО. Один движок на два входа: команда `/ai_review` в чате (маршрут
`POST /api/agent/review`, app/routers/chat.py) и пайплайн на pull request
(tools/ai_review.py, GitHub Action). Разница между ними только в том, ЧЕМ
подключены внешние вещи, и это передано параметрами (`Deps`): у чата документы
ищутся по базам знаний проекта, у пайплайна их заменяет документация
репозитория; настройка «включено» у чата лежит в проекте, у пайплайна — в
переменной репозитория.

ПОЧЕМУ ДВИЖОК, А НЕ ДВА КОДА. Порядок работы ревью — это контракт: сначала
состояние и список файлов, потом diff окнами, потом контекст, потом один вызов
модели, потом отчёт. Две копии этого порядка разошлись бы при первой правке
(ровно так уже было с ответом по документам), поэтому второй вход просто
собирает `Deps`.

ЧТО ВОЗВРАЩАЕТ. Асинхронный поток событий в словаре чата агента:
`review_start` (команда принята), `debug` (что делает ревью), `bot` (готовый
отчёт вместе с источниками), `error` (отказ с НАЗВАННОЙ причиной). Служебные
`answer_ready`, `usage` и `done` добавляет вызывающий: расход считает его
tracker, а движок о вызовах модели знает только «получилось/нет».
"""

import asyncio
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional

from app.ai import ai_review

# Готовый отчёт «изменений нет» и «режим выключен» — не ошибки, а честный ответ:
# ревью нечего делать. Пустой ответ модели, наоборот, ошибка: её причина
# называется (модель не ответила), и человек не должен думать, что «всё чисто».
EMPTY_CHANGES_TEXT = (
    "Разбирать нечего: ветка не отличается от базовой — изменений нет. "
    "Проверьте базу сравнения: `%s main` — сравнить с веткой main." % ai_review.COMMAND)
NO_DIFF_TEXT = (
    "Изменения есть, но diff не пришёл: сервер git не отдал ни одного окна "
    "diff'а. Причина — в строке выше; повторите команду или проверьте "
    "доступность сервера (кнопка «MCP»).")
EMPTY_ANSWER_TEXT = (
    "⚠ Ревью не получилось: модель не вернула ответ. Повторите команду — "
    "если повтор не удался, посмотрите строки диагностики выше.")


class Deps:
    """Внешние вещи прогона — всё параметрами (движок сам никуда не ходит).

    `call_tools(calls, limit)` — выполнить вызовы MCP git (в чате это
    `mcp_store.async_run_calls`, в пайплайне он же);
    `ask(messages, max_tokens, temperature, timeout)` — один вызов модели,
    возвращает (текст, метрики);
    `code_context(files, names, changed, meta)` — необязательный сбор КОНТЕКСТА
    КОДА: возвращает ту же форму, что `ai_review.context_block` ({"block",
    "whole", "related", "docs", "chars"}, плюс `chunks`, `base`, `note`). Так
    ревью получает RAG: индекс кода и документации репозитория ищется по самой
    правке (`app/ai/ai_review_rag.py`), и в модель уходят ТОЛЬКО найденные
    фрагменты. Не задан — движок отбирает контекст кодом (как раньше);
    `documents(query)` — необязательный поиск по документации проекта:
    возвращает {"block", "hits", "sources", "searched", "note"};
    `set_enabled(value)` — необязательная запись настройки проекта (только чат);
    `server_reason` — почему контекст репозитория недоступен ("" — доступен);
    `enabled` — включён ли режим у проекта (в пайплайне — по переменной).
    """

    def __init__(self, call_tools: Callable[..., Awaitable[List[Dict[str, Any]]]],
                 ask: Callable[..., Awaitable[Any]],
                 *, repo_dir: str = "", enabled: bool = True, base: str = "",
                 path: str = "", profile: str = "", model: str = "",
                 provider: str = "", max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None,
                 timeout: Optional[float] = None,
                 documents: Optional[Callable[[str], Awaitable[Dict[str, Any]]]] = None,
                 code_context: Optional[Callable[..., Awaitable[Dict[str, Any]]]] = None,
                 set_enabled: Optional[Callable[[bool], Awaitable[Dict[str, Any]]]] = None,
                 server_reason: str = "") -> None:
        self.call_tools = call_tools
        self.ask = ask
        self.repo_dir = str(repo_dir or "")
        self.enabled = bool(enabled)
        self.base = str(base or "")
        self.path = str(path or "")
        self.profile = str(profile or "")
        self.model = str(model or "")
        self.provider = str(provider or "")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.documents = documents
        self.code_context = code_context
        self.set_enabled = set_enabled
        self.server_reason = str(server_reason or "")


def git_hint(reason: str = "") -> str:
    """Почему изменений не видно и что с этим делать.

    Молча отвечать «изменений нет» нельзя: чаще всего сервер git просто не
    включён у проекта (кнопка «MCP»), и человек ищет причину в своей ветке.
    """
    text = ("Изменения берутся через MCP — сервер «%s» (ветка, список файлов, "
            "diff ветки). Он не подключён к проекту или недоступен, поэтому "
            "разбирать нечего. Включите его кнопкой «MCP» рядом с шестерёнкой "
            "проекта." % ai_review.SERVER_ID)
    if str(reason or "").strip():
        text += " Причина: %s." % str(reason).strip()
    return text


def _failed_lines(results: List[Dict[str, Any]]) -> List[str]:
    """Строки диагностики по отказавшим вызовам (по одной на вызов)."""
    out: List[str] = []
    for item in (results or []):
        if not isinstance(item, dict) or item.get("ok"):
            continue
        out.append("⚠ %s.%s — источник отказал: %s"
                   % (item.get("server_name") or ai_review.SERVER_ID,
                      item.get("tool"), item.get("error") or "без пояснения"))
    return out


def _texts(results: List[Dict[str, Any]], tool: str) -> List[str]:
    """Тексты успешных ответов одного инструмента (в порядке вызовов)."""
    return [str(item.get("text") or "") for item in (results or [])
            if isinstance(item, dict) and item.get("ok")
            and item.get("tool") == tool]


async def read_diff(deps: Deps, files: List[Dict[str, Any]], base: str,
                    emit: Callable[[str], None]) -> Dict[str, Any]:
    """Читает diff ветки ОКНАМИ по строкам: что прочитано, что нет и почему.

    Почему окнами и с бюджетом на файл. Ответ инструмента ограничен 4000
    символами: один вызов физически не может вернуть большой pull request, а
    молчаливая обрезка выглядела бы как «в файле больше ничего не менялось».
    Поэтому каждый файл читается своими окнами до конца, но не больше своей доли
    общего бюджета — иначе первый же огромный файл съел бы весь контекст, и
    ревью не увидело бы остальных файлов вовсе.

    Возвращает {"text", "files" (прочитанные пути), "chars", "calls", "failed"}.

    Файлов читается не больше `MAX_DIFF_FILES`: предел объявлен в отчёте
    («проверено N из M»), и он обязан быть настоящим — иначе человек считал бы
    проверенным то, что в разбор не попало.
    """
    picked = [item for item in (files or [])
              if isinstance(item, dict) and not item.get("closed")]
    picked = picked[:max(1, int(ai_review.MAX_DIFF_FILES or 1))]
    texts: List[str] = []
    read_files: List[str] = []
    calls = 0
    failed = 0
    used = 0
    for index, item in enumerate(picked):
        rel = str(item.get("path") or "").strip()
        if not rel or calls >= ai_review.MAX_DIFF_CALLS:
            break
        left = len(picked) - index
        share = max(2000, (ai_review.MAX_DIFF_CHARS - used) // max(1, left))
        chunks: List[str] = []
        size = 0
        from_line = 1
        while calls < ai_review.MAX_DIFF_CALLS and size < share:
            call = ai_review.diff_call(rel, base, from_line,
                                       ai_review.DIFF_WINDOW_LINES)
            results = await deps.call_tools([call], 1)
            calls += 1
            window = ai_review.parse_window(
                (results[0].get("text") if results else "") or "")
            if results and not results[0].get("ok"):
                failed += 1
                emit((_failed_lines(results) or ["⚠ diff не пришёл"])[0])
                break
            if not window["body"]:
                break
            # ПРЕДЕЛ ПРОВЕРЯЕТСЯ ДО ДОБАВЛЕНИЯ ОКНА: иначе последнее окно выводило
            # бы объём за бюджет, а в отчёте стояло бы «предел 24 000» — число,
            # которого ревью не соблюдало.
            if chunks and size + len(window["body"]) > share:
                break
            chunks.append(window["body"])
            size += len(window["body"])
            if window["done"] or not window["next_from"]:
                break
            from_line = window["next_from"]
        if chunks:
            body = "\n".join(chunks)
            texts.append("### diff: %s\n%s" % (rel, body))
            used += len(body)
            read_files.append(rel)
        if used >= ai_review.MAX_DIFF_CHARS:
            break
    return {"text": "\n\n".join(texts), "files": read_files, "chars": used,
            "calls": calls, "failed": failed}


async def run(deps: Deps, mode: str = ai_review.MODE_REVIEW, base: str = "",
              note: str = "") -> AsyncIterator[Dict[str, Any]]:
    """Прогон ревью (или переключение режима) — потоком событий.

    Включение и выключение режима идут тем же потоком: команда одна и та же по
    форме, и человек должен видеть её результат в чате, а не гадать, применилась
    ли настройка.
    """
    # --- ВКЛЮЧЕНИЕ И ВЫКЛЮЧЕНИЕ РЕЖИМА -------------------------------------
    if mode in (ai_review.MODE_ENABLE, ai_review.MODE_DISABLE):
        if deps.set_enabled is None:
            yield {"type": "error", "text": (
                "Переключить режим AI-ревью здесь нечем: настройка живёт у "
                "проекта, а не у пайплайна (в пайплайне режим задаёт переменная "
                "репозитория AI_REVIEW_ENABLED).")}
            return
        state = await deps.set_enabled(mode == ai_review.MODE_ENABLE)
        yield {"type": "review_start", "mode": mode,
               "enabled": bool(state.get("enabled")), "base": state.get("base") or ""}
        text = ai_review.state_text(bool(state.get("enabled")),
                                    str(state.get("base") or ""))
        # `note` от настройки — это то, что случилось СВЕРХ переключения: при
        # включении режима фоновая задача грузит код репозитория в базу знаний
        # (§5.11г). Молчать об этом нельзя: человек должен знать, что ревью пока
        # идёт по отбору контекста, а индекс ещё собирается.
        if str(state.get("note") or "").strip():
            text += "\n\n" + str(state["note"]).strip()
        yield {"type": "bot", "text": text}
        return

    # --- РЕВЬЮ --------------------------------------------------------------
    if not deps.enabled:
        yield {"type": "review_start", "mode": mode, "enabled": False, "base": ""}
        yield {"type": "error", "text": ai_review.DISABLED_TEXT}
        return
    if deps.server_reason:
        yield {"type": "review_start", "mode": mode, "enabled": True, "base": base}
        yield {"type": "error", "text": git_hint(deps.server_reason)}
        return

    chosen_base = str(base or deps.base or "").strip()
    yield {"type": "review_start", "mode": mode, "enabled": True,
           "base": chosen_base, "path": deps.path}

    notes: List[str] = []

    def emit(text: str) -> None:
        """Строка диагностики — копится и уходит событиями после шага."""
        if str(text or "").strip():
            notes.append(str(text).strip())

    # 1. СОСТОЯНИЕ И СПИСОК ФАЙЛОВ — через MCP git.
    discovery = ai_review.discovery_calls(chosen_base, deps.path)
    yield {"type": "debug", "text": (
        "🔧 Беру изменения через MCP (%s): %s."
        % (ai_review.SERVER_ID,
           ", ".join(str(call["tool"]) for call in discovery)))}
    results = await deps.call_tools(discovery, len(discovery))
    for line in _failed_lines(results):
        yield {"type": "debug", "text": line}
    files_text = "\n".join(_texts(results, ai_review.FILES_TOOL))
    files = ai_review.parse_files(files_text)
    where = ai_review.parse_files_header(files_text)
    if not files:
        if ai_review.changes_absent(files_text) or files_text.strip():
            yield {"type": "bot", "text": EMPTY_CHANGES_TEXT}
            return
        yield {"type": "error", "text": NO_DIFF_TEXT}
        return
    yield {"type": "debug", "text": "📄 " + ai_review.files_line(files)
           + (" База: %s." % where["base"] if where["base"] else "")}

    # 2. DIFF — окнами по строкам, с бюджетом на файл и честной строкой обрезки.
    diff = await read_diff(deps, files, chosen_base, emit)
    for line in notes:
        yield {"type": "debug", "text": line}
    notes.clear()
    if not diff["text"]:
        yield {"type": "error", "text": NO_DIFF_TEXT}
        return
    yield {"type": "debug", "text": (
        "📄 Diff прочитан по %d %s (%d символов, вызовов: %d)."
        % (len(diff["files"]), ai_review.files_word(len(diff["files"])),
           diff["chars"], diff["calls"]))}

    # 3. КОНТЕКСТ КОДА — либо поиск по ИНДЕКСУ кода и документации (RAG: в модель
    #    уходят только найденные фрагменты), либо отбор контекста кодом. Строка о
    #    начале работы уходит ДО вызова: индексация большого проекта — это минуты,
    #    и молчание в чате выглядело бы как «зависло».
    names = ai_review.symbols(diff["text"])
    changed = [item for item in files if item.get("path") in diff["files"]]
    yield {"type": "debug", "text": (
        "📚 Собираю контекст кода и документации (%s)…"
        % ("поиск по индексу кода — RAG" if deps.code_context is not None
           else "отбор кодом"))}
    if deps.code_context is not None:
        info = await deps.code_context(changed, names, diff["files"], {
            "branch": where["branch"], "base": where["base"] or chosen_base,
            "head": where["head"], "note": note})
    else:
        info = await asyncio.to_thread(ai_review.context_block, deps.repo_dir,
                                       changed, names, diff["files"])
    info = info if isinstance(info, dict) else {}
    if str(info.get("note") or "").strip():
        yield {"type": "debug", "text": str(info["note"])}
    if info.get("chunks"):
        yield {"type": "debug", "text": (
            "📚 Контекст собран поиском: фрагментов %d из базы «%s»%s."
            % (int(info["chunks"]), str(info.get("base") or ""),
               (", индекс: %d файлов, %d чанков, бэкенд %s"
                % (int((info.get("index") or {}).get("documents") or 0),
                   int((info.get("index") or {}).get("chunks") or 0),
                   str((info.get("index") or {}).get("backend") or "—")))
               if (info.get("index") or {}).get("chunks") else ""))}
    elif info.get("block"):
        yield {"type": "debug", "text": (
            "📚 Контекст кода: целиком %d, связанных %d, документации %d "
            "(%d символов)."
            % (len(info.get("whole") or []), len(info.get("related") or []),
               len(info.get("docs") or []), int(info.get("chars") or 0)))}
    else:
        yield {"type": "debug", "text": (
            "📚 Контекст кода не собран: каталог репозитория пуст или файлы не "
            "прочитались (%s)." % (deps.repo_dir or "каталог не задан"))}

    # 4. ДОКУМЕНТАЦИЯ ПРОЕКТА — поиск по базам знаний (у чата он есть, у
    #    пайплайна нет: там документация уже пришла из репозитория выше).
    doc_block = ""
    hits: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []
    searched = 0
    if deps.documents is not None:
        query = ai_review.query_for(files, names, note)
        yield {"type": "debug", "text": "🔎 Ищу в документации проекта по запросу: «%s»"
               % query[:200]}
        found = await deps.documents(query)
        doc_block = str((found or {}).get("block") or "")
        hits = list((found or {}).get("hits") or [])
        sources = list((found or {}).get("sources") or [])
        searched = int((found or {}).get("searched") or 0)
        if str((found or {}).get("note") or "").strip():
            yield {"type": "debug", "text": str(found["note"])}
        if hits:
            best = hits[0]
            yield {"type": "debug", "text": (
                "🔎 Найдено фрагментов: %d; лучший — %s"
                % (len(hits), str(best.get("address") or best.get("path") or "")))}

    # 5. ОДИН ВЫЗОВ МОДЕЛИ — ревью.
    truncated = ai_review.truncation_note(
        total_files=len(files), reviewed=len(diff["files"]),
        diff_chars=diff["chars"], budget=ai_review.MAX_DIFF_CHARS,
        failed=diff["failed"],
        closed=len([item for item in files if item.get("closed")]))
    message_list = ai_review.messages(
        diff_block=(ai_review.files_block(files, reviewed=len(diff["files"]))
                    + "\n\n" + diff["text"]),
        code_block=info["block"], doc_block=doc_block, profile=deps.profile,
        note=note)
    if truncated:
        # Модель обязана знать, что видит не всё: иначе она судит о правке
        # целиком по обрезанному diff и уверенно говорит «здесь всё в порядке».
        message_list.insert(1, {"role": "system", "content": "ВНИМАНИЕ: " + truncated})
    content, _metrics = await deps.ask(
        message_list, deps.max_tokens or ai_review.REVIEW_MAX_TOKENS,
        ai_review.REVIEW_TEMPERATURE if deps.temperature is None else deps.temperature,
        deps.timeout)
    if not str(content or "").strip():
        yield {"type": "error", "text": EMPTY_ANSWER_TEXT}
        return

    # 6. ОТЧЁТ: шапку с фактами собирает КОД, ответ модели идёт как есть.
    text = ai_review.answer(
        content, branch=where["branch"], base=where["base"] or chosen_base,
        head=where["head"], files=files, reviewed=len(diff["files"]),
        context_info=info, docs_used=len(hits), truncated=truncated,
        model=deps.model, provider=deps.provider, note=note)
    yield {"type": "bot", "text": text, "sources": sources, "hits": len(hits)}
