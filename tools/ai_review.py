#!/usr/bin/env python
"""AI-РЕВЬЮ ИЗМЕНЕНИЙ ДЛЯ ПАЙПЛАЙНА — вход, который зовёт GitHub Action на PR.

Запуск (в CI и локально одной командой):

    ./venv/bin/python tools/ai_review.py --base origin/main --out review.md

ЧТО ДЕЛАЕТ. Повторяет ровно то, что делает команда `/ai_review` в чате, но без
веб-приложения: берёт изменения ветки через MCP-сервер `git` (только чтение),
собирает контекст кода и документации репозитория, зовёт модель и печатает
отчёт с тремя разделами — потенциальные баги, архитектурные проблемы,
рекомендации. Движок ОДИН (app/ai/ai_review_run.py): здесь только собираются
`Deps` — «чем» подключены инструменты, документация и модель.

RAG ПО КОДОВОЙ БАЗЕ (и почему индекс, а не «весь код в промпт»). Ревью читает
правку вместе с кодовой базой, но отправлять в модель весь код нельзя: в проекте
десятки файлов и сотни тысяч символов. Поэтому скрипт САМ ЗАГРУЖАЕТ код и
документацию репозитория в базу знаний (обычный RAG проекта: чанки, эмбеддинги,
тот же поиск `rag_search.search`) и ищет по ней НЕСКОЛЬКИМИ запросами, собранными
из самой правки, — в модель уходят только найденные фрагменты. Индекс кэшируется
по коммиту (маркер в каталоге базы): повторный прогон на том же коммите не тратит
минуты на пересборку. Управление: `--rag auto|on|off`, `--reindex`,
`--index-dir`, `--embeddings`. Не собрался индекс (`auto`) — ревью всё равно
состоится: контекст отбирается кодом, и причина названа в отчёте и в диагностике.

ГДЕ ЭТО РАБОТАЕТ ЛУЧШЕ ВСЕГО. На self-hosted раннере — на своей машине: там есть
и пакет эмбеддингов (`sentence-transformers`, кэш моделей в `data/rag/models`), и
постоянный каталог индекса, а значит и семантический поиск, и кэш между прогонами.
На облачном раннере GitHub индексация тоже работает (встроенный офлайн-бэкенд
эмбеддингов, без сети), но индекс собирается заново на каждый прогон.

ПОЧЕМУ ЧЕРЕЗ MCP, А НЕ ЧЕРЕЗ `git diff` В СКРИПТЕ. Тот же сервер, что у агента в
чате: он один отвечает за то, ЧТО показывается модели (ветка против базовой, а
не рабочая копия; окна по строкам; закрытые пути — секреты и `data/`). Второй
способ читать репозиторий разошёлся бы с первым при первой правке — и ревью в CI
проверяло бы не то, что ревью в чате. Серверу нужен Node.js и его зависимости
(`npm ci --prefix mcp_servers`, в Actions это отдельный шаг).

ЧТО ЗНАЧИТ «ВКЛЮЧЕНО». Режим AI-ревью выключается и включается осознанно:
в чате — командами `/ai_review_enable` и `/ai_review_disable` (настройка
ПРОЕКТА), в пайплайне — переменной репозитория `AI_REVIEW_ENABLED`. Скрипт
понимает оба случая через `--enabled`: `auto` (по умолчанию) уважает настройку
проекта, если файл workspace есть (локальный запуск), а в CI, где рабочего файла
нет, считает режим включённым — там решение принято переменной репозитория.

КОДЫ ВОЗВРАТА (важно для пайплайна):
  * 0 — отчёт получен ИЛИ ревью честно не делалось по НАСТРОЙКЕ (режим выключен,
    нет ключа модели): это не проблема кода, и красить из-за неё pull request
    нельзя. Причина всё равно напечатана;
  * 1 — СБОЙ: сервер git недоступен, diff не пришёл, модель не ответила. Причина
    названа словами — молчаливый «зелёный» прогон без ревью хуже красного.

ЧЕГО СКРИПТ НЕ ДЕЛАЕТ. Не публикует отчёт (это шаг пайплайна: комментарий к pull
request и артефакт), не меняет репозиторий и не трогает `data/*.json` рабочего
приложения: у него нет ни записи в workspace, ни обновления настройки проекта.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import config  # noqa: E402
from app.ai import ai_review  # noqa: E402
from app.ai import ai_review_rag  # noqa: E402
from app.ai import ai_review_run  # noqa: E402
from app.ai import client as llm_client  # noqa: E402
from app.ai import mcp as mcp_store  # noqa: E402
from app.ai import rag_search  # noqa: E402
from app.ai import workspace as workspace_store  # noqa: E402

# Предел ожидания одного вызова модели: ревью — длинный ответ, и обрыв по
# короткому таймауту выглядел бы как «модель не ответила».
REVIEW_TIMEOUT = 300.0

# Что печатает заглушка модели в `--dry-run`: это НЕ ревью, и текст об этом
# говорит прямо — иначе «сухой» прогон приняли бы за настоящий отчёт.
DRY_RUN_ANSWER = ("DRY-RUN: ответ модели не запрашивался — проверялись сбор "
                  "данных, контекст и пределы прогона.")



def _log(text: str = "") -> None:
    """Строка хода работы — в stderr: stdout остаётся отчёту."""
    print(text, file=sys.stderr, flush=True)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Разбор аргументов пайплайна."""
    parser = argparse.ArgumentParser(
        description="AI-ревью изменений ветки (то же, что команда /ai_review в чате)")
    parser.add_argument("--repo", default=config.PROJECT_ROOT,
                        help="каталог репозитория (по умолчанию корень проекта)")
    parser.add_argument("--base", default="",
                        help="базовая ветка для сравнения (по умолчанию выбирает сервер git)")
    parser.add_argument("--path", default="",
                        help="ограничить разбор одним файлом или каталогом")
    parser.add_argument("--note", default="",
                        help="пожелание к ревью («проверь обработку ошибок»)")
    parser.add_argument("--out", default="", help="куда записать отчёт (markdown)")
    parser.add_argument("--json", default="", help="куда записать сводку (JSON)")
    parser.add_argument("--quiet", action="store_true",
                        help="не печатать отчёт в stdout (только файлы)")
    parser.add_argument("--enabled", choices=("auto", "on", "off"), default="auto",
                        help="состояние режима: auto — из настройки проекта, если она есть")
    parser.add_argument("--workspace", default="",
                        help="файл проекта для --enabled auto (по умолчанию — "
                             "%s из окружения, иначе рабочий файл проекта)"
                             % ai_review.WORKSPACE_ENV)
    parser.add_argument("--mode-report", action="store_true",
                        help="напечатать JSON о режиме у проекта и выйти: по нему "
                             "пайплайн решает, запускать ли ревью вообще")
    parser.add_argument("--max-files", type=int, default=0,
                        help="сколько файлов читать diff'ом (по умолчанию %d)"
                             % ai_review.MAX_DIFF_FILES)
    parser.add_argument("--max-chars", type=int, default=0,
                        help="предел объёма diff'а в символах (по умолчанию %d)"
                             % ai_review.MAX_DIFF_CHARS)
    parser.add_argument("--max-tokens", type=int, default=0,
                        help="предел ответа модели (по умолчанию %d)"
                             % ai_review.REVIEW_MAX_TOKENS)
    parser.add_argument("--no-context", action="store_true",
                        help="не собирать контекст кода вовсе (ни индексом, ни отбором)")
    parser.add_argument("--project-bases", action="store_true",
                        help="искать ещё и в базах знаний проекта (локально: "
                             "каталог data/rag и настройки проекта)")
    parser.add_argument("--rag", choices=("auto", "on", "off"), default="auto",
                        help="контекст кода: auto — индекс кода (RAG) с откатом на "
                             "отбор кодом, on — только индекс (сбой без него), "
                             "off — только отбор кодом")
    parser.add_argument("--reindex", action="store_true",
                        help="пересобрать индекс кода, даже если он свежий")
    parser.add_argument("--index-dir", default="",
                        help="каталог индексов ревью (по умолчанию RAG_DIR проекта); "
                             "на self-hosted раннере укажите постоянный каталог вне "
                             "рабочей копии — тогда индекс переживёт выкладку кода")
    parser.add_argument("--embeddings", choices=("auto", "hashing",
                                                 "sentence-transformers"),
                        default="", help="бэкенд эмбеддингов (RAG_EMBED_BACKEND)")
    parser.add_argument("--dry-run", action="store_true",
                        help="собрать данные и показать, что уйдёт в модель, без её вызова")
    parser.add_argument("--print-prompt", action="store_true",
                        help="с --dry-run: напечатать сами сообщения модели")
    return parser.parse_args(argv)


def apply_environment(args: argparse.Namespace) -> Dict[str, str]:
    """Переносит настройки индекса в окружение и возвращает их для сводки.

    Каталог индексов и бэкенд эмбеддингов читаются модулями RAG из окружения
    (`RAG_DIR`, `RAG_EMBED_BACKEND`), и это правильно: те же переменные
    действуют в приложении. Поэтому аргументы скрипта ВЫСТАВЛЯЮТ переменные, а не
    заводят второй способ настройки.
    """
    index_dir = str(args.index_dir or os.getenv(ai_review.INDEX_ENV) or "").strip()
    if index_dir:
        os.environ["RAG_DIR"] = os.path.abspath(index_dir)
    if str(args.embeddings or "").strip():
        os.environ["RAG_EMBED_BACKEND"] = str(args.embeddings).strip()
    return {"index_dir": os.environ.get("RAG_DIR", ""),
            "embeddings": os.environ.get("RAG_EMBED_BACKEND", "auto")}


def mode_report(args: argparse.Namespace) -> Dict[str, Any]:
    """Что пайплайн думает о режиме — одной строкой JSON (для гейта в workflow).

    Пайплайн читает настройку ПРОЕКТА (её меняют команды в чате) и выключается,
    когда режим выключен. Чтобы не повторять этот разбор в YAML, решение
    принимает ТОТ ЖЕ код, что и сам прогон: workflow лишь спрашивает и по ответу
    решает, делать ли остальные шаги.
    """
    environment = apply_environment(args)
    mode = project_mode(args)
    enabled = mode["enabled"] if args.enabled == "auto" else (args.enabled == "on")
    has_key, key_reason = model_key_present()
    return {
        "enabled": bool(enabled),
        "reason": mode["reason"],
        "workspace": mode["path"],
        "project": str(mode.get("project") or ""),
        "base": mode["base"],
        "index_dir": environment["index_dir"],
        "embeddings": environment["embeddings"],
        "key": bool(has_key),
        "key_reason": "" if has_key else key_reason,
        "runner_label": str(os.getenv("AI_REVIEW_RUNNER") or ""),
    }


def applied_limits(args: argparse.Namespace) -> Dict[str, int]:
    """Пределы прогона: аргументы пайплайна главнее умолчаний модуля.

    Переопределяются на САМОМ модуле (`ai_review`), потому что ими пользуется и
    прогон (`ai_review_run`): второй набор пределов в скрипте разошёлся бы с
    первым, и «--max-files 3» перестало бы что-либо значить.
    """
    if int(args.max_files or 0) > 0:
        ai_review.MAX_DIFF_FILES = max(1, int(args.max_files))
    if int(args.max_chars or 0) > 0:
        ai_review.MAX_DIFF_CHARS = max(2000, int(args.max_chars))
    return {"files": ai_review.MAX_DIFF_FILES, "chars": ai_review.MAX_DIFF_CHARS}


def _project_of(workspace: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Проект, настройки которого читает пайплайн (None — проектов нет).

    ТРИ ШАГА, и порядок важен. Проекты принадлежат ПРОФИЛЯМ (у каждого профиля
    свои), а пайплайну профиль неизвестен, поэтому:

      1. проект без владельца — `active_task(workspace)`;
      2. иначе проект, который КТО-ТО ДЕРЖИТ АКТИВНЫМ: значения `active_tasks`
         (один профиль — один текущий проект), самый поздний по времени создания;
      3. иначе последний проект файла.

    Шаг 2 появился из живого вопроса «а команды в чате точно управляют
    пайплайном?»: без него пайплайн читал ПОСЛЕДНИЙ СОЗДАННЫЙ проект, а человек
    в приложении мог работать в другом — и `/ai_review_disable` в его проекте не
    выключал бы ревью. Предел честный: если активные проекты есть у НЕСКОЛЬКИХ
    профилей, выбрать между ними нечем — берётся самый поздний, а причина с
    названием проекта уходит в журнал прогона.
    """
    task = workspace_store.active_task(workspace)
    if task is not None:
        return task
    tasks = [item for item in (workspace.get("tasks") or []) if isinstance(item, dict)]
    if not tasks:
        return None
    ids = [str(item.get("id") or "") for item in tasks]
    active = {str(value) for value in (workspace.get("active_tasks") or {}).values()}
    chosen = [index for index, task_id in enumerate(ids) if task_id and task_id in active]
    return tasks[chosen[-1]] if chosen else tasks[-1]


def project_mode(args: argparse.Namespace) -> Dict[str, Any]:
    """Что говорит настройка ПРОЕКТА о режиме ревью (для `--enabled auto`).

    Отдельный источник правды нужен потому, что локальный запуск и пайплайн —
    разные места: в CI файла `data/agent_workspace.json` нет вовсе (каталог
    `data/` не хранится в git), и «проект выключил ревью» там узнать неоткуда.
    Поэтому `auto` уважает настройку, КОГДА ОНА ЕСТЬ, и честно говорит, что
    смотрит её, — а не делает вид, что знает.
    """
    path = str(args.workspace or os.getenv(ai_review.WORKSPACE_ENV) or ""
               or config.AGENT_WORKSPACE_FILE)
    if not os.path.isfile(path):
        return {"found": False, "enabled": True, "base": "", "path": path,
                "project": "",
                "reason": "файла проекта нет (%s) — режим считаю включённым" % path}
    try:
        workspace = workspace_store.load_workspace(path)
    except Exception as exc:                # pragma: no cover - защита
        return {"found": False, "enabled": True, "base": "", "path": path,
                "reason": "файл проекта не прочитался (%s)" % str(exc)[:120]}
    task = _project_of(workspace)
    if task is None:
        return {"found": False, "enabled": True, "base": "", "path": path,
                "reason": "в файле проекта нет ни одного проекта"}
    settings = workspace_store.ai_review_settings(task)
    return {"found": True, "enabled": bool(settings.get("enabled")),
            "base": str(settings.get("base") or ""), "path": path,
            "project": str(task.get("name") or task.get("id") or ""),
            "reason": "настройка проекта «%s»" % str(task.get("name") or task.get("id") or "")}


def model_key_present() -> Tuple[bool, str]:
    """Есть ли чем звать модель (ключ действующего источника).

    Локальному источнику ключ не нужен — сервер модели стоит на этой же машине;
    удалённому нужен, и пустой ключ в CI (секрет репозитория не заведён) — это НЕ
    сбой кода, а неполная настройка: о ней и сообщается, чтобы прогон без ревью не
    выглядел успешным.
    """
    spec = config.provider_spec()
    if str(spec.get("provider") or "") == "local":
        return True, ""
    if str(spec.get("api_key") or "").strip():
        return True, ""
    env_name = "DEEPSEEK_API_KEY" if spec.get("provider") == "deepseek-official" \
        else "YANDEX_API_KEY"
    return False, ("нет ключа модели: задайте %s (в GitHub — секрет репозитория)"
                   % env_name)


async def rag_documents(query: str) -> Dict[str, Any]:
    """Поиск по включённым базам знаний ПРОЕКТА (необязательный, `--project-bases`).

    Это базы, которые человек подключил в приложении (кнопка «RAG»): его
    документы про проект. Индекс кода и документации РЕПОЗИТОРИЯ — другое дело и
    работает всегда (`ai_review_rag`), а этот поиск нужен лишь там, где каталог
    `data/rag` существует рядом с пайплайном.
    """
    workspace = workspace_store.load_workspace()
    task = _project_of(workspace)
    if task is None:
        return {"block": "", "hits": [], "sources": [], "searched": 0,
                "note": "📚 Проекта нет — базы знаний проекта не используются."}
    bases = workspace_store.rag_enabled(task)
    if not bases:
        return {"block": "", "hits": [], "sources": [], "searched": 0,
                "note": "📚 Базы знаний у проекта не подключены."}
    settings = workspace_store.rag_settings(task)
    data = await asyncio.to_thread(rag_search.search, bases, query,
                                   settings=settings)
    hits = rag_search.hits_of(data)
    return {"block": rag_search.block(data), "hits": hits,
            "sources": rag_search.sources(data), "searched": len(bases),
            "note": rag_search.results_note(data)}


def make_code_context(args: argparse.Namespace, used: Dict[str, Any]):
    """Провайдер контекста кода: индекс ревью (RAG) с откатом на отбор кодом.

    Порядок работы: убедиться, что индекс кода про ЭТОТ коммит есть (иначе
    собрать — «загрузить кодовую базу в RAG автоматически»), найти по нему нужные
    фрагменты ТРЕМЯ запросами из самой правки и отдать их модели. Не собрался —
    отбор контекста кодом (`ai_review.context_block`), и причина названа: ревью
    без контекста хуже, чем ревью по отобранному коду, но молчаливое «контекста
    нет» хуже обоих.

    Режим `--rag on` откат ЗАПРЕЩАЕТ: он для тех, кто настроил раннер и хочет
    знать, что индекс работает, а не догадываться.
    """
    # Настройки поиска по индексу — ОБЩИЕ с чатом (app/ai/ai_review_rag.py):
    # второй набор чисел разошёлся бы с первым при первой правке.
    search_settings = ai_review_rag.search_settings()

    async def provide(files: List[Dict[str, Any]], names: List[str],
                      changed: List[str], meta: Dict[str, Any]) -> Dict[str, Any]:
        root = str(args.repo or "")
        sha = str((meta or {}).get("head") or "")
        index = await asyncio.to_thread(
            ai_review_rag.ensure, root, profile=None, sha=sha,
            force=bool(args.reindex), settings=search_settings)
        used["index"] = index
        if index.get("ok"):
            query_list = ai_review_rag.queries(files, names, (meta or {}).get("note") or "")
            data = await asyncio.to_thread(
                ai_review_rag.search, [index["base_id"]], query_list,
                profile=None, settings=search_settings)
            used["rag"] = data
            built = ("индекс собран за %.1f с" % float(index.get("seconds") or 0.0)
                     if index.get("built") else "индекс был свежий")
            info = ai_review_rag.context_info(index, data)
            info["note"] = ("📚 Индекс кода: %d файлов → %d чанков, бэкенд %s; %s; "
                            "запросов к индексу: %d."
                            % (int(index.get("documents") or 0),
                               int(index.get("chunks") or 0),
                               str(index.get("backend") or "—"), built,
                               len(query_list)))
            return info
        info = await asyncio.to_thread(ai_review.context_block, root, files, names,
                                       changed)
        info["note"] = ("⚠ Индекс кода не собран (%s) — контекст отобран кодом: "
                        "целиком %d, связанных %d, документации %d."
                        % (str(index.get("reason") or "причина не названа"),
                           len(info.get("whole") or []), len(info.get("related") or []),
                           len(info.get("docs") or [])))
        return info

    return provide


async def run_review(args: argparse.Namespace) -> Dict[str, Any]:
    """Прогон ревью: события движка → (отчёт, сводка, причина отказа).

    Возвращает {"ok", "soft", "reason", "text", "meta", "events"} — по этому
    словарю и выбирается код возврата: «не делалось по настройке» (soft) — не
    сбой, а всё остальное несделанное — сбой.
    """
    limits = applied_limits(args)
    environment = apply_environment(args)
    mode = project_mode(args)
    enabled = mode["enabled"] if args.enabled == "auto" else (args.enabled == "on")
    base = str(args.base or "").strip() or (mode["base"] if args.enabled == "auto" else "")
    _log("AI-ревью: репозиторий %s, база %s, режим %s (%s)"
         % (args.repo, base or "по умолчанию", "включён" if enabled else "выключен",
            mode["reason"]))
    _log("Контекст кода: %s; индекс: %s; эмбеддинги: %s"
         % ({"auto": "индекс кода (RAG), откат на отбор кодом",
             "on": "только индекс кода (RAG)",
             "off": "только отбор кодом"}[args.rag],
            environment["index_dir"] or "по умолчанию (RAG_DIR)",
            environment["embeddings"]))
    if not enabled:
        return {"ok": False, "soft": True, "text": "", "events": [],
                "reason": "режим AI-ревью выключен (%s). Включите его командой "
                          "%s в чате проекта или переменной репозитория "
                          "AI_REVIEW_ENABLED в пайплайне."
                          % (mode["reason"], ai_review.COMMAND_ENABLE),
                "meta": {"enabled": False, "limits": limits}}
    has_key, key_reason = model_key_present()
    if not has_key and not args.dry_run:
        return {"ok": False, "soft": True, "text": "", "events": [],
                "reason": "ревью не сделано: %s" % key_reason,
                "meta": {"enabled": True, "limits": limits}}

    async def call_tools(calls: List[Dict[str, Any]],
                         limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Вызовы MCP git (сервер запускается отдельным процессом)."""
        return await mcp_store.async_run_calls(calls, limit=limit)

    used: Dict[str, Any] = {"content": "", "metrics": {}, "messages": []}

    async def ask(message_list: List[Dict[str, str]], max_tokens: int,
                  temperature: Optional[float],
                  timeout: Optional[float]) -> Any:
        """Вызов модели; в --dry-run модель НЕ зовётся — сообщения запоминаются.

        Сообщения сохраняются целиком: «сухой» прогон нужен ровно для того, чтобы
        увидеть, ЧТО уходит в модель (и сколько это символов), не платя за вызов.
        """
        used["messages"] = message_list
        if args.dry_run:
            return DRY_RUN_ANSWER, {}
        content, metrics = await llm_client.call_llm_async(
            user_text=(message_list[-1]["content"] if message_list else ""),
            model=config.active_model(), disable_thinking=True,
            max_tokens=max_tokens, temperature=temperature,
            messages=message_list, timeout=timeout or REVIEW_TIMEOUT)
        used["content"], used["metrics"] = content, metrics or {}
        return content, metrics

    entry = mcp_store.find_server(ai_review.SERVER_ID)
    reason = "" if entry else "сервер «%s» не найден в реестре серверов" % ai_review.SERVER_ID
    if entry and not reason:
        reason = mcp_store.availability_error(entry)
    deps = ai_review_run.Deps(
        call_tools, ask, repo_dir=str(args.repo or ""), enabled=True, base=base,
        path=str(args.path or ""), model=config.active_model(),
        provider=config.llm_source(),
        max_tokens=int(args.max_tokens or 0) or ai_review.REVIEW_MAX_TOKENS,
        temperature=ai_review.REVIEW_TEMPERATURE, timeout=REVIEW_TIMEOUT,
        documents=(rag_documents if args.project_bases else None),
        code_context=(None if args.no_context or args.rag == "off"
                      else make_code_context(args, used)),
        server_reason=reason)

    events: List[Dict[str, Any]] = []
    text = ""
    start = time.time()
    async for event in ai_review_run.run(deps, mode=ai_review.MODE_REVIEW,
                                         base=base, note=str(args.note or "")):
        events.append(event)
        kind = event.get("type")
        if kind == "debug":
            _log(str(event.get("text") or ""))
        elif kind == "error":
            _log("⚠ " + str(event.get("text") or ""))
        elif kind == "bot":
            text = str(event.get("text") or "")
        elif kind == "review_start":
            _log("Разбор: режим включён, база %s"
                 % (str(event.get("base") or "").strip() or "по умолчанию"))
    index = used.get("index") or {}
    meta = {
        "enabled": True, "base": base, "repo": str(args.repo or ""),
        "project": str(mode.get("project") or ""),
        "model": config.active_model(), "provider": config.llm_source(),
        "seconds": round(time.time() - start, 1), "limits": limits,
        "events": len(events), "dry_run": bool(args.dry_run),
        "rag": args.rag, "index_dir": environment["index_dir"],
        "embeddings": environment["embeddings"],
        # СВОДКА ПО ИНДЕКСУ — часть отчёта пайплайна: по ней видно, работал ли
        # RAG, чем считались векторы и сколько это заняло (в комментарии к PR это
        # не печатается, а в артефакте и итоге прогона — да).
        "index": {"ok": bool(index.get("ok")), "built": bool(index.get("built")),
                  "base_id": str(index.get("base_id") or ""),
                  "documents": int(index.get("documents") or 0),
                  "chunks": int(index.get("chunks") or 0),
                  "backend": str(index.get("backend") or ""),
                  "seconds": float(index.get("seconds") or 0.0),
                  "reason": str(index.get("reason") or "")},
        "chunks": int(((used.get("rag") or {}).get("hits") or []) and
                      len((used.get("rag") or {}).get("hits") or []) or 0),
    }
    if args.rag == "on" and not args.no_context and not index.get("ok"):
        # РЕЖИМ `--rag on` — обещание, что ревью работает по индексу: молча
        # откатиться на отбор кодом значило бы скрыть поломку раннера (нет пакета
        # эмбеддингов, не пишется каталог индекса) — а его и проверяют.
        return {"ok": False, "soft": False, "text": text, "events": events,
                "reason": ("индекс кода не собран, а --rag on требует именно его: %s"
                           % (str(index.get("reason") or "причина не названа"))),
                "meta": meta, "messages": used["messages"]}
    if text:
        return {"ok": True, "soft": False, "reason": "", "text": text,
                "meta": meta, "events": events, "messages": used["messages"]}
    failure = ""
    for event in events:
        if event.get("type") == "error":
            failure = str(event.get("text") or "")
    return {"ok": False, "soft": False, "text": "", "events": events,
            "reason": failure or "ревью не сделано: движок не вернул отчёт",
            "meta": meta, "messages": used["messages"]}


def dry_run_text(result: Dict[str, Any], args: argparse.Namespace) -> str:
    """Что уйдёт в модель — приписка к «сухому» прогону (`--dry-run`).

    Печатаются РАЗМЕРЫ блоков и, по просьбе, сами сообщения: по ним видно, что
    именно попало в ревью (diff, контекст кода, документация) и не съел ли один
    блок весь бюджет. Отчёт выше при этом уже настоящий — его шапку собирает код
    по фактическим данным сервера.
    """
    messages = result.get("messages") or []
    lines: List[str] = ["", "### Что уйдёт в модель (--dry-run)", ""]
    total = 0
    for index, message in enumerate(messages):
        size = len(str(message.get("content") or ""))
        total += size
        role = str(message.get("role") or "?")
        head = " ".join(str(message.get("content") or "").split())[:90]
        lines.append("- [%d] %s — %d символов: %s…" % (index, role, size, head))
    lines.append("")
    lines.append("Всего в модель: %d символов в %d сообщениях; пределы прогона: "
                 "файлов %s, символов diff %s."
                 % (total, len(messages), result["meta"]["limits"]["files"],
                    result["meta"]["limits"]["chars"]))
    if args.print_prompt:
        lines.append("")
        lines.append("### Сообщения целиком")
        for index, message in enumerate(messages):
            lines.append("")
            lines.append("#### [%d] %s" % (index, message.get("role")))
            lines.append("```")
            lines.append(str(message.get("content") or ""))
            lines.append("```")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    """Точка входа пайплайна: печатает отчёт, пишет файлы, возвращает код."""
    args = parse_args(argv)
    if args.mode_report:
        # Отчёт о режиме — ДО всякой работы и без обращений к модели: workflow
        # спрашивает его первым шагом и по ответу пропускает остальные.
        print(json.dumps(mode_report(args), ensure_ascii=False))
        return 0
    # Каталог репозитория — серверу `git` (он читает ту же переменную): ревью
    # обязано смотреть в ТО ЖЕ дерево, из которого собрано.
    os.environ[ai_review.REPO_ENV] = str(args.repo or "")
    result = asyncio.run(run_review(args))
    text = result.get("text") or ""
    if args.dry_run and result.get("ok"):
        text += dry_run_text(result, args)
    if args.out:
        if text:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(text)
            _log("Отчёт записан: %s (%d символов)" % (args.out, len(text)))
        else:
            # ПУСТОГО ФАЙЛА НЕ СОЗДАЁМ. Отчёта нет — значит и артефакта с
            # комментарием быть не должно: файл нулевой длины пайплайн принимает
            # за готовый отчёт и публикует ПУСТОЙ комментарий к pull request
            # (живой случай: ревью честно не делалось — режим выключен, а под PR
            # появился комментарий из одной строки-маркера).
            _log("Отчёта нет — файл %s не создан (ревью не выполнено)" % args.out)
    if not args.quiet and text:
        print(text)
    if args.json:
        summary = dict(result.get("meta") or {})
        summary.update({"ok": bool(result.get("ok")),
                        "soft": bool(result.get("soft")),
                        "reason": str(result.get("reason") or ""),
                        "chars": len(text or "")})
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
    if result.get("ok"):
        _log("AI-ревью готово (%d символов)" % len(text or ""))
        return 0
    reason = str(result.get("reason") or "")
    _log("AI-ревью не сделано: %s" % (reason or "причина не названа"))
    return 0 if result.get("soft") else 1


if __name__ == "__main__":
    sys.exit(main())
