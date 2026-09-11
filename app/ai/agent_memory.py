"""Файловое хранилище истории диалога режима «AI-агент».

Веб-слой (app/routers/chat.py) держит память агентского диалога в
модульной переменной, которая живёт только внутри процесса. Чтобы диалог
переживал перезапуск приложения (uvicorn, --reload), история сохраняется
в JSON-файл после каждого обмена репликами и загружается обратно при
старте процесса — диалог продолжается так, будто агент не выключался.

Формат файла:
    {"version": 2,
     "messages": [{"role": "user"|"assistant", "content": "..."}, ...],
     "usage": [{"requests": 1, "input": 0, "output": 0, "limit": null, "overflow": false}, ...],
     "summary": ["резюме старой части переписки (до 10 предложений)", ...]}

"summary" — резюме той части диалога, которая уже не хранится в "messages"
целиком (режим «суммаризация», см. app/ai/agent.py): каждый элемент — отдельное
резюме по N сообщениям, от старых к новым. Ключ необязателен: файл без него
читается как «резюме ещё нет» (обратная совместимость с version 1).

"usage" — расход токенов по каждому запросу пользователя (вход/выход всех
вызовов LLM, лимит вывода и признак его переполнения). Нужен для панели
«Токены диалога» в режиме AI-агента: после перезапуска приложения диаграмма
и таблица восстанавливаются вместе с историей переписки.

Запись атомарная (временный файл + os.replace), поэтому файл не повреждается
при обрыве записи. Подразумевается один процесс-писатель; одновременные
запросы внутри процесса уже сериализованы блокировкой в chat.py.
"""

import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

from app import config

logger = logging.getLogger(__name__)

# Версия формата файла (для будущих миграций структуры).
# 2 — добавлен ключ "summary" (резюме старой части переписки); файлы версии 1
# читаются без изменений, просто без резюме.
_VERSION = 2

# Страховочные лимиты на загрузку: сам агент обрезает историю до 24 реплик /
# ~16 000 символов (см. AgentConfig), поэтому эти значения срабатывают только
# для случайно повреждённого или вручную разросшегося файла.
_MAX_MESSAGES = 500
_MAX_TOTAL_CHARS = 200_000

# Сколько замеров расхода токенов хранить (сам агент помнит ≤ 12 запросов).
_MAX_USAGE = 500

# Сколько частей резюме хранить (режим «суммаризация»).
_MAX_SUMMARY = 500


def _read_payload(path: Optional[str] = None) -> Any:
    """Читает JSON-файл истории. None — файла нет или он повреждён."""
    file_path = path or config.AGENT_MEMORY_FILE
    try:
        if not os.path.isfile(file_path):
            return None
        with open(file_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("История AI-агента: не удалось прочитать %s: %s", file_path, exc)
        return None


def load_agent_memory(path: Optional[str] = None) -> List[Dict[str, str]]:
    """Загружает сохранённую историю диалога из JSON-файла.

    path — путь к файлу; None — путь по умолчанию из конфигурации
    (config.AGENT_MEMORY_FILE). Возвращает список {"role", "content"}
    (только user/assistant, непустые сообщения). Если файла нет или он
    повреждён — пустой список (история начинается заново), ошибок не бросает.
    """
    data = _read_payload(path)
    if data is None:
        return []

    # Принимаем как {"messages": [...]}, так и голый список (для простоты правки).
    messages = data.get("messages") if isinstance(data, dict) else data
    if not isinstance(messages, list):
        logger.warning(
            "История AI-агента: файл %s имеет неожиданную структуру",
            path or config.AGENT_MEMORY_FILE,
        )
        return []

    clean: List[Dict[str, str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = str(msg.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        clean.append({"role": role, "content": content})

    # Страховочная обрезка хвоста истории (голова — самые старые реплики).
    if len(clean) > _MAX_MESSAGES:
        clean = clean[-_MAX_MESSAGES:]
    total_chars = sum(len(m["content"]) for m in clean)
    while total_chars > _MAX_TOTAL_CHARS and clean:
        total_chars -= len(clean[0]["content"])
        clean.pop(0)
    return clean


def load_agent_usage(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Загружает сохранённый расход токенов по запросам пользователя.

    Возвращает список замеров {"requests", "input", "output", "limit",
    "overflow"} — по одному на запрос пользователя, в порядке диалога.
    Файла нет / он повреждён / ключа "usage" нет — пустой список.

    Замеры идут параллельно истории диалога (в истории для каждого запроса
    пользователя есть ответ ассистента), поэтому список нужен фронтенду
    вместе с сообщениями — для панели «Токены диалога».
    """
    data = _read_payload(path)
    if not isinstance(data, dict) or not isinstance(data.get("usage"), list):
        return []

    clean: List[Dict[str, Any]] = []
    for item in data["usage"]:
        if not isinstance(item, dict):
            continue
        try:
            record = {
                "requests": max(1, int(item.get("requests") or 1)),
                "input": max(0, int(item.get("input") or 0)),
                "output": max(0, int(item.get("output") or 0)),
                "limit": int(item["limit"]) if item.get("limit") else None,
                "overflow": item.get("overflow") is True,
                # Вклад служебных вызовов сжатия памяти (суммаризация): строка
                # «из них суммаризация» в панели. У старых замеров (до
                # появления суммаризации) этих полей нет — читаются как нули.
                "summary_requests": max(0, int(item.get("summary_requests") or 0)),
                "summary_input": max(0, int(item.get("summary_input") or 0)),
                "summary_output": max(0, int(item.get("summary_output") or 0)),
            }
        except (TypeError, ValueError):
            continue
        clean.append(record)
    return clean[-_MAX_USAGE:]


def load_agent_summary(path: Optional[str] = None) -> List[str]:
    """Загружает сохранённые части резюме (режим «суммаризация»).

    Возвращает список строк-резюме (от старых частей переписки к новым) —
    именно их агент отправляет в модель отдельным блоком вместе с последними
    N сообщениями «как есть». Файла нет / он повреждён / ключа "summary" нет
    (в т.ч. файл версии 1) — пустой список.
    """
    data = _read_payload(path)
    if not isinstance(data, dict) or not isinstance(data.get("summary"), list):
        return []

    clean: List[str] = []
    for part in data["summary"]:
        text = str(part or "").strip()
        if text:
            clean.append(text)
    return clean[-_MAX_SUMMARY:]


def save_agent_memory(
    messages: List[Dict[str, str]],
    usage: Optional[List[Dict[str, Any]]] = None,
    summary: Optional[List[str]] = None,
    path: Optional[str] = None,
) -> None:
    """Сохраняет историю диалога (расход токенов и резюме) в JSON-файл.

    Атомарная запись (временный файл + os.replace), поэтому файл не
    повреждается при обрыве записи.

    usage — список замеров токенов по запросам пользователя (может быть
    пустым); None — ключ "usage" в файл не пишется.
    summary — части резюме старой части переписки (режим «суммаризация»);
    None — ключ "summary" в файл не пишется.
    path — путь к файлу; None — путь по умолчанию из конфигурации. Каталог
    создаётся при необходимости. Ошибки ввода-вывода пробрасываются наверх —
    вызывающий код решает, как на них реагировать (лог и продолжение работы).
    """
    file_path = path or config.AGENT_MEMORY_FILE
    directory = os.path.dirname(os.path.abspath(file_path))
    os.makedirs(directory, exist_ok=True)

    payload: Dict[str, Any] = {"version": _VERSION, "messages": list(messages)}
    if usage is not None:
        payload["usage"] = [dict(item) for item in usage]
    if summary is not None:
        payload["summary"] = [str(part) for part in summary]
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=".agent_memory.", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        os.replace(tmp_path, file_path)  # атомарная замена — файл не портится
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
