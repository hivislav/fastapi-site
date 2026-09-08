"""Файловое хранилище истории диалога режима «AI-агент».

Веб-слой (app/routers/chat.py) держит память агентского диалога в
модульной переменной, которая живёт только внутри процесса. Чтобы диалог
переживал перезапуск приложения (uvicorn, --reload), история сохраняется
в JSON-файл после каждого обмена репликами и загружается обратно при
старте процесса — диалог продолжается так, будто агент не выключался.

Формат файла:
    {"version": 1, "messages": [{"role": "user"|"assistant", "content": "..."}, ...]}

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
_VERSION = 1

# Страховочные лимиты на загрузку: сам агент обрезает историю до 24 реплик /
# ~16 000 символов (см. AgentConfig), поэтому эти значения срабатывают только
# для случайно повреждённого или вручную разросшегося файла.
_MAX_MESSAGES = 500
_MAX_TOTAL_CHARS = 200_000


def load_agent_memory(path: Optional[str] = None) -> List[Dict[str, str]]:
    """Загружает сохранённую историю диалога из JSON-файла.

    path — путь к файлу; None — путь по умолчанию из конфигурации
    (config.AGENT_MEMORY_FILE). Возвращает список {"role", "content"}
    (только user/assistant, непустые сообщения). Если файла нет или он
    повреждён — пустой список (история начинается заново), ошибок не бросает.
    """
    file_path = path or config.AGENT_MEMORY_FILE
    try:
        if not os.path.isfile(file_path):
            return []
        with open(file_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("История AI-агента: не удалось прочитать %s: %s", file_path, exc)
        return []

    # Принимаем как {"messages": [...]}, так и голый список (для простоты правки).
    messages = data.get("messages") if isinstance(data, dict) else data
    if not isinstance(messages, list):
        logger.warning("История AI-агента: файл %s имеет неожиданную структуру", file_path)
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


def save_agent_memory(messages: List[Dict[str, str]], path: Optional[str] = None) -> None:
    """Сохраняет историю диалога в JSON-файл (атомарно).

    path — путь к файлу; None — путь по умолчанию из конфигурации. Каталог
    создаётся при необходимости. Ошибки ввода-вывода пробрасываются наверх —
    вызывающий код решает, как на них реагировать (лог и продолжение работы).
    """
    file_path = path or config.AGENT_MEMORY_FILE
    directory = os.path.dirname(os.path.abspath(file_path))
    os.makedirs(directory, exist_ok=True)

    payload: Dict[str, Any] = {"version": _VERSION, "messages": list(messages)}
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
