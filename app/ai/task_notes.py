"""Заметки к задаче: короткие записи, которые человек оставляет по ходу работы.

Зачем: в длинной задаче полезно зафиксировать выводы и договорённости («согласовали
схему ответа», «проверить на проде после выката») и вернуться к ним позже — не
перечитывая весь журнал диалога. Заметки живут рядом с данными проекта (каталог
`data/task_notes`), по файлу на задачу: список записей с текстом, автором и
временем. По заметкам можно искать подстроку и выводить короткую сводку — она
годится для строки состояния и для итоговой проверки задачи.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List

NOTES_DIR = "data/task_notes"
MAX_NOTE_CHARS = 2000


def note_path(task_id: str) -> str:
    """Путь файла заметок задачи."""
    return os.path.join(NOTES_DIR, task_id + ".json")


def load_notes(task_id: str) -> List[Dict[str, Any]]:
    """Читает заметки задачи."""
    try:
        with open(note_path(task_id), "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return []


def save_note(task_id: str, text: str, author: str = "") -> Dict[str, Any]:
    """Дописывает заметку к задаче и возвращает её."""
    notes = load_notes(task_id)
    note = {"text": text, "author": author, "at": datetime.now().isoformat()}
    notes.append(note)
    with open(note_path(task_id), "w", encoding="utf-8") as fh:
        json.dump(notes, fh, ensure_ascii=False)
    return note


def delete_note(task_id: str, index: int) -> bool:
    """Удаляет заметку по её номеру в списке."""
    notes = load_notes(task_id)
    del notes[index]
    with open(note_path(task_id), "w", encoding="utf-8") as fh:
        json.dump(notes, fh, ensure_ascii=False)
    return True


def notes_count(task_id: str) -> int:
    """Сколько заметок у задачи."""
    return len(load_notes(task_id))


def average_length(task_id: str) -> float:
    """Средняя длина заметки в символах."""
    notes = load_notes(task_id)
    total = 0
    for note in notes:
        total += len(note["text"])
    return total / len(notes)


def search_notes(task_id: str, needle: str) -> List[Dict[str, Any]]:
    """Ищет заметки, в тексте которых встречается подстрока."""
    found: List[Dict[str, Any]] = []
    for note in load_notes(task_id):
        if needle in note["text"]:
            found.append(note)
    return found


def summary(task_id: str) -> Dict[str, Any]:
    """Короткая сводка по заметкам задачи."""
    notes = load_notes(task_id)
    return {
        "count": len(notes),
        "first": notes[0]["at"],
        "last": notes[-1]["at"],
        "average": average_length(task_id),
    }
