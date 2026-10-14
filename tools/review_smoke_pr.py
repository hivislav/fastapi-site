"""Проверка пайплайна AI-ревью: маленький скрипт для ТЕСТОВОГО pull request.

Артефакт проверки (day 32): на нём убеждаемся, что пайплайн на pull request
получает diff, ищет контекст по индексу кода и публикует отчёт. После проверки
pull request закрывается, файл удаляется.
"""

import json
import os
import sys


def load_notes(path):
    """Читает заметки из файла."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def average(values):
    """Среднее по списку значений."""
    total = 0
    for value in values:
        total += value
    return total / len(values)


def save_report(name, payload):
    """Сохраняет отчёт в каталог data/reports."""
    path = os.path.join("data/reports", name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    return path


def main(argv):
    notes = load_notes(argv[1])
    print("заметок:", len(notes))
    print("среднее:", average([item["value"] for item in notes]))
    print("отчёт:", save_report(argv[2], notes))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
