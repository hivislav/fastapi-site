"""Утилиты для работы с JSON-ответами LLM.

Модель может вернуть обрезанный JSON (когда лимит max_tokens слишком мал и
валидный JSON не влезает целиком). Тогда мы «чиним» обрезанный фрагмент,
а если починить нельзя — оборачиваем в валидный JSON.
"""

import json
import re
from typing import Optional


def is_valid_json(text: str) -> bool:
    """True, если текст парсится как JSON."""
    if not text:
        return False
    try:
        json.loads(text)
        return True
    except (ValueError, TypeError):
        return False


def repair_json(text: str) -> Optional[str]:
    """Пытается превратить обрезанный JSON-фрагмент в валидный JSON.

    Закрывает незакрытые скобки/строки, убирает хвостовую запятую, добавляет
    значение после висящего двоеточия. Возвращает валидный JSON или None, если
    починить не удалось.
    """
    if not text:
        return None
    s = text.strip()
    if is_valid_json(s):
        return s

    # Висящая запятая в конце (незаконченный элемент массива/объекта).
    s = re.sub(r",\s*$", "", s)

    # «key:» без значения в конце — дописываем null.
    if re.search(r'"\s*:\s*$', s):
        s = s + " null"

    # Проход: ищем Реальные незакрытые строки и порядок незакрытых скобок.
    stack: list[str] = []
    in_str = False
    escaped = False
    for ch in s:
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch in "[{":
                stack.append(ch)
            elif (
                ch in "]}"
                and stack
                and ((ch == "]" and stack[-1] == "[") or (ch == "}" and stack[-1] == "{"))
            ):
                stack.pop()

    if in_str:  # незакрытая строка — закрываем кавычку
        s = s + '"'
    # Закрываем незакрытые скобки в обратном порядке вложенности.
    s = s + "".join("]" if o == "[" else "}" for o in reversed(stack))

    return s if is_valid_json(s) else None


def wrap_as_json(value: str) -> str:
    """Оборачивает произвольный текст в валидный JSON-объект reply.

    Используется как крайний fallback, когда обрезанный ответ не удалось починить.
    """
    return json.dumps({"reply": value.strip()}, ensure_ascii=False)