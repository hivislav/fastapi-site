"""Профили пользователя для режима «AI-агент»: хранилище и блок системного промпта.

Профилей может быть несколько — они полностью изолированы друг от друга: у
каждого СВОИ задачи и диалоги (см. app/ai/workspace.py) и свой системный
промпт сессии. Пользователь переключается между профилями в меню профиля
(иконка слева от заголовка панели Workspace). Идентификатор нового профиля —
«user_<цифры>» (например user_13213123): цифры — это уникальный номер
конкретного профиля.

У профиля есть название и пять полей сведений о пользователе:

    profile_name — название профиля (как он виден в списке): нужно ТОЛЬКО
                   пользователю, в системный промпт НЕ отправляется;
    user_name    — имя пользователя;
    occupation   — род деятельности;
    style        — стиль общения;
    answer_format— формат ответа;
    limits       — ограничения.

В системный промпт сессии уходят только пять полей сведений о пользователе
(см. profile_block).

Формат файла (по умолчанию data/profiles.json):

    {"version": 1,
     "active": "user_13213123",
     "profiles": [
       {"id": "user_13213123",
        "profile_name": "Работа",
        "user_name": "Иван",
        "occupation": "аналитик",
        "style": "кратко, по делу",
        "answer_format": "маркированный список",
        "limits": "без воды и шуток",
        "created": "2025-01-01T12:00:00"}
     ]}

Если профиля нет ни одного (файла нет / он пуст), при запуске приложения
создаётся профиль с идентификатором «user_<случайные цифры>» — тот же
механизм нужен, когда профилей нет в момент создания нового диалога/задачи
(см. ensure_profile).

Запись атомарная (временный файл + os.replace) — как в workspace.py и
agent_memory.py. Подразумевается один процесс-писатель: одновременные запросы
внутри процесса сериализованы блокировкой в chat.py.
"""

import json
import logging
import os
import random
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from app import config

logger = logging.getLogger(__name__)

# Версия формата файла (на будущее — для миграций структуры).
_VERSION = 1

# Страховочные лимиты: срабатывают только для повреждённого или вручную
# разросшегося файла, обычная работа до них не доходит.
MAX_PROFILES = 50
FIELD_LIMIT = 1000  # длина одного поля профиля (символов)
MAX_NAME = 120

# Название профиля (как он виден в списке профилей). Нужно ТОЛЬКО пользователю:
# в системный промпт сессии оно не уходит (см. profile_block).
NAME_KEY = "profile_name"
# Длина названия профиля.
NAME_LIMIT = 120
# Название профиля по умолчанию, создаваемого автоматически (напр. user1321412):
# приставка + номер профиля (те же цифры, что в id, только без подчёркивания).
DEFAULT_NAME_PREFIX = "user"
# Название профиля по умолчанию, если пользователь его не указал.
DEFAULT_PROFILE_LABEL = "Профиль"

# Поля СВЕДЕНИЙ О ПОЛЬЗОВАТЕЛЕ: ключ в файле/API -> подпись для системного
# блока агента. Только эти поля уходят в системный промпт сессии.
PROFILE_FIELDS = (
    ("user_name", "Имя"),
    ("occupation", "Род деятельности"),
    ("style", "Стиль общения"),
    ("answer_format", "Формат ответа"),
    ("limits", "Ограничения"),
)
FIELD_KEYS = tuple(key for key, _ in PROFILE_FIELDS)
# Что правится в меню профиля: название профиля + поля сведений о пользователе.
EDITABLE_KEYS = (NAME_KEY,) + FIELD_KEYS

# Заголовок блока профиля в контексте агента. Это ЕДИНСТВЕННАЯ инструкция,
# которую агент получает как системный промпт сессии: чем пользователь
# представился и в каком виде он хочет получать ответы.
PROFILE_HEADER = (
    "Профиль пользователя — сведения, которые он указал о себе сам. "
    "Учитывай их в КАЖДОМ ответе: обращайся и веди диалог с учётом имени и рода "
    "деятельности, соблюдай стиль общения, выдавай ответ в указанном формате и "
    "не нарушай ограничения. Сам профиль не пересказывай и не цитируй без просьбы."
)
# Название профиля в системный промпт НЕ уходит (и id тоже): в контекст агента
# попадают только пять полей сведений о пользователе — см. profile_block.


def _now() -> str:
    """Метка времени создания профиля (ISO, до секунд)."""
    return datetime.now().isoformat(timespec="seconds")


def new_profile_id() -> str:
    """Идентификатор профиля в формате «user_<цифры>» (напр. user_13213123).

    Цифры — уникальный номер конкретного профиля: 8 случайных цифр
    (без ведущих нулей), чтобы идентификатор читался как число и при этом
    практически не повторялся. Идентификатор служебный: пользователю в списке
    профилей показывается НАЗВАНИЕ профиля, а не он.
    """
    return "user_" + str(random.randint(10_000_000, 99_999_999))


def new_profile_name(profile_id: str = "") -> str:
    """Название профиля по умолчанию — «user<цифры>» (напр. user1321412).

    Цифры берутся из идентификатора профиля (тот же номер, что в id, только без
    подчёркивания) — так название профиля, созданного автоматически, читается
    как одно слово.
    """
    digits = "".join(ch for ch in str(profile_id or "") if ch.isdigit())
    return f"{DEFAULT_NAME_PREFIX}{digits or random.randint(1_000_000, 9_999_999)}"


def empty_profile(profile_id: Optional[str] = None) -> Dict[str, Any]:
    """Пустой профиль: сведения о пользователе не заполнены, есть название."""
    profile: Dict[str, Any] = {"id": profile_id or new_profile_id()}
    profile[NAME_KEY] = new_profile_name(profile["id"])
    profile.update({key: "" for key in FIELD_KEYS})
    profile["created"] = _now()
    return profile


def _clean_text(value: Any) -> str:
    """Значение поля профиля: строка без лишних пробелов, с ограничением длины."""
    return " ".join(str(value or "").split())[:FIELD_LIMIT]


def _clean_name(value: Any) -> str:
    """Название профиля: строка без лишних пробелов, короче полей сведений."""
    return " ".join(str(value or "").split())[:NAME_LIMIT]


def normalize_profile(raw: Any) -> Optional[Dict[str, Any]]:
    """Приводит прочитанный профиль к безопасному виду (None — запись пустая)."""
    if not isinstance(raw, dict):
        return None
    profile_id = str(raw.get("id") or "").strip()[:MAX_NAME] or new_profile_id()
    profile: Dict[str, Any] = {"id": profile_id}
    # Название профиля: если его нет (файл от прежней версии) — берём имя
    # пользователя, иначе название по умолчанию «user<цифры>».
    profile[NAME_KEY] = (
        _clean_name(raw.get(NAME_KEY))
        or _clean_name(raw.get("name"))
        or new_profile_name(profile_id)
    )
    for key in FIELD_KEYS:
        # Совместимость с прежним полем "name" (оно было именем пользователя).
        source = raw.get(key)
        if source in (None, "") and key == "user_name":
            source = raw.get("name")
        profile[key] = _clean_text(source)
    profile["created"] = str(raw.get("created") or _now())
    return profile


def normalize_profiles(raw: Any) -> Dict[str, Any]:
    """Приводит прочитанный файл профилей к безопасному виду."""
    if not isinstance(raw, dict):
        raw = {}
    profiles: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for item in (raw.get("profiles") if isinstance(raw.get("profiles"), list) else []):
        profile = normalize_profile(item)
        if profile is None or profile["id"] in seen:
            continue
        seen.add(profile["id"])
        profiles.append(profile)
    profiles = profiles[-MAX_PROFILES:]
    active = str(raw.get("active") or "").strip() or None
    if active not in seen:
        active = profiles[-1]["id"] if profiles else None
    return {"version": _VERSION, "active": active, "profiles": profiles}


# ---------------------------------------------------------------------------
# Чтение/запись файла
# ---------------------------------------------------------------------------
def _read_payload(path: Optional[str] = None) -> Any:
    file_path = path or config.AGENT_PROFILES_FILE
    try:
        if not os.path.isfile(file_path):
            return None
        with open(file_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("Профили пользователя: не удалось прочитать %s: %s", file_path, exc)
        return None


def load_profiles(path: Optional[str] = None) -> Dict[str, Any]:
    """Загружает профили из JSON-файла (или создаёт пустой набор).

    Файла нет / он повреждён — набор без профилей, без исключений: профиль
    создаст ensure_profile (см. ниже).
    """
    data = _read_payload(path)
    if data is None:
        return normalize_profiles({})
    return normalize_profiles(data)


def save_profiles(store: Dict[str, Any], path: Optional[str] = None) -> None:
    """Сохраняет профили в JSON-файл атомарно (временный файл + os.replace)."""
    file_path = path or config.AGENT_PROFILES_FILE
    directory = os.path.dirname(file_path) or "."
    os.makedirs(directory, exist_ok=True)
    payload = json.dumps(normalize_profiles(store), ensure_ascii=False, indent=2)
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".profiles-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp_path, file_path)
    except OSError:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Доступ к профилям
# ---------------------------------------------------------------------------
def find_profile(store: Dict[str, Any], profile_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """Профиль по id (None — такого профиля нет)."""
    for profile in store.get("profiles", []):
        if profile.get("id") == profile_id:
            return profile
    return None


def active_profile(store: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Текущий профиль (None — профилей ещё нет)."""
    return find_profile(store, store.get("active"))


def create_profile(store: Dict[str, Any], fields: Optional[Dict[str, Any]] = None,
                   profile_id: Optional[str] = None) -> Dict[str, Any]:
    """Создаёт профиль с идентификатором «user_<цифры>» и делает его текущим.

    fields — значения полей: название профиля ("profile_name", как профиль
    виден в списке) и сведения о пользователе (имя, род деятельности, стиль
    общения, формат ответа, ограничения). Название обязательно: если оно не
    передано, берётся название по умолчанию «user<цифры>» (см.
    new_profile_name; проверку обязательности делает веб-слой). Идентификатор
    генерируется сам; готовый профиль возвращается вызывающему.
    """
    used = {p.get("id") for p in store.get("profiles", [])}
    candidate = str(profile_id or "").strip()
    if not candidate or candidate in used:
        candidate = new_profile_id()
        while candidate in used:  # страховка от совпадения случайных номеров
            candidate = new_profile_id()
    profile = empty_profile(candidate)
    apply_fields(profile, fields or {})
    if not profile[NAME_KEY]:
        profile[NAME_KEY] = new_profile_name(candidate)
    store.setdefault("profiles", []).append(profile)
    store["profiles"] = store["profiles"][-MAX_PROFILES:]
    store["active"] = profile["id"]
    return profile


def delete_profile(store: Dict[str, Any], profile_id: str) -> bool:
    """Удаляет профиль. True — профиль был и удалён.

    Если удалён текущий профиль, текущим становится соседний (или ни одного:
    тогда следующий ensure_profile заведёт новый «user_<цифры>»).
    """
    profiles = store.get("profiles", [])
    for index, profile in enumerate(profiles):
        if profile.get("id") == profile_id:
            profiles.pop(index)
            if store.get("active") == profile_id:
                if profiles:
                    store["active"] = profiles[min(index, len(profiles) - 1)]["id"]
                else:
                    store["active"] = None
            return True
    return False


def apply_fields(profile: Dict[str, Any], fields: Dict[str, Any]) -> Dict[str, Any]:
    """Записывает значения полей в профиль (кнопка «Сохранить» в меню профиля).

    Обновляются только переданные ключи из EDITABLE_KEYS: название профиля
    (NAME_KEY) и пять полей сведений о пользователе. Пустая строка очищает поле
    сведений; пустое название отбрасывается (название обязательно). Возвращает
    тот же (изменённый) профиль.
    """
    for key, value in (fields or {}).items():
        if key == NAME_KEY:
            name = _clean_name(value)
            if name:
                profile[NAME_KEY] = name
        elif key in FIELD_KEYS:
            profile[key] = _clean_text(value)
    return profile


def ensure_profile(store: Dict[str, Any]) -> Dict[str, Any]:
    """Гарантирует, что профиль есть: если профилей нет — создаёт пустой.

    Именно так появляется профиль при создании нового диалога/задачи, когда
    пользователь ещё ни одного профиля не заводил: идентификатор генерируется
    автоматически в формате «user_<цифры>» (напр. user_13213123), а название —
    «user<цифры>» (напр. user1321412).
    """
    profile = active_profile(store)
    if profile is not None:
        return profile
    profile = create_profile(store)
    logger.info("Профили пользователя: создан профиль по умолчанию %s (%s)",
                profile["id"], profile[NAME_KEY])
    return profile


def profile_label(profile: Optional[Dict[str, Any]]) -> str:
    """Подпись профиля в интерфейсе — его НАЗВАНИЕ (id не показываем).

    Название обязательно, поэтому запасной вариант нужен только для
    повреждённых данных.
    """
    if not profile:
        return DEFAULT_PROFILE_LABEL
    return _clean_name(profile.get(NAME_KEY)) or DEFAULT_PROFILE_LABEL


def snapshot(store: Dict[str, Any]) -> Dict[str, Any]:
    """Снимок профилей для фронтенда: список, текущий профиль и его поля.

    Отдаёт {"active": id|None,
    "profiles": [{"id", "profile_name", "label", "user_name", "occupation",
    "style", "answer_format", "limits", "created"}, ...],
    "profile": {...}|None} — фронт рисует по нему выпадающий список профилей
    (только названия) и заполняет поля меню. Поле "label" — название профиля
    (id в списке не показывается).
    """
    profiles = [
        dict(profile, label=profile_label(profile))
        for profile in store.get("profiles", [])
    ]
    current = active_profile(store)
    return {
        "active": current["id"] if current else None,
        "profiles": profiles,
        "profile": dict(current) if current else None,
    }


def profile_block(profile: Optional[Dict[str, Any]]) -> str:
    """Системный блок профиля пользователя для контекста агента ("" — нет данных).

    В блок уходят ТОЛЬКО пять полей сведений о пользователе: имя, род
    деятельности, стиль общения, формат ответа и ограничения.

    Название профиля (NAME_KEY) и его id в системный промпт НЕ отправляются
    вообще: название служебное для интерфейса (по нему профиль выбирается в
    списке), id — внутренний. Поэтому у профиля с названием, но без заполненных
    сведений, блока нет и системный промпт сессии остаётся прежним. Не добавляй
    сюда название или id — однажды это уже привело к утечке названия в промпт.
    """
    lines = []
    if profile:
        for key, label in PROFILE_FIELDS:
            value = str(profile.get(key) or "").strip()
            if value:
                lines.append(f"{label}: {value}")
    if not lines:
        return ""
    return PROFILE_HEADER + "\n" + "\n".join(lines)
