"""Вложения MCP: бинарные результаты инструментов (xlsx, csv, pdf, png).

Инструмент MCP может вернуть ФАЙЛ. По протоколу это content-блок вида
`{"type": "resource", "resource": {"uri": ..., "mimeType": ..., "blob": "<base64>"}}`
(а также вложенный ресурс со ссылкой `resource_link`). Текстом такой результат
не отдать: данные бинарные, и модель их не прочитает.

Поэтому вложение СОХРАНЯЕТСЯ НА ДИСК (каталог `data/mcp_files`, в git не
попадает), а в результаты диалога, в системный блок «ДАННЫЕ MCP» и в журнал чата
уходит ССЫЛКА: имя файла, размер, mime, sha256 и адрес скачивания
(`GET /api/agent/files/{id}`). Пользователь получает файл карточкой в чате и
может его скачать; модель видит, что файл получен, и не выдумывает пути.

Модуль ничего не знает о конкретных серверах и инструментах: на вход идут
content-блоки как есть, поэтому он работает с любым MCP-сервером.

Файлы дедуплицируются по sha256: тот же самый xlsx, выгруженный повторно, не
плодит копии, а ссылается на уже сохранённый файл.
"""

import base64
import binascii
import hashlib
import logging
import mimetypes
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from app import config

logger = logging.getLogger(__name__)

# Каталог вложений. По умолчанию — data/mcp_files рядом с рабочим файлом
# агента; в тестах переопределяется переменной окружения (как AGENT_*_FILE).
ATTACH_DIR_ENV = "MCP_ATTACH_DIR"

# Пределы: один файл и сколько файлов берём из одного результата инструмента.
# Файл больше предела НЕ сохраняется (в результат уходит пометка с причиной):
# качать гигабайты в чат нельзя, а молча терять вложение — нельзя тем более.
MAX_BYTES = 25 * 1024 * 1024
MAX_PER_RESULT = 5

# СКОЛЬКО ВЛОЖЕНИЯ ЖИВУТ. Файлы копятся сами (каждая выгрузка — новый файл),
# поэтому у каталога есть срок и пределы: старые удаляются, а самые свежие
# остаются. Пределы настраиваются переменными окружения, потому что «сколько
# хранить» — вопрос политики развёртывания, а не кода.
# Ссылки в журнале чата живут дольше: задача, к которой вложение относилось,
# обычно уже закрыта (удалённая ссылка отвечает «Файл не найден»).
TTL_DAYS_ENV = "MCP_ATTACH_TTL_DAYS"
MAX_FILES_ENV = "MCP_ATTACH_MAX_FILES"
MAX_DIR_BYTES_ENV = "MCP_ATTACH_MAX_BYTES"
DEFAULT_TTL_DAYS = 30.0
DEFAULT_MAX_FILES = 500
DEFAULT_MAX_DIR_BYTES = 200 * 1024 * 1024
# Чистим не на каждом сохранении: обход каталога дешевле делать раз в 10 минут.
PRUNE_EVERY_S = 600.0
_last_prune = 0.0

# Имя файла: что оставляем от предложенного сервером и сколько символов.
NAME_CHARS = 80
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_UNSAFE_ID = re.compile(r"[^A-Za-z0-9._-]+")

# Расширение по умолчанию, когда сервер не назвал ни имени, ни mime.
DEFAULT_EXT = ".bin"

# MIME, которые считаем полезными для чата. Неизвестный mime не мешает сохранить
# файл — он просто не попадёт в белый список предпросмотра.
_PREVIEWABLE = ("image/", "text/", "application/pdf", "text/csv")


def _float_env(name: str, default: float, minimum: float) -> float:
    try:
        value = float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default
    return value if value >= minimum else default


def limits() -> Tuple[float, int, int]:
    """Пределы хранения вложений: (срок в днях, файлов, байт на каталог)."""
    return (
        _float_env(TTL_DAYS_ENV, DEFAULT_TTL_DAYS, 0.0),
        int(_float_env(MAX_FILES_ENV, float(DEFAULT_MAX_FILES), 1.0)),
        int(_float_env(MAX_DIR_BYTES_ENV, float(DEFAULT_MAX_DIR_BYTES), 1024.0)),
    )


def prune(force: bool = False, now: Optional[float] = None) -> List[str]:
    """Удаляет старые и лишние вложения; возвращает имена удалённых файлов.

    Порядок правил: сначала ПО ВОЗРАСТУ (старше срока хранения), затем — если
    файлов или байтов всё ещё больше предела — самые старые по времени
    изменения, пока каталог не войдёт в границы. Свежие файлы не удаляются
    никогда: активная задача продолжает работать со своими вложениями.

    Ошибки удаления не поднимаются: чистка не должна ломать запрос агента
    (файл занят, каталог только для чтения) — они логируются.
    """
    global _last_prune
    moment = time.time() if now is None else float(now)
    if not force and moment - _last_prune < PRUNE_EVERY_S:
        return []
    _last_prune = moment
    ttl_days, max_files, max_bytes = limits()
    folder = directory()
    entries: List[Tuple[float, int, str]] = []
    try:
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            try:
                stat = os.stat(path)
            except OSError:
                continue
            if not os.path.isfile(path):
                continue
            entries.append((stat.st_mtime, stat.st_size, name))
    except OSError as exc:
        logger.warning("MCP: каталог вложений не читается: %s", exc)
        return []
    entries.sort()  # старые первыми
    removed: List[str] = []
    keep: List[Tuple[float, int, str]] = []
    if ttl_days > 0:
        deadline = moment - ttl_days * 86400.0
        for item in entries:
            if item[0] < deadline:
                if _remove(folder, item[2]):
                    removed.append(item[2])
                continue
            keep.append(item)
    else:
        keep = list(entries)
    # Пределы по количеству и объёму: режем с самого старого конца.
    total = sum(item[1] for item in keep)
    while keep and (len(keep) > max_files or total > max_bytes):
        oldest = keep.pop(0)
        if _remove(folder, oldest[2]):
            removed.append(oldest[2])
        total -= oldest[1]
    if removed:
        logger.info("MCP: удалено вложений: %d (срок %.0f дн., не больше %d файлов)",
                    len(removed), ttl_days, max_files)
    return removed


def _remove(folder: str, name: str) -> bool:
    try:
        os.remove(os.path.join(folder, name))
        return True
    except OSError as exc:
        logger.warning("MCP: вложение %s не удалилось: %s", name, exc)
        return False


def directory() -> str:
    """Каталог вложений (создаётся при первом обращении)."""
    path = os.getenv(ATTACH_DIR_ENV, "").strip()
    if not path:
        data_dir = os.path.dirname(config.AGENT_WORKSPACE_FILE)
        path = os.path.join(data_dir, "mcp_files")
    os.makedirs(path, exist_ok=True)
    return path


def safe_name(raw: Any, mime: str = "") -> str:
    """Безопасное имя файла: без путей, только [A-Za-z0-9._-], не длиннее предела.

    Имя приходит от внешнего сервера, поэтому ему нельзя доверять: `..`,
    разделители пути и управляющие символы вырезаются. Пустое имя заменяется на
    «file» + расширение по mime — файл в чате должен быть узнаваемым.
    """
    text = os.path.basename(str(raw or "").strip().replace("\\", "/"))
    text = _UNSAFE_NAME.sub("_", text).strip("._-")
    if not text:
        ext = mimetypes.guess_extension(mime.split(";")[0].strip()) if mime else ""
        text = "file" + (ext or DEFAULT_EXT)
    return text[:NAME_CHARS]


def human_size(size: Any) -> str:
    """Размер файла для человека («12,3 КБ», «1,4 МБ»)."""
    try:
        value = float(size or 0)
    except (TypeError, ValueError):
        return ""
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if value < 1024 or unit == "ГБ":
            return (f"{int(value)} {unit}" if unit == "Б"
                    else f"{value:.1f}".replace(".", ",") + f" {unit}")
        value /= 1024
    return ""


def _file_id(digest: str, name: str) -> str:
    """Идентификатор вложения: хеш содержимого + безопасное имя."""
    return f"{digest[:16]}_{_UNSAFE_ID.sub('_', name)[:NAME_CHARS]}"


def store(data: bytes, name: Any = "", mime: str = "",
          origin: str = "") -> Optional[Dict[str, Any]]:
    """Сохраняет вложение и возвращает ссылку на него (None — сохранить нельзя).

    `origin` — откуда файл («сервер · инструмент»): попадает в карточку чата,
    чтобы пользователь видел источник. Ошибки записи не роняют запрос: они
    логируются, а инструмент возвращает пометку вместо ссылки.
    """
    if not data:
        return None
    if len(data) > MAX_BYTES:
        logger.warning("MCP: вложение %s больше предела (%d байт) — не сохраняю",
                       name, len(data))
        return None
    clean_mime = str(mime or "").split(";")[0].strip()[:100] or "application/octet-stream"
    clean_name = safe_name(name, clean_mime)
    digest = hashlib.sha256(data).hexdigest()
    file_id = _file_id(digest, clean_name)
    try:
        folder = directory()
        path = os.path.join(folder, file_id)
        # Дедупликация по содержимому: тот же файл уже сохранён — не перезаписываем.
        if not os.path.exists(path) or os.path.getsize(path) != len(data):
            with open(path, "wb") as handle:
                handle.write(data)
    except OSError as exc:
        logger.warning("MCP: вложение %s не сохранилось: %s", clean_name, exc)
        return None
    # Чистка каталога (не чаще раза в PRUNE_EVERY_S): вложения копятся сами, и
    # каталог не должен расти бесконечно. Сбой чистки не влияет на вложение.
    try:
        prune()
    except Exception:  # noqa: BLE001 — чистка не должна ломать сохранение файла
        logger.warning("MCP: чистка каталога вложений не удалась", exc_info=True)
    return {
        "id": file_id,
        "name": clean_name,
        "mime": clean_mime,
        "size": len(data),
        "size_text": human_size(len(data)),
        "sha256": digest,
        "url": f"/api/agent/files/{file_id}",
        "origin": str(origin or "")[:120],
        "previewable": str(clean_mime).startswith(_PREVIEWABLE),
    }


def from_resource(resource: Any, origin: str = "") -> Optional[Dict[str, Any]]:
    """Вложение из MCP-ресурса ({"uri", "mimeType", "blob"}). None — не файл.

    Разбираются три случая: вложенный blob (файл целиком), вложенный текст
    (`resource.text` — не файл, отдаётся как текст) и ссылка на внешний ресурс
    (`resource_link`/`uri` без содержимого — файл остаётся на сервере).
    """
    if not isinstance(resource, dict):
        return None
    blob = resource.get("blob")
    if isinstance(blob, str) and blob.strip():
        try:
            data = base64.b64decode(blob, validate=False)
        except (binascii.Error, ValueError) as exc:
            logger.warning("MCP: вложение %s не декодировалось: %s",
                           resource.get("uri"), exc)
            return None
        name = resource.get("name") or _name_from_uri(resource.get("uri"))
        return store(data, name=name, mime=str(resource.get("mimeType") or ""),
                     origin=origin)
    return None


def _name_from_uri(uri: Any) -> str:
    """Имя файла из uri ресурса (file:///tmp/report.xlsx -> report.xlsx)."""
    text = str(uri or "").strip()
    if not text:
        return ""
    return os.path.basename(text.replace("\\", "/").split("?")[0]) or ""


def name_from_uri(uri: Any) -> str:
    """Публичная обёртка: имя файла из uri (нужна и текстовому блоку ресурса)."""
    return _name_from_uri(uri)


def public(ref: Any) -> Optional[Dict[str, Any]]:
    """Вложение для модели и интерфейса: БЕЗ пути на диске и без base64.

    Модель получает ссылку и метаданные (имя, размер, адрес скачивания) — этого
    достаточно, чтобы сообщить пользователю о файле и не выдумывать путь.
    """
    if not isinstance(ref, dict):
        return None
    file_id = str(ref.get("id") or "").strip()
    if not file_id:
        return None
    return {
        "id": file_id,
        "name": str(ref.get("name") or "file")[:NAME_CHARS],
        "mime": str(ref.get("mime") or "")[:100],
        "size": int(ref.get("size") or 0),
        "size_text": str(ref.get("size_text") or human_size(ref.get("size")))[:20],
        "sha256": str(ref.get("sha256") or "")[:64],
        "url": str(ref.get("url") or f"/api/agent/files/{file_id}")[:200],
        "origin": str(ref.get("origin") or "")[:120],
        "previewable": bool(ref.get("previewable")),
    }


def normalize(raw: Any, limit: int = MAX_PER_RESULT) -> List[Dict[str, Any]]:
    """Приводит список вложений к безопасному виду (битое отбрасывается)."""
    out: List[Dict[str, Any]] = []
    for item in (raw if isinstance(raw, list) else []):
        clean = public(item)
        if clean is None or clean in out:
            continue
        out.append(clean)
        if len(out) >= limit:
            break
    return out


def resolve(file_id: Any) -> Optional[str]:
    """Путь к сохранённому вложению по его id (None — файла нет/путь небезопасен).

    Id проверяется шаблоном, поэтому «../../etc/passwd» не станет путём: каталог
    вложений — единственное место, откуда модуль отдаёт файлы.
    """
    name = str(file_id or "").strip()
    if not name or not re.fullmatch(r"[A-Za-z0-9._-]{1,120}", name):
        return None
    if name.startswith("."):
        return None
    path = os.path.join(directory(), name)
    if not os.path.isfile(path):
        return None
    return path


def summary_text(refs: Any) -> str:
    """Строка о вложениях для блока данных и карточки чата (пусто — вложений нет)."""
    items = normalize(refs)
    if not items:
        return ""
    parts = [f"{item['name']} ({item['size_text'] or human_size(item['size'])})"
             for item in items]
    return "Получены ФАЙЛЫ: " + "; ".join(parts) + "."
