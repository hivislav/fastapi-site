"""Конфигурация приложения.

Централизованно загружает настройки из переменных окружения (.env) и
предоставляет их остальным модулям. Здесь нет бизнес-логики и зависимостей
от FastAPI, поэтому модуль можно переиспользовать где угодно.
"""

import os

# Корень проекта: каталог, где лежат main.py, .env, data/ и т.п.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Простая загрузка .env (без внешней зависимости): читает KEY=VALUE строки
# из файла .env рядом с корнем проекта, не перезаписывая уже заданные
# переменные окружения.
# ---------------------------------------------------------------------------
def load_dotenv() -> None:
    env_path = os.path.join(PROJECT_ROOT, ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_dotenv()

# ---------------------------------------------------------------------------
# Настройки реальной LLM (Yandex Cloud / DeepSeek).
# Значения берутся из переменных окружения (.env) и могут быть переопределены.
# ---------------------------------------------------------------------------
LLM_BASE_URL = os.getenv(
    "LLM_BASE_URL", "https://ai.api.cloud.yandex.net/v1"
)
LLM_MODEL = os.getenv(
    "LLM_MODEL", "gpt://b1gkm5u908if6dc0focb/deepseek-v4-flash/latest"
)
LLM_API_KEY = os.getenv("YANDEX_API_KEY", "")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2000"))

# Сколько ДОПОЛНИТЕЛЬНЫХ попыток делает клиент модели при сбое (429/5xx/сеть).
# Один 429 на шаге раньше уничтожал всю задачу, хотя токены предыдущих шагов
# уже оплачены, поэтому по умолчанию две дополнительные попытки.
LLM_RETRIES = int(os.getenv("LLM_RETRIES", "2"))

# Повторять ли вызов, оборвавшийся по таймауту (0 — нет). Провайдер мог уже
# начать генерировать ответ, поэтому такой повтор дороже: по умолчанию выключен.
LLM_RETRY_TIMEOUT = int(os.getenv("LLM_RETRY_TIMEOUT", "0"))

# Читать ответ модели потоком (SSE). Снимает «общий» таймаут на весь ответ, но
# требует от провайдера usage в потоке (stream_options.include_usage): без него
# расход токенов неизвестен, а панель статистики считает именно его.
LLM_STREAM = int(os.getenv("LLM_STREAM", "0"))

# Тарифы моделей, руб. за 1000 токенов ({"input", "output"}). Нужны, чтобы
# показать СТОИМОСТЬ запросов: и в таблице аналитики («Тест моделей»,
# «Температура»), и в панели токенов агента. Цены — оценка по прайсу провайдера:
# при смене тарифов правится только этот словарь.
MODEL_PRICING = {
    "deepseek": {"input": 0.3, "output": 0.5},
    "alice": {"input": 0.5, "output": 1.2},
    "alice-flash": {"input": 0.1, "output": 0.2},
}


def model_price(model: str) -> dict:
    """Тариф модели по её идентификатору (неизвестная — нули, без выдумок)."""
    key = str(model or "").lower()
    for name, price in MODEL_PRICING.items():
        if name in key:
            return price
    return {"input": 0, "output": 0}


# Предел длины ответа ШАГА агента, если пользователь не задал «Длину» (0 — нет
# предела: действует ограничение провайдера, как было по умолчанию). Длинный
# ответ шага оплачивается ещё раз — он попадает в контекст следующих шагов.
LLM_AGENT_MAX_TOKENS = int(os.getenv("LLM_AGENT_MAX_TOKENS", "0"))

# Файл, в котором хранится история диалога режима «AI-агент» (JSON).
# По умолчанию — data/agent_memory.json в корне проекта; путь можно
# переопределить переменной окружения AGENT_MEMORY_FILE (например, в тестах).
# Файл остаётся только для обратной совместимости: при первом запуске с
# задачами (workspace) прежняя единая история переносится в задачу «Задача 1»
# (см. app/ai/workspace.py, _migrate_legacy).
AGENT_MEMORY_FILE = os.getenv(
    "AGENT_MEMORY_FILE",
    os.path.join(PROJECT_ROOT, "data", "agent_memory.json"),
)

# Файл «рабочего пространства» режима «AI-агент» (JSON): задачи и их сессии
# (диалоги). У каждой сессии своё состояние агентского диалога — память,
# замеры токенов, резюме, факты, ветви плана. По умолчанию —
# data/agent_workspace.json; путь переопределяется env AGENT_WORKSPACE_FILE.
AGENT_WORKSPACE_FILE = os.getenv(
    "AGENT_WORKSPACE_FILE",
    os.path.join(PROJECT_ROOT, "data", "agent_workspace.json"),
)

# Файл профилей пользователя (JSON): сведения о пользователе (имя, род
# деятельности, стиль общения, формат ответа, ограничения), которые уходят в
# системный промпт сессии режима «AI-агент». Профилей может быть несколько —
# пользователь создаёт, удаляет и переключает их в меню профиля. Пока профиля
# нет, он создаётся автоматически с идентификатором user_<цифры>. По умолчанию —
# data/profiles.json; путь переопределяется env AGENT_PROFILES_FILE.
AGENT_PROFILES_FILE = os.getenv(
    "AGENT_PROFILES_FILE",
    os.path.join(PROJECT_ROOT, "data", "profiles.json"),
)