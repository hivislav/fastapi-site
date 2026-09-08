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

# Файл, в котором хранится история диалога режима «AI-агент» (JSON).
# По умолчанию — data/agent_memory.json в корне проекта; путь можно
# переопределить переменной окружения AGENT_MEMORY_FILE (например, в тестах).
AGENT_MEMORY_FILE = os.getenv(
    "AGENT_MEMORY_FILE",
    os.path.join(PROJECT_ROOT, "data", "agent_memory.json"),
)