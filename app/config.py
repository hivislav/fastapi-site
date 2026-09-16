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