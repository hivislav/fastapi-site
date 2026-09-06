"""Pydantic-схемы запросов и ответов API."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

# Доступные форматы ответа бота.
# "free" — ИИ отвечает свободным текстом; "json" — ответ представляется в JSON.
ResponseFormat = Literal["free", "json"]

# Режимы экспертного поведения модели (выбираются радиокнопками в эксперте).
# "direct" — максимально сухой прямой ответ; "stepwise" — решение по шагам;
# "prompt" — составление промпта и ответ на него; "group" — группа экспертов.
ExpertModeType = Literal["direct", "stepwise", "prompt", "group"]


class ChatMessage(BaseModel):
    """Сообщение, отправляемое пользователем в чат."""

    content: str
    format: ResponseFormat = "free"
    # Ограничение длины ответа в токенах. Если не задано — используется
    # значение по умолчанию из конфигурации (LLM_MAX_TOKENS).
    max_tokens: Optional[int] = Field(default=None, ge=1)
    # Условие завершения: одна или несколько (через запятую) строк, на которых
    # LLM остановит генерацию (stop-последовательности).
    stop: Optional[str] = Field(default=None, max_length=200)
    # Экспертный режим: когда включён, применяется только выбранный
    # экспертный_mode, а format/max_tokens/stop не учитываются.
    expert_mode: bool = False
    expert_mode_type: ExpertModeType = "direct"
    # Роли экспертов для режима "group". Пустой список — ошибка.
    expert_roles: Optional[List[str]] = Field(default=None)
    # Настройка «Температура» (обычный режим). Каждое непустое значение —
    # отдельный запрос к LLM. Пустой/None — параметр не используется
    # (выполняется один запрос как обычно). Заполненные значения — числа.
    temperatures: Optional[List[float]] = Field(default=None)