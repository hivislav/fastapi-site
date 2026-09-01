"""Pydantic-схемы запросов и ответов API."""

from typing import Literal, Optional

from pydantic import BaseModel, Field

# Доступные форматы ответа бота.
# "free" — ИИ отвечает свободным текстом; "json" — ответ представляется в JSON.
ResponseFormat = Literal["free", "json"]


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