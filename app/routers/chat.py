"""Маршруты чата — приём сообщений и возврат ответа бота."""

from fastapi import APIRouter

from app.ai import service
from app.schemas import ChatMessage

router = APIRouter(prefix="/api")


@router.post("/chat")
def chat(msg: ChatMessage) -> dict:
    """Принимает сообщение пользователя и возвращает ответ бота."""
    if not msg.content.strip():
        return {"user": msg.content, "bot": "Пожалуйста, введите сообщение."}
    answer, correct = service.generate_response(
        msg.content,
        msg.format,
        msg.max_tokens,
        msg.stop,
        msg.expert_mode,
        msg.expert_mode_type,
        msg.expert_roles,
    )
    return {"user": msg.content, "bot": answer, "correct": correct}