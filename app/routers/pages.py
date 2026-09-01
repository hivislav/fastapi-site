"""Маршруты, отдающие пользовательский интерфейс и служебные endpoint'ы."""

from fastapi import APIRouter
from fastapi.responses import FileResponse

from app.web import frontend

router = APIRouter()


@router.get("/", response_class=FileResponse)
def home() -> str:
    """Главная страница — интерфейс чата."""
    return frontend.CHAT_HTML_PATH


@router.get("/health")
def health() -> dict:
    """Простой endpoint для проверки состояния сервиса."""
    return {"status": "ok"}