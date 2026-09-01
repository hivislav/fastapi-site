"""Точка входа приложения.

Здесь только собирается FastAPI-приложение из независимых модулей.
Вся логика вынесена в пакет app/: конфигурация, схемы, ИИ-слой, маршруты
и фронтенд. Этот файл однажды определяет app — запуск: uvicorn main:app.
"""

from fastapi import FastAPI

from app.routers import chat, pages

app = FastAPI(title="Чат-бот на FastAPI")

app.include_router(pages.router)
app.include_router(chat.router)