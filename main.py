"""Точка входа приложения.

Здесь только собирается FastAPI-приложение из независимых модулей.
Вся логика вынесена в пакет app/: конфигурация, схемы, ИИ-слой, маршруты
и фронтенд. Этот файл однажды определяет app — запуск: uvicorn main:app.
"""

from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware

from app.routers import chat, pages

app = FastAPI(title="Чат-бот на FastAPI")

# Сжатие ответов: страница чата и сохранённый диалог — крупные JSON/HTML
# (страница ~290 КБ, диалог с журналом до ~35 КБ), и они отдаются заново при
# каждом переключении задачи. С gzip это в 5–6 раз меньше трафика.
app.add_middleware(GZipMiddleware, minimum_size=1024)

app.include_router(pages.router)
app.include_router(chat.router)