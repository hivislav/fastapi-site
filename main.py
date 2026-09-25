"""Точка входа приложения.

Здесь только собирается FastAPI-приложение из независимых модулей.
Вся логика вынесена в пакет app/: конфигурация, схемы, ИИ-слой, маршруты
и фронтенд. Этот файл однажды определяет app — запуск: uvicorn main:app.

Вместе с приложением поднимается ПЛАНИРОВЩИК ПЕРИОДИЧЕСКИХ ЗАДАЧ
(app/periodic_runner.py): он повторяет запросы задач с расписанием (кнопка
«Новая периодическая задача») и складывает результат в чат этих задач. Работает
в том же процессе и цикле событий, поэтому видит те же задачи, что и API;
выключается настройкой PERIODIC_ENABLED=0.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware

from app import periodic_runner
from app.routers import chat, pages


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Запуск и остановка фоновых служб приложения.

    Планировщик стартует ВМЕСТЕ с приложением (первый тик — с небольшой
    задержкой, см. PERIODIC_START_DELAY) и останавливается вместе с ним: без
    этого повтор задачи шёл бы в уже закрытом цикле событий.
    """
    periodic_runner.start()
    try:
        yield
    finally:
        await periodic_runner.stop()


app = FastAPI(title="Чат-бот на FastAPI", lifespan=lifespan)

# Сжатие ответов: страница чата и сохранённый диалог — крупные JSON/HTML
# (страница ~290 КБ, диалог с журналом до ~35 КБ), и они отдаются заново при
# каждом переключении задачи. С gzip это в 5–6 раз меньше трафика.
app.add_middleware(GZipMiddleware, minimum_size=1024)

app.include_router(pages.router)
app.include_router(chat.router)
