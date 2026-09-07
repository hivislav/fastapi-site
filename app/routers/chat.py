"""Маршруты чата — приём сообщений и возврат ответа бота.

Классический POST /api/chat (обычный/экспертный/температура/тест моделей) и
потоковый POST /api/agent/chat для режима «AI-агент»: сервер отдаёт NDJSON,
каждая строка — событие агента (debug/bot/error), которое фронтенд выводит
в чат отдельным сообщением по мере появления.
"""

import asyncio
import json
import logging
from typing import Dict, List

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.ai import service
from app.ai.agent import Agent, AgentConfig
from app.schemas import ChatMessage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# Память диалога режима «AI-агент»: список {"role", "content"} всех реплик
# агентского диалога в рамках процесса. Сериализуется блокировкой — агент
# работает асинхронно, а память у нас одна на приложение.
_agent_memory: List[Dict[str, str]] = []
_agent_lock = asyncio.Lock()


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
        msg.temperatures,
        msg.models,
    )
    result = {"user": msg.content, "bot": answer, "correct": correct}
    # Настройка «Тест моделей»: ответы каждой модели по отдельности.
    if isinstance(answer, dict) and "model_responses" in answer:
        model_responses = answer["model_responses"]
        result["model_responses"] = model_responses
        result["bot"] = "\n".join(
            f"Ответ модели {r['model']}: {r['text']}" for r in model_responses
        ) if model_responses else "Пожалуйста, введите сообщение."
        # Аналитика судьи-аналитика (время, токены, стоимость) — отдельным полем.
        if answer.get("analytics"):
            result["analytics"] = answer["analytics"]
        return result
    # Настройка «Температура»: несколько независимых ответов + резюме судьи.
    # Фронтенд выводит каждый ответ с пометкой «Ответ при значении temperature …»,
    # а затем — резюме судьи-аналитика.
    if isinstance(answer, dict) and "responses" in answer:
        responses = answer["responses"]
        result["responses"] = responses
        result["bot"] = "\n".join(
            f"Ответ при значении temperature {r['temperature']}: {r['text']}"
            for r in responses
        ) if responses else "Пожалуйста, введите сообщение."
        if answer.get("judge"):
            result["judge"] = answer["judge"]
    return result


@router.post("/agent/chat")
async def agent_chat(msg: ChatMessage) -> StreamingResponse:
    """Режим «AI-агент»: потоковый ответ NDJSON.

    Каждая строка ответа — JSON-событие агента:
      {"type": "debug", "text": "чем агент сейчас занят"} — отдельное
        сообщение в чате (выводится по мере появления);
      {"type": "bot", "text": "..."} — финальный ответ;
      {"type": "error", "text": "..."} — понятное сообщение об ошибке;
      {"type": "done"} — служебный конец потока.

    Агент берёт из сообщения только формат/длину/завершение (настройки
    эксперта, температуры и теста моделей в этом режиме не участвуют),
    а историю диалога — из памяти приложения (ведётся самим агентом).
    """
    agent = Agent(AgentConfig(
        response_format=msg.format,
        max_tokens=msg.max_tokens,
        stop=msg.stop,
    ))

    def encode(event: dict) -> str:
        return json.dumps(event, ensure_ascii=False) + "\n"

    async def event_stream():
        try:
            async with _agent_lock:
                async for event in agent.stream_generate(msg.content, history=_agent_memory):
                    yield encode(event)
                # Обработка завершена — сохраняем обновлённую память диалога.
                _agent_memory[:] = list(agent.memory)
        except asyncio.CancelledError:
            raise  # клиент отключился — просто останавливаем поток
        except Exception as exc:  # noqa: BLE001
            logger.exception("Режим AI-агента: ошибка обработки запроса")
            yield encode({"type": "error", "text": "Внутренняя ошибка агента. Попробуйте ещё раз."})

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
