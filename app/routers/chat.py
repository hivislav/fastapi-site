"""Маршруты чата — приём сообщений и возврат ответа бота.

Классический POST /api/chat (обычный/экспертный/температура/тест моделей) и
потоковый POST /api/agent/chat для режима «AI-агент»: сервер отдаёт NDJSON,
каждая строка — событие агента (debug/bot/error), которое фронтенд выводит
в чат отдельным сообщением по мере появления.

История диалога «AI-агента» переживает перезапуск приложения: память
загружается из JSON-файла при старте процесса и сохраняется туда после
каждого обмена репликами (см. app/ai/agent_memory.py).
"""

import asyncio
import json
import logging
from typing import Any, Dict, List

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.ai import service
from app.ai.agent import Agent, AgentConfig
from app.ai.agent_memory import load_agent_memory, load_agent_usage, save_agent_memory
from app.schemas import ChatMessage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# Память диалога режима «AI-агент»: список {"role", "content"} всех реплик
# агентского диалога в рамках процесса. Сериализуется блокировкой — агент
# работает асинхронно, а память у нас одна на приложение.
# При старте процесса история восстанавливается из JSON-файла (если он есть),
# поэтому после перезапуска диалог продолжается с того же места.
_agent_memory: List[Dict[str, str]] = load_agent_memory()
# Расход токенов по запросам пользователя — по одному замеру на запрос, в том
# же порядке, что и запросы в _agent_memory (фронт рисует по ним панель
# «Токены диалога»). Тоже восстанавливается из файла при старте процесса.
_agent_usage: List[Dict[str, Any]] = load_agent_usage()
_agent_lock = asyncio.Lock()


def _usage_matches_history() -> None:
    """Синхронизирует число замеров токенов с числом запросов в истории.

    Агент обрезает память диалога (≤ 24 реплик), поэтому старые замеры тоже
    отбрасываем — панель показывает ТЕКУЩУЮ переписку.
    """
    requests = sum(1 for m in _agent_memory if m.get("role") == "user")
    while len(_agent_usage) > requests:
        _agent_usage.pop(0)


def _exchange_stored(memory: List[Dict[str, str]], user_text: str) -> bool:
    """True, если агент записал в память пару «запрос + ответ».

    Нужно, чтобы список замеров токенов не разъезжался с историей: пустой
    запрос, отказ по безопасности и прочие ранние выходы в память не пишут.
    """
    text = (user_text or "").strip()
    return (
        len(memory) >= 2
        and memory[-2].get("role") == "user"
        and memory[-2].get("content") == text
        and memory[-1].get("role") == "assistant"
    )


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
    # Формат ответа агенту не передаём: системных промптов у него нет,
    # запрос уходит в модель как есть (см. app/ai/agent.py).
    agent = Agent(AgentConfig(
        max_tokens=msg.max_tokens,
        stop=msg.stop,
    ))

    def encode(event: dict) -> str:
        return json.dumps(event, ensure_ascii=False) + "\n"

    async def event_stream():
        try:
            async with _agent_lock:
                usage: Dict[str, Any] = {}
                async for event in agent.stream_generate(msg.content, history=_agent_memory):
                    # Событие "done" несёт расход токенов текущего запроса.
                    if event.get("type") == "done" and isinstance(event.get("usage"), dict):
                        usage = event["usage"]
                    yield encode(event)
                # Обработка завершена — запоминаем расход токенов запроса и
                # сохраняем историю диалога в файл, чтобы она (вместе с
                # панелью токенов) пережила перезапуск приложения.
                memory = list(agent.memory)
                _agent_memory[:] = memory
                if usage and _exchange_stored(memory, msg.content):
                    _agent_usage.append(dict(usage))
                    _usage_matches_history()
                try:
                    await asyncio.to_thread(
                        save_agent_memory, memory, list(_agent_usage)
                    )
                except Exception:  # noqa: BLE001 — сбой сохранения не должен рвать диалог
                    logger.warning("Не удалось сохранить историю диалога AI-агента", exc_info=True)
        except asyncio.CancelledError:
            raise  # клиент отключился — просто останавливаем поток
        except Exception as exc:  # noqa: BLE001
            logger.exception("Режим AI-агента: ошибка обработки запроса")
            yield encode({"type": "error", "text": "Внутренняя ошибка агента. Попробуйте ещё раз."})

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@router.get("/agent/history")
async def agent_history() -> dict:
    """Сохранённая история диалога режима «AI-агент».

    Отдаёт {"messages": [{"role": "user"|"assistant", "content": "..."}, ...],
    "usage": [{"requests", "input", "output", "limit", "overflow"}, ...]} —
    сообщения агент использует как память диалога, usage — расход токенов по
    запросам пользователя (панель «Токены диалога»). Фронтенд вызывает этот
    маршрут при включении режима агента, чтобы нарисовать старые реплики в
    окне чата, в «Истории введённых данных» и в панели токенов — диалог
    выглядит так, будто агент не выключался.
    """
    async with _agent_lock:
        return {"messages": list(_agent_memory), "usage": list(_agent_usage)}


@router.delete("/agent/history")
async def agent_history_clear() -> dict:
    """Очищает историю диалога режима «AI-агент».

    Стирает память диалога и замеры токенов в процессе и сохраняет пустую
    историю в JSON-файл, чтобы после перезапуска приложения диалог не
    восстановился. Вызывается по кнопке «Очистить историю» рядом с
    «Отправить» в режиме агента.
    """
    async with _agent_lock:
        _agent_memory.clear()
        _agent_usage.clear()
        try:
            await asyncio.to_thread(save_agent_memory, [], [])
        except Exception:  # noqa: BLE001 — сбой записи не должен рвать запрос
            logger.warning("Не удалось сохранить очищенную историю AI-агента", exc_info=True)
    return {"ok": True}
