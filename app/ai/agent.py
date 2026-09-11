"""AI-агент: самостоятельная сущность поверх LLM.

Агент инкапсулирует ВСЮ логику взаимодействия с моделью: конфигурацию,
историю диалога, параметры генерации, обработку ошибок и самонаблюдение
(debug-события). Внешний мир (веб-приложение) видит только методы
generate()/stream_generate() и update_config() — внутри агент сам собирает
запрос к API и решает, как обрабатывать ответ.

Модуль асинхронный (async/await): HTTP-вызов модели выполняется в отдельном
потоке через asyncio.to_thread, поэтому агент безопасно использовать в
FastAPI/Flask без блокировки event loop.

Режим «суммаризация» (чекбокс в панели «Токены диалога»): память диалога
делится на две части — последние N сообщений хранятся «как есть», а всё, что
старше, сжимается в резюме (по одному резюме на каждые N сообщений). В запрос
к модели уходит: текущий запрос + последние сообщения «как есть» + резюме
прежней переписки отдельным блоком. N (поле «summary») агент получает из
интерфейса.
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional

from app import config
from app.ai import client, demo

logger = logging.getLogger(__name__)

# Жёсткий фильтр безопасности: прямые просьбы создать опасный контент.
# Срабатывает только на явные инструкции — учебные вопросы не блокируются.
_SAFETY_PATTERNS = (
    re.compile(r"взломай|взломать|укради|похить|вымогай", re.IGNORECASE),
    re.compile(
        r"(?:напиши|создай|сделай|дай|нужен|нужна|составь)\s+"
        r"(?:вирус|троян|бомбу|взрывчатк|наркотик|фишинг|вредоносн|эксплойт)",
        re.IGNORECASE,
    ),
    re.compile(
        r"как\s+(?:сделать|создать)\s+(?:вирус|бомбу|взрывчатк|наркотик|отмыть)",
        re.IGNORECASE,
    ),
)


# Вежливый отказ при опасном запросе (единый текст для локального фильтра
# и вердикта LLM).
SAFETY_REFUSAL = (
    "Я не могу помочь с этим запросом: он выглядит как инструкция по созданию "
    "потенциально опасного контента. Если нужна учебная информация по теме "
    "безопасности — переформулируйте, пожалуйста, без просьбы что-то создать."
)


# --- Суммаризация памяти (режим «суммаризация», чекбокс в панели) ----------
# Сколько последних сообщений хранить «как есть», если пользователь не задал
# своё значение в поле «summary» (один запрос юзера = 2 сообщения).
DEFAULT_SUMMARY_SIZE = 10

# Единственное место, где агенту нужен промпт: служебный вызов сжатия истории.
# Результат — не ответ пользователю, а данные для следующего запроса.
SUMMARIZE_PROMPT = (
    "Ты сжимаешь фрагмент истории диалога в краткое резюме. Сохрани только "
    "ключевую информацию: о чём спрашивал пользователь, что отвечала модель, "
    "какие факты, имена, числа, решения и договорённости важны для продолжения "
    "разговора. Ничего не выдумывай и не продолжай диалог. Без вступлений, "
    "оценок и обращений — только суть, не более 10 предложений."
)

# Заголовок блока резюме в контексте (не инструкция, а данные: сжатая часть
# той же переписки, поэтому роль system).
SUMMARY_HEADER = "Резюме предыдущей части этого же диалога (по частям, от старых к новым):"


# ---------------------------------------------------------------------------
# События агента (тип -> смысл):
#   debug — агент сообщает, чем сейчас занят (выводится в чат отдельным
#           сообщением, пользователю его можно показывать);
#   bot   — финальный ответ пользователю;
#   error — понятное сообщение об ошибке (без стек-трейса);
#   done  — служебный маркер конца обработки.
# ---------------------------------------------------------------------------
class AgentConfig:
    """Конфигурация агента.

    Содержит настройки, которые пользователь задал в интерфейсе (format,
    max_tokens, stop, ...) и которые агент получил от веб-приложения.
    Поля со значением None означают «пользователь ничего не задал» — тогда
    параметр в запрос к API не отправляется вовсе (никаких ограничений по
    умолчанию: ни temperature, ни max_tokens, ни stop).

    Отдельная группа полей — суммаризация памяти (summarization/summary_size):
    она управляет не запросом к API, а тем, какая часть памяти диалога уходит
    в контекст как есть, а какая — сжатой в резюме.
    """

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        stop: Optional[str] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        max_history_messages: int = 24,
        max_history_chars: int = 16000,
        summarization: bool = False,
        summary_size: int = DEFAULT_SUMMARY_SIZE,
        max_summaries_per_request: int = 3,
    ) -> None:
        # Ограничение длины ответа (токены); None — в API не отправляется.
        self.max_tokens = max_tokens
        # Stop-последовательности пользователя (строка через запятую).
        self.stop = stop
        # Явная температура пользователя; None — в API не отправляется.
        self.temperature = temperature
        # URI модели; None — модель по умолчанию из конфигурации приложения.
        self.model = model
        # Ограничения памяти диалога (обрезание истории).
        self.max_history_messages = max_history_messages
        self.max_history_chars = max_history_chars
        # Суммаризация памяти (чекбокс «суммаризация» в панели режима агента).
        self.summarization = summarization
        # N — сколько последних сообщений держать в контексте «как есть»
        # (поле «summary» в интерфейсе); всё, что старше, уходит в резюме.
        self.summary_size = summary_size
        # Сколько резюме разрешено составить за один запрос пользователя —
        # предохранитель от лавины вызовов LLM (например, если пользователь
        # резко уменьшил N при длинной истории). Лимит общий на сжатие до и
        # после ответа, счётчик — Agent._summaries_this_request.
        self.max_summaries_per_request = max_summaries_per_request

    def as_dict(self) -> Dict[str, Any]:
        """Все поля конфигурации словарём (для логов)."""
        return {
            "max_tokens": self.max_tokens,
            "stop": self.stop,
            "temperature": self.temperature,
            "model": self.model,
            "max_history_messages": self.max_history_messages,
            "max_history_chars": self.max_history_chars,
            "summarization": self.summarization,
            "summary_size": self.summary_size,
            "max_summaries_per_request": self.max_summaries_per_request,
        }


@dataclass
class AgentResult:
    """Результат работы агента (не потоковый режим)."""

    text: str  # финальный ответ пользователю
    params: Dict[str, Any]  # фактически использованные параметры генерации
    debug: List[Dict[str, Any]]  # список debug-событий обработки


Step = Dict[str, Any]
EmitFn = Callable[[Step], Awaitable[None]]


class Agent:
    """Самостоятельный AI-агент.

    Инкапсулирует всю работу с LLM: сборку запроса (текст пользователя +
    история диалога, без системного промпта), параметры генерации, вызов API,
    обработку ошибок.
    Внешний код использует generate() / stream_generate() и update_config().

    Потокобезопасность: экземпляр не рассчитан на одновременные вызовы
    generate(); веб-приложение должно создавать агента на запрос (или
    сериализовать обращения к одному экземпляру блокировкой).
    """

    def __init__(
        self,
        agent_config: Optional[AgentConfig] = None,
        name: str = "AI-агент",
    ) -> None:
        self.config = agent_config or AgentConfig()
        self.name = name
        # Память диалога: [{"role": "user"|"assistant", "content": str}, ...].
        # Если в generate() передана внешняя история — она авторитетна и
        # заменяет внутреннюю (веб-слой хранит диалог сам).
        self.memory: List[Dict[str, str]] = []
        # Резюме старой части диалога (режим «суммаризация»): список строк, по
        # одному резюме на каждые N сообщений, от старых к новым. Хранится
        # ОТДЕЛЬНО от memory и так же может прийти извне (веб-слой сохраняет
        # его в JSON-файл рядом с историей).
        self.summary: List[str] = []
        # Последние использованные параметры генерации (для generate()).
        self.last_params: Dict[str, Any] = {}
        # Сколько резюме составлено в ТЕКУЩЕМ запросе (обнуляется в начале
        # каждого запроса): общий предохранитель на сжатие памяти до и после
        # ответа, чтобы один запрос не породил лавину вызовов LLM.
        self._summaries_this_request = 0
        # Расход токенов ТЕКУЩЕГО запроса пользователя: вход/выход вызова LLM,
        # выставленный лимит и признак его превышения по входящим токенам.
        # Уезжает на фронт в событии "done" (панель «Токены диалога»).
        self.last_usage: Dict[str, Any] = self._new_usage()

    # ------------------------------------------------------------------
    # Конфигурация
    # ------------------------------------------------------------------
    def update_config(self, changes: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """Обновляет конфигурацию агента.

        Примеры: agent.update_config({"temperature": 0.9})
                 agent.update_config(max_tokens=500)
        """
        merged: Dict[str, Any] = dict(changes or {})
        merged.update(kwargs)
        for key, value in merged.items():
            if not hasattr(self.config, key):
                logger.warning("Агент %s: неизвестный параметр конфигурации %r", self.name, key)
                continue
            setattr(self.config, key, value)
            logger.info("Агент %s: конфигурация обновлена: %s = %r", self.name, key, value)

    # ------------------------------------------------------------------
    # Публичное API
    # ------------------------------------------------------------------
    async def generate(
        self,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
        summary: Optional[List[str]] = None,
    ) -> AgentResult:
        """Обрабатывает сообщение и возвращает результат целиком.

        history — внешняя история диалога (список {"role", "content"}).
        None — агент использует свою внутреннюю память (накопленную ранее).
        summary — внешние резюме прежней переписки (режим «суммаризация»);
        None — агент использует свои накопленные (self.summary).

        Собирает все debug-события в result.debug — пригодится, когда
        показывать их по одному не нужно (например, в API-ответах).
        """
        steps: List[Step] = []

        async def sink(event: Step) -> None:
            steps.append(event)

        await self._process(user_message, history, sink, summary)
        # Финальный текст — последний ответ бота; сообщение об ошибке берём
        # только если готового ответа в потоке не было (например, сбой LLM).
        final = next((e["text"] for e in reversed(steps) if e["type"] == "bot"), "")
        if not final:
            final = next((e["text"] for e in reversed(steps) if e["type"] == "error"), "")
        return AgentResult(
            text=final,
            params=dict(self.last_params),
            debug=steps,
        )

    async def stream_generate(
        self,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
        summary: Optional[List[str]] = None,
    ) -> AsyncIterator[Step]:
        """То же, что generate(), но отдаёт события по мере их возникновения.

        Каждое событие (dict) становится доступно сразу, как агент его
        создал — в том числе ДО вызова LLM (показываем пользователю, чем
        агент занят в реальном времени). Поток всегда заканчивается
        событием {"type": "done"} (в нём же — расход токенов запроса);
        события "error" (в т.ч. предупреждение о переполнении лимита
        токенов) поток НЕ прерывают.
        """
        queue: asyncio.Queue = asyncio.Queue()

        async def sink(event: Step) -> None:
            queue.put_nowait(event)

        runner = asyncio.create_task(self._process(user_message, history, sink, summary))
        try:
            while True:
                event = await queue.get()
                yield event
                # Поток завершает ТОЛЬКО "done" (его всегда шлёт finally в
                # _process): ошибка о переполнении лимита токенов приходит
                # перед финальным ответом и не должна обрывать поток.
                if event["type"] == "done":
                    break
        finally:
            if not runner.done():
                runner.cancel()
            await asyncio.gather(runner, return_exceptions=True)

    # ------------------------------------------------------------------
    # Внутренняя обработка
    # ------------------------------------------------------------------
    async def _process(
        self,
        user_message: str,
        history: Optional[List[Dict[str, str]]],
        emit: EmitFn,
        summary: Optional[List[str]] = None,
    ) -> None:
        """Полный цикл обработки одного сообщения (см. docstring класса)."""
        started = time.perf_counter()
        # Счётчик токенов обнуляем на каждый запрос пользователя.
        self.last_usage = self._new_usage()
        # Бюджет резюме тоже на запрос (см. _compact_memory).
        self._summaries_this_request = 0
        try:
            # Внешняя история авторитетна; без неё — продолжаем внутреннюю.
            if history is not None:
                self.memory = self._normalize_history(history)
            # То же для резюме прежней переписки (режим «суммаризация»).
            if summary is not None:
                self.summary = self._normalize_summary(summary)
            text = (user_message or "").strip()

            await emit(self._step("debug", f"{self.name}: принял сообщение ({len(text)} симв.) — запускаю обработку."))

            # Пустой запрос — уточняем, LLM не трогаем.
            if not text:
                await emit(self._step("debug", f"{self.name}: сообщение пустое — запрашиваю уточнение, LLM не вызываю."))
                await emit(self._step("bot", "Пожалуйста, введите сообщение."))
                return

            # 1. Быстрая локальная проверка безопасности — дешёвый предфильтр
            #    до вызова LLM (сам запрос уходит в модель как есть).
            if self._is_unsafe(text):
                await emit(self._step(
                    "debug",
                    f"{self.name}: проверка безопасности (локальный фильтр) — запрос похож "
                    "на инструкцию по созданию опасного контента, отклоняю.",
                ))
                await emit(self._step("bot", SAFETY_REFUSAL))
                return

            # 2. Параметры генерации: берём ТОЛЬКО то, что пользователь задал
            #    в интерфейсе; незаданные параметры в API не отправляются.
            params, notes = self._merge_params()
            self.last_params = params
            # Лимит исходящих токенов, который реально уйдёт провайдеру:
            # по нему же определяем переполнение (см. ниже). None — без лимита.
            self.last_usage["limit"] = params.get("max_tokens")
            for note in notes:
                await emit(self._step("debug", f"{self.name}: {note}"))

            # 3. Суммаризация памяти (если включён чекбокс): всё, что старше
            #    последних N сообщений, сжимаем в резюме ДО сборки контекста —
            #    в модель уйдут только свежие сообщения «как есть». Обычно тут
            #    работы нет: память уже сжата в конце прошлого запроса, но так
            #    учитывается и смена N пользователем по ходу диалога.
            if self.config.summarization:
                await self._compact_memory(emit)

            # 4. Память диалога: сколько контекста уже накоплено.
            history_len = len(self.memory)
            context_chars = sum(len(m["content"]) for m in self.memory)
            await emit(self._step(
                "debug",
                f"{self.name}: история диалога: {history_len} реплик "
                f"(≈{context_chars // 4} токенов эвристически) — добавляю текущий запрос."
                + (
                    f" Плюс резюме прежней переписки: {len(self.summary)} част."
                    if self.summary else ""
                ),
            ))

            # 5. Вызов LLM: запрос уходит в модель КАК ЕСТЬ (без системного
            #    промпта и формата) + последние реплики диалога + резюме
            #    прежней переписки, если суммаризация их уже составила. При
            #    включённой суммаризации вызовов два: сжатие старой части
            #    памяти (см. _compact_memory) и сам ответ. Вызов идёт в
            #    отдельном потоке и event loop не блокирует.
            messages: List[Dict[str, str]] = []
            if self.summary:
                messages.append({"role": "system", "content": self._summary_block()})
            messages.extend(self.memory)          # свежие реплики «как есть»
            messages.append({"role": "user", "content": text})  # текущий запрос

            short_model = (params["model"] or config.LLM_MODEL).split("/")[-1]
            await emit(self._step(
                "debug",
                f"{self.name}: отправляю запрос в LLM как есть (модель {short_model}, thinking выключен) — жду…",
            ))
            content, metrics, elapsed = await self._call_model(text, messages, params)
            self._track_usage(metrics)
            # Превышение лимита токенов: в модель ушло больше токенов, чем
            # задано настройкой «Длина» (считаем ВХОДЯЩИЕ — запрос + история).
            overflow = self._is_limit_exceeded(metrics, params.get("max_tokens"))
            self.last_usage["overflow"] = overflow
            if content:
                await emit(self._step(
                    "debug",
                    f"{self.name}: ответ получен за {elapsed:.1f} с"
                    + (f", токены: вход {metrics['prompt_tokens']} / выход {metrics['completion_tokens']}."
                       if metrics else "."),
                ))
            else:
                await emit(self._step(
                    "debug",
                    f"{self.name}: модель вернула пустой ответ ({elapsed:.1f} с) — включаю запасной сценарий.",
                ))

            # 6. Пустой ответ (сбой/нет ключа) — понятное сообщение вместо
            #    технического стека или пустоты.
            if not content:
                content = self._fallback_text(text)
                await emit(self._step(
                    "debug",
                    f"{self.name}: "
                    + ("API-ключ не задан — отвечаю по демо-правилам."
                       if not config.LLM_API_KEY else
                       "LLM не ответила — возвращаю пользователю понятное сообщение."),
                ))

            # 7. Запоминаем ход диалога (содержательный обмен).
            self.memory.append({"role": "user", "content": text})
            self.memory.append({"role": "assistant", "content": content})
            self._trim_memory()

            await emit(self._step("bot", content))

            # 8. Лимит токенов превышен — отдельным сообщением с ошибкой
            #    (сам ответ пользователь уже получил выше).
            if overflow and metrics:
                await emit(self._step(
                    "error",
                    self._limit_warning(
                        int(params["max_tokens"]),
                        int(metrics.get("prompt_tokens") or 0),
                    ),
                ))

            # 9. Суммаризация: после обмена репликами памяти стало больше N
            #    сообщений — самую старую часть сжимаем в резюме. Делаем это
            #    ПОСЛЕ ответа (пользователь его уже видит), чтобы служебный
            #    вызов LLM не задерживал ответ.
            if self.config.summarization:
                await self._compact_memory(emit)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — внешний код видит только понятное сообщение
            logger.exception("Агент %s: ошибка обработки запроса", self.name)
            await emit(self._step("error", f"Внутренняя ошибка агента: {exc.__class__.__name__}. Попробуйте ещё раз."))
        finally:
            await emit(self._step("done", "", {
                "params": self.last_params,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "memory_len": len(self.memory),
                # Сколько частей резюме накоплено (режим «суммаризация»).
                "summary_parts": len(self.summary),
                # Расход токенов текущего запроса — фронт рисует по нему
                # панель «Токены диалога».
                "usage": dict(self.last_usage),
            }))

    # ------------------------------------------------------------------
    # Вспомогательные шаги
    # ------------------------------------------------------------------
    @staticmethod
    def _step(event_type: str, text: str, extra: Optional[Dict[str, Any]] = None) -> Step:
        """Собирает событие агента с метаданными."""
        event: Step = {"type": event_type, "text": text}
        if extra:
            event.update(extra)
        return event

    # ------------------------------------------------------------------
    # Учёт токенов (панель «Токены диалога» на фронте)
    # ------------------------------------------------------------------
    @staticmethod
    def _new_usage() -> Dict[str, Any]:
        """Пустой замер расхода токенов одного запроса пользователя.

        requests/input/output — ВСЕ вызовы LLM по этому запросу (1 — обычный
        режим, 2 и больше — когда включена суммаризация: служебное сжатие
        памяти + сам ответ), их суммарные токены (промпт и ответ);
        summary_requests/summary_input/summary_output — вклад только служебных
        вызовов сжатия: он входит в общие числа и дополнительно считается
        отдельно (строка «из них суммаризация» в панели);
        limit — выставленный пользователем лимит токенов («Длина»); overflow —
        входящие токены запроса превысили этот лимит.
        """
        return {
            "requests": 0,
            "input": 0,
            "output": 0,
            "limit": None,
            "overflow": False,
            "summary_requests": 0,
            "summary_input": 0,
            "summary_output": 0,
        }

    def _track_usage(
        self,
        metrics: Optional[Dict[str, Any]],
        summarization: bool = False,
    ) -> None:
        """Добавляет метрики вызова LLM в расход текущего запроса.

        summarization=True — вызов служебный (сжатие памяти): его токены входят
        в общий расход запроса, но дополнительно копятся в отдельных полях,
        чтобы панель могла показать вклад суммаризации отдельной строкой.
        """
        if not metrics:
            return
        prompt = int(metrics.get("prompt_tokens") or 0)
        completion = int(metrics.get("completion_tokens") or 0)
        self.last_usage["requests"] += 1
        self.last_usage["input"] += prompt
        self.last_usage["output"] += completion
        if summarization:
            self.last_usage["summary_requests"] += 1
            self.last_usage["summary_input"] += prompt
            self.last_usage["summary_output"] += completion

    @staticmethod
    def _is_limit_exceeded(
        metrics: Optional[Dict[str, Any]],
        limit: Optional[int],
    ) -> bool:
        """True, если ВХОДЯЩИЕ токены запроса превысили выставленный лимит.

        Сравниваем с лимитом (настройка «Длина») именно промпт — текст запроса
        вместе с историей диалога, то есть то, сколько токенов реально ушло в
        модель. Ответ при этом НЕ обрезается: агент только предупреждает
        пользователя сообщением об ошибке (см. _limit_warning).
        """
        if not metrics or not limit:
            return False
        return int(metrics.get("prompt_tokens") or 0) > int(limit)

    @staticmethod
    def _limit_warning(limit: int, sent: int) -> str:
        """Текст ошибки о превышении лимита токенов (sent — входящие)."""
        return (
            f"Лимит токенов превышен: лимит — {limit} токенов, "
            f"отправлено — {sent} токенов (запрос вместе с историей диалога). "
            "Сократите историю или увеличьте «Длину»."
        )

    @staticmethod
    def _normalize_history(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Приводит внешнюю историю к безопасному виду (роли, непустые строки)."""
        clean: List[Dict[str, str]] = []
        for msg in history or []:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = (msg.get("content") or "").strip()
            if role not in ("user", "assistant") or not content:
                continue
            clean.append({"role": role, "content": content})
        return clean

    def _trim_memory(self) -> None:
        """Обрезает память диалога по количеству реплик и эвристике токенов.

        В режиме «суммаризация» размер памяти ограничивает само сжатие: старое
        уходит в резюме, а не выбрасывается молча, поэтому обычная обрезка тут
        не работает. Остаётся лишь страховочный предел на случай, когда резюме
        перестали получаться (например, LLM недоступна) — иначе контекст рос бы
        безгранично.
        """
        if self.config.summarization:
            safety = max(
                self.config.max_history_messages,
                self._summary_size() * 3,
            )
            while len(self.memory) > safety:
                self.memory.pop(0)
            return
        while len(self.memory) > self.config.max_history_messages:
            self.memory.pop(0)
        while sum(len(m["content"]) for m in self.memory) > self.config.max_history_chars:
            self.memory.pop(0)
        if self.memory:
            logger.info(
                "Агент %s: память диалога обрезана до %d реплик",
                self.name,
                len(self.memory),
            )

    # ------------------------------------------------------------------
    # Суммаризация памяти (чекбокс «суммаризация» в панели режима агента)
    # ------------------------------------------------------------------
    def _summary_size(self) -> int:
        """N — сколько последних сообщений памяти держать «как есть».

        Значение приходит из поля «summary» в интерфейсе (один запрос юзера —
        это два сообщения: запрос и ответ). Некорректное/незаданное значение
        заменяется значением по умолчанию.
        """
        try:
            size = int(self.config.summary_size)
        except (TypeError, ValueError):
            size = DEFAULT_SUMMARY_SIZE
        return max(2, size)

    def _needs_compaction(self, size: int) -> bool:
        """Пора ли сжимать: сообщений больше N.

        Дополнительное условие — после сжатия в памяти должно остаться не
        меньше пары сообщений, иначе из контекста пропал бы только что
        завершённый обмен «запрос + ответ».
        """
        return len(self.memory) - size >= 2

    async def _compact_memory(self, emit: EmitFn) -> None:
        """Сжимает самую старую часть памяти в резюме (режим «суммаризация»).

        Пока сообщений больше N, берём N самых старых и просим модель сжать их
        в одно резюме (до 10 предложений), после чего убираем их из памяти —
        они остаются только в резюме (self.summary, по элементу на сжатие).

        Вызывается ДО сборки контекста (память могла разрастись после смены N)
        и ПОСЛЕ ответа пользователю (свежий обмен перевалил за N). За один
        запрос составляется не больше max_summaries_per_request резюме —
        предохранитель от лавины вызовов LLM. Если резюме получить не удалось,
        сообщения остаются в памяти как есть: терять их нельзя.
        """
        size = self._summary_size()
        if not self._needs_compaction(size):
            return
        budget = max(1, int(self.config.max_summaries_per_request or 1))
        while self._needs_compaction(size):
            if self._summaries_this_request >= budget:
                await emit(self._step(
                    "debug",
                    f"{self.name}: суммаризация: за один запрос не больше {budget} резюме "
                    "— остальное сожму при следующем запросе.",
                ))
                break
            chunk = self.memory[:size]
            summary = await self._summarize_chunk(chunk)
            if not summary:
                await emit(self._step(
                    "debug",
                    f"{self.name}: суммаризация: не удалось сжать {len(chunk)} сообщений "
                    "— оставляю их в контексте как есть.",
                ))
                break
            del self.memory[:len(chunk)]
            self.summary.append(summary)
            self._summaries_this_request += 1
            logger.info(
                "Агент %s: суммаризация — %d сообщений сжаты в резюме №%d",
                self.name, len(chunk), len(self.summary),
            )
            await emit(self._step(
                "debug",
                f"{self.name}: суммаризация: {len(chunk)} сообщений сжаты в резюме "
                f"№{len(self.summary)} — «{self._shorten(summary)}» "
                f"(в контексте осталось {len(self.memory)} реплик как есть).",
            ))

    async def _summarize_chunk(self, chunk: List[Dict[str, str]]) -> str:
        """Сжимает фрагмент памяти в резюме (служебный вызов LLM).

        Единственный вызов, где агенту нужен промпт: результат — не ответ
        пользователю, а данные для следующего запроса. Лимит «Длина» и
        «Завершение» сюда НЕ передаются: они заданы для ответа пользователю, а
        стоп-последовательность или короткий лимит обрезали бы резюме.

        Возвращает текст резюме; пустая строка — вызов не удался (тогда память
        не трогаем, см. _compact_memory).
        """
        rendered = "\n".join(
            f"{'Пользователь' if msg['role'] == 'user' else 'Ассистент'}: {msg['content']}"
            for msg in chunk
        )
        content, metrics = await client.call_llm_async(
            user_text=rendered,
            model=self.config.model or config.LLM_MODEL,
            disable_thinking=True,  # служебный вызов — reasoning не нужен
            messages=[
                {"role": "system", "content": SUMMARIZE_PROMPT},
                {"role": "user", "content": rendered},
            ],
            omit_default_max_tokens=True,  # лимит приложения сюда не подставляем
        )
        # Токены служебного вызова: входят в общий расход запроса и отдельно —
        # в поля summary_* (панель показывает вклад суммаризации строкой).
        self._track_usage(metrics, summarization=True)
        return (content or "").strip()

    def _summary_block(self) -> str:
        """Текст резюме прежней переписки для контекста запроса.

        Уходит отдельным сообщением с ролью system (это не инструкция, а
        сжатая часть того же диалога) — до последних реплик «как есть».
        """
        parts = "\n".join(
            f"{i}) {text}" for i, text in enumerate(self.summary, 1)
        )
        return f"{SUMMARY_HEADER}\n{parts}"

    @staticmethod
    def _shorten(text: str, limit: int = 200) -> str:
        """Обрезает длинный текст для debug-сообщения."""
        one_line = " ".join((text or "").split())
        return one_line if len(one_line) <= limit else one_line[:limit].rstrip() + "…"

    @staticmethod
    def _normalize_summary(summary: Optional[List[str]]) -> List[str]:
        """Приводит внешнее резюме к списку непустых строк (по порядку частей)."""
        clean: List[str] = []
        for part in summary or []:
            text = str(part or "").strip()
            if text:
                clean.append(text)
        return clean

    def _merge_params(self):
        """Параметры генерации: ТОЛЬКО то, что задал пользователь.

        Незаданные параметры остаются None и в запрос к API не попадают
        вовсе — никаких ограничений по умолчанию (ни temperature, ни
        max_tokens, ни stop): провайдер применяет свои собственные значения.

        Возвращает (params, notes): params уходят в запрос к LLM, notes —
        строки-объяснения для debug-чата.
        """
        cfg = self.config
        params: Dict[str, Any] = {}
        notes: List[str] = []

        # --- temperature: только значение пользователя ---
        params["temperature"] = cfg.temperature
        notes.append(
            f"temperature={cfg.temperature} — задана пользователем."
            if cfg.temperature is not None
            else "temperature не задана — параметр в API не отправляется."
        )

        # --- max_tokens: только значение пользователя ---
        params["max_tokens"] = cfg.max_tokens
        notes.append(
            f"лимит ответа {cfg.max_tokens} токенов — задан пользователем."
            if cfg.max_tokens is not None
            else "лимит ответа не задан — max_tokens в API не отправляется."
        )

        # --- stop: только значение пользователя ---
        params["stop"] = cfg.stop or None
        notes.append(
            f"стоп-последовательности «{cfg.stop}» — заданы пользователем."
            if cfg.stop
            else "стоп-последовательности не заданы — параметр в API не отправляется."
        )

        # --- model: всегда пользователь/конфигурация ---
        params["model"] = cfg.model or config.LLM_MODEL

        return params, notes

    @staticmethod
    def _is_unsafe(text: str) -> bool:
        """True, если запрос — прямая инструкция создать опасный контент."""
        return any(pattern.search(text) for pattern in _SAFETY_PATTERNS)

    # ------------------------------------------------------------------
    # Вызов модели
    # ------------------------------------------------------------------
    async def _call_model(
        self,
        text: str,
        messages: List[Dict[str, str]],
        params: Dict[str, Any],
    ):
        """Выполняет запрос к LLM в отдельном потоке.

        Возвращает (content, metrics|None, elapsed_seconds). Пустой content
        означает сбой/отсутствие ключа — обрабатывается выше.

        Запрос уходит как есть: без системного промпта и формата, а из
        параметров — только те, что задал пользователь (см. _merge_params).
        """
        started = time.perf_counter()
        content, metrics = await client.call_llm_async(
            user_text=text,
            max_tokens=params.get("max_tokens"),
            stop=params.get("stop"),
            temperature=params.get("temperature"),
            model=params.get("model"),
            disable_thinking=True,  # reasoning-модель отвечает в разы быстрее
            messages=messages,      # история диалога + текущий запрос
            omit_default_max_tokens=True,  # лимита нет — API его не получает
        )
        elapsed = time.perf_counter() - started
        return content, metrics, elapsed

    def _fallback_text(self, original_text: str) -> str:
        """Понятный ответ пользователю, когда LLM недоступна."""
        if not config.LLM_API_KEY:
            return demo.demo_ai(original_text)
        return "Извините, не удалось получить ответ от модели. Попробуйте ещё раз."
