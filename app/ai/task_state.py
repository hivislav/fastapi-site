"""Ядро режима «AI-агент»: конечный автомат задачи (Task State Machine).

Каждый запрос пользователя в режиме «AI-агент» проходит через конечный автомат.
Состояние задачи ОБЯЗАТЕЛЬНО содержит три поля:

    stage           — этап задачи (крупная фаза);
    current_step    — текущий шаг внутри этапа;
    expected_action — что агент ожидает сделать/получить прямо сейчас.

Базовые этапы: planning → execution → validation → done. Дополнительные
(расширения): awaiting_user (ждём ввода пользователя), failed (ошибка),
cancelled (задача отменена).

Разрешены ТОЛЬКО переходы из ALLOWED_TRANSITIONS; всё остальное (например
planning → done) запрещено — попытка бросает IllegalTransition. Каждый переход
пишется в history записью {"from", "to", "step", "at", "reason"}; этап нельзя
сменить без явного перехода, а current_step не теряется при смене этапа (в
записи истории он сохраняется всегда, а на этапе validation переходит в «check»
вместе со ссылкой на проверяемый шаг в reason/expected_action).

Модуль чистый: ни файлов, ни сети, ни LLM — только состояние и его переходы.
Хранение состояния — в dialog["state"] сессии рабочего пространства
(app/ai/workspace.py), маршруты и контроллер переходов — в app/routers/chat.py,
системный блок состояния для модели — в app/ai/agent.py (Agent.task_state).
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

logger = logging.getLogger(__name__)

# Этап задачи. planning/execution/validation/done — базовые,
# awaiting_user/failed/cancelled — допустимые расширения.
Stage = Literal["planning", "execution", "validation",
                "awaiting_user", "failed", "cancelled", "done"]

# Базовые этапы (в порядке автомата) — их рисует полоса состояния в интерфейсе.
BASE_STAGES: Tuple[str, ...] = ("planning", "execution", "validation", "done")
# Расширения: показываются отдельным блоком, когда активны.
EXTRA_STAGES: Tuple[str, ...] = ("awaiting_user", "failed", "cancelled")
STAGES: Tuple[str, ...] = BASE_STAGES + EXTRA_STAGES
# Терминальные этапы: дальше переходов нет.
TERMINAL_STAGES: Tuple[str, ...] = ("done", "cancelled")

STAGE_LABELS: Dict[str, str] = {
    "planning": "Планирование",
    "execution": "Выполнение",
    "validation": "Проверка",
    "done": "Готово",
    "awaiting_user": "Ждём пользователя",
    "failed": "Ошибка",
    "cancelled": "Отменено",
}

# Разрешённые переходы — ровно те, что заданы спецификацией автомата.
# Перескок этапов (planning → done), смена этапа «напрямую» и возвраты, которых
# здесь нет, запрещены: transition() бросит IllegalTransition.
ALLOWED_TRANSITIONS: Dict[str, Tuple[str, ...]] = {
    # план готов → работаем; нужны уточнения → ждём пользователя.
    "planning": ("execution", "awaiting_user"),
    # шаги выполнены → проверка; ошибка → failed.
    "execution": ("validation", "failed"),
    # проверка пройдена → готово; не пройдена → возврат в работу; критическая
    # ошибка → failed.
    "validation": ("done", "execution", "failed"),
    # пользователь ответил → снова планируем.
    "awaiting_user": ("planning",),
    # пользователь перезапустил → планируем заново.
    "failed": ("planning",),
    # терминальные этапы: переходов нет.
    "cancelled": (),
    "done": (),
}

# Отмена — это остановка задачи, а не переход автомата: из любого незавершённого
# этапа задача может быть отменена (этап cancelled терминальный). Держим список
# отдельно, чтобы таблица ALLOWED_TRANSITIONS осталась ровно по спецификации.
CANCELLABLE_STAGES: Tuple[str, ...] = (
    "planning", "execution", "validation", "awaiting_user", "failed",
)

# Страховочные пределы (защита контекста и файла workspace от разрастания).
MAX_HISTORY = 200       # записей истории переходов
MAX_STEPS = 20          # шагов в плане
# Сколько раз ЗАДАЧУ можно вернуть на доработку (validation → execution) без
# вмешательства пользователя. Ограничение нужно потому, что шаги выполняет
# интерфейс сам: иначе «не принял проверку → доработал → снова не принял»
# крутилось бы до предохранителя цепочки, сжигая вызовы LLM. После MAX_REDO
# неудачных проверок задача уходит в failed — решение за пользователем.
MAX_REDO = 2
STEP_LIMIT = 300        # символов в шаге плана
ACTION_LIMIT = 300      # символов в expected_action
REQUEST_LIMIT = 2000    # символов в исходном запросе задачи
REASON_LIMIT = 300      # символов в причине перехода

# Подсказки ожидаемого действия по этапам (expected_action по умолчанию).
ACTION_PLANNING = "составить план и подтвердить его у пользователя"
ACTION_AWAITING = "подтвердить план («ок») или внести правки"
ACTION_FAILED = "перезапустить задачу или изменить запрос"
ACTION_PAUSED = "пауза: нажмите «Продолжить»"


class IllegalTransition(ValueError):
    """Попытка запрещённого перехода автомата (например planning → done).

    Наследник ValueError: вызывающий код может поймать его как обычную ошибку
    значения, но по типу видно, что сломан именно автомат.
    """


# ---------------------------------------------------------------------------
# Состояние задачи
# ---------------------------------------------------------------------------
@dataclass
class TaskState:
    """Состояние задачи: где мы находимся и что делаем прямо сейчас.

    task_id         — идентификатор задачи пользователя (у нас это сессия-диалог);
    stage           — этап задачи (см. ALLOWED_TRANSITIONS);
    current_step    — текущий шаг внутри этапа («step_2», «check», «»);
    expected_action — что агент ожидает сделать/получить прямо сейчас;
    steps           — план задачи: список шагов (строки), шаг N — steps[N-1];
    step_index      — индекс текущего шага в steps (0-based);
    paused          — пользователь нажал «Пауза»: переходы не выполняются;
    autonomous      — пользователь сказал «работай автономно»: подтверждение
                      плана не требуется;
    base_stage      — последний достигнутый БАЗОВЫЙ этап (planning/execution/
                      validation/done): интерфейс подсвечивает его в полосе,
                      когда текущий этап — расширение (например awaiting_user
                      после планирования);
    redo_count      — сколько раз проверка возвращала задачу на доработку
                      (validation → execution); ограничено MAX_REDO;
    request         — исходный запрос пользователя, с которого началась задача:
                      по нему проверка результата сверяет работу (см.
                      GET-проверку в chat.py), а не по последнему сообщению;
    reason          — причина текущего состояния (последний переход);
    history         — журнал переходов {"from", "to", "step", "at", "reason"};
    created_at/updated_at — время создания и последнего изменения.
    """

    task_id: str
    stage: Stage = "planning"
    current_step: str = ""
    expected_action: str = ""
    steps: List[str] = field(default_factory=list)
    step_index: int = 0
    paused: bool = False
    autonomous: bool = False
    base_stage: str = "planning"
    redo_count: int = 0
    request: str = ""
    reason: str = ""
    history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    # ------------------------------------------------------------------
    # Шаги плана
    # ------------------------------------------------------------------
    @property
    def steps_total(self) -> int:
        """Сколько шагов в плане."""
        return len(self.steps)

    @property
    def step_number(self) -> int:
        """Номер текущего шага (1-based; 0 — шаг не выбран)."""
        if not self.steps:
            return 0
        return max(1, min(self.step_index + 1, len(self.steps)))

    def step_text(self, index: Optional[int] = None) -> str:
        """Текст шага плана (по умолчанию — текущего)."""
        if not self.steps:
            return ""
        position = self.step_index if index is None else index
        if 0 <= position < len(self.steps):
            return self.steps[position]
        return ""

    def step_label(self) -> str:
        """«шаг 2 из 4» — человекочитаемое место в плане."""
        if not self.steps:
            return "шаг не выбран"
        return f"шаг {self.step_number} из {len(self.steps)}"

    def plan_text(self) -> str:
        """План нумерованным списком; текущий шаг помечен «→»."""
        if not self.steps:
            return "(план пуст)"
        lines: List[str] = []
        for position, step in enumerate(self.steps):
            mark = "→" if (self.stage == "execution" and position == self.step_index) else " "
            lines.append(f"{mark} {position + 1}) {step}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Переходы
    # ------------------------------------------------------------------
    def can_transition(self, to: str) -> bool:
        """True, если переход разрешён таблицей ALLOWED_TRANSITIONS."""
        return str(to) in ALLOWED_TRANSITIONS.get(self.stage, ())

    def transition(
        self,
        to: str,
        reason: str,
        current_step: Optional[str] = None,
        expected_action: Optional[str] = None,
    ) -> None:
        """Единственная точка смены этапа: проверяет переход и пишет историю.

        to — новый этап; reason — причина (обязательна: без причины перехода не
        бывает); current_step/expected_action — новое содержимое (None — оставить
        прежнее, чтобы шаг не «терялся» при смене этапа).

        Запрещённый переход (в т.ч. перескок этапа) → IllegalTransition;
        состояние при этом не меняется.
        """
        target = str(to or "").strip()
        if target not in STAGES:
            raise IllegalTransition(f"Неизвестный этап: {target!r}")
        if not self.can_transition(target):
            raise IllegalTransition(
                f"Переход {self.stage} → {target} запрещён "
                f"(разрешены: {', '.join(ALLOWED_TRANSITIONS.get(self.stage) or ('—',))})"
            )
        previous = self.stage
        step = self.current_step if current_step is None else str(current_step)
        action = self.expected_action if expected_action is None else str(expected_action)
        self.stage = target  # type: ignore[assignment]
        if target in BASE_STAGES:
            # Место в базовой цепочке (planning → execution → validation → done):
            # нужно интерфейсу, когда задача уходит в расширение (awaiting_user).
            self.base_stage = target
        self.current_step = step[:STEP_LIMIT]
        self.expected_action = action[:ACTION_LIMIT]
        self._log(previous, target, step, reason)
        logger.info(
            "TaskState %s: %s → %s (шаг %r, причина: %s)",
            self.task_id or "—", previous, target, step, _short(reason),
        )

    def advance_step(self, reason: str) -> bool:
        """Переводит выполнение на следующий шаг плана (смена current_step).

        Это НЕ смена этапа (execution → execution), но изменение состояния:
        тоже пишется в history, чтобы по журналу было видно движение по плану.
        False — следующий шаг не начат (шаги кончились).
        """
        if self.step_index + 1 >= len(self.steps):
            return False
        self.step_index += 1
        self.current_step = f"step_{self.step_index + 1}"
        self.expected_action = f"выполнить: {self.step_text()}"[:ACTION_LIMIT]
        self._log(self.stage, self.stage, self.current_step, reason)
        return True

    def pause(self, reason: str) -> None:
        """Ставит задачу на паузу: этап и шаг сохраняются, ожидание — «Продолжить».

        Этап НЕ меняется (иначе потерялось бы место в автомате): предыдущее
        expected_action запоминается в reason и восстанавливается при resume().
        """
        if self.paused:
            return
        self.paused = True
        self._log(self.stage, self.stage, self.current_step, reason, paused=True)
        self.expected_action = ACTION_PAUSED

    def resume(self, reason: str) -> None:
        """Снимает паузу: работа продолжается с того же этапа и шага."""
        if not self.paused:
            return
        self.paused = False
        self._log(self.stage, self.stage, self.current_step, reason, paused=False)
        self.expected_action = default_action(self)

    # ------------------------------------------------------------------
    # Служебное
    # ------------------------------------------------------------------
    def _log(
        self,
        from_stage: str,
        to_stage: str,
        step: str,
        reason: str,
        paused: Optional[bool] = None,
    ) -> None:
        """Пишет запись в журнал переходов (формат спецификации) и метит время.

        {"from", "to", "step", "at", "reason"} + "paused" (если менялась пауза).
        """
        record: Dict[str, Any] = {
            "from": from_stage,
            "to": to_stage,
            "step": step,
            "at": datetime.utcnow().isoformat(timespec="seconds"),
            "reason": _short(reason, REASON_LIMIT),
        }
        if paused is not None:
            record["paused"] = bool(paused)
        self.history.append(record)
        if len(self.history) > MAX_HISTORY:
            self.history = self.history[-MAX_HISTORY:]
        self.reason = record["reason"]
        self.updated_at = datetime.utcnow()


# ---------------------------------------------------------------------------
# Создание, чтение и запись состояния
# ---------------------------------------------------------------------------
def default_action(state: TaskState) -> str:
    """Ожидаемое действие (expected_action) по текущему этапу и шагу."""
    if state.stage == "planning":
        return ACTION_PLANNING
    if state.stage == "awaiting_user":
        return ACTION_AWAITING
    if state.stage == "execution":
        step = state.step_text()
        return f"выполнить: {step}"[:ACTION_LIMIT] if step else "выполнить шаг плана"
    if state.stage == "validation":
        step = state.step_text()
        return (f"проверить результат шага {state.step_number}: {step}"[:ACTION_LIMIT]
                if step else "проверить результат")
    if state.stage == "failed":
        return ACTION_FAILED
    return ""  # done / cancelled — ожидать нечего


def _apply_action(state: TaskState) -> None:
    """Пересчитывает expected_action по ТЕКУЩЕМУ этапу и шагу.

    Вызывается после смены этапа: подсказка зависит от нового этапа, поэтому
    вычислять её заранее (до transition) нельзя.
    """
    state.expected_action = default_action(state)[:ACTION_LIMIT]


def clean_steps(raw: Any, limit: int = MAX_STEPS) -> List[str]:
    """Приводит план к списку непустых коротких шагов (строки)."""
    steps: List[str] = []
    for item in (raw if isinstance(raw, (list, tuple)) else []):
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("step") or "").strip()
        else:
            text = str(item or "").strip()
        text = " ".join(text.split())[:STEP_LIMIT]
        if text and text not in steps:
            steps.append(text)
        if len(steps) >= limit:
            break
    return steps


def new_state(task_id: str, steps: Optional[List[str]] = None) -> TaskState:
    """Новое состояние задачи: этап planning, история пустая.

    Создаётся для НОВОЙ задачи пользователя — это не переход, а рождение
    автомата, поэтому запрет перескоков здесь не действует.
    """
    state = TaskState(task_id=str(task_id or ""), steps=clean_steps(steps))
    state.expected_action = ACTION_PLANNING
    state.reason = "задача создана: этап planning"
    state._log("planning", "planning", "", "задача создана: этап planning")
    return state


def reset(
    task_id: str,
    previous: Optional[TaskState] = None,
    reason: str = "новый запрос пользователя — задача начата заново",
    autonomous: bool = False,
) -> TaskState:
    """Новый автомат для той же сессии (прежняя задача завершена/сброшена).

    Прежнее состояние становится историей: в журнал нового автомата попадает
    запись о сбросе с этапа, на котором задача остановилась. Автомат при этом
    рождается заново (это не переход), поэтому недопустимых «перескоков» нет.
    """
    state = new_state(task_id)
    state.autonomous = bool(autonomous)
    if previous is not None:
        state.created_at = previous.created_at
        state.history.insert(0, {
            "from": previous.stage,
            "to": "planning",
            "step": previous.current_step,
            "at": datetime.utcnow().isoformat(timespec="seconds"),
            "reason": _short(reason, REASON_LIMIT),
            "reset": True,
        })
        state.reason = _short(reason, REASON_LIMIT)
    return state


def from_dict(raw: Any, task_id: str = "") -> TaskState:
    """Восстанавливает состояние из JSON-файла (мусор отбрасывается).

    Неизвестный этап, битые даты и «чужие» поля не ломают чтение: состояние
    приводится к безопасному виду, а испорченный этап считается planning.
    """
    if not isinstance(raw, dict):
        return new_state(task_id)
    stage = str(raw.get("stage") or "").strip()
    state = TaskState(
        task_id=str(raw.get("task_id") or task_id or ""),
        stage=stage if stage in STAGES else "planning",  # type: ignore[arg-type]
        current_step=str(raw.get("current_step") or "")[:STEP_LIMIT],
        expected_action=str(raw.get("expected_action") or "")[:ACTION_LIMIT],
        steps=clean_steps(raw.get("steps")),
        paused=raw.get("paused") is True,
        autonomous=raw.get("autonomous") is True,
        reason=str(raw.get("reason") or "")[:REASON_LIMIT],
        created_at=_parse_dt(raw.get("created_at")),
        updated_at=_parse_dt(raw.get("updated_at")),
    )
    try:
        index = int(raw.get("step_index") or 0)
    except (TypeError, ValueError):
        index = 0
    state.step_index = max(0, min(index, max(0, len(state.steps) - 1)))
    state.request = str(raw.get("request") or "")[:REQUEST_LIMIT]
    try:
        state.redo_count = max(0, int(raw.get("redo_count") or 0))
    except (TypeError, ValueError):
        state.redo_count = 0
    # Место в базовой цепочке: из файла (если корректно), иначе — сам этап.
    saved_base = str(raw.get("base_stage") or "").strip()
    if saved_base in BASE_STAGES:
        state.base_stage = saved_base
    elif state.stage in BASE_STAGES:
        state.base_stage = state.stage
    state.history = _clean_history(raw.get("history"))
    if not state.expected_action:
        state.expected_action = default_action(state)
    return state


def to_dict(state: TaskState) -> Dict[str, Any]:
    """Состояние как JSON-безопасный словарь (для файла workspace и API)."""
    return {
        "task_id": state.task_id,
        "stage": state.stage,
        "current_step": state.current_step,
        "expected_action": state.expected_action,
        "steps": list(state.steps),
        "step_index": state.step_index,
        "paused": bool(state.paused),
        "autonomous": bool(state.autonomous),
        "base_stage": state.base_stage,
        "redo_count": int(state.redo_count),
        "request": state.request,
        "reason": state.reason,
        "history": [dict(item) for item in state.history],
        "created_at": state.created_at.isoformat(timespec="seconds"),
        "updated_at": state.updated_at.isoformat(timespec="seconds"),
    }


def snapshot(state: TaskState, history_tail: int = 20) -> Dict[str, Any]:
    """Снимок состояния для интерфейса (полоса этапов и кнопка «Пауза»)."""
    return {
        # Три обязательных поля состояния.
        "stage": state.stage,
        "current_step": state.current_step,
        "expected_action": state.expected_action,
        # Остальное — для отрисовки полосы, шагов плана и кнопок.
        "stage_label": STAGE_LABELS.get(state.stage, state.stage),
        # Место в базовой цепочке: у расширений (awaiting_user/failed) оно своё,
        # поэтому полоса подсвечивает base_stage, а расширение — отдельным блоком.
        "base_stage": state.base_stage,
        "base_stages": [
            {"id": name, "label": STAGE_LABELS[name], "active": name == state.base_stage}
            for name in BASE_STAGES
        ],
        "extra_stage": ({
            "id": state.stage, "label": STAGE_LABELS.get(state.stage, state.stage),
        } if state.stage in EXTRA_STAGES else None),
        "steps": [
            {"number": position + 1, "text": step,
             # Текущий шаг подсвечен на всех незавершённых этапах: на этапе
             # планирования это шаг, который начнёт выполнение после «ок».
             "active": (position == state.step_index
                        and state.stage not in TERMINAL_STAGES),
             "done": (state.stage in ("validation", "done")
                      or position < state.step_index)}
            for position, step in enumerate(state.steps)
        ],
        "step_index": state.step_index,
        "step_number": state.step_number,
        "steps_total": len(state.steps),
        "paused": bool(state.paused),
        "autonomous": bool(state.autonomous),
        # Доработки по требованию проверки: «доработка 1 из 2» в полосе этапов.
        "redo_count": int(state.redo_count),
        "max_redo": MAX_REDO,
        "request": state.request,
        "can_redo": can_redo(state),
        "reason": state.reason,
        "can_pause": state.stage not in TERMINAL_STAGES and not state.paused,
        "can_resume": bool(state.paused),
        # Отменить задачу можно из любого незавершённого этапа (кнопка
        # «Отменить задачу»): cancelled — терминальный этап.
        "can_cancel": state.stage in CANCELLABLE_STAGES,
        "can_confirm": (state.stage in ("planning", "awaiting_user")
                        and bool(state.steps) and not state.paused),
        "terminal": state.stage in TERMINAL_STAGES,
        "task_id": state.task_id,
        "updated_at": state.updated_at.isoformat(timespec="seconds"),
        "history": [dict(item) for item in state.history[-history_tail:]],
    }


# ---------------------------------------------------------------------------
# Переходы-сценарии (используются контроллером в app/routers/chat.py)
# ---------------------------------------------------------------------------
def start_planning(state: TaskState, reason: str) -> None:
    """awaiting_user | failed → planning (пользователь ответил / перезапустил)."""
    state.transition("planning", reason, current_step="step_1")
    _apply_action(state)


def await_confirmation(state: TaskState, steps: List[str], reason: str) -> None:
    """planning → awaiting_user: план показан, ждём «ок» или правок."""
    state.steps = clean_steps(steps)
    state.step_index = 0
    state.transition("awaiting_user", reason, current_step="step_1")
    _apply_action(state)


def plan_ready(state: TaskState, steps: List[str], reason: str) -> None:
    """planning → execution: план подтверждён (или режим «работай автономно»).

    Подтверждение плана обнуляет счётчик доработок: это НОВЫЙ заход по задаче.
    """
    state.steps = clean_steps(steps)
    state.step_index = 0
    state.redo_count = 0
    state.transition("execution", reason, current_step="step_1")
    _apply_action(state)


def next_step(state: TaskState, reason: str) -> bool:
    """Переводит выполнение на следующий шаг плана (False — шагов больше нет)."""
    if state.stage != "execution":
        logger.warning("TaskState %s: следующий шаг запрошен вне этапа execution (%s)",
                       state.task_id or "—", state.stage)
        return False
    return state.advance_step(reason)


def to_validation(state: TaskState, reason: str) -> None:
    """execution → validation: все шаги выполнены, проверяем результат."""
    state.transition("validation", reason, current_step="check")
    _apply_action(state)


def validation_ok(state: TaskState, reason: str) -> None:
    """validation → done: проверка пройдена (шаг и ожидание очищаются)."""
    state.transition("done", reason, current_step="", expected_action="")


def validation_failed(state: TaskState, reason: str, step_index: Optional[int] = None,
                     redo: bool = True) -> None:
    """validation → execution: проверка не пройдена, возвращаемся на шаг.

    redo=True (обычный случай) увеличивает счётчик доработок: по нему
    контроллер понимает, что автоматические повторы пора прекращать
    (см. MAX_REDO и can_redo).
    """
    if step_index is not None and state.steps:
        state.step_index = max(0, min(int(step_index), len(state.steps) - 1))
    step = f"step_{state.step_number}" if state.steps else "step_1"
    state.transition("execution", reason, current_step=step)
    if redo:
        state.redo_count += 1
    _apply_action(state)


def can_redo(state: TaskState) -> bool:
    """True, если задачу ещё можно вернуть на доработку автоматически.

    False — доработок уже было MAX_REDO: контроллер переводит задачу в failed
    и ждёт решения пользователя.
    """
    return state.redo_count < MAX_REDO


def fail(state: TaskState, reason: str) -> None:
    """execution | validation → failed: ошибка, нужно вмешательство."""
    state.transition("failed", reason)  # current_step сохраняется (шаг не теряем)
    _apply_action(state)


def cancel(state: TaskState, reason: str) -> None:
    """Отмена задачи (остановка автомата): этап cancelled — терминальный.

    Спецификация не относит отмену к переходам автомата, поэтому здесь отдельная
    проверка источника (CANCELLABLE_STAGES) вместо ALLOWED_TRANSITIONS.
    """
    if state.stage not in CANCELLABLE_STAGES:
        raise IllegalTransition(f"Задачу на этапе {state.stage} отменить нельзя")
    previous = state.stage
    state.stage = "cancelled"  # type: ignore[assignment]
    state.current_step = ""
    state.expected_action = ""
    state._log(previous, "cancelled", "", reason)


def pause(state: TaskState, reason: str) -> None:
    """Пауза: автомат останавливается на текущем этапе и шаге."""
    if state.stage in TERMINAL_STAGES:
        raise IllegalTransition(f"Задача на этапе {state.stage} уже завершена")
    state.pause(reason)


def resume(state: TaskState, reason: str) -> None:
    """Снятие паузы: работа продолжается с того же места."""
    state.resume(reason)


# ---------------------------------------------------------------------------
# План: локальный запасной вариант (когда модель не дала план)
# ---------------------------------------------------------------------------
# Маркеры списка и нумерация в запросе пользователя.
_BULLETS = "-—–•*·"
_NUMBERING = re.compile(r"^\s*\d+[.)]\s*")
_SENTENCE_SPLIT = re.compile(r"(?<=[.;!?])\s+")
MIN_STEP_CHARS = 8
FALLBACK_STEPS_LIMIT = 5


def fallback_steps(text: str, limit: int = FALLBACK_STEPS_LIMIT) -> List[str]:
    """Разбивает запрос пользователя на шаги локально, без вызова LLM.

    Запасной вариант для этапа planning (основной — служебный вызов модели,
    см. Agent.build_plan): сначала по строкам списка (нумерация/маркеры), затем
    по границам предложений; если ничего не выделилось — один шаг с самим
    запросом. Слишком короткие обрывки отбрасываются, длинные обрезаются.
    """
    raw = str(text or "")
    items: List[str] = []
    for line in raw.splitlines():
        item = line.strip().lstrip(_BULLETS).strip()
        item = _NUMBERING.sub("", item).strip()
        if item:
            items.append(item)
    if len(items) < 2:
        items = [part.strip() for part in _SENTENCE_SPLIT.split(" ".join(raw.split()))
                 if part.strip()]
    if len(items) < 2:
        items = [" ".join(raw.split())]
    steps: List[str] = []
    for item in items:
        step = " ".join(item.split())[:STEP_LIMIT]
        if len(step) < MIN_STEP_CHARS and steps:
            continue  # обрывок вроде «ок.» шагом не считаем
        if step and step not in steps:
            steps.append(step)
        if len(steps) >= limit:
            break
    return steps


# ---------------------------------------------------------------------------
# Текстовые представления: системный блок для модели и ASCII-блок для чата
# ---------------------------------------------------------------------------
STATE_BLOCK_HEADER = (
    "СОСТОЯНИЕ ЗАДАЧИ (конечный автомат: planning → execution → validation → done). "
    "Это данные о ходе работы, а не новый запрос пользователя."
)


def state_block(state: TaskState) -> str:
    """Системный блок состояния задачи для модели (уходит в контекст всегда).

    По нему агент понимает, на каком этапе задача и какой шаг выполняется
    СЕЙЧАС: план целиком, текущий шаг помечен «→».
    """
    lines = [
        STATE_BLOCK_HEADER,
        f"Этап: {STAGE_LABELS.get(state.stage, state.stage)} ({state.stage}).",
    ]
    if state.steps:
        lines.append(f"Место в плане: {state.step_label()}.")
    if state.expected_action:
        lines.append(f"Сейчас ожидается: {state.expected_action}.")
    if state.steps:
        lines.append("План задачи (текущий шаг помечен «→»):")
        lines.append(state.plan_text())
    if state.stage == "execution":
        lines.append(
            "Выполни ИМЕННО текущий шаг: ответ пользователю должен относиться к нему. "
            "На следующие шаги не перескакивай — они будут выполнены в следующих ответах."
        )
    elif state.stage == "validation":
        lines.append(
            "Идёт проверка результата: коротко подтверди, что сделано по плану, "
            "и что осталось (без выдумывания новых требований)."
        )
    if state.paused:
        lines.append("Задача на паузе: пользователь нажал «Пауза».")
    return "\n".join(lines)


# Ширина ASCII-блока состояния (он же — формат ответа ассистента).
# 76 = 1 (рамка) + 4×(13 + 1) этапов + 18 (кнопка) + 1 (рамка).
_BOX_WIDTH = 76
_STAGE_COL = 13     # ширина блока этапа
_BUTTON_COL = 18    # ширина крайнего правого блока — кнопки «Пауза/Продолжить»


def _fit(text: Any, width: int, collapse: bool = True) -> str:
    """Обрезает текст до ширины блока (переводы строк — в пробелы).

    collapse=False — пробелы сохраняются (нужно для строки со стрелками, где
    пробелы и есть разметка).
    """
    value = " ".join(str(text if text is not None else "").split()) if collapse \
        else str(text if text is not None else "")
    if len(value) <= width:
        return value.ljust(width)
    return value[:max(1, width - 1)] + "…"


def steps_word(count: int) -> str:
    """«шаг» / «шага» / «шагов» — согласование числа шагов плана."""
    if count % 10 == 1 and count % 100 != 11:
        return "шаг"
    if count % 10 in (2, 3, 4) and count % 100 not in (12, 13, 14):
        return "шага"
    return "шагов"


def ascii_block(state: TaskState) -> str:
    """ASCII-блок состояния — тот же автомат, что в UI, но текстом.

    Печатается в лог и годится для ответа ассистента: этапы соединены стрелками,
    текущий выделен «», крайний правый блок — кнопка «Пауза»/«Продолжить»
    (стрелки к ней нет).
    """
    button = "Продолжить" if state.paused else "Пауза"
    top = "┌─ TASK STATE " + "─" * (_BOX_WIDTH - len("┌─ TASK STATE ") - 1) + "┐"
    lines = [top]
    lines.append("│ " + _fit(
        f"task: {state.task_id or '—'} · этап: {STAGE_LABELS.get(state.stage, state.stage)}"
        f" ({state.stage})", _BOX_WIDTH - 4) + " │")
    lines.append("│ " + _fit(
        f"current_step: {state.current_step or '—'}"
        + (f" ({state.step_label()})" if state.steps else "")
        + f" · paused: {'да' if state.paused else 'нет'}", _BOX_WIDTH - 4) + " │")
    lines.append("│ " + _fit(
        f"expected_action: {state.expected_action or '—'}", _BOX_WIDTH - 4) + " │")

    def divider(left: str, middle: str, right: str, fill: str = "─") -> str:
        cell = fill * _STAGE_COL
        tail = fill * _BUTTON_COL
        return left + middle.join([cell] * 4 + [tail]) + right

    def cell(name: str) -> str:
        active = name == state.stage
        text = f">{name}<" if active else name
        return _fit(text, _STAGE_COL)
    lines.append(divider("├", "┬", "┤"))
    lines.append("│" + "│".join(cell(name) for name in BASE_STAGES)
                 + "│" + _fit(f"  [ {button} ]", _BUTTON_COL) + "│")
    # Стрелки между этапами; к кнопке стрелки нет.
    arrows = "─▶".join(" " * _STAGE_COL for _ in BASE_STAGES)
    lines.append("│" + _fit(" " + arrows, _BOX_WIDTH - 2, collapse=False) + "│")
    lines.append(divider("└", "┴", "┘"))
    if state.steps:
        lines.append("│ " + _fit(
            f"план: {state.steps_total} {steps_word(state.steps_total)}"
            f" · текущий: {state.step_text() or '—'}", _BOX_WIDTH - 4) + " │")
    lines.append("└" + "─" * (_BOX_WIDTH - 2) + "┘")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Вспомогательное
# ---------------------------------------------------------------------------
def _short(text: Any, limit: int = 120) -> str:
    """Однострочная короткая версия текста (для логов и журнала)."""
    return " ".join(str(text or "").split())[:limit]


def _parse_dt(value: Any) -> datetime:
    """Разбирает ISO-дату из файла; битое значение → текущее время."""
    if isinstance(value, datetime):
        return value
    text = str(value or "").strip()
    if text:
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            pass
    return datetime.utcnow()


def _clean_history(raw: Any, limit: int = MAX_HISTORY) -> List[Dict[str, Any]]:
    """Приводит журнал переходов к безопасному виду (записи спецификации)."""
    clean: List[Dict[str, Any]] = []
    for item in (raw if isinstance(raw, list) else []):
        if not isinstance(item, dict):
            continue
        record: Dict[str, Any] = {
            "from": str(item.get("from") or "")[:40],
            "to": str(item.get("to") or "")[:40],
            "step": str(item.get("step") or "")[:STEP_LIMIT],
            "at": str(item.get("at") or ""),
            "reason": str(item.get("reason") or "")[:REASON_LIMIT],
        }
        if "paused" in item:
            record["paused"] = item.get("paused") is True
        if item.get("reset"):
            record["reset"] = True
        clean.append(record)
    return clean[-limit:]
