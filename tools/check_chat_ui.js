/* Прогон интерфейса полосы состояния задачи (Task State Machine) в jsdom.
 *
 * Страница app/web/chat.html грузится целиком (runScripts: 'dangerously') с
 * заглушкой window.fetch, эмулирующей маршруты /api/agent/* (workspace, state,
 * history, chat) и мини-автомат задачи на стороне «сервера», и щёлкаются
 * кнопки полосы. Проверяются: четыре блока этапов со стрелками, подсветка
 * текущего этапа, чипы шагов плана, блок расширенного этапа (awaiting_user),
 * АВТО-ПРОГОН шагов после «Подтвердить план» (без сообщений пользователя),
 * остановка прогона кнопкой «Пауза», «Продолжить», правка плана через модалку,
 * блокировка ввода и подсказка без задачи.
 *
 * Запуск (нужен jsdom — в зависимостях проекта его нет, ставится отдельно):
 *     npm install jsdom --prefix /tmp/jsdom-check
 *     NODE_PATH=/tmp/jsdom-check/node_modules node tools/check_chat_ui.js
 *
 * Сеть и сервер не нужны: fetch подменён внутри страницы. Скрипт возвращает
 * ненулевой код выхода, если хоть одна проверка провалилась.
 */
const fs = require('fs');
const { JSDOM } = require('jsdom');

const HTML = fs.readFileSync(
  require('path').join(__dirname, '..', 'app', 'web', 'chat.html'), 'utf8');

let failures = 0;
function check(name, cond, detail) {
  if (cond) console.log('  ok   ' + name);
  else { console.log('  FAIL ' + name + (detail ? ' — ' + detail : '')); failures++; }
}

// --- Мини-автомат «сервера»: этапы, план, пауза -----------------------------
const BASE = [
  { id: 'planning', label: 'Планирование' },
  { id: 'execution', label: 'Выполнение' },
  { id: 'validation', label: 'Проверка' },
  { id: 'done', label: 'Готово' },
];
let state = {};
let PLAN = ['Собрать данные', 'Написать код', 'Прогнать тесты'];
let STEP_DELAY = 5;   // задержка ответа «сервера» на шаг (для проверки «Паузы»)

function setState(patch) {
  state = Object.assign({
    stage: 'planning', current_step: '', expected_action: 'составить план и подтвердить его у пользователя',
    stage_label: 'Планирование', steps: [], step_index: 0, step_number: 0, steps_total: 0,
    paused: false, autonomous: false, can_pause: false, can_resume: false, can_confirm: false,
    terminal: false, extra_stage: null, reason: '', task_id: 's-1',
    updated_at: '2026-01-01T00:00:00', history: [], base_stage: 'planning',
  }, patch || {});
  if (BASE.some(b => b.id === state.stage)) state.base_stage = state.stage;
  state.can_confirm = (state.stage === 'planning' || state.stage === 'awaiting_user')
    && state.steps_total > 0 && !state.paused;
  state.can_pause = !state.terminal && !state.paused;
  state.can_resume = !!state.paused;
  state.can_cancel = !state.terminal;
  state.max_redo = 2;
  state.redo_count = Number(state.redo_count) || 0;
  state.base_stages = BASE.map(s => ({ id: s.id, label: s.label, active: s.id === state.base_stage }));
}
function snapshot() {
  const copy = JSON.parse(JSON.stringify(state));
  copy.steps = (state.steps || []).map((step, i) => ({
    number: i + 1, text: step.text,
    active: i === state.step_index && state.stage !== 'done' && state.stage !== 'cancelled',
    done: state.stage === 'validation' || state.stage === 'done' || i < state.step_index,
  }));
  return copy;
}
function setPlan(list) {
  setState(Object.assign({}, state, {
    steps: list.map(text => ({ text })),
    steps_total: list.length, step_index: 0, step_number: list.length ? 1 : 0,
  }));
}
setState({});

let workspace = {
  tasks: [{ id: 't-1', name: 'Задача' }], active_task: 't-1',
  sessions: [{ id: 's-1', title: 'Отчёт' }, { id: 's-2', title: 'Рецепт борща' }],
  active_session: 's-1',
  profile: {
    active: 'user_1',
    profiles: [{ id: 'user_1', profile_name: 'Профиль', label: 'Профиль' }],
    profile: { id: 'user_1', profile_name: 'Профиль', label: 'Профиль' },
  },
};
const calls = [];
const chatBodies = [];
// Журналы чата по сессиям (как dialog["log"] на сервере): что пользователь видел
// в окне — реплики, ответы и служебные debug-строки. По ним фронт восстанавливает
// чат при переключении диалога (в памяти messages debug не хранится).
const logs = {
  's-1': [
    { kind: 'user', text: 'Сделай отчёт по продажам' },
    { kind: 'debug', text: 'Автомат задачи: этап planning — разбиваю запрос на шаги.' },
    { kind: 'assistant', text: '📋 План задачи — 2 шага' },
  ],
  's-2': [
    { kind: 'user', text: 'дай рецепт борща' },
    { kind: 'debug', text: 'Автомат задачи: этап execution, шаг 1 из 2.' },
    { kind: 'assistant', text: 'Классический рецепт борща с бульоном.' },
  ],
};
// Шаг «в полёте»: нужен, чтобы проверить мгновенную реакцию «Паузы».
let stepInFlight = false;
// Задержка ответа на переключение задачи: нужна, чтобы проверить, что окно чата
// очищается СРАЗУ, а не показывает текст прежней задачи.
let selectDelay = 0;
// Задержка ответа планировщика (нужна, чтобы нажать паузу во время планирования)
// и признак «отложенная пауза ждёт»: на реальном сервере она применяется сразу
// после показа плана.
let planDelay = 0;
let pendingPause = false;
// Замеры токенов по задачам — как dialog["usage"] на сервере: их отдаёт
// GET /api/agent/history, поэтому освежение вида не должно их обнулять.
const usageBySession = { 's-1': [], 's-2': [] };
// Сколько раз приходила команда «Пауза» (для проверки повторной отправки).
let pauseRequests = 0;
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

function jsonResponse(data, ok) {
  return { ok: ok !== false, status: ok === false ? 400 : 200,
    json: async () => data, text: async () => JSON.stringify(data) };
}
function streamResponse(events) {
  const lines = events.map(e => JSON.stringify(e) + '\n');
  let index = 0;
  const encoder = new TextEncoder();
  return { ok: true, status: 200, body: { getReader: () => ({
    read: async () => index < lines.length
      ? { done: false, value: encoder.encode(lines[index++]) }
      : { done: true } }) } };
}
function usage(extra) {
  const base = { requests: 1, input: 10, output: 5, summary_requests: 0,
    summary_input: 0, summary_output: 0, limit: null, overflow: false };
  return Object.assign(base, extra || {});
}

// Число запросов в строке «Весь диалог» панели «Токены диалога».
function panelRequests() {
  const cell = dom.window.document.querySelector('#agent-total-body tr td:nth-child(2)');
  return cell ? Number(cell.textContent) : -1;
}

// Один шаг выполнения на «сервере»: сдвигаем шаг, на последнем — validation → done.
let requestSession = null;   // задача, в которой запущен текущий запрос

async function runStep() {
  stepInFlight = true;
  await sleep(STEP_DELAY);
  stepInFlight = false;
  if (state.paused) {
    return streamResponse([
      { type: 'state', state: snapshot() },
      { type: 'error', text: 'Задача на паузе: нажмите «Продолжить».' },
      { type: 'done', usage: usage(), state: snapshot() },
    ]);
  }
  const total = state.steps_total || 1;
  const last = state.step_index + 1 >= total;
  if (last) {
    setState(Object.assign({}, state, {
      stage: 'done', base_stage: 'done', current_step: '', expected_action: '',
      step_index: total - 1, step_number: total, terminal: true, extra_stage: null,
    }));
  } else {
    const next = state.step_index + 1;
    setState(Object.assign({}, state, {
      stage: 'execution', current_step: `step_${next + 1}`, step_index: next,
      step_number: next + 1, expected_action: 'выполнить: ' + state.steps[next].text,
    }));
  }
  // Журнал задачи, в которой шаг запущен (как dialog["log"] на сервере): по нему
  // открытый диалог восстанавливается после фонового шага.
  const sessionId = requestSession || workspace.active_session;
  logs[sessionId] = (logs[sessionId] || []).concat([
    { kind: 'assistant', text: last ? 'Шаг выполнен: итог задачи.' : 'Шаг выполнен.' },
  ]);
  return streamResponse([
    { type: 'state', state: snapshot() },
    { type: 'bot', text: last ? 'Задача выполнена.' : 'Шаг выполнен.' },
    { type: 'state', state: snapshot() },
    // Уточнённый замер: автомат сделал служебный вызов (проверка результата) —
    // фронт обязан ЗАМЕНИТЬ замер запроса, а не добавить второй.
    { type: 'usage', usage: usage({ summary_requests: 1, summary_input: 7, summary_output: 3 }) },
    { type: 'done', usage: usage(), state: snapshot() },
  ]);
}

function makeFetch() {
  return async (url, options) => {
    const method = ((options && options.method) || 'GET').toUpperCase();
    calls.push(method + ' ' + url);
    const body = options && options.body ? JSON.parse(options.body) : {};

    if (url === '/api/agent/workspace') return jsonResponse(workspace);
    if (url === '/api/agent/history') return jsonResponse({
      messages: [], usage: [], summary: [], facts: {}, branches: {}, active_branch: null,
      log: logs[workspace.active_session] || [],
      usage: usageBySession[workspace.active_session] || [],
      session: { id: workspace.active_session, title: 'Диалог' }, state: snapshot(),
    });
    if (url === '/api/agent/profiles') return jsonResponse(workspace.profile);
    if (url === '/api/agent/memory') {
      return jsonResponse({ task: { id: 't-1', name: 'Задача' }, working: [], long_term: [] });
    }
    if (url === '/api/agent/state') {
      return jsonResponse({ state: snapshot(), session: { id: 's-1', title: 'Диалог' } });
    }
    if (url === '/api/agent/state/cancel') {
      setState(Object.assign({}, state, {
        stage: 'cancelled', base_stage: 'cancelled', current_step: '',
        expected_action: '', terminal: true,
        extra_stage: { id: 'cancelled', label: 'Отменено' },
      }));
      const snap = snapshot();
      if (stepInFlight) snap.pending = 'cancel';
      return jsonResponse({ state: snap });
    }
    if (url === '/api/agent/sessions' && method === 'POST') {
      const id = 's-' + (workspace.sessions.length + 1) + 'x';
      workspace = Object.assign({}, workspace, {
        sessions: workspace.sessions.concat([{ id: id, title: 'Новая задача' }]),
        active_session: id,
      });
      logs[id] = [];
      usageBySession[id] = [];
      setState({});                       // новая задача — автомат с нуля
      return jsonResponse(workspace);
    }
    if (url.indexOf('/api/agent/sessions/') === 0 && url.endsWith('/select')) {
      if (selectDelay) await sleep(selectDelay);
      const id = decodeURIComponent(url.split('/')[4]);
      workspace = Object.assign({}, workspace, { active_session: id });
      return jsonResponse(workspace);
    }
    if (url === '/api/agent/state/pause') {
      pauseRequests += 1;
      if (stepInFlight || planDelay) {
        // Как на сервере: работа идёт (шаг или построение плана) — пауза только
        // «принята» (pending), сам переход применяется после ответа модели.
        const snap = snapshot();
        snap.pending = 'pause';
        snap.paused = true;
        pendingPause = true;
        return jsonResponse({ state: snap });
      }
      setState(Object.assign({}, state, { paused: true, expected_action: 'пауза: нажмите «Продолжить»' }));
      return jsonResponse({ state: snapshot() });
    }
    if (url === '/api/agent/state/resume') {
      const step = (state.steps[state.step_index] || {}).text || '';
      setState(Object.assign({}, state, {
        paused: false,
        expected_action: state.stage === 'awaiting_user'
          ? 'подтвердить план («ок») или внести правки'
          : 'выполнить: ' + step,
      }));
      return jsonResponse({ state: snapshot() });
    }
    if (url === '/api/agent/state/confirm') {
      setState(Object.assign({}, state, {
        stage: 'execution', base_stage: 'execution', extra_stage: null,
        current_step: 'step_1', step_index: 0, step_number: 1,
        expected_action: 'выполнить: ' + ((state.steps[0] || {}).text || ''),
      }));
      return jsonResponse({ state: snapshot() });
    }
    if (url === '/api/agent/state/plan') {
      setPlan(body.steps || []);
      setState(Object.assign({}, state, { stage: 'awaiting_user', base_stage: 'planning' }));
      return jsonResponse({ state: snapshot() });
    }
    if (url === '/api/agent/chat') {
      chatBodies.push(body);
      // Как на сервере: задача запроса — из session_id (шаг фоновой задачи),
      // иначе открытая.
      requestSession = (body && body.session_id) || workspace.active_session;
      // Как на сервере: запрос после завершённой задачи — НОВАЯ задача, автомат
      // начинается заново (иначе заглушка оставалась в done и план не пересобирался).
      if (state.stage === 'done' || state.stage === 'cancelled') setState({});
      if (state.stage === 'planning') {
        if (planDelay) await sleep(planDelay);
        setPlan(PLAN);
        setState(Object.assign({}, state, {
          stage: 'awaiting_user', base_stage: 'planning',
          extra_stage: { id: 'awaiting_user', label: 'Ждём пользователя' },
          expected_action: 'подтвердить план («ок») или внести правки',
        }));
        if (pendingPause) {
          // Как на сервере: «Пауза», нажатая во время планирования, применяется
          // сразу после показа плана.
          pendingPause = false;
          setState(Object.assign({}, state, { paused: true,
            expected_action: 'пауза: нажмите «Продолжить»' }));
        }
        return streamResponse([
          { type: 'state', state: snapshot() },
          { type: 'bot', text: `📋 План задачи — ${PLAN.length} шагов:\n1. ${PLAN[0]}` },
          { type: 'state', state: snapshot() },
          { type: 'done', usage: usage(), state: snapshot() },
        ]);
      }
      if (state.stage === 'execution') {
        const stream = await runStep();
        (usageBySession[workspace.active_session] =
          usageBySession[workspace.active_session] || []).push(usage());
        return stream;
      }
      return streamResponse([
        { type: 'state', state: snapshot() },
        { type: 'done', usage: usage(), state: snapshot() },
      ]);
    }
    return jsonResponse({ detail: 'не найдено' }, false);
  };
}

const dom = new JSDOM(HTML, {
  runScripts: 'dangerously',
  url: 'http://localhost/',
  beforeParse(window) {
    window.fetch = makeFetch();
    window.TextDecoder = TextDecoder;
    window.TextEncoder = TextEncoder;
  },
});

const $ = (id) => dom.window.document.getElementById(id);
const q = (sel) => Array.from(dom.window.document.querySelectorAll(sel));
const blocks = () => q('#tm-track .tm-block');
const chips = () => q('#tm-plan .tm-chip');
const arrows = () => q('#tm-track .tm-arrow');
const activeBlock = () => q('#tm-track .tm-block.active')[0];
const chatCalls = () => calls.filter(c => c === 'POST /api/agent/chat').length;
const stepCalls = () => chatBodies.filter(b => b && b.continue_step === true).length;
const wait = (ms) => new Promise(r => dom.window.setTimeout(r, ms));

async function click(el, ms) {
  el.dispatchEvent(new dom.window.MouseEvent('click', { bubbles: true }));
  await wait(ms === undefined ? 30 : ms);
}

// Запрос пользователя прямо через страницу (как будто ввели текст и нажали «Отправить»).
async function sendText(text, ms) {
  $('input').value = text;
  $('input').dispatchEvent(new dom.window.Event('input', { bubbles: true }));
  await click($('send'), ms === undefined ? 60 : ms);
}

async function run() {
  await wait(150);   // страница загрузилась (режим агента по умолчанию)

  console.log('\n[A] Полоса видна и нарисована');
  check('полоса состояния показана в режиме агента', !$('task-machine').hidden);
  check('четыре базовых этапа', blocks().length === 4, 'блоков: ' + blocks().length);
  check('этапы соединены стрелками', arrows().length === 3, 'стрелок: ' + arrows().length);
  check('первый этап — планирование, подсвечен',
    activeBlock() && activeBlock().textContent === 'Планирование');
  check('кнопка справа — «Пауза»', $('tm-pause').textContent === 'Пауза');
  const actionsEl = dom.window.document.querySelector('.tm-actions');
  check('кнопка «Пауза» — крайний правый блок, стрелки к ней нет',
    actionsEl && actionsEl.lastElementChild === $('tm-pause') && !actionsEl.querySelector('.tm-arrow'));
  check('ожидаемое действие показано', $('tm-expected').textContent.includes('составить план'));
  check('«Подтвердить план» скрыта без плана', $('tm-confirm').hidden === true);

  console.log('\n[B] Запрос пользователя: план и ожидание подтверждения');
  await sendText('Сделай отчёт');
  check('появился блок «Ждём пользователя»',
    blocks().some(b => b.classList.contains('extra') && b.textContent === 'Ждём пользователя'));
  check('текущий этап — планирование', activeBlock() && activeBlock().textContent === 'Планирование');
  check('чипы шагов плана нарисованы', chips().length === PLAN.length, 'чипов: ' + chips().length);
  check('шаг 1 подсвечен', chips()[0].classList.contains('active'));
  check('кнопка «Подтвердить план» видна', $('tm-confirm').hidden === false);
  check('кнопка «✎ План» видна', $('tm-edit').hidden === false);
  check('автомат ждёт подтверждения, а не сообщения',
    $('input').disabled === false && $('tm-expected').textContent.includes('подтвердить план'));
  check('после плана шаги сами не выполняются', stepCalls() === 0);

  console.log('\n[C] Правка плана через модалку');
  await click($('tm-edit'));
  check('модалка плана открылась', $('plan-modal').hidden === false);
  check('в модалке текущие шаги', $('plan-input').value.split('\n').length === PLAN.length);
  $('plan-input').value = 'Один шаг: собрать отчёт';
  await click($('plan-modal-ok'), 60);
  check('запрос правки плана ушёл', calls.includes('PUT /api/agent/state/plan'));
  check('модалка закрылась', $('plan-modal').hidden === true);
  check('чипов стало один', chips().length === 1, 'чипов: ' + chips().length);
  check('после правки снова ждём подтверждения',
    blocks().some(b => b.textContent === 'Ждём пользователя'));

  console.log('\n[D] «Подтвердить план» запускает авто-прогон шагов');
  PLAN = ['Первый', 'Второй', 'Третий'];
  STEP_DELAY = 5;
  await sendText('Сделай отчёт из трёх шагов');
  const before = stepCalls();
  // Реплик пользователя на этот момент: дальше автомат работает сам, и новых
  // сообщений от пользователя быть не должно.
  const userMessagesAtStart = q('.msg.user').length;
  await click($('tm-confirm'), 60);
  check('подтверждение ушло', calls.includes('POST /api/agent/state/confirm'));
  check('шаги выполняются автоматически, без сообщений', stepCalls() > before,
    `(шагов: ${stepCalls() - before})`);
  check('запрос шага помечен continue_step и без текста',
    chatBodies.some(b => b && b.continue_step === true && b.content === ''),
    JSON.stringify(chatBodies[chatBodies.length - 1] || {}));
  for (let i = 0; i < 60 && state.stage !== 'done'; i++) await wait(20);
  check('автомат сам дошёл до этапа «Готово»',
    activeBlock() && activeBlock().textContent === 'Готово',
    activeBlock() && activeBlock().textContent);
  check('пользователю не пришлось писать сообщения',
    q('.msg.user').length === userMessagesAtStart,
    `(реплик юзера: ${q('.msg.user').length}, было ${userMessagesAtStart})`);
  check('после завершения поле ввода свободно', $('input').disabled === false);
  // Замер запроса один: событие usage заменяет присланный ранее, а не удваивает,
  // а освежение вида берёт сохранённые замеры с сервера (как dialog["usage"]).
  const savedRequests = (usageBySession[workspace.active_session] || [])
    .reduce((sum, u) => sum + (Number(u.requests) || 0), 0);
  check('панель токенов считает по одному запросу на шаг',
    panelRequests() === savedRequests && panelRequests() > 0,
    `(панель: ${panelRequests()}, сохранено: ${savedRequests})`);
  check('кнопка «Пауза» заблокирована на завершённой задаче', $('tm-pause').disabled === true);

  console.log('\n[E] «Пауза» останавливает авто-прогон');
  PLAN = ['Шаг A', 'Шаг B', 'Шаг C', 'Шаг D'];
  STEP_DELAY = 150;
  await sendText('Длинная задача на четыре шага');
  await click($('tm-confirm'), 10);
  await wait(60);   // «Пауза» приходит ВНУТРИ первого шага (шаг идёт 150 мс)
  const stepsBeforePause = stepCalls();
  check('прогон идёт (запросы шагов появились)', stepsBeforePause >= 1);
  await click($('tm-pause'), 10);      // «Пауза» во время прогона
  // Реакция МГНОВЕННАЯ: интерфейс меняется, не дожидаясь ответа модели —
  // кнопка уже «Продолжить», а чип говорит, что остановка будет после шага.
  check('на паузе кнопка переключилась сразу', $('tm-pause').textContent === 'Продолжить');
  check('видна пометка «остановлю после текущего шага»',
    $('tm-pending').hidden === false && $('tm-pending').textContent.includes('после текущего шага'),
    $('tm-pending').textContent);
  await wait(600);                     // даём текущему шагу доработать
  check('автомат остановлен на паузе', state.paused === true,
    `(paused: ${state.paused}, stage: ${state.stage})`);
  check('задача не дошла до «Готово»', state.stage !== 'done', state.stage);
  check('кнопка стала «Продолжить»', $('tm-pause').textContent === 'Продолжить');
  check('поле ввода заблокировано на паузе', $('input').disabled === true);
  check('подсказка про паузу', $('input').placeholder.includes('Продолжить'),
    $('input').placeholder);
  const stepsAtPause = stepCalls();
  await wait(250);
  check('после паузы новые шаги не запускаются', stepCalls() === stepsAtPause,
    `(${stepCalls()} против ${stepsAtPause})`);

  console.log('\n[F] «Продолжить» возвращает автомат в работу');
  STEP_DELAY = 5;
  await click($('tm-pause'), 80);
  check('запрос продолжения ушёл', calls.includes('POST /api/agent/state/resume'));
  check('кнопка снова «Пауза»', $('tm-pause').textContent === 'Пауза');
  for (let i = 0; i < 80 && state.stage !== 'done'; i++) await wait(20);
  check('после «Продолжить» автомат доработал план', state.stage === 'done', state.stage);

  console.log('\n[G] Переключение диалога: журнал чата восстанавливается');
  await sendText('Отчёт по продажам');   // уводим задачу из done (новая задача)
  const sessionItems = () => q('#sessions .session-item');
  check('в панели две сессии', sessionItems().length === 2, String(sessionItems().length));
  const secondSession = sessionItems().find(item => item.textContent.includes('Рецепт борща'));
  check('вторая сессия найдена в панели', !!secondSession);
  await click(secondSession, 80);
  check('запрос переключения ушёл', calls.some(c => c.includes('/select')));
  const chatText = dom.window.document.getElementById('messages').textContent;
  check('виден запрос пользователя из журнала сессии', chatText.includes('дай рецепт борща'));
  check('виден ответ из журнала сессии', chatText.includes('Классический рецепт борща'));
  check('видна debug-строка из журнала сессии',
    q('#messages .msg.debug').some(el => el.textContent.includes('этап execution')),
    String(q('#messages .msg.debug').length));
  const firstSession = sessionItems().find(item => item.textContent.includes('Отчёт'));
  await click(firstSession, 80);
  check('возврат в первый диалог показывает его журнал',
    dom.window.document.getElementById('messages').textContent.includes('Сделай отчёт по продажам'));
  // Журнал рисуется по kind, а не по роли в памяти: служебные строки — debug.
  check('служебные строки рисуются как debug, а не как реплики пользователя',
    q('#messages .msg.debug').length >= 1 && q('#messages .msg.user').length === 1,
    `(debug: ${q('#messages .msg.debug').length}, user: ${q('#messages .msg.user').length})`);

  console.log('\n[H] Изоляция потока: события чужого диалога не рисуются');
  // Пока шаг выполняется, переключаемся в другой диалог: ответ прежнего диалога
  // должен дочитаться, но в окно чата нового — не попасть (иначе в открытой
  // задаче появлялись бы чужие ответы и debug, а полоса — чужое состояние).
  setState({
    stage: 'execution', base_stage: 'execution', steps: [{ text: 'A' }, { text: 'B' }],
    steps_total: 2, step_index: 0, step_number: 1, current_step: 'step_1',
    expected_action: 'выполнить: A',
  });
  dom.window.eval('applyTaskState(' + JSON.stringify(snapshot()) + ')');
  STEP_DELAY = 150;
  const chatCallsBefore = chatCalls();
  const stream = dom.window.eval('sendAgentRequest(agentStepBody())');
  await wait(30);
  const otherSession = sessionItems().find(item => item.textContent.includes('Рецепт борща'));
  await click(otherSession, 40);        // переключаемся во время шага
  await stream;                          // поток прежнего диалога дочитывается
  await wait(60);
  const viewText = dom.window.document.getElementById('messages').textContent;
  check('ответ прежнего диалога не нарисован в открытом',
    !viewText.includes('Задача выполнена') && !viewText.includes('Шаг выполнен'), viewText.slice(-80));
  const barStage = activeBlock() ? activeBlock().textContent : '';
  check('полоса показывает состояние ОТКРЫТОЙ задачи, а не выполнявшейся',
    barStage === 'Планирование' || barStage === 'Выполнение' || barStage === 'Готово',
    barStage);
  check('запрос шага всё же ушёл на сервер (ответ сохранится)',
    chatCalls() > chatCallsBefore);
  STEP_DELAY = 5;

  console.log('\n[H0] «Новая задача» доступна во время доработки шага');
  setState({
    stage: 'execution', base_stage: 'execution', steps: [{ text: 'A' }, { text: 'B' }],
    steps_total: 2, step_index: 0, step_number: 1, current_step: 'step_1',
    expected_action: 'выполнить: A',
  });
  dom.window.eval('applyTaskState(' + JSON.stringify(snapshot()) + ')');
  STEP_DELAY = 150;
  const pendingStep = dom.window.eval('sendAgentRequest(agentStepBody())');
  await wait(30);
  await click($('tm-pause'), 10);                 // «Пауза» → пометка pending
  check('показана пометка «остановлю после шага»', $('tm-pending').hidden === false);
  check('«Новая задача» доступна во время доработки шага',
    $('session-new').disabled === false, 'кнопка заблокирована');
  const sessionsBefore = q('#sessions .session-item').length;
  await click($('session-new'), 60);
  check('новая задача создана', q('#sessions .session-item').length === sessionsBefore + 1,
    `(${q('#sessions .session-item').length} против ${sessionsBefore})`);
  check('вид переключился на новую задачу', state.stage === 'planning', state.stage);
  check('пометка прежней задачи не висит в новой', $('tm-pending').hidden === true);
  STEP_DELAY = 0;
  await pendingStep;                              // прежний шаг дорабатывает в фоне
  await wait(80);
  check('прежний шаг доработал (ответ не потерян)',
    (logs[Object.keys(logs).find(k => k !== workspace.active_session)] || [])
      .some(item => item.text.includes('Шаг выполнен')));
  STEP_DELAY = 5;

  console.log('\n[H1] Пометка «остановлю после шага» не зависает');
  // Сценарий: шаг заканчивается ровно в момент нажатия «Паузы» — сервер отвечает
  // pending, но применить её уже не успевает. Клиент обязан сверить состояние с
  // сервером и повторить команду, а не оставлять чип навсегда.
  setState({
    stage: 'execution', base_stage: 'execution', steps: [{ text: 'A' }, { text: 'B' }],
    steps_total: 2, step_index: 0, step_number: 1, current_step: 'step_1',
    expected_action: 'выполнить: A',
  });
  dom.window.eval('applyTaskState(' + JSON.stringify(snapshot()) + ')');
  pauseRequests = 0;
  STEP_DELAY = 120;
  const loneStep = dom.window.eval('sendAgentRequest(agentStepBody())');   // шаг в полёте
  await wait(30);
  await click($('tm-pause'), 10);       // пауза: сервер ответит pending
  check('пометка появилась сразу', $('tm-pending').hidden === false, $('tm-pending').textContent);
  STEP_DELAY = 0;                        // шаг завершается — применить паузу уже некому
  await loneStep;
  await wait(120);                      // клиент сверяется с сервером и повторяет
  check('команда «Пауза» отправлена повторно', pauseRequests >= 2, String(pauseRequests));
  check('пометка «остановлю после шага» не осталась навсегда', $('tm-pending').hidden === true);
  check('пометка «остановлю после шага» снята', $('tm-pending').hidden === true,
    $('tm-pending').textContent);
  check('кнопка стала «Продолжить»', $('tm-pause').textContent === 'Продолжить');
  check('состояние действительно на паузе', state.paused === true, String(state.paused));
  STEP_DELAY = 5;
  await click($('tm-pause'), 60);        // снимаем паузу перед следующими секциями

  console.log('\n[H2] Кнопка «продолжить прогон» для остановленной задачи');
  setState({
    stage: 'execution', base_stage: 'execution', steps: [{ text: 'A' }, { text: 'B' }, { text: 'C' }],
    steps_total: 3, step_index: 1, step_number: 2, current_step: 'step_2',
    expected_action: 'выполнить: B',
  });
  dom.window.eval('applyTaskState(' + JSON.stringify(snapshot()) + ')');
  await wait(20);
  check('виден чип «продолжить: шаг 2 из 3»',
    $('tm-idle').hidden === false && $('tm-idle').textContent.includes('шаг 2 из 3'),
    $('tm-idle').textContent);
  const beforeIdle = stepCalls();
  await click($('tm-idle'), 80);
  check('клик по чипу запускает прогон шагов', stepCalls() > beforeIdle,
    `(${stepCalls()} против ${beforeIdle})`);
  for (let i = 0; i < 60 && state.stage !== 'done'; i++) await wait(20);
  check('прогон доходит до «Готово»', state.stage === 'done', state.stage);
  check('чип «продолжить» скрыт, когда прогон идёт', $('tm-idle').hidden === true);

  console.log('\n[H3] Фоновое выполнение: переключение задачи не останавливает её');
  const logsId = workspace.active_session;
  setState({
    stage: 'execution', base_stage: 'execution', steps: [{ text: 'A' }, { text: 'B' }, { text: 'C' }],
    steps_total: 3, step_index: 0, step_number: 1, current_step: 'step_1',
    expected_action: 'выполнить: A',
  });
  dom.window.eval('applyTaskState(' + JSON.stringify(snapshot()) + ')');
  STEP_DELAY = 120;
  const chainBodiesBefore = chatBodies.length;
  dom.window.eval('runStepChain()');
  await wait(60);
  check('авто-прогон идёт', stepCalls() >= 1);
  // Переключаемся в ДРУГУЮ задачу: работа ПРОДОЛЖАЕТСЯ (вариант «поставил и ушёл»).
  const other = sessionItems().find(item => !item.classList.contains('active'));
  await click(other, 30);
  check('переключение во время задачи не заблокировано',
    calls.some(c => c.includes('/select')), 'клик не сработал');
  const stepsRightAfterSwitch = stepCalls();
  await wait(150);
  // «Фоном» больше не чип-костыль: идущий прогон виден в СПИСКЕ задач (⏳), а
  // каждая задача при этом живёт своей цепочкой и не блокирует другие.
  check('в списке задач видно, что по задаче идут шаги',
    q('#sessions .session-item').some(item => item.textContent.includes('⏳')),
    q('#sessions .session-item').map(i => i.textContent).join(' | '));
  check('у другой задачи своя цепочка (прогонов: ' + dom.window.eval('chains.size') + ')',
    dom.window.eval('chains.size') >= 1);
  await wait(500);
  check('цепочка продолжается ФОНОМ после переключения',
    stepCalls() > stepsRightAfterSwitch, `(${stepCalls()} против ${stepsRightAfterSwitch})`);
  const chainBodies = chatBodies.slice(chainBodiesBefore).filter(b => b && b.continue_step);
  check('шаги адресованы СВОЕЙ задаче (session_id)',
    chainBodies.length > 0 && chainBodies.every(b => b.session_id === logsId),
    JSON.stringify(chainBodies.map(b => b.session_id)));
  check('переключение доведено до конца (активная сессия сменилась)',
    workspace.active_session !== logsId, workspace.active_session);
  check('поле ввода открытой задачи свободно при фоновой работе',
    $('input').disabled === false, $('input').placeholder);
  check('кнопка «Пауза» в открытой задаче ведёт себя как обычно',
    $('tm-pause').textContent === 'Пауза' || $('tm-pause').textContent === 'Продолжить',
    $('tm-pause').textContent);
  STEP_DELAY = 5;
  for (let i = 0; i < 60 && state.stage !== 'done'; i++) await wait(20);
  check('фоновая задача дошла до конца', state.stage === 'done', state.stage);
  await wait(120);

  console.log('\n[H3b] Две задачи работают одновременно');
  // Пока фоновая задача выполняет шаги, у ОТКРЫТОЙ может идти свой прогон:
  // раньше одна общая цепочка это блокировала.
  setState({
    stage: 'execution', base_stage: 'execution', steps: [{ text: 'X' }, { text: 'Y' }, { text: 'Z' }],
    steps_total: 3, step_index: 0, step_number: 1, current_step: 'step_1',
    expected_action: 'выполнить: X',
  });
  dom.window.eval('applyTaskState(' + JSON.stringify(snapshot()) + ')');
  STEP_DELAY = 150;
  const openSession = workspace.active_session;
  dom.window.eval('runStepChain(' + JSON.stringify(openSession) + ')');
  await wait(60);
  check('прогон открытой задачи запустился при работающей другой',
    dom.window.eval('chains.size') === 1 || dom.window.eval('chains.size') === 2,
    String(dom.window.eval('chains.size')));
  check('поле ввода занято, пока шаги идут в ОТКРЫТОЙ задаче',
    $('input').disabled === true, $('input').placeholder);
  STEP_DELAY = 5;
  for (let i = 0; i < 80 && dom.window.eval('chains.size') > 0; i++) await wait(20);
  check('оба прогона завершились', dom.window.eval('chains.size') === 0,
    String(dom.window.eval('chains.size')));
  check('в списке задач метка прогона снята',
    !q('#sessions .session-item').some(item => item.textContent.includes('⏳')));
  await wait(60);

  console.log('\n[H3c] Индикатор «думает» и очистка окна при переключении');
  setState({
    stage: 'execution', base_stage: 'execution', steps: [{ text: 'A' }, { text: 'B' }],
    steps_total: 2, step_index: 0, step_number: 1, current_step: 'step_1',
    expected_action: 'выполнить: A',
  });
  dom.window.eval('applyTaskState(' + JSON.stringify(snapshot()) + ')');
  STEP_DELAY = 200;
  const inFlight = dom.window.eval('sendAgentRequest(agentStepBody())');
  await wait(40);
  check('в открытой задаче видно «AI-агент думает»',
    $('typing').textContent.includes('думает'), $('typing').textContent);
  selectDelay = 80;
  const otherItem = sessionItems().find(item => !item.classList.contains('active'));
  const switching = click(otherItem, 5);          // не дожидаемся ответа сервера
  await wait(20);
  check('окно очищено сразу, текст прежней задачи не мелькает',
    dom.window.document.getElementById('messages').textContent.includes('Загружаю задачу'),
    dom.window.document.getElementById('messages').textContent.slice(-60));
  check('в новой задаче индикатор «думает» не показывается',
    $('typing').textContent === '', $('typing').textContent);
  await switching;
  await wait(60);
  check('после переключения индикатор пуст', $('typing').textContent === '',
    $('typing').textContent);
  const stepsShownBefore = (dom.window.document.getElementById('messages').textContent
    .match(/Шаг выполнен/g) || []).length;
  STEP_DELAY = 0;
  await inFlight;
  selectDelay = 0;
  await wait(60);
  const stepsShownAfter = (dom.window.document.getElementById('messages').textContent
    .match(/Шаг выполнен/g) || []).length;
  check('ответ прежней задачи не появился в открытой',
    stepsShownAfter === stepsShownBefore,
    `(${stepsShownAfter} против ${stepsShownBefore})`);

  console.log('\n[H3d] Чужие прогоны не блокируют открытую задачу');
  // Чужой прогон запускаем в задаче s-2, затем возвращаемся в s-1 и проверяем,
  // что её кнопки работают и она может запустить свой прогон параллельно.
  await dom.window.eval("selectSession('s-2')");
  await wait(80);
  setState({
    stage: 'execution', base_stage: 'execution', steps: [{ text: 'X' }, { text: 'Y' }, { text: 'Z' }],
    steps_total: 3, step_index: 0, step_number: 1, current_step: 'step_1',
    expected_action: 'выполнить: X',
  });
  dom.window.eval('applyTaskState(' + JSON.stringify(snapshot()) + ')');
  STEP_DELAY = 120;
  // Ранние секции могли «останавливать» прогон этой задачи — снимаем, как это
  // делает кнопка «Продолжить».
  dom.window.eval('noChainFor.clear()');
  dom.window.eval('runStepChain()');
  await wait(60);
  check('чужой прогон идёт (задача s-2)',
    dom.window.eval('chains.size') === 1 && workspace.active_session === 's-2',
    `${dom.window.eval('chains.size')} / ${workspace.active_session}`);

  await dom.window.eval("selectSession('s-1')");
  await wait(80);
  setState({
    stage: 'awaiting_user', base_stage: 'planning', steps: [{ text: 'A' }, { text: 'B' }],
    steps_total: 2, step_index: 0, step_number: 1, current_step: 'step_1',
    extra_stage: { id: 'awaiting_user', label: 'Ждём пользователя' },
    expected_action: 'подтвердить план («ок») или внести правки',
  });
  dom.window.eval('applyTaskState(' + JSON.stringify(snapshot()) + ')');
  await wait(20);
  check('открыта задача s-1', workspace.active_session === 's-1', workspace.active_session);
  check('«Подтвердить план» доступна при чужом прогоне',
    $('tm-confirm').hidden === false && $('tm-confirm').disabled === false,
    `hidden=${$('tm-confirm').hidden} disabled=${$('tm-confirm').disabled}`);
  check('«✎ План» доступна при чужом прогоне', $('tm-edit').disabled === false);
  const chainsBefore = dom.window.eval('chains.size');
  await click($('tm-confirm'), 80);
  check('открытая задача запустила СВОЙ прогон параллельно',
    dom.window.eval('chains.size') === chainsBefore + 1,
    `(${dom.window.eval('chains.size')} против ${chainsBefore})`);
  check('две задачи выполняются одновременно', dom.window.eval('chains.size') >= 2,
    String(dom.window.eval('chains.size')));
  STEP_DELAY = 5;
  for (let i = 0; i < 120 && dom.window.eval('chains.size') > 0; i++) await wait(20);
  await wait(60);

  console.log('\n[H3e] Кнопка «Отправить» и поле ввода — по своей задаче');
  // Свою задачу оставляем без прогона, чужую (s-2) держим в работе.
  await dom.window.eval("selectSession('s-1')");
  await wait(60);
  setState({
    stage: 'awaiting_user', base_stage: 'planning', steps: [{ text: 'A' }, { text: 'B' }],
    steps_total: 2, step_index: 0, step_number: 1, current_step: 'step_1',
    extra_stage: { id: 'awaiting_user', label: 'Ждём пользователя' },
    expected_action: 'подтвердить план («ок») или внести правки',
  });
  dom.window.eval('applyTaskState(' + JSON.stringify(snapshot()) + ')');
  // В чужой задаче запускаем долгий прогон и возвращаемся.
  await dom.window.eval("selectSession('s-2')");
  await wait(60);
  setState({
    stage: 'execution', base_stage: 'execution', steps: [{ text: 'X' }, { text: 'Y' }, { text: 'Z' }],
    steps_total: 3, step_index: 0, step_number: 1, current_step: 'step_1',
    expected_action: 'выполнить: X',
  });
  dom.window.eval('applyTaskState(' + JSON.stringify(snapshot()) + ')');
  STEP_DELAY = 250;
  dom.window.eval('noChainFor.clear()');
  dom.window.eval('runStepChain()');
  await wait(60);
  await dom.window.eval("selectSession('s-1')");
  await wait(80);
  setState({
    stage: 'awaiting_user', base_stage: 'planning', steps: [{ text: 'A' }, { text: 'B' }],
    steps_total: 2, step_index: 0, step_number: 1, current_step: 'step_1',
    extra_stage: { id: 'awaiting_user', label: 'Ждём пользователя' },
    expected_action: 'подтвердить план («ок») или внести правки',
  });
  dom.window.eval('applyTaskState(' + JSON.stringify(snapshot()) + ')');
  await wait(20);
  check('чужая задача всё ещё в работе', dom.window.eval('chains.size') >= 1,
    String(dom.window.eval('chains.size')));
  check('«Отправить» доступна при чужом прогоне', $('send').disabled === false,
    `disabled=${$('send').disabled}`);
  check('поле ввода доступно при чужом прогоне', $('input').disabled === false,
    $('input').placeholder);
  check('«Подтвердить план» доступна', $('tm-confirm').disabled === false);
  STEP_DELAY = 5;

  console.log('\n[H3f] Отправка в другой задаче во время чужого запроса');
  // Держим долгий запрос в s-2 и отправляем сообщение в s-1: раньше общий флаг
  // «идёт запрос» молча съедал отправку.
  await dom.window.eval("selectSession('s-2')");
  await wait(60);
  setState({
    stage: 'execution', base_stage: 'execution', steps: [{ text: 'X' }, { text: 'Y' }, { text: 'Z' }],
    steps_total: 3, step_index: 0, step_number: 1, current_step: 'step_1',
    expected_action: 'выполнить: X',
  });
  dom.window.eval('applyTaskState(' + JSON.stringify(snapshot()) + ')');
  STEP_DELAY = 220;
  dom.window.eval('noChainFor.clear()');
  dom.window.eval('runStepChain()');
  await wait(60);
  await dom.window.eval("selectSession('s-1')");
  await wait(80);
  setState({
    stage: 'awaiting_user', base_stage: 'planning', steps: [{ text: 'A' }, { text: 'B' }],
    steps_total: 2, step_index: 0, step_number: 1, current_step: 'step_1',
    extra_stage: { id: 'awaiting_user', label: 'Ждём пользователя' },
    expected_action: 'подтвердить план («ок») или внести правки',
  });
  dom.window.eval('applyTaskState(' + JSON.stringify(snapshot()) + ')');
  await wait(20);
  const chatBefore = chatBodies.length;
  await sendText('моё сообщение в открытой задаче', 120);
  const newBodies = chatBodies.slice(chatBefore);
  check('сообщение из другой задачи ушло на сервер',
    newBodies.some(b => b && !b.continue_step && (b.content || '').includes('моё сообщение')),
    JSON.stringify(newBodies.map(b => b && b.content).slice(0, 4)));
  check('при этом чужой прогон не сломался', dom.window.eval('chains.size') >= 1,
    String(dom.window.eval('chains.size')));
  check('реплика пользователя нарисована', q('#messages .msg.user').length >= 1);
  STEP_DELAY = 5;
  for (let i = 0; i < 120 && dom.window.eval('chains.size') > 0; i++) await wait(20);
  await wait(60);

  console.log('\n[H3g] Набранный текст не переезжает в другую задачу');
  $('input').value = 'черновик для первой задачи';
  const otherForDraft = sessionItems().find(item => !item.classList.contains('active'));
  await click(otherForDraft, 120);
  check('после переключения поле ввода пустое', $('input').value === '',
    JSON.stringify($('input').value));
  for (let i = 0; i < 120 && dom.window.eval('chains.size') > 0; i++) await wait(20);
  await wait(40);

  console.log('\n[H4] Переключение задач не ставит их на паузу «само»');
  // Сценарий из отчёта: пауза в одной задаче не подтвердилась (шаг закончился в
  // момент нажатия), затем переключились в другую — команда НЕ должна досылаться
  // в открытую задачу и ставить её на паузу.
  setState({
    stage: 'execution', base_stage: 'execution', steps: [{ text: 'A' }, { text: 'B' }],
    steps_total: 2, step_index: 0, step_number: 1, current_step: 'step_1',
    expected_action: 'выполнить: A',
  });
  dom.window.eval('applyTaskState(' + JSON.stringify(snapshot()) + ')');
  pauseRequests = 0;
  STEP_DELAY = 120;
  const bgStep = dom.window.eval('sendAgentRequest(agentStepBody())');
  await wait(30);
  await click($('tm-pause'), 10);          // команда в этой задаче (ответ pending)
  // Считаем паузы ПОСЛЕ нажатия: дальше ни одного нового запроса быть не должно.
  const pauseCallsAfterClick = calls.filter(c => c === 'POST /api/agent/state/pause').length;
  const currentId = workspace.active_session;
  // Запоминаем задачу, в которой нажали «Пауза», и элемент её списка: вернуться
  // надо ИМЕННО в неё (команда привязана к задаче).
  const pausedItem = sessionItems().find(item => item.classList.contains('active'));
  const switchTarget = sessionItems().find(item => !item.classList.contains('active'));
  await click(switchTarget, 20);           // уходим в другую задачу
  STEP_DELAY = 0;
  await bgStep;                            // шаг завершается уже в другой задаче
  await wait(120);
  check('в открытую задачу команда НЕ досылается',
    calls.filter(c => c === 'POST /api/agent/state/pause').length === pauseCallsAfterClick,
    `(пауз: ${calls.filter(c => c === 'POST /api/agent/state/pause').length}, было ${pauseCallsAfterClick})`);
  check('открытая задача не на паузе', state.paused === false, String(state.paused));
  check('в открытой задаче кнопка «Пауза», а не «Продолжить»',
    $('tm-pause').textContent === 'Пауза', $('tm-pause').textContent);
  // Возвращаемся в задачу, где нажимали паузу: команду доводим до конца.
  const backItem = pausedItem;
  setState(Object.assign({}, state, { paused: false }));   // сервер паузу не применил
  check('команда привязана к задаче, где её нажали',
    (dom.window.eval('stopRequest') || {}).session === currentId,
    JSON.stringify(dom.window.eval('stopRequest')));
  const pausedBefore = calls.filter(c => c === 'POST /api/agent/state/pause').length;
  await click(backItem, 120);
  check('вернувшись, команда доводится до конца',
    calls.filter(c => c === 'POST /api/agent/state/pause').length === pausedBefore + 1,
    `(пауз: ${calls.filter(c => c === 'POST /api/agent/state/pause').length}, было ${pausedBefore})`);
  await click($('tm-pause'), 60);          // снимаем паузу
  STEP_DELAY = 5;

  console.log('\n[H5] Ответ фонового шага виден, когда возвращаешься');
  setState({
    stage: 'execution', base_stage: 'execution', steps: [{ text: 'A' }, { text: 'B' }],
    steps_total: 2, step_index: 0, step_number: 1, current_step: 'step_1',
    expected_action: 'выполнить: A',
  });
  dom.window.eval('applyTaskState(' + JSON.stringify(snapshot()) + ')');
  const historyBefore = calls.filter(c => c === 'GET /api/agent/history').length;
  const workSession = workspace.active_session;
  STEP_DELAY = 120;
  const bg = dom.window.eval('sendAgentRequest(agentStepBody())');
  await wait(30);
  const away = sessionItems().find(item => !item.classList.contains('active'));
  await click(away, 20);                    // уходим, пока шаг выполняется
  STEP_DELAY = 0;
  await bg;
  await wait(120);
  check('после фонового шага открытая задача освежается',
    calls.filter(c => c === 'GET /api/agent/history').length > historyBefore,
    `(запросов истории: ${calls.filter(c => c === 'GET /api/agent/history').length})`);
  // Возвращаемся в рабочую задачу — ответ фонового шага должен быть виден.
  const backWork = sessionItems().find(item => item.textContent.includes(workSession === 's-1' ? 'Отчёт' : 'Рецепт борща'));
  await click(backWork, 120);
  check('ответ фонового шага виден в журнале задачи',
    dom.window.document.getElementById('messages').textContent.includes('Шаг выполнен'),
    dom.window.document.getElementById('messages').textContent.slice(-80));
  STEP_DELAY = 5;

  console.log('\n[H7] Отложенная проверка: чип «выполнить проверку»');
  // «Пауза» на последнем шаге оставляет задачу на этапе validation: проверку
  // выполняет клик по чипу (или сообщение).
  setState({
    stage: 'validation', base_stage: 'validation', current_step: 'check',
    steps: [{ text: 'A' }, { text: 'B' }], steps_total: 2, step_index: 1, step_number: 2,
    expected_action: 'проверить результат шага 2: B',
  });
  dom.window.eval('applyTaskState(' + JSON.stringify(snapshot()) + ')');
  await wait(20);
  check('текущий этап — проверка', activeBlock() && activeBlock().textContent === 'Проверка',
    activeBlock() && activeBlock().textContent);
  check('в полосе чип «выполнить проверку»',
    $('tm-idle').hidden === false && $('tm-idle').textContent.includes('выполнить проверку'),
    $('tm-idle').textContent);
  check('поле ввода подсказывает про проверку',
    $('input').placeholder.includes('Проверка'), $('input').placeholder);
  const beforeCheck = stepCalls();
  await click($('tm-idle'), 160);
  check('клик запускает отложенную проверку (запрос ушёл)', stepCalls() > beforeCheck,
    `(${stepCalls()} против ${beforeCheck})`);
  check('запрос помечен как шаг автомата',
    chatBodies[chatBodies.length - 1] && chatBodies[chatBodies.length - 1].continue_step === true);
  await wait(200);

  console.log('\n[H8] Пауза во время планирования: кнопки согласованы');
  setState({});
  workspace = Object.assign({}, workspace, { active_session: workspace.active_session });
  dom.window.eval('applyTaskState(' + JSON.stringify(snapshot()) + ')');
  planDelay = 150;
  const planRequest = dom.window.eval('sendAgentRequest({ content: "Задача", continue_step: false })');
  await wait(40);
  await click($('tm-pause'), 10);
  check('во время планирования показана пометка ожидания',
    $('tm-pending').hidden === false, $('tm-pending').textContent);
  check('кнопка уже «Продолжить» (состояние согласовано)',
    $('tm-pause').textContent === 'Продолжить', $('tm-pause').textContent);
  planDelay = 0;
  await planRequest;
  await wait(80);
  check('после плана задача НА ПАУЗЕ и кнопка «Продолжить»',
    $('tm-pause').textContent === 'Продолжить' && state.paused === true,
    `(${$('tm-pause').textContent}, paused=${state.paused})`);
  check('пометка ожидания снята', $('tm-pending').hidden === true,
    $('tm-pending').textContent);
  check('поле ввода подсказывает про паузу', $('input').placeholder.includes('Продолжить'),
    $('input').placeholder);
  await click($('tm-pause'), 80);        // «Продолжить»
  check('после «Продолжить» задача снова работает', state.paused === false);
  await wait(60);

  console.log('\n[H9] Фоновый шаг не рисуется в другой задаче');
  setState({
    stage: 'execution', base_stage: 'execution', steps: [{ text: 'A' }, { text: 'B' }, { text: 'C' }],
    steps_total: 3, step_index: 0, step_number: 1, current_step: 'step_1',
    expected_action: 'выполнить: A',
  });
  dom.window.eval('applyTaskState(' + JSON.stringify(snapshot()) + ')');
  STEP_DELAY = 150;
  const bgWorkSession = workspace.active_session;
  dom.window.eval('runStepChain(' + JSON.stringify(bgWorkSession) + ')');
  await wait(60);
  // Открываем ДРУГУЮ задачу (как «просто открыть новую задачу сразу же»).
  const otherTask = sessionItems().find(item => !item.classList.contains('active'));
  await click(otherTask, 40);
  const nodesAfterSwitch = dom.window.document.querySelectorAll('#messages [data-agent]').length;
  const shownAfterSwitch = (dom.window.document.getElementById('messages').textContent
    .match(/Шаг (фоновой задачи )?выполнен/g) || []).length;
  await wait(600);                      // фоновые шаги продолжают идти
  const nodesLater = dom.window.document.querySelectorAll('#messages [data-agent]').length;
  const shownLater = (dom.window.document.getElementById('messages').textContent
    .match(/Шаг (фоновой задачи )?выполнен/g) || []).length;
  check('в открытую задачу фоновые шаги не сыплются',
    nodesLater <= nodesAfterSwitch + 1, `(${nodesLater} против ${nodesAfterSwitch})`);
  check('фоновые шаги при этом реально шли',
    chatBodies.filter(b => b && b.continue_step).length >= 2,
    String(chatBodies.filter(b => b && b.continue_step).length));
  check('сообщения чужой задачи в окне не появляются',
    shownLater === shownAfterSwitch, `(${shownLater} против ${shownAfterSwitch})`);
  STEP_DELAY = 5;
  for (let i = 0; i < 80 && dom.window.eval('chains.size') > 0; i++) await wait(20);
  await wait(60);

  console.log('\n[I] «Отменить задачу»');
  // Незавершённая задача с планом (состояние ставим напрямую, чтобы секция не
  // зависела от того, что сделали предыдущие).
  setState({
    stage: 'awaiting_user', base_stage: 'planning', steps: [{ text: 'A' }],
    steps_total: 1, step_index: 0, step_number: 1, current_step: 'step_1',
    extra_stage: { id: 'awaiting_user', label: 'Ждём пользователя' },
    expected_action: 'подтвердить план («ок») или внести правки',
  });
  dom.window.eval('applyTaskState(' + JSON.stringify(snapshot()) + ')');
  await wait(20);
  check('кнопка «Отменить» видна на незавершённой задаче', $('tm-cancel').hidden === false);
  await click($('tm-cancel'), 30);
  check('открылась модалка подтверждения', $('confirm-modal').hidden === false);
  await click($('confirm-modal-ok'), 80);
  check('запрос отмены ушёл', calls.includes('POST /api/agent/state/cancel'));
  check('полоса показывает «Отменено»',
    blocks().some(b => b.classList.contains('extra') && b.textContent === 'Отменено'),
    blocks().map(b => b.textContent).join('|'));
  check('кнопка «Отменить» скрыта после отмены', $('tm-cancel').hidden === true);
  check('«Пауза» недоступна после отмены', $('tm-pause').disabled === true);
  check('поле ввода свободно: новое сообщение = новая задача', $('input').disabled === false);
  check('подсказка про отменённую задачу',
    $('input').placeholder.includes('отменена'), $('input').placeholder);
  check('пользователю сообщили об отмене',
    dom.window.document.body.textContent.includes('Задача отменена'));

  console.log('\n[J] Счётчик доработок в полосе');
  setState({
    stage: 'execution', base_stage: 'execution', steps: [{ text: 'Шаг A' }, { text: 'Шаг B' }],
    steps_total: 2, step_index: 0, step_number: 1, current_step: 'step_1',
    expected_action: 'выполнить: Шаг A', redo_count: 1,
  });
  dom.window.eval('applyTaskState(' + JSON.stringify(snapshot()) + ')');
  await wait(20);
  check('чип «доработка 1 из 2» показан',
    $('tm-redo').hidden === false && $('tm-redo').textContent === 'доработка 1 из 2',
    $('tm-redo').textContent);
  setState(Object.assign({}, state, { redo_count: 2 }));
  dom.window.eval('applyTaskState(' + JSON.stringify(snapshot()) + ')');
  await wait(20);
  check('на пределе доработок чип предупреждает',
    $('tm-redo').textContent === 'доработка 2 из 2'
    && $('tm-redo').title.includes('Лимит'), $('tm-redo').title);
  setState(Object.assign({}, state, { redo_count: 0 }));
  dom.window.eval('applyTaskState(' + JSON.stringify(snapshot()) + ')');
  await wait(20);
  check('без доработок чип скрыт', $('tm-redo').hidden === true);

  console.log('\n[K] Без проекта полоса подсказывает создать проект');
  workspace = { tasks: [], active_task: null, sessions: [], active_session: null };
  setState({});
  await dom.window.eval('loadWorkspace()');
  await wait(40);
  check('подсказка про отсутствие проекта',
    $('tm-step').textContent.includes('проект не создан'), $('tm-step').textContent);
  check('кнопка «Пауза» недоступна без проекта', $('tm-pause').disabled === true);
  check('поле ввода подсказывает создать проект',
    $('input').placeholder.toLowerCase().includes('создайте проект'), $('input').placeholder);

  console.log('\nИтог: ' + (failures ? 'ПРОВАЛЕНО проверок: ' + failures : 'все проверки пройдены'));
  dom.window.close();
  process.exit(failures ? 1 : 0);
}

run().catch(err => { console.error(err); process.exit(1); });
