/* Прогон интерфейса полосы состояния задачи (Task State Machine) в jsdom.
 *
 * Страница app/web/chat.html грузится целиком (runScripts: 'dangerously') с
 * заглушкой window.fetch, эмулирующей маршруты /api/agent/* (workspace, state,
 * history, chat) и мини-автомат задачи на стороне «сервера», и щёлкаются
 * кнопки полосы. Проверяются: четыре блока этапов со стрелками, подсветка
 * текущего этапа, чипы шагов плана, блок расширенного этапа (awaiting_user),
 * АВТО-ПРОГОН шагов после «Подтвердить план» (без сообщений пользователя),
 * остановка прогона кнопкой «Пауза», «Продолжить», правка плана через модалку,
 * блокировка ввода и подсказка без задачи. Раздел [P] проверяет MCP: кнопку
 * «MCP» рядом с шестерёнкой проекта, диалог со списком серверов (название,
 * описание, инструменты, причина недоступности), галочки и «применить».
 * КАТАЛОГ СЕРВЕРОВ ДЛЯ [P] берётся из реального реестра проекта
 * (app/ai/mcp.py, SERVERS) через ./venv/bin/python, а ожидания считаются от
 * него: серверы добавляются и убираются, и фикстур не должен от них отставать
 * (раньше здесь лежал список из четырёх серверов, и число 6 инструментов).
 * Раздел [S] проверяет RAG: кнопку «RAG» рядом с «MCP», диалог со списком баз
 * знаний (стратегия, чанки, средний размер чанка, вес базы, документы,
 * предупреждения по файлам), галочки-переключатели, «применить» вместе с
 * параметрами разбиения и ЗАГРУЗКУ своей базы: файл читается в base64,
 * уходит на индексацию, новая база появляется в списке, в чате — отчёт о
 * чанках. Стратегии и пределы размеров заглушка отдаёт как сервер
 * (app/ai/rag.py), поэтому интерфейс проверяется на реальном контракте.
 *
 * Запуск (нужен jsdom — в зависимостях проекта его нет, ставится отдельно):
 *     npm install --no-save jsdom      # в корне проекта (node_modules не в git)
 *     node tools/check_chat_ui.js
 * Прежний способ через NODE_PATH в Node.js 26 не работает (переменная больше не
 * подхватывается), поэтому jsdom ставится рядом с проектом.
 *
 * Сеть и сервер не нужны: fetch подменён внутри страницы. Скрипт возвращает
 * ненулевой код выхода, если хоть одна проверка провалилась.
 */
const fs = require('fs');
const { execFileSync } = require('child_process');
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
// «Сервер»: проверку результата выполнить не удалось — задача остаётся на этапе
// «Проверка» (check_blocked) и ждёт решения пользователя.
let CHECK_BLOCKED = false;

function setState(patch) {
  state = Object.assign({
    stage: 'planning', current_step: '', expected_action: 'составить план и подтвердить его у пользователя',
    stage_label: 'Планирование', steps: [], step_index: 0, step_number: 0, steps_total: 0,
    paused: false, autonomous: false, can_pause: false, can_resume: false, can_confirm: false,
    terminal: false, extra_stage: null, reason: '', task_id: 's-1',
    repeat_ready: false,
    updated_at: '2026-01-01T00:00:00', history: [], base_stage: 'planning',
    check_blocked: false, can_accept: false,
  }, patch || {});
  if (BASE.some(b => b.id === state.stage)) state.base_stage = state.stage;
  state.can_confirm = (state.stage === 'planning' || state.stage === 'awaiting_user')
    && state.steps_total > 0 && !state.paused;
  state.can_pause = !state.terminal && !state.paused;
  state.can_resume = !!state.paused;
  state.can_cancel = !state.terminal;
  // Проверку выполнить не удалось: задача не готова и ждёт решения пользователя
  // (как на сервере — см. task_state.snapshot).
  state.can_accept = state.stage === 'validation' && !!state.check_blocked && !state.paused;
  state.max_redo = 2;
  state.redo_count = Number(state.redo_count) || 0;
  state.base_stages = BASE.map(s => ({ id: s.id, label: s.label, active: s.id === state.base_stage }));
}
function snapshot() {
  const copy = JSON.parse(JSON.stringify(state));
  // ПЕРИОДИЧЕСКАЯ задача: сервер отдаёт полосу БЕЗ «проверки» и «готово» — этих
  // этапов она не проходит (см. task_state.PERIODIC_BASE_STAGES). Определяем
  // здесь, а не в setState: тот вызывается при инициализации заглушки, когда
  // workspace ещё не объявлен.
  if (typeof workspace !== 'undefined' && typeof PERIODIC !== 'undefined'
      && PERIODIC.tasks[workspace.active_session]) {
    copy.base_stages = copy.base_stages.filter(
      item => item.id === 'planning' || item.id === 'execution');
  }
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
// Время в журнале — как его пишет сервер (наивное локальное, до секунд).
// Часть записей БЕЗ времени: старые файлы времени не имеют, и подпись тогда не
// показывается (выдумывать его нельзя).
function logAt(hours, minutes) {
  const now = new Date();
  const pad = value => String(value).padStart(2, '0');
  return `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`
    + `T${pad(hours)}:${pad(minutes)}:00`;
}
const logs = {
  's-1': [
    { kind: 'user', text: 'Сделай отчёт по продажам', at: logAt(9, 7) },
    { kind: 'debug', text: 'Автомат задачи: этап planning — разбиваю запрос на шаги.', at: logAt(9, 8) },
    { kind: 'assistant', text: '📋 План задачи — 2 шага', at: logAt(9, 9) },
  ],
  's-2': [
    { kind: 'user', text: 'дай рецепт борща' },
    { kind: 'debug', text: 'Автомат задачи: этап execution, шаг 1 из 2.' },
    { kind: 'assistant', text: 'Классический рецепт борща с бульоном.' },
  ],
};
// Инварианты (правила, которые агент не должен нарушать): у проекта и у
// задачи-диалога свои списки + проверка пар «проект × задача». Заглушка
// повторяет поведение сервера: POST добавляет правило и (если правила есть с
// обеих сторон) возвращает противоречие; DELETE убирает правило и его проверки;
// resolve фиксирует решение пользователя.
let INV = {
  project: [{ id: 'i-p1', text: 'Только PostgreSQL' }],
  task: [{ id: 'i-t1', text: 'Только MongoDB' }],
  tasks: { 's-1': [{ id: 'i-t1', text: 'Только MongoDB' }], 's-2': [] },
  conflict: true,
  resolved: {},   // ключ пары -> 'project'|'task' (решение пользователя)
};
function invPairs() {
  const task = (workspace.active_session && INV.tasks[workspace.active_session]) || [];
  const pairs = [];
  INV.project.forEach(p => task.forEach(t => pairs.push({ p, t })));
  return pairs;
}
function invChecks() {
  // Вердикт проверки НЕ меняется от решения пользователя: конфликт остаётся
  // конфликтом, добавляется лишь отметка «решено, главнее X» — так же ведёт
  // себя сервер.
  return invPairs().map(({ p, t }) => {
    const key = p.id + '|' + t.id;
    const winner = INV.resolved[key] || '';
    return {
      key: key, project_id: p.id, task_id: t.id,
      verdict: INV.conflict ? 'conflict' : 'clear',
      reason: INV.conflict ? 'СУБД разная' : '',
      resolved: !!winner, winner: winner,
      checked_ok: true, project_text: p.text, task_text: t.text,
    };
  });
}
function invPayload(sessionId) {
  const session = sessionId || workspace.active_session;
  const list = invChecks();
  return {
    project: INV.project,
    task: (session && INV.tasks[session]) || [],
    project_id: workspace.active_task, task_id: session,
    pairs: list.length, checks: list,
    has_conflict: list.some(c => c.verdict === 'conflict' && !c.resolved),
    counts: {
      project: INV.project.length,
      task: ((session && INV.tasks[session]) || []).length,
      conflict: list.filter(c => c.verdict === 'conflict' && !c.resolved).length,
    },
  };
}

// Последний разбор запроса на соответствие инвариантам (заглушка арбитра).
let LAST_ANALYSIS = { verdict: '', kind: '', explanation: '', suggestions: [] };

// --- Каталог MCP: из РЕАЛЬНОГО реестра проекта ------------------------------
// Серверы добавляются и убираются (три локальных на stdio плюс свои на VPS),
// поэтому список берётся из app/ai/mcp.py (SERVERS), а НЕ переписывается здесь:
// иначе фикстур отстаёт от реестра, и проверка «зеленеет» на устаревших данных.
// Инструменты — СИНТЕТИЧЕСКИЕ: их объявляет живой сервер по сети, а интерфейсу
// важен сам список и число строк, поэтому сеть тут не нужна.
const ROOT = require('path').join(__dirname, '..');
const UNAVAILABLE_REASON = 'туннель не поднят (заглушка UI-проверки)';

function readRegistryServers() {
  // Переопределение каталога — чтобы проверить САМУ эту проверку на другом
  // наборе серверов (в реестре стало больше или меньше серверов):
  //   CHECK_CHAT_UI_REGISTRY=/tmp/servers.json node tools/check_chat_ui.js
  // Файл — массив записей {id, name, description, source, transport}.
  const fromFile = process.env.CHECK_CHAT_UI_REGISTRY;
  if (fromFile) {
    const servers = JSON.parse(fs.readFileSync(fromFile, 'utf8'));
    if (Array.isArray(servers) && servers.length) return servers;
    console.log('  FAIL каталог из ' + fromFile + ' пуст — серверы нужны хотя бы');
    console.log('       для одной строки списка.');
    process.exit(1);
  }
  const code = [
    'import json, sys',
    "sys.path.insert(0, '.')",
    'from app.ai import mcp',
    'print(json.dumps([{',
    "    'id': entry['id'],",
    "    'name': entry.get('name') or entry['id'],",
    "    'description': entry.get('description') or '',",
    "    'source': entry.get('source') or '',",
    "    'transport': mcp.transport_of(entry),",
    '} for entry in mcp.SERVERS], ensure_ascii=False))',
  ].join('\n');
  let lastError = '';
  for (const bin of [require('path').join(ROOT, 'venv', 'bin', 'python'), 'python3']) {
    try {
      const out = execFileSync(bin, ['-c', code], {
        cwd: ROOT, stdio: ['ignore', 'pipe', 'ignore'],
      }).toString('utf8').trim();
      const servers = JSON.parse(out);
      if (Array.isArray(servers) && servers.length) return servers;
      lastError = 'реестр пуст';
    } catch (err) {
      lastError = String(err && err.message ? err.message : err);
    }
  }
  console.log('  FAIL каталог MCP не прочитан из app/ai/mcp.py — ' + lastError);
  console.log('       UI-проверка берёт список серверов из реестра, поэтому он');
  console.log('       должен читаться: ./venv/bin/python (см. SESSION_PROMPT §7).');
  process.exit(1);
}

const REGISTRY = readRegistryServers();
const MCP_SERVERS = REGISTRY.map((server, index) => {
  // Один сервер показываем недоступным (последний в реестре): интерфейс обязан
  // показать ПРИЧИНУ, а не молчать. В реестре из одного сервера он остаётся
  // рабочим — тогда проверка причины пропускается, а не падает.
  const down = REGISTRY.length > 1 && index === REGISTRY.length - 1;
  const count = down ? 0 : (server.transport === 'http' ? 3 : 1);
  return {
    id: server.id, name: server.name, source: server.source,
    description: server.description,
    transport: server.transport,
    available: !down,
    error: down ? UNAVAILABLE_REASON : '',
    tools: Array.from({ length: count }, (unused, i) => ({
      name: server.id + '_tool_' + (i + 1),
      title: 'Инструмент ' + (i + 1),
      description: 'Демонстрационный инструмент ' + (i + 1)
        + ' сервера ' + server.id,
    })),
  };
});

// Заглушка повторяет сервер: GET отдаёт серверы с описанием, инструментами и
// состоянием галочек; POST применяет ПОЛНЫЙ набор галочек (неизвестные id
// отбрасываются) и возвращает тот же снимок.
let MCP = { enabled: [], servers: MCP_SERVERS };
function mcpPayload() {
  const servers = MCP.servers.map(server => Object.assign({}, server, {
    enabled: MCP.enabled.indexOf(server.id) >= 0,
  }));
  return {
    servers: servers,
    enabled: MCP.enabled.slice(),
    project_id: workspace.active_task,
    counts: {
      servers: servers.length,
      enabled: servers.filter(s => s.enabled).length,
      tools: servers.reduce((total, s) => total + s.tools.length, 0),
      available: servers.filter(s => s.available).length,
    },
  };
}

// --- Базы знаний (RAG): снимок «сервера» -------------------------------------
// Заглушка повторяет /api/agent/rag: список баз профиля с метриками, галочки
// проекта, доступные стратегии, пределы и состояние эмбеддингов. Метрики здесь
// НАРОЧНО разные у двух баз — интерфейс обязан показать их по каждой базе, а не
// одной строкой на всех.
const RAG_STRATEGIES = [
  { id: 'structure', name: 'По структуре (заголовки/разделы/файлы)',
    description: 'Режет по заголовкам и разделам документа.' },
  { id: 'fixed', name: 'Фиксированный размер',
    description: 'Режет текст окнами по N символов с перекрытием.' },
];

function ragBase(overrides) {
  return Object.assign({
    id: 'kb-00000001', name: 'Инструкции оператора', enabled: false,
    strategy: 'structure', strategy_name: 'По структуре (заголовки/разделы/файлы)',
    chunk_size: 1000, overlap: 150,
    chunks: 42, documents: 2, chars_total: 32768, chars_avg: 780,
    chars_min: 210, chars_max: 1180, sections: 11, est_tokens: 8192,
    vectors_bytes: 64512, dim: 384, backend: 'sentence-transformers',
    model: 'paraphrase-multilingual-MiniLM-L12-v2', fallback: '',
    size_bytes: 262144, size_human: '256.0 КБ', share: 0.6,
    created: '2026-01-02T10:00:00', updated: '2026-01-02T10:05:00',
    storage: { sqlite: true, json: true },
    sources: [
      { source: 'guide.md', format: 'Markdown', chunks: 30, chars: 24000,
        pages: 0, warning: '' },
      { source: 'scan.pdf', format: 'PDF', chunks: 12, chars: 8768,
        pages: 4, warning: 'в PDF нет текстового слоя' },
    ],
    failures: [],
  }, overrides || {});
}

let RAG = {
  bases: [ragBase({}), ragBase({
    id: 'kb-00000002', name: 'Регламенты', strategy: 'fixed',
    strategy_name: 'Фиксированный размер', chunk_size: 500, overlap: 50,
    chunks: 17, documents: 1, chars_avg: 470, chars_min: 300, chars_max: 500,
    sections: 0, size_bytes: 131072, size_human: '128.0 КБ', share: 0.4,
    fallback: 'модель недоступна: нет сети',
    sources: [{ source: 'rules.docx', format: 'Word (DOCX)', chunks: 17,
                chars: 7990, pages: 0, warning: '' }],
    failures: ['bad.pdf — PDF защищён паролем'],
  })],
  enabled: [],
  uploads: [],
  streams: [],
  settings: { strategy: 'structure', chunk_size: 1000, overlap: 150 },
};

// Чанки базы для проверки просмотра: 25 штук с разными документами и текстом,
// чтобы работали и страницы, и фильтр по документу, и поиск по тексту.
function ragChunkSet(base) {
  const names = (base.sources || []).map(item => item.source);
  const list = names.length ? names : ['guide.md'];
  const chunks = [];
  for (let i = 0; i < 25; i += 1) {
    const source = list[i % list.length];
    chunks.push({
      chunk_id: base.id + '-' + (i % 2) + '-' + String(i).padStart(4, '0'),
      index: i, doc_index: i % 2, position: i, source: source,
      title: source.replace(/\.[a-z]+$/, ''),
      section: i % 3 === 0 ? 'Глава 1 › Установка' : '',
      kind: 'section', start: i * 500, end: i * 500 + 420, chars: 420,
      text: 'Фрагмент № ' + (i + 1) + ' из ' + source
        + '. Резервное копирование выполняется командой backup.sh.',
    });
  }
  return chunks;
}

function ragChunksPayload(baseId, url) {
  const base = RAG.bases.filter(item => item.id === baseId)[0] || RAG.bases[0];
  const query = new URLSearchParams(url.split('?')[1] || '');
  const offset = Number(query.get('offset') || 0);
  const limit = Number(query.get('limit') || 10);
  const source = query.get('source') || '';
  const text = query.get('q') || '';
  let all = ragChunkSet(base);
  if (source) all = all.filter(chunk => chunk.source === source);
  if (text) {
    const needle = text.toLowerCase();
    all = all.filter(chunk => chunk.text.toLowerCase().indexOf(needle) >= 0);
  }
  const page = all.slice(offset, offset + limit);
  return {
    base: { id: base.id, name: base.name, strategy: base.strategy,
            strategy_name: base.strategy_name, chunk_size: base.chunk_size,
            overlap: base.overlap, chunks: base.chunks, chars_avg: base.chars_avg,
            dim: base.dim, backend: base.backend, model: base.model,
            sources: (base.sources || []).map(item => item.source) },
    chunks: page, total: all.length, offset: offset, limit: limit,
    has_more: offset + page.length < all.length,
    filter: { source: source, query: text },
  };
}

// Фоновые задачи индексации: заглушка повторяет сервер — задача заводится на
// загрузку, «идёт» несколько опросов (чтобы интерфейс успел показать прогресс),
// затем завершается. Прогресс выдаётся РАЗНЫЙ, иначе проверка не отличила бы
// живую полосу от застывшей.
let RAG_JOBS = [];
let RAG_JOB_SEQ = 0;

function ragStartJob(filename, baseId, name) {
  RAG_JOB_SEQ += 1;
  const job = {
    id: 'job-' + String(RAG_JOB_SEQ).padStart(8, '0'),
    state: 'running', stage: 'extract', stage_name: 'разбор документов',
    done: 0, total: 4, percent: 10,
    detail: '«' + filename + '»: страница 1 из 4',
    base_id: baseId || '', name: baseId ? '' : (name || 'База задач'),
    append: Boolean(baseId), chunks: 0, documents: 0, error: '',
    cancel_requested: false, started: '2026-01-02T10:00:00', finished: '',
    elapsed: 0.5, result: null,
    _startedAt: Date.now(), _filename: filename, _baseId: baseId || '',
  };
  RAG_JOBS.push(job);
  return job;
}

// Этапы задачи считаются ОТ ВРЕМЕНИ старта: 1.2 с разбор, потом эмбеддинги,
// потом запись, к 3 с — готово. Так проверка знает, что увидит в любой момент,
// и не зависит от того, сколько раз успел сработать таймер опроса.
const RAG_JOB_STAGES = [
  [2000, 'extract', 'разбор документов', 10, 0, 4, '«%s»: страница 1 из 4'],
  [4000, 'embed', 'эмбеддинги', 65, 2, 4, 'векторы: 2 из 4 чанков'],
  [5000, 'save', 'запись индекса', 97, 1, 1, 'пишу индекс: 3 чанков'],
];

function ragAdvanceJobs() {
  RAG_JOBS.forEach(job => {
    if (job.state !== 'running') return;
    if (job.cancel_requested) {
      job.state = 'cancelled';
      job.detail = 'отменено';
      job.elapsed = 3.0;
      return;
    }
    const age = Date.now() - job._startedAt;
    const stage = RAG_JOB_STAGES.filter(item => age < item[0])[0];
    if (stage) {
      job.stage = stage[1];
      job.stage_name = stage[2];
      job.percent = stage[3];
      job.done = stage[4];
      job.total = stage[5];
      job.detail = stage[6].replace('%s', job._filename);
      job.elapsed = Number((age / 1000).toFixed(1));
      return;
    }
    // Все этапы прошли — задача завершается и СОЗДАЁТ базу (или дописывает в неё).
    const target = job._baseId
      ? RAG.bases.filter(base => base.id === job._baseId)[0]
      : ragBase({ id: 'kb-0000000' + (RAG.bases.length + 1),
                  name: job.name || job._filename, chunks: 3, documents: 1,
                  chars_avg: 700, sections: 1 });
    if (job._baseId && target) {
      target.documents += 1;
      target.chunks += 3;
    } else if (target) {
      RAG.bases.push(target);
      RAG.enabled.push(target.id);
    }
    job.state = 'done';
    job.stage = 'save';
    job.percent = 100;
    job.detail = 'готово';
    job.elapsed = 3.0;
    job.base_id = target ? target.id : '';
    job.chunks = 3;
    job.documents = 1;
    job.result = {
      id: job.base_id, name: (target || {}).name || job.name,
      strategy_name: (target || {}).strategy_name || '',
      chunk_size: (target || {}).chunk_size || 0,
      overlap: (target || {}).overlap || 0,
      chunks: 3, documents: 1, chars_avg: 700,
      size_human: (target || {}).size_human || '10.0 КБ', failures: [],
    };
  });
}

// Проверке нужно уметь останавливать опрос задач индексации: он идёт по
// таймеру страницы и мешал бы следующим разделам (страница живёт одна на весь
// прогон). Останавливаем ШТАТНОЙ функцией страницы — своего таймера у проверки
// нет и быть не может.
function stopRagPollForCheck() {
  try { dom.window.eval('stopRagPoll()'); } catch (e) { /* страница ещё не готова */ }
}

function ragJobsPayload(activeOnly) {
  const jobs = activeOnly
    ? RAG_JOBS.filter(job => job.state === 'running')
    : RAG_JOBS.slice();
  return {
    jobs: jobs.slice().reverse(),
    active: RAG_JOBS.filter(job => job.state === 'running').length,
    summary: { jobs: [], active: 0, running: false, percent: 0 },
  };
}

function ragPayload() {
  const bases = RAG.bases.map(base => Object.assign({}, base, {
    enabled: RAG.enabled.indexOf(base.id) >= 0,
  }));
  const sum = key => bases.reduce((total, base) => total + Number(base[key] || 0), 0);
  return {
    bases: bases,
    enabled: RAG.enabled.slice(),
    project_id: workspace.active_task,
    counts: {
      bases: bases.length,
      enabled: bases.filter(base => base.enabled).length,
      chunks: sum('chunks'), documents: sum('documents'),
      chars_total: sum('chars_total'), est_tokens: sum('est_tokens'),
      size_bytes: sum('size_bytes'), size_human: '384.0 КБ',
    },
    strategies: RAG_STRATEGIES,
    settings: Object.assign({}, RAG.settings),
    defaults: { strategy: 'structure', chunk_size: 1000, overlap: 150 },
    limits: {
      chunk_size: { min: 100, max: 8000 }, overlap: { min: 0, max: 4000 },
      file_bytes: 268435456, file_size_human: '256.0 МБ',
      // Предел JSON-пути НАРОЧНО маленький: тогда проверка может показать
      // выбор транспорта на файле в пару килобайт, не выделяя 30 МБ в jsdom.
      json_file_bytes: 1024, json_file_size_human: '1.0 КБ',
      files_per_upload: 20, bases: 50, chunks_page: 10, chunks_page_max: 50,
    },
    formats: ['pdf', 'docx', 'md', 'txt'],
    embedding: {
      requested: 'auto', backend: 'sentence-transformers',
      backend_name: 'sentence-transformers (семантические эмбеддинги)',
      model: 'paraphrase-multilingual-MiniLM-L12-v2', dim: 384,
      sbert_installed: true, reason: '', cache_dir: 'data/rag/models',
      backends: [],
    },
    storage: { sqlite: true, json: true, vectors: 'numpy',
               vectors_name: 'numpy — счёт близости матрицей',
               vectors_reason: '', dir: 'data/rag' },
    dir: 'data/rag',
  };
}

// --- Периодические задачи: расписания на «сервере» ---------------------------
// Заглушка повторяет поведение сервера (app/ai/periodic.py, /api/agent/periodic):
// расписание есть у задачи, созданной кнопкой «Новая периодическая задача»; его
// период и остановку меняет POST /api/agent/periodic/{id}; в снимке задачи
// расписание видно как session.periodic, а опрос /api/agent/periodic отдаёт тот
// же снимок со свежим размером журнала (log_len) — по нему интерфейс понимает,
// что автозапуск дописал в задачу новое.
const PERIODIC = { tasks: {} };
const PERIODIC_LABELS = {
  '300': 'каждые 5 минут', '900': 'каждые 15 минут', '1800': 'каждые 30 минут',
  '3600': 'раз в час', '10800': 'каждые 3 часа', '21600': 'каждые 6 часов',
  '43200': 'каждые 12 часов', '86400': 'раз в сутки', '604800': 'раз в неделю',
};
const periodicBodies = [];    // тела POST /api/agent/sessions
const periodicUpdates = [];   // тела POST /api/agent/periodic/{id}

function periodicBrief(sessionId) {
  const meta = PERIODIC.tasks[sessionId];
  if (!meta) return null;
  return Object.assign({}, meta, { log_len: (logs[sessionId] || []).length });
}
function periodicPayload() {
  const tasks = Object.keys(PERIODIC.tasks).map(sessionId => ({
    task_id: workspace.active_task,
    session_id: sessionId,
    title: ((workspace.sessions || []).find(s => s.id === sessionId) || {}).title || 'Новая задача',
    periodic: periodicBrief(sessionId),
  }));
  return { tasks: tasks, now: '2026-01-01T00:00:00' };
}
// Снимок workspace: у периодических задач в списке — свежее расписание.
function workspacePayload() {
  return Object.assign({}, workspace, {
    sessions: (workspace.sessions || []).map(session => Object.assign({}, session,
      PERIODIC.tasks[session.id] ? { periodic: periodicBrief(session.id) } : {})),
  });
}

// Шаг «в полёте»: нужен, чтобы проверить мгновенную реакцию «Паузы».
let stepInFlight = false;// Задержка ответа на переключение задачи: нужна, чтобы проверить, что окно чата
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
  // ПЕРИОДИЧЕСКАЯ задача: её прогон заканчивается НЕ этапом «Готово» — цикл
  // заканчивается возвратом на первый шаг, план сохраняется (как на сервере,
  // task_state.cycle_done). Иначе интерфейс крутил бы повтор за повтором.
  const periodicSession = !!(PERIODIC.tasks[requestSession || workspace.active_session]);
  if (last && periodicSession) {
    setState(Object.assign({}, state, {
      stage: 'execution', base_stage: 'execution', current_step: 'step_1',
      step_index: 0, step_number: total ? 1 : 0, terminal: false, extra_stage: null,
      // Признак СЕРВЕРА: цикл пройден, ждём следующего повтора. По нему прогон
      // шагов останавливается (см. runStepChain).
      repeat_ready: true,
      expected_action: 'повтор по расписанию: ждём следующего повтора',
    }));
  } else if (last && CHECK_BLOCKED) {
    // Последний шаг выполнен, но проверку результата выполнить не удалось:
    // задача НЕ объявляется готовой — этап validation + признак check_blocked.
    setState(Object.assign({}, state, {
      stage: 'validation', base_stage: 'validation', current_step: 'check',
      step_index: total - 1, step_number: total, check_blocked: true,
      expected_action: 'повторить проверку или принять результат вручную',
    }));
  } else if (last) {
    setState(Object.assign({}, state, {
      stage: 'done', base_stage: 'done', current_step: '', expected_action: '',
      step_index: total - 1, step_number: total, terminal: true, extra_stage: null,
    }));
  } else {
    const next = state.step_index + 1;
    setState(Object.assign({}, state, {
      stage: 'execution', current_step: `step_${next + 1}`, step_index: next,
      step_number: next + 1, expected_action: 'выполнить: ' + state.steps[next].text,
      repeat_ready: false,   // цикл идёт — «ждём повтора» снимается
    }));
  }
  // Журнал задачи, в которой шаг запущен (как dialog["log"] на сервере): по нему
  // открытый диалог восстанавливается после фонового шага.
  const sessionId = requestSession || workspace.active_session;
  // Текст «итог задачи» — только у ОБЫЧНОЙ задачи: периодическая не завершается.
  const finalStep = last && !CHECK_BLOCKED && !periodicSession;
  logs[sessionId] = (logs[sessionId] || []).concat([
    { kind: 'assistant', text: finalStep ? 'Шаг выполнен: итог задачи.' : 'Шаг выполнен.' },
  ]);  const blockedEvent = CHECK_BLOCKED && last ? {
    type: 'error',
    text: '⚠️ Проверку результата выполнить не удалось: модель не ответила. Задачу '
      + 'готовой не объявляю — «▶ повторить проверку» запустит проверку снова, '
      + '«Принять вручную» завершит задачу без проверки.',
  } : null;
  return streamResponse([
    { type: 'state', state: snapshot() },
    { type: 'bot', text: finalStep ? 'Задача выполнена.' : 'Шаг выполнен.' },
    { type: 'state', state: snapshot() },
    // Уточнённый замер: автомат сделал служебный вызов (проверка результата) —
    // фронт обязан ЗАМЕНИТЬ замер запроса, а не добавить второй.
    { type: 'usage', usage: usage({ summary_requests: 1, summary_input: 7, summary_output: 3 }) },
    ...(blockedEvent ? [blockedEvent] : []),
    { type: 'done', usage: usage(), state: snapshot() },
  ]);
}

function makeFetch() {
  return async (url, options) => {
    const method = ((options && options.method) || 'GET').toUpperCase();
    calls.push(method + ' ' + url);
    // Тело бывает ДВУХ видов: строка JSON (обычные запросы) и сам файл
    // (потоковая загрузка). Разбирать второе как JSON нельзя — это Blob.
    const rawBody = options && options.body ? options.body : null;
    const body = typeof rawBody === 'string' ? JSON.parse(rawBody) : {};

    if (url === '/api/agent/workspace') return jsonResponse(workspacePayload());
    if (url === '/api/agent/periodic') return jsonResponse(periodicPayload());
    if (url.indexOf('/api/agent/periodic/') === 0 && method === 'POST') {
      // Правка расписания: как сервер — период (числом секунд) и включение-
      // выключение автозапуска. Ответ — снимок задач вместе с расписаниями.
      const id = decodeURIComponent(url.split('/')[4]);
      const meta = PERIODIC.tasks[id];
      if (!meta) return jsonResponse({ detail: 'Задача не периодическая' }, false);
      periodicUpdates.push(body);
      if (body && body.interval) {
        meta.interval = body.interval;
        meta.label = PERIODIC_LABELS[String(body.interval)] || 'каждые ' + body.interval + ' с';
      }
      if (body && typeof body.enabled === 'boolean') {
        meta.enabled = body.enabled;
        meta.when = body.enabled ? 'повтор через ' + meta.label : 'повтор остановлен';
      }
      return jsonResponse(Object.assign(workspacePayload(), { periodic: periodicPayload() }));
    }
    if (url === '/api/agent/history') return jsonResponse({
      messages: [], usage: [], summary: [], facts: {}, branches: {}, active_branch: null,
      log: logs[workspace.active_session] || [],
      usage: usageBySession[workspace.active_session] || [],
      session: { id: workspace.active_session, title: 'Диалог' }, state: snapshot(),
    });
    if (url === '/api/agent/profiles') return jsonResponse(workspace.profile);
    if (url.indexOf('/api/agent/invariants') === 0 && url.indexOf('?') > 0
        && method !== 'DELETE') {
      // Снимок правил КОНКРЕТНОЙ задачи (шестерёнка задачи в списке задач):
      // правило правится, даже если её диалог не открыт.
      const sessionId = decodeURIComponent(url.split('session_id=')[1] || '');
      return jsonResponse(invPayload(sessionId));
    }
    if (url === '/api/agent/invariants') {
      if (method === 'POST') {
        INV.newId = (INV.newId || 0) + 1;
        const entry = { id: 'i-new' + INV.newId, text: body.text };
        if (body.scope === 'task') {
          const session = workspace.active_session;
          INV.tasks[session] = (INV.tasks[session] || []).concat([entry]);
        } else {
          INV.project = INV.project.concat([entry]);
        }
        // Правила есть с обеих сторон — сервер проверяет пары и находит
        // противоречие (в заглушке — всегда, если INV.conflict).
        return jsonResponse(invPayload());
      }
      return jsonResponse(invPayload());
    }
    if (url === '/api/agent/invariants/choose' && method === 'POST') {
      // Выбор варианта из разбора: как сервер — берём вариант ПО НОМЕРУ из
      // последнего разбора (текст с фронта не принимаем).
      const item = (LAST_ANALYSIS.suggestions || [])[body.index] || {};
      return jsonResponse({ action: 'send', text: item.send || '' });
    }
    if (url === '/api/agent/invariants/check') return jsonResponse(invPayload());
    if (url === '/api/agent/invariants/conflicts/resolve') {
      INV.resolved[body.key] = body.winner;
      return jsonResponse(invPayload());
    }
    if (url.indexOf('/api/agent/invariants/') === 0 && method === 'DELETE') {
      const parts = url.split('?')[0].split('/');
      const scope = parts[4];
      const id = decodeURIComponent(parts[5]);
      // Задача может быть названа явно (?session_id=…): шестерёнка в списке
      // задач правит СВОЮ задачу, даже если её диалог не открыт.
      const explicit = url.indexOf('session_id=') > 0
        ? decodeURIComponent(url.split('session_id=')[1]) : '';
      if (scope === 'project') {
        INV.project = INV.project.filter(e => e.id !== id);
      } else {
        const session = explicit || workspace.active_session;
        INV.tasks[session] = (INV.tasks[session] || []).filter(e => e.id !== id);
      }
      return jsonResponse(invPayload(explicit || undefined));
    }
    if (url === '/api/agent/memory') {
      return jsonResponse({ task: { id: 't-1', name: 'Задача' }, working: [], long_term: [] });
    }
    if (url.indexOf('/api/agent/mcp') === 0) {
      if (method === 'POST') {
        // Применяется ПОЛНЫЙ набор галочек; неизвестный сервер не включается.
        const known = MCP.servers.map(s => s.id);
        MCP.enabled = (body.enabled || []).filter(id => known.indexOf(id) >= 0);
      }
      return jsonResponse(mcpPayload());
    }
    if (url.indexOf('/api/agent/rag') === 0) {
      if (url === '/api/agent/rag/upload' && method === 'POST') {
        // Загрузка базы: заглушка повторяет сервер — файлы приходят в base64,
        // разбираются, режутся на чанки и получают эмбеддинги. Числа чанков
        // считаются от РАЗМЕРА текста и размера чанка, как у настоящего
        // пайплайна, поэтому проверка видит согласованные метрики.
        const files = body.files || [];
        RAG.uploads.push(body);
        const size = Number(body.chunk_size) || RAG.settings.chunk_size || 1000;
        const strategy = body.strategy || RAG.settings.strategy || 'structure';
        let chars = 0;
        const sources = files.map(item => {
          const text = Buffer.from(String(item.content_base64 || ''), 'base64')
            .toString('utf8');
          chars += text.length;
          return {
            source: item.filename, format: 'Markdown',
            chunks: Math.max(1, Math.ceil(text.length / size)),
            chars: text.length, pages: 0, warning: '',
          };
        });
        const chunks = sources.reduce((total, item) => total + item.chunks, 0);
        const id = 'kb-0000000' + (RAG.bases.length + 1);
        const base = ragBase({
          id: id, name: body.name || (files[0] || {}).filename || 'База знаний',
          strategy: strategy,
          strategy_name: (RAG_STRATEGIES.filter(s => s.id === strategy)[0] || {}).name,
          chunk_size: size, overlap: Number(body.overlap) || 0,
          chunks: chunks, documents: files.length, chars_total: chars,
          chars_avg: chunks ? Math.round(chars / chunks) : 0,
          chars_min: size, chars_max: size, sections: 0,
          sources: sources, failures: [], fallback: '',
        });
        RAG.bases.push(base);
        RAG.enabled.push(id);
        RAG.settings = { strategy: strategy, chunk_size: size,
                         overlap: Number(body.overlap) || 0 };
        return jsonResponse({ base: Object.assign({}, base, { enabled: true }),
                              view: ragPayload() });
      }
      if (url.indexOf('/api/agent/rag/') === 0 && url.indexOf('/chunks') > 0) {
        const id = decodeURIComponent(url.split('/')[4]);
        return jsonResponse(ragChunksPayload(id, url));
      }
      if (url.indexOf('/api/agent/rag/jobs') === 0) {
        if (url.indexOf('/jobs/finish') > 0 && method === 'POST') {
          const job = RAG_JOBS.filter(item => item.id === body.job_id)[0];
          if (job && job.base_id && RAG.enabled.indexOf(job.base_id) < 0) {
            RAG.enabled.push(job.base_id);
          }
          return jsonResponse({ applied: Boolean(job), base_id: job ? job.base_id : '',
                                view: ragPayload() });
        }
        if (url.indexOf('/jobs/') > 0 && url.endsWith('/cancel') && method === 'POST') {
          const id = decodeURIComponent(url.split('/')[5]);
          const job = RAG_JOBS.filter(item => item.id === id)[0];
          if (job) job.cancel_requested = true;
          return jsonResponse({ job: job || null });
        }
        if (url.indexOf('/jobs/') > 0) {
          const id = decodeURIComponent(url.split('/')[5]);
          const job = RAG_JOBS.filter(item => item.id === id)[0];
          if (!job) return jsonResponse({ detail: 'Задача индексации не найдена' }, false);
          return jsonResponse({ job: job });
        }
        // Опрос списка: каждый опрос двигает задачи вперёд (как идёт время).
        ragAdvanceJobs();
        return jsonResponse(ragJobsPayload(url.indexOf('active=1') > 0));
      }
      if (url.indexOf('/api/agent/rag/upload/stream') === 0 && method === 'POST') {
        // Потоковая загрузка: тело — сам файл (не JSON!), параметры — в строке
        // запроса. Заглушка повторяет сервер: создаёт базу или дописывает файл.
        const query = new URLSearchParams(url.split('?')[1] || '');
        const filename = query.get('filename') || 'file.bin';
        const baseId = query.get('base_id') || '';
        const size = Number((options && options.body && options.body.size) || 0);
        RAG.streams.push({ filename: filename, baseId: baseId, size: size,
                           strategy: query.get('strategy') || '',
                           chunk_size: query.get('chunk_size') || '',
                           overlap: query.get('overlap') || '' });
        const job = ragStartJob(filename, baseId, query.get('name'));
        return jsonResponse({ job: job, view: ragPayload() });
      }
      if (url.indexOf('/api/agent/rag/') === 0 && method === 'DELETE') {
        const id = decodeURIComponent(url.split('/')[4]);
        RAG.bases = RAG.bases.filter(base => base.id !== id);
        RAG.enabled = RAG.enabled.filter(item => item !== id);
        return jsonResponse(ragPayload());
      }
      if (method === 'POST') {
        // Применяется ПОЛНЫЙ набор галочек (неизвестная база не включается) и
        // параметры разбиения — как на сервере они запоминаются на проекте.
        const known = RAG.bases.map(base => base.id);
        RAG.enabled = (body.enabled || []).filter(id => known.indexOf(id) >= 0);
        RAG.settings = {
          strategy: body.strategy || RAG.settings.strategy,
          chunk_size: Number(body.chunk_size) || RAG.settings.chunk_size,
          overlap: Number(body.overlap) || 0,
        };
      }
      return jsonResponse(ragPayload());
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
      periodicBodies.push(body);
      const id = 's-' + (workspace.sessions.length + 1) + 'x';
      const brief = { id: id, title: 'Новая задача' };
      if (body && body.periodic) {
        // Периодическая задача: расписание по умолчанию — раз в сутки (как на
        // сервере, см. periodic.DEFAULT_INTERVAL).
        PERIODIC.tasks[id] = {
          enabled: true, interval: 86400, label: 'раз в сутки',
          next_run: '2026-01-02T12:00:00', last_run: '', left: 86400,
          when: 'повтор через 24 часа', runs: 0, error: '', request: '',
          running: false, hold: '',
        };
      }
      workspace = Object.assign({}, workspace, {
        sessions: workspace.sessions.concat([brief]),
        active_session: id,
      });
      logs[id] = [];
      usageBySession[id] = [];
      setState({});                       // новая задача — автомат с нуля
      return jsonResponse(workspacePayload());
    }
    if (url.indexOf('/api/agent/sessions/') === 0 && method === 'DELETE') {
      // Удаление задачи: она уходит из списка вместе со своим журналом и
      // расписанием (как на сервере: задача — это и есть диалог).
      const id = decodeURIComponent(url.split('/')[4]);
      const rest = workspace.sessions.filter(s => s.id !== id);
      workspace = Object.assign({}, workspace, {
        sessions: rest,
        active_session: workspace.active_session === id
          ? ((rest[rest.length - 1] || {}).id || null)
          : workspace.active_session,
      });
      delete PERIODIC.tasks[id];
      return jsonResponse(workspacePayload());
    }
    if (url.indexOf('/api/agent/sessions/') === 0 && url.endsWith('/select')) {
      if (selectDelay) await sleep(selectDelay);
      const id = decodeURIComponent(url.split('/')[4]);
      workspace = Object.assign({}, workspace, { active_session: id });
      return jsonResponse(workspacePayload());
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
    if (url === '/api/agent/state/accept') {
      // «Принять вручную»: принимается только задача, у которой проверку
      // выполнить не удалось (как на сервере — иначе 400).
      if (state.stage !== 'validation' || !state.check_blocked) {
        return jsonResponse({ detail: 'Принимать вручную нечего' }, false);
      }
      setState(Object.assign({}, state, {
        stage: 'done', base_stage: 'done', current_step: '', expected_action: '',
        terminal: true, extra_stage: null, check_blocked: false,
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
      // Журнал чата задачи — как на сервере: реплика пользователя и всё, что
      // агент показал, попадают в dialog["log"]. Без этого перерисовка диалога
      // после запроса (loadActiveDialog) вернула бы СТАРЫЙ журнал и стёрла бы
      // только что показанные кликабельные варианты.
      const logSession = (body && body.session_id) || workspace.active_session;
      logs[logSession] = logs[logSession] || [];
      // РАЗБОР ЗАПРОСА ДО ПЛАНИРОВАНИЯ (как на сервере): если запрос требует
      // нарушить инвариант («веб»), агент отказывается и показывает варианты —
      // плана нет, шаги не выполняются.
      const requestText = String((body && body.content) || '');
      if (requestText && !body.continue_step) {
        logs[logSession].push({ kind: 'user', text: requestText });
      }
      if (!body.continue_step && /веб/i.test(requestText)) {
        LAST_ANALYSIS = {
          verdict: 'violation', kind: 'violation',
          message: '⛔ Запрос нарушает инвариант (правило, которое нарушать нельзя) — '
            + 'выполнять его не буду.\n\nВеб-приложение нарушает инвариант: разрешён '
            + 'только Kotlin и Android.',
          explanation: 'Веб-приложение нарушает инвариант: разрешён только Kotlin и Android.',
          suggestions: [
            { title: 'Нативное Android-приложение на Kotlin',
              details: 'Укладывается в стек: Kotlin + Compose.',
              send: 'Сделай нативное Android-приложение погоды на Kotlin', resolve: '' },
            { title: 'Kotlin Multiplatform с Android-таргетом',
              details: 'Общий код на Kotlin, нативный UI.',
              send: 'Сделай погодное приложение на Kotlin Multiplatform', resolve: '' },
          ],
        };
        logs[logSession].push({
          kind: 'suggestions', text: LAST_ANALYSIS.message, analysis: LAST_ANALYSIS,
        });
        return streamResponse([
          { type: 'state', state: snapshot() },
          { type: 'suggestions', text: LAST_ANALYSIS.message, analysis: LAST_ANALYSIS },
          { type: 'state', state: snapshot() },
          { type: 'done', usage: usage(), state: snapshot() },
        ]);
      }
      LAST_ANALYSIS = { verdict: '', kind: '', explanation: '', suggestions: [] };
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

const TRACE_HTML = HTML
  .replace(`            const list = (analysis.suggestions || []).filter(item => item && item.title);`,
    `            const list = (analysis.suggestions || []).filter(item => item && item.title);
            console.log("OPTIONS list=" + list.length + " raw=" + (analysis.suggestions || []).length + " wrap=" + (wrap ? wrap.className : "NULL"));`)
  .replace(`                box.appendChild(btn);`, `                box.appendChild(btn);
                console.log("OPT button appended", box.children.length);`)
  .replace(`            wrap.appendChild(box);`, `            console.log("BEFORE APPEND box.children=" + box.children.length + " forEachLen=" + list.length);
            wrap.appendChild(box);
            console.log("APPENDED box", wrap.querySelectorAll('.inv-option').length, wrap.className);`)
  .replace('let agentRootMessages = [];',
    'let agentRootMessages = [];\n        window.__trace = function (w) { console.log("ROOT@" + w, agentRootMessages.length); };')
  .replace('        function renderAgentDialog() {',
    '        function renderAgentDialog() { window.__trace("render");')
  .replace('        function drawAgentNode(role, text, analysis) {',
    '        function drawAgentNode(role, text, analysis) { if (analysis) console.log("DRAW analysis suggestions=" + (analysis.suggestions || []).length);')
  .replace('            } else {\n                const wrap = addMessage(\'bot\', text, true, true);',
    '            } else {\n                const wrap = addMessage(\'bot\', text, true, true); console.log("DRAW BOT inDom=" + document.body.contains(wrap) + " opts=" + wrap.querySelectorAll(".inv-option").length + " parent=" + (wrap.parentNode ? wrap.parentNode.className : "none"));')
  .replace('            agentRootMessages = (log.length ? log : msgs.map(m => ({',
    '            window.__trace("before set log=" + log.length + " msgs=" + msgs.length);\n            agentRootMessages = (log.length ? log : msgs.map(m => ({')
  .replace('                await loadWorkspace();\n            } catch (e) {',
    '                window.__trace("before loadWorkspace");\n                await loadWorkspace();\n                window.__trace("after loadWorkspace");\n            } catch (e) {');

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

// Значение CSS-свойства по каскаду: среди подходящих правил берём самое
// специфичное (при равенстве — последнее по порядку), как это делает браузер.
// Нужен потому, что jsdom возвращает по getComputedStyle значение базового
// правила (.modal-box), не разрешая каскад двух классов.
function selectorWeight(selector) {
  const classes = (selector.match(/\.[\w-]+/g) || []).length;
  const ids = (selector.match(/#[\w-]+/g) || []).length;
  const parts = (selector.match(/\[[^\]]+\]|:[\w-]+/g) || []).length;
  return ids * 100 + (classes + parts) * 10;
}

// Текст пояснения модалки инвариантов (первый .modal-hint самой модалки).
function invHintText() {
  const hints = $('invariants-modal').querySelectorAll('.modal-hint');
  // Переносы строк в разметке схлопываем: проверяем смысл, а не форматирование.
  return hints.length ? hints[0].textContent.replace(/\s+/g, ' ') : '';
}

// Секция модалки скрыта: атрибут hidden + поддержка в CSS ([hidden] снимаем
// display у grid-элемента) — проверяем оба условия (jsdom не всегда разрешает
// display для [hidden], поэтому стиль читаем из правил).
function sectionHidden(el) {
  if (el.hidden !== true) return false;
  const rule = declaredStyle(el, 'display');
  return rule === 'none';
}

function expectedWidthWins(el) {
  return parseFloat((declaredStyle(el, 'width') || '0').replace('px', '')) || 0;
}

function declaredStyle(el, prop) {
  let best = null;
  const sheets = dom.window.document.styleSheets;
  for (let i = 0; i < sheets.length; i++) {
    let rules;
    try { rules = sheets[i].cssRules; } catch (e) { continue; }
    for (let j = 0; j < rules.length; j++) {
      const rule = rules[j];
      if (!rule.selectorText || !rule.style) continue;
      const value = rule.style.getPropertyValue(prop);
      if (!value) continue;
      let matches = false;
      try { matches = el.matches(rule.selectorText); } catch (e) { matches = false; }
      if (!matches) continue;
      const weight = selectorWeight(rule.selectorText);
      if (best === null || weight > best.weight) best = { weight: weight, value: value };
    }
  }
  return best ? best.value : '';
}

async function click(el, ms) {
  el.dispatchEvent(new dom.window.MouseEvent('click', { bubbles: true }));
  await wait(ms === undefined ? 30 : ms);
}

// Запрос пользователя прямо через страницу (как будто ввели текст и нажали
// «Отправить»). Имя отличается от одноимённого помощника СТРАНИЦЫ (sendUserText),
// который используют варианты решения инвариантов.
async function sendRequest(text, ms) {
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
  await sendRequest('Сделай отчёт');
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
  await sendRequest('Сделай отчёт из трёх шагов');
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
  await sendRequest('Длинная задача на четыре шага');
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
  await sendRequest('Отчёт по продажам');   // уводим задачу из done (новая задача)
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
  // Журнал рисуется по kind, а не по роли в памяти: служебные строки — debug,
  // и они не превращаются в реплики пользователя (в т.ч. узел разбора инвариантов
  // «suggestions» — это сообщение АГЕНТА).
  check('служебные строки рисуются как debug, а не как реплики пользователя',
    q('#messages .msg.debug').some(el => el.textContent.includes('этап planning'))
    && !q('#messages .msg.user').some(el => el.textContent.includes('этап planning')),
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
  await sendRequest('моё сообщение в открытой задаче', 120);
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

  console.log('\n[L] Инварианты: шестерёнки проекта и задачи, модалка, противоречие');
  // Возвращаем проект и активный диалог: в разделе [K] workspace опустошали.
  workspace = {
    tasks: [{ id: 't-1', name: 'Задача' }], active_task: 't-1',
    sessions: [{ id: 's-1', title: 'Отчёт' }, { id: 's-2', title: 'Рецепт' }],
    active_session: 's-1',
    profile: workspace.profile,
  };
  INV.project = [{ id: 'i-p1', text: 'Только PostgreSQL' }];
  INV.tasks = { 's-1': [{ id: 'i-t1', text: 'Только MongoDB' }], 's-2': [] };
  INV.conflict = true;
  INV.resolved = {};
  setState({});
  await dom.window.eval('loadWorkspace()');
  await wait(40);

  // Шестерёнки: у проекта — рядом с карандашом и корзиной проекта, у каждой
  // задачи — рядом с её карандашом и корзиной (отдельной кнопки «Инварианты» нет).
  check('шестерёнка у проекта стоит среди иконок проекта',
    $('project-invariants').closest('.task-actions') !== null);
  // Она должна быть ВИДНА: атрибут hidden или display:none сделали бы настройку
  // недоступной (видимостью управляет только #task-block в обычном режиме).
  check('шестерёнка проекта действительно отрисована и видна',
    $('project-invariants').hidden === false
    && dom.window.getComputedStyle($('project-invariants')).display !== 'none'
    && $('project-invariants').className.indexOf('icon-btn') === 0,
    'display=' + dom.window.getComputedStyle($('project-invariants')).display
      + ', hidden=' + $('project-invariants').hidden
      + ', class=' + $('project-invariants').className);
  check('порядок иконок проекта: шестерёнка, MCP, RAG, карандаш, корзина',
    (function () {
      const actions = $('project-invariants').closest('.task-actions');
      return actions.children.length === 5
        && actions.children[0] === $('project-invariants')
        && actions.children[1] === $('project-mcp')
        && actions.children[2] === $('project-rag')
        && actions.children[3] === $('task-rename')
        && actions.children[4] === $('task-delete');
    })());
  check('отдельной кнопки «Инварианты» больше нет',
    dom.window.document.getElementById('invariants-btn') === null);
  check('у каждой задачи своя шестерёнка инвариантов',
    q('.session-item').length === 2
    && q('.session-item .session-invariants').length === 2,
    'шестерёнок задач: ' + q('.session-item .session-invariants').length);
  check('шестерёнка задачи стоит рядом с карандашом и корзиной',
    (function () {
      const actions = q('.session-item')[0].querySelector('.session-actions');
      return actions && actions.firstElementChild ===
        q('.session-item')[0].querySelector('.session-invariants')
        && actions.children.length === 3;
    })());
  // Тревожных меток от «проверок пар» больше нет: инварианты пишутся без
  // обращения к модели, противоречия выясняются в диалоге.
  check('шестерёнки без тревожных меток при записи правил',
    !$('project-invariants').classList.contains('conflict')
    && q('.session-item .session-invariants.conflict').length === 0);

  // Шестерёнка ЗАДАЧИ другой задачи: правила правятся, даже если диалог не открыт.
  await click(q('.session-item')[1].querySelector('.session-invariants'), 60);
  check('модалка открылась на правилах задачи', $('invariants-modal').hidden === false
    && $('inv-title').textContent === 'Инварианты задачи', $('inv-title').textContent);
  check('шестерёнка задачи запросила СВОЮ задачу',
    calls.some(c => c.indexOf('GET /api/agent/invariants?session_id=s-2') === 0),
    calls.slice(-2).join(' | '));
  check('правила этой задачи пусты',
    $('inv-task-list').textContent.includes('Правил задачи пока нет'),
    $('inv-task-list').textContent.slice(0, 80));
  // Области РАЗДЕЛЕНЫ: в модалке задачи секции проекта быть не должно.
  check('в модалке задачи нет секции правил проекта',
    sectionHidden($('inv-project-section')) && $('inv-project-list').textContent === '',
    'project-секция: hidden=' + $('inv-project-section').hidden
      + ', текст=' + JSON.stringify($('inv-project-list').textContent.slice(0, 40)));
  check('в модалке задачи правила проекта не упоминаются как редактируемые',
    $('inv-task-input').disabled === false && $('inv-task-add').disabled === false);
  check('модалка задачи — одна колонка правил',
    dom.window.getComputedStyle($('inv-cols')).gridTemplateColumns.split(' ').length === 1,
    dom.window.getComputedStyle($('inv-cols')).gridTemplateColumns);
  check('пояснение про инварианты есть в модалке задачи',
    $('inv-reminder').hidden === false
    && $('inv-reminder').textContent.includes('ЗАДАЧИ')
    && $('inv-reminder').textContent.includes('правила всего проекта задаются'));
  await click($('inv-close'), 40);

  // Шестерёнка ПРОЕКТА: правила всего проекта, правило задачи — только справка.
  await click($('project-invariants'), 60);
  check('модалка открылась на правилах проекта',
    $('inv-title').textContent === 'Инварианты проекта', $('inv-title').textContent);
  check('правила проекта перечислены',
    q('#inv-project-list .inv-item').length === 1
    && $('inv-project-list').textContent.includes('Только PostgreSQL'));
  // В модалке проекта секции правил задачи быть не должно.
  check('в модалке проекта нет секции правил задачи',
    sectionHidden($('inv-task-section')) && $('inv-task-list').textContent === '',
    'task-секция: hidden=' + $('inv-task-section').hidden
      + ', текст=' + JSON.stringify($('inv-task-list').textContent.slice(0, 40)));
  check('модалка проекта — одна колонка правил',
    dom.window.getComputedStyle($('inv-cols')).gridTemplateColumns.split(' ').length === 1,
    dom.window.getComputedStyle($('inv-cols')).gridTemplateColumns);
  check('пояснение объясняет, где правятся правила задачи',
    $('inv-reminder').textContent.includes('ПРОЕКТА')
    && $('inv-reminder').textContent.includes('шестерёнкой в списке задач'));
  // Пояснение должно быть в самой модалке (а не только в подсказке шестерёнки):
  // при открытых инвариантах задачи — какая это задача, при проекте — что здесь
  // правятся правила проекта.
  check('в модалке есть пояснение, что такое инварианты',
    invHintText().includes('не имеет права нарушить')
    && invHintText().includes('Одно поле — один инвариант')
    && invHintText().includes('бизнес-правила'),
    invHintText().slice(0, 90));
  check('в модалке нет блока «проверок пар» (их не существует)',
    dom.window.document.getElementById('inv-check') === null);

  // Модалка крупная: шире обычной и с двумя колонками правил.
  const modalBox = $('invariants-modal').querySelector('.inv-modal');
  const declaredWidth = parseFloat(
    (declaredStyle(modalBox, 'width') || '0').replace('px', '')) || 0;
  const plainWidth = parseFloat(
    (declaredStyle($('confirm-modal').querySelector('.modal-box'), 'width') || '0')
      .replace('px', '')) || 0;
  check('модалка инвариантов заметно больше обычной',
    declaredWidth >= plainWidth * 1.5, `ширина: ${declaredWidth} против ${plainWidth}`);
  check('ширина модалки важнее базовой ширины .modal-box',
    expectedWidthWins(modalBox) === declaredWidth,
    'каскад: ' + expectedWidthWins(modalBox));
  check('правила в модалке выводятся сеткой (своя область на всю ширину)',
    dom.window.getComputedStyle(modalBox.querySelector('.inv-cols')).display === 'grid');

  // Панель Workspace стала в полтора раза шире (260px → 390px).
  const layout = dom.window.getComputedStyle(dom.window.document.querySelector('.layout'));
  check('панель Workspace в полтора раза шире',
    layout.gridTemplateColumns.indexOf('390px') === 0, layout.gridTemplateColumns);

  check('запись правил не дёргает модель (счётчик вызовов не вырос)',
    !calls.some(c => c.indexOf('POST /api/agent/invariants/conflicts/resolve') === 0));

  // Добавление правила проекта из его шестерёнки.
  $('inv-project-input').value = 'Python 3.9';
  await click($('inv-project-add'), 60);
  check('правило проекта добавлено и поле очищено',
    $('inv-project-input').value === '' && INV.project.length === 2,
    'правил проекта: ' + INV.project.length);
  check('введённое правило видно в списке',
    $('inv-project-list').textContent.includes('Python 3.9'));

  // Правила задачи правятся из шестерёнки САМОЙ задачи (диалог может быть не открыт).
  await click($('inv-close'), 30);
  await click(q('.session-item')[0].querySelector('.session-invariants'), 60);
  check('в модалке задачи поле ввода доступно именно для правил задачи',
    $('inv-task-section').hidden === false && $('inv-task-input').disabled === false
    && $('inv-project-section').hidden === true);
  $('inv-task-input').value = 'Ответы только на русском';
  await click($('inv-task-add'), 60);
  check('правило добавлено в правила ЗАДАЧИ',
    (INV.tasks['s-1'] || []).length === 2
    && $('inv-task-list').textContent.includes('Ответы только на русском'),
    'правил задачи: ' + (INV.tasks['s-1'] || []).length);

  // Удаление правила (корзина рядом с правилом).
  const delButtons = q('#inv-task-list .inv-del');
  const taskBefore = (INV.tasks['s-1'] || []).length;
  await click(delButtons[0], 60);
  check('правило задачи удалено', (INV.tasks['s-1'] || []).length === taskBefore - 1,
    'правил задачи: ' + (INV.tasks['s-1'] || []).length);
  check('модалка осталась открытой после правки', $('invariants-modal').hidden === false);
  await click($('inv-close'), 40);
  check('модалка закрылась', $('invariants-modal').hidden === true);

  // Выключение агента: модалка закрывается, панель (и шестерёнки) убираются.
  await dom.window.eval('setAgentMode(false)');
  await wait(40);
  check('в обычном режиме панель Workspace скрыта', $('task-block').hidden === true);
  check('модалка инвариантов закрыта', $('invariants-modal').hidden === true);
  await dom.window.eval('setAgentMode(true)');
  await wait(60);
  check('при возврате в режим агента панель с шестерёнками снова видна',
    $('task-block').hidden === false);

  console.log('\n[M] Разбор запроса: отказ и кликабельные варианты');
  // Возвращаем проект и задачу: раздел [K] их опустошал.
  workspace = {
    tasks: [{ id: 't-1', name: 'Проект' }], active_task: 't-1',
    sessions: [{ id: 's-1', title: 'Погода' }], active_session: 's-1',
    profile: workspace.profile,
  };
  INV.project = [{ id: 'i-p1', text: 'Только Kotlin и Android, без веба' }];
  INV.tasks = { 's-1': [] };
  INV.conflict = false;
  INV.resolved = {};
  logs['s-1'] = [];                    // журнал этой задачи — как у новой задачи
  setState({});                        // планирование, без плана
  dom.window.eval('viewEpoch += 1');   // как при переключении задачи
  await dom.window.eval('loadWorkspace()');
  await wait(40);
  const optionsTsx = () => q('#messages .inv-options .inv-option');
  const lastOptions = () => {
    const boxes = q('#messages .inv-options');
    return boxes.length ? Array.from(boxes[boxes.length - 1].querySelectorAll('.inv-option')) : [];
  };

  // Запрос, который нарушает инвариант: агент отказывается и показывает варианты.
  await sendRequest('нужно веб-приложение погоды, открывается в браузере', 120);
  check('под сообщением появились кликабельные варианты', lastOptions().length === 2,
    'вариантов: ' + lastOptions().length);
  check('в отказе объяснено, что нарушено',
    q('#messages .msg.bot').slice(-1)[0].textContent.includes('нарушает инвариант'),
    q('#messages .msg.bot').slice(-1)[0].textContent.slice(0, 80));
  check('у варианта видно, какой запрос уйдёт при клике',
    lastOptions()[0].textContent.includes('отправить как запрос')
    && lastOptions()[0].textContent.includes('Kotlin'),
    lastOptions()[0].textContent.slice(0, 100));
  check('среди вариантов нет нарушающего правила',
    !lastOptions().some(btn => /веб-приложение/i.test(btn.textContent)),
    lastOptions().map(b => b.textContent.slice(0, 30)).join(' | '));
  check('варианты помечены как узлы диалога агента (убираются при выключении)',
    q('#messages .inv-options').every(box => box.closest('[data-agent]') !== null));
  // Отказ приходит ДО планирования: шагов в состоянии нет и автомат их не гонит.
  check('план при отказе не строится и шаги не идут',
    !(dom.window.eval('taskMachineState && taskMachineState.steps') || []).length,
    JSON.stringify(dom.window.eval('taskMachineState && taskMachineState.steps')));

  // Клик по варианту: текст берётся у сервера и уходит как новый запрос.
  const bodiesBefore = chatBodies.length;
  await click(lastOptions()[0], 200);
  check('клик по варианту спросил сервер, что делать с вариантом',
    calls.includes('POST /api/agent/invariants/choose'));
  const sentVariant = chatBodies.slice(bodiesBefore).some(
    b => b.continue_step !== true && String(b.content || '').includes('нативное Android-приложение'));
  check('вариант отправлен как запрос пользователя (обычным сообщением)', sentVariant,
    JSON.stringify(chatBodies.slice(bodiesBefore).map(
      b => ({ text: String(b.content || '').slice(0, 30), cs: b.continue_step }))));
  await wait(150);
  check('текст варианта виден в диалоге как реплика пользователя',
    q('#messages .msg.user').some(el => el.textContent.includes('нативное Android-приложение')),
    q('#messages .msg.user').map(el => el.textContent.slice(0, 30)).join(' | '));

  // Варианты переживают переключение диалога: они лежат в журнале чата.
  dom.window.eval('resetAgentDialogs()');
  logs['s-1'] = [{
    kind: 'suggestions',
    text: '⛔ Запрос нарушает инвариант — выполнять его не буду.',
    analysis: {
      kind: 'violation', explanation: 'Веб запрещён',
      suggestions: [
        { title: 'Нативное Android-приложение', details: 'В стеке',
          send: 'Сделай Android-приложение', resolve: '' },
        { title: 'Kotlin Multiplatform', details: 'Общий код',
          send: 'Сделай на Kotlin MPP', resolve: '' },
      ],
    },
  }];
  await dom.window.eval('loadActiveDialog()');
  await wait(60);
  check('после перерисовки диалога варианты снова кликабельны', lastOptions().length === 2,
    'вариантов: ' + lastOptions().length);

  // Конфликт правила задачи с правилом проекта: приоритет у проекта, агент
  // отказывается и даёт альтернативы — выбора «какое правило главнее» нет.
  INV.conflict = true;
  INV.resolved = {};
  LAST_ANALYSIS = {
    verdict: 'violation', kind: 'violation',
    message: '⛔ Запрос нарушает инвариант проекта — выполняю его не буду.',
    explanation: 'Правило проекта запрещает веб, правило задачи его просит: '
      + 'действует правило проекта.',
    suggestions: [
      { title: 'Нативное Android-приложение', details: 'В стеке проекта',
        send: 'Сделай Android-приложение погоды на Kotlin' },
      { title: 'План без веба', details: 'Только Android',
        send: 'Спланируй Android-приложение погоды' },
    ],
  };
  // Такой разбор приходит в журнале задачи — рисуем диалог из него.
  logs['s-1'] = [{
    kind: 'suggestions',
    text: LAST_ANALYSIS.message,
    analysis: LAST_ANALYSIS,
  }];
  await dom.window.eval('loadActiveDialog()');
  await wait(60);
  check('варианты-альтернативы показаны кликабельными',
    lastOptions().length === 2, 'вариантов: ' + lastOptions().length);
  check('выбора «главнее проект/задача» в интерфейсе нет',
    q('#messages .inv-option.resolve').length === 0);
  check('у альтернативы подписан запрос, который уйдёт',
    lastOptions()[0].textContent.includes('отправить как запрос')
    && lastOptions()[0].textContent.includes('Android'),
    lastOptions()[0].textContent.slice(0, 90));

  // Клик по альтернативе: обычный запрос, который правила не нарушает.
  const bodiesBeforeAlt = chatBodies.length;
  await click(lastOptions()[0], 200);
  await wait(150);
  check('альтернатива отправлена как запрос пользователя',
    chatBodies.slice(bodiesBeforeAlt).some(b => !b.continue_step
      && String(b.content || '').includes('Android-приложение')),
    JSON.stringify(chatBodies.slice(bodiesBeforeAlt).map(b => String(b.content || '').slice(0, 30))));

  console.log('\n[N] Панель «Токены задачи»: пустые замеры, разбивка, лимит');
  await dom.window.eval('setAgentMode(true)');
  await wait(60);
  // Замеры как их отдаёт сервер: обычный шаг со служебной проверкой результата
  // (лимит «Длина» превышен) и ПУСТОЙ замер — такой приходит в служебных ветках
  // (пауза, ошибка, «введите сообщение»).
  dom.window.eval('agentUsage = ' + JSON.stringify([
    { requests: 2, input: 900, output: 90, summary_requests: 1, summary_input: 700,
      summary_output: 70, failed_requests: 0, limit: 500, overflow: true,
      cost_rub: 0.31, service: { review: { requests: 1, input: 700, output: 70, failed: 0 } } },
    {}
  ]) + '; agentPanelView = "tokens"; renderAgentPanel();');
  await wait(30);
  const panelRows = (id) => Array.from($(id).querySelectorAll('tr'))
    .map(tr => Array.from(tr.querySelectorAll('td')).map(td => td.textContent));
  const stepRow = panelRows('agent-stats-body')[0];
  const totalRows = panelRows('agent-total-body');
  const chartLabel = $('agent-chart').textContent.replace(/\s+/g, ' ');
  check('пустой замер не попадает в расчёт «Ответ шага»',
    stepRow && stepRow[1] === '1', JSON.stringify(stepRow));
  check('в «Ответ шага» видны лимит «Длина» и переполнение',
    stepRow && stepRow[5] === '500'
    && $( 'agent-stats-body').querySelector('td.agent-overflow') !== null,
    JSON.stringify(stepRow));
  check('итог считается по обращениям к модели, а не по записям',
    totalRows[0] && totalRows[0][1] === '2', JSON.stringify(totalRows[0]));
  check('служебные вызовы разложены по видам',
    totalRows.some(r => r[0].indexOf('проверка результата') > 0),
    JSON.stringify(totalRows.map(r => r[0])));
  check('диаграмма считает запросы, а не записи массива',
    chartLabel.indexOf('запросов: 1') >= 0, chartLabel);
  check('под таблицей видна стоимость запросов (оценка по тарифам)',
    $('agent-cost').textContent.indexOf('за шаг') > 0
    && $('agent-cost').textContent.indexOf('за диалог') > 0,
    $('agent-cost').textContent);

  console.log('\n[N2] Проверка не удалась: задача не готова, прогон остановлен');
  // «Сервер»: проверку результата выполнить не удалось (модель не ответила).
  CHECK_BLOCKED = true;
  setState({});
  await wait(40);
  await sendRequest('Сделай отчёт');
  await click($('tm-confirm'), 40);
  for (let i = 0; i < 100 && state.stage !== 'validation'; i++) await wait(20);
  check('задача осталась на этапе «Проверка», а не ушла в «Готово»',
    state.stage === 'validation', state.stage);
  check('кнопка «Принять вручную» показана', $('tm-accept').hidden === false);
  check('в полосе видно, что проверка не выполнена',
    $('tm-redo').hidden === false
    && $('tm-redo').textContent.indexOf('проверка не выполнена') >= 0,
    $('tm-redo').textContent);
  check('подсказка предлагает повторить проверку',
    $('tm-idle').hidden === false
    && $('tm-idle').textContent.indexOf('повторить проверку') >= 0,
    $('tm-idle').textContent);
  check('в чате объяснено, почему задача не объявлена готовой',
    q('#messages .msg.bot').some(el => el.textContent.indexOf(
      'Проверку результата выполнить не удалось') >= 0),
    q('#messages .msg.bot').map(el => el.textContent.slice(0, 60)).join(' | '));
  // Прогон остановлен: повторять проверку по кругу автомат не должен (иначе это
  // лишние вызовы LLM до предохранителя цепочки).
  const chatsAfterBlock = chatBodies.length;
  await wait(600);
  check('прогон остановлен: новых запросов к модели нет',
    chatBodies.length === chatsAfterBlock,
    chatsAfterBlock + ' → ' + chatBodies.length);
  // «Принять вручную» закрывает задачу без проверки.
  await click($('tm-accept'), 60);
  check('«Принять вручную» завершает задачу', state.stage === 'done', state.stage);
  check('кнопка «Принять вручную» скрыта после завершения', $('tm-accept').hidden === true);
  CHECK_BLOCKED = false;

  console.log('\n[O] Экспертная статистика: счётчики приходят с сервера');
  dom.window.eval('setExpertMode(false); setAgentMode(false); setExpertMode(true)');
  dom.window.eval('applyStats({ direct: { correct: 2, incorrect: 1 }, group: { correct: 0, incorrect: 3 } })');
  await wait(30);
  const statsRows = () => Array.from($('stats-body').querySelectorAll('tr'))
    .map(tr => Array.from(tr.querySelectorAll('td')).map(td => td.textContent));
  check('статистика рисуется из снимка сервера',
    JSON.stringify(statsRows()[0]) === JSON.stringify(['Прямой ответ', '2', '1'])
    && JSON.stringify(statsRows()[3]) === JSON.stringify(['Экспертная группа', '0', '3']),
    JSON.stringify(statsRows()));
  // Переключение режима больше НЕ обнуляет статистику (она живёт на сервере).
  dom.window.eval('setExpertMode(false)');
  dom.window.eval('setExpertMode(true)');
  await wait(30);
  check('переключение режима статистику не обнуляет',
    JSON.stringify(statsRows()[0]) === JSON.stringify(['Прямой ответ', '2', '1']),
    JSON.stringify(statsRows()));

  console.log('\n[P] MCP: кнопка у проекта, список серверов, «применить»');
  // Возвращаемся в режим агента: кнопка «MCP» живёт в блоке проекта.
  dom.window.eval('setExpertMode(false); setAgentMode(true)');
  await wait(60);
  MCP.enabled = [];
  await dom.window.eval('loadMcp(false)');
  await wait(40);

  check('кнопка «MCP» стоит рядом с шестерёнкой проекта',
    $('project-mcp') !== null && $('project-mcp').closest('.task-actions') !== null
    && $('project-mcp').previousElementSibling === $('project-invariants'));
  check('кнопка «MCP» действительно видна',
    $('project-mcp').hidden === false
    && dom.window.getComputedStyle($('project-mcp')).display !== 'none',
    'display=' + dom.window.getComputedStyle($('project-mcp')).display);
  check('диалог MCP закрыт до нажатия', $('mcp-modal').hidden === true);
  check('выключенный MCP не помечает кнопку',
    $('project-mcp').classList.contains('on') === false,
    $('project-mcp').className + ' / ' + $('project-mcp').title);

  await click($('project-mcp'), 60);
  check('нажатие открывает диалог со списком MCP', $('mcp-modal').hidden === false);
  // Ожидания ВЫЧИСЛЯЮТСЯ из фикстура (а он собран из реестра проекта): добавили
  // или убрали сервер — проверка это увидит, а не завалится на старом числе.
  const items = q('#mcp-list .mcp-item');
  check('в диалоге перечислены все серверы проекта',
    items.length === MCP.servers.length,
    'на экране: ' + items.length + ', в реестре: ' + MCP.servers.length);
  const names = q('#mcp-list .mcp-name').map(el => el.textContent);
  check('у каждого сервера есть название',
    names.join('|') === MCP.servers.map(s => s.name).join('|'), names.join('|'));
  const descs = q('#mcp-list .mcp-desc').map(el => el.textContent);
  check('у каждого сервера есть краткое описание',
    descs.length === MCP.servers.filter(s => s.description).length
    && descs.every(text => text.length > 10), JSON.stringify(descs));
  const toolNames = MCP.servers.reduce(
    (all, server) => all.concat(server.tools.map(tool => tool.name)), []);
  check('серверы показаны с инструментами',
    q('#mcp-list .mcp-tools li').length === toolNames.length
    && toolNames.every(name => $('mcp-list').textContent.indexOf(name) >= 0),
    'на экране: ' + q('#mcp-list .mcp-tools li').length
      + ', в фикстуре: ' + toolNames.length);
  // Сверка фикстура с реестром: добавили или убрали сервер — видно здесь, и
  // дальше все ожидания считаются от этого же списка.
  check('в фикстуре ровно те серверы, что в реестре',
    MCP.servers.map(s => s.id).join('|') === REGISTRY.map(s => s.id).join('|')
    && MCP.servers.map(s => s.transport).join('|')
      === REGISTRY.map(s => s.transport).join('|'),
    MCP.servers.map(s => s.id + ':' + s.transport).join(', '));
  const remote = MCP.servers.filter(s => s.available && s.transport === 'http')[0];
  if (remote) {
    check('удалённый сервер показывает свои инструменты',
      remote.tools.length > 0
      && remote.tools.every(tool => $('mcp-list').textContent.indexOf(tool.name) >= 0)
      && $('mcp-list').textContent.indexOf(
        'доступен · инструментов: ' + remote.tools.length) >= 0,
      remote.id);
  } else {
    // Реестр без своих серверов на VPS (или единственный такой показан
    // недоступным) — проверять нечего, и падать из-за этого нельзя.
    console.log('  --   доступных серверов на VPS в фикстуре нет — '
      + 'проверка их инструментов пропущена');
  }
  const down = MCP.servers.filter(s => !s.available);
  if (down.length) {
    check('недоступный сервер показан причиной, а не молчанием',
      $('mcp-list').textContent.indexOf('недоступен') >= 0
      && down.every(s => $('mcp-list').textContent.indexOf('Причина: ' + s.error) >= 0),
      $('mcp-list').textContent.slice(0, 120));
  } else {
    console.log('  --   недоступных серверов в фикстуре нет — причина не проверяется');
  }
  check('у серверов есть галочки',
    q('#mcp-list input[type=checkbox]').length === MCP.servers.length);
  check('галочки сняты, пока MCP выключен',
    q('#mcp-list input[type=checkbox]').every(box => box.checked === false));

  // Включаем ДВА доступных сервера, применяем: на сервер уходит полный набор.
  const boxes = q('#mcp-list input[type=checkbox]');
  const indexOfServer = id => MCP.servers.findIndex(server => server.id === id);
  const picked = MCP.servers.filter(s => s.available).map(s => s.id).slice(0, 2);
  picked.forEach(id => { boxes[indexOfServer(id)].checked = true; });
  const mcpPostsBefore = calls.filter(c => c === 'POST /api/agent/mcp').length;
  await click($('mcp-apply'), 80);
  check('«применить» отправил набор на сервер',
    calls.filter(c => c === 'POST /api/agent/mcp').length === mcpPostsBefore + 1,
    calls.slice(-3).join(' | '));
  check('на сервер ушёл полный список галочек',
    JSON.stringify(MCP.enabled) === JSON.stringify(picked),
    JSON.stringify(MCP.enabled));
  check('после «применить» диалог закрывается', $('mcp-modal').hidden === true);
  check('включённый MCP помечает кнопку проекта',
    $('project-mcp').classList.contains('on') === true
    && $('project-mcp').title.indexOf(
      'включено ' + picked.length + ' из ' + MCP.servers.length) > 0,
    $('project-mcp').className + ' / ' + $('project-mcp').title);
  check('в чате сказано, что MCP включён',
    q('#messages .msg.bot').some(el => el.textContent.indexOf('MCP включён') >= 0));

  // Повторное открытие показывает сохранённые галочки (снимок с сервера).
  await click($('project-mcp'), 60);
  const checkedNow = q('#mcp-list input[type=checkbox]').map(box => box.checked);
  check('повторное открытие показывает включённые серверы',
    JSON.stringify(checkedNow)
      === JSON.stringify(MCP.servers.map(s => picked.indexOf(s.id) >= 0)),
    JSON.stringify(checkedNow) + ' при выбранных ' + JSON.stringify(picked));

  // Выключаем всё: набор снова пуст, кнопка без пометки.
  q('#mcp-list input[type=checkbox]').forEach(box => { box.checked = false; });
  await click($('mcp-apply'), 80);
  check('выключение всех серверов сохраняется',
    MCP.enabled.length === 0, JSON.stringify(MCP.enabled));
  check('пустой набор снимает пометку кнопки',
    $('project-mcp').classList.contains('on') === false,
    $('project-mcp').className + ' / ' + $('project-mcp').title);

  // В обычном режиме кнопка «MCP» скрыта вместе с блоком проекта, а диалог
  // (если был открыт) закрывается.
  await click($('project-mcp'), 60);
  dom.window.eval('setAgentMode(false)');
  await wait(60);
  check('вне режима агента диалог MCP закрыт', $('mcp-modal').hidden === true);
  check('кнопка «MCP» скрыта вместе с блоком проекта', $('task-block').hidden === true);
  dom.window.eval('setAgentMode(true)');
  await wait(40);

  // ---------------------------------------------------------------------
  // [S] БАЗЫ ЗНАНИЙ (RAG): кнопка рядом с MCP, список баз с метриками,
  // переключатели, загрузка своей базы со стратегией и размерами чанка.
  console.log('\n[S] RAG: базы знаний — список, переключатели, загрузка, стратегии');
  dom.window.eval('setExpertMode(false); setAgentMode(true)');
  await wait(60);
  RAG.enabled = [];
  RAG.bases = [ragBase({}), ragBase({
    id: 'kb-00000002', name: 'Регламенты', strategy: 'fixed',
    strategy_name: 'Фиксированный размер', chunk_size: 500, overlap: 50,
    chunks: 17, documents: 1, chars_avg: 470, chars_min: 300, chars_max: 500,
    sections: 0, size_bytes: 131072, size_human: '128.0 КБ', share: 0.4,
    fallback: 'модель недоступна: нет сети',
    sources: [{ source: 'rules.docx', format: 'Word (DOCX)', chunks: 17,
                chars: 7990, pages: 0, warning: '' }],
    failures: ['bad.pdf — PDF защищён паролем'],
  })];
  RAG.uploads = [];
  await dom.window.eval('loadRag(false)');
  await wait(40);

  check('кнопка «RAG» стоит рядом с кнопкой «MCP»',
    $('project-rag') !== null
    && $('project-rag').previousElementSibling === $('project-mcp')
    && $('project-rag').closest('.task-actions') !== null);
  check('кнопка «RAG» действительно видна',
    $('project-rag').hidden === false
    && dom.window.getComputedStyle($('project-rag')).display !== 'none',
    'display=' + dom.window.getComputedStyle($('project-rag')).display);
  check('диалог баз знаний закрыт до нажатия', $('rag-modal').hidden === true);
  check('выключенные базы не помечают кнопку',
    $('project-rag').classList.contains('on') === false,
    $('project-rag').className + ' / ' + $('project-rag').title);

  await click($('project-rag'), 80);
  check('нажатие открывает диалог со списком баз знаний',
    $('rag-modal').hidden === false);

  const ragItems = q('#rag-list .rag-item');
  check('в диалоге перечислены все базы профиля',
    ragItems.length === RAG.bases.length,
    'на экране: ' + ragItems.length + ', в фикстуре: ' + RAG.bases.length);
  check('у каждой базы есть название',
    q('#rag-list .rag-name').map(el => el.textContent).join('|')
      === RAG.bases.map(base => base.name).join('|'),
    q('#rag-list .rag-name').map(el => el.textContent).join('|'));
  check('у каждой базы есть галочка-переключатель',
    q('#rag-list input[type=checkbox]').length === RAG.bases.length);
  check('галочки сняты, пока базы выключены',
    q('#rag-list input[type=checkbox]').every(box => box.checked === false));

  // Метрики: стратегия, чанки, средний размер, вес — по КАЖДОЙ базе, своими
  // числами (две базы в фикстуре нарочно разные).
  const firstText = (ragItems[0] ? ragItems[0].textContent : '').replace(/\u00a0/g, ' ');
  const secondText = (ragItems[1] ? ragItems[1].textContent : '').replace(/\u00a0/g, ' ');
  check('в строке базы названа стратегия разбиения',
    firstText.indexOf(RAG.bases[0].strategy_name) >= 0
    && secondText.indexOf(RAG.bases[1].strategy_name) >= 0,
    firstText.slice(0, 80));
  check('в строке базы показано число чанков',
    firstText.indexOf(String(RAG.bases[0].chunks)) >= 0);
  check('в строке базы показан средний размер чанка',
    firstText.indexOf('Средний чанк') >= 0
    && firstText.indexOf(String(RAG.bases[0].chars_avg)) >= 0);
  check('в строке базы показан вес базы и её доля',
    firstText.indexOf(RAG.bases[0].size_human) >= 0
    && firstText.indexOf('60% баз') >= 0, firstText.slice(0, 120));
  check('в строке базы показано число документов и разделов',
    firstText.indexOf('Документов') >= 0 && firstText.indexOf('Разделов') >= 0);
  check('в строке базы показана размерность эмбеддингов',
    firstText.indexOf(String(RAG.bases[0].dim)) >= 0);
  check('документы базы перечислены с форматом и числом чанков',
    firstText.indexOf('guide.md') >= 0 && firstText.indexOf('24 000') >= 0,
    firstText.slice(0, 200));
  check('предупреждение по документу видно (скан без текстового слоя)',
    firstText.indexOf('нет текстового слоя') >= 0, firstText.slice(-160));
  check('непрочитанный файл показан причиной, а не молчанием',
    secondText.indexOf('PDF защищён паролем') >= 0, secondText.slice(-120));
  check('база, посчитанная запасным бэкендом, помечена',
    secondText.indexOf('запасным офлайн-бэкендом') >= 0);

  // Настройки разбиения: стратегии и пределы приходят С СЕРВЕРА.
  const strategyOptions = Array.from($('rag-strategy').options).map(o => o.value);
  check('в списке стратегий — обе стратегии с сервера',
    JSON.stringify(strategyOptions) === JSON.stringify(RAG_STRATEGIES.map(s => s.id)),
    JSON.stringify(strategyOptions));
  check('выбрана стратегия из настроек проекта',
    $('rag-strategy').value === RAG.settings.strategy, $('rag-strategy').value);
  check('под стратегией видно её описание',
    $('rag-strategy-hint').textContent.length > 10,
    $('rag-strategy-hint').textContent);
  check('размер чанка подставлен из настроек проекта',
    $('rag-size').value === String(RAG.settings.chunk_size), $('rag-size').value);
  check('перекрытие подставлено из настроек проекта',
    $('rag-overlap').value === String(RAG.settings.overlap), $('rag-overlap').value);
  check('пределы размеров взяты с сервера',
    $('rag-size').min === '100' && $('rag-size').max === '8000',
    $('rag-size').min + '…' + $('rag-size').max);

  // ВЫРАВНИВАНИЕ НАСТРОЕК: «размер чанка» и «перекрытие» обязаны стоять
  // симметрично. Раскладку задаёт СЕТКА (jsdom считает computed style), поэтому
  // проверяется не «как выглядит», а чем это обеспечено: один ряд, равные
  // колонки, выравнивание по нижнему краю и одинаковая подпись-обёртка.
  const paramsRow = $('rag-size').closest('.rag-row');
  check('настройки разбиения стоят в одном ряду',
    paramsRow !== null && paramsRow === $('rag-overlap').closest('.rag-row')
    && paramsRow === $('rag-strategy').closest('.rag-row'),
    paramsRow ? paramsRow.className : 'ряд не найден');
  const paramsStyle = paramsRow ? dom.window.getComputedStyle(paramsRow) : {};
  check('ряд настроек разложен сеткой, а не flex-переносами',
    paramsStyle.display === 'grid', String(paramsStyle.display));
  check('поля ряда выровнены по нижнему краю (подпись не сдвигает поле)',
    paramsStyle.alignItems === 'end', String(paramsStyle.alignItems));
  // Колонки разбираем с учётом скобок: «minmax(0, 1fr)» — одна колонка, а не
  // две (иначе проверка ломалась бы на самой записи правила).
  const columns = String(paramsStyle.gridTemplateColumns || '')
    .match(/minmax\([^)]*\)|[^\s]+/g) || [];
  const fixed = columns
    .map(value => (/^[\d.]+px$/.test(value) ? parseFloat(value) : 0))
    .filter(value => value > 0);
  check('ряд настроек — три колонки, из них две РАВНЫЕ фиксированные',
    columns.length === 3 && fixed.length === 2 && fixed[0] === fixed[1],
    columns.length + ' колонок: ' + JSON.stringify(paramsStyle.gridTemplateColumns));
  check('оба числовых поля одинаковой ширины (нет narrow/wide у одного из них)',
    $('rag-size').closest('.rag-field').className
      === $('rag-overlap').closest('.rag-field').className
    && $('rag-size').closest('.rag-field').className === 'rag-field',
    $('rag-size').closest('.rag-field').className + ' / '
      + $('rag-overlap').closest('.rag-field').className);
  const labelHeights = ['rag-strategy', 'rag-size', 'rag-overlap'].map(id => {
    const label = $(id).closest('.rag-field').querySelector('label');
    return label ? dom.window.getComputedStyle(label).minHeight : '';
  });
  check('у каждого поля есть подпись, и высота подписей одинакова',
    ['rag-strategy', 'rag-size', 'rag-overlap'].every(id => {
      const label = $(id).closest('.rag-field').querySelector('label');
      return label && label.textContent.trim().length > 3;
    })
    // Высота подписи зарезервирована под ДВЕ строки и у всех полей одна: иначе
    // односложная подпись «вытягивала» своё поле вверх относительно соседнего.
    && labelHeights.every(height => parseFloat(height) >= 20)
    && new Set(labelHeights).size === 1,
    labelHeights.join(' | '));
  check('в подписи числовых полей названы единицы измерения',
    $('rag-size').closest('.rag-field').querySelector('label')
      .textContent.indexOf('символов') >= 0
    && $('rag-overlap').closest('.rag-field').querySelector('label')
      .textContent.indexOf('символов') >= 0);

  // Включаем одну базу и меняем стратегию с размерами — «применить».
  const ragBoxes = q('#rag-list input[type=checkbox]');
  ragBoxes[0].checked = true;
  $('rag-strategy').value = 'fixed';
  $('rag-size').value = '600';
  $('rag-overlap').value = '90';
  const ragPostsBefore = calls.filter(c => c === 'POST /api/agent/rag').length;
  await click($('rag-apply'), 80);
  check('«применить» отправил набор на сервер',
    calls.filter(c => c === 'POST /api/agent/rag').length === ragPostsBefore + 1,
    calls.slice(-3).join(' | '));
  check('на сервер ушёл полный список галочек',
    JSON.stringify(RAG.enabled) === JSON.stringify([RAG.bases[0].id]),
    JSON.stringify(RAG.enabled));
  check('стратегия и размеры чанка ушли вместе с набором',
    RAG.settings.strategy === 'fixed' && RAG.settings.chunk_size === 600
    && RAG.settings.overlap === 90, JSON.stringify(RAG.settings));
  check('после «применить» диалог закрывается', $('rag-modal').hidden === true);
  check('включённая база помечает кнопку проекта',
    $('project-rag').classList.contains('on') === true
    && $('project-rag').title.indexOf('включено 1 из ' + RAG.bases.length) > 0,
    $('project-rag').className + ' / ' + $('project-rag').title);
  check('в чате сказано, что базы знаний включены',
    q('#messages .msg.bot').some(el => el.textContent.indexOf('Базы знаний включены') >= 0));

  // Повторное открытие показывает сохранённые галочки (снимок с сервера).
  await click($('project-rag'), 80);
  check('повторное открытие показывает включённые базы',
    JSON.stringify(q('#rag-list input[type=checkbox]').map(box => box.checked))
      === JSON.stringify(RAG.bases.map(base => RAG.enabled.indexOf(base.id) >= 0)),
    JSON.stringify(q('#rag-list input[type=checkbox]').map(box => box.checked)));

  // ---------------------------------------------------------------------
  // ПРОСМОТР ЧАНКОВ: у каждой базы есть кнопка, диалог показывает текст
  // чанков с адресом, работает фильтр по документу, поиск и «показать ещё».
  check('в строке базы есть кнопка просмотра чанков',
    q('#rag-list .rag-view').length === RAG.bases.length,
    'кнопок: ' + q('#rag-list .rag-view').length);
  check('просмотр чанков закрыт до нажатия', $('rag-chunks-modal').hidden === true);
  await click(q('#rag-list .rag-view')[0], 120);
  check('кнопка «чанки» открывает диалог просмотра', $('rag-chunks-modal').hidden === false);
  check('в заголовке просмотра названа база',
    $('rag-chunks-name').textContent === RAG.bases[0].name,
    $('rag-chunks-name').textContent);
  check('в шапке просмотра видны стратегия и размер чанка',
    $('rag-chunks-meta').textContent.indexOf(RAG.bases[0].strategy_name) >= 0
    && $('rag-chunks-meta').textContent.indexOf('чанк') >= 0,
    $('rag-chunks-meta').textContent);
  check('на первой странице показана ровно страница чанков',
    q('#rag-chunks-list .rag-chunk').length === 10,
    'чанков: ' + q('#rag-chunks-list .rag-chunk').length);
  check('в строке чанка есть номер, идентификатор и источник',
    q('#rag-chunks-list .rag-chunk-num')[0].textContent.indexOf('№ 1') >= 0
    && q('#rag-chunks-list .rag-chunk-chip.id')[0].textContent.length > 4
    && q('#rag-chunks-list .rag-chunk')[0].textContent.indexOf('guide.md') >= 0,
    q('#rag-chunks-list .rag-chunk')[0].textContent.slice(0, 120));
  check('в строке чанка виден раздел и границы символов',
    q('#rag-chunks-list .rag-chunk')[0].textContent.indexOf('раздел:') >= 0
    && q('#rag-chunks-list .rag-chunk')[0].textContent.indexOf('символы') >= 0);
  check('текст чанка показан как есть (переносы не потеряны)',
    q('#rag-chunks-list .rag-chunk-text').length === 10
    && q('#rag-chunks-list .rag-chunk-text')[0].textContent.indexOf('Резервное') >= 0);
  check('в статусе видно, сколько показано и сколько всего',
    $('rag-chunks-status').textContent.indexOf('Показано 10 из 25') >= 0,
    $('rag-chunks-status').textContent);
  check('кнопка «показать ещё» доступна, пока есть что показать',
    $('rag-chunks-more').hidden === false);

  await click($('rag-chunks-more'), 120);
  check('«показать ещё» догружает следующую страницу',
    q('#rag-chunks-list .rag-chunk').length === 20
    && $('rag-chunks-status').textContent.indexOf('Показано 20 из 25') >= 0,
    q('#rag-chunks-list .rag-chunk').length + ' / ' + $('rag-chunks-status').textContent);
  await click($('rag-chunks-more'), 120);
  check('на последней странице «показать ещё» скрывается',
    q('#rag-chunks-list .rag-chunk').length === 25 && $('rag-chunks-more').hidden === true);

  // Фильтр по документу: сервер отдаёт только чанки этого файла.
  const sourceOptions = Array.from($('rag-chunks-source').options).map(o => o.value);
  check('в фильтре перечислены документы базы',
    sourceOptions[0] === '' && sourceOptions.length === 1 + (RAG.bases[0].sources || []).length,
    JSON.stringify(sourceOptions));
  // Меняем выбор КАК ПОЛЬЗОВАТЕЛЬ — событием change: прямая установка value
  // обработчик не вызывает, и проверка прошла бы на невызванном фильтре.
  const pickedSource = sourceOptions[1];
  $('rag-chunks-source').value = pickedSource;
  $('rag-chunks-source').dispatchEvent(new dom.window.Event('change'));
  await wait(150);
  check('фильтр по документу показывает только его чанки',
    q('#rag-chunks-list .rag-chunk').length > 0
    && q('#rag-chunks-list .rag-chunk').every(el =>
      el.textContent.indexOf(pickedSource) >= 0),
    'фильтр ' + pickedSource + ', чанков: ' + q('#rag-chunks-list .rag-chunk').length);
  check('в статусе назван выбранный документ',
    $('rag-chunks-status').textContent.indexOf(pickedSource) >= 0,
    $('rag-chunks-status').textContent);

  // Поиск по тексту чанка.
  $('rag-chunks-query').value = 'backup';
  await click($('rag-chunks-find'), 120);
  check('поиск по тексту возвращает найденные чанки',
    q('#rag-chunks-list .rag-chunk').length > 0
    && $('rag-chunks-status').textContent.indexOf('поиск:') >= 0,
    $('rag-chunks-status').textContent);
  $('rag-chunks-query').value = 'такого-текста-нет';
  await click($('rag-chunks-find'), 120);
  check('поиск без совпадений говорит об этом прямо',
    q('#rag-chunks-list .rag-chunk').length === 0
    && q('#rag-chunks-list .rag-chunk-empty').length === 1
    && $('rag-chunks-more').hidden === true,
    $('rag-chunks-status').textContent);

  await click($('rag-chunks-close'), 40);
  check('кнопка «закрыть» закрывает просмотр чанков',
    $('rag-chunks-modal').hidden === true);

  // ---------------------------------------------------------------------
  // ФОНОВАЯ ИНДЕКСАЦИЯ: файл уходит потоком, сервер заводит задачу, интерфейс
  // опрашивает её состояние и рисует прогресс. Прежде чем начать, важно, чтобы
  // фоновых задач не осталось от прошлых проверок.
  stopRagPollForCheck();
  await click($('project-rag'), 80);
  check('диалог баз знаний открыт для проверки фоновой индексации',
    $('rag-modal').hidden === false);
  check('полоса прогресса скрыта, пока ничего не индексируется',
    $('rag-progress').hidden === true);

  const bigBlob = 'П'.repeat(4096);
  const bigFile = new dom.window.File([bigBlob], 'big.pdf', { type: 'application/pdf' });
  Object.defineProperty($('rag-file'), 'files', { value: [bigFile], configurable: true });
  $('rag-name').value = 'Крупная база';
  const streamsBefore = RAG.streams.length;
  const jobsBefore = RAG_JOBS.length;
  await click($('rag-upload'), 200);

  check('файл ушёл ПОТОКОМ (телом запроса, без base64)',
    RAG.streams.length === streamsBefore + 1
    && RAG.streams[RAG.streams.length - 1].filename === 'big.pdf',
    JSON.stringify(RAG.streams.slice(streamsBefore)));
  check('сервер завёл задачу индексации, а не ждал её в запросе',
    RAG_JOBS.length === jobsBefore + 1
    && RAG_JOBS[RAG_JOBS.length - 1].state === 'running',
    JSON.stringify(RAG_JOBS.slice(jobsBefore)));
  check('полоса прогресса появилась сразу после старта',
    $('rag-progress').hidden === false);
  const stageText = $('rag-progress-stage').textContent;
  check('на полосе назван этап индексации',
    stageText.indexOf('Индексация:') >= 0
    && ['разбор', 'эмбеддинги', 'запись'].some(word => stageText.indexOf(word) >= 0),
    stageText);
  check('в подробностях видно, ЧТО именно делается',
    $('rag-progress-detail').textContent.indexOf('big.pdf') >= 0
    || $('rag-progress-detail').textContent.indexOf('векторы') >= 0
    || $('rag-progress-detail').textContent.indexOf('пишу индекс') >= 0,
    $('rag-progress-detail').textContent);
  check('во время индексации доступна кнопка «остановить»',
    $('rag-progress-cancel').hidden === false);
  check('кнопка «RAG» помечена и показывает ход в подсказке',
    $('project-rag').classList.contains('on') === true
    && $('project-rag').title.indexOf('идёт индексация') > 0,
    $('project-rag').title);

  // Опрос двигает прогресс: этап, проценты и подробности меняются.
  await wait(2100);
  const stageAfterPoll = $('rag-progress-stage').textContent;
  const percentAfterPoll = $('rag-progress-percent').textContent;
  check('опрос обновляет этап и проценты (полоса не застывает)',
    stageAfterPoll.indexOf('эмбеддинги') >= 0 || stageAfterPoll.indexOf('запись') >= 0,
    stageAfterPoll + ' / ' + percentAfterPoll);
  check('проценты растут вместе с этапом',
    parseInt(percentAfterPoll, 10) > 40, percentAfterPoll);
  check('подробности меняются на ходу',
    $('rag-progress-detail').textContent.indexOf('векторы') >= 0
    || $('rag-progress-detail').textContent.indexOf('пишу индекс') >= 0,
    $('rag-progress-detail').textContent);

  // Завершение: база включается у проекта, в чате — отчёт, а полоса и строка
  // загрузки убираются СРАЗУ: результат уже виден (база в списке, числа в чате),
  // и оставленная «загрузка» выглядела как незаконченное действие (живое
  // замечание: базы в списке есть, а информация о загрузке не исчезает).
  await wait(3600);
  check('по завершении полоса прогресса убрана', $('rag-progress').hidden === true,
    'полоса hidden=' + $('rag-progress').hidden
      + ', этап=' + $('rag-progress-stage').textContent);
  check('по завершении кнопка «остановить» убрана', $('rag-progress-cancel').hidden === true);
  check('по завершении строка загрузки очищена',
    $('rag-upload-status').hidden === true, $('rag-upload-status').textContent);
  check('по завершении база появилась в списке диалога',
    q('#rag-list .rag-name').map(el => el.textContent).indexOf('Крупная база') >= 0,
    q('#rag-list .rag-name').map(el => el.textContent).join('|'));
  check('по завершении база включена у проекта', RAG.enabled.length > 0,
    JSON.stringify(RAG.enabled));
  check('после успеха поля загрузки очищены (имя и выбранные файлы)',
    $('rag-name').value === '' && $('rag-file').value === '',
    'имя=' + JSON.stringify($('rag-name').value));
  check('в чате появился отчёт об индексации с числами',
    q('#messages .msg.bot').some(el => el.textContent.indexOf('проиндексирована') >= 0
      && el.textContent.indexOf('чанков') >= 0));
  check('опрос остановлен — фоновых задач нет', RAG_JOBS.every(j => j.state !== 'running'));

  // НЕСКОЛЬКО ФАЙЛОВ: идут по одному, каждый следующий ДОПИСЫВАЕТСЯ в базу.
  const second = new dom.window.File(['П'.repeat(2048)], 'second.pdf',
    { type: 'application/pdf' });
  Object.defineProperty($('rag-file'), 'files',
    { value: [bigFile, second], configurable: true });
  $('rag-name').value = 'Два файла';
  const basesBefore = RAG.bases.length;
  const streamsBefore2 = RAG.streams.length;
  const jobsBefore2 = RAG_JOBS.length;
  await click($('rag-upload'), 200);
  check('первый файл уходит сразу, второй ждёт в очереди',
    RAG.streams.length === streamsBefore2 + 1
    && $('rag-progress-queue').textContent.indexOf('1') >= 0,
    RAG.streams.length - streamsBefore2 + ' / ' + $('rag-progress-queue').textContent);
  await wait(6500);            // первый файл доиндексировался, ушёл второй
  const paired = RAG.streams.slice(streamsBefore2);
  check('оба файла ушли по одному: первый создаёт базу, второй дописывается',
    paired.length === 2 && paired[0].baseId === '' && paired[1].baseId !== ''
    && paired[1].baseId === RAG.bases[RAG.bases.length - 1].id,
    JSON.stringify(paired.map(s => [s.filename, s.baseId])));
  await wait(6500);            // второй файл тоже доиндексировался
  check('два файла дали ОДНУ новую базу, а не две',
    RAG.bases.length === basesBefore + 1,
    'баз было ' + basesBefore + ', стало ' + RAG.bases.length);
  check('оба файла доиндексировались в одну базу',
    RAG_JOBS.length >= jobsBefore2 + 2
    && RAG_JOBS.filter(job => job.state === 'done').length >= jobsBefore2 + 2,
    JSON.stringify(RAG_JOBS.map(job => job.state)));
  check('очередь опустела', $('rag-progress-queue').hidden === true,
    $('rag-progress-queue').textContent);

  // ОТМЕНА: остановка задачи, индекс не меняется.
  Object.defineProperty($('rag-file'), 'files', { value: [bigFile], configurable: true });
  $('rag-name').value = 'Отменяемая';
  await click($('rag-upload'), 200);
  check('задача снова идёт', RAG_JOBS.some(j => j.state === 'running'));
  await click($('rag-progress-cancel'), 1200);
  await wait(1800);
  check('«остановить» отменяет задачу',
    RAG_JOBS.filter(j => j.state === 'cancelled').length === 1,
    JSON.stringify(RAG_JOBS.map(j => j.state)));
  check('после отмены полоса говорит об остановке, а не о завершении',
    $('rag-progress-stage').textContent.indexOf('остановлена') >= 0,
    $('rag-progress-stage').textContent);
  check('после отмены база НЕ включается и в чате нет отчёта об успехе',
    !q('#messages .msg.bot').some(el =>
      el.textContent.indexOf('Отменяемая') >= 0
      && el.textContent.indexOf('проиндексирована') >= 0));

  // Файл больше общего предела отклоняется ДО отправки, с числами в подсказке.
  const huge = { name: 'huge.pdf', size: 999 * 1024 * 1024 };
  Object.defineProperty($('rag-file'), 'files', { value: [huge], configurable: true });
  const streamsBefore3 = RAG.streams.length;
  await click($('rag-upload'), 120);
  check('файл больше общего предела не отправляется вовсе',
    RAG.streams.length === streamsBefore3
    && $('rag-upload-status').textContent.indexOf('huge.pdf') >= 0
    && $('rag-upload-status').textContent.indexOf('RAG_MAX_FILE_BYTES') >= 0,
    $('rag-upload-status').textContent);

  await click($('rag-close'), 40);
  // СТРОКА СОСТОЯНИЯ — ПРО СПИСОК БАЗ, а не про настройку проекта: у баз
  // стратегии могут быть РАЗНЫЕ, и подпись «стратегия: …» здесь читалась как
  // свойство сразу всех баз (живое замечание: две базы с разными стратегиями).
  const statusText = $('rag-status').textContent;
  check('в строке состояния нет параметров разбиения (у баз они разные)',
    statusText.indexOf('стратегия') < 0 && statusText.indexOf('перекрытие') < 0,
    statusText);
  check('в строке состояния есть счётчики баз, чанков, документов и объём',
    statusText.indexOf('Баз:') >= 0 && statusText.indexOf('чанков') >= 0
    && statusText.indexOf('документов') >= 0 && statusText.indexOf('объём') >= 0,
    statusText);
  check('стратегия каждой базы видна в ЕЁ строке, а не общим текстом',
    q('#rag-list .rag-item').length > 0
    && q('#rag-list .rag-item').every(item => item.textContent.indexOf('чанк') >= 0),
    q('#rag-list .rag-item').map(item => item.textContent.slice(0, 50)).join(' | '));

  // Чем считаются векторы и чем идёт поиск — это видно в диалоге (свойство
  // машины: с numpy счёт близости в разы быстрее, без него — перебор).
  check('в диалоге сказано, чем считаются эмбеддинги',
    $('rag-embed-note').textContent.indexOf('sentence-transformers') >= 0
    && $('rag-embed-note').textContent.indexOf('384') >= 0,
    $('rag-embed-note').textContent);
  check('в диалоге сказано, чем считается близость при поиске',
    $('rag-embed-note').textContent.indexOf('Поиск по базе:') >= 0
    && $('rag-embed-note').textContent.indexOf('numpy') >= 0,
    $('rag-embed-note').textContent);

  // УДАЛЕНИЕ базы: она уходит из списка, а в чате появляется строка об этом.
  const basesBeforeDelete = RAG.bases.length;
  await click(q('#rag-list .rag-delete')[0], 140);
  check('кнопка 🗑 удаляет базу знаний',
    RAG.bases.length === basesBeforeDelete - 1
    && q('#rag-list .rag-item').length === RAG.bases.length,
    'баз было ' + basesBeforeDelete + ', стало ' + RAG.bases.length);
  check('в чате сказано об удалении базы',
    q('#messages .msg.bot').some(el =>
      el.textContent.indexOf('удалена вместе с индексом') >= 0));

  // ЗАГРУЗКА БЕЗ ФАЙЛА: на сервер ничего не уходит, а пользователь видит причину.
  Object.defineProperty($('rag-file'), 'files', { value: [], configurable: true });
  const streamsBeforeEmpty = RAG.streams.length;
  await click($('rag-upload'), 80);
  check('загрузка без файла на сервер не уходит',
    RAG.streams.length === streamsBeforeEmpty
    && $('rag-upload-status').textContent.indexOf('Выберите') >= 0,
    $('rag-upload-status').textContent);

  // ПОСЛЕ ОТМЕНЫ имя базы ОСТАЁТСЯ в поле: загрузку логично повторить с тем же
  // именем (а файл всё равно придётся выбрать заново — поле файла не восстановить).
  check('после отмены имя базы остаётся в поле (повтор без набора заново)',
    $('rag-name').value !== '',
    JSON.stringify($('rag-name').value));

  check('кнопка «закрыть» закрывает диалог баз знаний', $('rag-modal').hidden === true);

  // Вне режима агента кнопка скрыта вместе с блоком проекта.
  await click($('project-rag'), 60);
  dom.window.eval('setAgentMode(false)');
  await wait(60);
  check('вне режима агента диалог баз знаний закрыт', $('rag-modal').hidden === true);
  dom.window.eval('setAgentMode(true)');
  await wait(40);

  // ---------------------------------------------------------------------
  // [Q] ПЕРИОДИЧЕСКИЕ ЗАДАЧИ: кнопка под «Новая задача», метка в списке,
  //     модалка периода и подхват того, что автозапуск дописал в задачу.
  // Повторы выполняет СЕРВЕР (app/periodic_runner.py): интерфейс только
  // показывает расписание и опрашивает /api/agent/periodic.
  // ---------------------------------------------------------------------
  console.log('\n[Q] Периодические задачи: кнопка, метка, расписание');
  const periodicBtn = $('session-new-periodic');
  check('кнопка «Новая периодическая задача» есть на странице', !!periodicBtn);
  check('подпись кнопки', periodicBtn.textContent === 'Новая периодическая задача',
    JSON.stringify(periodicBtn.textContent));
  check('кнопка стоит ПОД «Новая задача»',
    periodicBtn.previousElementSibling === $('session-new'),
    (periodicBtn.previousElementSibling || {}).id);
  check('класс кнопки — тот же, что у «Новая задача»',
    periodicBtn.classList.contains('session-new')
    && $('session-new').classList.contains('session-new'),
    periodicBtn.className);
  check('кнопка видна в режиме агента', periodicBtn.hidden === false);

  const periodicSessionsBefore = workspace.sessions.length;
  const periodicPostsBefore = calls.filter(
    c => c === 'POST /api/agent/sessions').length;
  await click(periodicBtn, 60);
  check('нажатие создаёт задачу на сервере',
    calls.filter(c => c === 'POST /api/agent/sessions').length
      === periodicPostsBefore + 1,
    calls.slice(-2).join(' | '));
  check('создание помечено как периодическое',
    periodicBodies.some(b => b && b.periodic === true),
    JSON.stringify(periodicBodies));
  check('задача появилась в списке', workspace.sessions.length === periodicSessionsBefore + 1);
  const periodicId = workspace.active_session;
  const periodicItem = q('#sessions .session-item')
    .filter(el => el.dataset.session === periodicId)[0];
  check('периодическая задача выделена в списке',
    !!periodicItem && periodicItem.classList.contains('periodic'),
    periodicItem ? periodicItem.className : '(элемент не найден)');
  check('у неё есть метка с периодом по умолчанию',
    !!periodicItem
    && periodicItem.querySelector('.session-period').textContent.indexOf('раз в сутки') > 0,
    periodicItem ? periodicItem.textContent : '');
  check('метку периода видно и в подсказке',
    !!periodicItem
    && periodicItem.querySelector('.session-period').title.indexOf('Периодическая задача') === 0,
    periodicItem ? periodicItem.querySelector('.session-period').title : '');
  // Полоса этапов периодической задачи: блоков «проверка» и «готово» быть не
  // должно — задача не проверяется и не завершается.
  setState({});
  dom.window.eval('applyTaskState(' + JSON.stringify(snapshot()) + ')');
  await wait(40);
  const periodicBlocks = blocks().map(el => el.textContent);
  check('в полосе периодической задачи нет «Проверки» и «Готово»',
    periodicBlocks.join('|') === 'Планирование|Выполнение', periodicBlocks.join('|'));
  check('стрелок в полосе на одну меньше блоков',
    arrows().length === periodicBlocks.length - 1,
    'блоков: ' + periodicBlocks.length + ', стрелок: ' + arrows().length);

  check('обычные задачи выделения не получают',
    q('#sessions .session-item').filter(
      el => el.dataset.session !== periodicId && el.classList.contains('periodic')
    ).length === 0);
  check('в чате объяснено, что задача периодическая',
    q('#messages .msg.bot').some(el => el.textContent.indexOf('Создана периодическая задача') >= 0));

  // Кнопка 🔁 открывает расписание: период, срок повтора, счётчик.
  const gear = periodicItem.querySelector('.icon-btn[title^="Периодическая задача"]');
  check('у периодической задачи есть кнопка расписания 🔁', !!gear);
  await click(gear, 60);
  check('нажатие открывает модалку расписания', $('periodic-modal').hidden === false);
  const status = $('periodic-status').textContent;
  check('в модалке видно, что автозапуск включён и с каким периодом',
    status.indexOf('Автозапуск включён') >= 0 && status.indexOf('раз в сутки') > 0,
    status.slice(0, 160));
  check('в модалке виден срок следующего повтора',
    status.indexOf('Следующий повтор') > 0, status.slice(0, 200));
  check('в модалке видно, что запрос ещё не написан',
    $('periodic-request').textContent.indexOf('Запрос ещё не написан') >= 0,
    $('periodic-request').textContent);
  check('селект показывает период задачи',
    $('periodic-period').value === '86400', $('periodic-period').value);

  // Правка периода уходит на сервер и перерисовывает список.
  const periodicPosts = calls.filter(
    c => c.indexOf('POST /api/agent/periodic/') === 0).length;
  $('periodic-period').value = '3600';
  await click($('periodic-save'), 80);
  check('«сохранить» отправил период на сервер',
    calls.filter(c => c.indexOf('POST /api/agent/periodic/') === 0).length
      === periodicPosts + 1,
    calls.slice(-2).join(' | '));
  check('на сервер ушёл выбранный период (раз в час)',
    periodicUpdates.some(u => u && u.interval === 3600),
    JSON.stringify(periodicUpdates));
  check('модалка закрылась после сохранения', $('periodic-modal').hidden === true);
  const hourItem = q('#sessions .session-item')
    .filter(el => el.dataset.session === periodicId)[0];
  check('метка задачи показывает новый период',
    hourItem.querySelector('.session-period').textContent.indexOf('раз в час') > 0,
    hourItem.querySelector('.session-period').textContent);

  // Остановка автозапуска: задача остаётся периодической, но помечена как
  // остановленная.
  await click(hourItem.querySelector('.icon-btn[title^="Периодическая задача"]'), 60);
  await click($('periodic-toggle'), 80);
  check('«остановить автозапуск» ушло на сервер с enabled=false',
    periodicUpdates.some(u => u && u.enabled === false),
    JSON.stringify(periodicUpdates));
  const stoppedItem = q('#sessions .session-item')
    .filter(el => el.dataset.session === periodicId)[0];
  check('остановленная задача остаётся периодической',
    stoppedItem.classList.contains('periodic')
    && stoppedItem.classList.contains('off'),
    stoppedItem.className);
  check('метка говорит, что повтор остановлен',
    stoppedItem.querySelector('.session-period').textContent.indexOf('остановлена') > 0,
    stoppedItem.querySelector('.session-period').textContent);

  // ВОЗВРАЩЕНИЕ В ЗАДАЧУ: автозапуск дописал в неё ответ — открытый диалог
  // должен подхватить это сам (опрос /api/agent/periodic), без перезагрузки.
  await click(stoppedItem, 60);          // открываем задачу
  const periodicChatCallsBefore = chatCalls();
  // Первый опрос только запоминает размер журнала: сравнивать ещё не с чем —
  // он ничего не перерисовывает.
  await dom.window.eval('pollPeriodic()');
  await wait(60);
  check('первый опрос не перерисовывает окно вслепую',
    !q('#messages .msg').some(el => el.textContent.indexOf('Повтор: +14') >= 0));
  const logBefore = (logs[periodicId] || []).length;
  // Автозапуск дописал в задачу новое: журнал вырос (на сервере это видно по
  // log_len в снимке расписания).
  logs[periodicId] = (logs[periodicId] || []).concat([
    { kind: 'periodic', text: '⏱ Автозапуск (раз в час): Сводка погоды в Москве' },
    { kind: 'assistant', text: 'Повтор: +14, облачно, воздух чистый.' },
  ]);
  await dom.window.eval('pollPeriodic()');
  await wait(80);
  const periodicMsgs = q('#messages .msg').filter(
    el => el.textContent.indexOf('Повтор: +14') >= 0);
  check('повтор автозапуска появился в открытом чате сам',
    periodicMsgs.length > 0, String(periodicMsgs.length));
  check('пометка автозапуска нарисована в чате',
    q('#messages .msg').some(el => el.textContent.indexOf('⏱ Автозапуск') >= 0));
  check('опрос не отправил запрос к агенту',
    chatCalls() === periodicChatCallsBefore, String(chatCalls()));
  check('новая запись журнала больше прошлой',
    (logs[periodicId] || []).length > logBefore);

  // Отменённая задача: расписание включено, но повторов не будет — модалка
  // обязана сказать это прямо (кнопка 🔁 сама задачу не «оживит»).
  PERIODIC.tasks[periodicId].hold = 'cancelled';
  await dom.window.eval('pollPeriodic()');
  await wait(60);
  await click(hourItem.querySelector('.icon-btn[title^="Периодическая задача"]')
    || q('#sessions .session-item').filter(el => el.dataset.session === periodicId)[0]
        .querySelector('.icon-btn[title^="Периодическая задача"]'), 60);
  check('модалка объясняет, что отменённая задача не повторяется',
    $('periodic-status').textContent.indexOf('Задача отменена') > 0,
    $('periodic-status').textContent.slice(0, 160));
  const cancelledItem = q('#sessions .session-item')
    .filter(el => el.dataset.session === periodicId)[0];
  check('метка задачи говорит про отмену',
    cancelledItem.querySelector('.session-period').title.indexOf('отменена') > 0,
    cancelledItem.querySelector('.session-period').title);
  await click($('periodic-close'), 40);
  PERIODIC.tasks[periodicId].hold = '';
  await dom.window.eval('pollPeriodic()');
  await wait(60);

  // РЕАЛЬНЫЙ ПОРЯДОК: сообщение → план → «Подтвердить план» → прогон шагов.
  // Проверяем, что прогон выполняет ТОЛЬКО ОДИН цикл и не уходит в следующий.
  setState({});
  dom.window.eval('applyTaskState(' + JSON.stringify(snapshot()) + ')');
  $('input').value = 'сводка погоды, раз в час';
  const confirmCallsBefore = chatCalls();
  await sendRequest('сводка погоды, раз в час', 120);
  const planCalls = chatCalls() - confirmCallsBefore;
  check('запрос построил план (один вызов)', planCalls === 1, 'вызовов: ' + planCalls);
  check('авто-прогон без подтверждения плана не начался',
    chatCalls() - confirmCallsBefore === 1, 'вызовов: ' + (chatCalls() - confirmCallsBefore));
  const cycleStepsPlanned = JSON.parse(dom.window.eval('JSON.stringify(taskMachineState.steps_total)'));
  await click($('tm-confirm'), 200);
  const cycleCallsReal = chatCalls() - confirmCallsBefore - 1;
  check('после подтверждения прогон выполнил ровно один цикл',
    cycleCallsReal === cycleStepsPlanned && cycleStepsPlanned > 0,
    'запросов шага: ' + cycleCallsReal + ', шагов в плане: ' + cycleStepsPlanned);
  const afterConfirm = JSON.parse(dom.window.eval('JSON.stringify(taskMachineState)'));
  check('после цикла прогон остановлен (состояние ждёт расписания)',
    afterConfirm.stage === 'execution' && afterConfirm.step_index === 0,
    JSON.stringify({ stage: afterConfirm.stage, index: afterConfirm.step_index }));
  await wait(200);
  check('прогон не пошёл на второй цикл сам',
    chatCalls() - confirmCallsBefore - 1 === cycleStepsPlanned,
    'запросов шага: ' + (chatCalls() - confirmCallsBefore - 1));

  // ЦИКЛ периодической задачи: прогон шагов заканчивается возвратом на первый шаг
  // («Готово» у периодической задачи не бывает), поэтому прогон НЕ крутит повтор
  // за повтором — он останавливается на конце цикла.
  setPlan(['Шаг A', 'Шаг B']);
  setState(Object.assign({}, state, {
    stage: 'execution', base_stage: 'execution', current_step: 'step_1',
    step_index: 0, step_number: 1, terminal: false, extra_stage: null,
    repeat_ready: false,
    expected_action: 'выполнить: Шаг A',
  }));
  dom.window.eval('applyTaskState(' + JSON.stringify(snapshot()) + ')');
  const cycleCallsBefore = chatCalls();
  await dom.window.eval('runStepChain(' + JSON.stringify(periodicId) + ')');
  await wait(150);
  const cycleCalls = chatCalls() - cycleCallsBefore;
  check('прогон периодической задачи выполнил ровно один цикл (2 шага)',
    cycleCalls === 2, 'запросов шага: ' + cycleCalls);
  const cycleState = JSON.parse(dom.window.eval('JSON.stringify(taskMachineState)'));
  check('после цикла задача НЕ в «Готово», а ждёт следующего повтора',
    cycleState.stage === 'execution' && cycleState.step_index === 0
    && cycleState.terminal === false && cycleState.repeat_ready === true,
    JSON.stringify({ stage: cycleState.stage, index: cycleState.step_index,
      terminal: cycleState.terminal, repeat_ready: cycleState.repeat_ready }));
  check('кнопка предлагает выполнить ПОВТОР, а не «продолжить шаг»',
    $('tm-idle').hidden === false
    && $('tm-idle').textContent.indexOf('выполнить повтор сейчас') > 0,
    $('tm-idle').textContent + ' | hidden=' + $('tm-idle').hidden);
  check('в поле ввода — подсказка про расписание, а не «задача в работе»',
    $('input').placeholder.indexOf('Периодическая задача') === 0,
    $('input').placeholder);

  // Задачу о периодичности удаляем: дальше проверяем обычные задачи.
  await click(q('#sessions .session-item')
    .filter(el => el.dataset.session === periodicId)[0]
    .querySelector('.icon-btn[title="Удалить задачу"]'), 60);
  await dom.window.eval('document.getElementById("confirm-modal-ok").click()');
  await wait(80);
  check('периодическая задача удаляется как обычная',
    !workspace.sessions.some(s => s.id === periodicId),
    workspace.sessions.map(s => s.id).join('|'));

  // ---------------------------------------------------------------------
  // [R] ВРЕМЯ СООБЩЕНИЙ: у реплик пользователя и ответов агента — как в
  // мессенджерах; у служебных (debug/error) подписи времени нет.
  // ---------------------------------------------------------------------
  console.log('\n[R] Время отправки/получения сообщений');
  // Кладём в журнал ОТКРЫТОЙ задачи записи с временем «от сервера» и перечитываем
  // диалог: так проверяется отрисовка времени ИМЕННО из журнала, а не «сейчас».
  const sessionNow = workspace.active_session;
  logs[sessionNow] = [
    { kind: 'user', text: 'Реплика с временем', at: logAt(9, 7) },
    { kind: 'debug', text: 'Служебная строка про работу автомата', at: logAt(9, 8) },
    { kind: 'assistant', text: 'Ответ агента с временем', at: logAt(9, 9) },
  ];
  await dom.window.eval('loadActiveDialog()');
  await wait(80);
  const timeNodes = el => (el ? Array.from(el.querySelectorAll('.msg-time')) : []);
  const pick = (selector, needle) => q(selector).filter(
    el => el.textContent.indexOf(needle) >= 0)[0];
  const userMsg = pick('#messages .msg.user', 'Реплика с временем');
  const botMsg = pick('#messages .msg.bot', 'Ответ агента с временем');
  const debugMsg = pick('#messages .msg.debug', 'Служебная строка');
  check('у реплики пользователя есть подпись времени',
    timeNodes(userMsg).length === 1,
    userMsg ? userMsg.textContent.slice(0, 60) : '(нет реплики)');
  check('время взято из журнала сервера (09:07), а не поставлено сейчас',
    timeNodes(userMsg).length === 1 && timeNodes(userMsg)[0].textContent === '09:07',
    JSON.stringify(timeNodes(userMsg).map(el => el.textContent)));
  check('подпись стоит ПОСЛЕ пузыря сообщения',
    !!userMsg && userMsg.querySelector('.bubble')
      && userMsg.querySelector('.bubble').nextElementSibling
      && userMsg.querySelector('.bubble').nextElementSibling.classList.contains('msg-time'));
  check('в подсказке — полная дата и время',
    timeNodes(userMsg).length === 1 && timeNodes(userMsg)[0].title.length > 10,
    JSON.stringify(timeNodes(userMsg).map(el => el.title)));
  check('у ответа агента тоже есть время (09:09)',
    timeNodes(botMsg).length === 1 && timeNodes(botMsg)[0].textContent === '09:09',
    botMsg ? JSON.stringify(timeNodes(botMsg).map(el => el.textContent)) : '(нет ответа)');
  check('у служебной строки времени НЕТ',
    !!debugMsg && timeNodes(debugMsg).length === 0,
    debugMsg ? debugMsg.textContent.slice(0, 60) : '(нет служебной строки)');
  // Запись из старого файла (без времени): подпись не показываем вовсе.
  logs[sessionNow] = [{ kind: 'user', text: 'Запись без времени' }];
  await dom.window.eval('loadActiveDialog()');
  await wait(60);
  const oldMsg = pick('#messages .msg.user', 'Запись без времени');
  check('запись без времени рисуется без подписи',
    !!oldMsg && timeNodes(oldMsg).length === 0,
    oldMsg ? oldMsg.textContent : '(нет реплики)');
  check('вместо времени не появляется «NaN»',
    !dom.window.document.getElementById('messages').textContent.includes('NaN'));

  console.log('\n[T] Тарифы и стоимость в интерфейсе');
  // Стоимость вызова у официального DeepSeek — тысячные доли рубля, поэтому
  // проверяем, что мелкие суммы НЕ превращаются в «0,00 руб».
  const rub = v => dom.window.eval('fmtRub(' + JSON.stringify(v) + ')');
  check('мелкая сумма не показывается нулём', rub(0.0034) === '0,0034 руб', rub(0.0034));
  check('сумма до рубля — три знака', rub(0.1234) === '0,123 руб', rub(0.1234));
  check('обычная сумма — два знака', rub(1.2345) === '1,23 руб', rub(1.2345));
  check('ноль показывается как ноль', rub(0) === '0 руб', rub(0));
  check('пустое значение не даёт «NaN»', rub(null) === '0 руб', rub(null));

  // Тариф — данные СЕРВЕРА (pricing у строки аналитики): интерфейс не должен
  // выдумывать цены сам, а обязан показать ставки, пиковость и курс.
  const dsPricing = {
    provider: 'deepseek-official', model: 'deepseek-v4-flash', peak: false,
    tariff: 'непиковый тариф (×0,5)', usd_rub: 84.0657,
    usd_per_mtok: { cache_hit: 0.003, cache_miss: 0.15, output: 0.6 },
    rub_per_mtok: { cache_hit: 0.2521971, cache_miss: 12.609855, output: 50.43942 },
    peak_note: 'Пиковые часы DeepSeek: 06:00–09:00 и 11:00–15:00 по Екатеринбургу',
  };
  const yandexPricing = {
    provider: 'yandex', model: 'gpt://x/aliceai-llm/latest', peak: false,
    tariff: 'тариф Yandex, руб. за 1000 токенов', usd_rub: null, usd_per_mtok: null,
    rub_per_mtok: { cache_hit: 500, cache_miss: 500, output: 1200 }, peak_note: '',
  };
  const dsText = dom.window.eval('fmtPricing(' + JSON.stringify(dsPricing) + ')');
  check('в строке тарифа видны ставки за 1M токенов',
    dsText.includes('12,61') && dsText.includes('50,44'), dsText);
  check('в строке тарифа видны цена из кэша и курс',
    dsText.includes('0,252') && dsText.includes('84,07'), dsText);
  check('в строке тарифа назван сам тариф',
    dsText.includes('непиковый') && dsText.includes('deepseek-v4-flash'), dsText);
  const yaText = dom.window.eval('fmtPricing(' + JSON.stringify(yandexPricing) + ')');
  check('тариф Yandex показывается без курса и без пика',
    yaText.includes('500,00') && !yaText.includes('курс'), yaText);
  check('без данных тарифа строка пустая', dom.window.eval('fmtPricing(null)') === '');

  // Пояснения под таблицей: по одной строке на КАЖДЫЙ тариф (модели могут быть
  // у разных провайдеров), повторы не дублируются.
  const notesHtml = dom.window.eval('pricingNotes(' + JSON.stringify([
    { label: 'A', cost_rub: 0.01, pricing: dsPricing },
    { label: 'B', cost_rub: 0.02, pricing: dsPricing },
    { label: 'C', cost_rub: 0.03, pricing: yandexPricing },
    { label: 'D', cost_rub: 0.04 },
  ]) + ')');
  const notesCount = (notesHtml.match(/tariff-note/g) || []).length;
  check('пояснение тарифа — по одному на тариф (не на строку)', notesCount === 2,
    String(notesCount));
  check('пояснение помечено классом и не ломает вёрстку',
    notesHtml.includes('judge-summary tariff-note'));
  check('часы пика ушли во всплывающую подсказку',
    notesHtml.includes('title="Пиковые часы DeepSeek'), notesHtml.slice(0, 120));
  check('строка без тарифа пояснения не создаёт',
    (dom.window.eval('pricingNotes([{ label: "D", cost_rub: 0.04 }])')
      .match(/tariff-note/g) || []).length === 0);

  console.log('\nИтог: ' + (failures ? 'ПРОВАЛЕНО проверок: ' + failures : 'все проверки пройдены'));
  dom.window.close();
  process.exit(failures ? 1 : 0);
}

run().catch(err => { console.error(err); process.exit(1); });
