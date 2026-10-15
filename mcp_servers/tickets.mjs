#!/usr/bin/env node
// MCP-сервер «Поддержка»: обращения пользователей (тикеты) и сами пользователи.
//
// ЗАЧЕМ. До него у ассистента не было НИ ОДНОГО источника про конкретного
// человека: базы знаний отвечают «как устроен продукт», а на вопрос «почему не
// работает авторизация?» нужны данные ОБРАЩЕНИЯ — что человек делал, что ему уже
// отвечали. Эти данные и отдаёт сервер: пять инструментов
// чтения (пользователи, обращения, поиск по тексту обращения) плюс два
// изменяющих — завести обращение (`create_ticket`) и дописать сообщение в
// переписку (`add_comment`).
//
// ИСТОЧНИК ДАННЫХ — ЛОКАЛЬНЫЙ JSON-ФАЙЛ, а не CRM и не сеть. По умолчанию
// `data/tickets.json` (путь переопределяет переменная TICKETS_FILE). Каталог
// `data/` в git НЕ хранится, поэтому демонстрационные данные лежат в репозитории
// отдельно — `mcp_servers/tickets.seed.json`, — и при первом обращении файл
// рабочей копии создаётся из образца: сервер работает сразу после `npm install`,
// без настройки и без ключей. Свой файл на это место кладётся как есть: структура
// описана ниже и в mcp_servers/README.md.
//
// СТРУКТУРА ФАЙЛА (обе ветки — массивы; лишние поля сохраняются):
//   {
//     "users":   [{"id", "name"}],
//     "tickets": [{"id", "user_id", "subject", "status", "priority", "product",
//                  "created_at", "updated_at", "description",
//                  "comments": [{"author", "at", "text"}]}]
//   }
//
// КЛИЕНТ — ТОЛЬКО id И ИМЯ. Ни тарифа, ни компании, ни почты здесь нет и не
// должно быть: режим поддержки отвечает по работе с ЭТОЙ кодовой базой, и
// «клиент» тут — человек с именем (в приложении имя приходит из профиля), а не
// запись из CRM. Лишние поля пришлось бы заполнять выдумкой, а выдуманный тариф
// в ответе поддержки хуже его отсутствия.
//
// ЧТО ЗДЕСЬ ВАЖНО ДЛЯ АГЕНТА (те же правила, что у сервера git):
//   * ЧТЕНИЕ — по умолчанию: пять инструментов из семи ничего не меняют;
//   * ЗАПИСЬ — `create_ticket` и `add_comment`. Имена начинаются с `create_` и
//     `add_`, поэтому клиент агента сам откладывает эти вызовы до подтверждения
//     плана (app/ai/mcp.py, RESULT_RE): побочный эффект не случается «по дороге»;
//   * ЗАЯВИТЕЛЯ ПРИ РАБОТЕ ИЗ ПРИЛОЖЕНИЯ НАЗЫВАЕТ КОД, А НЕ МОДЕЛЬ: имя клиента
//     подставляет веб-слой из профиля (chat.py, `_fill_requester`), в аргументе
//     `requester` оно приходит уже готовым. Модель не выбирает, от чьего имени
//     заводится обращение, — так же, как не она ставит флаг перезаписи;
//   * ДУБЛИКАТЫ НЕ ПЛОДЯТСЯ МОЛЧА: перед созданием сервер ищет похожие ОТКРЫТЫЕ
//     обращения и без `force` второе не создаёт — решение принимает человек;
//   * ЧЕСТНОСТЬ ВАЖНЕЕ ПОЛНОТЫ: «обращения не найдены» — это ОТВЕТ, и он
//     называет, по чему искали и что вообще есть в базе. Молчаливый пустой
//     результат модель достраивает догадкой, и человек получает выдуманный
//     статус или выдуманного пользователя;
//   * БЮДЖЕТ ОТВЕТА — BUDGET символов (клиент берёт не больше 4000). Что не
//     влезло, то не влезло: сервер говорит об этом словами, а не обрывает
//     строку на полуслове;
//   * ИСПОРЧЕННЫЙ ФАЙЛ НЕ ПЕРЕЗАПИСЫВАЕТСЯ: если JSON не разбирается, запись
//     отменяется с причиной — данные поддержки дороже удобства.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { z } from "zod";
import { createServer, failure, guard, lines, log, serve, text } from "./lib/common.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SEED_FILE = path.join(HERE, "tickets.seed.json");

// Предел ответа одного инструмента. Клиент агента берёт не больше 4000 символов
// (RESULT_CHARS в app/ai/mcp.py), поэтому режем сами и говорим, что показали не всё.
const BUDGET = 3000;
// Сколько обращений/пользователей возвращаем по умолчанию и максимум.
const LIST_DEFAULT = 5;
const LIST_MAX = 20;
// Сколько символов описания и комментария попадает в ответ.
const EXCERPT = 300;
const COMMENT_CHARS = 700;

const STATUSES = [
  "новый", "в работе", "ждём ответа", "решён", "закрыт",
];

// СИНОНИМЫ ЗНАЧЕНИЙ. Схема инструмента объявляет русские значения, а модель
// регулярно присылает английские («high», «closed») или пишет статус словами
// («in progress»). Строгая схема в этом случае ОТКЛОНЯЕТ вызов, и человек читает
// «Invalid option: expected one of …» вместо ответа поддержки (живой случай:
// обращение не завелось из-за `priority: "high"`). Поэтому значения принимаются
// строкой и приводятся к каноническим, а нераспознанное называется причиной.
const PRIORITIES = ["низкий", "средний", "высокий"];

// ГРУППЫ СТАТУСОВ. Человек говорит «что у нас в работе», «открытые обращения»,
// «ещё не закрытые» — а это НЕ один статус, а группа из нескольких. Без групп
// запрос «открыто» отклонялся («Статус «открыто» не распознан. Известные
// статусы: …»), и режим вместо списка просил человека выбрать статус (живой
// случай 14.10): «/support — что у нас в работе» → отказ инструмента → ответ
// «скажите, какой статус смотреть». Группа решает это одним значением.
const STATUS_GROUPS = {
  "open": ["новый", "в работе", "ждём ответа"],
  "closed": ["решён", "закрыт"],
};
const PRIORITY_ALIASES = {
  "низкий": "низкий", "low": "низкий", "minor": "низкий",
  "средний": "средний", "medium": "средний", "normal": "средний", "обычный": "средний",
  "высокий": "высокий", "high": "высокий", "major": "высокий",
  "критический": "высокий", "critical": "высокий", "blocker": "высокий",
};
const STATUS_ALIASES = {
  "новый": "новый", "new": "новый",
  "в работе": "в работе", "in progress": "в работе", "in_progress": "в работе",
  "inprogress": "в работе", "work": "в работе", "работа": "в работе",
  "ждём ответа": "ждём ответа", "ждем ответа": "ждём ответа",
  "waiting": "ждём ответа", "pending": "ждём ответа",
  "решён": "решён", "решен": "решён", "resolved": "решён", "solved": "решён",
  "закрыт": "закрыт", "closed": "закрыт",
  // Группы: «открытые» — это не статус «новый», а всё, что ещё не закончено.
  "open": "open", "opened": "open", "активн": "open", "active": "open",
  "все": "all", "all": "all", "любые": "all",
};

/** Канонический приоритет ("" — значение не распознано; пустое = средний). */
function normalizePriority(value) {
  const raw = String(value ?? "").trim().toLowerCase();
  if (!raw) return "средний";
  return PRIORITY_ALIASES[raw] || PRIORITY_ALIASES[raw.replace(/ё/g, "е")] || "";
}

/**
 * Канонический статус, КЛЮЧ ГРУППЫ («open», «closed», «all») или "" — не
 * распознано. Кроме точных синонимов распознаются начала слов: «открыт…»,
 * «активн…», «незакрыт…», «нерешён…» — человек пишет форму, которой нет в
 * списке, а смысл её понятен.
 */
function normalizeStatus(value) {
  const raw = String(value ?? "").trim().toLowerCase();
  if (!raw) return "";
  const direct = STATUS_ALIASES[raw] || STATUS_ALIASES[raw.replace(/ё/g, "е")];
  if (direct) return direct;
  const soft = raw.replace(/ё/g, "е");
  if (/^(открыт|активн|незакрыт|не закрыт|нерешен|незавершен|в процессе|в работе)/.test(soft)) {
    return "open";
  }
  if (/^(закрыт|решен|завершен|выполнен)/.test(soft)) return "closed";
  return "";
}

/** Подходит ли обращение под запрошенный статус (или группу статусов). */
function statusMatches(ticket, wanted) {
  if (!wanted) return true;
  const status = norm(ticket.status);
  const group = STATUS_GROUPS[wanted];
  if (group) return group.some((item) => norm(item) === status);
  if (wanted === "all") return true;
  return status === norm(wanted);
}

/** Человеческое пояснение допустимых значений — для отказов. */
function statusHint() {
  return "Известные статусы: " + STATUSES.join(", ")
    + "; можно назвать и группу: «открытые» (новый, в работе, ждём ответа) или "
    + "«закрытые» (решён, закрыт).";
}

/**
 * Путь к файлу данных: переменная TICKETS_FILE, иначе data/tickets.json проекта.
 * Путь считается от самого сервера, а не от текущего каталога процесса: клиент
 * запускает сервер из другого места, и относительный путь означал бы разные
 * файлы у проверки и у агента.
 */
function storePath() {
  const custom = String(process.env.TICKETS_FILE || "").trim();
  if (custom) return path.resolve(custom);
  return path.resolve(HERE, "..", "data", "tickets.json");
}

function emptyStore() {
  return { users: [], tickets: [] };
}

/** Проверить, что прочитанное — та структура, с которой работают инструменты. */
function checkStore(data) {
  if (!data || typeof data !== "object" || Array.isArray(data)) {
    throw new Error("в файле не объект JSON");
  }
  if (data.users !== undefined && !Array.isArray(data.users)) {
    throw new Error("поле users — не массив");
  }
  if (data.tickets !== undefined && !Array.isArray(data.tickets)) {
    throw new Error("поле tickets — не массив");
  }
  return {
    ...data,
    users: (data.users || []).filter((item) => item && typeof item === "object"),
    tickets: (data.tickets || []).filter((item) => item && typeof item === "object"),
  };
}

/**
 * Прочитать файл данных. Нет файла — создать из образца (демо-данные в
 * репозитории) либо пустую заготовку; образца нет тоже — работаем с пустой
 * базой, а не падаем: сервер обязан объявить инструменты даже без данных.
 */
function readStore() {
  const file = storePath();
  if (!fs.existsSync(file)) {
    let seed = emptyStore();
    let origin = "пустая база (файла данных и образца нет)";
    if (fs.existsSync(SEED_FILE)) {
      seed = checkStore(JSON.parse(fs.readFileSync(SEED_FILE, "utf8")));
      origin = "образец " + path.basename(SEED_FILE);
    }
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, JSON.stringify(seed, null, 2) + "\n", "utf8");
    log("создан файл данных", file, "из", origin);
    return seed;
  }
  return checkStore(JSON.parse(fs.readFileSync(file, "utf8")));
}

/**
 * Записать файл данных. Атомарно (временный файл + rename): читатель не увидит
 * половину JSON. Вызывается ТОЛЬКО после успешного чтения — испорченный файл
 * не перезаписывается, причина называется в ответе инструмента.
 */
function writeStore(data) {
  const file = storePath();
  const tmp = file + ".tmp";
  fs.writeFileSync(tmp, JSON.stringify(data, null, 2) + "\n", "utf8");
  fs.renameSync(tmp, file);
  return file;
}

/** Привести текст к виду для поиска: регистр, «ё», лишние пробелы. */
function norm(value) {
  return String(value ?? "")
    .toLowerCase()
    .replace(/ё/g, "е")
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim();
}

// Служебные слова запроса. Без них «почему не работает авторизация?» находило
// ПЯТЬ обращений: слово «не» есть в теме почти каждого («не приходит», «не
// сохраняется»), и в контекст модели уходил шум вместо обращения клиента. Список
// короткий и по делу: предлоги, союзы и вопросительные слова.
const STOPWORDS = new Set([
  "не", "ни", "и", "а", "но", "или", "же", "ли", "бы", "в", "во", "на", "за",
  "по", "из", "от", "до", "для", "при", "про", "над", "под", "о", "об", "у",
  "с", "со", "к", "ко", "что", "как", "чем", "кто", "где", "когда", "почему",
  "зачем", "это", "этот", "эта", "его", "ее", "их", "мы", "вы", "он", "она",
  "мне", "мой", "моя", "мои", "все", "уже", "еще", "есть", "был", "была",
  "можно", "нужно", "надо", "the", "and", "for", "you", "why", "not",
]);

/** Значимые слова запроса: 3+ символа и не служебное слово. */
function words(value) {
  return norm(value)
    .split(" ")
    .filter((word) => word.length > 2 && !STOPWORDS.has(word));
}

function cut(value, limit) {
  const body = String(value ?? "").replace(/\s+/g, " ").trim();
  return body.length > limit ? body.slice(0, limit).trimEnd() + " …" : body;
}

/** Местные дата и время в том же виде, что в данных: «2026-10-14 18:55».
 *
 * Не toISOString(): он даёт UTC, и в переписке появлялось время на пять часов
 * раньше местного — рядом с датами самого обращения это выглядело как ошибка.
 */
function stamp(date = new Date()) {
  const pad = (value) => String(value).padStart(2, "0");
  return (
    date.getFullYear() + "-" + pad(date.getMonth() + 1) + "-" + pad(date.getDate()) +
    " " + pad(date.getHours()) + ":" + pad(date.getMinutes())
  );
}

function plural(count, one, few, many) {
  const n = Math.abs(Number(count) || 0) % 100;
  const n1 = n % 10;
  if (n > 10 && n < 20) return many;
  if (n1 === 1) return one;
  if (n1 >= 2 && n1 <= 4) return few;
  return many;
}

function ticketById(store, id) {
  const needle = norm(id).replace(/ /g, "");
  if (!needle) return null;
  return (
    store.tickets.find((item) => norm(item.id).replace(/ /g, "") === needle) || null
  );
}

/**
 * Обращение по номеру: точное совпадение, а если назван ТОЛЬКО номер без
 * префикса («1041») — единственное обращение, которое им заканчивается.
 * Уникальность обязательна: при двух подходящих вернуть одно значило бы
 * ответить не про то обращение, о котором спросили.
 */
function resolveTicket(store, id) {
  const exact = ticketById(store, id);
  if (exact) return exact;
  const digits = norm(id).replace(/ /g, "");
  if (!/^\d+$/.test(digits)) return null;
  const found = store.tickets.filter(
    (item) => norm(item.id).replace(/ /g, "").endsWith(digits)
  );
  return found.length === 1 ? found[0] : null;
}

function userById(store, id) {
  const needle = String(id ?? "").trim().toLowerCase();
  return store.users.find((item) => String(item.id ?? "").toLowerCase() === needle) || null;
}

/** Следующий свободный номер вида «TCK-1043»: максимум по цифрам + 1.
 *
 * Не «последний + 1» и не количество: в файл кладут данные руками, номера могут
 * идти с пропусками и вперемешку, и нумерация по счётчику выдала бы уже занятый
 * номер — а номер обращения это ссылка, которой люди пользуются в письмах.
 */
function nextTicketId(store) {
  let max = 1000;
  for (const item of store.tickets) {
    const found = String(item.id ?? "").match(/(\d+)\s*$/);
    if (found) max = Math.max(max, Number(found[1]));
  }
  return "TCK-" + (max + 1);
}

/** Следующий свободный id клиента («u-1005»). */
function nextUserId(store) {
  let max = 1000;
  for (const item of store.users) {
    const found = String(item.id ?? "").match(/(\d+)\s*$/);
    if (found) max = Math.max(max, Number(found[1]));
  }
  return "u-" + (max + 1);
}

/**
 * Клиент по имени: сначала ТОЧНОЕ совпадение нормализованного имени, потом
 * вхождение (в обе стороны — «Анна» найдёт «Анна Литвинова», и наоборот).
 * Больше искать не по чему: у клиента есть только id и имя (см. шапку файла).
 */
function findUser(store, name) {
  const wanted = norm(name);
  if (!wanted) return null;
  const exact = store.users.find((item) => norm(item.name) === wanted);
  if (exact) return exact;
  const inside = store.users.filter((item) => {
    const key = norm(item.name);
    return key && (key.includes(wanted) || wanted.includes(key));
  });
  return inside.length === 1 ? inside[0] : null;
}

// Статусы, при которых обращение считается ЗАКОНЧЕННЫМ: похожее на них новое
// обращение не «дубликат», а повторная жалоба по той же теме — её как раз надо
// завести отдельно.
const CLOSED_STATUSES = new Set(["решён", "закрыт"]);

/**
 * Похожие ОТКРЫТЫЕ обращения: те же значимые слова в теме или описании.
 *
 * Нужны, чтобы не плодить дубликаты молча. Решение всё равно принимает человек:
 * сервер только показывает найденное и отказывается создавать второе обращение
 * без явного подтверждения (`force`).
 */
function similarOpen(store, subject, description) {
  const tokens = words(String(subject || "") + " " + String(description || ""));
  if (!tokens.length) return [];
  const found = [];
  for (const ticket of store.tickets) {
    if (CLOSED_STATUSES.has(String(ticket.status || ""))) continue;
    const body = norm(String(ticket.subject || "") + " " + String(ticket.description || ""));
    const hits = tokens.filter((token) => body.includes(token)).length;
    if (hits >= Math.max(2, Math.ceil(tokens.length / 2))) found.push(ticket);
  }
  return found.slice(0, 3);
}

/** Пользователь ОБРАЩЕНИЯ: данные клиента нужны в ответе про его вход в систему. */
function userOf(store, ticket) {
  return userById(store, ticket.user_id) || null;
}

function userName(store, ticket) {
  const user = userOf(store, ticket);
  return user ? String(user.name ?? "") : String(ticket.user_id ?? "неизвестен");
}

/**
 * Строка списка: id, тема, статус, приоритет, дата, клиент. Одна строка на
 * обращение — так в бюджет влезает больше обращений, а подробности человек (и
 * модель) получает вызовом get_ticket.
 */
function ticketLine(store, ticket) {
  const parts = [
    String(ticket.id ?? "без id"),
    String(ticket.subject ?? "без темы"),
  ];
  const extra = [];
  if (ticket.status) extra.push("статус: " + ticket.status);
  if (ticket.priority) extra.push("приоритет: " + ticket.priority);
  if (ticket.updated_at || ticket.created_at) {
    extra.push("обновлено: " + String(ticket.updated_at || ticket.created_at));
  }
  extra.push("клиент: " + userName(store, ticket));
  return "- " + parts.join(" · ") + (extra.length ? " (" + extra.join(", ") + ")" : "");
}

function commentLine(comment, limit = COMMENT_CHARS) {
  const at = comment.at ? " (" + comment.at + ")" : "";
  return "  · " + String(comment.author ?? "без автора") + at + ": " + cut(comment.text, limit);
}

/** Полное обращение: клиент, поля, описание, переписка. */
function ticketBlock(store, ticket, { comments = true } = {}) {
  const user = userOf(store, ticket);
  const blocks = [
    "ОБРАЩЕНИЕ " + String(ticket.id ?? "без id"),
    lines([
      ["Тема", ticket.subject],
      ["Статус", ticket.status],
      ["Приоритет", ticket.priority],
      ["Продукт", ticket.product],
      ["Создано", ticket.created_at],
      ["Обновлено", ticket.updated_at],
      ["Клиент", user ? user.name + " (" + user.id + ")" : ticket.user_id],
    ]),
  ];
  if (ticket.description) {
    blocks.push("ОПИСАНИЕ (со слов клиента):\n" + cut(ticket.description, 900));
  }
  const list = Array.isArray(ticket.comments) ? ticket.comments : [];
  if (comments && list.length) {
    blocks.push(
      "ПЕРЕПИСКА (" + list.length + " " + plural(list.length, "сообщение", "сообщения", "сообщений") + "):\n" +
        list.map((item) => commentLine(item)).join("\n")
    );
  } else if (comments) {
    blocks.push("ПЕРЕПИСКА: сообщений нет — в обращении только описание.");
  }
  return blocks.join("\n\n");
}

function clip(body, note) {
  const value = String(body ?? "");
  if (value.length <= BUDGET) return value;
  return value.slice(0, BUDGET - 120).trimEnd() + "\n\n[ответ обрезан по бюджету: " +
    (note || "показано не всё") + "]";
}

/**
 * Поиск обращений по СМЫСЛУ ЗАПРОСА: слова проверяются в теме, описании,
 * переписке, продукте и имени клиента. Тема весит больше переписки: «не работает
 * авторизация» — это тема, а «спасибо, помогло» встречается в каждом закрытом
 * обращении и без веса утопило бы поиск в шуме.
 */
function searchTickets(store, query, status = "", limit = LIST_DEFAULT) {
  const tokens = words(query);
  if (!tokens.length) return [];
  const wanted = norm(status);
  const scored = [];
  for (const ticket of store.tickets) {
    if (!statusMatches(ticket, wanted)) continue;
    const fields = [
      [norm(ticket.subject), 3],
      [norm(ticket.description), 2],
      [norm(ticket.product), 2],
      [norm(userName(store, ticket)), 2],
      [(Array.isArray(ticket.comments) ? ticket.comments : [])
        .map((item) => norm(item.text)).join(" "), 1],
      [norm(ticket.id), 3],
    ];
    let score = 0;
    const hitWords = new Set();
    for (const token of tokens) {
      for (const [body, weight] of fields) {
        if (body && body.includes(token)) {
          score += weight;
          hitWords.add(token);
          break;
        }
      }
    }
    if (score > 0) scored.push({ ticket, score, matched: hitWords.size });
  }
  scored.sort((a, b) => b.score - a.score || b.matched - a.matched);
  // ПОРОГ СЛАБЫХ СОВПАДЕНИЙ: в выдачу идут только совпадения по теме, описанию,
  // продукту, имени клиента или номеру (это 2 и больше баллов) и не более чем
  // вдвое слабее лучшего. Совпадение ТОЛЬКО в переписке (1 балл) — это «спасибо,
  // помогло» из закрытого обращения, а не ответ на вопрос клиента.
  const best = scored.length ? scored[0].score : 0;
  const floor = Math.max(2, best * 0.4);
  return scored
    .filter((item) => item.score >= floor)
    .slice(0, Math.max(1, limit))
    .map((item) => item.ticket);
}

/** Что вообще есть в базе — нужно, когда поиск ничего не нашёл (вместо догадки). */
function overview(store, limit = 8) {
  const byStatus = {};
  for (const ticket of store.tickets) {
    const key = String(ticket.status || "без статуса");
    byStatus[key] = (byStatus[key] || 0) + 1;
  }
  const statuses = Object.keys(byStatus)
    .map((key) => key + ": " + byStatus[key])
    .join(", ");
  const head =
    "В базе " + store.tickets.length + " " +
    plural(store.tickets.length, "обращение", "обращения", "обращений") + " и " +
    store.users.length + " " + plural(store.users.length, "клиент", "клиента", "клиентов") +
    (statuses ? " (" + statuses + ")" : "") + ".";
  const last = [...store.tickets]
    .slice(-limit)
    .reverse()
    .map((ticket) => ticketLine(store, ticket))
    .join("\n");
  return head + (last ? "\nПоследние обращения:\n" + last : "");
}

const server = createServer("support", "1.0.0");

server.registerTool(
  "list_users",
  {
    title: "Клиенты поддержки",
    description:
      "Список клиентов поддержки: id, имя и сколько у кого обращений. "
      + "Можно отфильтровать по имени или id (query).",
    inputSchema: {
      query: z.string().optional().describe("Часть имени или id клиента"),
      limit: z.number().int().min(1).max(LIST_MAX).default(LIST_DEFAULT)
        .describe("Сколько клиентов вернуть (1–20)"),
    },
  },
  async ({ query, limit }) =>
    guard(async () => {
      const store = readStore();
      const needle = norm(query);
      const found = store.users.filter((user) => {
        if (!needle) return true;
        return [user.id, user.name].some((value) => norm(value).includes(needle));
      });
      if (!found.length) {
        return failure(
          "Клиентов по «" + query + "» в базе нет. Всего клиентов: " +
          store.users.length + ". Имена: " +
          store.users.map((user) => String(user.name ?? user.id)).join(", ")
        );
      }
      const shown = found.slice(0, limit);
      const body = shown
        .map((user) => {
          const count = store.tickets.filter(
            (ticket) => String(ticket.user_id) === String(user.id)
          ).length;
          return (
            "- " + user.name + " (" + user.id + ")" + " · обращений: " + count
          );
        })
        .join("\n");
      const tail = found.length > shown.length
        ? "\nПоказано " + shown.length + " из " + found.length + " — уменьшите список аргументом limit."
        : "";
      return text(clip("КЛИЕНТЫ ПОДДЕРЖКИ (" + found.length + ")\n" + body + tail));
    })
);

server.registerTool(
  "get_user",
  {
    title: "Карточка клиента",
    description:
      "Карточка клиента (имя, id) и список его обращений. Клиент ищется по id "
      + "или имени — например «Анна».",
    inputSchema: {
      user: z.string().min(1).describe("id или имя клиента"),
    },
  },
  async ({ user }) =>
    guard(async () => {
      const store = readStore();
      const needle = norm(user);
      const exact = userById(store, user);
      const found = exact || store.users.find((item) =>
        norm(item.name).includes(needle)
      );
      if (!found) {
        return failure(
          "Клиент «" + user + "» не найден. Известные клиенты: " +
          store.users.map((item) => String(item.name ?? item.id)).join(", ") +
          ". Уточните имя или посмотрите список инструментом list_users."
        );
      }
      const own = store.tickets.filter(
        (ticket) => String(ticket.user_id) === String(found.id)
      );
      const body = lines([
        ["Имя", found.name],
        ["id", found.id],
      ]);
      const tickets = own.length
        ? "ОБРАЩЕНИЯ КЛИЕНТА (" + own.length + "):\n" +
          own.map((ticket) => ticketLine(store, ticket)).join("\n")
        : "ОБРАЩЕНИЙ У КЛИЕНТА НЕТ.";
      return text(clip("КАРТОЧКА КЛИЕНТА\n" + body + "\n\n" + tickets));
    })
);

server.registerTool(
  "search_tickets",
  {
    title: "Поиск обращений по тексту",
    description:
      "Найти обращения по словам запроса — по теме, описанию, переписке, продукту " +
      "и имени клиента. Это первый инструмент для вопроса вида «почему не работает " +
      "авторизация», «не приходит письмо», «ошибка при выгрузке отчёта»: он находит " +
      "обращение, даже когда номер неизвестен.",
    inputSchema: {
      query: z.string().min(1).describe("Слова запроса, например «не работает авторизация»"),
      status: z.string().optional()
        .describe("Оставить только обращения с этим статусом: %s" % STATUSES.join(", ")),
      limit: z.number().int().min(1).max(LIST_MAX).default(LIST_DEFAULT)
        .describe("Сколько обращений вернуть (1–20)"),
    },
  },
  async ({ query, status, limit }) =>
    guard(async () => {
      const store = readStore();
      const wanted = String(status || "").trim() ? normalizeStatus(status) : "";
      if (String(status || "").trim() && !wanted) {
        return failure("Статус «" + status + "» не распознан. " + statusHint());
      }
      const found = searchTickets(store, query, wanted, limit);
      if (!found.length) {
        // ПУСТОЙ ОТВЕТ БЕЗ ПРИЧИНЫ МОДЕЛЬ ДОСТРАИВАЕТ: «обращений нет» и «не
        // понял запрос» — разные вещи, и второе лечится другим запросом.
        return failure(
          "По запросу «" + query + "»" + (status ? " со статусом «" + status + "»" : "") +
          " обращений в базе нет." +
          (words(query).length < 2
            ? " Слов в запросе мало — назовите тему словами клиента («не работает"
              + " вход», «не приходит письмо»)."
            : "") +
          "\n" + overview(store)
        );
      }
      const head =
        "НАЙДЕННЫЕ ОБРАЩЕНИЯ (" + found.length + " по запросу «" + query + "»" +
        (status ? ", статус «" + status + "»" : "") + "):";
      const blocks = found.map((ticket) => {
        const excerpt = ticket.description
          ? "\n  описание: " + cut(ticket.description, EXCERPT)
          : "";
        const firstComment = (Array.isArray(ticket.comments) ? ticket.comments : [])[0];
        const comment = firstComment
          ? "\n  первое сообщение: " + cut(firstComment.text, EXCERPT)
          : "";
        return ticketLine(store, ticket) + excerpt + comment;
      });
      return text(clip(
        head + "\n" + blocks.join("\n") +
        "\n\nПодробности одного обращения (вся переписка и данные клиента) — " +
        "инструментом get_ticket с его id."
      ));
    })
);

server.registerTool(
  "get_ticket",
  {
    title: "Обращение целиком",
    description:
      "Одно обращение целиком: тема, статус, приоритет, продукт, даты, описание со " +
      "слов клиента, вся переписка и имя клиента. Нужен id — " +
      "например «TCK-1041»; если id неизвестен, сначала search_tickets. " +
      "В режиме поддержки возвращаются только обращения своего клиента.",
    inputSchema: {
      id: z.string().min(1).describe("Номер обращения, например «TCK-1041»"),
      // КТО СПРАШИВАЕТ. В режиме поддержки приложение передаёт сюда имя клиента
      // из профиля, и чужое обращение сервер НЕ отдаёт: профиль — это клиент, и
      // показывать ему обращения других клиентов нельзя. Пустое значение —
      // обычный вызов (так работают проверки и ручная диагностика).
      client: z.string().optional()
        .describe("Имя клиента, которому должно принадлежать обращение (подставляет приложение)"),
    },
  },
  async ({ id, client }) =>
    guard(async () => {
      const store = readStore();
      const found = resolveTicket(store, id);
      const viewer = String(client || "").trim();
      if (found && viewer) {
        const owner = userOf(store, found);
        const ownerName = owner ? String(owner.name || "") : "";
        if (norm(ownerName) !== norm(viewer)) {
          // Не «нет обращения», а именно «не ваше»: причина должна быть честной,
          // иначе человек решит, что номер устарел, и будет искать его заново.
          return failure(
            "Обращение " + String(found.id) + " принадлежит другому клиенту"
            + (ownerName ? " (" + ownerName + ")" : "")
            + " — в режиме поддержки показываются только ваши обращения. "
            + "Свои обращения можно посмотреть списком: «/support что у меня "
            + "в работе»."
          );
        }
      }
      if (!found) {
        // ПОХОЖИЕ — по САМОМУ НОМЕРУ, а не текстовым поиском: у «TCK-9999» слов
        // для поиска нет, и общий поиск возвращал бы все обращения подряд под
        // видом «похожих по номеру».
        const needle = norm(id).replace(/ /g, "");
        const near = store.tickets
          .filter((ticket) => {
            const key = norm(ticket.id).replace(/ /g, "");
            return key.includes(needle) || needle.includes(key);
          })
          .slice(0, 5);
        return failure(
          "Обращения с номером «" + id + "» нет. " +
          (near.length
            ? "Похожие по номеру: " + near.map((item) => item.id).join(", ") + "."
            : "Последние обращения в базе:\n" +
              store.tickets.slice(-5).reverse().map((item) => ticketLine(store, item)).join("\n") +
              "\nНомер обращения можно уточнить поиском по тексту — search_tickets.")
        );
      }
      return text(clip(ticketBlock(store, found)));
    })
);

server.registerTool(
  "list_tickets",
  {
    title: "Список обращений",
    description:
      "Обращения поддержки списком: все, по клиенту (user) или по статусу (status). " +
      "Для вопроса «что у клиента в работе» — user и status вместе.",
    inputSchema: {
      user: z.string().optional().describe("id или имя клиента"),
      status: z.string().optional()
        .describe("Статус обращения: %s" % STATUSES.join(", ")),
      limit: z.number().int().min(1).max(LIST_MAX).default(LIST_DEFAULT)
        .describe("Сколько обращений вернуть (1–20)"),
    },
  },
  async ({ user, status, limit }) =>
    guard(async () => {
      const store = readStore();
      let owner = null;
      if (user) {
        const needle = norm(user);
        owner = userById(store, user) || store.users.find((item) =>
          norm(item.name).includes(needle)
        );
        if (!owner) {
          return failure(
            "Клиент «" + user + "» не найден — список обращений по нему не построить. " +
            "Известные клиенты: " + store.users.map((item) => String(item.name ?? item.id)).join(", ")
          );
        }
      }
      const wanted = String(status || "").trim() ? normalizeStatus(status) : "";
      if (String(status || "").trim() && !wanted) {
        return failure("Статус «" + status + "» не распознан. " + statusHint());
      }
      const found = store.tickets.filter((ticket) => {
        if (owner && String(ticket.user_id) !== String(owner.id)) return false;
        if (!statusMatches(ticket, wanted)) return false;
        return true;
      });
      if (!found.length) {
        return failure(
          "Обращений" + (owner ? " у клиента " + owner.name : "") +
          (status ? " со статусом «" + status + "»" : "") + " нет.\n" + overview(store)
        );
      }
      const shown = found.slice(0, limit);
      const head =
        "ОБРАЩЕНИЯ (" + found.length + ")" +
        (owner ? " · клиент: " + owner.name : "") +
        (status ? " · статус: " + status : "") + ":";
      const tail = found.length > shown.length
        ? "\nПоказано " + shown.length + " из " + found.length + " (limit)."
        : "";
      return text(clip(head + "\n" + shown.map((ticket) => ticketLine(store, ticket)).join("\n") + tail));
    })
);

server.registerTool(
  "create_ticket",
  {
    title: "Создать обращение",
    description:
      "Завести новое обращение в поддержку: тема и описание со слов клиента. " +
      "Заявителя называть НЕ нужно при обычной работе — его подставляет " +
      "приложение из профиля клиента; в аргументе requester он приходит уже " +
      "готовым. Клиент ищется по имени, а если такого нет — заводится. " +
      "Перед созданием сервер проверяет похожие ОТКРЫТЫЕ обращения и без " +
      "подтверждения (force) второе не создаёт.",
    inputSchema: {
      // НЕ required — и это важно. Заявителя подставляет КОД из профиля клиента
      // (app/routers/chat.py, `_fill_requester`), а обязательный аргумент, которого
      // модель не знает, диспетчер трактует по своему правилу: «не хватает
      // обязательного сведения — не вызывай инструмент, пусть агент уточнит». На
      // живой просьбе «заведи обращение» это выглядело так: create_ticket вообще
      // не вызывался, и человек слышал «завести не могу, нет клиента» (живой
      // случай 15.10). Аргумент остаётся в схеме, чтобы его можно было назвать
      // явно («заведи обращение для Дмитрия Ковалёва»).
      requester: z.string().optional()
        .describe("Имя заявителя (клиента); при работе из приложения подставляется из профиля — заполнять не нужно"),
      subject: z.string().min(1).describe("Тема обращения словами клиента"),
      description: z.string().optional()
        .describe("Что случилось, со слов клиента: симптомы, когда началось, что уже пробовал"),
      product: z.string().optional().describe("Продукт или раздел, которого касается обращение"),
      priority: z.string().optional()
        .describe("Приоритет обращения: низкий, средний или высокий (можно по-английски)"),
      force: z.boolean().default(false)
        .describe("Создать, даже если найдено похожее открытое обращение (только после решения человека)"),
    },
  },
  async ({ requester, subject, description, product, priority, force }) =>
    guard(async () => {
      // Читаем ДО записи: испорченный файл не перезаписывается (см. add_comment).
      const store = readStore();
      const today = stamp().slice(0, 10);
      // ЗАЯВИТЕЛЬ ОБЯЗАТЕЛЕН ПО СМЫСЛУ: без него обращение было бы «от никого».
      // В приложении значение подставляет код из профиля, но если оно пустое
      // (профиль не заполнен, вызов пришёл не из приложения) — честный отказ с
      // причиной, а не карточка клиента с пустым именем.
      const who = String(requester || "").trim();
      if (!who) {
        return failure("Заявитель не назван — обращение не создано. Назовите клиента "
          + "(имя или id) или заполните поле «Имя» в профиле: при работе из "
          + "приложения заявитель подставляется оттуда.");
      }
      const level = normalizePriority(priority);
      if (!level) {
        return failure("Приоритет «" + priority + "» не распознан. Допустимые: "
          + PRIORITIES.join(", ") + " (или low/medium/high).");
      }

      // ПОХОЖИЕ ОТКРЫТЫЕ ОБРАЩЕНИЯ — не запрет, а вопрос человеку. Молча создать
      // второе обращение по той же теме значит раздвоить переписку: клиент
      // ответит в одном, поддержка будет смотреть в другое.
      const similar = similarOpen(store, subject, description);
      if (similar.length && !force) {
        return failure(
          "Похожее ОТКРЫТОЕ обращение уже есть — новое не создано:\n" +
          similar.map((ticket) => ticketLine(store, ticket)).join("\n") +
          "\nРешите: дописать в существующее (инструмент add_comment с его id) " +
          "или завести отдельное — тогда повторите вызов с force=true. " +
          "Закрытые обращения («решён», «закрыт») дубликатом не считаются: " +
          "повторная жалоба по той же теме заводится как новое обращение."
        );
      }

      // КЛИЕНТ: находим по имени, иначе заводим. Карточка нужна, чтобы видеть
      // ДРУГИЕ обращения этого человека — ради этого она и существует.
      let user = findUser(store, who);
      let created_user = false;
      if (!user) {
        user = { id: nextUserId(store), name: who };
        store.users.push(user);
        created_user = true;
      }

      const ticket = {
        id: nextTicketId(store),
        user_id: user.id,
        subject: String(subject).trim(),
        status: "новый",
        priority: level,
        product: String(product || "").trim(),
        created_at: today,
        updated_at: today,
        description: String(description || "").trim(),
        comments: [],
      };
      store.tickets.push(ticket);
      const file = writeStore(store);

      return text(clip(
        "ОБРАЩЕНИЕ СОЗДАНО: " + ticket.id + "\n" +
        lines([
          ["Тема", ticket.subject],
          ["Статус", ticket.status],
          ["Приоритет", ticket.priority],
          ["Продукт", ticket.product],
          ["Создано", ticket.created_at],
          ["Заявитель", user.name + " (" + user.id + ")"],
        ]) +
        "\n" +
        (created_user
          ? "Клиента с таким именем в базе не было — карточка заведена.\n"
          : "Обращение привязано к существующему клиенту.\n") +
        (ticket.description ? "ОПИСАНИЕ (со слов клиента):\n" + cut(ticket.description, 600) + "\n" : "") +
        "\nПЕРЕПИСКА ПУСТА: это только что заведённое обращение. Ответ клиенту " +
        "пишется отдельным инструментом add_comment (он откладывается до " +
        "подтверждения плана).\n" +
        "Файл данных: " + file + "\n" +
        "Запись сохранена на диск: это подтверждение самого инструмента, а не пересказ."
      ));
    })
);

server.registerTool(
  "add_comment",
  {
    title: "Добавить комментарий в обращение",
    description:
      "Записать сообщение в переписку обращения: ответ клиенту или пометку " +
      "поддержки. Нужны id обращения и текст. Это ЕДИНСТВЕННЫЙ инструмент, " +
      "который меняет данные: остальные только читают.",
    inputSchema: {
      id: z.string().min(1).describe("Номер обращения, например «TCK-1041»"),
      text: z.string().min(1).describe("Текст сообщения"),
      author: z.string().optional().describe("Автор сообщения (по умолчанию «поддержка»)"),
    },
  },
  async ({ id, text: body, author }) =>
    guard(async () => {
      // Читаем ДО записи: если файл испорчен, readStore бросит исключение, и
      // guard вернёт причину — перезаписи чужих данных не случится.
      const store = readStore();
      // Тот же разбор номера, что у get_ticket: «1041» без префикса — законный
      // номер, если он однозначен. Запись называет полный id в подтверждении,
      // поэтому человек видит, куда именно ушло сообщение.
      const found = resolveTicket(store, id);
      if (!found) {
        return failure(
          "Обращения с номером «" + id + "» нет — комментарий не записан. " +
          "Найдите обращение инструментом search_tickets и повторите с его id."
        );
      }
      const comment = {
        author: String(author || "поддержка").trim() || "поддержка",
        at: stamp(),
        text: String(body).trim(),
      };
      if (!comment.text) {
        return failure("Пустой текст сообщения — комментарий не записан.");
      }
      if (!Array.isArray(found.comments)) found.comments = [];
      found.comments.push(comment);
      found.updated_at = comment.at.slice(0, 10);
      const file = writeStore(store);
      const total = found.comments.length;
      return text(
        "СООБЩЕНИЕ ЗАПИСАНО в обращение " + String(found.id) + " (" +
        String(found.subject ?? "без темы") + ").\n" +
        lines([
          ["Автор", comment.author],
          ["Когда", comment.at],
          ["Текст", cut(comment.text, COMMENT_CHARS)],
          ["Сообщений в обращении", total],
          ["Статус обращения", found.status],
          ["Файл данных", file],
        ]) +
        "\nЗапись сохранена на диск: это подтверждение самого инструмента, а не пересказ."
      );
    })
);

await serve(server);
