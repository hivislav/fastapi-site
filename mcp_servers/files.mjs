#!/usr/bin/env node
// MCP-сервер «Файлы проекта» — РАБОЧИЕ операции с файлами рабочей копии.
//
// ЗАЧЕМ ОН НУЖЕН. До него агент умел только отвечать текстом: содержимое файлов
// он видел лишь урывками (git_show_file показывает один файл), а изменить файл не
// мог вовсе — «обнови документацию» заканчивалось текстом в чате, который надо
// было руками переносить в файл. Здесь у агента появляются НАСТОЯЩИЕ операции:
// он сам решает, какие файлы прочитать, ищет по нескольким сразу, разбирает
// структуру, а затем создаёт и правит файлы — изменения остаются на диске, и
// рядом с ними агент видит diff.
//
// ЧТО ЗДЕСЬ ЕСТЬ (9 инструментов):
//   чтения   — list_files, read_file, read_files, search_files, outline_file,
//              diff_files;
//   записи   — write_file (создать или переписать), edit_file (точная замена
//              фрагмента), append_file (дописать).
//
// ПИШУЩИЕ ВЫЗОВЫ ОТКЛАДЫВАЮТСЯ ДО ПОДТВЕРЖДЕНИЯ ПЛАНА, и это гарантия
// приложения, а не сервера: имена write_/edit_/append_ попадают в классификатор
// `RESULT_RE` (app/ai/mcp.py), поэтому до «ок» пользователя выполняются только
// чтения, а сама запись доигрывается на шаге выполнения (см. `is_deferred_call`,
// `split_calls`, `chain["deferred"]`). Пользователь утверждает план ДО того, как
// что-то изменится на диске.
//
// ГАРАНТИИ КОДА (не просьбы в промпте):
//   * пути проверяются: выход за корень (`../`), каталоги из DENY_DIRS (data/,
//     venv/, node_modules/, .git/ …) и файлы с секретами (.env, ключи) закрыты и
//     на чтение, и на запись; символические ссылки разыменовываются, поэтому
//     «ссылка наружу» проверку не обходит;
//   * запись АТОМАРНАЯ: содержимое сначала пишется в соседний временный файл и
//     затем переименовывается — оборванная запись не оставит половину файла;
//   * предохранитель от потери: если новая версия РЕЗКО меньше прежней (больше
//     чем втрое, а файл был от 4 КБ), рядом остаётся `<файл>.bak` с прежним
//     текстом, и об этом сказано в ответе — тот же приём, что у workspace-файла
//     приложения (workspace._keep_backup);
//   * ответы ограничены бюджетом, и обрезка НАЗВАНА: клиент MCP в приложении
//     (app/ai/mcp.py, RESULT_CHARS = 4000) молча обрезал бы длинный текст, а это
//     выглядело бы как «в файле больше ничего нет».
//
// ЧТО ВИДИТ МОДЕЛЬ. Инструменты отвечают ТЕКСТОМ (никаких вложений): чтения —
// содержимым с номерами строк, записи — фактом правки и её diff. Идентификаторы
// здесь не выдумываются: путь не является идентификатором (ID_ARG_RE его не
// ловит), поэтому охрана вызовов (guard_calls) записи не блокирует.

import { execFile } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { z } from "zod";
import { createServer, guard, serve, text } from "./lib/common.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));

// Корень рабочей копии: по умолчанию — сам проект (каталог серверов лежит внутри
// него). Другой каталог задаётся переменной окружения FILES_REPO_DIR — так
// проверки и демонстрационные прогоны работают на ВРЕМЕННОЙ копии и не трогают
// рабочие файлы проекта.
//
// Путь СРАЗУ разыменовывается (`realpathSync`): на macOS временный каталог
// `/var/folders/…` — это символическая ссылка на `/private/var/folders/…`, и без
// разыменования КАЖДЫЙ существующий файл внутри выглядел бы как «ссылка ведёт за
// пределы рабочей копии». Проверка «наружу» ниже сравнивает уже настоящие пути.
const ROOT = (() => {
  const raw = path.resolve(process.env.FILES_REPO_DIR || path.join(HERE, ".."));
  try {
    return fs.realpathSync(raw);
  } catch {
    return raw;
  }
})();

// Бюджет ответа одного инструмента (символы). Ниже клиентского предела (4000):
// место нужно ещё на заголовок, на подписи обрезки и на пустые строки.
const RESULT_BUDGET = 3400;
// Сколько показываем в списках, совпадениях и отрывках чтения.
const LIST_LIMIT_DEFAULT = 60;
const LIST_LIMIT_MAX = 400;
const MATCH_LIMIT_DEFAULT = 30;
const MATCH_LIMIT_MAX = 80;
// Контекст совпадений: больше двадцати строк на совпадение в бюджет не влезает,
// но СХЕМА это значение не отвергает — лишнее зажимается здесь (см. ниже).
const CONTEXT_LINES_MAX = 20;
const READ_LINES_DEFAULT = 120;
const READ_LINES_MAX = 400;
const READ_FILES_MAX = 8;              // сколько файлов за одно чтение
const DIFF_FILES_MAX = 12;
// Пределы самого файла: больше читать нечего (это уже не «файл проекта», а
// свалка), а писать такое одной правкой нельзя — приложение и не пропустит
// столько в аргументах вызова (предел ответа диспетчера).
const MAX_READ_BYTES = 2 * 1024 * 1024;
const MAX_SEARCH_BYTES = 1024 * 1024;
const MAX_WRITE_CHARS = 200000;
// Порог предохранителя: новая версия меньше прежней больше чем втрое — рядом
// остаётся .bak. Втрое — тот же коэффициент, что у workspace приложения.
const SHRINK_RATIO = 3;
const SHRINK_MIN_BYTES = 4096;

// ЗАКРЫТЫЕ ПУТИ. Проверяются в СЕРВЕРЕ, а не в промпте: данные и пути выбирает
// модель, а «не читай .env» просьбой не гарантируется (правило проекта: промпт —
// просьба, гарантия — код). Каталоги закрыты целиком: история диалогов (data/),
// окружение (venv/, node_modules/), служебный .git, кэши.
const DENY_DIRS = new Set([
  ".git", "data", "venv", ".venv", "env", "node_modules", "__pycache__",
  ".idea", ".vscode", ".pytest_cache", ".mypy_cache", ".ruff_cache",
]);
// Файлы с секретами: даже внутри разрешённого каталога их показывать нельзя.
const DENY_FILE = [
  /^\.env/i, /\.local$/i, /secret/i, /credential/i, /^auth_secret/i,
  /\.pem$/i, /\.key$/i, /\.p12$/i, /\.pfx$/i, /\.crt$/i, /\.cer$/i,
  /^id_(rsa|dsa|ecdsa|ed25519)/i, /^\.netrc$/i, /^\.htpasswd$/i,
];
// Двоичные расширения: их нельзя ни искать по тексту, ни ПЕРЕЗАПИСЫВАТЬ текстом
// (запись строки в .png или .zip — гарантированная порча файла).
const BINARY_EXT = new Set([
  ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".bmp", ".tiff", ".heic",
  ".pdf", ".zip", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar", ".jar",
  ".woff", ".woff2", ".ttf", ".otf", ".eot", ".so", ".dylib", ".dll", ".exe",
  ".bin", ".pyc", ".class", ".o", ".a", ".mp3", ".mp4", ".mov", ".wav", ".sqlite",
  ".db", ".xlsx", ".docx", ".pptx", ".odt",
]);

const server = createServer("files", "1.0.0");

// ---------------------------------------------------------------------------
// Пути и защита
// ---------------------------------------------------------------------------
/**
 * Проверить путь из аргумента инструмента: вернуть относительный путь внутри
 * корня. Пусто — «путь не задан» (пустая строка).
 *
 * Закрыты: выход за корень (`../`), каталоги из DENY_DIRS и файлы с секретами.
 * Символические ссылки разыменовываются: ссылка вида `link -> /etc/passwd`
 * проверку не обходит.
 */
function safePath(raw, { what = "чтения" } = {}) {
  const value = String(raw === undefined || raw === null ? "" : raw).trim();
  if (!value) return "";
  if (value.startsWith("-")) {
    throw new Error("путь не может начинаться с «-»: " + value);
  }
  const abs = path.resolve(ROOT, value);
  const rel = path.relative(ROOT, abs);
  if (!rel || rel.startsWith("..") || path.isAbsolute(rel)) {
    throw new Error("путь вне рабочей копии: " + value);
  }
  const parts = rel.split(path.sep).filter(Boolean);
  for (const part of parts.slice(0, -1)) {
    if (DENY_DIRS.has(part)) {
      throw new Error("каталог " + part + "/ закрыт для " + what + " сервером файлов");
    }
  }
  const base = parts[parts.length - 1] || "";
  if (DENY_DIRS.has(base)) {
    throw new Error("каталог " + base + "/ закрыт для " + what + " сервером файлов");
  }
  if (DENY_FILE.some((re) => re.test(base))) {
    throw new Error("файл " + base + " закрыт для " + what + " сервером файлов (секреты)");
  }
  let real = abs;
  try {
    real = fs.realpathSync(abs);
  } catch {
    // Файла ещё нет (новая запись) — проверяем существующий родительский каталог.
    let dir = path.dirname(abs);
    while (dir !== path.dirname(dir)) {
      try {
        real = path.join(fs.realpathSync(dir), path.relative(dir, abs));
        break;
      } catch {
        dir = path.dirname(dir);
      }
    }
  }
  const realRel = path.relative(ROOT, real);
  if (realRel.startsWith("..") || path.isAbsolute(realRel)) {
    throw new Error("ссылка ведёт за пределы рабочей копии: " + value);
  }
  return rel;
}

/** Похож ли файл на двоичный: NUL-байт в первых килобайтах — почти всегда «да». */
function looksBinary(abs) {
  let fd = null;
  try {
    fd = fs.openSync(abs, "r");
    const buf = Buffer.alloc(4096);
    const read = fs.readSync(fd, buf, 0, buf.length, 0);
    for (let i = 0; i < read; i += 1) {
      if (buf[i] === 0) return true;
    }
    return false;
  } catch {
    return false;
  } finally {
    if (fd !== null) {
      try {
        fs.closeSync(fd);
      } catch {
        /* закрытие не критично */
      }
    }
  }
}

/** Текст файла или понятная причина, почему его нельзя прочитать. */
function readText(rel, { maxBytes = MAX_READ_BYTES } = {}) {
  const abs = path.join(ROOT, rel);
  let stat;
  try {
    stat = fs.statSync(abs);
  } catch {
    throw new Error("файла нет: " + rel + " (список файлов — list_files)");
  }
  if (stat.isDirectory()) {
    throw new Error(rel + " — это каталог, а не файл (список файлов — list_files)");
  }
  if (stat.size > maxBytes) {
    throw new Error("файл " + rel + " слишком велик для чтения: " + fmtSize(stat.size)
      + " (предел " + fmtSize(maxBytes) + ")");
  }
  if (BINARY_EXT.has(path.extname(rel).toLowerCase()) || looksBinary(abs)) {
    throw new Error("файл " + rel + " двоичный — текстом его не читают");
  }
  return fs.readFileSync(abs, "utf8");
}

/** Отформатировать размер по-человечески. */
function fmtSize(bytes) {
  const value = Number(bytes) || 0;
  if (value < 1024) return value + " Б";
  if (value < 1024 * 1024) return (value / 1024).toFixed(1) + " КБ";
  return (value / (1024 * 1024)).toFixed(1) + " МБ";
}

// ---------------------------------------------------------------------------
// Текст ответа
// ---------------------------------------------------------------------------
const TRUNCATED_NOTE =
  "…обрезано: показано %shown% символов из %total%. Сузьте запрос "
  + "(path, glob, limit, from_line/to_line).";

/** Обрезать длинный текст по границе строки и честно сказать, что показано. */
function clip(body, budget, note = TRUNCATED_NOTE) {
  const source = String(body || "");
  if (source.length <= budget) return source;
  let cut = source.slice(0, budget);
  const nl = cut.lastIndexOf("\n");
  if (nl > budget * 0.5) cut = cut.slice(0, nl);
  return cut + "\n" + note.replace("%shown%", String(cut.length))
    .replace("%total%", String(source.length));
}

/** Строки текста (без завершающей пустой). */
function splitLines(body) {
  const value = String(body === undefined || body === null ? "" : body);
  const rows = value.split("\n");
  if (rows.length > 1 && rows[rows.length - 1] === "") rows.pop();
  return rows;
}

/** Пронумерованные строки: «  12| текст» — как в git_show_file. */
function numbered(rows, start) {
  const width = String(start + rows.length - 1).length;
  return rows.map((line, index) =>
    String(start + index).padStart(width, " ") + "| " + line);
}

/**
 * Простой построчный diff: LCS для небольшой середины, иначе — честная замена
 * блока целиком. Возвращает массив строк вида «  контекст», «-было», «+стало».
 *
 * Зачем свой, а не `git diff`: правка может идти в файле, который git не
 * отслеживает (новый), и в каталоге вообще без git — а показать, ЧТО изменилось,
 * нужно всегда (это и есть требование «изменения видны как diff»).
 */
function diffLines(before, after, context = 2) {
  const a = splitLines(before);
  const b = splitLines(after);
  let start = 0;
  while (start < a.length && start < b.length && a[start] === b[start]) start += 1;
  let endA = a.length;
  let endB = b.length;
  while (endA > start && endB > start && a[endA - 1] === b[endB - 1]) {
    endA -= 1;
    endB -= 1;
  }
  const midA = a.slice(start, endA);
  const midB = b.slice(start, endB);
  const ops = lcsOps(midA, midB);
  const out = [];
  const from = Math.max(0, start - context);
  for (let i = from; i < start; i += 1) out.push("  " + a[i]);
  for (const [kind, line] of ops) {
    out.push((kind === "=" ? "  " : kind === "-" ? "- " : "+ ") + line);
  }
  const to = Math.min(a.length, endA + context);
  for (let i = endA; i < to; i += 1) out.push("  " + a[i]);
  return { lines: out, removed: midA.length, added: midB.length, at: from + 1 };
}

/** Операции diff для середины: LCS, если она небольшая, иначе замена блоком. */
function lcsOps(a, b) {
  if (!a.length && !b.length) return [];
  if (!a.length) return b.map((line) => ["+", line]);
  if (!b.length) return a.map((line) => ["-", line]);
  // Предохранитель от квадратичной сложности: на больших блоках показываем
  // замену целиком — это честнее, чем съесть всё время сервера.
  if (a.length * b.length > 250000) {
    return a.map((line) => ["-", line]).concat(b.map((line) => ["+", line]));
  }
  const table = [];
  for (let i = 0; i <= a.length; i += 1) table.push(new Array(b.length + 1).fill(0));
  for (let i = a.length - 1; i >= 0; i -= 1) {
    for (let j = b.length - 1; j >= 0; j -= 1) {
      table[i][j] = a[i] === b[j]
        ? table[i + 1][j + 1] + 1
        : Math.max(table[i + 1][j], table[i][j + 1]);
    }
  }
  const ops = [];
  let i = 0;
  let j = 0;
  while (i < a.length && j < b.length) {
    if (a[i] === b[j]) {
      ops.push(["=", a[i]]);
      i += 1;
      j += 1;
    } else if (table[i + 1][j] >= table[i][j + 1]) {
      ops.push(["-", a[i]]);
      i += 1;
    } else {
      ops.push(["+", b[j]]);
      j += 1;
    }
  }
  while (i < a.length) {
    ops.push(["-", a[i]]);
    i += 1;
  }
  while (j < b.length) {
    ops.push(["+", b[j]]);
    j += 1;
  }
  return ops;
}

/** Строка-сводка о diff: «−3 +5» — то, что человек видит у git diff --stat. */
function diffStat(removed, added) {
  return "−" + removed + " +" + added + " строк";
}

/**
 * АТОМАРНАЯ запись: содержимое ложится в соседний временный файл и переименование
 * подменяет файл целиком. Оборванная запись (сбой, отмена, кончившееся место) не
 * оставит половину файла — либо прежняя версия, либо новая.
 */
function writeAtomic(rel, content) {
  const abs = path.join(ROOT, rel);
  const dir = path.dirname(abs);
  fs.mkdirSync(dir, { recursive: true });
  const tmp = path.join(dir, "." + path.basename(abs) + ".tmp-" + process.pid);
  try {
    fs.writeFileSync(tmp, content, "utf8");
    fs.renameSync(tmp, abs);
  } catch (err) {
    try {
      fs.unlinkSync(tmp);
    } catch {
      /* временного файла может уже не быть */
    }
    throw err;
  }
}

/** Существует ли файл (не каталог) внутри корня. */
function fileExists(rel) {
  try {
    return fs.statSync(path.join(ROOT, rel)).isFile();
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------------------
// Обход каталогов
// ---------------------------------------------------------------------------
/** Простое сопоставление с glob: `*`, `**`, `?` (без внешних зависимостей). */
function globRe(pattern) {
  const value = String(pattern || "").trim();
  if (!value) return null;
  let out = "";
  for (let i = 0; i < value.length; i += 1) {
    const ch = value[i];
    if (ch === "*") {
      if (value[i + 1] === "*") {
        out += ".*";
        i += 1;
        if (value[i + 1] === "/") i += 1;
      } else {
        out += "[^/]*";
      }
    } else if (ch === "?") {
      out += "[^/]";
    } else if ("\\^$.|+()[]{}".includes(ch)) {
      out += "\\" + ch;
    } else {
      out += ch;
    }
  }
  return new RegExp("^" + out + "$");
}

/**
 * Обойти каталог и вернуть относительные пути файлов (без закрытых каталогов).
 * Символические ссылки на каталоги не разворачиваются: ссылка — не часть проекта.
 */
function walk(dirRel, { limit = 5000 } = {}) {
  const out = [];
  const stack = [dirRel];
  while (stack.length && out.length < limit) {
    const current = stack.pop();
    const abs = path.join(ROOT, current);
    let entries;
    try {
      entries = fs.readdirSync(abs, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const entry of entries) {
      const rel = current ? current + "/" + entry.name : entry.name;
      if (entry.isDirectory()) {
        if (DENY_DIRS.has(entry.name) || entry.name.startsWith(".")) continue;
        stack.push(rel);
      } else if (entry.isFile()) {
        if (DENY_FILE.some((re) => re.test(entry.name))) continue;
        out.push(rel);
      }
    }
  }
  return out.sort();
}

/** Файлы для поиска: обход + glob, без двоичных и слишком больших. */
function searchableFiles(dirRel, pattern) {
  const re = globRe(pattern);
  const found = [];
  // `dir` может указывать и на ОДИН файл («поищи в app/ai/mcp.py») — тогда обход
  // не нужен: ищем ровно в нём.
  const candidates = [];
  let single = null;
  if (dirRel) {
    try {
      if (fs.statSync(path.join(ROOT, dirRel)).isFile()) single = dirRel;
    } catch {
      single = null;
    }
  }
  if (single) candidates.push(single);
  else candidates.push(...walk(dirRel || ""));
  for (const rel of candidates) {
    const base = rel.split("/").pop() || "";
    if (re && !re.test(rel) && !re.test(base)) continue;
    if (BINARY_EXT.has(path.extname(rel).toLowerCase())) continue;
    let stat;
    try {
      stat = fs.statSync(path.join(ROOT, rel));
    } catch {
      continue;
    }
    if (!stat.isFile() || stat.size > MAX_SEARCH_BYTES) continue;
    if (looksBinary(path.join(ROOT, rel))) continue;
    found.push(rel);
  }
  return found;
}

/** Структурные шаблоны для outline_file по расширению. */
function outlinePatterns(ext) {
  if (ext === ".md" || ext === ".markdown") {
    return [[/^#{1,6}\s+\S/, "раздел"]];
  }
  if (ext === ".py") {
    return [
      [/^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)/, "функция"],
      [/^\s*class\s+([A-Za-z_]\w*)/, "класс"],
    ];
  }
  if ([".js", ".mjs", ".cjs", ".ts", ".jsx", ".tsx"].includes(ext)) {
    return [
      [/registerTool\(\s*["'`]([^"'`]+)/, "инструмент"],
      [/^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)/, "функция"],
      [/^\s*(?:export\s+)?class\s+([A-Za-z_$][\w$]*)/, "класс"],
      [/^\s*(?:export\s+)?const\s+([A-Za-z_$][\w$]*)\s*=/, "константа"],
    ];
  }
  if (ext === ".json") {
    return [[/^\s{0,2}"[^"]+"\s*:/, "поле"]];
  }
  if (ext === ".css" || ext === ".html") {
    return [[/^\s*[.#@a-zA-Z][^{]{0,60}\{/, "правило"]];
  }
  return [];
}

/** Запуск git в корне (для diff_files): список аргументов, без оболочки. */
function runGit(args, { timeoutMs = 15000 } = {}) {
  return new Promise((resolve) => {
    execFile(
      "git",
      ["-C", ROOT, "-c", "core.quotepath=false"].concat(args),
      {
        cwd: ROOT,
        timeout: timeoutMs,
        maxBuffer: 8 * 1024 * 1024,
        env: { ...process.env, GIT_PAGER: "cat", GIT_OPTIONAL_LOCKS: "0" },
      },
      (err, stdout) => resolve({ ok: !err, out: String(stdout || "") }),
    );
  });
}

// ---------------------------------------------------------------------------
// Инструменты: ЧТЕНИЕ
// ---------------------------------------------------------------------------
server.registerTool(
  "list_files",
  {
    title: "Список файлов проекта",
    description:
      "Файлы рабочей копии: каталог (dir), шаблон имени (glob, например "
      + "**/*.py или *.md) и предел. Закрытые каталоги (data/, venv/, "
      + "node_modules/, .git/) в список не попадают. С этого стоит начинать "
      + "работу с файлами: сначала увидеть, что есть, потом читать нужное.",
    inputSchema: {
      dir: z.string().optional().describe("Каталог внутри проекта, например app/ai"),
      glob: z.string().optional().describe("Шаблон пути: *.md, **/*.py, tests/test_*"),
      limit: z.number().int().min(1).max(5000).default(LIST_LIMIT_DEFAULT)
        .describe("Сколько файлов показать (по умолчанию " + LIST_LIMIT_DEFAULT
                  + ", больше " + LIST_LIMIT_MAX + " не показываю)"),
    },
  },
  async ({ dir, glob, limit }) => guard(async () => {
    const rel = safePath(dir);
    const re = globRe(glob);
    const cap = Math.max(1, Math.min(LIST_LIMIT_MAX, Number(limit) || LIST_LIMIT_DEFAULT));
    const all = walk(rel);
    const files = [];
    for (const item of all) {
      const base = item.split("/").pop() || "";
      if (re && !re.test(item) && !re.test(base)) continue;
      files.push(item);
    }
    if (!files.length) {
      return text("Файлов не нашлось"
        + (rel ? " в каталоге " + rel : "")
        + (glob ? " по шаблону " + glob : "")
        + ". Попробуйте другой каталог или шаблон (dir/glob).");
    }
    const shown = files.slice(0, cap);
    const rows = shown.map((item) => {
      let size = "";
      try {
        size = "  " + fmtSize(fs.statSync(path.join(ROOT, item)).size);
      } catch {
        size = "";
      }
      return item + size;
    });
    const header = "Файлы рабочей копии" + (rel ? " в " + rel : "")
      + (glob ? " по шаблону " + glob : "")
      + ": найдено " + files.length + ", показано " + shown.length + "\n";
    const tail = files.length > shown.length
      ? "\n…показаны не все: всего " + files.length + " файлов. "
        + "Уточните dir/glob или поднимите limit."
      : "";
    return text(clip(header + rows.join("\n"), RESULT_BUDGET - tail.length) + tail);
  }),
);

server.registerTool(
  "read_file",
  {
    title: "Прочитать файл",
    description:
      "Содержимое файла с НОМЕРАМИ СТРОК (from_line/to_line — отрывок). Номера "
      + "нужны, чтобы потом точечно править файл (edit_file) и ссылаться на строки "
      + "в ответе. Большой файл читайте отрывками.",
    inputSchema: {
      path: z.string().describe("Путь файла внутри проекта, например app/ai/mcp.py"),
      from_line: z.number().int().min(1).optional().describe("Первая строка (нумерация с 1)"),
      to_line: z.number().int().min(1).optional().describe("Последняя строка"),
    },
  },
  async ({ path: rawPath, from_line, to_line }) => guard(async () => {
    const rel = safePath(rawPath);
    if (!rel) throw new Error("не указан путь файла (path)");
    const body = readText(rel);
    const rows = splitLines(body);
    const start = Math.max(1, Number(from_line) || 1);
    const wantEnd = Number(to_line) || 0;
    if (start > rows.length) {
      return text("В файле " + rel + " только " + rows.length + " строк — начало за его концом.");
    }
    const stop = wantEnd > 0 ? Math.min(rows.length, wantEnd) : rows.length;
    const slice = rows.slice(start - 1, stop);
    const header = "Файл: " + rel + " — строк " + rows.length
      + (slice.length !== rows.length
        ? ", показаны " + start + "–" + (start + slice.length - 1) : "") + "\n";
    const tail = start + slice.length - 1 < rows.length
      ? "\n\n…файл продолжается: всего " + rows.length + " строк. "
        + "Следующий отрывок — from_line=" + (start + slice.length) + "."
      : "";
    return text(clip(header + numbered(slice, start).join("\n"),
      RESULT_BUDGET - tail.length) + tail);
  }),
);

server.registerTool(
  "read_files",
  {
    title: "Прочитать несколько файлов",
    description:
      "Содержимое 2–8 файлов за ОДИН вызов (у каждого — отрывок from_line/to_line). "
      + "Нужен для работы по НЕСКОЛЬКИМ файлам: сравнить, найти общее, собрать "
      + "сводку. Бюджет ответа делится между файлами поровну.",
    inputSchema: {
      paths: z.array(z.string()).max(READ_FILES_MAX)
        .describe("Пути файлов внутри проекта (до " + READ_FILES_MAX + ")"),
      from_line: z.number().int().min(1).optional().describe("Первая строка для всех файлов"),
      to_line: z.number().int().min(1).optional().describe("Последняя строка для всех файлов"),
    },
  },
  async ({ paths, from_line, to_line }) => guard(async () => {
    const list = (Array.isArray(paths) ? paths : []).slice(0, READ_FILES_MAX);
    if (!list.length) throw new Error("не указаны файлы (paths)");
    const start = Math.max(1, Number(from_line) || 1);
    const stop = Number(to_line) || 0;
    const perFile = Math.max(400, Math.floor(RESULT_BUDGET / list.length) - 120);
    const blocks = [];
    const problems = [];
    for (const rawPath of list) {
      let rel;
      try {
        rel = safePath(rawPath);
        const body = readText(rel);
        const rows = splitLines(body);
        const slice = rows.slice(start - 1, stop > 0 ? Math.min(rows.length, stop) : rows.length);
        const head = "=== " + rel + " (строк " + rows.length
          + (slice.length !== rows.length ? ", показаны " + start + "–"
            + (start + slice.length - 1) : "") + ") ===";
        const shown = numbered(slice, start);
        // Строки режем по бюджету на файл, но не молча: подпись скажет, сколько не влезло.
        let used = 0;
        const kept = [];
        for (const line of shown) {
          if (used + line.length + 1 > perFile) break;
          kept.push(line);
          used += line.length + 1;
        }
        const more = shown.length - kept.length > 0
          ? "\n…в этом файле не показано строк: " + (shown.length - kept.length)
            + " (читайте отдельно read_file с from_line/to_line)"
          : "";
        blocks.push(head + "\n" + kept.join("\n") + more);
      } catch (err) {
        problems.push(rel || String(rawPath) + ": " + (err && err.message ? err.message : err));
      }
    }
    if (!blocks.length) {
      return text("Ни один файл не прочитан:\n" + problems.join("\n"));
    }
    const header = "Прочитано файлов: " + blocks.length + " из " + list.length + "\n";
    const tail = problems.length ? "\n\nНЕ прочитаны:\n" + problems.join("\n") : "";
    return text(clip(header + blocks.join("\n\n"), RESULT_BUDGET - tail.length) + tail);
  }),
);

server.registerTool(
  "search_files",
  {
    title: "Поиск по нескольким файлам",
    description:
      "Найти текст или регулярное выражение по ВСЕМ файлам проекта (или в "
      + "каталоге/по шаблону): «где используется этот компонент/функция/api». "
      + "Возвращает файл, номер строки и саму строку — это адрес правки для "
      + "edit_file. Без regex=true запрос ищется как обычный текст.",
    inputSchema: {
      query: z.string().describe("Что искать: текст или (при regex=true) выражение"),
      dir: z.string().optional().describe("Ограничить каталогом, например app/ai"),
      glob: z.string().optional().describe("Шаблон файлов: *.py, **/*.md"),
      regex: z.boolean().default(false).describe("true — query это регулярное выражение"),
      ignore_case: z.boolean().default(true).describe("Не различать регистр (по умолчанию да)"),
      context_lines: z.number().int().min(0).max(50).default(0)
        .describe("Сколько строк контекста показать после совпадения (до "
                  + CONTEXT_LINES_MAX + ")"),
      limit: z.number().int().min(1).max(500).default(MATCH_LIMIT_DEFAULT)
        .describe("Сколько совпадений показать (по умолчанию " + MATCH_LIMIT_DEFAULT
                  + ", больше " + MATCH_LIMIT_MAX + " не показываю)"),
    },
  },
  async ({ query, dir, glob, regex, ignore_case, context_lines, limit }) => guard(async () => {
    const needle = String(query || "");
    if (!needle) throw new Error("пустой запрос поиска (query)");
    let re;
    try {
      re = regex
        ? new RegExp(needle, ignore_case ? "iu" : "u")
        : new RegExp(needle.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), ignore_case ? "iu" : "u");
    } catch (err) {
      throw new Error("выражение не разобрано: " + (err && err.message ? err.message : err)
        + ". Упростите запрос или уберите regex=true");
    }
    const dirRel = safePath(dir);
    const cap = Math.max(1, Math.min(MATCH_LIMIT_MAX, Number(limit) || MATCH_LIMIT_DEFAULT));
    const context = Math.max(0, Math.min(CONTEXT_LINES_MAX, Number(context_lines) || 0));
    const files = searchableFiles(dirRel, glob);
    const found = [];
    let scanned = 0;
    let total = 0;
    let skipped = 0;
    for (const rel of files) {
      let body;
      try {
        body = fs.readFileSync(path.join(ROOT, rel), "utf8");
      } catch {
        skipped += 1;
        continue;
      }
      scanned += 1;
      const rows = splitLines(body);
      for (let i = 0; i < rows.length; i += 1) {
        // Сброс lastIndex: с флагом g один и тот же объект RegExp «помнит» позицию.
        re.lastIndex = 0;
        if (!re.test(rows[i])) continue;
        total += 1;
        if (found.length >= cap) continue;
        const extra = context > 0
          ? rows.slice(i + 1, i + 1 + context).map((line, k) => "      " + (i + 2 + k) + "| " + line)
          : [];
        found.push(rel + ":" + (i + 1) + ": " + rows[i].trim() + (extra.length ? "\n" + extra.join("\n") : ""));
      }
    }
    if (!total) {
      return text("Совпадений нет: «" + needle + "» не найдено в " + scanned + " файлах"
        + (dirRel ? " каталога " + dirRel : "")
        + (glob ? " по шаблону " + glob : "")
        + ". Стоит проверить формулировку запроса (другое имя, часть слова) "
        + "или расширить область поиска (dir/glob).");
    }
    const shownCount = found.length;
    const header = "Найдено совпадений: " + total + " (показано " + shownCount
      + "), просмотрено файлов: " + scanned
      + (skipped ? ", не прочитано: " + skipped : "") + "\n";
    const tail = total > shownCount
      ? "\n…показаны не все совпадения: всего " + total
        + ". Уточните query/dir/glob или поднимите limit."
      : "";
    return text(clip(header + found.join("\n"), RESULT_BUDGET - tail.length) + tail);
  }),
);

server.registerTool(
  "outline_file",
  {
    title: "Структура файла",
    description:
      "Каркас файла БЕЗ содержимого: разделы markdown, функции и классы Python, "
      + "функции/классы/инструменты в JS, поля JSON — с номерами строк. Дешёвый "
      + "способ понять устройство файла перед чтением или правкой.",
    inputSchema: {
      path: z.string().describe("Путь файла внутри проекта"),
    },
  },
  async ({ path: rawPath }) => guard(async () => {
    const rel = safePath(rawPath);
    if (!rel) throw new Error("не указан путь файла (path)");
    const ext = path.extname(rel).toLowerCase();
    // Двоичный файл отвергаем ДО разбора типа: «структуру .png не разбираю» —
    // неверный ответ, файл не текстовый вовсе.
    if (BINARY_EXT.has(ext)) {
      throw new Error("файл " + rel + " двоичный — текстом его не читают");
    }
    const patterns = outlinePatterns(ext);
    if (!patterns.length) {
      return text("Структуру файлов " + (ext || "без расширения")
        + " не разбираю — прочитайте файл целиком (read_file) или поищите по нему "
        + "(search_files).");
    }
    const body = readText(rel);
    const rows = splitLines(body);
    const items = [];
    for (let i = 0; i < rows.length; i += 1) {
      for (const [re, kind] of patterns) {
        const match = re.exec(rows[i]);
        if (!match) continue;
        const name = (match[1] || rows[i].trim()).slice(0, 90);
        items.push(String(i + 1).padStart(5, " ") + "  " + kind + ": " + name);
        break;
      }
    }
    if (!items.length) {
      return text("В файле " + rel + " структурных элементов не нашлось (" + rows.length
        + " строк). Прочитайте его целиком — read_file.");
    }
    const header = "Структура файла: " + rel + " — строк " + rows.length
      + ", элементов " + items.length + "\n";
    return text(clip(header + items.join("\n"), RESULT_BUDGET));
  }),
);

/**
 * НОВЫЕ (неотслеживаемые) файлы рабочей копии — с учётом фильтра по путям.
 *
 * Нужны в ЛЮБОМ случае, а не только при пустом diff: `git diff` не показывает
 * новые файлы вовсе, и если рядом с правкой завёлся новый файл, о нём иначе не
 * сказано ни слова — «в diff его нет» читается как «его не создавали».
 */
async function untrackedUnder(list) {
  const status = await runGit(["status", "--porcelain=v1", "-unormal"]);
  return String(status.out || "").split("\n")
    .filter((line) => line.startsWith("??"))
    .map((line) => line.slice(3).trim())
    .filter((name) => !list.length
      || list.some((item) => name === item || name.startsWith(item + "/")))
    .slice(0, 8);
}

server.registerTool(
  "diff_files",
  {
    title: "Что изменилось в файлах",
    description:
      "Изменения рабочей копии против последнего коммита (git diff HEAD), при "
      + "необходимости только по указанным файлам. Это способ ПОКАЗАТЬ результат "
      + "правок: что именно изменилось, строками. Новые (неотслеживаемые) файлы "
      + "git в diff не показывает — о них сказано отдельно, их содержимое смотрите "
      + "read_file.",
    inputSchema: {
      paths: z.array(z.string()).max(DIFF_FILES_MAX).optional()
        .describe("Только эти файлы (до " + DIFF_FILES_MAX + "); пусто — все изменения"),
      context_lines: z.number().int().min(0).max(50).default(3)
        .describe("Сколько строк контекста вокруг правки (по умолчанию 3)"),
    },
  },
  async ({ paths, context_lines }) => guard(async () => {
    const list = (Array.isArray(paths) ? paths : []).slice(0, DIFF_FILES_MAX)
      .map((item) => safePath(item)).filter(Boolean);
    const context = Math.max(0, Math.min(CONTEXT_LINES_MAX, Number(context_lines) || 3));
    const inside = await runGit(["rev-parse", "--is-inside-work-tree"]);
    if (!inside.ok || inside.out.trim() !== "true") {
      return text("Каталог " + ROOT + " не является git-репозиторием, поэтому diff "
        + "взять неоткуда. Что изменено — видно по ответам write_file/edit_file/"
        + "append_file, а содержимое файлов читает read_file.");
    }
    const args = ["diff", "HEAD", "--no-color", "--unified=" + context];
    if (list.length) args.push("--", ...list);
    const result = await runGit(args, { timeoutMs: 20000 });
    if (!result.ok) {
      throw new Error("git не смог показать diff (проверьте, что каталог — репозиторий)");
    }
    const body = result.out.trim();
    const untracked = await untrackedUnder(list);
    if (!body) {
      // Пустой diff — это ФАКТ, и у него бывают разные причины: важно назвать их,
      // иначе «изменений нет» читается как «правки не сохранились».
      let note = "Изменений против последнего коммита нет";
      note += list.length ? " (проверены: " + list.join(", ") + ")." : ".";
      if (untracked.length) {
        note += " НО есть НОВЫЕ файлы, которых нет в истории (их git в diff не "
          + "показывает вовсе): " + untracked.join(", ")
          + ". Это НЕ «изменений нет» — содержимое смотрите read_file.";
      }
      if (!list.length) note += " Рабочая копия совпадает с последним коммитом.";
      return text(note);
    }
    const changed = (body.match(/^diff --git /gm) || []).length;
    const stat = await runGit(["diff", "HEAD", "--shortstat"].concat(
      list.length ? ["--", ...list] : []));
    const header = "Изменения рабочей копии против последнего коммита"
      + (list.length ? " (" + list.join(", ") + ")" : "")
      + ": файлов " + changed + (stat.ok && stat.out.trim() ? ", " + stat.out.trim() : "")
      + "\n";
    // Изменения есть, но НОВЫЕ файлы в diff не видны — говорим и о них: иначе
    // созданный файл выглядел бы несозданным.
    const tail = untracked.length
      ? "\n\nКроме этого, есть НОВЫЕ файлы, которых нет в истории (git показывает их "
        + "только как «??»): " + untracked.join(", ")
        + ". Их содержимое — read_file."
      : "";
    return text(clip(header + body, RESULT_BUDGET - tail.length) + tail);
  }),
);

// ---------------------------------------------------------------------------
// Инструменты: ЗАПИСЬ (откладываются приложением до подтверждения плана)
// ---------------------------------------------------------------------------
/** Строка без отступов и повторных пробелов — для снисходительного сравнения. */
function normalizeLine(line) {
  return String(line || "").trim().replace(/\s+/g, " ");
}

/**
 * Где в тексте лежит фрагмент, если сравнивать строки БЕЗ отступов.
 * Возвращает {at, count} — единственное совпадение, либо null (нет или несколько).
 * Неоднозначность не угадываем: править не тот участок хуже, чем честно отказать.
 */
function fuzzyWindow(text, needle) {
  const lines = splitLines(text);
  const wanted = splitLines(needle).map(normalizeLine);
  if (!wanted.length || wanted.some((line) => !line)) return null;
  const hits = [];
  for (let i = 0; i + wanted.length <= lines.length; i += 1) {
    let ok = true;
    for (let k = 0; k < wanted.length; k += 1) {
      if (normalizeLine(lines[i + k]) !== wanted[k]) {
        ok = false;
        break;
      }
    }
    if (ok) hits.push(i);
  }
  return hits.length === 1 ? { at: hits[0], count: wanted.length } : null;
}

/** Общая часть ответов о записи: факт, размеры, diff, предупреждения. */
function writeReply({ rel, action, before, after, extra = "" }) {
  const beforeRows = before === null ? 0 : splitLines(before).length;
  const afterRows = splitLines(after).length;
  const diff = before === null ? null : diffLines(before, after);
  const head = action + " " + rel + ": строк " + (before === null ? 0 : beforeRows)
    + " → " + afterRows
    + (diff ? " (" + diffStat(diff.removed, diff.added) + ")" : "")
    + "; файл записан на диск (" + fmtSize(Buffer.byteLength(after, "utf8")) + ").";
  const parts = [head];
  if (diff && diff.lines.length) {
    const shown = diff.lines.slice(0, 40);
    parts.push("Изменения:");
    parts.push(shown.join("\n"));
    if (diff.lines.length > shown.length) {
      parts.push("…diff показан не весь: строк изменений " + diff.lines.length
        + " (полный — diff_files).");
    }
  }
  if (extra) parts.push(extra);
  return text(clip(parts.join("\n"), RESULT_BUDGET));
}

server.registerTool(
  "write_file",
  {
    title: "Создать или переписать файл",
    description:
      "Записать файл целиком: создаёт новый или ЗАМЕНЯЕТ прежнее содержимое "
      + "(содержимое передаётся аргументом content). Для правки существующего "
      + "файла лучше edit_file: он меняет только нужный фрагмент и не требует "
      + "переписывать файл целиком. Выполняется ПОСЛЕ подтверждения плана.",
    inputSchema: {
      path: z.string().describe("Путь файла внутри проекта, например docs/ADR-001.md"),
      content: z.string().describe("Полное новое содержимое файла (пустое нельзя)"),
    },
  },
  async ({ path: rawPath, content }) => guard(async () => {
    const rel = safePath(rawPath, { what: "записи" });
    if (!rel) throw new Error("не указан путь файла (path)");
    const body = String(content ?? "");
    if (!body.trim().length) {
      throw new Error("пустое содержимое: чтобы создать пустой файл, это не нужно, "
        + "а прежний текст так можно потерять");
    }
    if (body.length > MAX_WRITE_CHARS) {
      throw new Error("содержимое слишком велико: " + body.length + " символов "
        + "(предел " + MAX_WRITE_CHARS + "). Разделите файл на несколько вызовов "
        + "(write_file + append_file)");
    }
    if (body.includes("\u0000")) {
      throw new Error("содержимое не текстовое (есть нулевой байт)");
    }
    if (BINARY_EXT.has(path.extname(rel).toLowerCase())) {
      throw new Error("в файл " + rel + " текстом не пишут — это двоичный формат");
    }
    const existed = fileExists(rel);
    const before = existed ? readText(rel, { maxBytes: MAX_READ_BYTES }) : null;
    let extra = "";
    if (before !== null) {
      // ПРЕДОХРАНИТЕЛЬ ОТ ПОТЕРИ: резкое сокращение файла — почти всегда признак
      // того, что модель собрала новую версию по отрывку и «съела» остальное.
      // Прежний текст остаётся рядом в .bak, и об этом сказано прямо.
      const beforeBytes = Buffer.byteLength(before, "utf8");
      const afterBytes = Buffer.byteLength(body, "utf8");
      if (beforeBytes >= SHRINK_MIN_BYTES && afterBytes * SHRINK_RATIO < beforeBytes) {
        const backup = rel + ".bak";
        writeAtomic(backup, before);
        extra = "ВНИМАНИЕ: новая версия резко меньше прежней ("
          + fmtSize(beforeBytes) + " → " + fmtSize(afterBytes) + "). Прежний текст "
          + "сохранён рядом: " + backup + ". Если это не задумано, верните нужные "
          + "части из него.";
      }
    }
    writeAtomic(rel, body);
    return writeReply({
      rel,
      action: before === null ? "СОЗДАН" : "ОБНОВЛЁН",
      before,
      after: body,
      extra,
    });
  }),
);

server.registerTool(
  "edit_file",
  {
    title: "Заменить фрагмент в файле",
    description:
      "Точная замена фрагмента текста в файле (find → replace): правка раздела, "
      + "строки, значения. Фрагмент должен встречаться ОДИН раз, иначе вызов "
      + "отклоняется — уточните find или поставьте all=true. Выполняется ПОСЛЕ "
      + "подтверждения плана.",
    inputSchema: {
      path: z.string().describe("Путь файла внутри проекта"),
      find: z.string().describe("Что заменить — точный фрагмент из файла (пустой нельзя)"),
      replace: z.string().describe("Чем заменить (пустая строка — удалить фрагмент)"),
      all: z.boolean().default(false)
        .describe("true — заменить ВСЕ вхождения фрагмента (по умолчанию только одно)"),
    },
  },
  async ({ path: rawPath, find, replace, all }) => guard(async () => {
    const rel = safePath(rawPath, { what: "записи" });
    if (!rel) throw new Error("не указан путь файла (path)");
    if (!fileExists(rel)) {
      throw new Error("файла нет: " + rel + ". Новый файл создаёт write_file");
    }
    const needle = String(find ?? "");
    if (!needle) throw new Error("пустой фрагмент поиска (find)");
    const before = readText(rel, { maxBytes: MAX_READ_BYTES });
    const parts = before.split(needle);
    const count = parts.length - 1;
    if (!count) {
      // ТОЧНЫЙ ФРАГМЕНТ НЕ НАЙДЕН — ПРОБУЕМ ПО СМЫСЛУ СТРОК. Модель часто
      // воспроизводит текст с другим отступом или лишними пробелами, и правка
      // отклонялась, хотя нужное место она назвала верно (живой случай 16.10:
      // шаг «подключить команду» оставался невыполненным). Сравниваем строки БЕЗ
      // отступов и повторных пробелов, и правим ТОЛЬКО если такое место ОДНО.
      const fuzzy = fuzzyWindow(before, needle);
      if (fuzzy === null) {
        throw new Error("фрагмент не найден в " + rel + " — прочитайте файл (read_file) "
          + "и возьмите точный текст, включая отступы; для ВСТАВКИ рядом с "
          + "одной строкой есть insert_in_file (короткий якорь)");
      }
      const lines = splitLines(before);
      const replacementLines = splitLines(String(replace ?? ""));
      const afterLines = lines.slice(0, fuzzy.at)
        .concat(replacementLines, lines.slice(fuzzy.at + fuzzy.count));
      const after = afterLines.join("\n") + (before.endsWith("\n") ? "\n" : "");
      writeAtomic(rel, after);
      return writeReply({
        rel, action: "ИЗМЕНЁН", before, after,
        extra: "Совпадение найдено ПО НОРМАЛИЗОВАННОМУ тексту (строки "
          + (fuzzy.at + 1) + "–" + (fuzzy.at + fuzzy.count)
          + "): отступы или повторные пробелы отличались от указанных в find.",
      });
    }
    if (count > 1 && !all) {
      throw new Error("фрагмент встречается " + count + " раз — уточните find "
        + "(возьмите больше соседних строк) или поставьте all=true, если менять надо все");
    }
    // Замена передаётся ФУНКЦИЕЙ: строковая замена в JS трактует «$&», «$1» и
    // подобное как подстановку из совпадения — и текст правки приехал бы в файл
    // искажённым (например, «цена $& руб.» развернулась бы в найденный фрагмент).
    const replacement = String(replace ?? "");
    const after = all ? parts.join(replacement) : before.replace(needle, () => replacement);
    writeAtomic(rel, after);
    return writeReply({
      rel,
      action: "ИЗМЕНЁН",
      before,
      after,
      extra: count > 1 ? "Заменено вхождений: " + count : "",
    });
  }),
);

server.registerTool(
  "insert_in_file",
  {
    title: "Вставить строки рядом с якорем",
    description:
      "Вставить текст в СУЩЕСТВУЮЩИЙ файл рядом с ЯКОРЕМ: `anchor` — короткий "
      + "фрагмент одной строки (имя функции, команды, маркера), рядом с которой "
      + "вставляем. Так ПОДКЛЮЧАЮТ новое в готовый код: не нужно переписывать файл "
      + "и не нужен точный многострочный фрагмент (в отличие от edit_file). Якорь "
      + "обязан встречаться РОВНО один раз — иначе назову строки, где он есть, и "
      + "попрошу уточнить. Выполняется ПОСЛЕ подтверждения плана.",
    inputSchema: {
      path: z.string().describe("Путь файла внутри проекта"),
      anchor: z.string().describe("Фрагмент строки-якоря (короткий и уникальный)"),
      text: z.string().describe("Что вставить (несколько строк — через перевод строки)"),
      where: z.enum(["after", "before"]).default("after")
        .describe("after — сразу после строки-якоря, before — перед ней"),
      indent: z.boolean().default(true)
        .describe("Повторить отступ строки-якоря у вставляемых строк (по умолчанию да)"),
    },
  },
  async ({ path: rawPath, anchor, text, where, indent }) => guard(async () => {
    const rel = safePath(rawPath, { what: "записи" });
    if (!rel) throw new Error("не указан путь файла (path)");
    if (!fileExists(rel)) {
      throw new Error("файла нет: " + rel + ". Новый файл создаёт write_file");
    }
    const needle = String(anchor ?? "");
    if (!needle.trim()) throw new Error("пустой якорь (anchor)");
    const body = String(text ?? "");
    if (!body.trim()) throw new Error("нечего вставлять (пустой text)");
    if (BINARY_EXT.has(path.extname(rel).toLowerCase())) {
      throw new Error("в файл " + rel + " текстом не пишут — это двоичный формат");
    }
    const before = readText(rel, { maxBytes: MAX_READ_BYTES });
    const lines = splitLines(before);
    const hits = [];
    for (let i = 0; i < lines.length; i += 1) {
      if (lines[i].includes(needle)) hits.push(i);
    }
    if (!hits.length) {
      throw new Error("якорь не найден в " + rel + ": «" + needle + "» — возьмите "
        + "фрагмент строки из файла (search_files → read_file)");
    }
    if (hits.length > 1) {
      throw new Error("якорь встречается " + hits.length + " раз (строки "
        + hits.slice(0, 6).map((n) => n + 1).join(", ")
        + ") — уточните его: возьмите больше символов строки, чтобы место стало "
        + "единственным");
    }
    const at = hits[0];
    // ОТСТУП БЕРЁМ У ЯКОРЯ: вставляемый код должен встать на своё место в блоке,
    // а не с нулевым отступом (это частая причина «вставил, но файл сломан»).
    const base = indent ? (lines[at].match(/^[ \t]*/) || [""])[0] : "";
    const addition = splitLines(body).map((line) => (line.trim() ? base + line : line));
    const index = where === "before" ? at : at + 1;
    const afterLines = lines.slice(0, index).concat(addition, lines.slice(index));
    const after = afterLines.join("\n") + (before.endsWith("\n") ? "\n" : "");
    writeAtomic(rel, after);
    return writeReply({
      rel, action: "ИЗМЕНЁН", before, after,
      extra: "Вставлено " + addition.length + " " + (addition.length === 1 ? "строка" : "строк")
        + " после строки " + (at + 1) + " (якорь найден один раз).",
    });
  }),
);

server.registerTool(
  "append_file",
  {
    title: "Дописать в конец файла",
    description:
      "Добавить текст в КОНЕЦ файла (создаёт его, если файла нет). Нужен для "
      + "журналов и списков: запись в CHANGELOG, строка в отчёт, новый пункт в "
      + "перечень. Прежнее содержимое не трогается. Выполняется ПОСЛЕ "
      + "подтверждения плана.",
    inputSchema: {
      path: z.string().describe("Путь файла внутри проекта"),
      content: z.string().describe("Что дописать (без завершающего перевода строки)"),
    },
  },
  async ({ path: rawPath, content }) => guard(async () => {
    const rel = safePath(rawPath, { what: "записи" });
    if (!rel) throw new Error("не указан путь файла (path)");
    const body = String(content ?? "");
    if (!body.trim().length) throw new Error("нечего дописывать (пустой content)");
    if (BINARY_EXT.has(path.extname(rel).toLowerCase())) {
      throw new Error("в файл " + rel + " текстом не пишут — это двоичный формат");
    }
    const existed = fileExists(rel);
    const before = existed ? readText(rel, { maxBytes: MAX_READ_BYTES }) : "";
    if (before.length + body.length > MAX_WRITE_CHARS) {
      throw new Error("файл станет слишком большим: " + (before.length + body.length)
        + " символов (предел " + MAX_WRITE_CHARS + ")");
    }
    const glue = !before.length || before.endsWith("\n") ? "" : "\n";
    const after = before + glue + body + "\n";
    writeAtomic(rel, after);
    const rows = splitLines(after);
    const tail = rows.slice(-5).map((line) => "  " + line).join("\n");
    return text(clip("ДОПИСАНО в " + rel + ": строк " + splitLines(before).length
      + " → " + rows.length + " (+" + splitLines(body).length
      + "); файл записан на диск.\nКонец файла теперь:\n" + tail, RESULT_BUDGET));
  }),
);

await serve(server);
