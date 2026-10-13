#!/usr/bin/env node
// MCP-сервер «Git-репозиторий проекта».
//
// Источник: ЛОКАЛЬНЫЙ репозиторий самого проекта (внешней сети нет вовсе).
// Зачем он нужен агенту: у модели нет доступа к рабочей копии, поэтому «на какой
// ветке идёт работа», «что я изменил за сегодня» и «что делает такой-то модуль»
// она может только выдумать. Здесь она получает факты: ветку, последний коммит,
// список файлов, diff и содержимое файла.
//
// Инструменты: git_status, git_files, git_diff, git_log, git_show_file.
//
// ТОЛЬКО ЧТЕНИЕ, и это гарантия кода, а не обещание:
//   * выполняются исключительно читающие команды (status/ls-files/log/diff/show);
//   * любая запись в репозиторий, коммит, checkout, push и запуск произвольной
//     команды через этот сервер невозможны — аргументы передаются в git списком
//     (оболочка не участвует, `execFile` без shell), а имя команды задаёт код;
//   * GIT_OPTIONAL_LOCKS=0 — даже `git status` не обновляет индекс в .git;
//   * пути проверяются: выход за пределы репозитория и каталоги/файлы с
//     секретами (.env, data/, venv/, node_modules/, ключи) закрыты.
//
// ПОЧЕМУ ОТВЕТЫ ОГРАНИЧЕНЫ ПО РАЗМЕРУ. Клиент MCP в приложении
// (app/ai/mcp.py, RESULT_CHARS = 4000) берёт у инструмента не больше 4000
// символов. Поэтому сервер сам держит ответы в бюджете (RESULT_BUDGET) и ЧЕСТНО
// пишет, что показал не всё: молчаливая обрезка выглядела бы как «изменений
// больше нет» или «файл кончился».

import { execFile } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { z } from "zod";
import { createServer, guard, serve, text } from "./lib/common.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));

// Корень репозитория: по умолчанию — сам проект (каталог серверов лежит внутри
// него). Другой репозиторий задаётся переменной окружения GIT_REPO_DIR.
const REPO = path.resolve(process.env.GIT_REPO_DIR || path.join(HERE, ".."));

// Бюджет ответа одного инструмента (символы). Ниже клиентского предела: место
// нужно ещё на заголовок, на подписи обрезки и на пустые строки.
const RESULT_BUDGET = 3400;
// Сколько файлов показываем в списках и сколько строк — в diff и файле.
const MAX_FILES = 120;
const MAX_STATUS_FILES = 25;
const MAX_LOG = 20;
const SHOW_LINES_DEFAULT = 90;
const SHOW_LINES_MAX = 400;
// Пределы процесса git: буфер на большой diff и время на «тяжёлый» запрос.
const MAX_BUFFER = 16 * 1024 * 1024;
const GIT_TIMEOUT_MS = 15000;

// ЗАКРЫТЫЕ ПУТИ. Проверяются в сервере, а не в промпте: данные модели выбирает
// модель, и «не читай .env» просьбой не гарантируется (в проекте правило
// звучит так: промпт — просьба, гарантия — код). Каталоги закрыты целиком:
// история диалогов (data/), окружение (venv/, node_modules/), служебный .git.
const DENY_DIRS = new Set([
  ".git", "data", "venv", ".venv", "env", "node_modules", "__pycache__",
  ".idea", ".vscode", ".pytest_cache",
]);
// Файлы с секретами: даже внутри разрешённого каталога их показывать нельзя.
const DENY_FILE = [
  /^\.env/i, /\.local$/i, /secret/i, /credential/i, /^auth_secret/i,
  /\.pem$/i, /\.key$/i, /\.p12$/i, /\.pfx$/i, /\.crt$/i, /\.cer$/i,
  /^id_(rsa|dsa|ecdsa|ed25519)/i, /^\.netrc$/i, /^\.htpasswd$/i,
];

const server = createServer("git", "1.0.0");

// ---------------------------------------------------------------------------
// Запуск git
// ---------------------------------------------------------------------------
/**
 * Выполнить git в репозитории проекта. Возвращает {ok, code, stdout, stderr}.
 *
 * Аргументы передаются СПИСКОМ (`execFile`, без оболочки): ни одна подстановка
 * из аргументов инструмента не может стать второй командой. `-c
 * core.quotepath=false` — чтобы русские имена файлов не приезжали
 * восьмеричными escape-последовательностями.
 */
function git(args, { timeoutMs = GIT_TIMEOUT_MS } = {}) {
  const full = ["-C", REPO, "-c", "core.quotepath=false"].concat(args);
  return new Promise((resolve) => {
    execFile(
      "git",
      full,
      {
        cwd: REPO,
        timeout: timeoutMs,
        maxBuffer: MAX_BUFFER,
        env: {
          ...process.env,
          GIT_PAGER: "cat",
          GIT_OPTIONAL_LOCKS: "0",
          GIT_TERMINAL_PROMPT: "0",
        },
      },
      (err, stdout, stderr) => {
        const code = err && typeof err.code === "number" ? err.code : err ? -1 : 0;
        resolve({
          ok: !err,
          code,
          stdout: String(stdout || ""),
          stderr: String(stderr || ""),
          spawnError: err && err.code === "ENOENT" ? err : null,
        });
      }
    );
  });
}

/** Текст ошибки git для человека: причина, а не «команда не удалась». */
function gitError(result, doing) {
  if (result.spawnError) {
    return "не найден git: установите git или укажите путь в PATH (" + doing + ")";
  }
  const detail = String(result.stderr || "").trim().split("\n")[0] || "";
  return "git не смог выполнить «" + doing + "»" + (detail ? ": " + detail : "");
}

// Корень репозитория узнаём у самого git и запоминаем: каталог, куда указала
// переменная, может быть ПОДкаталогом репозитория, и тогда проверять пути надо
// от настоящего корня, а не от переданного пути.
let rootPromise = null;
function repoRoot() {
  if (rootPromise === null) {
    rootPromise = git(["rev-parse", "--show-toplevel"]).then((result) => {
      if (!result.ok) {
        throw new Error(
          "каталог " + REPO + " не является git-репозиторием"
          + (result.stderr ? " (" + result.stderr.trim().split("\n")[0] + ")" : "")
        );
      }
      return result.stdout.trim() || REPO;
    });
  }
  return rootPromise;
}

/**
 * Проверить путь из аргумента инструмента: вернуть абсолютный путь внутри
 * репозитория. Пусто — «путь не задан» (пустая строка).
 *
 * Закрыты: выход за корень (`../`), каталоги из DENY_DIRS и файлы с секретами
 * (DENY_FILE). Символические ссылки разыменовываются: ссылка вида `link ->
 * /etc/passwd` не должна обойти проверку.
 */
function safePath(raw, root) {
  const value = String(raw === undefined || raw === null ? "" : raw).trim();
  if (!value) return "";
  if (value.startsWith("-")) {
    throw new Error("путь не может начинаться с «-»: " + value);
  }
  const abs = path.resolve(root, value);
  const rel = path.relative(root, abs);
  if (!rel || rel.startsWith("..") || path.isAbsolute(rel)) {
    throw new Error("путь вне репозитория: " + value);
  }
  const parts = rel.split(path.sep).filter(Boolean);
  for (const part of parts.slice(0, -1)) {
    if (DENY_DIRS.has(part)) {
      throw new Error("каталог " + part + "/ закрыт для чтения сервером git");
    }
  }
  const base = parts[parts.length - 1] || "";
  if (DENY_DIRS.has(base)) {
    throw new Error("каталог " + base + "/ закрыт для чтения сервером git");
  }
  if (DENY_FILE.some((re) => re.test(base))) {
    throw new Error("файл " + base + " закрыт для чтения сервером git (секреты)");
  }
  let real = abs;
  try {
    real = fs.realpathSync(abs);
  } catch {
    real = abs;
  }
  const realRel = path.relative(root, real);
  if (realRel.startsWith("..") || path.isAbsolute(realRel)) {
    throw new Error("ссылка ведёт за пределы репозитория: " + value);
  }
  return rel;
}

/** Обрезать длинный текст по границе строки и честно сказать, что показано. */
function clip(body, budget, note) {
  const source = String(body || "");
  if (source.length <= budget) return source;
  let cut = source.slice(0, budget);
  const nl = cut.lastIndexOf("\n");
  if (nl > budget * 0.5) cut = cut.slice(0, nl);
  return cut + "\n" + note.replace("%shown%", String(cut.length))
    .replace("%total%", String(source.length));
}

const TRUNCATED_NOTE =
  "…обрезано: показано %shown% символов из %total%. Сузьте запрос "
  + "(path, limit) или попросите сводку (stat_only).";

/** Разбор строк porcelain-статуса: [{code, path, staged, unstaged, untracked}]. */
function parseStatus(output) {
  const rows = [];
  for (const line of String(output || "").split("\n")) {
    if (!line || line.startsWith("##")) continue;
    const code = line.slice(0, 2);
    let name = line.slice(3);
    // Переименование приезжает как «старое -> новое»: для работы важно новое.
    const arrow = name.indexOf(" -> ");
    if (arrow !== -1) name = name.slice(arrow + 4);
    rows.push({
      code,
      path: name.replace(/^"|"$/g, ""),
      staged: code[0] !== " " && code[0] !== "?",
      unstaged: code[1] !== " " && code[1] !== "?",
      untracked: code === "??",
    });
  }
  return rows;
}

/**
 * ФАКТЫ О ПУТИ: отслеживает ли его git, игнорируется ли он и где лежат правки.
 *
 * Зачем это нужно отдельно. Пустой `git diff` у НЕотслеживаемого файла (нового,
 * `??`) означает не «изменений нет», а «сравнивать не с чем»: у файла нет ни
 * одной версии в истории. Разница принципиальная, и модель её не видит — она
 * читает текст инструмента (живой случай 12.10: на вопрос «какие изменения в
 * README.md» агент ответил «изменений нет», хотя файл новый и весь его текст был
 * написан только что). Поэтому решение принимает СЕРВЕР, а не догадка модели.
 */
async function pathFacts(rel) {
  const facts = {
    tracked: false, untracked: false, ignored: false,
    staged: false, unstaged: false, changes: false,
  };
  if (!rel) return facts;
  const listed = await git(["ls-files", "--error-unmatch", "--", rel]);
  facts.tracked = listed.ok && Boolean(listed.stdout.trim());
  const ignore = await git(["check-ignore", "-q", "--", rel]);
  facts.ignored = ignore.ok;                 // код 0 у check-ignore = игнорируется
  const status = await git(["status", "--porcelain=v1", "-unormal", "--", rel]);
  if (status.ok) {
    for (const line of String(status.stdout).split("\n")) {
      if (!line) continue;
      const code = line.slice(0, 2);
      facts.changes = true;
      if (code === "??") {
        facts.untracked = true;
      } else {
        if (code[0] !== " ") facts.staged = true;
        if (code[1] !== " ") facts.unstaged = true;
      }
    }
  }
  return facts;
}

/** Новые (неотслеживаемые) файлы рабочей копии — для честной подписи пустого diff. */
async function untrackedFiles(limit = 5) {
  const result = await git(["status", "--porcelain=v1", "-unormal"]);
  if (!result.ok) return [];
  return parseStatus(result.stdout).filter((row) => row.untracked)
    .map((row) => row.path).slice(0, Math.max(1, limit));
}

/**
 * Что значит ПУСТОЙ diff у этого пути. Возвращает готовый текст для модели.
 *
 * Тексты разные намеренно: «изменений нет» (файл отслеживается и совпадает с
 * коммитом), «изменения в индексе, а не в рабочей копии», «файл не отслеживается
 * вовсе» и «файл в .gitignore» — четыре разных факта, и путать их нельзя.
 */
function emptyDiffNote(what, rel, facts, scope, untracked) {
  if (rel && facts.untracked) {
    return "Пустой diff у " + rel + " — потому что файл НЕ ОТСЛЕЖИВАЕТСЯ git "
      + "(новый, помечен `??`): у него нет ни одной версии в истории, и сравнивать "
      + "не с чем. Это НЕ «изменений нет»: файл есть в рабочей копии целиком. "
      + "Содержимое смотри git_show_file (path=" + rel + "), а сам факт новизны — "
      + "в git_status и git_files(changed_only=true).";
  }
  if (rel && facts.ignored) {
    return "Пустой diff у " + rel + " — файл в .gitignore: git его не отслеживает, "
      + "и diff по нему не бывает. Содержимое смотри git_show_file (path=" + rel + ").";
  }
  if (rel && facts.staged && scope === "working") {
    return "В рабочей копии " + rel + " не изменён, но изменения ЕСТЬ — они в "
      + "ИНДЕКСЕ (staged). Повтори вызов со scope=staged.";
  }
  if (rel && facts.unstaged && scope === "staged") {
    return "В индексе " + rel + " не изменён, но изменения ЕСТЬ — они в рабочей "
      + "копии (без индекса). Повтори вызов со scope=working или scope=head.";
  }
  let note = "Изменений нет (" + what + (rel ? ", " + rel : "") + "): ";
  if (rel) {
    note += facts.tracked
      ? "файл отслеживается git и совпадает с " + (scope === "branch"
        ? "базовой веткой" : "последним коммитом") + "."
      : "файл не изменён и не новый.";
  } else {
    note += "рабочая копия совпадает с последним коммитом.";
  }
  if (untracked && untracked.length) {
    // Самый частый источник недоразумения: новые файлы в diff НЕ видны.
    note += " В рабочей копии есть НОВЫЕ файлы, которых нет в истории (их diff "
      + "не показывается вовсе): " + untracked.join(", ")
      + ". Их содержимое смотри git_show_file.";
  }
  return note;
}

/** Базовая ветка для сравнения: отслеживаемая, иначе main/master. */
async function baseBranch(explicit) {
  const asked = String(explicit || "").trim();
  if (asked) {
    const check = await git(["rev-parse", "--verify", "--quiet", asked + "^{commit}"]);
    if (!check.ok) throw new Error("не нашёл ветку или коммит «" + asked + "»");
    return asked;
  }
  const upstream = await git(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"]);
  if (upstream.ok && upstream.stdout.trim()) return upstream.stdout.trim();
  for (const candidate of ["main", "master"]) {
    const check = await git(["rev-parse", "--verify", "--quiet", candidate + "^{commit}"]);
    if (check.ok) return candidate;
  }
  return "";
}

// ---------------------------------------------------------------------------
// Инструменты
// ---------------------------------------------------------------------------
server.registerTool(
  "git_status",
  {
    title: "Состояние репозитория",
    description:
      "Ветка проекта, последний коммит, отслеживаемая ветка (впереди/отстала) и " +
      "изменённые файлы рабочей копии. НАЧИНАЙТЕ С ЭТОГО инструмента: он " +
      "отвечает, на какой ветке идёт работа и что уже изменено.",
    inputSchema: {},
  },
  async () =>
    guard(async () => {
      const root = await repoRoot();
      const head = await git(["rev-parse", "--abbrev-ref", "HEAD"]);
      const sha = await git(["rev-parse", "--short", "HEAD"]);
      const last = await git([
        "log", "-1", "--no-color", "--date=short",
        "--pretty=format:%h\t%s\t%an\t%ad",
      ]);
      const statusResult = await git(["status", "--porcelain=v1", "-b", "-unormal"]);
      if (!statusResult.ok) throw new Error(gitError(statusResult, "git status"));
      const header = String(statusResult.stdout).split("\n").find((l) => l.startsWith("##")) || "";
      const rows = parseStatus(statusResult.stdout);
      const staged = rows.filter((row) => row.staged).length;
      const unstaged = rows.filter((row) => row.unstaged).length;
      const untracked = rows.filter((row) => row.untracked).length;

      const out = ["Репозиторий: " + root];
      if (sha.ok && sha.stdout.trim()) {
        const branch = head.ok ? head.stdout.trim() : "";
        out.push("Ветка: " + (branch && branch !== "HEAD" ? branch : "(отсоединённый HEAD)"));
        if (last.ok && last.stdout.trim()) {
          const parts = last.stdout.trim().split("\t");
          out.push("Последний коммит: " + parts[0] + " — " + (parts[1] || "")
            + (parts[3] ? " (" + parts[3] + ")" : "")
            + (parts[2] ? ", автор " + parts[2] : ""));
        }
        const upstream = await git(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"]);
        if (upstream.ok && upstream.stdout.trim()) {
          const counts = header.match(/\[(.+)\]/);
          out.push("Отслеживаемая ветка: " + upstream.stdout.trim()
            + (counts ? " (" + counts[1] + ")" : " (без расхождений)"));
        } else {
          out.push("Отслеживаемая ветка: нет (у ветки нет upstream)");
        }
        const base = await baseBranch("");
        if (base) out.push("Базовая ветка для сравнения: " + base);
      } else {
        out.push("Ветка: " + (head.ok && head.stdout.trim() && head.stdout.trim() !== "HEAD"
          ? head.stdout.trim() : "(неизвестна)"));
        out.push("Коммитов ещё нет (репозиторий только создан).");
      }

      const shortstat = await git(["diff", "HEAD", "--shortstat"]);
      const short = shortstat.ok ? shortstat.stdout.trim() : "";
      if (!rows.length) {
        out.push("Рабочая копия: изменений нет (совпадает с последним коммитом).");
      } else {
        out.push("Рабочая копия: изменено файлов — " + rows.length
          + " (в индексе " + staged + ", без индекса " + unstaged
          + ", новых " + untracked + ")"
          + (short ? "; " + short : ""));
        out.push("Изменённые файлы:");
        for (const row of rows.slice(0, MAX_STATUS_FILES)) {
          out.push("  " + row.code + " " + row.path);
        }
        if (rows.length > MAX_STATUS_FILES) {
          out.push("  …и ещё " + (rows.length - MAX_STATUS_FILES)
            + " (полный список — git_files с changed_only=true)");
        }
      }
      return text(clip(out.join("\n"), RESULT_BUDGET, TRUNCATED_NOTE));
    })
);

server.registerTool(
  "git_files",
  {
    title: "Список файлов проекта",
    description:
      "Файлы репозитория: все (включая новые, но без игнорируемых) или только " +
      "изменённые. Можно ограничить каталогом: path=\"app/ai\". Помогает понять " +
      "структуру проекта и найти нужный модуль.",
    inputSchema: {
      path: z.string().optional().describe("Каталог или файл внутри репозитория, например app/ai"),
      changed_only: z.boolean().default(false)
        .describe("true — только изменённые файлы (как git status), false — весь список"),
      limit: z.number().int().min(1).max(1000).default(MAX_FILES)
        .describe("Сколько файлов показать (по умолчанию " + MAX_FILES + ")"),
    },
  },
  async ({ path: rawPath, changed_only, limit }) =>
    guard(async () => {
      const root = await repoRoot();
      const rel = safePath(rawPath, root);
      const abs = rel ? path.join(root, rel) : "";
      const cap = Math.max(1, Number(limit) || MAX_FILES);
      const branchResult = await git(["rev-parse", "--abbrev-ref", "HEAD"]);
      const branch = branchResult.ok ? branchResult.stdout.trim() : "";

      if (changed_only) {
        const args = ["status", "--porcelain=v1", "-unormal"];
        if (abs) args.push("--", abs);
        const result = await git(args);
        if (!result.ok) throw new Error(gitError(result, "git status"));
        const rows = parseStatus(result.stdout);
        if (!rows.length) {
          return text("Изменённых файлов нет"
            + (rel ? " в " + rel : "") + ": рабочая копия совпадает с последним коммитом.");
        }
        const body = rows.slice(0, cap).map((row) => row.code + " " + row.path).join("\n");
        const tail = rows.length > cap
          ? "\n…и ещё " + (rows.length - cap) + " (сузьте path или поднимите limit)" : "";
        return text(clip(
          "Изменённые файлы" + (rel ? " в " + rel : "") + " ("
          + rows.length + (branch ? ", ветка " + branch : "") + "):\n" + body + tail,
          RESULT_BUDGET, TRUNCATED_NOTE));
      }

      const args = ["ls-files", "--cached", "--others", "--exclude-standard"];
      if (abs) args.push("--", abs);
      const result = await git(args);
      if (!result.ok) throw new Error(gitError(result, "git ls-files"));
      const files = String(result.stdout).split("\n").map((s) => s.trim()).filter(Boolean);
      files.sort();
      if (!files.length) {
        return text("Файлов не найдено" + (rel ? " в " + rel : "") + ".");
      }
      const body = files.slice(0, cap).join("\n");
      const tail = files.length > cap
        ? "\n…показаны первые " + cap + " из " + files.length
          + " (сузьте path или поднимите limit)" : "";
      return text(clip(
        "Файлы проекта" + (rel ? " в " + rel : "")
        + " (отслеживаемые и новые, игнорируемые не показаны; всего "
        + files.length + (branch ? "; ветка " + branch : "") + "):\n" + body + tail,
        RESULT_BUDGET, TRUNCATED_NOTE));
    })
);

server.registerTool(
  "git_diff",
  {
    title: "Изменения (diff)",
    description:
      "Что именно изменилось: незакоммиченные правки (scope=working), индекс " +
      "(staged), всё незакоммиченное (head) или коммиты ветки против базовой " +
      "(branch). stat_only=true — короткая сводка по файлам вместо самого diff.",
    inputSchema: {
      scope: z.enum(["working", "staged", "head", "branch"]).default("working")
        .describe("working — правки без индекса; staged — индекс; head — всё незакоммиченное; branch — против базовой ветки"),
      path: z.string().optional().describe("Ограничить одним файлом или каталогом"),
      stat_only: z.boolean().default(false)
        .describe("true — только сводка «файл: +строки −строки»"),
      base: z.string().optional().describe("Базовая ветка для scope=branch (по умолчанию main или отслеживаемая)"),
      limit: z.number().int().min(200).max(20000).default(RESULT_BUDGET)
        .describe("Предел символов ответа (по умолчанию " + RESULT_BUDGET + ")"),
    },
  },
  async ({ scope, path: rawPath, stat_only, base, limit }) =>
    guard(async () => {
      const root = await repoRoot();
      const rel = safePath(rawPath, root);
      const args = ["diff", "--no-color"];
      let what = "незакоммиченные правки (без индекса)";
      if (scope === "staged") {
        args.push("--cached");
        what = "изменения в индексе";
      } else if (scope === "head") {
        args.push("HEAD");
        what = "все незакоммиченные изменения";
      } else if (scope === "branch") {
        const chosen = await baseBranch(base);
        if (!chosen) {
          throw new Error("не нашёл базовую ветку для сравнения — назовите её "
            + "аргументом base (например base=\"main\")");
        }
        const mergeBase = await git(["merge-base", chosen, "HEAD"]);
        args.push(mergeBase.ok ? chosen + "...HEAD" : chosen);
        what = "коммиты ветки против " + chosen;
      }
      if (stat_only) args.push("--stat");
      else args.push("-U2");
      if (rel) args.push("--", path.join(root, rel));
      const result = await git(args);
      if (!result.ok) throw new Error(gitError(result, "git diff"));
      const body = String(result.stdout).trim();
      const head = "Diff (" + what + (rel ? ", " + rel : "") + ")"
        + (stat_only ? ", сводка" : "") + ":\n";
      if (!body) {
        // ПУСТОЙ DIFF — это ФАКТ, но у него РАЗНЫЕ причины, и молчать о них
        // нельзя: у нового (неотслеживаемого) файла diff пуст не потому, что
        // «изменений нет», а потому что сравнивать не с чем (см. pathFacts).
        const facts = await pathFacts(rel);
        const fresh = rel ? [] : await untrackedFiles();
        return text(emptyDiffNote(what, rel, facts, scope, fresh));
      }
      const cap = Math.max(200, Number(limit) || RESULT_BUDGET);
      return text(clip(head + body, cap, TRUNCATED_NOTE));
    })
);

server.registerTool(
  "git_log",
  {
    title: "История коммитов",
    description:
      "Последние коммиты ветки: короткий хеш, дата, автор, тема. По ним видно, " +
      "чем занимались в проекте в последнее время и какие правки уже вошли.",
    inputSchema: {
      limit: z.number().int().min(1).max(50).default(10)
        .describe("Сколько коммитов показать (по умолчанию 10)"),
      path: z.string().optional().describe("История только по этому файлу или каталогу"),
    },
  },
  async ({ limit, path: rawPath }) =>
    guard(async () => {
      const root = await repoRoot();
      const rel = safePath(rawPath, root);
      const count = Math.min(50, Math.max(1, Number(limit) || 10));
      const args = [
        "log", "--no-color", "--date=short",
        "--pretty=format:%h\t%ad\t%an\t%s", "-n", String(count),
      ];
      if (rel) args.push("--", path.join(root, rel));
      const result = await git(args);
      if (!result.ok) {
        if (/does not have any commits|unknown revision/.test(result.stderr || "")) {
          return text("В репозитории ещё нет коммитов.");
        }
        throw new Error(gitError(result, "git log"));
      }
      const body = String(result.stdout).trim();
      if (!body) {
        // «Коммитов нет» у НОВОГО файла означает не «файл не менялся», а «истории
        // у него не существует вовсе»: файл не отслеживается git. Разница та же,
        // что у пустого diff (см. pathFacts).
        const facts = await pathFacts(rel);
        if (rel && facts.untracked) {
          return text("Коммитов по пути " + rel + " нет: файл НЕ ОТСЛЕЖИВАЕТСЯ git "
            + "(новый) — в истории его нет ни в одном коммите, поэтому и истории "
            + "по нему не существует. Текущее содержимое — git_show_file.");
        }
        if (rel && facts.ignored) {
          return text("Коммитов по пути " + rel + " нет: файл в .gitignore — git его "
            + "не отслеживает, поэтому в истории он не появляется.");
        }
        return text("Коммитов не найдено" + (rel ? " по пути " + rel : "")
          + ": файл отслеживается, но ни один коммит его не менял.");
      }
      const rows = body.split("\n").slice(0, MAX_LOG).map((line) => {
        const parts = line.split("\t");
        return "  " + (parts[0] || "") + "  " + (parts[1] || "") + "  " + (parts[2] || "")
          + "  " + (parts[3] || "");
      });
      const extra = body.split("\n").length > MAX_LOG
        ? "\n…показаны первые " + MAX_LOG + " коммитов" : "";
      return text(clip("Коммиты" + (rel ? " по пути " + rel : "") + " (от новых к старым):\n"
        + rows.join("\n") + extra, RESULT_BUDGET, TRUNCATED_NOTE));
    })
);

server.registerTool(
  "git_show_file",
  {
    title: "Содержимое файла проекта",
    description:
      "Показать файл репозитория (рабочую копию или версию из ветки/коммита) с " +
      "номерами строк — так видно, что делает модуль, а не только его имя. " +
      "Файл показывается ЧАСТЬЮ (по умолчанию 90 строк): если нужного места в " +
      "показанном отрывке нет, попросите следующие строки (from/lines).",
    inputSchema: {
      path: z.string().min(1).describe("Путь файла в репозитории, например app/ai/mcp.py"),
      from: z.number().int().min(1).default(1).describe("Первая строка (нумерация с 1)"),
      lines: z.number().int().min(1).max(SHOW_LINES_MAX).default(SHOW_LINES_DEFAULT)
        .describe("Сколько строк показать (по умолчанию " + SHOW_LINES_DEFAULT + ")"),
      ref: z.string().optional()
        .describe("Версия файла: ветка или коммит (по умолчанию — рабочая копия)"),
    },
  },
  async ({ path: rawPath, from, lines, ref }) =>
    guard(async () => {
      const root = await repoRoot();
      const rel = safePath(rawPath, root);
      if (!rel) throw new Error("не указан путь файла");
      const abs = path.join(root, rel);
      const version = String(ref || "").trim();
      let content = "";
      let where = "рабочая копия";
      if (version) {
        const result = await git(["show", version + ":" + rel]);
        if (!result.ok) {
          throw new Error("не нашёл файл " + rel + " в " + version
            + ": " + (result.stderr || "").trim().split("\n")[0]);
        }
        content = result.stdout;
        where = version;
      } else {
        let stat = null;
        try {
          stat = fs.statSync(abs);
        } catch {
          throw new Error("файла нет в рабочей копии: " + rel);
        }
        if (!stat.isFile()) throw new Error("это не файл: " + rel);
        if (stat.size > 4 * 1024 * 1024) {
          throw new Error("файл слишком большой для показа (" + stat.size
            + " байт) — укажите конкретный модуль");
        }
        const buffer = fs.readFileSync(abs);
        if (buffer.includes(0)) {
          throw new Error("это двоичный файл — показываю только текстовые");
        }
        content = buffer.toString("utf8");
      }
      const all = content.split("\n");
      const start = Math.max(1, Number(from) || 1);
      const want = Math.min(SHOW_LINES_MAX, Math.max(1, Number(lines) || SHOW_LINES_DEFAULT));
      if (start > all.length) {
        return text("В файле " + rel + " только " + all.length + " строк — начало за его концом.");
      }
      const slice = all.slice(start - 1, start - 1 + want);
      const width = String(start + slice.length - 1).length;
      const numbered = slice.map((line, index) =>
        String(start + index).padStart(width, " ") + "| " + line);
      const header = "Файл: " + rel + " (" + where + "), строки " + start + "–"
        + (start + slice.length - 1) + " из " + all.length + "\n";
      const shownTo = start + slice.length - 1;
      const tail = shownTo < all.length
        ? "\n\n…файл продолжается: всего " + all.length + " строк. "
          + "Следующий отрывок — from=" + (shownTo + 1) + "."
        : "";
      // Подпись о продолжении входит в тот же бюджет: иначе ответ перевалил бы
      // клиентский предел (4000 символов) и хвост обрезался бы молча.
      const body = clip(header + numbered.join("\n"), RESULT_BUDGET - tail.length,
                        TRUNCATED_NOTE);
      return text(body + tail);
    })
);

await serve(server);
