// Общие помощники локальных MCP-серверов.
//
// ВАЖНО: stdout занят протоколом MCP (JSON-RPC построчно), поэтому всё
// диагностическое пишется только в stderr через log()/console.error.

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";

export const DEFAULT_TIMEOUT_MS = 15000;

export function log(...args) {
  console.error("[mcp]", ...args);
}

/** Создать сервер MCP с общими настройками. */
export function createServer(name, version = "1.0.0") {
  return new McpServer({ name, version });
}

/** Запустить сервер по stdio (единственный транспорт этих серверов). */
export async function serve(server) {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  log("started", server.serverInfo?.name ?? "server");
}

/** Текстовый результат инструмента. */
export function text(value) {
  return { content: [{ type: "text", text: String(value) }] };
}

/**
 * Результат-ошибка. isError=true — модель видит сбой инструмента, но
 * сервер остаётся живым (в отличие от брошенного исключения).
 */
export function failure(message) {
  return { content: [{ type: "text", text: String(message) }], isError: true };
}

/** Выполнить инструмент, превратив любое исключение в понятный ответ. */
export async function guard(work) {
  try {
    return await work();
  } catch (err) {
    const message = err && err.message ? err.message : String(err);
    log("tool failed:", message);
    return failure("Инструмент не смог получить данные: " + message);
  }
}

/** GET с таймаутом; возвращает текст ответа. */
export async function fetchText(url, { timeoutMs = DEFAULT_TIMEOUT_MS, headers = {} } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const resp = await fetch(url, {
      signal: controller.signal,
      headers: { "user-agent": "fastapi-site-mcp/1.0", accept: "*/*", ...headers },
    });
    if (!resp.ok) {
      throw new Error("источник ответил " + resp.status + " " + resp.statusText);
    }
    return await resp.text();
  } catch (err) {
    if (err && err.name === "AbortError") {
      throw new Error("источник не ответил за " + timeoutMs + " мс");
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

/** GET JSON с таймаутом. */
export async function fetchJson(url, options = {}) {
  const body = await fetchText(url, options);
  try {
    return JSON.parse(body);
  } catch {
    throw new Error("источник вернул не JSON");
  }
}

/** Округлить до n знаков, убрав хвостовые нули: 12.340 -> "12.34". */
export function round(value, digits = 4) {
  const num = Number(value);
  if (!Number.isFinite(num)) return String(value);
  return String(Number(num.toFixed(digits)));
}

/** Число из строки «59,8800» (запятая как разделитель). */
export function parseNumber(raw) {
  if (raw === null || raw === undefined) return NaN;
  return Number(String(raw).replace(/\s+/g, "").replace(",", "."));
}

/** Простейший разбор XML по тегам (внешних зависимостей не тянем). */
export function xmlAll(xml, tag) {
  const out = [];
  const re = new RegExp("<" + tag + "(?:\\s[^>]*)?>([\\s\\S]*?)</" + tag + ">", "g");
  let m;
  while ((m = re.exec(xml)) !== null) out.push(m[1]);
  return out;
}

export function xmlValue(block, tag) {
  const m = new RegExp("<" + tag + "(?:\\s[^>]*)?>([\\s\\S]*?)</" + tag + ">").exec(block);
  return m ? m[1] : "";
}

/** Декодировать windows-1251 (так отдаёт ЦБ РФ) в строку. */
export function decodeWindows1251(buffer) {
  return new TextDecoder("windows-1251").decode(buffer);
}

/** Скачать текст в кодировке windows-1251 (ЦБ РФ). */
export async function fetchWindows1251(url, options = {}) {
  const controller = new AbortController();
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const resp = await fetch(url, { signal: controller.signal, headers: { "user-agent": "fastapi-site-mcp/1.0" } });
    if (!resp.ok) throw new Error("источник ответил " + resp.status);
    const buf = await resp.arrayBuffer();
    return decodeWindows1251(buf);
  } catch (err) {
    if (err && err.name === "AbortError") throw new Error("источник не ответил за " + timeoutMs + " мс");
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

/** Отобразить «ключ: значение» списком строк, пропуская пустое. */
export function lines(pairs) {
  return pairs
    .filter(([, value]) => value !== undefined && value !== null && value !== "")
    .map(([key, value]) => key + ": " + value)
    .join("\n");
}
