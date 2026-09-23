#!/usr/bin/env node
// MCP-сервер «Курсы валют».
//
// Источник: официальные курсы Банка России (cbr.ru) — бесплатно, без ключа.
// Инструменты: get_rate, convert, list_rates.

import { z } from "zod";
import {
  createServer,
  failure,
  fetchWindows1251,
  guard,
  lines,
  parseNumber,
  round,
  serve,
  text,
  xmlAll,
  xmlValue,
} from "./lib/common.mjs";

const CBR_URL = "https://www.cbr.ru/scripts/XML_daily.asp";

const server = createServer("currency", "1.0.0");

// Кэш разобранной таблицы курсов: ключ — запрошенная дата ("" = текущая).
const cache = new Map();

/** «23.09.2026» / «2026-09-23» -> «23/09/2026» для cbr.ru. */
function cbrDate(date) {
  if (!date) return "";
  const iso = /^(\d{4})-(\d{2})-(\d{2})$/.exec(date.trim());
  if (iso) return iso[3] + "/" + iso[2] + "/" + iso[1];
  const dotted = /^(\d{2})[.\/](\d{2})[.\/](\d{4})$/.exec(date.trim());
  if (dotted) return dotted[1] + "/" + dotted[2] + "/" + dotted[3];
  throw new Error("дата должна быть в виде 2026-09-23 или 23.09.2026");
}

/** Разобрать XML Банка России в {date, rates: {USD: {...}}} + RUB. */
function parseCbr(xml) {
  const dateAttr = /Date="([^"]+)"/.exec(xml);
  const rates = {
    RUB: { code: "RUB", name: "Российский рубль", nominal: 1, unit: 1 },
  };
  for (const block of xmlAll(xml, "Valute")) {
    const code = xmlValue(block, "CharCode").trim();
    if (!code) continue;
    const nominal = parseNumber(xmlValue(block, "Nominal")) || 1;
    const unit = parseNumber(xmlValue(block, "VunitRate"));
    rates[code] = {
      code,
      name: xmlValue(block, "Name").trim(),
      nominal,
      unit: Number.isFinite(unit) && unit > 0 ? unit : parseNumber(xmlValue(block, "Value")) / nominal,
      value: parseNumber(xmlValue(block, "Value")),
    };
  }
  return { date: dateAttr ? dateAttr[1] : "", rates };
}

async function loadRates(date) {
  const key = date ? cbrDate(date) : "";
  if (cache.has(key)) return cache.get(key);
  const url = key ? CBR_URL + "?date_req=" + encodeURIComponent(key) : CBR_URL;
  const xml = await fetchWindows1251(url);
  if (!/<ValCurs/i.test(xml)) throw new Error("Банк России не отдал таблицу курсов на эту дату");
  const parsed = parseCbr(xml);
  cache.set(key, parsed);
  return parsed;
}

function currencyFrom(rates, code) {
  const key = String(code || "").trim().toUpperCase();
  const found = rates[key];
  if (!found) {
    throw new Error(
      "валюта «" + code + "» не найдена в таблице Банка России; доступны: " + Object.keys(rates).join(", ")
    );
  }
  return found;
}

server.registerTool(
  "get_rate",
  {
    title: "Курс валюты",
    description:
      "Официальный курс валюты Банка России к рублю (сколько рублей стоит одна единица валюты). " +
      "Поддерживаются коды USD, EUR, CNY, GBP, JPY, KZT, TRY, BYN и другие из таблицы ЦБ РФ.",
    inputSchema: {
      currency: z.string().min(1).describe("Код валюты по ISO, например USD или EUR"),
      date: z.string().optional().describe("Дата курса: 2026-09-23 или 23.09.2026 (по умолчанию — действующий)"),
    },
  },
  async ({ currency, date }) =>
    guard(async () => {
      const { date: effective, rates } = await loadRates(date);
      const item = currencyFrom(rates, currency);
      return text(
        lines([
          ["Дата курса", effective],
          ["Валюта", item.code + " — " + item.name],
          ["Номинал ЦБ", item.nominal],
          ["Курс, ₽ за 1 " + item.code, round(item.unit)],
        ])
      );
    })
);

server.registerTool(
  "convert",
  {
    title: "Перевести валюту",
    description:
      "Перевести сумму из одной валюты в другую по официальному курсу Банка России. " +
      "Рубль обозначается кодом RUB.",
    inputSchema: {
      amount: z.number().describe("Сумма перевода"),
      from: z.string().min(1).describe("Из какой валюты, например USD"),
      to: z.string().min(1).describe("В какую валюту, например RUB"),
      date: z.string().optional().describe("Дата курса: 2026-09-23 или 23.09.2026"),
    },
  },
  async ({ amount, from, to, date }) =>
    guard(async () => {
      const { date: effective, rates } = await loadRates(date);
      const source = currencyFrom(rates, from);
      const target = currencyFrom(rates, to);
      const value = (Number(amount) * source.unit) / target.unit;
      if (!Number.isFinite(value)) return failure("не удалось посчитать перевод");
      return text(
        lines([
          ["Дата курса", effective],
          [round(amount, 2) + " " + source.code + " =", round(value, 4) + " " + target.code],
          ["Курс " + source.code, round(source.unit) + " ₽"],
          ["Курс " + target.code, round(target.unit) + " ₽"],
        ])
      );
    })
);

server.registerTool(
  "list_rates",
  {
    title: "Список курсов",
    description: "Показать действующие курсы основных валют Банка России к рублю одним списком.",
    inputSchema: {
      limit: z.number().int().min(1).max(60).default(15).describe("Сколько валют показать"),
    },
  },
  async ({ limit }) =>
    guard(async () => {
      const { date: effective, rates } = await loadRates("");
      const rows = Object.values(rates)
        .filter((item) => item.code !== "RUB")
        .sort((a, b) => a.code.localeCompare(b.code))
        .slice(0, limit)
        .map((item) => item.code + " — " + round(item.unit) + " ₽ (" + item.name + ")");
      return text("Курсы Банка России на " + effective + ":\n" + rows.join("\n"));
    })
);

await serve(server);
