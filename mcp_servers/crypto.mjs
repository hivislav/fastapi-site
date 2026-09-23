#!/usr/bin/env node
// MCP-сервер «Криптовалюты».
//
// Источник: публичный API CoinGecko — бесплатно, без ключа.
// Инструменты: get_price, get_market.

import { z } from "zod";
import { createServer, fetchJson, guard, lines, round, serve, text } from "./lib/common.mjs";

const API = "https://api.coingecko.com/api/v3";

const server = createServer("crypto", "1.0.0");

// Частые названия (в том числе русские), чтобы модель могла спросить «биткоин».
const ALIASES = {
  btc: "bitcoin",
  биткоин: "bitcoin",
  биткойн: "bitcoin",
  bitcoin: "bitcoin",
  eth: "ethereum",
  эфир: "ethereum",
  эфириум: "ethereum",
  ethereum: "ethereum",
  usdt: "tether",
  tether: "tether",
  тезер: "tether",
  bnb: "binancecoin",
  sol: "solana",
  солана: "solana",
  solana: "solana",
  xrp: "ripple",
  ripple: "ripple",
  ada: "cardano",
  кардано: "cardano",
  doge: "dogecoin",
  догикоин: "dogecoin",
  ton: "the-open-network",
  тоник: "the-open-network",
  trx: "tron",
  ltc: "litecoin",
  dot: "polkadot",
  matic: "matic-network",
};

const COIN = z.string().min(1).describe("Монета: код (BTC, ETH) или идентификатор CoinGecko (bitcoin)");
const FIAT = z.string().min(1).default("usd").describe("Валюта цены: usd, eur, rub (по умолчанию usd)");

function coinId(raw) {
  const key = String(raw || "").trim().toLowerCase();
  return ALIASES[key] || key;
}

function coinList(raw) {
  const list = Array.isArray(raw) ? raw : [raw];
  const ids = list.map(coinId).filter(Boolean);
  if (!ids.length) throw new Error("не указана ни одна монета");
  return Array.from(new Set(ids));
}

function fiatList(raw) {
  const list = Array.isArray(raw) ? raw : raw ? [raw] : ["usd"];
  return Array.from(new Set(list.map((item) => String(item).trim().toLowerCase()).filter(Boolean)));
}

function money(value, currency) {
  if (value === null || value === undefined) return "нет данных";
  const digits = Number(value) >= 1000 ? 2 : Number(value) >= 1 ? 2 : 6;
  return round(value, digits) + " " + String(currency).toUpperCase();
}

server.registerTool(
  "get_price",
  {
    title: "Курс криптовалюты",
    description:
      "Текущая цена криптовалюты в обычных валютах, с изменением за сутки. " +
      "Можно перечислить несколько монет сразу: BTC, ETH, TON.",
    inputSchema: {
      coins: z.array(COIN).min(1).describe("Список монет, например [\"BTC\", \"ETH\"]"),
      currencies: z.array(z.string().min(1)).default(["usd"]).describe("Валюты цены, например [\"usd\", \"rub\"]"),
    },
  },
  async ({ coins, currencies }) =>
    guard(async () => {
      const ids = coinList(coins);
      const vs = fiatList(currencies);
      const url =
        API +
        "/simple/price?ids=" +
        encodeURIComponent(ids.join(",")) +
        "&vs_currencies=" +
        encodeURIComponent(vs.join(",")) +
        "&include_24hr_change=true&include_last_updated_at=true";
      const data = await fetchJson(url);
      const blocks = ids.map((id) => {
        const row = data[id];
        if (!row) return id + ": монета не найдена в CoinGecko";
        const parts = vs.map((cur) => [String(cur).toUpperCase(), money(row[cur], cur)]);
        const change = row[vs[0] + "_24h_change"];
        if (change !== undefined && change !== null) parts.push(["Изменение за 24 ч, %", round(change, 2)]);
        if (row.last_updated_at) {
          parts.push(["Обновлено", new Date(row.last_updated_at * 1000).toISOString()]);
        }
        return lines([["Монета", id]].concat(parts));
      });
      return text("Цены криптовалют (CoinGecko)\n\n" + blocks.join("\n\n"));
    })
);

server.registerTool(
  "get_market",
  {
    title: "Обзор рынка монеты",
    description: "Капитализация, объём торгов, максимум и минимум за сутки и исторический максимум монеты.",
    inputSchema: {
      coin: COIN,
      currency: FIAT,
    },
  },
  async ({ coin, currency }) =>
    guard(async () => {
      const id = coinId(coin);
      const cur = fiatList(currency)[0];
      const url =
        API +
        "/coins/markets?vs_currency=" +
        encodeURIComponent(cur) +
        "&ids=" +
        encodeURIComponent(id) +
        "&price_change_percentage=24h,7d";
      const data = await fetchJson(url);
      const row = Array.isArray(data) ? data[0] : null;
      if (!row) return text("Монета «" + coin + "» не найдена в CoinGecko.");
      return text(
        lines([
          ["Монета", row.name + " (" + String(row.symbol || "").toUpperCase() + ")"],
          ["Цена", money(row.current_price, cur)],
          ["Капитализация", money(row.market_cap, cur)],
          ["Объём за 24 ч", money(row.total_volume, cur)],
          ["Максимум 24 ч", money(row.high_24h, cur)],
          ["Минимум 24 ч", money(row.low_24h, cur)],
          ["Изменение 24 ч, %", round(row.price_change_percentage_24h, 2)],
          ["Изменение 7 дней, %", round(row.price_change_percentage_7d_in_currency, 2)],
          ["Исторический максимум", money(row.ath, cur)],
          ["В обороте", row.circulating_supply ? round(row.circulating_supply, 0) : ""],
        ])
      );
    })
);

await serve(server);
