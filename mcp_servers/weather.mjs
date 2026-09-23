#!/usr/bin/env node
// MCP-сервер «Погода».
//
// Источники (оба бесплатные, без ключей и регистрации):
//   * geocoding-api.open-meteo.com — название города -> координаты;
//   * 7timer.info (NOAA GFS) — текущая погода и прогноз по всему миру.
// Инструменты: get_weather (сейчас), get_forecast (по дням).

import { z } from "zod";
import { createServer, failure, fetchJson, guard, lines, serve, text } from "./lib/common.mjs";

const GEO = "https://geocoding-api.open-meteo.com/v1/search";
const TIMER = "https://www.7timer.info/bin/api.pl";

const server = createServer("weather", "1.0.0");

// Описания кодов погоды 7timer.
const WEATHER = {
  clear: "ясно",
  pcloudy: "переменная облачность",
  mcloudy: "значительная облачность",
  cloudy: "облачно",
  humid: "влажно, дымка",
  lightrain: "небольшой дождь",
  oshower: "ливневые осадки",
  rain: "дождь",
  prain: "временами дождь",
  rainsnow: "дождь со снегом",
  lightsnow: "небольшой снег",
  snow: "снег",
  icepellets: "ледяная крупа",
  ra1sn: "дождь, переходящий в снег",
  sn1ra: "снег, переходящий в дождь",
  fog: "туман",
  ts: "гроза",
  tsrain: "гроза с дождём",
};

// Шкала ветра 7timer (1–8) -> метры в секунду.
const WIND = {
  1: "0–0,3 м/с, штиль",
  2: "0,3–3,4 м/с, лёгкий",
  3: "3,4–8 м/с, слабый",
  4: "8–11 м/с, умеренный",
  5: "11–17 м/с, свежий",
  6: "17–21 м/с, сильный",
  7: "21–25 м/с, крепкий",
  8: "более 25 м/с, штормовой",
};

function weatherText(code) {
  const key = String(code || "").toLowerCase().replace(/(day|night)$/, "");
  const base = WEATHER[key] || key || "нет данных";
  if (/day$/.test(code)) return base + " (день)";
  if (/night$/.test(code)) return base + " (ночь)";
  return base;
}

function windText(value) {
  return WIND[value] || (value ? String(value) : "");
}

function formatDate(raw) {
  const text = String(raw);
  if (!/^\d{8}$/.test(text)) return text;
  return text.slice(6, 8) + "." + text.slice(4, 6) + "." + text.slice(0, 4);
}

/** Название города -> координаты (Open-Meteo Geocoding, понимает русский). */
async function geocode(city) {
  const url = GEO + "?name=" + encodeURIComponent(city) + "&count=1&language=ru&format=json";
  const data = await fetchJson(url);
  const hit = Array.isArray(data.results) ? data.results[0] : null;
  if (!hit) throw new Error("город «" + city + "» не найден");
  const title = [hit.name, hit.admin1, hit.country].filter(Boolean).join(", ");
  return { lat: hit.latitude, lon: hit.longitude, title, timezone: hit.timezone || "" };
}

function timerUrl(product, place) {
  return (
    TIMER +
    "?lon=" + encodeURIComponent(place.lon) +
    "&lat=" + encodeURIComponent(place.lat) +
    "&product=" + product +
    "&output=json"
  );
}

server.registerTool(
  "get_weather",
  {
    title: "Погода сейчас",
    description:
      "Текущая погода в городе: температура, облачность, влажность, ветер, осадки, а также " +
      "максимум и минимум на сегодня. Город указывается названием — «Москва», «Казань», «Sochi».",
    inputSchema: {
      city: z.string().min(1).describe("Название города, например «Москва»"),
    },
  },
  async ({ city }) =>
    guard(async () => {
      const place = await geocode(city);
      // «civil» — трёхчасовые срезы на 3 суток: первый срез и есть текущая погода.
      const civil = await fetchJson(timerUrl("civil", place));
      const series = Array.isArray(civil.dataseries) ? civil.dataseries : [];
      if (!series.length) return failure("источник не отдал данные о погоде для «" + city + "»");
      const now = series[0];
      const hours = Number(now.timepoint || 0);
      const when = hours ? "через " + hours + " ч" : "сейчас";

      let today = null;
      try {
        const light = await fetchJson(timerUrl("civillight", place));
        today = (light.dataseries || [])[0] || null;
      } catch {
        today = null;
      }

      return text(
        "Погода в " + place.title + "\n" +
          lines([
            ["Ближайший срез", when],
            ["Погода", weatherText(now.weather)],
            ["Температура, °C", now.temp2m],
            ["Влажность", now.rh2m],
            ["Ветер", windText(now.wind10m && now.wind10m.speed) + (now.wind10m && now.wind10m.direction ? ", " + now.wind10m.direction : "")],
            ["Облачность, баллов", now.cloudcover],
            ["Осадки", now.prec_type && now.prec_type !== "none" ? now.prec_type : "нет"],
            ["Сегодня днём, °C", today ? today.temp2m.max : ""],
            ["Сегодня ночью, °C", today ? today.temp2m.min : ""],
            ["Координаты", place.lat + ", " + place.lon],
          ])
      );
    })
);

server.registerTool(
  "get_forecast",
  {
    title: "Прогноз погоды",
    description: "Прогноз погоды в городе по дням (до 7 суток): максимум и минимум температуры, осадки, ветер.",
    inputSchema: {
      city: z.string().min(1).describe("Название города, например «Москва»"),
      days: z.number().int().min(1).max(7).default(3).describe("Сколько дней прогноза вернуть (1–7)"),
    },
  },
  async ({ city, days }) =>
    guard(async () => {
      const place = await geocode(city);
      const light = await fetchJson(timerUrl("civillight", place));
      const list = (light.dataseries || []).slice(0, days);
      if (!list.length) return failure("источник не отдал прогноз для «" + city + "»");
      const blocks = list.map((day) =>
        lines([
          ["Дата", formatDate(day.date)],
          ["Погода", weatherText(day.weather)],
          ["Днём, °C", day.temp2m && day.temp2m.max],
          ["Ночью, °C", day.temp2m && day.temp2m.min],
          ["Ветер", windText(day.wind10m_max)],
        ])
      );
      return text("Прогноз для " + place.title + "\n\n" + blocks.join("\n\n"));
    })
);

await serve(server);
