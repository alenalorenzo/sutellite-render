import { count, decimal, downloadCalendar, drawTrend, renderCalendar } from "./calendar.js";
import { makeGlobe } from "./globe.js";
import { makeAlertMap, MODIS } from "./map.js";
import { createCalendarWorkspace, getApiOrigin } from "./workspace.js";

const i18n = window.sutelliteI18n;
const { t, monthName, periodName, text } = i18n;

const REGIONS = {
  angola: { label: "Angola", country: "Angola", center: [-12.3, 17.7] },
  australia: { label: "Australia", country: "Australia", center: [-25.3, 134] },
  brasil: { label: "Brazil", country: "Brazil", center: [-11.1, -51.5] },
  california: { label: "California", country: "United States of America", center: [37, -119.5] },
  rdc: { label: "DR Congo", country: "Democratic Republic of the Congo", center: [-3.5, 23.7] },
  indonesia: { label: "Indonesia", country: "Indonesia", center: [-2.4, 115] },
  espana: { label: "Spain", country: "Spain", center: [39.8, -4.1] },
  zambia: { label: "Zambia", country: "Zambia", center: [-13.1, 27.9] },
};
const LAND_COVER = {
  10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland",
  50: "Built-up", 60: "Bare / sparse vegetation", 70: "Snow and ice",
  80: "Permanent water", 90: "Herbaceous wetland", 95: "Mangroves",
  100: "Moss and lichen",
};

const byId = id => document.getElementById(id);
const tabs = ["calendar", "alerts"];
const calendarMonths = [null, ...Array.from({ length: 12 }, (_, index) => index + 1)];
const labelMonth = month => month === null ? t("All months") : monthName(month);
const monthTick = month => month === null ? t("All") : monthName(month, "short");
const coordinate = i18n.numberFormat({ minimumFractionDigits: 2, maximumFractionDigits: 2, useGrouping: false });
const exampleYears = [2024, 2025];
const state = {
  page: "home",
  tab: "calendar",
  region: "angola",
  alertYear: 2025,
  alertMonth: 8,
  historyYear: 2025,
  historyMonth: 8,
  historicalView: true,
  archivePeriod: { year: 2025, month: null },
  uploadPeriod: { year: null, month: null },
};
const alertCache = new Map();
let calendar;
let countries;
let atlasMap;
let showLocator;
let visiblePoints = [];
let alertRequest = 0;
let history;
let historyUnavailable = false;
const historyCache = new Map();
const workspace = createCalendarWorkspace(uploaded => {
  if (uploaded) {
    state.uploadPeriod = { year: null, month: null };
    if (state.page !== "archive") navigate("upload/calendar");
  } else if (state.page === "upload" && !workspace.hasUpload()) {
    navigate("home");
  } else if (state.page !== "home" && state.tab === "calendar") {
    updateCalendar();
  }
});

function makeSlider(id, label, change, labelTick = label) {
  const input = byId(id);
  const output = byId(id + "-value");
  const ticks = byId(id + "-ticks");
  const ruler = input.closest(".range-ruler");
  let values = [];
  let lastLanguage;

  function showTicks() {
    const lastIndex = values.length - 1;
    const step = Math.max(1, Math.ceil(lastIndex / 5));
    ticks.replaceChildren(...values.map((value, index) => {
      const mark = document.createElement("span");
      mark.className = "ruler-tick";
      const first = index === 0;
      const last = index === lastIndex;
      if (first || last || (index % step === 0 && lastIndex - index >= step * .65)) {
        mark.textContent = labelTick(value);
        mark.classList.add("labelled");
      }
      if (first || last) mark.id = id + (first ? "-start" : "-end");
      mark.style.left = (lastIndex > 0 ? 100 * index / lastIndex : 0) + "%";
      return mark;
    }));
  }

  function showValue() {
    const value = values[Number(input.value)];
    const text = label(value);
    output.value = text;
    input.setAttribute("aria-valuetext", text);
    const position = values.length > 1 ? Number(input.value) / (values.length - 1) : 0;
    ruler.style.setProperty("--position", position);
    return value;
  }

  input.addEventListener("input", () => change(showValue()));
  return (options, selected) => {
    const changed = options.length !== values.length || options.some((value, index) => value !== values[index]);
    values = options;
    if (changed || lastLanguage !== i18n.language) {
      lastLanguage = i18n.language;
      showTicks();
    }
    input.max = Math.max(0, values.length - 1);
    input.value = Math.max(0, values.indexOf(selected));
    input.disabled = values.length < 2;
    showValue();
  };
}

let pendingCalendarFrame;
function scheduleCalendar() {
  cancelAnimationFrame(pendingCalendarFrame);
  pendingCalendarFrame = requestAnimationFrame(() => {
    if (state.page !== "home" && state.tab === "calendar") updateCalendar();
  });
}

let pendingAlertFrame;
function scheduleAlerts() {
  cancelAnimationFrame(pendingAlertFrame);
  pendingAlertFrame = requestAnimationFrame(() => {
    if (state.page === "archive" && state.tab === "alerts") updateAlerts();
  });
}

const setCalendarYear = makeSlider("calendar-year", String, value => {
  selectedCalendarPeriod(calendarSelection()).year = value;
  scheduleCalendar();
});
const setCalendarMonth = makeSlider("calendar-month", labelMonth, value => {
  selectedCalendarPeriod(calendarSelection()).month = value;
  scheduleCalendar();
}, monthTick);
const useHistoricalAlerts = () => Boolean(history?.regions[state.region] && state.historicalView);
const setAlertYear = makeSlider("alert-year", String, value => {
  if (useHistoricalAlerts()) state.historyYear = value;
  else state.alertYear = value;
  scheduleAlerts();
});
const setAlertMonth = makeSlider("alert-month", labelMonth, value => {
  if (useHistoricalAlerts()) state.historyMonth = value;
  else state.alertMonth = value;
  scheduleAlerts();
}, monthTick);

async function readJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    const error = new Error(path + " returned " + response.status);
    error.status = response.status;
    throw error;
  }
  return response.json();
}

async function getRegionAlerts(region) {
  if (!alertCache.has(region)) {
    const periods = await readJson("./data/alerts/" + region + ".json");
    const points = Object.values(periods).flatMap(year => Object.values(year).flat());
    const bounds = L.latLngBounds(points.map(point => [point.lat, point.lon]));
    alertCache.set(region, { periods, bounds });
  }
  return alertCache.get(region);
}

function formatCoordinates(point) {
  const latitude = coordinate.format(Math.abs(point.lat)) + "°" + (point.lat < 0 ? "S" : "N");
  const longitude = coordinate.format(Math.abs(point.lon)) + "°" + (point.lon < 0 ? "W" : "E");
  return latitude + ", " + longitude;
}

function explainPoint(point, month, year, archivePeriod) {
  const period = periodName(year, month);
  const observed = count.format(point.detections);
  if (archivePeriod) {
    const current = archivePeriod.source === "modis"
      ? t("MODIS picked up {observed} hotspots in this area. On the VIIRS scale, the estimate is {value} ({low}–{high}).", {
        observed, value: decimal.format(point.value), low: decimal.format(point.low), high: decimal.format(point.high),
      })
      : t("VIIRS picked up {observed} hotspots in this area.", { observed });
    const threshold = decimal.format(point.historicalUpper);
    const comparison = t(archivePeriod.source === "modis"
      ? "Even the low estimate is above the previous upper limit of {threshold}."
      : "That count is above the previous upper limit of {threshold}.", { threshold });
    return t("In {period}, {current} {comparison} We compared the same month and area across {years} earlier years with records.", {
      period, current, comparison, years: count.format(point.referenceYears),
    });
  }
  return t("In {period}, VIIRS picked up {observed} hotspots in this area. The previous high for {month} was {maximum} in 2013–2023. Sutellite’s model also flagged the count as unusual.", {
    period, observed, month: monthName(month), maximum: count.format(point.historicalMax),
  });
}

function currentArchivePeriod() {
  return state.historicalView && history
    ? history.regions[state.region]?.periods[state.historyYear]?.[state.historyMonth]
    : undefined;
}

function selectPoint(index, pan = false, showPopup = false) {
  const point = visiblePoints[index];
  const details = byId("point-details");
  const summary = byId("point-explanation");
  byId("point-picker").value = point ? String(index) : "";
  byId("point-instruction").textContent = point
    ? formatCoordinates(point) + " · " + t("0.1° grid cell center")
    : t("Select a point to inspect its 0.1° grid cell.");
  details.hidden = !point;
  summary.hidden = !point;
  atlasMap.highlight(point ? index : -1, pan);
  if (!point) {
    summary.textContent = "";
    details.replaceChildren();
    atlasMap.explain(-1, "", false);
    return;
  }

  const period = currentArchivePeriod();
  const month = period ? state.historyMonth : state.alertMonth;
  const year = period ? state.historyYear : state.alertYear;
  const explanation = explainPoint(point, month, year, period);
  summary.textContent = explanation;
  atlasMap.explain(index, explanation, showPopup);

  const fields = [];
  if (period) {
    fields.push(
      [period.source === "modis" ? "MODIS detections" : "VIIRS S-NPP detections", count.format(point.detections)],
    );
    if (period.source === "modis") {
      fields.push(
        ["Estimated on the VIIRS scale", decimal.format(point.value)],
        ["Calibration range", decimal.format(point.low) + "–" + decimal.format(point.high)],
      );
    }
    fields.push(
      [t("Previous high") + " · " + point.historicalYear, decimal.format(point.historicalMax)],
      [t("Previous upper limit") + " · " + point.historicalUpperYear, decimal.format(point.historicalUpper)],
      ["Earlier years with records", count.format(point.referenceYears)],
      ["Reference span", point.firstReferenceYear + "–" + point.lastReferenceYear],
    );
    if (point.frp !== null) {
      fields.push([point.validFrp === point.detections ? "Total FRP" : "Available FRP sum", decimal.format(point.frp) + " MW"]);
    }
  } else {
    fields.push(["VIIRS detections", count.format(point.detections)],
      ["Historical maximum", decimal.format(point.historicalMax)]);
    fields.push(
      ["Total FRP", decimal.format(point.frp) + " MW"],
      ["WorldCover 2021", t(LAND_COVER[point.landCover] ?? "No land cover data")],
      ["MODIS", t(MODIS[point.modis]), point.modis],
    );
    if (point.modis === "burned") {
      fields.push(["Mapped burned area", decimal.format(point.burnedHectares) + " ha"]);
    }
  }
  details.replaceChildren(...fields.map(([name, value, style]) => {
    const row = document.createElement("div");
    const term = document.createElement("dt");
    const description = document.createElement("dd");
    term.textContent = t(name);
    description.textContent = value;
    if (style) description.className = style;
    row.append(term, description);
    return row;
  }));
}

function setPointOptions(selection) {
  const picker = byId("point-picker");
  picker.replaceChildren(
    new Option(t(visiblePoints.length ? "Choose an alert point" : "No alert points"), ""),
    ...visiblePoints.map((point, index) =>
      new Option(String(index + 1).padStart(2, "0") + " · " + formatCoordinates(point), index)),
  );
  picker.disabled = !visiblePoints.length;
  selectPoint(selection?.index ?? (visiblePoints.length ? 0 : -1), false, selection?.popup);
}

function setMapStatus(message) {
  const status = byId("map-status");
  text(status, message);
  status.hidden = !message;
}

function syncAlertFilters(years, year, month) {
  setAlertYear(years, year);
  setAlertMonth(calendarMonths.slice(1), month);
}

function showArchiveMode(historical) {
  byId("historical-mode").disabled = !history?.regions[state.region];
  byId("historical-mode").setAttribute("aria-pressed", String(historical));
  byId("ai-mode").setAttribute("aria-pressed", String(!historical));
  byId("example-legend").hidden = historical;
  byId("history-legend").hidden = !historical;
  byId("alert-title").textContent = t(historical ? "Historical anomaly map" : "AI alert map");
  byId("alert-count-label").textContent = t(historical ? "Flagged cells" : "AI alerts");
  byId("overlap-label").textContent = t(historical ? "Compared cells" : "MODIS overlap");
  byId("alert-source").textContent = historical
    ? t("Canonical archive · Same calendar month, earlier years")
    : [t("2024–2025 AI results · Explore MODIS burned area and ESA WorldCover 2021."),
      historyUnavailable ? t("The 2000–2025 archive could not be loaded.")
        : state.historicalView ? t("The 2000–2025 archive for this region has not been imported yet.") : ""].filter(Boolean).join(" ");
}

async function showArchiveAlerts(request, selection) {
  const { region, historyYear, historyMonth } = state;
  const info = REGIONS[region];
  const regionData = history.regions[region];
  const period = currentArchivePeriod();
  if (period) {
    byId("alert-source").textContent = t(period.source === "modis"
      ? "MODIS records converted to the VIIRS S-NPP scale · Estimated values"
      : "VIIRS S-NPP records · Same calendar month, earlier years");
  }
  const years = Array.from({ length: history.endYear - history.startYear + 1 }, (_, i) => history.startYear + i);
  syncAlertFilters(years, historyYear, historyMonth);
  byId("map-selection").textContent = t(info.label) + " · " + periodName(historyYear, historyMonth);
  byId("alert-count").textContent = period?.compared ? count.format(period.alerts) : "—";
  byId("overlap-count").textContent = period ? count.format(period.compared) : "—";
  byId("overlap-detail").textContent = period
    ? t("{compared} of {cells} cells with detections had at least three comparable earlier years.", {
      compared: count.format(period.compared), cells: count.format(period.cells),
    })
    : t("No detections for the selected source in this region and month.");
  visiblePoints = [];
  if (!period?.alerts) {
    atlasMap.show(info, regionData.bounds, []);
    setPointOptions();
    if (!period) setMapStatus("No archive records for this region and month.");
    else if (period.status === "uncalibrated") setMapStatus("Calibration is unavailable for this region and month.");
    else if (!period.compared) setMapStatus("Insufficient earlier years for a historical comparison.");
    else setMapStatus("No compared cell exceeded the previous range.");
    return;
  }

  const key = region + "/" + historyYear + "/" + historyMonth;
  try {
    if (!historyCache.has(key)) {
      atlasMap.clear();
      setPointOptions();
      setMapStatus("Loading anomaly points…");
      const pending = readJson("./data/historical-alerts/" + key + ".json");
      historyCache.set(key, pending);
      if (historyCache.size > 18) historyCache.delete(historyCache.keys().next().value);
      pending.catch(() => {
        if (historyCache.get(key) === pending) historyCache.delete(key);
      });
    }
    const points = await historyCache.get(key);
    if (request !== alertRequest) return;
    if (!Array.isArray(points) || points.length !== period.alerts) {
      historyCache.delete(key);
      throw new Error("The anomaly file does not match its index.");
    }
    visiblePoints = points;
    atlasMap.show(info, regionData.bounds, visiblePoints, Boolean(selection));
    setPointOptions(selection);
    setMapStatus("");
  } catch (error) {
    if (request !== alertRequest) return;
    setMapStatus("Could not load these anomalies. Change the year or reload the page.");
    console.error(error);
  }
}

async function updateAlerts(selection) {
  const request = ++alertRequest;
  if (!atlasMap) atlasMap = makeAlertMap(countries, selectPoint);
  const historical = useHistoricalAlerts();
  showArchiveMode(historical);
  if (historical) {
    await showArchiveAlerts(request, selection);
    return;
  }

  const { region, alertYear, alertMonth } = state;
  const info = REGIONS[region];
  syncAlertFilters(exampleYears, alertYear, alertMonth);
  byId("map-selection").textContent = t(info.label) + " · " + periodName(alertYear, alertMonth);

  const cached = alertCache.get(region);
  if (!cached) {
    atlasMap.clear();
    visiblePoints = [];
    setPointOptions();
    byId("alert-count").textContent = "—";
    byId("overlap-count").textContent = "—";
    byId("overlap-detail").textContent = t("Share of alerts with valid MODIS data that overlap burned area.");
    setMapStatus("Loading alert points…");
  }

  try {
    const { periods, bounds } = cached ?? await getRegionAlerts(region);
    if (request !== alertRequest) return;
    visiblePoints = periods[alertYear]?.[alertMonth] ?? [];
    atlasMap.show(info, bounds, visiblePoints, Boolean(selection));
    setPointOptions(selection);

    const valid = visiblePoints.filter(point => point.modis !== "no-data").length;
    const burned = visiblePoints.filter(point => point.modis === "burned").length;
    byId("alert-count").textContent = count.format(visiblePoints.length);
    byId("overlap-count").textContent = valid ? decimal.format(100 * burned / valid) + "%" : "—";
    byId("overlap-detail").textContent = valid
      ? t("{burned} of {valid} alerts with valid MODIS data overlap mapped burned area.", {
        burned: count.format(burned), valid: count.format(valid),
      })
      : t(visiblePoints.length ? "MODIS has no valid data for these alert cells." : "No alerts for the selected month.");
    setMapStatus(visiblePoints.length ? "" : "No alerts for this region and month.");
  } catch (error) {
    if (request !== alertRequest) return;
    setMapStatus("Could not load these alerts. Change region or refresh to try again.");
    console.error(error);
  }
}

function calendarSelection() {
  return workspace.selection(state.region, t(REGIONS[state.region].label));
}

function selectedCalendarPeriod(selected) {
  return selected.archive ? state.archivePeriod : state.uploadPeriod;
}

function renderStatistics(selected, current) {
  renderCalendar(selected.years, current.year, current.month, (year, month) => {
    current.year = year;
    current.month = month;
    updateCalendar();
  }, selected.label);
  const flag = selected.monthlyFlags?.[current.year + "-" + current.month];
  const note = byId("workspace-month-flag");
  note.hidden = !flag;
  if (flag) note.textContent = t("High monthly activity compared with {years} earlier years. Historical median: {median} detections.", {
    years: count.format(flag.referenceYears), median: decimal.format(flag.median),
  });
}

function updateCalendar() {
  workspace.setActive(true, state.page === "upload");
  const selected = calendarSelection();
  const { years, name, label, error, filename } = selected;
  const loading = byId("calendar-data-status");
  byId("calendar-selection").textContent = [name, label, filename].filter(Boolean).join(" · ");
  byId("calendar-content").hidden = !years;
  loading.hidden = Boolean(years);
  loading.classList.toggle("workspace-error", Boolean(error));
  if (!years) {
    loading.textContent = t(error ? "The calendar could not be loaded. Reload the page to try again."
      : "Loading the historical calendar…");
    return;
  }
  const current = selectedCalendarPeriod(selected);
  const yearOptions = Object.keys(years).map(Number).sort((a, b) => a - b);
  if (!years[current.year]) current.year = yearOptions.at(-1);
  setCalendarYear(yearOptions, current.year);
  setCalendarMonth(calendarMonths, current.month);
  const months = years[current.year].months;
  const period = current.month ? [months[current.month - 1]] : months;
  const coverage = byId("calendar-coverage");
  const noObservations = period.every(entry => entry.detections === null);
  let notice = noObservations ? "No observations for this source and period." : "";
  if (selected.harmonized && !noObservations) {
    notice = "MODIS estimates on the VIIRS S-NPP scale through January 2012. Observed S-NPP counts from February 2012.";
  }
  coverage.hidden = !notice;
  coverage.textContent = t(notice);
  if (!byId("calendar-statistics").hidden) renderStatistics(selected, current);
  byId("calendar-land-source").textContent = t(selected.archive
    ? "Land type uses the dominant ESA WorldCover 2021 class in each 0.5° cell."
    : "Land type uses available ESA WorldCover 2021 classes.");
  workspace.showMap(selected, current.year, current.month);
}

function setStatistics(open, reveal = false) {
  const panel = byId("calendar-statistics");
  panel.hidden = !open;
  byId("toggle-statistics").setAttribute("aria-expanded", String(open));
  if (open) {
    const selected = calendarSelection();
    if (selected.years) renderStatistics(selected, selectedCalendarPeriod(selected));
    if (reveal && matchMedia("(max-width: 1010px)").matches) panel.scrollIntoView({ block: "start" });
  }
}

function navigate(route) {
  const hash = "#" + route;
  if (location.hash !== hash) window.history.pushState(null, "", hash);
  applyRoute(true);
}

function selectRegion(region, tab = "calendar") {
  navigate("region/" + region + "/" + tab);
}

function currentRoute() {
  const segments = location.hash.slice(1).split("/");
  if (segments[0] === "region" && Object.hasOwn(REGIONS, segments[1]) && calendar?.[segments[1]]) {
    const tab = segments[2] === "alerts" ? "alerts" : "calendar";
    return { page: "archive", region: segments[1], tab };
  }
  if (segments[0] === "upload" && workspace.hasUpload()) return { page: "upload", tab: "calendar" };
  return { page: "home", tab: "calendar" };
}

let appliedRoute;
function applyRoute(moveFocus = false) {
  if (!calendar) return;
  const route = currentRoute();
  const key = [route.page, route.region, route.tab, location.hash].join("/");
  if (key === appliedRoute) {
    if (route.page === "upload") updateCalendar();
    return;
  }
  appliedRoute = key;
  cancelAnimationFrame(pendingCalendarFrame);
  cancelAnimationFrame(pendingAlertFrame);
  alertRequest++;
  const differentRegion = route.region && route.region !== state.region;
  const differentPage = route.page !== state.page;
  if (differentRegion || differentPage) setStatistics(false);
  if (route.region) state.region = route.region;
  state.page = route.page;
  state.tab = route.tab;

  const home = route.page === "home";
  const uploaded = route.page === "upload";
  showLocator?.(route.region);
  byId("home-view").hidden = !home;
  byId("atlas").hidden = home;
  const heading = byId(home ? "page-title" : (state.tab === "calendar" ? "calendar-title" : "alert-title"));
  document.querySelector(".skip-link").href = "#" + heading.id;
  byId("region").value = state.region;
  renderCalendarTitle();
  byId("alerts-tab").hidden = uploaded;
  for (const tab of tabs) {
    const active = state.tab === tab;
    const button = byId(tab + "-tab");
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
    byId(tab + "-view").hidden = !active;
  }

  if (home) workspace.setActive(false);
  else if (state.tab === "calendar") updateCalendar();
  else {
    workspace.setActive(false);
    updateAlerts();
  }
  if (moveFocus && (!home || differentPage) && location.hash !== "#analyze-data") {
    window.scrollTo(0, 0);
    heading.focus({ preventScroll: true });
  }
}

function renderCalendarTitle() {
  byId("calendar-title").textContent = t(state.page === "upload" ? "Historical calendar · Uploaded file" : "Historical calendar");
}

function renderThemeLabel() {
  byId("theme-label").textContent = t(document.documentElement.dataset.theme === "light" ? "Night mode" : "Day mode");
}

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const day = theme === "light";
  byId("theme-toggle").setAttribute("aria-pressed", String(day));
  renderThemeLabel();
  byId("theme-icon").textContent = day ? "☾" : "☀";
  document.querySelector('meta[name="theme-color"]').content =
    getComputedStyle(document.documentElement).getPropertyValue("--header-bg").trim();
  workspace.setTheme(theme);
  drawSelectedTrend();
}

function drawSelectedTrend() {
  if (!calendar || state.page === "home" || state.tab !== "calendar" || byId("calendar-statistics").hidden) return;
  const selected = calendarSelection();
  const { year, month } = selectedCalendarPeriod(selected);
  drawTrend(selected.years?.[year], month);
}

function bindSuggestions() {
  const form = byId("suggestion-form");
  const field = byId("suggestion-message");
  const status = byId("suggestion-status");
  const button = form.querySelector('button[type="submit"]');
  const draftKey = "sutellite-suggestion-draft";
  field.addEventListener("invalid", event => {
    event.preventDefault();
    text(status, "Enter a suggestion before sending it.");
    field.focus();
  });
  try {
    field.value = localStorage.getItem(draftKey) ?? "";
  } catch {}

  form.addEventListener("submit", async event => {
    event.preventDefault();
    const message = field.value.trim();
    if (!message) {
      text(status, "Enter a suggestion before sending it.");
      field.focus();
      return;
    }
    button.disabled = true;
    field.readOnly = true;
    text(status, "Sending…");
    let staticPage = false;
    try {
      const response = await fetch(getApiOrigin() + "/api/suggestions", {
        method: "POST",
        body: new URLSearchParams({ message }),
        signal: AbortSignal.timeout(12000),
      });
      staticPage = [404, 405, 501].includes(response.status);
      if (response.status === 201 && (await response.json()).saved === true) {
        form.reset();
        try { localStorage.removeItem(draftKey); } catch {}
        text(status, "Suggestion sent.");
        return;
      }
    } catch {
      // Keep the message available when a request fails or times out.
    } finally {
      button.disabled = false;
      field.readOnly = false;
    }
    let draftSaved = false;
    try {
      localStorage.setItem(draftKey, field.value);
      draftSaved = true;
    } catch {}
    text(status, (draftSaved ? "Draft saved in this browser. " : "Your suggestion is still in the box. ")
      + (staticPage ? "Use the Sutellite server to send it." : "Try sending it again."));
  });
}

function bindControls() {
  bindSuggestions();
  for (const [index, tab] of tabs.entries()) {
    const button = byId(tab + "-tab");
    button.addEventListener("click", () => {
      if (state.page === "archive") selectRegion(state.region, tab);
      button.focus({ preventScroll: true });
    });
    button.addEventListener("keydown", event => {
      if (state.page !== "archive") return;
      const shortcuts = {
        ArrowRight: tabs[1 - index], ArrowLeft: tabs[1 - index], Home: tabs[0], End: tabs.at(-1),
      };
      if (!shortcuts[event.key]) return;
      event.preventDefault();
      const next = shortcuts[event.key];
      selectRegion(state.region, next);
      byId(next + "-tab").focus({ preventScroll: true });
    });
  }
  for (const link of document.querySelectorAll('a[href="#home"]')) {
    link.addEventListener("click", event => {
      event.preventDefault();
      navigate("home");
    });
  }
  document.querySelector(".skip-link").addEventListener("click", event => {
    event.preventDefault();
    document.querySelector(event.currentTarget.hash)?.focus();
  });
  window.addEventListener("popstate", () => applyRoute(true));
  byId("region").addEventListener("change", event => selectRegion(event.target.value, state.tab));
  byId("workspace-resume").addEventListener("click", () => navigate("upload/calendar"));
  byId("toggle-statistics").addEventListener("click", () => setStatistics(byId("calendar-statistics").hidden, true));
  byId("point-picker").addEventListener("change", event => {
    selectPoint(event.target.value ? Number(event.target.value) : -1, true, true);
  });
  byId("download").addEventListener("click", () => {
    const selected = calendarSelection();
    const { year } = selectedCalendarPeriod(selected);
    if (selected.years?.[year]) downloadCalendar(selected, year, selected.years[year]);
  });
  byId("open-methodology").addEventListener("click", () => byId("methodology").showModal());
  for (const [mode, historical] of [["historical-mode", true], ["ai-mode", false]]) {
    byId(mode).addEventListener("click", () => {
      state.historicalView = historical;
      updateAlerts();
    });
  }
  byId("theme-toggle").addEventListener("click", () => {
    const theme = document.documentElement.dataset.theme === "light" ? "dark" : "light";
    try {
      localStorage.setItem("sutellite-theme", theme);
    } catch {}
    setTheme(theme);
  });
  setTheme(document.documentElement.dataset.theme ?? "dark");
  document.addEventListener("sutellite-language-change", () => {
    renderThemeLabel();
    workspace.updateLanguage();
    for (const option of byId("region").options) option.text = t(REGIONS[option.value].label);
    if (!calendar || state.page === "home") return;
    renderCalendarTitle();
    if (state.tab === "calendar") updateCalendar();
    else {
      const picker = byId("point-picker");
      const selection = picker.disabled ? undefined : {
        index: picker.value === "" ? -1 : Number(picker.value), popup: atlasMap?.hasPopup(),
      };
      updateAlerts(selection);
    }
  });
  document.addEventListener("sutellite-accessibility-change", ({ detail }) => {
    workspace.setAccessibility(detail);
    atlasMap?.setColorVision(detail.colorVision);
    drawSelectedTrend();
  });
  new ResizeObserver(drawSelectedTrend).observe(byId("trend"));
  document.fonts.ready.then(drawSelectedTrend);
}

async function start() {
  bindControls();
  try {
    [calendar, countries, history] = await Promise.all([
      readJson("./data/calendar.json"),
      readJson("./assets/countries.json"),
      readHistory(),
    ]);
    const entries = Object.entries(REGIONS).filter(([id]) => id in calendar)
      .sort(([, a], [, b]) => a.label.localeCompare(b.label));
    const regionPicker = byId("region");
    regionPicker.replaceChildren(...entries.map(([id, info]) => new Option(t(info.label), id)));
    regionPicker.disabled = false;
    showLocator = makeGlobe(countries, Object.fromEntries(entries), selectRegion);
    byId("load-notice").hidden = true;
    byId("globe-card").hidden = false;
    applyRoute();
  } catch (error) {
    text(byId("load-notice"), "The atlas data could not be loaded. Open this folder with a local web server, or refresh this page.");
    console.error(error);
  }
}

async function readHistory() {
  try {
    const data = await readJson("./data/historical-alerts/index.json");
    if (data.schema !== 1 || data.canonical !== "canonical_v1.1"
        || data.startYear !== 2000 || data.endYear !== 2025
        || !data.regions || !Object.keys(data.regions).length) {
      throw new Error("Incomplete historical anomaly data.");
    }
    return data;
  } catch (error) {
    if (error.status !== 404) {
      historyUnavailable = true;
      console.error(error);
    }
    return null;
  }
}

start();
