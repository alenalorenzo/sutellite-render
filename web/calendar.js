const { t, monthName, periodName, numberFormat } = window.sutelliteI18n;
export const count = numberFormat({ maximumFractionDigits: 0 });
export const decimal = numberFormat({ maximumFractionDigits: 1 });

const compact = numberFormat({ notation: "compact", maximumFractionDigits: 1 });
const months = Array.from({ length: 12 }, (_, index) => index + 1);

function element(tag, text) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  return node;
}

function renderHeatmap(years, selectedYear, selectedMonth, chooseYear, sourceLabel) {
  const heatmap = document.getElementById("heatmap");
  const entries = Object.entries(years).reverse();
  const maximum = Math.max(0, ...entries.flatMap(([, year]) =>
    year.months.map(month => month.detections ?? 0)));
  document.getElementById("heat-maximum").textContent = compact.format(maximum);
  const hasEstimates = entries.some(([, data]) => data.months.some(month => month.estimated));
  const estimateKey = document.getElementById("calendar-estimate-key");
  estimateKey.hidden = !hasEstimates;

  const table = element("table");
  table.className = "heatmap";
  table.setAttribute("aria-label", t("Monthly detections by year") + " · " + sourceLabel);
  const headings = table.createTHead().insertRow();
  for (const title of [t("Year"), ...months.map(month => monthName(month, "short"))]) {
    const th = element("th", title);
    th.scope = "col";
    headings.append(th);
  }
  const body = table.createTBody();
  for (const [year, data] of entries) {
    const row = body.insertRow();
    const selected = Number(year) === selectedYear;
    row.classList.toggle("selected", selected);
    const yearHeader = element("th");
    yearHeader.scope = "row";
    const yearButton = element("button", year);
    yearButton.type = "button";
    yearButton.setAttribute("aria-pressed", String(selected));
    yearButton.addEventListener("click", () => {
      chooseYear(Number(year), null);
      heatmap.querySelector(".selected button")?.focus();
    });
    yearHeader.append(yearButton);
    row.append(yearHeader);
    for (const [i, month] of data.months.entries()) {
      const cell = row.insertCell();
      const value = month.detections;
      cell.title = periodName(year, i + 1) + " · " + detectionLabel(month);
      cell.setAttribute("aria-label", cell.title);
      if (value === null) {
        cell.classList.add("missing");
      } else {
        const ratio = maximum ? value / maximum : 0;
        cell.style.setProperty("--cell-intensity", ratio * 100 + "%");
      }
      const label = element("span", value === null ? "—" : (month.estimated ? "≈" : "") + compact.format(value));
      label.className = "heat-value";
      label.setAttribute("aria-hidden", "true");
      cell.append(label);
      cell.classList.toggle("active-month", selected && selectedMonth === i + 1);
      cell.classList.toggle("estimated", Boolean(month.estimated));
      cell.addEventListener("click", () => chooseYear(Number(year), i + 1));
    }
  }
  heatmap.replaceChildren(table);
}

function detectionLabel({ detections, estimated }) {
  if (detections === null) return t("No observation");
  return t(detections === 1 ? "{count} detection" : "{count} detections", {
    count: (estimated ? "≈" : "") + count.format(detections),
  }) + (estimated ? " (" + t("MODIS estimate") + ")" : "");
}

function renderComparison(data, month) {
  const container = document.getElementById("baseline-table");
  const hasReference = data.months.some(entry => entry.referenceYears >= 3);
  const notice = document.getElementById("history-notice");
  notice.hidden = hasReference;
  notice.textContent = t("Historical comparisons require data for the same month in at least three earlier years.");
  document.getElementById("reference-label").textContent =
    data.reference ? t("Reference period") + " " + data.reference.replace("-", "–") : t("No earlier years in the archive.");

  const table = element("table");
  table.className = "baseline-table";
  table.setAttribute("aria-label", t("Monthly detections and historical comparison"));
  const headers = table.createTHead().insertRow();
  const columns = ["Month", "Detections", "Years with data"];
  if (hasReference) columns.push("Median", "Mean", "Min", "Max", "Difference vs. median");
  for (const title of columns) {
    const th = element("th", t(title));
    th.scope = "col";
    headers.append(th);
  }

  const body = table.createTBody();
  for (const [i, entry] of data.months.entries()) {
    const row = body.insertRow();
    row.classList.toggle("selected", month === i + 1);
    row.insertCell().textContent = monthName(i + 1);
    const value = row.insertCell();
    value.textContent = entry.detections === null ? t("No observation")
      : (entry.estimated ? "≈" : "") + count.format(entry.detections);
    if (entry.estimated) value.title = detectionLabel(entry);
    row.insertCell().textContent = String(entry.referenceYears);
    if (!hasReference) continue;
    for (const value of [entry.median, entry.mean, entry.minimum, entry.maximum]) {
      const cell = row.insertCell();
      cell.textContent = value === null ? "—" : decimal.format(value);
      if (value === null) cell.title = t("Fewer than three earlier years with observations for this month");
    }
    const difference = row.insertCell();
    if (entry.difference === null) {
      difference.textContent = "—";
    } else {
      difference.textContent = (entry.difference > 0 ? "+" : "") + decimal.format(entry.difference) + "%";
      if (entry.difference !== 0) difference.className = entry.difference > 0 ? "positive" : "negative";
    }
  }
  container.replaceChildren(table);
}

export function renderCalendar(years, year, month, chooseYear, sourceLabel) {
  const data = years[year];
  const observedMonths = data.months.filter(entry => entry.detections !== null).length;
  const estimatedMonths = data.months.filter(entry => entry.estimated).length;
  document.getElementById("trend-title").textContent = t("Monthly trend") + " · " + year;
  document.getElementById("year-total").textContent = data.total === null ? "—"
    : (estimatedMonths ? "≈" : "") + count.format(data.total);
  const label = estimatedMonths === 0 ? "Detections"
    : estimatedMonths === observedMonths ? "Estimated detections" : "Includes MODIS estimates";
  document.getElementById("year-total-label").textContent = t(label)
    + (observedMonths === 12 ? "" : " · " + t("{months}/12 months", { months: observedMonths }));
  document.getElementById("peak-month").textContent = data.peakMonth ? monthName(data.peakMonth) : "—";
  document.getElementById("median-key").hidden = data.months.every(entry => entry.median === null);
  renderHeatmap(years, year, month, chooseYear, sourceLabel);
  renderComparison(data, month);
  drawTrend(data, month);
}

export function drawTrend(data, highlightedMonth) {
  const canvas = document.getElementById("trend");
  const { width, height } = canvas.getBoundingClientRect();
  if (!width || !height || !data) return;
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(width * pixelRatio);
  canvas.height = Math.round(height * pixelRatio);
  const ctx = canvas.getContext("2d");
  ctx.scale(pixelRatio, pixelRatio);
  const theme = getComputedStyle(canvas);
  const color = name => theme.getPropertyValue(name).trim();
  const readingMode = document.documentElement.hasAttribute("data-dyslexia");

  const values = data.months.map(month => month.detections);
  const medians = data.months.map(month => month.median);
  const maximum = Math.max(1, ...values.filter(value => value !== null),
    ...medians.filter(value => value !== null));
  const magnitude = 10 ** Math.floor(Math.log10(maximum / 4));
  const interval = [1, 2, 5, 10].find(value => value * magnitude >= maximum / 4) * magnitude;
  const upper = Math.ceil(maximum / interval) * interval;
  const left = readingMode ? 43 : 38;
  const right = width - 13;
  const top = 10;
  const bottom = height - 25;
  const x = i => left + i * (right - left) / 11;
  const y = value => bottom - value * (bottom - top) / upper;

  if (highlightedMonth) {
    ctx.fillStyle = color("--sky-tint");
    ctx.fillRect(x(highlightedMonth - 1) - 8, top, 16, bottom - top);
  }
  ctx.font = (readingMode ? "13px " : "12px ") + color("--sans");
  if (readingMode) ctx.letterSpacing = ".35px";
  ctx.fillStyle = color("--muted");
  ctx.strokeStyle = color("--trend-grid");
  ctx.textBaseline = "middle";
  ctx.textAlign = "right";
  for (let value = 0; value <= upper + interval / 100; value += interval) {
    ctx.beginPath();
    ctx.moveTo(left, y(value));
    ctx.lineTo(right, y(value));
    ctx.stroke();
    ctx.fillText(compact.format(value), left - 6, y(value));
  }
  ctx.textAlign = "center";
  for (const [index, month] of months.entries()) {
    if (width < (readingMode ? 420 : 360) && index % 2 !== 0) continue;
    ctx.fillText(monthName(month, "short"), x(index), bottom + 16);
  }

  function drawSeries(series, color, dash, markers) {
    ctx.beginPath();
    ctx.lineWidth = 2;
    ctx.strokeStyle = color;
    ctx.setLineDash(dash);
    let connected = false;
    for (const [i, value] of series.entries()) {
      if (value === null) {
        connected = false;
        continue;
      }
      if (connected) ctx.lineTo(x(i), y(value));
      else ctx.moveTo(x(i), y(value));
      connected = true;
    }
    ctx.stroke();
    ctx.setLineDash([]);
    if (markers) {
      ctx.fillStyle = color;
      for (const [i, value] of series.entries()) {
        if (value === null) continue;
        ctx.beginPath();
        ctx.arc(x(i), y(value), 3.2, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
  }

  drawSeries(medians, color("--trend-median") || color("--unburned"), [5, 4], false);
  drawSeries(values, color("--trend-detections") || color("--burned"), [], true);
}

export function downloadCalendar(selected, year, data) {
  const headers = [
    "month", "detections", "reference_period", "years_with_data", "median",
    "mean", "min", "max", "difference_vs_median_pct", "value_type",
  ];
  const rows = data.months.map((month, i) => [
    i + 1, month.detections, data.reference, month.referenceYears,
    month.median, month.mean, month.minimum, month.maximum, month.difference,
    month.detections === null ? "no_observation" : month.estimated ? "modis_estimate" : "observed",
  ]);
  const csv = [headers, ...rows].map(row => row.join(",")).join("\r\n") + "\r\n";
  const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
  const link = element("a");
  link.href = url;
  const source = [selected.archive ?? "upload", selected.platform,
    selected.stream === "UNSPECIFIED" ? null : selected.stream].filter(Boolean);
  link.download = "sutellite_calendar_" + source.join("_") + "_" + year + ".csv";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
