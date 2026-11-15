const i18n = window.sutelliteI18n;
const { t, text, periodName } = i18n;

const MAX_UPLOAD_BYTES = 25 * 1024 * 1024;
const ARCHIVE_SATELLITES = { HARMONIZED: ["HARMONIZED"], MODIS: ["TERRA", "AQUA"], VIIRS: ["SNPP", "NOAA-20", "NOAA-21"] };
const SATELLITE_LABELS = { HARMONIZED: "—", SNPP: "S-NPP", TERRA: "Terra", AQUA: "Aqua" };
const byId = id => document.getElementById(id);

function satelliteLabel(value) {
  return SATELLITE_LABELS[value] ?? (value === "UNSPECIFIED" ? t("Unspecified") : value);
}

function sensorLabel(value) {
  return value === "HARMONIZED" ? t("Harmonized") : value === "UNSPECIFIED" ? t("Unspecified") : value;
}

const streamLabel = value => t(value === "UNSPECIFIED" ? "Unspecified" : value);

function sourceLabel({ sensor, platform, stream }) {
  return [sensor === "UNSPECIFIED" ? "" : sensorLabel(sensor),
    platform === "UNSPECIFIED" ? "" : satelliteLabel(platform),
    stream === "UNSPECIFIED" ? "" : streamLabel(stream)]
    .filter(Boolean).join(" · ") || t("Source unspecified");
}

function showMessage(element, message, error = false, parameters) {
  element.hidden = !message;
  text(element, message, parameters);
  element.classList.toggle("workspace-error", error);
}

const uploadErrorPatterns = [
  [/^Invalid data: empty or null values found in '(?<column>.*)'\.$/u, "Invalid data: empty or null values found in '{column}'."],
  [/^Invalid year value detected: '(?<value>.*)'\.$/u, "Invalid year value detected: '{value}'."],
  [/^Invalid month value detected \(must be 1-12\): '(?<value>.*)'\.$/u, "Invalid month value detected (must be 1-12): '{value}'."],
  [/^Invalid date value detected in '(?<column>.*)': '(?<value>.*)'\.$/u, "Invalid date value detected in '{column}': '{value}'."],
  [/^Invalid count in '(?<column>.*)': empty or null values are not allowed\.$/u, "Invalid count in '{column}': empty or null values are not allowed."],
  [/^Invalid count in '(?<column>.*)': non-numeric text '(?<value>.*)' detected\.$/u, "Invalid count in '{column}': non-numeric text '{value}' detected."],
  [/^Invalid count in '(?<column>.*)': negative value '(?<value>.*)' detected\.$/u, "Invalid count in '{column}': negative value '{value}' detected."],
  [/^Invalid count in '(?<column>.*)': fractional value '(?<value>.*)' detected\.$/u, "Invalid count in '{column}': fractional value '{value}' detected."],
  [/^Invalid count in '(?<column>.*)': use integers below 2⁶³\.$/u, "Invalid count in '{column}': use integers below 2⁶³."],
  [/^Invalid (?<column>latitude|longitude|frp) value: (?<value>.*)\.$/u, "Invalid {column} value: {value}."],
];

function showUploadError(status, error) {
  const message = error instanceof TypeError ? "Could not connect to the upload service." : error.message;
  for (const [pattern, key] of uploadErrorPatterns) {
    const match = pattern.exec(message);
    if (match) {
      showMessage(status, key, true, match.groups);
      return;
    }
  }
  showMessage(status, message, true);
}

function setOptions(picker, values, label, preferred) {
  picker.replaceChildren(...values.map(value => new Option(label(value), value)));
  if (values.includes(preferred)) picker.value = preferred;
  picker.disabled = values.length < 2;
}

export function getApiOrigin() {
  const meta = document.querySelector('meta[name="sutellite-api-origin"]');
  return (meta?.content ?? "").trim().replace(/\/$/, "");
}

export function createCalendarWorkspace(refresh) {
  const form = byId("workspace-upload");
  const input = byId("workspace-file");
  const fileName = byId("workspace-file-name");
  const sourcePicker = byId("workspace-source");
  const sensorPicker = byId("calendar-sensor");
  const satellitePicker = byId("calendar-satellite");
  const regionPicker = byId("workspace-region");
  const submit = byId("workspace-submit");
  const reset = byId("workspace-reset");
  const resume = byId("workspace-resume");
  const status = byId("workspace-upload-status");
  const mapStatus = byId("workspace-map-status");
  const map = byId("workspace-map");
  const archiveCalendars = new Map();
  const archiveMaps = new Map();
  const archiveSatellite = { MODIS: "TERRA", VIIRS: "SNPP" };
  let archiveSensor = "HARMONIZED";
  const origin = getApiOrigin();
  let uploadedData;
  let dataset;
  let uploadFilters = {};
  let theme = "dark";
  let accessibility;
  let upload;
  let mapRequest = 0;
  let pendingMap;
  let fitMap;
  let mapResize;

  function source() {
    return dataset.series[Number(sourcePicker.value)];
  }

  function refreshRegions() {
    const previousRegion = regionPicker.selectedOptions[0]?.textContent;
    regionPicker.replaceChildren(...source().regions.map(({ name }, index) => new Option(name, index)));
    const matching = Array.from(regionPicker.options).find(option => option.textContent === previousRegion);
    if (matching) regionPicker.value = matching.value;
    regionPicker.disabled = regionPicker.options.length < 2;
  }

  function refreshStreams() {
    if (!dataset) return;
    const previous = sourcePicker.value;
    const options = dataset.series.flatMap((item, index) =>
      item.sensor === sensorPicker.value && item.platform === satellitePicker.value
        ? [new Option(streamLabel(item.stream), index)] : []);
    sourcePicker.replaceChildren(...options);
    if (options.some(option => option.value === previous)) sourcePicker.value = previous;
    byId("workspace-stream").hidden = options.length < 2;
    refreshRegions();
  }

  function refreshSatellites(preferred) {
    const sensor = sensorPicker.value;
    const satellites = dataset ? [...new Set(dataset.series.filter(item => item.sensor === sensor)
      .map(item => item.platform))] : ARCHIVE_SATELLITES[sensor];
    setOptions(satellitePicker, satellites, satelliteLabel, preferred);
    refreshStreams();
  }

  function setupSourceFilters() {
    const sensors = dataset ? [...new Set(dataset.series.map(item => item.sensor))]
      : Object.keys(ARCHIVE_SATELLITES);
    setOptions(sensorPicker, sensors, sensorLabel, dataset ? uploadFilters.sensor : archiveSensor);
    refreshSatellites(dataset ? uploadFilters.platform : archiveSatellite[archiveSensor]);
  }

  setupSourceFilters();
  sensorPicker.addEventListener("change", () => {
    if (!dataset) archiveSensor = sensorPicker.value;
    refreshSatellites(dataset ? undefined : archiveSatellite[archiveSensor]);
    refresh();
  });
  satellitePicker.addEventListener("change", () => {
    if (!dataset) archiveSatellite[archiveSensor] = satellitePicker.value;
    refreshStreams();
    refresh();
  });
  function renderFileName() {
    const name = input.files[0]?.name;
    text(fileName, name ? "{filename}" : "No file selected", { filename: name });
    fileName.title = name ?? "";
  }
  input.addEventListener("change", renderFileName);
  input.addEventListener("invalid", event => {
    event.preventDefault();
    showMessage(status, "Choose a CSV or Parquet file.", true);
    input.focus();
  });

  form.addEventListener("submit", async event => {
    event.preventDefault();
    if (!input.files.length) return;
    if (input.files[0].size > MAX_UPLOAD_BYTES) {
      showMessage(status, "Choose a file smaller than 25 MiB.", true);
      return;
    }
    upload?.abort();
    const request = new AbortController();
    upload = request;
    submit.disabled = true;
    reset.hidden = false;
    showMessage(status, "Processing file…");
    try {
      const response = await fetch(origin + "/api/analyze", {
        method: "POST", body: new FormData(form), signal: request.signal,
      });
      if ([404, 405, 501].includes(response.status)) {
        throw new Error("The upload service is unavailable. Open the site through the Sutellite server.");
      }
      const data = await response.json().catch(() => null);
      if (!response.ok || !data?.series?.length) {
        const detail = data?.detail;
        throw new Error(typeof detail === "string" ? detail : "The file could not be processed.");
      }
      if (request !== upload) return;
      uploadedData = data;
      uploadFilters = {};
      sourcePicker.replaceChildren();
      regionPicker.replaceChildren();
      resume.hidden = false;
      showMessage(status, "Loaded {filename}.", false, { filename: data.filename });
      refresh(true);
    } catch (error) {
      if (request !== upload) return;
      showUploadError(status, error);
      reset.hidden = !uploadedData;
    } finally {
      if (request === upload) {
        upload = undefined;
        submit.disabled = false;
      }
    }
  });
  reset.addEventListener("click", () => {
    upload?.abort();
    upload = undefined;
    uploadedData = undefined;
    uploadFilters = {};
    form.reset();
    renderFileName();
    submit.disabled = false;
    reset.hidden = true;
    resume.hidden = true;
    setActive(false);
    showMessage(status, "");
    refresh();
  });
  sourcePicker.addEventListener("change", () => {
    refreshRegions();
    refresh();
  });
  regionPicker.addEventListener("change", () => refresh());

  async function readArchive(name) {
    const response = await fetch("./data/spatial/" + name + ".json");
    if (!response.ok) throw new Error("Archive data is unavailable.");
    return response.json();
  }

  function selection(region, archiveLabel) {
    if (dataset) {
      const item = source();
      return {
        ...item.regions[Number(regionPicker.value)],
        label: sourceLabel(item),
        platform: item.platform,
        stream: item.stream,
        filename: dataset.filename,
      };
    }
    const platform = satellitePicker.value;
    const harmonized = platform === "HARMONIZED";
    const label = harmonized ? t("Harmonized · VIIRS S-NPP scale")
      : sourceLabel({ sensor: sensorPicker.value, platform, stream: "UNSPECIFIED" });
    if (!archiveCalendars.has(region)) {
      const entry = {};
      archiveCalendars.set(region, entry);
      readArchive(region + "-calendar").then(data => {
        entry.data = data;
      }).catch(() => {
        entry.error = true;
      }).finally(refresh);
    }
    const entry = archiveCalendars.get(region);
    return {
      years: entry.data?.[platform],
      error: entry.error || Boolean(entry.data && !entry.data[platform]),
      name: archiveLabel,
      label,
      archive: region,
      platform,
      harmonized,
    };
  }

  function setActive(active, useUpload = false) {
    const nextDataset = active && useUpload ? uploadedData : undefined;
    if (dataset !== nextDataset) {
      if (dataset && dataset === uploadedData) {
        uploadFilters = { sensor: sensorPicker.value, platform: satellitePicker.value };
      }
      dataset = nextDataset;
      setupSourceFilters();
    }
    byId("archive-region").hidden = Boolean(dataset);
    byId("workspace-filters").hidden = !dataset;
    if (!active) {
      ++mapRequest;
      pendingMap = undefined;
    }
  }

  function updateFrame() {
    if (pendingMap && map.contentWindow?.showSutelliteCells) {
      map.contentWindow.showSutelliteCells(...pendingMap);
      fitMap?.();
    }
  }

  function setTheme(value) {
    theme = value;
    map.contentWindow?.setSutelliteTheme?.(theme);
  }

  function setAccessibility(preferences) {
    accessibility = preferences;
    map.contentWindow?.setSutelliteAccessibility?.(accessibility);
  }

  function updateLanguage() {
    for (const option of sensorPicker.options) option.text = sensorLabel(option.value);
    for (const option of satellitePicker.options) option.text = satelliteLabel(option.value);
    for (const option of sourcePicker.options) {
      const item = dataset?.series[Number(option.value)];
      if (item) option.text = streamLabel(item.stream);
    }
    map.contentWindow?.sutelliteI18n?.setLanguage(i18n.language, false);
  }

  map.addEventListener("load", () => {
    // The original 2D/2.5D view is served from the same web folder.
    if (!map.contentWindow?.showSutelliteCells) return;
    setTheme(theme);
    updateLanguage();
    if (accessibility) setAccessibility(accessibility);
    mapResize?.disconnect();
    const main = map.contentDocument.querySelector("main");
    fitMap = () => {
      if (!map.hidden && main.offsetHeight) map.style.height = (main.offsetHeight + 6) + "px";
    };
    mapResize = new ResizeObserver(fitMap);
    mapResize.observe(main);
    updateFrame();
  });

  async function showMap(selected, year, month) {
    const request = ++mapRequest;
    pendingMap = undefined;
    const months = selected.years[year].months;
    const period = month ? [months[month - 1]] : months;
    if (!period.some(record => record.detections > 0)) {
      map.hidden = true;
      showMessage(mapStatus, period.some(record => record.detections !== null)
        ? "No mapped detections in this period." : "No observations for this source and period.");
      return;
    }
    if (!selected.archive && !selected.map) {
      map.hidden = true;
      showMessage(mapStatus, "This region and source have no detections with coordinates. The calendar is available in Statistics.");
      return;
    }
    const title = selected.name + " · " + selected.label + " · " + periodName(year, month);
    try {
      let data = selected.map;
      if (selected.archive) {
        const name = selected.harmonized ? selected.archive : selected.archive + "-" + selected.platform;
        let cached = archiveMaps.get(name);
        if (!cached) {
          cached = { data: null };
          cached.pending = readArchive(name).then(value => {
            cached.data = value;
            return value;
          }).catch(error => {
            if (archiveMaps.get(name) === cached) archiveMaps.delete(name);
            throw error;
          });
          archiveMaps.set(name, cached);
          if (archiveMaps.size > 6) archiveMaps.delete(archiveMaps.keys().next().value);
        }
        if (!cached.data) {
          showMessage(mapStatus, "Loading hotspot cells…");
          map.hidden = true;
        }
        data = await cached.pending;
      }
      if (request !== mapRequest) return;
      const estimated = period.some(record => record.estimated);
      const observed = period.some(record => record.detections !== null && !record.estimated);
      const measure = estimated ? (observed ? "mixed" : "estimated") : "observed";
      pendingMap = [data, { year, month, measure }, title];
      map.hidden = false;
      showMessage(mapStatus, "");
      if (!map.hasAttribute("src")) map.src = "map.html?embed&theme=" + theme;
      updateFrame();
    } catch {
      if (request !== mapRequest) return;
      pendingMap = undefined;
      map.hidden = true;
      showMessage(mapStatus, "Could not load the hotspot map. Change region or reload the page.", true);
    }
  }

  return { selection, setActive, setTheme, setAccessibility, updateLanguage, showMap, hasUpload: () => Boolean(uploadedData) };
}
