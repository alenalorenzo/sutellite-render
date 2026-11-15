const SATELLITE_TILES = "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/BlueMarble_NextGeneration/default/GoogleMapsCompatible_Level8/{z}/{y}/{x}.jpeg";
const { t, text, numberFormat } = window.sutelliteI18n;
const count = numberFormat({ maximumFractionDigits: 0 });

export const MODIS = {
  burned: "Burned area detected",
  unburned: "No burned area detected",
  "no-data": "No valid MODIS data",
};

function nameOf(feature) {
  return feature.properties.ADMIN;
}

export function makeAlertMap(countries, selectPoint) {
  const theme = getComputedStyle(document.documentElement);
  const color = name => theme.getPropertyValue(name).trim();
  const markerColors = {
    burned: color("--burned"), unburned: color("--unburned"), "no-data": color("--no-data"),
  };
  const markerOutline = color("--ink");
  const selectedOutline = color("--cream");
  const map = L.map("alert-map", {
    preferCanvas: true,
    scrollWheelZoom: false,
    minZoom: 1,
    maxZoom: 12,
    zoomSnap: 0.25,
    worldCopyJump: false,
    maxBounds: [[-85, -180], [85, 180]],
    maxBoundsViscosity: 0.85,
  }).setView([-12, 18], 5);
  map.attributionControl.setPrefix(
    '<a href="https://leafletjs.com/">Leaflet</a>',
  );
  map.attributionControl.addAttribution(
    '<a href="https://www.naturalearthdata.com/">Natural Earth</a>',
  );

  let region;
  let shownPoints = [];
  let markerLayers = [];
  let activeIndex = -1;
  let bounds;
  let popup;
  let popupIndex = -1;
  let colorVision = document.documentElement.hasAttribute("data-color-vision");
  map.createPane("outlineBasemap").style.zIndex = 190;
  const borders = L.geoJSON(countries, {
    interactive: false,
    pane: "outlineBasemap",
    style: feature => ({
      color: nameOf(feature) === region?.country ? color("--map-active-line") : color("--map-line"),
      weight: nameOf(feature) === region?.country ? 1.5 : 0.8,
      fillColor: nameOf(feature) === region?.country ? color("--map-active-land") : color("--map-land"),
      fillOpacity: 1,
    }),
  }).addTo(map);
  const countryOutline = L.geoJSON(null, {
    interactive: false,
    style: { color: color("--map-active-line"), weight: 1.35, opacity: 0.84, fill: false },
  }).addTo(map);
  const labels = L.layerGroup().addTo(map);
  const points = L.layerGroup().addTo(map);
  const satellite = L.tileLayer(SATELLITE_TILES, {
    maxNativeZoom: 8,
    maxZoom: 12,
    noWrap: true,
    attribution: '<a href="https://earthdata.nasa.gov/gibs">NASA GIBS</a> · <a href="https://science.nasa.gov/earth/earth-observatory/blue-marble-next-generation/">NASA Earth Observatory</a>',
  });
  const basemapButton = document.getElementById("map-basemap");
  const basemapStatus = document.getElementById("basemap-status");
  let tileFailed = false;
  satellite.on("loading", () => { tileFailed = false; });
  satellite.on("tileerror", () => { tileFailed = true; });
  satellite.on("load", () => {
    text(basemapStatus, "Outline map shown where satellite imagery is unavailable.");
    basemapStatus.hidden = !tileFailed;
  });
  basemapButton.addEventListener("click", () => {
    const showImagery = !map.hasLayer(satellite);
    if (showImagery) satellite.addTo(map);
    else map.removeLayer(satellite);
    basemapStatus.hidden = true;
    basemapButton.setAttribute("aria-pressed", String(showImagery));
  });
  satellite.addTo(map);

  function updateLabels() {
    labels.clearLayers();
    if (map.getZoom() < 3.75) return;
    const visible = map.getBounds().pad(0.1);
    for (const feature of countries.features) {
      const { LABEL_X: lon, LABEL_Y: lat } = feature.properties;
      const name = nameOf(feature);
      if (!Number.isFinite(lat) || !Number.isFinite(lon) || !visible.contains([lat, lon])) continue;
      if (region?.label === "California" && name === region.country) continue;
      const text = document.createElement("span");
      text.textContent = t(name === "Democratic Republic of the Congo" ? "DR Congo" : name);
      L.tooltip({
        permanent: true,
        direction: "center",
        className: "country-label" + (name === region?.country ? " active-country" : ""),
      }).setLatLng([lat, lon]).setContent(text).addTo(labels);
    }
    if (region?.label === "California") {
      L.tooltip({ permanent: true, direction: "center", className: "country-label active-country" })
        .setLatLng(region.center).setContent(t(region.label)).addTo(labels);
    }
  }

  function fit() {
    if (!bounds) return;
    map.fitBounds(bounds, { padding: [29, 29], maxZoom: 7, animate: false });
  }

  function show(nextRegion, regionBounds, selection, preserveView = false) {
    const refit = region?.label !== nextRegion.label || bounds !== regionBounds;
    if (preserveView && !refit && shownPoints === selection) return;
    region = nextRegion;
    bounds = regionBounds;
    map.closePopup();
    activeIndex = -1;
    if (refit) {
      borders.resetStyle();
      countryOutline.clearLayers();
      const country = region.country && region.label !== "California"
        ? countries.features.find(feature => nameOf(feature) === region.country)
        : null;
      if (country) countryOutline.addData(country);
    }
    shownPoints = selection;
    renderMarkers();
    map.invalidateSize({ pan: false });
    if (refit) fit();
    updateLabels();
  }

  function tooltipText(point) {
    const label = document.createElement("span");
    label.textContent = t(point.detections === 1 ? "{count} hotspot detection" : "{count} hotspot detections", {
      count: count.format(point.detections),
    });
    if (colorVision && MODIS[point.modis]) label.textContent += " · " + t(MODIS[point.modis]);
    return label;
  }

  function renderMarkers() {
    points.clearLayers();
    markerLayers = shownPoints.map((point, index) => {
      const category = Object.hasOwn(MODIS, point.modis) ? point.modis : "burned";
      const position = [point.lat, point.lon];
      const marker = colorVision ? L.marker(position, {
        icon: L.divIcon({ className: "alert-symbol " + category, iconSize: [14, 14], iconAnchor: [7, 7] }),
        keyboard: false,
      }) : L.circleMarker(position, {
        radius: shownPoints.length > 350 ? 3.6 : 4.7,
        color: markerOutline,
        weight: 1.15,
        fillColor: markerColors[category],
        fillOpacity: 0.94,
      });
      return marker.bindTooltip(tooltipText(point), { className: "point-tooltip", direction: "top", offset: [0, -5] })
        .on("click", () => selectPoint(index, false, true))
        .addTo(points);
    });
    styleMarker(markerLayers[activeIndex], true);
  }

  function setColorVision(enabled) {
    if (colorVision === enabled) return;
    colorVision = enabled;
    renderMarkers();
  }

  function styleMarker(marker, selected) {
    if (!marker) return;
    if (colorVision) {
      marker.setZIndexOffset(selected ? 500 : 0);
      marker.getElement()?.classList.toggle("is-highlighted", selected);
    } else {
      if (selected) marker.bringToFront();
      marker.setStyle({ weight: selected ? 2.5 : 1.15, color: selected ? selectedOutline : markerOutline });
    }
  }

  function highlight(index, pan = false) {
    styleMarker(markerLayers[activeIndex], false);
    const marker = markerLayers[index];
    activeIndex = index;
    if (!marker) return;
    styleMarker(marker, true);
    if (pan && !map.getBounds().pad(-0.12).contains(marker.getLatLng())) {
      map.panTo(marker.getLatLng(), { animate: false });
    }
  }

  function explain(index, message, showPopup) {
    const marker = markerLayers[index];
    if (!marker || !showPopup) {
      map.closePopup();
      return;
    }
    const content = document.createElement("div");
    const title = document.createElement("strong");
    const body = document.createElement("p");
    title.textContent = t("Why this point was flagged");
    body.textContent = message;
    content.append(title, body);
    if (popupIndex === index && popup && map.hasLayer(popup)) {
      popup.setContent(content);
    } else {
      map.closePopup();
      const width = map.getSize().x;
      popupIndex = index;
      popup = L.popup({
        className: "alert-popup",
        maxWidth: Math.min(300, Math.max(155, width - 112)),
        autoPanPaddingTopLeft: [53, 16],
        autoPanPaddingBottomRight: [18, 24],
        offset: [0, -5],
      }).setLatLng(marker.getLatLng()).setContent(content).openOn(map);
    }
    popup.getElement().querySelector(".leaflet-popup-close-button").setAttribute("aria-label", t("Close popup"));
  }

  function localizeZoom() {
    for (const [direction, label] of [["in", "Zoom in"], ["out", "Zoom out"]]) {
      const button = map.getContainer().querySelector(".leaflet-control-zoom-" + direction);
      button.title = t(label);
      button.setAttribute("aria-label", t(label));
    }
  }
  localizeZoom();
  document.addEventListener("sutellite-language-change", () => {
    localizeZoom();
    updateLabels();
    markerLayers.forEach((marker, index) => marker.setTooltipContent(tooltipText(shownPoints[index])));
  });
  map.on("moveend", updateLabels);
  document.getElementById("reset-map").addEventListener("click", fit);
  return { show, highlight, explain, setColorVision, hasPopup: () => Boolean(popup && map.hasLayer(popup)), clear: () => {
    map.closePopup();
    shownPoints = [];
    markerLayers = [];
    activeIndex = -1;
    points.clearLayers();
  } };
}
