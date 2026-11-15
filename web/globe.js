const WORLD_WIDTH = 1440;
const WORLD_HEIGHT = 720;
const RADIANS = Math.PI / 180;
const INITIAL_LONGITUDE = 2;
const INITIAL_LATITUDE = 15;
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
const mapX = longitude => (longitude + 180) * WORLD_WIDTH / 360;
const mapY = latitude => (90 - latitude) * WORLD_HEIGHT / 180;

function countryPath({ geometry }) {
  const path = new Path2D();
  if (!geometry) return path;
  const polygons = geometry.type === "MultiPolygon" ? geometry.coordinates : [geometry.coordinates];
  for (const polygon of polygons) {
    for (const ring of polygon) {
      for (const [index, [lon, lat]] of ring.entries()) {
        if (index === 0) path.moveTo(mapX(lon), mapY(lat));
        else path.lineTo(mapX(lon), mapY(lat));
      }
      path.closePath();
    }
  }
  return path;
}

function gridPath() {
  const grid = new Path2D();
  for (let lon = -165; lon <= 165; lon += 15) {
    grid.moveTo(mapX(lon), 0);
    grid.lineTo(mapX(lon), WORLD_HEIGHT);
  }
  for (let lat = -75; lat <= 75; lat += 15) {
    grid.moveTo(0, mapY(lat));
    grid.lineTo(WORLD_WIDTH, mapY(lat));
  }
  return grid;
}

function createMarker(id, region, selectRegion) {
  const button = document.createElement("button");
  const leader = document.createElement("span");
  const label = document.createElement("span");
  button.type = "button";
  button.className = "globe-marker";
  button.dataset.region = id;
  button.hidden = true;
  leader.className = "marker-leader";
  leader.setAttribute("aria-hidden", "true");
  label.className = "marker-label";
  label.textContent = window.sutelliteI18n.t(region.label);
  button.append(leader, label);
  button.addEventListener("click", () => selectRegion(id));
  return { button, label, region };
}

function placeLabel(marker, x, y, width, height, occupied) {
  const labelWidth = marker.label.offsetWidth;
  const labelHeight = marker.label.offsetHeight;
  const separatedBy = 15;
  const left = x - labelWidth - separatedBy;
  const right = x + separatedBy;
  const above = y - labelHeight - 12;
  const below = y + 12;
  const candidates = [
    [right, above], [right, below], [left, above], [left, below],
    [x - labelWidth / 2, above - 18], [x - labelWidth / 2, below + 18],
    [right + 20, above - 23], [right + 20, below + 23],
  ];
  const boxes = candidates.map(([px, py]) => ({
    x: clamp(px, 5, width - labelWidth - 5),
    y: clamp(py, 30, height - labelHeight - 8),
    width: labelWidth,
    height: labelHeight,
  }));
  const box = boxes.find(candidate => !occupied.some(other =>
    candidate.x < other.x + other.width + 3 && candidate.x + candidate.width + 3 > other.x
    && candidate.y < other.y + other.height + 3 && candidate.y + candidate.height + 3 > other.y,
  )) ?? boxes[0];
  occupied.push(box);
  const endX = clamp(x, box.x + 2, box.x + box.width - 2) - x;
  const endY = clamp(y, box.y + 2, box.y + box.height - 2) - y;
  marker.button.style.setProperty("--label-x", box.x - x + "px");
  marker.button.style.setProperty("--label-y", box.y - y + "px");
  marker.button.style.setProperty("--leader-size", Math.hypot(endX, endY) + "px");
  marker.button.style.setProperty("--leader-angle", Math.atan2(endY, endX) + "rad");
}

function resizeCanvas(canvas) {
  const { clientWidth: width, clientHeight: height } = canvas;
  const ratio = Math.min(devicePixelRatio || 1, 2);
  const physicalWidth = Math.round(width * ratio);
  const physicalHeight = Math.round(height * ratio);
  if (canvas.width !== physicalWidth) canvas.width = physicalWidth;
  if (canvas.height !== physicalHeight) canvas.height = physicalHeight;
  return { width, height, ratio };
}

function sphereView(x, y, radius, longitude, latitude) {
  const lon = longitude * RADIANS;
  const lat = latitude * RADIANS;
  return {
    x, y, radius, longitude: lon,
    cosLon: Math.cos(lon), sinLon: Math.sin(lon),
    cosLat: Math.cos(lat), sinLat: Math.sin(lat),
  };
}

function projectPoint([latitude, longitude], view) {
  const angle = longitude * RADIANS - view.longitude;
  const sinLat = Math.sin(latitude * RADIANS);
  const cosLat = Math.cos(latitude * RADIANS);
  const facing = cosLat * Math.cos(angle);
  return {
    x: view.x + view.radius * cosLat * Math.sin(angle),
    y: view.y - view.radius * (view.cosLat * sinLat - view.sinLat * facing),
    depth: view.cosLat * facing + view.sinLat * sinLat,
  };
}

export function makeGlobe(countries, regions, selectRegion) {
  const canvas = document.getElementById("region-globe");
  const locator = document.getElementById("locator");
  const layer = document.getElementById("globe-markers");
  const paths = countries.features.filter(feature =>
    ["Polygon", "MultiPolygon"].includes(feature.geometry?.type),
  ).map(feature => ({ name: feature.properties.ADMIN, path: countryPath(feature) }));
  const highlightedCountries = new Map(Object.entries(regions).flatMap(([id, region]) =>
    id === "california" ? [] : [[region.country, id]],
  ));
  const selectablePaths = paths.flatMap(({ name, path }) =>
    highlightedCountries.has(name) ? [[highlightedCountries.get(name), path]] : [],
  );
  const markers = Object.entries(regions).map(([id, region]) => createMarker(id, region, selectRegion));
  layer.replaceChildren(...markers.map(marker => marker.button));

  const texture = document.createElement("canvas");
  texture.width = WORLD_WIDTH;
  texture.height = WORLD_HEIGHT;
  const textureContext = texture.getContext("2d", { willReadFrequently: true });
  const raster = document.createElement("canvas");
  const rasterContext = raster.getContext("2d");
  const globeContext = canvas.getContext("2d");
  const locatorContext = locator.getContext("2d");
  const grid = gridPath();
  let frame = 0;
  let longitude = INITIAL_LONGITUDE;
  let latitude = INITIAL_LATITUDE;
  let projection;
  let drag;
  let shaded;
  const colors = getComputedStyle(document.documentElement);
  const color = name => colors.getPropertyValue("--" + name).trim();
  let locatorRegion;
  let locatorPixels;
  const locatorStyle = () => locatorRegion ? "locator" : "locator-neutral";

  function paintTexture(style, selected) {
    const ocean = color(style + "-ocean");
    const land = color(style + "-land");
    const coast = color(style + "-coast");
    const highlight = style === "globe" ? "globe" : "locator";
    const activeLand = color(highlight + "-region");
    const activeCoast = color(highlight + "-region-line");
    textureContext.fillStyle = ocean;
    textureContext.fillRect(0, 0, WORLD_WIDTH, WORLD_HEIGHT);
    for (const { name, path } of paths) {
      const region = highlightedCountries.get(name);
      const active = region && (style === "globe" || region === selected);
      textureContext.fillStyle = active ? activeLand : land;
      textureContext.strokeStyle = active ? activeCoast : coast;
      textureContext.lineWidth = active ? 1.55 : .85;
      textureContext.fill(path, "evenodd");
      textureContext.stroke(path);
    }
    textureContext.lineWidth = .8;
    textureContext.strokeStyle = color(style + "-grid");
    textureContext.stroke(grid);
    return textureContext.getImageData(0, 0, WORLD_WIDTH, WORLD_HEIGHT).data;
  }

  function textureOffset(east, north, front, view) {
    const horizontal = view.cosLat * front - view.sinLat * north;
    const worldX = view.cosLon * horizontal - view.sinLon * east;
    const worldY = view.sinLon * horizontal + view.cosLon * east;
    const worldZ = clamp(view.cosLat * north + view.sinLat * front, -1, 1);
    const x = clamp(Math.floor(mapX(Math.atan2(worldY, worldX) / RADIANS)), 0, WORLD_WIDTH - 1);
    const y = clamp(Math.floor(mapY(Math.asin(worldZ) / RADIANS)), 0, WORLD_HEIGHT - 1);
    return (y * WORLD_WIDTH + x) * 4;
  }

  function drawSphere(context, view, pixels, maximum, rim) {
    const { x, y, radius } = view;
    const size = Math.min(maximum, Math.max(2, Math.round(radius * 2)));
    if (!shaded || shaded.width !== size) {
      raster.width = raster.height = size;
      shaded = rasterContext.createImageData(size, size);
    }
    const image = shaded.data;
    for (let row = 0; row < size; row++) {
      const north = 1 - (row + .5) * 2 / size;
      for (let col = 0; col < size; col++) {
        const east = (col + .5) * 2 / size - 1;
        const depth = 1 - east * east - north * north;
        const offset = (row * size + col) * 4;
        if (depth <= 0) {
          image[offset + 3] = 0;
          continue;
        }
        const front = Math.sqrt(depth);
        const source = textureOffset(east, north, front, view);
        const light = .55 + .45 * Math.max(0, -.38 * east + .42 * north + .82 * front);
        image[offset] = pixels[source] * light;
        image[offset + 1] = pixels[source + 1] * light;
        image[offset + 2] = pixels[source + 2] * light;
        image[offset + 3] = 255;
      }
    }
    rasterContext.putImageData(shaded, 0, 0);
    context.drawImage(raster, x - radius, y - radius, radius * 2, radius * 2);
    context.beginPath();
    context.arc(x, y, radius, 0, Math.PI * 2);
    context.strokeStyle = rim;
    context.lineWidth = 1.1;
    context.stroke();
  }

  function drawLocator() {
    const { width, height, ratio } = resizeCanvas(locator);
    if (!width || !height || !locatorPixels) return;
    const ctx = locatorContext;
    const region = regions[locatorRegion];
    const style = locatorStyle();
    const [lat, lon] = region?.center ?? [INITIAL_LATITUDE, INITIAL_LONGITUDE];
    const view = sphereView(width * .52, height * .62, Math.min(width * .47, height * 1.01), lon, lat);
    const { x, y, radius } = view;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    drawSphere(ctx, view, locatorPixels, 660, color(style + "-rim"));

    ctx.beginPath();
    ctx.ellipse(x, y, radius * 1.24, radius * .36, -.47, .14, 6.02);
    ctx.strokeStyle = color(style + "-orbit");
    ctx.lineWidth = 1.35;
    ctx.stroke();

    for (const [id, entry] of Object.entries(regions)) {
      const point = projectPoint(entry.center, view);
      if (point.depth < .03) continue;
      const active = id === locatorRegion;
      ctx.beginPath();
      ctx.arc(point.x, point.y, active ? 12 : 5, 0, Math.PI * 2);
      ctx.fillStyle = color(active ? "locator-active-halo" : style + "-dot-halo");
      ctx.fill();
      ctx.beginPath();
      ctx.arc(point.x, point.y, active ? 4.6 : 2.4, 0, Math.PI * 2);
      ctx.fillStyle = active ? color("alert-symbol-burned") || color("coral-soft") : color(region ? "sky" : "locator-neutral-dot");
      ctx.fill();
      if (active) {
        ctx.font = "600 14px " + color("sans");
        ctx.fillText(window.sutelliteI18n.t(entry.label), point.x + 13, point.y - 9, Math.max(4, width - point.x - 17));
      }
    }
  }

  function showLocator(region) {
    if (locatorPixels && locatorRegion === region) return;
    locatorRegion = region;
    locatorPixels = paintTexture(locatorStyle(), region);
    if (region) locator.dataset.region = region;
    else locator.removeAttribute("data-region");
    drawLocator();
  }

  function draw() {
    frame = 0;
    const { width, height, ratio } = resizeCanvas(canvas);
    if (!width || !height) return;
    const radius = Math.min(width, height) * .469;
    projection = sphereView(width / 2, height / 2, radius, longitude, latitude);
    globeContext.setTransform(ratio, 0, 0, ratio, 0, 0);
    globeContext.clearRect(0, 0, width, height);
    drawSphere(globeContext, projection, pixels, drag?.moved ? 470 : 720, color("globe-rim"));

    const visible = [];
    for (const marker of markers) {
      const point = projectPoint(marker.region.center, projection);
      marker.button.hidden = point.depth <= .055;
      if (marker.button.hidden) continue;
      marker.button.style.left = point.x + "px";
      marker.button.style.top = point.y + "px";
      visible.push({ marker, x: point.x, y: point.y });
    }
    const occupied = visible.map(point => ({ x: point.x - 7, y: point.y - 7, width: 14, height: 14 }));
    for (const point of visible) placeLabel(point.marker, point.x, point.y, width, height, occupied);
  }

  function redraw() {
    if (!frame) frame = requestAnimationFrame(draw);
  }

  function regionAt(event) {
    if (!projection) return undefined;
    const bounds = canvas.getBoundingClientRect();
    const east = (event.clientX - bounds.left - projection.x) / projection.radius;
    const north = (projection.y - event.clientY + bounds.top) / projection.radius;
    const depth = 1 - east * east - north * north;
    if (depth <= 0) return undefined;
    const offset = textureOffset(east, north, Math.sqrt(depth), projection) / 4;
    return selectablePaths.find(([, path]) => textureContext.isPointInPath(
      path, offset % WORLD_WIDTH, Math.floor(offset / WORLD_WIDTH), "evenodd",
    ))?.[0];
  }

  function rotate(east, north = 0) {
    longitude = (longitude + east + 540) % 360 - 180;
    latitude = clamp(latitude + north, -85, 85);
    redraw();
  }

  canvas.addEventListener("pointerdown", event => {
    if (!event.isPrimary || event.button !== 0) return;
    drag = { id: event.pointerId, x: event.clientX, y: event.clientY, moved: false };
    canvas.setPointerCapture(event.pointerId);
    canvas.setAttribute("data-dragging", "");
  });
  canvas.addEventListener("pointermove", event => {
    if (!drag) {
      canvas.toggleAttribute("data-over-region", Boolean(regionAt(event)));
      return;
    }
    if (event.pointerId !== drag.id) return;
    const dx = event.clientX - drag.x;
    const dy = event.clientY - drag.y;
    if (!drag.moved && Math.hypot(dx, dy) < 5) return;
    drag.x = event.clientX;
    drag.y = event.clientY;
    drag.moved = true;
    const speed = 58 / Math.max(projection?.radius ?? 100, 80);
    rotate(-dx * speed, dy * speed);
  });
  canvas.addEventListener("pointerup", event => {
    if (event.pointerId !== drag?.id) return;
    const region = drag.moved ? undefined : regionAt(event);
    drag = undefined;
    canvas.removeAttribute("data-dragging");
    if (region) selectRegion(region);
    else redraw();
  });
  canvas.addEventListener("lostpointercapture", () => {
    drag = undefined;
    canvas.removeAttribute("data-dragging");
    redraw();
  });
  canvas.addEventListener("keydown", event => {
    const turn = { ArrowLeft: [-19, 0], ArrowRight: [19, 0], ArrowUp: [0, 16], ArrowDown: [0, -16] }[event.key];
    if (!turn) return;
    event.preventDefault();
    rotate(...turn);
  });
  for (const button of document.querySelectorAll("[data-globe-turn]")) {
    button.addEventListener("click", () => rotate(Number(button.dataset.globeTurn)));
  }
  document.getElementById("reset-globe").addEventListener("click", () => {
    longitude = INITIAL_LONGITUDE;
    latitude = INITIAL_LATITUDE;
    redraw();
  });
  let pixels = paintTexture("globe");
  document.addEventListener("sutellite-language-change", () => {
    for (const { label, region } of markers) label.textContent = window.sutelliteI18n.t(region.label);
    drawLocator();
    redraw();
  });
  let colorVision = document.documentElement.hasAttribute("data-color-vision");
  document.addEventListener("sutellite-accessibility-change", ({ detail }) => {
    if (colorVision !== detail.colorVision) {
      colorVision = detail.colorVision;
      pixels = paintTexture("globe");
      locatorPixels = paintTexture(locatorStyle(), locatorRegion);
    }
    drawLocator();
    redraw();
  });
  showLocator();
  new ResizeObserver(redraw).observe(canvas);
  new ResizeObserver(drawLocator).observe(locator);
  document.fonts.ready.then(() => {
    drawLocator();
    redraw();
  });
  redraw();
  return showLocator;
}
