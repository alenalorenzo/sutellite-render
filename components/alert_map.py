"""Sutellite alert map for Colab, web browsers, and Streamlit."""

import json
import math
from pathlib import Path
from string import Template

REGIONES = {
    "angola": "Angola",
    "australia": "Australia",
    "brasil": "Brazil",
    "california": "California",
    "espana": "Spain",
    "indonesia": "Indonesia",
    "rdc": "Democratic Republic of the Congo",
    "zambia": "Zambia",
}

MESES = (
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
)

WORLDCOVER_LABELS = {
    "Bosque": "Tree cover",
    "Matorral": "Shrubland",
    "Pastizal": "Grassland",
    "Cultivos": "Cropland",
    "Zona urbana": "Built-up",
    "Suelo desnudo": "Bare or sparse vegetation",
    "Nieve y hielo": "Snow and ice",
    "Agua": "Permanent water bodies",
    "Humedal herbaceo": "Herbaceous wetland",
    "Manglar": "Mangroves",
    "Musgos y liquenes": "Moss and lichen",
    "Sin datos WorldCover": "No WorldCover data",
}


def _leer_alertas(ruta_geojson):
    with Path(ruta_geojson).open(encoding="utf-8") as archivo:
        geojson = json.load(archivo)
    if geojson.get("type") != "FeatureCollection":
        raise ValueError("The GeoJSON file must be a FeatureCollection.")

    grupos, limites = {}, {}
    anios = set()
    for indice, alerta in enumerate(geojson["features"], start=1):
        datos = alerta["properties"]
        if datos.get("resultado") != "Alerta de aumento":
            continue
        geometria = alerta["geometry"]
        if geometria["type"] != "Point":
            raise ValueError(f"Alert {indice} must have Point geometry.")

        lon, lat = geometria["coordinates"][:2]
        if not (math.isfinite(lon) and math.isfinite(lat) and
                -180 <= lon <= 180 and -90 <= lat <= 90):
            raise ValueError(f"Alert {indice} has invalid coordinates.")
        region = datos["region_id"]
        anio, mes = int(datos["year"]), int(datos["month"])
        if not 1 <= mes <= 12:
            raise ValueError(f"Alert {indice} has an invalid month.")

        quemadura = datos.get("coincide_modis")
        if datos.get("modis_datos_validos") is False or quemadura is None:
            estado = "sin_datos"
        else:
            estado = "con_quemadura" if quemadura else "sin_quemadura"

        terreno = datos.get("worldcover_2021_clase") or "No WorldCover data"
        grupos.setdefault(f"{region}/{anio}/{mes}", []).append({
            "lat": lat,
            "lon": lon,
            "detecciones": datos["detections"],
            "maximo": datos.get("maximo_historico"),
            "terreno": WORLDCOVER_LABELS.get(terreno, terreno),
            "modis": estado,
        })
        anios.add(anio)

        sur, oeste, norte, este = limites.get(region, (lat, lon, lat, lon))
        limites[region] = [
            min(sur, lat), min(oeste, lon), max(norte, lat), max(este, lon)
        ]

    if not grupos:
        raise ValueError("The GeoJSON file contains no fire activity increase alerts.")
    regiones = {r: REGIONES.get(r, r) for r in limites}
    return grupos, limites, sorted(anios), regiones


def crear_mapa_alertas(ruta_geojson, *, region=None, anio=None, mes=None,
                       mostrar_filtros=True):
    """Return an interactive HTML map for the alert GeoJSON.

    When mostrar_filtros is False, the calling interface sets the selection
    with region, anio, and mes.
    """
    grupos, limites, anios, regiones = _leer_alertas(ruta_geojson)
    region = region if region is not None else (
        "espana" if "espana" in regiones else next(iter(regiones))
    )
    anio = anio if anio is not None else anios[-1]
    mes = mes if mes is not None else 8
    if region not in regiones or anio not in anios or not 1 <= mes <= 12:
        raise ValueError("Select a valid region, year, and month.")

    if not mostrar_filtros:
        clave = f"{region}/{anio}/{mes}"
        grupos = {clave: grupos.get(clave, [])}
        limites = {region: limites[region]}
        regiones = {region: regiones[region]}
        anios = [anio]

    datos = {
        "grupos": grupos,
        "limites": limites,
        "regiones": dict(sorted(regiones.items(), key=lambda par: par[1])),
        "anios": anios,
        "meses": MESES,
        "inicio": {"region": region, "anio": anio, "mes": mes},
        "mostrar_filtros": mostrar_filtros,
    }
    contenido = json.dumps(datos, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")).replace("<", "\\u003c")
    return PAGINA.substitute(datos=contenido)


PAGINA = Template(r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fire Activity Alert Map | Sutellite</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css">
<style>
  :root {
    color-scheme: light;
    --tinta: #152a33;
    --secundario: #465d68;
    --linea: #cfdbde;
    --quemadura: #b53e24;
    --sin-quemadura: #087785;
    --sin-dato: #5d6974;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    color: var(--tinta);
    background: #f5f8f9;
    font: 15px/1.45 "Public Sans", system-ui, sans-serif;
  }
  main { max-width: 1500px; margin: auto; padding: 24px; }
  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 16px;
    margin-bottom: 16px;
  }
  h1 { margin: 0; font-size: clamp(23px, 3.2vw, 34px); line-height: 1.15; }
  #periodo { margin: 5px 0 0; color: var(--secundario); }
  #cantidad {
    font-weight: 650;
    font-variant-numeric: tabular-nums;
    text-align: right;
    white-space: nowrap;
  }
  .filtros {
    display: grid;
    grid-template-columns: minmax(0, 2fr) minmax(90px, 0.75fr) minmax(116px, 1fr);
    gap: 12px;
    margin-bottom: 14px;
  }
  [hidden] { display: none !important; }
  .filtros label { display: grid; gap: 4px; font-weight: 600; min-width: 0; }
  select {
    width: 100%;
    min-height: 42px;
    border: 1px solid var(--linea);
    border-radius: 7px;
    background: white;
    color: var(--tinta);
    padding: 5px 9px;
    font: inherit;
    font-weight: 400;
  }
  .lienzo {
    position: relative;
    border-radius: 10px;
    border: 1px solid var(--linea);
    overflow: hidden;
  }
  #mapa { height: clamp(345px, 62vh, 720px); background: #deebf0; }
  #aviso {
    position: absolute;
    inset: 14px auto auto 50%;
    transform: translateX(-50%);
    margin: 0;
    padding: 8px 12px;
    background: white;
    border: 1px solid var(--linea);
    border-radius: 6px;
    text-align: center;
    width: max-content;
    max-width: calc(100% - 92px);
    z-index: 700;
  }
  .leyenda {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 6px 18px;
    margin: 12px 0 0;
    padding: 0;
    list-style: none;
    font-size: 13px;
  }
  .leyenda li { display: flex; align-items: center; gap: 7px; }
  .punto { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
  .punto.rojo { background: var(--quemadura); }
  .punto.azul { background: var(--sin-quemadura); }
  .punto.gris { background: var(--sin-dato); }
  .nota { color: var(--secundario); margin: 9px 0 0; font-size: 13px; }
  .ficha h2 { margin: 0; font-size: 16px; }
  .ficha dl { margin: 7px 0 0; display: grid; gap: 4px 12px;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); }
  .ficha dt { font-weight: 600; }
  .ficha dd { margin: 0; }
  .nombre-pais.leaflet-tooltip {
    background: transparent; border: 0; box-shadow: none;
    color: #50656c; font-size: 12px; font-weight: 650;
  }
  @media (max-width: 520px) {
    main { padding: 14px; }
    header { align-items: flex-start; gap: 9px; }
    #cantidad { padding-top: 3px; }
    .filtros { grid-template-columns: 1fr 1fr; gap: 8px; }
    .filtros label:first-child { grid-column: 1 / -1; }
    select { font-size: 16px; min-height: 44px; }
    #mapa { height: clamp(355px, 56vh, 620px); }
  }
</style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>Fire Activity Alert Map</h1>
      <p id="periodo"></p>
    </div>
    <output id="cantidad" role="status" aria-live="polite"></output>
  </header>
  <section class="filtros" id="filtros" aria-label="Filter alerts">
    <label>Region <select id="region"></select></label>
    <label>Year <select id="anio"></select></label>
    <label>Month <select id="mes"></select></label>
  </section>
  <section class="lienzo">
    <div id="mapa" aria-label="Locations of fire activity alerts"></div>
    <p id="aviso" role="status" hidden></p>
  </section>
  <ul class="leyenda" aria-label="MODIS comparison">
    <li><span class="punto rojo" aria-hidden="true"></span>Burned area detected by MODIS</li>
    <li><span class="punto azul" aria-hidden="true"></span>No burned area detected by MODIS</li>
    <li><span class="punto gris" aria-hidden="true"></span>No valid MODIS data</li>
  </ul>
  <p class="nota">Each point marks the center of a 0.1° grid cell. Alerts are based on VIIRS S-NPP detections.</p>
</main>
<script type="application/json" id="datos-alertas">$datos</script>
<!-- BASE CARTOGRAFICA -->
<script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
(() => {
  const datos = JSON.parse(document.getElementById("datos-alertas").textContent);
  const aviso = document.getElementById("aviso");
  if (typeof L === "undefined") {
    aviso.textContent = "Leaflet could not load. Check your connection and reload the page.";
    aviso.hidden = false;
    return;
  }

  const estilo = getComputedStyle(document.documentElement);
  const estados = {
    con_quemadura: {color: "--quemadura", texto: "Burned area detected"},
    sin_quemadura: {color: "--sin-quemadura", texto: "No burned area detected"},
    sin_datos: {color: "--sin-dato", texto: "No valid MODIS data"},
  };
  const mapa = L.map("mapa", {scrollWheelZoom: false, maxZoom: 16});
  const [sur, oeste, norte, este] = datos.limites[datos.inicio.region];
  mapa.fitBounds([[sur, oeste], [norte, este]], {maxZoom: 9, animate: false});
  const base = document.getElementById("base-cartografica");
  if (base) {
    L.geoJSON(JSON.parse(base.textContent), {
      interactive: false,
      style: f => f.properties.tipo === "estado" ?
        {color: "#718a91", weight: 1.2, fillOpacity: 0} :
        {color: "#a3b4b6", weight: 0.8, fillColor: "#f2f4e9", fillOpacity: 1},
      pointToLayer: (f, posicion) => L.circleMarker(posicion, {
        opacity: 0, fillOpacity: 0, radius: 0
      }).bindTooltip(f.properties.nombre, {
        permanent: true, direction: "center", className: "nombre-pais",
        opacity: 0.85,
      }),
    }).addTo(mapa);
    mapa.attributionControl.addAttribution(
      '<a href="https://www.naturalearthdata.com/">Natural Earth</a>'
    );
  } else {
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a>',
      maxZoom: 16,
    }).addTo(mapa);
  }
  const capaAlertas = L.layerGroup().addTo(mapa);

  const region = document.getElementById("region");
  const anio = document.getElementById("anio");
  const mes = document.getElementById("mes");
  const formato = new Intl.NumberFormat("en-US");
  for (const [valor, texto] of Object.entries(datos.regiones)) {
    region.add(new Option(texto, valor));
  }
  for (const valor of datos.anios) anio.add(new Option(valor, valor));
  datos.meses.forEach((texto, indice) => mes.add(new Option(texto, indice + 1)));
  region.value = datos.inicio.region;
  anio.value = datos.inicio.anio;
  mes.value = datos.inicio.mes;
  document.getElementById("filtros").hidden = !datos.mostrar_filtros;

  function ficha(punto) {
    const contenido = document.createElement("section");
    contenido.className = "ficha";
    const titulo = document.createElement("h2");
    titulo.textContent = "Fire activity increase alert";
    contenido.append(titulo);
    const detalle = document.createElement("dl");
    const campos = [
      ["VIIRS detections", formato.format(punto.detecciones)],
      ["Historical maximum for this month", punto.maximo == null ? "No data" : formato.format(punto.maximo)],
      ["WorldCover 2021", punto.terreno],
      ["MODIS", estados[punto.modis].texto],
    ];
    for (const [nombre, valor] of campos) {
      const dt = document.createElement("dt");
      const dd = document.createElement("dd");
      dt.textContent = nombre;
      dd.textContent = valor;
      detalle.append(dt, dd);
    }
    contenido.append(detalle);
    return contenido;
  }

  function actualizar() {
    const nombre = datos.regiones[region.value];
    const mesTexto = datos.meses[Number(mes.value) - 1];
    const puntos = datos.grupos[[region.value, anio.value, mes.value].join("/")] || [];
    document.getElementById("periodo").textContent = nombre + " · " + mesTexto + " " + anio.value;
    document.getElementById("cantidad").textContent = formato.format(puntos.length) +
      (puntos.length === 1 ? " alert" : " alerts");
    aviso.textContent = "No alerts for this selection.";
    aviso.hidden = puntos.length > 0;
    mapa.closePopup();
    capaAlertas.clearLayers();
    const extension = L.latLngBounds([]);

    for (const punto of puntos) {
      const posicion = [punto.lat, punto.lon];
      extension.extend(posicion);
      const color = estilo.getPropertyValue(estados[punto.modis].color).trim();
      const marca = L.circleMarker(posicion, {
        radius: 5, color: "#fff", weight: 1.15,
        fillColor: color, fillOpacity: 0.9,
      }).bindPopup(() => ficha(punto), {maxWidth: 310}).addTo(capaAlertas);
      const icono = marca.getElement();
      if (icono) {
        icono.setAttribute("tabindex", "0");
        icono.setAttribute("role", "button");
        const unidad = punto.detecciones === 1 ? " detection" : " detections";
        icono.setAttribute("aria-label", "Alert with " + formato.format(punto.detecciones) + unidad);
        icono.addEventListener("keydown", evento => {
          if (evento.key === "Enter" || evento.key === " ") {
            evento.preventDefault();
            marca.openPopup();
          }
        });
      }
    }
    if (!extension.isValid()) {
      const [sur, oeste, norte, este] = datos.limites[region.value];
      extension.extend([sur, oeste]).extend([norte, este]);
    }
    mapa.fitBounds(extension, {padding: [30, 30], maxZoom: 9, animate: false});
  }
  [region, anio, mes].forEach(filtro => filtro.addEventListener("change", actualizar));
  actualizar();
})();
</script>
</body>
</html>
""")
