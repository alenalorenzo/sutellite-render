
"""Consulta puntual de WorldCover 2021 en las claves de cuadrícula."""

from pathlib import Path
import re

import numpy as np
import pandas as pd

CLASES = {
    10: "Bosque", 20: "Matorral", 30: "Pastizal", 40: "Cultivos",
    50: "Zona urbana", 60: "Suelo desnudo", 70: "Nieve y hielo",
    80: "Agua", 90: "Humedal herbaceo", 95: "Manglar", 100: "Musgos y liquenes"
}
PATRON_TESELA = re.compile(r"([NS]\d{2}[EW]\d{3})_Map", re.IGNORECASE)


def _tesela(lat, lon):
    sur = int(np.floor(lat / 3) * 3)
    oeste = int(np.floor(lon / 3) * 3)
    ns = "S" if sur < 0 else "N"
    ew = "W" if oeste < 0 else "E"
    return f"{ns}{abs(sur):02d}{ew}{abs(oeste):03d}"


def anadir_worldcover(tabla, carpeta_teselas):
    """Añade la clase de la muestra en grid_lon, grid_lat."""
    import rasterio

    carpeta = Path(carpeta_teselas)
    if not carpeta.is_dir():
        raise FileNotFoundError(f"Carpeta WorldCover inexistente: {carpeta}")

    teselas = {}
    for ruta in sorted(carpeta.rglob("*.tif"), key=lambda p: (len(p.parts), str(p))):
        codigo = PATRON_TESELA.search(ruta.name)
        if codigo:
            teselas.setdefault(codigo.group(1).upper(), ruta)
    if not teselas:
        raise FileNotFoundError(f"No se encontraron teselas Map en {carpeta}")

    claves = ["grid_lat", "grid_lon"]
    puntos = tabla[claves].drop_duplicates().reset_index(drop=True)
    if not np.isfinite(puntos.to_numpy(dtype=float)).all():
        raise ValueError("Hay coordenadas inválidas para WorldCover.")

    puntos["tesela"] = [
        _tesela(lat, lon) for lat, lon in puntos[claves].itertuples(index=False, name=None)
    ]
    codigos = pd.Series(pd.NA, index=puntos.index, dtype="Int64")

    for tesela, grupo in puntos.groupby("tesela"):
        ruta = teselas.get(tesela)
        if ruta is None:
            continue
        with rasterio.open(ruta) as src:
            if src.crs is None or src.crs.to_epsg() != 4326:
                raise ValueError(f"CRS inesperado en {ruta.name}: se esperaba EPSG:4326.")
            coordenadas = grupo[["grid_lon", "grid_lat"]].itertuples(index=False, name=None)
            muestras = src.sample(coordenadas, indexes=1, masked=True)
            for indice, muestra in zip(grupo.index, muestras):
                valor = muestra[0]
                if not np.ma.is_masked(valor) and int(valor) in CLASES:
                    codigos.at[indice] = int(valor)

    puntos["worldcover_2021_codigo"] = codigos
    puntos["worldcover_2021_clase"] = codigos.map(CLASES).fillna("Sin datos WorldCover")
    salida = ["worldcover_2021_codigo", "worldcover_2021_clase"]
    return tabla.drop(columns=salida, errors="ignore").merge(
        puntos[claves + salida], on=claves, how="left", validate="many_to_one"
    )
