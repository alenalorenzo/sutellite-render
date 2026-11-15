
"""Ejecuta los modelos guardados de Sutellite desde una tabla mensual."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REGIONES = (
    "espana", "california", "indonesia", "zambia",
    "brasil", "australia", "angola", "rdc"
)
CLAVES = ["year", "month", "grid_lat", "grid_lon"]
REFERENCIAS_FLOAT32 = {"indonesia", "zambia"}


def leer_mensual(ruta):
    ruta = Path(ruta)
    if ruta.suffix.lower() == ".parquet":
        return pd.read_parquet(ruta)
    if ruta.suffix.lower() == ".csv":
        return pd.read_csv(ruta, low_memory=False, float_precision="round_trip")
    raise ValueError(f"Formato mensual desconocido: {ruta.name}")


def preparar_evaluacion(mensual, region):
    """Calcula las referencias 2013–2023 y las une a las filas de 2024–2025."""
    if region not in REGIONES:
        raise ValueError(f"Región desconocida: {region}")

    columnas = CLAVES + ["detections"]
    faltan = set(columnas) - set(mensual.columns)
    if faltan:
        raise ValueError(f"Faltan columnas en la tabla mensual: {sorted(faltan)}")

    datos = mensual.copy()
    for nombre in columnas:
        datos[nombre] = pd.to_numeric(datos[nombre], errors="raise")

    if not np.isfinite(datos[columnas].to_numpy(dtype=float)).all():
        raise ValueError("La tabla mensual contiene claves o recuentos inválidos.")
    if not datos["month"].isin(range(1, 13)).all() or datos["detections"].lt(0).any():
        raise ValueError("Hay meses o recuentos fuera de rango.")
    if datos.duplicated(CLAVES).any():
        raise ValueError("Hay cuadrículas mensuales duplicadas.")
    if "region_id" in datos and not datos["region_id"].eq(region).all():
        raise ValueError("La tabla mensual contiene registros de otra región.")

    claves_zona_mes = CLAVES[1:]
    historico = datos.loc[datos["year"].between(2013, 2023)]
    evaluacion = datos.loc[datos["year"].between(2024, 2025)].copy()
    if historico.empty or evaluacion.empty:
        raise ValueError("La tabla debe contener registros de ambos periodos.")

    referencias = historico.groupby(claves_zona_mes, as_index=False).agg(
        total=("detections", "sum"),
        maximo_historico=("detections", "max")
    )
    precision = np.float32 if region in REFERENCIAS_FLOAT32 else np.float64
    # En la matriz original, los años sin filas aportaban cero.
    referencias["media_referencia"] = referencias.pop("total").to_numpy(dtype=precision) / 11

    resultado = evaluacion.merge(
        referencias, on=claves_zona_mes, how="left", validate="many_to_one"
    )
    if "region_id" not in resultado:
        resultado.insert(0, "region_id", region)
    return resultado


def main():
    from ai.anomaly_detector import aplicar_modelos

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", choices=REGIONES, required=True)
    parser.add_argument("--mensual", type=Path, required=True)
    parser.add_argument("--modelo", type=Path, required=True)
    parser.add_argument("--salida", type=Path, required=True)
    parser.add_argument("--worldcover", type=Path, help="Carpeta de teselas de la región")
    args = parser.parse_args()

    if args.salida.exists():
        parser.error(f"Ya existe la salida: {args.salida}")
    tabla = preparar_evaluacion(leer_mensual(args.mensual), args.region)
    modelos = joblib.load(args.modelo)
    resultado = aplicar_modelos(tabla, modelos)

    if args.worldcover is not None:
        from geo.worldcover_lookup import anadir_worldcover
        resultado = anadir_worldcover(resultado, args.worldcover)

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(args.salida, index=False, mode="x")
    print(f"{len(resultado):,} filas guardadas en {args.salida}")


if __name__ == "__main__":
    main()
