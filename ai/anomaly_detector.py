
import numpy as np
import pandas as pd


def aplicar_modelos(tabla, modelos):
    """Aplica los modelos mensuales de una región con sus referencias históricas."""
    salida = tabla.copy().reset_index(drop=True)
    if "media_referencia" not in salida:
        salida = salida.rename(columns={"media_historica": "media_referencia"})

    necesarias = ["month", "detections", "media_referencia", "maximo_historico"]
    faltan = set(necesarias) - set(salida.columns)
    if faltan:
        raise ValueError(f"Faltan columnas: {sorted(faltan)}")
    if set(modelos) != set(range(1, 13)):
        raise ValueError("Se necesitan los 12 modelos mensuales de una región.")
    if "region_id" in salida and salida["region_id"].nunique(dropna=False) > 1:
        raise ValueError("Pasa una única región por llamada.")

    for columna in necesarias:
        salida[columna] = pd.to_numeric(salida[columna], errors="raise")

    if not salida["month"].isin(range(1, 13)).all():
        raise ValueError("Hay meses fuera de 1 a 12.")

    detecciones = salida["detections"].to_numpy(dtype=float)
    if not np.isfinite(detecciones).all() or (detecciones < 0).any():
        raise ValueError("Hay detecciones inválidas.")

    con_historial = salida["media_referencia"].notna()
    if not con_historial.eq(salida["maximo_historico"].notna()).all():
        raise ValueError("Hay referencias históricas incompletas.")

    referencias = salida.loc[
        con_historial, ["media_referencia", "maximo_historico"]
    ].to_numpy(dtype=float)
    if not np.isfinite(referencias).all() or (referencias < 0).any():
        raise ValueError("Hay referencias históricas inválidas.")

    salida["resultado"] = "Sin historial del mes"
    salida["puntuacion"] = np.nan

    for mes in range(1, 13):
        filas = salida["month"].eq(mes) & con_historial
        if not filas.any():
            continue

        actual = salida.loc[filas, "detections"].to_numpy(dtype=float)
        media = salida.loc[filas, "media_referencia"].to_numpy(dtype=float)
        maximo = salida.loc[filas, "maximo_historico"].to_numpy(dtype=float)
        log_actual = np.log1p(actual)
        log_media = np.log1p(media)
        variables = np.column_stack([log_actual, log_media, log_actual - log_media])
        puntuacion = modelos[mes].decision_function(variables)
        anomalia = puntuacion < 0

        etiquetas = np.full(len(actual), "Sin alerta", dtype=object)
        etiquetas[anomalia] = "Anomalia estadistica"
        etiquetas[anomalia & (actual > maximo)] = "Alerta de aumento"
        salida.loc[filas, "resultado"] = etiquetas
        salida.loc[filas, "puntuacion"] = puntuacion

    return salida
