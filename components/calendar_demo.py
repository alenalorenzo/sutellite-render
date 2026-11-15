"""
Standalone Demo Viewer for the Historical Calendar Component.

Loads the repository's precomputed monthly platform summary directly
or allows users to upload their own datasets for real-time harmonization.
"""

import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st

# 1. Configurar la ruta raíz para que el frontend encuentre tu backend
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Importar tu adaptador
from data_adapter import load_data
from components.calendar_view import render_calendar

DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "reports" / "sensor_comparison" / "monthly_platform_counts.csv"

st.set_page_config(page_title="Satellite Historical Calendar", layout="wide")
st.title("Satellite Data Analysis | Historical Calendar Viewer")

@st.cache_data(show_spinner=False)
def _cached_load_from_path(
    path_str: str, sensor: str, platform: str
) -> pd.DataFrame:
    return load_data(path_str, sensor=sensor, platform=platform)

# --- PANEL LATERAL: Carga de Datos y Filtros ---
st.sidebar.header("Carga de Datos")
archivo_usuario = st.sidebar.file_uploader("Sube tu propio archivo (CSV/Parquet)", type=["csv", "parquet"])

st.sidebar.header("Visualization Filters")
sensor_choice = st.sidebar.selectbox("Sensor", options=["VIIRS", "MODIS"], index=0)

# El filtro ahora admite toda la constelación VIIRS (SNPP, NOAA-21, NOAA-20)
platform_options: List[str] = ["SNPP", "NOAA-21", "NOAA-20"] if sensor_choice == "VIIRS" else ["TERRA", "AQUA"]
platform_choice = st.sidebar.selectbox("Platform", options=platform_options, index=0)

# --- MOTOR DE INGESTA (El puente Frontend/Backend) ---
try:
    if archivo_usuario is not None:
        # El usuario ha subido un archivo: pasa por tu adaptador para validar y armonizar
        data: Optional[pd.DataFrame] = load_data(
            archivo_usuario, 
            sensor=sensor_choice, 
            platform=platform_choice
        )
        st.sidebar.success("Archivo armonizado correctamente.")
        st.caption(f"Source: **Archivo de Usuario (Dinámico)** | Sensor: **{sensor_choice}** | Platform: **{platform_choice}**")
        
    else:
        # No hay archivo: usamos la base histórica fija de la demo
        if not DEFAULT_SUMMARY_PATH.exists():
            st.error(
                f"Preloaded repository dataset not found at: `{DEFAULT_SUMMARY_PATH}`. "
                "Please verify that `reports/sensor_comparison/monthly_platform_counts.csv` exists."
            )
            st.stop()
            
        data: Optional[pd.DataFrame] = _cached_load_from_path(
            str(DEFAULT_SUMMARY_PATH), sensor_choice, platform_choice
        )
        st.caption(
            f"Source: `reports/sensor_comparison/monthly_platform_counts.csv` | "
            f"Sensor: **{sensor_choice}** | Platform: **{platform_choice}** | Stream: **ARCHIVE**"
        )
        
except Exception as error:
    st.error(f"Error cargando o armonizando el dataset: {error}")
    st.stop()

if data is None or data.empty:
    st.warning(
        f"El dataset no devolvió registros para Sensor={sensor_choice} y Platform={platform_choice}."
    )
    st.stop()

# --- VISUALIZACIÓN PRINCIPAL ---
col_reg, col_yr = st.columns(2)

available_regions: List[str] = sorted(data["region"].unique().tolist())
if not available_regions:
    st.warning("No se encontraron regiones válidas en el dataset.")
    st.stop()
    
selected_region: str = col_reg.selectbox("Select Region", available_regions)

filtered_data: pd.DataFrame = data[data["region"] == selected_region]

available_years: List[int] = sorted(
    [int(y) for y in filtered_data["date"].dt.year.dropna().unique()],
    reverse=True,
)

if not available_years:
    st.warning("No valid year records found for the selected region.")
    st.stop()

default_year_idx = available_years.index(2025) if 2025 in available_years else 0
selected_year: int = col_yr.selectbox("Select Year", available_years, index=default_year_idx)

render_calendar(filtered_data, selected_year)