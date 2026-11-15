# Sutellite

**2026 NASA Space Apps Challenge — Harmonisation of MODIS and VIIRS active-fire observations**

Sutellite builds a reproducible data pipeline for inspecting, validating, preserving and comparing active-fire observations distributed through NASA FIRMS. The project brings MODIS and VIIRS records into a common Parquet schema while retaining source provenance and sensor-specific measurements.

The core implementation is a reproducible data preparation and exploratory comparison pipeline, with an optional experimental MODIS/VIIRS detection-count mapping model. Sutellite does **not** identify unique fire events, correct for satellite observation opportunity, estimate emissions or provide operational fire predictions.

## Project objectives

- Audit the expected structure and metadata of the raw FIRMS files.
- Record source-file provenance, row counts and SHA-256 checksums.
- Validate raw observations without modifying the source files.
- Convert source CSV records into a versioned canonical Parquet dataset.
- Preserve sensor-specific fields, source identifiers and quality flags.
- Profile the canonical dataset and compare selected MODIS and VIIRS platforms.
- Expose the processing workflow through a Streamlit prototype: upload FIRMS CSVs, create canonical Parquet, generate spatial cells, inspect statistics/anomalies and export cell products.

## Current dataset

The development dataset used for the current end-to-end processing run contains:

| Property | Observed inventory |
|---|---:|
| Study regions | 8 |
| Source product groups | 4 per region |
| Raw CSV files | 64 |
| ARCHIVE files | 32 |
| NRT files | 32 |
| Source observation records | 231,359,744 |
| Raw file size | 18,736,441,436 bytes |

The eight region identifiers are `angola`, `australia`, `brasil`, `california`, `espana`, `indonesia` and `rdc`, plus `zambia` (eight regions in total).

The inventory is a snapshot of the data currently used in development; it is not a promise that every future source download will have the same files, dates or row counts. A source manifest should be regenerated whenever the raw input collection changes.

## Repository layout

```text
sutellite/
├── web/                           # Atlas interface and bundled data
├── web_server.py                  # FastAPI atlas and CSV/Parquet upload
├── app.py                         # Streamlit regional explorer
├── .streamlit/config.toml         # Upload size and application theme
├── ai/                            # Experimental model and anomaly modules
├── archive/legacy_dashboard/      # Earlier static/Streamlit prototype (not active)
├── docs/                          # Scientific, schema and workflow documentation
├── geo/                           # Optional ESA WorldCover lookup
├── src/
│   ├── canonical_schema.py        # Versioned 35-field Arrow schema
│   ├── source_contracts.py        # FIRMS headers and satellite identity rules
│   ├── validate_data.py           # Row-level validation and quality flags
│   ├── process_data.py            # Batch CSV-to-canonical-Parquet pipeline
│   ├── upload_pipeline.py         # Website CSV upload-to-canonical-Parquet pipeline
│   ├── cell_aggregation.py        # Cell, monthly, daily, anomaly and GeoJSON products
│   └── ...                        # Audits, profiling and experimental analysis
├── tests/
├── requirements.txt
└── requirements-geo.txt           # Optional rasterio dependency for WorldCover
```

Raw CSV files and canonical Parquet datasets are external/local data, not included in Git. Generated upload outputs, caches and canonical Parquet files are written under `processed/` and excluded from commits. Small reviewed reports may be versioned; the large spatial comparison intermediate is excluded.

For the detailed module inventory, see [`docs/PROJECT_FILE_GUIDE.md`](docs/PROJECT_FILE_GUIDE.md). For the active website workflow, see [`docs/WEB_APP_WORKFLOW.md`](docs/WEB_APP_WORKFLOW.md).

## Requirements and setup

Use **Python 3.12 or newer** with the packages in `requirements.txt`. The website uses Streamlit, Plotly and PyDeck; the processing pipeline uses PyArrow, pandas and DuckDB. Shapely is used to clip uploads to custom GeoJSON regions.

### Windows PowerShell

1. Clone the repository and enter its directory:

   ```powershell
   git clone https://github.com/ainhoareycuesta/sutellite.git
   Set-Location .\sutellite
   ```

2. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install the project dependencies:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

4. Create a local environment file:

   ```powershell
   Copy-Item .env.example .env
   ```

5. Set `SUTELLITE_CANONICAL_DIR` in `.env` to the shared `canonical_v1.1` folder if the eight built-in regions should load immediately. Set `SUTELLITE_DATA_DIR` to the raw FIRMS dataset root only if you also intend to run the command-line batch pipeline. Optionally set `SUTELLITE_WORLDCOVER_DIR` to a local folder of ESA WorldCover GeoTIFF tiles and `SUTELLITE_AI_RESULTS_DIR` to the folder containing regional model-prediction CSVs.

Example paths (replace these with paths on your own machine):

```dotenv
SUTELLITE_DATA_DIR=D:/datasets/nasa_firms/raw
SUTELLITE_CANONICAL_DIR=D:/datasets/nasa_firms/processed/canonical_v1.1
SUTELLITE_WORLDCOVER_DIR=D:/datasets/worldcover
SUTELLITE_AI_RESULTS_DIR=D:/datasets/sutellite_ai/results
```

The raw and canonical datasets are not included in this repository. Each teammate must use the shared drive or obtain the source data separately. Do not commit `.env` or the data files.

### Start the integrated atlas

From the repository root, use Python 3.12 or newer.

```powershell
py -3 -m venv .venv-web
.\.venv-web\Scripts\python.exe -m pip install -r requirements-web.txt
.\.venv-web\Scripts\python.exe -m uvicorn web_server:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/>. The historical and AI anomaly maps use the bundled archive.

The **Historical Calendar** covers 2000–2025 for the eight study regions. Select MODIS (Terra or Aqua), VIIRS (S-NPP, NOAA-20 or NOAA-21), or the harmonized series. The harmonized series uses regionally calibrated MODIS estimates through January 2012 and observed S-NPP counts from February 2012. Months with no available FIRMS records are blank. The calendar, trend, CSV and 2D/2.5D map follow the same archive selection.

The archived map cells use the dominant valid [ESA WorldCover 2021 v200](https://doi.org/10.5281/zenodo.7254221) class within each 0.5° cell. It is a fixed land-cover reference. Contains modified Copernicus Sentinel data (2021) processed by the ESA WorldCover consortium under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Use **Analyze your data** to upload a CSV or Parquet of up to 25 MiB and 200,000 rows. The filter options reflect the sources in the file. Uploaded coordinates in the bundled lookup can use its WorldCover classes. For other coordinates, provide local tiles with `SUTELLITE_WORLDCOVER_DIR` and install `requirements-geo.txt`.

Use FIRMS observation records or a summary with a count and either an observation date or a year and month. If the file has no region column, enter a region name in the form. Coordinates are needed to display detections on the map. Counts and historical comparisons are calculated for each source in the uploaded file.

The generated Historical Calendar files live in `web/data/spatial/`. To rebuild them from the team data, run `python -m scripts.build_workspace_map --checkpoints <historical-checkpoints> --canonical <canonical_v1.1> --worldcover web/data/spatial/worldcover-2021.json`. The checkpoints contain the S-NPP cell totals and the canonical archive supplies Terra, Aqua, NOAA-20 and NOAA-21. The bundled land-class lookup was generated from the categorical overview of ESA WorldCover v200.

### Deploy the integrated atlas

For one public URL, deploy a Python web service on Render from this repository. Use the repository root, `pip install -r requirements-web.txt` for the build and `uvicorn web_server:app --host 0.0.0.0 --port $PORT` for startup. Set the health check path to `/api/health`.

For a Netlify frontend, publish `web/` and keep the same Python web service. Set `SUTELLITE_WEB_ORIGIN` on the service to the Netlify origin, such as `https://your-project.netlify.app`. Set the `sutellite-api-origin` meta tag in `web/index.html` to the Python service origin before publishing.

### Start the Streamlit explorer

From the repository root with `.venv` active:

```powershell
python -m streamlit run app.py
```

Streamlit prints a local address, normally `http://localhost:8501`. Open it in your browser. Stop the server with `Ctrl+C` in the terminal.

## Raw data layout

`SUTELLITE_DATA_DIR` must point to a directory whose immediate child folders follow the source contracts. The project expects eight region names, each combined with four platform suffixes:

```text
<raw-data-root>/
├── angola_modis/
├── angola_viirs_snpp/
├── angola_viirs_20/
├── angola_viirs_21/
├── australia_modis/
├── ...
└── zambia_viirs_21/
```

The complete region and folder contracts are defined in `src/source_contracts.py`. Each folder is expected to contain one ARCHIVE CSV and one NRT CSV. Filename product tokens are:

| Source | Filename token | Canonical platform(s) |
|---|---|---|
| MODIS Collection 6.1 | `M-C61` | Terra and Aqua |
| VIIRS Collection 2, S-NPP | `SV-C2` | SNPP |
| VIIRS Collection 2, NOAA-20 | `J1V-C2` | NOAA-20 |
| VIIRS Collection 2, NOAA-21 | `J2V-C2` | NOAA-21 |

Filenames follow the pattern `fire_archive_<token>_<numeric-id>.csv` or `fire_nrt_<token>_<numeric-id>.csv`. ARCHIVE and NRT have different source headers; the NRT contract does not include the ARCHIVE `type` field. Do not add or infer that field for NRT records.

See [`docs/data/RAW_DATA_CHARACTERISTICS.md`](docs/data/RAW_DATA_CHARACTERISTICS.md) for the source fields and sensor/product characteristics.

## Pipeline execution

Run commands from the repository root with the virtual environment activated. The `python -m src...` form is intentional: it runs the modules with the repository root on the import path.

### 1. Audit source folders and CSV headers

```powershell
python -m src.audit_sources
```

Checks the expected folders, filenames, source contracts, exact CSV headers, missing streams and duplicate streams. This is a metadata/header audit; it does not scan all observation rows.

Output: timestamped JSON under `reports/source_audits/`.

### 2. Build the source manifest

```powershell
python -m src.source_manifest
```

Reads the raw CSV files incrementally, calculates SHA-256 checksums and row counts, and records source identity and contract metadata. This step is required by the canonical processor. The processor verifies the current source files against the selected manifest before writing outputs.

Outputs under `reports/source_manifests/`:

- `source_manifest_<timestamp>.json`
- `source_manifest_files_<timestamp>.csv`

Regenerate the manifest whenever any source file is added, replaced or modified.

### 3. Run the independent raw-data validation report

```powershell
python -m src.validate_data
```

Scans the raw records and writes per-file and aggregate validation results under `reports/data_validation/`. This is a separate diagnostic validation pass. The canonical processor also validates records as it maps them, so this command is useful when a distinct validation report is required but it adds another full read of the raw dataset.

Outputs:

- `validation_report_<timestamp>.json`
- `validation_files_<timestamp>.csv`

### 4. Convert the raw CSVs to canonical Parquet

```powershell
python -m src.process_data
```

The processor requires a complete, valid source manifest. By default it selects the newest manifest in `reports/source_manifests/` and writes the canonical dataset to:

```text
processed/canonical_v1.1/
├── <region>/
│   └── <source-file-name>.parquet
├── quality_flags.parquet
├── rejected_records.jsonl
├── processing_report.json
└── processing_files.csv
```

Raw files are read only. Source rows that fail the implemented record rules are recorded in `rejected_records.jsonl`; non-fatal quality flags are recorded separately in `quality_flags.parquet` and linked by `observation_id`. The processing report includes source accounting and checksum verification.

The output directory must not already exist. The processor deliberately refuses to overwrite an existing output. Move or rename an earlier result, or supply a different `--output-dir`.

Optional arguments:

```powershell
python -m src.process_data --help
```

The processor supports `--manifest`, `--output-dir` and `--batch-size`.

### 5. Profile the canonical Parquet dataset

```powershell
python -m src.profile_data
```

Streams the canonical Parquet files and writes a dataset profile to `reports/data_profiles/`:

- `profile_summary.json`
- `group_summary.csv`
- `yearly_observations.csv`
- `monthly_observations.csv`
- `field_completeness.csv`
- `categorical_distributions.csv`
- `frp_summary.csv`
- `spatial_bounds.csv`

The default input is `processed/canonical_v1.1/`. The profiler checks the expected file and row inventory and warns if the observed dataset differs.

Optional arguments:

```powershell
python -m src.profile_data --help
```

### 6. Run the first MODIS/VIIRS comparison

```powershell
python -m src.compare_sensors
```

The default comparison uses:

- ARCHIVE observations only.
- Calendar years 2013–2025 inclusive.
- MODIS Terra, MODIS Aqua and VIIRS S-NPP, kept as separate platforms.
- All eight configured study regions.
- Day and night groups separately.
- An exploratory global 0.1-degree latitude/longitude grid.

Outputs under `reports/sensor_comparison/`:

| File | Contents |
|---|---|
| `monthly_platform_counts.csv` | Monthly platform counts, day/night counts, active detection days and observations per calendar day. |
| `monthly_sensor_comparison.csv` | Terra/SNPP and Aqua/SNPP descriptive observation-count ratios. |
| `monthly_frp_summary.csv` | Monthly FRP counts and descriptive statistics by platform and day/night. |
| `monthly_spatial_grid_counts.csv` | Source-observation counts grouped by 0.1-degree cell, month, platform and day/night. |
| `comparison_summary.json` | Filters, row accounting, methodology and limitations. |

Optional arguments:

```powershell
python -m src.compare_sensors --help
```

The comparison supports `--input-dir`, `--output-dir`, `--start-year`, `--end-year`, `--batch-size` and `--grid-size-degrees`.

### 7. Run the experimental full-dataset sensor count-mapping model

This experiment scans all canonical Parquet files recursively. It uses ARCHIVE
only (NRT is excluded to avoid mixing potentially overlapping source records),
keeps MODIS Aqua/Terra and VIIRS SNPP/NOAA-20/NOAA-21 counts separate, and then
constructs sensor-family totals on the common monthly 0.1-degree grid.

First prepare the global feature table:

```powershell
python -m src.sensor_harmonization_global prepare `
  --input-dir processed/canonical_v1.1 `
  --output-dir processed/harmonization_global `
  --grid-size-degrees 0.1 `
  --memory-limit 8GB
```

Then train and evaluate the candidate models:

```powershell
python -m src.sensor_harmonization_global train `
  --dataset-file processed/harmonization_global/global_monthly_cell_counts.parquet `
  --output-dir processed/harmonization_global/model
```

The default chronological split is training through 2022, validation 2023–2024,
and held-out test 2025. Data after the test window (such as an incomplete 2026
year) is not silently included in the training or test score. Model selection
uses validation MAE, with RMSE as a tie-breaker. Outputs include validation/test
metrics, predictions, a serialized model and a methodology summary.

For a faster first run, use `--skip-poisson-boosting` to train Ridge only.

Important: supervised fitting uses only grid/month/day-night groups with at
least one detection from **both** sensor families. One-sided groups are
excluded because a missing record is not a confirmed zero or proof of satellite
coverage. Estimates for MODIS-only groups are exploratory and are explicitly
flagged outside the validated population. The model estimates source detection
records, not unique fires or physical burning intensity. Read
[`docs/model_harmonization_global.md`](docs/model_harmonization_global.md)
before interpreting its results.

The global workflow uses DuckDB for out-of-core Parquet aggregation. Install
all dependencies with `pip install -r requirements.txt` before running it.

## Tests

Run the complete test suite from the repository root:

```powershell
python -m unittest discover -s tests -v
```

The suite covers source auditing, the canonical schema, comparison, processing, profiling, source contracts, source manifests, raw-data validation, validation-report aggregation and the experimental global count-mapping workflow. The global model tests can be run directly with `python -m pytest -q tests/test_sensor_harmonization_global.py`.

## Additional raw-data inspection tools

These are diagnostic utilities rather than required steps in the canonical pipeline. They read the raw dataset selected by `SUTELLITE_DATA_DIR`:

```powershell
python -m src.inventory_data
python -m src.inspect_schemas
python -m src.inspect_values
python -m src.inspect_pixel_dimensions
python -m src.inspect_negative_frp
python -m src.inspect_frp_context
python -m src.profile_raw_data
```

`inspect_negative_frp` rewrites `docs/data/NEGATIVE_FRP_RECORDS.csv`, and `inspect_frp_context` rewrites `docs/data/NEGATIVE_FRP_CONTEXT.csv`. Review the resulting Git diff before committing changes to those tracked evidence files.

## Scientific interpretation and limitations

The pipeline standardises storage and provenance; it does not make the sensors scientifically interchangeable. In particular:

- A row is one source observation record, not necessarily one unique fire event. Cross-sensor matching, event clustering and deduplication are not performed by the current comparison.
- MODIS confidence (numeric 0–100) and VIIRS confidence (`l`, `n`, `h`) are different conventions. They must not be treated as equivalent probabilities or merged into one numeric scale.
- MODIS and VIIRS have different nominal spatial resolutions and detection algorithms. The nominal resolution fields are not exact observation footprints. `scan_km` and `track_km` are retained as source-provided dimensions; the pipeline does not reconstruct exact pixel geometries.
- Terra, Aqua and SNPP have different overpass times and observation opportunities. The dataset does not currently contain a valid-observation opportunity denominator, so detection counts are not detection rates.
- `active_detection_days` means calendar dates with one or more detections. It does not mean the satellite successfully observed the region on those dates. Observations per calendar day are not coverage-normalised.
- MODIS/SNPP and Aqua/SNPP ratios are descriptive ratios of source-observation counts. They are not fire-event ratios, detection-probability estimates or sensor-bias correction factors.
- The 0.1-degree grid is angular, not equal-area; east-west cell width varies with latitude. Grid aggregation is exploratory and does not equalise sensor footprints.
- FRP distributions can be strongly right-skewed. Current comparison outputs include counts, minimum, mean, standard deviation and maximum, but not percentiles. Do not interpret mean differences alone as sensor bias.
- Negative FRP observations are retained and quality-flagged. They are excluded from nonnegative FRP summary statistics, not replaced with zero.
- ARCHIVE and NRT remain separate processing streams. The first comparison uses ARCHIVE only; it does not select between ARCHIVE/NRT duplicates or establish a source-precedence rule.
- The VIIRS I5 brightness-temperature mapping is marked provisional in the schema. Do not treat the mapped field as a fully verified MODIS BT31 equivalent.

These limitations should be preserved in presentations, reports and any downstream modelling work. The repository now includes an **experimental, paired-detection MODIS/VIIRS count-mapping model** (see `docs/model_harmonization_global.md`). It is not a fire-prediction model, does not correct for observation opportunity, and is not a validated correction for MODIS-only groups or operational fire-response decisions. Emissions estimation and production fire-event clustering are not implemented.

## Data and Git policy

Do not commit raw FIRMS CSV downloads, `.env` files, virtual environments, user uploads, canonical Parquet outputs or local WorldCover tiles. Generated application products and caches are written under ignored local folders. Small, reviewed reports and methodology documentation may be committed when they are useful to reproduce or audit the work. Do not commit the 600+ MB spatial-grid intermediate.

## Documentation notes

The file guide records known inconsistencies between some existing data documents and the executable implementation. In particular, the schema document's stated version/status and the historical validation report should be reconciled before those documents are treated as a single, fully synchronised specification.

## Website workflow

The Streamlit explorer in `app.py` opens on an interactive orthographic globe. The eight selectable study areas are Angola, Australia, Brasil, California, España, Indonesia, RDC and Zambia. Selecting a region opens a workspace with a 2D/2.5D spatial-cell map. A collapsible Analysis & monitoring panel contains Statistics, Anomalies, AI detections and a daily Hotspot calendar. Cell data and download products are in a separate expander.

The landing page also supports raw FIRMS CSV uploads for one of the eight built-in regions or a custom region with a WGS84 GeoJSON boundary. Built-in-region uploads must already be filtered to that area because precise study-area polygons are not bundled. AI detections are displayed only when regional prediction CSVs are available through `SUTELLITE_AI_RESULTS_DIR`; statistical anomaly flags are explicitly kept separate from AI model predictions.

The uploader accepts the FIRMS filename/product tokens already defined by the source contracts: `M-C61`, `SV-C2`, `J1V-C2` and `J2V-C2`, with `fire_archive_` or `fire_nrt_` prefixes. It harmonises the uploaded files only; it does not fetch missing satellites from NASA. Large historical collections should use the batch processing pipeline rather than the browser uploader.

See [`docs/WEB_APP_WORKFLOW.md`](docs/WEB_APP_WORKFLOW.md) for the exact input contracts, outputs, configuration and limitations.

## Data sources and attribution

The official satellite active-fire data used to build and harmonize the dataset in this pipeline was obtained from NASA Earthdata (FIRMS):

- **MODIS (Moderate Resolution Imaging Spectroradiometer):** [https://www.earthdata.nasa.gov/data/instruments/modis](https://www.earthdata.nasa.gov/data/instruments/modis)
- **VIIRS (Visible Infrared Imaging Radiometer Suite):** [https://www.earthdata.nasa.gov/data/instruments/viirs](https://www.earthdata.nasa.gov/data/instruments/viirs)
- **ESA WorldCover 2021 v200:** Zanaga, D., et al. (2022). *ESA WorldCover 10 m 2021 v200* ([https://doi.org/10.5281/zenodo.7254221](https://doi.org/10.5281/zenodo.7254221)), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## License

The source code and documentation in this repository are released into the public domain under the terms in [`LICENSE`](LICENSE).
