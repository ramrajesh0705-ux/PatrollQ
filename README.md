# PatrolIQ

PatrolIQ is a Streamlit-based urban safety analytics platform for exploring Chicago crime data and identifying geographic and temporal patterns that can support evidence-based patrol planning. The current implementation is an offline analytics and machine-learning prototype: it cleans a Chicago crime CSV, derives time and severity features, evaluates unsupervised clustering methods, applies PCA, tracks experiments with MLflow, and presents the resulting analysis through an interactive dashboard.

> **Scope note:** The repository currently implements historical crime-data analysis and hotspot exploration. It does not currently include live sensor ingestion, user-report intake, automated alert dispatch, emergency-service integrations, patrol-route optimization, authentication, or a persistent application database.

## Contents

- [Features](#features)
- [Architecture and data flow](#architecture-and-data-flow)
- [Technology stack](#technology-stack)
- [Repository structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data preparation](#data-preparation)
- [Run the pipeline](#run-the-pipeline)
- [Launch the dashboard](#launch-the-dashboard)
- [Usage and demo guide](#usage-and-demo-guide)
- [Outputs and experiment tracking](#outputs-and-experiment-tracking)
- [Configuration and environment variables](#configuration-and-environment-variables)
- [Known limitations](#known-limitations)

## Features

### Interactive safety analytics dashboard

The Streamlit application in `app/` provides:

- A PatrolIQ landing page describing the operational use cases.
- Exploratory analysis of crime frequency by primary type, police district, community area, year, month, day, hour, and season.
- Interactive Plotly charts for crime trends, arrest status, domestic incidents, and year-month heatmaps.
- Geographic density visualization using latitude/longitude coordinates and OpenStreetMap tiles.
- Geographic K-Means hotspot analysis with cluster summaries, dominant crime types, peak hours, and district-based cluster labels.
- PCA variance and feature-importance visualizations.
- A page for reviewing stored K-Means experiment results and MLflow guidance.

### Machine-learning workflow

The training workflow in `src/train.py`:

1. Loads Chicago crime records from `data/raw/crimes.csv`.
2. Removes records missing `Latitude`, `Longitude`, or `Date`.
3. Derives `Hour`, `DayOfWeek`, `Month`, `Year`, `IsWeekend`, `TimeOfDay`, and `CrimeSeverity`.
4. Removes identifier columns and limits the cleaned dataset to 500,000 records.
5. Samples up to 50,000 records for clustering.
6. Selects seven clustering features: `Latitude`, `Longitude`, `Hour`, `Month`, `DayOfWeek`, `IsWeekend`, and `CrimeSeverity`.
7. Scales the feature matrix and evaluates K-Means, DBSCAN, and hierarchical clustering.
8. Uses silhouette score and Davies-Bouldin index to compare clustering quality.
9. Applies PCA and saves explained variance and feature-importance results.
10. Logs experiment parameters, metrics, and the result artifact to MLflow.

## Architecture and data flow

PatrolIQ is organized as a local, file-backed analytics pipeline rather than a service-oriented application:

```text
Chicago crime CSV
  data/raw/crimes.csv
          |
          v
src/data_loader.py       Read CSV with pandas
          |
          v
src/preprocessing.py     Clean rows, parse dates, derive temporal/severity fields
          |
          +--> data/processed/crime_cleaned.csv
          |
          v
src/features.py          Select clustering feature matrix
          |
          v
src/train.py             StandardScaler + K-Means / DBSCAN / hierarchical clustering
          |                         |
          |                         +--> MLflow experiment runs
          |
          +--> outputs/clustering_results.json
          +--> outputs/pca_results.json
          +--> logs/training_*.log

Streamlit pages read the processed CSV and JSON outputs
          |
          v
Interactive dashboard in app/Home.py and app/pages/
```

The dashboard reads local artifacts on each page. `01_Exploratory_Analysis.py` and `02_Clustering.py` consume `data/processed/crime_cleaned.csv`; the clustering, dimensionality, and MLflow pages consume JSON results in `outputs/`. Plotly renders charts and maps in the browser. No backend HTTP API, database, message queue, GIS server, or external alerting API is configured in the current codebase.

## Technology stack

- **Language:** Python 3.11 in the provided development container.
- **Application framework:** Streamlit 1.26.0.
- **Data processing:** pandas 1.5.3 and NumPy 1.24.3.
- **Machine learning:** scikit-learn 1.3.0, including K-Means, DBSCAN, agglomerative clustering, StandardScaler, PCA, and t-SNE utilities.
- **Visualization:** Plotly 5.15.0, with Matplotlib and Seaborn also listed as dependencies.
- **Experiment tracking:** MLflow 2.7.0 with the local `mlflow.db` and `mlruns/` artifacts.
- **Storage:** CSV and JSON files on the local filesystem; there is no application database or ORM.
- **Mapping/GIS:** Plotly map figures using latitude/longitude data and `open-street-map` tiles. No Google Maps, Mapbox token, geocoding service, or other map API key is required by the checked-in code.
- **Configuration:** YAML is available through PyYAML, although no application configuration file or required environment variable is currently defined.

## Repository structure

```text
.
├── app/
│   ├── Home.py                         Streamlit landing page
│   └── pages/
│       ├── 01_Exploratory_Analysis.py  Filters, charts, density map, trends
│       ├── 02_Clustering.py             Hotspot clustering and cluster summaries
│       ├── 03_Dimensionlity.py          PCA results and feature importance
│       └── 04_Mlflow_Integration.py     Experiment/result review page
├── data/
│   ├── raw/                             Expected location for crimes.csv
│   └── processed/                       Generated crime_cleaned.csv
├── src/
│   ├── data_loader.py                   CSV loading helper
│   ├── preprocessing.py                 Cleaning and feature derivation
│   ├── features.py                      Clustering feature selection
│   ├── clustering.py                    K-Means and DBSCAN helpers
│   ├── dimensionality.py                PCA/t-SNE utilities and JSON export
│   └── train.py                         End-to-end training and MLflow pipeline
├── outputs/                             Generated clustering and PCA JSON results
├── logs/                                Generated timestamped training logs
├── mlruns/                              Local MLflow file store
├── mlflow.db                            Local MLflow tracking database
├── requirements.txt                     Python dependencies
└── .devcontainer/devcontainer.json      Python 3.11 Codespaces configuration
```

## Prerequisites

- Python 3.10+; the repository's dev container uses Python 3.11.
- A local copy of the Chicago crime dataset in the expected CSV format.
- Sufficient memory and disk space for preprocessing large CSV files. The recorded training run loaded 2,752,438 rows and generated a 500,000-row cleaned sample.
- Optional: VS Code Dev Containers or GitHub Codespaces.

## Installation

```bash
git clone https://github.com/ramrajesh0705-ux/PatrollQ.git
cd PatrollQ

python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The supplied `.devcontainer/devcontainer.json` uses the `mcr.microsoft.com/devcontainers/python:1-3.11-bookworm` image, installs `requirements.txt`, and forwards Streamlit port `8501`.

## Data preparation

The training pipeline expects:

```text
data/raw/crimes.csv
```

The source data must include at least these columns:

- `Latitude`
- `Longitude`
- `Date`
- `Primary Type`
- `ID`
- `Case Number`
- `Updated On`

The dashboard also expects fields such as `District`, `Community Area`, `Arrest`, and `Domestic` for the corresponding charts. Place the source file at the path above before running the pipeline. The repository does not define a database migration or seed process; CSV placement is the data-loading step.

## Run the pipeline

From the repository root:

```bash
python src/train.py
```

The pipeline creates or updates:

```text
data/processed/crime_cleaned.csv
outputs/clustering_results.json
outputs/pca_results.json
logs/training_YYYYMMDD_HHMMSS.log
```

The current training code evaluates K-Means values from `k=5` through `k=10`, runs DBSCAN with `eps=0.01` and `min_samples=50`, and runs agglomerative clustering with five clusters. It records metrics in the local MLflow experiment named `Chicago Crime Clustering`.

## Launch the dashboard

```bash
streamlit run app/Home.py
```

Then open [http://localhost:8501](http://localhost:8501). In Codespaces or the dev container, port `8501` is configured for automatic forwarding.

To inspect MLflow runs in a second terminal:

```bash
mlflow ui
```

Then open [http://localhost:5000](http://localhost:5000).

## Usage and demo guide

1. Run the training pipeline once so that the processed CSV and JSON artifacts exist.
2. Start Streamlit with `streamlit run app/Home.py`.
3. On **Exploratory Data Analysis**, choose a year, day, or season in the sidebar and inspect crime-type, district, temporal, seasonal, arrest, and domestic-incident charts.
4. On **Clustering**, review K-Means silhouette/Davies-Bouldin comparisons, then inspect the geographic cluster map, cluster counts, dominant crime types, and peak hours.
5. On **Dimensionality**, review the three-component PCA variance charts and the ranked feature-importance values.
6. On **MLflow Integration**, review the saved K-Means comparisons and the selected best K value. For full run metadata, use `mlflow ui`.

There is currently no implemented test-alert button or live incident-ingestion workflow. To demonstrate a new scenario, replace or update `data/raw/crimes.csv`, rerun `python src/train.py`, and refresh the Streamlit pages.

## Outputs and experiment tracking

The checked-in example artifacts include PCA results for 50,000 rows and three components explaining approximately 62.1% cumulative variance. The stored clustering results include K-Means comparisons, DBSCAN and hierarchical silhouette scores, and feature-importance values. Treat these as generated artifacts tied to the included dataset, not as a guarantee for future datasets.

MLflow stores local tracking data in the repository's `mlflow.db`/`mlruns` paths. For reproducible experiments, preserve the input-data version, training log, generated JSON files, and MLflow run metadata together.

## Configuration and environment variables

No required `.env` file, secret, database URL, map token, or external API key is referenced by the current Python code. Paths are currently hard-coded, including:

- `data/raw/crimes.csv` in `src/train.py`
- `data/processed/crime_cleaned.csv` in the Streamlit pages
- `outputs/*.json` for dashboard results

If the platform is extended with live feeds, authenticated APIs, or a commercial mapping provider, add documented environment variables and keep credentials out of the repository.

## Known limitations

- The platform is historical/offline analytics, not a real-time incident-response system.
- No patrol-route optimizer, alert dispatcher, emergency coordination service, sensor adapter, or citizen-report API is implemented.
- There is no database schema, migration tooling, or seed command.
- The preprocessing function assumes the expected Chicago crime columns exist and samples exactly 500,000 rows, so smaller datasets may require code changes.
- The dashboard uses generated local files and relative paths; start commands should be run from the repository root.
- Clustering and visualizations can be memory-intensive for large datasets.
- The model outputs are analytical clusters, not predictions of individual behavior or determinations of criminality. Operational use should include privacy, governance, bias, and human-review safeguards.

## License

No license file is currently included in the repository. Add an explicit license before distributing or deploying PatrolIQ.
