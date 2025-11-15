# Health Analytics Platform

A modular data platform for ingesting, transforming, and analyzing healthcare data — combining robust ETL pipelines, machine learning models, and data warehousing to enable advanced insights into health and safety events.

---

## Overview

The **Health Analytics Platform** is an end-to-end system designed to collect public health datasets (such as **OpenFDA** and **Synthea**), load them into a centralized data warehouse, and power downstream analytics and machine learning models.

This monorepo unifies all major components:
- **ETL Pipelines:** Extract and load public health data using Airflow and Python.
- **Machine Learning Pipelines:** Train predictive models (e.g., drug safety, reaction severity) using scikit-learn and Pandas.
- **Data Warehouse Integration:** Centralized PostgreSQL or Snowflake layer for structured data.
- **Infrastructure:** Docker- and Terraform-based setup for reproducible local and cloud deployments.
---

## Repository Structure

```
health-analytics-platform
├───health-etl-pipeline
│   ├───airflow
│   │   ├───dags
│   │   └───etl
│   │       ├───common
│   │       ├───openfda
│   │       └───synthea
│   ├───data
│   ├───include
│   ├───pgadmin-config
│   ├───plugins
│   │   └───operators
│   ├───scripts
│   ├───sql
│   ├───test
│   ├───visualization
│   │   └───utils
│   └───warehouse
└───health-ml-pipelines
    ├───archive
    └───pipelines
        ├───common
        ├───reaction_prediction
        └───serious_prediction
```
---

## Components

### 1. ETL Pipelines (etl/)
- Uses **Apache Airflow** to orchestrate ingestion from OpenFDA, Synthea, and other sources.
- Loads structured data into a **PostgreSQL** warehouse (with optional Snowflake integration).
- Handles data cleaning, normalization, and schema management.

### 2. Machine Learning Pipelines (ml/)
- Includes modular ML pipelines built in scikit-learn and pandas.
- Currently supports:
  - **Serious Adverse Event Prediction** using OpenFDA data.
  - (Planned) **Reaction Prediction** models.
- Exports models as `.joblib` artifacts for downstream applications.

### 3. Infrastructure (Coming Soon)
- Provides a **Docker Compose** setup for local development, running:
  - Postgres
  - Airflow
  - pgAdmin
- Contains optional Terraform/Helm scaffolding for cloud deployment.

### 4. Dashboard (Coming Soon)
- A Flask + Plotly.js web dashboard for visualizing trends and model results.

---

## Getting Started

### 1. Clone the Repository
git clone https://github.com/jacklayfield/health-analytics-platform.git
cd health-analytics-platform

### 2. Start the Infrastructure (Coming soon)
cd infra
docker compose up -d

This will launch:
- Postgres (data warehouse)
- Airflow (ETL orchestrator)
- pgAdmin (database management UI)

NOTE: IF needed to reset pgadmin account:
docker compose down
docker volume rm health-analytics-platform_pgadmin-data
docker compose up -d

### 3. Run the ETL Pipeline
Once Airflow is up, open the UI at http://localhost:8080 and trigger the DAG:
openfda_etl_dag

### 4. Train the ML Model
cd ml/pipelines/serious_prediction
python train_logreg.py

This will train the logistic regression model on OpenFDA data and save it to:
ml/models/openfda_serious_predictor.joblib

---

## Architecture Overview

[OpenFDA / Synthea APIs] 
        ↓
     [Airflow ETL]
        ↓
 [PostgreSQL Data Warehouse]
        ↓
 [ML Pipelines (Scikit-Learn)]
        ↓
 [Model Artifacts / Dashboard / Analytics]

---

## Stack

| Layer | Technology |
|-------|-------------|
| **Orchestration** | Apache Airflow |
| **Database** | PostgreSQL (Snowflake planned) |
| **ML/Analytics** | Python, scikit-learn, pandas |
| **Infrastructure** | Docker, Terraform (optional), Helm (optional) |
| **Visualization** | Plotly.js / Flask (future) |

---

## Development Notes
- Ensure Python 3.10+ and Docker are installed.
- Environment variables are stored in `.env` files inside each module (e.g. `etl/.env`, `ml/.env`).
- Common utilities are located in `ml/pipelines/common/`.

**Environment Files (per-component `.env`)**:

- **Pattern**: each subproject has a component-local `.env` file (`ml/.env`, `etl/.env`, `infra/.env`, `dashboard/.env`) that contains the environment variables used by that component's `docker-compose`. This keeps components self-contained and makes it easy to run a single component in isolation.
- **Do not commit real secrets**: all `.env` files should contain non-sensitive defaults or placeholders. Add any real, sensitive secrets to your team's secrets manager or use Docker secrets for production. The repository's top-level `.gitignore` already excludes `.env` files.

- `ml/.env`: `WAREHOUSE_DB_URI`, `MLFLOW_TRACKING_URI`
- `etl/.env`: `AIRFLOW__*` settings (e.g., `AIRFLOW__CORE__FERNET_KEY`), `AIRFLOW_UID`, `AIRFLOW_ADMIN_USER`, `AIRFLOW_ADMIN_PASSWORD`
- `infra/.env`: `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `PGADMIN_DEFAULT_EMAIL`, `PGADMIN_DEFAULT_PASSWORD`, `PGADMIN_DEFAULT_SERVER_PASSWORD`
- `dashboard/.env`: `MLFLOW_TRACKING_URI`, `POSTGRES_URI`, `AIRFLOW_BASE`, `AIRFLOW_USERNAME`, `AIRFLOW_PASSWORD`

**How to use (local dev)**:
       - Create or edit the component `.env` files with appropriate values. For example:

```bash
cp ml/.env.example ml/.env
```

       - Start a single component from its directory:

```bash
cd ml
docker compose up --build
```

       - Or run a component compose file from the repo root (compose will read the component's `.env` file):

```bash
docker compose -f ml/docker-compose.yml up --build
```

       - Validate compose before bringing services up:

```bash
docker compose -f ml/docker-compose.yml config
```

**Production recommendations**:
- Move the most sensitive values (DB passwords, Airflow Fernet key, API keys) to a secrets manager or Docker secrets and reference them in a `docker-compose.prod.yml`.
- Keep `.env` files as lightweight local overrides only. Use CI/CD pipeline secrets for build/deploy-time configuration.


# Call docker compose directly and merge files:
```bash
docker compose \
       -f docker-compose.yml \
       -f infra/docker-compose.yml \
       -f etl/docker-compose.yml \
       -f ml/docker-compose.yml \
       -f dashboard/docker-compose.yml \
       up --build
```

---

## Future Roadmap
- [ ] Add reaction severity prediction pipeline.
- [ ] Integrate Synthea patient simulation data.
- [ ] Deploy ML models with FastAPI endpoints.
- [ ] Add health analytics dashboard.
- [ ] Move warehouse to Snowflake.
- [ ] Automate cloud infra with Terraform + Helm.

---

## Author
Jack Layfield  
GitHub: https://github.com/jacklayfield
