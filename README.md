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
health-analytics-platform/
├── etl/                    # ETL pipelines using Apache Airflow for healthcare data ingestion
├── ml/                     # Machine learning pipelines with MLflow integration for health predictions
├── dashboard/              # Web dashboard with React frontend and Python backend
├── infra/                  # Infrastructure setup with Docker, Helm, and Terraform
├── data/                   # Data storage and processing artifacts
├── include/                # Shared configurations and utilities
├── plugins/                # Custom Airflow plugins and operators
└── requirements.txt/       # Python dependencies
```
---

## Components

### 1. ETL Pipelines (etl/)
- Robust ETL system built on **Apache Airflow** for orchestrating healthcare data ingestion from sources like OpenFDA and Synthea.
- Features modular pipeline design, data transformation, validation, and PostgreSQL warehouse integration.
- Includes Docker containerization, custom operators, SQL analytics queries, and a Plotly/Dash visualization dashboard.

### 2. Machine Learning Pipelines (ml/)
- Production-ready ML pipeline with comprehensive experiment tracking using **MLflow**.
- Supports multiple algorithms (Logistic Regression, Random Forest, etc.), hyperparameter optimization with Optuna, and model serving via FastAPI.
- Includes data validation, feature engineering, model evaluation, and CLI tools for all operations.

### 3. Infrastructure (infra/)
- Complete infrastructure setup with **Docker Compose** for local development, including Postgres, Airflow, and pgAdmin.
- Includes **Helm charts** for Kubernetes deployment and **Terraform** configurations for cloud infrastructure.
- Provides scripts and configurations for reproducible deployments.

### 4. Dashboard (dashboard/)
- Modern web dashboard with **React + TypeScript + Vite** frontend for data visualization and analytics.
- **Python backend** (Flask/FastAPI) for API endpoints, integrating with ML models and data warehouse.
- Features interactive charts and real-time insights into health data trends and model results.

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/jacklayfield/health-analytics-platform.git
cd health-analytics-platform
```

### 2. Start the Infrastructure
```bash
cd infra
docker compose up -d
```

This will launch:
- Postgres (data warehouse)
- Airflow (ETL orchestrator)
- pgAdmin (database management UI)

**NOTE:** If needed to reset pgadmin account:
```bash
docker compose down
docker volume rm health-analytics-platform_pgadmin-data
docker compose up -d
```

### 3. Run the ETL Pipeline
Once Airflow is up, open the UI at http://localhost:8080 and trigger the DAG:
- `openfda_etl_dag`

### 4. Train the ML Model
```bash
cd ml
# Follow the ML pipeline README for training commands
```

### 5. Launch the Dashboard
```bash
cd dashboard
docker compose up -d
```
Open the dashboard at http://localhost:3000 (frontend) and API at http://localhost:5000 (backend).

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
| **ML/Analytics** | Python, scikit-learn, pandas, MLflow, Optuna |
| **Infrastructure** | Docker, Docker Compose, Terraform, Helm, Kubernetes |
| **Visualization** | Plotly.js, Dash, React, TypeScript, Vite |
| **Backend** | Flask, FastAPI |

---

## Development Notes
- Ensure Python 3.10+, Node.js, and Docker are installed.
- Environment variables are stored in `.env` files inside each module (e.g. `etl/.env`, `ml/.env`).
- Common utilities are located in `etl/etl/common/` and `ml/src/utils/`.

**Environment Files (per-component `.env`)**:

- **Pattern**: each subproject has a component-local `.env` file (`ml/.env`, `etl/.env`, `infra/.env`, `dashboard/.env`) that contains the environment variables used by that component's `docker-compose`. This keeps components self-contained and makes it easy to run a single component in isolation.
- **Do not commit real secrets**: all `.env` files should contain non-sensitive defaults or placeholders. Add any real, sensitive secrets to your team's secrets manager or use Docker secrets for production. The repository's top-level `.gitignore` already excludes `.env` files.

- `ml/.env`: `WAREHOUSE_DB_URI`, `MLFLOW_TRACKING_URI`
- `etl/.env`: `AIRFLOW__*` settings (e.g., `AIRFLOW__CORE__FERNET_KEY`), `AIRFLOW_UID`, `AIRFLOW_ADMIN_USER`, `AIRFLOW_ADMIN_PASSWORD`
- `infra/.env`: `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `PGADMIN_DEFAULT_EMAIL`, `PGADMIN_DEFAULT_PASSWORD`, `PGADMIN_DEFAULT_SERVER_PASSWORD`
- `dashboard/.env`: `MLFLOW_TRACKING_URI`, `POSTGRES_URI`, `AIRFLOW_BASE`, `AIRFLOW_USERNAME`, `AIRFLOW_PASSWORD`

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
- [x] Implement ETL pipelines with Apache Airflow
- [x] Build ML pipelines with MLflow integration
- [ ] Set up infrastructure with Docker, Helm, Terraform
- [x] Create web dashboard with React frontend
- [ ] Add reaction severity prediction pipeline
- [ ] Integrate Synthea patient simulation data
- [ ] Deploy ML models with FastAPI endpoints
- [ ] Enhance dashboard with advanced analytics
- [ ] Move warehouse to Snowflake
- [ ] Automate cloud infra with Terraform + Helm

---

## Author
Jack Layfield  
GitHub: https://github.com/jacklayfield
