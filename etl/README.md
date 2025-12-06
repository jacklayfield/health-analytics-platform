# ETL (Extract, Transform, Load)

A robust ETL pipeline system for ingesting, transforming, and loading healthcare data into a centralized data warehouse. Built on **Apache Airflow**, this module orchestrates data workflows from multiple sources (OpenFDA, Synthea, etc.) and makes them available for downstream analytics and machine learning.

---

## Overview

The ETL module provides:
- **DAG orchestration** via Apache Airflow (v2.9.0)
- **Modular pipeline design** for healthcare data ingestion
- **Data transformation** and validation workflows
- **PostgreSQL warehouse integration** for persistent storage
- **Docker containerization** for reproducible local and cloud deployments

---

## Project Structure

```
etl/
├── README.md                    # This file
├── docker-compose.yml           # Docker Compose setup for local Airflow
├── requirements.txt             # Python dependencies
├── .env.example                 # Example environment variables
├── airflow/
│   ├── dags/                    # Airflow DAG definitions (entry point for workflows)
│   │   └── openfda_etl_dag.py   # Example: OpenFDA data ingestion DAG
│   └── etl/                     # ETL modules and helper functions
│       ├── common/              # Common utilities (logging, DB connectors, etc.)
│       ├── openfda/             # OpenFDA-specific ETL logic
│       └── synthea/             # Synthea-specific ETL logic (planned)
├── data/
│   ├── raw/                     # Raw data ingestion (not committed)
│   └── processed/               # Processed data artifacts (not committed)
├── include/                     # Airflow include directory (configs, macros)
├── plugins/                     # Airflow plugins and custom operators
│   └── operators/               # Custom operator definitions
├── scripts/                     # Utility scripts (setup, CI/CD, etc.)
├── sql/                         # SQL schemas, queries, and data warehouse definitions
│   ├── age_distribution.sql     # Analytics: age breakdown
│   ├── common_reactions.sql     # Analytics: most common adverse reactions
│   ├── most_reported_product.sql # Analytics: top reported products
│   ├── reports_by_month.sql     # Analytics: temporal trends
│   └── sex_distribution.sql     # Analytics: sex breakdown
├── test/                        # Unit and integration tests
├── visualization/               # Data visualization dashboards
│   ├── app.py                   # Plotly/Dash dashboard
│   ├── Dockerfile               # Containerized visualization
│   └── utils/                   # Visualization helpers
└── warehouse/
    └── queries.py               # Reusable database query utilities
```

---

## Getting Started

### Prerequisites

- Docker & Docker Compose (v1.29+)
- Python 3.10+ (for local Airflow development outside containers)
- PostgreSQL client tools (optional; for direct DB access)

### 1. Set Up the Environment

Copy the example environment file and configure for your environment:

```bash
cd etl
cp .env.example .env
# Edit .env to set:
#   - AIRFLOW__CORE__FERNET_KEY (generate with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
#   - AIRFLOW_WEBSERVER_SECRET_KEY (any random string for development)
#   - AIRFLOW_ADMIN_USER and AIRFLOW_ADMIN_PASSWORD
```

### 2. Start Airflow

From the `etl/` directory:

```bash
docker compose up --build
```

This will:
- Initialize the Airflow database
- Create the admin user
- Start the Airflow webserver (port 8080)
- Start the Airflow scheduler (continuously monitoring for DAGs to run)

Access Airflow at: **http://localhost:8080**

### 3. Trigger a DAG

1. Open the Airflow UI at http://localhost:8080
2. Log in with your admin credentials (default: `admin` / `admin`)
3. Find the DAG (e.g., `openfda_etl_dag`) in the DAG list
4. Click the DAG name to view details
5. Click the play button (▶) to trigger a run

---

## Environment Configuration

### Required Variables

- `AIRFLOW__CORE__FERNET_KEY` — Encryption key for Airflow secrets (generate with cryptography library)
- `AIRFLOW_WEBSERVER_SECRET_KEY` — Webserver session key (any random string)
- `AIRFLOW_ADMIN_USER` — Initial admin username
- `AIRFLOW_ADMIN_PASSWORD` — Initial admin password

### Optional Overrides

These have sensible defaults but can be overridden in `.env`:

- `AIRFLOW__CORE__EXECUTOR` (default: `LocalExecutor`)
- `AIRFLOW__DATABASE__SQL_ALCHEMY_CONN` (default: PostgreSQL on `postgres:5432`)
- `AIRFLOW__CORE__LOGGING_LEVEL` (default: `INFO`)
- `AIRFLOW__CORE__LOAD_EXAMPLES` (default: `False`)
- `AIRFLOW_UID` (default: `50000` — Linux user ID for mounted volumes)

See `etl/.env.example` for the complete list.

---

## Running the Full Stack

To bring up the entire health-analytics platform (infrastructure, ETL, ML, dashboard) from the repo root:

```bash
docker compose -f docker-compose.yml \
  -f infra/docker-compose.yml \
  -f etl/docker-compose.yml \
  -f ml/docker-compose.yml \
  -f dashboard/docker-compose.yaml up --build
```

Or use the helper script:

```bash
./scripts/start-all.sh
```

---

## Writing and Debugging DAGs

### Creating a New DAG

1. Create a Python file in `airflow/dags/` (e.g., `my_pipeline.py`):

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def extract_data():
    """Extract phase: fetch raw data."""
    print("Extracting data...")

def transform_data():
    """Transform phase: clean and reshape data."""
    print("Transforming data...")

def load_data():
    """Load phase: write to warehouse."""
    print("Loading data...")

dag = DAG(
    'my_etl_pipeline',
    start_date=datetime(2025, 1, 1),
    schedule_interval='@daily',  # Run daily
    catchup=False,
)

t1 = PythonOperator(task_id='extract', python_callable=extract_data, dag=dag)
t2 = PythonOperator(task_id='transform', python_callable=transform_data, dag=dag)
t3 = PythonOperator(task_id='load', python_callable=load_data, dag=dag)

t1 >> t2 >> t3  # Define task dependencies
```

2. Save the file; Airflow will auto-discover it within ~1 minute.

### Viewing DAG Code and Logs

- **Code view**: In the Airflow UI, click the DAG name → "Code" tab
- **Logs**: Click a task run in the "Graph View" or "Tree View" to see detailed logs
- **Local logs**: Check `etl/logs/` for Airflow logs (mounted in the container)

### Testing Locally

Use Airflow's testing tools:

```bash
# Test a single task
docker compose exec airflow-webserver airflow tasks test my_etl_pipeline extract 2025-01-01

# List all DAGs
docker compose exec airflow-webserver airflow dags list
```

---

## Data Warehouse Integration

### SQL Schemas and Queries

The `sql/` directory contains SQL files for:
- **Schema definitions**: Table structures for raw and processed data
- **Analytics queries**: Pre-built queries for common health metrics (age distribution, reactions by product, temporal trends, etc.)

### Using `warehouse/queries.py`

Reusable query utilities are in `warehouse/queries.py`. Example:

```python
from warehouse.queries import get_postgres_connection

conn = get_postgres_connection()
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM adverse_events;")
result = cursor.fetchone()
print(f"Total events: {result[0]}")
```

### Connecting to PostgreSQL

From outside the container:

```bash
psql -h localhost -U airflow -d airflow
# Enter password: airflow (default)
```

Or use pgAdmin at **http://localhost:5050** (default: `admin@example.com` / `password`).

---

## Development and Testing

### Running Tests

Tests are in `test/` and use pytest:

```bash
# Run all tests
docker compose exec airflow-webserver pytest test/

# Run a specific test file
docker compose exec airflow-webserver pytest test/test_openfda_etl.py -v
```

### Viewing Airflow Logs

Airflow logs are mounted at `etl/logs/`:

```bash
# View recent logs
tail -f etl/logs/dag_id/task_id/attempt_id/log.txt
```

### Restarting Services

```bash
# Restart all ETL services
docker compose restart

# Restart just the scheduler
docker compose restart airflow-scheduler

# Stop all services
docker compose down

# Remove volumes (careful: deletes data)
docker compose down -v
```

---

## Common Issues and Troubleshooting

### Issue: "Fernet key not set"

**Solution**: Generate and set `AIRFLOW__CORE__FERNET_KEY` in `etl/.env`:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

### Issue: DAG not appearing in UI

**Solution**: 
- Ensure the Python file is in `airflow/dags/` and is syntactically valid
- Check Airflow logs for import errors: `docker compose logs airflow-webserver | grep -i error`
- Airflow scans for DAGs every 1-2 minutes; wait and refresh the UI

### Issue: Database connection refused

**Solution**:
- Ensure the infra stack (PostgreSQL) is running: `docker compose ps | grep postgres`
- Check `AIRFLOW__DATABASE__SQL_ALCHEMY_CONN` in `.env` matches the database host/port
- From repo root, ensure the `health-net` network exists: `docker network ls | grep health-net`

### Issue: Out of disk space

**Solution**:
- Remove old containers and images: `docker system prune`
- Or delete unused volumes: `docker volume prune`

## Resources

- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [OpenFDA API](https://open.fda.gov/apis/)
- [Synthea Patient Simulator](https://github.com/synthetichealth/synthea)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
