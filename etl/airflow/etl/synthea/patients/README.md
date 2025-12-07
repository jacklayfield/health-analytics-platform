Synthea ETL
===========

Overview
--------
This directory contains a small Synthea ETL pipeline used by the Airflow DAG `synthea_etl_dag.py`.
It follows a simple extract -> transform -> load pattern mirroring the OpenFDA pipeline in this repo.

Files
-----
- `extract.py` — prepares a single JSON array file at `/opt/airflow/data/raw/synthea/patients.json`.
  - Behavior: if `patients.json` already exists the extractor returns immediately.
  - If not, it will search the configured raw directory for files with extensions `.json`, `.ndjson`, or `.csv`.
  - It will convert NDJSON/CSV to a JSON array and write `patients.json` for downstream tasks.
  - If no files are present it writes a small sample dataset to allow local development.

- `transform.py` — reads the JSON array (or CSV) and flattens patient records to a CSV
  at `/opt/airflow/data/processed/synthea_patients.csv` with these columns:
  - `patient_id`, `birth_date`, `gender`, `race`, `ethnicity`, `conditions`, `medications`
  - Multi-valued fields are serialized as semicolon-delimited strings.

- `load.py` — loads the processed CSV into the warehouse using `PostgresLoader`.
  - By default `load.py` looks for the `WAREHOUSE_DB_URI` environment variable and falls back to
    `postgresql+psycopg2://airflow:airflow@postgres:5432/airflow` which matches the local ETL compose setup.

How it integrates with Airflow
-----------------------------
- The DAG should call the extract -> transform -> load functions in order.
- The DAG can call the provided Python functions directly (via `PythonOperator`) or call the
  module CLIs via `BashOperator` if you prefer process isolation.

Where to put Synthea output
---------------------------
- Recommended: run Synthea (Docker or locally) and place output files in `etl/data/raw/synthea/` on the host.
- When running the ETL stack with Docker Compose, ensure the host `etl/data/raw/synthea` is mounted into
  the Airflow container at `/opt/airflow/data/raw/synthea` so the extractor can find the files.

Running Synthea (quick options)
------------------------------
1) Use the Synthea docker image (recommended for reproducibility):

```bash
# generate 100 patients as FHIR JSON
docker run --rm -v "$(pwd)/etl/data/raw/synthea:/output" synthetichealth/synthea:latest generate
# or run with Java options and output formats
```

2) Local Java install (example):

```bash
java -jar synthea-with-dependencies.jar -p 100 -o ./etl/data/raw/synthea
```

Notes & suggestions
-------------------
- The extractor will convert NDJSON/CSV into a single JSON array at
  `/opt/airflow/data/raw/synthea/patients.json`. That is the canonical input used by the transform.
- The loader currently replaces the target table (`if_exists='replace'`) — change to `append` if you
  want incremental loads in production.
- Sensitive connection strings should be provided via environment variables or Airflow connections
  (e.g., `WAREHOUSE_DB_URI`), not checked into the repo.

Testing locally
---------------
- You can exercise the pipeline locally without Airflow:

```bash
python etl/airflow/etl/synthea/extract.py --input-dir etl/data/raw/synthea --output /tmp/patients.json
python etl/airflow/etl/synthea/transform.py --input /tmp/patients.json --output /tmp/synthea_patients.csv
# set WAREHOUSE_DB_URI if you have a local Postgres instance
python etl/airflow/etl/synthea/load.py --input /tmp/synthea_patients.csv --db-uri postgresql+psycopg2://user:pw@host:5432/db
```

If you want, I can:
- Add a `BashOperator` to the DAG to run Synthea inside Docker,
- Add unit tests for `transform.py`, or
- Change the loader to use `append` mode and create migration-safe DDL.

