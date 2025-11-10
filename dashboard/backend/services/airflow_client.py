import requests
from config import AIRFLOW_BASE, AIRFLOW_USERNAME, AIRFLOW_PASSWORD
from typing import Dict, Any

auth = None
if AIRFLOW_USERNAME and AIRFLOW_PASSWORD:
    auth = (AIRFLOW_USERNAME, AIRFLOW_PASSWORD)

def list_dags(limit=100):
    url = f"{AIRFLOW_BASE}/dags?limit={limit}"
    r = requests.get(url, auth=auth)
    r.raise_for_status()
    return r.json()

def get_dag(dag_id: str):
    url = f"{AIRFLOW_BASE}/dags/{dag_id}"
    r = requests.get(url, auth=auth)
    r.raise_for_status()
    return r.json()

def list_dag_runs(dag_id: str, limit=50):
    url = f"{AIRFLOW_BASE}/dags/{dag_id}/dagRuns?limit={limit}"
    r = requests.get(url, auth=auth)
    r.raise_for_status()
    return r.json()

def trigger_dag(dag_id: str, conf: dict = None):
    url = f"{AIRFLOW_BASE}/dags/{dag_id}/dagRuns"
    payload = {"conf": conf or {}}
    r = requests.post(url, json=payload, auth=auth)
    r.raise_for_status()
    return r.json()