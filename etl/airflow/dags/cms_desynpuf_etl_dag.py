# CMS DE-SynPUF DAG

from datetime import datetime
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.append("/opt/airflow")

from etl.cms_desynpuf.extract import extract_cms_desynpuf
from etl.cms_desynpuf.transform import transform_cms_desynpuf
from etl.cms_desynpuf.load import load_cms_desynpuf

default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
}

with DAG(
    "cms_desynpuf_etl",
    default_args=default_args,
    schedule_interval="@monthly",
    catchup=False,
    description="Extract CMS DE-SynPUF Medicare claims data into the raw landing zone.",
) as dag:

    extract_cms_task = PythonOperator(
        task_id="extract_cms_desynpuf",
        python_callable=extract_cms_desynpuf,
    )

    transform_cms_task = PythonOperator(
        task_id="transform_cms_desynpuf",
        python_callable=transform_cms_desynpuf,
    )

    load_cms_task = PythonOperator(
        task_id="load_cms_desynpuf",
        python_callable=load_cms_desynpuf,
    )

    extract_cms_task >> transform_cms_task >> load_cms_task
