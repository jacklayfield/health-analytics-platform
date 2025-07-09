# OpenFDA dag

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from airflow.etl.openfda.extract import download_openfda_data
from airflow.etl.openfda.transform import transform_openfda_data
from airflow.etl.openfda.load import load_to_snowflake

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
}

with DAG('openfda_etl', default_args=default_args, schedule_interval='@monthly', catchup=False) as dag:

    extract = PythonOperator(
        task_id='extract_openfda',
        python_callable=download_openfda_data,
    )

    transform = PythonOperator(
        task_id='transform_openfda',
        python_callable=transform_openfda_data,
    )

    load = PythonOperator(
        task_id='load_openfda',
        python_callable=load_to_snowflake,
    )

    extract >> transform >> load
