# OpenFDA dag

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys

sys.path.append('/opt/airflow')

from etl.openfda.extract import download_openfda_data
from etl.openfda.transform import transform_openfda_data
from etl.openfda.load import load_openfda_data_to_postgres

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
        python_callable=load_openfda_data_to_postgres,
    )

    extract >> transform >> load
