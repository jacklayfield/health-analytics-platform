# Synthea DAG

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys

sys.path.append('/opt/airflow')

from etl.synthea.transform.patients import transform_patients
from etl.synthea.load.postgres import load_patients

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
}

with DAG('synthea_etl', default_args=default_args, schedule_interval='@monthly', catchup=False) as dag:

    transform_patients_task = PythonOperator(
        task_id='transform_patients',
        python_callable=transform_patients,
    )

    load_patients_task = PythonOperator(
        task_id='load_patients',
        python_callable=load_patients,
    )

    transform_patients_task >> load_patients_task