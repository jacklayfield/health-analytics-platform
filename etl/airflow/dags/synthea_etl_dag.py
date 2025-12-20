# Synthea DAG

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys

sys.path.append('/opt/airflow')

from etl.airflow.etl.synthea.extract.synthea import extract_synthea
from etl.airflow.etl.synthea.transform.patients import transform_patients
from etl.airflow.etl.synthea.load.postgres import load_patients

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
}

with DAG('synthea_etl', default_args=default_args, schedule_interval='@monthly', catchup=False) as dag:

    extract_synthea_task = PythonOperator(
        task_id='extract_synthea',
        python_callable=extract_synthea,
    )

    transform_patients_task = PythonOperator(
        task_id='transform_patients',
        python_callable=transform_patients,
    )

    load_patients_task = PythonOperator(
        task_id='load_patients',
        python_callable=load_patients,
    )

    extract_synthea_task >> transform_patients_task >> load_patients_task