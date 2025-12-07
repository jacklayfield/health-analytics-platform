# Synthea ETL DAG (Will need to reevaluate once Synthea ETL structure is finalized)
# # Synthea ETL DAG

# from airflow import DAG
# from airflow.operators.python import PythonOperator
# from datetime import datetime
# import sys

# sys.path.append('/opt/airflow')

# from etl.synthea.extract import download_synthea_data
# from etl.synthea.transform import transform_synthea_data
# from etl.synthea.load import load_synthea_data

# default_args = {
#     'owner': 'airflow',
#     'start_date': datetime(2024, 1, 1),
# }

# with DAG('synthea_etl', default_args=default_args, schedule_interval='@monthly', catchup=False) as dag:

#     extract = PythonOperator(
#         task_id='extract_synthea',
#         python_callable=download_synthea_data,
#     )

#     transform = PythonOperator(
#         task_id='transform_synthea',
#         python_callable=transform_synthea_data,
#     )

#     load = PythonOperator(
#         task_id='load_synthea',
#         python_callable=load_synthea_data,
#     )

#     extract >> transform >> load
# # Synthea dag (To be implemented)