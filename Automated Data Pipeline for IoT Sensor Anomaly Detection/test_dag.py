from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
from datetime import timedelta

default_args = {
    'owner': 'airflow',
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1)
}

dag = DAG(
    dag_id = 'test_dag', 
    tags = ['test'],
    description = 'Simple test DAG',
    start_date = datetime(2021, 6, 25),
    end_date = datetime(2021, 6, 26),
    schedule_interval = '*/5 * * * *',
    catchup = False,
    default_args = default_args)

t1 = BashOperator(
    task_id='task_1',
    bash_command='echo this is task 1',
    dag=dag)

t2 = BashOperator(
    task_id='task_2',
    bash_command='echo this is task 2',
    dag=dag)

t3 = BashOperator(
    task_id='run_model',
    bash_command='python3 ~/airflow/model_test_postgres.py',
    dag=dag)

t1 >> t2 >> t3
