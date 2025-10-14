from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.http.operators.http import SimpleHttpOperator
from datetime import datetime
import requests
import json

def get_auth_token(**context):
    """Get JWT token from gate-api"""
    try:
        response = requests.post(
            "http://gate-api:5000/login",
            json={"username": "user", "password": "user_pass"},
            timeout=30
        )
        if response.status_code == 200:
            token = response.json()["token"]
            context['ti'].xcom_push(key='auth_token', value=token)
            print(f"Successfully obtained token: {token[:20]}...")
            return token
        else:
            raise Exception(f"Authentication failed: {response.text}")
    except Exception as e:
        raise Exception(f"Failed to get auth token: {e}")

default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
}

with DAG(
    dag_id="test_auth_dag",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag:

    get_token = PythonOperator(
        task_id="get_auth_token",
        python_callable=get_auth_token,
        provide_context=True,
    )

    test_preprocess = SimpleHttpOperator(
        task_id="test_preprocess",
        http_conn_id="preprocess_api",
        endpoint="/health",
        method="GET",
        headers={
            "Authorization": "Bearer {{ ti.xcom_pull(task_ids='get_auth_token', key='auth_token') }}"
        },
        response_check=lambda response: response.status_code == 200,
    )

    get_token >> test_preprocess