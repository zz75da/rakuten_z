# airflow/plugins/jwt_http_operator.py
from airflow.providers.http.operators.http import SimpleHttpOperator
import jwt
from datetime import datetime, timedelta

class JWTHttpOperator(SimpleHttpOperator):
    def execute(self, context):
        # Generate token before each execution
        secret_key = "default_secret"
        payload = {
            'sub': 'airflow',
            'exp': datetime.utcnow() + timedelta(hours=1)
        }
        token = jwt.encode(payload, secret_key, algorithm='HS256')
        
        # Set headers
        self.headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        
        return super().execute(context)