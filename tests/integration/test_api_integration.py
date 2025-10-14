import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json

# Import your apps
from preprocess_api.app import app as preprocess_app
from gate_api.app import app as gate_app
from train_api.app import app as train_app

class TestAPIIntegration:
    
    @pytest.fixture
    def preprocess_client(self):
        return TestClient(preprocess_app)
    
    @pytest.fixture
    def gate_client(self):
        return TestClient(gate_app)
    
    @pytest.fixture
    def train_client(self):
        return TestClient(train_app)
    
    def test_gate_api_login(self, gate_client):
        """Test login endpoint"""
        response = gate_client.post("/login", json={
            "username": "admin",
            "password": "admin_pass"
        })
        
        assert response.status_code == 200
        assert "token" in response.json()
        assert response.json()["role"] == "admin"
    
    def test_gate_api_invalid_login(self, gate_client):
        """Test invalid login"""
        response = gate_client.post("/login", json={
            "username": "invalid",
            "password": "wrong_password"
        })
        
        assert response.status_code == 401
    
    @patch('preprocess_api.app.resnet_model')
    @patch('preprocess_api.app.nlp')
    def test_preprocess_api_endpoints(self, mock_nlp, mock_resnet_model, preprocess_client, mock_jwt_token, sample_image_data):
        """Test preprocess API endpoints with mocked models"""
        # Mock model responses
        mock_resnet_model.predict.return_value = np.random.rand(1, 1, 2048)
        mock_doc = MagicMock()
        mock_doc.text = "test description"
        mock_nlp.pipe.return_value = [mock_doc]
        
        headers = {"Authorization": f"Bearer {mock_jwt_token}"}
        
        # Test image feature extraction
        response = preprocess_client.post(
            "/extract-image-features",
            json={"image_data": sample_image_data},
            headers=headers
        )
        
        assert response.status_code == 200
        assert "image_features" in response.json()
        
        # Test text feature extraction
        response = preprocess_client.post(
            "/extract-text-features",
            json={"descriptions": ["test description"]},
            headers=headers
        )
        
        assert response.status_code == 200
        assert "text_features" in response.json()
    
    @patch('train_api.app.requests.post')
    @patch('train_api.app.mlflow')
    def test_train_api_integration(self, mock_mlflow, mock_requests, train_client, mock_jwt_token):
        """Test train API integration with mocked dependencies"""
        # Mock preprocess-api response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "features": {"/path/to/image.jpg": [0.1, 0.2, 0.3]}
        }
        mock_requests.return_value = mock_response
        
        # Mock MLflow
        mock_mlflow_client = MagicMock()
        mock_mlflow.tracking.MlflowClient.return_value = mock_mlflow_client
        
        headers = {"Authorization": f"Bearer {mock_jwt_token}"}
        
        # This will test the full integration flow
        response = train_client.post(
            "/train",
            json={
                "test_size": 0.2,
                "epochs": 2,
                "batch_size": 32
            },
            headers=headers
        )
        
        # Should reach the MLflow logging part
        assert mock_requests.called  # Should have called preprocess-api
        assert mock_mlflow_client.log_metric.called  # Should have logged metrics