import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

class TestWorkflow:
    
    @patch('train_api.app.pd.read_csv')
    @patch('train_api.app.requests.post')
    @patch('train_api.app.mlflow')
    def test_full_training_workflow(self, mock_mlflow, mock_requests, mock_read_csv):
        """Test the complete training workflow with mocked dependencies"""
        # Mock data loading
        mock_X_data = pd.DataFrame({
            'Unnamed: 0': [1, 2, 3],
            'designation': ['bag', 'shoes', 'jacket'],
            'description': ['leather bag', 'running shoes', 'winter jacket'],
            'imageid': [1, 2, 3],
            'productid': [100, 200, 300]
        })
        
        mock_Y_data = pd.DataFrame({
            'Unnamed: 0': [1, 2, 3],
            'prdtypecode': [40, 60, 1140]
        })
        
        mock_read_csv.side_effect = [mock_X_data, mock_Y_data]
        
        # Mock preprocess-api responses
        mock_image_response = MagicMock()
        mock_image_response.status_code = 200
        mock_image_response.json.return_value = {
            "features": {
                "/app/data/images/image_train/image_1_product_100.jpg": list(np.random.rand(2048)),
                "/app/data/images/image_train/image_2_product_200.jpg": list(np.random.rand(2048)),
                "/app/data/images/image_train/image_3_product_300.jpg": list(np.random.rand(2048))
            }
        }
        
        mock_text_response = MagicMock()
        mock_text_response.status_code = 200
        mock_text_response.json.return_value = {
            "text_features": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            "processed_descriptions": ["leather bag", "running shoes", "winter jacket"]
        }
        
        mock_requests.side_effect = [mock_image_response, mock_text_response]
        
        # Mock MLflow
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=None)
        
        # Import and test the training function
        from train_api.app import train_model
        
        # Mock request and authorization
        mock_request = MagicMock()
        mock_request.test_size = 0.2
        mock_request.random_state = 42
        mock_request.epochs = 2
        mock_request.batch_size = 32
        mock_request.model_name = "test_model"
        mock_request.experiment_name = "test_experiment"
        
        mock_authorization = "Bearer mock_token"
        
        # This should execute the full workflow without errors
        try:
            result = train_model(mock_request, mock_authorization)
            assert "accuracy" in result
            assert result["accuracy"] >= 0  # Should be a valid accuracy
        except Exception as e:
            pytest.fail(f"Workflow failed with error: {e}")
    
    @patch('predict_api.app.mlflow.pyfunc.load_model')
    @patch('predict_api.app.requests.post')
    def test_prediction_workflow(self, mock_requests, mock_load_model):
        """Test the prediction workflow"""
        # Mock model loading
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.1, 0.8, 0.1]])  # Mock predictions
        mock_load_model.return_value = mock_model
        
        # Mock preprocess-api response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "image_features": list(np.random.rand(2048))
        }
        mock_requests.return_value = mock_response
        
        # Import and test prediction
        from predict_api.app import predict_image
        
        mock_request = MagicMock()
        mock_request.image_data = "base64_test_image_data"
        
        mock_authorization = "Bearer mock_token"
        
        result = predict_image(mock_request, mock_authorization)
        
        assert "pred_class" in result
        assert "label" in result
        assert "probs" in result
        assert result["input_mode"] == "image_only"