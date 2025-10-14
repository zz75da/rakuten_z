import pytest
import numpy as np
from unittest.mock import Mock, patch
from preprocess_api.app import preprocess_text, _process_single_image_path, load_models

class TestPreprocessing:
    
    def test_preprocess_text(self):
        """Test text preprocessing function"""
        # Mock spaCy model
        mock_nlp = Mock()
        mock_doc = Mock()
        mock_doc.text = "High quality leather handbag"
        
        mock_token1 = Mock()
        mock_token1.is_stop = False
        mock_token1.is_alpha = True
        mock_token1.lemma_ = "high"
        
        mock_token2 = Mock()
        mock_token2.is_stop = True  # Should be filtered out
        mock_token2.is_alpha = True
        mock_token2.lemma_ = "quality"
        
        mock_token3 = Mock()
        mock_token3.is_stop = False
        mock_token3.is_alpha = False  # Should be filtered out
        mock_token3.lemma_ = "leather123"
        
        mock_doc.__iter__ = Mock(return_value=iter([mock_token1, mock_token2, mock_token3]))
        mock_nlp.return_value = mock_doc
        
        with patch('preprocess_api.app.nlp', mock_nlp):
            result = preprocess_text("High quality leather handbag")
            assert result == "high"
    
    @patch('preprocess_api.app.resnet_model')
    @patch('preprocess_api.app.load_img')
    @patch('preprocess_api.app.img_to_array')
    def test_process_single_image_path_success(self, mock_img_to_array, mock_load_img, mock_resnet_model):
        """Test successful image processing"""
        # Mock dependencies
        mock_image = Mock()
        mock_load_img.return_value = mock_image
        mock_img_array = np.random.rand(224, 224, 3)
        mock_img_to_array.return_value = mock_img_array
        mock_resnet_model.predict.return_value = np.random.rand(1, 2048)
        
        result = _process_single_image_path("/fake/path/image.jpg")
        
        assert result is not None
        assert len(result) == 2048
        mock_load_img.assert_called_once()
        mock_resnet_model.predict.assert_called_once()
    
    @patch('preprocess_api.app.load_img')
    def test_process_single_image_path_missing_file(self, mock_load_img):
        """Test image processing with missing file"""
        mock_load_img.side_effect = FileNotFoundError("File not found")
        
        result = _process_single_image_path("/nonexistent/path/image.jpg")
        
        assert result is None
    
    @patch('preprocess_api.app.spacy.load')
    @patch('preprocess_api.app.ResNet50')
    def test_load_models(self, mock_resnet, mock_spacy_load):
        """Test model loading function"""
        # Mock model loading
        mock_nlp = Mock()
        mock_spacy_load.return_value = mock_nlp
        mock_resnet_model = Mock()
        mock_resnet.return_value = mock_resnet_model
        
        # Clear globals to test loading
        import preprocess_api.app as app_module
        app_module.nlp = None
        app_module.resnet_model = None
        app_module.text_vectorizer = None
        
        load_models()
        
        assert app_module.nlp is not None
        assert app_module.resnet_model is not None
        assert app_module.text_vectorizer is not None
        mock_spacy_load.assert_called_once()
        mock_resnet.assert_called_once()