import pytest
import numpy as np
from unittest.mock import Mock, patch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

class TestModels:
    
    def test_count_vectorizer(self):
        """Test text vectorization"""
        vectorizer = CountVectorizer(max_features=10)
        texts = ["high quality leather", "mens athletic shoes", "leather handbag"]
        
        X = vectorizer.fit_transform(texts)
        
        assert X.shape[0] == 3  # 3 samples
        assert X.shape[1] <= 10  # Max 10 features
        assert "leather" in vectorizer.get_feature_names_out()
    
    def test_label_encoder(self):
        """Test label encoding"""
        encoder = LabelEncoder()
        labels = [40, 60, 1140, 40, 60]
        
        encoded = encoder.fit_transform(labels)
        
        assert len(encoded) == 5
        assert len(encoder.classes_) == 3  # 3 unique classes
        assert encoded[0] == encoded[3]  # Same class should have same encoding
    
    @patch('tensorflow.keras.models.Model')
    def test_model_compilation(self, mock_model):
        """Test model compilation setup"""
        from tensorflow.keras.layers import Input, Dense, Dropout
        from tensorflow.keras import Model
        
        # Create a simple test model
        input_layer = Input(shape=(100,))
        x = Dense(64, activation='relu')(input_layer)
        x = Dropout(0.5)(x)
        output_layer = Dense(10, activation='softmax')(x)
        model = Model(inputs=input_layer, outputs=output_layer)
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        assert model is not None
        assert len(model.layers) == 4  # Input, Dense, Dropout, Output
    
    def test_feature_concatenation(self):
        """Test feature concatenation for multimodal model"""
        # Simulate text and image features
        text_features = np.random.rand(10, 50)  # 10 samples, 50 features
        image_features = np.random.rand(10, 100)  # 10 samples, 100 features
        
        combined = np.hstack([text_features, image_features])
        
        assert combined.shape == (10, 150)  # Combined features
        assert np.array_equal(combined[:, :50], text_features)
        assert np.array_equal(combined[:, 50:], image_features)