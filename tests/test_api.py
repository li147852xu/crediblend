"""Tests for CrediBlend Python API."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from crediblend.api import (
    fit_blend, predict_blend, search_weights, quick_blend,
    BlendConfig, BlendModel, BlendResult, load_model, save_model
)


class TestBlendConfig:
    """Test BlendConfig validation."""
    
    def test_valid_config(self):
        """Test valid configuration."""
        config = BlendConfig(method="mean", metric="auc")
        assert config.method == "mean"
        assert config.metric == "auc"
        assert config.target_col == "target"
        assert config.decorrelate is False
    
    def test_invalid_method(self):
        """Test invalid method raises error."""
        with pytest.raises(ValueError, match="Invalid method"):
            BlendConfig(method="invalid_method")
    
    def test_invalid_metric(self):
        """Test invalid metric raises error."""
        with pytest.raises(ValueError, match="Invalid metric"):
            BlendConfig(metric="invalid_metric")


class TestFitBlend:
    """Test fit_blend function."""
    
    @pytest.fixture
    def sample_oof_data(self):
        """Create sample OOF data."""
        return [
            pd.DataFrame({
                'id': [1, 2, 3, 4, 5],
                'pred': [0.1, 0.2, 0.3, 0.4, 0.5],
                'target': [0, 1, 0, 1, 0],
                'fold': [0, 0, 1, 1, 1]
            }),
            pd.DataFrame({
                'id': [1, 2, 3, 4, 5],
                'pred': [0.2, 0.3, 0.4, 0.5, 0.6],
                'target': [0, 1, 0, 1, 0],
                'fold': [0, 0, 1, 1, 1]
            })
        ]
    
    def test_fit_blend_mean(self, sample_oof_data):
        """Test fitting mean blending model."""
        model = fit_blend(sample_oof_data, method="mean")
        
        assert isinstance(model, BlendModel)
        assert model.method == "mean"
        assert model.config.method == "mean"
        assert len(model.oof_metrics) == 2
        assert "model_0" in model.oof_metrics
        assert "model_1" in model.oof_metrics
    
    def test_fit_blend_with_config(self, sample_oof_data):
        """Test fitting with custom config."""
        config = BlendConfig(method="rank_mean", metric="mse", random_state=42)
        model = fit_blend(sample_oof_data, config=config)
        
        assert model.config.method == "rank_mean"
        assert model.config.metric == "mse"
        assert model.config.random_state == 42
    
    def test_fit_blend_weighted(self, sample_oof_data):
        """Test fitting weighted blending model."""
        model = fit_blend(sample_oof_data, method="weighted", random_state=42)
        
        assert model.method == "weighted"
        # Should have weights after optimization
        assert model.weights is not None
        assert len(model.weights) == 2
    
    def test_fit_blend_decorrelation(self, sample_oof_data):
        """Test fitting with decorrelation."""
        model = fit_blend(sample_oof_data, method="mean", decorrelate=True, random_state=42)
        
        assert model.config.decorrelate is True
        # Should have decorrelation info
        assert model.decorrelation_info is not None
    
    def test_fit_blend_empty_data(self):
        """Test fitting with empty data raises error."""
        with pytest.raises(ValueError, match="At least one OOF frame is required"):
            fit_blend([])
    
    def test_fit_blend_missing_columns(self):
        """Test fitting with missing columns raises error."""
        invalid_data = [pd.DataFrame({'id': [1, 2], 'pred': [0.1, 0.2]})]  # Missing target
        
        with pytest.raises(ValueError, match="missing target column"):
            fit_blend(invalid_data)


class TestPredictBlend:
    """Test predict_blend function."""
    
    @pytest.fixture
    def sample_model(self):
        """Create sample trained model."""
        oof_data = [
            pd.DataFrame({
                'id': [1, 2, 3, 4, 5],
                'pred': [0.1, 0.2, 0.3, 0.4, 0.5],
                'target': [0, 1, 0, 1, 0],
                'fold': [0, 0, 1, 1, 1]
            })
        ]
        return fit_blend(oof_data, method="mean")
    
    @pytest.fixture
    def sample_sub_data(self):
        """Create sample submission data."""
        return [
            pd.DataFrame({
                'id': [1, 2, 3, 4, 5],
                'pred': [0.15, 0.25, 0.35, 0.45, 0.55]
            })
        ]
    
    def test_predict_blend_basic(self, sample_model, sample_sub_data):
        """Test basic prediction."""
        result = predict_blend(sample_model, sample_sub_data)
        
        assert isinstance(result, BlendResult)
        assert isinstance(result.predictions, pd.DataFrame)
        assert 'id' in result.predictions.columns
        assert 'pred' in result.predictions.columns
        assert len(result.predictions) == 5
        assert result.model == sample_model
    
    def test_predict_blend_weighted(self, sample_sub_data):
        """Test prediction with weighted model."""
        oof_data = [
            pd.DataFrame({
                'id': [1, 2, 3, 4, 5],
                'pred': [0.1, 0.2, 0.3, 0.4, 0.5],
                'target': [0, 1, 0, 1, 0],
                'fold': [0, 0, 1, 1, 1]
            }),
            pd.DataFrame({
                'id': [1, 2, 3, 4, 5],
                'pred': [0.2, 0.3, 0.4, 0.5, 0.6],
                'target': [0, 1, 0, 1, 0],
                'fold': [0, 0, 1, 1, 1]
            })
        ]
        
        model = fit_blend(oof_data, method="weighted", random_state=42)
        result = predict_blend(model, sample_sub_data)
        
        assert result.model.method == "weighted"
        assert result.model.weights is not None
    
    def test_predict_blend_empty_data(self, sample_model):
        """Test prediction with empty data raises error."""
        with pytest.raises(ValueError, match="At least one submission frame is required"):
            predict_blend(sample_model, [])
    
    def test_predict_blend_missing_columns(self, sample_model):
        """Test prediction with missing columns raises error."""
        invalid_data = [pd.DataFrame({'id': [1, 2]})]  # Missing pred
        
        with pytest.raises(ValueError, match="missing required columns"):
            predict_blend(sample_model, invalid_data)


class TestSearchWeights:
    """Test search_weights function."""
    
    @pytest.fixture
    def sample_oof_data(self):
        """Create sample OOF data for weight search."""
        return [
            pd.DataFrame({
                'id': [1, 2, 3, 4, 5],
                'pred': [0.1, 0.2, 0.3, 0.4, 0.5],
                'target': [0, 1, 0, 1, 0],
                'fold': [0, 0, 1, 1, 1]
            }),
            pd.DataFrame({
                'id': [1, 2, 3, 4, 5],
                'pred': [0.2, 0.3, 0.4, 0.5, 0.6],
                'target': [0, 1, 0, 1, 0],
                'fold': [0, 0, 1, 1, 1]
            })
        ]
    
    def test_search_weights_basic(self, sample_oof_data):
        """Test basic weight search."""
        weights, score, info = search_weights(sample_oof_data, random_state=42)
        
        assert isinstance(weights, dict)
        assert len(weights) == 2
        assert 'model_0' in weights
        assert 'model_1' in weights
        assert abs(sum(weights.values()) - 1.0) < 1e-6  # Should sum to 1
        assert isinstance(score, float)
        assert isinstance(info, dict)
    
    def test_search_weights_insufficient_data(self):
        """Test weight search with insufficient data raises error."""
        single_model = [pd.DataFrame({
            'id': [1, 2, 3],
            'pred': [0.1, 0.2, 0.3],
            'target': [0, 1, 0]
        })]
        
        with pytest.raises(ValueError, match="At least 2 OOF frames required"):
            search_weights(single_model)


class TestQuickBlend:
    """Test quick_blend convenience function."""
    
    def test_quick_blend(self):
        """Test quick blending."""
        oof_data = [
            pd.DataFrame({
                'id': [1, 2, 3, 4, 5],
                'pred': [0.1, 0.2, 0.3, 0.4, 0.5],
                'target': [0, 1, 0, 1, 0],
                'fold': [0, 0, 1, 1, 1]
            })
        ]
        
        sub_data = [
            pd.DataFrame({
                'id': [1, 2, 3, 4, 5],
                'pred': [0.15, 0.25, 0.35, 0.45, 0.55]
            })
        ]
        
        result = quick_blend(oof_data, sub_data, method="mean")
        
        assert isinstance(result, BlendResult)
        assert isinstance(result.predictions, pd.DataFrame)
        assert len(result.predictions) == 5


class TestModelSerialization:
    """Test model save/load functionality."""
    
    def test_save_load_model(self):
        """Test saving and loading a model."""
        oof_data = [
            pd.DataFrame({
                'id': [1, 2, 3, 4, 5],
                'pred': [0.1, 0.2, 0.3, 0.4, 0.5],
                'target': [0, 1, 0, 1, 0],
                'fold': [0, 0, 1, 1, 1]
            })
        ]
        
        model = fit_blend(oof_data, method="mean", random_state=42)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save model
            save_model(model, temp_path)
            assert os.path.exists(temp_path)
            
            # Load model
            loaded_model = load_model(temp_path)
            
            assert isinstance(loaded_model, BlendModel)
            assert loaded_model.method == model.method
            assert loaded_model.config.method == model.config.method
            assert loaded_model.config.random_state == model.config.random_state
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestBlendResult:
    """Test BlendResult class."""
    
    def test_blend_result_creation(self):
        """Test creating BlendResult."""
        config = BlendConfig(method="mean")
        model = BlendModel(method="mean", config=config)
        
        predictions = pd.DataFrame({
            'id': [1, 2, 3],
            'pred': [0.1, 0.2, 0.3]
        })
        
        result = BlendResult(
            predictions=predictions,
            model=model,
            improvement_over_best_single=0.05,
            warnings=["Test warning"]
        )
        
        assert isinstance(result.predictions, pd.DataFrame)
        assert result.model == model
        assert result.improvement_over_best_single == 0.05
        assert "Test warning" in result.warnings


if __name__ == '__main__':
    pytest.main([__file__])
