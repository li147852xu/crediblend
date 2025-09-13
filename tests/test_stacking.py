"""Tests for stacking functionality."""

import pytest
import numpy as np
import pandas as pd
from crediblend.core.stacking import StackingBlender, stacking_blend, prepare_stacking_data, cross_validate_stacking


@pytest.fixture
def sample_oof_data():
    """Create sample OOF data for testing."""
    return {
        'oof_modelA': pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'pred': [0.65, 0.32, 0.78, 0.45, 0.89],
            'target': [1, 0, 1, 0, 1]
        }),
        'oof_modelB': pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'pred': [0.62, 0.35, 0.75, 0.48, 0.91],
            'target': [1, 0, 1, 0, 1]
        }),
        'oof_modelC': pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'pred': [0.68, 0.29, 0.81, 0.42, 0.87],
            'target': [1, 0, 1, 0, 1]
        })
    }


@pytest.fixture
def sample_sub_data():
    """Create sample submission data for testing."""
    return {
        'oof_modelA': pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'pred': [0.68, 0.29, 0.81, 0.42, 0.87]
        }),
        'oof_modelB': pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'pred': [0.61, 0.36, 0.77, 0.49, 0.92]
        }),
        'oof_modelC': pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'pred': [0.71, 0.25, 0.85, 0.38, 0.89]
        })
    }


def test_stacking_blender_initialization():
    """Test stacking blender initialization."""
    # Test LogisticRegression
    blender_lr = StackingBlender('lr', random_state=42)
    assert blender_lr.meta_learner_type == 'lr'
    assert blender_lr.random_state == 42
    assert not blender_lr.is_fitted
    
    # Test Ridge
    blender_ridge = StackingBlender('ridge', random_state=42)
    assert blender_ridge.meta_learner_type == 'ridge'
    assert blender_ridge.random_state == 42
    assert not blender_ridge.is_fitted
    
    # Test invalid meta-learner
    with pytest.raises(ValueError):
        StackingBlender('invalid')


def test_stacking_blender_fit_predict():
    """Test stacking blender fit and predict."""
    blender = StackingBlender('lr', random_state=42)
    
    # Create test data
    X_oof = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    y_oof = np.array([0, 1, 1, 0])
    X_sub = np.array([[0.2, 0.3], [0.4, 0.5]])
    
    # Fit
    blender.fit(X_oof, y_oof)
    assert blender.is_fitted
    
    # Predict
    predictions = blender.predict(X_sub)
    assert len(predictions) == 2
    assert all(0 <= p <= 1 for p in predictions)  # Should be probabilities


def test_stacking_blender_ridge():
    """Test Ridge stacking blender."""
    blender = StackingBlender('ridge', random_state=42)
    
    # Create test data
    X_oof = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    y_oof = np.array([0, 1, 1, 0])
    X_sub = np.array([[0.2, 0.3], [0.4, 0.5]])
    
    # Fit and predict
    blender.fit(X_oof, y_oof)
    predictions = blender.predict(X_sub)
    
    assert len(predictions) == 2
    # Ridge can produce values outside [0, 1], so we don't check bounds


def test_stacking_blender_coefficients():
    """Test getting stacking coefficients."""
    blender = StackingBlender('lr', random_state=42)
    
    # Create test data
    X_oof = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    y_oof = np.array([0, 1, 1, 0])
    
    # Fit
    blender.fit(X_oof, y_oof)
    
    # Get coefficients
    coefs = blender.get_coefficients()
    assert isinstance(coefs, dict)
    assert len(coefs) == 2  # Two models
    assert all(isinstance(v, float) for v in coefs.values())


def test_prepare_stacking_data(sample_oof_data, sample_sub_data):
    """Test data preparation for stacking."""
    X_oof, y_oof, X_sub, model_names = prepare_stacking_data(
        sample_oof_data, sample_sub_data
    )
    
    # Check shapes
    assert X_oof.shape[0] == 5  # 5 samples
    assert X_oof.shape[1] == 3  # 3 models
    assert X_sub.shape[0] == 5  # 5 samples
    assert X_sub.shape[1] == 3  # 3 models
    assert len(y_oof) == 5
    assert len(model_names) == 3
    
    # Check model names
    expected_models = ['oof_modelA', 'oof_modelB', 'oof_modelC']
    assert all(model in expected_models for model in model_names)


def test_prepare_stacking_data_mismatch():
    """Test data preparation with mismatched models."""
    oof_data = {
        'model1': pd.DataFrame({
            'id': [1, 2, 3],
            'pred': [0.1, 0.2, 0.3],
            'target': [0, 1, 0]
        })
    }
    
    sub_data = {
        'model2': pd.DataFrame({
            'id': [1, 2, 3],
            'pred': [0.4, 0.5, 0.6]
        })
    }
    
    with pytest.raises(ValueError):
        prepare_stacking_data(oof_data, sub_data)


def test_stacking_blend(sample_oof_data, sample_sub_data):
    """Test end-to-end stacking blend."""
    result_df, stacking_info = stacking_blend(
        sample_oof_data, sample_sub_data, 
        meta_learner='lr', random_state=42
    )
    
    # Check result structure
    assert 'id' in result_df.columns
    assert 'pred' in result_df.columns
    assert len(result_df) == 5
    
    # Check stacking info
    assert 'meta_learner' in stacking_info
    assert 'model_names' in stacking_info
    assert 'coefficients' in stacking_info
    assert 'n_models' in stacking_info
    
    # Check predictions are probabilities
    assert all(0 <= p <= 1 for p in result_df['pred'])


def test_stacking_blend_ridge(sample_oof_data, sample_sub_data):
    """Test Ridge stacking blend."""
    result_df, stacking_info = stacking_blend(
        sample_oof_data, sample_sub_data, 
        meta_learner='ridge', random_state=42
    )
    
    # Check result structure
    assert 'id' in result_df.columns
    assert 'pred' in result_df.columns
    assert len(result_df) == 5
    assert stacking_info['meta_learner'] == 'ridge'


def test_cross_validate_stacking():
    """Test cross-validation for stacking."""
    # Create test data
    X_oof = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7]
    ])
    y_oof = np.array([0, 1, 1, 0, 1])
    
    # Test cross-validation
    mean_score, cv_scores = cross_validate_stacking(
        X_oof, y_oof, meta_learner='lr', cv=3, random_state=42
    )
    
    assert isinstance(mean_score, float)
    assert len(cv_scores) == 3
    assert all(isinstance(s, float) for s in cv_scores)


def test_stacking_shape_consistency():
    """Test shape consistency in stacking operations."""
    blender = StackingBlender('lr', random_state=42)
    
    # Create test data with different shapes
    X_oof = np.random.rand(10, 3)
    y_oof = np.random.randint(0, 2, 10)
    X_sub = np.random.rand(5, 3)
    
    # Fit
    blender.fit(X_oof, y_oof)
    
    # Predict
    predictions = blender.predict(X_sub)
    
    # Check shapes
    assert predictions.shape == (5,)  # Should match X_sub rows
    assert len(blender.get_coefficients()) == 3  # Should match number of models


def test_stacking_with_missing_target():
    """Test stacking with missing target column."""
    oof_data = {
        'model1': pd.DataFrame({
            'id': [1, 2, 3],
            'pred': [0.1, 0.2, 0.3]
            # Missing target column
        }),
        'model2': pd.DataFrame({
            'id': [1, 2, 3],
            'pred': [0.4, 0.5, 0.6],
            'target': [0, 1, 0]
        })
    }
    
    sub_data = {
        'model1': pd.DataFrame({
            'id': [1, 2, 3],
            'pred': [0.7, 0.8, 0.9]
        }),
        'model2': pd.DataFrame({
            'id': [1, 2, 3],
            'pred': [0.1, 0.2, 0.3]
        })
    }
    
    # Should raise error when not enough valid OOF data
    with pytest.raises(ValueError, match="Not enough valid OOF data for stacking"):
        result_df, stacking_info = stacking_blend(
            oof_data, sub_data, meta_learner='lr', random_state=42
        )


if __name__ == '__main__':
    pytest.main([__file__])
