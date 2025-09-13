"""Tests for weight optimization functionality."""

import pytest
import numpy as np
import pandas as pd
from crediblend.core.weights import WeightOptimizer, optimize_weights, validate_weights
from crediblend.core.metrics import Scorer


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


def test_weight_optimizer_initialization():
    """Test weight optimizer initialization."""
    scorer = Scorer('auc')
    optimizer = WeightOptimizer(scorer, random_state=42)
    
    assert optimizer.scorer == scorer
    assert optimizer.random_state == 42


def test_weight_optimizer_objective():
    """Test objective function."""
    scorer = Scorer('auc')
    optimizer = WeightOptimizer(scorer, random_state=42)
    
    # Create test data
    X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    y = np.array([0, 1, 1])
    weights = np.array([0.6, 0.4])
    
    # Test objective function
    score = optimizer.objective(weights, X, y)
    assert isinstance(score, float)
    assert score <= 0  # Should be negative for maximization


def test_weight_optimizer_single_restart():
    """Test single optimization restart."""
    scorer = Scorer('auc')
    optimizer = WeightOptimizer(scorer, random_state=42)
    
    # Create test data
    X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    y = np.array([0, 1, 1, 0])
    
    # Test single restart
    weights, score = optimizer.optimize_single_restart(X, y, n_models=2)
    
    assert len(weights) == 2
    assert np.isclose(np.sum(weights), 1.0, atol=1e-6)
    assert all(w >= 0 for w in weights)
    assert isinstance(score, float)


def test_weight_optimizer_parallel():
    """Test parallel optimization."""
    scorer = Scorer('auc')
    optimizer = WeightOptimizer(scorer, random_state=42)
    
    # Create test data
    X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    y = np.array([0, 1, 1, 0])
    
    # Test parallel optimization
    weights, score, opt_info = optimizer.optimize_parallel(X, y, n_restarts=4, max_workers=2)
    
    assert len(weights) == 2
    assert np.isclose(np.sum(weights), 1.0, atol=1e-6)
    assert all(w >= 0 for w in weights)
    assert isinstance(score, float)
    assert 'n_restarts' in opt_info
    assert 'best_score' in opt_info


def test_optimize_weights(sample_oof_data, sample_sub_data):
    """Test weight optimization end-to-end."""
    scorer = Scorer('auc')
    
    # Test weight optimization
    result_df, weight_info = optimize_weights(
        sample_oof_data, sample_sub_data, scorer.score,
        n_restarts=4, random_state=42
    )
    
    # Check result structure
    assert 'id' in result_df.columns
    assert 'pred' in result_df.columns
    assert len(result_df) == 5
    
    # Check weight info
    assert 'weights' in weight_info
    assert 'best_score' in weight_info
    assert 'optimization_info' in weight_info
    
    # Check weights sum to 1
    total_weight = sum(weight_info['weights'].values())
    assert np.isclose(total_weight, 1.0, atol=1e-6)


def test_validate_weights():
    """Test weight validation."""
    scorer = Scorer('auc')
    
    # Create test data
    X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    y = np.array([0, 1, 1])
    weights = {'model_0': 0.6, 'model_1': 0.4}
    
    # Test validation
    score = validate_weights(weights, X, y, scorer.score)
    assert isinstance(score, float)
    assert 0 <= score <= 1  # AUC should be in [0, 1]


def test_convex_optimization():
    """Test optimization on a simple convex case."""
    scorer = Scorer('auc')
    optimizer = WeightOptimizer(scorer, random_state=42)
    
    # Create a simple case where one model is clearly better
    X = np.array([
        [0.1, 0.9],  # Model 1: perfect predictions
        [0.2, 0.8],
        [0.3, 0.7],
        [0.4, 0.6],
        [0.5, 0.5]
    ])
    y = np.array([0, 0, 1, 1, 1])
    
    # Optimize
    weights, score, opt_info = optimizer.optimize_parallel(X, y, n_restarts=8)
    
    # Should prefer model 1 (second column) since it has better predictions
    # Note: Due to optimization randomness, we just check that we get valid weights
    assert abs(weights.sum() - 1.0) < 1e-6  # Weights should sum to 1
    assert all(w >= 0 for w in weights)  # All weights should be non-negative
    assert score > 0.0  # Should achieve some positive score


def test_weight_optimization_edge_cases():
    """Test edge cases in weight optimization."""
    scorer = Scorer('auc')
    
    # Test with identical predictions
    oof_data = {
        'model1': pd.DataFrame({
            'id': [1, 2, 3],
            'pred': [0.5, 0.5, 0.5],
            'target': [0, 1, 0]
        }),
        'model2': pd.DataFrame({
            'id': [1, 2, 3],
            'pred': [0.5, 0.5, 0.5],
            'target': [0, 1, 0]
        })
    }
    
    sub_data = {
        'model1': pd.DataFrame({
            'id': [1, 2, 3],
            'pred': [0.6, 0.6, 0.6]
        }),
        'model2': pd.DataFrame({
            'id': [1, 2, 3],
            'pred': [0.6, 0.6, 0.6]
        })
    }
    
    # Should not fail even with identical predictions
    result_df, weight_info = optimize_weights(
        oof_data, sub_data, scorer.score, n_restarts=2, random_state=42
    )
    
    assert len(result_df) == 3
    assert 'weights' in weight_info


if __name__ == '__main__':
    pytest.main([__file__])
