"""Tests for stability analysis functionality."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from crediblend.core.stability import (
    parse_time_frequency, create_time_windows, compute_windowed_metrics,
    compute_stability_scores, detect_dominance_patterns, generate_stability_report
)
from crediblend.core.metrics import Scorer


@pytest.fixture
def sample_time_oof_data():
    """Create sample OOF data with timestamps for testing."""
    # Create synthetic time series data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    return {
        'oof_modelA': pd.DataFrame({
            'id': range(100),
            'pred': np.random.rand(100),
            'target': np.random.randint(0, 2, 100),
            'date': dates
        }),
        'oof_modelB': pd.DataFrame({
            'id': range(100),
            'pred': np.random.rand(100),
            'target': np.random.randint(0, 2, 100),
            'date': dates
        }),
        'oof_modelC': pd.DataFrame({
            'id': range(100),
            'pred': np.random.rand(100),
            'target': np.random.randint(0, 2, 100),
            'date': dates
        })
    }


def test_parse_time_frequency():
    """Test time frequency parsing."""
    assert parse_time_frequency('M') == 'M'
    assert parse_time_frequency('W') == 'W'
    assert parse_time_frequency('D') == 'D'
    assert parse_time_frequency('month') == 'M'
    assert parse_time_frequency('week') == 'W'
    assert parse_time_frequency('day') == 'D'
    
    with pytest.raises(ValueError):
        parse_time_frequency('invalid')


def test_create_time_windows():
    """Test time window creation."""
    # Create test data
    dates = pd.date_range('2023-01-01', periods=10, freq='D')
    df = pd.DataFrame({
        'id': range(10),
        'pred': np.random.rand(10),
        'target': np.random.randint(0, 2, 10),
        'date': dates
    })
    
    # Test daily windows
    df_windowed = create_time_windows(df, 'date', 'D')
    assert 'window' in df_windowed.columns
    assert len(df_windowed['window'].unique()) == 10  # 10 days
    
    # Test weekly windows
    df_windowed = create_time_windows(df, 'date', 'W')
    assert 'window' in df_windowed.columns
    assert len(df_windowed['window'].unique()) <= 3  # At most 3 weeks


def test_create_time_windows_invalid_column():
    """Test time window creation with invalid column."""
    df = pd.DataFrame({
        'id': range(10),
        'pred': np.random.rand(10),
        'target': np.random.randint(0, 2, 10)
    })
    
    with pytest.raises(ValueError):
        create_time_windows(df, 'nonexistent', 'D')


def test_compute_windowed_metrics(sample_time_oof_data):
    """Test windowed metrics computation."""
    scorer = Scorer('auc')
    
    window_metrics = compute_windowed_metrics(
        sample_time_oof_data, 'date', 'W', 'target', scorer.score
    )
    
    # Check structure
    assert 'model' in window_metrics.columns
    assert 'window' in window_metrics.columns
    assert 'auc' in window_metrics.columns
    assert 'n_samples' in window_metrics.columns
    
    # Check that we have data for all models
    assert len(window_metrics['model'].unique()) == 3
    
    # Check that AUC values are valid
    assert all(0 <= auc <= 1 for auc in window_metrics['auc'] if not pd.isna(auc))


def test_compute_windowed_metrics_missing_target():
    """Test windowed metrics with missing target column."""
    dates = pd.date_range('2023-01-01', periods=10, freq='D')
    oof_data = {
        'model1': pd.DataFrame({
            'id': range(10),
            'pred': np.random.rand(10),
            'date': dates
            # Missing target column
        })
    }
    
    scorer = Scorer('auc')
    window_metrics = compute_windowed_metrics(
        oof_data, 'date', 'W', 'target', scorer.score
    )
    
    # Should be empty due to missing target
    assert window_metrics.empty


def test_compute_stability_scores():
    """Test stability scores computation."""
    # Create sample window metrics
    window_metrics = pd.DataFrame({
        'model': ['model1', 'model1', 'model1', 'model2', 'model2', 'model2'],
        'window': ['2023-01', '2023-02', '2023-03', '2023-01', '2023-02', '2023-03'],
        'auc': [0.8, 0.85, 0.82, 0.9, 0.88, 0.91],
        'n_samples': [10, 10, 10, 10, 10, 10]
    })
    
    stability_scores = compute_stability_scores(window_metrics)
    
    # Check structure
    assert 'model1' in stability_scores
    assert 'model2' in stability_scores
    
    # Check that scores are computed
    for model, scores in stability_scores.items():
        assert 'mean_auc' in scores
        assert 'std_auc' in scores
        assert 'stability_score' in scores
        assert 'iqr_stability' in scores
        assert 'n_windows' in scores
        assert 'min_auc' in scores
        assert 'max_auc' in scores


def test_detect_dominance_patterns():
    """Test dominance pattern detection."""
    # Create sample window metrics with clear dominance
    window_metrics = pd.DataFrame({
        'model': ['model1', 'model1', 'model1', 'model2', 'model2', 'model2'],
        'window': ['2023-01', '2023-02', '2023-03', '2023-01', '2023-02', '2023-03'],
        'auc': [0.95, 0.96, 0.97, 0.6, 0.65, 0.62],  # model1 dominates
        'n_samples': [10, 10, 10, 10, 10, 10]
    })
    
    dominance_analysis = detect_dominance_patterns(window_metrics)
    
    # Check structure
    assert 'dominant_models' in dominance_analysis
    assert 'unstable_windows' in dominance_analysis
    assert 'leakage_candidates' in dominance_analysis
    
    # Check that model1 is detected as dominant
    assert 'model1' in dominance_analysis['dominant_models']
    assert dominance_analysis['dominant_models']['model1'] == 3  # All 3 windows


def test_detect_dominance_patterns_leakage():
    """Test leakage detection."""
    # Create sample window metrics with suspiciously high AUC
    window_metrics = pd.DataFrame({
        'model': ['model1', 'model1', 'model1', 'model2', 'model2', 'model2'],
        'window': ['2023-01', '2023-02', '2023-03', '2023-01', '2023-02', '2023-03'],
        'auc': [0.99, 0.98, 0.99, 0.7, 0.72, 0.71],  # model1 has very high AUC
        'n_samples': [10, 10, 10, 10, 10, 10]
    })
    
    dominance_analysis = detect_dominance_patterns(window_metrics)
    
    # Check that model1 is flagged as leakage candidate
    assert len(dominance_analysis['leakage_candidates']) > 0
    assert any(candidate['model'] == 'model1' for candidate in dominance_analysis['leakage_candidates'])


def test_generate_stability_report():
    """Test stability report generation."""
    # Create sample data
    window_metrics = pd.DataFrame({
        'model': ['model1', 'model1', 'model1', 'model2', 'model2', 'model2'],
        'window': ['2023-01', '2023-02', '2023-03', '2023-01', '2023-02', '2023-03'],
        'auc': [0.8, 0.85, 0.82, 0.9, 0.88, 0.91],
        'n_samples': [10, 10, 10, 10, 10, 10]
    })
    
    stability_scores = compute_stability_scores(window_metrics)
    dominance_analysis = detect_dominance_patterns(window_metrics)
    
    report = generate_stability_report(window_metrics, stability_scores, dominance_analysis)
    
    # Check structure
    assert 'summary' in report
    assert 'stability_scores' in report
    assert 'dominance_analysis' in report
    assert 'plots' in report
    assert 'warnings' in report
    
    # Check summary
    assert report['summary']['n_models'] == 2
    assert report['summary']['n_windows'] == 3
    assert report['summary']['total_samples'] == 60


def test_generate_stability_report_empty():
    """Test stability report generation with empty data."""
    window_metrics = pd.DataFrame()
    stability_scores = {}
    dominance_analysis = {'dominant_models': {}, 'unstable_windows': [], 'leakage_candidates': []}
    
    report = generate_stability_report(window_metrics, stability_scores, dominance_analysis)
    
    # Should still have structure
    assert 'summary' in report
    assert 'stability_scores' in report
    assert 'dominance_analysis' in report
    assert 'plots' in report
    assert 'warnings' in report


def test_synthetic_windows():
    """Test with synthetic window data."""
    # Create synthetic data with known patterns
    np.random.seed(42)
    
    # Model 1: stable performance
    model1_auc = np.random.normal(0.8, 0.02, 12)  # Low variance
    
    # Model 2: unstable performance
    model2_auc = np.random.normal(0.75, 0.1, 12)  # High variance
    
    # Model 3: suspiciously high performance (potential leakage)
    model3_auc = np.random.normal(0.98, 0.01, 12)  # Very high, low variance
    
    window_metrics = pd.DataFrame({
        'model': (['model1'] * 12 + ['model2'] * 12 + ['model3'] * 12),
        'window': (list(range(12)) * 3),
        'auc': list(model1_auc) + list(model2_auc) + list(model3_auc),
        'n_samples': [10] * 36
    })
    
    # Compute stability scores
    stability_scores = compute_stability_scores(window_metrics)
    
    # Model 1 should be most stable (lowest stability score)
    # Model 2 should be least stable (highest stability score)
    # Model 3 should be flagged for leakage
    
    assert stability_scores['model1']['stability_score'] < stability_scores['model2']['stability_score']
    
    # Check leakage detection
    dominance_analysis = detect_dominance_patterns(window_metrics)
    leakage_models = [candidate['model'] for candidate in dominance_analysis['leakage_candidates']]
    assert 'model3' in leakage_models


if __name__ == '__main__':
    pytest.main([__file__])
