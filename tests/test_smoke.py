"""Smoke tests for CrediBlend."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from crediblend.core.io import read_oof_files, read_sub_files, align_submission_ids
from crediblend.core.metrics import Scorer, compute_oof_metrics
from crediblend.core.blend import mean_blend, rank_mean_blend, logit_mean_blend
from crediblend.core.report import generate_report


@pytest.fixture
def sample_oof_data():
    """Create sample OOF data for testing."""
    return {
        'oof_modelA': pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'pred': [0.65, 0.32, 0.78, 0.45, 0.89],
            'target': [1, 0, 1, 0, 1],
            'fold': [0, 0, 1, 1, 1]
        }),
        'oof_modelB': pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'pred': [0.62, 0.35, 0.75, 0.48, 0.91],
            'target': [1, 0, 1, 0, 1],
            'fold': [0, 0, 1, 1, 1]
        })
    }


@pytest.fixture
def sample_sub_data():
    """Create sample submission data for testing."""
    return {
        'sub_modelA': pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'pred': [0.68, 0.29, 0.81, 0.42, 0.87]
        }),
        'sub_modelB': pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'pred': [0.61, 0.36, 0.77, 0.49, 0.92]
        })
    }


def test_scorer_auc():
    """Test AUC scorer."""
    scorer = Scorer('auc')
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0.1, 0.9, 0.2, 0.8, 0.7])
    
    score = scorer.score(y_true, y_pred)
    assert isinstance(score, float)
    assert 0 <= score <= 1


def test_scorer_mse():
    """Test MSE scorer."""
    scorer = Scorer('mse')
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
    
    score = scorer.score(y_true, y_pred)
    assert isinstance(score, float)
    assert score <= 0  # Negative MSE for maximization


def test_mean_blend(sample_sub_data):
    """Test mean blending."""
    result = mean_blend(sample_sub_data)
    
    assert 'id' in result.columns
    assert 'pred' in result.columns
    assert len(result) == 5
    
    # Check that predictions are means
    expected_mean = (sample_sub_data['sub_modelA']['pred'] + sample_sub_data['sub_modelB']['pred']) / 2
    np.testing.assert_array_almost_equal(result['pred'].values, expected_mean.values)


def test_rank_mean_blend(sample_sub_data):
    """Test rank-based mean blending."""
    result = rank_mean_blend(sample_sub_data)
    
    assert 'id' in result.columns
    assert 'pred' in result.columns
    assert len(result) == 5
    
    # Check that predictions are in [0, 1] range
    assert result['pred'].min() >= 0
    assert result['pred'].max() <= 1


def test_logit_mean_blend(sample_sub_data):
    """Test logit-space mean blending."""
    result = logit_mean_blend(sample_sub_data)
    
    assert 'id' in result.columns
    assert 'pred' in result.columns
    assert len(result) == 5
    
    # Check that predictions are in [0, 1] range
    assert result['pred'].min() >= 0
    assert result['pred'].max() <= 1


def test_compute_oof_metrics(sample_oof_data):
    """Test OOF metrics computation."""
    scorer = Scorer('auc')
    metrics = compute_oof_metrics(sample_oof_data, scorer)
    
    assert 'oof_modelA' in metrics
    assert 'oof_modelB' in metrics
    
    # Check that metrics are computed
    for model_name, model_metrics in metrics.items():
        assert 'overall_oof' in model_metrics
        assert 'mean_fold' in model_metrics
        assert isinstance(model_metrics['overall_oof'], (float, type(np.nan)))


def test_align_submission_ids():
    """Test submission ID alignment."""
    sub_data = {
        'model1': pd.DataFrame({'id': [1, 2, 3], 'pred': [0.1, 0.2, 0.3]}),
        'model2': pd.DataFrame({'id': [2, 3, 4], 'pred': [0.4, 0.5, 0.6]}),
    }
    
    aligned = align_submission_ids(sub_data)
    
    # Should only keep common IDs (2, 3)
    assert len(aligned['model1']) == 2
    assert len(aligned['model2']) == 2
    assert set(aligned['model1']['id']) == {2, 3}
    assert set(aligned['model2']['id']) == {2, 3}


def test_generate_report(sample_oof_data, sample_sub_data):
    """Test HTML report generation."""
    scorer = Scorer('auc')
    oof_metrics = compute_oof_metrics(sample_oof_data, scorer)
    
    methods_df = pd.DataFrame({
        'model': ['oof_modelA', 'oof_modelB'],
        'overall_oof': [0.8, 0.75],
        'mean_fold': [0.82, 0.77],
        'submission_rows': [5, 5]
    })
    
    blend_results = {
        'mean': mean_blend(sample_sub_data),
        'rank_mean': rank_mean_blend(sample_sub_data)
    }
    
    config = {
            'oof_dir': 'examples',
            'sub_dir': 'examples',
            'out_dir': 'runs/demo',
            'metric': 'auc',
            'timestamp': '2024-01-01 12:00:00',
            'decorrelate': False,
            'stacking': 'none',
            'search_params': None,
            'seed': None
        }
    
    html = generate_report(oof_metrics, methods_df, blend_results, config)
    
    assert isinstance(html, str)
    assert '<html' in html
    assert 'CrediBlend Report' in html
    assert 'oof_modelA' in html
    assert 'oof_modelB' in html


def test_integration_with_temp_files():
    """Test integration with temporary files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        oof_dir = Path(temp_dir) / "oof"
        sub_dir = Path(temp_dir) / "sub"
        out_dir = Path(temp_dir) / "out"
        
        oof_dir.mkdir()
        sub_dir.mkdir()
        
        # Create OOF files
        oof_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'pred': [0.65, 0.32, 0.78, 0.45, 0.89],
            'target': [1, 0, 1, 0, 1],
            'fold': [0, 0, 1, 1, 1]
        })
        oof_data.to_csv(oof_dir / "oof_modelA.csv", index=False)
        
        # Create submission files
        sub_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'pred': [0.68, 0.29, 0.81, 0.42, 0.87]
        })
        sub_data.to_csv(sub_dir / "sub_modelA.csv", index=False)
        
        # Test reading files
        oof_files = read_oof_files(str(oof_dir))
        sub_files = read_sub_files(str(sub_dir))
        
        assert len(oof_files) == 1
        assert len(sub_files) == 1
        assert 'oof_modelA' in oof_files
        assert 'sub_modelA' in sub_files


if __name__ == '__main__':
    pytest.main([__file__])
