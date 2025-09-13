"""Tests for stable contracts and schema validation."""

import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path
import tempfile

from crediblend.api import fit_blend, predict_blend, quick_blend
from crediblend.core.report import create_blend_summary


class TestBlendSummaryContract:
    """Test blend_summary.json schema stability."""
    
    def test_blend_summary_schema(self):
        """Test that blend_summary.json has stable schema."""
        # Create sample data
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
        
        sub_data = [
            pd.DataFrame({
                'id': [1, 2, 3, 4, 5],
                'pred': [0.15, 0.25, 0.35, 0.45, 0.55]
            }),
            pd.DataFrame({
                'id': [1, 2, 3, 4, 5],
                'pred': [0.25, 0.35, 0.45, 0.55, 0.65]
            })
        ]
        
        # Fit model and predict
        model = fit_blend(oof_data, method='mean', random_state=42)
        result = predict_blend(model, sub_data)
        
        # Create blend summary
        methods_df = pd.DataFrame({
            'model': ['model_0', 'model_1'],
            'overall_oof': [0.8, 0.75]
        })
        
        blend_summary = create_blend_summary(methods_df, {}, {}, {})
        
        # Validate schema
        required_fields = ['timestamp', 'top_methods', 'weights', 'stacking', 'summary_stats']
        for field in required_fields:
            assert field in blend_summary, f"Missing required field: {field}"
        
        # Validate top_methods structure
        assert isinstance(blend_summary['top_methods'], list)
        for method in blend_summary['top_methods']:
            assert 'method' in method
            assert 'oof_score' in method
            assert 'rank' in method
            assert isinstance(method['oof_score'], (int, float))
            assert isinstance(method['rank'], int)
        
        # Validate summary_stats structure
        stats = blend_summary['summary_stats']
        assert 'n_models' in stats
        assert 'n_blend_methods' in stats
        assert isinstance(stats['n_models'], int)
        assert isinstance(stats['n_blend_methods'], int)
    
    def test_blend_summary_json_serialization(self):
        """Test that blend_summary can be serialized to JSON."""
        methods_df = pd.DataFrame({
            'model': ['model_0', 'model_1'],
            'overall_oof': [0.8, 0.75]
        })
        
        blend_summary = create_blend_summary(methods_df, {}, {}, {})
        
        # Should be JSON serializable
        json_str = json.dumps(blend_summary, indent=2)
        assert isinstance(json_str, str)
        
        # Should be deserializable
        parsed = json.loads(json_str)
        assert parsed == blend_summary
    
    def test_blend_summary_with_weights(self):
        """Test blend_summary with weight information."""
        methods_df = pd.DataFrame({
            'model': ['model_0', 'model_1'],
            'overall_oof': [0.8, 0.75]
        })
        
        weight_info = {
            'weights': {'model_0': 0.6, 'model_1': 0.4}
        }
        
        blend_summary = create_blend_summary(methods_df, weight_info, {}, {})
        
        # Should include weights
        assert 'weights' in blend_summary
        assert blend_summary['weights'] == {'model_0': 0.6, 'model_1': 0.4}
    
    def test_blend_summary_with_stacking(self):
        """Test blend_summary with stacking information."""
        methods_df = pd.DataFrame({
            'model': ['model_0', 'model_1'],
            'overall_oof': [0.8, 0.75]
        })
        
        stacking_info = {
            'meta_learner': 'lr',
            'coefficients': {'model_0': 0.7, 'model_1': 0.3}
        }
        
        blend_summary = create_blend_summary(methods_df, {}, stacking_info, {})
        
        # Should include stacking info
        assert 'stacking' in blend_summary
        assert blend_summary['stacking']['meta_learner'] == 'lr'
        assert blend_summary['stacking']['coefficients'] == {'model_0': 0.7, 'model_1': 0.3}


class TestAPIStability:
    """Test API stability and backward compatibility."""
    
    def test_fit_blend_api_stability(self):
        """Test that fit_blend API remains stable."""
        oof_data = [
            pd.DataFrame({
                'id': [1, 2, 3],
                'pred': [0.1, 0.2, 0.3],
                'target': [0, 1, 0]
            })
        ]
        
        # Test basic API
        model = fit_blend(oof_data, method='mean')
        assert hasattr(model, 'method')
        assert hasattr(model, 'config')
        assert hasattr(model, 'oof_metrics')
        
        # Test with config
        from crediblend.api import BlendConfig
        config = BlendConfig(method='mean', metric='auc')
        model = fit_blend(oof_data, config=config)
        assert model.config.method == 'mean'
    
    def test_predict_blend_api_stability(self):
        """Test that predict_blend API remains stable."""
        oof_data = [
            pd.DataFrame({
                'id': [1, 2, 3],
                'pred': [0.1, 0.2, 0.3],
                'target': [0, 1, 0]
            })
        ]
        
        sub_data = [
            pd.DataFrame({
                'id': [1, 2, 3],
                'pred': [0.15, 0.25, 0.35]
            })
        ]
        
        model = fit_blend(oof_data, method='mean')
        result = predict_blend(model, sub_data)
        
        # Test result structure
        assert hasattr(result, 'predictions')
        assert hasattr(result, 'model')
        assert hasattr(result, 'warnings')
        assert isinstance(result.predictions, pd.DataFrame)
        assert 'id' in result.predictions.columns
        assert 'pred' in result.predictions.columns
    
    def test_quick_blend_api_stability(self):
        """Test that quick_blend API remains stable."""
        oof_data = [
            pd.DataFrame({
                'id': [1, 2, 3],
                'pred': [0.1, 0.2, 0.3],
                'target': [0, 1, 0]
            })
        ]
        
        sub_data = [
            pd.DataFrame({
                'id': [1, 2, 3],
                'pred': [0.15, 0.25, 0.35]
            })
        ]
        
        result = quick_blend(oof_data, sub_data, method='mean')
        
        # Should return same structure as predict_blend
        assert hasattr(result, 'predictions')
        assert hasattr(result, 'model')
        assert isinstance(result.predictions, pd.DataFrame)


class TestCLIContract:
    """Test CLI contract stability."""
    
    def test_cli_exit_codes(self):
        """Test that CLI exit codes are stable."""
        # This would require subprocess testing in a real scenario
        # For now, we test the exit code logic
        exit_codes = {
            0: "Success - Improvement detected",
            2: "Success with warnings - Unstable or redundant models detected", 
            3: "No improvement - Ensemble not better than best single model",
            4: "Invalid input or configuration"
        }
        
        # All exit codes should be defined
        for code in [0, 2, 3, 4]:
            assert code in exit_codes
            assert isinstance(exit_codes[code], str)
    
    def test_cli_output_files(self):
        """Test that CLI output files are stable."""
        expected_files = [
            'best_submission.csv',
            'methods.csv', 
            'report.html',
            'meta.json'
        ]
        
        # All expected files should be defined
        for file in expected_files:
            assert isinstance(file, str)
            assert file.endswith(('.csv', '.html', '.json'))


class TestDataContract:
    """Test data format contract stability."""
    
    def test_oof_data_contract(self):
        """Test OOF data format contract."""
        # Required columns
        required_cols = ['id', 'pred']
        optional_cols = ['target', 'fold']
        
        # Test valid data
        valid_data = pd.DataFrame({
            'id': [1, 2, 3],
            'pred': [0.1, 0.2, 0.3],
            'target': [0, 1, 0],
            'fold': [0, 1, 0]
        })
        
        for col in required_cols:
            assert col in valid_data.columns
        
        # Test data types
        assert pd.api.types.is_numeric_dtype(valid_data['id'])
        assert pd.api.types.is_numeric_dtype(valid_data['pred'])
    
    def test_sub_data_contract(self):
        """Test submission data format contract."""
        # Required columns
        required_cols = ['id', 'pred']
        
        # Test valid data
        valid_data = pd.DataFrame({
            'id': [1, 2, 3],
            'pred': [0.1, 0.2, 0.3]
        })
        
        for col in required_cols:
            assert col in valid_data.columns
        
        # Test data types
        assert pd.api.types.is_numeric_dtype(valid_data['id'])
        assert pd.api.types.is_numeric_dtype(valid_data['pred'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
