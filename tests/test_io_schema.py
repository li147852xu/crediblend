"""Tests for input schema validation."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from crediblend.core.io import validate_oof_schema, validate_sub_schema, create_meta_json


class TestOOFSchemaValidation:
    """Test OOF file schema validation."""
    
    def test_valid_oof_schema(self):
        """Test valid OOF schema passes validation."""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'pred': [0.1, 0.2, 0.3],
            'target': [0, 1, 0],
            'fold': [0, 1, 0]
        })
        
        # Should not raise any exception
        validate_oof_schema(df, "test.csv")
    
    def test_valid_oof_schema_with_time_col(self):
        """Test valid OOF schema with time column passes validation."""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'pred': [0.1, 0.2, 0.3],
            'target': [0, 1, 0],
            'fold': [0, 1, 0],
            'date': ['2023-01-01', '2023-01-02', '2023-01-03']
        })
        
        # Should not raise any exception
        validate_oof_schema(df, "test.csv", time_col="date")
    
    def test_missing_required_columns(self):
        """Test missing required columns raises error."""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'target': [0, 1, 0]
            # Missing 'pred' column
        })
        
        with pytest.raises(ValueError, match="missing required columns"):
            validate_oof_schema(df, "test.csv")
    
    def test_invalid_data_types(self):
        """Test invalid data types raise error."""
        df = pd.DataFrame({
            'id': ['a', 'b', 'c'],  # Should be numeric
            'pred': [0.1, 0.2, 0.3],
            'target': [0, 1, 0]
        })
        
        with pytest.raises(ValueError, match="must be numeric"):
            validate_oof_schema(df, "test.csv")
    
    def test_missing_values_in_required_columns(self):
        """Test missing values in required columns raise error."""
        df = pd.DataFrame({
            'id': [1, 2, np.nan],  # Missing value
            'pred': [0.1, 0.2, 0.3],
            'target': [0, 1, 0]
        })
        
        with pytest.raises(ValueError, match="missing values"):
            validate_oof_schema(df, "test.csv")
    
    def test_invalid_time_column(self):
        """Test invalid time column raises error."""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'pred': [0.1, 0.2, 0.3],
            'target': [0, 1, 0],
            'date': ['invalid', 'date', 'format']  # Invalid datetime
        })
        
        with pytest.raises(ValueError, match="must be parseable as datetime"):
            validate_oof_schema(df, "test.csv", time_col="date")
    
    def test_unexpected_columns_warning(self):
        """Test unexpected columns generate warning."""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'pred': [0.1, 0.2, 0.3],
            'target': [0, 1, 0],
            'unexpected_col': ['a', 'b', 'c']  # Unexpected column
        })
        
        with pytest.warns(UserWarning, match="unexpected columns"):
            validate_oof_schema(df, "test.csv")


class TestSubmissionSchemaValidation:
    """Test submission file schema validation."""
    
    def test_valid_sub_schema(self):
        """Test valid submission schema passes validation."""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'pred': [0.1, 0.2, 0.3]
        })
        
        # Should not raise any exception
        validate_sub_schema(df, "test.csv")
    
    def test_missing_required_columns(self):
        """Test missing required columns raises error."""
        df = pd.DataFrame({
            'id': [1, 2, 3]
            # Missing 'pred' column
        })
        
        with pytest.raises(ValueError, match="missing required columns"):
            validate_sub_schema(df, "test.csv")
    
    def test_invalid_data_types(self):
        """Test invalid data types raise error."""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'pred': ['a', 'b', 'c']  # Should be numeric
        })
        
        with pytest.raises(ValueError, match="must be numeric"):
            validate_sub_schema(df, "test.csv")
    
    def test_missing_values(self):
        """Test missing values raise error."""
        df = pd.DataFrame({
            'id': [1, 2, np.nan],  # Missing value
            'pred': [0.1, 0.2, 0.3]
        })
        
        with pytest.raises(ValueError, match="missing values"):
            validate_sub_schema(df, "test.csv")
    
    def test_unexpected_columns_warning(self):
        """Test unexpected columns generate warning."""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'pred': [0.1, 0.2, 0.3],
            'unexpected_col': ['a', 'b', 'c']  # Unexpected column
        })
        
        with pytest.warns(UserWarning, match="unexpected columns"):
            validate_sub_schema(df, "test.csv")


class TestMetaJSONCreation:
    """Test meta.json creation."""
    
    def test_create_meta_json(self):
        """Test meta.json creation with valid data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            args = {
                'oof_dir': 'examples',
                'sub_dir': 'examples',
                'out_dir': temp_dir,
                'metric': 'auc',
                'seed': 42
            }
            
            oof_files = ['oof_modelA', 'oof_modelB']
            sub_files = ['sub_modelA', 'sub_modelB']
            
            create_meta_json(args, 42, oof_files, sub_files, temp_dir)
            
            # Check if meta.json was created
            meta_path = Path(temp_dir) / 'meta.json'
            assert meta_path.exists()
            
            # Check content
            with open(meta_path, 'r') as f:
                meta = f.read()
                assert 'crediblend_version' in meta
                assert 'arguments' in meta
                assert 'random_seed' in meta
                assert 'input_files' in meta


if __name__ == '__main__':
    pytest.main([__file__])
