"""I/O utilities for reading OOF and submission files."""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
import json
import numpy as np
from datetime import datetime


def validate_oof_schema(df: pd.DataFrame, filename: str, time_col: Optional[str] = None) -> None:
    """Validate OOF file schema.
    
    Args:
        df: DataFrame to validate
        filename: Name of the file for error messages
        time_col: Optional time column name
        
    Raises:
        ValueError: If schema validation fails
    """
    # Required columns
    required_cols = ['id', 'pred']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"OOF file {filename} missing required columns: {missing_cols}")
    
    # Optional columns
    optional_cols = ['fold', 'target']
    if time_col:
        optional_cols.append(time_col)
    
    # Check for unexpected columns
    expected_cols = set(required_cols + optional_cols)
    unexpected_cols = set(df.columns) - expected_cols
    if unexpected_cols:
        warnings.warn(f"OOF file {filename} has unexpected columns: {unexpected_cols}")
    
    # Validate data types
    if not pd.api.types.is_numeric_dtype(df['pred']):
        raise ValueError(f"OOF file {filename} 'pred' column must be numeric")
    
    if not pd.api.types.is_numeric_dtype(df['id']):
        raise ValueError(f"OOF file {filename} 'id' column must be numeric")
    
    # Check for missing values in required columns
    if df['id'].isna().any():
        raise ValueError(f"OOF file {filename} has missing values in 'id' column")
    
    if df['pred'].isna().any():
        raise ValueError(f"OOF file {filename} has missing values in 'pred' column")
    
    # Validate fold column if present
    if 'fold' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['fold']):
            raise ValueError(f"OOF file {filename} 'fold' column must be numeric")
        if df['fold'].isna().any():
            raise ValueError(f"OOF file {filename} has missing values in 'fold' column")
    
    # Validate target column if present
    if 'target' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['target']):
            raise ValueError(f"OOF file {filename} 'target' column must be numeric")
        if df['target'].isna().any():
            raise ValueError(f"OOF file {filename} has missing values in 'target' column")
    
    # Validate time column if present
    if time_col and time_col in df.columns:
        try:
            pd.to_datetime(df[time_col])
        except Exception as e:
            raise ValueError(f"OOF file {filename} '{time_col}' column must be parseable as datetime: {e}")


def validate_sub_schema(df: pd.DataFrame, filename: str) -> None:
    """Validate submission file schema.
    
    Args:
        df: DataFrame to validate
        filename: Name of the file for error messages
        
    Raises:
        ValueError: If schema validation fails
    """
    # Required columns
    required_cols = ['id', 'pred']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Submission file {filename} missing required columns: {missing_cols}")
    
    # Check for unexpected columns
    expected_cols = set(required_cols)
    unexpected_cols = set(df.columns) - expected_cols
    if unexpected_cols:
        warnings.warn(f"Submission file {filename} has unexpected columns: {unexpected_cols}")
    
    # Validate data types
    if not pd.api.types.is_numeric_dtype(df['pred']):
        raise ValueError(f"Submission file {filename} 'pred' column must be numeric")
    
    if not pd.api.types.is_numeric_dtype(df['id']):
        raise ValueError(f"Submission file {filename} 'id' column must be numeric")
    
    # Check for missing values
    if df['id'].isna().any():
        raise ValueError(f"Submission file {filename} has missing values in 'id' column")
    
    if df['pred'].isna().any():
        raise ValueError(f"Submission file {filename} has missing values in 'pred' column")


def create_meta_json(args: Dict[str, Any], seed: Optional[int], 
                    oof_files: List[str], sub_files: List[str], 
                    output_dir: str) -> None:
    """Create meta.json with run information.
    
    Args:
        args: Command line arguments
        seed: Random seed used
        oof_files: List of OOF file names
        sub_files: List of submission file names
        output_dir: Output directory path
    """
    import platform
    import sys
    
    meta = {
        'run_info': {
            'timestamp': datetime.now().isoformat(),
            'python_version': sys.version,
            'platform': platform.platform(),
            'crediblend_version': '0.4.0'
        },
        'arguments': args,
        'random_seed': seed,
        'input_files': {
            'oof_files': oof_files,
            'submission_files': sub_files
        },
        'numpy_version': np.__version__,
        'pandas_version': pd.__version__
    }
    
    meta_path = Path(output_dir) / 'meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    
    print(f"Saved metadata: {meta_path}")


def read_oof_files(oof_dir: str, time_col: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """Read all OOF files from directory.
    
    Args:
        oof_dir: Directory containing OOF CSV files
        time_col: Optional time column name for validation
        
    Returns:
        Dictionary mapping filename to DataFrame with columns [id, pred, fold?, target?, time_col?]
    """
    oof_files = {}
    oof_path = Path(oof_dir)
    
    if not oof_path.exists():
        raise FileNotFoundError(f"OOF directory not found: {oof_dir}")
    
    for file_path in oof_path.glob("oof_*.csv"):
        df = pd.read_csv(file_path)
        
        # Validate schema
        validate_oof_schema(df, file_path.name, time_col)
        
        # Check if fold column exists
        has_fold = 'fold' in df.columns
        
        oof_files[file_path.stem] = df
        
        print(f"Loaded OOF file: {file_path.name} ({len(df)} rows, fold={'yes' if has_fold else 'no'})")
    
    if not oof_files:
        raise ValueError(f"No OOF files found in {oof_dir}")
    
    return oof_files


def read_sub_files(sub_dir: str) -> Dict[str, pd.DataFrame]:
    """Read all submission files from directory.
    
    Args:
        sub_dir: Directory containing submission CSV files
        
    Returns:
        Dictionary mapping filename to DataFrame with columns [id, pred]
    """
    sub_files = {}
    sub_path = Path(sub_dir)
    
    if not sub_path.exists():
        raise FileNotFoundError(f"Submission directory not found: {sub_dir}")
    
    for file_path in sub_path.glob("sub_*.csv"):
        df = pd.read_csv(file_path)
        
        # Validate schema
        validate_sub_schema(df, file_path.name)
        
        sub_files[file_path.stem] = df
        
        print(f"Loaded submission file: {file_path.name} ({len(df)} rows)")
    
    if not sub_files:
        raise ValueError(f"No submission files found in {sub_dir}")
    
    return sub_files


def align_submission_ids(sub_files: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Align submission files by ID using inner join.
    
    Args:
        sub_files: Dictionary of submission DataFrames
        
    Returns:
        Dictionary of aligned submission DataFrames
    """
    if len(sub_files) <= 1:
        return sub_files
    
    # Get all unique IDs
    all_ids = set()
    for df in sub_files.values():
        all_ids.update(df['id'].unique())
    
    # Find common IDs
    common_ids = set(sub_files[list(sub_files.keys())[0]]['id'].unique())
    for df in sub_files.values():
        common_ids = common_ids.intersection(set(df['id'].unique()))
    
    if len(common_ids) < len(all_ids):
        warnings.warn(f"ID mismatch detected: {len(all_ids)} total IDs, {len(common_ids)} common IDs")
    
    # Filter to common IDs
    aligned_files = {}
    for name, df in sub_files.items():
        aligned_df = df[df['id'].isin(common_ids)].copy()
        aligned_df = aligned_df.sort_values('id').reset_index(drop=True)
        aligned_files[name] = aligned_df
    
    return aligned_files


def save_outputs(output_dir: str, best_submission: pd.DataFrame, 
                methods_df: pd.DataFrame, report_html: str) -> None:
    """Save all outputs to directory.
    
    Args:
        output_dir: Directory to save outputs
        best_submission: Best submission DataFrame
        methods_df: Methods comparison DataFrame
        report_html: HTML report content
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save best submission
    best_submission.to_csv(output_path / "best_submission.csv", index=False)
    print(f"Saved best submission: {output_path / 'best_submission.csv'}")
    
    # Save methods comparison
    methods_df.to_csv(output_path / "methods.csv", index=False)
    print(f"Saved methods comparison: {output_path / 'methods.csv'}")
    
    # Save HTML report
    with open(output_path / "report.html", "w") as f:
        f.write(report_html)
    print(f"Saved HTML report: {output_path / 'report.html'}")
