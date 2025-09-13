"""I/O utilities for reading OOF and submission files."""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings


def read_oof_files(oof_dir: str) -> Dict[str, pd.DataFrame]:
    """Read all OOF files from directory.
    
    Args:
        oof_dir: Directory containing OOF CSV files
        
    Returns:
        Dictionary mapping filename to DataFrame with columns [id, pred, fold?]
    """
    oof_files = {}
    oof_path = Path(oof_dir)
    
    if not oof_path.exists():
        raise FileNotFoundError(f"OOF directory not found: {oof_dir}")
    
    for file_path in oof_path.glob("oof_*.csv"):
        df = pd.read_csv(file_path)
        
        # Validate required columns
        if 'id' not in df.columns or 'pred' not in df.columns:
            raise ValueError(f"OOF file {file_path.name} must have 'id' and 'pred' columns")
        
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
        
        # Validate required columns
        if 'id' not in df.columns or 'pred' not in df.columns:
            raise ValueError(f"Submission file {file_path.name} must have 'id' and 'pred' columns")
        
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
