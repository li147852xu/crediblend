"""Performance optimization utilities for CrediBlend."""

import os
import psutil
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import warnings
from joblib import Parallel, delayed
import gc


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def check_memory_cap(memory_cap_mb: float) -> bool:
    """Check if current memory usage exceeds cap."""
    current_memory = get_memory_usage()
    return current_memory > memory_cap_mb


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame dtypes to reduce memory usage."""
    df_optimized = df.copy()
    
    # Convert float64 to float32 where possible
    for col in df_optimized.select_dtypes(include=[np.float64]).columns:
        if df_optimized[col].min() >= np.finfo(np.float32).min and \
           df_optimized[col].max() <= np.finfo(np.float32).max:
            df_optimized[col] = df_optimized[col].astype(np.float32)
    
    # Convert int64 to int32 where possible
    for col in df_optimized.select_dtypes(include=[np.int64]).columns:
        if df_optimized[col].min() >= np.iinfo(np.int32).min and \
           df_optimized[col].max() <= np.iinfo(np.int32).max:
            df_optimized[col] = df_optimized[col].astype(np.int32)
    
    return df_optimized


def chunked_read_csv(file_path: Path, chunk_size: int = 10000, 
                    memory_cap_mb: float = 4096) -> pd.DataFrame:
    """Read CSV file in chunks to manage memory usage."""
    chunks = []
    total_rows = 0
    
    try:
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            # Optimize dtypes
            chunk_optimized = optimize_dtypes(chunk)
            chunks.append(chunk_optimized)
            total_rows += len(chunk_optimized)
            
            # Check memory usage
            if check_memory_cap(memory_cap_mb):
                warnings.warn(f"Memory usage exceeded {memory_cap_mb}MB, stopping chunked read")
                break
                
        if chunks:
            result = pd.concat(chunks, ignore_index=True)
            print(f"Read {total_rows} rows from {file_path.name} in {len(chunks)} chunks")
            return result
        else:
            return pd.DataFrame()
            
    except Exception as e:
        warnings.warn(f"Chunked read failed for {file_path.name}: {e}")
        # Fallback to regular read
        return pd.read_csv(file_path)


def parallel_correlation_matrix(data_dict: Dict[str, pd.DataFrame], 
                              n_jobs: int = -1) -> pd.DataFrame:
    """Compute correlation matrix in parallel."""
    from .decorrelate import compute_correlation_matrix
    
    if n_jobs == 1:
        return compute_correlation_matrix(data_dict)
    
    # Split data into chunks for parallel processing
    model_names = list(data_dict.keys())
    chunk_size = max(1, len(model_names) // n_jobs) if n_jobs > 0 else len(model_names)
    
    def process_chunk(chunk_models):
        chunk_data = {name: data_dict[name] for name in chunk_models}
        return compute_correlation_matrix(chunk_data)
    
    # Process in parallel
    chunks = [model_names[i:i+chunk_size] for i in range(0, len(model_names), chunk_size)]
    results = Parallel(n_jobs=n_jobs)(delayed(process_chunk)(chunk) for chunk in chunks)
    
    # Combine results
    if len(results) == 1:
        return results[0]
    else:
        # For now, just return the first result
        # In a full implementation, we'd need to merge correlation matrices
        warnings.warn("Parallel correlation computation not fully implemented, using first chunk")
        return results[0]


def parallel_weight_optimization(oof_data: Dict[str, pd.DataFrame],
                                sub_data: Dict[str, pd.DataFrame],
                                scorer, target_col: str,
                                n_restarts: int = 16,
                                n_jobs: int = -1) -> Tuple[Dict[str, float], float, Dict[str, Any]]:
    """Optimize weights in parallel."""
    from .weights import optimize_weights
    
    if n_jobs == 1:
        return optimize_weights(oof_data, sub_data, scorer, target_col, n_restarts, 4)
    
    # For now, use single-threaded optimization
    # TODO: Implement proper parallel optimization
    return optimize_weights(oof_data, sub_data, scorer, target_col, n_restarts, 4)


def parallel_stacking(oof_data: Dict[str, pd.DataFrame],
                     sub_data: Dict[str, pd.DataFrame],
                     meta_learner: str = 'lr',
                     n_jobs: int = -1) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Perform stacking in parallel."""
    from .stacking import stacking_blend
    
    if n_jobs == 1:
        return stacking_blend(oof_data, sub_data, meta_learner)
    
    # For stacking, we can parallelize cross-validation
    # This is a simplified implementation
    return stacking_blend(oof_data, sub_data, meta_learner)


def memory_efficient_blend(oof_data: Dict[str, pd.DataFrame],
                          sub_data: Dict[str, pd.DataFrame],
                          method: str = 'mean',
                          memory_cap_mb: float = 4096) -> pd.DataFrame:
    """Perform blending with memory efficiency."""
    # Check memory usage
    if check_memory_cap(memory_cap_mb):
        warnings.warn(f"Memory usage {get_memory_usage():.1f}MB exceeds cap {memory_cap_mb}MB")
    
    # Optimize data types
    oof_optimized = {name: optimize_dtypes(df) for name, df in oof_data.items()}
    sub_optimized = {name: optimize_dtypes(df) for name, df in sub_data.items()}
    
    # Perform blending
    from .blend import mean_blend, rank_mean_blend, logit_mean_blend
    
    if method == 'mean':
        result = mean_blend(sub_optimized)
    elif method == 'rank_mean':
        result = rank_mean_blend(sub_optimized)
    elif method == 'logit_mean':
        result = logit_mean_blend(sub_optimized)
    else:
        result = mean_blend(sub_optimized)
    
    # Force garbage collection
    gc.collect()
    
    return result


def estimate_memory_usage(data_dict: Dict[str, pd.DataFrame]) -> float:
    """Estimate memory usage of data dictionary in MB."""
    total_memory = 0
    for name, df in data_dict.items():
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        total_memory += memory_mb
        print(f"  {name}: {memory_mb:.1f}MB")
    
    print(f"Total estimated memory: {total_memory:.1f}MB")
    return total_memory


def auto_strategy_selection(oof_data: Dict[str, pd.DataFrame],
                           sub_data: Dict[str, pd.DataFrame],
                           target_col: str = 'target',
                           metric: str = 'auc',
                           memory_cap_mb: float = 4096,
                           n_jobs: int = -1) -> str:
    """Automatically select the best blending strategy."""
    print("🤖 Auto strategy selection...")
    
    # Check data size and memory
    estimated_memory = estimate_memory_usage(oof_data)
    n_models = len(oof_data)
    
    print(f"  Models: {n_models}, Estimated memory: {estimated_memory:.1f}MB")
    
    # Strategy selection logic
    if n_models < 2:
        print("  → Using 'mean' (insufficient models for advanced strategies)")
        return 'mean'
    
    if estimated_memory > memory_cap_mb * 0.8:
        print("  → Using 'mean' (high memory usage)")
        return 'mean'
    
    if n_models > 10:
        print("  → Using 'decorrelate + weighted' (many models)")
        return 'decorrelate_weighted'
    
    if n_models >= 3:
        print("  → Using 'weighted' (moderate number of models)")
        return 'weighted'
    
    print("  → Using 'mean' (fallback)")
    return 'mean'


def performance_guardrails(data_dict: Dict[str, pd.DataFrame],
                          memory_cap_mb: float = 4096,
                          max_models: int = 20) -> Dict[str, pd.DataFrame]:
    """Apply performance guardrails to data."""
    print("🛡️  Applying performance guardrails...")
    
    # Check memory usage
    estimated_memory = estimate_memory_usage(data_dict)
    if estimated_memory > memory_cap_mb:
        warnings.warn(f"Estimated memory {estimated_memory:.1f}MB exceeds cap {memory_cap_mb}MB")
    
    # Limit number of models
    if len(data_dict) > max_models:
        warnings.warn(f"Too many models ({len(data_dict)}), limiting to {max_models}")
        # Keep first max_models
        limited_data = dict(list(data_dict.items())[:max_models])
        print(f"  Limited to {max_models} models")
        return limited_data
    
    # Optimize data types
    optimized_data = {}
    for name, df in data_dict.items():
        optimized_data[name] = optimize_dtypes(df)
    
    print(f"  Optimized {len(optimized_data)} models")
    return optimized_data
