"""Blending methods for combining predictions."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy.special import expit, logit


def mean_blend(sub_files: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Simple mean blending of predictions.
    
    Args:
        sub_files: Dictionary of submission DataFrames with columns [id, pred]
        
    Returns:
        DataFrame with mean predictions
    """
    if not sub_files:
        raise ValueError("No submission files provided for blending")
    
    # Get the first file to use as base
    first_name, first_df = next(iter(sub_files.items()))
    result = first_df[['id']].copy()
    
    # Collect all predictions
    pred_arrays = []
    for name, df in sub_files.items():
        pred_arrays.append(df['pred'].values)
    
    # Compute mean
    pred_matrix = np.column_stack(pred_arrays)
    mean_pred = np.mean(pred_matrix, axis=1)
    
    result['pred'] = mean_pred
    
    return result


def rank_mean_blend(sub_files: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Rank-based mean blending.
    
    Args:
        sub_files: Dictionary of submission DataFrames with columns [id, pred]
        
    Returns:
        DataFrame with rank-mean predictions
    """
    if not sub_files:
        raise ValueError("No submission files provided for blending")
    
    # Get the first file to use as base
    first_name, first_df = next(iter(sub_files.items()))
    result = first_df[['id']].copy()
    
    # Collect all predictions and convert to ranks
    rank_arrays = []
    for name, df in sub_files.items():
        # Convert to ranks (higher prediction = higher rank)
        ranks = df['pred'].rank(method='average')
        rank_arrays.append(ranks.values)
    
    # Compute mean of ranks
    rank_matrix = np.column_stack(rank_arrays)
    mean_ranks = np.mean(rank_matrix, axis=1)
    
    # Convert back to predictions by normalizing ranks to [0, 1]
    n_samples = len(result)
    normalized_pred = (mean_ranks - 1) / (n_samples - 1)
    
    result['pred'] = normalized_pred
    
    return result


def logit_mean_blend(sub_files: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Logit-space mean blending for binary probabilities.
    
    Args:
        sub_files: Dictionary of submission DataFrames with columns [id, pred]
        
    Returns:
        DataFrame with logit-mean predictions
    """
    if not sub_files:
        raise ValueError("No submission files provided for blending")
    
    # Get the first file to use as base
    first_name, first_df = next(iter(sub_files.items()))
    result = first_df[['id']].copy()
    
    # Collect all predictions and convert to logits
    logit_arrays = []
    for name, df in sub_files.items():
        preds = df['pred'].values
        
        # Clamp predictions to avoid logit issues
        preds = np.clip(preds, 1e-7, 1 - 1e-7)
        
        # Convert to logits
        logits = logit(preds)
        logit_arrays.append(logits)
    
    # Compute mean of logits
    logit_matrix = np.column_stack(logit_arrays)
    mean_logits = np.mean(logit_matrix, axis=1)
    
    # Convert back to probabilities
    mean_probs = expit(mean_logits)
    
    result['pred'] = mean_probs
    
    return result


def get_best_blend(sub_files: Dict[str, pd.DataFrame], 
                   oof_metrics: Dict[str, Dict[str, float]],
                   method: str = "overall_oof") -> pd.DataFrame:
    """Get the best single model prediction based on OOF metrics.
    
    Args:
        sub_files: Dictionary of submission DataFrames
        oof_metrics: OOF metrics dictionary
        method: Metric to use for selection ('overall_oof', 'mean_fold')
        
    Returns:
        DataFrame with best single model prediction
    """
    if not sub_files:
        raise ValueError("No submission files provided")
    
    # Find best model
    best_model = None
    best_score = -np.inf
    
    for model_name, metrics in oof_metrics.items():
        if model_name in sub_files:
            score = metrics.get(method, -np.inf)
            if not np.isnan(score) and score > best_score:
                best_score = score
                best_model = model_name
    
    if best_model is None:
        # Fallback to first available model
        best_model = list(sub_files.keys())[0]
        print(f"Warning: No valid OOF metrics found, using first model: {best_model}")
    else:
        print(f"Best model based on {method}: {best_model} (score: {best_score:.4f})")
    
    return sub_files[best_model].copy()


def blend_predictions(sub_files: Dict[str, pd.DataFrame],
                     oof_metrics: Dict[str, Dict[str, float]],
                     methods: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """Apply multiple blending methods.
    
    Args:
        sub_files: Dictionary of submission DataFrames
        oof_metrics: OOF metrics dictionary
        methods: List of blending methods to apply
        
    Returns:
        Dictionary mapping method name to blended predictions
    """
    if methods is None:
        methods = ["mean", "rank_mean", "logit_mean", "best_single"]
    
    results = {}
    
    for method in methods:
        if method == "mean":
            results[method] = mean_blend(sub_files)
        elif method == "rank_mean":
            results[method] = rank_mean_blend(sub_files)
        elif method == "logit_mean":
            results[method] = logit_mean_blend(sub_files)
        elif method == "weighted":
            # For weighted blending, we'll use mean as fallback
            # The actual weighted blending is handled separately
            results[method] = mean_blend(sub_files)
        elif method == "best_single":
            results[method] = get_best_blend(sub_files, oof_metrics)
        else:
            print(f"Warning: Unknown blending method: {method}")
            continue
        
        print(f"Computed {method} blend: {len(results[method])} predictions")
    
    return results
