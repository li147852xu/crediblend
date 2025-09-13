"""Metrics computation for model evaluation."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from typing import Dict, List, Optional, Union


class Scorer:
    """Simple scorer for different metrics."""
    
    def __init__(self, metric: str = "auc"):
        """Initialize scorer with specified metric.
        
        Args:
            metric: Metric name ('auc', 'mse', 'mae')
        """
        self.metric = metric.lower()
        self._validate_metric()
    
    def _validate_metric(self):
        """Validate that metric is supported."""
        supported_metrics = ["auc", "mse", "mae"]
        if self.metric not in supported_metrics:
            raise ValueError(f"Unsupported metric: {self.metric}. Supported: {supported_metrics}")
    
    def score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute metric score.
        
        Args:
            y_true: True labels
            y_pred: Predicted values
            
        Returns:
            Metric score
        """
        if self.metric == "auc":
            if len(np.unique(y_true)) != 2:
                raise ValueError("AUC requires binary classification labels")
            return roc_auc_score(y_true, y_pred)
        elif self.metric == "mse":
            return -mean_squared_error(y_true, y_pred)  # Negative for maximization
        elif self.metric == "mae":
            return -mean_absolute_error(y_true, y_pred)  # Negative for maximization
        else:
            raise ValueError(f"Unknown metric: {self.metric}")


def compute_oof_metrics(oof_data: Dict[str, pd.DataFrame], 
                        scorer: Scorer,
                        target_col: str = "target") -> Dict[str, Dict[str, float]]:
    """Compute OOF metrics for each model.
    
    Args:
        oof_data: Dictionary of OOF DataFrames with columns [id, pred, fold?, target?]
        scorer: Scorer instance
        target_col: Name of target column (if available)
        
    Returns:
        Dictionary mapping model name to metrics dict
    """
    metrics = {}
    
    for model_name, df in oof_data.items():
        model_metrics = {}
        
        # Check if target column exists
        if target_col in df.columns:
            y_true = df[target_col].values
            y_pred = df['pred'].values
            
            # Overall OOF score
            try:
                overall_score = scorer.score(y_true, y_pred)
                model_metrics['overall_oof'] = overall_score
            except Exception as e:
                print(f"Warning: Could not compute overall OOF for {model_name}: {e}")
                model_metrics['overall_oof'] = np.nan
            
            # Per-fold scores if fold column exists
            if 'fold' in df.columns:
                fold_scores = []
                for fold in sorted(df['fold'].unique()):
                    fold_df = df[df['fold'] == fold]
                    fold_y_true = fold_df[target_col].values
                    fold_y_pred = fold_df['pred'].values
                    
                    try:
                        fold_score = scorer.score(fold_y_true, fold_y_pred)
                        fold_scores.append(fold_score)
                        model_metrics[f'fold_{fold}'] = fold_score
                    except Exception as e:
                        print(f"Warning: Could not compute fold {fold} score for {model_name}: {e}")
                        model_metrics[f'fold_{fold}'] = np.nan
                
                # Mean fold score
                valid_scores = [s for s in fold_scores if not np.isnan(s)]
                if valid_scores:
                    model_metrics['mean_fold'] = np.mean(valid_scores)
                else:
                    model_metrics['mean_fold'] = np.nan
        else:
            print(f"Warning: No target column '{target_col}' found in {model_name}, skipping OOF metrics")
            model_metrics['overall_oof'] = np.nan
            model_metrics['mean_fold'] = np.nan
        
        metrics[model_name] = model_metrics
    
    return metrics


def create_methods_table(metrics: Dict[str, Dict[str, float]], 
                        sub_files: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create methods comparison table.
    
    Args:
        metrics: OOF metrics dictionary
        sub_files: Submission files dictionary
        
    Returns:
        DataFrame with methods comparison
    """
    rows = []
    
    for model_name in metrics.keys():
        row = {
            'model': model_name,
            'overall_oof': metrics[model_name].get('overall_oof', np.nan),
            'mean_fold': metrics[model_name].get('mean_fold', np.nan),
            'submission_rows': len(sub_files.get(model_name, []))
        }
        
        # Add fold scores if available
        fold_cols = [k for k in metrics[model_name].keys() if k.startswith('fold_')]
        for fold_col in sorted(fold_cols):
            row[fold_col] = metrics[model_name][fold_col]
        
        rows.append(row)
    
    return pd.DataFrame(rows)
