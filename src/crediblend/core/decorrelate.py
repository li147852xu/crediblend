"""De-correlation utilities for reducing redundant models."""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from typing import Dict, List, Tuple, Optional
import warnings


def compute_correlation_matrix(oof_data: Dict[str, pd.DataFrame], 
                              target_col: str = "target") -> pd.DataFrame:
    """Compute Spearman correlation matrix of OOF predictions.
    
    Args:
        oof_data: Dictionary of OOF DataFrames with columns [id, pred, target?, fold?]
        target_col: Name of target column for filtering
        
    Returns:
        Correlation matrix DataFrame
    """
    # Collect predictions from all models
    pred_data = {}
    for model_name, df in oof_data.items():
        if target_col in df.columns:
            # Use all data for correlation
            pred_data[model_name] = df['pred'].values
        else:
            warnings.warn(f"No target column '{target_col}' in {model_name}, skipping")
    
    if len(pred_data) < 2:
        warnings.warn("Need at least 2 models for correlation analysis")
        return pd.DataFrame()
    
    # Compute correlation matrix
    model_names = list(pred_data.keys())
    n_models = len(model_names)
    corr_matrix = np.eye(n_models)
    
    for i in range(n_models):
        for j in range(i + 1, n_models):
            pred_i = pred_data[model_names[i]]
            pred_j = pred_data[model_names[j]]
            
            # Compute Spearman correlation
            corr, _ = spearmanr(pred_i, pred_j)
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr
    
    # Create DataFrame
    corr_df = pd.DataFrame(corr_matrix, index=model_names, columns=model_names)
    
    return corr_df


def hierarchical_clustering(corr_matrix: pd.DataFrame, 
                          threshold: float = 0.8) -> Dict[str, int]:
    """Perform hierarchical clustering on correlation matrix.
    
    Args:
        corr_matrix: Correlation matrix DataFrame
        threshold: Correlation threshold for clustering
        
    Returns:
        Dictionary mapping model name to cluster ID
    """
    if corr_matrix.empty:
        return {}
    
    # Convert correlation to distance (1 - |correlation|)
    distance_matrix = 1 - np.abs(corr_matrix.values)
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(squareform(distance_matrix), method='average')
    
    # Get cluster assignments
    cluster_ids = fcluster(linkage_matrix, 1 - threshold, criterion='distance')
    
    # Map to model names
    model_names = corr_matrix.index.tolist()
    cluster_map = {model_names[i]: int(cluster_ids[i]) for i in range(len(model_names))}
    
    return cluster_map


def select_cluster_medoids(oof_data: Dict[str, pd.DataFrame],
                          cluster_map: Dict[str, int],
                          oof_metrics: Dict[str, Dict[str, float]],
                          metric_key: str = "overall_oof") -> List[str]:
    """Select medoid (best performing model) from each cluster.
    
    Args:
        oof_data: Dictionary of OOF DataFrames
        cluster_map: Dictionary mapping model name to cluster ID
        oof_metrics: OOF metrics dictionary
        metric_key: Metric to use for selection
        
    Returns:
        List of selected model names
    """
    if not cluster_map:
        return list(oof_data.keys())
    
    # Group models by cluster
    clusters = {}
    for model_name, cluster_id in cluster_map.items():
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(model_name)
    
    selected_models = []
    
    for cluster_id, models in clusters.items():
        if len(models) == 1:
            # Single model in cluster, keep it
            selected_models.append(models[0])
        else:
            # Multiple models, select best performing
            best_model = None
            best_score = -np.inf
            
            for model in models:
                if model in oof_metrics:
                    score = oof_metrics[model].get(metric_key, -np.inf)
                    if not np.isnan(score) and score > best_score:
                        best_score = score
                        best_model = model
            
            if best_model:
                selected_models.append(best_model)
            else:
                # Fallback to first model if no metrics available
                selected_models.append(models[0])
    
    return selected_models


def filter_redundant_models(oof_data: Dict[str, pd.DataFrame],
                           oof_metrics: Dict[str, Dict[str, float]],
                           target_col: str = "target",
                           correlation_threshold: float = 0.8,
                           metric_key: str = "overall_oof") -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, float]], Dict]:
    """Filter redundant models using hierarchical clustering.
    
    Args:
        oof_data: Dictionary of OOF DataFrames
        oof_metrics: OOF metrics dictionary
        target_col: Name of target column
        correlation_threshold: Correlation threshold for clustering
        metric_key: Metric to use for medoid selection
        
    Returns:
        Tuple of (filtered_oof_data, filtered_metrics, decorrelation_info)
    """
    print(f"🔍 Analyzing correlations (threshold: {correlation_threshold})...")
    
    # Compute correlation matrix
    corr_matrix = compute_correlation_matrix(oof_data, target_col)
    
    if corr_matrix.empty:
        print("⚠️  No valid models for correlation analysis")
        return oof_data, oof_metrics, {}
    
    # Print correlation matrix
    print("📊 Correlation Matrix:")
    print(corr_matrix.round(3))
    
    # Perform clustering
    cluster_map = hierarchical_clustering(corr_matrix, correlation_threshold)
    
    # Select medoids
    selected_models = select_cluster_medoids(oof_data, cluster_map, oof_metrics, metric_key)
    
    # Filter data
    filtered_oof_data = {model: oof_data[model] for model in selected_models if model in oof_data}
    filtered_metrics = {model: oof_metrics[model] for model in selected_models if model in oof_metrics}
    
    # Prepare decorrelation info
    decorrelation_info = {
        'correlation_matrix': corr_matrix,
        'cluster_map': cluster_map,
        'selected_models': selected_models,
        'original_count': len(oof_data),
        'filtered_count': len(filtered_oof_data),
        'threshold': correlation_threshold
    }
    
    print(f"✅ Selected {len(selected_models)} models from {len(oof_data)} original models")
    print(f"📋 Selected models: {', '.join(selected_models)}")
    
    return filtered_oof_data, filtered_metrics, decorrelation_info


def get_cluster_summary(cluster_map: Dict[str, int], 
                       oof_metrics: Dict[str, Dict[str, float]],
                       metric_key: str = "overall_oof") -> pd.DataFrame:
    """Create cluster summary table.
    
    Args:
        cluster_map: Dictionary mapping model name to cluster ID
        oof_metrics: OOF metrics dictionary
        metric_key: Metric to use for ranking
        
    Returns:
        DataFrame with cluster summary
    """
    if not cluster_map:
        return pd.DataFrame()
    
    # Group models by cluster
    clusters = {}
    for model_name, cluster_id in cluster_map.items():
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(model_name)
    
    rows = []
    for cluster_id, models in clusters.items():
        # Get metrics for models in this cluster
        cluster_metrics = []
        for model in models:
            if model in oof_metrics:
                score = oof_metrics[model].get(metric_key, np.nan)
                cluster_metrics.append((model, score))
        
        # Sort by metric score
        cluster_metrics.sort(key=lambda x: x[1] if not np.isnan(x[1]) else -np.inf, reverse=True)
        
        # Create row for each model in cluster
        for i, (model, score) in enumerate(cluster_metrics):
            rows.append({
                'cluster_id': cluster_id,
                'model': model,
                'rank_in_cluster': i + 1,
                'oof_score': score,
                'is_medoid': i == 0,  # Best model in cluster
                'cluster_size': len(models)
            })
    
    return pd.DataFrame(rows)
