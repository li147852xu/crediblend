"""Plotting utilities for CrediBlend reports."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import base64
from io import BytesIO
import warnings

# Set style
plt.style.use('default')
sns.set_palette("husl")


def create_correlation_heatmap(corr_matrix: pd.DataFrame, 
                             title: str = "Model Correlation Matrix") -> str:
    """Create correlation heatmap as base64 encoded image.
    
    Args:
        corr_matrix: Correlation matrix DataFrame
        title: Plot title
        
    Returns:
        Base64 encoded image string
    """
    if corr_matrix.empty:
        return ""
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True,
                fmt='.3f',
                cbar_kws={"shrink": .8},
                ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Models', fontsize=12)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return image_base64


def create_weights_barplot(weights: Dict[str, float], 
                          title: str = "Optimized Model Weights") -> str:
    """Create weights bar plot as base64 encoded image.
    
    Args:
        weights: Dictionary of model weights
        title: Plot title
        
    Returns:
        Base64 encoded image string
    """
    if not weights:
        return ""
    
    # Prepare data
    models = list(weights.keys())
    weight_values = list(weights.values())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar plot
    bars = ax.bar(models, weight_values, color=sns.color_palette("husl", len(models)))
    
    # Add value labels on bars
    for bar, value in zip(bars, weight_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Weight', fontsize=12)
    ax.set_ylim(0, max(weight_values) * 1.2)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return image_base64


def create_performance_comparison(methods_df: pd.DataFrame,
                                metric_col: str = "overall_oof",
                                title: str = "Model Performance Comparison") -> str:
    """Create performance comparison plot.
    
    Args:
        methods_df: Methods comparison DataFrame
        metric_col: Metric column to plot
        title: Plot title
        
    Returns:
        Base64 encoded image string
    """
    if methods_df.empty or metric_col not in methods_df.columns:
        return ""
    
    # Filter valid data
    valid_data = methods_df[methods_df[metric_col].notna()].copy()
    if valid_data.empty:
        return ""
    
    # Sort by metric
    valid_data = valid_data.sort_values(metric_col, ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create bar plot
    bars = ax.bar(range(len(valid_data)), valid_data[metric_col], 
                  color=sns.color_palette("husl", len(valid_data)))
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, valid_data[metric_col])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel(metric_col.replace('_', ' ').title(), fontsize=12)
    ax.set_xticks(range(len(valid_data)))
    ax.set_xticklabels(valid_data['model'], rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return image_base64


def create_cluster_visualization(cluster_summary: pd.DataFrame,
                               title: str = "Model Clustering Results") -> str:
    """Create cluster visualization.
    
    Args:
        cluster_summary: Cluster summary DataFrame
        title: Plot title
        
    Returns:
        Base64 encoded image string
    """
    if cluster_summary.empty:
        return ""
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Cluster sizes
    cluster_sizes = cluster_summary.groupby('cluster_id').size()
    ax1.bar(cluster_sizes.index, cluster_sizes.values, 
            color=sns.color_palette("husl", len(cluster_sizes)))
    ax1.set_title('Cluster Sizes', fontweight='bold')
    ax1.set_xlabel('Cluster ID')
    ax1.set_ylabel('Number of Models')
    
    # Plot 2: OOF scores by cluster
    for cluster_id in cluster_summary['cluster_id'].unique():
        cluster_data = cluster_summary[cluster_summary['cluster_id'] == cluster_id]
        ax2.scatter([cluster_id] * len(cluster_data), cluster_data['oof_score'], 
                   label=f'Cluster {cluster_id}', s=100, alpha=0.7)
    
    ax2.set_title('OOF Scores by Cluster', fontweight='bold')
    ax2.set_xlabel('Cluster ID')
    ax2.set_ylabel('OOF Score')
    ax2.legend()
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return image_base64


def create_ensemble_comparison(blend_results: Dict[str, pd.DataFrame],
                             title: str = "Ensemble Method Comparison") -> str:
    """Create ensemble method comparison plot.
    
    Args:
        blend_results: Dictionary of blending results
        title: Plot title
        
    Returns:
        Base64 encoded image string
    """
    if not blend_results:
        return ""
    
    # Prepare data
    methods = []
    mean_preds = []
    std_preds = []
    
    for method, df in blend_results.items():
        methods.append(method.replace('_', ' ').title())
        mean_preds.append(df['pred'].mean())
        std_preds.append(df['pred'].std())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create bar plot with error bars
    x_pos = np.arange(len(methods))
    bars = ax.bar(x_pos, mean_preds, yerr=std_preds, 
                  color=sns.color_palette("husl", len(methods)),
                  capsize=5, alpha=0.7)
    
    # Add value labels
    for bar, mean_val in zip(bars, mean_preds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Ensemble Methods', fontsize=12)
    ax.set_ylabel('Mean Prediction', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return image_base64


def create_all_plots(corr_matrix: pd.DataFrame,
                    weights: Dict[str, float],
                    methods_df: pd.DataFrame,
                    cluster_summary: pd.DataFrame,
                    blend_results: Dict[str, pd.DataFrame]) -> Dict[str, str]:
    """Create all plots for the report.
    
    Args:
        corr_matrix: Correlation matrix
        weights: Model weights
        methods_df: Methods comparison DataFrame
        cluster_summary: Cluster summary DataFrame
        blend_results: Blending results
        
    Returns:
        Dictionary mapping plot name to base64 image
    """
    plots = {}
    
    try:
        # Correlation heatmap
        if not corr_matrix.empty:
            plots['correlation_heatmap'] = create_correlation_heatmap(corr_matrix)
    except Exception as e:
        warnings.warn(f"Failed to create correlation heatmap: {e}")
    
    try:
        # Weights bar plot
        if weights:
            plots['weights_barplot'] = create_weights_barplot(weights)
    except Exception as e:
        warnings.warn(f"Failed to create weights barplot: {e}")
    
    try:
        # Performance comparison
        if not methods_df.empty:
            plots['performance_comparison'] = create_performance_comparison(methods_df)
    except Exception as e:
        warnings.warn(f"Failed to create performance comparison: {e}")
    
    try:
        # Cluster visualization
        if not cluster_summary.empty:
            plots['cluster_visualization'] = create_cluster_visualization(cluster_summary)
    except Exception as e:
        warnings.warn(f"Failed to create cluster visualization: {e}")
    
    try:
        # Ensemble comparison
        if blend_results:
            plots['ensemble_comparison'] = create_ensemble_comparison(blend_results)
    except Exception as e:
        warnings.warn(f"Failed to create ensemble comparison: {e}")
    
    return plots
