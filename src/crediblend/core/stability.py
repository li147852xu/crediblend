"""Time-sliced evaluation and stability diagnostics."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings
from datetime import datetime, timedelta
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


def parse_time_frequency(freq: str) -> str:
    """Parse time frequency string.
    
    Args:
        freq: Frequency string (M/W/D for month/week/day)
        
    Returns:
        Pandas frequency string
    """
    freq_map = {
        'M': 'M',  # Monthly
        'W': 'W',  # Weekly
        'D': 'D',  # Daily
        'month': 'M',
        'week': 'W', 
        'day': 'D'
    }
    
    if freq not in freq_map:
        raise ValueError(f"Unsupported frequency: {freq}. Supported: M/W/D")
    
    return freq_map[freq]


def create_time_windows(df: pd.DataFrame, time_col: str, freq: str) -> pd.DataFrame:
    """Create time windows for analysis.
    
    Args:
        df: DataFrame with time column
        time_col: Name of time column
        freq: Frequency for windowing
        
    Returns:
        DataFrame with window assignments
    """
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found in DataFrame")
    
    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Create time windows
    df = df.copy()
    df['window'] = df[time_col].dt.to_period(freq)
    
    return df


def compute_windowed_metrics(oof_data: Dict[str, pd.DataFrame],
                           time_col: str,
                           freq: str,
                           target_col: str = "target",
                           metric_func: callable = roc_auc_score) -> pd.DataFrame:
    """Compute metrics for each time window.
    
    Args:
        oof_data: Dictionary of OOF DataFrames
        time_col: Name of time column
        freq: Frequency for windowing
        target_col: Name of target column
        metric_func: Metric function to use
        
    Returns:
        DataFrame with windowed metrics
    """
    window_metrics = []
    
    for model_name, df in oof_data.items():
        if target_col not in df.columns:
            warnings.warn(f"No target column '{target_col}' in {model_name}, skipping")
            continue
        
        # Create time windows
        try:
            df_windowed = create_time_windows(df, time_col, freq)
        except Exception as e:
            warnings.warn(f"Failed to create time windows for {model_name}: {e}")
            continue
        
        # Compute metrics for each window
        for window in df_windowed['window'].unique():
            window_df = df_windowed[df_windowed['window'] == window]
            
            if len(window_df) < 2:  # Need at least 2 samples for AUC
                continue
            
            y_true = window_df[target_col].values
            y_pred = window_df['pred'].values
            
            try:
                metric_value = metric_func(y_true, y_pred)
                window_metrics.append({
                    'model': model_name,
                    'window': str(window),
                    'auc': metric_value,
                    'n_samples': len(window_df)
                })
            except Exception as e:
                warnings.warn(f"Failed to compute metric for {model_name} window {window}: {e}")
                continue
    
    return pd.DataFrame(window_metrics)


def compute_stability_scores(window_metrics: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute stability scores for each model.
    
    Args:
        window_metrics: DataFrame with windowed metrics
        
    Returns:
        Dictionary with stability scores
    """
    stability_scores = {}
    
    for model in window_metrics['model'].unique():
        model_metrics = window_metrics[window_metrics['model'] == model]['auc'].values
        
        if len(model_metrics) < 2:
            continue
        
        # Compute stability metrics
        mean_auc = np.mean(model_metrics)
        std_auc = np.std(model_metrics)
        iqr_auc = np.percentile(model_metrics, 75) - np.percentile(model_metrics, 25)
        
        # Stability score: lower is more stable
        stability_score = std_auc / mean_auc if mean_auc > 0 else np.inf
        iqr_stability = iqr_auc / mean_auc if mean_auc > 0 else np.inf
        
        stability_scores[model] = {
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'iqr_auc': iqr_auc,
            'stability_score': stability_score,
            'iqr_stability': iqr_stability,
            'n_windows': len(model_metrics),
            'min_auc': np.min(model_metrics),
            'max_auc': np.max(model_metrics)
        }
    
    return stability_scores


def detect_dominance_patterns(window_metrics: pd.DataFrame, 
                            dominance_threshold: float = 0.7) -> Dict[str, Any]:
    """Detect model dominance patterns across windows.
    
    Args:
        window_metrics: DataFrame with windowed metrics
        dominance_threshold: Threshold for considering a model dominant
        
    Returns:
        Dictionary with dominance analysis
    """
    dominance_analysis = {
        'dominant_models': {},
        'unstable_windows': [],
        'leakage_candidates': []
    }
    
    # Analyze each window
    for window in window_metrics['window'].unique():
        window_df = window_metrics[window_metrics['window'] == window]
        
        if len(window_df) < 2:
            continue
        
        # Find best model in this window
        best_model = window_df.loc[window_df['auc'].idxmax()]
        best_auc = best_model['auc']
        second_best_auc = window_df['auc'].nlargest(2).iloc[1] if len(window_df) > 1 else 0
        
        # Check for dominance
        dominance_ratio = best_auc / (second_best_auc + 1e-8)
        
        if dominance_ratio > 1 / dominance_threshold:
            model_name = best_model['model']
            if model_name not in dominance_analysis['dominant_models']:
                dominance_analysis['dominant_models'][model_name] = 0
            dominance_analysis['dominant_models'][model_name] += 1
            
            dominance_analysis['unstable_windows'].append({
                'window': window,
                'dominant_model': model_name,
                'dominance_ratio': dominance_ratio,
                'best_auc': best_auc,
                'second_best_auc': second_best_auc
            })
    
    # Detect potential leakage (unrealistically high AUC)
    for model in window_metrics['model'].unique():
        model_metrics = window_metrics[window_metrics['model'] == model]['auc'].values
        
        if len(model_metrics) > 0:
            mean_auc = np.mean(model_metrics)
            max_auc = np.max(model_metrics)
            
            # Flag if mean AUC > 0.95 or max AUC > 0.99
            if mean_auc > 0.95 or max_auc > 0.99:
                dominance_analysis['leakage_candidates'].append({
                    'model': model,
                    'mean_auc': mean_auc,
                    'max_auc': max_auc,
                    'n_windows': len(model_metrics)
                })
    
    return dominance_analysis


def create_windowed_auc_plot(window_metrics: pd.DataFrame) -> str:
    """Create windowed AUC line chart.
    
    Args:
        window_metrics: DataFrame with windowed metrics
        
    Returns:
        Base64 encoded image string
    """
    if window_metrics.empty:
        return ""
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot each model
    for model in window_metrics['model'].unique():
        model_data = window_metrics[window_metrics['model'] == model]
        ax.plot(model_data['window'], model_data['auc'], 
                marker='o', label=model, linewidth=2, markersize=4)
    
    ax.set_title('AUC by Time Window / 时间窗口AUC', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time Window / 时间窗口', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Convert to base64
    import base64
    from io import BytesIO
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return image_base64


def create_stability_heatmap(stability_scores: Dict[str, Dict[str, float]]) -> str:
    """Create stability scores heatmap.
    
    Args:
        stability_scores: Dictionary with stability scores
        
    Returns:
        Base64 encoded image string
    """
    if not stability_scores:
        return ""
    
    # Prepare data for heatmap
    models = list(stability_scores.keys())
    metrics = ['mean_auc', 'std_auc', 'stability_score', 'iqr_stability']
    
    data = []
    for model in models:
        row = [stability_scores[model].get(metric, np.nan) for metric in metrics]
        data.append(row)
    
    data = np.array(data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create heatmap
    im = ax.imshow(data, cmap='RdYlBu_r', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(['Mean AUC', 'Std AUC', 'Stability Score', 'IQR Stability'])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Score Value', rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f'{data[i, j]:.3f}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title('Model Stability Scores / 模型稳定性评分', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Convert to base64
    import base64
    from io import BytesIO
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return image_base64


def generate_stability_report(window_metrics: pd.DataFrame,
                            stability_scores: Dict[str, Dict[str, float]],
                            dominance_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive stability report.
    
    Args:
        window_metrics: DataFrame with windowed metrics
        stability_scores: Dictionary with stability scores
        dominance_analysis: Dictionary with dominance analysis
        
    Returns:
        Dictionary with stability report
    """
    report = {
        'summary': {
            'n_models': len(window_metrics['model'].unique()) if not window_metrics.empty else 0,
            'n_windows': len(window_metrics['window'].unique()) if not window_metrics.empty else 0,
            'total_samples': window_metrics['n_samples'].sum() if not window_metrics.empty else 0
        },
        'stability_scores': stability_scores,
        'dominance_analysis': dominance_analysis,
        'plots': {}
    }
    
    # Generate plots
    report['plots']['windowed_auc'] = create_windowed_auc_plot(window_metrics)
    report['plots']['stability_heatmap'] = create_stability_heatmap(stability_scores)
    
    # Add warnings
    warnings = []
    
    # Check for unstable models
    for model, scores in stability_scores.items():
        if scores['stability_score'] > 0.2:  # 20% coefficient of variation
            warnings.append(f"Model {model} shows high instability (CV: {scores['stability_score']:.3f})")
    
    # Check for dominant models
    if dominance_analysis['dominant_models']:
        most_dominant = max(dominance_analysis['dominant_models'].items(), key=lambda x: x[1])
        if most_dominant[1] > len(window_metrics['window'].unique()) * 0.5:
            warnings.append(f"Model {most_dominant[0]} dominates in {most_dominant[1]} windows")
    
    # Check for leakage candidates
    if dominance_analysis['leakage_candidates']:
        for candidate in dominance_analysis['leakage_candidates']:
            warnings.append(f"Model {candidate['model']} shows suspiciously high AUC (mean: {candidate['mean_auc']:.3f})")
    
    report['warnings'] = warnings
    
    return report


def save_window_metrics(window_metrics: pd.DataFrame, output_dir: str) -> None:
    """Save window metrics to CSV.
    
    Args:
        window_metrics: DataFrame with windowed metrics
        output_dir: Output directory
    """
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    window_metrics.to_csv(output_path / "window_metrics.csv", index=False)
    print(f"Saved window metrics: {output_path / 'window_metrics.csv'}")
