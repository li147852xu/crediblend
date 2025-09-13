"""HTML report generation using Jinja2 templates."""

import os
from pathlib import Path
from typing import Dict, Any
from jinja2 import Environment, FileSystemLoader
import pandas as pd


def load_template(template_name: str = "report.html.j2") -> str:
    """Load Jinja2 template.
    
    Args:
        template_name: Name of template file
        
    Returns:
        Template content
    """
    # Get template directory
    template_dir = Path(__file__).parent.parent / "templates"
    template_path = template_dir / template_name
    
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
    
    # Load template
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = env.get_template(template_name)
    
    return template


def generate_report(oof_metrics: Dict[str, Dict[str, float]],
                   methods_df: pd.DataFrame,
                   blend_results: Dict[str, pd.DataFrame],
                   config: Dict[str, Any],
                   decorrelation_info: Dict = None,
                   cluster_summary: pd.DataFrame = None,
                   stacking_info: Dict = None,
                   weight_info: Dict = None,
                   plots: Dict[str, str] = None) -> str:
    """Generate HTML report.
    
    Args:
        oof_metrics: OOF metrics dictionary
        methods_df: Methods comparison DataFrame
        blend_results: Blending results dictionary
        config: Configuration dictionary
        
    Returns:
        HTML report content
    """
    template = load_template()
    
    # Prepare data for template
    context = {
        'config': config,
        'oof_metrics': oof_metrics,
        'methods_df': methods_df,
        'blend_results': blend_results,
        'n_models': len(oof_metrics),
        'n_blend_methods': len(blend_results),
        'decorrelation_info': decorrelation_info or {},
        'cluster_summary': cluster_summary or pd.DataFrame(),
        'stacking_info': stacking_info or {},
        'weight_info': weight_info or {},
        'plots': plots or {},
    }
    
    # Add summary statistics
    if not methods_df.empty and 'overall_oof' in methods_df.columns:
        valid_oof = methods_df[methods_df['overall_oof'].notna()]
        if not valid_oof.empty:
            context['best_model'] = valid_oof.loc[valid_oof['overall_oof'].idxmax(), 'model']
            context['best_oof_score'] = valid_oof['overall_oof'].max()
        else:
            context['best_model'] = None
            context['best_oof_score'] = None
    else:
        context['best_model'] = None
        context['best_oof_score'] = None
    
    # Generate HTML
    html_content = template.render(**context)
    
    return html_content


def create_summary_stats(methods_df: pd.DataFrame) -> Dict[str, Any]:
    """Create summary statistics for the report.
    
    Args:
        methods_df: Methods comparison DataFrame
        
    Returns:
        Dictionary with summary statistics
    """
    stats = {}
    
    if methods_df.empty:
        return stats
    
    # Basic counts
    stats['n_models'] = len(methods_df)
    stats['n_with_oof'] = methods_df['overall_oof'].notna().sum() if 'overall_oof' in methods_df.columns else 0
    stats['n_with_folds'] = methods_df['mean_fold'].notna().sum() if 'mean_fold' in methods_df.columns else 0
    
    # OOF scores
    if 'overall_oof' in methods_df.columns:
        oof_scores = methods_df['overall_oof'].dropna()
        if len(oof_scores) > 0:
            stats['best_oof_score'] = oof_scores.max()
            stats['worst_oof_score'] = oof_scores.min()
            stats['mean_oof_score'] = oof_scores.mean()
            stats['std_oof_score'] = oof_scores.std()
    
    # Fold scores
    if 'mean_fold' in methods_df.columns:
        fold_scores = methods_df['mean_fold'].dropna()
        if len(fold_scores) > 0:
            stats['best_fold_score'] = fold_scores.max()
            stats['worst_fold_score'] = fold_scores.min()
            stats['mean_fold_score'] = fold_scores.mean()
            stats['std_fold_score'] = fold_scores.std()
    
    return stats
