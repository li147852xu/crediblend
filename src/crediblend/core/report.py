"""HTML report generation using Jinja2 templates."""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader
import pandas as pd


def load_template(template_name: str = "report.html.j2"):
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
                   decorrelation_info: Optional[Dict] = None,
                   cluster_summary: Optional[pd.DataFrame] = None,
                   stacking_info: Optional[Dict] = None,
                   weight_info: Optional[Dict] = None,
                   plots: Optional[Dict[str, str]] = None,
                   stability_report: Optional[Dict] = None,
                   window_metrics: pd.DataFrame = None) -> str:
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
        'cluster_summary': cluster_summary if cluster_summary is not None and not cluster_summary.empty else None,
        'stacking_info': stacking_info or {},
        'weight_info': weight_info or {},
        'plots': plots or {},
        'stability_report': stability_report or {},
        'window_metrics': window_metrics if window_metrics is not None and not window_metrics.empty else None,
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


def export_to_pdf(html_content: str, output_path: str) -> bool:
    """Export HTML report to PDF using WeasyPrint.
    
    Args:
        html_content: HTML content to export
        output_path: Path for PDF output
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from weasyprint import HTML, CSS
        from weasyprint.text.fonts import FontConfiguration
        
        # Create font configuration for better Unicode support
        font_config = FontConfiguration()
        
        # Convert HTML to PDF
        html_doc = HTML(string=html_content)
        css = CSS(string="""
            @page {
                size: A4;
                margin: 1in;
            }
            body {
                font-family: Arial, sans-serif;
                line-height: 1.4;
            }
            .metric-value {
                font-weight: bold;
            }
            .good-score {
                color: #28a745;
            }
            .bad-score {
                color: #dc3545;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 1rem;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
                font-weight: bold;
            }
        """, font_config=font_config)
        
        html_doc.write_pdf(output_path, stylesheets=[css], font_config=font_config)
        return True
        
    except ImportError:
        print("⚠️  WeasyPrint not available. Install with: pip install weasyprint")
        return False
    except Exception as e:
        print(f"⚠️  PDF export failed: {e}")
        return False


def create_blend_summary(methods_df: pd.DataFrame, weight_info: Optional[Dict] = None,
                        stacking_info: Optional[Dict] = None, 
                        blend_results: Optional[Dict] = None) -> Dict[str, Any]:
    """Create blend summary for JSON export.
    
    Args:
        methods_df: Methods comparison DataFrame
        weight_info: Weight optimization info
        stacking_info: Stacking info
        blend_results: Blending results
        
    Returns:
        Dictionary with blend summary
    """
    summary = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'top_methods': [],
        'weights': {},
        'stacking': {},
        'summary_stats': {}
    }
    
    # Get top 3 methods by OOF score
    if not methods_df.empty and 'overall_oof' in methods_df.columns:
        valid_methods = methods_df[methods_df['overall_oof'].notna()].copy()
        if not valid_methods.empty:
            top_methods = valid_methods.nlargest(3, 'overall_oof')
            summary['top_methods'] = [
                {
                    'method': row['model'],
                    'oof_score': float(row['overall_oof']),
                    'rank': i + 1
                }
                for i, (_, row) in enumerate(top_methods.iterrows())
            ]
    
    # Add weights if available
    if weight_info and weight_info.get('weights'):
        summary['weights'] = {k: float(v) for k, v in weight_info['weights'].items()}
    
    # Add stacking info if available
    if stacking_info and stacking_info.get('coefficients'):
        summary['stacking'] = {
            'meta_learner': stacking_info.get('meta_learner', 'unknown'),
            'coefficients': {k: float(v) for k, v in stacking_info['coefficients'].items()}
        }
    
    # Add summary statistics
    if not methods_df.empty:
        summary['summary_stats'] = {
            'n_models': len(methods_df),
            'n_blend_methods': len(blend_results) if blend_results else 0,
            'best_oof_score': float(methods_df['overall_oof'].max()) if 'overall_oof' in methods_df.columns else None
        }
    
    return summary
