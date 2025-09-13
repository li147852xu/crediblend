"""Command-line interface for CrediBlend."""

import click
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from .core.io import read_oof_files, read_sub_files, align_submission_ids, save_outputs, create_meta_json
from .core.metrics import Scorer, compute_oof_metrics, create_methods_table
from .core.blend import blend_predictions
from .core.report import generate_report, export_to_pdf, create_blend_summary
from .core.decorrelate import filter_redundant_models, get_cluster_summary
from .core.stacking import stacking_blend
from .core.weights import optimize_weights
from .core.plots import create_all_plots
from .core.stability import (compute_windowed_metrics, compute_stability_scores,
                           detect_dominance_patterns, generate_stability_report,
                           save_window_metrics)


@click.command()
@click.option('--oof_dir', required=True, help='Directory containing OOF CSV files')
@click.option('--sub_dir', required=True, help='Directory containing submission CSV files')
@click.option('--out', 'out_dir', required=True, help='Output directory for results')
@click.option('--metric', default='auc', help='Metric to use for evaluation (auc, mse, mae)')
@click.option('--target_col', default='target', help='Name of target column in OOF files')
@click.option('--methods', default='mean,rank_mean,logit_mean,best_single', 
              help='Comma-separated list of blending methods')
@click.option('--decorrelate', type=click.Choice(['on', 'off']), default='off',
              help='Enable decorrelation via clustering (on/off)')
@click.option('--stacking', type=click.Choice(['lr', 'ridge', 'none']), default='none',
              help='Enable stacking with meta-learner (lr/ridge/none)')
@click.option('--search', default='iters=200,restarts=16',
              help='Weight search parameters (iters=N,restarts=M)')
@click.option('--seed', type=int, default=None,
              help='Random seed for reproducibility')
@click.option('--time-col', default=None,
              help='Time column name for time-sliced analysis (e.g., date)')
@click.option('--freq', default='M', type=click.Choice(['M', 'W', 'D']),
              help='Time frequency for windowing (M=month, W=week, D=day)')
@click.option('--export', type=click.Choice(['pdf', 'none']), default='none',
              help='Export format for report (pdf/none)')
@click.option('--summary-json', help='Path to save blend summary JSON')
def main(oof_dir: str, sub_dir: str, out_dir: str, metric: str,
         target_col: str, methods: str, decorrelate: str, stacking: str,
         search: str, seed: int, time_col: str, freq: str, export: str, 
         summary_json: str) -> None:
    """CrediBlend: Blend machine learning predictions.
    
    This tool reads OOF (out-of-fold) and submission files, computes various
    blending methods, and generates a comprehensive report.
    """
    print("🎯 CrediBlend - Machine Learning Prediction Blending")
    print("=" * 50)
    print(f"Using metric: {metric}")
    
    # Set random seeds for reproducibility
    if seed is not None:
        print(f"Setting random seed: {seed}")
        np.random.seed(seed)
        import random
        random.seed(seed)
    
    # Parse methods
    method_list = [m.strip() for m in methods.split(',')]
    
    # Parse search parameters
    search_params = {}
    for param in search.split(','):
        if '=' in param:
            key, value = param.split('=')
            search_params[key.strip()] = int(value.strip())
    
    # Configuration
    config = {
        'oof_dir': oof_dir,
        'sub_dir': sub_dir,
        'out_dir': out_dir,
        'metric': metric,
        'target_col': target_col,
        'methods': method_list,
        'decorrelate': decorrelate == 'on',
        'stacking': stacking,
        'search_params': search_params,
        'seed': seed,
        'time_col': time_col,
        'freq': freq,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    try:
        # Initialize scorer
        scorer = Scorer(metric=metric)
        print(f"Using metric: {metric}")
        
        # Read OOF files
        print(f"\n📁 Reading OOF files from: {oof_dir}")
        oof_files = read_oof_files(oof_dir, time_col)
        
        # Read submission files
        print(f"\n📁 Reading submission files from: {sub_dir}")
        sub_files = read_sub_files(sub_dir)
        
        # Align submission IDs
        print(f"\n🔗 Aligning submission IDs...")
        aligned_sub_files = align_submission_ids(sub_files)
        
        # Compute OOF metrics
        print(f"\n📊 Computing OOF metrics...")
        oof_metrics = compute_oof_metrics(oof_files, scorer, target_col)
        
        # Apply decorrelation if enabled
        decorrelation_info = {}
        cluster_summary = pd.DataFrame()
        if config['decorrelate']:
            print(f"\n🔍 Applying decorrelation...")
            filtered_oof_files, filtered_metrics, decorrelation_info = filter_redundant_models(
                oof_files, oof_metrics, target_col, correlation_threshold=0.8
            )
            oof_files = filtered_oof_files
            oof_metrics = filtered_metrics
            
            # Create cluster summary
            if decorrelation_info.get('cluster_map'):
                cluster_summary = get_cluster_summary(
                    decorrelation_info['cluster_map'], oof_metrics
                )
        
        # Create methods table
        print(f"\n📋 Creating methods comparison table...")
        methods_df = create_methods_table(oof_metrics, aligned_sub_files)
        
        # Apply blending methods
        print(f"\n🔄 Applying blending methods: {', '.join(method_list)}")
        blend_results = blend_predictions(aligned_sub_files, oof_metrics, method_list)
        
        # Apply stacking if enabled
        stacking_info = {}
        if config['stacking'] != 'none':
            print(f"\n📚 Applying stacking with {config['stacking']}...")
            try:
                stacking_result, stacking_info = stacking_blend(
                    oof_files, aligned_sub_files, 
                    meta_learner=config['stacking'], 
                    target_col=target_col, 
                    random_state=seed
                )
                blend_results['stacking'] = stacking_result
            except Exception as e:
                print(f"⚠️  Stacking failed: {e}")
        
        # Apply weight optimization
        weight_info = {}
        if 'weight_optimization' in method_list or config['search_params']:
            print(f"\n⚖️  Applying weight optimization...")
            try:
                n_restarts = config['search_params'].get('restarts', 16)
                weight_result, weight_info = optimize_weights(
                    oof_files, aligned_sub_files, scorer.score,
                    target_col=target_col, n_restarts=n_restarts, random_state=seed
                )
                blend_results['weight_optimization'] = weight_result
            except Exception as e:
                print(f"⚠️  Weight optimization failed: {e}")
                weight_info = {}
        
        # Select best submission (use best single model for now)
        if 'best_single' in blend_results:
            best_submission = blend_results['best_single']
        else:
            # Fallback to mean blend
            best_submission = blend_results.get('mean', list(blend_results.values())[0])
        
        # Time-sliced analysis if time column is provided
        stability_report = {}
        window_metrics = pd.DataFrame()
        if time_col:
            print(f"\n⏰ Performing time-sliced analysis...")
            try:
                # Compute windowed metrics
                window_metrics = compute_windowed_metrics(
                    oof_files, time_col, freq, target_col, scorer.score
                )
                
                if not window_metrics.empty:
                    # Compute stability scores
                    stability_scores = compute_stability_scores(window_metrics)
                    
                    # Detect dominance patterns
                    dominance_analysis = detect_dominance_patterns(window_metrics)
                    
                    # Generate stability report
                    stability_report = generate_stability_report(
                        window_metrics, stability_scores, dominance_analysis
                    )
                    
                    # Save window metrics
                    save_window_metrics(window_metrics, out_dir)
                    
                    print(f"✅ Time-sliced analysis completed: {len(window_metrics)} window-method combinations")
                else:
                    print("⚠️  No valid time windows found for analysis")
            except Exception as e:
                print(f"⚠️  Time-sliced analysis failed: {e}")
                stability_report = {}

        # Create visualizations
        print(f"\n📊 Creating visualizations...")
        plots = create_all_plots(
            decorrelation_info.get('correlation_matrix', pd.DataFrame()),
            weight_info.get('weights', {}),
            methods_df,
            cluster_summary,
            blend_results
        )
        
        # Add stability plots if available
        if stability_report.get('plots'):
            plots.update(stability_report['plots'])
        
        # Generate HTML report
        print(f"\n📄 Generating HTML report...")
        report_html = generate_report(
            oof_metrics, methods_df, blend_results, config,
            decorrelation_info=decorrelation_info,
            cluster_summary=cluster_summary,
            stacking_info=stacking_info,
            weight_info=weight_info,
            plots=plots,
            stability_report=stability_report,
            window_metrics=window_metrics
        )
        
        # Save outputs
        print(f"\n💾 Saving outputs to: {out_dir}")
        save_outputs(out_dir, best_submission, methods_df, report_html)
        
        # Create meta.json
        args_dict = {
            'oof_dir': oof_dir,
            'sub_dir': sub_dir,
            'out_dir': out_dir,
            'metric': metric,
            'target_col': target_col,
            'methods': methods,
            'decorrelate': decorrelate,
            'stacking': stacking,
            'search': search,
            'time_col': time_col,
            'freq': freq,
            'export': export,
            'summary_json': summary_json
        }
        create_meta_json(args_dict, seed, list(oof_files.keys()), list(sub_files.keys()), out_dir)
        
        # Save additional outputs
        output_path = Path(out_dir)
        
        # Save weights if available
        if weight_info.get('weights'):
            import json
            with open(output_path / "weights.json", "w") as f:
                json.dump(weight_info, f, indent=2)
            print(f"Saved weights: {output_path / 'weights.json'}")
        
        # Save stacking coefficients if available
        if stacking_info.get('coefficients'):
            import json
            with open(output_path / "stacking_coefficients.json", "w") as f:
                json.dump(stacking_info, f, indent=2)
            print(f"Saved stacking coefficients: {output_path / 'stacking_coefficients.json'}")
        
        # Save decorrelation info if available
        if decorrelation_info:
            import json
            # Convert numpy arrays to lists for JSON serialization
            serializable_info = {}
            for key, value in decorrelation_info.items():
                if isinstance(value, pd.DataFrame):
                    serializable_info[key] = value.to_dict('records')
                elif isinstance(value, np.ndarray):
                    serializable_info[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    serializable_info[key] = value.item()
                else:
                    serializable_info[key] = value
            
            with open(output_path / "decorrelation_info.json", "w") as f:
                json.dump(serializable_info, f, indent=2)
            print(f"Saved decorrelation info: {output_path / 'decorrelation_info.json'}")
        
        # Export PDF if requested
        if export == 'pdf':
            print(f"\n📄 Exporting PDF report...")
            pdf_path = output_path / "report.pdf"
            if export_to_pdf(report_html, str(pdf_path)):
                print(f"Saved PDF report: {pdf_path}")
            else:
                print("PDF export failed or WeasyPrint not available")
        
        # Create blend summary JSON
        if summary_json:
            print(f"\n📊 Creating blend summary...")
            import json
            blend_summary = create_blend_summary(methods_df, weight_info, stacking_info, blend_results)
            with open(summary_json, "w") as f:
                json.dump(blend_summary, f, indent=2)
            print(f"Saved blend summary: {summary_json}")
        
        # Print summary
        print(f"\n✅ Success! Generated:")
        print(f"   • best_submission.csv ({len(best_submission)} predictions)")
        print(f"   • methods.csv ({len(methods_df)} models)")
        print(f"   • report.html")
        
        # Print best model info
        if not methods_df.empty and 'overall_oof' in methods_df.columns:
            best_model = methods_df.loc[methods_df['overall_oof'].idxmax()]
            print(f"\n🏆 Best model: {best_model['model']} (OOF: {best_model['overall_oof']:.4f})")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise click.ClickException(str(e))


if __name__ == '__main__':
    main()
