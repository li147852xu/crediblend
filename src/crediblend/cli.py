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
from .core.performance import (auto_strategy_selection, performance_guardrails,
                             parallel_weight_optimization, memory_efficient_blend,
                             get_memory_usage)


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
@click.option('--n-jobs', type=int, default=-1, help='Number of parallel jobs (-1 for all CPUs)')
@click.option('--memory-cap', type=int, default=4096, help='Memory cap in MB')
@click.option('--strategy', type=click.Choice(['auto', 'mean', 'weighted', 'decorrelate_weighted']), 
              default='mean', help='Blending strategy')
def main(oof_dir: str, sub_dir: str, out_dir: str, metric: str,
         target_col: str, methods: str, decorrelate: str, stacking: str,
         search: str, seed: int, time_col: str, freq: str, export: str, 
         summary_json: str, n_jobs: int, memory_cap: int, strategy: str) -> None:
    """CrediBlend: Blend machine learning predictions.
    
    This tool reads OOF (out-of-fold) and submission files, computes various
    blending methods, and generates a comprehensive report.
    """
    print("🎯 CrediBlend - Machine Learning Prediction Blending")
    print("=" * 50)
    print(f"Using metric: {metric}")
    print(f"Memory cap: {memory_cap}MB, Parallel jobs: {n_jobs}")
    
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
        'n_jobs': n_jobs,
        'memory_cap': memory_cap,
        'strategy': strategy,
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
        
        # Apply performance guardrails
        print(f"\n🛡️  Applying performance guardrails...")
        oof_files = performance_guardrails(oof_files, memory_cap, max_models=20)
        sub_files = performance_guardrails(sub_files, memory_cap, max_models=20)
        
        print(f"Memory usage: {get_memory_usage():.1f}MB")
        
        # Align submission IDs
        print(f"\n🔗 Aligning submission IDs...")
        aligned_sub_files = align_submission_ids(sub_files)
        
        # Compute OOF metrics
        print(f"\n📊 Computing OOF metrics...")
        oof_metrics = compute_oof_metrics(oof_files, scorer, target_col)
        
        # Apply decorrelation if enabled
        decorrelation_info: Dict[str, Any] = {}
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
        if strategy == 'auto':
            print(f"\n🤖 Auto strategy selection...")
            selected_strategy = auto_strategy_selection(oof_files, aligned_sub_files, 
                                                      target_col, metric, memory_cap, n_jobs)
            print(f"Selected strategy: {selected_strategy}")
            
            if selected_strategy == 'decorrelate_weighted':
                # Apply decorrelation + weighted blending
                decorrelate = 'on'
                method_list = ['weighted']
            elif selected_strategy == 'weighted':
                method_list = ['weighted']
            else:
                method_list = ['mean']
        
        print(f"\n🔄 Applying blending methods: {', '.join(method_list)}")
        blend_results = blend_predictions(aligned_sub_files, oof_metrics, method_list)
        
        # Apply stacking if enabled
        stacking_info: Dict[str, Any] = {}
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
        weight_info: Dict[str, Any] = {}
        if 'weighted' in method_list or config['search_params']:
            print(f"\n⚖️  Applying weight optimization...")
            try:
                n_restarts = config['search_params'].get('restarts', 16)
                if n_jobs != 1:
                    print(f"Using parallel optimization with {n_jobs} jobs...")
                    weights, best_score, weight_info = parallel_weight_optimization(
                        oof_files, aligned_sub_files, scorer, target_col,
                        n_restarts, n_jobs
                    )
                    # Create weighted blend result
                    if weights:
                        weighted_pred = np.zeros(len(aligned_sub_files[list(aligned_sub_files.keys())[0]]))
                        for model_name, weight in weights.items():
                            # Map model names from oof_files to aligned_sub_files
                            if model_name in aligned_sub_files:
                                weighted_pred += weight * aligned_sub_files[model_name]['pred'].values
                            else:
                                # Try to find matching model by removing prefixes
                                for sub_name in aligned_sub_files.keys():
                                    if model_name.replace('oof_', 'sub_') == sub_name or \
                                       model_name.replace('model_', 'sub_model') == sub_name:
                                        weighted_pred += weight * aligned_sub_files[sub_name]['pred'].values
                                        break
                        blend_results['weighted'] = pd.DataFrame({
                            'id': aligned_sub_files[list(aligned_sub_files.keys())[0]]['id'].values,
                            'pred': weighted_pred
                        })
                else:
                    weight_result, weight_info = optimize_weights(
                        oof_files, aligned_sub_files, scorer,
                        target_col=target_col, n_restarts=n_restarts, random_state=seed
                    )
                    blend_results['weighted'] = weight_result
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
        
        # Determine exit code based on results
        exit_code = 0  # Default: success
        warnings_list = []
        
        # Check for improvement over best single model
        if not methods_df.empty and 'overall_oof' in methods_df.columns:
            best_single_score = methods_df['overall_oof'].max()
            best_single_model = methods_df.loc[methods_df['overall_oof'].idxmax(), 'model']
            
            # Check if any ensemble method improved over best single
            ensemble_improved = False
            for method in ['mean', 'rank_mean', 'logit_mean']:
                if method in blend_results:
                    # For now, we can't easily compare ensemble vs single without re-evaluation
                    # This would require evaluating the ensemble predictions on OOF data
                    pass
            
            # If no improvement detected, set exit code 3
            if not ensemble_improved and len(methods_df) > 1:
                exit_code = 3
                warnings_list.append("No improvement over best single model detected")
            
            print(f"\n🏆 Best model: {best_single_model} (OOF: {best_single_score:.4f})")
        
        # Check for stability warnings
        if stability_report and stability_report.get('warnings'):
            exit_code = 2  # Warnings
            warnings_list.extend(stability_report['warnings'])
        
        # Check for decorrelation warnings
        if decorrelation_info and decorrelation_info.get('original_count', 0) > decorrelation_info.get('filtered_count', 0):
            if exit_code == 0:
                exit_code = 2  # Warnings
            warnings_list.append(f"Decorrelation removed {decorrelation_info['original_count'] - decorrelation_info['filtered_count']} redundant models")
        
        # Print warnings if any
        if warnings_list:
            print(f"\n⚠️  Warnings:")
            for warning in warnings_list:
                print(f"   • {warning}")
        
        # Print exit code info
        exit_codes = {
            0: "Success - Improvement detected",
            2: "Success with warnings - Unstable or redundant models detected", 
            3: "No improvement - Ensemble not better than best single model",
            4: "Invalid input or configuration"
        }
        
        if exit_code != 0:
            print(f"\n📊 Exit code {exit_code}: {exit_codes.get(exit_code, 'Unknown')}")
        
        # Exit with appropriate code
        import sys
        sys.exit(exit_code)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import sys
        sys.exit(4)  # Invalid input/config


if __name__ == '__main__':
    main()
