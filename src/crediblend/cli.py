"""Command-line interface for CrediBlend."""

import click
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from .core.io import read_oof_files, read_sub_files, align_submission_ids, save_outputs
from .core.metrics import Scorer, compute_oof_metrics, create_methods_table
from .core.blend import blend_predictions
from .core.report import generate_report


@click.command()
@click.option('--oof_dir', required=True, help='Directory containing OOF CSV files')
@click.option('--sub_dir', required=True, help='Directory containing submission CSV files')
@click.option('--out', 'out_dir', required=True, help='Output directory for results')
@click.option('--metric', default='auc', help='Metric to use for evaluation (auc, mse, mae)')
@click.option('--target_col', default='target', help='Name of target column in OOF files')
@click.option('--methods', default='mean,rank_mean,logit_mean,best_single', 
              help='Comma-separated list of blending methods')
def main(oof_dir: str, sub_dir: str, out_dir: str, metric: str, 
         target_col: str, methods: str) -> None:
    """CrediBlend: Blend machine learning predictions.
    
    This tool reads OOF (out-of-fold) and submission files, computes various
    blending methods, and generates a comprehensive report.
    """
    print("🎯 CrediBlend - Machine Learning Prediction Blending")
    print("=" * 50)
    
    # Parse methods
    method_list = [m.strip() for m in methods.split(',')]
    
    # Configuration
    config = {
        'oof_dir': oof_dir,
        'sub_dir': sub_dir,
        'out_dir': out_dir,
        'metric': metric,
        'target_col': target_col,
        'methods': method_list,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    try:
        # Initialize scorer
        scorer = Scorer(metric=metric)
        print(f"Using metric: {metric}")
        
        # Read OOF files
        print(f"\n📁 Reading OOF files from: {oof_dir}")
        oof_files = read_oof_files(oof_dir)
        
        # Read submission files
        print(f"\n📁 Reading submission files from: {sub_dir}")
        sub_files = read_sub_files(sub_dir)
        
        # Align submission IDs
        print(f"\n🔗 Aligning submission IDs...")
        aligned_sub_files = align_submission_ids(sub_files)
        
        # Compute OOF metrics
        print(f"\n📊 Computing OOF metrics...")
        oof_metrics = compute_oof_metrics(oof_files, scorer, target_col)
        
        # Create methods table
        print(f"\n📋 Creating methods comparison table...")
        methods_df = create_methods_table(oof_metrics, aligned_sub_files)
        
        # Apply blending methods
        print(f"\n🔄 Applying blending methods: {', '.join(method_list)}")
        blend_results = blend_predictions(aligned_sub_files, oof_metrics, method_list)
        
        # Select best submission (use best single model for now)
        if 'best_single' in blend_results:
            best_submission = blend_results['best_single']
        else:
            # Fallback to mean blend
            best_submission = blend_results.get('mean', list(blend_results.values())[0])
        
        # Generate HTML report
        print(f"\n📄 Generating HTML report...")
        report_html = generate_report(oof_metrics, methods_df, blend_results, config)
        
        # Save outputs
        print(f"\n💾 Saving outputs to: {out_dir}")
        save_outputs(out_dir, best_submission, methods_df, report_html)
        
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
