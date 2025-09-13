# CrediBlend CLI Flags and Exit Codes

## Command Line Interface

### Basic Usage
```bash
crediblend --oof_dir <path> --sub_dir <path> --out <path> [OPTIONS]
```

### Required Arguments

| Flag | Type | Description |
|------|------|-------------|
| `--oof_dir` | string | Directory containing OOF CSV files (format: `oof_*.csv`) |
| `--sub_dir` | string | Directory containing submission CSV files (format: `sub_*.csv`) |
| `--out` | string | Output directory for results |

### Optional Arguments

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--metric` | string | `auc` | Evaluation metric (`auc`, `mse`, `mae`) |
| `--target_col` | string | `target` | Name of target column in OOF files |
| `--methods` | string | `mean,rank_mean,logit_mean,best_single` | Comma-separated list of blending methods |
| `--decorrelate` | choice | `off` | Enable decorrelation via clustering (`on`/`off`) |
| `--stacking` | choice | `none` | Enable stacking with meta-learner (`lr`/`ridge`/`none`) |
| `--search` | string | `iters=200,restarts=16` | Weight search parameters |
| `--seed` | integer | `None` | Random seed for reproducibility |
| `--time-col` | string | `None` | Time column name for time-sliced analysis |
| `--freq` | choice | `M` | Time frequency for windowing (`M`/`W`/`D`) |
| `--export` | choice | `none` | Export format for report (`pdf`/`none`) |
| `--summary-json` | string | `None` | Path to save blend summary JSON |

### Blending Methods

| Method | Description |
|--------|-------------|
| `mean` | Simple arithmetic mean of predictions |
| `rank_mean` | Mean of rank-transformed predictions |
| `logit_mean` | Mean in logit space (for binary classification) |
| `best_single` | Best performing single model |
| `weighted` | Weight-optimized ensemble (requires `--search`) |
| `stacking` | Stacked ensemble (requires `--stacking`) |

### Time Frequencies

| Code | Description |
|------|-------------|
| `M` | Monthly windows |
| `W` | Weekly windows |
| `D` | Daily windows |

## Exit Codes

CrediBlend uses meaningful exit codes for CI/CD integration and automated decision making:

| Code | Meaning | Description |
|------|---------|-------------|
| `0` | Success | Operation completed successfully with improvement detected |
| `2` | Success with warnings | Operation completed but with warnings (unstable models, redundant models) |
| `3` | No improvement | Ensemble methods did not improve over best single model |
| `4` | Invalid input | Invalid input data or configuration error |

### Exit Code Details

#### Code 0: Success
- All operations completed successfully
- Ensemble methods show improvement over best single model
- No critical warnings

#### Code 2: Success with warnings
- Operation completed but with non-critical issues:
  - Unstable models detected in time-sliced analysis
  - Redundant models removed by decorrelation
  - Weight optimization or stacking failed (fell back to simpler methods)
- Results are still valid and usable

#### Code 3: No improvement
- Ensemble methods did not provide improvement over the best single model
- May indicate:
  - Models are too similar (high correlation)
  - Insufficient diversity in predictions
  - Best single model is already optimal
- Consider using different blending strategies or model diversity

#### Code 4: Invalid input
- Critical errors that prevent successful execution:
  - Missing required files or directories
  - Invalid file formats or schemas
  - Configuration errors
  - Missing required columns in data
- Requires fixing input data or configuration before retry

## Examples

### Basic Usage
```bash
crediblend --oof_dir data/oof --sub_dir data/sub --out results/basic
```

### Advanced Ensemble
```bash
crediblend --oof_dir data/oof --sub_dir data/sub --out results/advanced \
  --decorrelate on --stacking lr --search iters=500,restarts=32 --seed 42
```

### Time-Sliced Analysis
```bash
crediblend --oof_dir data/oof --sub_dir data/sub --out results/time \
  --time-col date --freq M --decorrelate on
```

### Production Pipeline
```bash
crediblend --oof_dir data/oof --sub_dir data/sub --out results/prod \
  --export pdf --summary-json results/prod/blend_summary.json --seed 123
```

### CI/CD Integration
```bash
# In CI pipeline
crediblend --oof_dir data/oof --sub_dir data/sub --out results/ci --seed 42
EXIT_CODE=$?

case $EXIT_CODE in
  0) echo "✅ Ensemble improved - Deploy to production" ;;
  2) echo "⚠️  Warnings detected - Review and consider deployment" ;;
  3) echo "❌ No improvement - Skip deployment, use best single model" ;;
  4) echo "💥 Invalid input - Fix data and retry" ;;
esac
```

## Input File Requirements

### OOF Files (`oof_*.csv`)
**Required columns:**
- `id`: Sample identifier (numeric)
- `pred`: Model prediction (numeric)

**Optional columns:**
- `target`: Target values for evaluation (numeric)
- `fold`: Fold identifier for cross-validation (numeric)
- `{time_col}`: Time column for time-sliced analysis (datetime-parseable)

### Submission Files (`sub_*.csv`)
**Required columns:**
- `id`: Sample identifier (numeric)
- `pred`: Model prediction (numeric)

## Output Files

| File | Description |
|------|-------------|
| `best_submission.csv` | Best blended predictions |
| `methods.csv` | Model performance comparison |
| `report.html` | Comprehensive HTML report |
| `report.pdf` | PDF version of report (if `--export pdf`) |
| `meta.json` | Run metadata and configuration |
| `blend_summary.json` | Top-3 methods and weights summary |
| `weights.json` | Optimized ensemble weights |
| `stacking_coefficients.json` | Stacking meta-learner coefficients |
| `decorrelation_info.json` | Decorrelation analysis results |
| `window_metrics.csv` | Time-sliced AUC metrics |

## Error Handling

CrediBlend provides detailed error messages and suggestions:

```bash
# Example error messages
❌ Error: OOF file model1.csv missing required columns: ['pred']
❌ Error: Submission file model2.csv has missing values in 'id' column
❌ Error: Need at least 2 common models for weight optimization, got 0
⚠️  Weight optimization failed: Need at least 2 common models for weight optimization, got 0
```

## Performance Considerations

- **Memory**: Large datasets may require significant RAM for decorrelation and stacking
- **Time**: Weight optimization and stacking can be computationally expensive
- **Parallelization**: Weight search uses parallel processing when available
- **Caching**: Results are cached to avoid recomputation

## Troubleshooting

### Common Issues

1. **"No improvement over best single model"**
   - Check model diversity and correlation
   - Try different blending methods
   - Consider feature engineering

2. **"Weight optimization failed"**
   - Ensure sufficient model diversity
   - Check for identical predictions
   - Verify OOF data quality

3. **"Stacking failed"**
   - Ensure models have different predictions
   - Check for sufficient training data
   - Verify target column presence

4. **"Decorrelation removed all models"**
   - Models are too similar
   - Lower correlation threshold
   - Use different model architectures
