# CrediBlend

A minimal CLI tool for blending machine learning predictions.

## Features

- **Multiple Blending Methods**: Mean, rank-based mean, and logit-space mean blending
- **OOF Evaluation**: Out-of-fold metrics with per-fold analysis
- **Flexible Metrics**: Support for AUC, MSE, and MAE
- **HTML Reports**: Beautiful, comprehensive reports with Jinja2 templates
- **ID Alignment**: Automatic alignment of submission files by ID

## Installation

```bash
pip install -e .
```

## Usage

```bash
crediblend run --oof_dir examples --sub_dir examples --out runs/demo
```

### Options

- `--oof_dir`: Directory containing OOF CSV files (format: `oof_*.csv`)
- `--sub_dir`: Directory containing submission CSV files (format: `sub_*.csv`)
- `--out`: Output directory for results
- `--metric`: Metric to use for evaluation (`auc`, `mse`, `mae`) [default: `auc`]
- `--target_col`: Name of target column in OOF files [default: `target`]
- `--methods`: Comma-separated list of blending methods [default: `mean,rank_mean,logit_mean,best_single`]

## File Formats

### OOF Files (`oof_*.csv`)
```csv
id,pred,target,fold
1,0.65,1,0
2,0.32,0,0
...
```

### Submission Files (`sub_*.csv`)
```csv
id,pred
1,0.68
2,0.29
...
```

## Output Files

- `best_submission.csv`: Best blended predictions
- `methods.csv`: Model performance comparison table
- `report.html`: Comprehensive HTML report

## Development

```bash
# Install in development mode
pip install -e .

# Run tests
pytest -q

# Run example
crediblend run --oof_dir examples --sub_dir examples --out runs/demo
```

## License

MIT
