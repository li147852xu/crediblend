# CrediBlend

> A fast, reproducible ensembling toolkit for tabular ML: merge multiple OOF/SUB predictions, de-correlate models, and search optimal blends with solid diagnostics.

## Why CrediBlend?

- **OOF ensembles often inflate or collapse** due to correlation and unstable folds. CrediBlend quantifies correlation, filters redundancy, and searches robust weights/stacking.
- **Production-ready** with parallel processing, memory optimization, and stable APIs for CI/CD integration.
- **Comprehensive diagnostics** with time-sliced analysis, stability scoring, and visual reports.

## What it does

- **Strategies**: mean / rank_mean / logit_mean / stacking (LR/Ridge) / weight search
- **Diagnostics**: Spearman heatmap, clustering, time-sliced AUC, stability score
- **Outputs**: best submission, method leaderboard, blend_summary.json, HTML/PDF report
- **DX**: CLI & Python SDK, deterministic runs via meta.json, CI-ready exit codes

## Quickstart

```bash
pip install crediblend
crediblend --oof_dir path/to/oof --sub_dir path/to/sub --out runs/demo
```

## 🚀 Features

### Core Blending Methods
- **Mean Blending**: Simple arithmetic mean of predictions
- **Rank-based Blending**: Mean of rank-transformed predictions
- **Logit-space Blending**: Mean in logit space for probability predictions
- **Weight Optimization**: Parallel search for optimal ensemble weights
- **Stacking**: Meta-learning with LogisticRegression/Ridge

### Advanced Diagnostics
- **Correlation Analysis**: Spearman correlation matrix and hierarchical clustering
- **Time-sliced Evaluation**: Per-window AUC analysis for temporal stability
- **Stability Scoring**: Standard deviation and IQR of windowed metrics
- **Dominance Detection**: Identify models that dominate across time windows
- **Leakage Hints**: Flag models with suspiciously high performance

### Performance & Production
- **Parallel Processing**: Multi-core optimization with joblib
- **Memory Optimization**: Automatic dtype optimization and chunked reading
- **Auto Strategy**: Intelligent strategy selection based on data characteristics
- **Docker Support**: Production-ready containerization
- **CI/CD Integration**: Meaningful exit codes and stable contracts

## 📊 Usage Examples

### Command Line Interface

```bash
# Basic usage
crediblend --oof_dir data/oof --sub_dir data/sub --out results

# Advanced features
crediblend --oof_dir data/oof --sub_dir data/sub --out results \
  --decorrelate on --stacking lr --search iters=200,restarts=16 --seed 42

# Performance optimized
crediblend --oof_dir data/oof --sub_dir data/sub --out results \
  --n-jobs 8 --memory-cap 4096 --strategy auto --seed 42

# Time-sliced analysis
crediblend --oof_dir data/oof --sub_dir data/sub --out results \
  --time-col date --freq M --decorrelate on

# PDF export with summary
crediblend --oof_dir data/oof --sub_dir data/sub --out results \
  --export pdf --summary-json results/blend_summary.json --seed 123
```

### Python API

```python
from crediblend.api import fit_blend, predict_blend, quick_blend
import pandas as pd

# Load your data
oof_data = [pd.read_csv('oof_model1.csv'), pd.read_csv('oof_model2.csv')]
sub_data = [pd.read_csv('sub_model1.csv'), pd.read_csv('sub_model2.csv')]

# Quick blending
result = quick_blend(oof_data, sub_data, method='mean')
print(result.predictions)

# Advanced blending with configuration
from crediblend.api import BlendConfig
config = BlendConfig(method='weighted', metric='auc', random_state=42)
model = fit_blend(oof_data, config=config)
result = predict_blend(model, sub_data)

# Weight optimization
from crediblend.api import search_weights
weights, score, info = search_weights(oof_data, sub_data, metric='auc')
print(f"Best score: {score:.4f}")
print(f"Weights: {weights}")
```

### Docker Usage

```bash
# Build image
docker build -t crediblend .

# Run with data mounted
docker run -v $(pwd)/data:/data -v $(pwd)/results:/results \
  crediblend --oof_dir /data/oof --sub_dir /data/sub --out /results

# Performance-optimized run
docker run -v $(pwd)/data:/data -v $(pwd)/results:/results \
  crediblend --oof_dir /data/oof --sub_dir /data/sub --out /results \
  --n-jobs 4 --memory-cap 2048 --strategy auto
```

## 📁 File Formats

### OOF Files (`oof_*.csv`)
**Required columns**: `id`, `pred`  
**Optional columns**: `target`, `fold`, `{time_col}`

```csv
id,pred,target,fold
1,0.1,0,0
2,0.2,1,0
3,0.3,0,1
```

### Submission Files (`sub_*.csv`)
**Required columns**: `id`, `pred`

```csv
id,pred
1,0.15
2,0.25
3,0.35
```

## ⚙️ Configuration Options

### CLI Flags
- `--oof_dir`: Directory containing OOF CSV files
- `--sub_dir`: Directory containing submission CSV files
- `--out`: Output directory for results
- `--metric`: Evaluation metric (`auc`, `mse`, `mae`) [default: `auc`]
- `--target_col`: Target column name [default: `target`]
- `--methods`: Comma-separated list of blending methods
- `--decorrelate`: Enable decorrelation (`on`/`off`) [default: `off`]
- `--stacking`: Enable stacking (`lr`/`ridge`/`none`) [default: `none`]
- `--search`: Weight search parameters (`iters=N,restarts=M`)
- `--seed`: Random seed for reproducibility
- `--time-col`: Time column name for time-sliced analysis
- `--freq`: Time frequency (`M`/`W`/`D`) [default: `M`]
- `--export`: Export format (`pdf`/`none`) [default: `none`]
- `--summary-json`: Path to save blend summary JSON
- `--n-jobs`: Number of parallel jobs (-1 for all CPUs) [default: -1]
- `--memory-cap`: Memory cap in MB [default: 4096]
- `--strategy`: Blending strategy (`auto`/`mean`/`weighted`/`decorrelate_weighted`) [default: `mean`]

### Exit Codes
- `0`: Success - Improvement detected
- `2`: Success with warnings - Unstable or redundant models detected
- `3`: No improvement - Ensemble not better than best single model
- `4`: Invalid input or configuration

## 📈 Performance Benchmarks

- **200k rows × 8 models**: Completes in 1-5 minutes
- **Memory usage**: Configurable cap, default 4GB
- **Parallel processing**: Multi-core optimization support
- **Data type optimization**: 50%+ memory reduction

## 🔧 Installation

```bash
# From PyPI (coming soon)
pip install crediblend

# From source
git clone https://github.com/li147852xu/crediblend.git
cd crediblend
pip install -e .

# With development dependencies
pip install -e .[dev]
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_api.py -v          # API tests
pytest tests/test_contracts.py -v    # Contract stability tests
pytest tests/perf/ -v                # Performance tests (slow)

# Run with coverage
pytest tests/ --cov=src/crediblend --cov-report=html
```

## 📚 API Reference

### Core Functions

#### `fit_blend(oof_frames, method="mean", config=None, **kwargs) -> BlendModel`
Fit a blending model on OOF data.

**Parameters:**
- `oof_frames`: List of OOF DataFrames
- `method`: Blending method (`mean`, `rank_mean`, `logit_mean`, `weighted`, `stacking`, `best_single`)
- `config`: Optional BlendConfig object
- `**kwargs`: Additional configuration parameters

**Returns:**
- `BlendModel`: Trained blending model

#### `predict_blend(model, sub_frames) -> BlendResult`
Generate predictions using a trained model.

**Parameters:**
- `model`: Trained BlendModel
- `sub_frames`: List of submission DataFrames

**Returns:**
- `BlendResult`: Predictions and metadata

#### `quick_blend(oof_frames, sub_frames, method="mean", **kwargs) -> BlendResult`
Quick one-step blending without explicit model fitting.

#### `search_weights(oof_frames, sub_frames, metric="auc", **kwargs) -> Tuple[Dict, float, Dict]`
Search for optimal ensemble weights.

### Configuration Classes

#### `BlendConfig`
Configuration for blending operations.

```python
config = BlendConfig(
    method="weighted",
    metric="auc",
    target_col="target",
    random_state=42,
    decorrelate=True,
    correlation_threshold=0.8,
    stacking="lr",
    weight_search=True,
    search_params={"iters": 200, "restarts": 16}
)
```

## 🐳 Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml ./
RUN pip install -e .

COPY src/ ./src/
COPY examples/ ./examples/

USER crediblend
ENTRYPOINT ["crediblend"]
CMD ["--help"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [pandas](https://pandas.pydata.org/), [scikit-learn](https://scikit-learn.org/), and [pydantic](https://pydantic.dev/)
- Inspired by the need for robust ensemble methods in tabular ML competitions
- Special thanks to the open-source ML community

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/li147852xu/crediblend/issues)
- **Discussions**: [GitHub Discussions](https://github.com/li147852xu/crediblend/discussions)
- **Documentation**: [GitHub Wiki](https://github.com/li147852xu/crediblend/wiki)

---

<div align="center">

**CrediBlend** - Making ensemble learning fast, reliable, and production-ready 🚀

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/li147852xu/crediblend/workflows/CI/badge.svg)](https://github.com/li147852xu/crediblend/actions)
[![Coverage](https://codecov.io/gh/li147852xu/crediblend/branch/main/graph/badge.svg)](https://codecov.io/gh/li147852xu/crediblend)

</div>