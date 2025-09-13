# CrediBlend / 可信混合

> A fast, reproducible ensembling toolkit for tabular ML: merge multiple OOF/SUB predictions, de-correlate models, and search optimal blends with solid diagnostics.  
> 一个快速、可重现的表格机器学习集成工具包：合并多个OOF/SUB预测，去相关模型，并通过可靠的诊断搜索最优混合。

## Why CrediBlend? / 为什么选择CrediBlend？

- **OOF ensembles often inflate or collapse** due to correlation and unstable folds. CrediBlend quantifies correlation, filters redundancy, and searches robust weights/stacking.  
- **OOF集成经常因相关性和不稳定的折数而膨胀或崩溃**。CrediBlend量化相关性，过滤冗余，并搜索鲁棒的权重/堆叠。

- **Production-ready** with parallel processing, memory optimization, and stable APIs for CI/CD integration.  
- **生产就绪**，具有并行处理、内存优化和用于CI/CD集成的稳定API。

- **Comprehensive diagnostics** with time-sliced analysis, stability scoring, and visual reports.  
- **全面的诊断**，包括时间切片分析、稳定性评分和可视化报告。

## What it does / 功能特性

- **Strategies**: mean / rank_mean / logit_mean / stacking (LR/Ridge) / weight search  
- **策略**：均值 / 排名均值 / 逻辑均值 / 堆叠（逻辑回归/岭回归）/ 权重搜索

- **Diagnostics**: Spearman heatmap, clustering, time-sliced AUC, stability score  
- **诊断**：Spearman热图、聚类、时间切片AUC、稳定性评分

- **Outputs**: best submission, method leaderboard, blend_summary.json, HTML/PDF report  
- **输出**：最佳提交、方法排行榜、blend_summary.json、HTML/PDF报告

- **DX**: CLI & Python SDK, deterministic runs via meta.json, CI-ready exit codes  
- **开发体验**：CLI和Python SDK、通过meta.json的确定性运行、CI就绪的退出代码

## Quickstart / 快速开始

```bash
pip install crediblend
crediblend --oof_dir path/to/oof --sub_dir path/to/sub --out runs/demo
```

## 🚀 Features / 功能特性

### Core Blending Methods / 核心混合方法
- **Mean Blending**: Simple arithmetic mean of predictions  
- **均值混合**：预测的简单算术平均

- **Rank-based Blending**: Mean of rank-transformed predictions  
- **基于排名的混合**：排名转换预测的均值

- **Logit-space Blending**: Mean in logit space for probability predictions  
- **逻辑空间混合**：概率预测在逻辑空间中的均值

- **Weight Optimization**: Parallel search for optimal ensemble weights  
- **权重优化**：并行搜索最优集成权重

- **Stacking**: Meta-learning with LogisticRegression/Ridge  
- **堆叠**：使用逻辑回归/岭回归的元学习

### Advanced Diagnostics / 高级诊断
- **Correlation Analysis**: Spearman correlation matrix and hierarchical clustering  
- **相关性分析**：Spearman相关性矩阵和层次聚类

- **Time-sliced Evaluation**: Per-window AUC analysis for temporal stability  
- **时间切片评估**：用于时间稳定性的每窗口AUC分析

- **Stability Scoring**: Standard deviation and IQR of windowed metrics  
- **稳定性评分**：窗口化指标的标准差和四分位距

- **Dominance Detection**: Identify models that dominate across time windows  
- **主导性检测**：识别在时间窗口中占主导地位的模型

- **Leakage Hints**: Flag models with suspiciously high performance  
- **泄露提示**：标记性能异常高的模型

### Performance & Production / 性能与生产
- **Parallel Processing**: Multi-core optimization with joblib  
- **并行处理**：使用joblib的多核优化

- **Memory Optimization**: Automatic dtype optimization and chunked reading  
- **内存优化**：自动数据类型优化和分块读取

- **Auto Strategy**: Intelligent strategy selection based on data characteristics  
- **自动策略**：基于数据特征的智能策略选择

- **Docker Support**: Production-ready containerization  
- **Docker支持**：生产就绪的容器化

- **CI/CD Integration**: Meaningful exit codes and stable contracts  
- **CI/CD集成**：有意义的退出代码和稳定合约

## 📊 Usage Examples / 使用示例

### Command Line Interface / 命令行界面

```bash
# Basic usage / 基础用法
crediblend --oof_dir data/oof --sub_dir data/sub --out results

# Advanced features / 高级功能
crediblend --oof_dir data/oof --sub_dir data/sub --out results \
  --decorrelate on --stacking lr --search iters=200,restarts=16 --seed 42

# Performance optimized / 性能优化
crediblend --oof_dir data/oof --sub_dir data/sub --out results \
  --n-jobs 8 --memory-cap 4096 --strategy auto --seed 42

# Time-sliced analysis / 时间切片分析
crediblend --oof_dir data/oof --sub_dir data/sub --out results \
  --time-col date --freq M --decorrelate on

# PDF export with summary / PDF导出与摘要
crediblend --oof_dir data/oof --sub_dir data/sub --out results \
  --export pdf --summary-json results/blend_summary.json --seed 123
```

### Python API / Python API

```python
from crediblend.api import fit_blend, predict_blend, quick_blend
import pandas as pd

# Load your data / 加载数据
oof_data = [pd.read_csv('oof_model1.csv'), pd.read_csv('oof_model2.csv')]
sub_data = [pd.read_csv('sub_model1.csv'), pd.read_csv('sub_model2.csv')]

# Quick blending / 快速混合
result = quick_blend(oof_data, sub_data, method='mean')
print(result.predictions)

# Advanced blending with configuration / 高级混合配置
from crediblend.api import BlendConfig
config = BlendConfig(method='weighted', metric='auc', random_state=42)
model = fit_blend(oof_data, config=config)
result = predict_blend(model, sub_data)

# Weight optimization / 权重优化
from crediblend.api import search_weights
weights, score, info = search_weights(oof_data, sub_data, metric='auc')
print(f"Best score: {score:.4f}")
print(f"Weights: {weights}")
```

### Docker Usage / Docker使用

```bash
# Build image / 构建镜像
docker build -t crediblend .

# Run with data mounted / 挂载数据运行
docker run -v $(pwd)/data:/data -v $(pwd)/results:/results \
  crediblend --oof_dir /data/oof --sub_dir /data/sub --out /results

# Performance-optimized run / 性能优化运行
docker run -v $(pwd)/data:/data -v $(pwd)/results:/results \
  crediblend --oof_dir /data/oof --sub_dir /data/sub --out /results \
  --n-jobs 4 --memory-cap 2048 --strategy auto
```

## 📁 File Formats / 文件格式

### OOF Files (`oof_*.csv`) / OOF文件
**Required columns / 必需列**: `id`, `pred`  
**Optional columns / 可选列**: `target`, `fold`, `{time_col}`

```csv
id,pred,target,fold
1,0.1,0,0
2,0.2,1,0
3,0.3,0,1
```

### Submission Files (`sub_*.csv`) / 提交文件
**Required columns / 必需列**: `id`, `pred`

```csv
id,pred
1,0.15
2,0.25
3,0.35
```

## ⚙️ Configuration Options / 配置选项

### CLI Flags / CLI标志
- `--oof_dir`: Directory containing OOF CSV files / 包含OOF CSV文件的目录
- `--sub_dir`: Directory containing submission CSV files / 包含提交CSV文件的目录
- `--out`: Output directory for results / 结果输出目录
- `--metric`: Evaluation metric (`auc`, `mse`, `mae`) [default: `auc`] / 评估指标
- `--target_col`: Target column name [default: `target`] / 目标列名称
- `--methods`: Comma-separated list of blending methods / 混合方法列表（逗号分隔）
- `--decorrelate`: Enable decorrelation (`on`/`off`) [default: `off`] / 启用去相关
- `--stacking`: Enable stacking (`lr`/`ridge`/`none`) [default: `none`] / 启用堆叠
- `--search`: Weight search parameters (`iters=N,restarts=M`) / 权重搜索参数
- `--seed`: Random seed for reproducibility / 随机种子
- `--time-col`: Time column name for time-sliced analysis / 时间切片分析的时间列名称
- `--freq`: Time frequency (`M`/`W`/`D`) [default: `M`] / 时间频率
- `--export`: Export format (`pdf`/`none`) [default: `none`] / 导出格式
- `--summary-json`: Path to save blend summary JSON / 混合摘要JSON保存路径
- `--n-jobs`: Number of parallel jobs (-1 for all CPUs) [default: -1] / 并行作业数
- `--memory-cap`: Memory cap in MB [default: 4096] / 内存限制（MB）
- `--strategy`: Blending strategy (`auto`/`mean`/`weighted`/`decorrelate_weighted`) [default: `mean`] / 混合策略

### Exit Codes / 退出代码
- `0`: Success - Improvement detected / 成功 - 检测到改进
- `2`: Success with warnings - Unstable or redundant models detected / 成功但警告 - 检测到不稳定或冗余模型
- `3`: No improvement - Ensemble not better than best single model / 无改进 - 集成不比最佳单模型好
- `4`: Invalid input or configuration / 无效输入或配置

## 📈 Performance Benchmarks / 性能基准

- **200k rows × 8 models**: Completes in 1-5 minutes / 20万行×8个模型：1-5分钟内完成
- **Memory usage**: Configurable cap, default 4GB / 内存使用：可配置上限，默认4GB
- **Parallel processing**: Multi-core optimization support / 并行处理：多核优化支持
- **Data type optimization**: 50%+ memory reduction / 数据类型优化：50%+内存减少

## 🔧 Installation / 安装

```bash
# From PyPI (coming soon) / 从PyPI安装（即将推出）
pip install crediblend

# From source / 从源码安装
git clone https://github.com/li147852xu/crediblend.git
cd crediblend
pip install -e .

# With development dependencies / 安装开发依赖
pip install -e .[dev]
```

## 🧪 Testing / 测试

```bash
# Run all tests / 运行所有测试
pytest tests/ -v

# Run specific test categories / 运行特定测试类别
pytest tests/test_api.py -v          # API tests / API测试
pytest tests/test_contracts.py -v    # Contract stability tests / 合约稳定性测试
pytest tests/perf/ -v                # Performance tests (slow) / 性能测试（慢）

# Run with coverage / 运行覆盖率测试
pytest tests/ --cov=src/crediblend --cov-report=html
```

## 📚 API Reference / API参考

### Core Functions / 核心函数

#### `fit_blend(oof_frames, method="mean", config=None, **kwargs) -> BlendModel`
Fit a blending model on OOF data. / 在OOF数据上拟合混合模型。

**Parameters / 参数:**
- `oof_frames`: List of OOF DataFrames / OOF DataFrame列表
- `method`: Blending method (`mean`, `rank_mean`, `logit_mean`, `weighted`, `stacking`, `best_single`) / 混合方法
- `config`: Optional BlendConfig object / 可选的BlendConfig对象
- `**kwargs`: Additional configuration parameters / 额外配置参数

**Returns / 返回:**
- `BlendModel`: Trained blending model / 训练好的混合模型

#### `predict_blend(model, sub_frames) -> BlendResult`
Generate predictions using a trained model. / 使用训练好的模型生成预测。

**Parameters / 参数:**
- `model`: Trained BlendModel / 训练好的BlendModel
- `sub_frames`: List of submission DataFrames / 提交DataFrame列表

**Returns / 返回:**
- `BlendResult`: Predictions and metadata / 预测和元数据

#### `quick_blend(oof_frames, sub_frames, method="mean", **kwargs) -> BlendResult`
Quick one-step blending without explicit model fitting. / 无需显式模型拟合的快速一步混合。

#### `search_weights(oof_frames, sub_frames, metric="auc", **kwargs) -> Tuple[Dict, float, Dict]`
Search for optimal ensemble weights. / 搜索最优集成权重。

### Configuration Classes / 配置类

#### `BlendConfig`
Configuration for blending operations. / 混合操作的配置。

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

## 🐳 Docker / Docker

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

## 🤝 Contributing / 贡献

1. Fork the repository / Fork仓库
2. Create a feature branch (`git checkout -b feature/amazing-feature`) / 创建功能分支
3. Commit your changes (`git commit -m 'Add amazing feature'`) / 提交更改
4. Push to the branch (`git push origin feature/amazing-feature`) / 推送到分支
5. Open a Pull Request / 打开Pull Request

## 📄 License / 许可证

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.  
本项目采用MIT许可证 - 详见[LICENSE](LICENSE)文件。

## 🙏 Acknowledgments / 致谢

- Built with [pandas](https://pandas.pydata.org/), [scikit-learn](https://scikit-learn.org/), and [pydantic](https://pydantic.dev/) / 基于pandas、scikit-learn和pydantic构建
- Inspired by the need for robust ensemble methods in tabular ML competitions / 受表格ML竞赛中对鲁棒集成方法需求的启发
- Special thanks to the open-source ML community / 特别感谢开源ML社区

## 📞 Support / 支持

- **Issues**: [GitHub Issues](https://github.com/li147852xu/crediblend/issues) / 问题反馈
- **Discussions**: [GitHub Discussions](https://github.com/li147852xu/crediblend/discussions) / 讨论
- **Documentation**: [GitHub Wiki](https://github.com/li147852xu/crediblend/wiki) / 文档

---

<div align="center">

**CrediBlend** - Making ensemble learning fast, reliable, and production-ready 🚀  
**CrediBlend** - 让集成学习快速、可靠、生产就绪 🚀

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/li147852xu/crediblend/workflows/CI/badge.svg)](https://github.com/li147852xu/crediblend/actions)
[![Coverage](https://codecov.io/gh/li147852xu/crediblend/branch/main/graph/badge.svg)](https://codecov.io/gh/li147852xu/crediblend)

</div>