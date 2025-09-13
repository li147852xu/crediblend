# CrediBlend / 可信混合

A minimal CLI tool for blending machine learning predictions.  
一个用于混合机器学习预测的最小化CLI工具。

## Features / 功能特性

### v0.1 - Basic Blending / 基础混合
- **Multiple Blending Methods / 多种混合方法**: Mean, rank-based mean, and logit-space mean blending
- **OOF Evaluation / OOF评估**: Out-of-fold metrics with per-fold analysis
- **Flexible Metrics / 灵活指标**: Support for AUC, MSE, and MAE
- **HTML Reports / HTML报告**: Beautiful, comprehensive reports with Jinja2 templates
- **ID Alignment / ID对齐**: Automatic alignment of submission files by ID

### v0.2 - Advanced Ensemble / 高级集成
- **Decorrelation / 去相关**: Hierarchical clustering to remove redundant models
- **Stacking / 堆叠**: LogisticRegression and Ridge meta-learners
- **Weight Optimization / 权重优化**: Parallel random restarts with coordinate descent
- **Visualizations / 可视化**: Correlation heatmaps, weight plots, and performance charts
- **Bilingual Reports / 双语报告**: Chinese-English interface and documentation

### v0.3 - Time-Sliced Diagnostics / 时间切片诊断
- **Time-Sliced Analysis / 时间切片分析**: Per-window AUC computation and trend analysis
- **Stability Diagnostics / 稳定性诊断**: Model stability scoring and instability detection
- **Dominance Analysis / 主导性分析**: Identify models that dominate across time windows
- **Leakage Detection / 泄露检测**: Flag models with suspiciously high performance
- **Windowed Visualizations / 窗口可视化**: Time series charts and stability heatmaps

### v0.4 - CLI Polish & Reproducibility / CLI优化与可重现性
- **Strict Input Validation / 严格输入验证**: Comprehensive schema validation with descriptive errors
- **Deterministic Runs / 确定性运行**: Reproducible results with random seed control
- **Metadata Tracking / 元数据跟踪**: Complete run information in meta.json
- **PDF Export / PDF导出**: Professional PDF reports with WeasyPrint
- **Blend Summary / 混合摘要**: Top-3 methods and weights in JSON format
- **Enhanced Error Handling / 增强错误处理**: Clear, actionable error messages

## Installation / 安装

```bash
pip install -e .
```

## Usage / 使用方法

### Basic Usage / 基础用法
```bash
crediblend --oof_dir examples --sub_dir examples --out runs/demo
```

### Advanced Usage / 高级用法
```bash
# v0.2 - Advanced Ensemble / 高级集成
crediblend --oof_dir examples --sub_dir examples --out runs/v02 \
  --decorrelate on --stacking lr --search iters=200,restarts=8 --seed 42

# v0.3 - Time-Sliced Analysis / 时间切片分析
crediblend --oof_dir examples --sub_dir examples --out runs/v03 \
  --time-col date --freq M --decorrelate on --stacking lr

# v0.4 - CLI Polish & Reproducibility / CLI优化与可重现性
crediblend --oof_dir examples --sub_dir examples --out runs/v04 \
  --export pdf --summary-json runs/v04/blend_summary.json --seed 123
```

### Options / 选项

- `--oof_dir`: Directory containing OOF CSV files (format: `oof_*.csv`) / OOF CSV文件目录
- `--sub_dir`: Directory containing submission CSV files (format: `sub_*.csv`) / 提交文件目录
- `--out`: Output directory for results / 输出结果目录
- `--metric`: Metric to use for evaluation (`auc`, `mse`, `mae`) [default: `auc`] / 评估指标
- `--target_col`: Name of target column in OOF files [default: `target`] / 目标列名称
- `--methods`: Comma-separated list of blending methods / 混合方法列表
- `--decorrelate`: Enable decorrelation via clustering (`on`/`off`) [default: `off`] / 启用去相关
- `--stacking`: Enable stacking with meta-learner (`lr`/`ridge`/`none`) [default: `none`] / 启用堆叠
- `--search`: Weight search parameters (`iters=N,restarts=M`) / 权重搜索参数
- `--seed`: Random seed for reproducibility / 随机种子
- `--time-col`: Time column name for time-sliced analysis (e.g., `date`) / 时间列名称
- `--freq`: Time frequency for windowing (`M`/`W`/`D`) [default: `M`] / 时间窗口频率

## File Formats / 文件格式

### OOF Files (`oof_*.csv`) / OOF文件
**Required columns / 必需列**: `id`, `pred`  
**Optional columns / 可选列**: `target`, `fold`, `{time_col}`

```csv
id,pred,target,fold,date
1,0.65,1,0,2023-01-01
2,0.32,0,0,2023-01-02
...
```

### Submission Files (`sub_*.csv`) / 提交文件
**Required columns / 必需列**: `id`, `pred`

```csv
id,pred
1,0.68
2,0.29
...
```

**Schema Validation (v0.4+) / 模式验证**:
- All required columns must be present and numeric / 所有必需列必须存在且为数值类型
- No missing values in required columns / 必需列不能有缺失值
- Time columns must be parseable as datetime / 时间列必须可解析为日期时间
- Unexpected columns generate warnings / 意外列会生成警告

**Note / 注意**: For time-sliced analysis, include a time column (e.g., `date`) in your OOF files. The time column should be parseable by pandas (e.g., `YYYY-MM-DD` format). / 对于时间切片分析，请在OOF文件中包含时间列（如`date`）。时间列应能被pandas解析（如`YYYY-MM-DD`格式）。

## Output Files / 输出文件

- `best_submission.csv`: Best blended predictions / 最佳混合预测
- `methods.csv`: Model performance comparison table / 模型性能对比表
- `report.html`: Comprehensive HTML report / 综合HTML报告
- `report.pdf`: PDF version of report (v0.4+) / PDF版本报告
- `meta.json`: Run metadata and configuration (v0.4+) / 运行元数据和配置
- `blend_summary.json`: Top-3 methods and weights summary (v0.4+) / 前3名方法和权重摘要
- `weights.json`: Optimized ensemble weights / 优化集成权重
- `stacking_coefficients.json`: Stacking meta-learner coefficients / 堆叠元学习器系数
- `decorrelation_info.json`: Decorrelation analysis results / 去相关分析结果
- `window_metrics.csv`: Time-sliced AUC metrics (v0.3+) / 时间切片AUC指标

## Advanced Features / 高级功能

### Decorrelation / 去相关
Removes redundant models using hierarchical clustering on Spearman correlation matrix.  
使用层次聚类在Spearman相关性矩阵上移除冗余模型。

### Stacking / 堆叠
Uses meta-learners (LogisticRegression/Ridge) to combine base model predictions.  
使用元学习器（逻辑回归/岭回归）组合基础模型预测。

### Weight Optimization / 权重优化
Optimizes ensemble weights using parallel random restarts and coordinate descent.  
使用并行随机重启和坐标下降优化集成权重。

### Time-Sliced Analysis / 时间切片分析
Analyzes model performance across time windows to detect stability issues and potential data leakage.  
分析模型在不同时间窗口的性能，检测稳定性问题和潜在的数据泄露。

**Features / 功能**:
- **Windowed AUC**: Compute AUC for each time window / 计算每个时间窗口的AUC
- **Stability Scoring**: Measure model consistency across time / 测量模型在时间上的一致性
- **Dominance Detection**: Identify models that dominate specific periods / 识别在特定时期占主导的模型
- **Leakage Flags**: Flag suspiciously high performance models / 标记性能异常高的模型

## Development / 开发

```bash
# Install in development mode / 开发模式安装
pip install -e .

# Run tests / 运行测试
pytest -q

# Run example / 运行示例
crediblend --oof_dir examples --sub_dir examples --out runs/demo

# Run advanced example / 运行高级示例
crediblend --oof_dir examples --sub_dir examples --out runs/v02 \
  --decorrelate on --stacking lr --search iters=200,restarts=8

# Run time-sliced analysis / 运行时间切片分析
crediblend --oof_dir examples --sub_dir examples --out runs/v03 \
  --time-col date --freq M --decorrelate on --stacking lr
```

## License / 许可证

MIT
