# CrediBlend / 可信混合

> Fast, reproducible ensembling toolkit for tabular ML | 表格机器学习的快速可重现集成工具包

[![PyPI version](https://img.shields.io/pypi/v/crediblend.svg)](https://pypi.org/project/crediblend/)
[![Python 3.10+](https://img.shields.io/pypi/pyversions/crediblend.svg)](https://pypi.org/project/crediblend/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI Status](https://github.com/li147852xu/crediblend/workflows/CI/badge.svg)](https://github.com/li147852xu/crediblend/actions)
[![Coverage](https://codecov.io/gh/li147852xu/crediblend/branch/main/graph/badge.svg)](https://codecov.io/gh/li147852xu/crediblend)

**Professional Ensemble Blending with Advanced Diagnostics | 具有高级诊断的专业集成混合**

Detects correlation, optimizes weights, and searches robust blends with time-sliced stability analysis.  
检测相关性、优化权重，并通过时间切片稳定性分析搜索鲁棒混合。

## 🚀 Quick Start / 快速开始

```bash
# Install from PyPI / 从PyPI安装
pip install crediblend

# Basic usage / 基础用法
crediblend --oof_dir data/oof --sub_dir data/sub --out results

# Advanced features / 高级功能
crediblend --oof_dir data/oof --sub_dir data/sub --out results \
  --decorrelate on --stacking lr --search iters=200,restarts=16 --seed 42
```

## 📦 Installation / 安装

```bash
# Standard installation / 标准安装
pip install crediblend

# With PDF export support / 带PDF导出支持
pip install "crediblend[pdf]"

# With performance optimizations / 带性能优化
pip install "crediblend[perf]"

# With all features / 所有功能
pip install "crediblend[pdf,perf]"
```

## ✅ Automatic Dependency Detection / 自动依赖检测

When you install via `pip install crediblend`, pip automatically:  
当您通过`pip install crediblend`安装时，pip会自动：

• Installs all required dependencies (pandas, numpy, scikit-learn, jinja2, etc.)  
• 安装所有必需依赖（pandas、numpy、scikit-learn、jinja2等）

• Resolves version conflicts  
• 解决版本冲突

• Creates proper dependency tree  
• 创建正确的依赖树

• No manual dependency management needed!  
• 无需手动依赖管理！

## 🎯 Core Features / 核心功能

### Blending Methods / 混合方法
• **Mean Blending**: Simple arithmetic mean of predictions  
• **均值混合**：预测的简单算术平均

• **Rank-based Blending**: Mean of rank-transformed predictions  
• **基于排名的混合**：排名转换预测的均值

• **Weight Optimization**: Parallel search for optimal ensemble weights  
• **权重优化**：并行搜索最优集成权重

• **Stacking**: Meta-learning with LogisticRegression/Ridge  
• **堆叠**：使用逻辑回归/岭回归的元学习

### Advanced Diagnostics / 高级诊断
• **Correlation Analysis**: Spearman correlation matrix and hierarchical clustering  
• **相关性分析**：Spearman相关性矩阵和层次聚类

• **Time-sliced Evaluation**: Per-window AUC analysis for temporal stability  
• **时间切片评估**：用于时间稳定性的每窗口AUC分析

• **Stability Scoring**: Standard deviation and IQR of windowed metrics  
• **稳定性评分**：窗口化指标的标准差和四分位距

• **Leakage Detection**: Flag models with suspiciously high performance  
• **泄露检测**：标记性能异常高的模型

### Performance & Production / 性能与生产
• **Parallel Processing**: Multi-core optimization with joblib  
• **并行处理**：使用joblib的多核优化

• **Memory Optimization**: Automatic dtype optimization and chunked reading  
• **内存优化**：自动数据类型优化和分块读取

• **Auto Strategy**: Intelligent strategy selection based on data characteristics  
• **自动策略**：基于数据特征的智能策略选择

• **Docker Support**: Production-ready containerization  
• **Docker支持**：生产就绪的容器化

• **CI/CD Integration**: Meaningful exit codes and stable contracts  
• **CI/CD集成**：有意义的退出代码和稳定合约

## 📊 Usage Examples / 使用示例

### Command Line Interface / 命令行界面

```bash
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
```

### Docker Usage / Docker使用

```bash
# Build and run / 构建并运行
docker build -t crediblend .
docker run -v $(pwd)/data:/data -v $(pwd)/results:/results \
  crediblend --oof_dir /data/oof --sub_dir /data/sub --out /results
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

### Key CLI Flags / 关键CLI标志
• `--oof_dir`: Directory containing OOF CSV files / 包含OOF CSV文件的目录
• `--sub_dir`: Directory containing submission CSV files / 包含提交CSV文件的目录
• `--out`: Output directory for results / 结果输出目录
• `--metric`: Evaluation metric (`auc`, `mse`, `mae`) [default: `auc`] / 评估指标
• `--decorrelate`: Enable decorrelation (`on`/`off`) [default: `off`] / 启用去相关
• `--stacking`: Enable stacking (`lr`/`ridge`/`none`) [default: `none`] / 启用堆叠
• `--search`: Weight search parameters (`iters=N,restarts=M`) / 权重搜索参数
• `--strategy`: Blending strategy (`auto`/`mean`/`weighted`/`decorrelate_weighted`) [default: `mean`] / 混合策略
• `--n-jobs`: Number of parallel jobs (-1 for all CPUs) [default: -1] / 并行作业数
• `--memory-cap`: Memory cap in MB [default: 4096] / 内存限制（MB）

### Exit Codes / 退出代码
• `0`: Success - Improvement detected / 成功 - 检测到改进
• `2`: Success with warnings - Unstable or redundant models detected / 成功但警告 - 检测到不稳定或冗余模型
• `3`: No improvement - Ensemble not better than best single model / 无改进 - 集成不比最佳单模型好
• `4`: Invalid input or configuration / 无效输入或配置

## 📈 Performance Benchmarks / 性能基准

• **200k rows × 8 models**: Completes in 1-5 minutes / 20万行×8个模型：1-5分钟内完成
• **Memory usage**: Configurable cap, default 4GB / 内存使用：可配置上限，默认4GB
• **Parallel processing**: Multi-core optimization support / 并行处理：多核优化支持
• **Data type optimization**: 50%+ memory reduction / 数据类型优化：50%+内存减少

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

## 🔧 Development / 开发

```bash
# Install from source / 从源码安装
git clone https://github.com/li147852xu/crediblend.git
cd crediblend
pip install -e .

# With development dependencies / 安装开发依赖
pip install -e .[dev]
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

## 📞 Support / 支持

• **Issues**: [GitHub Issues](https://github.com/li147852xu/crediblend/issues) / 问题反馈
• **Discussions**: [GitHub Discussions](https://github.com/li147852xu/crediblend/discussions) / 讨论
• **Documentation**: [GitHub Wiki](https://github.com/li147852xu/crediblend/wiki) / 文档

---

<div align="center">

**CrediBlend** - Making ensemble learning fast, reliable, and production-ready 🚀  
**CrediBlend** - 让集成学习快速、可靠、生产就绪 🚀

</div>