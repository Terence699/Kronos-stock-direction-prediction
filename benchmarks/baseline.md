# 📋 模块化基准模型对比系统 - 文件功能大纲

## 🏗️ 核心架构文件

### 1. **benchmark_models.py** - 基准模型实现库

**主要功能**: 实现各种类型的预测模型

- **技术分析模型** (8个)
  - `MovingAverageCrossModel` - 移动平均线交叉策略
  - `RSIModel` - RSI超买超卖策略
  - `MACDModel` - MACD信号策略
  - `BollingerBandsModel` - 布林带策略
  - `MomentumModel` - 动量策略
- **机器学习模型** (5个)
  - `RandomForestModel` - 随机森林
  - `GradientBoostingModel` - 梯度提升
  - `SVMModel` - 支持向量机
  - `LogisticRegressionModel` - 逻辑回归
  - `XGBoostModel` - XGBoost
- **深度学习模型** (2个)
  - `LSTMModel` - LSTM神经网络
  - `GRUModel` - GRU神经网络

### 2. **model_factory.py** - 模型工厂和管理器

**主要功能**: 统一创建、管理和比较所有模型

- **ModelFactory类**
  - 注册所有可用模型
  - 按类别组织模型
  - 创建模型实例
  - 管理模型生命周期
- **ModelComparator类**
  - 比较多个模型性能
  - 生成性能排名
  - 输出比较报告

### 3. **comprehensive_evaluator.py** - 综合评估框架

**主要功能**: 对所有模型进行统一评估和排名

- **ComprehensiveModelEvaluator类**
  - 加载和准备数据
  - 评估单个模型
  - 评估所有模型
  - 生成性能可视化
  - 模型排名和选择
  - 保存评估结果

## 🚀 执行和测试文件

### 4. **main_benchmark_comparison.py** - 主执行脚本

**主要功能**: 一键运行完整的模型对比流程

- **执行步骤**:
  1. 测试基准模型
  2. 测试模型工厂
  3. 运行综合评估
  4. 运行集成优化
  5. 生成最终报告
- **输出**: 完整的对比报告和可视化

### 5. **test_benchmark_system.py** - 快速测试脚本

**主要功能**: 验证系统各组件是否正常工作

- **测试内容**:
  - 导入测试
  - 模型创建测试
  - 数据加载测试
  - 模型评估测试
- **用途**: 在运行完整流程前验证系统状态

## 📚 文档文件

### 6. **README_benchmark_system.md** - 使用说明文档

**主要功能**: 详细的系统使用说明

- 系统架构说明
- 使用方法指南
- API文档
- 扩展指南
- 故障排除

## 🔄 与现有系统的集成

### 7. **score_ensemble.py** - 现有集成分析模块

**主要功能**: 现有的模型集成和权重优化

- 集成Kronos AAPL、VGT和情感分析模型
- 网格搜索最优权重
- 生成集成结果

### 8. **main_ensemble_analysis.py** - 现有主执行脚本

**主要功能**: 运行现有的集成分析流程

## 📊 数据文件

### 9. **data/** 目录

- `stock_105.AAPL_2025-09.csv` - AAPL股票数据
- `etf_VGT_2025-09.csv` - VGT ETF数据
- `tweets_105.AAPL_2025-09.csv` - AAPL推文数据

### 10. **result/** 目录 (输出)

- `comprehensive_evaluation_results.pkl` - 详细评估结果
- `evaluation_summary.txt` - 评估摘要
- `comprehensive_benchmark_report.txt` - 综合报告
- `comprehensive_model_comparison.png` - 模型对比图
- `category_performance_analysis.png` - 类别性能分析
- `horizon_comparison_analysis.png` - 时间跨度对比
- `performance_distribution.png` - 性能分布

## 🎯 使用流程

### 快速开始

```bash
# 1. 测试系统
python test_benchmark_system.py

# 2. 运行完整对比
python main_benchmark_comparison.py
```

### 分步使用

```python
# 1. 创建模型工厂
from model_factory import ModelFactory
factory = ModelFactory()

# 2. 运行评估
from comprehensive_evaluator import ComprehensiveModelEvaluator
evaluator = ComprehensiveModelEvaluator()
results = evaluator.evaluate_all_models()
```

## 🔧 扩展性

- **添加新模型**: 在 `benchmark_models.py` 中实现，在 `model_factory.py` 中注册
- **自定义评估**: 在 `comprehensive_evaluator.py` 中添加新指标
- **新可视化**: 在评估框架中添加新的图表类型

这个系统设计为完全模块化，每个文件都有明确的职责，可以独立使用也可以组合使用！
