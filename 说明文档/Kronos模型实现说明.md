# Kronos模型程序运行解读

## 📋 **程序运行逻辑简介**

### 🎯 **程序目标**
本程序实现了一个基于技术分析的股票价格方向预测系统，使用Kronos模型对AAPL股票进行1天、3天、5天的价格方向预测，并通过60天滚动窗口验证模型性能。

### 🔄 **核心运行逻辑**
```
📊 数据输入 → 🧮 技术指标计算 → 🔮 多时间跨度预测 → 📈 滚动验证 → 📊 性能评估 → 🎨 可视化输出
```

### ⚡ **关键特点**
1. **统一特征策略**：所有预测时间跨度使用相同的30个OHLCV + 15个技术指标
2. **时间偏置机制**：1天保守、3天中性、5天趋势的不同偏置策略
3. **滚动验证**：每天计算过去60天的预测准确度，观察模型性能变化
4. **技术分析结合**：综合价格动量、EMA交叉、RSI超买超卖等信号
5. **多维度评估**：准确率、精确率、召回率、F1分数、累积收益等指标

### 📊 **运行结果概览**
- **数据规模**：2015-2025年AAPL股票数据，2535条记录
- **预测窗口**：1天、3天、5天三个时间跨度
- **验证方式**：60天滚动窗口，2470次预测
- **性能表现**：1D(50.40%)、3D(51.50%)、5D(53.04%)准确率
- **输出内容**：6张可视化图表 + 详细性能报告

### 🎯 **程序价值**
- **学术研究**：验证技术分析在股票预测中的有效性
- **策略开发**：为量化交易策略提供预测基础
- **风险控制**：通过多时间跨度预测降低单一预测风险
- **性能监控**：实时跟踪模型表现，及时调整策略

---

## 🎯 **程序整体架构**

### 核心文件

- **`kronos_aapl_analysis.py`** - 主程序，包含完整的分析流程
- **`kronos_model.py`** - Kronos模型实现（当前使用简化版本）

### 程序运行流程

```
数据加载 → 技术指标计算 → 滚动分析 → 性能评估 → 可视化生成
```

## 📊 **第一阶段：数据准备与预处理**

### 1. 数据加载

```python
# 加载AAPL股票数据 (2015-2025年，2535条记录)
data = pd.read_csv('data/stock_105.AAPL_2025-09.csv')
data['date'] = pd.to_datetime(data['date'])
```

### 2. 技术指标计算

```python
class TechnicalIndicators:
    def calculate_all_indicators(self):
        # EMA指标
        self.data['EMA_12'] = talib.EMA(close, timeperiod=12)
        self.data['EMA_26'] = talib.EMA(close, timeperiod=26)
      
        # MACD指标
        macd, macd_signal, macd_hist = talib.MACD(close)
      
        # RSI指标
        self.data['RSI'] = talib.RSI(close, timeperiod=14)
      
        # 布林带、随机指标、威廉指标等
```

**关键指标**：EMA_12, EMA_26, MACD, RSI, 布林带, ATR, 随机指标, Williams %R, 波动率

## 🧠 **第二阶段：预测模型核心逻辑**

### 1. 统一特征策略

```python
class KronosPredictor:
    def __init__(self, lookback_window=60):
        # 统一使用30个OHLCV + 15个技术指标
        self.token_strategy = {
            'ohlcv_tokens': 30,
            'tech_tokens': 15
        }
      
        # 不同时间跨度的偏置设置
        self.horizon_bias = {
            1: {'bias': 'conservative', 'factor': -0.2, 'threshold': 0.1},
            3: {'bias': 'neutral', 'factor': 0.0, 'threshold': 0.0},
            5: {'bias': 'trend', 'factor': 0.2, 'threshold': -0.1}
        }
```

### 2. 预测算法核心

```python
def predict_direction(self, features, horizon_days=1):
    base_score = 0
  
    # 1. 价格动量分析
    recent_momentum = np.mean(np.diff(close_prices[-3:]))
    base_score += recent_momentum * 10
  
    # 2. 技术指标分析
    if EMA_12 > EMA_26: base_score += 0.3      # 均线交叉
    if EMA_12上升: base_score += 0.2             # 均线趋势
  
    if RSI < 30: base_score += 0.3              # 超卖信号
    elif 30 < RSI < 70: base_score += 0.1       # 中性区域
    elif RSI > 70: base_score -= 0.2             # 超买信号
  
    # 3. 时间偏置调整
    bias_config = self.horizon_bias[horizon_days]
    final_score = base_score + bias_config['factor']
  
    # 4. 最终预测
    return 1 if final_score > bias_config['threshold'] else 0
```

## 🔄 **第三阶段：60天滚动分析**

### 1. 滚动窗口机制

```python
class RollingAnalysis:
    def run_rolling_analysis(self):
        # 60天窗口，每天向前滚动
        start_idx = 60
        end_idx = len(data) - 5  # 为5天预测预留空间
      
        for i in range(start_idx, end_idx):
            # 取过去60天数据
            features = self.predictor.prepare_features(
                data, i - 60, i
            )
          
            # 预测1天、3天、5天后价格方向
            pred_1d = predictor.predict_direction(features, 1)
            pred_3d = predictor.predict_direction(features, 3)
            pred_5d = predictor.predict_direction(features, 5)
```

### 2. 每日准确度计算

```python
# 每天计算过去60天的预测准确度
if i >= start_idx + 60:
    past_results = self.results[-60:]  # 过去60天结果
  
    for horizon in ['1d', '3d', '5d']:
        actual_values = [r[f'actual_dir_{horizon}'] for r in past_results]
        pred_values = [r[f'pred_dir_{horizon}'] for r in past_results]
        accuracy = accuracy_score(actual_values, pred_values)
        daily_accuracies[f'accuracy_{horizon}'] = accuracy
```

## 📈 **第四阶段：性能评估**

### 1. 关键指标计算

```python
class PerformanceEvaluator:
    def calculate_metrics(self):
        for horizon in ['1d', '3d', '5d']:
            # 准确率、精确率、召回率、F1分数
            accuracy = accuracy_score(actual, predicted)
            precision = precision_score(actual, predicted)
            recall = recall_score(actual, predicted)
            f1 = f1_score(actual, predicted)
          
            # 命中率（方向正确预测的百分比）
            hit_rate = np.mean((predicted_returns > 0) == (actual_returns > 0))
```

### 2. 当前性能表现

| 预测窗口     | 准确率 | 上涨预测 | 下跌预测 | 偏置策略 |
| ------------ | ------ | -------- | -------- | -------- |
| **1D** | 50.40% | 1381     | 1089     | 保守偏置 |
| **3D** | 51.50% | 1426     | 1044     | 中性偏置 |
| **5D** | 53.04% | 1489     | 981      | 趋势偏置 |

## 🎨 **第五阶段：可视化生成**

### 1. 图片生成顺序

```python
def create_all_plots(self):
    # 1. 价格走势 + 技术指标
    self.plot_price_and_indicators()
  
    # 2. 60天滚动准确度变化
    self.plot_prediction_accuracy()
  
    # 3. 混淆矩阵对比
    self.plot_confusion_matrices()
  
    # 4. 性能指标对比
    self.plot_performance_metrics()
  
    # 5. 累积收益对比
    self.plot_cumulative_returns()
  
    # 6. 预测分布统计
    self.plot_prediction_distribution()
```

### 2. 关键图表解读

- **第二张图**：显示每天计算的60天滚动准确度，观察模型性能的时间变化
- **第四张图**：对比1D、3D、5D预测的各项性能指标
- **第五张图**：展示不同策略的累积收益表现

## 🔍 **核心算法特点**

### 1. **统一特征策略**

- 所有预测时间跨度使用相同的特征：30个OHLCV + 15个技术指标
- 便于比较不同偏置策略的效果

### 2. **技术分析结合**

- 价格动量分析（最近3天平均变化）
- EMA交叉信号（12日与26日均线）
- RSI超买超卖信号
- 综合评分机制

### 3. **滚动验证**

- 60天滚动窗口提供稳定的准确度计算
- 每天更新过去60天的表现评估
- 动态观察模型性能变化

## ⚠️ **当前问题与改进方向**

### 1. **交易成本问题**

- 频繁交易导致成本过高
- 需要优化交易频率和止损止盈策略

### 2. **预测精度提升**

- 虽然超过50%随机基线，但不足以克服交易成本
- 可考虑集成更多市场情绪指标

### 3. **策略优化**

- 减少交易频率
- 增加置信度阈值
- 考虑市场状态适应性

## 🚀 **程序运行命令**

```bash
cd /Users/vincentwu/Desktop/BAP
python kronos_aapl_analysis.py
```

**输出结果**：

- 6张可视化图片（results/文件夹）
- 预测结果CSV文件
- 控制台性能报告

这个程序本质上是一个**基于技术分析的多时间跨度预测系统**，通过60天滚动窗口验证和不同偏置策略来适应不同的市场环境。🎯
