#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征构建和预处理脚本
根据AAPL股票数据列解释文档，对AAPL和VGT数据进行特征构建和预处理
"""

import pandas as pd
import numpy as np
import talib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def calculate_basic_features(df):
    """
    计算基础特征：turnover, amplitude, pct_change, change_amount, turnover_rate
    """
    df = df.copy()
    
    # 计算成交额 turnover = volume × close
    df['turnover'] = df['volume'] * df['close']
    
    # 计算振幅 amplitude = ((high - low) / close) × 100
    df['amplitude'] = ((df['high'] - df['low']) / df['close']) * 100
    
    # 计算涨跌幅 pct_change = ((close - open) / open) × 100
    df['pct_change'] = ((df['close'] - df['open']) / df['open']) * 100
    
    # 计算涨跌额 change_amount = close - open
    df['change_amount'] = df['close'] - df['open']
    
    # 计算换手率 turnover_rate (假设流通股本为总股本的80%)
    # 这里使用一个估算值，实际应用中需要真实的流通股本数据
    estimated_shares_outstanding = df['volume'].rolling(window=252).mean() * 10  # 估算
    df['turnover_rate'] = (df['volume'] / estimated_shares_outstanding) * 100
    
    return df

def calculate_technical_indicators(df):
    """
    计算技术指标
    """
    df = df.copy()
    
    # 确保数据按日期排序
    df = df.sort_values('date').reset_index(drop=True)
    
    # 提取价格和成交量数据
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    
    # 移动平均线指标
    df['SMA_20'] = talib.SMA(close, timeperiod=20)
    df['SMA_50'] = talib.SMA(close, timeperiod=50)
    df['EMA_12'] = talib.EMA(close, timeperiod=12)
    df['EMA_26'] = talib.EMA(close, timeperiod=26)
    df['EMA_50'] = talib.EMA(close, timeperiod=50)
    
    # MACD指标
    macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_Signal'] = macd_signal
    df['MACD_Histogram'] = macd_hist
    
    # RSI指标
    df['RSI'] = talib.RSI(close, timeperiod=14)
    
    # 布林带指标
    bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
    df['BB_Upper'] = bb_upper
    df['BB_Middle'] = bb_middle
    df['BB_Lower'] = bb_lower
    
    # 随机振荡器
    stoch_k, stoch_d = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
    df['Stoch_K'] = stoch_k
    df['Stoch_D'] = stoch_d
    
    # 威廉指标
    df['Williams_R'] = talib.WILLR(high, low, close, timeperiod=14)
    
    # ATR指标
    df['ATR'] = talib.ATR(high, low, close, timeperiod=14)
    
    # 成交量移动平均线
    df['Volume_SMA'] = talib.SMA(volume.astype(float), timeperiod=20)
    
    # 价格变化指标
    df['Price_Change'] = df['close'].pct_change()
    df['Price_Change_5d'] = df['close'].pct_change(periods=5)
    df['Price_Change_10d'] = df['close'].pct_change(periods=10)
    
    # 20日波动率
    df['Volatility_20d'] = df['Price_Change'].rolling(window=20).std()
    
    return df

def handle_null_values(df):
    """
    将所有空值填充为字符串 "null"
    """
    df = df.copy()
    
    # 将所有NaN值替换为字符串 "null"
    df = df.fillna('null')
    
    # 确保日期列保持为字符串格式
    df['date'] = df['date'].astype(str)
    
    return df

def process_data(file_path, output_path):
    """
    处理单个数据文件
    """
    print(f"处理文件: {file_path}")
    
    # 读取数据
    df = pd.read_csv(file_path)
    print(f"原始数据形状: {df.shape}")
    
    # 计算基础特征
    df = calculate_basic_features(df)
    print("基础特征计算完成")
    
    # 计算技术指标
    df = calculate_technical_indicators(df)
    print("技术指标计算完成")
    
    # 处理空值
    df = handle_null_values(df)
    print("空值处理完成")
    
    # 保存处理后的数据
    df.to_csv(output_path, index=False)
    print(f"处理后的数据已保存到: {output_path}")
    print(f"处理后数据形状: {df.shape}")
    
    return df

def main():
    """
    主函数
    """
    print("开始特征构建和预处理...")
    
    # 处理AAPL数据
    aapl_input = "Data/stock_105.AAPL_2025-09.csv"
    aapl_output = "Data/processed_stock_105.AAPL_2025-09.csv"
    
    print("\n" + "="*50)
    print("处理AAPL数据")
    print("="*50)
    aapl_df = process_data(aapl_input, aapl_output)
    
    # 处理VGT数据
    vgt_input = "Data/etf_VGT_2025-09.csv"
    vgt_output = "Data/processed_etf_VGT_2025-09.csv"
    
    print("\n" + "="*50)
    print("处理VGT数据")
    print("="*50)
    vgt_df = process_data(vgt_input, vgt_output)
    
    print("\n" + "="*50)
    print("处理完成!")
    print("="*50)
    
    # 显示处理结果摘要
    print(f"\nAAPL数据摘要:")
    print(f"- 数据形状: {aapl_df.shape}")
    print(f"- 列数: {len(aapl_df.columns)}")
    print(f"- 空值数量: {(aapl_df == 'null').sum().sum()}")
    
    print(f"\nVGT数据摘要:")
    print(f"- 数据形状: {vgt_df.shape}")
    print(f"- 列数: {len(vgt_df.columns)}")
    print(f"- 空值数量: {(vgt_df == 'null').sum().sum()}")
    
    # 显示列名对比
    print(f"\n列名对比:")
    print(f"AAPL列数: {len(aapl_df.columns)}")
    print(f"VGT列数: {len(vgt_df.columns)}")
    print(f"列名是否一致: {list(aapl_df.columns) == list(vgt_df.columns)}")

if __name__ == "__main__":
    main()
