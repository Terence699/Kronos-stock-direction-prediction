#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AAPL+VGT双模型 vs AAPL+VGT+情感三模型对比分析系统
生成详细的性能对比图表和报告
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 设置英文字体
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class ModelComparisonAnalyzer:
    """
    双模型与三模型集成对比分析器
    """
    
    def __init__(self):
        """
        初始化对比分析器
        """
        self.dual_results = None
        self.triple_results = None
        self.comparison_metrics = None
        
    def load_model_results(self, dual_path, triple_path):
        """
        加载双模型和三模型的结果数据
        
        Args:
            dual_path: 双模型结果文件路径
            triple_path: 三模型结果文件路径
        """
        print("加载模型结果数据...")
        
        # 加载双模型结果
        self.dual_results = pd.read_csv(dual_path)
        self.dual_results['date'] = pd.to_datetime(self.dual_results['date'])
        
        # 加载三模型结果
        self.triple_results = pd.read_csv(triple_path)
        self.triple_results['date'] = pd.to_datetime(self.triple_results['date'])
        
        print(f"双模型数据形状: {self.dual_results.shape}")
        print(f"三模型数据形状: {self.triple_results.shape}")
        
        # 确保数据对齐
        common_dates = pd.merge(self.dual_results[['date']], 
                              self.triple_results[['date']], 
                              on='date', how='inner')
        
        self.dual_results = pd.merge(common_dates, self.dual_results, on='date')
        self.triple_results = pd.merge(common_dates, self.triple_results, on='date')
        
        print(f"对齐后双模型数据形状: {self.dual_results.shape}")
        print(f"对齐后三模型数据形状: {self.triple_results.shape}")
        
    def calculate_performance_comparison(self):
        """
        计算两个模型的性能对比指标
        """
        print("计算性能对比指标...")
        
        comparison_metrics = {}
        
        for horizon in ['1d', '3d', '5d']:
            # 双模型指标
            dual_pred_col = f'dual_ensemble_pred_{horizon}'
            dual_score_col = f'dual_ensemble_score_{horizon}'
            actual_col = f'actual_{horizon}_aapl'
            actual_return_col = f'actual_return_{horizon}_aapl'
            
            # 三模型指标
            triple_pred_col = f'triple_ensemble_pred_{horizon}'
            triple_score_col = f'triple_ensemble_score_{horizon}'
            
            # 计算双模型性能
            dual_accuracy = accuracy_score(self.dual_results[actual_col], 
                                         self.dual_results[dual_pred_col])
            dual_precision = precision_score(self.dual_results[actual_col], 
                                           self.dual_results[dual_pred_col], zero_division=0)
            dual_recall = recall_score(self.dual_results[actual_col], 
                                     self.dual_results[dual_pred_col], zero_division=0)
            dual_f1 = f1_score(self.dual_results[actual_col], 
                             self.dual_results[dual_pred_col], zero_division=0)
            
            # 计算双模型分数相关性
            try:
                dual_correlation = np.corrcoef(self.dual_results[dual_score_col], 
                                            self.dual_results[actual_return_col])[0, 1]
                if np.isnan(dual_correlation):
                    dual_correlation = 0.0
            except:
                dual_correlation = 0.0
            
            # 计算三模型性能
            triple_accuracy = accuracy_score(self.triple_results[actual_col], 
                                           self.triple_results[triple_pred_col])
            triple_precision = precision_score(self.triple_results[actual_col], 
                                            self.triple_results[triple_pred_col], zero_division=0)
            triple_recall = recall_score(self.triple_results[actual_col], 
                                       self.triple_results[triple_pred_col], zero_division=0)
            triple_f1 = f1_score(self.triple_results[actual_col], 
                               self.triple_results[triple_pred_col], zero_division=0)
            
            # 计算三模型分数相关性
            try:
                triple_correlation = np.corrcoef(self.triple_results[triple_score_col], 
                                               self.triple_results[actual_return_col])[0, 1]
                if np.isnan(triple_correlation):
                    triple_correlation = 0.0
            except:
                triple_correlation = 0.0
            
            comparison_metrics[horizon] = {
                'dual': {
                    'accuracy': dual_accuracy,
                    'precision': dual_precision,
                    'recall': dual_recall,
                    'f1_score': dual_f1,
                    'correlation': dual_correlation
                },
                'triple': {
                    'accuracy': triple_accuracy,
                    'precision': triple_precision,
                    'recall': triple_recall,
                    'f1_score': triple_f1,
                    'correlation': triple_correlation
                }
            }
        
        self.comparison_metrics = comparison_metrics
        
        # 打印对比结果
        print("\n" + "="*60)
        print("模型性能对比结果")
        print("="*60)
        
        for horizon in ['1d', '3d', '5d']:
            dual_metrics = comparison_metrics[horizon]['dual']
            triple_metrics = comparison_metrics[horizon]['triple']
            
            print(f"\n{horizon.upper()}预测性能对比:")
            print(f"  准确率: 双模型={dual_metrics['accuracy']:.4f}, 三模型={triple_metrics['accuracy']:.4f}, 差异={triple_metrics['accuracy']-dual_metrics['accuracy']:+.4f}")
            print(f"  精确率: 双模型={dual_metrics['precision']:.4f}, 三模型={triple_metrics['precision']:.4f}, 差异={triple_metrics['precision']-dual_metrics['precision']:+.4f}")
            print(f"  召回率: 双模型={dual_metrics['recall']:.4f}, 三模型={triple_metrics['recall']:.4f}, 差异={triple_metrics['recall']-dual_metrics['recall']:+.4f}")
            print(f"  F1分数: 双模型={dual_metrics['f1_score']:.4f}, 三模型={triple_metrics['f1_score']:.4f}, 差异={triple_metrics['f1_score']-dual_metrics['f1_score']:+.4f}")
            print(f"  相关性: 双模型={dual_metrics['correlation']:.4f}, 三模型={triple_metrics['correlation']:.4f}, 差异={triple_metrics['correlation']-dual_metrics['correlation']:+.4f}")
    
    def create_comparison_visualizations(self, save_path="Result_compare"):
        """
        创建模型对比可视化图表
        
        Args:
            save_path: 保存路径
        """
        import os
        os.makedirs(save_path, exist_ok=True)
        
        print(f"创建对比可视化图表，保存到 {save_path} 文件夹...")
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8')
        
        # 1. 性能指标对比柱状图
        self._plot_performance_comparison(save_path)
        
        # 2. 准确率趋势对比
        self._plot_accuracy_trend_comparison(save_path)
        
        # 3. 分数分布对比
        self._plot_score_distribution_comparison(save_path)
        
        # 4. 混淆矩阵对比
        self._plot_confusion_matrix_comparison(save_path)
        
        # 5. 预测一致性分析
        self._plot_prediction_consistency(save_path)
        
        # 6. 性能提升分析
        self._plot_performance_improvement(save_path)
        
        # 7. 相关性对比
        self._plot_correlation_comparison(save_path)
        
        # 8. 综合对比雷达图
        self._plot_comprehensive_radar(save_path)
        
        print(f"对比可视化图表已保存到 {save_path} 文件夹")
    
    def _plot_performance_comparison(self, save_path):
        """绘制性能指标对比柱状图"""
        horizons = ['1d', '3d', '5d']
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            dual_values = [self.comparison_metrics[h]['dual'][metric] for h in horizons]
            triple_values = [self.comparison_metrics[h]['triple'][metric] for h in horizons]
            
            x = np.arange(len(horizons))
            width = 0.35
            
            bars1 = axes[i].bar(x - width/2, dual_values, width, label='Dual Model (A+V)', 
                               alpha=0.8, color='skyblue')
            bars2 = axes[i].bar(x + width/2, triple_values, width, label='Triple Model (A+V+S)', 
                               alpha=0.8, color='lightcoral')
            
            axes[i].set_title(f'{metric.title()} Comparison by Horizon', fontsize=12, fontweight='bold')
            axes[i].set_ylabel(metric.title())
            axes[i].set_xlabel('Prediction Horizon')
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(horizons)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
            # 添加数值标签
            for j, (v1, v2) in enumerate(zip(dual_values, triple_values)):
                axes[i].text(j - width/2, v1 + 0.01, f'{v1:.3f}', ha='center', va='bottom', fontsize=9)
                axes[i].text(j + width/2, v2 + 0.01, f'{v2:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_accuracy_trend_comparison(self, save_path):
        """绘制准确率趋势对比"""
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # 计算滚动准确率
        window_size = 50
        
        for horizon in ['1d', '3d', '5d']:
            dual_pred_col = f'dual_ensemble_pred_{horizon}'
            triple_pred_col = f'triple_ensemble_pred_{horizon}'
            actual_col = f'actual_{horizon}_aapl'
            
            # 计算滚动准确率
            dual_rolling_acc = []
            triple_rolling_acc = []
            
            for i in range(window_size, len(self.dual_results)):
                dual_window_acc = accuracy_score(
                    self.dual_results[actual_col].iloc[i-window_size:i],
                    self.dual_results[dual_pred_col].iloc[i-window_size:i]
                )
                triple_window_acc = accuracy_score(
                    self.triple_results[actual_col].iloc[i-window_size:i],
                    self.triple_results[triple_pred_col].iloc[i-window_size:i]
                )
                
                dual_rolling_acc.append(dual_window_acc)
                triple_rolling_acc.append(triple_window_acc)
            
            dates = self.dual_results['date'].iloc[window_size:]
            
            ax.plot(dates, dual_rolling_acc, label=f'Dual Model {horizon}', alpha=0.7, linewidth=2)
            ax.plot(dates, triple_rolling_acc, label=f'Triple Model {horizon}', alpha=0.7, linewidth=2, linestyle='--')
        
        ax.set_title('Rolling Accuracy Comparison (50-day window)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Rolling Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{save_path}/accuracy_trend_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_score_distribution_comparison(self, save_path):
        """绘制分数分布对比"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, horizon in enumerate(['1d', '3d', '5d']):
            dual_scores = self.dual_results[f'dual_ensemble_score_{horizon}']
            triple_scores = self.triple_results[f'triple_ensemble_score_{horizon}']
            
            axes[i].hist(dual_scores, bins=30, alpha=0.6, label='Dual Model', color='skyblue', density=True)
            axes[i].hist(triple_scores, bins=30, alpha=0.6, label='Triple Model', color='lightcoral', density=True)
            
            axes[i].axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Neutral Line')
            axes[i].set_title(f'{horizon} Score Distribution Comparison', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Ensemble Score')
            axes[i].set_ylabel('Density')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
            # 添加统计信息
            dual_mean = dual_scores.mean()
            triple_mean = triple_scores.mean()
            axes[i].text(0.05, 0.95, f'Dual Mean: {dual_mean:.3f}\nTriple Mean: {triple_mean:.3f}', 
                        transform=axes[i].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/score_distribution_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrix_comparison(self, save_path):
        """绘制混淆矩阵对比"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        for i, horizon in enumerate(['1d', '3d', '5d']):
            # 双模型混淆矩阵
            dual_pred_col = f'dual_ensemble_pred_{horizon}'
            triple_pred_col = f'triple_ensemble_pred_{horizon}'
            actual_col = f'actual_{horizon}_aapl'
            
            dual_cm = confusion_matrix(self.dual_results[actual_col], 
                                     self.dual_results[dual_pred_col])
            triple_cm = confusion_matrix(self.triple_results[actual_col], 
                                       self.triple_results[triple_pred_col])
            
            # 绘制双模型混淆矩阵
            sns.heatmap(dual_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, i],
                       xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
            axes[0, i].set_title(f'Dual Model {horizon} Confusion Matrix', fontsize=12, fontweight='bold')
            axes[0, i].set_xlabel('Predicted')
            axes[0, i].set_ylabel('Actual')
            
            # 绘制三模型混淆矩阵
            sns.heatmap(triple_cm, annot=True, fmt='d', cmap='Reds', ax=axes[1, i],
                       xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
            axes[1, i].set_title(f'Triple Model {horizon} Confusion Matrix', fontsize=12, fontweight='bold')
            axes[1, i].set_xlabel('Predicted')
            axes[1, i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/confusion_matrix_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_prediction_consistency(self, save_path):
        """绘制预测一致性分析"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, horizon in enumerate(['1d', '3d', '5d']):
            dual_pred_col = f'dual_ensemble_pred_{horizon}'
            triple_pred_col = f'triple_ensemble_pred_{horizon}'
            
            # 计算预测一致性
            consistency = (self.dual_results[dual_pred_col] == self.triple_results[triple_pred_col]).astype(int)
            
            # 绘制一致性分布
            axes[i].hist(consistency, bins=2, alpha=0.7, color='lightblue')
            axes[i].set_title(f'{horizon} Prediction Consistency', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Consistency (0=Different, 1=Same)')
            axes[i].set_ylabel('Count')
            axes[i].set_xticks([0, 1])
            axes[i].set_xticklabels(['Different', 'Same'])
            axes[i].grid(True, alpha=0.3)
            
            # 添加一致性统计
            consistency_rate = consistency.mean()
            axes[i].text(0.5, 0.95, f'Consistency Rate: {consistency_rate:.3f}', 
                        transform=axes[i].transAxes, ha='center', va='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/prediction_consistency.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_improvement(self, save_path):
        """绘制性能提升分析"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        horizons = ['1d', '3d', '5d']
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        improvements = {}
        for metric in metrics:
            improvements[metric] = []
            for horizon in horizons:
                dual_val = self.comparison_metrics[horizon]['dual'][metric]
                triple_val = self.comparison_metrics[horizon]['triple'][metric]
                improvement = (triple_val - dual_val) / dual_val * 100  # 百分比提升
                improvements[metric].append(improvement)
        
        x = np.arange(len(horizons))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            ax.bar(x + i*width, improvements[metric], width, 
                  label=f'{metric.title()} Improvement (%)', alpha=0.8)
        
        ax.set_title('Performance Improvement: Triple vs Dual Model', fontsize=14, fontweight='bold')
        ax.set_xlabel('Prediction Horizon')
        ax.set_ylabel('Improvement (%)')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(horizons)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/performance_improvement.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_comparison(self, save_path):
        """绘制相关性对比"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, horizon in enumerate(['1d', '3d', '5d']):
            dual_score_col = f'dual_ensemble_score_{horizon}'
            triple_score_col = f'triple_ensemble_score_{horizon}'
            actual_return_col = f'actual_return_{horizon}_aapl'
            
            # 绘制双模型相关性
            axes[i].scatter(self.dual_results[dual_score_col], 
                          self.dual_results[actual_return_col], 
                          alpha=0.5, s=10, color='skyblue', label='Dual Model')
            
            # 绘制三模型相关性
            axes[i].scatter(self.triple_results[triple_score_col], 
                          self.triple_results[actual_return_col], 
                          alpha=0.5, s=10, color='lightcoral', label='Triple Model')
            
            dual_corr = self.comparison_metrics[horizon]['dual']['correlation']
            triple_corr = self.comparison_metrics[horizon]['triple']['correlation']
            
            axes[i].set_title(f'{horizon} Score vs Return Correlation\nDual: {dual_corr:.3f}, Triple: {triple_corr:.3f}', 
                             fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Ensemble Score')
            axes[i].set_ylabel('Actual Return')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/correlation_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_comprehensive_radar(self, save_path):
        """绘制综合对比雷达图"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(projection='polar'))
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        for i, horizon in enumerate(['1d', '3d', '5d']):
            dual_values = [self.comparison_metrics[horizon]['dual'][metric] for metric in metrics]
            triple_values = [self.comparison_metrics[horizon]['triple'][metric] for metric in metrics]
            
            dual_values += dual_values[:1]  # 闭合图形
            triple_values += triple_values[:1]  # 闭合图形
            
            axes[i].plot(angles, dual_values, 'o-', linewidth=2, label='Dual Model', color='skyblue')
            axes[i].fill(angles, dual_values, alpha=0.25, color='skyblue')
            
            axes[i].plot(angles, triple_values, 'o-', linewidth=2, label='Triple Model', color='lightcoral')
            axes[i].fill(angles, triple_values, alpha=0.25, color='lightcoral')
            
            axes[i].set_xticks(angles[:-1])
            axes[i].set_xticklabels(metrics)
            axes[i].set_title(f'{horizon} Comprehensive Comparison', fontsize=12, fontweight='bold')
            axes[i].legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            axes[i].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/comprehensive_radar_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_comparison_results(self, save_path="Result_compare"):
        """
        保存对比分析结果
        
        Args:
            save_path: 保存路径
        """
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # 保存性能对比数据
        comparison_data = []
        for horizon in ['1d', '3d', '5d']:
            for model in ['dual', 'triple']:
                metrics = self.comparison_metrics[horizon][model]
                comparison_data.append({
                    'horizon': horizon,
                    'model': model,
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1_score'],
                    'correlation': metrics['correlation']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(f'{save_path}/model_comparison_metrics.csv', index=False)
        
        # 保存预测一致性数据
        consistency_data = []
        for horizon in ['1d', '3d', '5d']:
            dual_pred_col = f'dual_ensemble_pred_{horizon}'
            triple_pred_col = f'triple_ensemble_pred_{horizon}'
            
            consistency = (self.dual_results[dual_pred_col] == self.triple_results[triple_pred_col]).astype(int)
            consistency_rate = consistency.mean()
            
            consistency_data.append({
                'horizon': horizon,
                'consistency_rate': consistency_rate,
                'total_predictions': len(consistency),
                'consistent_predictions': consistency.sum(),
                'different_predictions': len(consistency) - consistency.sum()
            })
        
        consistency_df = pd.DataFrame(consistency_data)
        consistency_df.to_csv(f'{save_path}/prediction_consistency_analysis.csv', index=False)
        
        print(f"对比分析结果已保存到 {save_path} 文件夹")


def main():
    """
    主函数
    """
    print("="*60)
    print("AAPL+VGT双模型 vs AAPL+VGT+情感三模型对比分析")
    print("="*60)
    
    # 初始化对比分析器
    analyzer = ModelComparisonAnalyzer()
    
    # 加载模型结果
    dual_results_path = "Result_a+v/dual_ensemble_prediction_results.csv"
    triple_results_path = "Result_triple/triple_ensemble_prediction_results.csv"
    
    analyzer.load_model_results(dual_results_path, triple_results_path)
    
    # 计算性能对比
    analyzer.calculate_performance_comparison()
    
    # 创建对比可视化图表
    print("\n" + "="*60)
    print("生成对比可视化图表...")
    print("="*60)
    
    analyzer.create_comparison_visualizations()
    
    # 保存对比分析结果
    analyzer.save_comparison_results()
    
    print("\n" + "="*60)
    print("模型对比分析完成!")
    print("="*60)


if __name__ == "__main__":
    main()
