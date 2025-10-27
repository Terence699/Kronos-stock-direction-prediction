#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于AAPL和VGT的双模型集成预测系统
结合AAPL预测器和VGT预测器进行1天、3天、5天的涨跌预测
使用网格搜索找到最佳权重组合，并与三模型集成进行对比
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import ParameterGrid
import warnings
warnings.filterwarnings('ignore')

# 设置英文字体
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class DualModelEnsemble:
    """
    基于AAPL和VGT的双模型集成预测器
    """
    
    def __init__(self):
        """
        初始化双模型集成预测器
        """
        self.best_weights = None
        self.best_performance = None
        self.grid_search_results = None
        
    def load_data(self, aapl_results_path, vgt_results_path):
        """
        加载AAPL和VGT预测结果数据
        
        Args:
            aapl_results_path: AAPL预测结果文件路径
            vgt_results_path: VGT预测结果文件路径
            
        Returns:
            pd.DataFrame: 合并后的数据
        """
        print("加载双模型数据...")
        
        # 加载AAPL预测结果
        aapl_df = pd.read_csv(aapl_results_path)
        aapl_df['date'] = pd.to_datetime(aapl_df['date'])
        
        # 加载VGT预测结果
        vgt_df = pd.read_csv(vgt_results_path)
        vgt_df['date'] = pd.to_datetime(vgt_df['date'])
        
        # 合并数据
        merged_df = pd.merge(aapl_df, vgt_df, on='date', how='inner', suffixes=('_aapl', '_vgt'))
        
        print(f"AAPL数据形状: {aapl_df.shape}")
        print(f"VGT数据形状: {vgt_df.shape}")
        print(f"合并后数据形状: {merged_df.shape}")
        print(f"数据时间范围: {merged_df['date'].min()} 到 {merged_df['date'].max()}")
        
        return merged_df
    
    def create_dual_features(self, df):
        """
        创建双模型集成特征
        
        Args:
            df: 合并后的数据
            
        Returns:
            pd.DataFrame: 包含集成特征的数据
        """
        print("创建双模型集成特征...")
        
        # 复制数据
        ensemble_df = df.copy()
        
        # 标准化各模型分数到[-1, 1]范围
        for horizon in ['1d', '3d', '5d']:
            aapl_col = f'score_{horizon}_aapl'
            vgt_col = f'score_{horizon}_vgt'
            
            # 确保分数在[-1, 1]范围内
            ensemble_df[f'aapl_score_{horizon}'] = np.clip(ensemble_df[aapl_col], -1, 1)
            ensemble_df[f'vgt_score_{horizon}'] = np.clip(ensemble_df[vgt_col], -1, 1)
        
        return ensemble_df
    
    def grid_search_weights(self, df, param_grid=None):
        """
        使用网格搜索找到最佳双模型权重组合
        
        Args:
            df: 包含集成特征的数据
            param_grid: 参数网格，如果为None则使用默认网格
            
        Returns:
            dict: 最佳权重和性能结果
        """
        print("开始双模型网格搜索...")
        
        if param_grid is None:
            # 默认参数网格：AAPL权重从30%到80%，步长5%
            param_grid = {
                'aapl_weight': [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
            }
        
        # 计算VGT权重（权重总和为1）
        valid_params = []
        for params in ParameterGrid(param_grid):
            vgt_weight = 1.0 - params['aapl_weight']
            params['vgt_weight'] = vgt_weight
            valid_params.append(params)
        
        print(f"总共测试 {len(valid_params)} 个权重组合...")
        
        best_performance = -np.inf
        best_weights = None
        grid_results = []
        
        for i, params in enumerate(valid_params):
            aapl_weight = params['aapl_weight']
            vgt_weight = params['vgt_weight']
            
            # 计算双模型集成预测分数
            ensemble_scores = self._calculate_dual_ensemble_scores(df, aapl_weight, vgt_weight)
            
            # 计算性能指标
            performance = self._evaluate_performance(df, ensemble_scores)
            
            # 使用加权平均准确率作为主要指标
            avg_accuracy = np.mean([performance[h]['accuracy'] for h in ['1d', '3d', '5d']])
            
            grid_results.append({
                'aapl_weight': aapl_weight,
                'vgt_weight': vgt_weight,
                'avg_accuracy': avg_accuracy,
                'performance': performance
            })
            
            if avg_accuracy > best_performance:
                best_performance = avg_accuracy
                best_weights = params.copy()
            
            if (i + 1) % 5 == 0:
                print(f"已完成 {i + 1}/{len(valid_params)} 个组合")
        
        self.best_weights = best_weights
        self.best_performance = best_performance
        self.grid_search_results = grid_results
        
        print(f"双模型网格搜索完成!")
        print(f"最佳权重组合:")
        print(f"  AAPL权重: {best_weights['aapl_weight']:.3f}")
        print(f"  VGT权重: {best_weights['vgt_weight']:.3f}")
        print(f"最佳平均准确率: {best_performance:.4f}")
        
        return {
            'best_weights': best_weights,
            'best_performance': best_performance,
            'grid_results': grid_results
        }
    
    def _calculate_dual_ensemble_scores(self, df, aapl_weight, vgt_weight):
        """
        计算双模型集成预测分数
        
        Args:
            df: 数据
            aapl_weight: AAPL权重
            vgt_weight: VGT权重
            
        Returns:
            dict: 各时间窗口的集成分数
        """
        ensemble_scores = {}
        
        for horizon in ['1d', '3d', '5d']:
            aapl_col = f'aapl_score_{horizon}'
            vgt_col = f'vgt_score_{horizon}'
            
            # 计算双模型加权集成分数
            ensemble_score = (aapl_weight * df[aapl_col] + vgt_weight * df[vgt_col])
            
            ensemble_scores[horizon] = ensemble_score
        
        return ensemble_scores
    
    def _evaluate_performance(self, df, ensemble_scores):
        """
        评估双模型集成性能
        
        Args:
            df: 数据
            ensemble_scores: 集成分数
            
        Returns:
            dict: 性能指标
        """
        performance = {}
        
        for horizon in ['1d', '3d', '5d']:
            # 获取预测和实际值
            pred_scores = ensemble_scores[horizon]
            pred_direction = (pred_scores > 0).astype(int)
            actual_direction = df[f'actual_{horizon}_aapl']  # 使用AAPL的实际值作为基准
            
            # 计算分类指标
            accuracy = accuracy_score(actual_direction, pred_direction)
            precision = precision_score(actual_direction, pred_direction, zero_division=0)
            recall = recall_score(actual_direction, pred_direction, zero_division=0)
            f1 = f1_score(actual_direction, pred_direction, zero_division=0)
            
            # 计算分数相关性
            actual_returns = df[f'actual_return_{horizon}_aapl']
            try:
                correlation = np.corrcoef(pred_scores, actual_returns)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
            except:
                correlation = 0.0
            
            performance[horizon] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'score_correlation': correlation,
                'total_predictions': len(df),
                'up_predictions': pred_direction.sum(),
                'down_predictions': len(df) - pred_direction.sum()
            }
        
        return performance
    
    def run_dual_ensemble_analysis(self, aapl_results_path, vgt_results_path):
        """
        运行完整的双模型集成分析
        
        Args:
            aapl_results_path: AAPL预测结果文件路径
            vgt_results_path: VGT预测结果文件路径
            
        Returns:
            dict: 分析结果
        """
        print("="*60)
        print("AAPL-VGT双模型集成预测分析")
        print("="*60)
        
        # 加载和准备数据
        df = self.load_data(aapl_results_path, vgt_results_path)
        ensemble_df = self.create_dual_features(df)
        
        # 网格搜索最佳权重
        grid_results = self.grid_search_weights(ensemble_df)
        
        # 使用最佳权重计算最终结果
        best_weights = grid_results['best_weights']
        final_scores = self._calculate_dual_ensemble_scores(ensemble_df, 
                                                           best_weights['aapl_weight'], 
                                                           best_weights['vgt_weight'])
        
        # 添加集成预测结果到数据框
        for horizon in ['1d', '3d', '5d']:
            ensemble_df[f'dual_ensemble_score_{horizon}'] = final_scores[horizon]
            ensemble_df[f'dual_ensemble_pred_{horizon}'] = (final_scores[horizon] > 0).astype(int)
        
        # 计算最终性能
        final_performance = self._evaluate_performance(ensemble_df, final_scores)
        
        return {
            'data': ensemble_df,
            'best_weights': best_weights,
            'final_performance': final_performance,
            'grid_results': grid_results['grid_results']
        }
    
    def create_visualizations(self, analysis_results, save_path="Result_a+v"):
        """
        创建双模型集成可视化图表
        
        Args:
            analysis_results: 分析结果
            save_path: 保存路径
        """
        import os
        os.makedirs(save_path, exist_ok=True)
        
        df = analysis_results['data']
        best_weights = analysis_results['best_weights']
        final_performance = analysis_results['final_performance']
        grid_results = analysis_results['grid_results']
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8')
        
        # 1. 双模型权重优化图
        self._plot_dual_weight_optimization(grid_results, best_weights, save_path)
        
        # 2. 双模型集成性能对比
        self._plot_dual_performance_comparison(final_performance, save_path)
        
        # 3. 双模型分数相关性分析
        self._plot_dual_model_correlation(df, save_path)
        
        # 4. 双模型集成预测分数分布
        self._plot_dual_score_distribution(df, save_path)
        
        # 5. 双模型集成混淆矩阵
        self._plot_dual_confusion_matrices(df, save_path)
        
        # 6. 双模型分数对比
        self._plot_dual_model_comparison(df, save_path)
        
        # 7. 权重性能热图
        self._plot_dual_weight_heatmap(grid_results, best_weights, save_path)
        
        print(f"双模型可视化图表已保存到 {save_path} 文件夹")
    
    def _plot_dual_weight_optimization(self, grid_results, best_weights, save_path):
        """绘制双模型权重优化过程"""
        # 按平均准确率排序
        sorted_results = sorted(grid_results, key=lambda x: x['avg_accuracy'], reverse=True)
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # 绘制所有结果
        x_pos = range(len(sorted_results))
        accuracies = [r['avg_accuracy'] for r in sorted_results]
        aapl_weights = [r['aapl_weight'] for r in sorted_results]
        
        # 使用颜色映射显示AAPL权重
        scatter = ax.scatter(x_pos, accuracies, c=aapl_weights, cmap='viridis', s=100, alpha=0.7)
        
        # 标记最佳结果
        best_idx = 0
        ax.scatter(best_idx, accuracies[best_idx], color='red', s=200, marker='*', label='Best Weights')
        
        ax.set_xlabel('Weight Combination Rank')
        ax.set_ylabel('Average Accuracy')
        ax.set_title('Dual Model Weight Optimization Results', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('AAPL Weight')
        
        # 添加最佳权重信息
        ax.text(0.02, 0.98, f'Best AAPL Weight: {best_weights["aapl_weight"]:.3f}\nBest VGT Weight: {best_weights["vgt_weight"]:.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/dual_weight_optimization.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_dual_performance_comparison(self, performance, save_path):
        """绘制双模型集成性能对比图"""
        horizons = ['1d', '3d', '5d']
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [performance[h][metric] for h in horizons]
            
            bars = axes[i].bar(horizons, values, alpha=0.8, color=['skyblue', 'lightgreen', 'lightcoral'])
            axes[i].set_title(f'Dual Model {metric.title()} by Horizon', fontsize=12, fontweight='bold')
            axes[i].set_ylabel(metric.title())
            axes[i].set_xlabel('Prediction Horizon')
            axes[i].grid(True, alpha=0.3)
            
            # 添加数值标签
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/dual_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_dual_model_correlation(self, df, save_path):
        """绘制双模型分数相关性分析"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, horizon in enumerate(['1d', '3d', '5d']):
            aapl_scores = df[f'aapl_score_{horizon}']
            vgt_scores = df[f'vgt_score_{horizon}']
            
            # 绘制散点图
            axes[i].scatter(aapl_scores, vgt_scores, alpha=0.5, s=10)
            
            # 计算相关系数
            correlation = np.corrcoef(aapl_scores, vgt_scores)[0, 1]
            
            axes[i].set_title(f'AAPL vs VGT Scores ({horizon})\nCorrelation: {correlation:.3f}', 
                             fontsize=12, fontweight='bold')
            axes[i].set_xlabel('AAPL Score')
            axes[i].set_ylabel('VGT Score')
            axes[i].grid(True, alpha=0.3)
            
            # 添加对角线
            min_val = min(aapl_scores.min(), vgt_scores.min())
            max_val = max(aapl_scores.max(), vgt_scores.max())
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect Correlation')
            axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/dual_model_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_dual_score_distribution(self, df, save_path):
        """绘制双模型集成预测分数分布"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, horizon in enumerate(['1d', '3d', '5d']):
            scores = df[f'dual_ensemble_score_{horizon}']
            
            axes[i].hist(scores, bins=30, alpha=0.7, edgecolor='black', color='lightblue')
            axes[i].axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Neutral Line')
            axes[i].set_title(f'{horizon} Dual Ensemble Score Distribution', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Ensemble Score')
            axes[i].set_ylabel('Frequency')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
            # 添加统计信息
            mean_score = scores.mean()
            std_score = scores.std()
            axes[i].text(0.05, 0.95, f'Mean: {mean_score:.3f}\nStd: {std_score:.3f}', 
                        transform=axes[i].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/dual_score_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_dual_confusion_matrices(self, df, save_path):
        """绘制双模型集成混淆矩阵"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, horizon in enumerate(['1d', '3d', '5d']):
            pred_col = f'dual_ensemble_pred_{horizon}'
            actual_col = f'actual_{horizon}_aapl'
            
            cm = confusion_matrix(df[actual_col], df[pred_col])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                       xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
            axes[i].set_title(f'{horizon} Dual Ensemble Confusion Matrix', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/dual_confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_dual_model_comparison(self, df, save_path):
        """绘制双模型分数对比"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, horizon in enumerate(['1d', '3d', '5d']):
            aapl_scores = df[f'aapl_score_{horizon}']
            vgt_scores = df[f'vgt_score_{horizon}']
            
            # 绘制两个模型的分数分布
            axes[i].hist(aapl_scores, bins=20, alpha=0.5, label='AAPL', color='blue')
            axes[i].hist(vgt_scores, bins=20, alpha=0.5, label='VGT', color='green')
            
            axes[i].axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Neutral Line')
            axes[i].set_title(f'{horizon} Model Scores Comparison', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Prediction Score')
            axes[i].set_ylabel('Frequency')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/dual_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_dual_weight_heatmap(self, grid_results, best_weights, save_path):
        """绘制双模型权重性能热图"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 提取权重和性能数据
        aapl_weights = [r['aapl_weight'] for r in grid_results]
        accuracies = [r['avg_accuracy'] for r in grid_results]
        
        # 创建散点图
        scatter = ax.scatter(aapl_weights, accuracies, c=accuracies, cmap='viridis', s=100, alpha=0.8)
        
        # 标记最佳点
        ax.scatter(best_weights['aapl_weight'], self.best_performance, 
                  color='red', s=200, marker='*', label='Best Weights')
        
        ax.set_xlabel('AAPL Weight')
        ax.set_ylabel('Average Accuracy')
        ax.set_title('Dual Model Weight vs Performance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Average Accuracy')
        
        # 添加最佳权重信息
        ax.text(0.02, 0.98, f'Best AAPL Weight: {best_weights["aapl_weight"]:.3f}\nBest VGT Weight: {best_weights["vgt_weight"]:.3f}\nBest Accuracy: {self.best_performance:.4f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/dual_weight_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """
    主函数
    """
    print("="*60)
    print("AAPL-VGT双模型集成预测系统")
    print("="*60)
    
    # 初始化双模型集成预测器
    ensemble = DualModelEnsemble()
    
    # 运行双模型集成分析
    aapl_results_path = "Result/kronos_prediction_results.csv"
    vgt_results_path = "Result-vgt/kronos_prediction_results.csv"
    
    analysis_results = ensemble.run_dual_ensemble_analysis(aapl_results_path, vgt_results_path)
    
    # 显示结果
    print("\n" + "="*60)
    print("双模型集成性能评估结果")
    print("="*60)
    
    best_weights = analysis_results['best_weights']
    final_performance = analysis_results['final_performance']
    
    print(f"最佳权重组合:")
    print(f"  AAPL权重: {best_weights['aapl_weight']:.3f}")
    print(f"  VGT权重: {best_weights['vgt_weight']:.3f}")
    print(f"  权重总和: {best_weights['aapl_weight'] + best_weights['vgt_weight']:.3f}")
    
    for horizon in ['1d', '3d', '5d']:
        metrics = final_performance[horizon]
        print(f"\n{horizon.upper()}预测性能:")
        print(f"  准确率: {metrics['accuracy']:.4f}")
        print(f"  精确率: {metrics['precision']:.4f}")
        print(f"  召回率: {metrics['recall']:.4f}")
        print(f"  F1分数: {metrics['f1_score']:.4f}")
        print(f"  分数相关性: {metrics['score_correlation']:.4f}")
        print(f"  总预测数: {metrics['total_predictions']}")
        print(f"  上涨预测: {metrics['up_predictions']}")
        print(f"  下跌预测: {metrics['down_predictions']}")
    
    # 创建可视化图表
    print("\n" + "="*60)
    print("生成双模型可视化图表...")
    print("="*60)
    
    ensemble.create_visualizations(analysis_results)
    
    # 保存双模型集成预测结果
    results_df = analysis_results['data']
    results_df.to_csv('Result_a+v/dual_ensemble_prediction_results.csv', index=False)
    print("双模型集成预测结果已保存到 Result_a+v/dual_ensemble_prediction_results.csv")
    
    # 保存网格搜索结果
    grid_df = pd.DataFrame(analysis_results['grid_results'])
    grid_df.to_csv('Result_a+v/dual_grid_search_results.csv', index=False)
    print("双模型网格搜索结果已保存到 Result_a+v/dual_grid_search_results.csv")
    
    print("\n" + "="*60)
    print("AAPL-VGT双模型集成预测分析完成!")
    print("="*60)


if __name__ == "__main__":
    main()
