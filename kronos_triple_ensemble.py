#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于AAPL、VGT和情感分析的三模型集成预测系统
结合AAPL预测器、VGT预测器和情感分数进行1天、3天、5天的涨跌预测
使用网格搜索找到最佳权重组合
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

class TripleModelEnsemble:
    """
    基于AAPL、VGT和情感分析的三模型集成预测器
    """
    
    def __init__(self):
        """
        初始化三模型集成预测器
        """
        self.best_weights = None
        self.best_performance = None
        self.grid_search_results = None
        
    def load_data(self, aapl_results_path, vgt_results_path, sentiment_path):
        """
        加载AAPL、VGT预测结果和情感分数数据
        
        Args:
            aapl_results_path: AAPL预测结果文件路径
            vgt_results_path: VGT预测结果文件路径
            sentiment_path: 情感分数文件路径
            
        Returns:
            pd.DataFrame: 合并后的数据
        """
        print("加载三模型数据...")
        
        # 加载AAPL预测结果
        aapl_df = pd.read_csv(aapl_results_path)
        aapl_df['date'] = pd.to_datetime(aapl_df['date'])
        
        # 加载VGT预测结果
        vgt_df = pd.read_csv(vgt_results_path)
        vgt_df['date'] = pd.to_datetime(vgt_df['date'])
        
        # 加载情感分数数据
        sentiment_df = pd.read_csv(sentiment_path)
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        
        # 合并数据
        merged_df = pd.merge(aapl_df, vgt_df, on='date', how='inner', suffixes=('_aapl', '_vgt'))
        merged_df = pd.merge(merged_df, sentiment_df, on='date', how='inner')
        
        print(f"AAPL数据形状: {aapl_df.shape}")
        print(f"VGT数据形状: {vgt_df.shape}")
        print(f"情感数据形状: {sentiment_df.shape}")
        print(f"合并后数据形状: {merged_df.shape}")
        print(f"数据时间范围: {merged_df['date'].min()} 到 {merged_df['date'].max()}")
        
        return merged_df
    
    def create_ensemble_features(self, df):
        """
        创建三模型集成特征
        
        Args:
            df: 合并后的数据
            
        Returns:
            pd.DataFrame: 包含集成特征的数据
        """
        print("创建三模型集成特征...")
        
        # 复制数据
        ensemble_df = df.copy()
        
        # 标准化各模型分数到[-1, 1]范围
        for horizon in ['1d', '3d', '5d']:
            aapl_col = f'score_{horizon}_aapl'
            vgt_col = f'score_{horizon}_vgt'
            sentiment_col = f'sentiment_pred_{horizon}'
            
            # 确保分数在[-1, 1]范围内
            ensemble_df[f'aapl_score_{horizon}'] = np.clip(ensemble_df[aapl_col], -1, 1)
            ensemble_df[f'vgt_score_{horizon}'] = np.clip(ensemble_df[vgt_col], -1, 1)
            ensemble_df[f'sentiment_score_{horizon}'] = np.clip(ensemble_df[sentiment_col], -1, 1)
        
        return ensemble_df
    
    def grid_search_weights(self, df, param_grid=None):
        """
        使用网格搜索找到最佳三模型权重组合
        
        Args:
            df: 包含集成特征的数据
            param_grid: 参数网格，如果为None则使用默认网格
            
        Returns:
            dict: 最佳权重和性能结果
        """
        print("开始三模型网格搜索...")
        
        if param_grid is None:
            # 默认参数网格：AAPL权重50%+，Sentiment权重25%-35%
            param_grid = {
                'aapl_weight': [0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
                'sentiment_weight': [0.25, 0.27, 0.30, 0.32, 0.35]
            }
        
        # 过滤掉权重和超过1的组合
        valid_params = []
        for params in ParameterGrid(param_grid):
            vgt_weight = 1.0 - params['aapl_weight'] - params['sentiment_weight']
            if vgt_weight >= 0.05:  # VGT权重至少5%
                params['vgt_weight'] = vgt_weight
                valid_params.append(params)
        
        print(f"总共测试 {len(valid_params)} 个权重组合...")
        
        best_performance = -np.inf
        best_weights = None
        grid_results = []
        
        for i, params in enumerate(valid_params):
            aapl_weight = params['aapl_weight']
            vgt_weight = params['vgt_weight']
            sentiment_weight = params['sentiment_weight']
            
            # 计算三模型集成预测分数
            ensemble_scores = self._calculate_triple_ensemble_scores(df, aapl_weight, vgt_weight, sentiment_weight)
            
            # 计算性能指标
            performance = self._evaluate_performance(df, ensemble_scores)
            
            # 使用加权平均准确率作为主要指标
            avg_accuracy = np.mean([performance[h]['accuracy'] for h in ['1d', '3d', '5d']])
            
            grid_results.append({
                'aapl_weight': aapl_weight,
                'vgt_weight': vgt_weight,
                'sentiment_weight': sentiment_weight,
                'avg_accuracy': avg_accuracy,
                'performance': performance
            })
            
            if avg_accuracy > best_performance:
                best_performance = avg_accuracy
                best_weights = params.copy()
            
            if (i + 1) % 20 == 0:
                print(f"已完成 {i + 1}/{len(valid_params)} 个组合")
        
        self.best_weights = best_weights
        self.best_performance = best_performance
        self.grid_search_results = grid_results
        
        print(f"三模型网格搜索完成!")
        print(f"最佳权重组合:")
        print(f"  AAPL权重: {best_weights['aapl_weight']:.3f}")
        print(f"  VGT权重: {best_weights['vgt_weight']:.3f}")
        print(f"  情感权重: {best_weights['sentiment_weight']:.3f}")
        print(f"最佳平均准确率: {best_performance:.4f}")
        
        return {
            'best_weights': best_weights,
            'best_performance': best_performance,
            'grid_results': grid_results
        }
    
    def _calculate_triple_ensemble_scores(self, df, aapl_weight, vgt_weight, sentiment_weight):
        """
        计算三模型集成预测分数
        
        Args:
            df: 数据
            aapl_weight: AAPL权重
            vgt_weight: VGT权重
            sentiment_weight: 情感权重
            
        Returns:
            dict: 各时间窗口的集成分数
        """
        ensemble_scores = {}
        
        for horizon in ['1d', '3d', '5d']:
            aapl_col = f'aapl_score_{horizon}'
            vgt_col = f'vgt_score_{horizon}'
            sentiment_col = f'sentiment_score_{horizon}'
            
            # 计算三模型加权集成分数
            ensemble_score = (aapl_weight * df[aapl_col] + 
                            vgt_weight * df[vgt_col] + 
                            sentiment_weight * df[sentiment_col])
            
            ensemble_scores[horizon] = ensemble_score
        
        return ensemble_scores
    
    def _evaluate_performance(self, df, ensemble_scores):
        """
        评估三模型集成性能
        
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
    
    def run_triple_ensemble_analysis(self, aapl_results_path, vgt_results_path, sentiment_path):
        """
        运行完整的三模型集成分析
        
        Args:
            aapl_results_path: AAPL预测结果文件路径
            vgt_results_path: VGT预测结果文件路径
            sentiment_path: 情感分数文件路径
            
        Returns:
            dict: 分析结果
        """
        print("="*60)
        print("AAPL-VGT-情感三模型集成预测分析")
        print("="*60)
        
        # 加载和准备数据
        df = self.load_data(aapl_results_path, vgt_results_path, sentiment_path)
        ensemble_df = self.create_ensemble_features(df)
        
        # 网格搜索最佳权重
        grid_results = self.grid_search_weights(ensemble_df)
        
        # 使用最佳权重计算最终结果
        best_weights = grid_results['best_weights']
        final_scores = self._calculate_triple_ensemble_scores(ensemble_df, 
                                                             best_weights['aapl_weight'], 
                                                             best_weights['vgt_weight'],
                                                             best_weights['sentiment_weight'])
        
        # 添加集成预测结果到数据框
        for horizon in ['1d', '3d', '5d']:
            ensemble_df[f'triple_ensemble_score_{horizon}'] = final_scores[horizon]
            ensemble_df[f'triple_ensemble_pred_{horizon}'] = (final_scores[horizon] > 0).astype(int)
        
        # 计算最终性能
        final_performance = self._evaluate_performance(ensemble_df, final_scores)
        
        return {
            'data': ensemble_df,
            'best_weights': best_weights,
            'final_performance': final_performance,
            'grid_results': grid_results['grid_results']
        }
    
    def create_visualizations(self, analysis_results, save_path="Result_triple"):
        """
        创建三模型集成可视化图表
        
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
        
        # 1. 三模型权重优化热图
        self._plot_triple_weight_heatmap(grid_results, best_weights, save_path)
        
        # 2. 三模型集成性能对比
        self._plot_triple_performance_comparison(final_performance, save_path)
        
        # 3. 三模型分数相关性分析
        self._plot_triple_model_correlation(df, save_path)
        
        # 4. 三模型集成预测分数分布
        self._plot_triple_score_distribution(df, save_path)
        
        # 5. 三模型集成混淆矩阵
        self._plot_triple_confusion_matrices(df, save_path)
        
        # 6. 权重优化过程
        self._plot_triple_weight_optimization(grid_results, save_path)
        
        # 7. 三模型分数对比
        self._plot_triple_model_comparison(df, save_path)
        
        print(f"三模型可视化图表已保存到 {save_path} 文件夹")
    
    def _plot_triple_weight_heatmap(self, grid_results, best_weights, save_path):
        """绘制三模型权重优化热图"""
        # 创建权重矩阵
        aapl_weights = sorted(list(set([r['aapl_weight'] for r in grid_results])))
        sentiment_weights = sorted(list(set([r['sentiment_weight'] for r in grid_results])))
        
        accuracy_matrix = np.zeros((len(aapl_weights), len(sentiment_weights)))
        
        for result in grid_results:
            i = aapl_weights.index(result['aapl_weight'])
            j = sentiment_weights.index(result['sentiment_weight'])
            accuracy_matrix[i, j] = result['avg_accuracy']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        im = ax.imshow(accuracy_matrix, cmap='viridis', aspect='auto')
        
        # 设置标签
        ax.set_xticks(range(len(sentiment_weights)))
        ax.set_yticks(range(len(aapl_weights)))
        ax.set_xticklabels([f'{w:.2f}' for w in sentiment_weights])
        ax.set_yticklabels([f'{w:.2f}' for w in aapl_weights])
        
        # 标记最佳点
        best_i = aapl_weights.index(best_weights['aapl_weight'])
        best_j = sentiment_weights.index(best_weights['sentiment_weight'])
        ax.scatter(best_j, best_i, color='red', s=200, marker='*', label='Best Weights')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Average Accuracy')
        
        ax.set_xlabel('Sentiment Weight')
        ax.set_ylabel('AAPL Weight')
        ax.set_title('Triple Model Weight Optimization Heatmap', fontsize=14, fontweight='bold')
        ax.legend()
        
        # 添加VGT权重信息
        vgt_weight = best_weights['vgt_weight']
        ax.text(0.02, 0.98, f'Best VGT Weight: {vgt_weight:.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/triple_weight_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_triple_performance_comparison(self, performance, save_path):
        """绘制三模型集成性能对比图"""
        horizons = ['1d', '3d', '5d']
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [performance[h][metric] for h in horizons]
            
            bars = axes[i].bar(horizons, values, alpha=0.8, color=['skyblue', 'lightgreen', 'lightcoral'])
            axes[i].set_title(f'Triple Model {metric.title()} by Horizon', fontsize=12, fontweight='bold')
            axes[i].set_ylabel(metric.title())
            axes[i].set_xlabel('Prediction Horizon')
            axes[i].grid(True, alpha=0.3)
            
            # 添加数值标签
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/triple_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_triple_model_correlation(self, df, save_path):
        """绘制三模型分数相关性分析"""
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        
        models = ['aapl', 'vgt', 'sentiment']
        model_names = ['AAPL', 'VGT', 'Sentiment']
        
        for i, horizon in enumerate(['1d', '3d', '5d']):
            for j, (model1, name1) in enumerate(zip(models, model_names)):
                for k, (model2, name2) in enumerate(zip(models, model_names)):
                    if j <= k:  # 只绘制上三角矩阵
                        if j == k:
                            # 对角线：显示分布
                            scores = df[f'{model1}_score_{horizon}']
                            axes[j, k].hist(scores, bins=20, alpha=0.7, edgecolor='black')
                            axes[j, k].set_title(f'{name1} Score Distribution ({horizon})', fontsize=10)
                        else:
                            # 非对角线：显示相关性
                            scores1 = df[f'{model1}_score_{horizon}']
                            scores2 = df[f'{model2}_score_{horizon}']
                            
                            axes[j, k].scatter(scores1, scores2, alpha=0.5, s=5)
                            
                            # 计算相关系数
                            correlation = np.corrcoef(scores1, scores2)[0, 1]
                            
                            axes[j, k].set_title(f'{name1} vs {name2} ({horizon})\nCorrelation: {correlation:.3f}', 
                                               fontsize=10, fontweight='bold')
                            axes[j, k].grid(True, alpha=0.3)
                    else:
                        # 下三角：空白
                        axes[j, k].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/triple_model_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_triple_score_distribution(self, df, save_path):
        """绘制三模型集成预测分数分布"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, horizon in enumerate(['1d', '3d', '5d']):
            scores = df[f'triple_ensemble_score_{horizon}']
            
            axes[i].hist(scores, bins=30, alpha=0.7, edgecolor='black', color='lightblue')
            axes[i].axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Neutral Line')
            axes[i].set_title(f'{horizon} Triple Ensemble Score Distribution', fontsize=12, fontweight='bold')
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
        plt.savefig(f'{save_path}/triple_score_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_triple_confusion_matrices(self, df, save_path):
        """绘制三模型集成混淆矩阵"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, horizon in enumerate(['1d', '3d', '5d']):
            pred_col = f'triple_ensemble_pred_{horizon}'
            actual_col = f'actual_{horizon}_aapl'
            
            cm = confusion_matrix(df[actual_col], df[pred_col])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                       xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
            axes[i].set_title(f'{horizon} Triple Ensemble Confusion Matrix', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/triple_confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_triple_weight_optimization(self, grid_results, save_path):
        """绘制三模型权重优化过程"""
        # 按平均准确率排序
        sorted_results = sorted(grid_results, key=lambda x: x['avg_accuracy'], reverse=True)
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # 绘制前15个最佳结果
        top_results = sorted_results[:15]
        x_pos = range(len(top_results))
        accuracies = [r['avg_accuracy'] for r in top_results]
        
        bars = ax.bar(x_pos, accuracies, alpha=0.7, color='lightgreen')
        
        # 标记最佳结果
        bars[0].set_color('red')
        
        ax.set_xlabel('Weight Combination Rank')
        ax.set_ylabel('Average Accuracy')
        ax.set_title('Top 15 Triple Model Weight Combinations by Performance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 添加标签
        labels = []
        for r in top_results:
            label = f"A:{r['aapl_weight']:.2f}, V:{r['vgt_weight']:.2f}, S:{r['sentiment_weight']:.2f}"
            labels.append(label)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        # 添加数值标签
        for i, v in enumerate(accuracies):
            ax.text(i, v + 0.001, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/triple_weight_optimization.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_triple_model_comparison(self, df, save_path):
        """绘制三模型分数对比"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, horizon in enumerate(['1d', '3d', '5d']):
            aapl_scores = df[f'aapl_score_{horizon}']
            vgt_scores = df[f'vgt_score_{horizon}']
            sentiment_scores = df[f'sentiment_score_{horizon}']
            
            # 绘制三个模型的分数分布
            axes[i].hist(aapl_scores, bins=20, alpha=0.5, label='AAPL', color='blue')
            axes[i].hist(vgt_scores, bins=20, alpha=0.5, label='VGT', color='green')
            axes[i].hist(sentiment_scores, bins=20, alpha=0.5, label='Sentiment', color='orange')
            
            axes[i].axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Neutral Line')
            axes[i].set_title(f'{horizon} Model Scores Comparison', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Prediction Score')
            axes[i].set_ylabel('Frequency')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/triple_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """
    主函数
    """
    print("="*60)
    print("AAPL-VGT-情感三模型集成预测系统")
    print("="*60)
    
    # 初始化三模型集成预测器
    ensemble = TripleModelEnsemble()
    
    # 运行三模型集成分析
    aapl_results_path = "Result/kronos_prediction_results.csv"
    vgt_results_path = "Result-vgt/kronos_prediction_results.csv"
    sentiment_path = "Data/sentiment_scores.csv"
    
    analysis_results = ensemble.run_triple_ensemble_analysis(aapl_results_path, vgt_results_path, sentiment_path)
    
    # 显示结果
    print("\n" + "="*60)
    print("三模型集成性能评估结果")
    print("="*60)
    
    best_weights = analysis_results['best_weights']
    final_performance = analysis_results['final_performance']
    
    print(f"最佳权重组合:")
    print(f"  AAPL权重: {best_weights['aapl_weight']:.3f}")
    print(f"  VGT权重: {best_weights['vgt_weight']:.3f}")
    print(f"  情感权重: {best_weights['sentiment_weight']:.3f}")
    print(f"  权重总和: {best_weights['aapl_weight'] + best_weights['vgt_weight'] + best_weights['sentiment_weight']:.3f}")
    
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
    print("生成三模型可视化图表...")
    print("="*60)
    
    ensemble.create_visualizations(analysis_results)
    
    # 保存三模型集成预测结果
    results_df = analysis_results['data']
    results_df.to_csv('Result_triple/triple_ensemble_prediction_results.csv', index=False)
    print("三模型集成预测结果已保存到 Result_triple/triple_ensemble_prediction_results.csv")
    
    # 保存网格搜索结果
    grid_df = pd.DataFrame(analysis_results['grid_results'])
    grid_df.to_csv('Result_triple/triple_grid_search_results.csv', index=False)
    print("三模型网格搜索结果已保存到 Result_triple/triple_grid_search_results.csv")
    
    print("\n" + "="*60)
    print("AAPL-VGT-情感三模型集成预测分析完成!")
    print("="*60)


if __name__ == "__main__":
    main()
