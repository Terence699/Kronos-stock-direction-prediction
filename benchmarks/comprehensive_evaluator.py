#!/usr/bin/env python3
"""
Comprehensive Model Comparison Framework
=======================================

This module provides a comprehensive framework for comparing all models:
1. Benchmark Models (Technical, ML, DL)
2. Kronos Models (AAPL, VGT)
3. Sentiment Analysis Model
4. Ensemble Models

Features:
- Unified evaluation metrics
- Cross-validation
- Performance visualization
- Model ranking and selection
- Ensemble optimization

Date: 2025
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, ParameterGrid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

# Import our modules
from benchmarks.model_factory import ModelFactory, ModelComparator
from benchmarks.benchmark_models import *
from analysis.kronos_aapl_analysis import AAPLScoreGenerator
from analysis.kronos_vgt_analysis import VGTScoreGenerator
from sentiment.sentiment_analysis import SentimentScoreGenerator
from config import DATA_DIR, AAPL_FILE, OUTPUT_DIR, BENCHMARKS_OUTPUT_DIR

# Configure matplotlib to not show plots
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ComprehensiveModelEvaluator:
    """Comprehensive model evaluation framework"""
    
    def __init__(self, data_path: str = None):
        self.data_path = str(data_path or DATA_DIR)
        self.factory = ModelFactory()
        self.comparator = ModelComparator(self.factory)
        self.evaluation_results = {}
        self.ensemble_results = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load and prepare data for evaluation"""
        print("Loading data for model evaluation...")
        
        # Load AAPL stock data
        aapl_file = str(AAPL_FILE)
        if os.path.exists(aapl_file):
            data = pd.read_csv(aapl_file)
            data['date'] = pd.to_datetime(data['date'])
            print(f"✓ Loaded AAPL data: {len(data)} records")
        else:
            print(f"✗ AAPL data file not found: {aapl_file}")
            return None
        
        # Create target variables for different horizons
        horizons = ['1d', '3d', '5d']
        for horizon in horizons:
            days = int(horizon.replace('d', ''))
            data[f'target_{horizon}'] = (data['close'].shift(-days) > data['close']).astype(int)
        
        # Remove rows with NaN targets
        data = data.dropna(subset=[f'target_{h}' for h in horizons])
        
        print(f"Final dataset: {len(data)} records")
        print(f"Date range: {data['date'].min()} to {data['date'].max()}")
        
        return data
    
    def evaluate_single_model(self, model_name: str, data: pd.DataFrame, 
                            horizon: str = '1d', cv_folds: int = 5) -> Dict[str, Any]:
        """Evaluate a single model with cross-validation"""
        print(f"\nEvaluating {model_name} for {horizon} horizon...")
        
        try:
            # Get model
            model = self.factory.get_model(model_name, horizon)
            if model is None:
                return {'error': f'Failed to create model {model_name}'}
            
            # Prepare features
            if hasattr(model, 'prepare_features'):
                X_features = model.prepare_features(data)
            else:
                X_features = data
            
            # Get target
            target_col = f'target_{horizon}'
            if target_col not in data.columns:
                return {'error': f'Target column {target_col} not found'}
            
            y = data[target_col]
            
            # Handle different model types
            if model_name in ['kronos_aapl', 'kronos_vgt', 'sentiment_analysis']:
                # These are score generators, not direct predictors
                return self._evaluate_score_generator(model, data, horizon)
            
            elif hasattr(model, 'train'):
                # Trainable models (ML/DL)
                return self._evaluate_trainable_model(model, X_features, y, cv_folds)
            
            else:
                # Technical analysis models
                return self._evaluate_technical_model(model, X_features, y)
                
        except Exception as e:
            print(f"✗ Error evaluating {model_name}: {e}")
            return {'error': str(e)}
    
    def _evaluate_score_generator(self, model, data: pd.DataFrame, horizon: str) -> Dict[str, Any]:
        """Evaluate score generator models"""
        try:
            # Check for cached scores first
            model_class_name = model.__class__.__name__.lower()
            score_file = None
            
            if 'aapl' in model_class_name:
                score_file = BENCHMARKS_OUTPUT_DIR / "aapl_scores.csv"
            elif 'vgt' in model_class_name:
                score_file = BENCHMARKS_OUTPUT_DIR / "vgt_scores.csv"
            elif 'sentiment' in model_class_name:
                score_file = BENCHMARKS_OUTPUT_DIR / "sentiment_scores.csv"
            
            # Try to load cached scores
            scores_df = None
            if score_file and os.path.exists(score_file):
                print(f"  Loading cached scores from {score_file.name}...")
                scores_df = pd.read_csv(score_file)
                scores_df['date'] = pd.to_datetime(scores_df['date'])
                print(f"  ✓ Loaded {len(scores_df)} cached records")
            
            # Generate scores if not cached
            if scores_df is None:
                if hasattr(model, 'generate_scores'):
                    print(f"  Generating scores for {model_class_name}...")
                    scores_df = model.generate_scores()
                    if scores_df is None:
                        return {'error': 'Failed to generate scores'}
                    
                    # Cache the scores for future use
                    if score_file:
                        scores_df.to_csv(score_file, index=False)
                        print(f"  ✓ Cached scores to {score_file.name}")

            # Ensure 'date' dtypes match for safe merge
            if 'date' in data.columns:
                data_date = data['date']
                if not pd.api.types.is_datetime64_any_dtype(data_date):
                    try:
                        data = data.copy()
                        data['date'] = pd.to_datetime(data['date'])
                    except Exception:
                        pass
            if 'date' in scores_df.columns:
                scores_date = scores_df['date']
                if not pd.api.types.is_datetime64_any_dtype(scores_date):
                    try:
                        scores_df = scores_df.copy()
                        scores_df['date'] = pd.to_datetime(scores_df['date'])
                    except Exception:
                        pass

            # Merge with main data
            merged_data = data.merge(scores_df, on='date', how='inner')

            # Get prediction column
            pred_col = f'{model.__class__.__name__.lower().replace("scoregenerator", "")}_pred_{horizon}'
            if pred_col not in merged_data.columns:
                # Try alternative column names
                possible_cols = [col for col in merged_data.columns if f'pred_{horizon}' in col]
                if possible_cols:
                    pred_col = possible_cols[0]
                else:
                    return {'error': f'Prediction column not found for {horizon}'}

            # Calculate metrics
            y_true = merged_data[f'target_{horizon}'].values
            y_pred = (merged_data[pred_col] > 0).astype(int)

            return self._calculate_metrics(y_true, y_pred, model.__class__.__name__)
                
        except Exception as e:
            return {'error': f'Score generator evaluation failed: {e}'}
    
    def _evaluate_trainable_model(self, model, X_features: pd.DataFrame, y: pd.Series, cv_folds: int) -> Dict[str, Any]:
        """Evaluate trainable models with cross-validation"""
        try:
            # Prepare features for ML models
            if hasattr(model, 'prepare_features'):
                X_processed = model.prepare_features(X_features)
            else:
                X_processed = X_features
            
            # Drop target columns from features to prevent leakage
            target_cols = [c for c in X_processed.columns if c.startswith('target_')]
            if target_cols:
                X_processed = X_processed.drop(columns=target_cols, errors='ignore')
            
            # Select numeric columns only
            numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
            X_numeric = X_processed[numeric_cols].fillna(0)
            
            # Use TimeSeriesSplit for time series data
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            
            # Cross-validation scores
            cv_scores = []
            cv_precisions = []
            cv_recalls = []
            cv_f1s = []
            
            for train_idx, val_idx in tscv.split(X_numeric):
                X_train, X_val = X_numeric.iloc[train_idx], X_numeric.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train model
                train_metrics = model.train(X_train, y_train)
                
                # Make predictions and align lengths if needed (e.g., sequence models)
                y_pred = model.predict(X_val)
                min_len = min(len(y_val), len(y_pred))
                y_val = y_val.iloc[:min_len]
                y_pred = y_pred[:min_len]
                
                # Calculate metrics
                accuracy = accuracy_score(y_val, y_pred)
                precision = precision_score(y_val, y_pred, average='binary', zero_division=0)
                recall = recall_score(y_val, y_pred, average='binary', zero_division=0)
                f1 = f1_score(y_val, y_pred, average='binary', zero_division=0)
                
                cv_scores.append(accuracy)
                cv_precisions.append(precision)
                cv_recalls.append(recall)
                cv_f1s.append(f1)
            
            return {
                'accuracy_mean': np.mean(cv_scores),
                'accuracy_std': np.std(cv_scores),
                'precision_mean': np.mean(cv_precisions),
                'precision_std': np.std(cv_precisions),
                'recall_mean': np.mean(cv_recalls),
                'recall_std': np.std(cv_recalls),
                'f1_mean': np.mean(cv_f1s),
                'f1_std': np.std(cv_f1s),
                'cv_scores': cv_scores,
                'model_type': 'trainable'
            }
            
        except Exception as e:
            return {'error': f'Trainable model evaluation failed: {e}'}
    
    def _evaluate_technical_model(self, model, X_features: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Evaluate technical analysis models"""
        try:
            # Make predictions
            predictions = model.predict(X_features)
            
            # Align predictions with targets
            min_len = min(len(predictions), len(y))
            y_true = y.iloc[:min_len].values
            y_pred = predictions[:min_len]
            
            return self._calculate_metrics(y_true, y_pred, model.__class__.__name__)
            
        except Exception as e:
            return {'error': f'Technical model evaluation failed: {e}'}
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Dict[str, Any]:
        """Calculate standard metrics"""
        try:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
            
            return {
                'accuracy_mean': accuracy,
                'accuracy_std': 0.0,
                'precision_mean': precision,
                'precision_std': 0.0,
                'recall_mean': recall,
                'recall_std': 0.0,
                'f1_mean': f1,
                'f1_std': 0.0,
                'model_type': 'technical',
                'predictions': y_pred,
                'targets': y_true
            }
            
        except Exception as e:
            return {'error': f'Metrics calculation failed: {e}'}
    
    def evaluate_all_models(self, horizons: List[str] = None, 
                          categories: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """Evaluate all available models"""
        if horizons is None:
            horizons = ['1d', '3d', '5d']
        
        if categories is None:
            categories = ['ml', 'dl', 'existing']
        
        print("Starting comprehensive model evaluation...")
        print("=" * 60)
        
        # Load data
        data = self.load_data()
        if data is None:
            print("✗ Failed to load data")
            return {}
        
        all_results = {}
        
        for horizon in horizons:
            print(f"\n{'='*20} {horizon.upper()} HORIZON {'='*20}")
            
            horizon_results = {}
            
            # Get models by category
            for category in categories:
                print(f"\nEvaluating {category.upper()} models...")
                
                category_models = self.factory.get_available_models(category)
                
                # Ignore old names if present
                for model_name in category_models.keys():
                    result = self.evaluate_single_model(model_name, data, horizon)
                    
                    if 'error' not in result:
                        horizon_results[model_name] = result
                        print(f"  ✓ {model_name}: Accuracy={result['accuracy_mean']:.4f}")
                    else:
                        print(f"  ✗ {model_name}: {result['error']}")
            
            all_results[horizon] = horizon_results
        
        self.evaluation_results = all_results
        return all_results
    
    def create_performance_visualization(self, output_dir: str = None):
        """Create comprehensive performance visualizations"""
        if output_dir is None:
            output_dir = str(BENCHMARKS_OUTPUT_DIR)
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nCreating performance visualizations...")
        
        # 1. Model Performance Comparison
        self._create_model_comparison_plot(output_dir)
        
        # 2. Category Performance Analysis
        self._create_category_analysis_plot(output_dir)
        
        # 3. Horizon Comparison
        self._create_horizon_comparison_plot(output_dir)
        
        # 4. Performance Distribution
        self._create_performance_distribution_plot(output_dir)
        
        print(f"✓ Visualizations saved to {output_dir}")
    
    def _create_model_comparison_plot(self, output_dir: str):
        """Create model comparison plot"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        horizons = ['1d', '3d', '5d']
        metrics = ['accuracy_mean', 'precision_mean', 'recall_mean', 'f1_mean']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i//2, i%2]
            
            # Collect data for all models across horizons
            model_data = {}
            
            for horizon in horizons:
                if horizon in self.evaluation_results:
                    for model_name, result in self.evaluation_results[horizon].items():
                        if metric in result:
                            if model_name not in model_data:
                                model_data[model_name] = {}
                            model_data[model_name][horizon] = result[metric]
            
            # Create bar plot
            if model_data:
                model_names = list(model_data.keys())
                x = np.arange(len(model_names))
                width = 0.25
                
                for j, horizon in enumerate(horizons):
                    values = [model_data[model].get(horizon, 0) for model in model_names]
                    ax.bar(x + j*width, values, width, label=f'{horizon} horizon', alpha=0.8)
                
                ax.set_xlabel('Models')
                ax.set_ylabel(metric_name)
                ax.set_title(f'{metric_name} Comparison Across Horizons')
                ax.set_xticks(x + width)
                ax.set_xticklabels(model_names, rotation=45, ha='right')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to free memory
    
    def _create_category_analysis_plot(self, output_dir: str):
        """Create category analysis plot"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        horizons = ['1d', '3d', '5d']
        metrics = ['accuracy_mean', 'precision_mean', 'recall_mean', 'f1_mean']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i//2, i%2]
            
            # Group by category
            category_data = {}
            
            for horizon in horizons:
                if horizon in self.evaluation_results:
                    for model_name, result in self.evaluation_results[horizon].items():
                        if metric in result:
                            # Get category
                            model_info = self.factory.get_model_info(model_name)
                            category = model_info['category']
                            
                            if category not in category_data:
                                category_data[category] = {}
                            if horizon not in category_data[category]:
                                category_data[category][horizon] = []
                            
                            category_data[category][horizon].append(result[metric])
            
            # Create box plot
            if category_data:
                categories = list(category_data.keys())
                x = np.arange(len(categories))
                width = 0.25
                
                for j, horizon in enumerate(horizons):
                    values = []
                    for category in categories:
                        if horizon in category_data[category]:
                            values.append(np.mean(category_data[category][horizon]))
                        else:
                            values.append(0)
                    
                    ax.bar(x + j*width, values, width, label=f'{horizon} horizon', alpha=0.8)
                
                ax.set_xlabel('Model Categories')
                ax.set_ylabel(metric_name)
                ax.set_title(f'{metric_name} by Category')
                ax.set_xticks(x + width)
                ax.set_xticklabels(categories)
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/category_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to free memory
    
    def _create_horizon_comparison_plot(self, output_dir: str):
        """Create horizon comparison plot"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        horizons = ['1d', '3d', '5d']
        metrics = ['accuracy_mean', 'precision_mean', 'recall_mean', 'f1_mean']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i//2, i%2]
            
            # Collect data for each horizon
            horizon_data = {}
            
            for horizon in horizons:
                if horizon in self.evaluation_results:
                    values = []
                    for model_name, result in self.evaluation_results[horizon].items():
                        if metric in result:
                            values.append(result[metric])
                    horizon_data[horizon] = values
            
            # Create violin plot
            if horizon_data:
                data_for_plot = []
                labels = []
                
                for horizon in horizons:
                    if horizon in horizon_data and horizon_data[horizon]:
                        data_for_plot.append(horizon_data[horizon])
                        labels.append(f'{horizon} horizon')
                
                if data_for_plot:
                    ax.violinplot(data_for_plot, positions=range(len(data_for_plot)))
                    ax.set_xticks(range(len(labels)))
                    ax.set_xticklabels(labels)
                    ax.set_ylabel(metric_name)
                    ax.set_title(f'{metric_name} Distribution by Horizon')
                    ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/horizon_comparison_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to free memory
    
    def _create_performance_distribution_plot(self, output_dir: str):
        """Create performance distribution plot"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        horizons = ['1d', '3d', '5d']
        metrics = ['accuracy_mean', 'precision_mean', 'recall_mean', 'f1_mean']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i//2, i%2]
            
            # Collect all values across all models and horizons
            all_values = []
            
            for horizon in horizons:
                if horizon in self.evaluation_results:
                    for model_name, result in self.evaluation_results[horizon].items():
                        if metric in result:
                            all_values.append(result[metric])
            
            # Create histogram
            if all_values:
                ax.hist(all_values, bins=20, alpha=0.7, edgecolor='black')
                ax.axvline(np.mean(all_values), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(all_values):.3f}')
                ax.axvline(np.median(all_values), color='green', linestyle='--', 
                          label=f'Median: {np.median(all_values):.3f}')
                ax.set_xlabel(metric_name)
                ax.set_ylabel('Frequency')
                ax.set_title(f'{metric_name} Distribution')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/performance_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to free memory
    
    def get_top_models(self, horizon: str = '1d', metric: str = 'accuracy_mean', top_k: int = 10) -> List[Dict[str, Any]]:
        """Get top-k performing models"""
        if horizon not in self.evaluation_results:
            return []
        
        results = self.evaluation_results[horizon]
        
        # Filter models with valid results
        valid_results = {name: result for name, result in results.items() 
                        if metric in result and 'error' not in result}
        
        # Sort by metric
        sorted_models = sorted(valid_results.items(), 
                             key=lambda x: x[1].get(metric, 0), 
                             reverse=True)
        
        # Return top-k
        top_models = []
        for i, (model_name, metrics) in enumerate(sorted_models[:top_k]):
            model_info = self.factory.get_model_info(model_name)
            top_models.append({
                'rank': i + 1,
                'model_name': model_name,
                'metrics': metrics,
                'model_info': model_info
            })
        
        return top_models
    
    def print_evaluation_summary(self):
        """Print comprehensive evaluation summary"""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE MODEL EVALUATION SUMMARY")
        print(f"{'='*80}")
        
        for horizon in ['1d', '3d', '5d']:
            if horizon in self.evaluation_results:
                print(f"\n{horizon.upper()} HORIZON:")
                print("-" * 40)
                
                # Get top models
                top_models = self.get_top_models(horizon, 'accuracy_mean', 5)
                
                print(f"{'Rank':<4} {'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
                print("-" * 80)
                
                for model in top_models:
                    metrics = model['metrics']
                    print(f"{model['rank']:<4} {model['model_name']:<25} "
                          f"{metrics['accuracy_mean']:<10.4f} {metrics['precision_mean']:<10.4f} "
                          f"{metrics['recall_mean']:<10.4f} {metrics['f1_mean']:<10.4f}")
                
                print("-" * 80)
                
                # Category breakdown
                print(f"\nCategory Breakdown for {horizon}:")
                categories = {}
                for model_name, result in self.evaluation_results[horizon].items():
                    if 'error' not in result:
                        model_info = self.factory.get_model_info(model_name)
                        category = model_info['category']
                        if category not in categories:
                            categories[category] = []
                        categories[category].append(result['accuracy_mean'])
                
                for category, accuracies in categories.items():
                    print(f"  {category.upper()}: Mean={np.mean(accuracies):.4f}, "
                          f"Best={np.max(accuracies):.4f}, Count={len(accuracies)}")
    
    def save_results(self, output_dir: str = None):
        """Save evaluation results"""
        if output_dir is None:
            output_dir = str(BENCHMARKS_OUTPUT_DIR)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        results_file = os.path.join(output_dir, 'comprehensive_evaluation_results.pkl')
        joblib.dump(self.evaluation_results, results_file)
        print(f"✓ Detailed results saved to: {results_file}")
        
        # Save summary report
        summary_file = os.path.join(output_dir, 'evaluation_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("COMPREHENSIVE MODEL EVALUATION SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            for horizon in ['1d', '3d', '5d']:
                if horizon in self.evaluation_results:
                    f.write(f"{horizon.upper()} HORIZON:\n")
                    f.write("-" * 40 + "\n")
                    
                    # Get top models
                    top_models = self.get_top_models(horizon, 'accuracy_mean', 10)
                    
                    f.write(f"{'Rank':<4} {'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}\n")
                    f.write("-" * 80 + "\n")
                    
                    for model in top_models:
                        metrics = model['metrics']
                        f.write(f"{model['rank']:<4} {model['model_name']:<25} "
                               f"{metrics['accuracy_mean']:<10.4f} {metrics['precision_mean']:<10.4f} "
                               f"{metrics['recall_mean']:<10.4f} {metrics['f1_mean']:<10.4f}\n")
                    
                    f.write("-" * 80 + "\n\n")
        
        print(f"✓ Summary report saved to: {summary_file}")

        # Save a plain-text full dump for custom visualizations later
        full_txt = os.path.join(output_dir, 'evaluation_results_full.txt')
        with open(full_txt, 'w') as f:
            f.write("FULL EVALUATION RESULTS\n")
            f.write("="*80 + "\n\n")
            for horizon in ['1d', '3d', '5d']:
                f.write(f"{horizon.upper()} HORIZON\n")
                f.write("-"*80 + "\n")
                if horizon in self.evaluation_results:
                    for model_name, metrics in self.evaluation_results[horizon].items():
                        if 'error' in metrics:
                            continue
                        f.write(f"Model: {model_name}\n")
                        for k in ['accuracy_mean','precision_mean','recall_mean','f1_mean','accuracy_std','precision_std','recall_std','f1_std']:
                            if k in metrics:
                                f.write(f"  {k}: {metrics[k]:.6f}\n")
                        f.write("\n")
                f.write("\n")
        print(f"✓ Full text results saved to: {full_txt}")

    def tune_model(self, model_name: str, horizon: str = '1d', param_grid: Dict[str, Any] = None, cv_folds: int = 5, output_dir: str = None):
        """Time-series CV tuning for a single ML/DL model; writes TXT/CSV outputs."""
        info = self.factory.get_model_info(model_name)
        if info['category'] not in ('ml', 'dl'):
            raise ValueError('Tuning is only supported for ML/DL models')

        data = self.load_data()
        target_col = f'target_{horizon}'
        if target_col not in data.columns:
            raise ValueError(f'Missing target column {target_col}')
        y = data[target_col]

        # Build features via model's own pipeline
        base_model = self.factory.create_model(model_name, horizon)
        X_features = base_model.prepare_features(data) if hasattr(base_model, 'prepare_features') else data
        # Drop labels to prevent leakage
        X_features = X_features.drop(columns=[c for c in X_features.columns if c.startswith('target_')], errors='ignore')
        X_numeric = X_features.select_dtypes(include=[np.number]).fillna(0)

        tscv = TimeSeriesSplit(n_splits=cv_folds)
        grid = param_grid or self.factory.get_param_grid(model_name)
        if not grid:
            raise ValueError('Param grid is empty. Provide param_grid or define it in ModelFactory.')

        results = []
        for params in ParameterGrid(grid):
            model = self.factory.create_model_with_params(model_name, horizon, params)
            fold_scores = []
            for train_idx, val_idx in tscv.split(X_numeric):
                X_train, X_val = X_numeric.iloc[train_idx], X_numeric.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                model.train(X_train, y_train)
                y_pred = model.predict(X_val)
                n = min(len(y_val), len(y_pred))
                fold_scores.append(accuracy_score(y_val.iloc[:n], y_pred[:n]))
            results.append({'params': params, 'cv_mean': float(np.mean(fold_scores)), 'cv_std': float(np.std(fold_scores))})

        # Sort and save
        results.sort(key=lambda r: r['cv_mean'], reverse=True)
        base_dir = output_dir or str(BENCHMARKS_OUTPUT_DIR)
        out_dir = os.path.join(base_dir, 'tuning')
        os.makedirs(out_dir, exist_ok=True)
        txt_path = os.path.join(out_dir, f'tuning_{model_name}_{horizon}.txt')
        with open(txt_path, 'w') as f:
            for r in results:
                f.write(f"{r['params']} => mean={r['cv_mean']:.4f}, std={r['cv_std']:.4f}\n")
        try:
            import csv
            csv_path = os.path.join(out_dir, f'tuning_{model_name}_{horizon}.csv')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['cv_mean','cv_std','params'])
                writer.writeheader()
                for r in results:
                    writer.writerow({'cv_mean': r['cv_mean'], 'cv_std': r['cv_std'], 'params': r['params']})
        except Exception:
            pass
        print(f"Saved tuning results to: {txt_path}")
        return results

def main():
    """Main execution function"""
    print("Comprehensive Model Comparison Framework")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = ComprehensiveModelEvaluator()
    
    # Evaluate all models
    results = evaluator.evaluate_all_models()
    
    if results:
        # Print summary
        evaluator.print_evaluation_summary()
        
        # Create visualizations
        evaluator.create_performance_visualization()
        
        # Save results
        evaluator.save_results()
        
        print("\n🎉 Comprehensive model evaluation completed successfully!")
        
        return evaluator
    else:
        print("✗ Model evaluation failed!")
        return None

if __name__ == "__main__":
    evaluator = main()
