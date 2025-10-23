#!/usr/bin/env python3
"""
Model Factory - Unified Management of All Models
===============================================

This module provides a factory pattern for creating and managing all models:
1. Benchmark Models (Technical, ML, DL)
2. Kronos Models (AAPL, VGT)
3. Sentiment Analysis Model

All models follow a unified interface for easy comparison and ensemble learning.

Date: 2025
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Type
import warnings
warnings.filterwarnings('ignore')

# Import benchmark models
from benchmarks.benchmark_models import (
    # Baseline Machine Learning Models
    RandomForestModel, LogisticRegressionModel, XGBoostModel,
    # Baseline Deep Learning Model
    LSTMModel
)

# Import existing models
from analysis.kronos_aapl_analysis import AAPLScoreGenerator
from analysis.kronos_vgt_analysis import VGTScoreGenerator
from sentiment.sentiment_analysis import SentimentScoreGenerator

class ModelFactory:
    """Factory class for creating and managing all models"""
    
    def __init__(self):
        self.available_models = self._register_models()
        self.best_params = self._load_best_params()
        self.model_instances = {}

    def _best_params_path(self) -> str:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_params.json')

    def _load_best_params(self) -> dict:
        """Load team-agreed best parameters if present."""
        path = self._best_params_path()
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        return {}
        
    def _register_models(self) -> Dict[str, Dict[str, Any]]:
        """Register all available models"""
        models = {
            # Baseline Machine Learning Models
            'logistic_regression': {
                'class': LogisticRegressionModel,
                'params': {'C': 1.0},
                'category': 'ml',
                'description': 'Logistic Regression',
                'param_grid': {
                    'C': [0.01, 0.1, 1.0, 10.0]
                }
            },
            'xgboost': {
                'class': XGBoostModel,
                'params': {'n_estimators': 100, 'learning_rate': 0.1},
                'category': 'ml',
                'description': 'XGBoost',
                'param_grid': {
                    'n_estimators': [100, 300],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5]
                }
            },
            'random_forest': {
                'class': RandomForestModel,
                'params': {'n_estimators': 100, 'max_depth': 10},
                'category': 'ml',
                'description': 'Random Forest',
                'param_grid': {
                    'n_estimators': [100, 300, 500],
                    'max_depth': [None, 10, 20]
                }
            },

            # Baseline Deep Learning Model
            'lstm': {
                'class': LSTMModel,
                'params': {
                    'hidden_size': 64,
                    'num_layers': 2,
                    'sequence_length': 20,
                    'epochs': 50,
                    'lr': 0.001,
                    'batch_size': 32
                },
                'category': 'dl',
                'description': 'LSTM',
                'param_grid': {
                    'hidden_size': [32, 64, 128],
                    'num_layers': [1, 2],
                    'sequence_length': [20, 30],
                    'epochs': [20, 40],
                    'lr': [1e-3, 5e-4],
                    'batch_size': [32, 64]
                }
            },
            
            # Existing Models
            'kronos_aapl': {
                'class': AAPLScoreGenerator,
                'params': {},
                'category': 'existing',
                'description': 'Kronos AAPL Model'
            },
            'kronos_vgt': {
                'class': VGTScoreGenerator,
                'params': {},
                'category': 'existing',
                'description': 'Kronos VGT Model'
            },
            'sentiment_analysis': {
                'class': SentimentScoreGenerator,
                'params': {},
                'category': 'existing',
                'description': 'Sentiment Analysis Model'
            }
        }
        
        return models
    
    def get_available_models(self, category: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Get available models, optionally filtered by category"""
        if category is None:
            return self.available_models
        
        return {name: info for name, info in self.available_models.items() 
                if info['category'] == category}
    
    def get_param_grid(self, model_name: str) -> Dict[str, Any]:
        info = self.available_models.get(model_name, {})
        return info.get('param_grid', {})
    
    def create_model_with_params(self, model_name: str, horizon: str = '1d', override_params: Optional[Dict[str, Any]] = None) -> Any:
        """Create a model instance with explicit parameter overrides."""
        if model_name not in self.available_models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.available_models.keys())}")
        model_info = self.available_models[model_name]
        model_class = model_info['class']
        params = model_info['params'].copy()
        # merge best params then overrides
        if model_name in self.best_params and isinstance(self.best_params[model_name], dict):
            bp = self.best_params[model_name]
            # Prefer horizon-specific block if present
            if horizon in bp and isinstance(bp[horizon], dict):
                params.update(bp[horizon])
            else:
                params.update(bp)
        if override_params:
            params.update(override_params)
        if model_info['category'] in ('ml', 'dl'):
            params['horizon'] = horizon
        try:
            instance = model_class(**params)
            self.model_instances[f"{model_name}_{horizon}"] = instance
            return instance
        except Exception as e:
            print(f"Error creating model {model_name} with params {params}: {e}")
            return None

    def create_model(self, model_name: str, horizon: str = '1d') -> Any:
        """Create a model instance with best_params merged if present"""
        if model_name not in self.available_models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.available_models.keys())}")
        
        model_info = self.available_models[model_name]
        model_class = model_info['class']
        params = model_info['params'].copy()
        category = model_info.get('category')
        # merge best params if available
        if model_name in self.best_params and isinstance(self.best_params[model_name], dict):
            bp = self.best_params[model_name]
            if horizon in bp and isinstance(bp[horizon], dict):
                params.update(bp[horizon])
            else:
                params.update(bp)
        
        # Add horizon parameter only for ML/DL models
        if category in ('ml', 'dl'):
            params['horizon'] = horizon
        
        # Create model instance
        try:
            model_instance = model_class(**params)
            
            # Store instance
            instance_key = f"{model_name}_{horizon}"
            self.model_instances[instance_key] = model_instance
            
            return model_instance
            
        except Exception as e:
            print(f"Error creating model {model_name}: {e}")
            return None
    
    def get_model(self, model_name: str, horizon: str = '1d') -> Any:
        """Get existing model instance or create new one"""
        instance_key = f"{model_name}_{horizon}"
        
        if instance_key in self.model_instances:
            return self.model_instances[instance_key]
        
        return self.create_model(model_name, horizon)
    
    def list_models_by_category(self) -> Dict[str, List[str]]:
        """List models grouped by category"""
        categories = {}
        
        for model_name, model_info in self.available_models.items():
            category = model_info['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(model_name)
        
        return categories
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a model"""
        if model_name not in self.available_models:
            raise ValueError(f"Model '{model_name}' not found")
        
        return self.available_models[model_name]
    
    def create_model_group(self, model_names: List[str], horizon: str = '1d') -> Dict[str, Any]:
        """Create multiple models at once"""
        models = {}
        
        for model_name in model_names:
            model = self.create_model(model_name, horizon)
            if model is not None:
                models[model_name] = model
        
        return models
    
    def create_all_models(self, horizon: str = '1d', categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """Create all available models"""
        if categories is None:
            categories = ['ml', 'dl', 'existing']
        
        models = {}
        
        for model_name, model_info in self.available_models.items():
            if model_info['category'] in categories:
                model = self.create_model(model_name, horizon)
                if model is not None:
                    models[model_name] = model
        
        return models

class ModelComparator:
    """Class for comparing multiple models"""
    
    def __init__(self, factory: ModelFactory):
        self.factory = factory
        self.comparison_results = {}
    
    def compare_models(self, model_names: List[str], data: pd.DataFrame, 
                      target_col: str = 'target', horizon: str = '1d') -> Dict[str, Dict[str, float]]:
        """Compare multiple models on the same dataset"""
        print(f"Comparing {len(model_names)} models for {horizon} horizon...")
        
        results = {}
        
        for model_name in model_names:
            print(f"\nEvaluating {model_name}...")
            
            try:
                # Get or create model
                model = self.factory.get_model(model_name, horizon)
                
                if model is None:
                    print(f"✗ Failed to create model {model_name}")
                    continue
                
                # Prepare data
                if hasattr(model, 'prepare_features'):
                    X_features = model.prepare_features(data)
                else:
                    X_features = data
                
                # Split data for training and testing
                split_idx = int(len(data) * 0.8)
                train_data = data.iloc[:split_idx]
                test_data = data.iloc[split_idx:]
                
                # Train model (if it's a trainable model)
                if hasattr(model, 'train') and model_name not in ['kronos_aapl', 'kronos_vgt', 'sentiment_analysis']:
                    train_metrics = model.train(train_data, train_data[target_col])
                    print(f"  Training accuracy: {train_metrics.get('accuracy', 0):.4f}")
                
                # Make predictions
                if hasattr(model, 'predict'):
                    predictions = model.predict(test_data)
                    
                    # Calculate metrics
                    y_true = test_data[target_col].values
                    y_pred = predictions
                    
                    # Handle different prediction formats
                    if len(y_pred) != len(y_true):
                        # Adjust predictions to match test data length
                        y_pred = y_pred[:len(y_true)]
                    
                    # Calculate metrics
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    
                    accuracy = accuracy_score(y_true, y_pred)
                    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
                    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
                    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
                    
                    results[model_name] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'predictions': y_pred,
                        'model_info': self.factory.get_model_info(model_name)
                    }
                    
                    print(f"  ✓ Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
                
                else:
                    print(f"  ✗ Model {model_name} doesn't have predict method")
                
            except Exception as e:
                print(f"  ✗ Error evaluating {model_name}: {e}")
                import traceback
                traceback.print_exc()
        
        self.comparison_results[horizon] = results
        return results
    
    def get_best_models(self, horizon: str = '1d', metric: str = 'accuracy', top_k: int = 5) -> List[Dict[str, Any]]:
        """Get top-k performing models"""
        if horizon not in self.comparison_results:
            return []
        
        results = self.comparison_results[horizon]
        
        # Sort by metric
        sorted_models = sorted(results.items(), 
                             key=lambda x: x[1].get(metric, 0), 
                             reverse=True)
        
        # Return top-k
        top_models = []
        for i, (model_name, metrics) in enumerate(sorted_models[:top_k]):
            top_models.append({
                'rank': i + 1,
                'model_name': model_name,
                'metrics': metrics,
                'model_info': metrics['model_info']
            })
        
        return top_models
    
    def print_comparison_summary(self, horizon: str = '1d'):
        """Print comparison summary"""
        if horizon not in self.comparison_results:
            print(f"No comparison results available for {horizon}")
            return
        
        results = self.comparison_results[horizon]
        
        print(f"\n{'='*60}")
        print(f"MODEL COMPARISON SUMMARY - {horizon.upper()} HORIZON")
        print(f"{'='*60}")
        
        # Sort by accuracy
        sorted_results = sorted(results.items(), 
                              key=lambda x: x[1].get('accuracy', 0), 
                              reverse=True)
        
        print(f"{'Rank':<4} {'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 80)
        
        for i, (model_name, metrics) in enumerate(sorted_results):
            print(f"{i+1:<4} {model_name:<25} {metrics['accuracy']:<10.4f} "
                  f"{metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                  f"{metrics['f1_score']:<10.4f}")
        
        print("-" * 80)
        
        # Category breakdown
        print(f"\nCATEGORY BREAKDOWN:")
        categories = {}
        for model_name, metrics in results.items():
            category = metrics['model_info']['category']
            if category not in categories:
                categories[category] = []
            categories[category].append((model_name, metrics['accuracy']))
        
        for category, models in categories.items():
            print(f"\n{category.upper()} MODELS:")
            sorted_models = sorted(models, key=lambda x: x[1], reverse=True)
            for model_name, accuracy in sorted_models:
                print(f"  {model_name}: {accuracy:.4f}")

def test_model_factory():
    """Test the model factory"""
    print("Testing Model Factory")
    print("=" * 50)
    
    # Create factory
    factory = ModelFactory()
    
    # List available models
    print("\nAvailable Models by Category:")
    categories = factory.list_models_by_category()
    for category, models in categories.items():
        print(f"\n{category.upper()}:")
        for model in models:
            info = factory.get_model_info(model)
            print(f"  {model}: {info['description']}")
    
    # Test creating models
    print(f"\nTesting Model Creation:")
    test_models = ['logistic_regression', 'random_forest', 'xgboost']
    
    for model_name in test_models:
        model = factory.create_model(model_name, '1d')
        if model is not None:
            print(f"✓ Created {model_name}: {type(model).__name__}")
        else:
            print(f"✗ Failed to create {model_name}")
    
    # Test model groups
    print(f"\nTesting Model Groups:")
    # (technical models disabled)
    print("Technical models are disabled in this configuration")
    
    print("\nModel factory test completed!")

def test_model_comparator():
    """Test the model comparator"""
    print("\nTesting Model Comparator")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate sample OHLCV data
    base_price = 100
    prices = []
    for i in range(n_samples):
        change = np.random.normal(0, 0.02)
        base_price *= (1 + change)
        prices.append(base_price)
    
    # Create OHLCV data
    data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=n_samples),
        'open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, n_samples)
    })
    
    # Create target (simple trend following)
    data['target'] = (data['close'].shift(-1) > data['close']).astype(int)
    
    # Create factory and comparator
    factory = ModelFactory()
    comparator = ModelComparator(factory)
    
    # Test comparison
    test_models = ['logistic_regression', 'random_forest', 'xgboost']
    
    results = comparator.compare_models(test_models, data, 'target', '1d')
    
    # Print summary
    comparator.print_comparison_summary('1d')
    
    # Get best models
    best_models = comparator.get_best_models('1d', 'accuracy', 3)
    print(f"\nTop 3 Models:")
    for model in best_models:
        print(f"  {model['rank']}. {model['model_name']}: {model['metrics']['accuracy']:.4f}")
    
    print("\nModel comparator test completed!")

if __name__ == "__main__":
    test_model_factory()
    test_model_comparator()
