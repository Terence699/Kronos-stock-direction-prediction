#!/usr/bin/env python3
"""
Score Ensemble Module - Grid Search for Optimal Classification Parameters
======================================================================

This module combines scores from three sources:
1. AAPL Kronos scores (kronos_aapl_analysis.py)
2. VGT Kronos scores (kronos_vgt_analysis.py)  
3. Sentiment scores (sentiment_analysis.py)

Performs grid search to find optimal parameters for 1d, 3d, 5d classification.

Author: AI Assistant
Date: 2025
"""

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

# Import our scoring modules
from kronos_aapl_analysis import AAPLScoreGenerator
from kronos_vgt_analysis import VGTScoreGenerator
from sentiment_analysis import SentimentScoreGenerator

def get_project_root():
    """Get the project root directory"""
    return os.path.dirname(os.path.abspath(__file__))

class ScoreEnsemble:
    """Ensemble learning module for combining multiple score sources"""
    
    def __init__(self):
        self.aapl_scores = None
        self.vgt_scores = None
        self.sentiment_scores = None
        self.combined_data = None
        self.best_models = {}
        self.grid_search_results = {}
        
    def load_all_scores(self):
        """Load pre-computed scores from CSV files or generate them if missing"""
        print("Loading pre-computed scores from files...")
        
        # Load AAPL scores
        aapl_file = os.path.join(get_project_root(), 'aapl_scores.csv')
        if os.path.exists(aapl_file):
            self.aapl_scores = pd.read_csv(aapl_file)
            self.aapl_scores['date'] = pd.to_datetime(self.aapl_scores['date'])
            print(f"✓ Loaded AAPL scores: {len(self.aapl_scores)} records")
        else:
            print(f"⚠️  AAPL scores file not found: {aapl_file}")
            print("   Generating AAPL scores...")
            try:
                from kronos_aapl_analysis import AAPLScoreGenerator
                generator = AAPLScoreGenerator()
                self.aapl_scores = generator.generate_scores()
                if self.aapl_scores is not None:
                    self.aapl_scores['date'] = pd.to_datetime(self.aapl_scores['date'])
                    print(f"✓ Generated AAPL scores: {len(self.aapl_scores)} records")
                else:
                    print("✗ Failed to generate AAPL scores")
                    return False
            except Exception as e:
                print(f"✗ Error generating AAPL scores: {e}")
                return False
        
        # Load VGT scores
        vgt_file = os.path.join(get_project_root(), 'vgt_scores.csv')
        if os.path.exists(vgt_file):
            self.vgt_scores = pd.read_csv(vgt_file)
            self.vgt_scores['date'] = pd.to_datetime(self.vgt_scores['date'])
            print(f"✓ Loaded VGT scores: {len(self.vgt_scores)} records")
        else:
            print(f"⚠️  VGT scores file not found: {vgt_file}")
            print("   Generating VGT scores...")
            try:
                from kronos_vgt_analysis import VGTScoreGenerator
                generator = VGTScoreGenerator()
                self.vgt_scores = generator.generate_scores()
                if self.vgt_scores is not None:
                    self.vgt_scores['date'] = pd.to_datetime(self.vgt_scores['date'])
                    print(f"✓ Generated VGT scores: {len(self.vgt_scores)} records")
                else:
                    print("✗ Failed to generate VGT scores")
                    return False
            except Exception as e:
                print(f"✗ Error generating VGT scores: {e}")
                return False
        
        # Load sentiment scores
        sentiment_file = os.path.join(get_project_root(), 'sentiment_scores.csv')
        if os.path.exists(sentiment_file):
            self.sentiment_scores = pd.read_csv(sentiment_file)
            self.sentiment_scores['date'] = pd.to_datetime(self.sentiment_scores['date'])
            print(f"✓ Loaded sentiment scores: {len(self.sentiment_scores)} records")
        else:
            print(f"⚠️  Sentiment scores file not found: {sentiment_file}")
            print("   Generating sentiment scores...")
            try:
                from sentiment_analysis import SentimentScoreGenerator
                generator = SentimentScoreGenerator()
                self.sentiment_scores = generator.generate_scores()
                if self.sentiment_scores is not None:
                    self.sentiment_scores['date'] = pd.to_datetime(self.sentiment_scores['date'])
                    print(f"✓ Generated sentiment scores: {len(self.sentiment_scores)} records")
                else:
                    print("✗ Failed to generate sentiment scores")
                    return False
            except Exception as e:
                print(f"✗ Error generating sentiment scores: {e}")
                return False
        
        print("All scores loaded/generated successfully!")
        return True
    
    def combine_scores(self):
        """Combine all scores into a unified dataset"""
        print("Combining scores from all sources...")
        
        # Start with AAPL scores as base
        combined = self.aapl_scores.copy()
        
        # Merge VGT scores
        vgt_merge_cols = ['date'] + [col for col in self.vgt_scores.columns if col.startswith('vgt_')]
        combined = combined.merge(
            self.vgt_scores[vgt_merge_cols], 
            on='date', 
            how='inner'
        )
        
        # Merge sentiment scores
        sentiment_merge_cols = ['date'] + [col for col in self.sentiment_scores.columns if col.startswith('sentiment_')]
        combined = combined.merge(
            self.sentiment_scores[sentiment_merge_cols], 
            on='date', 
            how='inner'
        )
        
        # Sort by date
        combined = combined.sort_values('date').reset_index(drop=True)
        
        print(f"Combined dataset: {len(combined)} records")
        print(f"Date range: {combined['date'].min()} to {combined['date'].max()}")
        
        self.combined_data = combined
        return combined
    
    def prepare_features(self, horizon='1d'):
        """Prepare features for machine learning"""
        if self.combined_data is None:
            raise ValueError("No combined data available. Run combine_scores() first.")
        
        # Select feature columns
        feature_cols = []
        
        # AAPL features
        aapl_pred_col = f'aapl_pred_{horizon}'
        if aapl_pred_col in self.combined_data.columns:
            feature_cols.append(aapl_pred_col)
        
        # VGT features
        vgt_pred_col = f'vgt_pred_{horizon}'
        if vgt_pred_col in self.combined_data.columns:
            feature_cols.append(vgt_pred_col)
        
        # Sentiment features
        sentiment_pred_col = f'sentiment_pred_{horizon}'
        if sentiment_pred_col in self.combined_data.columns:
            feature_cols.append(sentiment_pred_col)
        
        # Additional features
        additional_features = [
            'sentiment_score', 'sentiment_strength', 'volume_weighted_sentiment'
        ]
        for feat in additional_features:
            if feat in self.combined_data.columns:
                feature_cols.append(feat)
        
        # Prepare X and y
        X = self.combined_data[feature_cols].fillna(0)
        
        # Target variable (actual direction)
        actual_col = f'aapl_actual_{horizon}'
        if actual_col in self.combined_data.columns:
            y = self.combined_data[actual_col]
        else:
            # Fallback: calculate from price changes
            return_col = f'aapl_return_{horizon}'
            if return_col in self.combined_data.columns:
                y = (self.combined_data[return_col] > 0).astype(int)
            else:
                raise ValueError(f"No target variable available for horizon {horizon}")
        
        # Remove rows with NaN targets
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"Features for {horizon}: {feature_cols}")
        print(f"Dataset shape: {X.shape}, Target shape: {y.shape}")
        
        return X, y, feature_cols
    
    def define_weight_combinations(self):
        """Define weight combinations for ensemble scoring"""
        # Grid search for optimal weight combinations
        # Weights should sum to 1.0
        weight_combinations = []
        
        # Generate weight combinations
        for aapl_weight in np.arange(0.1, 0.9, 0.1):
            for vgt_weight in np.arange(0.1, 0.9, 0.1):
                for sentiment_weight in np.arange(0.1, 0.9, 0.1):
                    # Check if weights sum to approximately 1.0
                    total_weight = aapl_weight + vgt_weight + sentiment_weight
                    if 0.95 <= total_weight <= 1.05:  # Allow small tolerance
                        # Normalize weights
                        normalized_weights = np.array([aapl_weight, vgt_weight, sentiment_weight])
                        normalized_weights = normalized_weights / normalized_weights.sum()
                        
                        weight_combinations.append({
                            'aapl_weight': normalized_weights[0],
                            'vgt_weight': normalized_weights[1], 
                            'sentiment_weight': normalized_weights[2]
                        })
        
        # Add some specific combinations that might work well
        specific_combinations = [
            {'aapl_weight': 0.5, 'vgt_weight': 0.3, 'sentiment_weight': 0.2},  # Equal-ish
            {'aapl_weight': 0.6, 'vgt_weight': 0.2, 'sentiment_weight': 0.2},  # AAPL heavy
            {'aapl_weight': 0.4, 'vgt_weight': 0.4, 'sentiment_weight': 0.2},  # VGT heavy
            {'aapl_weight': 0.3, 'vgt_weight': 0.3, 'sentiment_weight': 0.4},  # Sentiment heavy
            {'aapl_weight': 0.7, 'vgt_weight': 0.2, 'sentiment_weight': 0.1},  # Very AAPL heavy
        ]
        
        weight_combinations.extend(specific_combinations)
        
        print(f"Generated {len(weight_combinations)} weight combinations for grid search")
        return weight_combinations
    
    def perform_weight_grid_search(self, horizon='1d'):
        """Perform grid search for optimal weight combinations"""
        print(f"\nPerforming weight grid search for {horizon} horizon...")
        
        # Prepare features
        X, y, feature_cols = self.prepare_features(horizon)
        
        # Define weight combinations
        weight_combinations = self.define_weight_combinations()
        
        # Use TimeSeriesSplit for time series data
        tscv = TimeSeriesSplit(n_splits=5)
        
        results = []
        
        for i, weights in enumerate(weight_combinations):
            if i % 20 == 0:
                print(f"Testing weight combination {i+1}/{len(weight_combinations)}: "
                      f"AAPL={weights['aapl_weight']:.2f}, "
                      f"VGT={weights['vgt_weight']:.2f}, "
                      f"Sentiment={weights['sentiment_weight']:.2f}")
            
            # Calculate ensemble score using weights
            ensemble_scores = self._calculate_weighted_ensemble_score(X, weights, horizon)
            
            # Convert scores to predictions (threshold at 0)
            predictions = (ensemble_scores > 0).astype(int)
            
            # Calculate cross-validation accuracy
            cv_scores = []
            for train_idx, val_idx in tscv.split(X):
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                pred_train, pred_val = predictions[train_idx], predictions[val_idx]
                
                # Calculate accuracy for this fold
                accuracy = accuracy_score(y_val, pred_val)
                cv_scores.append(accuracy)
            
            avg_cv_score = np.mean(cv_scores)
            
            results.append({
                'weights': weights,
                'cv_score': avg_cv_score,
                'cv_scores': cv_scores,
                'ensemble_scores': ensemble_scores,
                'predictions': predictions
            })
        
        # Find best weight combination
        best_result = max(results, key=lambda x: x['cv_score'])
        
        print(f"\nBest weight combination for {horizon}:")
        print(f"  AAPL Weight: {best_result['weights']['aapl_weight']:.3f}")
        print(f"  VGT Weight: {best_result['weights']['vgt_weight']:.3f}")
        print(f"  Sentiment Weight: {best_result['weights']['sentiment_weight']:.3f}")
        print(f"  CV Score: {best_result['cv_score']:.4f}")
        
        # Store results
        self.grid_search_results[horizon] = {
            'all_results': results,
            'best_result': best_result,
            'best_weights': best_result['weights'],
            'best_cv_score': best_result['cv_score']
        }
        
        return results
    
    def _calculate_weighted_ensemble_score(self, X, weights, horizon):
        """Calculate weighted ensemble score"""
        # Get individual model scores
        aapl_col = f'aapl_pred_{horizon}'
        vgt_col = f'vgt_pred_{horizon}'
        sentiment_col = f'sentiment_pred_{horizon}'
        
        ensemble_scores = np.zeros(len(X))
        
        if aapl_col in X.columns:
            ensemble_scores += weights['aapl_weight'] * X[aapl_col].values
        
        if vgt_col in X.columns:
            ensemble_scores += weights['vgt_weight'] * X[vgt_col].values
            
        if sentiment_col in X.columns:
            ensemble_scores += weights['sentiment_weight'] * X[sentiment_col].values
        
        return ensemble_scores
    
    def evaluate_best_weights(self, horizon='1d'):
        """Evaluate best weight combination on test data"""
        if horizon not in self.grid_search_results:
            raise ValueError(f"No weight search results for horizon {horizon}")
        
        print(f"\nEvaluating best weights for {horizon}...")
        
        # Prepare features
        X, y, feature_cols = self.prepare_features(horizon)
        
        # Split data (use last 20% for testing)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Get best weights
        best_weights = self.grid_search_results[horizon]['best_weights']
        
        # Calculate ensemble scores for test set
        ensemble_scores = self._calculate_weighted_ensemble_score(X_test, best_weights, horizon)
        
        # Convert to predictions
        y_pred = (ensemble_scores > 0).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        
        # Print results
        print(f"\nTest Results for {horizon}:")
        print(f"Best Weights: AAPL={best_weights['aapl_weight']:.3f}, "
              f"VGT={best_weights['vgt_weight']:.3f}, "
              f"Sentiment={best_weights['sentiment_weight']:.3f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Classification report
        print(f"\nClassification Report for {horizon}:")
        print(classification_report(y_test, y_pred))
        
        # Store evaluation results
        self.best_models[horizon] = {
            'weights': best_weights,
            'cv_score': self.grid_search_results[horizon]['best_cv_score'],
            'test_metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'y_test': y_test,
                'y_pred': y_pred,
                'ensemble_scores': ensemble_scores
            }
        }
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'weights': best_weights
        }
    
    def run_full_analysis(self):
        """Run complete ensemble analysis for all horizons"""
        print("Starting ensemble weight optimization analysis...")
        print("="*60)
        
        # Load all scores
        if not self.load_all_scores():
            return None
        
        # Combine scores
        self.combine_scores()
        
        # Run weight grid search for each horizon
        horizons = ['1d', '3d', '5d']
        all_results = {}
        
        for horizon in horizons:
            print(f"\n{'='*20} {horizon.upper()} HORIZON {'='*20}")
            
            # Perform weight grid search
            grid_results = self.perform_weight_grid_search(horizon)
            
            # Evaluate best weights
            test_metrics = self.evaluate_best_weights(horizon)
            
            all_results[horizon] = {
                'grid_results': grid_results,
                'test_metrics': test_metrics
            }
        
        # Print summary
        self.print_summary()
        
        # Save results
        self.save_results()
        
        return all_results
    
    def print_summary(self):
        """Print summary of all results"""
        print("\n" + "="*60)
        print("ENSEMBLE WEIGHT OPTIMIZATION SUMMARY")
        print("="*60)
        
        # Find best performing horizon
        best_horizon = None
        best_accuracy = 0
        
        for horizon in ['1d', '3d', '5d']:
            if horizon in self.best_models:
                model_info = self.best_models[horizon]
                accuracy = model_info['test_metrics']['accuracy']
                
                print(f"\n{horizon.upper()} HORIZON:")
                print(f"  Best Weights:")
                print(f"    AAPL: {model_info['weights']['aapl_weight']:.3f}")
                print(f"    VGT: {model_info['weights']['vgt_weight']:.3f}")
                print(f"    Sentiment: {model_info['weights']['sentiment_weight']:.3f}")
                print(f"  CV Score: {model_info['cv_score']:.4f}")
                print(f"  Test Accuracy: {accuracy:.4f}")
                print(f"  Test F1-Score: {model_info['test_metrics']['f1_score']:.4f}")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_horizon = horizon
        
        print(f"\n🏆 BEST PERFORMING HORIZON: {best_horizon.upper()}")
        print(f"   Accuracy: {best_accuracy:.4f}")
        
        # Print weight comparison
        print(f"\nWEIGHT COMPARISON ACROSS HORIZONS:")
        print("-" * 50)
        print(f"{'Horizon':<8} {'AAPL':<8} {'VGT':<8} {'Sentiment':<10} {'Accuracy':<10}")
        print("-" * 50)
        
        for horizon in ['1d', '3d', '5d']:
            if horizon in self.best_models:
                weights = self.best_models[horizon]['weights']
                accuracy = self.best_models[horizon]['test_metrics']['accuracy']
                print(f"{horizon:<8} {weights['aapl_weight']:<8.3f} {weights['vgt_weight']:<8.3f} "
                      f"{weights['sentiment_weight']:<10.3f} {accuracy:<10.4f}")
        
        print("-" * 50)
    
    def save_results(self):
        """Save all results to files"""
        print("\nSaving results...")
        
        # Save combined data
        combined_file = os.path.join(get_project_root(), 'combined_scores.csv')
        self.combined_data.to_csv(combined_file, index=False)
        print(f"Combined scores saved to: {combined_file}")
        
        # Save model results
        results_file = os.path.join(get_project_root(), 'ensemble_results.pkl')
        joblib.dump({
            'best_models': self.best_models,
            'grid_search_results': self.grid_search_results,
            'combined_data': self.combined_data
        }, results_file)
        print(f"Model results saved to: {results_file}")
        
        # Save summary report
        summary_file = os.path.join(get_project_root(), 'ensemble_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("ENSEMBLE WEIGHT OPTIMIZATION SUMMARY\n")
            f.write("="*60 + "\n\n")
            
            # Find best performing horizon
            best_horizon = None
            best_accuracy = 0
            
            for horizon in ['1d', '3d', '5d']:
                if horizon in self.best_models:
                    model_info = self.best_models[horizon]
                    accuracy = model_info['test_metrics']['accuracy']
                    
                    f.write(f"{horizon.upper()} HORIZON:\n")
                    f.write(f"  Best Weights:\n")
                    f.write(f"    AAPL: {model_info['weights']['aapl_weight']:.3f}\n")
                    f.write(f"    VGT: {model_info['weights']['vgt_weight']:.3f}\n")
                    f.write(f"    Sentiment: {model_info['weights']['sentiment_weight']:.3f}\n")
                    f.write(f"  CV Score: {model_info['cv_score']:.4f}\n")
                    f.write(f"  Test Accuracy: {accuracy:.4f}\n")
                    f.write(f"  Test F1-Score: {model_info['test_metrics']['f1_score']:.4f}\n\n")
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_horizon = horizon
            
            f.write(f"BEST PERFORMING HORIZON: {best_horizon.upper()}\n")
            f.write(f"Accuracy: {best_accuracy:.4f}\n\n")
            
            # Weight comparison table
            f.write("WEIGHT COMPARISON ACROSS HORIZONS:\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Horizon':<8} {'AAPL':<8} {'VGT':<8} {'Sentiment':<10} {'Accuracy':<10}\n")
            f.write("-" * 50 + "\n")
            
            for horizon in ['1d', '3d', '5d']:
                if horizon in self.best_models:
                    weights = self.best_models[horizon]['weights']
                    accuracy = self.best_models[horizon]['test_metrics']['accuracy']
                    f.write(f"{horizon:<8} {weights['aapl_weight']:<8.3f} {weights['vgt_weight']:<8.3f} "
                           f"{weights['sentiment_weight']:<10.3f} {accuracy:<10.4f}\n")
            
            f.write("-" * 50 + "\n")
        
        print(f"Summary report saved to: {summary_file}")

def main():
    """Main execution function"""
    print("Score Ensemble Analysis - Weight Optimization Grid Search")
    print("="*60)
    
    # Initialize ensemble
    ensemble = ScoreEnsemble()
    
    # Run full analysis
    results = ensemble.run_full_analysis()
    
    if results is not None:
        print("\nEnsemble weight optimization completed successfully!")
        return ensemble
    else:
        print("Ensemble weight optimization failed!")
        return None

if __name__ == "__main__":
    ensemble = main()
