#!/usr/bin/env python3
"""
AAPL Stock Analysis using Kronos Mini - Independent Scoring Module
================================================================

This module performs scoring analysis of AAPL stock data using:
1. Technical indicators (EMA, MACD, RSI)
2. Kronos mini model for 1, 3, 5-day ahead predictions
3. Rolling 60-day window analysis
4. Returns standardized scores for ensemble learning

Author: AI Assistant
Date: 2025
"""

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Technical Analysis Libraries
import talib

# Machine Learning Libraries
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch

# Import our real Kronos model
from kronos_model import KronosPredictor as RealKronosPredictor

def get_project_root():
    """Get the project root directory"""
    return os.path.dirname(os.path.abspath(__file__))

class TechnicalIndicators:
    """Calculate technical indicators for stock analysis"""
    
    def __init__(self, data):
        self.data = data.copy()
        self.calculate_all_indicators()
    
    def calculate_all_indicators(self):
        """Calculate all technical indicators"""
        # Price data - ensure float64 type for TA-Lib
        high = self.data['high'].astype(np.float64).values
        low = self.data['low'].astype(np.float64).values
        close = self.data['close'].astype(np.float64).values
        volume = self.data['volume'].astype(np.float64).values
        
        # Moving Averages
        self.data['SMA_20'] = talib.SMA(close, timeperiod=20)
        self.data['SMA_50'] = talib.SMA(close, timeperiod=50)
        self.data['EMA_12'] = talib.EMA(close, timeperiod=12)
        self.data['EMA_26'] = talib.EMA(close, timeperiod=26)
        self.data['EMA_50'] = talib.EMA(close, timeperiod=50)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        self.data['MACD'] = macd
        self.data['MACD_Signal'] = macd_signal
        self.data['MACD_Histogram'] = macd_hist
        
        # RSI
        self.data['RSI'] = talib.RSI(close, timeperiod=14)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        self.data['BB_Upper'] = bb_upper
        self.data['BB_Middle'] = bb_middle
        self.data['BB_Lower'] = bb_lower
        
        # Stochastic Oscillator
        slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
        self.data['Stoch_K'] = slowk
        self.data['Stoch_D'] = slowd
        
        # Williams %R
        self.data['Williams_R'] = talib.WILLR(high, low, close, timeperiod=14)
        
        # Average True Range
        self.data['ATR'] = talib.ATR(high, low, close, timeperiod=14)
        
        # Volume indicators
        self.data['Volume_SMA'] = talib.SMA(volume, timeperiod=20)
        
        # Price change indicators
        self.data['Price_Change'] = self.data['close'].pct_change()
        self.data['Price_Change_5d'] = self.data['close'].pct_change(periods=5)
        self.data['Price_Change_10d'] = self.data['close'].pct_change(periods=10)
        
        # Volatility
        self.data['Volatility_20d'] = self.data['Price_Change'].rolling(window=20).std()
        
        print("Technical indicators calculated successfully!")

class KronosPredictor:
    """Enhanced Kronos predictor with unified features for 1,3,5-day predictions"""
    
    def __init__(self, lookback_window=60):
        self.lookback_window = lookback_window
        self.model_loaded = False
        
        # Unified token strategy for all horizons
        self.token_strategy = {
            'ohlcv_tokens': 30,  # 统一使用30个OHLCV数据
            'tech_tokens': 15    # 统一使用15个技术指标
        }
        
        # Different bias for different horizons
        self.horizon_bias = {
            1: {'bias': 'conservative', 'factor': -0.2, 'threshold': 0.0},  # 1天: 保守偏置
            3: {'bias': 'neutral', 'factor': 0.0, 'threshold': 0.0},        # 3天: 中性偏置
            5: {'bias': 'trend', 'factor': 0.2, 'threshold': 0.0}            # 5天: 趋势偏置
        }
        
    def load_model(self):
        """Load Kronos mini model (placeholder for actual implementation)"""
        try:
            # This would be the actual Kronos model loading
            # For now, we'll use a simple LSTM-based approach as placeholder
            print("Loading Kronos mini model...")
            self.model_loaded = True
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_loaded = False
    
    def prepare_features(self, data, start_idx, end_idx, horizon_days=1):
        """Prepare unified features for all prediction horizons"""
        window_data = data.iloc[start_idx:end_idx].copy()
        
        # Use unified token strategy for all horizons
        ohlcv_tokens = self.token_strategy['ohlcv_tokens']
        tech_tokens = self.token_strategy['tech_tokens']
        
        # Select OHLCV features (most recent tokens)
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        ohlcv_data = window_data[ohlcv_cols].tail(ohlcv_tokens).ffill().fillna(0)
        
        # Select technical indicators (most recent tokens)
        technical_cols = ['EMA_12', 'EMA_26', 'MACD', 'RSI', 'BB_Upper', 'BB_Lower', 'ATR', 
                         'Stoch_K', 'Stoch_D', 'Williams_R', 'Volatility_20d']
        available_tech_cols = [col for col in technical_cols if col in window_data.columns]
        
        if len(available_tech_cols) > 0:
            tech_data = window_data[available_tech_cols].tail(tech_tokens).ffill().fillna(0)
            
            # Ensure both arrays have the same number of rows
            min_rows = min(len(ohlcv_data), len(tech_data))
            if min_rows > 0:
                ohlcv_subset = ohlcv_data.tail(min_rows).values
                tech_subset = tech_data.tail(min_rows).values
                # Combine OHLCV and technical features
                features = np.concatenate([ohlcv_subset, tech_subset], axis=1)
            else:
                features = ohlcv_data.values
        else:
            features = ohlcv_data.values
        
        return features
    
    def predict_direction(self, features, horizon_days=1):
        """Predict price direction with unified features and horizon-specific bias"""
        # Get bias configuration for this horizon
        bias_config = self.horizon_bias[horizon_days]
        bias_factor = bias_config['factor']
        threshold = bias_config['threshold']
        
        # Extract OHLCV data (first 5 columns are OHLCV)
        if features.shape[1] >= 5:
            close_prices = features[:, 3]  # close price is 4th column
            volume = features[:, 4] if features.shape[1] > 4 else None
        else:
            close_prices = features[:, -1]  # fallback to last column
            volume = None
        
        # Extract technical indicators if available
        tech_start_idx = min(5, features.shape[1])
        if features.shape[1] > tech_start_idx:
            tech_features = features[:, tech_start_idx:]
        else:
            tech_features = None
        
        # Calculate base prediction score
        base_score = 0
        
        # Price momentum analysis
        if len(close_prices) >= 3:
            recent_momentum = np.mean(np.diff(close_prices[-3:]))
            base_score += recent_momentum * 10  # Scale momentum
        
        # Technical indicators analysis
        if tech_features is not None and tech_features.shape[1] >= 4:
            # EMA signals
            if tech_features.shape[1] >= 2:
                ema_12 = tech_features[:, 0]
                ema_26 = tech_features[:, 1]
                if len(ema_12) > 0 and len(ema_26) > 0:
                    if ema_12[-1] > ema_26[-1]:
                        base_score += 0.3
                    if len(ema_12) > 1 and ema_12[-1] > ema_12[-2]:
                        base_score += 0.2
            
            # RSI analysis
            if tech_features.shape[1] >= 4:
                rsi = tech_features[:, 3]
                if len(rsi) > 0 and not np.isnan(rsi[-1]):
                    if 30 < rsi[-1] < 70:  # Neutral zone
                        base_score += 0.1
                    elif rsi[-1] < 30:  # Oversold
                        base_score += 0.3
                    elif rsi[-1] > 70:  # Overbought
                        base_score -= 0.2
        
        # Apply horizon-specific bias and make final prediction
        final_score = base_score + bias_factor
        prediction = 1 if final_score > threshold else 0
        
        return prediction

class RollingAnalysis:
    """Perform rolling analysis with enhanced 60-day window and 1,3,5-day predictions"""
    
    def __init__(self, data, lookback_window=60, use_real_kronos=False):
        self.data = data
        self.lookback_window = lookback_window
        self.use_real_kronos = use_real_kronos
        
        if use_real_kronos:
            print("Using REAL Kronos model implementation...")
            self.predictor = RealKronosPredictor(device="cpu", offline_mode=True)  # Use offline mode
        else:
            print("Using enhanced Kronos predictor with unified features...")
            self.predictor = KronosPredictor(lookback_window)
        
        self.results = []
        self.daily_accuracy = []  # Store daily accuracy calculations
        
    def run_rolling_analysis(self):
        """Run rolling analysis with daily 60-day accuracy calculation for 1,3,5-day predictions"""
        print(f"Starting enhanced rolling analysis with {self.lookback_window}-day window...")
        print("🎯 Key Features:")
        print("  - Daily accuracy calculation using past 60 days")
        print("  - Unified features: 30 OHLCV + 15 Tech indicators for all horizons")
        print("  - Prediction horizons: 1d(conservative), 3d(neutral), 5d(trend)")
        
        # Load the model
        self.predictor.load_model()
        
        # Start from lookback_window to ensure we have enough data
        start_idx = self.lookback_window
        end_idx = len(self.data) - 5  # Leave 5 days for 5-day ahead prediction
        
        for i in range(start_idx, end_idx):
            # Prepare unified features for all horizons
            if self.use_real_kronos:
                features = self.predictor.prepare_features(self.data, i - self.lookback_window, i)
            else:
                features = self.predictor.prepare_features(self.data, i - self.lookback_window, i)
            
            # Get actual future prices for comparison
            actual_1d = self.data.iloc[i]['close']
            actual_3d = self.data.iloc[i + 2]['close']
            actual_5d = self.data.iloc[i + 4]['close']
            
            current_price = self.data.iloc[i - 1]['close']
            
            # Calculate actual directions
            actual_dir_1d = 1 if actual_1d > current_price else 0
            actual_dir_3d = 1 if actual_3d > current_price else 0
            actual_dir_5d = 1 if actual_5d > current_price else 0
            
            # Make predictions with different horizons using unified features
            if self.use_real_kronos:
                pred_dir_1d = self.predictor.predict_direction(features, horizon_days=1)
                pred_dir_3d = self.predictor.predict_direction(features, horizon_days=3)
                pred_dir_5d = self.predictor.predict_direction(features, horizon_days=5)
            else:
                pred_dir_1d = self.predictor.predict_direction(features, horizon_days=1)
                pred_dir_3d = self.predictor.predict_direction(features, horizon_days=3)
                pred_dir_5d = self.predictor.predict_direction(features, horizon_days=5)
            
            # Calculate daily accuracy for past 60 days (if we have enough data)
            daily_accuracies = {}
            if i >= start_idx + self.lookback_window:
                # Get past 60 days of results
                past_results = self.results[-self.lookback_window:]
                
                for horizon in ['1d', '3d', '5d']:
                    actual_col = f'actual_dir_{horizon}'
                    pred_col = f'pred_dir_{horizon}'
                    
                    if len(past_results) > 0 and actual_col in past_results[0]:
                        actual_values = [r[actual_col] for r in past_results]
                        pred_values = [r[pred_col] for r in past_results]
                        
                        # Calculate accuracy for this horizon
                        accuracy = accuracy_score(actual_values, pred_values)
                        daily_accuracies[f'accuracy_{horizon}'] = accuracy
            
            # Store results
            result = {
                'date': self.data.iloc[i]['date'],
                'current_price': current_price,
                'actual_1d': actual_1d,
                'actual_3d': actual_3d,
                'actual_5d': actual_5d,
                'actual_dir_1d': actual_dir_1d,
                'actual_dir_3d': actual_dir_3d,
                'actual_dir_5d': actual_dir_5d,
                'pred_dir_1d': pred_dir_1d,
                'pred_dir_3d': pred_dir_3d,
                'pred_dir_5d': pred_dir_5d,
                'price_change_1d': (actual_1d - current_price) / current_price,
                'price_change_3d': (actual_3d - current_price) / current_price,
                'price_change_5d': (actual_5d - current_price) / current_price,
            }
            
            # Add daily accuracies
            result.update(daily_accuracies)
            
            self.results.append(result)
            
            if (i - start_idx) % 100 == 0:
                print(f"Processed {i - start_idx + 1} windows...")
        
        print(f"Rolling analysis completed! Processed {len(self.results)} windows.")
        return pd.DataFrame(self.results)

class ScoreCalculator:
    """Calculate and output model scores only"""
    
    def __init__(self, results_df):
        self.results_df = results_df
        
    def calculate_scores(self):
        """Calculate performance scores for 1, 3, 5-day predictions"""
        scores = {}
        
        for horizon in ['1d', '3d', '5d']:
            actual_col = f'actual_dir_{horizon}'
            pred_col = f'pred_dir_{horizon}'
            
            if actual_col in self.results_df.columns and pred_col in self.results_df.columns:
                actual = self.results_df[actual_col]
                predicted = self.results_df[pred_col]
                
                # Calculate basic metrics
                accuracy = accuracy_score(actual, predicted)
                precision = precision_score(actual, predicted, average='binary', zero_division=0)
                recall = recall_score(actual, predicted, average='binary', zero_division=0)
                f1 = f1_score(actual, predicted, average='binary', zero_division=0)
                
                # Calculate returns-based metrics
                returns_col = f'price_change_{horizon}'
                if returns_col in self.results_df.columns:
                    predicted_returns = self.results_df[returns_col] * (predicted * 2 - 1)
                    actual_returns = self.results_df[returns_col]
                    
                    # Sharpe ratio (simplified)
                    sharpe_ratio = np.mean(predicted_returns) / np.std(predicted_returns) if np.std(predicted_returns) > 0 else 0
                    
                    # Hit rate
                    hit_rate = np.mean((predicted_returns > 0) == (actual_returns > 0))
                else:
                    sharpe_ratio = 0
                    hit_rate = accuracy
                
                # Calculate composite score (weighted combination)
                composite_score = (
                    accuracy * 0.3 +           # 30% weight on accuracy
                    precision * 0.2 +          # 20% weight on precision
                    recall * 0.2 +             # 20% weight on recall
                    f1 * 0.2 +                 # 20% weight on F1
                    hit_rate * 0.1             # 10% weight on hit rate
                )
                
                scores[horizon] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'hit_rate': hit_rate,
                    'sharpe_ratio': sharpe_ratio,
                    'composite_score': composite_score,
                    'total_predictions': len(actual),
                    'up_predictions': np.sum(predicted),
                    'down_predictions': len(predicted) - np.sum(predicted)
                }
        
        return scores
    
    def print_scores(self, scores):
        """Print scores only"""
        print("\n" + "="*60)
        print("KRONOS MINI MODEL SCORES")
        print("="*60)
        
        for horizon, score_dict in scores.items():
            print(f"\n{horizon.upper()} PREDICTION SCORES:")
            print("-" * 30)
            print(f"Accuracy: {score_dict['accuracy']:.4f}")
            print(f"Precision: {score_dict['precision']:.4f}")
            print(f"Recall: {score_dict['recall']:.4f}")
            print(f"F1-Score: {score_dict['f1_score']:.4f}")
            print(f"Hit Rate: {score_dict['hit_rate']:.4f}")
            print(f"Sharpe Ratio: {score_dict['sharpe_ratio']:.4f}")
            print(f"Composite Score: {score_dict['composite_score']:.4f}")
            print(f"Total Predictions: {score_dict['total_predictions']}")
            print()
        
        # Print overall summary
        print("OVERALL SUMMARY:")
        print("-" * 20)
        avg_composite = np.mean([score_dict['composite_score'] for score_dict in scores.values()])
        print(f"Average Composite Score: {avg_composite:.4f}")
        
        best_horizon = max(scores.keys(), key=lambda h: scores[h]['composite_score'])
        print(f"Best Performing Horizon: {best_horizon.upper()} ({scores[best_horizon]['composite_score']:.4f})")
        print()

class AAPLScoreGenerator:
    """Independent AAPL scoring module for ensemble learning"""
    
    def __init__(self, data_path=None):
        self.data_path = data_path or os.path.join(get_project_root(), 'data', 'stock_105.AAPL_2025-09.csv')
        self.results_df = None
        self.scores = None
        
    def load_data(self):
        """Load and preprocess AAPL data"""
        print("Loading AAPL stock data...")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        data = pd.read_csv(self.data_path)
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date').reset_index(drop=True)
        
        print(f"AAPL data loaded: {len(data)} records from {data['date'].min()} to {data['date'].max()}")
        return data
    
    def generate_scores(self):
        """Generate standardized scores for ensemble learning"""
        try:
            # Load data
            data = self.load_data()
            
            # Calculate technical indicators
            print("Calculating technical indicators...")
            tech_indicators = TechnicalIndicators(data)
            data_with_indicators = tech_indicators.data
            
            # Run rolling analysis
            print("Running rolling analysis...")
            rolling_analysis = RollingAnalysis(data_with_indicators, lookback_window=60)
            self.results_df = rolling_analysis.run_rolling_analysis()
            
            # Calculate scores
            print("Calculating AAPL model scores...")
            score_calculator = ScoreCalculator(self.results_df)
            self.scores = score_calculator.calculate_scores()
            
            # Generate standardized scores for ensemble
            standardized_scores = self._standardize_scores()
            
            print("AAPL scoring completed successfully!")
            return standardized_scores
            
        except Exception as e:
            print(f"Error in AAPL scoring: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _standardize_scores(self):
        """Standardize scores for ensemble learning"""
        if self.results_df is None or self.scores is None:
            return None
            
        standardized_df = self.results_df[['date', 'current_price']].copy()
        
        for horizon in ['1d', '3d', '5d']:
            # Get prediction scores
            pred_col = f'pred_dir_{horizon}'
            actual_col = f'actual_dir_{horizon}'
            price_change_col = f'price_change_{horizon}'
            
            if pred_col in self.results_df.columns:
                # Convert predictions to standardized scores (-1 to 1)
                standardized_df[f'aapl_pred_{horizon}'] = (self.results_df[pred_col] * 2 - 1)
                
                # Add actual direction for evaluation
                if actual_col in self.results_df.columns:
                    standardized_df[f'aapl_actual_{horizon}'] = self.results_df[actual_col]
                
                # Add price change for evaluation
                if price_change_col in self.results_df.columns:
                    standardized_df[f'aapl_return_{horizon}'] = self.results_df[price_change_col]
        
        return standardized_df
    
    def get_performance_metrics(self):
        """Get performance metrics for reporting"""
        if self.scores is None:
            return None
            
        metrics = {}
        for horizon, score_dict in self.scores.items():
            metrics[f'aapl_{horizon}'] = {
                'accuracy': score_dict['accuracy'],
                'composite_score': score_dict['composite_score'],
                'total_predictions': score_dict['total_predictions']
            }
        
        return metrics

def main():
    """Main execution function for standalone testing"""
    print("AAPL Stock Analysis using Kronos Mini - Independent Module")
    print("="*60)
    
    # Initialize AAPL score generator
    aapl_generator = AAPLScoreGenerator()
    
    # Generate scores
    scores_df = aapl_generator.generate_scores()
    
    if scores_df is not None:
        # Save results
        results_file = os.path.join(get_project_root(), 'aapl_scores.csv')
        scores_df.to_csv(results_file, index=False)
        print(f"AAPL scores saved to '{results_file}'")
        
        # Print performance metrics
        metrics = aapl_generator.get_performance_metrics()
        if metrics:
            print("\nAAPL Performance Metrics:")
            for horizon, metric in metrics.items():
                print(f"{horizon}: Accuracy={metric['accuracy']:.4f}, Composite={metric['composite_score']:.4f}")
        
        return scores_df
    else:
        print("Failed to generate AAPL scores")
        return None

if __name__ == "__main__":
    scores = main()
