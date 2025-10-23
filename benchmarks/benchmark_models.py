#!/usr/bin/env python3
"""
Benchmark Models Module - Comprehensive Collection of Baseline Models
===================================================================

This module implements various benchmark models for stock prediction:
1. Technical Analysis Models (MA Cross, RSI, MACD, Bollinger Bands, Momentum)
2. Machine Learning Models (RF, GB, SVM, LR, XGBoost)
3. Deep Learning Models (LSTM, GRU, Transformer, CNN-LSTM)

All models follow a unified interface for easy comparison and ensemble learning.

Date: 2025
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Deep Learning Libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Deep learning models will be disabled.")

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. XGBoost models will be disabled.")
    # Create a dummy xgb module to avoid import errors
    class DummyXGB:
        class XGBClassifier:
            def __init__(self, **kwargs):
                raise ImportError("XGBoost is not installed")
    xgb = DummyXGB()

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import os
from config import BENCHMARKS_OUTPUT_DIR, SENTIMENT_SCORES

def merge_sentiment_into_features(features: pd.DataFrame, horizon: str) -> pd.DataFrame:
    """Merge cached sentiment features into the given feature DataFrame if available.
    Looks first in benchmarks output dir, then falls back to ensemble sentiment_scores.
    """
    if 'date' not in features.columns:
        return features
    
    # Determine sentiment scores path
    bench_sent_path = str(BENCHMARKS_OUTPUT_DIR / "sentiment_scores.csv")
    ensemble_sent_path = str(SENTIMENT_SCORES)
    sentiment_path = bench_sent_path if os.path.exists(bench_sent_path) else (
        ensemble_sent_path if os.path.exists(ensemble_sent_path) else None
    )
    
    if sentiment_path is None:
        return features
    
    try:
        sentiment_df = pd.read_csv(sentiment_path)
    except Exception:
        return features
    
    if 'date' not in sentiment_df.columns:
        return features
    
    # Ensure dates are comparable
    try:
        features = features.copy()
        features['date'] = pd.to_datetime(features['date'])
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    except Exception:
        pass
    
    # Select useful sentiment columns
    base_cols = ['sentiment_score', 'sentiment_strength', 'volume_weighted_sentiment']
    pred_col = f'sentiment_pred_{horizon}'
    selected_cols = ['date'] + [c for c in base_cols if c in sentiment_df.columns]
    if pred_col in sentiment_df.columns:
        selected_cols.append(pred_col)
    
    # If nothing to merge, return as-is
    if len(selected_cols) == 1:
        return features
    
    merged = features.merge(sentiment_df[selected_cols], on='date', how='left')
    return merged

class BaseModel(ABC):
    """Abstract base class for all benchmark models"""
    
    def __init__(self, name: str, horizon: str = '1d'):
        self.name = name
        self.horizon = horizon
        self.is_trained = False
        self.model = None
        self.scaler = None
        
    @abstractmethod
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for the model"""
        pass
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'name': self.name,
            'horizon': self.horizon,
            'is_trained': self.is_trained,
            'model_type': self.__class__.__name__
        }

class TechnicalAnalysisModel(BaseModel):
    """Base class for technical analysis models"""
    
    def __init__(self, name: str, horizon: str = '1d'):
        super().__init__(name, horizon)
        self.threshold = 0.0
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare technical features"""
        features = data.copy()
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in features.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        return features
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Technical models don't need training, just validation"""
        self.is_trained = True
        return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities for technical models"""
        predictions = self.predict(X)
        # Convert to probabilities (simple approach)
        proba = np.zeros((len(predictions), 2))
        proba[np.arange(len(predictions)), predictions] = 1.0
        return proba

class MovingAverageCrossModel(TechnicalAnalysisModel):
    """Moving Average Cross Strategy"""
    
    def __init__(self, short_window: int = 5, long_window: int = 20, horizon: str = '1d'):
        super().__init__(f"MA_Cross_{short_window}_{long_window}", horizon)
        self.short_window = short_window
        self.long_window = long_window
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate moving averages"""
        features = super().prepare_features(data)
        
        # Calculate moving averages
        features['MA_short'] = features['close'].rolling(window=self.short_window).mean()
        features['MA_long'] = features['close'].rolling(window=self.long_window).mean()
        
        # Calculate MA cross signal
        features['MA_signal'] = features['MA_short'] - features['MA_long']
        
        return features
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict based on MA cross"""
        if 'MA_signal' not in X.columns:
            X = self.prepare_features(X)
        
        # MA cross signal: 1 if short MA > long MA, 0 otherwise
        predictions = (X['MA_signal'] > 0).astype(int)
        return predictions.values

class RSIModel(TechnicalAnalysisModel):
    """RSI Strategy"""
    
    def __init__(self, window: int = 14, oversold: float = 30, overbought: float = 70, horizon: str = '1d'):
        super().__init__(f"RSI_{window}_{oversold}_{overbought}", horizon)
        self.window = window
        self.oversold = oversold
        self.overbought = overbought
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI"""
        features = super().prepare_features(data)
        
        # Calculate price changes
        delta = features['close'].diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=self.window).mean()
        avg_losses = losses.rolling(window=self.window).mean()
        
        # Calculate RSI
        rs = avg_gains / avg_losses
        features['RSI'] = 100 - (100 / (1 + rs))
        
        return features
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict based on RSI"""
        if 'RSI' not in X.columns:
            X = self.prepare_features(X)
        
        # RSI strategy: 1 if oversold (buy signal), 0 if overbought (sell signal)
        predictions = (X['RSI'] < self.oversold).astype(int)
        return predictions.values

class MACDModel(TechnicalAnalysisModel):
    """MACD Strategy"""
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, horizon: str = '1d'):
        super().__init__(f"MACD_{fast}_{slow}_{signal}", horizon)
        self.fast = fast
        self.slow = slow
        self.signal = signal
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD"""
        features = super().prepare_features(data)
        
        # Calculate EMAs
        ema_fast = features['close'].ewm(span=self.fast).mean()
        ema_slow = features['close'].ewm(span=self.slow).mean()
        
        # Calculate MACD line
        features['MACD'] = ema_fast - ema_slow
        
        # Calculate signal line
        features['MACD_signal'] = features['MACD'].ewm(span=self.signal).mean()
        
        # Calculate histogram
        features['MACD_histogram'] = features['MACD'] - features['MACD_signal']
        
        return features
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict based on MACD"""
        if 'MACD_histogram' not in X.columns:
            X = self.prepare_features(X)
        
        # MACD strategy: 1 if histogram > 0 (bullish), 0 otherwise
        predictions = (X['MACD_histogram'] > 0).astype(int)
        return predictions.values

class BollingerBandsModel(TechnicalAnalysisModel):
    """Bollinger Bands Strategy"""
    
    def __init__(self, window: int = 20, std_dev: float = 2, horizon: str = '1d'):
        super().__init__(f"BB_{window}_{std_dev}", horizon)
        self.window = window
        self.std_dev = std_dev
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        features = super().prepare_features(data)
        
        # Calculate moving average
        features['BB_middle'] = features['close'].rolling(window=self.window).mean()
        
        # Calculate standard deviation
        bb_std = features['close'].rolling(window=self.window).std()
        
        # Calculate upper and lower bands
        features['BB_upper'] = features['BB_middle'] + (bb_std * self.std_dev)
        features['BB_lower'] = features['BB_middle'] - (bb_std * self.std_dev)
        
        # Calculate position within bands
        features['BB_position'] = (features['close'] - features['BB_lower']) / (features['BB_upper'] - features['BB_lower'])
        
        return features
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict based on Bollinger Bands"""
        if 'BB_position' not in X.columns:
            X = self.prepare_features(X)
        
        # BB strategy: 1 if price near lower band (oversold), 0 if near upper band (overbought)
        predictions = (X['BB_position'] < 0.2).astype(int)
        return predictions.values

class MomentumModel(TechnicalAnalysisModel):
    """Momentum Strategy"""
    
    def __init__(self, window: int = 10, horizon: str = '1d'):
        super().__init__(f"Momentum_{window}", horizon)
        self.window = window
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum"""
        features = super().prepare_features(data)
        
        # Calculate momentum
        features['momentum'] = features['close'] - features['close'].shift(self.window)
        
        # Calculate momentum percentage
        features['momentum_pct'] = features['momentum'] / features['close'].shift(self.window)
        
        return features
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict based on momentum"""
        if 'momentum' not in X.columns:
            X = self.prepare_features(X)
        
        # Momentum strategy: 1 if positive momentum, 0 otherwise
        predictions = (X['momentum'] > 0).astype(int)
        return predictions.values

class MachineLearningModel(BaseModel):
    """Base class for machine learning models"""
    
    def __init__(self, name: str, horizon: str = '1d'):
        super().__init__(name, horizon)
        self.scaler = StandardScaler()
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML models"""
        features = data.copy()
        
        # Technical indicators
        features['returns'] = features['close'].pct_change()
        features['volatility'] = features['returns'].rolling(20).std()
        features['volume_ma'] = features['volume'].rolling(20).mean()
        features['volume_ratio'] = features['volume'] / features['volume_ma']
        
        # Price features
        features['high_low_ratio'] = features['high'] / features['low']
        features['close_open_ratio'] = features['close'] / features['open']
        
        # Moving averages
        for window in [5, 10, 20]:
            features[f'ma_{window}'] = features['close'].rolling(window).mean()
            features[f'ma_ratio_{window}'] = features['close'] / features[f'ma_{window}']
        
        # RSI
        delta = features['close'].diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        avg_gains = gains.rolling(14).mean()
        avg_losses = losses.rolling(14).mean()
        rs = avg_gains / avg_losses
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = features['close'].ewm(span=12).mean()
        ema_26 = features['close'].ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands
        bb_middle = features['close'].rolling(20).mean()
        bb_std = features['close'].rolling(20).std()
        features['bb_upper'] = bb_middle + (bb_std * 2)
        features['bb_lower'] = bb_middle - (bb_std * 2)
        features['bb_position'] = (features['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        # Merge sentiment features if available
        features = merge_sentiment_into_features(features, self.horizon)
        return features
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Train the ML model"""
        # Prepare features
        X_features = self.prepare_features(X)
        
        # Select numeric columns only
        numeric_cols = X_features.select_dtypes(include=[np.number]).columns
        X_numeric = X_features[numeric_cols].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_numeric)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Calculate training metrics
        y_pred = self.model.predict(X_scaled)
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y, y_pred, average='binary', zero_division=0),
            'f1': f1_score(y, y_pred, average='binary', zero_division=0)
        }
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare features
        X_features = self.prepare_features(X)
        
        # Select numeric columns only
        numeric_cols = X_features.select_dtypes(include=[np.number]).columns
        X_numeric = X_features[numeric_cols].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.transform(X_numeric)
        
        # Make predictions
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare features
        X_features = self.prepare_features(X)
        
        # Select numeric columns only
        numeric_cols = X_features.select_dtypes(include=[np.number]).columns
        X_numeric = X_features[numeric_cols].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.transform(X_numeric)
        
        # Make predictions
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_scaled)
        else:
            # Fallback for models without predict_proba
            predictions = self.model.predict(X_scaled)
            proba = np.zeros((len(predictions), 2))
            proba[np.arange(len(predictions)), predictions] = 1.0
            return proba

class RandomForestModel(MachineLearningModel):
    """Random Forest Model"""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10, horizon: str = '1d'):
        super().__init__(f"RF_{n_estimators}_{max_depth}", horizon)
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )

class GradientBoostingModel(MachineLearningModel):
    """Gradient Boosting Model"""
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, horizon: str = '1d'):
        super().__init__(f"GB_{n_estimators}_{learning_rate}", horizon)
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42
        )

class SVMModel(MachineLearningModel):
    """Support Vector Machine Model"""
    
    def __init__(self, C: float = 1.0, kernel: str = 'rbf', horizon: str = '1d'):
        super().__init__(f"SVM_{C}_{kernel}", horizon)
        self.model = SVC(
            C=C,
            kernel=kernel,
            probability=True,
            random_state=42
        )

class LogisticRegressionModel(MachineLearningModel):
    """Logistic Regression Model"""
    
    def __init__(self, C: float = 1.0, horizon: str = '1d'):
        super().__init__(f"LR_{C}", horizon)
        self.model = LogisticRegression(
            C=C,
            random_state=42,
            max_iter=1000
        )

class XGBoostModel(MachineLearningModel):
    """XGBoost Model"""
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 3, horizon: str = '1d'):
        super().__init__(f"XGB_{n_estimators}_{learning_rate}", horizon)
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not available")
        
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42,
            eval_metric='logloss'
        )

class DeepLearningModel(BaseModel):
    """Base class for deep learning models"""
    
    def __init__(self, name: str, horizon: str = '1d', sequence_length: int = 20, epochs: int = 50, lr: float = 1e-3, batch_size: int = 32, verbose: bool = True, log_every: int = 10):
        super().__init__(name, horizon)
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = MinMaxScaler()
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        # logging controls
        self.verbose = verbose
        self.log_every = max(1, int(log_every))
        self.context: str = ""
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for deep learning models"""
        features = data.copy()
        
        # Basic features
        features['returns'] = features['close'].pct_change()
        features['volatility'] = features['returns'].rolling(20).std()
        features['volume_ma'] = features['volume'].rolling(20).mean()
        features['volume_ratio'] = features['volume'] / features['volume_ma']
        
        # Price ratios
        features['high_low_ratio'] = features['high'] / features['low']
        features['close_open_ratio'] = features['close'] / features['open']
        
        # Moving averages
        for window in [5, 10, 20]:
            features[f'ma_{window}'] = features['close'].rolling(window).mean()
            features[f'ma_ratio_{window}'] = features['close'] / features[f'ma_{window}']
        
        # Merge sentiment features if available
        features = merge_sentiment_into_features(features, self.horizon)
        return features
    
    def create_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series data"""
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(target[i])
        
        return np.array(X), np.array(y)
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Train the deep learning model"""
        # Prepare features
        X_features = self.prepare_features(X)
        
        # Select numeric columns only
        numeric_cols = X_features.select_dtypes(include=[np.number]).columns
        X_numeric = X_features[numeric_cols].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_numeric)
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X_scaled, y.values)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.LongTensor(y_seq).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        input_size = X_seq.shape[2]
        self.model = self._create_model(input_size).to(self.device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            if self.verbose and (epoch % self.log_every == 0 or epoch == self.epochs - 1):
                accuracy = 100 * correct / max(1, total)
                prefix = f"{self.context} " if self.context else ""
                print(f'{prefix}Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}, Accuracy: {accuracy:.2f}%')
        
        self.is_trained = True
        
        # Calculate training metrics
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
            accuracy = accuracy_score(y_seq, predicted.cpu().numpy())
        
        return {'accuracy': accuracy, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare features
        X_features = self.prepare_features(X)
        
        # Select numeric columns only
        numeric_cols = X_features.select_dtypes(include=[np.number]).columns
        X_numeric = X_features[numeric_cols].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.transform(X_numeric)
        
        # Create sequences
        X_seq, _ = self.create_sequences(X_scaled, np.zeros(len(X_scaled)))
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
            return predicted.cpu().numpy()
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare features
        X_features = self.prepare_features(X)
        
        # Select numeric columns only
        numeric_cols = X_features.select_dtypes(include=[np.number]).columns
        X_numeric = X_features[numeric_cols].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.transform(X_numeric)
        
        # Create sequences
        X_seq, _ = self.create_sequences(X_scaled, np.zeros(len(X_scaled)))
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            return probabilities.cpu().numpy()
    
    @abstractmethod
    def _create_model(self, input_size: int) -> nn.Module:
        """Create the neural network model"""
        pass

class LSTMModel(DeepLearningModel):
    """LSTM Model"""
    
    def __init__(self, hidden_size: int = 64, num_layers: int = 2, horizon: str = '1d', sequence_length: int = 20, epochs: int = 50, lr: float = 1e-3, batch_size: int = 32):
        super().__init__(f"LSTM_{hidden_size}_{num_layers}", horizon, sequence_length=sequence_length, epochs=epochs, lr=lr, batch_size=batch_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
    
    def _create_model(self, input_size: int) -> nn.Module:
        """Create LSTM model"""
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, num_classes=2):
                super(LSTMModel, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, num_classes)
                self.dropout = nn.Dropout(0.2)
            
            def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                
                out, _ = self.lstm(x, (h0, c0))
                out = self.dropout(out[:, -1, :])
                out = self.fc(out)
                return out
        
        return LSTMModel(input_size, self.hidden_size, self.num_layers)

class GRUModel(DeepLearningModel):
    """GRU Model"""
    
    def __init__(self, hidden_size: int = 64, num_layers: int = 2, horizon: str = '1d'):
        super().__init__(f"GRU_{hidden_size}_{num_layers}", horizon)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
    
    def _create_model(self, input_size: int) -> nn.Module:
        """Create GRU model"""
        class GRUModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, num_classes=2):
                super(GRUModel, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, num_classes)
                self.dropout = nn.Dropout(0.2)
            
            def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                out, _ = self.gru(x, h0)
                out = self.dropout(out[:, -1, :])
                out = self.fc(out)
                return out
        
        return GRUModel(input_size, self.hidden_size, self.num_layers)

def test_benchmark_models():
    """Test all benchmark models"""
    print("Testing Benchmark Models")
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
    
    # Test technical analysis models
    print("\nTesting Technical Analysis Models:")
    tech_models = [
        MovingAverageCrossModel(),
        RSIModel(),
        MACDModel(),
        BollingerBandsModel(),
        MomentumModel()
    ]
    
    for model in tech_models:
        try:
            X_features = model.prepare_features(data)
            predictions = model.predict(X_features)
            print(f"✓ {model.name}: {len(predictions)} predictions")
        except Exception as e:
            print(f"✗ {model.name}: {e}")
    
    # Test machine learning models
    print("\nTesting Machine Learning Models:")
    ml_models = [
        RandomForestModel(),
        GradientBoostingModel(),
        SVMModel(),
        LogisticRegressionModel()
    ]
    
    if XGBOOST_AVAILABLE:
        ml_models.append(XGBoostModel())
    
    for model in ml_models:
        try:
            # Use first 800 samples for training, last 200 for testing
            train_data = data.iloc[:800]
            test_data = data.iloc[800:]
            
            # Train model
            metrics = model.train(train_data, train_data['target'])
            print(f"✓ {model.name}: Accuracy={metrics['accuracy']:.4f}")
            
            # Test predictions
            predictions = model.predict(test_data)
            print(f"  Test predictions: {len(predictions)}")
            
        except Exception as e:
            print(f"✗ {model.name}: {e}")
    
    # Test deep learning models
    if TORCH_AVAILABLE:
        print("\nTesting Deep Learning Models:")
        dl_models = [
            LSTMModel(),
            GRUModel()
        ]
        
        for model in dl_models:
            try:
                # Use first 800 samples for training, last 200 for testing
                train_data = data.iloc[:800]
                test_data = data.iloc[800:]
                
                # Train model
                metrics = model.train(train_data, train_data['target'])
                print(f"✓ {model.name}: Accuracy={metrics['accuracy']:.4f}")
                
                # Test predictions
                predictions = model.predict(test_data)
                print(f"  Test predictions: {len(predictions)}")
                
            except Exception as e:
                print(f"✗ {model.name}: {e}")
    
    print("\nBenchmark models test completed!")

if __name__ == "__main__":
    test_benchmark_models()
