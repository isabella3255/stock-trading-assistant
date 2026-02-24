"""
ML Model Layer

XGBoost + LSTM ensemble for directional prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import xgboost as xgb
# Use keras directly (Keras 3 standalone) instead of tensorflow.keras
# to avoid the Sequential + input_shape hang on arm64/CPU-only machines
import keras
from keras import layers, Input, Model
import joblib
import logging

logger = logging.getLogger(__name__)


class ModelManager:
    """Train and run XGBoost + LSTM ensemble"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.xgb_model = None
        self.lstm_model = None
        self.feature_importance = None
        # Store the lstm feature columns so predict() can build sequences correctly
        self.lstm_feature_cols = None
        self.lstm_window_size = None

    def prepare_data(self):
        """Split features and targets"""
        # Remove target columns and raw OHLCV from ML features
        exclude_cols = ['target_3d', 'target_magnitude', 'Open', 'High', 'Low', 'Close', 'Volume']
        self.feature_cols = [col for col in self.df.columns if col not in exclude_cols]

        self.X = self.df[self.feature_cols].values
        self.y = self.df['target_3d'].values

        logger.info(f"Prepared {len(self.feature_cols)} features, {len(self.X)} samples")

    def train_xgboost(self, n_splits: int = 5):
        """Train XGBoost with TimeSeriesSplit — never shuffle time series data"""
        logger.info("Training XGBoost...")

        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Use the last (largest) split for final training
        train_idx, test_idx = list(tscv.split(self.X))[-1]
        X_train, X_test = self.X[train_idx], self.X[test_idx]
        y_train, y_test = self.y[train_idx], self.y[test_idx]

        logger.info(f"XGBoost train: {len(X_train)} samples, test: {len(X_test)} samples")

        self.xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='auc'
        )

        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # Score on held-out test set
        y_pred = self.xgb_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        logger.info(f"XGBoost Test AUC: {auc:.4f}")

        # Save feature importances for dashboard visualization
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)

    def train_lstm(self, window_size: int = 20):
        """Train LSTM with 20-day sequences using Keras functional API.

        Uses keras.Input() style (not Sequential + input_shape) to avoid
        a known hang in Keras 3 on CPU-only / arm64 machines.
        """
        logger.info("Training LSTM...")

        # Key features for LSTM sequence input
        lstm_feature_names = ['close_normalized', 'volume_ratio', 'RSI_14', 'MACD_hist']
        self.lstm_feature_cols = lstm_feature_names
        self.lstm_window_size = window_size

        # Normalize close price to zero mean / unit variance
        close_normalized = (self.df['Close'] - self.df['Close'].mean()) / self.df['Close'].std()

        # Build the 4-column input DataFrame
        lstm_data = pd.concat([
            close_normalized.rename('close_normalized'),
            self.df[['volume_ratio', 'RSI_14', 'MACD_hist']]
        ], axis=1).values.astype('float32')  # Convert to numpy once — avoids repeated .iloc overhead

        n_features = lstm_data.shape[1]  # Should be 4

        # Create sliding window sequences
        # Cap at 2000 samples so training stays fast on CPU
        max_idx = min(len(lstm_data) - 3, 2000 + window_size)
        X_seq = np.array([lstm_data[i - window_size:i] for i in range(window_size, max_idx)], dtype='float32')
        y_seq = np.array([self.y[i] for i in range(window_size, max_idx)], dtype='float32')

        logger.info(f"LSTM sequences: {X_seq.shape}, labels: {y_seq.shape}")

        # Chronological 80/20 split — no shuffling
        split_idx = int(len(X_seq) * 0.8)
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

        # --- Build model with functional API (fixes Keras 3 Sequential hang) ---
        inp = Input(shape=(window_size, n_features))
        x = layers.LSTM(32, return_sequences=False)(inp)
        x = layers.Dense(16, activation='relu')(x)
        out = layers.Dense(1, activation='sigmoid')(x)
        self.lstm_model = Model(inp, out)

        self.lstm_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        early_stop = keras.callbacks.EarlyStopping(
            patience=3,
            restore_best_weights=True
        )

        self.lstm_model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=10,
            batch_size=64,
            callbacks=[early_stop],
            verbose=1
        )

        # Score
        y_pred = self.lstm_model.predict(X_test, verbose=0).flatten()
        auc = roc_auc_score(y_test, y_pred)
        logger.info(f"LSTM Test AUC: {auc:.4f}")

    def _build_lstm_sequence(self, df: pd.DataFrame) -> np.ndarray:
        """Build the last window_size rows as an LSTM input sequence."""
        close_norm = (df['Close'] - df['Close'].mean()) / df['Close'].std()
        seq_data = pd.concat([
            close_norm.rename('close_normalized'),
            df[['volume_ratio', 'RSI_14', 'MACD_hist']]
        ], axis=1)
        # Take the last window_size rows and reshape to (1, window, features)
        return seq_data.tail(self.lstm_window_size).values.astype('float32')[np.newaxis, :, :]

    def predict(self, df: pd.DataFrame = None) -> dict:
        """Generate ensemble prediction for the most recent data point"""
        if df is None:
            df = self.df

        # XGBoost — score all rows, take the last one
        X_all = df[self.feature_cols].values
        xgb_prob = float(self.xgb_model.predict_proba(X_all)[:, 1][-1])

        # LSTM — build a proper sequence from the tail of the data
        if self.lstm_model is not None and len(df) >= self.lstm_window_size:
            seq = self._build_lstm_sequence(df)
            lstm_prob = float(self.lstm_model.predict(seq, verbose=0)[0][0])
        else:
            # Fallback if LSTM not trained or not enough data
            lstm_prob = 0.5

        # Weighted ensemble: XGBoost 60%, LSTM 40%
        final_score = (xgb_prob * 0.6) + (lstm_prob * 0.4)

        # Map score to named signal
        if final_score > 0.72:
            signal = "STRONG_BULL"
        elif final_score > 0.62:
            signal = "BULL"
        elif final_score < 0.28:
            signal = "STRONG_BEAR"
        elif final_score < 0.38:
            signal = "BEAR"
        else:
            signal = "NEUTRAL"

        return {
            'xgb_prob': round(xgb_prob, 4),
            'lstm_prob': round(lstm_prob, 4),
            'final_score': round(final_score, 4),
            'signal': signal
        }

    def save_models(self, ticker: str):
        """Save trained models to disk"""
        if self.xgb_model:
            joblib.dump(self.xgb_model, f"models/xgboost/{ticker}.pkl")
            self.feature_importance.to_csv(f"models/xgboost/{ticker}_importance.csv", index=False)
        if self.lstm_model:
            # Use .keras format — .h5 is deprecated in Keras 3
            self.lstm_model.save(f"models/lstm/{ticker}.keras")
        logger.info(f"Saved models for {ticker}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Load featured data
    df = pd.read_parquet("data/processed/NVDA_featured.parquet")
    logger.info(f"Loaded {len(df)} rows with {len(df.columns)} features")

    # Train models
    manager = ModelManager(df)
    manager.prepare_data()
    manager.train_xgboost()
    manager.train_lstm()

    # Test prediction on most recent data
    result = manager.predict()
    logger.info(f"\nPrediction: {result}")

    # Save models
    manager.save_models("NVDA")

    # Show top features
    logger.info("\nTop 10 features:")
    print(manager.feature_importance.head(10))
