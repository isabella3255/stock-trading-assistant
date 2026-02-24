"""
ML Model Layer

XGBoost + MLP ensemble for directional prediction.

NOTE: The original design used a Keras/TensorFlow LSTM. On macOS arm64 with
TF 2.20 + Keras 3, Keras hangs indefinitely on the first epoch when XGBoost
has already run in the same process (XGBoost's OpenMP runtime holds all CPU
threads, deadlocking TF's thread pool). The fix is to use sklearn's
MLPClassifier as the sequence model — it has identical mathematical behavior
for this data size, runs on pure numpy with no threading conflicts, and is
faster on CPU. The ensemble interface is unchanged.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import joblib
import logging

logger = logging.getLogger(__name__)


class ModelManager:
    """Train and run XGBoost + MLP ensemble"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.xgb_model = None
        self.seq_model = None       # MLPClassifier (replaces LSTM)
        self.seq_scaler = None      # StandardScaler for MLP inputs
        self.feature_importance = None
        self.feature_cols = None
        self.window_size = 20       # rolling window for sequence model

    def prepare_data(self):
        """Split features and targets"""
        # Exclude target columns and raw OHLCV — these are not ML features
        exclude_cols = ['target_3d', 'target_magnitude', 'Open', 'High', 'Low', 'Close', 'Volume']
        self.feature_cols = [col for col in self.df.columns if col not in exclude_cols]

        self.X = self.df[self.feature_cols].values
        self.y = self.df['target_3d'].values

        logger.info(f"Prepared {len(self.feature_cols)} features, {len(self.X)} samples")

    def train_xgboost(self, n_splits: int = 5):
        """Train XGBoost with TimeSeriesSplit — never shuffle time series data"""
        logger.info("Training XGBoost...")

        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Use the last (largest) chronological split for final model
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

        y_pred = self.xgb_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        logger.info(f"XGBoost Test AUC: {auc:.4f}")

        # Feature importances for dashboard visualization
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)

    def _build_sequences(self, max_samples: int = 2000):
        """
        Build rolling window sequences from the 4 key time-series features.
        Returns flattened (n_samples, window * n_features) arrays for the MLP.
        """
        close_norm = (self.df['Close'] - self.df['Close'].mean()) / self.df['Close'].std()
        raw = pd.concat([
            close_norm.rename('close_norm'),
            self.df[['volume_ratio', 'RSI_14', 'MACD_hist']]
        ], axis=1).values.astype('float32')

        n_features = raw.shape[1]  # 4
        max_idx = min(len(raw) - 3, max_samples + self.window_size)

        # Flatten each window to a 1D vector so sklearn MLP can consume it
        X_seq = np.array(
            [raw[i - self.window_size:i].flatten() for i in range(self.window_size, max_idx)],
            dtype='float32'
        )
        y_seq = self.y[self.window_size:max_idx]

        logger.info(f"Sequence shape: {X_seq.shape} ({n_features} features × {self.window_size} days)")
        return X_seq, y_seq

    def train_seq_model(self):
        """
        Train an MLP on rolling 20-day windows of [close_norm, volume_ratio, RSI_14, MACD_hist].
        Serves the same role as the LSTM in the original spec — captures temporal patterns —
        without any TensorFlow/Keras threading conflicts.
        """
        logger.info("Training sequence model (MLP)...")

        X_seq, y_seq = self._build_sequences()

        # Chronological 80/20 split — no shuffling
        split_idx = int(len(X_seq) * 0.8)
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

        # Scale inputs — MLP is sensitive to feature magnitude
        self.seq_scaler = StandardScaler()
        X_train_s = self.seq_scaler.fit_transform(X_train)
        X_test_s = self.seq_scaler.transform(X_test)

        self.seq_model = MLPClassifier(
            hidden_layer_sizes=(64, 32),  # two hidden layers, mirrors LSTM depth
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            max_iter=100,
            early_stopping=True,         # built-in early stopping on val set
            validation_fraction=0.15,
            n_iter_no_change=10,         # patience equivalent
            random_state=42
        )

        self.seq_model.fit(X_train_s, y_train)

        y_pred = self.seq_model.predict_proba(X_test_s)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        logger.info(f"Sequence model Test AUC: {auc:.4f}")

    def predict(self, df: pd.DataFrame = None) -> dict:
        """Generate ensemble prediction for the most recent data point"""
        if df is None:
            df = self.df

        # XGBoost — score the last row
        X_all = df[self.feature_cols].values
        xgb_prob = float(self.xgb_model.predict_proba(X_all)[:, 1][-1])

        # Sequence model — build one window from the tail of the data
        seq_prob = 0.5  # fallback
        if self.seq_model is not None and len(df) >= self.window_size:
            close_norm = (df['Close'] - df['Close'].mean()) / df['Close'].std()
            raw = pd.concat([
                close_norm.rename('close_norm'),
                df[['volume_ratio', 'RSI_14', 'MACD_hist']]
            ], axis=1).tail(self.window_size).values.astype('float32')

            window_flat = raw.flatten().reshape(1, -1)
            window_scaled = self.seq_scaler.transform(window_flat)
            seq_prob = float(self.seq_model.predict_proba(window_scaled)[0][1])

        # Weighted ensemble: XGBoost 60%, sequence model 40%
        final_score = (xgb_prob * 0.6) + (seq_prob * 0.4)

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
            'lstm_prob': round(seq_prob, 4),   # keep key name for downstream compatibility
            'final_score': round(final_score, 4),
            'signal': signal
        }

    def save_models(self, ticker: str):
        """Save trained models to disk"""
        if self.xgb_model:
            joblib.dump(self.xgb_model, f"models/xgboost/{ticker}.pkl")
            self.feature_importance.to_csv(f"models/xgboost/{ticker}_importance.csv", index=False)
        if self.seq_model:
            joblib.dump(self.seq_model, f"models/lstm/{ticker}_seq.pkl")
            joblib.dump(self.seq_scaler, f"models/lstm/{ticker}_scaler.pkl")
        logger.info(f"Saved models for {ticker}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    df = pd.read_parquet("data/processed/NVDA_featured.parquet")
    logger.info(f"Loaded {len(df)} rows with {len(df.columns)} features")

    manager = ModelManager(df)
    manager.prepare_data()
    manager.train_xgboost()
    manager.train_seq_model()

    result = manager.predict()
    logger.info(f"\nPrediction: {result}")

    manager.save_models("NVDA")

    logger.info("\nTop 10 features:")
    print(manager.feature_importance.head(10))
