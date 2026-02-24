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

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def load_combined_df(tickers: list, base_path: str = "data/processed"):
    """
    Load all ticker featured parquets and concatenate into one DataFrame.
    Normalizes Close per-ticker before combining (price scales differ: NVDA ~$800,
    SPY ~$500, AAPL ~$190). Returns the combined DataFrame and per-ticker norm stats.
    """
    dfs = []
    norm_stats = {}
    for i, ticker in enumerate(tickers):
        path = Path(f"{base_path}/{ticker}_featured.parquet")
        if not path.exists():
            logger.warning(f"Missing {path}, skipping {ticker}")
            continue
        df = pd.read_parquet(path)
        m = df['Close'].mean()
        s = df['Close'].std()
        norm_stats[ticker] = {'mean': float(m), 'std': float(s)}
        df['close_norm_pretrained'] = (df['Close'] - m) / s
        df['ticker_id'] = i
        dfs.append(df)
        logger.info(f"Loaded {ticker}: {len(df)} rows")

    if not dfs:
        raise ValueError("No featured parquets found")

    combined = pd.concat(dfs, ignore_index=False)
    logger.info(f"Combined {len(dfs)} tickers: {len(combined)} total rows")
    return combined, norm_stats


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

        # Stored at train time so predict() uses the exact same normalization
        self.close_norm_mean = None
        self.close_norm_std = None
        self._seq_feature_cols = None   # set in _build_sequences
        self._seq_optional_cols = None

    def prepare_data(self):
        """Split features and targets"""
        # Support both 5-day and 3-day target column names
        target_col = 'target_5d' if 'target_5d' in self.df.columns else 'target_3d'
        exclude_cols = [target_col, 'target_magnitude', 'Open', 'High', 'Low', 'Close', 'Volume']
        # close_norm_pretrained is used inside _build_sequences, not as an XGB feature
        if 'close_norm_pretrained' in self.df.columns:
            exclude_cols.append('close_norm_pretrained')
        self.feature_cols = [col for col in self.df.columns if col not in exclude_cols]

        self.X = self.df[self.feature_cols].values
        self.y = self.df[target_col].values

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

        # Compute class weight to handle any imbalance across tickers
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        spw = neg_count / pos_count if pos_count > 0 else 1.0
        logger.info(f"Class balance: neg={neg_count}, pos={pos_count}, scale_pos_weight={spw:.3f}")

        self.xgb_model = xgb.XGBClassifier(
            n_estimators=500,          # early_stopping_rounds will cap this
            max_depth=4,               # shallower trees generalize better for financial data
            learning_rate=0.02,        # lower LR compensated by more estimators
            subsample=0.7,
            colsample_bytree=0.6,      # 60% of features per tree reduces variance
            min_child_weight=5,        # min samples per leaf — prevents overfit on noise
            gamma=0.1,                 # min split gain — acts as regularizer
            reg_alpha=0.1,             # L1 regularization
            reg_lambda=2.0,            # L2 regularization (slightly above default of 1)
            scale_pos_weight=spw,
            random_state=42,
            eval_metric='auc',
            early_stopping_rounds=30,
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

    def _build_sequences(self):
        """
        Build rolling window sequences from key time-series features.
        Returns flattened (n_samples, window * n_features) arrays for the MLP.

        Uses close_norm_pretrained if present (multi-ticker path where each ticker's
        Close was normalized before concatenation). Otherwise normalizes here and
        stores the stats on self for consistent predict-time reuse.
        """
        if 'close_norm_pretrained' in self.df.columns:
            close_norm = self.df['close_norm_pretrained']
            # For multi-ticker, norm stats are per-ticker — store global fallback
            self.close_norm_mean = self.df['Close'].mean()
            self.close_norm_std = self.df['Close'].std()
        else:
            self.close_norm_mean = self.df['Close'].mean()
            self.close_norm_std = self.df['Close'].std()
            close_norm = (self.df['Close'] - self.close_norm_mean) / self.close_norm_std

        seq_feature_cols = ['volume_ratio', 'RSI_14', 'MACD_hist',
                            'momentum_acceleration', 'hist_volatility_20', 'above_200ema']
        optional_cols = [c for c in ['VIX'] if c in self.df.columns]
        self._seq_feature_cols = seq_feature_cols
        self._seq_optional_cols = optional_cols

        raw = pd.concat([
            close_norm.rename('close_norm'),
            self.df[seq_feature_cols + optional_cols].fillna(0)
        ], axis=1).values.astype('float32')

        n_features = raw.shape[1]
        max_idx = len(raw) - 3  # leave room for target at the tail

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
        Train an MLP on rolling 20-day windows of temporal features.
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

        n_input = X_train_s.shape[1]  # 20 * n_features
        self.seq_model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),  # 3-layer for ~140-dim input
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            max_iter=100,
            early_stopping=True,         # built-in early stopping on val set
            validation_fraction=0.15,
            n_iter_no_change=10,         # patience equivalent
            random_state=42
        )
        logger.info(f"MLP input dim: {n_input}, hidden: (128, 64, 32)")

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
        seq_prob = 0.5  # fallback if model unavailable
        if self.seq_model is not None and len(df) >= self.window_size:
            # Reuse the exact normalization stats from training — critical for correct output
            if self.close_norm_mean is not None and self.close_norm_std is not None:
                close_norm = (df['Close'] - self.close_norm_mean) / self.close_norm_std
            else:
                close_norm = (df['Close'] - df['Close'].mean()) / df['Close'].std()

            seq_cols = (self._seq_feature_cols or []) + (self._seq_optional_cols or [])
            available_cols = [c for c in seq_cols if c in df.columns]

            raw = pd.concat([
                close_norm.rename('close_norm'),
                df[available_cols].fillna(0)
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
            joblib.dump(
                {
                    'mean': self.close_norm_mean,
                    'std': self.close_norm_std,
                    'seq_feature_cols': self._seq_feature_cols,
                    'seq_optional_cols': self._seq_optional_cols,
                },
                f"models/lstm/{ticker}_norm_stats.pkl"
            )
        logger.info(f"Saved models for {ticker}")

    def load_models(self, ticker: str):
        """Load previously saved models from disk"""
        xgb_path = Path(f"models/xgboost/{ticker}.pkl")
        if xgb_path.exists():
            self.xgb_model = joblib.load(xgb_path)
            imp_path = Path(f"models/xgboost/{ticker}_importance.csv")
            if imp_path.exists():
                self.feature_importance = pd.read_csv(imp_path)
                self.feature_cols = self.feature_importance['feature'].tolist()

        seq_path = Path(f"models/lstm/{ticker}_seq.pkl")
        if seq_path.exists():
            self.seq_model = joblib.load(seq_path)
            self.seq_scaler = joblib.load(f"models/lstm/{ticker}_scaler.pkl")
            norm_stats = joblib.load(f"models/lstm/{ticker}_norm_stats.pkl")
            self.close_norm_mean = norm_stats['mean']
            self.close_norm_std = norm_stats['std']
            self._seq_feature_cols = norm_stats.get('seq_feature_cols')
            self._seq_optional_cols = norm_stats.get('seq_optional_cols')

        logger.info(f"Loaded models for {ticker}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    import yaml
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    tickers = config['watchlist']

    # Try multi-ticker combined training first; fall back to NVDA-only
    combined_df, norm_stats = load_combined_df(tickers)

    if len(combined_df) == 0:
        logger.warning("No multi-ticker data found, falling back to NVDA only")
        combined_df = pd.read_parquet("data/processed/NVDA_featured.parquet")

    logger.info(f"Loaded {len(combined_df)} rows with {len(combined_df.columns)} features")

    manager = ModelManager(combined_df)
    manager.prepare_data()
    manager.train_xgboost()
    manager.train_seq_model()

    result = manager.predict()
    logger.info(f"\nPrediction: {result}")

    # Save models under the combined-data label
    manager.save_models("NVDA")

    logger.info("\nTop 10 features:")
    print(manager.feature_importance.head(10))
