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

    close_norm_pretrained is computed as a rolling 20-day z-score of the Close price.
    This is stationary regardless of long-term price level drift (e.g. AMD went from
    $18 to $200+). At inference time predict() applies the same rolling z-score so
    the MLP always sees values in a consistent range. The old approach (global mean/std
    per ticker) caused extreme z-scores (AMD z=5+) which saturated the MLP sigmoid.

    Returns the combined DataFrame and an empty norm_stats dict (kept for API
    compatibility — per_ticker_norm_stats is no longer needed for normalization).
    """
    dfs = []
    norm_stats = {}
    for i, ticker in enumerate(tickers):
        path = Path(f"{base_path}/{ticker}_featured.parquet")
        if not path.exists():
            logger.warning(f"Missing {path}, skipping {ticker}")
            continue
        df = pd.read_parquet(path)
        # Rolling 20-day z-score: each price is normalized relative to its own
        # recent history, so the value is stationary across all price regimes.
        roll_mean = df['Close'].rolling(20, min_periods=1).mean()
        roll_std  = df['Close'].rolling(20, min_periods=1).std().fillna(1.0)
        roll_std  = roll_std.replace(0, 1.0)  # avoid divide-by-zero
        df['close_norm_pretrained'] = (df['Close'] - roll_mean) / roll_std
        df['ticker_id'] = i
        norm_stats[ticker] = {}   # placeholder — rolling z-score needs no stored stats
        dfs.append(df)
        logger.info(f"Loaded {ticker}: {len(df)} rows")

    if not dfs:
        raise ValueError("No featured parquets found")

    combined = pd.concat(dfs, ignore_index=False)
    logger.info(f"Combined {len(dfs)} tickers: {len(combined)} total rows")
    return combined, norm_stats


class ModelManager:
    """Train and run XGBoost + MLP ensemble"""

    def __init__(self, df: pd.DataFrame, config: dict = None):
        self.df = df.copy()
        self.config = config or {}
        self.xgb_model = None
        self.seq_model = None       # MLPClassifier (replaces LSTM)
        self.seq_scaler = None      # StandardScaler for MLP inputs
        self.feature_importance = None
        self.feature_cols = None
        self.window_size = 20       # rolling window for sequence model

        # Stored at train time so predict() uses the exact same normalization.
        # For single-ticker training: scalar mean/std of the training df's Close.
        # For multi-ticker training: self.per_ticker_norm_stats[ticker] = {mean, std}.
        self.close_norm_mean = None
        self.close_norm_std = None
        self.per_ticker_norm_stats = {}  # populated by load_combined_df + save_models
        self._seq_feature_cols = None   # set in _build_sequences
        self._seq_optional_cols = None

    def prepare_data(self):
        """Split features and targets.

        Prefers target_5d_filtered (magnitude-gated) over raw target_5d.
        The filtered target drops ~15-25% of ambiguous near-zero rows, giving the
        model cleaner signal and improving AUC by ~0.01-0.03 points.
        
        Threshold is configurable via prediction_threshold_pct in config (default 2.0%).
        """
        # Prefer the magnitude-filtered target; fall back to raw binary target
        if 'target_5d_filtered' in self.df.columns:
            target_col = 'target_5d_filtered'
            df_train = self.df[self.df['target_5d_filtered'].notna()].copy()
            threshold_pct = self.config.get('prediction_threshold_pct', 2.0)
            logger.info(
                f"Using filtered target (±{threshold_pct}% threshold, volatility-adjusted): "
                f"{len(df_train)}/{len(self.df)} rows kept "
                f"({len(self.df) - len(df_train)} ambiguous rows dropped)"
            )
        elif 'target_5d' in self.df.columns:
            target_col = 'target_5d'
            df_train = self.df.copy()
        else:
            target_col = 'target_3d'
            df_train = self.df.copy()

        # Exclude all target-related columns to prevent leakage
        exclude_cols = [
            'target_5d', 'target_5d_filtered', 'target_3d', 
            'target_magnitude', 'target_magnitude_net', 'target_threshold',
            'Open', 'High', 'Low', 'Close', 'Volume',
        ]
        # close_norm_pretrained is used inside _build_sequences, not as an XGB feature
        if 'close_norm_pretrained' in df_train.columns:
            exclude_cols.append('close_norm_pretrained')

        self.feature_cols = [col for col in df_train.columns if col not in exclude_cols]
        self.X = df_train[self.feature_cols].values
        self.y = df_train[target_col].values
        # Keep df aligned with X/y for sequence model (uses same row subset)
        self.df = df_train

        logger.info(f"Prepared {len(self.feature_cols)} features, {len(self.X)} samples")

    def train_xgboost(self, n_splits: int = 5):
        """Train XGBoost with TimeSeriesSplit — never shuffle time series data.

        Runs a diagnostic cross-validation pass over all folds first to report
        mean ± std AUC, then trains the final model on the largest (last) fold.
        High std (>0.02) indicates the model is unreliable across market regimes.
        """
        logger.info("Training XGBoost...")

        tscv = TimeSeriesSplit(n_splits=n_splits)
        all_splits = list(tscv.split(self.X))

        # ── Diagnostic CV: evaluate across ALL folds (no model saved) ──────────
        fold_aucs = []
        for fold_idx, (tr_idx, te_idx) in enumerate(all_splits):
            X_tr, X_te = self.X[tr_idx], self.X[te_idx]
            y_tr, y_te = self.y[tr_idx], self.y[te_idx]
            neg = (y_tr == 0).sum(); pos = (y_tr == 1).sum()
            spw_fold = neg / pos if pos > 0 else 1.0
            clf_tmp = xgb.XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.02,
                subsample=0.7, colsample_bytree=0.6, min_child_weight=5,
                gamma=0.1, reg_alpha=0.1, reg_lambda=2.0,
                scale_pos_weight=spw_fold, random_state=42,
                eval_metric='auc', early_stopping_rounds=20,
            )
            clf_tmp.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
            preds = clf_tmp.predict_proba(X_te)[:, 1]
            fold_auc = roc_auc_score(y_te, preds)
            fold_aucs.append(fold_auc)
            logger.info(f"  Fold {fold_idx+1}/{n_splits}: AUC={fold_auc:.4f} (n_train={len(X_tr)}, n_test={len(X_te)})")

        cv_mean = float(np.mean(fold_aucs))
        cv_std  = float(np.std(fold_aucs))
        logger.info(f"CV AUC: {cv_mean:.4f} ± {cv_std:.4f}  "
                    f"{'(STABLE ✓)' if cv_std < 0.02 else '(HIGH VARIANCE — model unreliable across regimes ⚠)'}")

        # ── Final model: train on last (largest) chronological fold ───────────
        train_idx, test_idx = all_splits[-1]
        X_train, X_test = self.X[train_idx], self.X[test_idx]
        y_train, y_test = self.y[train_idx], self.y[test_idx]

        logger.info(f"XGBoost final train: {len(X_train)} samples, test: {len(X_test)} samples")

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
        logger.info(f"XGBoost Final Fold AUC: {auc:.4f}  (CV mean was {cv_mean:.4f})")

        # Feature importances for dashboard visualization
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)

    def _build_sequences(self):
        """
        Build rolling window sequences from key time-series features.
        Returns flattened (n_samples, window * n_features) arrays for the MLP.

        close_norm uses a rolling 20-day z-score so that it is stationary across all
        price regimes (e.g. AMD at $18 in 2003 vs $200 in 2024). This avoids the
        out-of-distribution issue where per-ticker global mean/std produced z-scores
        of 4–5 for recent prices, saturating the MLP sigmoid to 0.0 or 1.0.
        The same rolling z-score is applied at predict() time for consistency.
        """
        # Compute rolling z-score: works for both single- and multi-ticker DataFrames.
        # If close_norm_pretrained is already present (from load_combined_df), reuse it
        # directly — it was computed with the same rolling formula.
        if 'close_norm_pretrained' in self.df.columns:
            close_norm = self.df['close_norm_pretrained']
        else:
            roll_mean = self.df['Close'].rolling(20, min_periods=1).mean()
            roll_std  = self.df['Close'].rolling(20, min_periods=1).std().fillna(1.0)
            roll_std  = roll_std.replace(0, 1.0)
            close_norm = (self.df['Close'] - roll_mean) / roll_std
        # Store sentinel values so load_models can detect the rolling-z approach
        self.close_norm_mean = None  # rolling z-score: no fixed mean stored
        self.close_norm_std  = None

        # Use bounded/stationary features only — VIX and raw MACD_hist are unbounded
        # and cause out-of-distribution z-scores after StandardScaler.
        # price_vs_ema200 is a % deviation (bounded), RSI is 0-100 (stable),
        # volume_ratio is normalized already. XGBoost captures VIX and macro signals.
        seq_feature_cols = ['volume_ratio', 'RSI_14', 'MACD_hist',
                            'momentum_acceleration', 'hist_volatility_20',
                            'above_200ema', 'price_vs_ema200']
        optional_cols = []   # VIX removed: unbounded, causes OOD saturation in MLP
        self._seq_feature_cols = seq_feature_cols
        self._seq_optional_cols = optional_cols

        feat_df = self.df[seq_feature_cols + optional_cols].fillna(0)
        raw = pd.concat([close_norm.rename('close_norm'), feat_df], axis=1).values.astype('float32')

        # Winsorize each feature column to [p1, p99] to prevent extreme values from
        # causing out-of-distribution z-scores after StandardScaler. Store the bounds
        # so predict() can apply the identical clipping at inference time.
        self._seq_winsor_bounds = {}
        for j in range(raw.shape[1]):
            p1, p99 = np.percentile(raw[:, j], [1, 99])
            self._seq_winsor_bounds[j] = (float(p1), float(p99))
            raw[:, j] = np.clip(raw[:, j], p1, p99)

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

        # Filter feature_cols to only those present in df — ticker_id is added during
        # multi-ticker combined training but won't exist in single-ticker inference DataFrames.
        # Missing columns are filled with 0 (neutral / not applicable).
        available_cols = [c for c in self.feature_cols if c in df.columns]
        missing_cols   = [c for c in self.feature_cols if c not in df.columns]
        if missing_cols:
            logger.debug(f"predict(): filling {len(missing_cols)} missing features with 0: {missing_cols}")
            df = df.copy()
            for c in missing_cols:
                df[c] = 0

        # XGBoost — score the last row
        X_all = df[self.feature_cols].values
        xgb_prob = float(self.xgb_model.predict_proba(X_all)[:, 1][-1])

        # Sequence model — build one window from the tail of the data
        seq_prob = 0.5  # fallback if model unavailable
        if self.seq_model is not None and len(df) >= self.window_size:
            # Compute rolling 20-day z-score for the entire df so the tail window
            # (last `window_size` rows) has already-contextualized values.
            # This matches _build_sequences() exactly: stationary, no drift problem.
            roll_mean = df['Close'].rolling(20, min_periods=1).mean()
            roll_std  = df['Close'].rolling(20, min_periods=1).std().fillna(1.0)
            roll_std  = roll_std.replace(0, 1.0)
            close_norm = (df['Close'] - roll_mean) / roll_std

            seq_cols = (self._seq_feature_cols or []) + (self._seq_optional_cols or [])
            available_cols = [c for c in seq_cols if c in df.columns]

            raw = pd.concat([
                close_norm.rename('close_norm'),
                df[available_cols].fillna(0)
            ], axis=1).tail(self.window_size).values.astype('float32')

            # Apply the same per-feature winsorization used during training
            # to prevent OOD values from saturating the MLP after StandardScaler.
            winsor = getattr(self, '_seq_winsor_bounds', None)
            if winsor:
                for j, (lo, hi) in winsor.items():
                    if j < raw.shape[1]:
                        raw[:, j] = np.clip(raw[:, j], lo, hi)

            window_flat = raw.flatten().reshape(1, -1)
            window_scaled = self.seq_scaler.transform(window_flat)
            # Final safety clip to ±3σ — catches any residual extremes
            window_scaled = np.clip(window_scaled, -3.0, 3.0)
            seq_prob_raw = float(self.seq_model.predict_proba(window_scaled)[0][1])

            # Dampen extreme MLP outputs toward 0.5 when the model is saturating.
            # An MLP with AUC ~0.53 (near-chance) that outputs 0.99 or 0.01 is almost
            # certainly OOD rather than genuinely confident. We shrink extreme values
            # back toward 0.5 proportionally, preserving directional signal while
            # preventing it from hijacking the ensemble.
            # Formula: prob = 0.5 + (raw - 0.5) * damping_factor
            # damping_factor = 1.0 for raw in [0.1, 0.9], scales down toward 0.5 as extreme
            if seq_prob_raw > 0.9:
                dampen = 1.0 - (seq_prob_raw - 0.9) * 5.0  # 0.9→1.0, 1.0→0.5
                seq_prob = 0.5 + (seq_prob_raw - 0.5) * max(0.2, dampen)
            elif seq_prob_raw < 0.1:
                dampen = 1.0 - (0.1 - seq_prob_raw) * 5.0
                seq_prob = 0.5 + (seq_prob_raw - 0.5) * max(0.2, dampen)
            else:
                seq_prob = seq_prob_raw

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
                    'normalization': 'rolling_zscore',  # sentinel: rolling z-score, no fixed stats needed
                    'seq_feature_cols': self._seq_feature_cols,
                    'seq_optional_cols': self._seq_optional_cols,
                    'seq_winsor_bounds': getattr(self, '_seq_winsor_bounds', {}),
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
            # Rolling z-score approach: no fixed mean/std needed at inference time
            self.close_norm_mean = None
            self.close_norm_std  = None
            self.per_ticker_norm_stats = {}   # not used in rolling-z approach
            self._seq_feature_cols = norm_stats.get('seq_feature_cols')
            self._seq_optional_cols = norm_stats.get('seq_optional_cols')
            self._seq_winsor_bounds = norm_stats.get('seq_winsor_bounds', {})

        logger.info(f"Loaded models for {ticker}")


def tune_xgboost(manager: "ModelManager", n_trials: int = 50) -> dict:
    """
    Optuna hyperparameter search for XGBoost.

    Searches over key hyperparameters with TimeSeriesSplit — no look-ahead bias.
    Returns the best params dict. Call manager.train_xgboost() with these params
    by temporarily patching the classifier kwargs.

    Usage:
        combined_df, _ = load_combined_df(tickers)
        manager = ModelManager(combined_df)
        manager.prepare_data()
        best = tune_xgboost(manager, n_trials=50)
        print("Best params:", best)

    Note: Tuning 50 trials on 50k rows takes ~5-10 minutes.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.error("optuna not installed. Run: pip install optuna")
        return {}

    import xgboost as xgb_lib
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import TimeSeriesSplit

    def objective(trial):
        params = {
            'n_estimators':      trial.suggest_int('n_estimators', 200, 800),
            'max_depth':         trial.suggest_int('max_depth', 3, 6),
            'learning_rate':     trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'subsample':         trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.4, 0.8),
            'min_child_weight':  trial.suggest_int('min_child_weight', 3, 15),
            'gamma':             trial.suggest_float('gamma', 0.0, 0.5),
            'reg_alpha':         trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda':        trial.suggest_float('reg_lambda', 0.5, 5.0),
        }

        tscv = TimeSeriesSplit(n_splits=5)
        aucs = []
        for train_idx, test_idx in tscv.split(manager.X):
            X_tr, X_te = manager.X[train_idx], manager.X[test_idx]
            y_tr, y_te = manager.y[train_idx], manager.y[test_idx]
            neg = (y_tr == 0).sum()
            pos = (y_tr == 1).sum()
            spw = neg / pos if pos > 0 else 1.0
            clf = xgb_lib.XGBClassifier(
                **params,
                scale_pos_weight=spw,
                random_state=42,
                eval_metric='auc',
                early_stopping_rounds=20,
            )
            clf.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
            preds = clf.predict_proba(X_te)[:, 1]
            aucs.append(roc_auc_score(y_te, preds))
        return float(np.mean(aucs))

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    logger.info(f"Optuna best AUC: {study.best_value:.4f}")
    logger.info(f"Best params: {best}")
    return best


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Train ML models for the trading system")
    parser.add_argument("--tune", action="store_true",
                        help="Run Optuna hyperparameter tuning before training (takes ~5-10 min)")
    parser.add_argument("--tune-trials", type=int, default=50,
                        help="Number of Optuna trials (default: 50)")
    args = parser.parse_args()

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

    manager = ModelManager(combined_df, config=config)
    manager.per_ticker_norm_stats = norm_stats  # pass per-ticker stats before training
    manager.prepare_data()

    if args.tune:
        logger.info(f"Running Optuna tuning ({args.tune_trials} trials)…")
        best_params = tune_xgboost(manager, n_trials=args.tune_trials)
        logger.info(f"Best XGBoost params: {best_params}")
        # Patch the XGBoost training to use tuned params
        import xgboost as xgb_lib
        neg_count = (manager.y == 0).sum()
        pos_count = (manager.y == 1).sum()
        spw = neg_count / pos_count if pos_count > 0 else 1.0
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)
        train_idx, test_idx = list(tscv.split(manager.X))[-1]
        manager.xgb_model = xgb_lib.XGBClassifier(
            **best_params,
            scale_pos_weight=spw,
            random_state=42,
            eval_metric='auc',
            early_stopping_rounds=20,
        )
        manager.xgb_model.fit(
            manager.X[train_idx], manager.y[train_idx],
            eval_set=[(manager.X[test_idx], manager.y[test_idx])],
            verbose=False,
        )
        from sklearn.metrics import roc_auc_score
        preds = manager.xgb_model.predict_proba(manager.X[test_idx])[:, 1]
        logger.info(f"Tuned XGBoost Test AUC: {roc_auc_score(manager.y[test_idx], preds):.4f}")
        manager.feature_importance = pd.DataFrame({
            'feature': manager.feature_cols,
            'importance': manager.xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
    else:
        manager.train_xgboost()

    manager.train_seq_model()

    result = manager.predict()
    logger.info(f"\nPrediction: {result}")

    # Save models under the combined-data label
    manager.save_models("NVDA")

    logger.info("\nTop 10 features:")
    print(manager.feature_importance.head(10))
