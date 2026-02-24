"""
Feature Engineering Module

Computes 70+ technical indicators and features from OHLCV data.
"""

import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, SMAIndicator, MACD
from ta.momentum import RSIIndicator, ROCIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Compute technical features from price data"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
    def add_price_trend_features(self):
        """EMAs, SMAs, price distances, crossovers"""
        close = self.df['Close']
        
        # EMAs
        for period in [9, 21, 50, 200]:
            self.df[f'EMA_{period}'] = EMAIndicator(close, window=period).ema_indicator()
        
        # SMAs
        for period in [20, 50]:
            self.df[f'SMA_{period}'] = SMAIndicator(close, window=period).sma_indicator()
        
        # Price vs EMAs
        self.df['price_vs_ema9'] = ((close - self.df['EMA_9']) / self.df['EMA_9']) * 100
        self.df['price_vs_ema200'] = ((close - self.df['EMA_200']) / self.df['EMA_200']) * 100
        self.df['above_200ema'] = (close > self.df['EMA_200']).astype(int)
        
        # EMA crossovers
        self.df['ema9_cross_ema21'] = ((self.df['EMA_9'] > self.df['EMA_21']) & 
                                       (self.df['EMA_9'].shift(1) <= self.df['EMA_21'].shift(1))).astype(int)
        
        # Trend strength (EMA9 slope)
        self.df['trend_strength'] = self.df['EMA_9'].diff(5)
        
    def add_momentum_features(self):
        """ROC, MACD, RSI, Stochastic"""
        close = self.df['Close']
        
        # Rate of Change
        self.df['ROC_10'] = ROCIndicator(close, window=10).roc()
        self.df['ROC_20'] = ROCIndicator(close, window=20).roc()
        self.df['momentum_acceleration'] = self.df['ROC_10'] - self.df['ROC_10'].shift(5)
        
        # MACD
        macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
        self.df['MACD'] = macd.macd()
        self.df['MACD_signal'] = macd.macd_signal()
        self.df['MACD_hist'] = macd.macd_diff()
        self.df['MACD_cross'] = ((self.df['MACD'] > self.df['MACD_signal']) & 
                                 (self.df['MACD'].shift(1) <= self.df['MACD_signal'].shift(1))).astype(int)
        
        # RSI
        rsi = RSIIndicator(close, window=14)
        self.df['RSI_14'] = rsi.rsi()
        self.df['RSI_oversold'] = (self.df['RSI_14'] < 35).astype(int)
        self.df['RSI_overbought'] = (self.df['RSI_14'] > 70).astype(int)
        
        # Stochastic
        stoch = StochasticOscillator(self.df['High'], self.df['Low'], close, window=14, smooth_window=3)
        self.df['Stoch_K'] = stoch.stoch()
        self.df['Stoch_D'] = stoch.stoch_signal()
        
    def add_volatility_features(self):
        """ATR, Bollinger Bands, Historical Volatility"""
        close = self.df['Close']
        high = self.df['High']
        low = self.df['Low']
        
        # ATR
        atr = AverageTrueRange(high, low, close, window=14)
        self.df['ATR_14'] = atr.average_true_range()
        
        # Bollinger Bands
        bb = BollingerBands(close, window=20, window_dev=2)
        self.df['BB_upper'] = bb.bollinger_hband()
        self.df['BB_lower'] = bb.bollinger_lband()
        self.df['BB_mid'] = bb.bollinger_mavg()
        
        # BB Squeeze (band width in lowest 20% of last 90 days)
        bb_width = self.df['BB_upper'] - self.df['BB_lower']
        bb_width_percentile = bb_width.rolling(90).apply(lambda x: (x.iloc[-1] > x).mean())
        self.df['BB_squeeze'] = (bb_width_percentile < 0.20).astype(int)
        
        # Historical Volatility (annualized)
        log_returns = np.log(close / close.shift(1))
        self.df['hist_volatility_20'] = log_returns.rolling(20).std() * np.sqrt(252)
        self.df['hist_volatility_60'] = log_returns.rolling(60).std() * np.sqrt(252)
        self.df['hv_ratio'] = self.df['hist_volatility_20'] / self.df['hist_volatility_60']
        
    def add_volume_features(self):
        """VWAP, volume ratios, OBV"""
        close = self.df['Close']
        volume = self.df['Volume']
        
        # VWAP (daily proxy using rolling)
        typical_price = (self.df['High'] + self.df['Low'] + close) / 3
        self.df['VWAP'] = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum()
        
        # Volume ratio
        avg_volume_20 = volume.rolling(20).mean()
        self.df['volume_ratio'] = volume / avg_volume_20
        self.df['volume_spike'] = (self.df['volume_ratio'] > 2.0).astype(int)
        
        # OBV
        obv = OnBalanceVolumeIndicator(close, volume)
        self.df['OBV'] = obv.on_balance_volume()
        self.df['OBV_trend'] = self.df['OBV'].diff(10)
        
        # Price-volume divergence
        price_change = close.pct_change(5)
        obv_change = self.df['OBV'].pct_change(5)
        self.df['price_volume_divergence'] = ((price_change > 0) & (obv_change < 0)).astype(int)
        
    def add_pattern_features(self):
        """Higher highs/lows, inside days, bull flags, breakouts"""
        high = self.df['High']
        low = self.df['Low']
        close = self.df['Close']
        
        # Higher high/low
        self.df['higher_high'] = (high > high.shift(1).rolling(5).max()).astype(int)
        self.df['higher_low'] = (low > low.rolling(5).min().shift(1)).astype(int)
        
        # Inside day
        self.df['inside_day'] = ((high < high.shift(1)) & (low > low.shift(1))).astype(int)
        
        # Bull flag setup
        strong_uptrend = ((close > self.df['EMA_9']) & 
                         (self.df['EMA_9'] > self.df['EMA_21']) & 
                         (self.df['ROC_10'] > 5))
        
        consolidation = ((self.df['ATR_14'] < self.df['ATR_14'].rolling(14).mean()) & 
                        (self.df['inside_day'].rolling(3).sum() >= 2))
        
        volume_declining = (self.df['Volume'] < 0.7 * self.df['Volume'].rolling(5).mean())
        
        self.df['bull_flag_setup'] = (strong_uptrend & consolidation & volume_declining).astype(int)
        
        # Breakout signal
        high_20d = high.rolling(20).max()
        self.df['breakout_signal'] = ((close > high_20d.shift(1)) & 
                                      (self.df['volume_ratio'] > 1.5)).astype(int)
        
    def add_target_variables(self, horizon: int = 3):
        """Target for ML: price up/down in N days"""
        close = self.df['Close']
        
        # Binary target: 1 if price goes up
        future_price = close.shift(-horizon)
        self.df['target_3d'] = (future_price > close).astype(int)
        
        # Magnitude target: % change
        self.df['target_magnitude'] = ((future_price - close) / close) * 100
        
    def compute_all_features(self) -> pd.DataFrame:
        """Run all feature computations"""
        logger.info("Computing price & trend features...")
        self.add_price_trend_features()
        
        logger.info("Computing momentum features...")
        self.add_momentum_features()
        
        logger.info("Computing volatility features...")
        self.add_volatility_features()
        
        logger.info("Computing volume features...")
        self.add_volume_features()
        
        logger.info("Computing pattern features...")
        self.add_pattern_features()
        
        logger.info("Adding target variables...")
        self.add_target_variables()
        
        # Forward-fill macro columns before dropping — they update infrequently
        # and would otherwise wipe out years of historical price data
        macro_cols = ['VIX', 'FED_RATE', 'TREASURY_10Y', 'TREASURY_2Y', 'CPI', 'UNEMPLOYMENT']
        existing_macro = [c for c in macro_cols if c in self.df.columns]
        if existing_macro:
            self.df[existing_macro] = self.df[existing_macro].ffill().bfill()

        # Drop rows with NaN (from indicators that need warmup)
        initial_rows = len(self.df)
        self.df = self.df.dropna()
        logger.info(f"Dropped {initial_rows - len(self.df)} warmup rows, {len(self.df)} remaining")
        
        return self.df


if __name__ == "__main__":
    # Test with NVDA data
    logging.basicConfig(level=logging.INFO)
    
    df = pd.read_parquet("data/processed/NVDA.parquet")
    logger.info(f"Loaded {len(df)} rows")
    
    engineer = FeatureEngineer(df)
    df_featured = engineer.compute_all_features()
    
    # Save it
    output_file = "data/processed/NVDA_featured.parquet"
    df_featured.to_parquet(output_file)
    logger.info(f"Saved {len(df_featured)} rows to {output_file}")
    
    logger.info(f"\nFeatures: {len(df_featured.columns)} columns")
    logger.info(f"Columns: {list(df_featured.columns)}")
    logger.info(f"\nLast 5 rows:")
    print(df_featured.tail())
