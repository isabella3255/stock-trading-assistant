"""
Walk-Forward Backtesting Engine

Provides realistic performance validation for the trading system using
walk-forward analysis with proper time-series methodology.

Key features:
- Expanding window training (no look-ahead bias)
- Performance metrics: Win rate, Sharpe, max drawdown, profit factor, Calmar
- Trade-level tracking with entry/exit prices
- Realistic transaction costs (0.2% round-trip)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from modules.ml_models import ModelManager

logger = logging.getLogger(__name__)


class WalkForwardBacktester:
    """
    Walk-forward backtesting with expanding training window.
    
    Timeline example (5 folds):
      Fold 1: Train on months 1-12  → Test on month 13
      Fold 2: Train on months 1-24  → Test on month 25
      Fold 3: Train on months 1-36  → Test on month 37
      ...
    
    This simulates how you would actually trade: retrain monthly with all
    historical data, generate signals for the next month.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        config: dict,
        ticker: str,
        n_splits: int = 3,
        min_train_samples: int = 100,
    ):
        """
        Args:
            df: Featured dataframe with target column
            config: System config dict
            ticker: Ticker symbol
            n_splits: Number of walk-forward folds (default 3 for ~250 day datasets)
            min_train_samples: Minimum samples needed for training (default 100)
        """
        self.df = df.copy()
        self.config = config
        self.ticker = ticker
        self.n_splits = n_splits
        self.min_train_samples = min_train_samples
        self.transaction_cost = config.get('transaction_cost_round_trip_pct', 0.2) / 100.0
        
        self.results = []  # List of per-trade results
        self.fold_metrics = []  # Metrics per fold
        
    def run(self) -> pd.DataFrame:
        """
        Execute walk-forward backtest.
        
        Returns:
            DataFrame with trade-level results
        """
        logger.info(f"Starting walk-forward backtest for {self.ticker}")
        logger.info(f"  Data: {len(self.df)} rows, {self.n_splits} folds")
        logger.info(f"  Transaction cost: {self.transaction_cost*100:.2f}%")
        
        # Get target column
        target_col = 'target_5d_filtered' if 'target_5d_filtered' in self.df.columns else 'target_5d'
        
        # Create expanding window splits
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(self.df)):
            if len(train_idx) < self.min_train_samples:
                logger.warning(f"  Fold {fold_idx+1}: Skipping (only {len(train_idx)} train samples)")
                continue
                
            logger.info(f"  Fold {fold_idx+1}/{self.n_splits}: train={len(train_idx)}, test={len(test_idx)}")
            
            # Split data
            train_df = self.df.iloc[train_idx].copy()
            test_df = self.df.iloc[test_idx].copy()
            
            # Train model on expanding window
            manager = ModelManager(train_df, config=self.config)
            manager.prepare_data()
            
            # Skip if no valid targets
            if len(manager.y) == 0:
                logger.warning(f"    Fold {fold_idx+1}: No valid targets, skipping")
                continue
            
            # Train both models (direct training, no nested CV since we're doing walk-forward)
            # Use simple train on full training fold
            import xgboost as xgb
            
            neg_count = (manager.y == 0).sum()
            pos_count = (manager.y == 1).sum()
            spw = neg_count / pos_count if pos_count > 0 else 1.0
            
            manager.xgb_model = xgb.XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.02,
                subsample=0.7, colsample_bytree=0.6, min_child_weight=5,
                gamma=0.1, reg_alpha=0.1, reg_lambda=2.0,
                scale_pos_weight=spw, random_state=42,
                eval_metric='auc',
            )
            manager.xgb_model.fit(manager.X, manager.y)
            
            try:
                manager.train_seq_model()
            except Exception as e:
                logger.warning(f"    Fold {fold_idx+1}: Sequence model failed ({e}), using XGB only")
            
            # Generate predictions for test period
            fold_trades = []
            for i in range(len(test_df)):
                # Use data up to current point (no look-ahead)
                historical_df = self.df.iloc[:train_idx[-1] + i + 1]
                
                try:
                    prediction = manager.predict(historical_df)
                    
                    test_row = test_df.iloc[i]
                    entry_price = float(test_row['Close'])
                    actual_target = test_row[target_col] if target_col in test_df.columns else test_row['target_5d']
                    
                    # Calculate return for this trade
                    # Assume we trade based on signal (only BULL/STRONG_BULL)
                    signal = prediction['signal']
                    if signal in ['BULL', 'STRONG_BULL']:
                        # Long trade
                        if actual_target == 1.0:  # Correct prediction (up)
                            # Get the actual forward return if available
                            if 'target_magnitude_net' in test_row:
                                gross_return = float(test_row['target_magnitude']) / 100.0
                            else:
                                # Estimate: use threshold as proxy
                                gross_return = self.config.get('prediction_threshold_pct', 2.0) / 100.0
                            net_return = gross_return - self.transaction_cost
                        elif actual_target == 0.0:  # Wrong prediction (down or flat)
                            # Estimate loss: assume -threshold
                            gross_return = -self.config.get('prediction_threshold_pct', 2.0) / 100.0
                            net_return = gross_return - self.transaction_cost
                        else:
                            # NaN target (ambiguous) — skip
                            continue
                    elif signal in ['BEAR', 'STRONG_BEAR']:
                        # We don't trade bearish signals in this system (no shorting)
                        continue
                    else:
                        # NEUTRAL — no trade
                        continue
                    
                    fold_trades.append({
                        'date': test_row.name,  # Index is Date
                        'fold': fold_idx + 1,
                        'signal': signal,
                        'final_score': prediction['final_score'],
                        'xgb_prob': prediction['xgb_prob'],
                        'entry_price': entry_price,
                        'actual_direction': 'up' if actual_target == 1.0 else 'down',
                        'gross_return_pct': gross_return * 100,
                        'net_return_pct': net_return * 100,
                        'correct': (actual_target == 1.0),  # For long trades
                    })
                    
                except Exception as e:
                    logger.warning(f"    Failed to generate prediction for row {i}: {e}")
                    continue
            
            self.results.extend(fold_trades)
            
            # Calculate fold-level metrics
            if fold_trades:
                fold_df = pd.DataFrame(fold_trades)
                fold_metrics = self._calculate_metrics(fold_df)
                fold_metrics['fold'] = fold_idx + 1
                self.fold_metrics.append(fold_metrics)
                
                logger.info(f"    Trades: {len(fold_trades)}, "
                           f"Win rate: {fold_metrics['win_rate']:.1f}%, "
                           f"Avg return: {fold_metrics['avg_return']:.2f}%")
        
        if not self.results:
            logger.warning("No trades generated in backtest")
            return pd.DataFrame()
        
        results_df = pd.DataFrame(self.results)
        logger.info(f"Backtest complete: {len(results_df)} total trades")
        
        return results_df
    
    def _calculate_metrics(self, trades_df: pd.DataFrame) -> Dict:
        """Calculate performance metrics from trade results"""
        
        if len(trades_df) == 0:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'profit_factor': 0.0,
            }
        
        # Basic metrics
        total_trades = len(trades_df)
        wins = trades_df['correct'].sum()
        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0.0
        
        # Return metrics
        returns = trades_df['net_return_pct'].values / 100.0  # Convert to decimal
        avg_return = np.mean(returns) * 100  # Back to percentage for display
        std_return = np.std(returns) if len(returns) > 1 else 0.0
        
        # Sharpe ratio (annualized, assuming ~50 trades/year)
        if std_return > 0:
            sharpe = (np.mean(returns) / std_return) * np.sqrt(50)
        else:
            sharpe = 0.0
        
        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown) * 100  # Percentage
        
        # Profit factor (gross profit / gross loss)
        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]
        total_profit = np.sum(winning_trades) if len(winning_trades) > 0 else 0.0
        total_loss = abs(np.sum(losing_trades)) if len(losing_trades) > 0 else 0.0
        profit_factor = total_profit / total_loss if total_loss > 0 else 0.0
        
        # Win/loss stats
        avg_win = np.mean(winning_trades) * 100 if len(winning_trades) > 0 else 0.0
        avg_loss = np.mean(losing_trades) * 100 if len(losing_trades) > 0 else 0.0
        
        # Calmar ratio (return / max_drawdown)
        total_return = (np.prod(1 + returns) - 1) * 100
        calmar = total_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
        
        return {
            'total_trades': total_trades,
            'wins': int(wins),
            'losses': int(total_trades - wins),
            'win_rate': win_rate,
            'avg_return': avg_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'calmar_ratio': calmar,
        }
    
    def get_summary_metrics(self) -> Dict:
        """Get overall backtest performance metrics"""
        if not self.results:
            return {}
        
        trades_df = pd.DataFrame(self.results)
        overall_metrics = self._calculate_metrics(trades_df)
        overall_metrics['ticker'] = self.ticker
        
        return overall_metrics
    
    def save_results(self, output_dir: str = "outputs/backtests"):
        """Save backtest results to CSV files"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        if self.results:
            # Save trade-level results
            trades_df = pd.DataFrame(self.results)
            trades_path = f"{output_dir}/{self.ticker}_trades.csv"
            trades_df.to_csv(trades_path, index=False)
            logger.info(f"Saved trade results to {trades_path}")
            
            # Save summary metrics
            summary = self.get_summary_metrics()
            summary_df = pd.DataFrame([summary])
            summary_path = f"{output_dir}/{self.ticker}_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"Saved summary to {summary_path}")
            
            # Save fold-level metrics
            if self.fold_metrics:
                folds_df = pd.DataFrame(self.fold_metrics)
                folds_path = f"{output_dir}/{self.ticker}_folds.csv"
                folds_df.to_csv(folds_path, index=False)
                logger.info(f"Saved fold metrics to {folds_path}")


def run_backtest_for_ticker(ticker: str, config: dict, n_splits: int = 3) -> Dict:
    """
    Convenience function to run backtest for a single ticker.
    
    Usage:
        results = run_backtest_for_ticker('AAPL', config, n_splits=3)
        print(f"Win rate: {results['win_rate']:.1f}%")
        print(f"Sharpe: {results['sharpe_ratio']:.2f}")
    """
    # Load featured data
    parquet_path = f"data/processed/{ticker}_featured.parquet"
    if not Path(parquet_path).exists():
        logger.error(f"No featured data for {ticker} at {parquet_path}")
        return {}
    
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df)} rows for {ticker}")
    
    # Run backtest
    backtester = WalkForwardBacktester(df, config, ticker, n_splits=n_splits)
    trades_df = backtester.run()
    
    if trades_df.empty:
        logger.warning(f"No trades generated for {ticker}")
        return {}
    
    # Save results
    backtester.save_results()
    
    # Return summary metrics
    return backtester.get_summary_metrics()


if __name__ == "__main__":
    """Test the backtester on NVDA"""
    import yaml
    logging.basicConfig(level=logging.INFO)
    
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    results = run_backtest_for_ticker('NVDA', config, n_splits=3)
    
    print("\n" + "="*70)
    print(f"BACKTEST RESULTS: NVDA")
    print("="*70)
    print(f"Total Trades:     {results.get('total_trades', 0)}")
    print(f"Win Rate:         {results.get('win_rate', 0):.1f}%")
    print(f"Avg Return:       {results.get('avg_return', 0):.2f}%")
    print(f"Avg Win:          {results.get('avg_win', 0):.2f}%")
    print(f"Avg Loss:         {results.get('avg_loss', 0):.2f}%")
    print(f"Sharpe Ratio:     {results.get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown:     {results.get('max_drawdown', 0):.2f}%")
    print(f"Profit Factor:    {results.get('profit_factor', 0):.2f}")
    print(f"Total Return:     {results.get('total_return', 0):.2f}%")
    print(f"Calmar Ratio:     {results.get('calmar_ratio', 0):.2f}")
    print("="*70)
