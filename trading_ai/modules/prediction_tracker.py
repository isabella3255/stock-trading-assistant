"""
Prediction Tracking System

Logs all predictions made by the system and tracks their outcomes over time.
This provides real-world validation of model performance (not just backtesting).

Key difference from backtesting:
  - Backtesting: Tests on historical data (what would have happened)
  - Tracking: Logs live predictions and validates as time passes (what IS happening)

Usage in main.py:
    from modules.prediction_tracker import PredictionTracker
    tracker = PredictionTracker()
    tracker.log_prediction(
        ticker='AAPL',
        signal='BULL',
        score=0.68,
        entry_price=180.50,
        # ... other fields
    )
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class PredictionTracker:
    """
    Logs predictions and validates outcomes.
    
    The prediction log grows over time as you use the system. Each prediction
    includes all relevant context (signal, confidence, market conditions).
    After 5 days, run validate_predictions.py to check actual outcomes.
    """
    
    def __init__(self, log_path: str = "outputs/prediction_log.csv"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize log file if it doesn't exist
        if not self.log_path.exists():
            self._create_log_file()
    
    def _create_log_file(self):
        """Create empty prediction log with headers"""
        columns = [
            # Prediction metadata
            'prediction_date',
            'ticker',
            'signal',
            'final_score',
            'xgb_prob',
            'seq_prob',
            
            # Price data at prediction time
            'entry_price',
            'hist_vol_20',
            'rsi_14',
            'vix',
            'vix_3m',
            'vix_term_slope',
            
            # Predicted outcomes
            'predicted_direction',  # 'up' for BULL/STRONG_BULL
            'confidence_level',     # 'high', 'medium', 'low'
            
            # Context
            'days_to_earnings',
            'iv_crush_risk',
            'fear_greed_index',
            
            # Actual outcomes (filled in after 5 days)
            'outcome_date',
            'actual_price_5d',
            'actual_return_pct',
            'prediction_correct',
            'target_hit',
            'outcome_verified',
        ]
        
        df = pd.DataFrame(columns=columns)
        df.to_csv(self.log_path, index=False)
        logger.info(f"Created prediction log at {self.log_path}")
    
    def log_prediction(
        self,
        ticker: str,
        signal: str,
        final_score: float,
        xgb_prob: float,
        seq_prob: float,
        entry_price: float,
        hist_vol: float,
        rsi: float = None,
        vix: float = None,
        vix_3m: float = None,
        vix_term_slope: float = None,
        days_to_earnings: int = None,
        iv_crush_risk: str = None,
        fear_greed_index: float = None,
    ) -> None:
        """
        Log a prediction to the tracking file.
        
        Call this immediately after generating a prediction in main.py.
        """
        # Determine predicted direction and confidence
        if signal in ['BULL', 'STRONG_BULL']:
            predicted_direction = 'up'
        elif signal in ['BEAR', 'STRONG_BEAR']:
            predicted_direction = 'down'
        else:
            predicted_direction = 'neutral'
        
        # Confidence level based on score
        if final_score > 0.72 or final_score < 0.28:
            confidence_level = 'high'
        elif final_score > 0.62 or final_score < 0.38:
            confidence_level = 'medium'
        else:
            confidence_level = 'low'
        
        new_row = {
            'prediction_date': datetime.now().strftime('%Y-%m-%d'),
            'ticker': ticker,
            'signal': signal,
            'final_score': round(final_score, 4),
            'xgb_prob': round(xgb_prob, 4),
            'seq_prob': round(seq_prob, 4),
            'entry_price': round(entry_price, 2),
            'hist_vol_20': round(hist_vol, 2) if hist_vol else None,
            'rsi_14': round(rsi, 2) if rsi else None,
            'vix': round(vix, 2) if vix else None,
            'vix_3m': round(vix_3m, 2) if vix_3m else None,
            'vix_term_slope': round(vix_term_slope, 2) if vix_term_slope else None,
            'predicted_direction': predicted_direction,
            'confidence_level': confidence_level,
            'days_to_earnings': days_to_earnings,
            'iv_crush_risk': iv_crush_risk,
            'fear_greed_index': round(fear_greed_index, 1) if fear_greed_index else None,
            # Outcomes (to be filled in later)
            'outcome_date': None,
            'actual_price_5d': None,
            'actual_return_pct': None,
            'prediction_correct': None,
            'target_hit': None,
            'outcome_verified': False,
        }
        
        # Append to log
        if self.log_path.exists():
            existing = pd.read_csv(self.log_path)
            updated = pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)
        else:
            updated = pd.DataFrame([new_row])
        
        updated.to_csv(self.log_path, index=False)
        logger.info(f"Logged prediction: {ticker} {signal} (score={final_score:.3f})")
    
    def get_unvalidated_predictions(self, days_ago: int = 5) -> pd.DataFrame:
        """
        Get predictions that were made N days ago and haven't been validated yet.
        
        Args:
            days_ago: How many days ago to look for predictions (default 5)
        
        Returns:
            DataFrame of unvalidated predictions
        """
        if not self.log_path.exists():
            return pd.DataFrame()
        
        log = pd.read_csv(self.log_path)
        log['prediction_date'] = pd.to_datetime(log['prediction_date'])
        
        # Find predictions from N days ago that haven't been validated
        target_date = datetime.now() - timedelta(days=days_ago)
        mask = (
            (log['prediction_date'].dt.date == target_date.date()) &
            (log['outcome_verified'] == False)
        )
        
        return log[mask]
    
    def update_outcome(
        self,
        row_idx: int,
        actual_price: float,
        actual_return_pct: float,
        prediction_correct: bool,
        target_hit: bool,
    ) -> None:
        """
        Update a prediction row with actual outcome.
        
        Args:
            row_idx: Index of the prediction in the CSV
            actual_price: Actual stock price 5 days later
            actual_return_pct: Actual % return
            prediction_correct: Was the prediction direction correct?
            target_hit: Did the stock hit the target move (>2%)?
        """
        log = pd.read_csv(self.log_path)
        
        log.loc[row_idx, 'outcome_date'] = datetime.now().strftime('%Y-%m-%d')
        log.loc[row_idx, 'actual_price_5d'] = round(actual_price, 2)
        log.loc[row_idx, 'actual_return_pct'] = round(actual_return_pct, 2)
        log.loc[row_idx, 'prediction_correct'] = prediction_correct
        log.loc[row_idx, 'target_hit'] = target_hit
        log.loc[row_idx, 'outcome_verified'] = True
        
        log.to_csv(self.log_path, index=False)
        
        ticker = log.loc[row_idx, 'ticker']
        signal = log.loc[row_idx, 'signal']
        result = "CORRECT" if prediction_correct else "WRONG"
        logger.info(f"Validated: {ticker} {signal} → {result} ({actual_return_pct:+.2f}%)")
    
    def get_performance_stats(self, days: int = 30) -> Dict:
        """
        Calculate performance statistics over the last N days.
        
        Returns dict with:
          - total_predictions
          - validated_predictions
          - overall_win_rate
          - by_confidence_level
          - by_ticker
          - by_signal_type
        """
        if not self.log_path.exists():
            return {}
        
        log = pd.read_csv(self.log_path)
        log['prediction_date'] = pd.to_datetime(log['prediction_date'])
        
        # Filter to last N days
        cutoff = datetime.now() - timedelta(days=days)
        recent = log[log['prediction_date'] >= cutoff]
        
        # Only look at validated predictions for accuracy
        validated = recent[recent['outcome_verified'] == True]
        
        if len(validated) == 0:
            return {
                'total_predictions': len(recent),
                'validated_predictions': 0,
                'overall_win_rate': None,
                'message': f"No validated predictions in last {days} days (need 5 days for outcomes)"
            }
        
        # Overall stats
        overall_win_rate = (validated['prediction_correct'].sum() / len(validated)) * 100
        
        # By confidence level
        by_confidence = validated.groupby('confidence_level').agg({
            'prediction_correct': ['sum', 'count']
        })
        by_confidence.columns = ['wins', 'total']
        by_confidence['win_rate'] = (by_confidence['wins'] / by_confidence['total']) * 100
        
        # By ticker
        by_ticker = validated.groupby('ticker').agg({
            'prediction_correct': ['sum', 'count'],
            'actual_return_pct': 'mean'
        })
        by_ticker.columns = ['wins', 'total', 'avg_return']
        by_ticker['win_rate'] = (by_ticker['wins'] / by_ticker['total']) * 100
        
        return {
            'total_predictions': len(recent),
            'validated_predictions': len(validated),
            'pending_validation': len(recent) - len(validated),
            'overall_win_rate': round(overall_win_rate, 1),
            'avg_return': round(validated['actual_return_pct'].mean(), 2),
            'by_confidence': by_confidence.to_dict('index'),
            'by_ticker': by_ticker.to_dict('index'),
            'days_analyzed': days,
        }


if __name__ == "__main__":
    """Test the prediction tracker"""
    logging.basicConfig(level=logging.INFO)
    
    tracker = PredictionTracker("outputs/prediction_log_test.csv")
    
    # Log a test prediction
    tracker.log_prediction(
        ticker='AAPL',
        signal='BULL',
        final_score=0.68,
        xgb_prob=0.72,
        seq_prob=0.60,
        entry_price=180.50,
        hist_vol=25.0,
        rsi=62.0,
        vix=14.5,
        vix_3m=16.2,
        vix_term_slope=-1.7,
        days_to_earnings=30,
        iv_crush_risk='LOW',
        fear_greed_index=65.0,
    )
    
    print("✅ Prediction logged successfully")
    print(f"Check: outputs/prediction_log_test.csv")
