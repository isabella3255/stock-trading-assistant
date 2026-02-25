"""
Prediction Outcome Validator

Runs daily (after 4 PM ET market close) to validate predictions made 5 days ago.
Fetches actual stock prices and calculates whether predictions were correct.

Usage:
    cd trading_ai
    source ../.venv/bin/activate
    python validate_predictions.py

Recommended: Run this via cron job daily at 4:05 PM ET:
    5 16 * * 1-5 cd /path/to/trading_ai && source ../.venv/bin/activate && python validate_predictions.py
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from modules.prediction_tracker import PredictionTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_actual_price(ticker: str, target_date: datetime) -> float:
    """
    Fetch the closing price for a ticker on a specific date.
    
    Args:
        ticker: Stock symbol
        target_date: Date to fetch price for
    
    Returns:
        Closing price, or None if unavailable
    """
    try:
        # Fetch 10 days of data to ensure we get the target date
        end_date = target_date + timedelta(days=10)
        stock = yf.Ticker(ticker)
        hist = stock.history(start=target_date, end=end_date)
        
        if hist.empty:
            logger.warning(f"No price data for {ticker} on {target_date.date()}")
            return None
        
        # Get closest date (handles weekends/holidays)
        hist.index = hist.index.tz_localize(None)  # Remove timezone for comparison
        closest_date = min(hist.index, key=lambda x: abs(x - target_date))
        
        return float(hist.loc[closest_date, 'Close'])
        
    except Exception as e:
        logger.error(f"Failed to fetch price for {ticker}: {e}")
        return None


def validate_predictions(days_ago: int = 5, dry_run: bool = False):
    """
    Validate predictions made N days ago by fetching actual outcomes.
    
    Args:
        days_ago: How many days ago to validate (default 5, matches prediction horizon)
        dry_run: If True, show what would be validated but don't update the log
    """
    tracker = PredictionTracker()
    
    # Get unvalidated predictions from N days ago
    unvalidated = tracker.get_unvalidated_predictions(days_ago=days_ago)
    
    if unvalidated.empty:
        logger.info(f"No unvalidated predictions from {days_ago} days ago")
        return
    
    logger.info(f"Found {len(unvalidated)} predictions to validate")
    logger.info(f"Prediction date: {unvalidated.iloc[0]['prediction_date']}")
    
    # Load full log to update
    log = pd.read_csv(tracker.log_path)
    log['prediction_date'] = pd.to_datetime(log['prediction_date'])
    
    validated_count = 0
    for idx, row in unvalidated.iterrows():
        ticker = row['ticker']
        entry_price = row['entry_price']
        predicted_direction = row['predicted_direction']
        signal = row['signal']
        
        # Calculate target date (5 trading days ~= 7 calendar days)
        prediction_date = pd.to_datetime(row['prediction_date'])
        target_date = prediction_date + timedelta(days=7)
        
        # Fetch actual price
        actual_price = fetch_actual_price(ticker, target_date)
        
        if actual_price is None:
            logger.warning(f"Skipping {ticker}: No price data available")
            continue
        
        # Calculate actual return
        actual_return_pct = ((actual_price - entry_price) / entry_price) * 100
        
        # Determine if prediction was correct
        if predicted_direction == 'up':
            # For bullish predictions: correct if stock went up
            prediction_correct = (actual_return_pct > 0)
            # Target hit if moved >2% (configurable threshold)
            target_hit = (actual_return_pct > 2.0)
        elif predicted_direction == 'down':
            prediction_correct = (actual_return_pct < 0)
            target_hit = (actual_return_pct < -2.0)
        else:
            # Neutral predictions: correct if stayed within ±2%
            prediction_correct = (abs(actual_return_pct) < 2.0)
            target_hit = False
        
        result_emoji = "✅" if prediction_correct else "❌"
        logger.info(f"{result_emoji} {ticker} {signal}: {actual_return_pct:+.2f}% "
                   f"(predicted {predicted_direction}, entry ${entry_price:.2f} → ${actual_price:.2f})")
        
        if not dry_run:
            # Update the log
            log.loc[idx, 'outcome_date'] = datetime.now().strftime('%Y-%m-%d')
            log.loc[idx, 'actual_price_5d'] = round(actual_price, 2)
            log.loc[idx, 'actual_return_pct'] = round(actual_return_pct, 2)
            log.loc[idx, 'prediction_correct'] = prediction_correct
            log.loc[idx, 'target_hit'] = target_hit
            log.loc[idx, 'outcome_verified'] = True
            
            validated_count += 1
    
    if not dry_run and validated_count > 0:
        log.to_csv(tracker.log_path, index=False)
        logger.info(f"✅ Validated {validated_count} predictions and updated log")
    elif dry_run:
        logger.info(f"DRY RUN: Would have validated {len(unvalidated)} predictions")
    
    # Print summary
    if validated_count > 0:
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        
        # Calculate stats for newly validated
        correct_count = log.loc[unvalidated.index, 'prediction_correct'].sum()
        total_count = len(unvalidated)
        win_rate = (correct_count / total_count) * 100 if total_count > 0 else 0
        avg_return = log.loc[unvalidated.index, 'actual_return_pct'].mean()
        
        print(f"Validated: {validated_count} predictions")
        print(f"Win Rate:  {win_rate:.1f}% ({int(correct_count)}W / {total_count - int(correct_count)}L)")
        print(f"Avg Return: {avg_return:+.2f}%")
        print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate predictions made N days ago")
    parser.add_argument("--days", type=int, default=5,
                       help="How many days ago to validate (default: 5)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be validated but don't update log")
    args = parser.parse_args()
    
    logger.info(f"Starting prediction validation for predictions made {args.days} days ago")
    validate_predictions(days_ago=args.days, dry_run=args.dry_run)
    
    # Show overall performance stats
    tracker = PredictionTracker()
    stats = tracker.get_performance_stats(days=30)
    
    if stats and 'overall_win_rate' in stats and stats['overall_win_rate'] is not None:
        print("\n" + "="*70)
        print("PERFORMANCE STATS - LAST 30 DAYS")
        print("="*70)
        print(f"Total Predictions:    {stats['total_predictions']}")
        print(f"Validated:            {stats['validated_predictions']}")
        print(f"Pending (< 5 days):   {stats['pending_validation']}")
        print(f"Overall Win Rate:     {stats['overall_win_rate']:.1f}%")
        print(f"Avg Return:           {stats['avg_return']:+.2f}%")
        print("="*70)
