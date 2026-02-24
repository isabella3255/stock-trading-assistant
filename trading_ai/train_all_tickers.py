#!/usr/bin/env python3
"""
Train ML models for all tickers in watchlist.

This creates individual models per ticker, each trained on that ticker's
historical data. Alternative: use modules/ml_models.py directly to train
one combined model on all 10 tickers (~50k rows).
"""

import sys
import logging
from pathlib import Path

import pandas as pd
import yaml

from modules.ml_models import ModelManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_ticker(ticker: str) -> bool:
    """Train XGBoost + MLP for one ticker. Returns True on success."""
    try:
        logger.info(f"\n{'='*70}")
        logger.info(f"Training models for {ticker}")
        logger.info(f"{'='*70}")
        
        # Load featured data
        df_path = Path(f"data/processed/{ticker}_featured.parquet")
        if not df_path.exists():
            logger.error(f"Missing {df_path} - run run_pipeline.py first")
            return False
        
        df = pd.read_parquet(df_path)
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} features")
        
        # Initialize model manager
        manager = ModelManager(df)
        manager.prepare_data()
        
        # Train both models
        logger.info("Training XGBoost...")
        manager.train_xgboost()
        
        logger.info("Training sequence model (MLP)...")
        manager.train_seq_model()
        
        # Save models
        manager.save_models(ticker)
        logger.info(f"✓ {ticker} models saved successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ {ticker} training failed: {e}", exc_info=True)
        return False


def main():
    """Train all tickers in watchlist"""
    
    # Load config
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    tickers = config['watchlist']
    
    print(f"\n{'='*70}")
    print(f"TRAINING MODELS FOR {len(tickers)} TICKERS")
    print(f"{'='*70}\n")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Estimated time: ~{len(tickers) * 0.5}-{len(tickers)} minutes\n")
    
    results = {'success': [], 'failed': []}
    
    for ticker in tickers:
        success = train_ticker(ticker)
        if success:
            results['success'].append(ticker)
        else:
            results['failed'].append(ticker)
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"✓ Success: {results['success']}")
    print(f"✗ Failed:  {results['failed']}")
    print(f"{'='*70}\n")
    
    return 0 if not results['failed'] else 1


if __name__ == "__main__":
    sys.exit(main())
