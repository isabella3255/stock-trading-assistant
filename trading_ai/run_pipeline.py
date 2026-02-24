#!/usr/bin/env python3
"""
run_pipeline.py

Runs DataPipeline + FeatureEngineer for every ticker in config/config.yaml
and saves {ticker}_featured.parquet to data/processed/.

Run from the trading_ai/ directory:
    ./venv/bin/python3 run_pipeline.py
    ./venv/bin/python3 run_pipeline.py --tickers NVDA AAPL   # subset
    ./venv/bin/python3 run_pipeline.py --skip-existing        # skip already-processed
    ./venv/bin/python3 run_pipeline.py --delay 3.0            # longer API delay
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml

from modules.data_pipeline import DataPipeline
from modules.feature_engineering import FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_for_ticker(ticker: str, pipeline: DataPipeline) -> bool:
    """Run data pipeline + feature engineering for one ticker. Returns True on success."""
    try:
        logger.info(f"{'='*60}")
        logger.info(f"  {ticker}: data pipeline")
        logger.info(f"{'='*60}")
        df = pipeline.process_ticker(ticker)

        logger.info(f"  {ticker}: feature engineering")
        engineer = FeatureEngineer(df)
        df_featured = engineer.compute_all_features()

        out_path = f"data/processed/{ticker}_featured.parquet"
        df_featured.to_parquet(out_path)
        logger.info(f"  {ticker}: saved {len(df_featured)} rows -> {out_path}")
        return True
    except Exception as e:
        logger.error(f"  {ticker}: FAILED — {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run data pipeline for configured tickers")
    parser.add_argument("--tickers", nargs="+", help="Override tickers (default: all from config)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip tickers that already have a _featured.parquet")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Seconds to sleep between tickers (default: 2.0 for API rate limiting)")
    args = parser.parse_args()

    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    tickers = args.tickers if args.tickers else config['watchlist']
    logger.info(f"Processing {len(tickers)} tickers: {tickers}")

    pipeline = DataPipeline()

    results = {'success': [], 'skipped': [], 'failed': []}

    for i, ticker in enumerate(tickers):
        out_path = Path(f"data/processed/{ticker}_featured.parquet")
        if args.skip_existing and out_path.exists():
            logger.info(f"  {ticker}: skipping (already exists)")
            results['skipped'].append(ticker)
            continue

        ok = run_for_ticker(ticker, pipeline)
        if ok:
            results['success'].append(ticker)
        else:
            results['failed'].append(ticker)

        # Rate limiting — don't hammer yfinance/FRED between tickers
        if i < len(tickers) - 1:
            logger.info(f"Waiting {args.delay}s before next ticker...")
            time.sleep(args.delay)

    print(f"\n{'='*60}")
    print(f"Pipeline complete")
    print(f"Success:  {results['success']}")
    print(f"Skipped:  {results['skipped']}")
    print(f"Failed:   {results['failed']}")
    print(f"{'='*60}")

    return 0 if not results['failed'] else 1


if __name__ == "__main__":
    sys.exit(main())
