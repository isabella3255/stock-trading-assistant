#!/usr/bin/env python3
"""
AI Trading Signal System - Main Orchestrator

Runs full pipeline: Data → Features → Models → Options → LLM → Report
"""

import sys
import os
import argparse
import yaml
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import joblib

from modules.data_pipeline import DataPipeline
from modules.feature_engineering import FeatureEngineer
from modules.ml_models import ModelManager
from modules.options_analyzer import OptionsAnalyzer
from modules.llm_synthesis import LLMSynthesizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_ticker(ticker: str, config: dict, use_llm: bool = True) -> dict:
    """Run full analysis pipeline for one ticker"""
    
    logger.info(f"\n{'='*70}")
    logger.info(f"ANALYZING {ticker}")
    logger.info(f"{'='*70}")
    
    try:
        # Phase 2: Data Pipeline
        pipeline = DataPipeline()
        df = pipeline.process_ticker(ticker)
        
        # Phase 3: Feature Engineering
        engineer = FeatureEngineer(df)
        df_featured = engineer.compute_all_features()
        df_featured.to_parquet(f"data/processed/{ticker}_featured.parquet")
        
        # Phase 4: ML Models
        manager = ModelManager(df_featured)
        manager.prepare_data()
        
        # Try to load existing models, otherwise train
        xgb_path = f"models/xgboost/{ticker}.pkl"
        if os.path.exists(xgb_path):
            logger.info(f"Loading existing XGBoost model for {ticker}")
            manager.xgb_model = joblib.load(xgb_path)
            manager.feature_importance = pd.read_csv(f"models/xgboost/{ticker}_importance.csv")
        else:
            manager.train_xgboost()
            manager.save_models(ticker)
        
        # Get prediction
        ml_result = manager.predict()
        
        # Phase 5: Options Analysis
        latest = df_featured.iloc[-1]
        current_price = latest['Close']
        hist_vol = latest['hist_volatility_20']
        
        options_result = {}
        if ml_result['signal'] in ['BULL', 'STRONG_BULL']:
            analyzer = OptionsAnalyzer(ticker, current_price, hist_vol)
            options_result = analyzer.analyze(ml_result['signal'])
        
        # Prepare data package for LLM
        data_package = {
            'price_data': {
                'current_price': current_price,
                'prev_close': df_featured['Close'].iloc[-2],
                'week_52_high': df_featured['Close'].rolling(252).max().iloc[-1],
                'week_52_low': df_featured['Close'].rolling(252).min().iloc[-1],
                'volume_ratio': latest['volume_ratio']
            },
            'ml_signals': ml_result,
            'technical': {
                'rsi': latest['RSI_14'],
                'macd_hist': latest['MACD_hist'],
                'price_vs_ema200': latest['price_vs_ema200'],
                'above_200ema': latest['above_200ema'],
                'atr': latest['ATR_14'],
                'bb_squeeze': latest['BB_squeeze'],
                'bull_flag_setup': latest['bull_flag_setup'],
                'volume_spike': latest['volume_spike'],
                'hist_vol_20': latest['hist_volatility_20']
            },
            'options': options_result,
            'fundamentals': {},
            'headlines': []
        }
        
        # Phase 6: LLM Synthesis
        llm_result = {}
        if use_llm:
            synthesizer = LLMSynthesizer(config)
            llm_result = synthesizer.synthesize(ticker, data_package)
        
        return {
            'ticker': ticker,
            'signal': ml_result['signal'],
            'score': ml_result['final_score'],
            'xgb_prob': ml_result['xgb_prob'],
            'lstm_prob': ml_result['lstm_prob'],
            'options': options_result,
            'llm': llm_result,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Failed to analyze {ticker}: {e}")
        return {'ticker': ticker, 'success': False, 'error': str(e)}


def main():
    """Main orchestrator"""
    parser = argparse.ArgumentParser(description="AI Trading Signal System")
    parser.add_argument("--ticker", type=str, help="Analyze single ticker")
    parser.add_argument("--backtest", action="store_true", help="Run backtesting")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM synthesis")
    args = parser.parse_args()
    
    # Load config
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n" + "="*70)
    print("AI TRADING SIGNAL SYSTEM")
    print("="*70)
    
    # Get tickers
    if args.ticker:
        tickers = [args.ticker.upper()]
    else:
        tickers = config['watchlist']
    
    print(f"\nAnalyzing {len(tickers)} ticker(s): {', '.join(tickers)}")
    print(f"LLM Synthesis: {'Disabled' if args.no_llm else 'Enabled'}")
    print()
    
    # Analyze each ticker
    results = []
    for ticker in tickers:
        result = analyze_ticker(ticker, config, use_llm=not args.no_llm)
        results.append(result)
    
    # Display summary table
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Ticker':<8} {'Signal':<12} {'Score':<8} {'XGB':<8} {'MLP':<8}")
    print("-"*70)
    
    for r in results:
        if r['success']:
            print(f"{r['ticker']:<8} {r['signal']:<12} {r['score']:<8.3f} {r['xgb_prob']:<8.3f} {r['lstm_prob']:<8.3f}")
        else:
            print(f"{r['ticker']:<8} ERROR: {r.get('error', 'Unknown')}")
    
    # Save report
    today = datetime.now().strftime('%Y-%m-%d')
    report_file = f"outputs/signals/daily_report_{today}.csv"
    
    df_results = pd.DataFrame([r for r in results if r['success']])
    if not df_results.empty:
        df_results.to_csv(report_file, index=False)
        print(f"\nReport saved to: {report_file}")
    
    print("\n" + "="*70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
