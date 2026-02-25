#!/usr/bin/env python3
"""
AI Trading Signal System - Main Orchestrator

Runs full pipeline: Data → Features → Models → Options → LLM → Report
"""

import sys
import os
import argparse
import json
import yaml
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
import pandas as pd

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


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy types to native Python so options data serializes cleanly."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def analyze_ticker(ticker: str, config: dict, use_llm: bool = True) -> dict:
    """Run full analysis pipeline for one ticker"""

    logger.info(f"\n{'='*70}")
    logger.info(f"ANALYZING {ticker}")
    logger.info(f"{'='*70}")

    try:
        # Phase 2: Data Pipeline
        pipeline = DataPipeline()
        df = pipeline.process_ticker(ticker)

        # Pull live headlines from pipeline (cached, no extra API quota cost)
        # load_newsapi_headlines() returns a list directly (not a dict)
        try:
            headlines = pipeline.load_newsapi_headlines(ticker)[:5]
        except Exception:
            headlines = []

        # Phase 3: Feature Engineering
        engineer = FeatureEngineer(df)
        df_featured = engineer.compute_all_features()
        df_featured.to_parquet(f"data/processed/{ticker}_featured.parquet")

        # Phase 4: ML Models
        manager = ModelManager(df_featured)
        manager.prepare_data()

        # Model loading priority:
        #   1. Per-ticker model (e.g. TSLA.pkl) — use if it exists
        #   2. Combined model (NVDA.pkl) — trained on all 10 tickers (50,513 rows,
        #      XGB AUC 0.534, MLP AUC 0.537). predict() fills missing ticker_id with 0.
        #   3. Train fresh (last resort — small datasets produce sub-random AUC)
        xgb_path = Path(f"models/xgboost/{ticker}.pkl")
        combined_path = Path("models/xgboost/NVDA.pkl")
        if xgb_path.exists():
            logger.info(f"Loading existing models for {ticker}")
            manager.load_models(ticker)
        elif combined_path.exists():
            logger.info(f"No per-ticker model for {ticker} — loading combined model (NVDA, 50k rows)")
            manager.load_models("NVDA")
        else:
            logger.info(f"No saved models found for {ticker}, training now...")
            manager.train_xgboost()
            manager.train_seq_model()
            manager.save_models(ticker)

        # Get prediction
        ml_result = manager.predict()

        # Phase 5: Options Analysis (only for bullish signals)
        latest = df_featured.iloc[-1]
        current_price = float(latest['Close'])
        hist_vol = float(latest['hist_volatility_20'])

        options_result = {}
        if ml_result['signal'] in ['BULL', 'STRONG_BULL']:
            analyzer = OptionsAnalyzer(ticker, current_price, hist_vol)
            options_result = analyzer.analyze(
                ml_result['signal'],
                signal_score=ml_result['final_score'],   # used for half-Kelly sizing
                days_to_earnings=int(days_to_earnings) if days_to_earnings else None,
            )

        # Build macro dict from latest row — FRED series + Fear & Greed in the featured df
        macro = {
            'vix':              float(latest.get('VIX', 0) or 0),
            'vix_3m':           float(latest.get('VIX3M', 0) or 0),
            'vix_term_slope':   float(latest.get('vix_term_slope', 0) or 0),
            'fear_greed_index': float(latest.get('FEAR_GREED_INDEX', 50) or 50),
            'fed_rate':         float(latest.get('FED_RATE', 0) or 0),
            'treasury_10y':     float(latest.get('TREASURY_10Y', 0) or 0),
            'treasury_2y':      float(latest.get('TREASURY_2Y', 0) or 0),
            'yield_curve':      float(latest.get('TREASURY_10Y', 0) or 0) - float(latest.get('TREASURY_2Y', 0) or 0),
            'cpi':              float(latest.get('CPI', 0) or 0),
            'unemployment':     float(latest.get('UNEMPLOYMENT', 0) or 0),
        }

        # Build fundamentals from last row — PE_RATIO / PEG_RATIO / PROFIT_MARGIN /
        # REVENUE_GROWTH_YOY / EARNINGS_SURPRISE_PCT are merged in from FMP via the data pipeline
        def _safe_float(val):
            try:
                v = float(val)
                return v if v != 0 else None
            except (TypeError, ValueError):
                return None

        fundamentals = {
            'pe_ratio':             _safe_float(latest.get('PE_RATIO')),
            'peg_ratio':            _safe_float(latest.get('PEG_RATIO')),
            'profit_margin':        _safe_float(latest.get('PROFIT_MARGIN')),
            'revenue_growth_yoy':   _safe_float(latest.get('REVENUE_GROWTH_YOY')),
            'earnings_surprise_pct':_safe_float(latest.get('EARNINGS_SURPRISE_PCT')),
            'debt_to_equity':       _safe_float(latest.get('DEBT_TO_EQUITY')),
            'fcf_yield':            _safe_float(latest.get('FCF_YIELD')),
        }

        # Pass days_to_earnings to options analyzer so it can warn on IV crush
        days_to_earnings = _safe_float(latest.get('days_to_earnings'))

        # Prepare data package for LLM
        data_package = {
            'price_data': {
                'current_price': current_price,
                'prev_close':    float(df_featured['Close'].iloc[-2]),
                'week_52_high':  float(df_featured['Close'].rolling(252).max().iloc[-1]),
                'week_52_low':   float(df_featured['Close'].rolling(252).min().iloc[-1]),
                'volume_ratio':  float(latest['volume_ratio']),
            },
            'ml_signals': ml_result,
            'technical': {
                'rsi':            float(latest['RSI_14']),
                'macd_hist':      float(latest['MACD_hist']),
                'price_vs_ema200': float(latest['price_vs_ema200']),
                'above_200ema':   float(latest['above_200ema']),
                'atr':            float(latest['ATR_14']),
                'bb_squeeze':     float(latest['BB_squeeze']),
                'bull_flag_setup': float(latest['bull_flag_setup']),
                'volume_spike':   float(latest['volume_spike']),
                'hist_vol_20':    float(latest['hist_volatility_20']),
            },
            'options':      options_result,
            'macro':        macro,
            'fundamentals': fundamentals,
            'headlines':    headlines,
            # Finnhub composite sentiment score — now wired through to LLM
            'finnhub_sentiment': float(latest.get('NEWS_SENTIMENT', 0) or 0),
        }

        # Phase 6: LLM Synthesis
        llm_result = {}
        if use_llm:
            synthesizer = LLMSynthesizer(config)
            llm_result = synthesizer.synthesize(ticker, data_package)

        # Serialize options to clean JSON (avoids np.float64() repr in CSV)
        options_json = json.dumps(options_result, cls=_NumpyEncoder) if options_result else ''

        # Flatten LLM result — recommendation text is already saved to .txt and .json cache;
        # store only the first 500 chars of the verdict in the CSV for dashboard display.
        llm_verdict = (llm_result.get('recommendation', '') or '')[:500] if llm_result else ''

        return {
            'ticker':       ticker,
            'signal':       ml_result['signal'],
            'score':        ml_result['final_score'],
            'xgb_prob':     ml_result['xgb_prob'],
            'seq_prob':     ml_result['lstm_prob'],
            'hist_vol':     hist_vol,                  # used by portfolio sizing below
            'options_json': options_json,              # clean JSON string, parseable by dashboard
            'llm_verdict':  llm_verdict,               # first 500 chars of LLM recommendation
            'success':      True,
        }

    except Exception as e:
        logger.error(f"Failed to analyze {ticker}: {e}", exc_info=True)
        return {'ticker': ticker, 'success': False, 'error': str(e)}


def delete_per_ticker_models():
    """
    Delete all per-ticker model files, keeping only the combined NVDA model.
    This forces main.py to use the 50,513-row combined model for all tickers
    instead of stale per-ticker files with sub-random AUC.
    """
    tickers_to_remove = ['QQQ', 'SPY', 'AAPL', 'MSFT', 'TSLA', 'AMD', 'META', 'AMZN', 'GOOGL']
    patterns = [
        ("models/xgboost", ["{t}.pkl", "{t}_importance.csv"]),
        ("models/lstm",    ["{t}_seq.pkl", "{t}_scaler.pkl", "{t}_norm_stats.pkl"]),
    ]
    deleted = []
    for folder, templates in patterns:
        for ticker in tickers_to_remove:
            for tmpl in templates:
                path = Path(folder) / tmpl.replace("{t}", ticker)
                if path.exists():
                    path.unlink()
                    deleted.append(str(path))

    if deleted:
        logger.info(f"Deleted {len(deleted)} stale per-ticker model file(s):")
        for f in deleted:
            logger.info(f"  removed: {f}")
    else:
        logger.info("No stale per-ticker model files found to delete.")


def main():
    """Main orchestrator"""
    parser = argparse.ArgumentParser(description="AI Trading Signal System")
    parser.add_argument("--ticker", type=str, help="Analyze single ticker")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM synthesis")
    parser.add_argument(
        "--delete-models",
        action="store_true",
        help="Delete all per-ticker model files before running (keeps combined NVDA model)",
    )
    args = parser.parse_args()

    # Delete stale per-ticker models if requested
    if args.delete_models:
        logger.info("--delete-models flag set: removing stale per-ticker model files...")
        delete_per_ticker_models()

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
    print(f"\n{'Ticker':<8} {'Signal':<14} {'Score':<8} {'XGB':<8} {'SEQ':<8}")
    print("-"*46)

    for r in results:
        if r['success']:
            print(f"{r['ticker']:<8} {r['signal']:<14} {r['score']:<8.3f} {r['xgb_prob']:<8.3f} {r['seq_prob']:<8.3f}")
        else:
            print(f"{r['ticker']:<8} ERROR: {r.get('error', 'Unknown')}")

    # ── Portfolio-level risk summary ─────────────────────────────────────────
    # Groups known-correlated tickers and warns when multiple correlated
    # signals fire simultaneously — independent signals that look correlated
    # should be sized down proportionally.
    SECTOR_GROUPS = {
        'Tech Mega-Cap': {'NVDA', 'AMD', 'MSFT', 'AAPL', 'GOOGL', 'META', 'AMZN'},
        'Crypto-Adjacent': {'COIN', 'MSTR', 'RIOT'},
        'Broad Market ETF': {'SPY', 'QQQ', 'VOO', 'SCHG', 'SPMO'},
        'Energy': {'XLE', 'ENPH'},
        'Gold/Silver': {'GLD', 'SLV'},
        'Fintech/Growth': {'SOFI', 'UBER', 'PLTR', 'SHOP'},
    }

    successful = [r for r in results if r.get('success')]
    bull_set = {r['ticker'] for r in successful if r.get('signal') in ('BULL', 'STRONG_BULL')}
    bear_set = {r['ticker'] for r in successful if r.get('signal') in ('BEAR', 'STRONG_BEAR')}

    print("\n" + "="*70)
    print("PORTFOLIO RISK ANALYSIS")
    print("="*70)

    if not bull_set and not bear_set:
        print("\n  No actionable signals today — all tickers NEUTRAL.")
    else:
        # Concentration warnings
        warnings_issued = False
        for group_name, group_tickers in SECTOR_GROUPS.items():
            bull_in_group = bull_set & group_tickers
            if len(bull_in_group) >= 3:
                print(f"\n  ⚠️  HIGH CONCENTRATION — {group_name}: {len(bull_in_group)} tickers BULLISH")
                print(f"     {', '.join(sorted(bull_in_group))}")
                print(f"     These signals are correlated — treat as ONE position, not {len(bull_in_group)}.")
                warnings_issued = True

        if not warnings_issued:
            print("\n  ✅ No sector concentration warnings.")

        # Vol-adjusted position sizing for actionable signals
        print(f"\n  {'Ticker':<8} {'Signal':<14} {'HV-20':<8} {'Suggested Size'}")
        print(f"  {'-'*50}")
        for r in sorted(successful, key=lambda x: x.get('score', 0), reverse=True):
            sig = r.get('signal', '')
            if sig not in ('BULL', 'STRONG_BULL', 'BEAR', 'STRONG_BEAR'):
                continue
            hv = r.get('hist_vol', 0.0) or 0.02  # annualized HV-20
            ann_vol = hv * (252 ** 0.5) if hv < 1 else hv  # already annualized in feature eng
            # Vol-adjusted equal-risk sizing: target 1% portfolio daily vol per position
            # size = (target_daily_vol / asset_daily_vol) capped at 5%
            daily_vol = ann_vol / (252 ** 0.5)
            raw_size = 0.01 / max(daily_vol, 0.001) * 100  # as % of portfolio
            size_pct = round(min(5.0, max(0.5, raw_size)), 1)
            direction = "LONG" if sig in ('BULL', 'STRONG_BULL') else "SHORT/PUT"
            print(f"  {r['ticker']:<8} {sig:<14} {ann_vol:<8.1%} ≤{size_pct}% ({direction})")

    # Save report (hist_vol column used for risk display, not needed in CSV)
    today = datetime.now().strftime('%Y-%m-%d')
    report_file = f"outputs/signals/daily_report_{today}.csv"
    Path("outputs/signals").mkdir(parents=True, exist_ok=True)

    csv_rows = []
    for r in results:
        if r.get('success'):
            csv_rows.append({k: v for k, v in r.items() if k != 'hist_vol'})

    df_results = pd.DataFrame(csv_rows)
    if not df_results.empty:
        df_results.to_csv(report_file, index=False)
        print(f"\nReport saved to: {report_file}")

    print("\n" + "="*70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
