"""
Data Pipeline Module

Loads and merges data from 5 sources:
1. yfinance - Recent data + options chains + earnings dates
2. FRED - Macro indicators (VIX, Fed Rate, Treasury yields, CPI, Unemployment,
           plus VIX term structure: VXST/VIX9D, VXMT/VIX3M)
3. Financial Modeling Prep (FMP) - Fundamentals (replaces Alpha Vantage)
   Provides PE, profit margin, revenue growth YoY, earnings surprise %, debt/equity, FCF yield
4. Finnhub - News sentiment & insider trading
5. NewsAPI - Headlines
Plus: CNN Fear & Greed Index (free, single HTTP call)
"""

import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, List

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPipeline:
    """Loads and merges data from multiple sources"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize with config"""
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.cache_dir = Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.fred_api_key = os.getenv('FRED_API_KEY')
        self.fmp_api_key  = os.getenv('FMP_API_KEY')

    def load_yfinance_data(self, ticker: str) -> pd.DataFrame:
        """Load recent data from yfinance (2 years of OHLCV)"""
        try:
            logger.info(f"Fetching yfinance data for {ticker}")
            stock = yf.Ticker(ticker)
            period = self.config['data_pipeline'].get('recent_years', 2)
            df = stock.history(period=f"{period}y")
            if df.empty:
                logger.warning(f"No yfinance data for {ticker}")
                return pd.DataFrame()
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df.index.name = 'Date'
            return df
        except Exception as e:
            logger.warning(f"yfinance failed for {ticker}: {e}")
            return pd.DataFrame()

    def merge_price_data(self, ticker: str) -> pd.DataFrame:
        """Load yfinance price data (single source — Kaggle removed)"""
        df = self.load_yfinance_data(ticker)
        if df.empty:
            raise ValueError(f"No price data found for {ticker}")
        # Ensure tz-naive index for consistent merging with macro data
        df.index = pd.to_datetime(df.index).tz_localize(None)
        logger.info(f"Loaded {len(df)} rows for {ticker} "
                    f"({df.index.min().date()} to {df.index.max().date()})")
        return df
    
    def load_fred_macro_data(self) -> pd.DataFrame:
        """Load macro indicators from FRED"""
        cache_file = self.cache_dir / "fred_macro.json"
        
        # Check cache (24hr TTL)
        if cache_file.exists():
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age < 24 * 3600:
                logger.info("Loading FRED data from cache")
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
                if 'index' in df.columns:
                    df.index = pd.to_datetime(df['index'])
                    df = df.drop('index', axis=1)
                return df
        
        logger.info("Fetching FRED macro data")

        if not self.fred_api_key:
            logger.warning("FRED_API_KEY not set, skipping macro data")
            return pd.DataFrame()

        indicators = {
            'VIX':          'VIXCLS',
            'VIX3M':        'VXVCLS',    # CBOE 3-Month Volatility Index (VIX3M proxy)
            'FED_RATE':     'DFF',
            'TREASURY_10Y': 'DGS10',
            'TREASURY_2Y':  'DGS2',
            'CPI':          'CPIAUCSL',
            'UNEMPLOYMENT': 'UNRATE',
        }

        macro_data = {}
        start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')

        for name, series_id in indicators.items():
            try:
                # Use requests directly to avoid fredapi's SSL cert issues on macOS
                url = (
                    f"https://api.stlouisfed.org/fred/series/observations"
                    f"?series_id={series_id}&api_key={self.fred_api_key}"
                    f"&file_type=json&observation_start={start_date}"
                )
                response = requests.get(url, timeout=15)
                obs = response.json().get('observations', [])
                # Filter out missing values (FRED uses '.' for N/A)
                valid = {o['date']: float(o['value']) for o in obs if o['value'] != '.'}
                if valid:
                    series = pd.Series(valid)
                    series.index = pd.to_datetime(series.index)
                    macro_data[name] = series
                    logger.info(f"FRED {name}: {len(series)} observations")
            except Exception as e:
                logger.warning(f"Failed to fetch FRED {name}: {e}")
        
        if not macro_data:
            return pd.DataFrame()

        df = pd.DataFrame(macro_data)

        # Compute VIX term structure slope: VIX (30-day) - VIX3M (90-day)
        # Negative = normal contango (calm — near-term vol < long-term), Positive = inverted (fear spike)
        if 'VIX' in df.columns and 'VIX3M' in df.columns:
            df['vix_term_slope'] = df['VIX'] - df['VIX3M']

        # Cache it — convert Timestamp index to strings so json.dump doesn't crash
        cache_data = df.copy()
        cache_data.index = cache_data.index.astype(str)
        with open(cache_file, 'w') as f:
            json.dump(cache_data.to_dict(), f)

        return df
    
    def load_fmp_fundamentals(self, ticker: str) -> Dict:
        """Load fundamentals from Financial Modeling Prep (FMP).

        Replaces Alpha Vantage — FMP free tier gives 250 req/day vs Alpha Vantage's 25.
        Returns: PE_RATIO, PEG_RATIO, PROFIT_MARGIN, REVENUE_GROWTH_YOY,
                 EARNINGS_SURPRISE_PCT, DEBT_TO_EQUITY, FCF_YIELD
        """
        cache_file = self.cache_dir / f"fmp_{ticker}.json"

        # Check cache (24h TTL)
        if cache_file.exists():
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age < 24 * 3600:
                with open(cache_file, 'r') as f:
                    return json.load(f)

        api_key = self.fmp_api_key
        if not api_key:
            logger.warning(f"FMP_API_KEY not set — skipping fundamentals for {ticker}. "
                           "Get a free key at https://financialmodelingprep.com/developer/docs/")
            return {}

        result = {}
        try:
            logger.info(f"Fetching FMP fundamentals for {ticker}")

            # 1. Company profile — PE, beta, market cap, description
            url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={api_key}"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    p = data[0]
                    result['PE_RATIO']      = float(p.get('pe') or 0)
                    result['PROFIT_MARGIN'] = float(p.get('netProfitMargin') or 0) / 100  # FMP returns %
                    result['DEBT_TO_EQUITY'] = float(p.get('debtToEquity') or 0)

            # 2. Income statement — revenue growth YoY
            url = (f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}"
                   f"?limit=2&apikey={api_key}")
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                stmts = resp.json()
                if len(stmts) >= 2:
                    rev_now  = float(stmts[0].get('revenue') or 0)
                    rev_prev = float(stmts[1].get('revenue') or 1)
                    if rev_prev != 0:
                        result['REVENUE_GROWTH_YOY'] = (rev_now - rev_prev) / abs(rev_prev)

            # 3. Earnings surprises — most recent quarter
            url = (f"https://financialmodelingprep.com/api/v3/earnings-surprises/{ticker}"
                   f"?apikey={api_key}")
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                surprises = resp.json()
                if surprises:
                    s = surprises[0]
                    actual   = float(s.get('actualEarningResult') or 0)
                    estimate = float(s.get('estimatedEarning') or 0)
                    if estimate != 0:
                        result['EARNINGS_SURPRISE_PCT'] = (actual - estimate) / abs(estimate)

            # 4. Key metrics — PEG ratio, FCF yield
            url = (f"https://financialmodelingprep.com/api/v3/key-metrics/{ticker}"
                   f"?limit=1&apikey={api_key}")
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                metrics = resp.json()
                if metrics:
                    m = metrics[0]
                    result['PEG_RATIO']  = float(m.get('pegRatio') or 0)
                    result['FCF_YIELD']  = float(m.get('freeCashFlowYield') or 0)

            # Cache result
            if result:
                with open(cache_file, 'w') as f:
                    json.dump(result, f)
                logger.info(f"FMP fundamentals cached for {ticker}: {list(result.keys())}")

        except Exception as e:
            logger.warning(f"FMP fundamentals failed for {ticker}: {e}")

        return result

    def load_fear_greed_index(self) -> float:
        """Load CNN Fear & Greed Index (0 = Extreme Fear, 100 = Extreme Greed).

        Uses CNN's public data endpoint. Cached for 24h since it's a single daily reading.
        Free, no API key required.
        """
        cache_file = self.cache_dir / "fear_greed.json"

        # Check cache (24h TTL — Fear & Greed updates once daily)
        if cache_file.exists():
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age < 24 * 3600:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                return float(data.get('score', 50))

        try:
            url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
            resp = requests.get(url, timeout=10,
                                headers={'User-Agent': 'Mozilla/5.0'})
            if resp.status_code == 200:
                data = resp.json()
                score = float(data.get('fear_and_greed', {}).get('score', 50))
                with open(cache_file, 'w') as f:
                    json.dump({'score': score}, f)
                logger.info(f"Fear & Greed Index: {score:.1f}")
                return score
        except Exception as e:
            logger.warning(f"Fear & Greed fetch failed: {e}")

        return 50.0  # neutral fallback
    
    def load_finnhub_data(self, ticker: str) -> Dict:
        """Load sentiment and insider trading from Finnhub"""
        cache_file = self.cache_dir / f"finnhub_{ticker}.json"
        
        # Check cache
        if cache_file.exists():
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age < 24 * 3600:
                with open(cache_file, 'r') as f:
                    return json.load(f)
        
        api_key = os.getenv('FINNHUB_API_KEY')
        if not api_key:
            return {}
        
        try:
            logger.info(f"Fetching Finnhub data for {ticker}")
            
            data = {}
            
            # Get news sentiment
            url = f"https://finnhub.io/api/v1/news-sentiment?symbol={ticker}&token={api_key}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data['sentiment'] = response.json()
            
            # Get insider transactions
            url = f"https://finnhub.io/api/v1/stock/insider-transactions?symbol={ticker}&token={api_key}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data['insider'] = response.json()
            
            # Cache it
            if data:
                with open(cache_file, 'w') as f:
                    json.dump(data, f)
            
            return data
        except Exception as e:
            logger.warning(f"Finnhub failed: {e}")
            return {}
    
    def load_newsapi_headlines(self, ticker: str) -> List[str]:
        """Load recent headlines from NewsAPI"""
        cache_file = self.cache_dir / f"news_{ticker}.json"
        
        # Check cache
        if cache_file.exists():
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age < 24 * 3600:
                with open(cache_file, 'r') as f:
                    return json.load(f)
        
        api_key = os.getenv('NEWS_API_KEY')
        if not api_key:
            return []
        
        try:
            logger.info(f"Fetching NewsAPI headlines for {ticker}")
            
            # Get last 7 days
            from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            url = f"https://newsapi.org/v2/everything?q={ticker}&from={from_date}&sortBy=publishedAt&apiKey={api_key}"
            
            response = requests.get(url, timeout=10)
            data = response.json()
            
            headlines = []
            if data.get('status') == 'ok':
                articles = data.get('articles', [])
                headlines = [article['title'] for article in articles[:10]]
            
            # Cache it
            with open(cache_file, 'w') as f:
                json.dump(headlines, f)
            
            return headlines
        except Exception as e:
            logger.warning(f"NewsAPI failed: {e}")
            return []
    
    def get_yfinance_metadata(self, ticker: str) -> Dict:
        """Get earnings dates, sector, etc from yfinance.

        Also computes days_to_earnings — passed to options_analyzer to warn on IV crush risk.
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            metadata = {
                'sector':        info.get('sector', 'Unknown'),
                'market_cap':    info.get('marketCap', 0),
                'beta':          info.get('beta', 1.0),
                'forward_pe':    info.get('forwardPE', None),
                'week_52_high':  info.get('fiftyTwoWeekHigh', None),
                'week_52_low':   info.get('fiftyTwoWeekLow', None),
                'days_to_earnings': None,
            }

            # Extract next earnings date and compute days_to_earnings
            try:
                calendar = stock.calendar
                if calendar is not None and 'Earnings Date' in calendar:
                    next_earnings = calendar['Earnings Date'][0]
                    metadata['next_earnings_date'] = next_earnings
                    # Compute days from today
                    next_dt = pd.to_datetime(next_earnings)
                    days = (next_dt.tz_localize(None) - pd.Timestamp.now()).days
                    metadata['days_to_earnings'] = max(0, int(days))
                    logger.info(f"{ticker} next earnings: {next_earnings} ({days} days)")
            except Exception:
                metadata['next_earnings_date'] = None

            return metadata
        except Exception as e:
            logger.warning(f"yfinance metadata failed: {e}")
            return {}
    
    def process_ticker(self, ticker: str) -> pd.DataFrame:
        """Main pipeline: load and merge all data sources for a ticker"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {ticker}")
        logger.info(f"{'='*60}")

        # 1. Load price data (yfinance only — Kaggle removed)
        df = self.merge_price_data(ticker)

        # 2. Load FRED macro data (VIX, rates, inflation, VIX term structure)
        macro_df = self.load_fred_macro_data()
        if not macro_df.empty:
            # Ensure both indices are timezone-naive datetime
            df.index    = pd.to_datetime(df.index).tz_localize(None)
            macro_df.index = pd.to_datetime(macro_df.index).tz_localize(None)
            macro_df = macro_df.reindex(df.index, method='ffill')
            df = pd.concat([df, macro_df], axis=1)
        else:
            logger.warning("No macro data available, skipping")

        # 3. Load Fear & Greed Index (free, single HTTP call)
        fear_greed = self.load_fear_greed_index()
        df['FEAR_GREED_INDEX'] = fear_greed   # scalar broadcast across all rows

        # 4. Load FMP fundamentals (replaces Alpha Vantage — 250 req/day vs 25)
        fmp_data = self.load_fmp_fundamentals(ticker)
        if fmp_data:
            df['PE_RATIO']             = float(fmp_data.get('PE_RATIO', 0) or 0)
            df['PEG_RATIO']            = float(fmp_data.get('PEG_RATIO', 0) or 0)
            df['PROFIT_MARGIN']        = float(fmp_data.get('PROFIT_MARGIN', 0) or 0)
            df['REVENUE_GROWTH_YOY']   = float(fmp_data.get('REVENUE_GROWTH_YOY', 0) or 0)
            df['EARNINGS_SURPRISE_PCT'] = float(fmp_data.get('EARNINGS_SURPRISE_PCT', 0) or 0)
            df['DEBT_TO_EQUITY']       = float(fmp_data.get('DEBT_TO_EQUITY', 0) or 0)
            df['FCF_YIELD']            = float(fmp_data.get('FCF_YIELD', 0) or 0)

        # 5. Load Finnhub sentiment (bullish - bearish %, range -1 to +1)
        #    Previously fetched but never wired into the dataframe or LLM — fixed here.
        finnhub_data = self.load_finnhub_data(ticker)
        if finnhub_data.get('sentiment'):
            sent_data = finnhub_data['sentiment'].get('sentiment', {})
            bull_pct  = float(sent_data.get('bullishPercent', 0) or 0)
            bear_pct  = float(sent_data.get('bearishPercent', 0) or 0)
            df['NEWS_SENTIMENT'] = bull_pct - bear_pct  # -1 bearish … +1 bullish

        # 6. Store metadata and earnings date
        metadata = self.get_yfinance_metadata(ticker)
        headlines = self.load_newsapi_headlines(ticker)

        # Inject days_to_earnings as a numeric column for feature engineering
        days_to_earn = metadata.get('days_to_earnings')
        if days_to_earn is not None:
            df['days_to_earnings'] = int(days_to_earn)
        else:
            df['days_to_earnings'] = 999   # sentinel: no upcoming earnings known

        # Fill missing values
        df = df.ffill().bfill()

        # Save to parquet
        output_file = f"data/processed/{ticker}.parquet"
        df.to_parquet(output_file)
        logger.info(f"Saved {len(df)} rows to {output_file}")

        # Save metadata separately (headlines for LLM context)
        metadata['headlines'] = headlines
        metadata_file = f"data/processed/{ticker}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, default=str)

        return df


if __name__ == "__main__":
    # Test with one ticker
    pipeline = DataPipeline()
    df = pipeline.process_ticker("NVDA")
    print(f"\nLoaded {len(df)} rows")
    new_cols = [c for c in df.columns if c in (
        'VIX9D', 'VIX3M', 'vix_term_slope', 'FEAR_GREED_INDEX',
        'REVENUE_GROWTH_YOY', 'EARNINGS_SURPRISE_PCT', 'DEBT_TO_EQUITY',
        'FCF_YIELD', 'NEWS_SENTIMENT', 'days_to_earnings',
    )]
    print(f"New data columns present: {new_cols}")
    print(f"\nLast 5 rows (key columns):")
    show = ['Close'] + new_cols
    print(df[[c for c in show if c in df.columns]].tail())
