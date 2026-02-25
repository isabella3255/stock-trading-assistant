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
import pytz
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

    def load_yfinance_data(self, ticker: str, force_refresh: bool = False) -> pd.DataFrame:
        """Load recent data from yfinance (2 years of OHLCV) with 24-hour caching.
        
        Caching prevents signal instability caused by refreshing data every 5 minutes.
        Data is cached for 24 hours or until after market close (4 PM ET), whichever is longer.
        
        Args:
            ticker: Stock ticker symbol
            force_refresh: If True, bypass cache and fetch fresh data
        """
        cache_file = self.cache_dir / f"yfinance_{ticker}.parquet"
        
        # Check cache (24h TTL or until after market close)
        if not force_refresh and cache_file.exists():
            cache_age = time.time() - cache_file.stat().st_mtime
            cache_age_hours = cache_age / 3600
            
            # Check if we're past market close today (4 PM ET = 9 PM UTC)
            now_et = datetime.now(tz=pytz.timezone('America/New_York'))
            market_close_today = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
            
            # If cache is <24h old OR we haven't crossed market close yet today, use cache
            if cache_age_hours < 24:
                logger.info(f"Using cached yfinance data for {ticker} ({cache_age_hours:.1f}h old)")
                return pd.read_parquet(cache_file)
            elif now_et < market_close_today:
                # Before market close, keep using today's cache even if >24h
                logger.info(f"Using cached yfinance data for {ticker} (before market close)")
                return pd.read_parquet(cache_file)
        
        try:
            logger.info(f"Fetching fresh yfinance data for {ticker}")
            stock = yf.Ticker(ticker)
            period = self.config['data_pipeline'].get('recent_years', 2)
            df = stock.history(period=f"{period}y")
            if df.empty:
                logger.warning(f"No yfinance data for {ticker}")
                return pd.DataFrame()
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df.index.name = 'Date'
            
            # Cache result
            df.to_parquet(cache_file)
            logger.info(f"Cached fresh yfinance data for {ticker}: {len(df)} rows")
            
            return df
        except Exception as e:
            logger.warning(f"yfinance failed for {ticker}: {e}")
            # If fetch fails and cache exists, return stale cache as fallback
            if cache_file.exists():
                logger.info(f"Returning stale cache for {ticker} after fetch failure")
                return pd.read_parquet(cache_file)
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
    
    def load_fmp_fundamentals(self, ticker: str) -> pd.DataFrame:
        """Load quarterly fundamentals time-series from Financial Modeling Prep (FMP).

        Replaces Alpha Vantage — FMP free tier gives 250 req/day vs Alpha Vantage's 25.
        Returns DataFrame with quarterly data indexed by date:
            PE_RATIO, PEG_RATIO, PROFIT_MARGIN, REVENUE_GROWTH_YOY,
            EARNINGS_SURPRISE_PCT, DEBT_TO_EQUITY, FCF_YIELD
        """
        cache_file = self.cache_dir / f"fmp_ts_{ticker}.parquet"

        # Check cache (24h TTL)
        if cache_file.exists():
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age < 24 * 3600:
                return pd.read_parquet(cache_file)

        api_key = self.fmp_api_key
        if not api_key:
            logger.warning(f"FMP_API_KEY not set — skipping fundamentals for {ticker}. "
                           "Get a free key at https://financialmodelingprep.com/developer/docs/")
            return pd.DataFrame()

        try:
            logger.info(f"Fetching FMP quarterly fundamentals for {ticker}")

            # 1. Key metrics (quarterly, last 40 quarters = 10 years)
            url = (f"https://financialmodelingprep.com/api/v3/key-metrics/{ticker}"
                   f"?period=quarter&limit=40&apikey={api_key}")
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                logger.warning(f"FMP key metrics failed for {ticker}: {resp.status_code}")
                return pd.DataFrame()
            
            metrics = resp.json()
            if not metrics:
                logger.warning(f"No FMP data returned for {ticker}")
                return pd.DataFrame()

            # Build DataFrame from quarterly metrics
            records = []
            for m in metrics:
                date = m.get('date')
                if not date:
                    continue
                records.append({
                    'date': pd.to_datetime(date),
                    'PE_RATIO': float(m.get('peRatio') or 0),
                    'PEG_RATIO': float(m.get('pegRatio') or 0),
                    'DEBT_TO_EQUITY': float(m.get('debtToEquity') or 0),
                    'FCF_YIELD': float(m.get('freeCashFlowYield') or 0)
                })

            # 2. Income statement (quarterly) — for profit margin and revenue growth
            url = (f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}"
                   f"?period=quarter&limit=40&apikey={api_key}")
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                stmts = resp.json()
                income_data = {}
                for stmt in stmts:
                    date = stmt.get('date')
                    if not date:
                        continue
                    revenue = float(stmt.get('revenue') or 0)
                    net_income = float(stmt.get('netIncome') or 0)
                    margin = (net_income / revenue) if revenue else 0
                    income_data[pd.to_datetime(date)] = {
                        'PROFIT_MARGIN': margin,
                        'revenue': revenue
                    }
                
                # Add profit margins to records
                for rec in records:
                    if rec['date'] in income_data:
                        rec['PROFIT_MARGIN'] = income_data[rec['date']]['PROFIT_MARGIN']
                
                # Calculate revenue growth YoY (Q vs Q-4)
                sorted_dates = sorted(income_data.keys())
                for i, date in enumerate(sorted_dates):
                    if i >= 4:
                        prev_date = sorted_dates[i - 4]
                        rev_now = income_data[date]['revenue']
                        rev_prev = income_data[prev_date]['revenue']
                        growth = ((rev_now - rev_prev) / abs(rev_prev)) if rev_prev else 0
                        for rec in records:
                            if rec['date'] == date:
                                rec['REVENUE_GROWTH_YOY'] = growth

            # 3. Earnings surprises (quarterly)
            url = (f"https://financialmodelingprep.com/api/v3/earnings-surprises/{ticker}"
                   f"?apikey={api_key}")
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                surprises = resp.json()
                surprise_data = {}
                for s in surprises[:40]:
                    date = s.get('date')
                    if not date:
                        continue
                    actual = float(s.get('actualEarningResult') or 0)
                    estimate = float(s.get('estimatedEarning') or 0)
                    surprise_pct = ((actual - estimate) / abs(estimate)) if estimate else 0
                    surprise_data[pd.to_datetime(date)] = surprise_pct
                
                # Add to records
                for rec in records:
                    if rec['date'] in surprise_data:
                        rec['EARNINGS_SURPRISE_PCT'] = surprise_data[rec['date']]

            if not records:
                logger.warning(f"No fundamental records built for {ticker}")
                return pd.DataFrame()

            # Create DataFrame, sort by date, set index
            df = pd.DataFrame(records).sort_values('date')
            df = df.set_index('date')
            
            # Fill missing columns with 0
            for col in ['PE_RATIO', 'PEG_RATIO', 'PROFIT_MARGIN', 'REVENUE_GROWTH_YOY',
                        'EARNINGS_SURPRISE_PCT', 'DEBT_TO_EQUITY', 'FCF_YIELD']:
                if col not in df.columns:
                    df[col] = 0.0
                else:
                    df[col] = df[col].fillna(0.0)

            # Cache result
            df.to_parquet(cache_file)
            logger.info(f"FMP fundamentals cached for {ticker}: {len(df)} quarters")

            return df

        except Exception as e:
            logger.warning(f"FMP fundamentals failed for {ticker}: {e}")
            return pd.DataFrame()

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
    
    def load_finnhub_sentiment_timeseries(self, ticker: str) -> pd.DataFrame:
        """Load daily news sentiment time-series from Finnhub.
        
        Fetches company news for last 30 days and aggregates daily sentiment.
        Returns DataFrame with date index and NEWS_SENTIMENT column (-1 to +1).
        """
        cache_file = self.cache_dir / f"finnhub_sentiment_ts_{ticker}.parquet"
        
        # Check cache (24h TTL)
        if cache_file.exists():
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age < 24 * 3600:
                return pd.read_parquet(cache_file)
        
        api_key = os.getenv('FINNHUB_API_KEY')
        if not api_key:
            logger.warning("FINNHUB_API_KEY not set — skipping sentiment time-series")
            return pd.DataFrame()
        
        try:
            logger.info(f"Fetching Finnhub daily sentiment for {ticker}")
            
            # Get company news for last 30 days
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            
            url = (f"https://finnhub.io/api/v1/company-news?"
                   f"symbol={ticker}&from={start_date}&to={end_date}&token={api_key}")
            response = requests.get(url, timeout=15)
            
            if response.status_code != 200:
                logger.warning(f"Finnhub sentiment failed for {ticker}: {response.status_code}")
                return pd.DataFrame()
            
            articles = response.json()
            if not articles:
                logger.warning(f"No Finnhub articles for {ticker}")
                return pd.DataFrame()
            
            # Group articles by date and calculate daily sentiment
            # Finnhub provides sentiment scores with each article
            daily_sentiments = {}
            for article in articles:
                if 'datetime' not in article or 'sentiment' not in article:
                    continue
                
                # Convert Unix timestamp to date
                date = pd.to_datetime(article['datetime'], unit='s').date()
                sentiment = float(article.get('sentiment', 0))
                
                if date not in daily_sentiments:
                    daily_sentiments[date] = []
                daily_sentiments[date].append(sentiment)
            
            # Average sentiment per day
            records = []
            for date, sentiments in daily_sentiments.items():
                avg_sentiment = np.mean(sentiments)
                records.append({
                    'date': pd.to_datetime(date),
                    'NEWS_SENTIMENT': avg_sentiment,
                    'article_count': len(sentiments)
                })
            
            if not records:
                logger.warning(f"No sentiment records for {ticker}")
                return pd.DataFrame()
            
            # Create DataFrame
            df = pd.DataFrame(records).sort_values('date')
            df = df.set_index('date')
            
            # Add sentiment momentum (7-day change)
            df['SENTIMENT_MOMENTUM_7D'] = df['NEWS_SENTIMENT'].diff(7)
            
            # Cache result
            df.to_parquet(cache_file)
            logger.info(f"Finnhub sentiment cached for {ticker}: {len(df)} days")
            
            return df
            
        except Exception as e:
            logger.warning(f"Finnhub sentiment time-series failed for {ticker}: {e}")
            return pd.DataFrame()
    
    def load_finnhub_data(self, ticker: str) -> Dict:
        """Load aggregate sentiment and insider trading from Finnhub (legacy, kept for compatibility)"""
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
            logger.info(f"Fetching Finnhub aggregate data for {ticker}")
            
            data = {}
            
            # Get news sentiment (aggregate)
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

        # 4. Load FMP fundamentals time-series (replaces Alpha Vantage — 250 req/day vs 25)
        fmp_df = self.load_fmp_fundamentals(ticker)
        if not fmp_df.empty:
            # Ensure both indices are timezone-naive datetime
            fmp_df.index = pd.to_datetime(fmp_df.index).tz_localize(None)
            
            # Merge on index (left join to keep all price data)
            # Forward-fill fundamentals but limit to 95 days (1 quarter + margin)
            # This prevents stale data from leaking across years
            fmp_df = fmp_df.reindex(df.index, method='ffill', limit=95)
            
            # Merge into main dataframe
            for col in fmp_df.columns:
                df[col] = fmp_df[col]
            
            # Fill any remaining NaNs with 0 (for dates before first fundamental data point)
            fundamental_cols = ['PE_RATIO', 'PEG_RATIO', 'PROFIT_MARGIN', 'REVENUE_GROWTH_YOY',
                                'EARNINGS_SURPRISE_PCT', 'DEBT_TO_EQUITY', 'FCF_YIELD']
            for col in fundamental_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(0.0)
        else:
            # No fundamentals available, fill with zeros
            logger.warning(f"No FMP fundamentals for {ticker}, filling with zeros")
            for col in ['PE_RATIO', 'PEG_RATIO', 'PROFIT_MARGIN', 'REVENUE_GROWTH_YOY',
                        'EARNINGS_SURPRISE_PCT', 'DEBT_TO_EQUITY', 'FCF_YIELD']:
                df[col] = 0.0

        # 5. Load Finnhub sentiment time-series (daily sentiment + momentum)
        sentiment_df = self.load_finnhub_sentiment_timeseries(ticker)
        if not sentiment_df.empty:
            # Ensure both indices are timezone-naive datetime
            sentiment_df.index = pd.to_datetime(sentiment_df.index).tz_localize(None)
            
            # Merge sentiment data (reindex to match price data, forward-fill up to 7 days)
            sentiment_df = sentiment_df.reindex(df.index, method='ffill', limit=7)
            
            # Add sentiment columns
            df['NEWS_SENTIMENT'] = sentiment_df['NEWS_SENTIMENT'].fillna(0.0)
            df['SENTIMENT_MOMENTUM_7D'] = sentiment_df['SENTIMENT_MOMENTUM_7D'].fillna(0.0)
            
            logger.info(f"Merged sentiment time-series for {ticker}")
        else:
            # Fallback: try to load aggregate sentiment (legacy)
            finnhub_data = self.load_finnhub_data(ticker)
            if finnhub_data.get('sentiment'):
                sent_data = finnhub_data['sentiment'].get('sentiment', {})
                bull_pct  = float(sent_data.get('bullishPercent', 0) or 0)
                bear_pct  = float(sent_data.get('bearishPercent', 0) or 0)
                df['NEWS_SENTIMENT'] = bull_pct - bear_pct  # -1 bearish … +1 bullish
                df['SENTIMENT_MOMENTUM_7D'] = 0.0
            else:
                df['NEWS_SENTIMENT'] = 0.0
                df['SENTIMENT_MOMENTUM_7D'] = 0.0

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
