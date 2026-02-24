"""
Data Pipeline Module

Loads and merges data from 6 sources:
1. Kaggle - Historical OHLCV (20+ years)
2. yfinance - Recent data + options chains
3. FRED - Macro indicators
4. Alpha Vantage - Fundamentals
5. Finnhub - Sentiment & insider trading
6. NewsAPI - Headlines
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
from fredapi import Fred
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
        
        # API clients
        self.fred = Fred(api_key=os.getenv('FRED_API_KEY'))
        
    def load_kaggle_data(self, ticker: str) -> pd.DataFrame:
        """Load historical data from Kaggle CSV files"""
        kaggle_path = self.config['data_pipeline']['kaggle_data_path']
        
        # Try different file formats and locations
        possible_files = [
            f"{kaggle_path}/Stocks/{ticker.lower()}.us.txt",
            f"{kaggle_path}/Stocks/{ticker.upper()}.us.txt",
            f"{kaggle_path}/ETFs/{ticker.lower()}.us.txt",
            f"{kaggle_path}/ETFs/{ticker.upper()}.us.txt",
            f"{kaggle_path}/{ticker.lower()}.us.txt",
            f"{kaggle_path}/{ticker.upper()}.us.txt",
        ]
        
        for file_path in possible_files:
            if os.path.exists(file_path):
                logger.info(f"Loading Kaggle data from {file_path}")
                df = pd.read_csv(file_path)
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                return df
        
        logger.warning(f"No Kaggle file found for {ticker}")
        return pd.DataFrame()
    
    def load_yfinance_data(self, ticker: str) -> pd.DataFrame:
        """Load recent data from yfinance"""
        try:
            logger.info(f"Fetching yfinance data for {ticker}")
            
            stock = yf.Ticker(ticker)
            
            # Get 2 years of recent data
            period = self.config['data_pipeline']['recent_years']
            df = stock.history(period=f"{period}y")
            
            if df.empty:
                logger.warning(f"No yfinance data for {ticker}")
                return pd.DataFrame()
            
            # Standardize column names
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df.index.name = 'Date'
            
            return df
        except Exception as e:
            logger.warning(f"yfinance failed for {ticker}: {e}")
            return pd.DataFrame()
    
    def merge_price_data(self, ticker: str) -> pd.DataFrame:
        """Merge Kaggle historical + yfinance recent data"""
        
        # Load both sources
        kaggle_df = self.load_kaggle_data(ticker)
        yfinance_df = self.load_yfinance_data(ticker)
        
        if kaggle_df.empty and yfinance_df.empty:
            raise ValueError(f"No data found for {ticker}")
        
        if kaggle_df.empty:
            return yfinance_df
        
        if yfinance_df.empty:
            return kaggle_df
        
        # Merge: yfinance overwrites overlapping dates
        combined = pd.concat([kaggle_df, yfinance_df])
        combined = combined[~combined.index.duplicated(keep='last')]
        combined = combined.sort_index()
        
        logger.info(f"Merged {len(combined)} rows for {ticker}")
        return combined
    
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
        
        indicators = {
            'VIX': 'VIXCLS',
            'FED_RATE': 'DFF',
            'TREASURY_10Y': 'DGS10',
            'TREASURY_2Y': 'DGS2',
            'CPI': 'CPIAUCSL',
            'UNEMPLOYMENT': 'UNRATE'
        }
        
        macro_data = {}
        start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
        
        for name, series_id in indicators.items():
            try:
                series = self.fred.get_series(series_id, observation_start=start_date)
                macro_data[name] = series
            except Exception as e:
                logger.warning(f"Failed to fetch {name}: {e}")
        
        if not macro_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(macro_data)
        
        # Cache it
        cache_data = df.copy()
        cache_data['index'] = df.index.astype(str)
        with open(cache_file, 'w') as f:
            json.dump(cache_data.to_dict(), f)
        
        return df
    
    def load_alpha_vantage_data(self, ticker: str) -> Dict:
        """Load fundamentals from Alpha Vantage"""
        cache_file = self.cache_dir / f"av_{ticker}.json"
        
        # Check cache
        if cache_file.exists():
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age < 24 * 3600:
                with open(cache_file, 'r') as f:
                    return json.load(f)
        
        api_key = os.getenv('ALPHA_VANTAGE_KEY')
        if not api_key:
            return {}
        
        try:
            logger.info(f"Fetching Alpha Vantage data for {ticker}")
            
            # Get company overview
            url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={api_key}"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if 'Symbol' in data:
                # Cache it
                with open(cache_file, 'w') as f:
                    json.dump(data, f)
                return data
            
            logger.warning(f"Alpha Vantage returned: {data}")
        except Exception as e:
            logger.warning(f"Alpha Vantage failed: {e}")
        
        return {}
    
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
        """Get earnings dates, sector, etc from yfinance"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            metadata = {
                'sector': info.get('sector', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'beta': info.get('beta', 1.0),
                'forward_pe': info.get('forwardPE', None),
            }
            
            # Get next earnings date
            try:
                calendar = stock.calendar
                if calendar is not None and 'Earnings Date' in calendar:
                    next_earnings = calendar['Earnings Date'][0]
                    metadata['next_earnings_date'] = next_earnings
            except Exception:
                metadata['next_earnings_date'] = None
            
            return metadata
        except Exception as e:
            logger.warning(f"yfinance metadata failed: {e}")
            return {}
    
    def process_ticker(self, ticker: str) -> pd.DataFrame:
        """Main pipeline: load and merge all data for a ticker"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {ticker}")
        logger.info(f"{'='*60}")
        
        # 1. Load price data (Kaggle + yfinance)
        df = self.merge_price_data(ticker)
        
        # 2. Load macro data
        macro_df = self.load_fred_macro_data()
        
        # Merge macro data (forward fill to daily) - skip if empty
        if not macro_df.empty:
            # Ensure both indices are timezone-naive datetime
            df.index = pd.to_datetime(df.index).tz_localize(None)
            macro_df.index = pd.to_datetime(macro_df.index).tz_localize(None)
            
            macro_df = macro_df.reindex(df.index, method='ffill')
            df = pd.concat([df, macro_df], axis=1)
        else:
            logger.warning("No macro data available, skipping")
        
        # 3. Load fundamentals
        av_data = self.load_alpha_vantage_data(ticker)
        if av_data:
            df['PE_RATIO'] = float(av_data.get('PERatio', 0) or 0)
            df['PEG_RATIO'] = float(av_data.get('PEGRatio', 0) or 0)
            df['PROFIT_MARGIN'] = float(av_data.get('ProfitMargin', 0) or 0)
        
        # 4. Load sentiment
        finnhub_data = self.load_finnhub_data(ticker)
        if finnhub_data.get('sentiment'):
            sentiment = finnhub_data['sentiment']
            df['NEWS_SENTIMENT'] = sentiment.get('sentiment', {}).get('score', 0)
        
        # 5. Store metadata (for later use)
        metadata = self.get_yfinance_metadata(ticker)
        headlines = self.load_newsapi_headlines(ticker)
        
        # Fill missing values
        df = df.ffill().bfill()
        
        # Save to parquet
        output_file = f"data/processed/{ticker}.parquet"
        df.to_parquet(output_file)
        logger.info(f"Saved {len(df)} rows to {output_file}")
        
        # Save metadata separately
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
    print(f"Columns: {list(df.columns)}")
    print(f"\nLast 5 rows:")
    print(df.tail())
