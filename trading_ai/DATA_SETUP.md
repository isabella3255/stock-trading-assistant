# Kaggle Historical Data Setup

## Overview

Since you already have the Kaggle dataset downloaded, we'll use local files instead of the API. This is faster and more reliable.

## Dataset Information

**Dataset**: `borismarjanovic/price-volume-data-for-all-us-stocks-etfs`
- Contains 20+ years of historical OHLCV data
- Individual CSV files per ticker (e.g., `nvda.us.txt`, `qqq.us.txt`)

## Setup Instructions

### Option 1: Put Files in `data/raw/kaggle/` (Recommended)

1. Create the kaggle subdirectory:
   ```bash
   mkdir -p data/raw/kaggle
   ```

2. Copy your downloaded Kaggle files to this folder:
   ```bash
   # If your downloaded folder is ~/Downloads/archive/
   cp ~/Downloads/archive/*.txt data/raw/kaggle/
   
   # Or drag and drop the files into:
   # /Users/isabellanelsen/Documents/GitHub/stock-trading-assistant/trading_ai/data/raw/kaggle/
   ```

3. Expected structure:
   ```
   data/raw/kaggle/
   ├── nvda.us.txt
   ├── qqq.us.txt
   ├── spy.us.txt
   ├── aapl.us.txt
   ├── msft.us.txt
   ├── tsla.us.txt
   ├── amd.us.txt
   ├── meta.us.txt
   ├── amzn.us.txt
   └── googl.us.txt
   ```

### Option 2: Keep Files Elsewhere and Point to Them

If you prefer to keep the files in their current location, we can configure the data pipeline to read from there.

Just tell me the path where your Kaggle files are, and I'll update the config.

## File Format

The Kaggle CSV files have this format:
```
Date,Open,High,Low,Close,Volume,OpenInt
2020-01-02,100.5,102.3,99.8,101.2,5000000,0
```

## Next Steps

Once files are in place, the DataPipeline (Phase 2) will:
1. Read from `data/raw/kaggle/{ticker}.us.txt`
2. Parse Date, OHLC, Volume columns
3. Merge with yfinance recent data
4. Add macro data from FRED
5. Save to `data/processed/{ticker}.parquet`

## Verification

To verify files are in the right place:
```bash
cd trading_ai
ls -lh data/raw/kaggle/ | head -15
```

You should see your ticker files listed.
