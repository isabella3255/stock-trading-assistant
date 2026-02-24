# AI Trading Signal System

A professional-grade AI-powered trading signal system that combines machine learning models (XGBoost + LSTM), multi-source data integration, options analysis, and Claude AI synthesis to generate high-confidence trade recommendations.

## Overview

This system analyzes stocks and ETFs using:
- **6 Data Sources**: Kaggle historical data, yfinance, FRED macro data, Alpha Vantage fundamentals, Finnhub sentiment, and NewsAPI headlines
- **70+ Technical Features**: Price/trend, momentum, volatility, volume, patterns, macro context, and sentiment indicators
- **Dual ML Models**: XGBoost classifier (60% weight) + LSTM neural network (40% weight)
- **Options Analysis**: IV crush risk assessment, strike/expiration scoring, and spread recommendations
- **Claude AI Synthesis**: Human-level trade recommendation synthesis using Claude Opus

The system generates signals (STRONG_BULL, BULL, NEUTRAL, BEAR, STRONG_BEAR) with confidence intervals and provides specific entry/exit recommendations.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                              │
├──────────┬──────────┬──────────┬──────────┬──────────┬──────────┤
│  Kaggle  │ yfinance │   FRED   │  Alpha   │ Finnhub  │ NewsAPI  │
│Historical│   Live   │   Macro  │ Vantage  │Sentiment │Headlines │
└─────┬────┴─────┬────┴─────┬────┴─────┬────┴─────┬────┴─────┬────┘
      │          │          │          │          │          │
      └──────────┴──────────┴──────────┴──────────┴──────────┘
                            │
                    ┌───────▼────────┐
                    │ Data Pipeline  │
                    │  (Merge/Clean) │
                    └───────┬────────┘
                            │
                    ┌───────▼────────┐
                    │    Feature     │
                    │  Engineering   │
                    │   (70+ Feats)  │
                    └───┬────────┬───┘
                        │        │
            ┌───────────┘        └───────────┐
            │                                │
    ┌───────▼────────┐            ┌─────────▼────────┐
    │   XGBoost      │            │      LSTM        │
    │  Classifier    │            │  Neural Network  │
    │   (60% wt)     │            │    (40% wt)      │
    └───────┬────────┘            └─────────┬────────┘
            │                                │
            └──────────┬──────────┬──────────┘
                       │          │
                ┌──────▼──────┐   │
                │  Ensemble   │   │
                │   Scorer    │   │
                └──────┬──────┘   │
                       │          │
                ┌──────▼──────────▼─────┐
                │   Options Analyzer    │
                │  (IV Risk, Greeks)    │
                └──────┬────────────────┘
                       │
                ┌──────▼──────────┐
                │  Claude Opus    │
                │   Synthesis     │
                └──────┬──────────┘
                       │
                ┌──────▼──────────┐
                │ Trade Recommend │
                │  + Dashboard    │
                └─────────────────┘
```

## Features

### Multi-Source Data Integration
- **Kaggle**: 20+ years of historical OHLCV data
- **yfinance**: Live/recent data, options chains, earnings, fundamentals
- **FRED**: Macro indicators (VIX, Fed rate, yields, CPI, unemployment)
- **Alpha Vantage**: EPS history, fundamentals, P/E ratios
- **Finnhub**: News sentiment, insider transactions, analyst ratings
- **NewsAPI**: Recent headlines for sentiment analysis

### Advanced Feature Engineering
- **Price & Trend**: EMAs (9/21/50/200), SMAs, crossovers, trend strength
- **Momentum**: ROC, MACD, RSI, Stochastic, momentum acceleration
- **Volatility**: ATR, Bollinger Bands, squeeze detection, historical volatility
- **Volume**: VWAP, volume ratios, OBV, price-volume divergence
- **Patterns**: Higher highs/lows, inside days, bull flags, breakouts
- **Macro**: VIX regimes, yield curve, Fed policy changes
- **Sentiment**: Earnings beat rates, insider buying, analyst upgrades

### Machine Learning Models
- **XGBoost Classifier**: Trained on 70+ features with Optuna hyperparameter tuning
- **LSTM Network**: 30-day sequence memory for temporal patterns
- **Ensemble**: Weighted combination with bootstrap confidence intervals

### Options Strategy Engine
- IV crush risk assessment (compares current IV to historical volatility)
- Strike/expiration scoring (optimal DTE: 21-45 days, delta: 0.35-0.55)
- Automatic spread recommendations for high IV environments
- Liquidity filtering and Greek analysis

### Claude AI Integration
- Synthesizes all signals, technicals, fundamentals, and sentiment
- Provides human-level trade recommendations
- Specific entry/exit/stop-loss prices
- Risk factor analysis and alternative strategies
- Confidence scoring (1-10 scale)

## Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager
- Git

### Setup Instructions

1. **Clone the repository**
   ```bash
   cd stock-trading-assistant/trading_ai
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up API keys**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys (see API Keys section below)
   ```

5. **Verify installation**
   ```bash
   python main.py --help
   ```

## API Keys Setup

### Required API Keys

#### 1. Anthropic API (Claude) - **REQUIRED**
- **Purpose**: LLM synthesis engine for trade recommendations
- **Get key at**: https://console.anthropic.com/
- **Cost**: Pay-as-you-go (~$0.01-0.05 per ticker analysis)
- **Add to `.env`**: `ANTHROPIC_API_KEY=your_key_here`

### Free Tier API Keys (Recommended)

#### 2. FRED API (Federal Reserve Economic Data)
- **Purpose**: Macro economic indicators (VIX, yields, CPI, etc.)
- **Get key at**: https://fred.stlouisfed.org/docs/api/api_key.html
- **Limits**: Unlimited (free)
- **Add to `.env`**: `FRED_API_KEY=your_key_here`

#### 3. Alpha Vantage
- **Purpose**: Earnings data, fundamentals, P/E ratios
- **Get key at**: https://www.alphavantage.co/support/#api-key
- **Limits**: 25 requests/day (free tier)
- **Add to `.env`**: `ALPHA_VANTAGE_KEY=your_key_here`

#### 4. Finnhub
- **Purpose**: News sentiment, insider transactions, analyst ratings
- **Get key at**: https://finnhub.io/register
- **Limits**: 60 calls/minute (free tier)
- **Add to `.env`**: `FINNHUB_API_KEY=your_key_here`

#### 5. NewsAPI
- **Purpose**: Recent news headlines for sentiment
- **Get key at**: https://newsapi.org/register
- **Limits**: 100 requests/day (free tier)
- **Add to `.env`**: `NEWS_API_KEY=your_key_here`

### Optional API Keys

#### 6. Alpaca (Paper Trading)
- **Purpose**: Live paper trading integration
- **Get key at**: https://alpaca.markets/
- **Add to `.env`**: 
  ```
  ALPACA_API_KEY=your_key_here
  ALPACA_SECRET_KEY=your_secret_here
  ```

#### 7. Kaggle (Historical Data)
- **Purpose**: Download historical OHLCV datasets
- **Get credentials at**: https://www.kaggle.com/settings/account
- **Add to `.env`**: 
  ```
  KAGGLE_USERNAME=your_username
  KAGGLE_KEY=your_key_here
  ```
- **Note**: `kagglehub` has built-in authentication as alternative

## Usage

### Basic Usage

**Analyze all tickers in watchlist:**
```bash
python main.py
```

**Analyze a single ticker:**
```bash
python main.py --ticker NVDA
```

**Run without LLM synthesis (faster, cheaper):**
```bash
python main.py --no-llm
```

**Run in backtest mode:**
```bash
python main.py --backtest
```

### Launch Dashboard

```bash
streamlit run dashboard.py
```

The dashboard will open at `http://localhost:8501` with:
- Interactive charts with technical indicators
- Real-time signal gauges
- Feature importance visualization
- Options chain analysis
- Claude recommendations
- Backtest results

### Configuration

Edit `config/config.yaml` to customize:
- **Watchlist**: Add/remove tickers
- **Thresholds**: Adjust confidence and signal thresholds
- **Options**: Modify DTE ranges, IV crush thresholds
- **Risk**: Change position sizing and stop-loss rules
- **LLM**: Configure Claude model and token limits

### Example Output

```
╔══════════════════════════════════════════════════════════════════════╗
║                    AI TRADING SIGNAL SYSTEM                          ║
║                    Daily Report: 2026-02-23                          ║
╚══════════════════════════════════════════════════════════════════════╝

Ticker | Signal      | Score | IV Risk | Top Option    | Verdict     | Conf
─────────────────────────────────────────────────────────────────────────
NVDA   | STRONG_BULL | 0.78  | MEDIUM  | $140c Mar21   | STRONG BUY  | 8/10
QQQ    | BULL        | 0.68  | LOW     | $485c Mar14   | BUY         | 7/10
AAPL   | NEUTRAL     | 0.55  | LOW     | -             | HOLD        | 5/10
TSLA   | BEAR        | 0.35  | HIGH    | -             | AVOID       | 6/10
...

Full reports saved to: outputs/signals/daily_report_2026-02-23.csv
```

## Project Structure

```
trading_ai/
├── config/
│   └── config.yaml           # System configuration
├── data/
│   ├── raw/                  # Raw data from APIs
│   ├── processed/            # Cleaned DataFrames
│   └── cache/                # API response cache
├── models/
│   ├── xgboost/              # Trained XGBoost models
│   └── lstm/                 # Trained LSTM models
├── modules/
│   ├── data_pipeline.py      # Multi-source data loader
│   ├── feature_engineering.py # Technical indicators
│   ├── ml_models.py          # XGBoost + LSTM
│   ├── options_analyzer.py   # Options scoring
│   ├── llm_synthesis.py      # Claude integration
│   └── backtester.py         # Backtesting engine
├── outputs/
│   └── signals/              # Daily reports
├── tests/                    # Unit tests
├── logs/                     # Application logs
├── main.py                   # CLI entry point
├── dashboard.py              # Streamlit dashboard
├── requirements.txt          # Dependencies
├── .env.example              # API key template
└── README.md                 # This file
```

## Development Phases

This project is built in 8 phases:

- [x] **Phase 1**: Project structure & environment setup
- [ ] **Phase 2**: Data pipeline (6 data sources)
- [ ] **Phase 3**: Feature engineering (70+ features)
- [ ] **Phase 4**: ML model layer (XGBoost + LSTM)
- [ ] **Phase 5**: Options analyzer
- [ ] **Phase 6**: LLM synthesis engine
- [ ] **Phase 7**: Main orchestrator & CLI
- [ ] **Phase 8**: Streamlit dashboard

Each phase is tested and confirmed before proceeding to the next.

## Trading Methodology

This system implements principles from:

1. **Van Tharp** - *Trade Your Way to Financial Freedom*
   - Position sizing based on ATR
   - Risk management (max 5% per trade)
   - Volatility-based stop losses

2. **Andrew Aziz** - *How to Day Trade for a Living*
   - Volume and price action analysis
   - VWAP and momentum indicators
   - Pattern recognition (flags, breakouts)

3. **Options Trading Success**
   - IV crush risk management
   - DTE optimization (21-45 days sweet spot)
   - Spread strategies for high IV environments

## Risk Warnings

⚠️ **IMPORTANT DISCLAIMERS**

- This system is for **educational and research purposes only**
- **Past performance does not guarantee future results**
- **Trading involves substantial risk of loss**
- **Never trade with money you cannot afford to lose**
- **Always do your own due diligence**
- ML models are trained on historical data and may not capture regime changes
- Options trading is particularly risky and not suitable for all investors
- API reliability can affect system performance

**This is NOT financial advice. Always consult with a licensed financial advisor.**

## Testing

Run unit tests:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=modules tests/
```

## Logging

Logs are written to `logs/trading_ai.log` with rotation:
- Max file size: 10 MB
- Backup count: 5 files
- Format: `YYYY-MM-DD HH:MM:SS - module - LEVEL - message`

View logs:
```bash
tail -f logs/trading_ai.log
```

## Troubleshooting

### Common Issues

**API Rate Limits**
- System caches responses for 24 hours
- Alpha Vantage: Only 25 requests/day on free tier
- Solution: Reduce watchlist size or upgrade API tier

**Missing Data**
- Some tickers may not have complete historical data
- Solution: System handles gracefully with forward/backward fill

**TensorFlow/LSTM Errors**
- Ensure Python 3.10+ and compatible TensorFlow version
- Solution: `pip install --upgrade tensorflow`

**Module Import Errors**
- Verify virtual environment is activated
- Solution: `source venv/bin/activate && pip install -r requirements.txt`

## Performance

**Expected Runtime (10 tickers):**
- Data pipeline: ~2-3 minutes (first run, then cached)
- Feature engineering: ~30 seconds
- ML models: ~1-2 minutes (training), <5 seconds (inference)
- Options analysis: ~10 seconds
- LLM synthesis: ~30 seconds
- **Total**: ~5-7 minutes (first run), ~2 minutes (subsequent runs with cache)

## Contributing

This is a personal trading system project. If you fork it:
1. Never commit `.env` files
2. Test thoroughly before using with real money
3. Respect API rate limits
4. Follow the phase-by-phase development approach

## License

This project is provided as-is for educational purposes. Use at your own risk.

## Acknowledgments

- **Trading Books**: Van Tharp, Andrew Aziz, and various options trading texts
- **Data Providers**: Kaggle, yfinance, FRED, Alpha Vantage, Finnhub, NewsAPI
- **ML Libraries**: XGBoost, TensorFlow, scikit-learn, Optuna
- **Claude AI**: Anthropic's Claude for intelligent synthesis

## Contact & Support

For questions or issues:
1. Check the troubleshooting section above
2. Review API documentation for data source issues
3. Ensure all dependencies are correctly installed
4. Verify API keys are valid and have sufficient quota

---

**Built with Python 🐍 | Powered by Claude 🤖 | Trade Responsibly 📈**
