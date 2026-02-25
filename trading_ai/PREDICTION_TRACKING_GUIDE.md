# Prediction Tracking System - Quick Reference

## What Is This?

**Prediction Tracking** logs every signal the system generates and validates the outcome 5 days later. This shows your REAL win rate in current market conditions (not just backtested).

### The Difference

| Feature | What It Does | When To Use |
|---------|--------------|-------------|
| **Backtesting** | Tests on historical data (2024 data) | Validate strategy logic |
| **Prediction Tracking** | Tracks live predictions → validates outcomes | Validate real-world performance |

**Key insight**: A model with 65% backtested win rate might only achieve 48% real win rate due to overfitting. Prediction tracking tells you the truth.

---

## Daily Workflow

### After Market Close (4 PM ET)

```bash
cd trading_ai
source ../.venv/bin/activate

# Step 1: Generate today's predictions (automatically logged)
python main.py

# Step 2: Validate predictions made 5 days ago
python validate_predictions.py

# Step 3: View results in dashboard
streamlit run dashboard.py
# → Go to "📈 Live Track Record" tab
```

---

## What Gets Logged

Every prediction includes:

**Core Data:**
- Date, ticker, signal, confidence score
- Entry price, predicted direction
- XGBoost prob, Sequence prob

**Market Context:**
- VIX, VIX 3M, term slope
- RSI, historical volatility
- Days to earnings, IV crush risk
- Fear & Greed Index

**Outcomes (after 5 days):**
- Actual price 5 days later
- Actual return %
- Was prediction correct?
- Did it hit target (>2% move)?

---

## Interpreting Results

### After 1 Week (5-10 predictions)
- **Too early** - sample size too small
- Action: Keep logging

### After 1 Month (30-50 predictions)
- **Initial validation** - patterns emerging
- Look for: Win rate by confidence level
- Action: Identify which signals work

### After 3 Months (80-120 predictions)
- **Statistical significance** - reliable data
- Decision point: Trade or refine?

---

## Success Criteria

### GREEN LIGHT (Trade Small Size):
- Overall win rate: >55%
- High confidence (>0.72): >65% win rate
- Avg return: >+1% per trade
- Stable over 30+ days

### YELLOW LIGHT (Paper Trade Only):
- Overall win rate: 50-55%
- Inconsistent across confidence levels
- Need more data or refinement

### RED LIGHT (Don't Trade):
- Overall win rate: <50%
- High confidence signals <55% win rate
- Negative avg return
- Action: Debug model, check for bugs

---

## Common Insights You'll Discover

**By Confidence Level:**
```
High (>0.72):    68% win rate → TRADE THESE
Medium (0.62-0.72): 52% win rate → BREAKEVEN, SKIP
Low (<0.62):     42% win rate → IGNORE
```
→ **Action**: Only trade high-confidence signals

**By Ticker:**
```
NVDA:  71% win rate → Model works great
AAPL:  58% win rate → Decent
TSLA:  38% win rate → Model fails, remove from watchlist
```
→ **Action**: Focus capital on tickers where model has edge

**By Market Regime:**
```
VIX < 15 (calm):     62% win rate
VIX 15-25 (normal):  54% win rate
VIX > 25 (fear):     41% win rate
```
→ **Action**: Reduce size or don't trade when VIX >25

---

## Validation Commands

### Basic Validation (5 days ago)
```bash
python validate_predictions.py
```

### Custom Date Range
```bash
python validate_predictions.py --days 7  # Validate 7-day-old predictions
```

### Dry Run (Preview)
```bash
python validate_predictions.py --dry-run
# Shows what would be validated without updating log
```

---

## Files & Locations

```
trading_ai/
  outputs/
    prediction_log.csv          ← Master prediction log (grows over time)
    signals/
      daily_report_*.csv        ← Daily signal summaries
  
  modules/
    prediction_tracker.py       ← Logging system
  
  validate_predictions.py       ← Outcome validator (run daily)
  
  dashboard.py                  ← View results in "Live Track Record" tab
```

---

## Automation (Optional)

### Cron Job for Auto-Validation

Add to crontab (`crontab -e`):

```bash
# Validate predictions daily at 4:05 PM ET (Mon-Fri)
5 16 * * 1-5 cd /Users/yourusername/trading_ai && source ../.venv/bin/activate && python validate_predictions.py >> logs/validation.log 2>&1
```

This ensures outcomes are checked automatically every trading day.

---

## FAQ

**Q: How long until I have useful data?**
A: 30 days minimum for initial validation, 90 days for confidence.

**Q: What if my win rate is only 52%?**
A: You're barely above breakeven. Need >55% to cover transaction costs and have edge.

**Q: Should I retrain the model with tracked outcomes?**
A: Yes! After 100+ validated predictions, you can use this data to:
  - Identify which features cause failures
  - Filter training data to similar market conditions
  - Calibrate confidence scores

**Q: What's a good win rate for options trading?**
A: 55-60% is excellent for long-only directional options. 65%+ is exceptional.

**Q: Can I backfill historical predictions?**
A: No - this tracks forward-looking predictions only. That's what makes it valuable (no cheating).

---

## Next Steps

1. **Start tracking TODAY** - run `python main.py` daily
2. **Validate weekly** - run `python validate_predictions.py` every 5 trading days
3. **Review monthly** - check "Live Track Record" tab for patterns
4. **Decide in 90 days** - trade, refine, or abandon based on real results

**Remember**: Past performance ≠ future results, but a validated 60% win rate over 90 days is strong evidence of edge.
