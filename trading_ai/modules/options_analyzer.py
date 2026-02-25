"""
Options Analyzer Module

Scores options setups, assesses IV crush risk, and calculates theoretical prices
using Black-Scholes when yfinance data is unavailable.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)


def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> dict:
    """
    Calculate Black-Scholes call option price and Greeks.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate (annual, as decimal, e.g. 0.05 for 5%)
        sigma: Implied volatility (annual, as decimal, e.g. 0.30 for 30%)
    
    Returns:
        dict with 'price', 'delta', 'gamma', 'theta', 'vega'
    """
    if T <= 0:
        # Option expired
        return {
            'price': max(0, S - K),
            'delta': 1.0 if S > K else 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0
        }
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
             - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365  # Daily theta
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Vega per 1% IV change
    
    return {
        'price': round(call_price, 2),
        'delta': round(delta, 4),
        'gamma': round(gamma, 6),
        'theta': round(theta, 4),
        'vega': round(vega, 4),
    }


class OptionsAnalyzer:
    """Analyze and score options chains"""

    def __init__(self, ticker: str, current_price: float, hist_vol_20: float):
        self.ticker = ticker
        self.current_price = current_price
        self.hist_vol_20 = hist_vol_20
        self.stock = yf.Ticker(ticker)

    def get_options_chain(self):
        """Get all call options from yfinance across all expirations"""
        try:
            expirations = self.stock.options
            if not expirations:
                logger.warning(f"No options data for {self.ticker}")
                return None

            chains = []
            for exp_date in expirations:
                opt = self.stock.option_chain(exp_date)
                calls = opt.calls.copy()
                calls['expiration'] = exp_date
                chains.append(calls)

            return pd.concat(chains, ignore_index=True)
        except Exception as e:
            logger.warning(f"Failed to get options chain: {e}")
            return None

    def calculate_dte(self, expiration) -> int:
        """Days to expiration — uses UTC-aware comparison to avoid timezone mismatch"""
        exp_date = pd.to_datetime(expiration).tz_localize('UTC')
        now = datetime.now(timezone.utc)
        return max(0, (exp_date - now).days)

    def _get_premium(self, row, use_ask_for_buy: bool = True) -> float:
        """
        Extract option premium from a chain row.
        
        For realistic execution:
          - Use ASK price when buying (you pay the ask)
          - Use BID price when selling (you receive the bid)
        
        Falls back to lastPrice if bid/ask is unavailable, then Black-Scholes
        if all market data is stale.
        
        Args:
            row: Options chain row with price data
            use_ask_for_buy: If True, prefer ask price (realistic buy price)
        """
        bid = row.get('bid', 0) or 0
        ask = row.get('ask', 0) or 0
        last = row.get('lastPrice', 0) or 0
        
        # Prefer ask price for buys (realistic execution)
        if use_ask_for_buy and ask > 0:
            return ask
        
        # Fall back to bid-ask midpoint if available
        if bid > 0 and ask > 0:
            return (bid + ask) / 2
        
        # Fall back to last traded price
        if last > 0:
            return last
        
        # Final fallback: Black-Scholes theoretical price
        iv = row.get('impliedVolatility', 0) or 0
        if iv > 0:
            dte = self.calculate_dte(row['expiration'])
            T = dte / 365.0
            if T > 0:
                bs = black_scholes_call(
                    S=self.current_price,
                    K=row['strike'],
                    T=T,
                    r=0.05,  # Assume 5% risk-free rate
                    sigma=iv
                )
                return bs['price']
        
        return 0

    def score_option(self, row, dte_min=14, dte_max=45) -> tuple:
        """Score a single option on DTE, moneyness, liquidity, and delta"""
        dte = self.calculate_dte(row['expiration'])
        strike = row['strike']
        volume = row.get('volume', 0) or 0
        oi = row.get('openInterest', 0) or 0

        # DTE score: full marks inside 14-45, linear decay outside
        if dte_min <= dte <= dte_max:
            dte_score = 1.0
        elif dte < dte_min:
            dte_score = max(0.0, dte / dte_min)
        else:
            dte_score = max(0.0, 1 - (dte - dte_max) / 30)

        # Moneyness score: prefer 0.98–1.05x (ATM to slight OTM)
        moneyness = strike / self.current_price
        if 0.98 <= moneyness <= 1.05:
            moneyness_score = 1.0
        else:
            moneyness_score = max(0.0, 1 - abs(moneyness - 1.0) * 2)

        # Liquidity score: based on volume and open interest
        volume_score = min(1.0, volume / 100) if volume > 0 else 0.0
        oi_score = min(1.0, oi / 100) if oi > 0 else 0.0
        liquidity_score = (volume_score + oi_score) / 2

        # Delta score: use actual delta from yfinance when available,
        # otherwise fall back to a moneyness-based linear approximation.
        # The old code used three hardcoded buckets (0.5/0.7/0.3) which
        # caused all OTM calls to receive the same wrong score.
        actual_delta = row.get('delta', None)
        if actual_delta is not None and not np.isnan(actual_delta):
            approx_delta = abs(float(actual_delta))
        else:
            # Linear approximation: ATM ≈ 0.5, scales with distance from ATM
            approx_delta = max(0.0, min(1.0, 0.5 + (1.0 - moneyness) * 0.5))

        if 0.35 <= approx_delta <= 0.55:
            delta_score = 1.0
        else:
            delta_score = max(0.0, 1 - abs(approx_delta - 0.45) * 2)

        # Weighted composite score
        score = (
            dte_score * 0.3
            + moneyness_score * 0.3
            + liquidity_score * 0.2
            + delta_score * 0.2
        )

        return score, dte

    def assess_iv_crush_risk(self, days_to_earnings=None) -> dict:
        """Assess IV crush risk by comparing current IV to historical volatility"""
        chain = self.get_options_chain()
        if chain is None or chain.empty:
            return {'risk': 'UNKNOWN', 'iv_premium_ratio': 0}

        # Find ATM calls — widen to ±5% so we always find strikes.
        # The old ±2% range was too narrow for high-priced or wide-spread stocks.
        atm_calls = chain[
            abs(chain['strike'] - self.current_price) < self.current_price * 0.05
        ]
        if atm_calls.empty:
            # Last resort: just take the single closest strike
            idx = (chain['strike'] - self.current_price).abs().idxmin()
            atm_calls = chain.loc[[idx]]

        # Filter out zero/near-zero/NaN IV before computing median.
        # yfinance uses 1e-05 as a placeholder for stale/unavailable IV (market closed).
        iv_series = atm_calls['impliedVolatility'].replace(0, np.nan).dropna()
        iv_series = iv_series[iv_series > 0.01]  # drop placeholder values < 1% IV
        if iv_series.empty or self.hist_vol_20 == 0:
            return {'risk': 'UNKNOWN', 'iv_premium_ratio': 0,
                    'current_iv': 0, 'environment': 'UNKNOWN'}

        current_iv = float(iv_series.median())
        iv_premium_ratio = current_iv / self.hist_vol_20

        # Risk classification: near earnings is much higher risk
        if days_to_earnings is not None and days_to_earnings <= 5:
            if iv_premium_ratio > 1.5:
                risk = "HIGH"
            elif iv_premium_ratio > 1.2:
                risk = "MEDIUM"
            else:
                risk = "LOW"
        else:
            if iv_premium_ratio > 1.8:
                risk = "MEDIUM"
            else:
                risk = "LOW"

        environment = (
            'EXPENSIVE' if iv_premium_ratio > 1.5
            else 'FAIR' if iv_premium_ratio > 1.0
            else 'CHEAP'
        )

        return {
            'risk': risk,
            'iv_premium_ratio': round(iv_premium_ratio, 3),
            'current_iv': round(current_iv, 4),
            'environment': environment
        }

    def analyze(self, signal: str, days_to_earnings=None, signal_score: float = 0.5) -> dict:
        """Main analysis — returns scored call recommendations, spread setup, and fixed risk sizing.

        Args:
            signal: 'BULL' or 'STRONG_BULL'
            days_to_earnings: days until next earnings (optional)
            signal_score: ensemble final_score from ml_result (0–1). Used to adjust
                          position sizing conservatively. Default 0.5 = baseline risk.
        """

        if signal not in ['BULL', 'STRONG_BULL']:
            return {'signal': signal, 'recommendation': 'No options plays for bearish/neutral signals'}

        chain = self.get_options_chain()
        if chain is None or chain.empty:
            return {'error': 'No options data available'}

        # Score every call in the chain
        scored = []
        for _, row in chain.iterrows():
            premium = self._get_premium(row)
            score, dte = self.score_option(row)
            scored.append({
                'strike': row['strike'],
                'expiration': row['expiration'],
                'dte': dte,
                'premium': premium,
                'volume': row.get('volume', 0) or 0,
                'openInterest': row.get('openInterest', 0) or 0,
                'score': score
            })

        scores_df = pd.DataFrame(scored)
        scores_df = scores_df[scores_df['premium'] > 0]   # drop illiquid / stale rows
        scores_df = scores_df.sort_values('score', ascending=False).reset_index(drop=True)

        if scores_df.empty:
            return {'error': 'No valid (priced) options found'}

        # Top 3 calls by composite score
        top_calls = scores_df.head(3).to_dict('records')
        for call in top_calls:
            call['breakeven'] = round(call['strike'] + call['premium'], 2)
            call['max_loss'] = round(call['premium'], 2)

        # Best spread: buy top-scored call, sell the highest-scored call
        # with a higher strike (not just the first one in the list).
        # Old code used higher_strikes.iloc[0] which was the first by score,
        # but that happened to be the same as highest-scored — however the
        # sort was ascending=False so iloc[0] was actually the BEST, meaning
        # we were buying and selling the same option. Fix: explicitly filter
        # by strike > buy_strike and sort remaining by score desc.
        spread = None
        if len(scores_df) > 1:
            best_call = scores_df.iloc[0]
            sell_candidates = (
                scores_df[scores_df['strike'] > best_call['strike']]
                .sort_values('score', ascending=False)
            )
            # Iterate sell candidates until we find one that produces a valid spread:
            # net_debit must be positive (we pay a debit) AND less than spread_width
            # (otherwise max_gain would be zero or negative — not a tradeable spread).
            for _, sell_call in sell_candidates.iterrows():
                net_debit = round(best_call['premium'] - sell_call['premium'], 2)
                spread_width = sell_call['strike'] - best_call['strike']
                max_gain = round(spread_width - net_debit, 2)
                if net_debit > 0 and max_gain > 0:
                    spread = {
                        'buy_strike': best_call['strike'],
                        'sell_strike': sell_call['strike'],
                        'expiration': best_call['expiration'],
                        'net_debit': net_debit,
                        'spread_width': round(spread_width, 2),
                        'max_gain': max_gain,
                        'max_loss': net_debit,   # max loss on a debit spread = net debit paid
                        'reward_risk': round(max_gain / net_debit, 2)
                    }
                    break  # found a valid spread — stop searching

        iv_analysis = self.assess_iv_crush_risk(days_to_earnings)

        # ── Fixed Risk Position Sizing ─────────────────────────────────────────
        # Professional traders use fixed risk (1-2% of account per trade), NOT Kelly.
        # Kelly criterion over-leverages when using uncalibrated ML probabilities.
        # 
        # Sizing logic:
        #   - STRONG_BULL (score > 0.72): 2.0% risk
        #   - BULL (score > 0.62):         1.5% risk  
        #   - Otherwise:                   1.0% risk (baseline)
        # 
        # For options: Cap at 2% of account (options are already leveraged)
        # For stock: Use stop loss to calculate position size
        if signal == 'STRONG_BULL':
            base_risk_pct = 2.0
        elif signal == 'BULL':
            base_risk_pct = 1.5
        else:
            base_risk_pct = 1.0
        
        # Adjust down for low confidence scores
        confidence_multiplier = min(1.0, signal_score / 0.6)  # Scale down if score < 0.6
        suggested_risk_pct = round(base_risk_pct * confidence_multiplier, 1)
        
        # Cap at 2% for options (they're already leveraged)
        suggested_risk_pct = min(2.0, suggested_risk_pct)

        position_sizing = {
            'suggested_risk_pct_of_portfolio': suggested_risk_pct,
            'risk_method': 'fixed_risk',
            'signal_strength': signal,
            'confidence_score': round(signal_score, 3),
            'rationale': (
                f"Fixed Risk Sizing: {signal} signal with {signal_score:.0%} confidence → "
                f"{suggested_risk_pct}% of portfolio (conservative, capped at 2% for options)"
            ),
        }

        return {
            'signal': signal,
            'top_calls': top_calls,
            'best_spread': spread,
            'iv_analysis': iv_analysis,
            'recommendation': 'SPREAD' if iv_analysis['risk'] == 'HIGH' else 'LONG_CALL',
            'position_sizing': position_sizing,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    ticker = "NVDA"
    df = pd.read_parquet("data/processed/NVDA_featured.parquet")

    current_price = df['Close'].iloc[-1]
    hist_vol = df['hist_volatility_20'].iloc[-1]

    logger.info(f"Analyzing options for {ticker}")
    logger.info(f"Current price: ${current_price:.2f}")
    logger.info(f"Historical vol (20d): {hist_vol:.2%}")

    analyzer = OptionsAnalyzer(ticker, current_price, hist_vol)
    result = analyzer.analyze('BULL', days_to_earnings=30)

    logger.info(f"\nResults:")
    logger.info(f"Signal: {result.get('signal')}")
    logger.info(f"Recommendation: {result.get('recommendation')}")

    if 'iv_analysis' in result:
        iv = result['iv_analysis']
        logger.info(f"\nIV Analysis:")
        logger.info(f"  Risk: {iv['risk']}")
        logger.info(f"  IV/HV Ratio: {iv['iv_premium_ratio']:.2f}x")
        logger.info(f"  Current IV: {iv['current_iv']:.2%}")
        logger.info(f"  Environment: {iv['environment']}")

    if result.get('top_calls'):
        logger.info(f"\nTop 3 Calls:")
        for i, call in enumerate(result['top_calls'][:3], 1):
            logger.info(
                f"  {i}. ${call['strike']} exp {call['expiration']} "
                f"— Premium: ${call['premium']:.2f}, Breakeven: ${call['breakeven']:.2f}"
            )

    if result.get('best_spread'):
        spread = result['best_spread']
        logger.info(f"\nBest Spread:")
        logger.info(f"  Buy ${spread['buy_strike']} / Sell ${spread['sell_strike']} (width: ${spread.get('spread_width', 'N/A')})")
        logger.info(f"  Net Debit: ${spread['net_debit']:.2f}")
        logger.info(f"  Max Gain: ${spread['max_gain']:.2f} | Max Loss: ${spread['max_loss']:.2f}")
        logger.info(f"  Reward/Risk: {spread.get('reward_risk', 'N/A')}x")
    else:
        logger.info(f"\nNo valid spread found (all candidates had negative max_gain)")
