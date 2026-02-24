"""
LLM Synthesis Engine

Uses Claude API to synthesize all signals into final trade recommendation.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class LLMSynthesizer:
    """Synthesize all signals using Claude API"""
    
    def __init__(self, config: dict):
        self.config = config
        self.client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self.cache_dir = Path("data/cache")
        
    def build_context(self, ticker: str, data: dict) -> str:
        """Build comprehensive context for Claude"""
        
        # Extract all the data
        price_data = data.get('price_data', {})
        ml_signals = data.get('ml_signals', {})
        technical = data.get('technical', {})
        options = data.get('options', {})
        fundamentals = data.get('fundamentals', {})
        headlines = data.get('headlines', [])
        
        context = f"""You are a seasoned quantitative trader with 20 years of experience.
You have studied Van Tharp's position sizing principles (Trade Your Way to Financial Freedom),
Andrew Aziz's momentum and pattern strategies (How to Day Trade for a Living),
and options trading principles from Simple Steps to Option Trading Success.

Analyze the following data for {ticker} and give a clear, brutally honest trade recommendation.

=== MARKET DATA ===
Current Price: ${price_data.get('current_price', 0):.2f}
Previous Close: ${price_data.get('prev_close', 0):.2f}
52-Week High/Low: ${price_data.get('week_52_high', 0):.2f} / ${price_data.get('week_52_low', 0):.2f}
Today's Volume vs Avg: {price_data.get('volume_ratio', 1):.1f}x

=== ML MODEL SIGNALS ===
XGBoost Bullish Probability (3-day): {ml_signals.get('xgb_prob', 0):.1%}
Sequence Model Probability (3-day): {ml_signals.get('lstm_prob', 0):.1%}
Ensemble Score: {ml_signals.get('final_score', 0):.1%}
Signal: {ml_signals.get('signal', 'NEUTRAL')}

=== TECHNICAL PICTURE ===
RSI(14): {technical.get('rsi', 50):.1f}
MACD Histogram: {technical.get('macd_hist', 0):.3f} ({'positive momentum' if technical.get('macd_hist', 0) > 0 else 'negative momentum'})
Price vs EMA200: {technical.get('price_vs_ema200', 0):+.1f}% ({'above' if technical.get('above_200ema', 0) else 'below'} long-term trend)
ATR(14): ${technical.get('atr', 0):.2f} ({(technical.get('atr', 0)/price_data.get('current_price', 1)*100):.1f}% daily range)
Bollinger Band Squeeze: {'YES - coiling for breakout' if technical.get('bb_squeeze', 0) else 'No'}
Bull Flag Setup: {'YES' if technical.get('bull_flag_setup', 0) else 'No'}
Volume Spike: {'YES' if technical.get('volume_spike', 0) else 'No'}
Historical Volatility (20d): {technical.get('hist_vol_20', 0):.1%}

=== FUNDAMENTALS ===
Earnings Beat Rate: {fundamentals.get('earnings_beat_rate', 0):.0%}
Days to Earnings: {fundamentals.get('days_to_earnings', 'Unknown')}
Sector: {fundamentals.get('sector', 'Unknown')}

=== NEWS SENTIMENT ===
Recent Headlines: {', '.join(headlines[:5]) if headlines else 'No headlines available'}

=== OPTIONS ANALYSIS ===
IV Environment: {options.get('iv_analysis', {}).get('environment', 'UNKNOWN')}
IV Crush Risk: {options.get('iv_analysis', {}).get('risk', 'UNKNOWN')}
IV/HV Ratio: {options.get('iv_analysis', {}).get('iv_premium_ratio', 0):.2f}x
Recommendation: {options.get('recommendation', 'N/A')}
"""
        
        # Add top option details if available
        if options.get('top_calls'):
            top_call = options['top_calls'][0]
            context += f"\nTop Call: ${top_call['strike']} exp {top_call['expiration']}, Premium: ${top_call['premium']:.2f}"
        
        if options.get('best_spread'):
            spread = options['best_spread']
            context += f"\nTop Spread: Buy ${spread['buy_strike']} / Sell ${spread['sell_strike']}, Net Debit: ${spread['net_debit']:.2f}"
        
        context += f"""

=== POSITION SIZING (Van Tharp Method) ===
Account Size: ${self.config.get('risk_management', {}).get('position_size_account', 1000)}
Max Risk Per Trade: {self.config.get('risk_management', {}).get('max_portfolio_risk_pct', 0.05)*100:.0f}%
ATR Stop Distance: ${technical.get('atr', 0) * 2:.2f}

Please provide:
1. OVERALL VERDICT: [STRONG BUY / BUY / HOLD / AVOID / SHORT] with one sentence rationale
2. HIGHEST PROBABILITY TRADE SETUP: Exact entry, target, stop loss
3. OPTIONS RECOMMENDATION: Specific call or spread to buy with exact strike and expiry
4. RISK FACTORS: Top 3 reasons this trade could fail
5. ALTERNATIVE: If you would NOT take this trade, what would you do instead?
6. CONFIDENCE: Your overall confidence in this setup on a scale of 1-10

Be direct, specific, and numerical. No vague language.
"""
        
        return context
    
    def synthesize(self, ticker: str, data: dict) -> dict:
        """Send to Claude and get recommendation"""
        
        # Check cache
        today = datetime.now().strftime('%Y-%m-%d')
        cache_file = self.cache_dir / f"llm_{ticker}_{today}.json"
        
        if cache_file.exists():
            logger.info(f"Using cached LLM response for {ticker}")
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        # Build context
        context = self.build_context(ticker, data)
        
        logger.info(f"Calling Claude API for {ticker}...")
        
        try:
            response = self.client.messages.create(
                model=self.config.get('llm', {}).get('model', 'claude-opus-4-6'),
                max_tokens=self.config.get('llm', {}).get('max_tokens', 1500),
                messages=[{"role": "user", "content": context}]
            )
            
            recommendation = response.content[0].text
            
            result = {
                'ticker': ticker,
                'date': today,
                'recommendation': recommendation,
                'usage': {
                    'input_tokens': response.usage.input_tokens,
                    'output_tokens': response.usage.output_tokens
                }
            }
            
            # Cache it
            with open(cache_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Also save to outputs
            output_file = f"outputs/signals/{ticker}_{today}.txt"
            with open(output_file, 'w') as f:
                f.write(f"=== {ticker} Trade Recommendation - {today} ===\n\n")
                f.write(recommendation)
            
            logger.info(f"Saved recommendation to {output_file}")
            
            return result
            
        except Exception as e:
            logger.error(f"Claude API failed: {e}")
            return {'error': str(e)}


if __name__ == "__main__":
    import yaml
    import pandas as pd
    
    logging.basicConfig(level=logging.INFO)
    
    # Load config
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    ticker = "NVDA"
    df = pd.read_parquet("data/processed/NVDA_featured.parquet")
    
    # Build sample data package
    latest = df.iloc[-1]
    
    data = {
        'price_data': {
            'current_price': latest['Close'],
            'prev_close': df['Close'].iloc[-2],
            'week_52_high': df['Close'].rolling(252).max().iloc[-1],
            'week_52_low': df['Close'].rolling(252).min().iloc[-1],
            'volume_ratio': latest['volume_ratio']
        },
        'ml_signals': {
            'xgb_prob': 0.45,
            'lstm_prob': 0.50,
            'final_score': 0.47,
            'signal': 'NEUTRAL'
        },
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
        'options': {
            'iv_analysis': {'environment': 'FAIR', 'risk': 'MEDIUM', 'iv_premium_ratio': 1.2},
            'recommendation': 'LONG_CALL'
        },
        'fundamentals': {
            'earnings_beat_rate': 0.75,
            'days_to_earnings': 30,
            'sector': 'Technology'
        },
        'headlines': ['NVDA announces new AI chip', 'Tech stocks rally on earnings']
    }
    
    synthesizer = LLMSynthesizer(config)
    result = synthesizer.synthesize(ticker, data)
    
    if 'recommendation' in result:
        logger.info(f"\n{'='*70}")
        logger.info(f"CLAUDE RECOMMENDATION:")
        logger.info(f"{'='*70}\n")
        print(result['recommendation'])
        logger.info(f"\nTokens used: {result['usage']['input_tokens']} in, {result['usage']['output_tokens']} out")
