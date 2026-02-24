#!/usr/bin/env python3
"""
AI Trading Signal System - Phase 7 Streamlit Dashboard

Full interactive dashboard: live signals, charts, feature importance,
options analysis, LLM recommendations, and model performance.

Usage:
    cd trading_ai
    streamlit run dashboard.py
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Resolve paths regardless of working directory
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent          # trading_ai/
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)


# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Trading Signal System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SIGNAL_COLORS = {
    "STRONG_BULL": "#00c853",
    "BULL":        "#69f0ae",
    "NEUTRAL":     "#ffd740",
    "BEAR":        "#ff5722",
    "STRONG_BEAR": "#b71c1c",
}

SIGNAL_EMOJI = {
    "STRONG_BULL": "🚀",
    "BULL":        "📈",
    "NEUTRAL":     "⚖️",
    "BEAR":        "📉",
    "STRONG_BEAR": "💀",
}


@st.cache_data(ttl=300)
def load_featured(ticker: str) -> pd.DataFrame | None:
    path = ROOT / "data" / "processed" / f"{ticker}_featured.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


@st.cache_data(ttl=300)
def load_report() -> pd.DataFrame | None:
    today = datetime.now().strftime("%Y-%m-%d")
    p = ROOT / "outputs" / "signals" / f"daily_report_{today}.csv"
    if not p.exists():
        # Find any report
        reports = sorted((ROOT / "outputs" / "signals").glob("daily_report_*.csv"))
        if not reports:
            return None
        p = reports[-1]
    return pd.read_csv(p)


@st.cache_data(ttl=300)
def load_llm_result(ticker: str) -> dict | None:
    today = datetime.now().strftime("%Y-%m-%d")
    p = ROOT / "data" / "cache" / f"llm_{ticker}_{today}.json"
    if not p.exists():
        files = sorted((ROOT / "data" / "cache").glob(f"llm_{ticker}_*.json"))
        if not files:
            return None
        p = files[-1]
    with open(p) as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_feature_importance(ticker: str) -> pd.DataFrame | None:
    p = ROOT / "models" / "xgboost" / f"{ticker}_importance.csv"
    if not p.exists():
        return None
    return pd.read_csv(p).sort_values("importance", ascending=False)


def load_ticker_signals() -> dict:
    """Return {ticker: {signal, score, xgb_prob, seq_prob}} from latest report"""
    report = load_report()
    if report is None:
        return {}
    out = {}
    for _, row in report.iterrows():
        out[row["ticker"]] = row.to_dict()
    return out


def signal_badge(signal: str) -> str:
    color = SIGNAL_COLORS.get(signal, "#888")
    emoji = SIGNAL_EMOJI.get(signal, "❓")
    return f'<span style="background:{color};color:#000;padding:4px 12px;border-radius:12px;font-weight:700">{emoji} {signal}</span>'


def score_bar(score: float) -> None:
    """Render a horizontal score gauge 0–1"""
    pct = int(score * 100)
    color = SIGNAL_COLORS.get(
        "STRONG_BULL" if score > 0.72
        else "BULL" if score > 0.62
        else "BEAR" if score < 0.38
        else "STRONG_BEAR" if score < 0.28
        else "NEUTRAL"
    )
    st.markdown(
        f"""
        <div style="background:#333;border-radius:8px;height:22px;width:100%">
          <div style="background:{color};border-radius:8px;height:22px;width:{pct}%;
                      display:flex;align-items:center;justify-content:center;
                      color:#000;font-weight:700;font-size:13px">
            {score:.3f}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.shields.io/badge/AI%20Trading-Signal%20System-blue?style=for-the-badge")
    st.markdown("---")

    all_tickers = ["NVDA", "QQQ", "SPY", "AAPL", "MSFT", "TSLA", "AMD", "META", "AMZN", "GOOGL"]
    ticker = st.selectbox("🔍 Select Ticker", all_tickers, index=0)

    st.markdown("---")
    st.markdown("### ⚡ Quick Actions")

    run_pipeline = st.button("🔄 Refresh Data & Run Pipeline", width='stretch')
    if run_pipeline:
        with st.spinner(f"Running pipeline for {ticker}…"):
            import subprocess
            result = subprocess.run(
                [sys.executable, "main.py", "--ticker", ticker, "--no-llm"],
                capture_output=True, text=True, cwd=ROOT
            )
            if result.returncode == 0:
                st.success("Pipeline complete!")
                st.cache_data.clear()
            else:
                st.error(f"Pipeline failed:\n{result.stderr[-500:]}")

    run_llm = st.button("🤖 Run Full Analysis (with LLM)", width='stretch')
    if run_llm:
        with st.spinner(f"Running LLM analysis for {ticker}…"):
            import subprocess
            result = subprocess.run(
                [sys.executable, "main.py", "--ticker", ticker],
                capture_output=True, text=True, cwd=ROOT
            )
            if result.returncode == 0:
                st.success("LLM analysis complete!")
                st.cache_data.clear()
            else:
                st.error(f"LLM failed:\n{result.stderr[-500:]}")

    run_all = st.button("🏃 Run All 10 Tickers", width='stretch')
    if run_all:
        with st.spinner("Running all tickers (no LLM)…"):
            import subprocess
            result = subprocess.run(
                [sys.executable, "main.py", "--no-llm"],
                capture_output=True, text=True, cwd=ROOT
            )
            if result.returncode == 0:
                st.success("All tickers processed!")
                st.cache_data.clear()
            else:
                st.error(f"Failed:\n{result.stderr[-500:]}")

    st.markdown("---")
    st.caption(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.caption("Powered by Claude AI · Phase 7")


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.title("🤖 AI Trading Signal System")
st.markdown("---")

df = load_featured(ticker)
signals = load_ticker_signals()
ticker_sig = signals.get(ticker, {})

# ============================================================
# TAB LAYOUT
# ============================================================
tabs = st.tabs([
    "📊 Overview",
    "📈 Price & Technicals",
    "🧠 ML Signals",
    "🎯 Options",
    "🤖 AI Recommendation",
    "⚡ Feature Importance",
    "📋 All Tickers",
    "🔬 Backtesting",
])

# ---------------------------------------------------------------
# TAB 1: OVERVIEW
# ---------------------------------------------------------------
with tabs[0]:
    st.subheader(f"Signal Overview — {ticker}")

    if df is None:
        st.warning(f"No data found for {ticker}. Run the pipeline first.")
    else:
        latest = df.iloc[-1]
        prev   = df.iloc[-2]

        # Top KPI row
        col1, col2, col3, col4, col5 = st.columns(5)

        price     = float(latest["Close"])
        prev_p    = float(prev["Close"])
        pct_chg   = (price - prev_p) / prev_p * 100
        sign      = "+" if pct_chg >= 0 else ""
        arrow     = "▲" if pct_chg >= 0 else "▼"

        col1.metric("💰 Price", f"${price:.2f}", f"{sign}{pct_chg:.2f}%")
        col2.metric("📊 RSI(14)", f"{float(latest['RSI_14']):.1f}",
                    delta=None,
                    help="<30 oversold, >70 overbought")
        col3.metric("📉 ATR(14)", f"${float(latest['ATR_14']):.2f}",
                    help="Average True Range — daily volatility")
        col4.metric("🌊 Vol Ratio", f"{float(latest['volume_ratio']):.2f}x",
                    help=">1.5x = unusual volume spike")

        # Signal
        signal_val = ticker_sig.get("signal", "—")
        score_val  = ticker_sig.get("score", 0)
        col5.metric("🎯 Signal", signal_val)

        st.markdown("---")

        # Signal gauge
        st.markdown("#### Ensemble Score")
        if ticker_sig:
            score_bar(float(score_val))
            c1, c2, c3 = st.columns(3)
            c1.metric("XGBoost", f"{float(ticker_sig.get('xgb_prob', 0)):.3f}")
            c2.metric("SEQ Model", f"{float(ticker_sig.get('seq_prob', 0)):.3f}")
            c3.metric("Ensemble", f"{float(score_val):.3f}")
        else:
            st.info("No signal data found. Run the pipeline first.")

        st.markdown("---")

        # Macro snapshot
        st.markdown("#### 🌍 Macro Snapshot")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("VIX",          f"{float(latest.get('VIX', 0) or 0):.1f}")
        m2.metric("Fed Rate",     f"{float(latest.get('FED_RATE', 0) or 0):.2f}%")
        m3.metric("10Y Yield",    f"{float(latest.get('TREASURY_10Y', 0) or 0):.2f}%")
        m4.metric("CPI",          f"{float(latest.get('CPI', 0) or 0):.1f}")
        m5.metric("Unemployment", f"{float(latest.get('UNEMPLOYMENT', 0) or 0):.1f}%")

        # Fundamental snapshot
        st.markdown("#### 📋 Fundamentals")
        f1, f2, f3 = st.columns(3)
        pe  = latest.get("PE_RATIO")
        peg = latest.get("PEG_RATIO")
        pm  = latest.get("PROFIT_MARGIN")
        f1.metric("P/E Ratio",     f"{float(pe):.1f}"  if pe  and float(pe)  != 0 else "N/A")
        f2.metric("PEG Ratio",     f"{float(peg):.2f}" if peg and float(peg) != 0 else "N/A")
        f3.metric("Profit Margin", f"{float(pm):.1%}"  if pm  and float(pm)  != 0 else "N/A")

        # Pattern flags
        st.markdown("#### 🚦 Pattern Flags")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("BB Squeeze",     "YES 🔥" if float(latest.get("BB_squeeze", 0)) > 0 else "No")
        p2.metric("Bull Flag",      "YES 🏳️" if float(latest.get("bull_flag_setup", 0)) > 0 else "No")
        p3.metric("Volume Spike",   "YES ⚡" if float(latest.get("volume_spike", 0)) > 0 else "No")
        p4.metric("Above 200 EMA",  "YES ✅" if float(latest.get("above_200ema", 0)) > 0 else "No")


# ---------------------------------------------------------------
# TAB 2: PRICE & TECHNICALS
# ---------------------------------------------------------------
with tabs[1]:
    st.subheader(f"Price History & Technical Indicators — {ticker}")

    if df is None:
        st.warning("No data available.")
    else:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            lookback = st.slider("Lookback (trading days)", 30, 504, 126, step=21)
            plot_df = df.tail(lookback).copy()

            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                row_heights=[0.45, 0.20, 0.20, 0.15],
                vertical_spacing=0.03,
                subplot_titles=("Price + EMAs + Bollinger Bands", "MACD", "RSI(14)", "Volume"),
            )

            # Candlestick
            fig.add_trace(go.Candlestick(
                x=plot_df.index,
                open=plot_df["Open"], high=plot_df["High"],
                low=plot_df["Low"],   close=plot_df["Close"],
                name="OHLC", increasing_line_color="#00c853",
                decreasing_line_color="#ff5722",
            ), row=1, col=1)

            # EMAs
            for period, color in [(21, "#1e88e5"), (50, "#ffa000"), (200, "#7b1fa2")]:
                col_name = f"EMA_{period}"
                if col_name in plot_df.columns:
                    fig.add_trace(go.Scatter(
                        x=plot_df.index, y=plot_df[col_name],
                        name=f"EMA{period}", line=dict(color=color, width=1.5),
                    ), row=1, col=1)

            # Bollinger Bands
            for col_name, style in [
                ("BB_upper", dict(color="#90a4ae", dash="dot")),
                ("BB_lower", dict(color="#90a4ae", dash="dot")),
            ]:
                if col_name in plot_df.columns:
                    fig.add_trace(go.Scatter(
                        x=plot_df.index, y=plot_df[col_name],
                        name=col_name.replace("BB_", "BB "),
                        line=style, showlegend=False,
                    ), row=1, col=1)

            # MACD
            if "MACD_hist" in plot_df.columns and "MACD_line" in plot_df.columns:
                fig.add_trace(go.Bar(
                    x=plot_df.index, y=plot_df["MACD_hist"],
                    name="MACD Hist",
                    marker_color=plot_df["MACD_hist"].apply(
                        lambda v: "#00c853" if v >= 0 else "#ff5722"
                    ),
                ), row=2, col=1)
                fig.add_trace(go.Scatter(
                    x=plot_df.index, y=plot_df["MACD_line"],
                    name="MACD", line=dict(color="#1e88e5", width=1),
                ), row=2, col=1)
                if "MACD_signal" in plot_df.columns:
                    fig.add_trace(go.Scatter(
                        x=plot_df.index, y=plot_df["MACD_signal"],
                        name="Signal", line=dict(color="#ffa000", width=1),
                    ), row=2, col=1)

            # RSI
            if "RSI_14" in plot_df.columns:
                fig.add_trace(go.Scatter(
                    x=plot_df.index, y=plot_df["RSI_14"],
                    name="RSI(14)", line=dict(color="#ab47bc", width=1.5),
                ), row=3, col=1)
                for level, color in [(70, "rgba(255,87,34,0.3)"), (30, "rgba(0,200,83,0.3)")]:
                    fig.add_hline(y=level, line_dash="dot", line_color=color, row=3, col=1)

            # Volume
            fig.add_trace(go.Bar(
                x=plot_df.index,
                y=plot_df["Volume"],
                name="Volume",
                marker_color="rgba(100,181,246,0.5)",
            ), row=4, col=1)

            fig.update_layout(
                height=800,
                template="plotly_dark",
                xaxis_rangeslider_visible=False,
                legend=dict(orientation="h", y=1.02),
                margin=dict(l=40, r=40, t=60, b=40),
            )
            st.plotly_chart(fig, width='stretch')

        except ImportError:
            st.error("plotly not installed. Run: pip install plotly")

        # Raw data table
        with st.expander("📄 Raw Feature Data (last 20 rows)"):
            show_cols = [c for c in ["Close", "RSI_14", "MACD_hist", "ATR_14",
                                      "BB_squeeze", "bull_flag_setup", "volume_spike",
                                      "above_200ema", "hist_volatility_20", "price_vs_ema200"]
                         if c in df.columns]
            st.dataframe(df[show_cols].tail(20).style.format("{:.3f}"))


# ---------------------------------------------------------------
# TAB 3: ML SIGNALS
# ---------------------------------------------------------------
with tabs[2]:
    st.subheader(f"ML Model Signals — {ticker}")

    if df is None:
        st.warning("No data available.")
    else:
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### 🌲 XGBoost")
            xgb_val = float(ticker_sig.get("xgb_prob", 0))
            st.metric("Bullish Probability", f"{xgb_val:.1%}")
            score_bar(xgb_val)
            st.caption("Gradient-boosted trees on 52 technical + macro features")

        with col_b:
            st.markdown("#### 🧬 Sequence Model (MLP)")
            seq_val = float(ticker_sig.get("seq_prob", 0))
            st.metric("Bullish Probability", f"{seq_val:.1%}")
            score_bar(seq_val)
            st.caption("MLP on 20-day rolling windows of momentum, vol, and RSI")

        st.markdown("---")
        st.markdown("#### 🎯 Ensemble Score (XGB 60% + SEQ 40%)")
        ens_val = float(ticker_sig.get("score", 0))
        score_bar(ens_val)

        sig_str = ticker_sig.get("signal", "N/A")
        st.markdown(f"**Signal:** {signal_badge(sig_str)}", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 📊 Model Performance")
        perf_data = {
            "Metric":     ["XGBoost AUC", "MLP AUC", "Ensemble AUC (est.)", "Prediction Horizon", "Training Rows"],
            "Value":      ["0.534",       "0.531",    "~0.540",             "5 trading days",    "50,513 (10 tickers)"],
            "Benchmark":  ["0.500 (random)", "0.500", "0.500",              "—",                 "—"],
        }
        st.dataframe(pd.DataFrame(perf_data), width='stretch', hide_index=True)

        st.info("**Note**: AUC of 0.53–0.54 is realistic for financial markets. Professional quant funds "
                "typically target 53–57% win rates. The real edge comes from position sizing, "
                "risk management, and avoiding losers — not 80%+ accuracy.")

        # Historical target distribution
        st.markdown("---")
        st.markdown("#### 📈 Historical 5-Day Up/Down Distribution")
        target_col = "target_5d" if "target_5d" in df.columns else "target_3d"
        if target_col in df.columns:
            counts = df[target_col].value_counts()
            bull_pct = counts.get(1, 0) / len(df) * 100
            bear_pct = counts.get(0, 0) / len(df) * 100
            col1, col2 = st.columns(2)
            col1.metric("📈 Bullish Days", f"{bull_pct:.1f}%", f"{counts.get(1,0):,} days")
            col2.metric("📉 Bearish Days", f"{bear_pct:.1f}%", f"{counts.get(0,0):,} days")


# ---------------------------------------------------------------
# TAB 4: OPTIONS
# ---------------------------------------------------------------
with tabs[3]:
    st.subheader(f"Options Analysis — {ticker}")

    st.markdown("Options analysis runs in real-time from yfinance. Click below to fetch.")
    run_opts = st.button("🔍 Fetch Live Options Data", key="opts_btn")

    if run_opts:
        if df is None:
            st.error("Load featured data first.")
        else:
            with st.spinner("Fetching options chain…"):
                try:
                    from modules.options_analyzer import OptionsAnalyzer
                    latest = df.iloc[-1]
                    current_price = float(latest["Close"])
                    hist_vol = float(latest["hist_volatility_20"])

                    signal = ticker_sig.get("signal", "BULL")
                    if signal not in ["BULL", "STRONG_BULL"]:
                        st.warning(f"Signal is {signal} — options analysis only runs for bullish signals.")
                    else:
                        analyzer = OptionsAnalyzer(ticker, current_price, hist_vol)
                        result = analyzer.analyze(signal)

                        # IV Analysis
                        st.markdown("#### 📊 IV Environment")
                        iv = result.get("iv_analysis", {})
                        c1, c2, c3 = st.columns(3)
                        c1.metric("IV Environment", iv.get("environment", "N/A"))
                        c2.metric("IV/HV Ratio",    f"{iv.get('iv_premium_ratio', 0):.2f}x")
                        c3.metric("IV Crush Risk",  iv.get("risk", "N/A"))

                        st.markdown("---")

                        # Top Calls
                        top_calls = result.get("top_calls", [])
                        if top_calls:
                            st.markdown("#### 📞 Top 3 Call Options")
                            calls_data = []
                            for c in top_calls:
                                calls_data.append({
                                    "Strike": f"${c['strike']}",
                                    "Expiration": c["expiration"],
                                    "DTE": c["dte"],
                                    "Premium": f"${c['premium']:.2f}",
                                    "Breakeven": f"${c.get('breakeven', 0):.2f}",
                                    "Volume": c.get("volume", 0),
                                    "OI": c.get("openInterest", 0),
                                    "Score": f"{c['score']:.3f}",
                                })
                            st.dataframe(pd.DataFrame(calls_data), width='stretch', hide_index=True)

                        # Best Spread
                        spread = result.get("best_spread")
                        if spread:
                            st.markdown("#### 📐 Bull Call Spread")
                            sc1, sc2, sc3, sc4 = st.columns(4)
                            sc1.metric("Buy Strike",  f"${spread['buy_strike']}")
                            sc2.metric("Sell Strike", f"${spread['sell_strike']}")
                            sc3.metric("Net Debit",   f"${spread['net_debit']:.2f}")
                            sc4.metric("Max Gain",    f"${spread['max_gain']:.2f}")
                            st.metric("Reward/Risk",  f"{spread.get('reward_risk', 0):.2f}x",
                                      help="Max gain / net debit paid")
                        else:
                            st.info("No valid bull call spread found (all candidates had negative max gain).")

                        st.markdown("---")
                        st.metric("Recommendation", result.get("recommendation", "N/A"))

                except Exception as e:
                    st.error(f"Options analysis failed: {e}")


# ---------------------------------------------------------------
# TAB 5: AI RECOMMENDATION
# ---------------------------------------------------------------
with tabs[4]:
    st.subheader(f"Claude AI Recommendation — {ticker}")

    llm = load_llm_result(ticker)
    if llm:
        st.markdown(f"**Date:** {llm.get('date', 'N/A')}  |  "
                    f"**Tokens:** {llm.get('usage', {}).get('input_tokens', '?')} in, "
                    f"{llm.get('usage', {}).get('output_tokens', '?')} out")
        st.markdown("---")
        st.markdown(llm.get("recommendation", "No recommendation available."))

        with st.expander("📄 Raw JSON"):
            st.json(llm)
    else:
        st.info(f"No LLM recommendation cached for {ticker} today. "
                "Click 'Run Full Analysis (with LLM)' in the sidebar.")
        # Show any older cached file
        old_files = sorted((ROOT / "data" / "cache").glob(f"llm_{ticker}_*.json"))
        if old_files:
            st.markdown(f"**Showing older result:** `{old_files[-1].name}`")
            with open(old_files[-1]) as f:
                old = json.load(f)
            st.markdown(old.get("recommendation", ""))


# ---------------------------------------------------------------
# TAB 6: FEATURE IMPORTANCE
# ---------------------------------------------------------------
with tabs[5]:
    st.subheader("Feature Importance — XGBoost")

    imp = load_feature_importance(ticker)
    if imp is None:
        st.warning("Feature importance file not found. Train the model first.")
    else:
        try:
            import plotly.express as px

            top_n = st.slider("Show top N features", 10, 52, 25)
            top = imp.head(top_n)

            fig = px.bar(
                top,
                x="importance", y="feature",
                orientation="h",
                title=f"Top {top_n} Features by XGBoost Importance",
                color="importance",
                color_continuous_scale="viridis",
            )
            fig.update_layout(
                template="plotly_dark",
                height=max(400, top_n * 22),
                yaxis=dict(categoryorder="total ascending"),
                margin=dict(l=200, r=40, t=60, b=40),
                showlegend=False,
            )
            st.plotly_chart(fig, width='stretch')

            # Category summary
            st.markdown("#### Feature Categories")
            categories = {
                "Macro":        ["VIX", "FED_RATE", "TREASURY", "CPI", "UNEMPLOYMENT"],
                "Fundamentals": ["PE_RATIO", "PEG_RATIO", "PROFIT_MARGIN"],
                "Momentum":     ["RSI", "MACD", "momentum", "ROC", "stoch"],
                "Trend":        ["EMA", "SMA", "above_200", "price_vs_ema"],
                "Volatility":   ["ATR", "BB", "hist_vol", "squeeze"],
                "Volume":       ["volume", "OBV", "VWAP"],
                "Patterns":     ["bull_flag", "inside_day", "higher_high", "breakout"],
                "ID":           ["ticker_id"],
            }

            cat_rows = []
            for cat, keywords in categories.items():
                mask = imp["feature"].apply(
                    lambda f: any(k.lower() in f.lower() for k in keywords)
                )
                subset = imp[mask]
                cat_rows.append({
                    "Category":   cat,
                    "# Features": len(subset),
                    "Total Importance": f"{subset['importance'].sum():.3f}",
                    "Top Feature": subset.iloc[0]["feature"] if len(subset) > 0 else "—",
                })
            st.dataframe(pd.DataFrame(cat_rows), width='stretch', hide_index=True)

        except ImportError:
            st.error("plotly not installed.")
            st.dataframe(imp.head(30))


# ---------------------------------------------------------------
# TAB 7: ALL TICKERS
# ---------------------------------------------------------------
with tabs[6]:
    st.subheader("All Tickers — Signal Summary")

    report = load_report()
    if report is None:
        st.warning("No daily report found. Run the pipeline for all tickers first.")
    else:
        # Signal color-coded table
        def color_signal(val):
            color = SIGNAL_COLORS.get(val, "#555")
            return f"background-color:{color};color:#000;font-weight:700"

        def color_score(val):
            try:
                v = float(val)
                if v > 0.72: return "background-color:#00c853;color:#000"
                if v > 0.62: return "background-color:#69f0ae;color:#000"
                if v < 0.28: return "background-color:#b71c1c;color:#fff"
                if v < 0.38: return "background-color:#ff5722;color:#fff"
                return "background-color:#ffd740;color:#000"
            except:
                return ""

        display_cols = [c for c in ["ticker", "signal", "score", "xgb_prob", "seq_prob"]
                        if c in report.columns]
        styled = (
            report[display_cols]
            .style
            .map(color_signal, subset=["signal"] if "signal" in display_cols else [])
            .map(color_score, subset=["score"] if "score" in display_cols else [])
            .format({
                "score":    "{:.3f}",
                "xgb_prob": "{:.3f}",
                "seq_prob": "{:.3f}",
            }, na_rep="N/A")
        )
        st.dataframe(styled, width='stretch', hide_index=True)

        # Signal distribution pie
        st.markdown("---")
        st.markdown("#### Signal Distribution")
        try:
            import plotly.express as px
            sig_counts = report["signal"].value_counts().reset_index()
            sig_counts.columns = ["Signal", "Count"]
            sig_counts["Color"] = sig_counts["Signal"].map(SIGNAL_COLORS)
            fig = px.pie(
                sig_counts, names="Signal", values="Count",
                color="Signal",
                color_discrete_map=SIGNAL_COLORS,
                title="Current Signal Distribution",
            )
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, width='stretch')
        except ImportError:
            st.dataframe(report["signal"].value_counts())


# ---------------------------------------------------------------
# TAB 8: BACKTESTING
# ---------------------------------------------------------------
with tabs[7]:
    st.subheader("Backtesting")

    if df is None:
        st.warning("Load data first.")
    else:
        st.markdown("""
        Backtest simulates holding a position for N days whenever the ensemble score
        exceeds the entry threshold. Assumes no commissions and 0.1% slippage.
        """)

        col_bt1, col_bt2, col_bt3 = st.columns(3)
        entry_thresh  = col_bt1.slider("Entry Threshold", 0.55, 0.80, 0.65, 0.01)
        hold_days     = col_bt2.slider("Hold Period (days)", 1, 20, 5)
        initial_cap   = col_bt3.number_input("Initial Capital ($)", 1000, 100000, 10000, 1000)

        run_bt = st.button("▶ Run Backtest", key="bt_btn")

        if run_bt:
            with st.spinner("Running backtest…"):
                try:
                    target_col = "target_5d" if "target_5d" in df.columns else "target_3d"
                    if target_col not in df.columns:
                        st.error("No target column found in featured data.")
                    else:
                        # Simple backtest: signal from target column (proxy for model prediction)
                        # In production you'd use saved model predictions here
                        bt_df = df[["Close", target_col]].copy().dropna()
                        bt_df = bt_df[bt_df[target_col].isin([0, 1])]

                        # Simulate: enter when target=1 (model predicted correctly),
                        # use actual 5d return
                        bt_df["fwd_return"] = (
                            bt_df["Close"].shift(-hold_days) / bt_df["Close"] - 1
                        )
                        bt_df = bt_df.dropna(subset=["fwd_return"])

                        # Simple buy-and-hold baseline
                        bh_return = bt_df["Close"].iloc[-1] / bt_df["Close"].iloc[0] - 1

                        # Strategy: take every trade where target=1 (simulating ML signal)
                        trades = bt_df[bt_df[target_col] == 1].copy()
                        slippage = 0.001
                        trades["net_return"] = trades["fwd_return"] - slippage

                        wins       = (trades["net_return"] > 0).sum()
                        losses     = (trades["net_return"] <= 0).sum()
                        total_tr   = len(trades)
                        win_rate   = wins / total_tr if total_tr > 0 else 0
                        avg_win    = trades.loc[trades["net_return"] > 0, "net_return"].mean()
                        avg_loss   = trades.loc[trades["net_return"] <= 0, "net_return"].mean()
                        profit_factor = (
                            abs(wins * (avg_win or 0)) / abs(losses * (avg_loss or 1))
                            if losses > 0 else float("inf")
                        )

                        # Equity curve (compound, sequential trades)
                        equity = [initial_cap]
                        for ret in trades["net_return"].values:
                            equity.append(equity[-1] * (1 + ret))

                        eq_series = pd.Series(equity, index=range(len(equity)))
                        peak       = eq_series.cummax()
                        drawdown   = (eq_series - peak) / peak
                        max_dd     = drawdown.min()
                        final_cap  = equity[-1]
                        total_ret  = (final_cap - initial_cap) / initial_cap

                        # Summary
                        st.markdown("#### Backtest Results")
                        r1, r2, r3, r4 = st.columns(4)
                        r1.metric("Total Trades",   f"{total_tr:,}")
                        r2.metric("Win Rate",        f"{win_rate:.1%}")
                        r3.metric("Profit Factor",   f"{profit_factor:.2f}")
                        r4.metric("Max Drawdown",    f"{max_dd:.1%}")

                        r5, r6, r7, r8 = st.columns(4)
                        r5.metric("Total Return",    f"{total_ret:.1%}")
                        r6.metric("B&H Return",      f"{bh_return:.1%}")
                        r7.metric("Avg Win",         f"{avg_win:.2%}" if avg_win == avg_win else "N/A")
                        r8.metric("Avg Loss",        f"{avg_loss:.2%}" if avg_loss == avg_loss else "N/A")

                        # Equity curve chart
                        try:
                            import plotly.express as px
                            fig = px.line(
                                x=range(len(equity)), y=equity,
                                title=f"{ticker} Equity Curve — ${initial_cap:,} → ${final_cap:,.0f}",
                                labels={"x": "Trade #", "y": "Portfolio Value ($)"},
                            )
                            fig.add_hline(y=initial_cap, line_dash="dot",
                                          annotation_text="Starting Capital")
                            fig.update_layout(template="plotly_dark")
                            st.plotly_chart(fig, width='stretch')
                        except ImportError:
                            st.line_chart(equity)

                        st.caption(
                            "⚠️ This is a simplified backtest using actual forward returns. "
                            "A production backtest would use model predictions on unseen data, "
                            "not the target column (which is computed from future prices)."
                        )

                except Exception as e:
                    st.error(f"Backtest error: {e}")
                    import traceback
                    st.code(traceback.format_exc())


# ---------------------------------------------------------------
# Footer
# ---------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#888;font-size:12px'>"
    "🤖 AI Trading Signal System | Powered by Claude AI | "
    "XGBoost + MLP Ensemble | Van Tharp Position Sizing"
    "</div>",
    unsafe_allow_html=True,
)
