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
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import yaml

# ---------------------------------------------------------------------------
# Resolve paths regardless of working directory
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent          # trading_ai/
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)


# ---------------------------------------------------------------------------
# Config helpers — read/write config.yaml so all modules stay in sync
# ---------------------------------------------------------------------------
CONFIG_PATH = ROOT / "config" / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def save_config(cfg: dict) -> None:
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def get_watchlist() -> list[str]:
    return load_config().get("watchlist", [])


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
def _run_cmd(cmd: list, label: str, success_msg: str) -> None:
    """Run a subprocess command and show live log output in an expander."""
    with st.spinner(f"{label}…"):
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=ROOT
        )
    if result.returncode == 0:
        st.success(success_msg)
        st.cache_data.clear()
    else:
        st.error(f"{label} failed (exit {result.returncode})")
    # Show combined stdout+stderr in a collapsible log
    log_text = (result.stdout or "") + (result.stderr or "")
    if log_text.strip():
        with st.expander("📋 View log output", expanded=result.returncode != 0):
            st.code(log_text[-4000:], language="text")   # last 4000 chars


with st.sidebar:
    st.image("https://img.shields.io/badge/AI%20Trading-Signal%20System-blue?style=for-the-badge")
    st.markdown("---")

    # ── Ticker selector — reads live from config.yaml ─────────────────────
    all_tickers = get_watchlist()
    if not all_tickers:
        st.warning("Watchlist is empty. Add tickers in the ⚙️ Manage Watchlist tab.")
        all_tickers = ["NVDA"]
    ticker = st.selectbox("🔍 Select Ticker", all_tickers, index=0)

    st.markdown("---")
    
    # ── Account Size for Position Sizing ─────────────────────────────────
    st.markdown("### 💰 Account Settings")
    account_size = st.number_input(
        "Account Size ($)",
        min_value=1000,
        max_value=10000000,
        value=10000,
        step=1000,
        help="Your total trading account size for position sizing calculations"
    )
    risk_per_trade = st.slider(
        "Risk Per Trade (%)",
        min_value=0.5,
        max_value=5.0,
        value=1.5,
        step=0.5,
        help="Maximum % of account to risk on a single trade (1-2% is conservative)"
    )
    
    st.markdown("---")
    st.markdown("### ⚡ Quick Actions")

    # ── Per-ticker actions ─────────────────────────────────────────────────
    st.markdown("**Single Ticker**")

    if st.button("📥 Pull Data Only", key="pull_data", use_container_width=True,
                 help="Runs run_pipeline.py for this ticker — fetches fresh OHLCV, macro, fundamentals"):
        _run_cmd(
            [sys.executable, "run_pipeline.py", "--tickers", ticker],
            f"Pulling data for {ticker}",
            f"Data pulled for {ticker}!",
        )

    if st.button("🔄 Analyze (No LLM)", key="run_ticker_nollm", use_container_width=True,
                 help="Runs main.py --ticker {ticker} --no-llm — fast signal refresh"):
        _run_cmd(
            [sys.executable, "main.py", "--ticker", ticker, "--no-llm"],
            f"Analyzing {ticker}",
            f"Signal updated for {ticker}!",
        )

    if st.button("🤖 Analyze + LLM", key="run_ticker_llm", use_container_width=True,
                 help="Runs main.py --ticker {ticker} — full pipeline with Claude AI recommendation"):
        _run_cmd(
            [sys.executable, "main.py", "--ticker", ticker],
            f"LLM analysis for {ticker}",
            f"LLM recommendation generated for {ticker}!",
        )

    st.markdown("---")
    st.markdown("**All Watchlist Tickers**")
    
    if st.button("🔄 DAILY REFRESH (Full Pipeline)", key="daily_refresh", use_container_width=True,
                 type="primary",
                 help="Complete workflow: Pull data → Retrain model → Generate signals with LLM"):
        with st.spinner("Running full pipeline..."):
            # Step 1: Pull all data
            st.info("Step 1/3: Pulling fresh data...")
            result1 = subprocess.run(
                [sys.executable, "run_pipeline.py"],
                cwd=str(ROOT),
                capture_output=True,
                text=True
            )
            if result1.returncode != 0:
                st.error(f"Data pull failed: {result1.stderr[:500]}")
            else:
                st.success("✓ Data pulled")
            
            # Step 2: Retrain model
            st.info("Step 2/3: Retraining model...")
            result2 = subprocess.run(
                [sys.executable, "-m", "modules.ml_models"],
                cwd=str(ROOT),
                capture_output=True,
                text=True
            )
            if result2.returncode != 0:
                st.error(f"Model training failed: {result2.stderr[:500]}")
            else:
                st.success("✓ Model retrained")
            
            # Step 3: Generate signals
            st.info("Step 3/3: Generating AI recommendations...")
            result3 = subprocess.run(
                [sys.executable, "main.py"],
                cwd=str(ROOT),
                capture_output=True,
                text=True
            )
            if result3.returncode != 0:
                st.error(f"Signal generation failed: {result3.stderr[:500]}")
            else:
                st.success("✅ Daily refresh complete!")
                st.info(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M ET')}")
                st.rerun()
    
    st.markdown("---")

    if st.button("📥 Pull All Data", key="pull_all_data", use_container_width=True,
                 help="Runs run_pipeline.py for every ticker in your watchlist"):
        _run_cmd(
            [sys.executable, "run_pipeline.py"],
            "Pulling data for all tickers",
            "All data pulled!",
        )

    if st.button("🏃 Analyze All (No LLM)", key="run_all_nollm", use_container_width=True,
                 help="Runs main.py --no-llm — signals for all watchlist tickers"):
        _run_cmd(
            [sys.executable, "main.py", "--no-llm"],
            f"Analyzing all {len(all_tickers)} tickers",
            "All tickers analyzed!",
        )

    if st.button("🤖 Analyze All + LLM", key="run_all_llm", use_container_width=True,
                 help="Runs main.py — full pipeline with LLM for every ticker (slow — uses API credits)"):
        _run_cmd(
            [sys.executable, "main.py"],
            f"Full LLM analysis for all {len(all_tickers)} tickers",
            "Full analysis complete!",
        )

    st.markdown("---")
    st.markdown("**Model Training**")

    if st.button("🧠 Retrain Model", key="retrain", use_container_width=True,
                 help="Runs python -m modules.ml_models — retrains XGBoost + MLP on all parquet files"):
        _run_cmd(
            [sys.executable, "-m", "modules.ml_models"],
            "Retraining combined model",
            "Model retrained!",
        )

    if st.button("🔬 Retrain + Optuna Tune", key="retrain_tune", use_container_width=True,
                 help="Retrains with Optuna hyperparameter search (~5-10 min). Uses --tune --tune-trials 30"):
        _run_cmd(
            [sys.executable, "-m", "modules.ml_models", "--tune", "--tune-trials", "30"],
            "Retraining with Optuna tuning (this takes ~5-10 min)",
            "Model retrained with tuned hyperparameters!",
        )

    if st.button("🗑️ Clean Stale Models", key="clean_models", use_container_width=True,
                 help="Runs main.py --delete-models --no-llm --ticker NVDA — removes per-ticker model files"):
        _run_cmd(
            [sys.executable, "main.py", "--delete-models", "--ticker", ticker, "--no-llm"],
            "Cleaning stale per-ticker model files",
            "Stale models cleaned!",
        )

    st.markdown("---")
    st.caption(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.caption(f"Watchlist: {len(all_tickers)} tickers · Powered by Claude AI")


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
    "📊 Performance & Backtest",
    "📈 Live Track Record",
    "⚡ Feature Importance",
    "📋 All Tickers",
    "⚙️ Manage Watchlist",
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
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("VIX",          f"{float(latest.get('VIX', 0) or 0):.1f}")
        m2.metric("Fed Rate",     f"{float(latest.get('FED_RATE', 0) or 0):.2f}%")
        m3.metric("10Y Yield",    f"{float(latest.get('TREASURY_10Y', 0) or 0):.2f}%")
        m4.metric("CPI",          f"{float(latest.get('CPI', 0) or 0):.1f}")
        m5.metric("Unemployment", f"{float(latest.get('UNEMPLOYMENT', 0) or 0):.1f}%")
        fg = float(latest.get('FEAR_GREED_INDEX', 50) or 50)
        fg_label = "Extr.Fear" if fg < 25 else "Fear" if fg < 45 else "Neutral" if fg < 55 else "Greed" if fg < 75 else "Extr.Greed"
        m6.metric("Fear & Greed", f"{fg:.0f} — {fg_label}")

        # VIX term structure
        vix30 = float(latest.get('VIX', 0) or 0)
        vix3m = float(latest.get('VIX3M', 0) or 0)
        slope = float(latest.get('vix_term_slope', 0) or 0)
        if vix3m > 0:
            st.markdown("#### 📐 VIX Term Structure")
            vt1, vt2, vt3 = st.columns(3)
            vt1.metric("VIX (30-Day)",  f"{vix30:.1f}")
            vt2.metric("VIX 3-Month",   f"{vix3m:.1f}")
            vt3.metric("Slope (30D-3M)", f"{slope:+.2f}",
                       help="Negative = normal contango/calm, Positive = inverted/fear spike")

        # Fundamental snapshot
        st.markdown("#### 📋 Fundamentals")
        f1, f2, f3, f4, f5 = st.columns(5)
        pe   = latest.get("PE_RATIO")
        peg  = latest.get("PEG_RATIO")
        pm   = latest.get("PROFIT_MARGIN")
        rg   = latest.get("REVENUE_GROWTH_YOY")
        es   = latest.get("EARNINGS_SURPRISE_PCT")
        f1.metric("P/E Ratio",      f"{float(pe):.1f}"   if pe  and float(pe)  != 0 else "N/A")
        f2.metric("PEG Ratio",      f"{float(peg):.2f}"  if peg and float(peg) != 0 else "N/A")
        f3.metric("Profit Margin",  f"{float(pm):.1%}"   if pm  and float(pm)  != 0 else "N/A")
        f4.metric("Revenue Growth", f"{float(rg):+.1%}"  if rg  and float(rg)  != 0 else "N/A")
        f5.metric("EPS Surprise",   f"{float(es):+.1%}"  if es  and float(es)  != 0 else "N/A")

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

            # MACD — feature engineering creates 'MACD' (not 'MACD_line')
            if "MACD_hist" in plot_df.columns and "MACD" in plot_df.columns:
                fig.add_trace(go.Bar(
                    x=plot_df.index, y=plot_df["MACD_hist"],
                    name="MACD Hist",
                    marker_color=plot_df["MACD_hist"].apply(
                        lambda v: "#00c853" if v >= 0 else "#ff5722"
                    ),
                ), row=2, col=1)
                fig.add_trace(go.Scatter(
                    x=plot_df.index, y=plot_df["MACD"],
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
                        
                        # ── Position Sizing Calculator ─────────────────────────────
                        st.markdown("#### 💰 Position Sizing (Based on Your Account)")
                        
                        max_risk_dollars = account_size * (risk_per_trade / 100)
                        st.info(f"**Risk Budget:** ${max_risk_dollars:,.0f} ({risk_per_trade}% of ${account_size:,.0f})")
                        
                        # For options: recommend 2% max position size
                        max_position_size = account_size * 0.02
                        
                        # Calculate number of contracts based on top call
                        if top_calls:
                            best_call = top_calls[0]
                            premium = best_call.get('premium', 0)
                            contracts_per_lot = 100  # Standard option contract
                            
                            if premium > 0:
                                # Max contracts based on 2% position size rule
                                max_contracts = int(max_position_size / (premium * contracts_per_lot))
                                total_cost = max_contracts * premium * contracts_per_lot
                                pct_of_account = (total_cost / account_size) * 100
                                
                                st.success(f"""
**Recommended Position:**
- **Buy {max_contracts} contract(s)** of ${best_call['strike']} call expiring {best_call['expiration']}
- **Cost:** ${total_cost:,.0f} ({pct_of_account:.1f}% of account)
- **Premium per contract:** ${premium:.2f} × 100 shares = ${premium * contracts_per_lot:,.0f}
- **Breakeven:** ${best_call.get('breakeven', 0):.2f} (stock must be above this at expiry)
                                """)
                                
                                st.warning("""
**Risk Management:**
- This position risks ${total_cost:,.0f} (max loss if worthless at expiry)
- Set stop loss at 50% premium loss: Exit if option drops to ${(premium * 0.5):.2f}
- Take profits at 100% gain: Exit if option doubles to ${(premium * 2):.2f}
                                """.replace("${total_cost:,.0f}", f"${total_cost:,.0f}").replace("${(premium * 0.5):.2f}", f"${(premium * 0.5):.2f}").replace("${(premium * 2):.2f}", f"${(premium * 2):.2f}"))
                        
                        # For spread
                        if spread:
                            net_debit = spread.get('net_debit', 0)
                            if net_debit > 0:
                                spread_contracts = int(max_position_size / (net_debit * 100))
                                spread_cost = spread_contracts * net_debit * 100
                                spread_pct = (spread_cost / account_size) * 100
                                
                                st.info(f"""
**Alternative: Bull Call Spread**
- **Buy {spread_contracts} spread(s)**: Buy ${spread['buy_strike']}, Sell ${spread['sell_strike']}
- **Cost:** ${spread_cost:,.0f} ({spread_pct:.1f}% of account)
- **Max Gain:** ${spread_contracts * spread.get('max_gain', 0):,.0f}
- **Risk/Reward:** {spread.get('reward_risk', 0):.2f}x
                                """)
                        
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
    
    # ── Chat Interface for Follow-up Questions ────────────────────────────
    st.markdown("---")
    st.subheader("💬 Ask Follow-up Questions")
    
    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a follow-up question about this recommendation..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    from anthropic import Anthropic
                    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                    
                    # Build context with current recommendation
                    context = f"""You are a professional trading assistant. The user is viewing a trading recommendation for {ticker}.
                    
Current Recommendation:
{llm.get("recommendation", "No recommendation available") if llm else "No recommendation generated yet"}

Current Signal: {ticker_sig.get('signal', 'N/A')}
ML Score: {ticker_sig.get('score', 0):.3f}

Answer the user's follow-up question based on this context. Be concise and actionable."""
                    
                    # Create message history for Claude
                    messages = [{"role": "user", "content": context}]
                    for msg in st.session_state.chat_history[-10:]:  # Last 10 messages
                        messages.append({"role": msg["role"], "content": msg["content"]})
                    
                    response = client.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=1000,
                        messages=messages
                    )
                    
                    assistant_message = response.content[0].text
                    st.markdown(assistant_message)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": assistant_message
                    })
                    
                except Exception as e:
                    st.error(f"Chat failed: {e}")
                    st.info("Make sure ANTHROPIC_API_KEY is set in your .env file")


# ---------------------------------------------------------------
# TAB 6: PERFORMANCE & BACKTEST
# ---------------------------------------------------------------
with tabs[5]:
    st.subheader("📊 Performance & Backtest Results")
    
    # Check if backtest results exist
    backtest_summary_path = f"outputs/backtests/{ticker}_summary.csv"
    backtest_trades_path = f"outputs/backtests/{ticker}_trades.csv"
    
    if not os.path.exists(backtest_summary_path):
        st.info(f"No backtest results available for {ticker}")
        st.markdown("""
        **To generate backtest results:**
        ```bash
        cd trading_ai
        source ../.venv/bin/activate
        python -m modules.backtester
        ```
        
        This will run walk-forward validation on historical data and generate:
        - Win rate
        - Sharpe ratio
        - Max drawdown
        - Profit factor
        - Trade-level analysis
        """)
    else:
        # Load backtest summary
        summary = pd.read_csv(backtest_summary_path).iloc[0].to_dict()
        
        # Display key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", f"{summary.get('total_trades', 0):.0f}")
            st.metric("Win Rate", f"{summary.get('win_rate', 0):.1f}%")
        
        with col2:
            st.metric("Avg Return", f"{summary.get('avg_return', 0):.2f}%")
            st.metric("Total Return", f"{summary.get('total_return', 0):.2f}%")
        
        with col3:
            st.metric("Sharpe Ratio", f"{summary.get('sharpe_ratio', 0):.2f}")
            st.metric("Max Drawdown", f"{summary.get('max_drawdown', 0):.2f}%")
        
        with col4:
            st.metric("Profit Factor", f"{summary.get('profit_factor', 0):.2f}")
            st.metric("Calmar Ratio", f"{summary.get('calmar_ratio', 0):.2f}")
        
        st.markdown("---")
        
        # Load and display trade history
        if os.path.exists(backtest_trades_path):
            trades = pd.read_csv(backtest_trades_path)
            
            st.subheader("Trade History")
            
            # Create equity curve
            trades['cumulative_return'] = (1 + trades['net_return_pct']/100).cumprod() - 1
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trades['date'],
                y=trades['cumulative_return'] * 100,
                mode='lines',
                name='Equity Curve',
                line=dict(color='green' if trades['cumulative_return'].iloc[-1] > 0 else 'red', width=2)
            ))
            fig.update_layout(
                title="Cumulative Return Over Time",
                xaxis_title="Date",
                yaxis_title="Cumulative Return (%)",
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Trade table
            st.subheader("Individual Trades")
            display_trades = trades[['date', 'signal', 'final_score', 'entry_price', 
                                     'actual_direction', 'net_return_pct', 'correct']].copy()
            display_trades['net_return_pct'] = display_trades['net_return_pct'].round(2)
            display_trades['final_score'] = display_trades['final_score'].round(3)
            display_trades['entry_price'] = display_trades['entry_price'].round(2)
            
            st.dataframe(display_trades, use_container_width=True)
        
        # Performance interpretation
        st.markdown("---")
        st.subheader("Performance Interpretation")
        
        win_rate = summary.get('win_rate', 0)
        sharpe = summary.get('sharpe_ratio', 0)
        profit_factor = summary.get('profit_factor', 0)
        
        if win_rate > 55 and sharpe > 1.0 and profit_factor > 1.5:
            st.success("✅ **Strong Performance** - System shows consistent edge")
        elif win_rate > 45 and sharpe > 0.5 and profit_factor > 1.0:
            st.warning("⚠️ **Modest Performance** - System has slight edge, needs refinement")
        else:
            st.error("🔴 **Poor Performance** - System needs improvement or more data")
        
        st.markdown(f"""
        **Metrics Explained:**
        - **Win Rate**: {win_rate:.1f}% (target: >50% for long-only strategy)
        - **Sharpe Ratio**: {sharpe:.2f} (target: >1.0 for good risk-adjusted returns)
        - **Profit Factor**: {profit_factor:.2f} (target: >1.5, meaning wins are 1.5x losses)
        - **Max Drawdown**: {summary.get('max_drawdown', 0):.1f}% (max decline from peak)
        
        **Note**: These results are from walk-forward validation on historical data.
        Past performance does not guarantee future results.
        """)


# ---------------------------------------------------------------
# TAB 7: LIVE TRACK RECORD (Forward Validation)
# ---------------------------------------------------------------
with tabs[6]:
    st.subheader("📈 Live Track Record — Real Prediction Accuracy")
    
    st.info("""
    **This tracks YOUR ACTUAL predictions and outcomes over time.**
    
    Unlike backtesting (which uses old data), this shows:
    - What you predicted on each date
    - What actually happened 5 days later
    - Your real win rate in current market conditions
    """)
    
    prediction_log_path = ROOT / "outputs" / "prediction_log.csv"
    
    if not prediction_log_path.exists():
        st.warning("No predictions logged yet. Predictions are automatically logged each time you run `main.py`.")
        st.markdown("""
        **To start tracking:**
        1. Run daily analysis: `python main.py` or click sidebar buttons
        2. After 5 days, validate outcomes: `python validate_predictions.py`
        3. Come back here to see your real track record
        
        **Validation Schedule (Recommended):**
        - Run `python validate_predictions.py` daily after 4 PM ET
        - Or set up a cron job to auto-validate
        """)
    else:
        # Load prediction log
        pred_log = pd.read_csv(prediction_log_path)
        pred_log['prediction_date'] = pd.to_datetime(pred_log['prediction_date'])
        
        # Overall stats
        st.markdown("### 📊 Overall Performance")
        
        total_preds = len(pred_log)
        validated = pred_log[pred_log['outcome_verified'] == True]
        pending = total_preds - len(validated)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Predictions", total_preds)
        col2.metric("Validated", len(validated))
        col3.metric("Pending", pending, help="< 5 days old, waiting for outcome")
        
        if len(validated) > 0:
            win_rate = (validated['prediction_correct'].sum() / len(validated)) * 100
            col4.metric("Win Rate", f"{win_rate:.1f}%")
            
            st.markdown("---")
            
            # Performance by time period
            st.markdown("### 📅 Performance Over Time")
            
            periods = {
                "Last 7 Days": 7,
                "Last 30 Days": 30,
                "Last 90 Days": 90,
                "All Time": 9999
            }
            
            period_stats = []
            for period_name, days in periods.items():
                cutoff = datetime.now() - timedelta(days=days)
                period_validated = validated[validated['prediction_date'] >= cutoff]
                
                if len(period_validated) > 0:
                    period_win_rate = (period_validated['prediction_correct'].sum() / len(period_validated)) * 100
                    period_avg_return = period_validated['actual_return_pct'].mean()
                    
                    period_stats.append({
                        'Period': period_name,
                        'Trades': len(period_validated),
                        'Win Rate': f"{period_win_rate:.1f}%",
                        'Avg Return': f"{period_avg_return:+.2f}%"
                    })
            
            if period_stats:
                st.dataframe(pd.DataFrame(period_stats), use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            # Performance by confidence level
            st.markdown("### 🎯 Accuracy by Confidence Level")
            
            confidence_stats = validated.groupby('confidence_level').agg({
                'prediction_correct': ['sum', 'count', 'mean'],
                'actual_return_pct': 'mean'
            }).round(2)
            
            if not confidence_stats.empty:
                confidence_display = pd.DataFrame({
                    'Confidence': confidence_stats.index,
                    'Trades': confidence_stats[('prediction_correct', 'count')].values,
                    'Win Rate (%)': (confidence_stats[('prediction_correct', 'mean')] * 100).round(1).values,
                    'Avg Return (%)': confidence_stats[('actual_return_pct', 'mean')].round(2).values
                })
                st.dataframe(confidence_display, use_container_width=True, hide_index=True)
                
                st.caption("**Key Insight:** High confidence signals should have >60% win rate. If not, model is miscalibrated.")
            
            st.markdown("---")
            
            # Performance by ticker
            st.markdown("### 📊 Performance by Ticker")
            
            ticker_stats = validated.groupby('ticker').agg({
                'prediction_correct': ['sum', 'count', 'mean'],
                'actual_return_pct': 'mean'
            }).round(2)
            
            if not ticker_stats.empty:
                ticker_display = pd.DataFrame({
                    'Ticker': ticker_stats.index,
                    'Trades': ticker_stats[('prediction_correct', 'count')].values,
                    'Win Rate (%)': (ticker_stats[('prediction_correct', 'mean')] * 100).round(1).values,
                    'Avg Return (%)': ticker_stats[('actual_return_pct', 'mean')].round(2).values
                })
                # Sort by win rate descending
                ticker_display = ticker_display.sort_values('Win Rate (%)', ascending=False)
                st.dataframe(ticker_display, use_container_width=True, hide_index=True)
                
                st.caption("**Key Insight:** Focus on tickers where model consistently outperforms. Avoid tickers with <45% win rate.")
            
            st.markdown("---")
            
            # Recent predictions (last 10)
            st.markdown("### 📋 Recent Predictions")
            recent = pred_log.nlargest(10, 'prediction_date')
            display_recent = recent[['prediction_date', 'ticker', 'signal', 'final_score', 
                                     'entry_price', 'outcome_verified', 'actual_return_pct', 'prediction_correct']].copy()
            display_recent['final_score'] = display_recent['final_score'].round(3)
            display_recent['entry_price'] = display_recent['entry_price'].round(2)
            display_recent['actual_return_pct'] = display_recent['actual_return_pct'].fillna(0).round(2)
            st.dataframe(display_recent, use_container_width=True, hide_index=True)
            
            st.caption("**Note:** Predictions <5 days old show 'outcome_verified = False' (still pending)")
        else:
            st.info("Predictions logged, but no outcomes validated yet (need 5+ days). Run `python validate_predictions.py`")


# ---------------------------------------------------------------
# TAB 8: FEATURE IMPORTANCE
# ---------------------------------------------------------------
with tabs[7]:
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
# TAB 9: ALL TICKERS
# ---------------------------------------------------------------
with tabs[8]:
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
# TAB 10: MANAGE WATCHLIST
# ---------------------------------------------------------------
with tabs[9]:
    st.subheader("⚙️ Manage Watchlist")
    st.markdown(
        "Add or remove tickers from your watchlist. Changes are written to "
        "`config/config.yaml` and immediately reflected across all modules "
        "(main.py, run_pipeline.py, ml_models.py) and this dashboard."
    )

    # Re-read live so this tab always shows the current state
    current_watchlist = get_watchlist()

    col_left, col_right = st.columns([3, 2])

    with col_left:
        # ── Current watchlist with remove buttons ──────────────────────────
        st.markdown("### 📋 Current Watchlist")
        if not current_watchlist:
            st.info("No tickers in watchlist. Add some below.")
        else:
            # Show each ticker with its data status and a remove button
            for i, t in enumerate(current_watchlist):
                parquet_ok = (ROOT / "data" / "processed" / f"{t}_featured.parquet").exists()
                model_ok   = (ROOT / "models" / "xgboost" / "NVDA.pkl").exists()
                data_icon  = "✅" if parquet_ok else "⚠️"
                data_tip   = "Data ready" if parquet_ok else "No data — pull data after adding"

                col_t, col_status, col_btn = st.columns([3, 2, 1])
                col_t.markdown(f"**{t}**")
                col_status.markdown(
                    f'<span title="{data_tip}">{data_icon} {"Ready" if parquet_ok else "No data"}</span>',
                    unsafe_allow_html=True,
                )
                if col_btn.button("➖", key=f"remove_{t}_{i}", help=f"Remove {t} from watchlist"):
                    cfg = load_config()
                    cfg["watchlist"] = [x for x in cfg["watchlist"] if x != t]
                    save_config(cfg)
                    st.success(f"Removed **{t}** from watchlist.")
                    st.cache_data.clear()
                    st.rerun()

        st.markdown(f"**Total: {len(current_watchlist)} tickers**")

    with col_right:
        # ── Add new ticker ─────────────────────────────────────────────────
        st.markdown("### ➕ Add Ticker")
        new_ticker = st.text_input(
            "Ticker symbol",
            placeholder="e.g. NVDA, MSFT, BTC-USD",
            max_chars=12,
            key="new_ticker_input",
        ).strip().upper()

        validate_first = st.checkbox(
            "Validate with yfinance before adding",
            value=True,
            help="Checks that yfinance can find price data for this ticker. Uncheck for crypto or ETFs that may have unusual symbols.",
        )

        col_add1, col_add2 = st.columns(2)
        add_btn = col_add1.button("➕ Add to Watchlist", type="primary", use_container_width=True)
        pull_after = col_add2.checkbox("Pull data immediately after adding", value=True)

        if add_btn and new_ticker:
            if new_ticker in current_watchlist:
                st.warning(f"**{new_ticker}** is already in your watchlist.")
            else:
                # Optional yfinance validation
                valid = True
                if validate_first:
                    with st.spinner(f"Validating {new_ticker} with yfinance…"):
                        try:
                            import yfinance as yf
                            info = yf.Ticker(new_ticker).fast_info
                            last_price = getattr(info, "last_price", None)
                            if last_price is None or last_price == 0:
                                hist = yf.Ticker(new_ticker).history(period="5d")
                                valid = not hist.empty
                            else:
                                valid = True
                        except Exception:
                            valid = False

                if not valid:
                    st.error(
                        f"❌ **{new_ticker}** could not be validated via yfinance. "
                        "Check the symbol or uncheck 'Validate' to add anyway."
                    )
                else:
                    # Write to config.yaml
                    cfg = load_config()
                    cfg["watchlist"].append(new_ticker)
                    save_config(cfg)
                    st.success(f"✅ Added **{new_ticker}** to watchlist.")
                    st.cache_data.clear()

                    if pull_after:
                        _run_cmd(
                            [sys.executable, "run_pipeline.py", "--tickers", new_ticker],
                            f"Pulling data for {new_ticker}",
                            f"Data ready for {new_ticker}! You can now run analysis.",
                        )

                    st.rerun()

        st.markdown("---")
        # ── Bulk operations ────────────────────────────────────────────────
        st.markdown("### 🔧 Bulk Operations")

        if st.button("📥 Pull Data for All Missing Tickers", use_container_width=True,
                     help="Runs run_pipeline.py --skip-existing — only processes tickers without data files"):
            _run_cmd(
                [sys.executable, "run_pipeline.py", "--skip-existing"],
                "Pulling data for tickers with missing parquet files",
                "Done! All tickers with missing data have been refreshed.",
            )

        if st.button("🔄 Refresh All Data (Force)", use_container_width=True,
                     help="Runs run_pipeline.py for ALL tickers — overwrites existing data"):
            _run_cmd(
                [sys.executable, "run_pipeline.py"],
                f"Force-refreshing data for all {len(current_watchlist)} tickers",
                "All data refreshed!",
            )

        if st.button("🧠 Retrain Model (All Tickers)", use_container_width=True,
                     help="After adding/removing tickers, retrain the combined model to include new data"):
            _run_cmd(
                [sys.executable, "-m", "modules.ml_models"],
                "Retraining combined model on all watchlist tickers",
                "Model retrained with updated watchlist!",
            )

        st.markdown("---")
        # ── Config preview ─────────────────────────────────────────────────
        st.markdown("### 📄 Current config.yaml Watchlist")
        st.code("\n".join(f"- {t}" for t in current_watchlist), language="yaml")

        st.download_button(
            "💾 Download config.yaml",
            data=CONFIG_PATH.read_text(),
            file_name="config.yaml",
            mime="text/yaml",
            use_container_width=True,
        )


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
