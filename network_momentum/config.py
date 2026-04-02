"""
config.py — Configuration for Network Momentum replication
Paper: "Network Momentum across Asset Classes" (Pu, Roberts, Dong, Zohren, 2023)
arXiv: 2308.11294
"""

# ─── Data ────────────────────────────────────────────────────────────────────
DATA_START = "1995-01-01"      # fetch start (need burn-in before backtest)
BACKTEST_START = "2000-01-01"  # OOS period start
BACKTEST_END   = "2022-12-31"  # OOS period end

# ─── Asset universe (Yahoo Finance tickers, grouped by class) ─────────────────
# Note: paper uses Pinnacle Data continuous futures; these are ETF/index proxies.
# Replace with actual continuous futures prices via CSV if available.
ASSETS = {
    "Commodity": [
        "GLD",   # Gold
        "SLV",   # Silver
        "USO",   # WTI Crude Oil
        "UNG",   # Natural Gas
        "DBA",   # Agriculture (broad basket)
        "DBB",   # Base Metals
        "CORN",  # Corn
        "WEAT",  # Wheat
        "SOYB",  # Soybeans
        "BAL",   # Cotton
        "NIB",   # Cocoa
        "SGG",   # Sugar
    ],
    "Equity": [
        "SPY",   # S&P 500
        "EFA",   # MSCI EAFE (developed ex-US)
        "EEM",   # MSCI Emerging Markets
        "EWJ",   # Japan
        "EWG",   # Germany
        "EWU",   # UK
        "EWA",   # Australia
        "EWZ",   # Brazil
    ],
    "FixedIncome": [
        "TLT",   # 20+ yr US Treasury
        "IEF",   # 7-10 yr US Treasury
        "SHY",   # 1-3 yr US Treasury
        "LQD",   # Investment Grade Corporate
        "HYG",   # High Yield Corporate
        "EMB",   # EM USD Bonds
        "BWX",   # International Treasuries
    ],
    "FX": [
        "FXE",   # EUR/USD
        "FXY",   # JPY/USD
        "FXB",   # GBP/USD
        "FXA",   # AUD/USD
        "FXC",   # CAD/USD
        "FXF",   # CHF/USD
    ],
}
# Flatten and clean
ASSETS_FLAT = []
ASSET_CLASS = {}
for cls, tickers in ASSETS.items():
    for t in (tickers or []):
        if t and t != "CurrencyShares":
            ASSETS_FLAT.append(t)
            ASSET_CLASS[t] = cls

# ─── Momentum features ───────────────────────────────────────────────────────
VOL_RETURN_WINDOWS = [1, 21, 63, 126, 252]   # Δ in trading days
MACD_PAIRS = [(8, 24), (16, 48), (32, 96)]   # (S, L) short/long spans
N_FEATURES = len(VOL_RETURN_WINDOWS) + len(MACD_PAIRS)  # = 8
VOL_SPAN = 60          # EWMA span for daily vol estimation (σ_t)
WINSOR_HALFLIFE = 252  # half-life for winsorization EWMA stats
WINSOR_THRESHOLD = 5   # ± 5 EWMA std

# ─── Graph learning ──────────────────────────────────────────────────────────
LOOKBACK_WINDOWS = [252, 504, 756, 1008, 1260]  # δ ∈ {1y, 2y, 3y, 4y, 5y}
ALPHA_GRID = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
BETA_GRID  = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]

# Solver preference: 'MOSEK' > 'CLARABEL' > 'SCS' (in order of accuracy/speed)
SOLVER_PREFERENCE = ["SCS", "MOSEK", "CLARABEL"]

# Refit graph every N trading days (paper: daily; set higher for speed)
# 21 = monthly, 5 = weekly, 1 = daily (slowest)
GRAPH_REFIT_FREQ = 21

# ─── Training schedule ───────────────────────────────────────────────────────
# Paper: retrain every 5 years; validate on last 10% of training data
RETRAIN_YEARS = 5
VAL_FRACTION  = 0.10   # last 10% of training set is validation

# ─── Portfolio ───────────────────────────────────────────────────────────────
VOL_TARGET  = 0.10     # annualised target volatility (σ_tgt)
TRADING_DAYS_PER_YEAR = 252

# ─── Transaction costs ───────────────────────────────────────────────────────
# Cost sensitivity sweep (basis points)
COST_BPS_SWEEP = [0, 0.5, 1, 2, 3, 4, 5]

# ─── Output ──────────────────────────────────────────────────────────────────
import os
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(BASE_DIR, "output")
CACHE_DIR   = os.path.join(BASE_DIR, "cache")
