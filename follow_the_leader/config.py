"""
config.py — Configuration for Follow The Leader replication
Paper: "Follow The Leader: Enhancing Systematic Trend-Following Using Network Momentum"
Authors: Linze Li, William Ferreira (Imperial College London / UCL, January 2025)

Data note: Paper uses Bloomberg settlement prices for 28 commodity futures.
This replication uses Yahoo Finance ETF proxies; replace ASSETS with actual
futures prices via --csv if available.
"""
import os

# ─── Data ────────────────────────────────────────────────────────────────────
DATA_START      = "2005-01-01"   # fetch start (need burn-in before backtest)
BACKTEST_START  = "2010-01-01"  # OOS period start (2010 ensures most ETFs have history)
BACKTEST_END    = "2024-06-30"  # OOS period end (matches paper's June 2024)

# ─── Asset Universe ───────────────────────────────────────────────────────────
# ETF/ETN proxies for commodity futures (Yahoo Finance tickers).
# Paper uses 28 Bloomberg futures: 14 agriculture, 3 energy, metals, equity indices.
ASSETS = {
    "Agriculture": [
        "DBA",   # Invesco DB Agriculture Fund (broad basket)
        "CORN",  # Teucrium Corn Fund
        "WEAT",  # Teucrium Wheat Fund
        "SOYB",  # Teucrium Soybean Fund
        "SGG",   # iPath Sugar ETN
        "BAL",   # iPath Cotton ETN
        "NIB",   # iPath Cocoa ETN
    ],
    "Energy": [
        "USO",   # United States Oil Fund (WTI Crude)
        "UNG",   # United States Natural Gas Fund
        "UGA",   # United States Gasoline Fund
    ],
    "Metals": [
        "GLD",   # SPDR Gold Shares
        "SLV",   # iShares Silver Trust
        "CPER",  # United States Copper Index Fund
    ],
    "EquityIndex": [
        "SPY",   # SPDR S&P 500 (proxy for CME S&P futures)
        "QQQ",   # Invesco Nasdaq 100 (proxy for NASDAQ futures)
        "FEZ",   # SPDR Euro Stoxx 50 (proxy for EURO STOXX futures)
        "EWG",   # iShares Germany (proxy for DAX futures)
        "EWQ",   # iShares France (proxy for CAC 40 futures)
    ],
}

ASSETS_FLAT = []
ASSET_CLASS = {}
for _cls, _tickers in ASSETS.items():
    for _t in _tickers:
        ASSETS_FLAT.append(_t)
        ASSET_CLASS[_t] = _cls

# ─── Feature Parameters ──────────────────────────────────────────────────────
VOL_SPAN     = 22    # EWMA vol span for σ^22 (22 trading days ≈ 1 month)
MACD_SPEEDS  = [1, 2, 3, 4, 5]  # k: α_fast(k) = 1/2^k, α_slow(k) = 1/(M·2^k)
MACD_M_RATIO = 4                 # slow/fast ratio M; slow is 4x longer than fast

# ─── Lead-Lag Detection ──────────────────────────────────────────────────────
# Ensemble lookback windows δ (trading days) — paper: {22,44,66,88,110,132}
LOOKBACK_WINDOWS  = [22, 44, 66, 88, 110, 132]

# Lead-lag detection method
# 'levy'  : Lévy area (fast, captures non-linear lead-lag; no DTW dependency)
# 'dtw'   : Standard Dynamic Time Warping (requires dtaidistance for speed)
# 'ddtw'  : Derivative DTW — uses local derivatives instead of raw values
# 'sdtw'  : Shape DTW — uses multidimensional local shape descriptors
# 'sddtw' : Shape + Derivative DTW
# 'xcorr' : Cross-correlation lag (fastest approximation; bonus method)
DEFAULT_METHOD    = "dtw"
RUN_ALL_METHODS   = True   # If True, backtest all methods and produce comparison table

# Shape descriptor window size for SDTW / SDDTW
SHAPE_WINDOW = 5

# ─── Graph Learning ──────────────────────────────────────────────────────────
# FTL objective: min ||V - A||^2_F + alpha*sum(A) + beta*||A||^2_F
#   s.t.  A >= 0,  A = A^T
ALPHA_GRID        = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
BETA_GRID         = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
SOLVER_PREFERENCE = ["MOSEK", "CLARABEL", "SCS"]

# Refit graph every N trading days (1=daily, 5=weekly, 22=monthly)
# Paper implicitly uses daily; set higher for speed.
GRAPH_REFIT_FREQ = 22   # monthly — change to 5 for weekly (slower)

# Years of initial in-sample data used for hyperparameter grid search
HYPERPARAM_TRAIN_YEARS = 3

# Whether to add self-loops before normalising (recommended: True)
# Self-loops mean R̃ includes own oscillator as well as neighbours'
ADD_SELF_LOOPS = True

# ─── Portfolio ───────────────────────────────────────────────────────────────
VOL_TARGET            = 0.10     # 10% annualised target vol (σ_tgt in paper)
TRADING_DAYS_PER_YEAR = 252
TRANSACTION_COST_BPS  = 3.0     # basis points per unit of turnover (for net Sharpe)

# ─── Bootstrap Validation ────────────────────────────────────────────────────
BOOTSTRAP_SAMPLES    = 100    # 100 stationary block bootstrap samples (paper)
BOOTSTRAP_BLOCK_SIZE = 22     # ~1 month block preserves autocorrelation structure

# ─── Output ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CACHE_DIR  = os.path.join(BASE_DIR, "cache")
