"""
config.py — Configuration for Deep Momentum replication
Paper: "Bimodality Everywhere: International Evidence of Deep Momentum"
       Han & Qin (2026), SSRN 4452964
"""

# ─── Data ─────────────────────────────────────────────────────────────────────
DATA_START     = "1993-01-01"   # fetch start (need burn-in for 10-yr training)
BACKTEST_START = "2010-01-01"   # OOS start (paper: Jan 2010)
BACKTEST_END   = "2023-12-31"   # OOS end   (paper: Dec 2023)

# ─── Asset universe ───────────────────────────────────────────────────────────
# Paper uses Datastream (WRDS) with 4,985 US stocks on average.
# Here we use a representative set of S&P 500 constituents from Yahoo Finance.
# For a real replication, provide a CSV with monthly returns for all stocks.
#
# To use your own data:  python main.py --csv path/to/monthly_returns.csv
#
# The CSV should have:
#   - Index: "Date" column (YYYY-MM-DD, last trading day of each month)
#   - Columns: one column per stock ticker, containing total returns (not prices)
#   - Values: monthly returns as decimals (e.g. 0.05 for +5%)

USE_SP500_SAMPLE = True   # download a sample of large-cap US stocks via yfinance

# A representative sample of ~100 S&P 500 stocks across sectors (for quick tests)
# Replace or expand for production use
SP500_SAMPLE = [
    # Technology
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "AVGO", "ORCL",
    "AMD", "INTC", "QCOM", "TXN", "MU", "AMAT", "ADI", "LRCX", "KLAC",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "C", "AXP", "COF", "USB",
    "PNC", "TFC", "SCHW", "ICE", "CME", "CB", "MMC", "AON",
    # Healthcare
    "JNJ", "UNH", "LLY", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY",
    "AMGN", "GILD", "CVS", "CI", "HUM", "MDT", "BSX", "SYK",
    # Consumer Discretionary
    "HD", "MCD", "NKE", "SBUX", "TJX", "LOW", "TGT", "BKNG", "MAR", "HLT",
    # Consumer Staples
    "PG", "KO", "PEP", "WMT", "COST", "CL", "MDLZ", "KHC", "GIS",
    # Industrials
    "RTX", "HON", "UPS", "CAT", "DE", "BA", "GE", "LMT", "MMM", "EMR",
    # Energy
    "XOM", "CVX", "COP", "EOG", "SLB", "PSX", "MPC", "VLO",
    # Materials
    "LIN", "APD", "SHW", "FCX", "NEM", "NUE",
    # Real Estate
    "AMT", "PLD", "EQIX", "CCI", "PSA", "O",
    # Utilities
    "NEE", "DUK", "SO", "D", "AEP", "EXC",
    # Communication Services
    "T", "VZ", "DIS", "NFLX", "CMCSA",
]

# ─── Momentum features ────────────────────────────────────────────────────────
# Paper: 16 features per stock-month
#   - 5 normalized m-month momentum: m ∈ {1, 3, 6, 9, 12}
#   - 5 cross-sectional means:  mean(MOM^m) across all stocks
#   - 5 cross-sectional stds:   std(MOM^m)  across all stocks
#   - 1 size decile:            categorical 1-10

MOM_WINDOWS = [1, 3, 6, 9, 12]   # months
N_FEATURES  = 16                   # 5 + 5 + 5 + 1

# ─── XGBoost classifier ───────────────────────────────────────────────────────
N_CLASSES          = 10     # predict return decile (1=worst, 10=best)
N_ENSEMBLE_RUNS    = 100    # paper: train 100 times per month, average probabilities
TRAIN_VAL_SPLIT    = 0.80   # random 80/20 split (NOT chronological)
EARLY_STOPPING_ROUNDS = 20  # XGBoost early stopping on val set

# XGBoost hyperparameters (paper uses defaults + early stopping)
XGB_PARAMS = {
    "n_estimators":    300,
    "max_depth":       6,
    "learning_rate":   0.1,
    "subsample":       0.8,
    "colsample_bytree": 0.8,
    "use_label_encoder": False,
    "eval_metric":     "mlogloss",
    "random_state":    42,        # overridden per ensemble run
    "n_jobs":          -1,
}

# ─── Training schedule ────────────────────────────────────────────────────────
MIN_TRAIN_YEARS = 10    # require at least 10 years before first prediction
RETRAIN_FREQ    = 12    # retrain every N months (paper: annually = 12)

# ─── Reclassification ─────────────────────────────────────────────────────────
CLASS_MEAN_LOOKBACK = 10  # years of history to estimate µ^k (class mean returns)

# ─── Portfolio construction ───────────────────────────────────────────────────
LONG_FRAC  = 0.10   # top decile (10%) → long
SHORT_FRAC = 0.10   # bottom decile (10%) → short
# Equal-weighted within each leg; long-short combined

# ─── Output ───────────────────────────────────────────────────────────────────
import os
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CACHE_DIR  = os.path.join(BASE_DIR, "cache")
