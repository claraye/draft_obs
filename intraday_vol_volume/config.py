"""
config.py — Futures-focused configuration for the Volume-Driven ToD replication.

Paper: "Volume-Driven Time-of-Day Effects in Intraday Volatility Models"
       Martins, Virbickaite, Nguyen, Lopes (2025)

This implementation follows the paper's market set-up as closely as Yahoo data allows:
E-mini S&P 500, E-mini Nasdaq 100, Euro FX, and WTI crude futures on 5-minute bars.
"""

# ── Data ─────────────────────────────────────────────────────────────────────
# Yahoo continuous/front-month futures proxies for the paper's contracts.
TICKERS = ["ES=F", "NQ=F", "6E=F", "CL=F"]

# Yahoo Finance supports 5-minute bars for roughly the most recent 60 days.
DATA_INTERVAL = "5m"
DATA_PERIOD = "60d"

CACHE_DIR  = "cache"
OUTPUT_DIR = "output"

# ── Intraday Interval Definition ──────────────────────────────────────────────
# Paper session: 18:00 ET previous day to 17:00 ET current day, excluding the
# daily 17:00–18:00 maintenance break. That leaves 23 hours = 276 five-minute bars.
N_INTRADAY_INTERVALS = 276
INTERVAL_LABELS = [
    hour * 100 + minute
    for hour in tuple(range(18, 24)) + tuple(range(0, 17))
    for minute in range(0, 60, 5)
]

# ── Model Parameters ──────────────────────────────────────────────────────────
EWMA_LAMBDA   = 0.94   # EWMA decay for persistent volatility component
MIN_OBS_OLS   = 120    # minimum bars required to fit OLS (per interval × day)
LOG_VAR_CLIP  = -20.0  # floor for log(r²) to avoid -inf from zero returns
SUNDAY_OPEN_LAGS  = 12
MCMC_N_ITER       = 120
MCMC_BURN_IN      = 40
MCMC_THIN         = 2
MCMC_RANDOM_STATE = 42

# ── Backtest ──────────────────────────────────────────────────────────────────
TRAIN_FRAC    = 0.70   # fraction of sample used for model fitting
# Vol-managed portfolio scaling constant:
# position size = c / forecast_vol_bar; scaled back to daily metrics afterward
TARGET_ANN_VOL = 0.15  # 15% annualized vol target
TC_BPS         = 1.0   # one-way transaction cost in bps (1 bp = 0.0001)

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"

