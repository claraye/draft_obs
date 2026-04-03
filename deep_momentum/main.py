"""
main.py — Deep Momentum Replication
Paper: "Bimodality Everywhere: International Evidence of Deep Momentum"
       Han & Qin (2026), SSRN 4452964

Usage:
  python main.py                         # full run on S&P 500 sample
  python main.py --fast                  # 10 ensemble runs instead of 100
  python main.py --csv path/to/rets.csv  # use your own monthly returns
  python main.py --start 2010 --end 2023 # custom OOS period

Output (in output/):
  performance_summary.csv
  monthly_returns.csv
  cumulative_returns.png
  bimodality_over_time.png
"""

import os
import argparse
import logging
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Deep Momentum Replication (Han & Qin 2026)")
    p.add_argument("--fast",  action="store_true",
                   help="10 ensemble runs instead of 100 (faster, less stable)")
    p.add_argument("--csv",   type=str, default=None,
                   help="Path to CSV of monthly returns (Date index, ticker columns)")
    p.add_argument("--start", type=str, default=None, help="OOS start (YYYY-MM-DD)")
    p.add_argument("--end",   type=str, default=None, help="OOS end   (YYYY-MM-DD)")
    p.add_argument("--nocache", action="store_true", help="Recompute everything, ignore cache")
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    import config as cfg

    if args.start: cfg.BACKTEST_START = args.start
    if args.end:   cfg.BACKTEST_END   = args.end
    n_runs = 10 if args.fast else cfg.N_ENSEMBLE_RUNS

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(cfg.CACHE_DIR,  exist_ok=True)

    if args.nocache:
        import shutil
        shutil.rmtree(cfg.CACHE_DIR, ignore_errors=True)
        os.makedirs(cfg.CACHE_DIR)

    # ── 1. Load monthly returns ───────────────────────────────────────────
    logger.info("Step 1: Loading monthly return data")
    if args.csv:
        from data_loader import load_csv
        returns = load_csv(args.csv)
    else:
        from data_loader import fetch_yahoo_monthly
        returns = fetch_yahoo_monthly(
            cfg.SP500_SAMPLE,
            start=cfg.DATA_START,
            end=cfg.BACKTEST_END,
            cache_dir=cfg.CACHE_DIR,
        )

    from data_loader import apply_data_filters, compute_market_cap_proxy
    returns = apply_data_filters(returns)
    size_proxy = compute_market_cap_proxy(returns)

    logger.info(f"Universe: {returns.shape[1]} stocks, {returns.shape[0]} months")
    logger.info(f"Date range: {returns.index[0].date()} → {returns.index[-1].date()}")

    # ── 2. Build feature matrix ───────────────────────────────────────────
    logger.info("Step 2: Building 16-feature matrix ...")
    from features import build_features
    feature_df = build_features(returns, size_proxy)
    logger.info(f"Feature matrix: {len(feature_df):,} stock-month observations")

    # ── 3. Run walk-forward backtest ──────────────────────────────────────
    logger.info("Step 3: Running walk-forward backtest ...")
    logger.info(f"  OOS period:    {cfg.BACKTEST_START} → {cfg.BACKTEST_END}")
    logger.info(f"  Ensemble runs: {n_runs}")
    logger.info(f"  Retrain freq:  every {cfg.RETRAIN_FREQ} months")

    from backtest import run_backtest
    result = run_backtest(
        feature_df        = feature_df,
        returns           = returns,
        backtest_start    = cfg.BACKTEST_START,
        backtest_end      = cfg.BACKTEST_END,
        min_train_years   = cfg.MIN_TRAIN_YEARS,
        retrain_freq      = cfg.RETRAIN_FREQ,
        n_ensemble_runs   = n_runs,
        xgb_params        = cfg.XGB_PARAMS,
        early_stopping    = cfg.EARLY_STOPPING_ROUNDS,
        long_frac         = cfg.LONG_FRAC,
        short_frac        = cfg.SHORT_FRAC,
        class_mean_lb_yrs = cfg.CLASS_MEAN_LOOKBACK,
        n_classes         = cfg.N_CLASSES,
        cache_dir         = cfg.CACHE_DIR,
    )

    if not result.dates:
        logger.error("No OOS predictions generated. Check date range and data availability.")
        return

    # ── 4. Performance metrics ────────────────────────────────────────────
    logger.info("Step 4: Computing performance metrics")
    from metrics import print_summary, sharpe_ratio, annualized_return, crash_rate

    perf_df = print_summary(result.returns)
    perf_path = os.path.join(cfg.OUTPUT_DIR, "performance_summary.csv")
    perf_df.to_csv(perf_path)
    logger.info(f"Performance summary saved: {perf_path}")

    # Save monthly returns
    rets_df = pd.DataFrame(result.returns, index=result.dates)
    rets_df.to_csv(os.path.join(cfg.OUTPUT_DIR, "monthly_returns.csv"))

    # ── 5. Plots ──────────────────────────────────────────────────────────
    logger.info("Step 5: Generating plots")
    _plot_cumulative_returns(result, cfg.OUTPUT_DIR)
    _plot_bimodality(result, cfg.OUTPUT_DIR)

    logger.info(f"\n✓ All outputs saved to: {cfg.OUTPUT_DIR}")


# ─── Plotting ─────────────────────────────────────────────────────────────────

def _plot_cumulative_returns(result, output_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        colors = {
            "MOM": "gray",
            "XGB": "blue",
            "DPR": "orange",
            "RET": "red",
            "SRP": "green",
        }
        styles = {
            "MOM": ":",
            "XGB": "--",
            "DPR": "-.",
            "RET": "-",
            "SRP": "--",
        }

        fig, ax = plt.subplots(figsize=(13, 6))
        for strat, rets in result.returns.items():
            rets = np.array(rets, dtype=float)
            valid = np.where(np.isfinite(rets), rets, 0.0)
            cum = np.cumprod(1 + valid)
            ax.plot(
                result.dates[:len(cum)], cum,
                label=strat,
                color=colors.get(strat, "black"),
                linestyle=styles.get(strat, "-"),
                linewidth=2.0 if strat == "RET" else 1.2,
            )

        ax.set_title("Deep Momentum — Cumulative Returns (Equal-Weighted Long-Short)", fontsize=12)
        ax.set_ylabel("Cumulative Return (base = 1)")
        ax.legend(loc="upper left", fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(alpha=0.3)
        ax.axhline(1, color="black", linewidth=0.5, linestyle="-")

        plt.tight_layout()
        path = os.path.join(output_dir, "cumulative_returns.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Plot saved: {path}")

    except Exception as e:
        logger.warning(f"Plotting failed: {e}. Install matplotlib: pip install matplotlib")


def _plot_bimodality(result, output_dir: str) -> None:
    """Plot bimodality measure over time for MOM and RET strategies."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        if not any(result.bimodality.values()):
            return

        fig, ax = plt.subplots(figsize=(13, 4))
        colors = {"MOM": "gray", "XGB": "blue", "RET": "red"}

        for strat, bm_vals in result.bimodality.items():
            if bm_vals:
                n = len(bm_vals)
                dates = result.dates[:n]
                bm_series = pd.Series(bm_vals, index=dates).rolling(6).mean()
                ax.plot(
                    bm_series.index, bm_series.values,
                    label=f"{strat} (6m avg)",
                    color=colors.get(strat, "black"),
                    linewidth=1.5,
                )

        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_title("Bimodality Measure (BM) Over Time — Lower = Less Bimodal", fontsize=11)
        ax.set_ylabel("BM")
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(alpha=0.3)

        plt.tight_layout()
        path = os.path.join(output_dir, "bimodality_over_time.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Plot saved: {path}")

    except Exception as e:
        logger.warning(f"Bimodality plot failed: {e}")


if __name__ == "__main__":
    main()
