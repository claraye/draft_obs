"""
main.py — Follow The Leader replication entry point

Paper: "Follow The Leader: Enhancing Systematic Trend-Following Using Network Momentum"
Authors: Linze Li, William Ferreira (Imperial College London / UCL, January 2025)

Usage:
  python main.py                              # full run, DTW method
  python main.py --fast                       # skip grid search, monthly refit
  python main.py --method levy                # use Lévy area (faster than DTW)
  python main.py --method xcorr              # cross-correlation approximation (fastest)
  python main.py --all-methods               # run all 5 methods and compare
  python main.py --bootstrap                  # run 100-sample bootstrap validation
  python main.py --csv path/to/prices.csv    # use own futures price data
  python main.py --start 2015 --end 2024     # custom OOS period

Outputs:
  output/performance_table.csv
  output/cumulative_returns.png
  output/bootstrap_sharpes.csv        (if --bootstrap)
  output/significance_table.csv       (if --bootstrap)
  output/skewness_by_horizon.png      (if --bootstrap)
"""
import os
import sys
import argparse
import logging
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ALL_METHODS = ["levy", "dtw", "ddtw", "sdtw", "sddtw", "xcorr"]
METHOD_LABELS = {
    "levy":  "NMM-LEVY",
    "dtw":   "NMM-DTW-E",
    "ddtw":  "NMM-DDTW-E",
    "sdtw":  "NMM-SDTW-E",
    "sddtw": "NMM-SDDTW-E",
    "xcorr": "NMM-XCORR-E",
}


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Follow The Leader — Network Momentum Replication"
    )
    p.add_argument("--method",    type=str,  default=None,
                   help="Lead-lag method: levy|dtw|ddtw|sdtw|sddtw|xcorr")
    p.add_argument("--all-methods", action="store_true",
                   help="Run all lead-lag methods and produce comparison table")
    p.add_argument("--fast",      action="store_true",
                   help="Skip grid search; use default alpha/beta; monthly refit")
    p.add_argument("--bootstrap", action="store_true",
                   help="Run 100-sample stationary block bootstrap validation")
    p.add_argument("--n-boot",    type=int,  default=None,
                   help="Number of bootstrap samples (default from config)")
    p.add_argument("--csv",       type=str,  default=None,
                   help="CSV file with prices (Date index, asset columns)")
    p.add_argument("--start",     type=str,  default=None, help="OOS start date YYYY-MM-DD")
    p.add_argument("--end",       type=str,  default=None, help="OOS end date YYYY-MM-DD")
    p.add_argument("--refit",     type=int,  default=None,
                   help="Graph refit frequency in trading days")
    p.add_argument("--nocache",   action="store_true", help="Ignore disk cache")
    p.add_argument("--cost",      type=float, default=None,
                   help="Transaction cost in bps (overrides config)")
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    import config as cfg

    # Apply CLI overrides
    if args.start:  cfg.BACKTEST_START   = args.start
    if args.end:    cfg.BACKTEST_END     = args.end
    if args.refit:  cfg.GRAPH_REFIT_FREQ = args.refit
    if args.cost is not None:
        cfg.TRANSACTION_COST_BPS = args.cost
    if args.n_boot is not None:
        cfg.BOOTSTRAP_SAMPLES = args.n_boot

    # Lead-lag method
    method = args.method or cfg.DEFAULT_METHOD
    run_all = args.all_methods or cfg.RUN_ALL_METHODS
    methods_to_run = ALL_METHODS if run_all else [method]

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(cfg.CACHE_DIR,  exist_ok=True)

    if args.nocache:
        import shutil
        shutil.rmtree(cfg.CACHE_DIR, ignore_errors=True)
        os.makedirs(cfg.CACHE_DIR)

    # ── 1. Load prices ────────────────────────────────────────────────────
    logger.info("Step 1: Loading price data")
    if args.csv:
        from data_loader import load_csv
        prices = load_csv(args.csv)
        assets = list(prices.columns)
        asset_class = {a: "Unknown" for a in assets}
    else:
        from data_loader import fetch_yahoo
        prices = fetch_yahoo(
            cfg.ASSETS_FLAT, cfg.DATA_START, cfg.BACKTEST_END, cfg.CACHE_DIR
        )
        # Remove assets still NaN at backtest start
        from data_loader import align_to_backtest
        from data_loader import compute_log_returns
        returns_raw = compute_log_returns(prices)
        prices, returns_raw = align_to_backtest(
            prices, returns_raw, cfg.BACKTEST_START, cfg.BACKTEST_END
        )
        assets = list(prices.columns)
        asset_class = {a: cfg.ASSET_CLASS.get(a, "Unknown") for a in assets}

    logger.info(f"Universe: {len(assets)} assets — {assets}")
    logger.info(f"Date range: {prices.index[0].date()} → {prices.index[-1].date()}")

    # ── 2. Returns and volatility ─────────────────────────────────────────
    logger.info("Step 2: Computing log returns and EWMA volatility")
    from data_loader import compute_log_returns
    from features import ewma_vol

    returns_df = compute_log_returns(prices)
    vol_df     = ewma_vol(returns_df, span=cfg.VOL_SPAN)

    # ── 3. MACD oscillators ───────────────────────────────────────────────
    logger.info(f"Step 3: Building MACD oscillators (speeds k={cfg.MACD_SPEEDS})")
    from features import build_oscillators

    oscillators = build_oscillators(
        returns_df, vol_df,
        k_values=cfg.MACD_SPEEDS,
        M=cfg.MACD_M_RATIO,
    )
    logger.info(f"Oscillator matrix shape: ({len(returns_df)}, {len(assets)}, {len(cfg.MACD_SPEEDS)})")

    # ── 4. Hyperparameter search ──────────────────────────────────────────
    logger.info("Step 4: Hyperparameter grid search")
    from backtest import run_hyperparam_search

    alpha_opt, beta_opt = run_hyperparam_search(
        returns_df=returns_df,
        vol_df=vol_df,
        oscillators=oscillators,
        assets=assets,
        backtest_start=cfg.BACKTEST_START,
        lookback_windows=cfg.LOOKBACK_WINDOWS,
        alpha_grid=cfg.ALPHA_GRID,
        beta_grid=cfg.BETA_GRID,
        method=methods_to_run[0],     # use primary method for hyperparameter tuning
        shape_window=cfg.SHAPE_WINDOW,
        train_years=cfg.HYPERPARAM_TRAIN_YEARS,
        sigma_tgt=cfg.VOL_TARGET,
        tdpy=cfg.TRADING_DAYS_PER_YEAR,
        solver_prefs=cfg.SOLVER_PREFERENCE,
        add_self_loops=cfg.ADD_SELF_LOOPS,
        fast_mode=args.fast,
    )

    # ── 5. Walk-forward backtest (all methods) ────────────────────────────
    logger.info(f"Step 5: Walk-forward backtest ({methods_to_run})")
    from backtest import run_backtest

    all_results = {}
    for m in methods_to_run:
        logger.info(f"  Running method: {m}")
        result = run_backtest(
            prices=prices,
            returns_df=returns_df,
            vol_df=vol_df,
            oscillators=oscillators,
            assets=assets,
            backtest_start=cfg.BACKTEST_START,
            backtest_end=cfg.BACKTEST_END,
            lookback_windows=cfg.LOOKBACK_WINDOWS,
            alpha_opt=alpha_opt,
            beta_opt=beta_opt,
            method=m,
            shape_window=cfg.SHAPE_WINDOW,
            sigma_tgt=cfg.VOL_TARGET,
            tdpy=cfg.TRADING_DAYS_PER_YEAR,
            graph_refit_freq=cfg.GRAPH_REFIT_FREQ,
            cost_bps=cfg.TRANSACTION_COST_BPS,
            solver_prefs=cfg.SOLVER_PREFERENCE,
            add_self_loops=cfg.ADD_SELF_LOOPS,
            cache_dir=cfg.CACHE_DIR,
        )
        all_results[m] = result

    # Use primary method's result as the main result
    primary_result = all_results[methods_to_run[0]]

    # ── 6. Performance metrics ────────────────────────────────────────────
    logger.info("Step 6: Performance metrics")
    from metrics import print_summary

    # Merge all strategy returns into one dict for the summary table
    summary_rets = {}
    summary_rets["MACD"]  = primary_result.returns["MACD"]
    for m, res in all_results.items():
        label = METHOD_LABELS.get(m, f"NMM-{m.upper()}-E")
        summary_rets[label] = res.returns["NMM-E"]

    perf_df = print_summary(summary_rets)
    perf_path = os.path.join(cfg.OUTPUT_DIR, "performance_table.csv")
    perf_df.to_csv(perf_path)
    logger.info(f"Performance table saved: {perf_path}")

    # ── 7. Plots ──────────────────────────────────────────────────────────
    logger.info("Step 7: Generating plots")
    _plot_cumulative_returns(primary_result, summary_rets, cfg.OUTPUT_DIR)
    _plot_skewness_by_horizon(summary_rets, cfg.OUTPUT_DIR)

    # ── 8. Bootstrap validation (optional) ───────────────────────────────
    if args.bootstrap:
        logger.info(f"Step 8: Bootstrap validation ({cfg.BOOTSTRAP_SAMPLES} samples)")
        _run_bootstrap(
            returns_df=returns_df,
            prices=prices,
            vol_df=vol_df,
            oscillators=oscillators,
            assets=assets,
            primary_result=primary_result,
            all_results=all_results,
            methods_to_run=methods_to_run,
            cfg=cfg,
            alpha_opt=alpha_opt,
            beta_opt=beta_opt,
        )

    logger.info(f"\nAll outputs saved to: {cfg.OUTPUT_DIR}")


# ─── Bootstrap helper ─────────────────────────────────────────────────────────

def _run_bootstrap(
    returns_df, prices, vol_df, oscillators, assets,
    primary_result, all_results, methods_to_run, cfg,
    alpha_opt, beta_opt,
):
    from backtest import make_backtest_fn
    from bootstrap import (
        stationary_block_bootstrap,
        run_bootstrap_comparison,
        print_significance_table,
    )
    from metrics import sharpe_ratio

    oos_start = cfg.BACKTEST_START
    oos_end   = cfg.BACKTEST_END
    oos_mask  = (returns_df.index >= oos_start) & (returns_df.index <= oos_end)
    oos_returns = returns_df.loc[oos_mask].values
    oos_prices  = prices.loc[oos_mask].values

    # Create backtest function for primary method
    m = methods_to_run[0]
    backtest_fn = make_backtest_fn(
        returns_df=returns_df, vol_df=vol_df, oscillators=oscillators,
        assets=assets,
        backtest_start=oos_start, backtest_end=oos_end,
        lookback_windows=cfg.LOOKBACK_WINDOWS,
        alpha_opt=alpha_opt, beta_opt=beta_opt,
        method=m, shape_window=cfg.SHAPE_WINDOW,
        sigma_tgt=cfg.VOL_TARGET, tdpy=cfg.TRADING_DAYS_PER_YEAR,
        graph_refit_freq=cfg.GRAPH_REFIT_FREQ,
        cost_bps=cfg.TRANSACTION_COST_BPS,
        solver_prefs=cfg.SOLVER_PREFERENCE,
        add_self_loops=cfg.ADD_SELF_LOOPS,
    )

    bootstrap_df = run_bootstrap_comparison(
        returns=oos_returns,
        prices=oos_prices,
        assets=assets,
        backtest_fn=backtest_fn,
        n_samples=cfg.BOOTSTRAP_SAMPLES,
        block_size=cfg.BOOTSTRAP_BLOCK_SIZE,
    )
    # Rename columns
    bootstrap_df.rename(columns={
        "MACD": "MACD",
        "NMM-E": METHOD_LABELS.get(m, "NMM-E"),
    }, inplace=True)

    boot_path = os.path.join(cfg.OUTPUT_DIR, "bootstrap_sharpes.csv")
    bootstrap_df.to_csv(boot_path)
    logger.info(f"Bootstrap Sharpe distribution saved: {boot_path}")

    sig_df = print_significance_table(bootstrap_df, baseline_col="MACD")
    sig_path = os.path.join(cfg.OUTPUT_DIR, "significance_table.csv")
    sig_df.to_csv(sig_path)
    logger.info(f"Significance table saved: {sig_path}")

    _plot_bootstrap_distributions(bootstrap_df, cfg.OUTPUT_DIR)


# ─── Plotting helpers ─────────────────────────────────────────────────────────

def _plot_cumulative_returns(
    primary_result,
    summary_rets: dict,
    output_dir: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        dates = primary_result.dates

        fig, ax = plt.subplots(figsize=(14, 6))
        colors = {
            "MACD":         "#555555",
            "NMM-LEVY-E":   "#1f77b4",
            "NMM-DTW-E":    "#d62728",
            "NMM-DDTW-E":   "#ff7f0e",
            "NMM-SDTW-E":   "#2ca02c",
            "NMM-SDDTW-E":  "#9467bd",
            "NMM-XCORR-E":  "#8c564b",
        }
        styles = {"MACD": "--"}

        for label, rets in summary_rets.items():
            rets_clean = np.where(np.isfinite(rets), rets, 0.0)
            cum = np.cumprod(1.0 + rets_clean)
            color  = colors.get(label, None)
            style  = styles.get(label, "-")
            lw     = 2.0 if "NMM" in label else 1.2
            ax.plot(dates[:len(cum)], cum, label=label,
                    color=color, linestyle=style, linewidth=lw)

        ax.axhline(1, color="black", linewidth=0.5, linestyle=":")
        ax.set_title("Cumulative Returns — Follow The Leader vs MACD Baseline",
                     fontsize=13)
        ax.set_ylabel("Cumulative Return")
        ax.legend(loc="upper left", fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(alpha=0.3)
        plt.tight_layout()

        path = os.path.join(output_dir, "cumulative_returns.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Cumulative returns plot: {path}")
    except Exception as e:
        logger.warning(f"Plotting failed: {e}")


def _plot_skewness_by_horizon(
    summary_rets: dict,
    output_dir: str,
) -> None:
    """
    Rolling skewness at different return aggregation horizons.
    Matches Figure 4 in the paper (skewness profiles across 1d to ~60d horizons).
    """
    try:
        import matplotlib.pyplot as plt
        from scipy.stats import skew

        horizons = [1, 5, 10, 21, 42, 63]   # trading days
        horizon_labels = ["1D", "1W", "2W", "1M", "2M", "3M"]

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = {"MACD": "#555555", "NMM-DTW-E": "#d62728",
                  "NMM-LEVY-E": "#1f77b4", "NMM-DDTW-E": "#ff7f0e"}

        for label, rets in summary_rets.items():
            rets_clean = np.where(np.isfinite(rets), rets, 0.0)
            skews = []
            for h in horizons:
                # Aggregate returns over h-day non-overlapping windows
                n_windows = len(rets_clean) // h
                agg = np.array([
                    rets_clean[i * h:(i + 1) * h].sum()
                    for i in range(n_windows)
                ])
                skews.append(float(skew(agg)))
            color = colors.get(label, None)
            ax.plot(horizon_labels, skews, label=label, marker="o",
                    markersize=5, color=color,
                    linewidth=2 if "NMM" in label else 1)

        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Return Horizon")
        ax.set_ylabel("Skewness")
        ax.set_title("Return Skewness by Horizon: NMM vs MACD Baseline", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()

        path = os.path.join(output_dir, "skewness_by_horizon.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Skewness plot: {path}")
    except Exception as e:
        logger.warning(f"Skewness plot failed: {e}")


def _plot_bootstrap_distributions(
    bootstrap_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """
    Distribution of bootstrapped Sharpe ratios for each strategy.
    Matches Figure 1 in the paper.
    """
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = {"MACD": "#555555"}
        nmm_colors = ["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
        nmm_idx = 0

        for col in bootstrap_df.columns:
            c = colors.get(col, nmm_colors[nmm_idx % len(nmm_colors)])
            if col not in colors:
                nmm_idx += 1
            ax.hist(bootstrap_df[col], bins=20, alpha=0.5, color=c, label=col, density=True)

        ax.set_xlabel("Net Sharpe Ratio (Bootstrapped)")
        ax.set_ylabel("Density")
        ax.set_title("Distribution of Bootstrapped Net Sharpe Ratios", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()

        path = os.path.join(output_dir, "bootstrap_sharpe_distributions.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Bootstrap distribution plot: {path}")
    except Exception as e:
        logger.warning(f"Bootstrap distribution plot failed: {e}")


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
