"""
main.py — Volume-Driven Time-of-Day Intraday Volatility Replication
Paper: "Volume-Driven Time-of-Day Effects in Intraday Volatility Models"
       Martins, Virbickaite, Nguyen, Lopes (2025)

Usage:
  python main.py                         # all futures (ES=F, NQ=F, 6E=F, CL=F)
  python main.py --tickers ES=F NQ=F    # specific futures
  python main.py --nocache           # force re-download
  python main.py --no-plot           # skip chart generation

Outputs:
  output/performance_<TICKER>.csv    # daily returns by strategy
  output/cumulative_returns.png      # cumulative return chart (all tickers)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd


def configure_stdio() -> None:
    """Avoid Windows console encoding failures when printing Unicode summaries."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Volume-Driven Intraday Vol — US Market Replication"
    )
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Override ticker list (default: from config)")
    parser.add_argument("--nocache", action="store_true",
                        help="Force re-download all intraday data")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip chart generation")
    parser.add_argument("--train-frac", type=float, default=None,
                        help="Train fraction 0-1 (default: from config)")
    return parser.parse_args()


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )


def resolve_local_dir(base_dir: str, path: str) -> str:
    """Resolve config directories relative to this strategy directory."""
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


def plot_results(all_results: dict, output_dir: str) -> None:
    """Save cumulative return chart for all tickers."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logging.getLogger(__name__).warning("matplotlib not available; skipping plot")
        return

    n_tickers = len(all_results)
    if n_tickers == 0:
        return

    fig, axes = plt.subplots(n_tickers, 1, figsize=(13, 5 * n_tickers), squeeze=False)

    strat_colors = {
        "buy_hold": "#7f7f7f",
        "vtod_vm":  "#1f77b4",
        "garch_vm": "#ff7f0e",
        "rv21_vm":  "#2ca02c",
    }
    strat_labels = {
        "buy_hold": "Buy & Hold",
        "vtod_vm":  "Volume-ToD Vol-Managed",
        "garch_vm": "GARCH(1,1) Vol-Managed",
        "rv21_vm":  "Rolling-RV21 Vol-Managed",
    }

    for ax_idx, (ticker, res) in enumerate(all_results.items()):
        ax = axes[ax_idx][0]
        if not res.dates:
            ax.set_title(f"{ticker} — no data")
            continue

        dates = pd.to_datetime(res.dates)
        for strat, rets in res.returns.items():
            rets  = np.array(rets, dtype=float)
            valid = np.isfinite(rets)
            if valid.sum() < 2:
                continue
            cum   = np.cumprod(1 + np.where(valid, rets, 0))
            ax.plot(dates[valid], cum[valid],
                    label=strat_labels.get(strat, strat),
                    color=strat_colors.get(strat),
                    linewidth=1.2)

        ax.set_title(
            f"{ticker}  |  OOS MZ R²={res.mz_r2_oos:.3f}  (train MZ R²={res.mz_r2_train:.3f})",
            fontsize=11,
        )
        ax.set_ylabel("Cumulative Return")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Volume-Driven ToD Intraday Vol — Replication\n"
        "(OLS+EWMA approximation of Fruet Dias et al. 2025)",
        fontsize=12,
    )
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "cumulative_returns.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    logging.getLogger(__name__).info(f"[plot] Saved to {path}")


def save_outputs(all_results: dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for ticker, res in all_results.items():
        if not res.dates:
            continue
        rets_df = pd.DataFrame(res.returns, index=res.dates)
        rets_df.to_csv(os.path.join(output_dir, f"performance_{ticker}.csv"))

    logging.getLogger(__name__).info(f"[output] Saved to {output_dir}/")


def main() -> None:
    args = parse_args()

    configure_stdio()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, base_dir)
    import config as cfg
    setup_logging(cfg.LOG_LEVEL)
    log = logging.getLogger(__name__)

    tickers    = args.tickers or cfg.TICKERS
    train_frac = args.train_frac or cfg.TRAIN_FRAC
    cache_dir = resolve_local_dir(base_dir, cfg.CACHE_DIR)
    output_dir = resolve_local_dir(base_dir, cfg.OUTPUT_DIR)

    os.makedirs(cache_dir,  exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Download intraday data ────────────────────────────────────────
    from data_loader import fetch_intraday
    intraday_data = fetch_intraday(
        tickers, period=cfg.DATA_PERIOD,
        interval=cfg.DATA_INTERVAL,
        cache_dir=cache_dir, nocache=args.nocache
    )
    if not intraday_data:
        log.error("No intraday data downloaded. Check connectivity or --nocache.")
        sys.exit(1)

    log.info(f"Loaded data for: {list(intraday_data.keys())}")

    # ── Step 2: Run backtest for each ticker ──────────────────────────────────
    from backtest import run_backtest
    all_results = {}

    for tk in tickers:
        log.info(f"── {tk} ──────────────────────────────────────────────────")
        res = run_backtest(
            intraday_data   = intraday_data,
            ticker          = tk,
            interval_labels = cfg.INTERVAL_LABELS,
            train_frac      = train_frac,
            target_ann_vol  = cfg.TARGET_ANN_VOL,
            tc_bps          = cfg.TC_BPS,
            log_var_clip    = cfg.LOG_VAR_CLIP,
            mcmc_n_iter     = cfg.MCMC_N_ITER,
            mcmc_burn_in    = cfg.MCMC_BURN_IN,
            mcmc_thin       = cfg.MCMC_THIN,
            sunday_open_lags = cfg.SUNDAY_OPEN_LAGS,
            random_state    = cfg.MCMC_RANDOM_STATE,
        )
        all_results[tk] = res
        if res.dates:
            log.info(f"  OOS period: {res.dates[0]} → {res.dates[-1]} ({len(res.dates)} days)")

    # ── Step 3: Print metrics ─────────────────────────────────────────────────
    from metrics import print_summary
    print_summary(all_results, rf_annual=0.04)

    # ── Step 4: Save outputs ──────────────────────────────────────────────────
    save_outputs(all_results, output_dir)

    if not args.no_plot:
        plot_results(all_results, output_dir)

    log.info("Done.")


if __name__ == "__main__":
    main()
