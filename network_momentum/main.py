"""
main.py — Network Momentum Replication
Paper: "Network Momentum across Asset Classes" (Pu, Roberts, Dong, Zohren; arXiv:2308.11294)

Usage:
  python main.py                    # full run (slow)
  python main.py --fast             # fast mode: skip grid search
  python main.py --csv path/to/data.csv   # use own price data
  python main.py --start 2010 --end 2022  # custom OOS period

Output:
  output/performance_table.csv
  output/cumulative_returns.png
  output/graph_statistics.csv
  output/cost_sensitivity.png
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
    p = argparse.ArgumentParser(description="Network Momentum Replication (arXiv:2308.11294)")
    p.add_argument("--fast",   action="store_true",
                   help="Skip hyperparameter grid search (α=0.01, β=0.01)")
    p.add_argument("--csv",    type=str, default=None,
                   help="Path to CSV price file (Date index, asset columns)")
    p.add_argument("--start",  type=str, default=None, help="OOS start date (YYYY-MM-DD)")
    p.add_argument("--end",    type=str, default=None, help="OOS end date (YYYY-MM-DD)")
    p.add_argument("--refit",  type=int, default=None,
                   help="Graph refit frequency in trading days (default from config)")
    p.add_argument("--nocache", action="store_true", help="Ignore cache, recompute everything")
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    import config as cfg

    # Apply CLI overrides
    if args.start:  cfg.BACKTEST_START = args.start
    if args.end:    cfg.BACKTEST_END   = args.end
    if args.refit:  cfg.GRAPH_REFIT_FREQ = args.refit

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
        # Infer asset classes from columns or assign all to "Unknown"
        assets = list(prices.columns)
        asset_class = {a: "Unknown" for a in assets}
    else:
        from data_loader import fetch_yahoo
        prices = fetch_yahoo(
            cfg.ASSETS_FLAT, cfg.DATA_START, cfg.BACKTEST_END, cfg.CACHE_DIR
        )
        assets = list(prices.columns)
        asset_class = {a: cfg.ASSET_CLASS.get(a, "Unknown") for a in assets}

    logger.info(f"Asset universe: {len(assets)} assets")
    logger.info(f"Date range: {prices.index[0].date()} → {prices.index[-1].date()}")
    logger.info(f"Asset classes: { {k: sum(1 for a in assets if asset_class[a]==k) for k in set(asset_class.values())} }")

    # ── 2. Compute returns and volatility ─────────────────────────────────
    logger.info("Step 2: Computing returns and volatility")
    from data_loader import compute_returns, ewm_vol

    returns_df = compute_returns(prices)          # (T, N) log returns
    sigma_df   = ewm_vol(returns_df, cfg.VOL_SPAN) # (T, N) daily EWMA vol

    # ── 3. Build feature matrix ───────────────────────────────────────────
    logger.info("Step 3: Building 8-factor momentum feature matrix")
    from features import build_feature_matrix, get_feature_tensor

    feature_df = build_feature_matrix(prices, returns_df, sigma_df)
    feature_tensor = get_feature_tensor(feature_df, assets, prices.index)
    # feature_tensor: (T, N, 8)
    logger.info(f"Feature tensor shape: {feature_tensor.shape}")

    # ── 4. Run backtest ───────────────────────────────────────────────────
    logger.info("Step 4: Running walk-forward backtest")
    from backtest import run_backtest

    result = run_backtest(
        prices           = prices,
        feature_df       = feature_df,
        feature_tensor   = feature_tensor,
        returns_df       = returns_df,
        sigma_df         = sigma_df,
        assets           = assets,
        backtest_start   = cfg.BACKTEST_START,
        backtest_end     = cfg.BACKTEST_END,
        retrain_years    = cfg.RETRAIN_YEARS,
        val_fraction     = cfg.VAL_FRACTION,
        lookback_windows = cfg.LOOKBACK_WINDOWS,
        alpha_grid       = cfg.ALPHA_GRID,
        beta_grid        = cfg.BETA_GRID,
        vol_target       = cfg.VOL_TARGET,
        graph_refit_freq = cfg.GRAPH_REFIT_FREQ,
        solver_prefs     = cfg.SOLVER_PREFERENCE,
        cache_dir        = cfg.CACHE_DIR,
        fast_mode        = args.fast,
    )

    # ── 5. Performance metrics ────────────────────────────────────────────
    logger.info("Step 5: Computing performance metrics")
    from metrics import print_summary, cost_adjusted_returns, sharpe_ratio

    # Drop NaN periods (burn-in)
    valid_mask = np.isfinite(result.returns["GMOM"])
    for s in result.returns:
        result.returns[s] = np.where(valid_mask, result.returns[s], 0.0)

    perf_df = print_summary(result.returns)
    perf_path = os.path.join(cfg.OUTPUT_DIR, "performance_table.csv")
    perf_df.to_csv(perf_path)
    logger.info(f"Performance table saved: {perf_path}")

    # ── 6. Transaction cost sensitivity ──────────────────────────────────
    logger.info("Step 6: Transaction cost sensitivity analysis")
    cost_rows = []
    for c_bps in cfg.COST_BPS_SWEEP:
        row = {"Cost_bps": c_bps}
        for strat in ["GMOM", "LinReg", "MACD", "LongOnly"]:
            rets_adj = cost_adjusted_returns(
                result.returns[strat],
                result.turnovers[strat],
                c_bps
            )
            row[f"Sharpe_{strat}"] = round(sharpe_ratio(rets_adj[np.isfinite(rets_adj)]), 4)
        cost_rows.append(row)

    cost_df = pd.DataFrame(cost_rows)
    cost_path = os.path.join(cfg.OUTPUT_DIR, "cost_sensitivity.csv")
    cost_df.to_csv(cost_path, index=False)
    print("\nTransaction Cost Sensitivity:")
    print(cost_df.to_string(index=False))

    # ── 7. Plots ──────────────────────────────────────────────────────────
    logger.info("Step 7: Generating plots")
    _plot_cumulative_returns(result, cfg.OUTPUT_DIR)
    _plot_cost_sensitivity(cost_df, cfg.OUTPUT_DIR)

    logger.info(f"\n✓ All outputs saved to: {cfg.OUTPUT_DIR}")


# ─── Plotting ─────────────────────────────────────────────────────────────────

def _plot_cumulative_returns(result: "BacktestResult", output_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        colors = {"GMOM": "red", "LinReg": "blue", "MACD": "green", "LongOnly": "gray"}
        styles = {"GMOM": "-", "LinReg": "--", "MACD": "-.", "LongOnly": ":"}

        for ax_idx, (ax, scale_label) in enumerate(
            zip(axes, ["Raw Signals", "Vol-Scaled Signals (15% target)"])
        ):
            for strat, rets in result.returns.items():
                cum = np.cumprod(1 + rets)
                ax.plot(result.dates[:len(cum)], cum,
                        label=strat, color=colors[strat], linestyle=styles[strat],
                        linewidth=1.5 if strat == "GMOM" else 1.0)

            ax.set_title(f"Cumulative Returns — {scale_label}", fontsize=12)
            ax.set_ylabel("Cumulative Return")
            ax.legend(loc="upper left", fontsize=9)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.grid(alpha=0.3)
            ax.axhline(1, color="black", linewidth=0.5)

        plt.tight_layout()
        path = os.path.join(output_dir, "cumulative_returns.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Cumulative returns plot saved: {path}")

    except Exception as e:
        logger.warning(f"Plotting failed: {e}. Install matplotlib: pip install matplotlib")


def _plot_cost_sensitivity(cost_df: pd.DataFrame, output_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        colors = {"GMOM": "red", "LinReg": "blue", "MACD": "green", "LongOnly": "gray"}
        for strat in ["GMOM", "LinReg", "MACD", "LongOnly"]:
            col = f"Sharpe_{strat}"
            if col in cost_df.columns:
                ax.plot(cost_df["Cost_bps"], cost_df[col],
                        label=strat, color=colors[strat],
                        marker="o", markersize=4)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Transaction Cost (bps)")
        ax.set_ylabel("Net Sharpe Ratio")
        ax.set_title("Cost-Adjusted Sharpe Ratio", fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)

        path = os.path.join(output_dir, "cost_sensitivity.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Cost sensitivity plot saved: {path}")

    except Exception as e:
        logger.warning(f"Plotting failed: {e}")


# ─── Graph topology analysis (Section 5.1) ────────────────────────────────────

def analyse_graph_topology(result: "BacktestResult", output_dir: str) -> None:
    """
    Compute time series of: edge sparsity, avg node degree, clustering coefficient,
    community ratio, Jaccard index. Matches Figure 6 in the paper.
    """
    from config import ASSET_CLASS

    stats = []
    prev_edges = None

    for date, (A_tilde, valid_idx) in sorted(result.graphs.items()):
        if A_tilde is None or len(valid_idx) < 5:
            continue

        N = len(valid_idx)
        edges = set(zip(*np.where(A_tilde > 0)))
        edges = {(i, j) for i, j in edges if i < j}  # undirected

        n_edges = len(edges)
        n_possible = N * (N - 1) / 2

        sparsity = n_edges / n_possible if n_possible > 0 else 0
        avg_degree = 2 * n_edges / N if N > 0 else 0

        # Jaccard with previous graph
        if prev_edges is not None:
            intersection = len(edges & prev_edges)
            union = len(edges | prev_edges)
            jaccard = intersection / union if union > 0 else 1.0
        else:
            jaccard = 1.0
        prev_edges = edges

        stats.append({
            "Date": date,
            "N_assets": N,
            "N_edges": n_edges,
            "EdgeSparsity": sparsity,
            "AvgDegree": avg_degree,
            "Jaccard": jaccard,
        })

    if stats:
        stats_df = pd.DataFrame(stats).set_index("Date")
        path = os.path.join(output_dir, "graph_statistics.csv")
        stats_df.to_csv(path)
        logger.info(f"Graph topology statistics saved: {path}")
        print("\nGraph Topology Summary:")
        print(stats_df.describe().round(4).to_string())
        return stats_df
    return None


if __name__ == "__main__":
    main()
