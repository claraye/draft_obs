"""
bootstrap.py — Stationary block bootstrap and statistical significance tests

Paper: Section 6 — "Bootstrap Validation"

The paper generates 100 stationary block bootstrap samples from the
actual price data, preserving autocorrelation structure (using ~22 day blocks).
For each sample, the full pipeline is run and metrics are recorded.
Statistical significance: Wilcoxon signed-rank test and KS test.

Reference: Politis & Romano (1994), "The Stationary Bootstrap"
"""
import logging
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


def stationary_block_bootstrap(
    data: np.ndarray,
    block_size: int = 22,
    n_samples: int = 100,
    rng: np.random.Generator = None,
) -> list:
    """
    Stationary block bootstrap: resample the time series preserving
    local autocorrelation structure.

    Algorithm:
      1. Draw random start indices uniformly from [0, T)
      2. Extract blocks of length `block_size` starting at each index
         (with wrap-around to ensure stationarity)
      3. Concatenate blocks to length T

    Args:
        data:       (T, N) return array
        block_size: average block length (~22 for monthly, preserves autocorr)
        n_samples:  number of bootstrap samples to generate
        rng:        numpy random Generator (for reproducibility)

    Returns:
        samples: list of n_samples (T, N) bootstrapped return arrays
    """
    if rng is None:
        rng = np.random.default_rng(42)

    T, N = data.shape
    samples = []

    for _ in range(n_samples):
        # Number of blocks needed (ceil)
        n_blocks = int(np.ceil(T / block_size)) + 1
        # Random start indices (with wrap-around)
        starts = rng.integers(0, T, size=n_blocks)
        # Build bootstrap sample by concatenating blocks
        indices = []
        for s in starts:
            block_idx = [(s + i) % T for i in range(block_size)]
            indices.extend(block_idx)
            if len(indices) >= T:
                break
        indices = indices[:T]
        samples.append(data[indices])

    return samples


def bootstrap_prices_from_returns(
    price_init: np.ndarray,
    boot_returns: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct bootstrapped prices from initial prices and bootstrapped returns.

    price_boot[t] = price_init * exp(cumsum(boot_returns[:t]))

    Args:
        price_init:   (N,) initial price vector
        boot_returns: (T, N) bootstrapped log returns

    Returns:
        prices: (T+1, N) reconstructed price series
    """
    T, N = boot_returns.shape
    prices = np.zeros((T + 1, N))
    prices[0] = price_init
    for t in range(T):
        prices[t + 1] = prices[t] * np.exp(boot_returns[t])
    return prices


def wilcoxon_test(
    sharpes_nmm: np.ndarray,
    sharpes_macd: np.ndarray,
) -> tuple:
    """
    Wilcoxon signed-rank test: H0: NMM Sharpe = MACD Sharpe (paired samples).
    One-sided: tests if NMM Sharpe > MACD Sharpe.

    Returns:
        (statistic, p_value) — p < 0.05 indicates significant improvement
    """
    diff = np.asarray(sharpes_nmm) - np.asarray(sharpes_macd)
    stat, p = sp_stats.wilcoxon(diff, alternative="greater")
    return float(stat), float(p)


def ks_test(
    sharpes_nmm: np.ndarray,
    sharpes_macd: np.ndarray,
) -> tuple:
    """
    Kolmogorov-Smirnov test: H0: NMM and MACD Sharpe distributions are equal.
    One-sided: tests if NMM Sharpe distribution stochastically dominates MACD.

    Returns:
        (statistic, p_value)
    """
    stat, p = sp_stats.ks_2samp(
        np.asarray(sharpes_nmm), np.asarray(sharpes_macd),
        alternative="less"      # NMM CDF is to the right of MACD CDF
    )
    return float(stat), float(p)


def run_bootstrap_comparison(
    returns: np.ndarray,
    prices: np.ndarray,
    assets: list,
    backtest_fn,
    n_samples: int = 100,
    block_size: int = 22,
    rng: np.random.Generator = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the full bootstrap comparison pipeline.

    For each bootstrap sample:
      1. Generate bootstrapped returns
      2. Run backtest_fn(returns_boot, prices_boot) → (macd_rets, nmm_rets)
      3. Record Sharpe ratio for each strategy

    Args:
        returns:      (T, N) actual log returns
        prices:       (T, N) actual prices (for price reconstruction)
        assets:       list of N asset names
        backtest_fn:  callable(returns_boot, prices_boot) -> dict {strategy: daily_rets}
        n_samples:    number of bootstrap iterations
        block_size:   bootstrap block size
        rng:          random generator for reproducibility
        verbose:      print progress

    Returns:
        bootstrap_df: DataFrame with Sharpe ratios for each strategy × sample
    """
    from metrics import sharpe_ratio

    samples = stationary_block_bootstrap(returns, block_size, n_samples, rng)
    price_init = prices[0]

    all_sharpes = []
    for i, boot_rets in enumerate(samples):
        if verbose and (i + 1) % 10 == 0:
            logger.info(f"Bootstrap sample {i + 1}/{n_samples}")
        boot_prices = bootstrap_prices_from_returns(price_init, boot_rets)
        try:
            strategy_rets = backtest_fn(boot_rets, boot_prices)
            row = {k: sharpe_ratio(np.asarray(v)[np.isfinite(np.asarray(v))])
                   for k, v in strategy_rets.items()}
            row["sample"] = i
            all_sharpes.append(row)
        except Exception as e:
            logger.warning(f"Bootstrap sample {i} failed: {e}")

    return pd.DataFrame(all_sharpes).set_index("sample")


def print_significance_table(
    bootstrap_df: pd.DataFrame,
    baseline_col: str = "MACD",
) -> pd.DataFrame:
    """
    Print Wilcoxon and KS test results for all strategies vs baseline.
    Matches Table 2 / Table 3 in the paper.

    Args:
        bootstrap_df: DataFrame(n_samples, n_strategies) of Sharpe ratios
        baseline_col: column name of the baseline strategy (MACD)

    Returns:
        sig_df: DataFrame with test statistics and p-values
    """
    base = bootstrap_df[baseline_col].values
    rows = []
    for col in bootstrap_df.columns:
        if col == baseline_col:
            continue
        sharpes = bootstrap_df[col].values
        w_stat, w_p = wilcoxon_test(sharpes, base)
        ks_stat, ks_p = ks_test(sharpes, base)
        rows.append({
            "Strategy":        col,
            "Mean_Sharpe":     round(sharpes.mean(), 4),
            "MACD_Mean_Sharpe":round(base.mean(), 4),
            "Wilcoxon_stat":   round(w_stat, 2),
            "Wilcoxon_p":      round(w_p, 4),
            "KS_stat":         round(ks_stat, 4),
            "KS_p":            round(ks_p, 4),
            "Significant_5pct":(w_p < 0.05 and ks_p < 0.05),
        })

    df = pd.DataFrame(rows).set_index("Strategy")
    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE: NMM vs MACD Baseline")
    print("=" * 70)
    print(df.to_string())
    print("=" * 70)
    return df
