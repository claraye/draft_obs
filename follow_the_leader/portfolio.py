"""
portfolio.py — Position sizing and PnL computation

Paper: Section 5 — "Portfolio Construction" (Definition 5.2)

Position formula (simplified for ETF proxies, F=price, E=1, Γ=1):

  X_{t,m} = (1/M) · (1/K) · Σ_k φ(S^k_{t,m}) · σ_tgt / (σ^22_{t,m} · √252)

where:
  S^k_{t,m}  = R^k_{t,m}  (MACD baseline) or R̃^k_{t,m} (NMM model)
  φ(·)       = response function (Baz et al.)
  σ_tgt      = 10% annualised target vol
  σ^22_{t,m} = 22-day EWMA daily vol
  M          = number of markets (equal contribution)
  K          = number of oscillator speeds

PnL with 2-day execution lag:
  r^portfolio_{t+2} = Σ_m X_{t,m} · r_{t+2,m}

The 2-day lag (signal@t → trade@t+1 → PnL@t+2) is a conservative assumption
more realistic than the 1-day lag prevalent in much of the literature.
"""
import numpy as np
from features import response_function


def compute_positions(
    signal_matrix: np.ndarray,
    vol_t: np.ndarray,
    sigma_tgt: float = 0.10,
    tdpy: int = 252,
) -> np.ndarray:
    """
    Compute portfolio positions (weights) at time t.

    Position formula:
      X_{t,m} = (1/M) · (1/K) · Σ_k φ(S^k_{t,m}) · σ_tgt / (σ^22_{t,m} · √252)

    Args:
        signal_matrix: (N, K) signal matrix (MACD or NMM oscillators)
        vol_t:         (N,) daily EWMA standard deviation at time t
        sigma_tgt:     annualised target volatility (default 0.10 = 10%)
        tdpy:          trading days per year (default 252)

    Returns:
        positions: (N,) portfolio weights (can be negative for short positions)
    """
    N, K = signal_matrix.shape

    # Vol-scaling factor per asset: σ_tgt / (σ^22_t · √252)
    vol_scale = sigma_tgt / (vol_t * np.sqrt(tdpy) + 1e-10)   # (N,)

    # Apply response function to each (N, K) element
    phi_S = response_function(signal_matrix)   # (N, K)

    # Average across speeds K, then vol-scale and equal-weight across M markets
    avg_signal = phi_S.mean(axis=1)            # (N,)
    positions  = avg_signal * vol_scale / N    # (N,)

    return positions


def compute_turnover(
    pos_t: np.ndarray,
    pos_tm1: np.ndarray,
) -> float:
    """
    Daily portfolio turnover = Σ_m |X_{t,m} - X_{t-1,m}| / Σ_m |X_{t,m}|.
    Returns absolute sum of position changes (dollar turnover proxy).
    """
    return float(np.nansum(np.abs(pos_t - pos_tm1)))


def apply_transaction_costs(
    daily_returns: np.ndarray,
    daily_turnovers: np.ndarray,
    cost_bps: float,
) -> np.ndarray:
    """
    Deduct transaction costs from daily portfolio returns.

    Net return_t = gross return_t - (cost_bps / 10000) * turnover_t

    Args:
        daily_returns:  (T,) gross daily portfolio returns
        daily_turnovers:(T,) daily turnover (sum of |Δposition|)
        cost_bps:       transaction cost in basis points per unit turnover

    Returns:
        net_returns: (T,) net daily returns after costs
    """
    cost = (cost_bps / 10_000) * daily_turnovers
    return daily_returns - cost
