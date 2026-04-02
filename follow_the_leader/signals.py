"""
signals.py — Network momentum feature propagation

Paper: Section 4.2 — "Network Momentum Features"

Network momentum feature (Definition 4.2):
  R̃^k_{t,m} = Σ_{n ∈ N_t(m)} Ã_{m,n} · R^k_{t,n}

In matrix form:
  R̃^k_t = Ã_t · R^k_t

where Ã_t is the (N, N) normalised graph adjacency (with self-loops),
and R^k_t is the (N,) vector of MACD oscillator values at speed k.

Interpretation: Market m's NMM signal = weighted sum of its graph
neighbours' oscillators, where weights reflect lead-lag strength.
With self-loops (add_self_loops=True), own oscillator is also included.
"""
import numpy as np


def propagate_oscillators(
    A_tilde: np.ndarray,
    oscillators_t: dict,
) -> dict:
    """
    Propagate MACD oscillators through the normalised graph:
      R̃^k_t = Ã_t · R^k_t  for each speed k.

    Args:
        A_tilde:       (N, N) normalised adjacency matrix
        oscillators_t: dict {k: (N,) array} — oscillator vector at time t

    Returns:
        R_tilde_t: dict {k: (N,) array} — network momentum oscillators
    """
    return {k: A_tilde @ np.asarray(v) for k, v in oscillators_t.items()}


def propagate_oscillators_matrix(
    A_tilde: np.ndarray,
    R_matrix: np.ndarray,
) -> np.ndarray:
    """
    Propagate oscillator matrix: R̃ = Ã · R.

    Args:
        A_tilde:  (N, N) normalised adjacency
        R_matrix: (N, K) oscillator matrix at time t — one column per speed

    Returns:
        R_tilde:  (N, K) network momentum oscillator matrix
    """
    return A_tilde @ R_matrix


def build_signal_matrix(
    oscillators_t: dict,
    assets: list,
    use_network: bool = False,
    A_tilde: np.ndarray = None,
) -> np.ndarray:
    """
    Build (N, K) signal matrix at time t for position sizing.

    For MACD baseline:   S = R  (own oscillators)
    For NMM:             S = Ã · R  (graph-propagated oscillators)

    Args:
        oscillators_t: dict {k: (N,) array or pd.Series}
        assets:        list of N asset names (for alignment)
        use_network:   whether to apply graph propagation
        A_tilde:       (N, N) normalised adjacency (required if use_network=True)

    Returns:
        S: (N, K) signal matrix
    """
    k_values = sorted(oscillators_t.keys())
    K = len(k_values)
    N = len(assets)

    # Build (N, K) raw oscillator matrix
    R = np.empty((N, K))
    for ki, k in enumerate(k_values):
        v = oscillators_t[k]
        R[:, ki] = v.values if hasattr(v, "values") else np.asarray(v)

    # Replace NaN with zero (missing data / burn-in)
    R = np.where(np.isfinite(R), R, 0.0)

    if use_network:
        if A_tilde is None:
            raise ValueError("A_tilde must be provided when use_network=True")
        return propagate_oscillators_matrix(A_tilde, R)

    return R
