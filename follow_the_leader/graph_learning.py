"""
graph_learning.py — Sparse graph learning from lead-lag matrix

Paper: Section 4 — "Network Momentum Matrix"

FTL objective (Equation from paper):
  min_{A ≥ 0, A = A^T}  ||V - A||^2_F  +  α · 1^T A 1  +  β · ||A||^2_F

Differences from Dong et al. (2019) / Network Momentum paper:
  - No Laplacian smoothness term (tr(V^T L V))
  - No log-barrier term (−α log d)
  - V is an integer lead-lag matrix (not a feature similarity matrix)
  - Simpler convex QP: regularised non-negative least squares with symmetry
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)


def learn_graph_cvxpy(
    V: np.ndarray,
    alpha: float,
    beta: float,
    solver_prefs: list = None,
) -> np.ndarray:
    """
    Solve the FTL graph learning problem via CVXPY:

      min_{A}  ||V - A||^2_F  +  alpha · sum(A)  +  beta · ||A||^2_F
      s.t.     A ≥ 0,  A = A^T

    The α·sum(A) term (L1-style) promotes sparsity — many edges become exactly zero.
    The β·||A||^2_F term (L2) prevents excessively large edge weights.
    The non-negativity constraint A ≥ 0 combined with skew-symmetric V means
    only positive lead-lag values (|V[i,j]| where V[i,j] > 0) survive in A.

    Args:
        V:            (N, N) lead-lag matrix (skew-symmetric for DTW/Lévy)
        alpha:        sparsity regularisation strength
        beta:         weight regularisation strength
        solver_prefs: preferred solver list (MOSEK > CLARABEL > SCS)

    Returns:
        A: (N, N) sparse symmetric adjacency matrix with zero diagonal
    """
    if solver_prefs is None:
        solver_prefs = ["CLARABEL", "SCS"]

    try:
        import cvxpy as cp
    except ImportError:
        logger.warning("CVXPY not available; falling back to analytical solution")
        return _learn_graph_analytical(V, alpha, beta)

    N = V.shape[0]
    A = cp.Variable((N, N), symmetric=True)
    objective = cp.Minimize(
        cp.sum_squares(V - A)           # ||V - A||^2_F  (data fidelity)
        + alpha * cp.sum(A)             # alpha · sum(A)  (sparsity)
        + beta  * cp.sum_squares(A)     # beta · ||A||^2_F  (smoothness)
    )
    constraints = [A >= 0]
    problem = cp.Problem(objective, constraints)

    for solver_name in solver_prefs:
        try:
            problem.solve(solver=solver_name, warm_start=True)
            if problem.status in ("optimal", "optimal_inaccurate") and A.value is not None:
                result = np.maximum(A.value, 0.0)
                np.fill_diagonal(result, 0.0)
                return result
        except Exception as e:
            logger.debug(f"Solver {solver_name} failed: {e}")

    logger.warning("All CVXPY solvers failed; falling back to analytical solution")
    return _learn_graph_analytical(V, alpha, beta)


def _learn_graph_analytical(
    V: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """
    Closed-form approximate solution when CVXPY is unavailable.

    Ignoring the symmetry constraint, the unconstrained minimiser of
      ||V - A||^2_F + alpha·sum(A) + beta·||A||^2_F
    is element-wise:
      A*_{ij} = (V_{ij} - alpha/2) / (1 + beta)

    We symmetrise: A_sym = (A* + A*^T) / 2, then project to non-negative orthant.
    This is exact when V is symmetric; for skew-symmetric V it captures only the
    positive part (leaders in each direction).
    """
    A = (V - alpha / 2.0) / (1.0 + beta)
    A = (A + A.T) / 2.0       # enforce symmetry
    A = np.maximum(A, 0.0)    # project to A ≥ 0
    np.fill_diagonal(A, 0.0)  # zero diagonal (no self-loops before normalisation)
    return A


def ensemble_graph(
    V_list: list,
    alpha: float,
    beta: float,
    solver_prefs: list = None,
) -> np.ndarray:
    """
    Ensemble: learn one adjacency matrix per lookback window, then average.

    Ā_t = (1/S) · Σ_{s=1}^{S} A^{(s)}_t

    Args:
        V_list:       list of S (N, N) lead-lag matrices (one per lookback δ)
        alpha, beta:  graph learning hyperparameters
        solver_prefs: CVXPY solver preference list

    Returns:
        A_bar: (N, N) averaged adjacency matrix
    """
    if not V_list:
        raise ValueError("V_list is empty")
    A_list = [learn_graph_cvxpy(V, alpha, beta, solver_prefs) for V in V_list]
    return np.mean(A_list, axis=0)


def normalise_graph(A: np.ndarray, add_self_loops: bool = True) -> np.ndarray:
    """
    Symmetric degree normalisation: Ã = D^{-1/2} Ā D^{-1/2}

    If add_self_loops=True (default), Ā = A + I before normalising.
    Self-loops ensure each market m's own oscillator contributes to its
    network momentum feature R̃^k_m, alongside its graph neighbours.

    Without self-loops, R̃^k_m is purely neighbours' oscillators (zero own signal).

    Args:
        A:               (N, N) symmetric adjacency matrix (A ≥ 0)
        add_self_loops:  whether to add identity before normalising

    Returns:
        A_tilde: (N, N) normalised adjacency matrix
    """
    A_bar = A.copy()
    if add_self_loops:
        A_bar = A_bar + np.eye(A.shape[0])
    d = A_bar.sum(axis=1)
    d_inv_sqrt = np.where(d > 1e-10, d ** (-0.5), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return D_inv_sqrt @ A_bar @ D_inv_sqrt


def grid_search_hyperparams(
    V_list: list,
    returns_val: np.ndarray,
    oscillators_val: dict,
    vol_val: np.ndarray,
    alpha_grid: list,
    beta_grid: list,
    sigma_tgt: float = 0.10,
    tdpy: int = 252,
    solver_prefs: list = None,
    add_self_loops: bool = True,
) -> tuple:
    """
    In-sample grid search for best (alpha, beta) by maximising Sharpe ratio.

    For each (alpha, beta):
      1. Build ensemble graph from V_list
      2. Propagate oscillators: R̃ = Ã @ R
      3. Compute portfolio returns for validation period
      4. Record annualised Sharpe

    Uses the midpoint of the validation period's oscillators as a snapshot
    for computational efficiency (avoids recomputing V_list for each date).

    Args:
        V_list:          list of S (N, N) lead-lag matrices (pre-computed)
        returns_val:     (T_val, N) log returns for validation period
        oscillators_val: dict {k: (T_val, N) DataFrame or array}
        vol_val:         (T_val, N) daily EWMA vol for validation period
        alpha_grid, beta_grid: hyperparameter grids to search
        sigma_tgt:       target annualised vol

    Returns:
        (best_alpha, best_beta): hyperparameters maximising in-sample Sharpe
    """
    from features import response_function

    T_val, N = returns_val.shape
    k_values = sorted(oscillators_val.keys())
    K = len(k_values)

    best_sharpe = -np.inf
    best_alpha  = alpha_grid[len(alpha_grid) // 2]
    best_beta   = beta_grid[len(beta_grid) // 2]

    n_candidates = len(alpha_grid) * len(beta_grid)
    logger.info(f"Grid search: {n_candidates} (alpha, beta) candidates ...")

    for alpha in alpha_grid:
        for beta in beta_grid:
            try:
                A_bar   = ensemble_graph(V_list, alpha, beta, solver_prefs)
                A_tilde = normalise_graph(A_bar, add_self_loops=add_self_loops)
            except Exception as e:
                logger.debug(f"Graph failed for alpha={alpha}, beta={beta}: {e}")
                continue

            # Evaluate portfolio on validation returns
            pnl = []
            for t in range(T_val - 2):
                # Stack oscillators at t: (N, K)
                R_t = np.column_stack([
                    oscillators_val[k][t] if isinstance(oscillators_val[k], np.ndarray)
                    else oscillators_val[k].iloc[t].values
                    for k in k_values
                ])
                R_tilde = A_tilde @ R_t          # (N, K)
                vol_t   = vol_val[t]              # (N,)
                scale   = sigma_tgt / (vol_t * np.sqrt(tdpy) + 1e-10)  # (N,)
                weights = (response_function(R_tilde) * scale[:, None]).mean(axis=1) / N
                pnl.append(float(np.nansum(weights * returns_val[t + 2])))

            pnl = np.array(pnl)
            pnl = pnl[np.isfinite(pnl)]
            if len(pnl) < 20:
                continue
            sharpe = float(pnl.mean() / (pnl.std(ddof=1) + 1e-12) * np.sqrt(tdpy))
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_alpha, best_beta = alpha, beta

    logger.info(f"Best: alpha={best_alpha}, beta={best_beta}, "
                f"in-sample Sharpe={best_sharpe:.4f}")
    return best_alpha, best_beta
