"""
graph_learning.py — Convex graph learning (Dong et al., 2019)
Paper Section 3.1: learns sparse dynamic adjacency matrix from momentum features.

Optimisation problem (Eq. 4):
  min_{A} tr(V^T (D - A) V) - alpha * 1^T log(A @ 1) + beta * ||A||_F^2
  s.t.  A_ij = A_ji >= 0  for all i!=j,   diag(A) = 0

where:
  V in R^(N x 8*delta): stacked momentum features over lookback delta
  D: diagonal matrix with D_ii = sum_j A_ij
  tr(V^T (D-A) V) = (1/2) sum_{ij} A_ij ||v_i - v_j||^2   [Laplacian smoothness]

Ensemble (Eq. 5): A_bar = (1/K) sum_k A^(k)  for K=5 lookback windows
Normalisation (Eq. 6): A_tilde = D_bar^{-1/2} A_bar D_bar^{-1/2}
"""
from __future__ import annotations

import numpy as np
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

_SOLVER_AVAILABLE = {}  # cache solver availability checks


def _check_solver(name: str) -> bool:
    if name in _SOLVER_AVAILABLE:
        return _SOLVER_AVAILABLE[name]
    try:
        import cvxpy as cp
        prob = cp.Problem(cp.Minimize(cp.Variable()))
        prob.solve(solver=getattr(cp, name, None), verbose=False)
        _SOLVER_AVAILABLE[name] = True
    except Exception:
        _SOLVER_AVAILABLE[name] = False
    return _SOLVER_AVAILABLE[name]


def _get_solver(preferences: list[str]) -> str:
    try:
        import cvxpy as cp
        for name in preferences:
            if hasattr(cp, name) and _check_solver(name):
                return name
    except ImportError:
        pass
    return "SCS"  # fallback


# ─── Core graph learning ──────────────────────────────────────────────────────

def learn_graph_cvxpy(
    V: np.ndarray,
    alpha: float,
    beta: float,
    solver: str = "SCS",
    verbose: bool = False,
) -> np.ndarray | None:
    """
    Solve the Dong et al. graph learning problem via CVXPY.

    Args:
      V:     (N, 8δ) feature matrix (rows = assets, columns = stacked features)
      alpha: log-barrier weight (controls connectivity)
      beta:  L2 regularisation weight (controls sparsity/smoothness)
      solver: CVXPY solver name

    Returns:
      A: (N, N) adjacency matrix, or None if solver fails
    """
    try:
        import cvxpy as cp
    except ImportError:
        raise ImportError("cvxpy not installed. Run: pip install cvxpy")

    N = V.shape[0]
    ones = np.ones(N)

    # Pairwise squared Euclidean distances: Θ_ij = ||v_i - v_j||^2
    # tr(V^T L V) = (1/2) Σ_{ij} A_ij Θ_ij
    sq_norms = np.sum(V ** 2, axis=1, keepdims=True)  # (N,1)
    Theta = sq_norms + sq_norms.T - 2 * (V @ V.T)    # (N,N)
    Theta = np.maximum(Theta, 0)                       # numerical safety

    # CVXPY variable: symmetric N×N matrix
    A = cp.Variable((N, N), symmetric=True)

    # Degree vector: d = A @ 1
    d = A @ ones  # (N,)

    # Objective terms
    laplacian_term = 0.5 * cp.sum(cp.multiply(A, Theta))
    log_term       = -alpha * cp.sum(cp.log(d))
    l2_term        = beta * cp.sum_squares(A)

    objective = cp.Minimize(laplacian_term + log_term + l2_term)

    constraints = [
        A >= 0,
        cp.diag(A) == 0,
        d >= 1e-6,  # prevent log(-inf); ensure connectivity
    ]

    prob = cp.Problem(objective, constraints)

    solver_obj = getattr(cp, solver, cp.SCS)
    try:
        prob.solve(solver=solver_obj, verbose=verbose, max_iters=5000)
    except Exception as e:
        logger.warning(f"Solver {solver} failed: {e}. Trying SCS.")
        try:
            prob.solve(solver=cp.SCS, verbose=False, max_iters=10000)
        except Exception as e2:
            logger.error(f"SCS also failed: {e2}")
            return None

    if A.value is None:
        logger.warning("Graph learning returned None (infeasible/unbounded).")
        return None

    A_val = np.array(A.value)
    A_val = np.maximum(A_val, 0)          # enforce non-negativity numerically
    np.fill_diagonal(A_val, 0)            # zero diagonal
    A_val = (A_val + A_val.T) / 2        # enforce exact symmetry

    return A_val


def learn_graph_closed_form(V: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """
    Fast approximate graph learning via correlation-based proxy.
    Used as fallback when CVXPY is unavailable.

    Constructs A_ij ∝ max(0, v_i · v_j / (||v_i|| ||v_j||)) with
    L2 sparsification. This captures the Laplacian smoothness intuition
    (connecting assets with similar momentum features) without the solver.
    """
    N = V.shape[0]

    # Normalise rows
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    V_normed = V / (norms + 1e-10)

    # Cosine similarity
    sim = V_normed @ V_normed.T  # (N,N) ∈ [-1,1]

    # Keep only positive similarities (momentum co-movement, not anti-movement)
    A = np.maximum(sim, 0)
    np.fill_diagonal(A, 0)

    # Soft-threshold by beta (higher beta → sparser)
    threshold = np.percentile(A[A > 0], min(beta * 10, 90)) if (A > 0).any() else 0
    A = np.maximum(A - threshold, 0)

    # Ensure minimum connectivity (log-barrier proxy for alpha)
    if A.sum() == 0:
        # Connect each node to its 3 most similar neighbours
        for i in range(N):
            top3 = np.argsort(sim[i])[-4:-1]  # top 3 excluding self
            A[i, top3] = np.maximum(sim[i, top3], 0)
            A[top3, i] = A[i, top3]

    return A


def learn_graph(
    V: np.ndarray,
    alpha: float,
    beta: float,
    solver_prefs: list[str] = None,
) -> np.ndarray:
    """
    Learn graph adjacency from feature matrix V, using CVXPY if available.
    Falls back to correlation-based approximation otherwise.
    """
    if solver_prefs is None:
        from config import SOLVER_PREFERENCE
        solver_prefs = SOLVER_PREFERENCE

    try:
        import cvxpy as cp  # noqa — check availability
        solver = _get_solver(solver_prefs)
        A = learn_graph_cvxpy(V, alpha, beta, solver=solver)
        if A is not None:
            return A
    except ImportError:
        logger.warning("CVXPY not available; using closed-form approximation.")

    return learn_graph_closed_form(V, alpha, beta)


# ─── Ensemble ─────────────────────────────────────────────────────────────────

def ensemble_graph(
    feature_tensor: np.ndarray,
    t_idx: int,
    valid_idx: np.ndarray,
    alpha: float,
    beta: float,
    lookback_windows: list[int],
    solver_prefs: list[str] = None,
) -> np.ndarray | None:
    """
    Compute ensemble adjacency matrix Ā_t = (1/K) Σ_k A^(k)  [Eq.5]
    by fitting one graph per lookback window δ_k.

    Args:
      feature_tensor: (T, N_full, 8) array of all features
      t_idx:          current time index in feature_tensor
      valid_idx:      indices of assets available at t
      alpha, beta:    graph learning hyperparameters
      lookback_windows: [δ_1, ..., δ_K]

    Returns:
      Ā: (N_valid, N_valid) ensemble adjacency, or None if all windows fail
    """
    from features import stack_lookback

    N_valid = len(valid_idx)
    A_sum = np.zeros((N_valid, N_valid))
    n_success = 0

    # Create a boolean mask over all assets for stack_lookback
    N_full = feature_tensor.shape[1]
    assets_mask = np.zeros(N_full, dtype=bool)
    assets_mask[valid_idx] = True

    for delta in lookback_windows:
        V, idx = stack_lookback(feature_tensor, t_idx, delta, assets_mask)
        if V is None or len(idx) < 5:
            continue

        # If the valid assets differ from our expected set, skip
        if len(idx) != N_valid:
            continue

        A_k = learn_graph(V, alpha, beta, solver_prefs)
        A_sum += A_k
        n_success += 1

    if n_success == 0:
        return None

    A_bar = A_sum / n_success  # Eq.5
    return A_bar


# ─── Normalisation ────────────────────────────────────────────────────────────

def normalise_graph(A_bar: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Symmetric degree normalisation (Eq. 6):
      Ã = D̄^{-1/2} Ā D̄^{-1/2}

    where D̄_ii = Σ_j Ā_ij.
    """
    d = A_bar.sum(axis=1)                   # degree vector
    d_inv_sqrt = 1.0 / (np.sqrt(d) + eps)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    A_tilde = D_inv_sqrt @ A_bar @ D_inv_sqrt
    return A_tilde


# ─── Hyperparameter selection ─────────────────────────────────────────────────

def select_hyperparams(
    feature_tensor: np.ndarray,
    train_dates_idx: np.ndarray,
    val_dates_idx: np.ndarray,
    all_assets_idx: np.ndarray,
    alpha_grid: list[float],
    beta_grid: list[float],
    lookback_windows: list[int],
    solver_prefs: list[str] = None,
    subsample: int = 5,
) -> tuple[float, float]:
    """
    Grid search over (alpha, beta) using validation set Sharpe ratio as criterion.
    To reduce computation, evaluates on a subsampled set of validation dates.

    Returns best (alpha, beta).
    """
    from signals import ols_train

    logger.info(f"Hyperparameter search: {len(alpha_grid)*len(beta_grid)} combinations")

    # Use last lookback only for speed in grid search
    delta_search = [lookback_windows[-1]]

    best_sharpe = -np.inf
    best_params = (alpha_grid[0], beta_grid[0])

    # Subsample validation dates for speed
    val_idx_sub = val_dates_idx[::subsample]

    for alpha in alpha_grid:
        for beta in beta_grid:
            try:
                sharpe = _evaluate_params(
                    feature_tensor, train_dates_idx, val_idx_sub,
                    all_assets_idx, alpha, beta, delta_search, solver_prefs
                )
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = (alpha, beta)
            except Exception as e:
                logger.debug(f"  α={alpha}, β={beta} failed: {e}")

    logger.info(f"Best params: α={best_params[0]}, β={best_params[1]}, Sharpe={best_sharpe:.3f}")
    return best_params


def _evaluate_params(
    feature_tensor, train_idx, val_idx, asset_idx,
    alpha, beta, lookback_windows, solver_prefs
) -> float:
    """Helper: compute net Sharpe on val set for given (α,β)."""
    from signals import propagate_features, ols_train

    N_full = feature_tensor.shape[1]
    assets_mask = np.zeros(N_full, dtype=bool)
    assets_mask[asset_idx] = True

    # Compute network momentum features on training set (subsample for speed)
    train_features, train_targets, train_valid = [], [], []

    # Use training data to fit OLS
    t_step = max(1, len(train_idx) // 50)  # limit to ~50 train points
    for t in train_idx[::t_step]:
        V, valid = _get_V_and_propagate(
            feature_tensor, t, assets_mask, alpha, beta,
            [lookback_windows[-1]], solver_prefs
        )
        if V is None:
            continue
        train_features.append(V["net_features"])
        train_targets.append(V["target"])
        train_valid.append(V["valid_idx"])

    if not train_features:
        return -np.inf

    # Fit OLS
    X_train = np.vstack([f for f in train_features if f is not None])
    y_train = np.concatenate([tgt for tgt in train_targets if tgt is not None])
    if len(X_train) < 20:
        return -np.inf

    beta_coef, intercept = ols_train(X_train, y_train)

    # Evaluate on validation set
    returns = []
    for t in val_idx:
        V = _get_V_and_propagate(
            feature_tensor, t, assets_mask, alpha, beta,
            [lookback_windows[-1]], solver_prefs
        )
        if V is None or V["net_features"] is None:
            continue
        preds = V["net_features"] @ beta_coef + intercept
        signs = np.sign(preds)
        if "day_return" in V:
            returns.append(np.mean(signs * V["day_return"]))

    if len(returns) < 10:
        return -np.inf

    returns = np.array(returns)
    sharpe = returns.mean() / (returns.std() + 1e-10) * np.sqrt(252)
    return sharpe


def _get_V_and_propagate(
    feature_tensor, t_idx, assets_mask, alpha, beta, lookback_windows, solver_prefs
):
    """Minimal helper used in hyperparameter search."""
    from features import stack_lookback
    N_full = feature_tensor.shape[1]

    V_raw, valid_idx = stack_lookback(feature_tensor, t_idx, lookback_windows[-1], assets_mask)
    if V_raw is None:
        return None

    A = learn_graph(V_raw, alpha, beta, solver_prefs)
    if A is None:
        return None
    A_tilde = normalise_graph(A)

    # Individual features at time t
    u_t = feature_tensor[t_idx, valid_idx, :]  # (N_valid, 8)

    # Propagate
    net_features = A_tilde @ u_t  # (N_valid, 8)

    # Target: next-day vol-scaled return (proxy: just raw return for now)
    if t_idx + 1 < feature_tensor.shape[0]:
        target = feature_tensor[t_idx + 1, valid_idx, 0]  # use 1d vsr as proxy
    else:
        target = np.zeros(len(valid_idx))

    return {
        "net_features": net_features,
        "target": target,
        "valid_idx": valid_idx,
        "day_return": target,
    }
