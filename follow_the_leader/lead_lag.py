"""
lead_lag.py — Lead-lag detection methods for commodity futures

Paper: Section 3 — "Lead-Lag Relationships in Commodity Futures"

Methods implemented:
  1. Lévy Area (path signature): O(T·N²), captures non-linear lead-lag
  2. DTW   — standard Dynamic Time Warping (1D)
  3. DDTW  — Derivative DTW (Keogh & Pazzani 2001)
  4. SDTW  — Shape DTW (Zhao & Itti 2018) — multidimensional local descriptors
  5. SDDTW — Shape + Derivative DTW
  6. XCorr — Cross-correlation (fast approximation bonus method)

All methods produce a skew-symmetric (N×N) lead-lag matrix V where:
  V[i, j] > 0  →  asset i leads asset j
  V[j, i] = -V[i, j]  (skew-symmetric)

Computational notes:
  - DTW variants are O(T²) per pair in Python. Install dtaidistance for ~100x speedup.
  - Lévy area is O(T·N²), much faster for large T.
  - xcorr is O(T·log(T)·N²) via FFT, practical without any C extensions.
  - With GRAPH_REFIT_FREQ=22 (monthly) and N=19, DTW is feasible (~3-5 min per run).
"""
import logging
import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

# ─── Backend detection ────────────────────────────────────────────────────────

try:
    from dtaidistance import dtw as _dtaidist_dtw
    _DTAIDIST_AVAILABLE = True
    logger.debug("dtaidistance available: using C-accelerated DTW")
except ImportError:
    _DTAIDIST_AVAILABLE = False

try:
    import numba as _numba
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False


# ─── 1. Lévy Area ─────────────────────────────────────────────────────────────

def _levy_area_pair(path_x: np.ndarray, path_y: np.ndarray) -> float:
    """
    Discrete Lévy area between two cumulative paths using midpoint quadrature:

      A^Lévy_{x,y} = (1/2)(∫ X dY  −  ∫ Y dX)
                   ≈ (1/2) Σ_t [x_mid_t · Δy_t  −  y_mid_t · Δx_t]

    Theorem (Cartea, Cucuringu, Jin 2023):
      sgn(A^Lévy_{x,y}) = sgn(lag) × sgn(β)
    where lag > 0 means x leads y with lead-lag coefficient β.

    Positive return: x is the leader, y is the follower.
    """
    dx = np.diff(path_x)
    dy = np.diff(path_y)
    x_mid = 0.5 * (path_x[:-1] + path_x[1:])
    y_mid = 0.5 * (path_y[:-1] + path_y[1:])
    return 0.5 * (np.dot(x_mid, dy) - np.dot(y_mid, dx))


def levy_area_matrix(returns_window: np.ndarray) -> np.ndarray:
    """
    Compute (N, N) skew-symmetric Lévy area matrix V from a return window.

    V[i, j] > 0  →  asset i leads asset j.
    Complexity: O(T·N²).

    Args:
        returns_window: (T, N) log returns over lookback window

    Returns:
        V: (N, N) skew-symmetric float matrix
    """
    T, N = returns_window.shape
    paths = np.cumsum(returns_window, axis=0)   # cumulative paths (T, N)
    V = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            a = _levy_area_pair(paths[:, i], paths[:, j])
            V[i, j] = a
            V[j, i] = -a
    return V


# ─── 2. DTW Utilities ─────────────────────────────────────────────────────────

def _dtw_path_numpy(x: np.ndarray, y: np.ndarray) -> list:
    """
    Pure-numpy DTW warping path via dynamic programming.
    Cost: squared Euclidean distance at each aligned point.

    Returns list of (i, j) index pairs along optimal warping path.
    Complexity: O(|x|·|y|) time and memory.
    """
    n, m = len(x), len(y)
    C = (x[:, None] - y[None, :]) ** 2   # (n, m) local cost matrix
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = C[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    # Backtrack from (n, m) to (0, 0)
    path = []
    i, j = n, m
    while i > 0 or j > 0:
        path.append((i - 1, j - 1))
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            best = int(np.argmin([D[i - 1, j], D[i, j - 1], D[i - 1, j - 1]]))
            if best == 0:
                i -= 1
            elif best == 1:
                j -= 1
            else:
                i -= 1
                j -= 1
    return path[::-1]


def _dtw_path_nd_numpy(X: np.ndarray, Y: np.ndarray) -> list:
    """
    DTW warping path for multivariate sequences.
    X: (n, d), Y: (m, d) — local cost is squared L2 norm between descriptors.
    """
    n, m = len(X), len(Y)
    diff = X[:, None, :] - Y[None, :, :]    # (n, m, d)
    C = np.sum(diff ** 2, axis=-1)           # (n, m)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = C[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    path = []
    i, j = n, m
    while i > 0 or j > 0:
        path.append((i - 1, j - 1))
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            best = int(np.argmin([D[i - 1, j], D[i, j - 1], D[i - 1, j - 1]]))
            if best == 0:
                i -= 1
            elif best == 1:
                j -= 1
            else:
                i -= 1
                j -= 1
    return path[::-1]


def _dtw_path_fast(x: np.ndarray, y: np.ndarray) -> list:
    """
    DTW warping path using dtaidistance (C extension) if available,
    falling back to pure numpy otherwise.
    """
    if _DTAIDIST_AVAILABLE:
        try:
            return list(_dtaidist_dtw.warping_path(
                x.astype(np.double), y.astype(np.double)
            ))
        except Exception:
            pass
    return _dtw_path_numpy(x, y)


def _lag_from_path(path: list) -> int:
    """
    Estimate lead-lag as the mode of (j − i) over the warping path.
    Positive lag: j is behind i → i is the leader.
    """
    if not path:
        return 0
    lags = [j - i for (i, j) in path]
    return int(sp_stats.mode(lags, keepdims=False).mode)


# ─── 3. Series Transformations ────────────────────────────────────────────────

def _derivative_transform(x: np.ndarray) -> np.ndarray:
    """
    DDTW derivative estimate (Keogh & Pazzani 2001):
      D(x)_t = ((x_t − x_{t-1}) + (x_{t+1} − x_{t-1}) / 2) / 2  for interior points
      D(x)_0 = x_1 − x_0                                           (forward diff)
      D(x)_T = x_T − x_{T-1}                                       (backward diff)
    """
    n = len(x)
    d = np.empty(n)
    d[0]    = x[1] - x[0]
    d[-1]   = x[-1] - x[-2]
    d[1:-1] = ((x[1:-1] - x[:-2]) + (x[2:] - x[:-2]) / 2.0) / 2.0
    return d


def _shape_descriptor(x: np.ndarray, window: int = 5) -> np.ndarray:
    """
    SDTW shape descriptor (Zhao & Itti 2018):
    Each time point is represented by its local neighbourhood of size `window`,
    z-score normalised, forming a (T, window) array for multivariate DTW.

    Args:
        x:      1D time series of length T
        window: descriptor half-width (total window = window, centred on each point)

    Returns:
        desc: (T, window) array of local z-score descriptors
    """
    n = len(x)
    half = window // 2
    desc = np.empty((n, window))
    for t in range(n):
        start = max(0, t - half)
        end   = min(n, t + half + 1)
        seg   = x[start:end]
        pad_l = max(0, half - t)
        pad_r = max(0, (t + half + 1) - n)
        seg   = np.pad(seg, (pad_l, pad_r), mode="edge")[:window]
        mu, sd = seg.mean(), seg.std()
        desc[t] = (seg - mu) / (sd + 1e-10)
    return desc


# ─── 4. Per-pair lag functions ────────────────────────────────────────────────

def _lag_dtw(xi: np.ndarray, xj: np.ndarray) -> int:
    """Standard DTW lag estimate."""
    return _lag_from_path(_dtw_path_fast(xi, xj))


def _lag_ddtw(xi: np.ndarray, xj: np.ndarray) -> int:
    """Derivative DTW lag: apply derivative transform before DTW."""
    return _lag_from_path(_dtw_path_fast(
        _derivative_transform(xi), _derivative_transform(xj)
    ))


def _lag_sdtw(xi: np.ndarray, xj: np.ndarray, window: int = 5) -> int:
    """Shape DTW lag: use local shape descriptors for multivariate DTW."""
    return _lag_from_path(_dtw_path_nd_numpy(
        _shape_descriptor(xi, window), _shape_descriptor(xj, window)
    ))


def _lag_sddtw(xi: np.ndarray, xj: np.ndarray, window: int = 5) -> int:
    """Shape + Derivative DTW: apply derivative transform then shape descriptor."""
    dxi = _derivative_transform(xi)
    dxj = _derivative_transform(xj)
    return _lag_from_path(_dtw_path_nd_numpy(
        _shape_descriptor(dxi, window), _shape_descriptor(dxj, window)
    ))


def _lag_xcorr(xi: np.ndarray, xj: np.ndarray, max_lag: int = None) -> int:
    """
    Cross-correlation lag estimate via FFT (O(T log T), fastest method).
    Returns the lag that maximises the linear cross-correlation.
    Positive lag: xi leads xj.
    This is an approximation — accurate only for linear lead-lag relationships.
    """
    n = len(xi)
    if max_lag is None:
        max_lag = n // 4
    # Zero-mean
    xi_z = xi - xi.mean()
    xj_z = xj - xj.mean()
    # Normalise
    xi_n = xi_z / (np.std(xi_z) + 1e-10)
    xj_n = xj_z / (np.std(xj_z) + 1e-10)
    # FFT cross-correlation
    n_fft = 1 << int(np.ceil(np.log2(2 * n - 1)))
    Fxi = np.fft.fft(xi_n, n=n_fft)
    Fxj = np.fft.fft(xj_n, n=n_fft)
    xcorr = np.real(np.fft.ifft(Fxi * np.conj(Fxj)))
    # Map to lags
    lags = np.concatenate([np.arange(0, n), np.arange(-(n - 1), 0)])[:n_fft]
    # Restrict to max_lag
    mask = np.abs(lags) <= max_lag
    best_idx = np.argmax(xcorr * mask)
    return int(lags[best_idx])


# ─── 5. Full lead-lag matrix ──────────────────────────────────────────────────

_LAG_FUNCTIONS = {
    "dtw":   _lag_dtw,
    "ddtw":  _lag_ddtw,
    "xcorr": _lag_xcorr,
}


def lead_lag_matrix(
    returns_window: np.ndarray,
    method: str = "dtw",
    shape_window: int = 5,
) -> np.ndarray:
    """
    Compute (N, N) skew-symmetric lead-lag matrix V from a window of returns.

    V[i, j] > 0  →  asset i leads asset j by |V[i,j]| steps.
    V[j, i] = -V[i, j]  (skew-symmetric by construction).

    Args:
        returns_window: (T, N) log returns over lookback window
        method:         'levy' | 'dtw' | 'ddtw' | 'sdtw' | 'sddtw' | 'xcorr'
        shape_window:   descriptor window size for SDTW / SDDTW

    Returns:
        V: (N, N) skew-symmetric matrix
    """
    T, N = returns_window.shape
    V = np.zeros((N, N))

    if method == "levy":
        return levy_area_matrix(returns_window)

    # Z-score normalise each series within the window for DTW variants
    std = returns_window.std(axis=0, ddof=1)
    std = np.where(std < 1e-10, 1.0, std)
    normed = returns_window / std

    if method in ("sdtw", "sddtw"):
        lag_fn = lambda xi, xj: (
            _lag_sdtw(xi, xj, shape_window) if method == "sdtw"
            else _lag_sddtw(xi, xj, shape_window)
        )
    elif method in _LAG_FUNCTIONS:
        lag_fn = _LAG_FUNCTIONS[method]
    else:
        raise ValueError(f"Unknown lead-lag method: {method}. "
                         f"Choose from: levy, dtw, ddtw, sdtw, sddtw, xcorr")

    for i in range(N):
        for j in range(i + 1, N):
            lag = lag_fn(normed[:, i], normed[:, j])
            V[i, j] = lag      # i leads j if lag > 0
            V[j, i] = -lag

    return V


def compute_ensemble_lead_lag(
    returns: np.ndarray,
    t_idx: int,
    lookback_windows: list,
    method: str = "dtw",
    shape_window: int = 5,
) -> list:
    """
    Compute S lead-lag matrices, one per lookback window, ending at t_idx.

    Args:
        returns:          (T_full, N) full return history
        t_idx:            current time index (exclusive end)
        lookback_windows: list of δ values (trading days)
        method:           lead-lag detection method

    Returns:
        V_list: list of S (N, N) lead-lag matrices
    """
    V_list = []
    for delta in lookback_windows:
        start = max(0, t_idx - delta)
        window = returns[start:t_idx]
        if len(window) < 5:
            N = returns.shape[1]
            V_list.append(np.zeros((N, N)))
        else:
            V_list.append(lead_lag_matrix(window, method=method, shape_window=shape_window))
    return V_list
