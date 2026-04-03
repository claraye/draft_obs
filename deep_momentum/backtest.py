"""
backtest.py — Walk-forward backtest engine for Deep Momentum replication
Paper: Han & Qin (2026), SSRN 4452964

Walk-forward schedule:
  - Require MIN_TRAIN_YEARS of data before first prediction
  - Retrain XGBoost ensemble every RETRAIN_FREQ months (annually)
  - Predict monthly, construct long-short portfolio
  - Compare: MOM, XGB (naive), RET (Deep Momentum), DPR, SRP
"""

import os
import logging
import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    dates:       list
    returns:     dict   # strategy_name → np.ndarray of monthly returns
    bimodality:  dict   # strategy_name → np.ndarray of monthly BM values
    predictions: dict   # date → {"proba": array, "scores": dict, "long": list, "short": list}


# ─── Bimodality measure ───────────────────────────────────────────────────────

def compute_bimodality(y_true: np.ndarray, y_pred_classes: np.ndarray, n_classes: int = 10) -> float:
    """
    Compute bimodality measure BM (Eq. 1–5) for a given month.

    'Positive' = top decile (class n_classes-1), 'Negative' = bottom decile (class 0).

    BM = -[(HH - HL) + (LL - LH)] / 2
    """
    H_true = (y_true == n_classes - 1)
    L_true = (y_true == 0)
    H_pred = (y_pred_classes == n_classes - 1)
    L_pred = (y_pred_classes == 0)

    TP = np.sum( H_pred &  H_true)
    FP = np.sum( H_pred & ~H_true & L_true)
    TN = np.sum(~H_pred &  L_true)
    FN = np.sum(~H_pred & ~L_true & H_true)

    denom_pos = TP + FP
    denom_neg = TN + FN

    HH = (TP / denom_pos - 0.1) if denom_pos > 0 else 0.0
    HL = (FP / denom_pos - 0.1) if denom_pos > 0 else 0.0
    LL = (TN / denom_neg - 0.1) if denom_neg > 0 else 0.0
    LH = (FN / denom_neg - 0.1) if denom_neg > 0 else 0.0

    return -((HH - HL) + (LL - LH)) / 2.0


# ─── Portfolio builder ────────────────────────────────────────────────────────

def build_portfolio(scores: pd.Series, long_frac: float, short_frac: float) -> tuple[list, list]:
    """
    Select long (top quantile) and short (bottom quantile) stocks by score.
    """
    valid = scores.dropna().sort_values(ascending=False)
    n = len(valid)
    n_long  = max(1, int(np.ceil(n * long_frac)))
    n_short = max(1, int(np.ceil(n * short_frac)))
    long_tickers  = valid.iloc[:n_long].index.tolist()
    short_tickers = valid.iloc[-n_short:].index.tolist()
    return long_tickers, short_tickers


def compute_portfolio_return(
    returns_t: pd.Series,
    long_tickers: list,
    short_tickers: list,
) -> float:
    """
    Equal-weighted long-short portfolio return for a single month.
    """
    long_ret  = returns_t.reindex(long_tickers).mean()
    short_ret = returns_t.reindex(short_tickers).mean()
    if np.isnan(long_ret) or np.isnan(short_ret):
        return np.nan
    return long_ret - short_ret


# ─── Main backtest ────────────────────────────────────────────────────────────

def run_backtest(
    feature_df:        pd.DataFrame,
    returns:           pd.DataFrame,
    backtest_start:    str,
    backtest_end:      str,
    min_train_years:   int,
    retrain_freq:      int,
    n_ensemble_runs:   int,
    xgb_params:        dict,
    early_stopping:    int,
    long_frac:         float,
    short_frac:        float,
    class_mean_lb_yrs: int,
    n_classes:         int,
    cache_dir:         str,
) -> BacktestResult:
    """
    Walk-forward backtest.

    Parameters
    ----------
    feature_df      : output of features.build_features(), with columns
                      ['date', 'ticker', 'zMOM1m', ..., 'SIZE', 'ret_fwd']
    returns         : (T, N) monthly returns DataFrame
    backtest_start  : first OOS month (str)
    backtest_end    : last OOS month (str)
    """
    from features import FEATURE_COLS
    from classifier import (
        assign_return_classes, train_xgb_ensemble, predict_ensemble,
        reclassify_DPR, reclassify_RET, reclassify_SRP, estimate_class_statistics
    )

    os.makedirs(cache_dir, exist_ok=True)

    all_dates = sorted(feature_df["date"].unique())
    oos_dates = [d for d in all_dates
                 if pd.Timestamp(backtest_start) <= d <= pd.Timestamp(backtest_end)]

    result = BacktestResult(
        dates=[],
        returns={"MOM": [], "XGB": [], "DPR": [], "RET": [], "SRP": []},
        bimodality={"MOM": [], "XGB": [], "RET": []},
        predictions={},
    )

    models          = None   # list of XGBClassifier
    last_retrain_dt = None

    for t_idx, pred_date in enumerate(oos_dates):
        logger.info(f"  {pred_date.strftime('%Y-%m')}  ({t_idx+1}/{len(oos_dates)})")

        # ── Training data: all months before pred_date ────────────────────
        train_df = feature_df[feature_df["date"] < pred_date].copy()
        if train_df["date"].nunique() < min_train_years * 12:
            logger.debug(f"    Insufficient training data, skipping.")
            continue

        # Assign cross-sectional return class labels
        train_df["class_label"] = (
            train_df
            .groupby("date")["ret_fwd"]
            .transform(lambda s: assign_return_classes(s, n_classes))
        )
        train_df = train_df.dropna(subset=["class_label", "ret_fwd"])
        train_df["class_label"] = train_df["class_label"].astype(int)

        # ── Retrain if needed ─────────────────────────────────────────────
        should_retrain = (
            models is None
            or last_retrain_dt is None
            or (pred_date.year - last_retrain_dt.year) * 12
               + (pred_date.month - last_retrain_dt.month) >= retrain_freq
        )

        if should_retrain:
            cache_key = pred_date.strftime("%Y%m")
            cache_path = os.path.join(cache_dir, f"models_{cache_key}.pkl")

            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    models = pickle.load(f)
                logger.info(f"    Loaded cached models from {cache_path}")
            else:
                logger.info(f"    Retraining ensemble ({n_ensemble_runs} runs) ...")

                X_all = train_df[FEATURE_COLS].values.astype(float)
                y_all = train_df["class_label"].values

                # Random 80/20 train/val split (not chronological — paper Section 3.3.3)
                rng = np.random.default_rng(seed=42)
                idx = rng.permutation(len(X_all))
                n_val = max(1, int(0.2 * len(X_all)))
                val_idx   = idx[:n_val]
                train_idx = idx[n_val:]

                X_tr, y_tr = X_all[train_idx], y_all[train_idx]
                X_va, y_va = X_all[val_idx],   y_all[val_idx]

                models = train_xgb_ensemble(
                    X_tr, y_tr, X_va, y_va,
                    n_runs=n_ensemble_runs,
                    xgb_params=xgb_params,
                    early_stopping_rounds=early_stopping,
                )

                with open(cache_path, "wb") as f:
                    pickle.dump(models, f)
                logger.info(f"    Models cached to {cache_path}")

            last_retrain_dt = pred_date

        # ── Class statistics for RET and SRP ─────────────────────────────
        # Use rolling lookback_years of training data
        lookback_start = pred_date - pd.DateOffset(years=class_mean_lb_yrs)
        stats_df = train_df[train_df["date"] >= lookback_start]
        from classifier import estimate_class_statistics
        class_means, class_stds = estimate_class_statistics(stats_df, class_mean_lb_yrs, n_classes)

        # ── Inference on pred_date ────────────────────────────────────────
        pred_df = feature_df[feature_df["date"] == pred_date].copy()
        if len(pred_df) < 10:
            logger.debug(f"    Too few stocks at {pred_date}, skipping.")
            continue

        X_pred = pred_df[FEATURE_COLS].values.astype(float)
        tickers = pred_df["ticker"].values

        proba = predict_ensemble(models, X_pred)   # (N, K)

        # XGB (naive): argmax class
        xgb_class = np.argmax(proba, axis=1)

        # Scores for each reclassification method
        dpr_scores = reclassify_DPR(proba)
        ret_scores = reclassify_RET(proba, class_means)
        srp_scores = reclassify_SRP(proba, class_means, class_stds)

        # MOM score: 12-month momentum (zMOM12m)
        mom_scores = pred_df["zMOM12m"].values

        # ── Build portfolios ──────────────────────────────────────────────
        def scores_series(arr):
            return pd.Series(arr, index=tickers)

        # Forward returns for this month
        if pred_date in returns.index:
            ret_next_idx = returns.index.get_loc(pred_date)
            if ret_next_idx + 1 < len(returns):
                ret_t = returns.iloc[ret_next_idx + 1]
            else:
                continue
        else:
            continue

        portfolios = {
            "MOM": scores_series(mom_scores),
            "XGB": scores_series(xgb_class.astype(float)),
            "DPR": scores_series(dpr_scores),
            "RET": scores_series(ret_scores),
            "SRP": scores_series(srp_scores),
        }

        for strat, scores_s in portfolios.items():
            long_t, short_t = build_portfolio(scores_s, long_frac, short_frac)
            pret = compute_portfolio_return(ret_t, long_t, short_t)
            result.returns[strat].append(pret)

            # Bimodality (for strategies with class predictions)
            if strat in ("MOM", "XGB", "RET") and pred_date in feature_df["date"].values:
                true_classes = (
                    feature_df[feature_df["date"] == pred_date]
                    .set_index("ticker")["class_label"]
                    if "class_label" in feature_df.columns else None
                )
                # Re-assign classes on pred month features
                true_cls = assign_return_classes(
                    pred_df.set_index("ticker")["ret_fwd"], n_classes
                ).dropna()
                if len(true_cls) > 2 * n_classes:
                    pred_cls = scores_series(xgb_class.astype(float)).reindex(true_cls.index)
                    bm = compute_bimodality(
                        true_cls.values.astype(int),
                        pred_cls.fillna(0).values.astype(int),
                        n_classes,
                    )
                    result.bimodality[strat if strat != "XGB" else "XGB"].append(bm)

        result.dates.append(pred_date)

    # Convert lists to arrays
    for strat in result.returns:
        result.returns[strat] = np.array(result.returns[strat], dtype=float)

    return result
