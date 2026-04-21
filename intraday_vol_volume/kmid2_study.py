"""
kmid2_study.py — KMID2 turnover-reduction study on ES1 5m bars.

Compares all five proposal families:
  1. Smoothers         — MA-n and S-n (sum-ratio)
  2. Hysteresis bands  — position stays until z-score exits band
  3. Minimum hold      — no re-evaluation for H bars
  4. Multi-scale       — equal-weight 5m + 15m + 30m KMID2
  5. K-Bar consensus   — require N-of-4 signed K-bar factors to agree

Metrics per variant:
  IC           — full-sample Spearman vs 1-bar forward return
  Sign SR      — gross annualised Sharpe (daily-aggregated P&L)
  Net SR       — after 0.5-tick round-trip cost per flip
  Avg Hold     — bars  (formula: 2·Σ|pos| / Σ|Δpos|)
  Avg Hold min — above × 5 (minutes)
  TO/day       — mean |Δpos| per trading day (turnover intensity)

Run:
  cd C:/Users/qingy/.claude/skills/factor-screen/scripts
  py312 kmid2_study.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS))

from blp_loader import load_blp_intraday
from compute_alpha158 import compute_alpha

# ── Load & prepare ─────────────────────────────────────────────────────────────
print("[load] ES1 Index 5m ...")
df = load_blp_intraday("ES1 Index", "2025-10-06", "2026-04-17",
                       bar_size_min=5, blp_root=None)
df = df.dropna(subset=["close"])
df = df[df["volume"].fillna(0) > 0]

o     = df["open"].astype(float)
h     = df["high"].astype(float)
l     = df["low"].astype(float)
close = df["close"].astype(float)
vol   = df["volume"].astype(float)

bar_ret  = close.pct_change()
bar_dates = bar_ret.index.normalize()
date_ser  = pd.Series(bar_dates, index=bar_ret.index)
session_open = (date_ser != date_ser.shift(1))
bar_ret = bar_ret.where(~session_open, 0.0)

n_days       = max(bar_dates.nunique(), 1)
bars_per_day = len(bar_dates) / n_days
ZS_W         = max(int(20 * bars_per_day), 60)
ZS_MP        = max(60, ZS_W // 4)
EXEC_LAG     = 2
TICK_PTS     = 0.25                              # one ES tick in index points
COST_PER_DIFF = TICK_PTS / close.mean() / 2     # cost per |dpos|=1 (half-tick per side)

print(f"  {len(df)} bars, {n_days} days, {bars_per_day:.1f} bars/day")
print(f"  ZS_WINDOW={ZS_W} bars (~20 days), COST_PER_DIFF={COST_PER_DIFF*1e4:.3f} bp\n")

# ── Helpers ────────────────────────────────────────────────────────────────────
rng  = (h - l).clip(lower=1e-8)
body = close - o

fwd = close.pct_change(-EXEC_LAG)   # 1-bar forward return (horizon=1, shifted by exec_lag)


def zscore(s: pd.Series) -> pd.Series:
    mu = s.rolling(ZS_W, min_periods=ZS_MP).mean()
    sd = s.rolling(ZS_W, min_periods=ZS_MP).std().replace(0, np.nan)
    return (s - mu) / sd


def spearman_ic(factor: pd.Series) -> float:
    both = pd.concat([factor, fwd], axis=1).dropna()
    if len(both) < 30:
        return np.nan
    return float(both.iloc[:, 0].rank().corr(both.iloc[:, 1].rank()))


def daily_sr(ret: pd.Series) -> float:
    daily = ret.groupby(ret.index.normalize()).sum()
    if daily.std() == 0 or len(daily) < 10:
        return np.nan
    return float(daily.mean() / daily.std() * np.sqrt(252))


def avg_hold_bars(pos: pd.Series) -> float:
    denom = pos.diff().abs().sum()
    return float(2 * pos.abs().sum() / denom) if denom > 0 else np.inf


def turnover_per_day(pos: pd.Series) -> float:
    return float(pos.diff().abs().groupby(pos.index.normalize()).sum().mean())


def net_sr(pos: pd.Series) -> float:
    strat = pos.mul(bar_ret)
    tc    = pos.diff().abs() * COST_PER_DIFF
    return daily_sr((strat - tc).dropna())


# ── Direction: KMID2 is a reversal factor (gross SR < 0 for raw signal).
# Determine direction from the raw sign strategy gross SR; flip all variants.
def _raw_sign_sr(factor: pd.Series) -> float:
    p = np.sign(factor).shift(EXEC_LAG)
    return daily_sr(p.mul(bar_ret).dropna())


def sign_pos(factor: pd.Series, direction: int = 1) -> pd.Series:
    return direction * np.sign(factor)


def stats(label: str, factor: pd.Series | None, pos: pd.Series) -> dict:
    """Compute all metrics for a position series."""
    p_exec = pos.shift(EXEC_LAG)
    ret    = p_exec.mul(bar_ret).dropna()
    tc     = pos.shift(EXEC_LAG).diff().abs() * COST_PER_DIFF
    net    = (p_exec.mul(bar_ret) - tc).dropna()
    ah     = avg_hold_bars(pos.dropna())
    return {
        "Variant":        label,
        "IC":             round(spearman_ic(factor), 4) if factor is not None else "—",
        "Sign SR":        round(daily_sr(ret), 3),
        "Net SR":         round(daily_sr(net), 3),
        "Avg Hold (bar)": round(ah, 1),
        "Avg Hold (min)": round(ah * 5, 0),
        "TO/day":         round(turnover_per_day(pos), 2),
    }


# ── 0. Baseline ────────────────────────────────────────────────────────────────
print("[compute] raw KMID2 ...")
KMID2 = body / rng
DIR   = -1 if _raw_sign_sr(KMID2) < 0 else 1
print(f"  direction={DIR:+d} (gross SR of raw signal: {_raw_sign_sr(KMID2):.3f})")
rows = [stats("KMID2_raw", KMID2, sign_pos(KMID2, DIR))]

# ── 1. Smoothers ───────────────────────────────────────────────────────────────
print("[compute] smoothers ...")
for n in [3, 5, 10, 20]:
    ma = KMID2.rolling(n, min_periods=1).mean()
    rows.append(stats(f"KMID2_MA{n}", ma, sign_pos(ma, DIR)))

for n in [3, 5, 10, 20]:
    sr = body.rolling(n, min_periods=1).sum() / (rng.rolling(n, min_periods=1).sum() + 1e-12)
    rows.append(stats(f"KMID2_S{n}", sr, sign_pos(sr, DIR)))

# ── 2. Hysteresis bands (on z-score of raw KMID2) ────────────────────────────
print("[compute] hysteresis ...")
z_raw = zscore(KMID2)

def hysteresis(z: pd.Series, enter: float, exit_: float) -> pd.Series:
    pos = np.zeros(len(z))
    p   = 0.0
    for i, zi in enumerate(z.to_numpy()):
        if np.isnan(zi):
            pos[i] = 0.0
            continue
        if zi > enter:
            p = 1.0
        elif zi < -enter:
            p = -1.0
        elif abs(zi) < exit_:
            p = 0.0
        pos[i] = p
    return pd.Series(pos, index=z.index)

for enter, exit_, lbl in [(0.4, 0.1, "hyst0.4"), (0.6, 0.2, "hyst0.6"), (1.0, 0.3, "hyst1.0")]:
    p = DIR * hysteresis(z_raw, enter, exit_)
    rows.append(stats(f"KMID2_{lbl}", None, p))

# ── 3. Minimum holding period ─────────────────────────────────────────────────
print("[compute] min-hold ...")

def min_hold(factor: pd.Series, H: int) -> pd.Series:
    sig = np.sign(factor.to_numpy())
    pos = np.zeros(len(sig))
    p   = 0.0
    cnt = H  # start ready to act
    for i, s in enumerate(sig):
        if np.isnan(s):
            pos[i] = 0.0
            continue
        if cnt >= H and s != p:
            p   = s
            cnt = 0
        pos[i] = p
        cnt += 1
    return pd.Series(pos, index=factor.index)

for H in [5, 10, 20]:
    p = DIR * min_hold(KMID2, H)
    rows.append(stats(f"KMID2_minhold{H}", None, p))

# ── 4. Multi-scale (5m + 15m + 30m resampled) ────────────────────────────────
print("[compute] multi-scale ...")

def resample_kmid2(df_in: pd.DataFrame, freq: str) -> pd.Series:
    r = df_in.resample(freq, closed="right", label="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna(subset=["close"])
    b = r["close"] - r["open"]
    rn = (r["high"] - r["low"]).clip(lower=1e-8)
    return (b / rn).reindex(close.index).ffill()

k15 = resample_kmid2(df, "15min")
k30 = resample_kmid2(df, "30min")

for label, fac in [
    ("KMID2_15m",    k15),
    ("KMID2_30m",    k30),
    ("KMID2_multi3", (KMID2 + k15 + k30) / 3),
]:
    rows.append(stats(label, fac, sign_pos(fac, DIR)))

# ── 5. K-Bar signed consensus (KMID + KMID2 + KSFT + KSFT2) ─────────────────
print("[compute] K-bar consensus ...")
KMID  = compute_alpha(df, "KMID")
KSFT  = compute_alpha(df, "KSFT")
KSFT2 = compute_alpha(df, "KSFT2")

vote = (np.sign(KMID) + np.sign(KMID2) + np.sign(KSFT) + np.sign(KSFT2))

for thresh, lbl in [(2, "cons2of4"), (3, "cons3of4"), (4, "cons4of4")]:
    p = DIR * pd.Series(
        np.where(vote >= thresh, 1.0, np.where(vote <= -thresh, -1.0, 0.0)),
        index=KMID2.index,
    )
    rows.append(stats(f"KMID2_{lbl}", None, p))

# ── Print results ──────────────────────────────────────────────────────────────
tbl = pd.DataFrame(rows).set_index("Variant")

# Section order
order = (
    ["KMID2_raw"] +
    [f"KMID2_MA{n}" for n in [3,5,10,20]] +
    [f"KMID2_S{n}"  for n in [3,5,10,20]] +
    ["KMID2_hyst0.4","KMID2_hyst0.6","KMID2_hyst1.0"] +
    [f"KMID2_minhold{H}" for H in [5,10,20]] +
    ["KMID2_15m","KMID2_30m","KMID2_multi3"] +
    ["KMID2_cons2of4","KMID2_cons3of4","KMID2_cons4of4"]
)
tbl = tbl.loc[[r for r in order if r in tbl.index]]

print("\n" + "=" * 75)
print("  KMID2 Turnover-Reduction Study -- ES1 5m (Oct-2025 to Apr-2026)")
print(f"  Cost model: {COST_PER_DIFF*1e4:.3f} bp per |dpos|=1  (0.5 tick round-trip)")
print("=" * 75)
print(tbl.to_string())
print()
