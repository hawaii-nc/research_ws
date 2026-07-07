"""
probe_utils.py — Shared utilities for the phi probe-evaluation pipeline.

Dependency-light: numpy only (torch/matplotlib used in the step scripts).

Contains:
  - PARAM_KEYS: canonical 7-D physics parameter ordering (matches phase2_adaptation.py)
  - ridge_fit / ridge_predict: closed-form ridge regression with standardization
  - r2_score: coefficient of determination
  - split_by_episode: train/test split that NEVER leaks timesteps across episodes
  - build_histories: sliding-window (history -> target) extraction, episode-boundary safe
  - ema_per_episode: exponential moving average, reset at each episode boundary
  - excitation: lateral-demand proxy |v * yaw_rate|
  - quantile_bins / stratified_r2: excitation-stratified evaluation
"""

import numpy as np

# Canonical ordering of the 7 physics parameters (et).
# MUST match the order used in phase2_adaptation.py's env_params_list.
PARAM_KEYS = [
    'grip_factor',
    'mass_scale',
    'inertia_scale',
    'motor_steering_scale',
    'motor_drive_scale',
    'delay_steering',
    'delay_drive',
]


# ----------------------------------------------------------------------------
# Ridge regression (closed form, standardized inputs, intercept via centering)
# ----------------------------------------------------------------------------

def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = 1.0) -> dict:
    """
    Fit ridge regression y ~ X with L2 penalty lam.

    Inputs are standardized (zero mean, unit std) internally so lam is
    scale-invariant; the intercept is handled by centering y.

    Returns a dict of fitted parameters for ridge_predict().
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()

    mu_x = X.mean(axis=0)
    sd_x = X.std(axis=0) + 1e-8
    Xs = (X - mu_x) / sd_x

    mu_y = y.mean()
    yc = y - mu_y

    d = Xs.shape[1]
    A = Xs.T @ Xs + lam * np.eye(d)
    b = Xs.T @ yc
    w = np.linalg.solve(A, b)

    return {'w': w, 'mu_x': mu_x, 'sd_x': sd_x, 'mu_y': mu_y, 'lam': lam}


def ridge_predict(model: dict, X: np.ndarray) -> np.ndarray:
    """Predict with a model returned by ridge_fit()."""
    Xs = (np.asarray(X, dtype=np.float64) - model['mu_x']) / model['sd_x']
    return Xs @ model['w'] + model['mu_y']


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coefficient of determination. R^2 = 1 - SS_res / SS_tot.
    Returns np.nan if the target has (near-)zero variance — an R^2 against a
    constant target is meaningless and we refuse to report it.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot < 1e-12:
        return float('nan')
    ss_res = np.sum((y_true - y_pred) ** 2)
    return float(1.0 - ss_res / ss_tot)


# ----------------------------------------------------------------------------
# Episode-aware splitting (prevents temporal leakage between train and test)
# ----------------------------------------------------------------------------

def split_by_episode(episode_ids: np.ndarray, test_frac: float = 0.2,
                     seed: int = 0):
    """
    Split timestep indices into train/test by WHOLE EPISODES.

    Splitting by raw timestep would leak: adjacent timesteps within an episode
    are nearly identical, so a random timestep split massively inflates test
    R^2. Episode-level splitting is the honest protocol.

    Returns (train_mask, test_mask) boolean arrays over timesteps.
    """
    episode_ids = np.asarray(episode_ids)
    uniq = np.unique(episode_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_test = max(1, int(round(test_frac * len(uniq))))
    test_eps = set(uniq[:n_test].tolist())
    test_mask = np.isin(episode_ids, list(test_eps))
    return ~test_mask, test_mask


# ----------------------------------------------------------------------------
# History building (for phi-style estimators and the observability regressor)
# ----------------------------------------------------------------------------

def build_histories(features: np.ndarray, episode_ids: np.ndarray,
                    window: int):
    """
    Build sliding-window histories that never cross an episode boundary.

    Args:
        features:    (N, F) per-timestep feature matrix (e.g. concat(obs, act))
        episode_ids: (N,)   episode id per timestep
        window:      history length k

    Returns:
        histories:   (M, window, F) — history ending at (and including) step t
        target_idx:  (M,) index into the original N-length arrays for the step
                     each history ends at (use to fetch targets / excitation)
    """
    features = np.asarray(features)
    episode_ids = np.asarray(episode_ids)
    hists, tidx = [], []

    for ep in np.unique(episode_ids):
        idx = np.where(episode_ids == ep)[0]
        f = features[idx]
        if len(f) < window:
            continue
        # sliding_window_view over the time axis -> (L-w+1, w, F)
        sw = np.lib.stride_tricks.sliding_window_view(f, (window, f.shape[1]))
        sw = sw[:, 0, :, :]  # squeeze the feature-window axis
        hists.append(sw)
        tidx.append(idx[window - 1:])  # history [t-w+1 .. t] targets step t

    if not hists:
        raise ValueError(f"No episode has >= {window} steps; cannot build histories.")
    return np.concatenate(hists, axis=0), np.concatenate(tidx, axis=0)


# ----------------------------------------------------------------------------
# EMA smoothing (reset at episode boundaries)
# ----------------------------------------------------------------------------

def ema_per_episode(x: np.ndarray, episode_ids: np.ndarray,
                    beta: float = 0.9) -> np.ndarray:
    """
    EMA smooth a (N, D) signal, restarting at every episode boundary.
    s[t] = beta * s[t-1] + (1-beta) * x[t];  s[0] = x[0] per episode.
    beta=0.9 at 50 Hz ~ 0.2 s time constant; beta=0.98 ~ 1 s.
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    episode_ids = np.asarray(episode_ids)
    for ep in np.unique(episode_ids):
        idx = np.where(episode_ids == ep)[0]
        s = x[idx[0]].copy()
        out[idx[0]] = s
        for j in idx[1:]:
            s = beta * s + (1.0 - beta) * x[j]
            out[j] = s
    return out


# ----------------------------------------------------------------------------
# Excitation (lateral-demand proxy) and stratified evaluation
# ----------------------------------------------------------------------------

def excitation(v: np.ndarray, yaw_rate: np.ndarray) -> np.ndarray:
    """
    Lateral-demand proxy: |a_lat| ~= |v * yaw_rate|  (m/s^2).
    Grip is only observable from behavior when lateral demand approaches the
    friction limit — this is the stratification variable for Steps 2 and 3.
    """
    return np.abs(np.asarray(v) * np.asarray(yaw_rate))


def quantile_bins(x: np.ndarray, n_bins: int = 5):
    """
    Quantile bin edges for x. Returns (edges, bin_index_per_sample).
    Duplicate edges (heavily zero-inflated excitation) are collapsed.
    """
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(x, qs))
    bin_idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    return edges, bin_idx


def stratified_r2(y_true: np.ndarray, y_pred: np.ndarray,
                  strat: np.ndarray, n_bins: int = 5):
    """
    R^2 per quantile bin of the stratification variable.

    Returns list of dicts: {bin, lo, hi, n, r2}.
    NOTE: within-bin R^2 uses the within-bin target mean as baseline, which is
    the strict version — it asks "does the estimate explain variance *within*
    this excitation regime", exactly the observability question.
    """
    edges, bin_idx = quantile_bins(strat, n_bins)
    rows = []
    for b in range(len(edges) - 1):
        m = bin_idx == b
        rows.append({
            'bin': b,
            'lo': float(edges[b]),
            'hi': float(edges[b + 1]),
            'n': int(m.sum()),
            'r2': r2_score(y_true[m], y_pred[m]) if m.sum() > 10 else float('nan'),
        })
    return rows
