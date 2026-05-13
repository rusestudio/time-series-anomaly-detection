"""
utils/preprocessing.py
Shared preprocessing logic — mirrors exactly what was done in HVAC_Preprocessing.ipynb
"""

import numpy as np
import pandas as pd


# ── Constants ─────────────────────────────────────────────────────────────────
FEATURES     = ['T_Supply', 'T_Return', 'T_Outdoor', 'Power', 'T_delta']
WINDOW_SIZE  = 96    # 96 × 15min = 24 hours
STEP_SIZE    = 4     # 1 hour stride

# IF feature names (29 statistical features per window)
IF_FEATURE_NAMES = []
for feat in FEATURES:
    IF_FEATURE_NAMES.extend([
        f'{feat}_mean', f'{feat}_std', f'{feat}_min',
        f'{feat}_max',  f'{feat}_range'
    ])
IF_FEATURE_NAMES.extend([
    'T_delta_p95', 'T_delta_p5', 'T_delta_std', 'supply_return_corr'
])


# ── Normalization ─────────────────────────────────────────────────────────────
def minmax_scale(df_in: pd.DataFrame,
                 p1_min: np.ndarray,
                 p1_max: np.ndarray) -> pd.DataFrame:
    """Apply min-max scaling using Period-1 params."""
    min_s = pd.Series(p1_min, index=FEATURES)
    max_s = pd.Series(p1_max, index=FEATURES)
    return (df_in - min_s) / (max_s - min_s + 1e-8)


# ── Sliding window ─────────────────────────────────────────────────────────────
def create_windows(df_scaled: pd.DataFrame,
                   window_size: int = WINDOW_SIZE,
                   step: int = STEP_SIZE):
    """Convert a scaled DataFrame into overlapping windows."""
    data = df_scaled.values
    X, timestamps = [], []
    for i in range(0, len(data) - window_size + 1, step):
        X.append(data[i:i + window_size])
        timestamps.append(df_scaled.index[i + window_size - 1])
    return np.array(X, dtype=np.float32), timestamps


# ── IF feature extraction ─────────────────────────────────────────────────────
def extract_if_features(X: np.ndarray) -> np.ndarray:
    """
    Convert windows (n, 96, 5) → statistical feature vectors (n, 29).
    Must match exactly what HVAC_IsolationForest.ipynb used.
    """
    features = []
    for i in range(len(X)):
        window = X[i]   # (96, 5)
        row = []
        for f in range(X.shape[2]):
            col = window[:, f]
            row.extend([col.mean(), col.std(), col.min(),
                        col.max(), col.max() - col.min()])
        t_supply = window[:, 0]
        t_return = window[:, 1]
        t_delta  = window[:, 4]
        row.extend([
            np.percentile(t_delta, 95),
            np.percentile(t_delta, 5),
            t_delta.std(),
            np.corrcoef(t_supply, t_return)[0, 1]
        ])
        features.append(row)
    return np.array(features, dtype=np.float32)


def fill_nan(F: np.ndarray, ref: np.ndarray = None) -> np.ndarray:
    """Fill NaN with column median."""
    F = F.copy()
    medians = np.nanmedian(ref if ref is not None else F, axis=0)
    for j in range(F.shape[1]):
        mask = np.isnan(F[:, j])
        if mask.any():
            F[mask, j] = medians[j]
    return F


# ── CSV ingestion ─────────────────────────────────────────────────────────────
def load_and_prepare_csv(uploaded_file,
                         p1_min: np.ndarray,
                         p1_max: np.ndarray):
    """
    Load an uploaded CSV, engineer T_delta, normalize, create windows.
    Returns (X_windows, timestamps, error_message).
    """
    try:
        df = pd.read_csv(uploaded_file)

        # Parse timestamp
        ts_col = next((c for c in df.columns
                       if 'time' in c.lower() or 'stamp' in c.lower()), None)
        if ts_col:
            df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
            df = df.set_index(ts_col).sort_index()
        else:
            df.index = pd.date_range('2021-01-01', periods=len(df),
                                     freq='15min', tz='UTC')

        # Engineer T_delta if not present
        if 'T_delta' not in df.columns:
            if 'T_Supply' in df.columns and 'T_Return' in df.columns:
                df['T_delta'] = df['T_Supply'] - df['T_Return']
            else:
                return None, None, "CSV must contain T_Supply and T_Return columns."

        # Check all features present
        missing = [f for f in FEATURES if f not in df.columns]
        if missing:
            return None, None, f"Missing columns: {missing}"

        df = df[FEATURES].dropna()
        if len(df) < WINDOW_SIZE:
            return None, None, (
                f"Need at least {WINDOW_SIZE} rows (24h at 15-min). "
                f"Got {len(df)}."
            )

        df_scaled = minmax_scale(df, p1_min, p1_max)
        X, ts = create_windows(df_scaled)
        return X, ts, None

    except Exception as e:
        return None, None, str(e)
