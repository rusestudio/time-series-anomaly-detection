"""
utils/model_loader.py
Cached model and scaler loading — runs once per session.
"""

import os
import numpy as np
import joblib
import streamlit as st


# ── Isolation Forest ──────────────────────────────────────────────────────────
@st.cache_resource
def load_isolation_forest():
    """Load trained IF model + StandardScaler + threshold."""
    model_path  = "models/isolation_forest.joblib"
    scaler_path = "models/if_scaler.joblib"
    meta_path   = "models/if_meta.joblib"

    if not os.path.exists(model_path):
        return None, None, None, (
            "isolation_forest.joblib not found in models/. "
            "Run HVAC_IsolationForest.ipynb and save the model first."
        )

    model     = joblib.load(model_path)
    scaler    = joblib.load(scaler_path)   if os.path.exists(scaler_path) else None
    meta      = joblib.load(meta_path)     if os.path.exists(meta_path)   else {}
    threshold = meta.get("threshold", None)
    return model, scaler, threshold, None


# ── LSTM Autoencoder (PyTorch) ────────────────────────────────────────────────
@st.cache_resource
def load_lstm():
    """Load trained LSTM model + scaling params + threshold."""
    checkpoint_path = "models/lstm_autoencoder.pt"
    meta_path       = "models/lstm_meta.joblib"

    if not os.path.exists(checkpoint_path):
        return None, None, None, None, (
            "lstm_autoencoder.pt not found in models/. "
            "Run HVAC_LSTM_Autoencoder.ipynb and save the model first."
        )

    try:
        import torch
        from utils.lstm_model import LSTMAutoencoder

        checkpoint = torch.load(checkpoint_path, map_location="cpu",
                                weights_only=False)
        hp = checkpoint.get("hyperparams", {})

        model = LSTMAutoencoder(
            n_features     = 5,
            hidden_dim     = hp.get("hidden_dim",     64),
            bottleneck_dim = hp.get("bottleneck_dim", 32),
            n_layers       = hp.get("n_layers",       2),
            dropout        = hp.get("dropout",        0.2),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        threshold = checkpoint.get("threshold", None)

        meta   = joblib.load(meta_path) if os.path.exists(meta_path) else {}
        p1_min = meta.get("p1_min", None)
        p1_max = meta.get("p1_max", None)

        return model, threshold, p1_min, p1_max, None

    except Exception as e:
        return None, None, None, None, str(e)


# ── Scaling params (shared) ───────────────────────────────────────────────────
@st.cache_resource
def load_scaling_params():
    """Load Period-1 min/max arrays saved during preprocessing."""
    min_path = "models/p1_min.npy"
    max_path = "models/p1_max.npy"
    if os.path.exists(min_path) and os.path.exists(max_path):
        return np.load(min_path), np.load(max_path)
    # Fallback: values computed during preprocessing notebook
    p1_min = np.array([12.26, 12.34,  2.30, 0.0, -14.0])
    p1_max = np.array([30.30, 25.99, 32.80, 5.32,  14.0])
    return p1_min, p1_max
