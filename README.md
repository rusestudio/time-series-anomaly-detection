# 🌡️ HVAC Anomaly Detection Dashboard

Streamlit web app for real-time HVAC anomaly detection using:
- **Isolation Forest** (Recall-optimised baseline)
- **LSTM Autoencoder** (F1-optimised deep learning model)

---

## Project Structure

```
hvac_app/
├── app.py                    ← Main Streamlit application
├── requirements.txt          ← Python dependencies
├── save_models.py            ← Helper: save models from notebooks
├── README.md
│
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py      ← Shared preprocessing (mirrors training notebooks)
│   ├── model_loader.py       ← Cached model loading
│   └── lstm_model.py         ← LSTM architecture (must match training)
│
├── models/                   ← Trained model files (you create these)
│   ├── isolation_forest.joblib
│   ├── if_scaler.joblib
│   ├── if_meta.joblib
│   ├── lstm_autoencoder.pt
│   ├── lstm_meta.joblib
│   ├── p1_min.npy
│   └── p1_max.npy
│
├── assets/                   ← EDA images for Tab 1
│   ├── 03_correlation.png
│   ├── 04_temporal_patterns.png
│   ├── if_02_anomaly_timeline.png
│   └── lstm_03_anomaly_timeline.png
│
└── .streamlit/
    └── config.toml           ← Dark theme config
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Save trained models from notebooks
Open `save_models.py` and copy each section into the corresponding notebook:
- **Section A** → `HVAC_IsolationForest.ipynb`
- **Section B** → `HVAC_LSTM_Autoencoder.ipynb`
- **Section C** → copies EDA images to `assets/`

### 3. Run the app locally
```bash
streamlit run app.py
```

---

## Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → select `app.py`
4. Add model files via GitHub LFS or use `st.secrets` for cloud storage paths

> **Note:** Model files (`.pt`, `.joblib`) are large. For deployment,
> consider storing them in Google Drive / S3 and loading via URL,
> or use Git LFS.

---

## Features

### Tab 1 — Overview & EDA
- Project description and dataset stats
- Feature table with importance reasoning
- EDA visualisations (correlation, temporal patterns, anomaly timelines)
- Model comparison table + interactive bar chart
- Known anomaly events summary

### Tab 2 — Isolation Forest
- 29-feature statistical input form
- Live prediction with anomaly score
- Signal importance bar chart
- Demo mode when model not loaded

### Tab 3 — LSTM Autoencoder
- CSV upload → batch prediction → downloadable results
- Manual slider input → synthetic window → actual vs reconstructed plot
- Paste raw 96-value sequences
- Per-feature reconstruction error chart
- Interactive Plotly timeline

---

## Input CSV Format (Tab 3)

```csv
Timestamp,T_Supply,T_Return,T_Outdoor,Power
2021-01-05 00:00:00+00:00,19.8,20.1,8.5,0.0
2021-01-05 00:15:00+00:00,19.9,20.2,8.4,0.0
...
```
Minimum 96 rows required (24 hours at 15-minute intervals).
`T_delta` is computed automatically.
