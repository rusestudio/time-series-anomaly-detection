# HVAC Time-Series Anomaly Detection

This repository contains an end-to-end pipeline to detect anomalies in HVAC telemetry using **unsupervised** methods and a **temporal** model.

The project is organized as 4 notebooks:
1. **EDA**: inspect signals, find an anomaly proxy, and choose features
2. **Preprocessing**: handle a large dataset gap, normalize, and build sliding windows
3. **Isolation Forest**: fast baseline anomaly detector on window summary statistics
4. **LSTM Autoencoder**: sequence-aware anomaly detector on full 24h windows
5. **Comparison**: compare both detectors and combine their outputs into high-confidence episodes

---

## Data

Expected raw input file (used by the notebooks):

- `HVAC_NE_EC_19-21.csv`

This is an HVAC time-series dataset spanning **Oct 2019 – Apr 2021**.

---

## Key idea

Because true expert anomaly labels are not provided, the notebooks use **EDA-identified anomaly events as a pseudo ground truth**:
- **Jan 2021**: large **ΔT spike** (`T_Supply - T_Return`)
- **Nov 2020**: elevated **restart volatility** and control instability

---

## Repository structure

- `HVAC_EDA.ipynb` — exploratory analysis + feature selection
- `HVAC_Preprocessing.ipynb` — scaling, gap handling, sliding windows, train/val/test split
- `HVAC_IsolationForest.ipynb` — Isolation Forest baseline
- `HVAC_LSTM_Autoencoder.ipynb` — LSTM autoencoder detector
- `HVAC_Comparison.ipynb` — compare and combine both models

Artifacts produced by notebooks:

- `preprocessed/`
  - `X_train.npy`, `X_val.npy`, `X_test.npy`
  - `p1_min.npy`, `p1_max.npy`
  - `test_timestamps.csv`
- `result/` (final CSV reports)
  - `if_results.csv`
  - `lstm_results.csv`
  - `combined_results.csv`
  - `final_anomaly_report.csv`

Figures are saved into `img/`.

---

## Modeling details

### Windowing

- **Window size**: 96 steps
- **Sampling**: 15-minute frequency
- => each window represents **24 hours** of 5 features

The sliding window stride is **4 steps** (1 hour).

### Features

From EDA, the final set of 5 features is:

- `T_Supply`
- `T_Return`
- `T_Outdoor`
- `Power`
- `T_delta = T_Supply - T_Return`

### Preprocessing (important)

The dataset contains a **large 183-day gap** (~May 2020 to ~Oct 2020). To prevent the model from learning this discontinuity as an anomaly:

- Split the timeline into:
  - **Period 1** (pre-shutdown): used for **train + validation**
  - **Period 2** (post-restart): used for **test**
- Reindex to a strict 15-min grid
- Interpolate smaller gaps (≤ 2 hours)
- **Min-Max normalization is fit only on Period 1** and applied to Period 2 (no data leakage)

---

## Pipeline (notebook sequence)

### 1) HVAC_EDA (feature selection)

- Visualize the dataset and operating modes
- Analyze distributions, correlations, and temporal patterns
- Select features and save the filtered dataset:
  - `hvac_eda_selected.csv`

### 2) HVAC_Preprocessing

Produces:
- `preprocessed/X_train.npy`
- `preprocessed/X_val.npy`
- `preprocessed/X_test.npy`
- `preprocessed/p1_min.npy`, `preprocessed/p1_max.npy`
- `preprocessed/test_timestamps.csv`

### 3) HVAC_IsolationForest (baseline)

Isolation Forest works on **flat feature vectors**, so each 24h window is summarized into statistical descriptors (mean/std/min/max/range + additional ΔT features).

Key outputs:
- `if_results.csv` (per-window anomaly score + flag)

### 4) HVAC_LSTM_Autoencoder (temporal model)

LSTM autoencoder learns to reconstruct normal windows.

- Input: full sequence `(96, 5)`
- Loss: **MSE reconstruction error**
- Threshold is selected using validation errors:
  - `THRESHOLD = val_mean + k * val_std` (chosen `k=2.5` in the notebook)

Key outputs:
- `lstm_results.csv` (per-window MSE + anomaly flag)
- `lstm_autoencoder.pt` (trained model + threshold)

### 5) HVAC_Comparison (combine outputs)

Combines both detectors into a single confidence label:
- **0 = Normal**
- **1 = Low confidence anomaly** (either model flags)
- **2 = High confidence anomaly** (both flag)

Saves:
- `combined_results.csv`
- `final_anomaly_report.csv` (grouped high-confidence anomaly episodes)

---

## Outputs

The comparison notebook saves multiple evaluation plots, including:
- `img/comparison_01_timeline.png`
- `img/comparison_02_agreement.png`
- `img/comparison_03_performance.png`

Final CSV reports:
- `result/combined_results.csv`
- `result/final_anomaly_report.csv`

---

## Running the notebooks

This project is notebook-first (Jupyter).

Typical workflow:
1. Run `HVAC_EDA.ipynb`
2. Run `HVAC_Preprocessing.ipynb`
3. Run `HVAC_IsolationForest.ipynb`
4. Run `HVAC_LSTM_Autoencoder.ipynb`
5. Run `HVAC_Comparison.ipynb`

Notes:
- Isolation Forest uses `scikit-learn`.
- LSTM autoencoder uses **PyTorch** (CPU is supported).

---

## Limitations

- Ground truth is **EDA-based pseudo labels**, not expert-verified labels.
- Anomaly timestamps are produced at the **window level** (24h windows with 1h stride), then grouped into episodes.

---

## Files produced by the repo

Preprocessed arrays:
- `preprocessed/X_train.npy`
- `preprocessed/X_val.npy`
- `preprocessed/X_test.npy`
- `preprocessed/p1_min.npy`
- `preprocessed/p1_max.npy`

Model/analysis results:
- `result/if_results.csv`
- `result/lstm_results.csv`
- `result/combined_results.csv`
- `result/final_anomaly_report.csv`

