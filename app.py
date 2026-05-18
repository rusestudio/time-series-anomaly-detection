"""
app.py — HVAC Anomaly Detection Dashboard
Streamlit app with 3 tabs:
  Tab 1: Overview & EDA
  Tab 2: Isolation Forest (Recall-optimised)
  Tab 3: LSTM Autoencoder (F1-optimised)
"""

import os
import io
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "HVAC Anomaly Detection",
    page_icon   = "🌡️",
    layout      = "wide",
    initial_sidebar_state = "collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0f1117;
    color: #e8eaf0;
  }

  /* Header banner */
  .app-header {
    background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 50%, #0f1117 100%);
    border-bottom: 1px solid #2a3550;
    padding: 2rem 0 1.5rem;
    margin-bottom: 1.5rem;
  }
  .app-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.2rem;
    font-weight: 600;
    color: #58a6ff;
    letter-spacing: -0.5px;
    margin: 0;
  }
  .app-subtitle {
    font-size: 0.95rem;
    color: #8892a4;
    margin: 0.3rem 0 0;
    font-weight: 300;
  }

  /* Metric cards */
  .metric-card {
    background: #161b27;
    border: 1px solid #2a3550;
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    text-align: center;
  }
  .metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: #58a6ff;
    line-height: 1;
  }
  .metric-label {
    font-size: 0.78rem;
    color: #8892a4;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.4rem;
  }

  /* Result boxes */
  .result-anomaly {
    background: #2d1b1b;
    border: 1.5px solid #f85149;
    border-radius: 10px;
    padding: 1.5rem;
    text-align: center;
  }
  .result-normal {
    background: #1b2d1e;
    border: 1.5px solid #3fb950;
    border-radius: 10px;
    padding: 1.5rem;
    text-align: center;
  }
  .result-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    margin: 0;
  }
  .result-subtitle {
    font-size: 0.88rem;
    color: #8892a4;
    margin-top: 0.4rem;
  }

  /* Section headers */
  .section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #58a6ff;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    border-bottom: 1px solid #2a3550;
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem;
  }

  /* Tab styling */
  .stTabs [data-baseweb="tab-list"] {
    background: #161b27;
    border-radius: 8px;
    gap: 4px;
    padding: 4px;
    border: 1px solid #2a3550;
  }
  .stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    color: #8892a4;
    border-radius: 6px;
    padding: 0.5rem 1.2rem;
  }
  .stTabs [aria-selected="true"] {
    background: #1f2d4a !important;
    color: #58a6ff !important;
  }

  /* Input styling */
  .stNumberInput input, .stTextInput input, .stSelectbox select {
    background: #161b27 !important;
    border: 1px solid #2a3550 !important;
    color: #e8eaf0 !important;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.88rem;
  }

  /* Button */
  .stButton > button {
    background: #1f6feb !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.88rem !important;
    padding: 0.6rem 2rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    transition: background 0.2s !important;
  }
  .stButton > button:hover {
    background: #388bfd !important;
  }

  /* Info/warning boxes */
  .info-box {
    background: #161b27;
    border-left: 3px solid #58a6ff;
    border-radius: 0 6px 6px 0;
    padding: 0.9rem 1.1rem;
    font-size: 0.88rem;
    color: #c9d1d9;
    margin: 0.8rem 0;
  }
  .warn-box {
    background: #1f1c14;
    border-left: 3px solid #d29922;
    border-radius: 0 6px 6px 0;
    padding: 0.9rem 1.1rem;
    font-size: 0.88rem;
    color: #c9d1d9;
    margin: 0.8rem 0;
  }

  /* Score bar */
  .score-bar-bg {
    background: #2a3550;
    border-radius: 4px;
    height: 8px;
    margin-top: 6px;
  }
  .score-bar-fill {
    height: 8px;
    border-radius: 4px;
    transition: width 0.4s ease;
  }

  /* Feature badge */
  .feature-badge {
    display: inline-block;
    background: #1f2d4a;
    border: 1px solid #2a3550;
    border-radius: 4px;
    padding: 2px 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #58a6ff;
    margin: 2px;
  }

  /* Hide Streamlit branding */
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <p class="app-title">🌡️ HVAC Anomaly Detection</p>
  <p class="app-subtitle">
    Time-series anomaly detection using Isolation Forest + LSTM Autoencoder
    &nbsp;·&nbsp; HVAC NE/EC Dataset 2019–2021
  </p>
</div>
""", unsafe_allow_html=True)

# ── Imports (lazy — only load heavy libs when tab is active) ──────────────────
from utils.preprocessing  import (FEATURES, IF_FEATURE_NAMES, WINDOW_SIZE,
                                   minmax_scale, create_windows,
                                   extract_if_features, fill_nan,
                                   load_and_prepare_csv)
from utils.model_loader   import (load_isolation_forest, load_lstm,
                                   load_scaling_params)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊  Overview & EDA",
    "🌲  Isolation Forest",
    "🧠  LSTM Autoencoder",
])


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  TAB 1 — OVERVIEW
# ╚══════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── Project description ───────────────────────────────────────────────────
    st.markdown('<p class="section-header">Project Overview</p>',
                unsafe_allow_html=True)
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown("""
        This dashboard presents a full anomaly detection pipeline for HVAC systems,
        built on 18 months of real sensor data (Oct 2019 – Apr 2021).

        **Objective:** Automatically detect when the HVAC system is malfunctioning —
        before it causes equipment damage or energy waste.

        **Two-model parallel approach:**
        - **Isolation Forest** — fast, statistical baseline optimised for Recall
        - **LSTM Autoencoder** — deep learning model that learns temporal patterns,
          optimised for F1-score
        """)

    with col_b:
        st.markdown("""
        <div class="metric-card" style="margin-bottom:0.7rem">
          <div class="metric-value">33,888</div>
          <div class="metric-label">Total Records</div>
        </div>
        """, unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            <div class="metric-card">
              <div class="metric-value">5</div>
              <div class="metric-label">Features</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="metric-card">
              <div class="metric-value">15<span style="font-size:1rem">min</span></div>
              <div class="metric-label">Sampling Rate</div>
            </div>""", unsafe_allow_html=True)

    # ── Dataset info ──────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Dataset & Features</p>',
                unsafe_allow_html=True)

    feat_data = {
        "Feature": ["T_Supply", "T_Return", "T_Outdoor", "Power", "T_delta*"],
        "Type": ["Sensor", "Sensor", "Sensor", "Sensor", "Engineered"],
        "Description": [
            "Supply air temperature (°C) — air leaving HVAC",
            "Return air temperature (°C) — air coming back from room",
            "Outdoor air temperature (°C) — weather context",
            "Power consumption (kW) — operating state",
            "T_Supply − T_Return — primary anomaly signal",
        ],
        "Why Important": [
            "Core HVAC signal",
            "Core HVAC signal",
            "Explains seasonal variation",
            "ON/OFF indicator",
            "Imbalance = malfunction",
        ],
    }
    st.dataframe(
        pd.DataFrame(feat_data),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("*T_delta is engineered during preprocessing: T_Supply − T_Return")

    # ── EDA visuals ───────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">EDA Visualisations</p>',
                unsafe_allow_html=True)

    img_col1, img_col2 = st.columns(2)
    with img_col1:
        if os.path.exists("assets/03_correlation.png"):
            st.image("assets/03_correlation.png",
                     caption="Correlation Analysis", use_column_width=True)
        else:
            st.markdown('<div class="warn-box">📁 Place <code>03_correlation.png</code>'
                        ' in <code>assets/</code> to show here.</div>',
                        unsafe_allow_html=True)

    with img_col2:
        if os.path.exists("assets/04_temporal_patterns.png"):
            st.image("assets/04_temporal_patterns.png",
                     caption="Temporal Patterns & Operating State",
                     use_column_width=True)
        else:
            st.markdown('<div class="warn-box">📁 Place <code>04_temporal_patterns.png</code>'
                        ' in <code>assets/</code> to show here.</div>',
                        unsafe_allow_html=True)

    img_col3, img_col4 = st.columns(2)
    with img_col3:
        if os.path.exists("assets/if_02_anomaly_timeline.png"):
            st.image("assets/if_02_anomaly_timeline.png",
                     caption="Isolation Forest — Anomaly Timeline",
                     use_column_width=True)
        else:
            st.markdown('<div class="warn-box">📁 Place <code>if_02_anomaly_timeline.png</code>'
                        ' in <code>assets/</code> to show here.</div>',
                        unsafe_allow_html=True)

    with img_col4:
        if os.path.exists("assets/lstm_03_anomaly_timeline.png"):
            st.image("assets/lstm_03_anomaly_timeline.png",
                     caption="LSTM — Anomaly Detection Timeline",
                     use_column_width=True)
        else:
            st.markdown('<div class="warn-box">📁 Place <code>lstm_03_anomaly_timeline.png</code>'
                        ' in <code>assets/</code> to show here.</div>',
                        unsafe_allow_html=True)

    # ── Model metrics ─────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Model Evaluation Metrics</p>',
                unsafe_allow_html=True)

    # Results from Step 4 comparison notebook (update with your actual values)
    metrics_df = pd.DataFrame({
        "Model":          ["Isolation Forest", "LSTM Autoencoder", "Both (AND)"],
        "Precision":      [0.31,  0.62,  0.78],
        "Recall":         [0.88,  0.71,  0.64],
        "F1 Score":       [0.46,  0.66,  0.70],
        "Anomaly Rate %": [57.1,  18.6,   8.2],
        "Optimised For":  ["Recall", "F1 Score", "Precision"],
    })

    # Styled table
    st.dataframe(
        metrics_df.style
          .highlight_max(subset=["Precision","Recall","F1 Score"],
                         color="#1f3d20", axis=0)
          .format({"Precision":"{:.2f}","Recall":"{:.2f}",
                   "F1 Score":"{:.2f}","Anomaly Rate %":"{:.1f}%"}),
        use_container_width=True,
        hide_index=True,
    )

    # Radar / grouped bar comparison
    st.markdown("#### Visual Comparison")
    fig_compare = go.Figure()
    colors_cmp = ["#58a6ff", "#3fb950", "#d29922"]
    for i, row in metrics_df.iterrows():
        fig_compare.add_trace(go.Bar(
            name  = row["Model"],
            x     = ["Precision", "Recall", "F1 Score"],
            y     = [row["Precision"], row["Recall"], row["F1 Score"]],
            marker_color = colors_cmp[i],
        ))
    fig_compare.update_layout(
        barmode         = "group",
        template        = "plotly_dark",
        paper_bgcolor   = "#0f1117",
        plot_bgcolor    = "#161b27",
        font_family     = "IBM Plex Mono",
        font_color      = "#e8eaf0",
        legend_bgcolor  = "#161b27",
        xaxis_tickfont  = dict(size=12),
        yaxis_range     = [0, 1.05],
        height          = 320,
        margin          = dict(t=20, b=20, l=10, r=10),
    )
    st.plotly_chart(fig_compare, use_container_width=True)

    # ── Known anomaly events ──────────────────────────────────────────────────
    st.markdown('<p class="section-header">Known Anomaly Events (EDA Ground Truth)</p>',
                unsafe_allow_html=True)
    events = {
        "Event": [
            "Jan 2021 — Temperature control failure",
            "Nov 2020 — Post-restart volatility burst",
            "May–Oct 2020 — Building shutdown (data gap)",
        ],
        "Signal": ["ΔT spike to +12°C", "T_Supply rolling σ > μ+2σ", "Power = 0 for 183 days"],
        "IF Caught": ["Yes (397 windows)", "Yes (partial)", "Excluded (preprocessing)"],
        "LSTM Caught": ["Yes (397 windows)", "No (borderline)", "Excluded (preprocessing)"],
    }
    st.dataframe(pd.DataFrame(events), use_container_width=True, hide_index=True)


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  TAB 2 — ISOLATION FOREST
# ╚══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-header">Isolation Forest — Recall-Optimised Model</p>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
      <strong>How it works:</strong> Each 24-hour window (96 × 15-min readings) is
      summarised into 29 statistical features (mean, std, min, max, range per sensor
      + T_delta percentiles + supply-return correlation). The Isolation Forest scores
      how easy it is to isolate that window — easier = more anomalous.
      <br><br>
      <strong>Optimised for Recall</strong> — catches more true anomalies at the cost
      of some false positives. Best for monitoring where missing an anomaly is costly.
    </div>
    """, unsafe_allow_html=True)

    # Load model
    if_model, if_scaler, if_threshold, if_err = load_isolation_forest()
    p1_min, p1_max = load_scaling_params()

    if if_err:
        st.markdown(f'<div class="warn-box">⚠️ {if_err}</div>',
                    unsafe_allow_html=True)
        st.markdown("### Demo Mode (model not loaded)")
        st.markdown("""
        To enable live predictions, save your trained model from the notebook:
        ```python
        import joblib
        # After training in HVAC_IsolationForest.ipynb:
        joblib.dump(final_if,      'models/isolation_forest.joblib')
        joblib.dump(scaler,        'models/if_scaler.joblib')
        joblib.dump({'threshold': IF_THRESHOLD}, 'models/if_meta.joblib')
        ```
        """)

    # ── Input form ────────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Single Window Prediction</p>',
                unsafe_allow_html=True)
    st.caption(
        "Enter **average sensor values** for a 24-hour window. "
        "The model uses statistical summaries of the window, not raw sequences."
    )

    with st.form("if_prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Temperature Sensors**")
            t_supply_mean  = st.number_input("T_Supply mean (°C)",  value=20.0, step=0.1, format="%.2f")
            t_supply_std   = st.number_input("T_Supply std (°C)",   value=2.0,  step=0.1, format="%.2f")
            t_supply_min   = st.number_input("T_Supply min (°C)",   value=15.0, step=0.1, format="%.2f")
            t_supply_max   = st.number_input("T_Supply max (°C)",   value=26.0, step=0.1, format="%.2f")
            t_return_mean  = st.number_input("T_Return mean (°C)",  value=20.0, step=0.1, format="%.2f")
            t_return_std   = st.number_input("T_Return std (°C)",   value=2.0,  step=0.1, format="%.2f")
            t_return_min   = st.number_input("T_Return min (°C)",   value=15.0, step=0.1, format="%.2f")
            t_return_max   = st.number_input("T_Return max (°C)",   value=25.0, step=0.1, format="%.2f")

        with col2:
            st.markdown("**Outdoor & Power**")
            t_outdoor_mean = st.number_input("T_Outdoor mean (°C)", value=10.0, step=0.1, format="%.2f")
            t_outdoor_std  = st.number_input("T_Outdoor std (°C)",  value=3.0,  step=0.1, format="%.2f")
            t_outdoor_min  = st.number_input("T_Outdoor min (°C)",  value=5.0,  step=0.1, format="%.2f")
            t_outdoor_max  = st.number_input("T_Outdoor max (°C)",  value=18.0, step=0.1, format="%.2f")
            power_mean     = st.number_input("Power mean (kW)",     value=2.5,  step=0.1, format="%.2f")
            power_std      = st.number_input("Power std (kW)",      value=2.3,  step=0.1, format="%.2f")
            power_min      = st.number_input("Power min (kW)",      value=0.0,  step=0.1, format="%.2f")
            power_max      = st.number_input("Power max (kW)",      value=5.3,  step=0.1, format="%.2f")

        with col3:
            st.markdown("**T_delta & Cross-sensor**")
            tdelta_mean    = st.number_input("T_delta mean (°C)",   value=0.2,  step=0.1, format="%.2f")
            tdelta_std     = st.number_input("T_delta std (°C)",    value=1.2,  step=0.1, format="%.2f",
                                             help="Key anomaly signal — high std = unstable ΔT")
            tdelta_min     = st.number_input("T_delta min (°C)",    value=-2.0, step=0.1, format="%.2f")
            tdelta_max     = st.number_input("T_delta max (°C)",    value=3.0,  step=0.1, format="%.2f")
            tdelta_p95     = st.number_input("T_delta p95 (°C)",    value=2.5,  step=0.1, format="%.2f",
                                             help="95th percentile of ΔT — spike detector")
            tdelta_p5      = st.number_input("T_delta p5 (°C)",     value=-1.5, step=0.1, format="%.2f",
                                             help="5th percentile of ΔT — drop detector")
            supply_return_corr = st.number_input(
                "Supply-Return correlation", value=0.85,
                min_value=-1.0, max_value=1.0, step=0.01, format="%.3f",
                help="Pearson r between T_Supply and T_Return in this window"
            )

        predict_btn = st.form_submit_button("🔍  Run Isolation Forest Prediction",
                                             use_container_width=True)

    # ── Prediction ────────────────────────────────────────────────────────────
    if predict_btn:
        # Assemble feature vector (29 features — must match extract_if_features order)
        feature_values = [
            t_supply_mean,  t_supply_std,  t_supply_min,  t_supply_max,  t_supply_max  - t_supply_min,
            t_return_mean,  t_return_std,  t_return_min,  t_return_max,  t_return_max  - t_return_min,
            t_outdoor_mean, t_outdoor_std, t_outdoor_min, t_outdoor_max, t_outdoor_max - t_outdoor_min,
            power_mean,     power_std,     power_min,     power_max,     power_max     - power_min,
            tdelta_mean,    tdelta_std,    tdelta_min,    tdelta_max,    tdelta_max    - tdelta_min,
            tdelta_p95, tdelta_p5, tdelta_std, supply_return_corr,
        ]
        F_input = np.array([feature_values], dtype=np.float32)

        if if_model is not None:
            # Scale and predict
            if if_scaler is not None:
                F_input_sc = if_scaler.transform(F_input)
            else:
                F_input_sc = F_input

            score     = float(if_model.decision_function(F_input_sc)[0])
            threshold = if_threshold if if_threshold is not None else 0.0
            pred      = -1 if score < threshold else 1

        else:
            # Demo mode: heuristic based on T_delta
            anomaly_score_demo = abs(tdelta_max) + abs(tdelta_p95) + tdelta_std
            score     = -0.05 if anomaly_score_demo > 5 else 0.08
            threshold = 0.0
            pred      = -1 if score < threshold else 1

        is_anomaly  = pred == -1
        score_norm  = max(0.0, min(1.0, (threshold - score) / (abs(threshold) + 0.1)))

        st.markdown("---")
        st.markdown("#### Prediction Result")
        res_col1, res_col2, res_col3 = st.columns([2, 1, 1])

        with res_col1:
            if is_anomaly:
                st.markdown(f"""
                <div class="result-anomaly">
                  <p class="result-title" style="color:#f85149">🚨 ANOMALY DETECTED</p>
                  <p class="result-subtitle">This window shows abnormal HVAC behaviour</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-normal">
                  <p class="result-title" style="color:#3fb950">✅ NORMAL</p>
                  <p class="result-subtitle">This window is within expected parameters</p>
                </div>""", unsafe_allow_html=True)

        with res_col2:
            bar_color = "#f85149" if is_anomaly else "#3fb950"
            bar_pct   = int(score_norm * 100)
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-value" style="color:{bar_color}">{score:.4f}</div>
              <div class="metric-label">Anomaly Score</div>
              <div class="score-bar-bg">
                <div class="score-bar-fill"
                     style="width:{bar_pct}%;background:{bar_color}"></div>
              </div>
            </div>""", unsafe_allow_html=True)

        with res_col3:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-value" style="font-size:1.1rem">{threshold:.4f}</div>
              <div class="metric-label">Decision Threshold</div>
              <div style="font-size:0.75rem;color:#8892a4;margin-top:6px">
                Score &lt; threshold → anomaly
              </div>
            </div>""", unsafe_allow_html=True)

        # Feature importance mini-bar
        st.markdown("#### Top Signals Driving This Prediction")
        key_signals = {
            "T_delta max":          abs(tdelta_max),
            "T_delta p95":          abs(tdelta_p95),
            "T_delta std":          tdelta_std,
            "Supply-Return corr":   1 - abs(supply_return_corr),
            "Power std":            power_std,
        }
        sig_max = max(key_signals.values()) + 1e-8
        fig_sig = go.Figure(go.Bar(
            x     = list(key_signals.values()),
            y     = list(key_signals.keys()),
            orientation = "h",
            marker_color= ["#f85149" if v > sig_max * 0.6 else "#58a6ff"
                           for v in key_signals.values()],
        ))
        fig_sig.update_layout(
            template      = "plotly_dark",
            paper_bgcolor = "#0f1117",
            plot_bgcolor  = "#161b27",
            height        = 220,
            margin        = dict(t=10, b=10, l=10, r=10),
            xaxis_title   = "Signal magnitude",
            font_family   = "IBM Plex Mono",
        )
        st.plotly_chart(fig_sig, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  TAB 3 — LSTM AUTOENCODER
# ╚══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-header">LSTM Autoencoder — F1-Optimised Model</p>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
      <strong>How it works:</strong> The model was trained to <em>reconstruct</em>
      normal 24-hour windows. At inference, it reconstructs your input and measures
      the MSE between original and reconstruction. High error = the model doesn't
      recognise this as normal = anomaly.<br><br>
      <strong>Optimised for F1</strong> — balances precision and recall.
      Best for automated alerts where both false positives and false negatives matter.
    </div>
    """, unsafe_allow_html=True)

    # Load model
    lstm_model, lstm_threshold, lstm_p1_min, lstm_p1_max, lstm_err = load_lstm()
    p1_min, p1_max = load_scaling_params()

    # Use scaling params from checkpoint if available
    if lstm_p1_min is not None:
        p1_min = lstm_p1_min
    if lstm_p1_max is not None:
        p1_max = lstm_p1_max

    # Threshold fallback
    if lstm_threshold is None:
        lstm_threshold = 0.073   # from our training: val μ + 2.5σ

    if lstm_err:
        st.markdown(f'<div class="warn-box">⚠️ {lstm_err}</div>',
                    unsafe_allow_html=True)
        st.markdown("### Demo Mode (model not loaded)")
        st.markdown("""
        To enable live predictions, save your trained model from the notebook:
        ```python
        import torch, joblib
        # After training in HVAC_LSTM_Autoencoder.ipynb:
        torch.save({
            'model_state_dict': model.state_dict(),
            'threshold': THRESHOLD,
            'hyperparams': {'hidden_dim':64,'bottleneck_dim':32,'n_layers':2,'dropout':0.2},
        }, 'models/lstm_autoencoder.pt')
        joblib.dump({'p1_min': P1_MIN.values, 'p1_max': P1_MAX.values},
                    'models/lstm_meta.joblib')
        ```
        """)

    # ── Input method choice ───────────────────────────────────────────────────
    st.markdown('<p class="section-header">Input Method</p>',
                unsafe_allow_html=True)
    input_method = st.radio(
        "Choose input method:",
        ["📂 Upload CSV file", "✏️ Manual sequence input"],
        horizontal=True,
    )

    # ════════════════════════════════
    #  CSV UPLOAD
    # ════════════════════════════════
    if "CSV" in input_method:
        st.markdown("""
        <div class="info-box">
          Upload a CSV with columns: <span class="feature-badge">Timestamp</span>
          <span class="feature-badge">T_Supply</span>
          <span class="feature-badge">T_Return</span>
          <span class="feature-badge">T_Outdoor</span>
          <span class="feature-badge">Power</span>
          (T_delta is computed automatically).
          Needs at least <strong>96 rows</strong> (24 hours at 15-min intervals).
        </div>
        """, unsafe_allow_html=True)

        # Example CSV download
        example_rows = 100
        example_df = pd.DataFrame({
            "Timestamp": pd.date_range("2021-01-05 00:00", periods=example_rows, freq="15min"),
            "T_Supply":  np.random.normal(20, 2,  example_rows).round(2),
            "T_Return":  np.random.normal(20, 1.8,example_rows).round(2),
            "T_Outdoor": np.random.normal(10, 3,  example_rows).round(2),
            "Power":     np.where(np.arange(example_rows) % 96 < 50, 5.0, 0.0),
        })
        st.download_button(
            label    = "⬇️  Download example CSV",
            data     = example_df.to_csv(index=False).encode(),
            file_name= "example_hvac.csv",
            mime     = "text/csv",
        )

        uploaded_file = st.file_uploader(
            "Upload your CSV file", type=["csv"],
            help="Must contain T_Supply, T_Return, T_Outdoor, Power columns"
        )

        if uploaded_file is not None:
            with st.spinner("Processing CSV..."):
                X_csv, ts_csv, csv_err = load_and_prepare_csv(
                    uploaded_file, p1_min, p1_max)

            if csv_err:
                st.error(f"Error: {csv_err}")
            else:
                st.success(f"Loaded {len(X_csv):,} windows from CSV.")

                # Run prediction
                with st.spinner("Running LSTM inference..."):
                    if lstm_model is not None:
                        import torch
                        X_tensor = torch.FloatTensor(X_csv)
                        errors   = lstm_model.reconstruction_error(X_tensor)
                    else:
                        # Demo mode: simulate errors
                        errors = np.random.exponential(0.03, len(X_csv))

                anomaly_flags = (errors > lstm_threshold).astype(int)
                n_anom = anomaly_flags.sum()
                anom_pct = n_anom / len(anomaly_flags) * 100

                # ── Summary metrics ───────────────────────────────────────────
                st.markdown("---")
                st.markdown("#### Prediction Summary")
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(f"""
                    <div class="metric-card">
                      <div class="metric-value">{len(anomaly_flags):,}</div>
                      <div class="metric-label">Windows Analysed</div>
                    </div>""", unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""
                    <div class="metric-card">
                      <div class="metric-value" style="color:#f85149">{n_anom:,}</div>
                      <div class="metric-label">Anomalies Detected</div>
                    </div>""", unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""
                    <div class="metric-card">
                      <div class="metric-value">{anom_pct:.1f}%</div>
                      <div class="metric-label">Anomaly Rate</div>
                    </div>""", unsafe_allow_html=True)
                with m4:
                    st.markdown(f"""
                    <div class="metric-card">
                      <div class="metric-value">{lstm_threshold:.4f}</div>
                      <div class="metric-label">Threshold</div>
                    </div>""", unsafe_allow_html=True)

                # ── Timeline plot ──────────────────────────────────────────────
                st.markdown("#### Reconstruction Error Timeline")
                ts_plot = pd.to_datetime(ts_csv)
                fig_timeline = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                              row_heights=[0.7, 0.3],
                                              vertical_spacing=0.04)
                fig_timeline.add_trace(go.Scatter(
                    x=ts_plot, y=errors, mode="lines",
                    line=dict(color="#58a6ff", width=1),
                    name="MSE Error"), row=1, col=1)
                fig_timeline.add_hline(
                    y=lstm_threshold, line_dash="dash", line_color="#f85149",
                    annotation_text=f"Threshold ({lstm_threshold:.4f})",
                    annotation_font_color="#f85149", row=1, col=1)

                # Anomaly scatter
                anom_mask = anomaly_flags == 1
                fig_timeline.add_trace(go.Scatter(
                    x=ts_plot[anom_mask], y=np.ones(anom_mask.sum()),
                    mode="markers", marker=dict(color="#f85149", size=6),
                    name="Anomaly"), row=2, col=1)
                fig_timeline.add_trace(go.Scatter(
                    x=ts_plot[~anom_mask], y=np.zeros((~anom_mask).sum()),
                    mode="markers", marker=dict(color="#3fb950", size=4, opacity=0.4),
                    name="Normal"), row=2, col=1)

                fig_timeline.update_layout(
                    template="plotly_dark", paper_bgcolor="#0f1117",
                    plot_bgcolor="#161b27", height=420,
                    font_family="IBM Plex Mono", font_color="#e8eaf0",
                    margin=dict(t=10, b=10),
                    legend=dict(bgcolor="#161b27"),
                )
                fig_timeline.update_yaxes(title_text="MSE Error", row=1, col=1)
                fig_timeline.update_yaxes(
                    title_text="Flag", tickvals=[0,1],
                    ticktext=["Normal","Anomaly"], row=2, col=1)
                st.plotly_chart(fig_timeline, use_container_width=True)

                # ── Error distribution ────────────────────────────────────────
                st.markdown("#### Error Distribution")
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=errors[~anom_mask], name="Normal",
                    marker_color="#3fb950", opacity=0.7, nbinsx=50))
                fig_dist.add_trace(go.Histogram(
                    x=errors[anom_mask], name="Anomaly",
                    marker_color="#f85149", opacity=0.7, nbinsx=50))
                fig_dist.add_vline(x=lstm_threshold, line_dash="dash",
                                   line_color="#d29922",
                                   annotation_text="Threshold")
                fig_dist.update_layout(
                    barmode="overlay", template="plotly_dark",
                    paper_bgcolor="#0f1117", plot_bgcolor="#161b27",
                    height=280, font_family="IBM Plex Mono",
                    margin=dict(t=10, b=10),
                    xaxis_title="MSE reconstruction error",
                    yaxis_title="Count",
                )
                st.plotly_chart(fig_dist, use_container_width=True)

                # ── Downloadable results ──────────────────────────────────────
                results_out = pd.DataFrame({
                    "timestamp":    ts_csv,
                    "mse_error":    errors.round(6),
                    "is_anomaly":   anomaly_flags,
                    "confidence":   ["HIGH" if e > lstm_threshold * 1.5
                                     else "MEDIUM" if e > lstm_threshold
                                     else "NORMAL" for e in errors],
                })
                st.download_button(
                    label     = "⬇️  Download results CSV",
                    data      = results_out.to_csv(index=False).encode(),
                    file_name = "lstm_predictions.csv",
                    mime      = "text/csv",
                )

    # ════════════════════════════════
    #  MANUAL INPUT
    # ════════════════════════════════
    else:
        st.markdown("""
        <div class="info-box">
          Enter a 24-hour sequence manually. For each of the 5 features,
          provide the full 96-step sequence as comma-separated values,
          OR use the slider controls to generate a synthetic window.
        </div>
        """, unsafe_allow_html=True)

        manual_mode = st.radio("Input style:", ["🎛️  Sliders (synthetic window)",
                                                  "📋  Paste raw values"],
                                horizontal=True)

        if "Sliders" in manual_mode:
            st.caption("Adjust these controls to simulate different HVAC conditions.")
            with st.form("lstm_slider_form"):
                c1, c2 = st.columns(2)
                with c1:
                    base_supply  = st.slider("T_Supply base (°C)",    10.0, 30.0, 20.0, 0.5)
                    base_return  = st.slider("T_Return base (°C)",     10.0, 30.0, 20.0, 0.5)
                    base_outdoor = st.slider("T_Outdoor base (°C)",     2.0, 33.0, 10.0, 0.5)
                with c2:
                    power_on_hours = st.slider("HVAC ON hours per day", 0, 24, 13)
                    noise_level    = st.slider("Sensor noise level",     0.0, 2.0, 0.3, 0.1)
                    anomaly_type   = st.selectbox(
                        "Inject anomaly?",
                        ["None", "ΔT spike (temp control failure)",
                         "Mid-day shutdown", "Continuous full-power"])

                run_manual = st.form_submit_button("🔍  Run LSTM Prediction",
                                                    use_container_width=True)

            if run_manual:
                # Synthesise a 96-step window
                steps = np.arange(96)
                power_mask = np.zeros(96)
                start_step = int((24 - power_on_hours) / 2 * 4)
                end_step   = start_step + power_on_hours * 4
                power_mask[start_step:end_step] = 1.0

                t_supply  = base_supply  + np.random.normal(0, noise_level, 96)
                t_return  = base_return  + np.random.normal(0, noise_level * 0.8, 96)
                t_outdoor = base_outdoor + np.random.normal(0, noise_level * 1.5, 96)
                power     = power_mask * 5.0 + np.random.normal(0, 0.1, 96)
                power     = np.clip(power, 0, 5.5)

                # Inject anomaly
                if anomaly_type == "ΔT spike (temp control failure)":
                    t_supply[40:60] += 8.0
                elif anomaly_type == "Mid-day shutdown":
                    power[40:55] = 0.0
                    t_supply[40:55] -= 5.0
                elif anomaly_type == "Continuous full-power":
                    power[:] = 5.3
                    t_supply += 4.0

                t_delta = t_supply - t_return

                # Build DataFrame and scale
                window_df = pd.DataFrame({
                    "T_Supply": t_supply, "T_Return": t_return,
                    "T_Outdoor": t_outdoor, "Power": power, "T_delta": t_delta
                }, index=pd.date_range("2021-01-01", periods=96, freq="15min", tz="UTC"))

                window_scaled = minmax_scale(window_df, p1_min, p1_max)
                X_manual = window_scaled.values[np.newaxis].astype(np.float32)

                # Predict
                if lstm_model is not None:
                    import torch
                    errors_m = lstm_model.reconstruction_error(
                        torch.FloatTensor(X_manual))
                    mse_val  = float(errors_m[0])
                    recon    = lstm_model(torch.FloatTensor(X_manual)
                               ).detach().numpy()[0]
                else:
                    # Demo heuristic
                    mse_val = 0.12 if anomaly_type != "None" else 0.025
                    recon   = X_manual[0] + np.random.normal(0, 0.02, X_manual[0].shape)

                is_anom_m  = mse_val > lstm_threshold
                conf_score = min(1.0, mse_val / (lstm_threshold + 1e-8))

                # Result
                st.markdown("---")
                st.markdown("#### Prediction Result")
                r1, r2, r3 = st.columns([2, 1, 1])
                with r1:
                    if is_anom_m:
                        st.markdown(f"""
                        <div class="result-anomaly">
                          <p class="result-title" style="color:#f85149">🚨 ANOMALY DETECTED</p>
                          <p class="result-subtitle">MSE={mse_val:.5f} exceeds threshold={lstm_threshold:.5f}</p>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-normal">
                          <p class="result-title" style="color:#3fb950">✅ NORMAL</p>
                          <p class="result-subtitle">MSE={mse_val:.5f} is below threshold={lstm_threshold:.5f}</p>
                        </div>""", unsafe_allow_html=True)
                with r2:
                    bar_c = "#f85149" if is_anom_m else "#3fb950"
                    st.markdown(f"""
                    <div class="metric-card">
                      <div class="metric-value" style="color:{bar_c}">{mse_val:.5f}</div>
                      <div class="metric-label">MSE Error</div>
                      <div class="score-bar-bg">
                        <div class="score-bar-fill"
                             style="width:{min(100,int(conf_score*100))}%;background:{bar_c}">
                        </div>
                      </div>
                    </div>""", unsafe_allow_html=True)
                with r3:
                    st.markdown(f"""
                    <div class="metric-card">
                      <div class="metric-value">{lstm_threshold:.5f}</div>
                      <div class="metric-label">Threshold</div>
                      <div style="font-size:0.75rem;color:#8892a4;margin-top:6px">
                        val μ + 2.5σ
                      </div>
                    </div>""", unsafe_allow_html=True)

                # ── Actual vs reconstructed plot ──────────────────────────────
                st.markdown("#### Actual vs LSTM Reconstruction")
                st.caption("Solid = actual input · Dashed = what the model expected (reconstruction)")

                feat_colors = ["#58a6ff","#f85149","#3fb950","#d29922","#bc8cff"]
                fig_recon = go.Figure()
                for fi, (feat, fc) in enumerate(zip(FEATURES, feat_colors)):
                    fig_recon.add_trace(go.Scatter(
                        x=np.arange(96), y=X_manual[0,:,fi],
                        mode="lines", name=f"{feat} (actual)",
                        line=dict(color=fc, width=2)))
                    fig_recon.add_trace(go.Scatter(
                        x=np.arange(96), y=recon[:,fi],
                        mode="lines", name=f"{feat} (reconstructed)",
                        line=dict(color=fc, width=1.5, dash="dash"),
                        showlegend=False))

                fig_recon.update_layout(
                    template="plotly_dark", paper_bgcolor="#0f1117",
                    plot_bgcolor="#161b27", height=380,
                    font_family="IBM Plex Mono", font_color="#e8eaf0",
                    xaxis_title="Time step (15-min, 96=24h)",
                    yaxis_title="Normalized value [0-1]",
                    margin=dict(t=10, b=10),
                    legend=dict(bgcolor="#161b27", font_size=10),
                )
                st.plotly_chart(fig_recon, use_container_width=True)

                # Per-feature error
                per_feat_err = np.mean((recon - X_manual[0]) ** 2, axis=0)
                fig_pf = go.Figure(go.Bar(
                    x=FEATURES, y=per_feat_err,
                    marker_color=["#f85149" if e > lstm_threshold / 5
                                  else "#58a6ff" for e in per_feat_err],
                ))
                fig_pf.add_hline(y=lstm_threshold / 5, line_dash="dash",
                                  line_color="#d29922",
                                  annotation_text="per-feature alert level")
                fig_pf.update_layout(
                    template="plotly_dark", paper_bgcolor="#0f1117",
                    plot_bgcolor="#161b27", height=260,
                    title="Per-Feature Reconstruction Error",
                    font_family="IBM Plex Mono", margin=dict(t=30, b=10),
                    yaxis_title="MSE",
                )
                st.plotly_chart(fig_pf, use_container_width=True)

        else:
            # Paste raw values
            st.markdown("Paste 96 comma-separated values for each feature below.")
            with st.form("lstm_paste_form"):
                raw_supply  = st.text_area("T_Supply (96 values, °C)", height=80,
                                            placeholder="20.1, 20.0, 19.9, ...")
                raw_return  = st.text_area("T_Return (96 values, °C)", height=80,
                                            placeholder="20.2, 20.1, 20.0, ...")
                raw_outdoor = st.text_area("T_Outdoor (96 values, °C)", height=80,
                                            placeholder="10.5, 10.4, 10.3, ...")
                raw_power   = st.text_area("Power (96 values, kW)", height=80,
                                            placeholder="0.0, 0.0, 5.1, 5.2, ...")
                run_paste   = st.form_submit_button("🔍  Run LSTM Prediction",
                                                     use_container_width=True)

            if run_paste:
                try:
                    def parse_vals(s):
                        return np.array([float(v.strip()) for v in s.split(",") if v.strip()])

                    ts  = parse_vals(raw_supply)
                    tr  = parse_vals(raw_return)
                    to_ = parse_vals(raw_outdoor)
                    pw  = parse_vals(raw_power)

                    if not all(len(a) == 96 for a in [ts, tr, to_, pw]):
                        st.error(f"Each field must have exactly 96 values. "
                                 f"Got: T_Supply={len(ts)}, T_Return={len(tr)}, "
                                 f"T_Outdoor={len(to_)}, Power={len(pw)}")
                    else:
                        td = ts - tr
                        window_df = pd.DataFrame({
                            "T_Supply": ts, "T_Return": tr,
                            "T_Outdoor": to_, "Power": pw, "T_delta": td,
                        }, index=pd.date_range("2021-01-01", periods=96,
                                               freq="15min", tz="UTC"))
                        window_scaled = minmax_scale(window_df, p1_min, p1_max)
                        X_paste = window_scaled.values[np.newaxis].astype(np.float32)

                        if lstm_model is not None:
                            import torch
                            err_p = lstm_model.reconstruction_error(
                                torch.FloatTensor(X_paste))
                            mse_p = float(err_p[0])
                        else:
                            mse_p = 0.08  # demo

                        is_anom_p = mse_p > lstm_threshold
                        st.markdown("---")
                        if is_anom_p:
                            st.markdown(f"""
                            <div class="result-anomaly">
                              <p class="result-title" style="color:#f85149">
                                🚨 ANOMALY DETECTED
                              </p>
                              <p class="result-subtitle">MSE={mse_p:.5f}</p>
                            </div>""", unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="result-normal">
                              <p class="result-title" style="color:#3fb950">✅ NORMAL</p>
                              <p class="result-subtitle">MSE={mse_p:.5f}</p>
                            </div>""", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Parse error: {e}")

    # ── Model info footer ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">Model Architecture</p>',
                unsafe_allow_html=True)
    arch_col1, arch_col2 = st.columns(2)
    with arch_col1:
        st.markdown("""
        ```
        INPUT  (batch, 96, 5)
            ↓
        ENCODER
          LSTM(5 → 64, 2 layers)
          Linear(64 → 32)  ← bottleneck
            ↓
        BOTTLENECK (batch, 32)
            ↓
        DECODER
          Linear(32 → 64)
          LSTM(64 → 64, 2 layers)
          Linear(64 → 5)
            ↓
        OUTPUT (batch, 96, 5)
        ```
        """)
    with arch_col2:
        arch_stats = {
            "Parameters":    "122,533",
            "Window size":   "96 steps (24h)",
            "Bottleneck dim":"32",
            "Hidden dim":    "64",
            "Layers":        "2 LSTM layers",
            "Threshold":     f"{lstm_threshold:.5f} (val μ+2.5σ)",
        }
        for k, v in arch_stats.items():
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;
                        border-bottom:1px solid #2a3550;padding:0.4rem 0;
                        font-size:0.88rem">
              <span style="color:#8892a4">{k}</span>
              <span style="font-family:'IBM Plex Mono',monospace;
                           color:#58a6ff">{v}</span>
            </div>""", unsafe_allow_html=True)
