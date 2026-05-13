"""
save_models.py
Run this script IN your training notebook environment (where models are in memory)
to save everything the Streamlit app needs.

Copy-paste each section into the corresponding notebook cell.
"""

# ════════════════════════════════════════════════
# SECTION A — Run in HVAC_IsolationForest.ipynb
# ════════════════════════════════════════════════
"""
import joblib, os
os.makedirs('models', exist_ok=True)

# Save model
joblib.dump(final_if, 'models/isolation_forest.joblib')

# Save scaler
joblib.dump(scaler, 'models/if_scaler.joblib')

# Save threshold and metadata
joblib.dump({
    'threshold'    : IF_THRESHOLD,
    'contamination': CONTAMINATION,
    'n_features'   : F_train_sc.shape[1],
    'feature_names': feat_names,
}, 'models/if_meta.joblib')

print("Saved: isolation_forest.joblib, if_scaler.joblib, if_meta.joblib")
"""

# ════════════════════════════════════════════════
# SECTION B — Run in HVAC_LSTM_Autoencoder.ipynb
# ════════════════════════════════════════════════
"""
import torch, joblib, os
os.makedirs('models', exist_ok=True)

# Save model checkpoint
torch.save({
    'model_state_dict': model.state_dict(),
    'threshold'       : THRESHOLD,
    'k'               : K,
    'features'        : FEATURES,
    'hyperparams': {
        'hidden_dim'    : 64,
        'bottleneck_dim': 32,
        'n_layers'      : 2,
        'dropout'       : 0.2,
        'window_size'   : 96,
        'batch_size'    : BATCH_SIZE,
        'best_val_loss' : best_val_loss,
    }
}, 'models/lstm_autoencoder.pt')

# Save scaling params
joblib.dump({
    'p1_min': P1_MIN.values,
    'p1_max': P1_MAX.values,
}, 'models/lstm_meta.joblib')

# Also save as .npy (used as fallback)
import numpy as np
np.save('models/p1_min.npy', P1_MIN.values)
np.save('models/p1_max.npy', P1_MAX.values)

print("Saved: lstm_autoencoder.pt, lstm_meta.joblib, p1_min.npy, p1_max.npy")
"""

# ════════════════════════════════════════════════
# SECTION C — Copy EDA images to assets/
# ════════════════════════════════════════════════
"""
import shutil, os
os.makedirs('assets', exist_ok=True)

images_to_copy = [
    '03_correlation.png',
    '04_temporal_patterns.png',
    '05_feature_analysis.png',
    'if_02_anomaly_timeline.png',
    'if_03_feature_importance.png',
    'lstm_03_anomaly_timeline.png',
    'lstm_04_per_feature_errors.png',
]
for img in images_to_copy:
    if os.path.exists(img):
        shutil.copy(img, f'assets/{img}')
        print(f"Copied: {img}")
    else:
        print(f"Not found: {img}")
"""
