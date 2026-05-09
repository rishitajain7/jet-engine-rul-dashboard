"""
============================================================
Jet Engine RUL Prediction - LSTM Training Pipeline
NASA CMAPSS FD001 Dataset
============================================================
Run this script ONCE to generate:
  - rul_model.h5
  - scaler.pkl

Then launch the dashboard:
  py -3.12 -m streamlit run "dashboard (2) (1).py"
============================================================
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
from pathlib import Path

# ── TensorFlow / Keras ──────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================================
# CONFIGURATION
# ============================================================

WINDOW_SIZE   = 50          # must match dashboard
MAX_RUL       = 130         # piecewise-linear RUL cap (common CMAPSS practice)
EPOCHS        = 60
BATCH_SIZE    = 64
VAL_SPLIT     = 0.15
RANDOM_SEED   = 42

# ── Paths ────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
TRAIN_FILE   = SCRIPT_DIR / "train_FD001.txt"
TEST_FILE    = SCRIPT_DIR / "test_FD001.txt"
RUL_FILE     = SCRIPT_DIR / "RUL_FD001.txt"
MODEL_OUT    = SCRIPT_DIR / "rul_model.h5"
SCALER_OUT   = SCRIPT_DIR / "scaler.pkl"

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================================================
# 1.  LOAD DATA
# ============================================================

COLS = (
    ["engine", "cycle"]
    + [f"op{i}" for i in range(1, 4)]
    + [f"s{i}"  for i in range(1, 22)]
)

print("► Loading training data …")
train = pd.read_csv(TRAIN_FILE, sep=r"\s+", names=COLS, header=None)
test  = pd.read_csv(TEST_FILE,  sep=r"\s+", names=COLS, header=None)
true_rul = pd.read_csv(RUL_FILE, header=None, names=["rul"])

print(f"  Train shape : {train.shape}")
print(f"  Test  shape : {test.shape}")
print(f"  Engines (train): {train['engine'].nunique()}")

# ============================================================
# 2.  FEATURE SELECTION
#     Drop constant / near-constant sensors (standard for FD001)
# ============================================================

DROP_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]
FEATURES = [c for c in COLS if c not in ["engine", "cycle"] + DROP_SENSORS]

print(f"\n► Using {len(FEATURES)} features: {FEATURES}")

# ============================================================
# 3.  RUL LABEL GENERATION  (piecewise-linear / capped)
# ============================================================

print("\n► Generating RUL labels …")

def add_rul(df, max_rul=MAX_RUL):
    grouped = df.groupby("engine")["cycle"].max().reset_index()
    grouped.columns = ["engine", "max_cycle"]
    df = df.merge(grouped, on="engine")
    df["rul"] = df["max_cycle"] - df["cycle"]
    df["rul"] = df["rul"].clip(upper=max_rul)   # piecewise-linear cap
    df.drop(columns=["max_cycle"], inplace=True)
    return df

train = add_rul(train)

# ============================================================
# 4.  FEATURE SCALING
# ============================================================

print("► Fitting StandardScaler on training features …")
scaler = StandardScaler()
train[FEATURES] = scaler.fit_transform(train[FEATURES])

# Save scaler immediately
with open(SCALER_OUT, "wb") as f:
    pickle.dump(scaler, f)
print(f"  ✓ Scaler saved → {SCALER_OUT}")

# ============================================================
# 5.  SEQUENCE GENERATION  (sliding window)
# ============================================================

def make_sequences(df, window=WINDOW_SIZE):
    X, y = [], []
    for eng in df["engine"].unique():
        data = df[df["engine"] == eng]
        vals = data[FEATURES].values
        ruls = data["rul"].values
        for i in range(len(vals)):
            end = i + 1
            start = max(0, end - window)
            seq = vals[start:end]
            # zero-pad at the front if shorter than window
            if len(seq) < window:
                pad = np.zeros((window - len(seq), len(FEATURES)))
                seq = np.vstack([pad, seq])
            X.append(seq)
            y.append(ruls[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

print("► Building training sequences (this may take ~30 s) …")
X_train, y_train = make_sequences(train)
print(f"  X_train : {X_train.shape}   y_train : {y_train.shape}")

# ============================================================
# 6.  LSTM MODEL
# ============================================================

print("\n► Building LSTM model …")

n_features = len(FEATURES)

model = Sequential([
    Input(shape=(WINDOW_SIZE, n_features)),

    # ── Named LSTM layer required by Grad-CAM in dashboard ──
    LSTM(128, return_sequences=True, name="lstm_feature_map"),
    Dropout(0.2),

    LSTM(64, return_sequences=False, name="lstm_2"),
    Dropout(0.2),

    Dense(32, activation="relu", name="dense_1"),
    Dense(1,  activation="linear", name="output")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="mse",
    metrics=["mae"]
)

model.summary()

# ============================================================
# 7.  TRAINING
# ============================================================

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]

print("\n► Training …")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VAL_SPLIT,
    callbacks=callbacks,
    verbose=1
)

# ============================================================
# 8.  SAVE MODEL
# ============================================================

model.save(str(MODEL_OUT))
print(f"\n  ✓ Model saved  → {MODEL_OUT}")

# ============================================================
# 9.  EVALUATION ON TEST SET
# ============================================================

print("\n► Evaluating on test set …")

# Scale test features with the SAME fitted scaler
test[FEATURES] = scaler.transform(test[FEATURES])

# Build one window per engine (last WINDOW_SIZE cycles, zero-padded if needed)
X_test, y_test = [], []
for i, eng in enumerate(test["engine"].unique()):
    vals = test[test["engine"] == eng][FEATURES].values
    pad  = WINDOW_SIZE - len(vals)
    if pad > 0:
        vals = np.vstack([np.zeros((pad, n_features)), vals])
    X_test.append(vals[-WINDOW_SIZE:])
    y_test.append(true_rul.iloc[i, 0])

X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)

y_pred = model.predict(X_test).flatten()

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print(f"\n  RMSE : {rmse:.2f} cycles")
print(f"  MAE  : {mae:.2f} cycles")
print(f"  R²   : {r2:.4f}")

# ============================================================
# 10.  PLOTS
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Training / Validation Loss
axes[0].plot(history.history["loss"],     label="Train Loss")
axes[0].plot(history.history["val_loss"], label="Val Loss")
axes[0].set_title("Training vs Validation Loss (MSE)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE")
axes[0].legend()
axes[0].grid(True)

# Actual vs Predicted RUL
axes[1].scatter(y_test, y_pred, alpha=0.5, s=20, label="Predictions")
lims = [0, max(y_test.max(), y_pred.max()) + 10]
axes[1].plot(lims, lims, "r--", label="Perfect fit")
axes[1].set_title(f"Actual vs Predicted RUL  (R²={r2:.3f})")
axes[1].set_xlabel("Actual RUL")
axes[1].set_ylabel("Predicted RUL")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plot_path = SCRIPT_DIR / "training_results.png"
plt.savefig(str(plot_path), dpi=120)
print(f"\n  ✓ Training plot saved → {plot_path}")
plt.show()

# ============================================================
# DONE
# ============================================================

print("\n" + "="*60)
print("  Training complete!")
print(f"  Model  : {MODEL_OUT}")
print(f"  Scaler : {SCALER_OUT}")
print("="*60)
print("\nNow launch the dashboard:")
print('  py -3.12 -m streamlit run "dashboard (2) (1).py"')