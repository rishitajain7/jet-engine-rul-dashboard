import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
import pickle
import os

st.set_page_config(layout="wide")

# ============================================================
# FIXED ABSOLUTE FILE PATHS
# ============================================================

DATA_FILES = {
    "test": "test_FD001.txt",
    "rul": "RUL_FD001.txt",
    "model": "rul_model.h5",
    "scaler": "scaler.pkl"
}

WINDOW_SIZE = 50
TARGET_LAYER_NAME = "lstm_feature_map"

# ============================================================
# LOAD DATA + MODEL
# ============================================================

@st.cache_data
def load_data_and_model():

    # Check files exist
    for key, path in DATA_FILES.items():

        if not os.path.exists(path):
            st.error(f"❌ File not found: {path}")
            st.stop()

    # Load test dataset
    cols = (
        ["engine", "cycle"]
        + [f"op{i}" for i in range(1, 4)]
        + [f"s{i}" for i in range(1, 22)]
    )

    test = pd.read_csv(
        DATA_FILES["test"],
        sep=r"\s+",
        names=cols,
        header=None
    )

    # Load true RUL
    true_rul = pd.read_csv(
        DATA_FILES["rul"],
        header=None
    )

    # Feature selection
    DROP_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]

    features = [
        c for c in test.columns
        if c not in ["engine", "cycle"] + DROP_SENSORS
    ]

    # Load trained model
    model = tf.keras.models.load_model(
        DATA_FILES["model"],
        custom_objects={
            "mse": tf.keras.losses.MeanSquaredError()
        }
    )

    # Load scaler
    with open(DATA_FILES["scaler"], "rb") as f:
        scaler = pickle.load(f)

    # Scale test features
    test[features] = scaler.transform(test[features])

    # Create test windows
    X, y = [], []

    for i, eng in enumerate(test["engine"].unique()):

        temp = test[test["engine"] == eng][features].values

        pad = WINDOW_SIZE - len(temp)

        if pad > 0:
            temp = np.vstack([
                np.zeros((pad, len(features))),
                temp
            ])

        X.append(temp[-WINDOW_SIZE:])
        y.append(true_rul.iloc[i, 0])

    return model, np.array(X), np.array(y), features


# ============================================================
# GRAD-CAM FOR TIME SERIES
# ============================================================

def compute_grad_cam(model, input_data, target_layer_name):

    feature_map_model = Model(
        model.inputs,
        model.get_layer(target_layer_name).output
    )

    # Find target layer index
    target_idx = None

    for i, layer in enumerate(model.layers):

        if layer.name == target_layer_name:
            target_idx = i
            break

    with tf.GradientTape() as tape:

        feature_maps = feature_map_model(input_data)
        tape.watch(feature_maps)

        x = feature_maps

        for layer in model.layers[target_idx + 1:]:
            x = layer(x)

        prediction = x[0][0]

    grads = tape.gradient(prediction, feature_maps)

    pooled = tf.reduce_mean(grads, axis=1)

    weighted = feature_maps * pooled[:, tf.newaxis, :]

    heatmap = tf.reduce_sum(weighted, axis=-1)[0].numpy()

    heatmap = np.maximum(heatmap, 0)

    heatmap /= (heatmap.max() + 1e-8)

    return heatmap


# ============================================================
# STREAMLIT DASHBOARD UI
# ============================================================

def main():

    st.title("🛠️ Jet Engine RUL Monitoring Dashboard")

    st.write(
        "Real-time Remaining Useful Life estimation with Grad-CAM interpretability."
    )

    model, X_test, y_test, features = load_data_and_model()

    # Sidebar
    st.sidebar.header("Engine Selection")

    engine_id = st.sidebar.slider(
        "Select Engine",
        1,
        len(X_test),
        1
    )

    idx = engine_id - 1

    input_window = X_test[idx:idx+1]

    # Prediction
    with st.spinner("Computing prediction + Grad-CAM..."):

        pred_rul = float(model.predict(input_window)[0][0])

        true_val = int(y_test[idx])

        heatmap = compute_grad_cam(
            model,
            input_window,
            TARGET_LAYER_NAME
        )

    # Layout
    col1, col2 = st.columns([1, 2])

    # ========================================================
    # LEFT PANEL
    # ========================================================

    with col1:

        st.subheader(f"Engine {engine_id} Health Status")

        if pred_rul > 100:
            color = "#2ecc71"
            status = "HEALTHY"

        elif pred_rul > 50:
            color = "#f1c40f"
            status = "WARNING"

        else:
            color = "#e74c3c"
            status = "CRITICAL"

        st.markdown(
            f"""
            <div style="
                padding:20px;
                border-radius:10px;
                border:4px solid {color};
                text-align:center;
            ">
                <h3 style="color:{color};">
                    Predicted RUL
                </h3>

                <h1 style="color:{color};">
                    {pred_rul:.1f} cycles
                </h1>

                <p style="
                    font-weight:bold;
                    color:{color};
                ">
                    {status}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.metric(
            "True RUL (Dataset)",
            true_val
        )

    # ========================================================
    # RIGHT PANEL
    # ========================================================

    with col2:

        st.subheader("Grad-CAM: Cycle Importance Map")

        fig, ax = plt.subplots(figsize=(10, 4))

        ax.bar(
            np.arange(WINDOW_SIZE),
            heatmap,
            color=plt.cm.jet(heatmap)
        )

        ax.set_title(
            "Most Influential Timesteps in the 50-cycle Window"
        )

        ax.set_xlabel(
            "Cycle Index (49 = Most Recent)"
        )

        ax.set_ylabel(
            "Importance"
        )

        st.pyplot(fig)

        st.info("""
        **Interpretation:**
        - Red/Yellow bars → cycles that most influenced the RUL prediction
        - Blue bars → low influence
        - The model typically focuses on the most recent 10–15 cycles
        """)


# ============================================================
# RUN APP
# ============================================================

if __name__ == "__main__":

    tf.config.run_functions_eagerly(True)

    main()