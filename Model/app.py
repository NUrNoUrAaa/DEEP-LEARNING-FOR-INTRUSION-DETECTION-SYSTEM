import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Page Setup
st.set_page_config(page_title="IDS Final Project", layout="centered")

# --- 1. Load Artifacts ---
@st.cache_resource
def load_artifacts():
    try:
        model = load_model('binary_model_final.keras')
        scaler = joblib.load('final_scaler.pkl')
        features = joblib.load('deployment_features.pkl')

        scaler_feature_count = getattr(scaler, 'n_features_in_', len(getattr(scaler, 'data_min_', [])))
        safe_feature_count = min(len(features), scaler_feature_count)
        if safe_feature_count == 0:
            st.error("Scaler metadata is empty. Cannot determine feature count.")
            return None, None, None

        if len(features) != safe_feature_count:
            st.warning(f"Feature list trimmed from {len(features)} to {safe_feature_count} "
                       "to stay aligned with the scaler/model.")
        features = features[:safe_feature_count]

        return model, scaler, features
    except Exception as e:
        st.error("Error loading model/scaler/features.")
        return None, None, None

model, scaler, model_features = load_artifacts()

st.title("Network Intrusion Detection System (IDS)")
st.write("Deep Learning Based Intrusion Detection System")

uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])

# ============================================================
# ==================== DATASET UPLOAD ========================
# ============================================================

if uploaded_file and model_features is not None:
    df = pd.read_csv(uploaded_file)

    if df.columns[0].startswith('Unnamed'):
        df = df.drop(df.columns[0], axis=1)

    st.write(f"Uploaded Data Shape: {df.shape}")

    try:
        df_ready = df[model_features].copy()
        st.success("Columns matched successfully.")

        if st.button("Start Analysis"):
            X = df_ready.replace([np.inf, -np.inf], np.nan).fillna(0)
            X_scaled = scaler.transform(X)

            preds = model.predict(X_scaled, verbose=0)

            results = pd.DataFrame()
            results["Prediction"] = ["Attack" if p > 0.5 else "Benign" for p in preds]
            results["Confidence"] = preds

            attacks = (preds > 0.5).sum()

            if attacks > 0:
                st.error(f"Alert: {int(attacks)} Threats Detected!")
            else:
                st.success("System Secure: No Threats Found.")

            st.dataframe(results.head(20), use_container_width=True)

    except KeyError as e:
        st.error(f"Missing Column: {e}")
        st.warning("Ensure your dataset contains correct feature names.")

else:
    if model_features is None:
        st.error("System files missing. Check folder.")


# ============================================================
# =========== REALISTIC RANDOM DATA (MIN-MAX BASED) ===========
# ============================================================

st.subheader("Generate Realistic Synthetic Data")

if st.button("Generate Realistic Random Data (Min–Max Based)"):
    try:
        feature_min = scaler.data_min_
        feature_max = scaler.data_max_
    except:
        st.error("Scaler is not MinMaxScaler.")
        st.stop()

    rows = 20
    realistic_data = np.zeros((rows, len(model_features)))

    for i in range(len(model_features)):
        realistic_data[:, i] = np.random.uniform(feature_min[i], feature_max[i], rows)

    realistic_df = pd.DataFrame(realistic_data, columns=model_features)

    st.info("Realistic Data Generated from Min–Max")
    st.dataframe(realistic_df, use_container_width=True)

    X_scaled = scaler.transform(realistic_df)
    preds = model.predict(X_scaled, verbose=0)

    results = pd.DataFrame()
    results["Prediction"] = ["Attack" if p > 0.5 else "Benign" for p in preds]
    results["Confidence"] = preds

    st.subheader("Prediction Results")
    st.dataframe(results, use_container_width=True)

    st.download_button(
        "Download Generated Data (CSV)",
        realistic_df.to_csv(index=False),
        file_name="realistic_minmax_data.csv"
    )


# ============================================================
# =============== ADVANCED REALISTIC GENERATORS ==============
# ============================================================

st.subheader("Advanced Synthetic Data Generator")

def get_stats_from_scaler(scaler):
    try:
        data_min = scaler.data_min_
        data_max = scaler.data_max_
        data_range = data_max - data_min
        mean = data_min + data_range / 2
        std = data_range / 6
        return data_min, data_max, mean, std
    except:
        return None, None, None, None

data_min, data_max, mean_vec, std_vec = get_stats_from_scaler(scaler)


# -------------------------------------------------------------
# 1) Generate Realistic Normal Distribution (Mean/Std)
# -------------------------------------------------------------
if st.button("Generate Mean/Std Realistic Data (Normal Distribution)"):
    rows = 20
    synthetic = np.zeros((rows, len(model_features)))

    for i in range(len(model_features)):
        synthetic[:, i] = np.random.normal(mean_vec[i], std_vec[i], rows)

    df_syn = pd.DataFrame(synthetic, columns=model_features)

    st.info("Generated Mean/Std Normal Distributed Data")
    st.dataframe(df_syn, use_container_width=True)

    X_scaled = scaler.transform(df_syn)
    preds = model.predict(X_scaled, verbose=0)

    results = pd.DataFrame()
    results["Prediction"] = ["Attack" if p > 0.5 else "Benign" for p in preds]
    results["Confidence"] = preds

    st.subheader("Prediction Results")
    st.dataframe(results, use_container_width=True)

    st.download_button("Download CSV (Mean/Std Synthetic)", 
                       df_syn.to_csv(index=False), 
                       file_name="synthetic_mean_std.csv")


# -------------------------------------------------------------
# 2) Generate Benign-like Data
# -------------------------------------------------------------
if st.button("Generate Benign-Like Data"):
    rows = 20
    benign_like = np.zeros((rows, len(model_features)))

    for i in range(len(model_features)):
        low = data_min[i]
        high = data_min[i] + (data_max[i] - data_min[i]) * 0.3
        benign_like[:, i] = np.random.uniform(low, high, rows)

    df_benign = pd.DataFrame(benign_like, columns=model_features)

    st.info("Generated Benign-like Synthetic Data")
    st.dataframe(df_benign, use_container_width=True)

    X_scaled = scaler.transform(df_benign)
    preds = model.predict(X_scaled, verbose=0)

    results = pd.DataFrame()
    results["Prediction"] = ["Attack" if p > 0.5 else "Benign" for p in preds]
    results["Confidence"] = preds

    st.subheader("Prediction Results (Benign-like)")
    st.dataframe(results, use_container_width=True)

    st.download_button("Download CSV (Benign-like)", 
                       df_benign.to_csv(index=False), 
                       file_name="synthetic_benign_like.csv")


# -------------------------------------------------------------
# 3) Generate Attack-like Data
# -------------------------------------------------------------
if st.button("Generate Attack-Like Data"):
    rows = 20
    attack_like = np.zeros((rows, len(model_features)))

    for i in range(len(model_features)):
        low = data_min[i] + (data_max[i] - data_min[i]) * 0.7
        high = data_max[i]
        attack_like[:, i] = np.random.uniform(low, high, rows)

    df_attack = pd.DataFrame(attack_like, columns=model_features)

    st.warning("Generated Attack-like Synthetic Data")
    st.dataframe(df_attack, use_container_width=True)

    X_scaled = scaler.transform(df_attack)
    preds = model.predict(X_scaled, verbose=0)

    results = pd.DataFrame()
    results["Prediction"] = ["Attack" if p > 0.5 else "Benign" for p in preds]
    results["Confidence"] = preds

    st.subheader("Prediction Results (Attack-like)")
    st.dataframe(results, use_container_width=True)

    st.download_button("Download CSV (Attack-like)", 
                       df_attack.to_csv(index=False), 
                       file_name="synthetic_attack_like.csv")
