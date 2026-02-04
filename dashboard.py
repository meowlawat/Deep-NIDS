import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import time
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(page_title="Deep-NIDS Dashboard", layout="wide", page_icon="🛡️")

# --- LOAD ARTIFACTS ---
@st.cache_resource
def load_system():
    try:
        # UPDATED PATHS: Pointing to the 'models/' folder now
        model = tf.keras.models.load_model("models/nids_model.keras")
        scaler = joblib.load("models/nids_scaler.pkl")
        threshold = joblib.load("models/nids_threshold.pkl")
        return model, scaler, threshold
    except Exception as e:
        st.error(f"Error loading files: {e}. Did you run main.py to train the models?")
        return None, None, None

model, scaler, threshold = load_system()

# --- HEADER ---
st.title("🛡️ DEEP-NIDS: AI-Powered Network Defense")
if threshold is not None:
    st.markdown(f"**System Status:** ACTIVE | **Model Type:** LSTM Autoencoder | **Anomaly Threshold:** `{threshold:.5f}`")

# --- SIDEBAR ---
st.sidebar.header("Simulation Controls")
upload_file = st.sidebar.file_uploader("Upload Network Traffic (CSV)", type=["txt", "csv"])
speed = st.sidebar.slider("Simulation Speed (sec)", 0.05, 1.0, 0.1)
run_sim = st.sidebar.button("🚨 START LIVE MONITORING")

# --- MAIN DISPLAY ---
col1, col2 = st.columns([3, 1])
with col1:
    chart_placeholder = st.empty()
with col2:
    alert_placeholder = st.empty()

# --- LOGIC ---
if upload_file is not None and run_sim and model is not None:
    # Load Data
    col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"]
    
    df = pd.read_csv(upload_file, names=col_names)
    
    # Quick cleanup
    df_clean = df.drop(['difficulty', 'protocol_type', 'service', 'flag', 'label'], axis=1)
    
    # Scale and Reshape
    X_scaled = scaler.transform(df_clean)
    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    
    # STREAMING LOOP
    loss_history = []
    
    for i in range(len(X_lstm)):
        packet = X_lstm[i:i+1]
        
        # Predict
        reconstruction = model.predict(packet, verbose=0)
        mse = np.mean(np.power(packet - reconstruction, 2))
        loss_history.append(mse)
        
        # Keep chart clean (last 50 points)
        if len(loss_history) > 50: loss_history.pop(0)
        
        # Check Attack
        is_attack = mse > threshold
        
        # Update Chart
        with chart_placeholder:
            fig = px.line(y=loss_history, title="Real-Time Reconstruction Error (MSE)")
            fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Threshold")
            
            # LOCK AXIS TO PREVENT JUMPING
            fig.update_layout(
                yaxis_range=[0, 0.05], 
                xaxis_title="Time Window",
                yaxis_title="Reconstruction Error"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        # Update Alert
        with alert_placeholder:
            if is_attack:
                st.error(f"🔴 ATTACK DETECTED\n\nPacket: {i}\nError: {mse:.5f}")
            else:
                st.success(f"🟢 SYSTEM SECURE\n\nPacket: {i}\nError: {mse:.5f}")
        
        time.sleep(speed)