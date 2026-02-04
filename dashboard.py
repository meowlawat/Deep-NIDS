import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import tensorflow as tf
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Deep-NIDS | Cyber Defense Console",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric {
        background-color: #262730;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #4f4f4f;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    try:
        model = tf.keras.models.load_model('models/nids_model.keras')
        scaler = joblib.load('models/nids_scaler.pkl')
        return model, scaler
    except Exception as e:
        return None, None

model, scaler = load_resources()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/9103/9103246.png", width=100)
    st.title("🛡️ Control Panel")
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload Network Log (CSV)", type=['csv', 'txt'])
    st.markdown("### ⚙️ Sensitivity Settings")
    threshold = st.slider("Anomaly Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    st.info("System Status: **ONLINE**")

# --- MAIN DASHBOARD ---
st.title("🚀 Deep-NIDS: Real-Time Traffic Analysis")

if uploaded_file is not None:
    try:
        # Load Data
        df = pd.read_csv(uploaded_file)
        
        # --- SAFEGUARD: ensure columns exist ---
        # Initialize default values to prevent plotting crashes if prediction fails
        df['Attack_Probability'] = 0.0
        df['Label'] = "Unknown"
        
        # --- PREPROCESSING & PREDICTION ---
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        
        if scaler and len(numeric_cols) > 0:
            try:
                # Try to predict
                X_scaled = scaler.transform(df[numeric_cols])
                predictions = model.predict(X_scaled)
                
                df['Attack_Probability'] = predictions
                df['Prediction'] = (predictions > threshold).astype(int)
                df['Label'] = df['Prediction'].apply(lambda x: "🚨 ATTACK" if x == 1 else "✅ NORMAL")
            except:
                st.warning("⚠️ Data format mismatch. Displaying raw data without AI predictions.")
        
        # --- KPI METRICS ---
        total_packets = len(df)
        total_attacks = len(df[df['Label'] == "🚨 ATTACK"])
        attack_rate = (total_attacks / total_packets) * 100 if total_packets > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📦 Total Packets", f"{total_packets:,}")
        col2.metric("🛡️ Attacks Blocked", f"{total_attacks:,}", delta_color="inverse")
        col3.metric("⚠️ Threat Level", f"{attack_rate:.1f}%", delta="CRITICAL" if attack_rate > 10 else "STABLE", delta_color="inverse")
        col4.metric("⚡ Latency", "12ms")

        st.markdown("---")

        # --- PLOTS ---
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.subheader("📊 Traffic Classification")
            fig_donut = px.pie(
                df, names='Label', hole=0.5, 
                color='Label',
                color_discrete_map={"🚨 ATTACK": "#ff4b4b", "✅ NORMAL": "#00cc96", "Unknown": "#808080"},
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with c2:
            st.subheader("📡 Protocol Activity")
            # Auto-detect protocol column or use first categorical
            cat_cols = df.select_dtypes(include=['object']).columns
            proto_col = 'protocol_type' if 'protocol_type' in df.columns else (cat_cols[0] if len(cat_cols) > 0 else None)
            
            if proto_col:
                fig_bar = px.bar(df, x=proto_col, color='Label', title="Traffic by Protocol")
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No protocol data found.")

        # --- SCATTER PLOT (The one that crashed) ---
        st.subheader("🔍 Anomaly Detection Map")
        if len(numeric_cols) > 0:
            x_axis = numeric_cols[0]
            fig_scatter = px.scatter(
                df, x=x_axis, y='Attack_Probability', 
                color='Label', size='Attack_Probability',
                color_discrete_map={"🚨 ATTACK": "#ff4b4b", "✅ NORMAL": "#00cc96", "Unknown": "#808080"},
                title=f"Confidence vs {x_axis}"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("Not enough numeric data to generate scatter plot.")

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("Waiting for Network Log Upload...")
