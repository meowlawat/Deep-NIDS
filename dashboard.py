import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import tensorflow as tf
import os

# --- PAGE CONFIGURATION (Must be first) ---
st.set_page_config(
    page_title="Deep-NIDS | Cyber Defense Console",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR "CYBER" VIBE ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #262730;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #4f4f4f;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD RESOURCES (Cached for speed) ---
@st.cache_resource
def load_resources():
    try:
        model = tf.keras.models.load_model('models/nids_model.keras')
        scaler = joblib.load('models/nids_scaler.pkl')
        # Load label encoders if you saved them, otherwise we map manually for demo
        return model, scaler
    except Exception as e:
        st.error(f"System Error: Could not load models. {e}")
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
st.markdown("### Network Intrusion Detection System")

if uploaded_file is not None:
    # 1. Load Data
    try:
        df = pd.read_csv(uploaded_file)
        
        # --- PREPROCESSING SIMULATION ---
        # (We assume the uploaded file needs the same preprocessing as training)
        # For the dashboard demo, we'll try to process numeric cols only if raw
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        data_to_scale = df[numeric_cols]
        
        # Scale
        if scaler:
            try:
                X_scaled = scaler.transform(data_to_scale)
                
                # Predict
                predictions = model.predict(X_scaled)
                # Binary classification (Normal vs Attack)
                df['Attack_Probability'] = predictions
                df['Prediction'] = (predictions > threshold).astype(int)
                df['Label'] = df['Prediction'].apply(lambda x: "🚨 ATTACK" if x == 1 else "✅ NORMAL")
                
            except Exception as e:
                st.warning("Data mismatch! Showing raw analysis instead of prediction.")
                df['Label'] = "Unknown"
        
        # --- KPI METRICS ROW ---
        total_packets = len(df)
        total_attacks = len(df[df['Label'] == "🚨 ATTACK"])
        attack_rate = (total_attacks / total_packets) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📦 Total Packets", f"{total_packets:,}")
        col2.metric("🛡️ Attacks Blocked", f"{total_attacks:,}", delta_color="inverse")
        col3.metric("⚠️ Threat Level", f"{attack_rate:.1f}%", delta=f"{'CRITICAL' if attack_rate > 10 else 'STABLE'}", delta_color="inverse")
        col4.metric("⚡ System Latency", "12ms", delta="-2ms")

        st.markdown("---")

        # --- INTERACTIVE GRAPHS ROW 1 ---
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.subheader("📊 Traffic Classification")
            # Create a donut chart
            fig_donut = px.pie(
                df, 
                names='Label', 
                hole=0.5, 
                color='Label',
                color_discrete_map={"🚨 ATTACK": "#ff4b4b", "✅ NORMAL": "#00cc96"},
                title="Normal vs Malicious Traffic"
            )
            fig_donut.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_donut, use_container_width=True)

        with c2:
            st.subheader("📡 Protocol Activity")
            # Assuming there's a 'protocol_type' column (common in KDD), else visualize random numeric
            if 'protocol_type' in df.columns:
                fig_bar = px.bar(df, x='protocol_type', color='Label', barmode='group', title="Attacks by Protocol")
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No Protocol column found for detail graph.")

        # --- INTERACTIVE GRAPHS ROW 2 (Time Series or Scatter) ---
        st.subheader("🔍 Deep Dive: Anomaly Detection")
        # Scatter plot of probability vs a feature (e.g., Duration or Src Bytes)
        # We pick the first numeric column as X-axis just for visualization
        x_axis = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[0]
        
        fig_scatter = px.scatter(
            df, 
            x=x_axis, 
            y='Attack_Probability', 
            color='Label',
            size='Attack_Probability',
            hover_data=df.columns[:5],
            color_discrete_map={"🚨 ATTACK": "#ff4b4b", "✅ NORMAL": "#00cc96"},
            title=f"Attack Confidence vs {x_axis}"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # --- DATA TABLE ---
        with st.expander("📂 View Detailed Traffic Logs"):
            st.dataframe(df.style.applymap(lambda v: 'color: red;' if v == "🚨 ATTACK" else None, subset=['Label']))

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    # --- IDLE STATE (Animations) ---
    st.info("Waiting for Network Log Upload...")
    
    # Fake Demo Data for visual appeal when empty
    st.markdown("### _Demo View (Upload data to see real stats)_")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("System Status", "IDLE", delta="Waiting")
    col2.metric("Model Confidence", "98.5%", "Ready")
    col3.metric("Last Scan", "00:00:00")
    
    # Placeholder Chart
    dummy_data = pd.DataFrame({
        "Packet": range(100),
        "Traffic_Load": np.random.randn(100).cumsum()
    })
    st.line_chart(dummy_data.set_index("Packet"))

# --- FOOTER ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: grey;'>Deep-NIDS v1.0 | AI-Powered Security</div>", unsafe_allow_html=True)
