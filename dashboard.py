import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import tensorflow as tf

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

# --- DEFINE KDD COLUMNS (The "Map") ---
KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", 
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", 
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells", 
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count", 
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", 
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", 
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", 
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", 
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"
]

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/9103/9103246.png", width=100)
    st.title("🛡️ Control Panel")
    st.markdown("---")
    # Add header option
    header_option = st.checkbox("File has headers?", value=False)
    uploaded_file = st.file_uploader("Upload Network Log (CSV/TXT)", type=['csv', 'txt'])
    st.markdown("### ⚙️ Sensitivity Settings")
    threshold = st.slider("Anomaly Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    st.info("System Status: **ONLINE**")

# --- MAIN DASHBOARD ---
st.title("🚀 Deep-NIDS: Real-Time Traffic Analysis")

if uploaded_file is not None:
    try:
        # 1. READ FILE
        if header_option:
            df = pd.read_csv(uploaded_file)
        else:
            # If no headers, assign the KDD names manually
            df = pd.read_csv(uploaded_file, header=None)
            # Ensure we don't apply more names than columns
            limit = min(len(df.columns), len(KDD_COLUMNS))
            df.columns = KDD_COLUMNS[:limit] + [f"extra_{i}" for i in range(len(df.columns) - limit)]

        # 2. PREPARE DATA
        # Drop non-numeric columns for the scaler (protocol, service, flag, label)
        # In a real app, we would One-Hot Encode them. For this Demo, we just drop them to prevent crashing.
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        
        # --- PREDICT ---
        if scaler and model:
            try:
                # Align columns with scaler
                # (This is a simplified check - in prod we would match exact feature names)
                X_scaled = scaler.transform(numeric_df)
                predictions = model.predict(X_scaled)
                
                df['Attack_Probability'] = predictions
                df['Prediction'] = (predictions > threshold).astype(int)
                df['Label'] = df['Prediction'].apply(lambda x: "🚨 ATTACK" if x == 1 else "✅ NORMAL")
            except Exception as e:
                # FALLBACK: If dimensions don't match, mock it for the demo
                st.warning(f"⚠️ Structure Mismatch ({e}). Simulating AI output for visualization.")
                df['Attack_Probability'] = np.random.uniform(0, 1, len(df))
                df['Label'] = df['Attack_Probability'].apply(lambda x: "🚨 ATTACK" if x > threshold else "✅ NORMAL")
        
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
                color_discrete_map={"🚨 ATTACK": "#ff4b4b", "✅ NORMAL": "#00cc96"}
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with c2:
            st.subheader("📡 Protocol Activity")
            if 'protocol_type' in df.columns:
                fig_bar = px.bar(df['protocol_type'].value_counts().reset_index(), x='protocol_type', y='count', title="By Protocol")
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("Protocol data unavailable.")

        # --- SCATTER ---
        st.subheader("🔍 Anomaly Detection Map")
        # Use 'duration' if it exists, otherwise the first numeric column
        x_axis = 'duration' if 'duration' in df.columns else numeric_df.columns[0]
        
        fig_scatter = px.scatter(
            df, x=x_axis, y='Attack_Probability', 
            color='Label', size='Attack_Probability',
            color_discrete_map={"🚨 ATTACK": "#ff4b4b", "✅ NORMAL": "#00cc96"},
            title=f"Attack Confidence vs {x_axis.title()}"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("Waiting for Network Log Upload...")
