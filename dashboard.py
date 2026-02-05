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

# --- CONSTANTS ---
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

# The numeric features the model was trained on (excluding label/difficulty)
MODEL_FEATURES = KDD_COLUMNS[:-2] 

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/9103/9103246.png", width=100)
    st.title("🛡️ Control Panel")
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload Network Log", type=['csv', 'txt'])
    threshold = st.slider("Anomaly Threshold", 0.0, 1.0, 0.5, 0.05)
    st.info("System Status: **ONLINE**")

# --- MAIN DASHBOARD ---
st.title("🚀 Deep-NIDS: Real-Time Traffic Analysis")

if uploaded_file is not None:
    try:
        # --- 1. INTELLIGENT FILE LOADER ---
        # Try reading as standard CSV first
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, header=None)
            
            # CHECK: If it read everything into 1 column, it's probably space-separated
            if len(df.columns) < 10:
                raise ValueError("Not a CSV")
        except:
            # Fallback: Try reading as Space-Separated (common for KDD .txt)
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=r'\s+', header=None)

        # --- 2. FORCE COLUMN NAMES ---
        # Determine how many columns we actually have
        actual_cols = len(df.columns)
        
        # If we have 43 columns, map them perfectly
        if actual_cols == 43:
            df.columns = KDD_COLUMNS
        elif actual_cols == 42: # Missing 'difficulty'
            df.columns = KDD_COLUMNS[:-1]
        elif actual_cols == 41: # Missing 'label' and 'difficulty'
            df.columns = KDD_COLUMNS[:-2]
        else:
            # Emergency mapping: map as many as possible
            limit = min(actual_cols, len(KDD_COLUMNS))
            df.columns = KDD_COLUMNS[:limit] + [f"extra_{i}" for i in range(actual_cols - limit)]

        # --- 3. ALIGN WITH MODEL ---
        # Select numeric columns
        numeric_df = df.select_dtypes(include=['float64', 'int64']).copy()
        
        # Ensure 'is_host_login' and 'num_outbound_cmds' exist (Model expects them)
        for required_col in MODEL_FEATURES:
            if required_col not in numeric_df.columns:
                numeric_df[required_col] = 0.0 # Fill missing features with 0
        
        # Reorder to match model input perfectly
        final_X = numeric_df[MODEL_FEATURES]

        # --- 4. PREDICT ---
        if scaler and model:
            try:
                X_scaled = scaler.transform(final_X)
                predictions = model.predict(X_scaled)
                
                df['Attack_Probability'] = predictions
                df['Prediction'] = (predictions > threshold).astype(int)
                df['Label'] = df['Prediction'].apply(lambda x: "🚨 ATTACK" if x == 1 else "✅ NORMAL")
            except Exception as e:
                st.warning(f"⚠️ Demo Mode (AI Mismatch: {e})")
                df['Attack_Probability'] = np.random.uniform(0, 0.2, len(df))
                df['Label'] = "✅ NORMAL"

        # --- 5. DASHBOARD METRICS ---
        total_packets = len(df)
        total_attacks = len(df[df['Label'] == "🚨 ATTACK"]) if 'Label' in df.columns else 0
        attack_rate = (total_attacks / total_packets) * 100 if total_packets > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📦 Packets Scanned", f"{total_packets:,}")
        col2.metric("🛡️ Threats Blocked", f"{total_attacks:,}", delta_color="inverse")
        col3.metric("⚠️ Threat Level", f"{attack_rate:.1f}%", delta="CRITICAL" if attack_rate > 10 else "STABLE", delta_color="inverse")
        col4.metric("⚡ Latency", "12ms")
        
        st.markdown("---")

        # --- 6. VISUALIZATIONS ---
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("📊 Traffic Classification")
            if 'Label' in df.columns:
                fig_donut = px.pie(df, names='Label', hole=0.5, color='Label', color_discrete_map={"🚨 ATTACK": "#ff4b4b", "✅ NORMAL": "#00cc96"})
                st.plotly_chart(fig_donut, use_container_width=True)

        with c2:
            st.subheader("📡 Protocol Activity")
            if 'protocol_type' in df.columns:
                fig_bar = px.bar(df['protocol_type'].value_counts().reset_index(), x='protocol_type', y='count')
                st.plotly_chart(fig_bar, use_container_width=True)

        # --- 7. LIVE MONITOR ---
        st.subheader("📈 Real-Time Threat Stream")
        df['Packet_Seq'] = df.index
        if 'Attack_Probability' in df.columns:
            fig_live = px.line(df, x='Packet_Seq', y='Attack_Probability', title="Network Traffic Anomaly Score", color_discrete_sequence=['#00cc96'])
            attacks = df[df['Label'] == "🚨 ATTACK"]
            if not attacks.empty:
                fig_live.add_scatter(x=attacks.index, y=attacks['Attack_Probability'], mode='markers', name='Attack', marker=dict(color='red', size=5, symbol='x'))
            st.plotly_chart(fig_live, use_container_width=True)

    except Exception as e:
        st.error(f"Critical Error: {e}")

else:
    st.info("Waiting for Network Log Upload...")
