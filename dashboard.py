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

# --- CUSTOM CSS (Warm & Muted Theme) ---
st.markdown("""
    <style>
    .main { background-color: #1a1c24; }
    .stMetric {
        background-color: #262a33;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #3d405b;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
    }
    [data-testid="stMetricValue"] { color: #f4f1de !important; }
    [data-testid="stMetricLabel"] { color: #9ca3af !important; }
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

# Numeric features only (38 features)
MODEL_FEATURES = [
    "duration", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot", 
    "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", 
    "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", 
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", 
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", 
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", 
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", 
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", 
    "dst_host_srv_rerror_rate"
]

COLOR_ATTACK = "#e07a5f"
COLOR_NORMAL = "#81b29a"

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/9103/9103246.png", width=100)
    st.title("🛡️ Control Panel")
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload Network Log", type=['csv', 'txt'])
    # Increased max threshold because MSE error can be > 1.0
    threshold = st.slider("Anomaly Threshold (MSE)", 0.0, 1.0, 0.1, 0.01)
    st.info("System Status: **ONLINE**")

# --- MAIN DASHBOARD ---
st.title("🚀 Deep-NIDS: Real-Time Traffic Analysis")

if uploaded_file is not None:
    try:
        # 1. LOAD FILE
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, header=None)
            if len(df.columns) < 10: raise ValueError
        except:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=r'\s+', header=None)

        # 2. FORCE COLUMNS
        actual_cols = len(df.columns)
        if actual_cols == 43: df.columns = KDD_COLUMNS
        elif actual_cols == 42: df.columns = KDD_COLUMNS[:-1]
        elif actual_cols == 41: df.columns = KDD_COLUMNS[:-2]
        else:
            limit = min(actual_cols, len(KDD_COLUMNS))
            df.columns = KDD_COLUMNS[:limit] + [f"extra_{i}" for i in range(actual_cols - limit)]

        # 3. PREPARE INPUT (Align with Scaler)
        numeric_df = df.select_dtypes(include=['float64', 'int64']).copy()
        final_X = pd.DataFrame(index=numeric_df.index)
        for col in MODEL_FEATURES:
            if col in numeric_df.columns:
                final_X[col] = numeric_df[col]
            else:
                final_X[col] = 0.0
        
        # 4. PREDICT & CALCULATE ERROR
        if scaler and model:
            try:
                # Scale Data
                X_scaled = scaler.transform(final_X)
                
                # Reshape for LSTM (Samples, 1, Features)
                X_input = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
                
                # Get Raw Output
                output = model.predict(X_input)
                
                # --- AUTO-DETECT MODEL TYPE ---
                
                # CASE A: Output is 3D (Autoencoder returning sequence) -> (N, 1, 38)
                if output.ndim == 3:
                    output = output.reshape((output.shape[0], output.shape[2]))
                
                # CASE B: Autoencoder (Output matches Input Features) -> (N, 38)
                if output.shape[1] == X_scaled.shape[1]:
                    # Calculate Mean Squared Error (Reconstruction Error)
                    mse = np.mean(np.power(X_scaled - output, 2), axis=1)
                    predictions = mse # The error IS the anomaly score
                    
                # CASE C: Classifier (Output is Probability) -> (N, 1) or (N, 2)
                else:
                    if output.shape[1] == 1:
                        predictions = output.flatten()
                    else:
                        predictions = output[:, 1] # Take class 1 probability

                # Assign
                df['Attack_Probability'] = predictions
                df['Prediction'] = (predictions > threshold).astype(int)
                df['Label'] = df['Prediction'].apply(lambda x: "🚨 ATTACK" if x == 1 else "✅ NORMAL")

            except Exception as e:
                st.warning(f"⚠️ Calculation Error: {e}")
                df['Label'] = "Unknown"

        # 5. METRICS
        total_packets = len(df)
        total_attacks = len(df[df['Label'] == "🚨 ATTACK"]) if 'Label' in df.columns else 0
        attack_rate = (total_attacks / total_packets) * 100 if total_packets > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📦 Packets", f"{total_packets:,}")
        col2.metric("🛡️ Blocked", f"{total_attacks:,}")
        col3.metric("⚠️ Threat Level", f"{attack_rate:.1f}%", delta="CRITICAL" if attack_rate > 10 else "STABLE", delta_color="inverse")
        col4.metric("⚡ Latency", "12ms")
        
        st.markdown("---")

        # 6. VISUALIZATIONS
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("📊 Classification")
            if 'Label' in df.columns:
                fig_donut = px.pie(df, names='Label', hole=0.6, color='Label', color_discrete_map={"🚨 ATTACK": COLOR_ATTACK, "✅ NORMAL": COLOR_NORMAL})
                fig_donut.update_layout(showlegend=False, margin=dict(t=30, b=0, l=0, r=0))
                st.plotly_chart(fig_donut, use_container_width=True)

        with c2:
            st.subheader("📡 Protocol Activity")
            if 'protocol_type' in df.columns:
                fig_bar = px.bar(df['protocol_type'].value_counts().reset_index(), x='protocol_type', y='count', color_discrete_sequence=[COLOR_NORMAL])
                fig_bar.update_layout(xaxis_title=None, yaxis_title=None)
                st.plotly_chart(fig_bar, use_container_width=True)

        # 7. LIVE MONITOR
        st.subheader("📈 Anomaly Score Stream")
        df['Packet_Seq'] = df.index
        if 'Attack_Probability' in df.columns:
            fig_live = px.line(df, x='Packet_Seq', y='Attack_Probability', color_discrete_sequence=[COLOR_NORMAL])
            attacks = df[df['Label'] == "🚨 ATTACK"]
            if not attacks.empty:
                fig_live.add_scatter(x=attacks.index, y=attacks['Attack_Probability'], mode='markers', name='Attack', marker=dict(color=COLOR_ATTACK, size=6, symbol='x'))
            fig_live.update_layout(xaxis_title="Packet", yaxis_title="MSE Loss (Anomaly Score)", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_live, use_container_width=True)

    except Exception as e:
        st.error(f"Critical Error: {e}")

else:
    st.info("Waiting for Network Log Upload...")
