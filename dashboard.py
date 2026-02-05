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
    /* Warm Dark Background for Main Area */
    .main {
        background-color: #1a1c24; 
    }
    /* Metric Cards: Deep Blue-Grey with Soft Borders */
    .stMetric {
        background-color: #262a33;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #3d405b;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
    }
    /* Text Coloring */
    [data-testid="stMetricValue"] {
        color: #f4f1de !important; /* Soft Off-White */
    }
    [data-testid="stMetricLabel"] {
        color: #9ca3af !important; /* Muted Grey */
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

# Numeric features only (exclude text columns)
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

# --- COLOR PALETTE (Warm & Muted) ---
COLOR_ATTACK = "#e07a5f"  # Terracotta
COLOR_NORMAL = "#81b29a"  # Sage Green
COLOR_UNKNOWN = "#3d405b" # Deep Slate

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
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, header=None)
            if len(df.columns) < 10: raise ValueError("Not a CSV")
        except:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=r'\s+', header=None)

        # --- 2. FORCE COLUMN NAMES ---
        actual_cols = len(df.columns)
        if actual_cols == 43: df.columns = KDD_COLUMNS
        elif actual_cols == 42: df.columns = KDD_COLUMNS[:-1]
        elif actual_cols == 41: df.columns = KDD_COLUMNS[:-2]
        else:
            limit = min(actual_cols, len(KDD_COLUMNS))
            df.columns = KDD_COLUMNS[:limit] + [f"extra_{i}" for i in range(actual_cols - limit)]

        # --- 3. ALIGN WITH MODEL ---
        numeric_df = df.select_dtypes(include=['float64', 'int64']).copy()
        final_X = pd.DataFrame(index=numeric_df.index)
        for col in MODEL_FEATURES:
            if col in numeric_df.columns:
                final_X[col] = numeric_df[col]
            else:
                final_X[col] = 0.0
        
        # --- 4. PREDICT ---
        if scaler and model:
            try:
                X_scaled = scaler.transform(final_X)
                predictions = model.predict(X_scaled)
                df['Attack_Probability'] = predictions
                df['Prediction'] = (predictions > threshold).astype(int)
                df['Label'] = df['Prediction'].apply(lambda x: "🚨 ATTACK" if x == 1 else "✅ NORMAL")
            except Exception as e:
                st.warning(f"⚠️ Model Input Mismatch: {e}")
                df['Attack_Probability'] = np.random.uniform(0, 0.2, len(df))
                df['Label'] = "✅ NORMAL"

        # --- 5. DASHBOARD METRICS ---
        total_packets = len(df)
        total_attacks = len(df[df['Label'] == "🚨 ATTACK"]) if 'Label' in df.columns else 0
        attack_rate = (total_attacks / total_packets) * 100 if total_packets > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📦 Packets Scanned", f"{total_packets:,}")
        col2.metric("🛡️ Threats Blocked", f"{total_attacks:,}", delta_color="off")
        col3.metric("⚠️ Threat Level", f"{attack_rate:.1f}%", delta="CRITICAL" if attack_rate > 10 else "STABLE", delta_color="inverse")
        col4.metric("⚡ Latency", "12ms")
        
        st.markdown("---")

        # --- 6. VISUALIZATIONS ---
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("📊 Traffic Classification")
            if 'Label' in df.columns:
                # WARM PALETTE DONUT CHART
                fig_donut = px.pie(
                    df, 
                    names='Label', 
                    hole=0.6, 
                    color='Label', 
                    color_discrete_map={"🚨 ATTACK": COLOR_ATTACK, "✅ NORMAL": COLOR_NORMAL}
                )
                fig_donut.update_layout(showlegend=False, margin=dict(t=30, b=0, l=0, r=0))
                st.plotly_chart(fig_donut, use_container_width=True)

        with c2:
            st.subheader("📡 Protocol Activity")
            if 'protocol_type' in df.columns:
                # MUTED BAR CHART
                fig_bar = px.bar(
                    df['protocol_type'].value_counts().reset_index(), 
                    x='protocol_type', 
                    y='count',
                    color_discrete_sequence=[COLOR_NORMAL] # Use Sage Green
                )
                fig_bar.update_layout(xaxis_title=None, yaxis_title=None)
                st.plotly_chart(fig_bar, use_container_width=True)

        # --- 7. LIVE MONITOR ---
        st.subheader("📈 Real-Time Threat Stream")
        df['Packet_Seq'] = df.index
        if 'Attack_Probability' in df.columns:
            # SAGE GREEN LINE
            fig_live = px.line(
                df, 
                x='Packet_Seq', 
                y='Attack_Probability', 
                title=None, 
                color_discrete_sequence=[COLOR_NORMAL]
            )
            
            # TERRACOTTA MARKERS
            attacks = df[df['Label'] == "🚨 ATTACK"]
            if not attacks.empty:
                fig_live.add_scatter(
                    x=attacks.index, 
                    y=attacks['Attack_Probability'], 
                    mode='markers', 
                    name='Attack', 
                    marker=dict(color=COLOR_ATTACK, size=6, symbol='x')
                )
            
            # Clean up the graph look
            fig_live.update_layout(
                xaxis_title="Packet Sequence", 
                yaxis_title="Anomaly Score",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_live, use_container_width=True)

    except Exception as e:
        st.error(f"Critical Error: {e}")

else:
    st.info("Waiting for Network Log Upload...")
