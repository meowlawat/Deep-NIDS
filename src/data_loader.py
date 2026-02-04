import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Define standard NSL-KDD columns
COL_NAMES = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"]

def load_and_preprocess(train_path, test_path):
    print("[*] Loading Raw Data...")
    df_train = pd.read_csv(train_path, names=COL_NAMES)
    df_test = pd.read_csv(test_path, names=COL_NAMES)
    
    # Drop categorical/unnecessary columns
    drop_cols = ['difficulty', 'protocol_type', 'service', 'flag']
    df_train = df_train.drop(drop_cols, axis=1)
    df_test = df_test.drop(drop_cols, axis=1)
    
    # Encode Label (Normal=0, Attack=1)
    df_train['label'] = df_train['label'].apply(lambda x: 0 if x == 'normal' else 1)
    df_test['label'] = df_test['label'].apply(lambda x: 0 if x == 'normal' else 1)
    
    # Split Features/Labels
    x_train = df_train.drop('label', axis=1)
    y_train = df_train['label']
    
    # Fit Scaler
    print("[*] Scaling Data...")
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    
    # Reshape for LSTM (Samples, Timesteps, Features)
    x_train_lstm = x_train_scaled.reshape((x_train_scaled.shape[0], 1, x_train_scaled.shape[1]))
    
    # Filter Normal Traffic for Training
    x_train_normal = x_train_lstm[y_train == 0]
    
    return x_train_normal, scaler, x_train.shape[1]