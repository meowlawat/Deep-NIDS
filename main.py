import numpy as np
import joblib
from src.data_loader import load_and_preprocess
from src.model_builder import build_lstm_autoencoder

# Config
TRAIN_FILE = "data/KDDTrain+.txt"
TEST_FILE = "data/KDDTest+.txt"
MODEL_PATH = "models/nids_model.keras"
SCALER_PATH = "models/nids_scaler.pkl"
THRESH_PATH = "models/nids_threshold.pkl"

def train_system():
    # 1. Get Data
    X_normal, scaler, input_dim = load_and_preprocess(TRAIN_FILE, TEST_FILE)
    
    # 2. Get Model
    model = build_lstm_autoencoder(input_dim)
    
    # 3. Train
    print("[*] Training (This may take a moment)...")
    model.fit(X_normal, X_normal, epochs=5, batch_size=32, verbose=1)
    
    # 4. Calculate Threshold
    print("[*] Calculating Threshold...")
    reconstructions = model.predict(X_normal)
    mse = np.mean(np.power(X_normal - reconstructions, 2), axis=2).flatten()
    threshold = np.percentile(mse, 95) # 95th percentile
    
    # 5. Save Everything
    print(f"[SUCCESS] System Trained. Threshold: {threshold:.5f}")
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(threshold, THRESH_PATH)
    print(f"[*] Artifacts saved to /models/")

if __name__ == "__main__":
    train_system()