# 🛡️ Deep-NIDS: AI-Powered Network Intrusion Detection System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)

## 📌 Overview
**Deep-NIDS** is a modern Intrusion Detection System (IDS) that uses **Unsupervised Deep Learning (LSTM Autoencoders)** to detect network attacks. Unlike traditional firewalls that rely on "signatures" (known bad guys), this system learns the pattern of "Normal" traffic and flags **anything** that deviates from it.

This approach allows it to detect **Zero-Day Attacks** (new, unknown threats) that standard systems miss.

## 🚀 Features
-   **Unsupervised Learning:** Trained only on normal traffic; catches unknown anomalies.
-   **LSTM Autoencoder:** Analyzes traffic as a *sequence* (time-series), not just isolated packets.
-   **Real-Time Dashboard:** A Streamlit interface simulating live network monitoring.
-   **Red Alert Logic:** Automatically flags packets with high Reconstruction Error (MSE).

## 🛠️ Project Structure
```text
Deep-NIDS/
├── data/                   # Raw NSL-KDD dataset
├── models/                 # Saved LSTM model & Scalers
├── src/                    # Source code (Data Loader, Model Builder)
├── dashboard.py            # Streamlit Dashboard (The UI)
├── main.py                 # Training Script
└── requirements.txt        # Dependencies