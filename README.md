# Deep-NIDS: AI-Powered Network Intrusion Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA_Enabled-red.svg)](https://pytorch.org/)

## 💡 TL;DR
Deep-NIDS is a deep learning-based Network Intrusion Detection System architected for localized, real-time packet analysis and threat detection. By shifting inference to the edge and leveraging hardware acceleration, this model identifies anomalous network traffic patterns and malicious signatures without the latency overhead of cloud-based round trips.

## ⚡ Architecture & Performance
The system is built to handle the rigorous demands of continuous live traffic analysis, ensuring the intrusion detection mechanism does not become a network bottleneck.

* **Real-Time Inference Engine:** The core execution layers are built on the **PyTorch** framework and aggressively optimized with **CUDA** acceleration. This minimizes pipeline inference latency, allowing the model to score packets at line rate.
* **Optimized Data Pipeline:** Raw network traffic is messy. This project utilizes highly structured preprocessing components built with **NumPy** and **Pandas** workflows to clean, extract, and format real-time network features into normalized tensors efficiently.
* **Localized Deployment:** Designed for immediate threat detection, keeping packet analysis localized to the host network to ensure strict data privacy and rapid response times.

## 📂 Repository Structure
* `/src`: Core PyTorch neural network architectures, inference engines, and live traffic evaluation scripts.
* `/data`: Sanitized sample network datasets and feature layouts for model evaluation.
* `/scripts`: Data preprocessing pipelines (NumPy/Pandas) for cleaning and vectorizing raw network features.

## ⚙️ System Requirements & Reproduction
To reliably run the real-time inference pipeline without latency degradation, a CUDA-enabled environment is highly recommended.

* **GPU Acceleration:** NVIDIA GPU with CUDA support.
* **Backend:** PyTorch, NumPy, Pandas.
* **Environment:** Python 3.10+

### Build Instructions
1. Clone this repository: `git clone https://github.com/meowlawat/Deep-NIDS.git`
2. Create a virtual environment and install dependencies: `pip install -r requirements.txt`
3. Run the feature extraction pipeline to format your network data: `python scripts/preprocess_traffic.py`
4. Execute the intrusion detection inference engine: `python src/detect.py`
