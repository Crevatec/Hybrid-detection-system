# Hybrid Behavioral Detection System v1

## Real-Time Mitigation of Business Logic & Credential Stuffing Attacks in APIs

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

This project implements a hybrid machine learning system for real-time detection and classification of API attacks. It combines four individual models and four hybrid ensemble configurations to detect:

- **Normal** traffic (Class 0)
- **Credential Stuffing** attacks (Class 1) — brute force, SSH-Patator
- **Business Logic** attacks (Class 2) — DDoS, XSS, SQLi, Bots, DoS

---

## Dataset

- **Name:** `hybrid_master_dataset.csv`
- **Size:** ~3.2 million rows
- **Source:** Combined multi-source network traffic dataset
- **Class Distribution:**
  - Normal: 82.9%
  - Credential Stuffing: 0.4%
  - Business Logic: 16.7%
- **Challenge:** Heavily imbalanced — class weights applied during training

### Label Mapping

| Raw Label | Mapped Class |
|---|---|
| Normal | Normal (0) |
| Web Attack – Brute Force, SSH-Patator | Credential Stuffing (1) |
| Web Attack – XSS, SQL Injection, Bot, DDoS, DoS | Business Logic Attack (2) |
| All error labels (DB failure, latency, etc.) | Normal (0) |

---

## Models Implemented

### Individual Models (Phase 2)

| Model | Description |
|---|---|
| Random Forest (RF) | Supervised ensemble, structured feature classification |
| Isolation Forest (IF) | Unsupervised anomaly detection |
| Artificial Neural Network (ANN) | Dense network, non-linear pattern recognition |
| Long Short-Term Memory (LSTM) | Sequential behavioural time-series analysis |

### Hybrid Ensemble Models (Phase 3)

| Model | Strategy |
|---|---|
| RF + ANN | Stacking — RF probabilities fed into ANN meta-learner |
| IF + ANN | Anomaly score augmentation — IF score appended as ANN feature |
| RF + LSTM | Parallel weighted voting (RF=45%, LSTM=55%) |
| ANN + LSTM + RF (Master) | Triple weighted ensemble (RF=30%, ANN=35%, LSTM=35%) |

---

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Random Forest | 0.8792 | 0.9253 | 0.8792 | 0.8899 |
| Isolation Forest | 0.6961 | 0.6823 | 0.6961 | 0.6891 |
| ANN | 0.7867 | 0.7148 | 0.7867 | 0.7442 |
| LSTM | 0.4914 | 0.7415 | 0.4914 | 0.5843 |
| RF + ANN | 0.8792 | 0.9251 | 0.8792 | 0.8899 |
| IF + ANN | 0.7760 | 0.8908 | 0.7760 | 0.8031 |
| RF + LSTM | 0.8757 | 0.9250 | 0.8757 | 0.8870 |
| **ANN + LSTM + RF (Master)** | **0.8368** | **0.9167** | **0.8368** | **0.8598** |

> Individual ANN and LSTM models underperformed due to severe class imbalance (0.4% Credential Stuffing). Hybrid combinations compensate by combining RF's structured classification strength with neural network flexibility, producing superior F1 scores.

---

## Project Structure

```
hybrid_detection_system/
│
├── README.md                          # This file
├── requirements.txt                   # All Python dependencies
│
├── data/
│   └── (place hybrid_master_dataset.csv here)
│
├── utils/
│   ├── __init__.py
│   ├── data_loader.py                 # Data loading & preprocessing
│   ├── feature_engineering.py        # Feature extraction & encoding
│   └── metrics.py                    # Evaluation utilities
│
├── models/
│   ├── __init__.py
│   ├── random_forest_model.py        # Phase 2: RF standalone
│   ├── isolation_forest_model.py     # Phase 2: IF standalone
│   ├── ann_model.py                  # Phase 2: ANN standalone
│   ├── lstm_model.py                 # Phase 2: LSTM standalone
│   ├── hybrid_rf_ann.py              # Phase 3: RF + ANN
│   ├── hybrid_if_ann.py              # Phase 3: IF + ANN
│   ├── hybrid_rf_lstm.py             # Phase 3: RF + LSTM
│   └── hybrid_master.py             # Phase 3: ANN + LSTM + RF (Master)
│
├── outputs/
│   ├── metrics/                      # Saved metrics JSON files
│   ├── plots/                        # Saved comparison charts
│   └── saved_models/                 # Persisted trained models
│
├── dashboard/
│   ├── __init__.py
│   ├── realtime_simulator.py         # Real-time login stream simulation
│   └── dashboard_app.py              # Streamlit monitoring dashboard
│
├── train_individual.py               # Run Phase 2: train all individual models
├── train_hybrid.py                   # Run Phase 3: train all hybrid models
├── evaluate_all.py                   # Run Phase 4: metrics + charts
└── run_dashboard.py                  # Run Phase 5: launch dashboard
```

---

## Installation & Usage

### 1. Create Virtual Environment (Python 3.11 required)

```bash
py -3.11 -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Dataset

Place `hybrid_master_dataset.csv` into the `data/` folder.

### 4. Run the System (in order)

```bash
# Train all individual models (~20 mins)
python train_individual.py

# Train all hybrid/ensemble models (~35 mins)
python train_hybrid.py

# Generate metrics and comparison charts
python evaluate_all.py

# Launch real-time monitoring dashboard
streamlit run run_dashboard.py
```

---

## Dashboard Features

- Live login attempt feed with colour-coded threat classification
- Real-time attack rate gauge
- Threat distribution pie chart
- Attack timeline bar chart
- All 8 model metrics comparison chart
- Master Hybrid (ANN+LSTM+RF) confidence breakdown panel

---

## Tech Stack

| Library | Version |
|---|---|
| Python | 3.11 |
| TensorFlow / Keras | 2.15 |
| Scikit-learn | 1.4 |
| Pandas | 2.1 |
| NumPy | 1.26 |
| Streamlit | 1.31 |
| Plotly | 5.18 |
| Matplotlib | 3.8 |

---

## License

MIT License — free to use, modify, and distribute.