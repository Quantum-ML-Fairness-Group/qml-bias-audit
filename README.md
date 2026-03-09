# Fairness in Quantum Machine Learning
### IML Project — University of Illinois

> *Investigating whether quantum machine learning models exhibit, amplify, or mitigate bias compared to classical baselines — using the COMPAS recidivism dataset.*

**Faculty Mentor:** Theshani Gallage  
**Graduate Mentor:** Sujeet Bhalerao  
**Scholars:** Ryan Santosh, Giannis Kostellos, Stanley Cheung, Zihan Li

---

## Overview

This project benchmarks **Variational Quantum Classifiers (VQC)** against classical ML models (Logistic Regression, Random Forest, MLP) on the COMPAS recidivism dataset, a domain where algorithmic bias has real-world consequences. We compute a suite of group fairness metrics across race and sex subgroups, and trace the *sources* of QML bias to data encoding, circuit architecture, and measurement processes.

---

## Project Structure

```
qml-bias/
├── data/
│   ├── load_compas.py          # Dataset loading & preprocessing
│   └── bias_aware_split.py     # Stratified train/test splits preserving group balance
├── models/
│   ├── classical/
│   │   ├── logistic_regression.py
│   │   ├── random_forest.py
│   │   └── mlp.py
│   └── quantum/
│       ├── vqc_angle.py        # VQC with angle encoding
│       ├── vqc_amplitude.py    # VQC with amplitude encoding
│       └── vqc_iqp.py          # VQC with IQP encoding
├── experiments/
│   ├── run_classical.py        # Train & evaluate all classical models
│   ├── run_quantum.py          # Train & evaluate all quantum models
│   └── ablation_encoding.py    # Ablation: encoding scheme vs. bias
├── utils/
│   ├── fairness_metrics.py     # Demographic parity, equalized odds, etc.
│   ├── bias_attribution.py     # Shapley-based bias attribution
│   └── visualization.py        # All plotting utilities
├── results/                    # Auto-generated JSON + figures
├── requirements.txt
└── README.md
```

---

## Research Questions

1. **Do QML models exhibit less, equal, or greater demographic bias than classical models on COMPAS?**
2. **Which component of QML (encoding, ansatz depth, measurement) contributes most to observed bias?**
3. **Do standard bias mitigation techniques (reweighting, post-processing) transfer from classical to quantum settings?**

---


## Models

### Classical Baselines
- **Logistic Regression** — L2 regularized, calibrated
- **Random Forest** — 200 estimators, tuned via GridSearch
- **MLP** — 3-layer (128→64→32), dropout 0.3, Adam optimizer

### Quantum Models (PennyLane)
- **VQC-Angle** — Angle encoding + strongly entangling layers ansatz
- **VQC-Amplitude** — Amplitude encoding (full state initialization)
- **VQC-IQP** — IQP feature map (closer to quantum advantage regime)

All QVCs use the `default.qubit` simulator; swap to `qiskit.aer` for noise modeling.

---
