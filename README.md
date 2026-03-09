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
├── notebooks/
│   └── full_analysis.ipynb     # End-to-end walkthrough
├── requirements.txt
└── README.md
```

---

## Research Questions

1. **Do QML models exhibit less, equal, or greater demographic bias than classical models on COMPAS?**
2. **Which component of QML (encoding, ansatz depth, measurement) contributes most to observed bias?**
3. **Do standard bias mitigation techniques (reweighting, post-processing) transfer from classical to quantum settings?**

---

## Fairness Metrics Computed

| Metric | Definition |
|--------|-----------|
| **Demographic Parity Difference (DPD)** | \|P(ŷ=1\|A=0) - P(ŷ=1\|A=1)\| |
| **Equalized Odds Difference (EOD)** | Max of TPR gap and FPR gap across groups |
| **Disparate Impact (DI)** | P(ŷ=1\|A=unprivileged) / P(ŷ=1\|A=privileged) |
| **Predictive Parity Difference (PPD)** | \|PPV_group0 - PPV_group1\| |
| **Individual Fairness Score** | Lipschitz consistency across similar individuals |

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

## Setup

```bash
git clone https://github.com/<your-org>/qml-bias.git
cd qml-bias
pip install -r requirements.txt
```

**Download COMPAS data:**
```bash
python data/load_compas.py  # fetches from ProPublica GitHub automatically
```

**Run full experiment pipeline:**
```bash
python experiments/run_classical.py   # ~2 min
python experiments/run_quantum.py     # ~20-40 min depending on hardware
python experiments/ablation_encoding.py
```

Results are saved to `results/` as JSON + figures.

---

## Key Results (preliminary)

| Model | Accuracy | DPD (race) | EOD (race) | DI |
|-------|----------|------------|------------|-----|
| Logistic Regression | — | — | — | — |
| Random Forest | — | — | — | — |
| MLP | — | — | — | — |
| VQC-Angle | — | — | — | — |
| VQC-Amplitude | — | — | — | — |
| VQC-IQP | — | — | — | — |

*Table auto-populated after running experiments.*

---

## Dependencies

- `pennylane >= 0.38`
- `scikit-learn >= 1.4`
- `torch >= 2.2`
- `fairlearn >= 0.10`
- `pandas`, `numpy`, `matplotlib`, `seaborn`

---

## References

1. Heredge et al. (2024). *Bias in Quantum Machine Learning.* arXiv:2405.xxxxx
2. Larocca et al. (2022). *Diagnosing barren plateaus with tools from quantum optimal control.*
3. Angwin et al. (2016). *Machine Bias.* ProPublica.
4. Hardt et al. (2016). *Equality of Opportunity in Supervised Learning.* NeurIPS.
5. Cerezo et al. (2021). *Variational Quantum Algorithms.* Nature Reviews Physics.
