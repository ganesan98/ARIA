# ARIA — Adversarially Robust IoT Authentication

A machine-learning pipeline that detects malicious IoT network traffic, stress-tests the detector using adversarial attacks, and hardens it using SHAP-guided feature selection + adversarial retraining.

---

## 📦 Requirements

```bash
pip install -r requirements.txt
```

Dependencies: `pandas`, `numpy`, `scikit-learn`, `shap`, `matplotlib`, `seaborn`, `requests`, `tqdm`, `joblib`

---

## 🚀 Run the Full Pipeline

```bash
python run_pipeline.py
```

Or run each step manually in order:

| Step | Script | What It Does |
|------|--------|-------------|
| 1 | `download_data.py` | Downloads CTU-IoT-23 Scenario 34 dataset |
| 2 | `preprocess.py` | Cleans data, splits train/test, scales features |
| 3 | `baseline_model.py` | Trains Random Forest baseline classifier |
| 4 | `attack.py` | Generates adversarial samples via FGSM-style attack |
| 5 | `shap_defense.py` | Uses SHAP to identify and select stable features |
| 6 | `adversarial_train.py` | Retrains model on clean + adversarial data |
| 7 | `visualize.py` | Generates comparison plots |

---

## 📂 Output

```
output/
├── results.json              ← All accuracy & attack success metrics
├── baseline_model.pkl        ← Trained baseline Random Forest
├── defended_model.pkl        ← ARIA hardened model
├── X_train/X_test/y_*.npy   ← Train/test splits
├── X_adv.npy                 ← Adversarial test samples
├── scaler.pkl                ← StandardScaler (for deployment)
└── plots/
    ├── confusion_matrices.png    ← Baseline vs ARIA under attack
    ├── accuracy_comparison.png   ← Before/after defense accuracy
    ├── shap_importance.png       ← Feature importance (SHAP)
    └── attack_success_rate.png   ← Attack success: undefended vs defended
```

---

## 🔬 Dataset

**CTU-IoT-23** (Stratosphere Lab, Czech Technical University)  
Scenario 34 — Real IoT malware capture (Mirai botnet + benign traffic)  
Source: https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/

---

## 🧠 How It Works

1. **Preprocessing** — 8 traffic features extracted from Zeek `conn.log`. Train/test split happens before scaling to prevent data leakage.
2. **Baseline** — Random Forest (100 trees) trained on clean traffic.
3. **Attack** — Feature-importance-weighted FGSM: malicious samples are perturbed toward the benign distribution with larger shifts on decision-critical features.
4. **SHAP Defense** — Features with below-average SHAP importance are dropped, reducing the attack surface.
5. **Adversarial Training** — Model retrained on 50% clean + 50% adversarially perturbed data, teaching it to handle both.

---

## 📈 Results

| Metric | Baseline | ARIA Defended |
|--------|----------|---------------|
| Clean Accuracy | 99.72% | 99.72% |
| Adversarial Accuracy | < baseline | 99.72% |
| Attack Success Rate | > 0% | 0.00% |
