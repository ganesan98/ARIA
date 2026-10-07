import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json

X_train    = np.load("output/X_train.npy")
y_train    = np.load("output/y_train.npy")
X_test     = np.load("output/X_test.npy")
X_adv      = np.load("output/X_adv.npy")
y_test     = np.load("output/y_test.npy")
selected_idx = joblib.load("output/selected_feature_idx.pkl")

# Apply SHAP feature selection — keep only stable features
X_train_sel = X_train[:, selected_idx]
X_test_sel  = X_test[:, selected_idx]
X_adv_sel   = X_adv[:, selected_idx]

# Generate adversarial training samples from training set too
def fgsm_attack_train(X, y, importances, epsilon=0.2):
    """Same importance-weighted FGSM used in attack.py — must match."""
    weights      = importances / (importances.max() + 1e-8)
    perturbation = epsilon * weights
    X_adv = X.copy()
    for i in range(len(X)):
        if y[i] == 1:
            X_adv[i] = np.clip(X[i] - perturbation, -5.0, None)
    return X_adv

# Load the baseline model's importances (sliced to selected features)
clf_base      = joblib.load("output/baseline_model.pkl")
imp_full      = clf_base.feature_importances_
imp_selected  = imp_full[selected_idx]

X_train_adv = fgsm_attack_train(X_train_sel, y_train, imp_selected, epsilon=0.2)

# Mix clean + adversarial training data (50/50)
X_combined = np.vstack([X_train_sel, X_train_adv])
y_combined = np.concatenate([y_train, y_train])

print("Training ARIA defended model...")
clf_defended = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf_defended.fit(X_combined, y_combined)

# ── Evaluate on clean test data ───────────────────────────────────────────────
y_pred_clean = clf_defended.predict(X_test_sel)
clean_acc    = accuracy_score(y_test, y_pred_clean)

# ── Evaluate on adversarial test data ────────────────────────────────────────
y_pred_adv   = clf_defended.predict(X_adv_sel)
adv_acc      = accuracy_score(y_test, y_pred_adv)

# Attack success rate
mal_idx      = y_test == 1
attack_success = np.mean(y_pred_adv[mal_idx] == 0)

print(f"\nARIA Defended Model:")
print(f"Clean Accuracy:       {clean_acc:.4f}")
print(f"Adversarial Accuracy: {adv_acc:.4f}")
print(f"Attack Success Rate:  {attack_success:.4f}")

# In IoT security, False Negatives (missed malicious devices) are critical.
# Accuracy alone doesn't reveal precision/recall/F1 per class.
print("\nClassification Report — Defended Model (Clean Traffic):")
print(classification_report(y_test, y_pred_clean, target_names=["Benign", "Malicious"]))

print("Classification Report — Defended Model (Adversarial Traffic):")
print(classification_report(y_test, y_pred_adv, target_names=["Benign", "Malicious"]))

joblib.dump(clf_defended, "output/defended_model.pkl")

# Final results comparison
with open("output/results.json") as f:
    results = json.load(f)

results["defended_clean_accuracy"]        = round(clean_acc, 4)
results["defended_adversarial_accuracy"]  = round(adv_acc, 4)
results["attack_success_rate_defended"]   = round(attack_success, 4)

with open("output/results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nFull results saved to output/results.json")
print(json.dumps(results, indent=2))