import numpy as np
from sklearn.metrics import accuracy_score
import joblib
import json

X_test = np.load("output/X_test.npy")
y_test = np.load("output/y_test.npy")
clf    = joblib.load("output/baseline_model.pkl")

# Improved approach: use Random Forest's feature_importances_ as a gradient
# proxy.  Features with higher importance get a larger perturbation — analogous
# to FGSM's gradient-sign weighting.  Perturbation is still subtracted (pushing
# malicious traffic toward the benign distribution), but now it is proportional
# to how much each feature contributes to the "Malicious" decision.

def fgsm_attack(X, clf, epsilon=0.2):
    """
    Improved FGSM-style feature-level perturbation for tabular data.

    Since Random Forests have no gradients, we are using feature importances
    as a proxy for the gradient magnitude.  Each feature is perturbed by
    epsilon * (importance / max_importance), so the most decision-relevant
    features receive the full epsilon shift while less-important ones are
    perturbed proportionally less.

    Direction: subtract (push malicious sample toward benign distribution).
    Constraint: clip standardized values to a floor of -5.0 to prevent
                physically impossible values (e.g., negative packet counts).
    """
    importances = clf.feature_importances_
    # Normalize so the most important feature receives exactly epsilon
    weights = importances / (importances.max() + 1e-8)
    perturbation = epsilon * weights          # shape: (n_features,)

    X_adv = X.copy()
    for i in range(len(X)):
        if y_test[i] == 1:                   # only attack malicious samples
            X_adv[i] = X[i] - perturbation   # nudge toward benign distribution

            # Features like packet counts and byte counts cannot be negative.
            # After StandardScaler, the practical lower bound is ~-5 std devs.
            X_adv[i] = np.clip(X_adv[i], -5.0, None)

    return X_adv

print("Running improved FGSM attack (epsilon=0.2, importance-weighted)...")
X_adv = fgsm_attack(X_test, clf, epsilon=0.2)

y_pred_clean = clf.predict(X_test)
y_pred_adv   = clf.predict(X_adv)

clean_acc = accuracy_score(y_test, y_pred_clean)
adv_acc   = accuracy_score(y_test, y_pred_adv)

# Attack success rate = % of malicious samples now predicted as benign
mal_idx      = y_test == 1
attack_success = np.mean(y_pred_adv[mal_idx] == 0)

print(f"Clean Accuracy:       {clean_acc:.4f}")
print(f"Adversarial Accuracy: {adv_acc:.4f}")
print(f"Attack Success Rate:  {attack_success:.4f}")
print(f"Accuracy Drop:        {clean_acc - adv_acc:.4f}")

# Update results
with open("output/results.json") as f:
    results = json.load(f)

results["adversarial_accuracy_undefended"]  = round(adv_acc, 4)
results["attack_success_rate_undefended"]   = round(attack_success, 4)

with open("output/results.json", "w") as f:
    json.dump(results, f, indent=2)

np.save("output/X_adv.npy", X_adv)
print("\nAdversarial samples saved to output/X_adv.npy")