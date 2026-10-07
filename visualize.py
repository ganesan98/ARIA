import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib
import json
import os

os.makedirs("output/plots", exist_ok=True)

# Load everything
X_test          = np.load("output/X_test.npy")
X_adv           = np.load("output/X_adv.npy")
y_test          = np.load("output/y_test.npy")
clf             = joblib.load("output/baseline_model.pkl")
clf_def         = joblib.load("output/defended_model.pkl")
selected_idx    = joblib.load("output/selected_feature_idx.pkl")
selected_features = joblib.load("output/selected_features.pkl")
mean_shap       = joblib.load("output/mean_shap.pkl")
feature_cols    = joblib.load("output/feature_cols.pkl")

with open("output/results.json") as f:
    results = json.load(f)

X_test_sel = X_test[:, selected_idx]
X_adv_sel  = X_adv[:, selected_idx]

# ── Plot 1: Confusion Matrices ────────────────────────────────────────────────
# ── FIX #3: Fair apples-to-apples comparison ─────────────────────────────────
# Original code compared Baseline(clean) vs ARIA(adversarial) — different
# conditions.  We now show BOTH models under the SAME adversarial condition
# so the viewer sees the true benefit of the defense.
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("ARIA — Confusion Matrices Under Adversarial Attack", fontsize=14, fontweight='bold')

comparisons = [
    (clf,     X_adv,    "Baseline (Under Attack — Undefended)"),
    (clf_def, X_adv_sel, "ARIA Defended (Under Attack)"),
]

for ax, (model, X_in, title) in zip(axes, comparisons):
    cm = confusion_matrix(y_test, model.predict(X_in))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=["Benign", "Malicious"],
                yticklabels=["Benign", "Malicious"])
    ax.set_title(title)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")

plt.tight_layout()
plt.savefig("output/plots/confusion_matrices.png", dpi=150)
plt.close()
print("Saved: confusion_matrices.png")

# ── Plot 2: Accuracy Comparison Bar Chart ────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
categories = ["Baseline\n(Clean)", "Baseline\n(Adversarial)", "ARIA\n(Clean)", "ARIA\n(Adversarial)"]
values = [
    results["baseline_clean_accuracy"],
    results["adversarial_accuracy_undefended"],
    results["defended_clean_accuracy"],
    results["defended_adversarial_accuracy"]
]
colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800']
bars = ax.bar(categories, values, color=colors, width=0.5, edgecolor='black', linewidth=0.5)

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.2%}', ha='center', va='bottom', fontweight='bold')

ax.set_ylim(0, 1.1)
ax.set_ylabel("Accuracy")
ax.set_title("ARIA — Accuracy Before vs After Defense", fontweight='bold')
ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='90% threshold')
ax.legend()
plt.tight_layout()
plt.savefig("output/plots/accuracy_comparison.png", dpi=150)
plt.close()
print("Saved: accuracy_comparison.png")

# ── Plot 3: SHAP Feature Importance ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
sorted_pairs = sorted(zip(feature_cols, mean_shap), key=lambda x: x[1])
names, vals = zip(*sorted_pairs)
colors = ['#FF7043' if n in selected_features else '#90CAF9' for n in names]
ax.barh(names, vals, color=colors, edgecolor='black', linewidth=0.5)
ax.set_xlabel("Mean |SHAP Value|")
ax.set_title("SHAP Feature Importance\n(Orange = selected for defense)", fontweight='bold')
plt.tight_layout()
plt.savefig("output/plots/shap_importance.png", dpi=150)
plt.close()
print("Saved: shap_importance.png")

# ── Plot 4: Attack Success Rate Comparison ────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
labels = ["Undefended", "ARIA Defended"]
asr    = [results["attack_success_rate_undefended"], results["attack_success_rate_defended"]]
bars   = ax.bar(labels, asr, color=['#F44336', '#4CAF50'], width=0.4, edgecolor='black')
for bar, val in zip(bars, asr):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.2%}', ha='center', fontweight='bold')
ax.set_ylim(0, 1.1)
ax.set_ylabel("Attack Success Rate")
ax.set_title("Attack Success Rate: Before vs After ARIA", fontweight='bold')
plt.tight_layout()
plt.savefig("output/plots/attack_success_rate.png", dpi=150)
plt.close()
print("Saved: attack_success_rate.png")

print("\nAll plots saved to output/plots/")
print("Use these as your output screenshots in the Word document.")