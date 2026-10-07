import numpy as np
import shap
import joblib

X_train = np.load("output/X_train.npy")
X_test = np.load("output/X_test.npy")
clf = joblib.load("output/baseline_model.pkl")
feature_cols = joblib.load("output/feature_cols.pkl")

print("Computing SHAP values (this takes ~1-2 mins)...")

# TreeExplainer is optimized for Random Forests
explainer = shap.TreeExplainer(clf)

# Use a sample for speed — 500 is enough to get stable feature importance
sample = X_train[:500]
shap_values = explainer.shap_values(sample)

# shap_values is a list [class_0_shap, class_1_shap]
# We care about class 1 (Malicious) importance
shap_class1 = np.abs(shap_values[1])
mean_shap = shap_class1.mean(axis=0)

print("\nFeature Importance (SHAP):")
for name, val in sorted(zip(feature_cols, mean_shap), key=lambda x: -x[1]):
    print(f"  {name:20s}: {val:.4f}")

# Select top-N most important AND most stable features
# We keep features with above-average SHAP importance
threshold = mean_shap.mean()
selected_idx = [i for i, v in enumerate(mean_shap) if v >= threshold]
selected_features = [feature_cols[i] for i in selected_idx]

print(f"\nSelected {len(selected_features)} stable features: {selected_features}")

joblib.dump(selected_idx, "output/selected_feature_idx.pkl")
joblib.dump(selected_features, "output/selected_features.pkl")
joblib.dump(shap_values, "output/shap_values.pkl")
joblib.dump(mean_shap, "output/mean_shap.pkl")

print("SHAP analysis saved.")