import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import json

# The split now lives in preprocess.py so the scaler was fitted correctly.
# baseline_model.py no longer splits — it just trains and evaluates.
X_train = np.load("output/X_train.npy")
X_test  = np.load("output/X_test.npy")
y_train = np.load("output/y_train.npy")
y_test  = np.load("output/y_test.npy")

print("Training Random Forest baseline...")
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nBaseline Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Benign", "Malicious"]))

# Save model
joblib.dump(clf, "output/baseline_model.pkl")

# Save accuracy for comparison later
results = {"baseline_clean_accuracy": round(acc, 4)}
with open("output/results.json", "w") as f:
    json.dump(results, f)

print("\nModel saved to output/baseline_model.pkl")