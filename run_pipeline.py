"""
ARIA — Adversarially Robust IoT Authentication
run_pipeline.py: Run the full pipeline end-to-end in one command.

Usage:
    python run_pipeline.py
"""

import subprocess
import sys
import time

STEPS = [
    ("📥 Downloading dataset",          "download_data.py"),
    ("🔧 Preprocessing & splitting",    "preprocess.py"),
    ("🌲 Training baseline model",      "baseline_model.py"),
    ("⚔️  Running FGSM attack",          "attack.py"),
    ("🔍 SHAP feature selection",       "shap_defense.py"),
    ("🛡️  Adversarial retraining",       "adversarial_train.py"),
    ("📊 Generating visualizations",    "visualize.py"),
]

def run_step(label, script):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Script: {script}")
    print(f"{'='*60}")
    start = time.time()
    result = subprocess.run([sys.executable, script])
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\n❌ FAILED: {script} exited with code {result.returncode}")
        sys.exit(result.returncode)
    print(f"\n✅ Done in {elapsed:.1f}s")

if __name__ == "__main__":
    total_start = time.time()
    print("\n🛡️  ARIA — Full Pipeline Starting")
    print(f"   Python: {sys.executable}")

    for label, script in STEPS:
        run_step(label, script)

    total = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  ✅ All steps complete in {total:.1f}s")
    print(f"  📂 Results → output/results.json")
    print(f"  🖼️  Plots   → output/plots/")
    print(f"{'='*60}\n")
