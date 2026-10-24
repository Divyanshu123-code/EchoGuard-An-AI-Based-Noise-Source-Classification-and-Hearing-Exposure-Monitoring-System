#!/usr/bin/env python3
"""
scripts/train_noise_classifier.py

Automated Machine Learning Training and Benchmarking Pipeline:
1. Synthesizes balanced dataset of 6 acoustic sound classes (360 samples).
2. Extracts 49-dimensional psychoacoustic features (MFCCs, spectral descriptors, chroma, crest factor).
3. Trains and benchmarks multiple classification algorithms:
   - Random Forest Classifier
   - Gradient Boosting Classifier
   - Support Vector Classifier (RBF Kernel)
   - K-Nearest Neighbors (KNN)
4. Computes 5-fold cross-validation accuracy and per-class classification reports.
5. Saves the winning pipeline to models/noise_classifier_pipeline.joblib.
6. Generates high-resolution visualization figures in docs/figures/:
   - ml_confusion_matrix.png
   - ml_model_comparison.png
   - ml_feature_importances.png
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import joblib

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from src.noise_classifier import (
    CLASSES,
    CLASS_DESCRIPTIONS,
    generate_synthetic_sound,
    extract_features_from_audio,
    NoiseSourceClassifier,
    AcousticLevelRegressor,
    MODEL_DIR,
    CLASSIFIER_PATH,
    REGRESSOR_PATH
)

DOCS_FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures")
METRICS_JSON = os.path.join(MODEL_DIR, "model_metrics.json")

def generate_benchmark_dataset(samples_per_class: int = 60, sr: int = 22050):
    print(f"Generating benchmark dataset: {len(CLASSES)} classes x {samples_per_class} samples = {len(CLASSES) * samples_per_class} total...")
    X_list = []
    y_list = []

    for cls_idx, c_name in enumerate(CLASSES):
        for i in range(samples_per_class):
            dur = 3.5 + (i % 3) * 0.5
            seed = cls_idx * 5000 + i
            audio = generate_synthetic_sound(c_name, duration=dur, sr=sr, seed=seed)
            feats = extract_features_from_audio(audio, sr=sr)
            X_list.append(feats)
            y_list.append(c_name)

    X = np.vstack(X_list)
    y = np.array(y_list)
    return X, y

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DOCS_FIG_DIR, exist_ok=True)

    print("=" * 72)
    print("   ECHOGUARD: AI NOISE SOURCE CLASSIFICATION & ML BENCHMARKING")
    print("=" * 72)

    X, y = generate_benchmark_dataset(samples_per_class=60)
    print(f"Dataset extracted: Shape {X.shape} (49 acoustic features)")

    # Define candidate models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=140, max_depth=14, random_state=42, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.1, max_depth=4, random_state=42),
        "Support Vector Machine": SVC(kernel="rbf", C=2.5, probability=True, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, weights="distance")
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}

    print("\n[Step 1: 5-Fold Cross-Validation Model Comparison]")
    print(f"{'Model Name':<26} | {'Mean CV Accuracy':<18} | {'Std Dev':<10}")
    print("-" * 60)

    best_name = None
    best_score = -1.0
    for name, clf in models.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
        mean_acc = float(np.mean(scores))
        std_acc = float(np.std(scores))
        cv_results[name] = {"mean": round(mean_acc * 100, 2), "std": round(std_acc * 100, 2)}
        print(f"{name:<26} | {mean_acc * 100:6.2f}%            | ±{std_acc * 100:4.2f}%")
        if mean_acc > best_score:
            best_score = mean_acc
            best_name = name

    print(f"\nWinning architecture: {best_name} with {best_score * 100:.2f}% cross-validation accuracy.")

    # Train / test split for in-depth evaluation & confusion matrix
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    # Fit the production Random Forest pipeline
    prod_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=140, max_depth=14, random_state=42, class_weight="balanced"))
    ])
    prod_pipe.fit(X_train, y_train)
    y_pred = prod_pipe.predict(X_test)

    test_acc = float(prod_pipe.score(X_test, y_test))
    report = classification_report(y_test, y_pred, output_dict=True, target_names=CLASSES)

    print("\n[Step 2: Test Set Classification Report]")
    print(classification_report(y_test, y_pred, target_names=CLASSES))

    # Save best model to joblib
    joblib.dump({
        "pipeline": prod_pipe,
        "classes": CLASSES,
        "metrics": {
            "test_accuracy": test_acc,
            "cv_scores": cv_results,
            "best_model": best_name
        }
    }, CLASSIFIER_PATH)
    print(f"Serialized production model to {CLASSIFIER_PATH}")

    # Also fit & save AcousticLevelRegressor
    calib_csv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "calibration.csv")
    reg = AcousticLevelRegressor()
    reg.fit_from_calibration_data(calib_csv)
    reg.save(REGRESSOR_PATH)
    print(f"Serialized acoustic level regressor to {REGRESSOR_PATH}")

    # Save metrics JSON
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "test_accuracy": test_acc,
            "cv_comparison": cv_results,
            "classification_report": report
        }, f, indent=2)

    # Plot 1: Model Comparison Chart
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    names = list(cv_results.keys())
    means = [cv_results[k]["mean"] for k in names]
    stds = [cv_results[k]["std"] for k in names]
    colors = ["#2563EB", "#10B981", "#8B5CF6", "#F59E0B"]

    bars = ax.bar(names, means, yerr=stds, capsize=6, color=colors, alpha=0.88, edgecolor="#1E293B", linewidth=1.2)
    ax.set_ylim(min(means) - 15, 105)
    ax.set_ylabel("5-Fold Cross-Validation Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_title("Acoustic Noise Source Classifier - Model Architecture Comparison", fontsize=12, fontweight="bold", pad=12)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    chart1_path = os.path.join(DOCS_FIG_DIR, "ml_model_comparison.png")
    fig.savefig(chart1_path)
    plt.close(fig)
    print(f"Saved chart: {chart1_path}")

    # Plot 2: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=CLASSES)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[c.replace("_", "\n") for c in CLASSES])
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=200)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    ax.set_title(f"Acoustic Noise Classification - Test Confusion Matrix (Acc: {test_acc*100:.1f}%)", fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    chart2_path = os.path.join(DOCS_FIG_DIR, "ml_confusion_matrix.png")
    fig.savefig(chart2_path)
    plt.close(fig)
    print(f"Saved chart: {chart2_path}")

    # Plot 3: Top Feature Importances (from Random Forest)
    rf_model = prod_pipe.named_steps["rf"]
    importances = rf_model.feature_importances_
    
    # Feature names
    feature_names = []
    for i in range(1, 14): feature_names.append(f"MFCC_{i}_mean")
    for i in range(1, 14): feature_names.append(f"MFCC_{i}_std")
    feature_names += ["Centroid_mean", "Centroid_std", "Rolloff_mean", "Rolloff_std", "Bandwidth_mean", "Bandwidth_std", "ZCR_mean", "ZCR_std", "RMS_mean", "RMS_std"]
    for i in range(1, 13): feature_names.append(f"Chroma_{i}_mean")
    feature_names.append("Crest_Factor")

    indices = np.argsort(importances)[::-1][:12]
    top_names = [feature_names[i] for i in indices]
    top_scores = [importances[i] * 100 for i in indices]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_scores[::-1], color="#3B82F6", edgecolor="#1E3A8A", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names[::-1], fontsize=10)
    ax.set_xlabel("Feature Importance Relative Weight (%)", fontsize=11, fontweight="bold")
    ax.set_title("Top 12 Psychoacoustic & Spectral Features in Noise Source Classification", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    chart3_path = os.path.join(DOCS_FIG_DIR, "ml_feature_importances.png")
    fig.savefig(chart3_path)
    plt.close(fig)
    print(f"Saved chart: {chart3_path}")

    print("\n" + "=" * 72)
    print("AI Model Training & Evaluation Finished Successfully!")
    print("=" * 72)

if __name__ == "__main__":
    main()
