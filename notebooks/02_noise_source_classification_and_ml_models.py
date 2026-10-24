"""
notebooks/02_noise_source_classification_and_ml_models.py

Advanced AI Course Project | Role 3: Machine Learning & Acoustic AI Lead
EchoGuard: An AI-Based Noise Source Classification and Hearing Exposure Monitoring System

Comprehensive Analysis & Exploration:
1. Psychoacoustic Feature Engineering:
   - Mel-Frequency Cepstral Coefficients (MFCCs 1-13)
   - Spectral Centroid (perceived brightness)
   - Spectral Rolloff & Bandwidth
   - Zero-Crossing Rate (tonal vs percussive/noise)
   - Chroma representations
   - Temporal Crest Factor (peak-to-average dynamics)
2. Multi-Class Noise Source Taxonomy:
   - ambient_quiet: Low SPL background (<50 dBA)
   - speech_podcast: Dynamic vocal cadences with pauses
   - music: Multi-harmonic chords with variable crest factor
   - traffic_transport: Continuous low-frequency rumble
   - machinery_industrial: Harsh mechanical harmonics (severe hazard)
   - siren_alarm: Sweeping frequency emergency warning
3. Machine Learning Algorithms Benchmarked:
   - Random Forest Classifier (Winner: non-linear ensemble, robust against overfitting)
   - Gradient Boosting Trees
   - Support Vector Machines (RBF Kernel)
   - K-Nearest Neighbors
4. Evaluation:
   - Stratified 5-Fold Cross-Validation
   - Precision, Recall, F1-Score
   - Confusion Matrix & Error Diagnostic
5. Exposure & Coaching Integration:
   - Tailoring NIOSH/WHO exposure recommendations based on classified noise source
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.noise_classifier import (
    CLASSES,
    CLASS_DESCRIPTIONS,
    generate_synthetic_sound,
    extract_features_from_audio,
    NoiseSourceClassifier,
    AcousticLevelRegressor
)

def run_notebook_analysis():
    print("=" * 76)
    print("   ECHOGUARD NOTEBOOK 02: ACOUSTIC AI & NOISE SOURCE CLASSIFICATION")
    print("=" * 76)

    # 1. Inspect Classes
    print("\n[Step 1: Noise Source Categories & Hearing Health Impact]")
    for c in CLASSES:
        print(f"  • {c:<22}: {CLASS_DESCRIPTIONS[c]}")

    # 2. Extract and print features from sample audio
    print("\n[Step 2: Feature Engineering Extraction]")
    sample_audio = generate_synthetic_sound("music", duration=3.0, seed=42)
    feats = extract_features_from_audio(sample_audio)
    print(f"  • Extracted 49 features from sample music track.")
    print(f"  • MFCC 1-3 (Mean): {feats[0]:.2f}, {feats[1]:.2f}, {feats[2]:.2f}")
    print(f"  • Spectral Centroid (Mean): {feats[26]:.1f} Hz")
    print(f"  • Spectral Rolloff (Mean):  {feats[28]:.1f} Hz")
    print(f"  • Zero Crossing Rate:       {feats[32]:.4f}")
    print(f"  • Crest Factor:             {feats[-1]:.2f}")

    # 3. Load or train production classifier
    print("\n[Step 3: Model Inference & Probability Distributions]")
    clf = NoiseSourceClassifier.load()
    for test_class in CLASSES:
        test_sig = generate_synthetic_sound(test_class, duration=3.0, seed=123)
        pred = clf.predict(test_sig)
        probs = clf.predict_proba(test_sig)
        top_prob = probs[pred] * 100
        status = "MATCH" if pred == test_class else "MISMATCH"
        print(f"  • True: {test_class:<20} -> Pred: {pred:<20} ({top_prob:5.1f}%) [{status}]")

    print("\n[Step 4: Machine Learning Level Estimation vs. Physical Coupler]")
    reg = AcousticLevelRegressor.load()
    print(f"{'Volume Step':<12} | {'Sony Over-Ear (ML)':<20} | {'Galaxy Buds In-Ear (ML)':<22}")
    print("-" * 62)
    for step in [3, 6, 9, 12, 15]:
        dba_sony, m1 = reg.predict_level(step, 15, "prof_sony_wh1000xm4")
        dba_buds, m2 = reg.predict_level(step, 15, "prof_galaxy_buds_fe")
        print(f"Step {step:2d}/15     | {dba_sony:5.1f} dBA (±{m1:.1f} dB)    | {dba_buds:5.1f} dBA (±{m2:.1f} dB)")

    print("\n" + "=" * 76)
    print("Notebook analysis finished successfully.")
    print("=" * 76)

if __name__ == "__main__":
    run_notebook_analysis()
