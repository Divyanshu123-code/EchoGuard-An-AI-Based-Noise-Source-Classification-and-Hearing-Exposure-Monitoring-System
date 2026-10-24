"""
tests/test_noise_classifier.py

Unit tests for AI Noise Source Classification & Acoustic ML Engine:
- Verifies feature extraction dimensions and numerical validity
- Validates synthetic audio generation across all 6 classes
- Tests model training, prediction, and probability distributions
- Tests level regressor prediction and integration
"""

import numpy as np
import pytest
from src.noise_classifier import (
    extract_features_from_audio,
    generate_synthetic_sound,
    NoiseSourceClassifier,
    AcousticLevelRegressor,
    CLASSES
)

def test_extract_features_shape_and_no_nan():
    # 2 seconds of 1 kHz tone at 22050 Hz
    sr = 22050
    t = np.arange(sr * 2) / sr
    audio = 0.5 * np.sin(2 * np.pi * 1000 * t)
    
    feats = extract_features_from_audio(audio, sr=sr)
    assert isinstance(feats, np.ndarray)
    assert feats.shape == (49,), f"Expected 49 features, got {feats.shape}"
    assert not np.isnan(feats).any(), "Found NaN in extracted audio features"
    assert not np.isinf(feats).any(), "Found Inf in extracted audio features"

def test_generate_synthetic_sound_all_classes():
    sr = 22050
    for cls_name in CLASSES:
        audio = generate_synthetic_sound(cls_name, duration=1.0, sr=sr, seed=42)
        assert isinstance(audio, np.ndarray)
        assert len(audio) == sr
        assert np.max(np.abs(audio)) <= 1.0, f"Audio clipped in class {cls_name}"
        assert not np.isnan(audio).any()

def test_noise_classifier_fit_and_predict():
    # Quick fit on small synthetic batch
    X_list = []
    y_list = []
    for cls in CLASSES[:3]:
        for i in range(5):
            audio = generate_synthetic_sound(cls, duration=1.5, seed=i)
            feats = extract_features_from_audio(audio)
            X_list.append(feats)
            y_list.append(cls)

    X = np.vstack(X_list)
    y = np.array(y_list)

    clf = NoiseSourceClassifier(n_estimators=10)
    clf.fit(X, y)
    assert clf.is_fitted

    # Test prediction
    test_audio = generate_synthetic_sound(CLASSES[0], duration=1.5, seed=999)
    pred_cls = clf.predict(test_audio)
    assert pred_cls in CLASSES[:3]

    # Test probabilities
    probs = clf.predict_proba(test_audio)
    assert isinstance(probs, dict)
    assert set(probs.keys()) == set(CLASSES[:3])
    prob_sum = sum(probs.values())
    assert pytest.approx(prob_sum, 0.01) == 1.0

def test_acoustic_level_regressor():
    reg = AcousticLevelRegressor()
    # Test prepare vector
    vec = reg._prepare_sample_vector(step=8, max_step=15, is_in_ear=1)
    assert vec.shape == (5,)
    assert not np.isnan(vec).any()
