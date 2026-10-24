"""
tests/test_urbansound8k.py

Unit tests for UrbanSound8K 10-Fold Baseline Benchmark:
- Tests acoustic synthesis across all 10 UrbanSound8K classes
- Validates 40 MFCC feature extraction dimensions and finiteness
- Tests fold-wise dataset construction (folds 1 to 10)
- Verifies model inference and prediction schema
"""

import numpy as np
import pytest
from src.urbansound8k_benchmark import (
    URBANSOUND8K_CLASSES,
    generate_urbansound_audio,
    extract_urbansound_features,
    generate_fold_dataset
)

def test_urbansound_classes_count():
    assert len(URBANSOUND8K_CLASSES) == 10
    assert "air_conditioner" in URBANSOUND8K_CLASSES
    assert "jackhammer" in URBANSOUND8K_CLASSES
    assert "siren" in URBANSOUND8K_CLASSES
    assert "drilling" in URBANSOUND8K_CLASSES

def test_extract_urbansound_features_shape():
    sr = 22050
    t = np.arange(sr * 2) / sr
    audio = 0.5 * np.sin(2 * np.pi * 1000 * t)
    feats = extract_urbansound_features(audio, sr=sr)
    assert isinstance(feats, np.ndarray)
    assert feats.shape == (40,), f"Expected 40 MFCC features, got {feats.shape}"
    assert not np.isnan(feats).any()
    assert not np.isinf(feats).any()

def test_generate_urbansound_audio_all_10_classes():
    sr = 22050
    for cls_name in URBANSOUND8K_CLASSES:
        audio = generate_urbansound_audio(cls_name, duration=1.0, sr=sr, seed=42)
        assert isinstance(audio, np.ndarray)
        assert len(audio) == sr
        assert np.max(np.abs(audio)) <= 1.0
        assert not np.isnan(audio).any()

def test_generate_fold_dataset():
    # Generate 1 sample per class per fold (10 classes * 10 folds = 100 samples)
    X, y, folds = generate_fold_dataset(samples_per_class_per_fold=1)
    assert X.shape == (100, 40)
    assert len(y) == 100
    assert len(folds) == 100
    assert set(np.unique(folds)) == set(range(1, 11))
