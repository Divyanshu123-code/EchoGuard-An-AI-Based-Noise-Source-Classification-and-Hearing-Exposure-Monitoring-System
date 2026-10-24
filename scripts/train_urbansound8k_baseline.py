#!/usr/bin/env python3
"""
scripts/train_urbansound8k_baseline.py

Runs the official UrbanSound8K 10-fold cross-validation baseline:
- Extracts 40 MFCC features per audio slice
- Evaluates across all 10 predefined folds (leaving out fold k for testing)
- Produces fold accuracy table, mean accuracy, std deviation, confusion matrix
- Exports model to models/urbansound8k_rf_pipeline.joblib
- Saves evaluation charts to docs/figures/
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.urbansound8k_benchmark import train_and_export_urbansound8k_pipeline

if __name__ == "__main__":
    train_and_export_urbansound8k_pipeline()
