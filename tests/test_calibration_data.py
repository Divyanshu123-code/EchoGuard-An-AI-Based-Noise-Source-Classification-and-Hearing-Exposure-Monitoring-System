"""
tests/test_calibration_data.py

Unit tests validating Role 2 (Data and Calibration Lead) deliverables:
- Schema, completeness, and cleanliness of data/calibration.csv
- Monotonic rise of sound level with volume step
- Schema and completeness of data/headphone_profiles.csv
- Repeatability standard deviation thresholds (< 2 dB)
- Presence and integrity of documentation and charts
"""

import os
import pandas as pd
import numpy as np
import pytest

CALIBRATION_CSV = "data/calibration.csv"
PROFILES_CSV = "data/headphone_profiles.csv"
CALIBRATION_REPORT = "docs/calibration_report.md"
CALIBRATION_NOTES = "docs/calibration_notes.md"
FIGURES_DIR = "docs/figures"

def test_calibration_csv_exists_and_valid_schema():
    assert os.path.exists(CALIBRATION_CSV), f"Missing {CALIBRATION_CSV}"
    df = pd.read_csv(CALIBRATION_CSV)
    
    expected_cols = ["recording_id", "date", "headphone", "signal", "volume_step", "max_step", "dba", "note"]
    assert list(df.columns) == expected_cols, f"Columns mismatch: {df.columns}"
    
    # Must have no empty/NaN cells in mandatory fields
    assert not df["recording_id"].isnull().any(), "Empty recording_id found"
    assert not df["headphone"].isnull().any(), "Empty headphone found"
    assert not df["signal"].isnull().any(), "Empty signal found"
    assert not df["volume_step"].isnull().any(), "Empty volume_step found"
    assert not df["dba"].isnull().any(), "Empty dba found"

def test_calibration_csv_record_counts():
    df = pd.read_csv(CALIBRATION_CSV)
    # Check baseline recordings per headphone (~60 rows per headphone)
    sony_rows = df[(df["headphone"] == "sony_wh1000xm4") & (~df["note"].str.startswith("repeatability"))]
    galaxy_rows = df[df["headphone"] == "galaxy_buds_fe"]
    
    assert len(sony_rows) == 60, f"Expected 60 Sony recordings, got {len(sony_rows)}"
    assert len(galaxy_rows) == 60, f"Expected 60 Galaxy recordings, got {len(galaxy_rows)}"
    
    # Repeatability trials (15 recordings)
    rep_rows = df[df["note"].str.startswith("repeatability")]
    assert len(rep_rows) == 15, f"Expected 15 repeatability recordings, got {len(rep_rows)}"

def test_calibration_sound_level_bounds_and_monotonicity():
    df = pd.read_csv(CALIBRATION_CSV)
    # Levels must fall between realistic listening bounds (40 to 105 dBA)
    assert (df["dba"] >= 40.0).all(), f"Found dBA below 40: {df[df['dba'] < 40.0]}"
    assert (df["dba"] <= 105.0).all(), f"Found dBA above 105: {df[df['dba'] > 105.0]}"

    # Test monotonicity: mean dBA must rise with volume step
    for hp in ["sony_wh1000xm4", "galaxy_buds_fe"]:
        sub = df[(df["headphone"] == hp) & (~df["note"].str.startswith("repeatability"))]
        step_means = sub.groupby("volume_step")["dba"].mean()
        diffs = np.diff(step_means.values)
        assert (diffs > 0).all(), f"Volume curve for {hp} is not strictly monotonic: diffs = {diffs}"

def test_repeatability_variance():
    df = pd.read_csv(CALIBRATION_CSV)
    rep_df = df[df["note"].str.startswith("repeatability")]
    assert not rep_df.empty, "No repeatability data found"
    
    stds = rep_df.groupby("volume_step")["dba"].std()
    for step, std_val in stds.items():
        assert std_val < 2.0, f"Repeatability std dev at step {step} is too high: {std_val:.2f} dB (must be < 2 dB)"

def test_headphone_profiles_csv():
    assert os.path.exists(PROFILES_CSV), f"Missing {PROFILES_CSV}"
    df = pd.read_csv(PROFILES_CSV)
    
    expected_cols = ["profile_id", "headphone", "type", "step", "dba", "margin_db", "source"]
    assert list(df.columns) == expected_cols
    
    # Both headphones represented for steps 1 to 15
    assert len(df) == 30
    assert set(df["headphone"].unique()) == {"sony_wh1000xm4", "galaxy_buds_fe"}
    assert (df["source"] == "measured").all()
    assert (df["margin_db"] == 5.0).all()

def test_report_headings_and_content():
    assert os.path.exists(CALIBRATION_REPORT), f"Missing {CALIBRATION_REPORT}"
    with open(CALIBRATION_REPORT, "r", encoding="utf-8") as f:
        content = f.read()

    # Step 2.4 required headings
    required_headings = [
        "What we did",
        "Equipment",
        "Method",
        "Results",
        "How accurate is it",
        "Limits"
    ]
    for h in required_headings:
        assert h.lower() in content.lower(), f"Missing required section heading: '{h}' in report"

    # Step 2.4 required phrases & content
    assert "estimated, about plus or minus 5 db unless a reference meter is used" in content.lower()
    assert "one microphone" in content.lower()
    assert "paper cup is not a real ear" in content.lower()
    assert "sound-meter app" in content.lower()

def test_figures_exist():
    expected_figures = [
        "dba_vs_volume_step.png",
        "repeatability_test.png",
        "headphone_comparison.png"
    ]
    for fig in expected_figures:
        path = os.path.join(FIGURES_DIR, fig)
        assert os.path.exists(path), f"Figure {path} not found"
        assert os.path.getsize(path) > 10000, f"Figure {path} is suspiciously small"
