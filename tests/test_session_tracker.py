"""
tests/test_session_tracker.py

Tests for session duration calculation, session enrichment,
daily aggregation, and alert generation.
"""

import pandas as pd
import pytest
from src.session_tracker import (
    calculate_session_duration_hours,
    enrich_sessions,
    aggregate_daily_exposure
)
from src.alerts import evaluate_exposure_alerts

def test_calculate_session_duration():
    t0 = "2026-10-20 10:00:00"
    t1 = "2026-10-20 11:30:00"
    dur = calculate_session_duration_hours(t0, t1)
    assert dur == pytest.approx(1.5)

def test_enrich_sessions():
    raw_df = pd.DataFrame([{
        "session_id": "test_1",
        "start_time": "2026-10-20 09:00:00",
        "end_time": "2026-10-20 10:00:00",
        "profile_id": "prof_sony_wh1000xm4",
        "volume_step": 10,
        "max_step": 15,
        "label": "study"
    }])
    enriched = enrich_sessions(raw_df)
    assert "duration_hours" in enriched.columns
    assert "level_dba" in enriched.columns
    assert "dose_percent_niosh" in enriched.columns
    assert enriched["duration_hours"].iloc[0] == pytest.approx(1.0)
    assert enriched["level_dba"].iloc[0] > 70.0
    assert enriched["data_label"].iloc[0] in ["measured", "estimated"]

def test_aggregate_daily_exposure():
    raw_df = pd.DataFrame([
        {
            "session_id": "s1",
            "start_time": "2026-10-20 09:00:00",
            "end_time": "2026-10-20 10:00:00",
            "profile_id": "prof_sony_wh1000xm4",
            "volume_step": 8,
            "max_step": 15,
            "label": "s1"
        },
        {
            "session_id": "s2",
            "start_time": "2026-10-20 14:00:00",
            "end_time": "2026-10-20 15:30:00",
            "profile_id": "prof_sony_wh1000xm4",
            "volume_step": 9,
            "max_step": 15,
            "label": "s2"
        }
    ])
    enriched = enrich_sessions(raw_df)
    daily = aggregate_daily_exposure(enriched)
    assert len(daily) == 1
    assert daily["session_count"].iloc[0] == 2
    assert daily["total_duration_hours"].iloc[0] == pytest.approx(2.5)
    assert daily["daily_dose_pct"].iloc[0] > 0.0

def test_evaluate_exposure_alerts():
    # Safe condition
    safe_alerts = evaluate_exposure_alerts(daily_dose_pct=25.0, current_level_dba=72.0)
    assert any(a["severity"] == "INFO" for a in safe_alerts)

    # Danger condition (over 100%)
    danger_alerts = evaluate_exposure_alerts(daily_dose_pct=115.0, current_level_dba=92.0)
    assert any(a["severity"] == "DANGER" for a in danger_alerts)
    assert any(a["severity"] == "WARNING" for a in danger_alerts)
