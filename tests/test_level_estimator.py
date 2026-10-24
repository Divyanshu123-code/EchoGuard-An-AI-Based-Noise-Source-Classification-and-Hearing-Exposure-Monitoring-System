import pytest
from src.level_estimator import estimate_level

def test_estimate_level_normal():
    res = estimate_level(volume_step=10, max_step=20, profile_id="test_profile")
    assert res["level_dba"] == pytest.approx(80.0)
    assert res["margin_db"] == 5.0
    assert res["method"] in ["lookup", "ml"]
    assert res["label"] in ["estimated", "simulated"]

def test_estimate_level_missing_volume():
    # When volume is None, must return None level with 'unavailable' label
    res = estimate_level(volume_step=None, max_step=20, profile_id="test_profile")
    assert res["level_dba"] is None
    assert res["label"] == "unavailable"

def test_estimate_level_ml():
    res = estimate_level(volume_step=8, max_step=15, profile_id="prof_sony_wh1000xm4", method="ml")
    assert res["level_dba"] is not None
    assert 60.0 <= res["level_dba"] <= 85.0
    assert res["method"] == "ml"
    assert res["label"] == "estimated"
    assert res["margin_db"] > 0

