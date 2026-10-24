import pytest
from src.dose_calculator import dose_percent, allowed_time_hours

def test_reference_cases():
    # NIOSH reference cases
    assert dose_percent([(85, 8)]) == pytest.approx(100.0)
    assert dose_percent([(91, 2)]) == pytest.approx(100.0)
    assert dose_percent([(100, 0.25)]) == pytest.approx(100.0)
    assert dose_percent([(82, 2), (91, 1), (97, 0.25)]) == pytest.approx(112.5)

def test_who_reference_cases():
    # WHO-ITU reference cases: 80 dBA for 40h/week
    assert dose_percent([(80, 40)], profile="who") == pytest.approx(100.0)
    assert dose_percent([(86, 10)], profile="who") == pytest.approx(100.0)
    assert dose_percent([(92, 2.5)], profile="who") == pytest.approx(100.0)

def test_missing_levels_skipped_never_zero():
    # Critical honesty requirement: None levels must contribute 0 dose (skipped),
    # but must NOT convert to 0 dBA (which would falsely imply silence)
    assert dose_percent([(None, 2), (85, 8)]) == pytest.approx(100.0)
    assert dose_percent([(None, 5)]) == pytest.approx(0.0)

def test_calculate_laeq():
    from src.dose_calculator import calculate_laeq
    # Equal duration of 80 dBA and 80 dBA -> 80.0 dBA
    assert calculate_laeq([(80, 2), (80, 2)]) == pytest.approx(80.0, 0.1)
    # Energy sum of two equal 80 dBA components doubles energy -> +3 dB = 83.0 dBA if combined simultaneously,
    # or equal sequential levels remain 80.0 dBA
    assert calculate_laeq([(85, 4), (91, 2)]) is not None
    assert calculate_laeq([]) is None

def test_combined_daily_dose():
    from src.dose_calculator import combined_daily_dose
    # Headphone: 85 dBA for 4 hours (50% NIOSH dose)
    # Ambient: 85 dBA for 4 hours (50% NIOSH dose)
    # Combined: 100% dose
    res = combined_daily_dose([(85, 4)], [(85, 4)], hp_profile="niosh", amb_profile="niosh")
    assert res["headphone_dose_pct"] == pytest.approx(50.0, 0.1)
    assert res["ambient_dose_pct"] == pytest.approx(50.0, 0.1)
    assert res["total_dose_pct"] == pytest.approx(100.0, 0.1)
    assert res["total_hours"] == pytest.approx(8.0)
    assert res["laeq_combined"] == pytest.approx(85.0, 0.1)

