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
