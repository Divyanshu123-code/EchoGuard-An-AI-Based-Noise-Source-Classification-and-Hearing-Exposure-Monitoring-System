"""
src/level_estimator.py

Headphone sound-level estimation module (Stage 4 / Features F4, FR-05, FR-06).
Estimates ear-level sound in dBA from volume step and headphone profile,
tracking measurement source and confidence error margins.
"""

from src.headphone_profiles import lookup_profile_dba

def estimate_level(
    volume_step: int | None,
    max_step: int,
    profile_id: str,
    content: str = "typical"
) -> dict:
    """
    Estimates ear-level sound in dBA from volume step and headphone profile.
    
    Parameters:
        volume_step: Current device volume integer step (or None if unavailable)
        max_step: Maximum volume step on device (e.g. 15 or 100)
        profile_id: Identifier for headphone profile
        content: Audio content loudness bucket ('typical', 'loud', 'quiet', etc.)
        
    Returns:
        dict:
            level_dba: float or None
            margin_db: float (uncertainty margin)
            method: 'lookup' or 'ml'
            label: 'measured', 'estimated', 'simulated', or 'unavailable'
    """
    if volume_step is None:
        return {
            "level_dba": None,
            "margin_db": 5.0,
            "method": "lookup",
            "label": "unavailable"
        }
    
    # Backwards compatibility for synthetic/test profiles
    if profile_id == "test_profile":
        fraction = volume_step / max_step
        simulated_dba = float(60.0 + 40.0 * fraction)
        return {
            "level_dba": simulated_dba,
            "margin_db": 5.0,
            "method": "lookup",
            "label": "simulated"
        }

    # Query calibrated profile lookup
    dba, margin_db, label = lookup_profile_dba(profile_id, volume_step, max_step)
    
    # Content type adjustment (crest factor / dynamic range)
    content_adj = {
        "typical": 0.0,
        "loud": +1.5,
        "quiet": -2.0,
        "podcast": -0.8
    }.get(content.lower(), 0.0)
    
    final_dba = round(dba + content_adj, 1)

    return {
        "level_dba": final_dba,
        "margin_db": margin_db,
        "method": "lookup",
        "label": label
    }
