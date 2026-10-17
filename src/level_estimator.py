"""
Headphone sound-level estimation module (Stage 4 / Feature F4).
Initial Day 1 stub to unblock Dashboard (Role 3).
"""

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
            label: 'estimated', 'simulated', or 'unavailable'
    """
    if volume_step is None:
        return {
            "level_dba": None,
            "margin_db": 5.0,
            "method": "lookup",
            "label": "unavailable"
        }
    
    fraction = volume_step / max_step
    # Temporary Day 1 simulated linear mapping between 60 dBA and 100 dBA
    simulated_dba = float(60.0 + 40.0 * fraction)
    
    return {
        "level_dba": simulated_dba,
        "margin_db": 5.0,
        "method": "lookup",
        "label": "simulated"
    }
