"""
Hearing-exposure dose calculator module (Stage 5 / Feature F5).
Implements switchable NIOSH and WHO-ITU reference exposure models.
"""

def allowed_time_hours(level_dba: float, profile: str = "niosh") -> float:
    """
    Computes allowed exposure time in hours at a given sound level.
    
    Parameters:
        level_dba: Sound exposure level in dBA
        profile: Reference standard ('niosh' or 'who')
                 - NIOSH: 85 dBA allows 8 hours, 3 dB exchange rate.
                 - WHO: 80 dBA allows 40 hours per week, 3 dB exchange rate.
                 
    Returns:
        Allowed time T(L) in hours.
    """
    ref, base = {"niosh": (85.0, 8.0), "who": (80.0, 40.0)}[profile.lower()]
    return float(base / (2.0 ** ((level_dba - ref) / 3.0)))

def dose_percent(periods: list, profile: str = "niosh") -> float:
    """
    Calculates accumulated dose percentage across periods of exposure.
    
    Parameters:
        periods: List of tuples [(level_dba, duration_hours), ...]
                 If level_dba is None, the period is counted as unknown/unavailable
                 and skipped in dose sum (never coerced to 0).
        profile: Reference standard ('niosh' or 'who')
        
    Returns:
        Cumulative dose as a percentage (e.g. 100.0 means reference limit reached).
    """
    return float(100.0 * sum(
        h / allowed_time_hours(l, profile)
        for l, h in periods
        if l is not None
    ))
