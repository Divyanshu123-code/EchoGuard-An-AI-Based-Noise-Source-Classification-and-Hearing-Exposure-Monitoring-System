import math

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
        if l is not None and h > 0
    ))

def calculate_laeq(periods: list) -> float | None:
    """
    Computes energy-equivalent continuous A-weighted sound level (L_Aeq) in dBA:
        L_Aeq = 10 * log10( (1 / sum(t)) * sum( t_i * 10^(L_i / 10) ) )
    
    Parameters:
        periods: List of tuples [(level_dba, duration_hours), ...]
        
    Returns:
        L_Aeq in dBA rounded to 1 decimal place, or None if no valid periods.
    """
    valid = [(float(l), float(h)) for l, h in periods if l is not None and h > 0]
    if not valid:
        return None
    total_time = sum(h for _, h in valid)
    if total_time <= 0:
        return None
    energy_sum = sum(h * (10.0 ** (l / 10.0)) for l, h in valid)
    laeq = 10.0 * math.log10(energy_sum / total_time)
    return round(laeq, 1)

def combined_daily_dose(
    headphone_periods: list,
    ambient_periods: list,
    hp_profile: str = "niosh",
    amb_profile: str = "niosh"
) -> dict:
    """
    Calculates combined daily hearing exposure across Headphone listening and
    Ambient environmental noise sources (FR8, FR9, Section 1 & A2):
        Total Daily Ear Exposure = Headphone Dose + Ambient Dose
        
    Parameters:
        headphone_periods: List of [(level_dba, duration_hours)] for headphone usage
        ambient_periods: List of [(level_dba, duration_hours)] for ambient environment
        hp_profile: Reference standard for headphones ('niosh' or 'who')
        amb_profile: Reference standard for ambient noise ('niosh')
        
    Returns:
        dict:
            headphone_dose_pct: float
            ambient_dose_pct: float
            total_dose_pct: float
            headphone_hours: float
            ambient_hours: float
            total_hours: float
            laeq_combined: float | None
            headphone_standard: str
            ambient_standard: str
    """
    hp_dose = dose_percent(headphone_periods, profile=hp_profile)
    amb_dose = dose_percent(ambient_periods, profile=amb_profile)
    total_dose = hp_dose + amb_dose

    hp_hrs = sum(h for l, h in headphone_periods if l is not None and h > 0)
    amb_hrs = sum(h for l, h in ambient_periods if l is not None and h > 0)
    total_hrs = hp_hrs + amb_hrs

    all_periods = [p for p in headphone_periods + ambient_periods if p[0] is not None and p[1] > 0]
    laeq = calculate_laeq(all_periods)

    return {
        "headphone_dose_pct": round(hp_dose, 2),
        "ambient_dose_pct": round(amb_dose, 2),
        "total_dose_pct": round(total_dose, 2),
        "headphone_hours": round(hp_hrs, 2),
        "ambient_hours": round(amb_hrs, 2),
        "total_hours": round(total_hrs, 2),
        "laeq_combined": laeq,
        "headphone_standard": hp_profile.upper(),
        "ambient_standard": amb_profile.upper()
    }

