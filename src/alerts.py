"""
src/alerts.py

Smart Hearing-Exposure Alert and Habit Coach module (Features F7, FR-10, FR-13).
Monitors exposure dose accumulation and instantaneous sound levels,
providing actionable, non-medical preventive coaching.
"""

from src.dose_calculator import allowed_time_hours

def evaluate_exposure_alerts(
    daily_dose_pct: float,
    current_level_dba: float | None = None,
    current_duration_hours: float = 0.0,
    profile_standard: str = "niosh"
) -> list[dict]:
    """
    Evaluates current exposure status and produces prioritized coaching alerts.
    
    Alert severity levels: 'INFO', 'CAUTION', 'WARNING', 'DANGER'
    """
    alerts = []
    
    # 1. Daily cumulative dose alerts
    if daily_dose_pct >= 100.0:
        alerts.append({
            "severity": "DANGER",
            "title": f"100% Daily Sound Exposure Limit Reached ({daily_dose_pct:.0f}%)",
            "message": (
                f"You have reached or exceeded 100% of your recommended daily sound dose "
                f"under the {profile_standard.upper()} guideline. Continued headphone listening today "
                f"significantly increases your cumulative acoustic fatigue."
            ),
            "recommendation": "Take off your headphones for the remainder of the day or give your ears an extended rest.",
            "icon": "🚨"
        })
    elif daily_dose_pct >= 80.0:
        alerts.append({
            "severity": "WARNING",
            "title": f"High Daily Sound Dose Approaching Limit ({daily_dose_pct:.0f}%)",
            "message": (
                f"You have consumed {daily_dose_pct:.0f}% of your safe daily allowance. "
                f"At current levels, you have limited remaining listening budget."
            ),
            "recommendation": "Lower your volume by 2–3 steps to dramatically extend your safe listening time.",
            "icon": "⚠️"
        })
    elif daily_dose_pct >= 50.0:
        alerts.append({
            "severity": "CAUTION",
            "title": f"Halfway Sound Dose Milestone ({daily_dose_pct:.0f}%)",
            "message": f"You have reached 50% of your {profile_standard.upper()} daily sound exposure target.",
            "recommendation": "Consider pausing media playback for a 10-minute ear break.",
            "icon": "⏸️"
        })
    else:
        alerts.append({
            "severity": "INFO",
            "title": "Exposure Within Healthy Daily Bounds",
            "message": f"Your cumulative exposure today is {daily_dose_pct:.1f}% of the {profile_standard.upper()} daily limit.",
            "recommendation": "Keep up your healthy listening habits!",
            "icon": "✅"
        })

    # 2. Instantaneous / session level alerts
    if current_level_dba is not None:
        if current_level_dba >= 91.0:
            limit = allowed_time_hours(current_level_dba, profile_standard)
            alerts.append({
                "severity": "WARNING",
                "title": f"High Sound Pressure Detected ({current_level_dba:.1f} dBA)",
                "message": (
                    f"Listening at {current_level_dba:.1f} dBA allows only {limit * 60:.0f} minutes "
                    f"of total daily exposure before 100% dose is reached."
                ),
                "recommendation": "Decrease volume by 3 steps immediately to protect hearing comfort.",
                "icon": "🔊"
            })
        elif current_level_dba >= 85.0:
            alerts.append({
                "severity": "CAUTION",
                "title": f"Volume at NIOSH Baseline ({current_level_dba:.1f} dBA)",
                "message": "Sound level is at or above the 85 dBA workplace reference threshold.",
                "recommendation": "Volume steps 1–2 notches lower will allow much longer safe listening.",
                "icon": "🔉"
            })

    # 3. Extended session duration alert
    if current_duration_hours >= 2.0:
        alerts.append({
            "severity": "CAUTION",
            "title": "Continuous Listening Session > 2 Hours",
            "message": f"You have been listening continuously for {current_duration_hours:.1f} hours.",
            "recommendation": "Take a 15-minute auditory rest to prevent ear fatigue and reset listening sensitivity.",
            "icon": "🎧"
        })

    return alerts
