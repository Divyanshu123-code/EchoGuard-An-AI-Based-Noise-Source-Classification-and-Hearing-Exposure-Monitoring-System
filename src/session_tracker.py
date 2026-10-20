"""
src/session_tracker.py

Session tracking and aggregation module (Features F1, FR-01, FR-02).
Tracks listening session start/end times, computes duration, session gaps,
and generates daily and weekly exposure metrics.
"""

import datetime
import pandas as pd
from src.level_estimator import estimate_level
from src.dose_calculator import dose_percent

def parse_iso_time(ts_str: str) -> datetime.datetime:
    """Parses ISO timestamp strings."""
    try:
        return datetime.datetime.fromisoformat(ts_str)
    except Exception:
        # Fallback format
        return datetime.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")

def calculate_session_duration_hours(start_time: str, end_time: str) -> float:
    """Calculates duration between start and end timestamps in hours."""
    t0 = parse_iso_time(start_time)
    t1 = parse_iso_time(end_time)
    duration_sec = max(0.0, (t1 - t0).total_seconds())
    return float(duration_sec / 3600.0)

def enrich_sessions(sessions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches a sessions DataFrame with duration, estimated dBA, and session dose %.
    Input schema: session_id, start_time, end_time, profile_id, volume_step, max_step, label
    """
    if sessions_df.empty:
        return sessions_df

    df = sessions_df.copy()
    durations = []
    levels = []
    margins = []
    labels = []

    for _, row in df.iterrows():
        dur = calculate_session_duration_hours(str(row["start_time"]), str(row["end_time"]))
        durations.append(round(dur, 3))
        
        step = int(row["volume_step"]) if pd.notnull(row["volume_step"]) else None
        max_step = int(row["max_step"]) if pd.notnull(row["max_step"]) else 15
        pid = str(row["profile_id"])

        est = estimate_level(step, max_step, pid)
        levels.append(est["level_dba"])
        margins.append(est["margin_db"])
        labels.append(est["label"])

    df["duration_hours"] = durations
    df["level_dba"] = levels
    df["margin_db"] = margins
    df["data_label"] = labels
    
    # Calculate per-session dose percent (NIOSH)
    session_doses = []
    for _, row in df.iterrows():
        lvl = row["level_dba"]
        dur = row["duration_hours"]
        if lvl is not None and dur > 0:
            d = dose_percent([(lvl, dur)], profile="niosh")
            session_doses.append(round(d, 2))
        else:
            session_doses.append(0.0)
    df["dose_percent_niosh"] = session_doses
    return df

def aggregate_daily_exposure(enriched_df: pd.DataFrame, standard: str = "niosh") -> pd.DataFrame:
    """
    Groups enriched sessions by calendar day to compute daily totals:
    - total_listening_hours
    - session_count
    - max_level_dba
    - cumulative_dose_pct
    """
    if enriched_df.empty:
        return pd.DataFrame()

    df = enriched_df.copy()
    df["date"] = df["start_time"].apply(lambda s: parse_iso_time(str(s)).strftime("%Y-%m-%d"))

    daily = []
    for date_str, grp in df.groupby("date"):
        periods = list(zip(grp["level_dba"], grp["duration_hours"]))
        total_dur = grp["duration_hours"].sum()
        valid_levels = [l for l in grp["level_dba"] if l is not None]
        max_lvl = max(valid_levels) if valid_levels else None
        cum_dose = dose_percent(periods, profile=standard)

        daily.append({
            "date": date_str,
            "session_count": len(grp),
            "total_duration_hours": round(total_dur, 2),
            "max_level_dba": round(max_lvl, 1) if max_lvl else None,
            "daily_dose_pct": round(cum_dose, 1)
        })

    return pd.DataFrame(daily).sort_values("date")
