"""
src/session_tracker.py

Session tracking and aggregation module (Features F1, FR-01, FR-02).
Tracks listening session start/end times, computes duration, session gaps,
and generates daily and weekly exposure metrics.
"""

import datetime
import pandas as pd
from src.level_estimator import estimate_level
from src.dose_calculator import dose_percent, combined_daily_dose, calculate_laeq

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
    Enriches a sessions DataFrame with duration, source_type ('headphone' vs 'ambient'),
    estimated dBA, uncertainty margin, and per-session dose %.
    Input schema: session_id, start_time, end_time, profile_id, volume_step, max_step, label [, source_type, level_dba]
    """
    if sessions_df.empty:
        return sessions_df

    df = sessions_df.copy()
    durations = []
    levels = []
    margins = []
    labels = []
    source_types = []

    for _, row in df.iterrows():
        dur = calculate_session_duration_hours(str(row["start_time"]), str(row["end_time"]))
        durations.append(round(dur, 3))
        
        pid = str(row["profile_id"]) if pd.notnull(row.get("profile_id")) else ""
        lbl = str(row["label"]) if pd.notnull(row.get("label")) else ""
        
        # Determine source type (FR9: separate ambient vs headphone exposure)
        stype = row.get("source_type")
        if pd.notnull(stype) and str(stype).strip():
            src = str(stype).strip().lower()
        elif "ambient" in pid.lower() or "ambient" in lbl.lower() or "city" in lbl.lower() or "traffic" in lbl.lower():
            src = "ambient"
        else:
            src = "headphone"
        source_types.append(src)

        # Level determination
        if pd.notnull(row.get("level_dba")):
            lvl = float(row["level_dba"])
            levels.append(round(lvl, 1))
            margins.append(2.0 if src == "ambient" else 5.0)
            labels.append("measured" if src == "ambient" else "estimated")
        elif src == "ambient":
            # Direct dBA reading or ambient preset
            raw_val = float(row["volume_step"]) if pd.notnull(row.get("volume_step")) else 70.0
            amb_dba = raw_val if raw_val > 30.0 else (55.0 + raw_val * 2.5)
            levels.append(round(amb_dba, 1))
            margins.append(2.0)
            labels.append("measured_ambient")
        else:
            step = int(row["volume_step"]) if pd.notnull(row["volume_step"]) else None
            max_step = int(row["max_step"]) if pd.notnull(row["max_step"]) else 15
            est = estimate_level(step, max_step, pid)
            levels.append(est["level_dba"])
            margins.append(est["margin_db"])
            labels.append(est["label"])

    df["duration_hours"] = durations
    df["source_type"] = source_types
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
    Groups enriched sessions by calendar day to compute daily totals with
    unified Headphone + Ambient exposure combination (FR8, FR9):
    - headphone_duration_hours & ambient_duration_hours
    - headphone_dose_pct & ambient_dose_pct
    - daily_dose_pct (combined total dose)
    - laeq_combined
    - max_level_dba
    """
    if enriched_df.empty:
        return pd.DataFrame()

    df = enriched_df.copy()
    df["date"] = df["start_time"].apply(lambda s: parse_iso_time(str(s)).strftime("%Y-%m-%d"))

    daily = []
    for date_str, grp in df.groupby("date"):
        hp_grp = grp[grp["source_type"] == "headphone"]
        amb_grp = grp[grp["source_type"] == "ambient"]

        hp_periods = list(zip(hp_grp["level_dba"], hp_grp["duration_hours"]))
        amb_periods = list(zip(amb_grp["level_dba"], amb_grp["duration_hours"]))

        comb = combined_daily_dose(hp_periods, amb_periods, hp_profile=standard, amb_profile="niosh")

        valid_levels = [l for l in grp["level_dba"] if l is not None]
        max_lvl = max(valid_levels) if valid_levels else None

        daily.append({
            "date": date_str,
            "session_count": len(grp),
            "headphone_session_count": len(hp_grp),
            "ambient_session_count": len(amb_grp),
            "total_duration_hours": comb["total_hours"],
            "headphone_duration_hours": comb["headphone_hours"],
            "ambient_duration_hours": comb["ambient_hours"],
            "headphone_dose_pct": comb["headphone_dose_pct"],
            "ambient_dose_pct": comb["ambient_dose_pct"],
            "daily_dose_pct": comb["total_dose_pct"],
            "max_level_dba": round(max_lvl, 1) if max_lvl else None,
            "laeq_combined": comb["laeq_combined"]
        })

    return pd.DataFrame(daily).sort_values("date")

