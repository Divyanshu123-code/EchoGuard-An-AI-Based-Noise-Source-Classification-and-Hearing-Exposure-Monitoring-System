"""
src/headphone_profiles.py

Headphone profile management module (Features F4, FR-04).
Loads calibrated profiles from data/headphone_profiles.csv and provides
generic fallback profiles for untested headphone categories.
"""

import os
import pandas as pd

PROFILES_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "headphone_profiles.csv")

GENERIC_PROFILES = {
    "prof_generic_overear": {
        "profile_id": "prof_generic_overear",
        "name": "Generic Over-Ear / Circumaural",
        "type": "over_ear",
        "base_at_step1": 45.0,
        "max_at_step15": 97.0,
        "margin_db": 6.5,
        "source": "simulated"
    },
    "prof_generic_inear": {
        "profile_id": "prof_generic_inear",
        "name": "Generic In-Ear / Ear Canal Sealed",
        "type": "in_ear",
        "base_at_step1": 47.0,
        "max_at_step15": 101.0,
        "margin_db": 7.0,
        "source": "simulated"
    },
    "prof_generic_earbuds": {
        "profile_id": "prof_generic_earbuds",
        "name": "Generic Open-Fit Earbuds",
        "type": "earbud",
        "base_at_step1": 44.0,
        "max_at_step15": 95.0,
        "margin_db": 7.5,
        "source": "simulated"
    }
}

def load_headphone_profiles() -> pd.DataFrame:
    """Loads profiles from data/headphone_profiles.csv if available."""
    if os.path.exists(PROFILES_CSV) and os.path.getsize(PROFILES_CSV) > 0:
        return pd.read_csv(PROFILES_CSV)
    return pd.DataFrame(columns=["profile_id", "headphone", "type", "step", "dba", "margin_db", "source"])

def get_available_profiles() -> list[dict]:
    """Returns a list of profile metadata dictionaries for UI dropdowns."""
    profiles = []
    df = load_headphone_profiles()
    if not df.empty:
        for pid, grp in df.groupby("profile_id"):
            hp_name = grp["headphone"].iloc[0]
            hp_type = grp["type"].iloc[0]
            display_name = "Sony WH-1000XM4 (Over-Ear ANC)" if "sony" in pid else "Samsung Galaxy Buds FE (In-Ear TWS)"
            profiles.append({
                "profile_id": pid,
                "name": display_name,
                "type": hp_type,
                "source": "measured",
                "margin_db": float(grp["margin_db"].iloc[0])
            })
    
    for gen in GENERIC_PROFILES.values():
        profiles.append({
            "profile_id": gen["profile_id"],
            "name": f"{gen['name']} (Generic Reference)",
            "type": gen["type"],
            "source": gen["source"],
            "margin_db": gen["margin_db"]
        })
    return profiles

def lookup_profile_dba(profile_id: str, step: int, max_step: int = 15) -> tuple[float, float, str]:
    """
    Looks up or interpolates sound level (dBA) for a given volume step.
    
    Returns:
        (dba, margin_db, label)
    """
    df = load_headphone_profiles()
    sub = df[df["profile_id"] == profile_id]
    
    if not sub.empty:
        # Scale input step if max_step differs from calibrated 15 steps
        norm_step = int(round((step / max_step) * 15.0))
        norm_step = max(1, min(15, norm_step))
        match = sub[sub["step"] == norm_step]
        if not match.empty:
            dba = float(match["dba"].iloc[0])
            margin = float(match["margin_db"].iloc[0])
            label = "measured" if max_step == 15 and step == norm_step else "estimated"
            return dba, margin, label
    
    # Check generic profiles
    if profile_id in GENERIC_PROFILES:
        gen = GENERIC_PROFILES[profile_id]
        frac = step / max_step
        dba = gen["base_at_step1"] + (gen["max_at_step15"] - gen["base_at_step1"]) * (frac ** 0.86)
        return float(dba), gen["margin_db"], "simulated"
    
    # Generic fallback
    frac = step / max_step
    dba = 60.0 + 40.0 * frac
    return float(dba), 5.0, "simulated"
