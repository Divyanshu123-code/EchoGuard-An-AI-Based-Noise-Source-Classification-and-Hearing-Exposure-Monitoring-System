"""
src/laptop_monitor.py

Real-Time macOS Audio & Headphone Exposure Monitor (Feature FR-01, FR-02, FR-03).
Detects active audio playback on macOS, tracks system volume, matches output devices
to calibrated headphone profiles, calculates real-time dBA and cumulative exposure,
and logs completed listening sessions directly into EchoGuard.
"""

import os
import re
import sys
import time
import datetime
import subprocess
import pandas as pd

from src.dose_calculator import allowed_time_hours, dose_percent
from src.level_estimator import estimate_level

# Path to persistent session log
SESSIONS_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample_sessions.csv")

def get_system_volume() -> tuple[int, bool]:
    """
    Returns current macOS output volume percentage (0-100) and mute status.
    Uses native osascript without external dependencies.
    """
    try:
        vol_str = subprocess.check_output(
            ["osascript", "-e", "output volume of (get volume settings)"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        vol = int(vol_str)
        
        mute_str = subprocess.check_output(
            ["osascript", "-e", "output muted of (get volume settings)"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        is_muted = mute_str.lower() == "true"
        return vol, is_muted
    except Exception:
        return 50, False

def get_default_output_device() -> str:
    """
    Queries macOS CoreAudio for currently active default output audio device.
    """
    try:
        import sounddevice as sd
        default_out_idx = sd.default.device[1]
        device_info = sd.query_devices(default_out_idx)
        return str(device_info.get("name", "Default Audio Device"))
    except Exception:
        return "Built-in / USB Audio"

def is_audio_playing() -> bool:
    """
    Checks if any process on macOS currently holds an active audio playback assertion
    (e.g., Spotify, Chrome YouTube, Apple Music, VLC, Zoom, etc.).
    """
    try:
        out = subprocess.check_output(["pmset", "-g", "assertions"], stderr=subprocess.DEVNULL).decode()
        # Look for active audio output assertions
        match = re.search(r"Playing audio|audio-out|AppleUSBAudioEngine|com\.apple\.audio", out, re.IGNORECASE)
        return bool(match)
    except Exception:
        return False

def volume_pct_to_step(vol_pct: int, max_step: int = 15) -> int:
    """Converts a 0-100 volume percentage to a discrete volume step."""
    if vol_pct <= 0:
        return 1
    step = int(round((vol_pct / 100.0) * max_step))
    return max(1, min(max_step, step))

def resolve_profile_for_device(device_name: str) -> tuple[str, str]:
    """
    Matches detected audio device name to a calibrated or generic EchoGuard profile.
    Returns (profile_id, display_label).
    """
    dev_lower = device_name.lower()
    if "sony" in dev_lower or "wh-1000" in dev_lower or "xm4" in dev_lower or "xm5" in dev_lower:
        return "prof_sony_wh1000xm4", "Sony WH-1000XM4 (Calibrated Over-Ear)"
    elif "bud" in dev_lower or "galaxy" in dev_lower or "airpod" in dev_lower or "earphone" in dev_lower or "ab13x" in dev_lower or "usb" in dev_lower:
        return "prof_galaxy_buds_fe", f"{device_name} (Calibrated In-Ear Profile)"
    elif "speaker" in dev_lower or "built-in" in dev_lower:
        return "prof_generic_overear", "Laptop Built-in Speakers (Ambient / Open)"
    else:
        return "prof_generic_inear", f"{device_name} (Generic In-Ear)"

class LaptopExposureMonitor:
    """
    Tracks audio playback sessions, calculates exposure, and writes to data/sample_sessions.csv.
    """
    def __init__(self, poll_interval_sec: float = 2.0, pause_timeout_sec: float = 12.0):
        self.poll_interval = poll_interval_sec
        self.pause_timeout = pause_timeout_sec
        
        self.is_session_active = False
        self.session_id = None
        self.start_time = None
        self.last_audio_active_time = None
        self.volume_samples = []
        self.current_device = ""
        self.current_profile_id = "prof_galaxy_buds_fe"
        self.current_profile_label = ""
        self.last_logged_time = None

    def start_session(self, current_time: datetime.datetime, vol_step: int, device_name: str):
        self.is_session_active = True
        self.session_id = f"mac_{int(current_time.timestamp())}"
        self.start_time = current_time
        self.last_audio_active_time = current_time
        self.volume_samples = [vol_step]
        self.current_device = device_name
        self.current_profile_id, self.current_profile_label = resolve_profile_for_device(device_name)
        print(f"\n▶️ [Session Started] ID: {self.session_id} | Device: {device_name}")

    def update_session(self, current_time: datetime.datetime, vol_step: int):
        self.last_audio_active_time = current_time
        self.volume_samples.append(vol_step)

    def close_and_save_session(self, end_time: datetime.datetime):
        if not self.is_session_active:
            return

        duration_sec = (end_time - self.start_time).total_seconds()
        # Only log sessions that ran for at least 5 seconds
        if duration_sec >= 5.0 and len(self.volume_samples) > 0:
            avg_step = int(round(sum(self.volume_samples) / len(self.volume_samples)))
            new_row = {
                "session_id": self.session_id,
                "start_time": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                "profile_id": self.current_profile_id,
                "volume_step": avg_step,
                "max_step": 15,
                "label": f"macos_{self.current_device.replace(' ', '_').lower()}"
            }
            
            # Append to CSV
            os.makedirs(os.path.dirname(SESSIONS_CSV), exist_ok=True)
            df = pd.DataFrame([new_row])
            if os.path.exists(SESSIONS_CSV) and os.path.getsize(SESSIONS_CSV) > 0:
                df.to_csv(SESSIONS_CSV, mode="a", header=False, index=False)
            else:
                df.to_csv(SESSIONS_CSV, index=False)
                
            est = estimate_level(avg_step, 15, self.current_profile_id)
            print(f"⏹️ [Session Completed & Saved] Duration: {duration_sec:.1f}s ({duration_sec/60:.2f} min) | Avg Step: {avg_step}/15 | Est: {est['level_dba']:.1f} dBA")

        self.is_session_active = False
        self.session_id = None
        self.volume_samples = []

    def tick(self) -> dict:
        """
        Executes one polling cycle.
        Returns state dictionary for HUD / terminal display.
        """
        now = datetime.datetime.now()
        vol_pct, is_muted = get_system_volume()
        device_name = get_default_output_device()
        is_playing = is_audio_playing() and not is_muted and vol_pct > 0

        vol_step = volume_pct_to_step(vol_pct, max_step=15)
        profile_id, profile_label = resolve_profile_for_device(device_name)

        if is_playing:
            if not self.is_session_active:
                self.start_session(now, vol_step, device_name)
            else:
                self.update_session(now, vol_step)
        else:
            if self.is_session_active:
                # Check pause timeout
                pause_duration = (now - self.last_audio_active_time).total_seconds()
                if pause_duration >= self.pause_timeout:
                    self.close_and_save_session(self.last_audio_active_time)

        # Compute live instantaneous metrics
        est = estimate_level(vol_step if not is_muted else None, 15, profile_id)
        current_dba = est["level_dba"]
        margin_db = est["margin_db"]
        data_label = est["label"]
        
        session_duration_sec = (now - self.start_time).total_seconds() if self.is_session_active else 0.0
        allowed_hrs = allowed_time_hours(current_dba, "niosh") if current_dba else None

        return {
            "timestamp": now.strftime("%H:%M:%S"),
            "device": device_name,
            "profile_label": profile_label,
            "volume_pct": vol_pct,
            "volume_step": vol_step,
            "is_playing": is_playing,
            "is_muted": is_muted,
            "current_dba": current_dba,
            "margin_db": margin_db,
            "data_label": data_label,
            "session_active": self.is_session_active,
            "session_duration_sec": session_duration_sec,
            "allowed_hours": allowed_hrs
        }
