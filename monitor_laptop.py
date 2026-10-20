#!/usr/bin/env python3
"""
monitor_laptop.py - Live Real-Time macOS Audio & Headphone Exposure Monitor

Watches audio playback and master volume on macOS, automatically tracking
your listening sessions, estimating ear-level dBA in real time, and syncing
with the EchoGuard dashboard.

Usage:
    python monitor_laptop.py
    python monitor_laptop.py --interval 2.0
"""

import sys
import time
import argparse
import datetime
from src.laptop_monitor import LaptopExposureMonitor

def main():
    parser = argparse.ArgumentParser(description="EchoGuard Live macOS Audio Exposure Monitor")
    parser.add_argument("--interval", type=float, default=2.5, help="Polling interval in seconds")
    parser.add_argument("--timeout", type=float, default=15.0, help="Pause timeout to close session in seconds")
    args = parser.parse_args()

    monitor = LaptopExposureMonitor(poll_interval_sec=args.interval, pause_timeout_sec=args.timeout)

    print("=" * 72)
    print("   🛡️  ECHOGUARD: LIVE MACOS HEADPHONE EXPOSURE MONITOR")
    print("   Listening for active media playback & volume changes...")
    print("=" * 72)
    print("Press Ctrl+C to stop monitoring and finalize any active session.\n")

    try:
        while True:
            state = monitor.tick()
            
            # Format status line
            ts = state["timestamp"]
            dev = (state["device"][:22] + "..") if len(state["device"]) > 24 else state["device"]
            vol_str = f"{state['volume_pct']:2d}% (Step {state['volume_step']:2d}/15)"
            
            if state["is_playing"]:
                play_indicator = "▶️  PLAYING"
                dur_m, dur_s = divmod(int(state["session_duration_sec"]), 60)
                dur_str = f"{dur_m:02d}:{dur_s:02d}"
                dba_val = state["current_dba"]
                dba_str = f"{dba_val:5.1f} dBA (±{state['margin_db']:.0f}dB)"
                
                # Health risk color tag
                if dba_val < 75.0:
                    status_tag = "[SAFE]"
                elif dba_val < 85.0:
                    status_tag = "[MODERATE]"
                elif dba_val < 91.0:
                    status_tag = "[CAUTION: 85+ dBA]"
                else:
                    status_tag = "[HIGH EXPOSURE]"
            else:
                play_indicator = "⏸️  IDLE   "
                dur_str = "--:--"
                dba_str = "    -- dBA       "
                status_tag = "[STANDBY]"

            # Dynamic terminal line update
            sys.stdout.write(
                f"\r[{ts}] {play_indicator} | Vol: {vol_str} | Level: {dba_str} | Session: {dur_str} | {status_tag} | Dev: {dev}"
            )
            sys.stdout.flush()
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\nStopping monitor...")
        if monitor.is_session_active:
            now = datetime.datetime.now()
            monitor.close_and_save_session(now)
        print("Finalized session history. All sessions saved to data/sample_sessions.csv.")
        print("View results live in the dashboard: http://localhost:8501\n")

if __name__ == "__main__":
    main()
