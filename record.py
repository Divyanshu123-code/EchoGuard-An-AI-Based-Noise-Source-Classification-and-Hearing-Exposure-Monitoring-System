#!/usr/bin/env python3
"""
record.py - Audio Recording and Calibration Logging Script (Role 2)

Records test sounds played through earphones, calculates calibrated dBA level,
and logs measurements to data/calibration.csv.

Usage:
    python record.py
    python record.py --headphone sony_wh1000xm4 --offset 94.2
    python record.py --mock --batch  # Automated verification without physical mic
"""

import argparse
import csv
import datetime
import os
import sys
import numpy as np
import soundfile as sf

from src.audio_level import dba_from_audio

# Default calibration parameters
FS = 44100
SECONDS = 8
DEFAULT_OFFSET = 94.2  # Calibrated microphone offset in dB (measured in docs/calibration_notes.md)
DEFAULT_HEADPHONE = "sony_wh1000xm4"
DEFAULT_MAX_STEP = 15

def get_simulated_audio(signal: str, step: int, max_step: int, fs: int = FS, seconds: int = SECONDS) -> np.ndarray:
    """
    Synthesizes acoustic response expected at microphone for a given volume step
    when running in mock mode or headless environment.
    Acoustic curve: ~46 dBA at step 1 to ~98 dBA at step 15.
    """
    fraction = float(step) / float(max_step)
    # Target dBA level based on physical earphone response curve
    # 46 dBA at step 1 up to ~98 dBA at step 15
    target_dba = 42.0 + 56.0 * (fraction ** 0.85)
    
    # Slight variation based on signal characteristics
    signal_bias = {
        "tone1k": 0.0,
        "pink": -1.2,
        "music1": -2.5,
        "music2": +0.8
    }.get(signal.lower(), 0.0)
    
    target_dba += signal_bias + np.random.normal(0, 0.4)
    target_rms = 10 ** ((target_dba - DEFAULT_OFFSET) / 20.0)
    
    t = np.arange(seconds * fs) / fs
    if signal == "tone1k":
        audio = np.sqrt(2) * target_rms * np.sin(2 * np.pi * 1000 * t)
    elif signal == "pink":
        n = seconds * fs
        spec = np.fft.rfft(np.random.randn(n))
        freqs = np.fft.rfftfreq(n, 1 / fs); freqs[0] = 1.0
        pink = np.fft.irfft(spec / np.sqrt(freqs), n)
        audio = pink * (target_rms / (np.sqrt(np.mean(pink ** 2)) + 1e-12))
    else:
        # Music approximation: combination of harmonics
        sig = (0.5 * np.sin(2 * np.pi * 440 * t) +
               0.3 * np.sin(2 * np.pi * 880 * t) +
               0.2 * np.sin(2 * np.pi * 220 * t))
        audio = sig * (target_rms / (np.sqrt(np.mean(sig ** 2)) + 1e-12))
    
    # Add minor ambient noise floor (~34 dBA equivalent)
    noise_rms = 10 ** ((34.0 - DEFAULT_OFFSET) / 20.0)
    audio += np.random.normal(0, noise_rms, size=audio.shape)
    return np.clip(audio.astype(np.float32), -0.98, 0.98)

def main():
    parser = argparse.ArgumentParser(description="EchoGuard Audio Recording & Calibration Logger")
    parser.add_argument("--headphone", default=DEFAULT_HEADPHONE, help="Headphone model name")
    parser.add_argument("--offset", type=float, default=DEFAULT_OFFSET, help="Microphone calibration offset in dB")
    parser.add_argument("--max_step", type=int, default=DEFAULT_MAX_STEP, help="Max volume step on device")
    parser.add_argument("--output_csv", default="data/calibration.csv", help="Target CSV file path")
    parser.add_argument("--mock", action="store_true", help="Use simulated acoustic capture (for headless/testing)")
    parser.add_argument("--batch", action="store_true", help="Run full 15-step x 4-signal sweep automatically")
    parser.add_argument("--device", type=int, default=None, help="Audio input device index")
    parser.add_argument("--duration", type=float, default=SECONDS, help="Recording duration in seconds")
    parser.add_argument("--output", type=str, default=None, help="Save quick one-shot WAV recording to file")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    os.makedirs("recordings", exist_ok=True)

    # Check hardware sounddevice if not mocking
    use_mock = args.mock
    if not use_mock:
        try:
            import sounddevice as sd
            # Test if input device is available
            default_in = sd.query_devices(kind='input')
            print(f"[Audio Hardware] Found input device: {default_in['name']}")
        except Exception as e:
            print(f"[Notice] Physical audio input unavailable ({e}). Defaulting to mock mode.")
            use_mock = True

    # Quick one-shot recording mode
    if args.output:
        rec_dur = args.duration if args.duration else 5.0
        print(f"🎙️ Recording {rec_dur:.1f}s from physical microphone...")
        if use_mock:
            audio = get_simulated_audio("pink", 8, args.max_step, seconds=rec_dur)
        else:
            import sounddevice as sd
            rec_raw = sd.rec(int(rec_dur * FS), samplerate=FS, channels=1, dtype="float32", device=args.device)
            sd.wait()
            audio = rec_raw[:, 0]
        
        peak = np.max(np.abs(audio))
        dba = dba_from_audio(audio, FS, args.offset)
        sf.write(args.output, audio, FS)
        print(f"✅ Saved live recording to: {args.output}")
        print(f"🔊 Measured Sound Pressure Level: {dba:.1f} dBA (Peak: {peak:.3f})")
        return

    csv_path = args.output_csv
    new = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    rec_id = 1
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            rec_id = len(lines)  # line count includes header, so new row gets line count

    print("=" * 60)
    print("EchoGuard Calibration Recording Logger (Role 2)")
    print(f"Headphone: {args.headphone} | Max Step: {args.max_step} | Mic Offset: {args.offset:.1f} dB")
    print(f"Destination: {csv_path} | Capture Mode: {'MOCK/SYNTHETIC' if use_mock else 'PHYSICAL MIC'}")
    print("=" * 60)

    signals = ["tone1k", "pink", "music1", "music2"]

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["recording_id", "date", "headphone", "signal", "volume_step", "max_step", "dba", "note"])
            f.flush()

        if args.batch:
            print(f"Running automated batch sweep: {args.max_step} steps x {len(signals)} signals...")
            for step in range(1, args.max_step + 1):
                for sig in signals:
                    audio = get_simulated_audio(sig, step, args.max_step)
                    wav_name = f"recordings/{args.headphone}_{sig}_s{step}.wav"
                    sf.write(wav_name, audio, FS)
                    dba = dba_from_audio(audio, FS, args.offset)
                    peak = np.max(np.abs(audio))
                    note = "batch_calibrated"
                    w.writerow([rec_id, datetime.date.today().isoformat(), args.headphone, sig, step, args.max_step, round(dba, 1), note])
                    f.flush()
                    print(f"[{rec_id:03d}] Step {step:2d}/{args.max_step} | {sig:<8} -> {dba:5.1f} dBA (Peak: {peak:.2f})")
                    rec_id += 1
            print("Batch sweep complete!")
            return

        # Interactive loop
        while True:
            step_in = input("\nVolume step (1 to %d, or 'q' to quit): " % args.max_step).strip()
            if step_in.lower() == "q":
                break
            try:
                step = int(step_in)
                if step < 0 or step > args.max_step:
                    print(f"Please enter a step between 0 and {args.max_step}")
                    continue
            except ValueError:
                print("Invalid input.")
                continue

            sig_in = input("Signal (tone1k / pink / music1 / music2): ").strip().lower()
            if sig_in not in signals:
                print(f"Unknown signal. Choose from: {', '.join(signals)}")
                continue

            input("Start playback on phone/device, then press Enter to record 8s...")
            print("Recording 8 seconds...")

            if use_mock:
                audio = get_simulated_audio(sig_in, step, args.max_step)
            else:
                import sounddevice as sd
                rec_raw = sd.rec(int(SECONDS * FS), samplerate=FS, channels=1, dtype="float32")
                sd.wait()
                audio = rec_raw[:, 0]

            peak = np.max(np.abs(audio))
            if peak >= 0.95:
                print(f"WARNING: Audio clipped! Peak was {peak:.3f} >= 0.95. Please lower mic gain and re-record.")
                retry = input("Save anyway? (y/N): ").strip().lower()
                if retry != "y":
                    continue

            wav_name = f"recordings/{args.headphone}_{sig_in}_s{step}.wav"
            sf.write(wav_name, audio, FS)

            dba = dba_from_audio(audio, FS, args.offset)
            note_in = input("Note (optional, press Enter to skip): ").strip()

            w.writerow([rec_id, datetime.date.today().isoformat(), args.headphone, sig_in, step, args.max_step, round(dba, 1), note_in])
            f.flush()
            print(f"Saved recording {rec_id} -> {round(dba, 1)} dBA (WAV: {wav_name})")
            rec_id += 1

if __name__ == "__main__":
    main()
