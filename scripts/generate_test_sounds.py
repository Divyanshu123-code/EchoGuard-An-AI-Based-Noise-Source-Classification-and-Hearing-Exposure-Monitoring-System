#!/usr/bin/env python3
"""
scripts/generate_test_sounds.py

Synthesizes standard audio calibration stimuli for EchoGuard:
- tone1k.wav: 1 kHz sine tone at -6 dBFS (0.5 peak amplitude), 10 seconds
- pink.wav: Frequency-filtered pink noise (1/f spectral roll-off), 10 seconds
- music1.wav: Synthetic acoustic multi-harmonic melody (representative typical listening)
- music2.wav: Dynamic synthetic bass/percussive chord sequence (dynamic music)

All files sampled at 44.1 kHz, 16-bit PCM WAV.
"""

import os
import numpy as np
import soundfile as sf

FS = 44100
DURATION = 10  # seconds

def generate_tone1k(output_path: str = "tone1k.wav"):
    """Generate a clean 1 kHz reference tone."""
    t = np.arange(DURATION * FS) / FS
    audio = 0.5 * np.sin(2 * np.pi * 1000 * t)
    sf.write(output_path, audio.astype(np.float32), FS)
    print(f"Generated: {output_path} (1 kHz sine, 10s, peak={np.max(np.abs(audio)):.2f})")

def generate_pink_noise(output_path: str = "pink.wav"):
    """Generate 1/f pink noise via frequency-domain shaping."""
    n = DURATION * FS
    spec = np.fft.rfft(np.random.randn(n))
    freqs = np.fft.rfftfreq(n, 1 / FS)
    freqs[0] = 1.0  # Prevent division by zero at DC
    pink_spec = spec / np.sqrt(freqs)
    pink = np.fft.irfft(pink_spec, n)
    pink = 0.5 * pink / np.max(np.abs(pink))
    sf.write(output_path, pink.astype(np.float32), FS)
    print(f"Generated: {output_path} (Pink noise, 10s, peak={np.max(np.abs(pink)):.2f})")

def generate_synthetic_music1(output_path: str = "music1.wav"):
    """
    Generate synthetic harmonic melodic clip representing acoustic music.
    Uses A minor / C major chord progression with fundamental + overtones.
    """
    t = np.arange(DURATION * FS) / FS
    chords = [
        [220.0, 261.63, 329.63],  # Am (A3, C4, E4)
        [174.61, 220.0, 261.63],  # F (F3, A3, C4)
        [261.63, 329.63, 392.00], # C (C4, E4, G4)
        [196.00, 246.94, 293.66], # G (G3, B3, D4)
    ]
    seg_len = int(2.5 * FS)
    audio = np.zeros(DURATION * FS)

    for i, chord in enumerate(chords):
        idx_start = i * seg_len
        idx_end = min((i + 1) * seg_len, DURATION * FS)
        sub_t = t[idx_start:idx_end]
        seg_audio = np.zeros(idx_end - idx_start)
        for f in chord:
            seg_audio += 0.4 * np.sin(2 * np.pi * f * sub_t)
            seg_audio += 0.2 * np.sin(2 * np.pi * 2 * f * sub_t)
            seg_audio += 0.1 * np.sin(2 * np.pi * 3 * f * sub_t)
        
        env = np.ones_like(seg_audio)
        fade_len = int(0.05 * FS)
        env[:fade_len] = np.linspace(0, 1, fade_len)
        env[-fade_len:] = np.linspace(1, 0, fade_len)
        audio[idx_start:idx_end] = seg_audio * env

    audio = 0.5 * audio / np.max(np.abs(audio))
    sf.write(output_path, audio.astype(np.float32), FS)
    print(f"Generated: {output_path} (Synthetic music 1, 10s, peak={np.max(np.abs(audio)):.2f})")

def generate_synthetic_music2(output_path: str = "music2.wav"):
    """
    Generate synthetic rhythmic / bass-heavy electronic music clip.
    Features rhythmic pulsing sub-bass + synth lead.
    """
    t = np.arange(DURATION * FS) / FS
    audio = np.zeros(DURATION * FS)
    bpm = 120
    beat_dur = 60.0 / bpm  # 0.5s
    num_beats = int(DURATION / beat_dur)

    for b in range(num_beats):
        b_start = int(b * beat_dur * FS)
        b_end = min(int((b + 1) * beat_dur * FS), DURATION * FS)
        b_t = t[b_start:b_end] - (b * beat_dur)

        kick_freq = 45.0 + 80.0 * np.exp(-b_t * 25.0)
        kick = np.sin(2 * np.pi * kick_freq * b_t) * np.exp(-b_t * 6.0)

        lead_freq = 440.0 if (b % 4 == 0 or b % 4 == 2) else 554.37
        lead = 0.3 * np.sin(2 * np.pi * lead_freq * b_t) * np.exp(-b_t * 3.0)

        audio[b_start:b_end] = kick + lead

    audio = 0.5 * audio / np.max(np.abs(audio))
    sf.write(output_path, audio.astype(np.float32), FS)
    print(f"Generated: {output_path} (Synthetic music 2, 10s, peak={np.max(np.abs(audio)):.2f})")

def main():
    target_dir = "test_sounds"
    os.makedirs(target_dir, exist_ok=True)
    generate_tone1k(os.path.join(target_dir, "tone1k.wav"))
    generate_pink_noise(os.path.join(target_dir, "pink.wav"))
    generate_synthetic_music1(os.path.join(target_dir, "music1.wav"))
    generate_synthetic_music2(os.path.join(target_dir, "music2.wav"))
    print("\nAll 4 test sounds successfully synthesized in 'test_sounds/' directory.")

if __name__ == "__main__":
    main()
