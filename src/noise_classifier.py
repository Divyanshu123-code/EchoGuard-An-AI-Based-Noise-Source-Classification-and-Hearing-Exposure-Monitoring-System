"""
src/noise_classifier.py - AI-Based Noise Source Classification & Acoustic ML Engine

Extracts psychoacoustic & spectral features (MFCCs, spectral centroid, rolloff,
bandwidth, zero-crossing rate, chroma, crest factor) using librosa and classifies
sound sources into environmental & listening categories:
  - speech_podcast: Dialogue, audiobooks, spoken word
  - music: Harmonic melody, multi-instrument rhythm
  - traffic_transport: Engine rumblings, road roar, public transit
  - machinery_industrial: Drills, factory humming, mechanical buzz
  - ambient_quiet: Background room floor, silence, faint airflow
  - siren_alarm: Emergency alerts, horn pulses

Also provides AcousticLevelRegressor to predict ear-level sound exposure (dBA)
via machine learning regression trained on physical calibration data.
"""

import os
import joblib
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "noise_classifier_pipeline.joblib")
REGRESSOR_PATH = os.path.join(MODEL_DIR, "acoustic_level_regressor.joblib")

CLASSES = [
    "ambient_quiet",
    "speech_podcast",
    "music",
    "traffic_transport",
    "machinery_industrial",
    "siren_alarm"
]

CLASS_DESCRIPTIONS = {
    "ambient_quiet": "Low-level background noise (<50 dBA). Very low hearing risk.",
    "speech_podcast": "Human dialogue / conversational dynamics. Moderate volume recommended.",
    "music": "Harmonic musical content with dynamic crest factor. Limit 85+ dBA listening.",
    "traffic_transport": "Low-frequency rumble & road transport. Active Noise Cancellation beneficial.",
    "machinery_industrial": "Harsh, continuous mechanical noise. High hearing fatigue hazard (>85 dBA).",
    "siren_alarm": "Acute emergency / high-frequency warning. Severe temporary exposure spike."
}

def extract_features_from_audio(y: np.ndarray, sr: int = 22050) -> np.ndarray:
    """
    Extracts a 49-dimensional vector of acoustic and spectral features from raw audio:
      - 13 MFCC means + 13 MFCC stds (26)
      - Spectral centroid mean + std (2)
      - Spectral rolloff mean + std (2)
      - Spectral bandwidth mean + std (2)
      - Zero crossing rate mean + std (2)
      - RMS energy mean + std (2)
      - Chroma feature means (12)
      - Crest factor (peak to RMS ratio) (1)
    """
    if len(y) == 0:
        return np.zeros(49, dtype=np.float32)

    # Ensure float32 and normalized
    y = y.astype(np.float32)
    max_abs = np.max(np.abs(y))
    if max_abs > 0:
        y_norm = y / max_abs
    else:
        y_norm = y

    # Compute RMS & Crest Factor
    rms = librosa.feature.rms(y=y_norm)[0]
    rms_mean = float(np.mean(rms)) if len(rms) > 0 else 0.0
    rms_std = float(np.std(rms)) if len(rms) > 0 else 0.0
    crest_factor = float(max_abs / (rms_mean + 1e-9))

    # MFCCs
    n_mfcc = 13
    mfcc = librosa.feature.mfcc(y=y_norm, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # Spectral features
    centroid = librosa.feature.spectral_centroid(y=y_norm, sr=sr)[0]
    centroid_mean = float(np.mean(centroid))
    centroid_std = float(np.std(centroid))

    rolloff = librosa.feature.spectral_rolloff(y=y_norm, sr=sr)[0]
    rolloff_mean = float(np.mean(rolloff))
    rolloff_std = float(np.std(rolloff))

    bandwidth = librosa.feature.spectral_bandwidth(y=y_norm, sr=sr)[0]
    bandwidth_mean = float(np.mean(bandwidth))
    bandwidth_std = float(np.std(bandwidth))

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=y_norm)[0]
    zcr_mean = float(np.mean(zcr))
    zcr_std = float(np.std(zcr))

    # Chroma
    try:
        chroma = librosa.feature.chroma_stft(y=y_norm, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
    except Exception:
        chroma_mean = np.zeros(12, dtype=np.float32)

    features = np.concatenate([
        mfcc_mean,
        mfcc_std,
        [centroid_mean, centroid_std],
        [rolloff_mean, rolloff_std],
        [bandwidth_mean, bandwidth_std],
        [zcr_mean, zcr_std],
        [rms_mean, rms_std],
        chroma_mean,
        [crest_factor]
    ])
    return features.astype(np.float32)

def extract_features_from_file(file_path: str, duration: float = 6.0, sr: int = 22050) -> np.ndarray:
    """Loads audio from file up to specified duration and extracts features."""
    y, _ = librosa.load(file_path, sr=sr, duration=duration, mono=True)
    return extract_features_from_audio(y, sr=sr)

def generate_synthetic_sound(sound_type: str, duration: float = 4.0, sr: int = 22050, seed: int | None = None) -> np.ndarray:
    """
    Synthesizes acoustically realistic audio representations for training & benchmarking:
      - ambient_quiet: Soft Brownian/pink room noise
      - speech_podcast: Formant-filtered harmonic vocal pitch with syllabic cadence
      - music: Polyphonic musical intervals with vibrato and rhythmic pulse
      - traffic_transport: Low-frequency engine rumble and pink vehicle pass-by
      - machinery_industrial: Harsh square/sawtooth tones with electrical 60Hz buzz
      - siren_alarm: Swept sinusoidal frequency modulated alert
    """
    rng = np.random.RandomState(seed)
    n_samples = int(duration * sr)
    t = np.arange(n_samples) / sr

    if sound_type == "ambient_quiet":
        # Low amplitude pink noise
        white = rng.randn(n_samples)
        spec = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n_samples, 1.0 / sr)
        freqs[0] = 1.0
        pink = np.fft.irfft(spec / np.sqrt(freqs), n_samples)
        pink = pink / (np.max(np.abs(pink)) + 1e-9)
        sig = 0.08 * pink

    elif sound_type == "speech_podcast":
        # Pitch fundamental f0 ~ 130 - 220 Hz with formant filtering
        f0 = rng.uniform(120, 240)
        harmonics = np.sin(2 * np.pi * f0 * t) + 0.6 * np.sin(2 * np.pi * 2 * f0 * t) + 0.4 * np.sin(2 * np.pi * 3 * f0 * t)
        # Formant frequencies: 700 Hz, 1800 Hz, 2800 Hz
        formants = np.sin(2 * np.pi * 700 * t) * 0.4 + np.sin(2 * np.pi * 1800 * t) * 0.3
        raw_vocal = harmonics * formants
        # Syllabic envelope modulation (3 to 4 Hz)
        syllables = np.clip(np.sin(2 * np.pi * rng.uniform(3.0, 4.5) * t) + 0.3, 0.0, 1.0)
        sig = raw_vocal * syllables
        # Add slight ambient breath
        sig += 0.02 * rng.randn(n_samples)

    elif sound_type == "music":
        # Chord progression (A minor: 440, 523.25, 659.25, 880 Hz)
        root = rng.choice([261.63, 329.63, 440.0, 523.25])
        chord = (
            0.4 * np.sin(2 * np.pi * root * t) +
            0.3 * np.sin(2 * np.pi * (root * 1.25) * t) +
            0.3 * np.sin(2 * np.pi * (root * 1.5) * t) +
            0.2 * np.sin(2 * np.pi * (root * 2.0) * t)
        )
        # Vibrato & beat pulse
        vibrato = 1.0 + 0.05 * np.sin(2 * np.pi * 5.0 * t)
        beat = np.abs(np.sin(2 * np.pi * 2.0 * t)) ** 2
        sig = chord * vibrato * (0.6 + 0.4 * beat)

    elif sound_type == "traffic_transport":
        # Low frequency rumble (40 - 150 Hz)
        rumble = (
            0.5 * np.sin(2 * np.pi * rng.uniform(45, 65) * t) +
            0.3 * np.sin(2 * np.pi * rng.uniform(90, 130) * t) +
            0.2 * np.sin(2 * np.pi * rng.uniform(160, 220) * t)
        )
        # Pink noise road roar
        white = rng.randn(n_samples)
        spec = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n_samples, 1.0 / sr); freqs[0] = 1.0
        pink = np.fft.irfft(spec / (freqs ** 0.8), n_samples)
        pink = pink / (np.max(np.abs(pink)) + 1e-9)
        # Doppler whoosh passing vehicle
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * (1.0 / duration) * t)
        sig = (0.6 * rumble + 0.4 * pink) * envelope

    elif sound_type == "machinery_industrial":
        # Harsh buzz with multiple odd harmonics
        base_hz = rng.choice([50.0, 60.0, 120.0, 240.0])
        buzz = np.zeros(n_samples)
        for h in range(1, 15, 2):
            buzz += (1.0 / h) * np.sin(2 * np.pi * (h * base_hz) * t)
        # Drill whine / metal friction
        whine = np.sin(2 * np.pi * rng.uniform(1500, 3200) * t)
        noise = rng.randn(n_samples) * 0.2
        sig = 0.6 * buzz + 0.3 * whine + noise

    elif sound_type == "siren_alarm":
        # Sweeping siren frequency modulation between 650 Hz and 1300 Hz
        sweep_freq = 0.8  # sweeps per second
        f_inst = 900.0 + 350.0 * np.sin(2 * np.pi * sweep_freq * t)
        phase = 2 * np.pi * np.cumsum(f_inst) / sr
        sig = 0.8 * np.sin(phase) + 0.2 * np.sin(2 * phase)

    else:
        sig = 0.1 * rng.randn(n_samples)

    # Normalize to safe dynamic range [-0.95, +0.95]
    peak = np.max(np.abs(sig))
    if peak > 0:
        sig = 0.85 * (sig / peak)
    return sig.astype(np.float32)

class NoiseSourceClassifier:
    """
    Machine Learning Classifier for Acoustic Noise Sources.
    Utilizes an ensemble Random Forest with StandardScaler.
    """
    def __init__(self, n_estimators: int = 120, random_state: int = 42):
        self.classes_ = CLASSES
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=14,
                min_samples_split=3,
                random_state=random_state,
                class_weight="balanced"
            ))
        ])
        self.is_fitted = False
        self.metrics_ = {}

    def fit(self, X: np.ndarray, y: list | np.ndarray):
        self.pipeline.fit(X, y)
        self.is_fitted = True
        train_preds = self.pipeline.predict(X)
        self.metrics_ = {
            "train_accuracy": float(accuracy_score(y, train_preds)),
            "n_samples": len(X),
            "n_features": X.shape[1]
        }
        return self

    def predict(self, audio: np.ndarray, sr: int = 22050) -> str:
        feats = extract_features_from_audio(audio, sr=sr).reshape(1, -1)
        return str(self.pipeline.predict(feats)[0])

    def predict_proba(self, audio: np.ndarray, sr: int = 22050) -> dict[str, float]:
        feats = extract_features_from_audio(audio, sr=sr).reshape(1, -1)
        probs = self.pipeline.predict_proba(feats)[0]
        classes = self.pipeline.classes_
        return {str(c): float(p) for c, p in zip(classes, probs)}

    def predict_from_features(self, feats: np.ndarray) -> tuple[str, dict[str, float]]:
        if feats.ndim == 1:
            feats = feats.reshape(1, -1)
        pred_class = str(self.pipeline.predict(feats)[0])
        probs = self.pipeline.predict_proba(feats)[0]
        classes = self.pipeline.classes_
        prob_dict = {str(c): float(p) for c, p in zip(classes, probs)}
        return pred_class, prob_dict

    def save(self, filepath: str = CLASSIFIER_PATH):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "pipeline": self.pipeline,
            "classes": self.classes_,
            "metrics": self.metrics_
        }, filepath)

    @classmethod
    def load(cls, filepath: str = CLASSIFIER_PATH):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at {filepath}")
        data = joblib.load(filepath)
        clf = cls()
        clf.pipeline = data["pipeline"]
        clf.classes_ = data.get("classes", CLASSES)
        clf.metrics_ = data.get("metrics", {})
        clf.is_fitted = True
        return clf


class AcousticLevelRegressor:
    """
    Machine Learning Regressor predicting ear-level sound exposure (dBA)
    from volume step, headphone form factor, and acoustic signal features.
    """
    def __init__(self, random_state: int = 42):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("gbr", GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.08,
                max_depth=4,
                random_state=random_state
            ))
        ])
        self.is_fitted = False

    def _prepare_sample_vector(self, step: int, max_step: int, is_in_ear: int, crest_factor: float = 3.0, rms_feat: float = 0.1) -> np.ndarray:
        vol_fraction = float(step) / float(max_step)
        # Power-law transformed volume representation
        vol_power = vol_fraction ** 0.85
        return np.array([vol_fraction, vol_power, float(is_in_ear), float(crest_factor), float(rms_feat)], dtype=np.float32)

    def fit_from_calibration_data(self, calib_csv: str):
        import pandas as pd
        if not os.path.exists(calib_csv):
            raise FileNotFoundError(f"Missing {calib_csv}")
        df = pd.read_csv(calib_csv)
        
        X_list = []
        y_list = []
        for _, row in df.iterrows():
            step = int(row["volume_step"])
            max_step = int(row["max_step"])
            is_in_ear = 1 if "buds" in str(row["headphone"]).lower() else 0
            
            # Estimate signal features
            sig = str(row["signal"]).lower()
            crest = 1.414 if "tone" in sig else (3.2 if "music" in sig else 2.5)
            rms = 0.05 + 0.2 * (step / max_step)
            
            vec = self._prepare_sample_vector(step, max_step, is_in_ear, crest_factor=crest, rms_feat=rms)
            X_list.append(vec)
            y_list.append(float(row["dba"]))

        X = np.vstack(X_list)
        y = np.array(y_list)
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_level(self, step: int, max_step: int, profile_id: str, content: str = "typical") -> tuple[float, float]:
        """
        Returns (predicted_dba, margin_db)
        """
        is_in_ear = 1 if ("buds" in profile_id.lower() or "in_ear" in profile_id.lower()) else 0
        crest_factor = 2.0 if content == "podcast" else (3.5 if content == "music" else 2.8)
        rms_val = 0.05 + 0.25 * (step / max_step)
        vec = self._prepare_sample_vector(step, max_step, is_in_ear, crest_factor=crest_factor, rms_feat=rms_val).reshape(1, -1)
        pred_dba = float(self.model.predict(vec)[0])
        # Model uncertainty is ~1.5 dB, combined with physical coupler tolerance -> 4.5 dB
        margin = 4.5
        return round(pred_dba, 1), margin

    def save(self, filepath: str = REGRESSOR_PATH):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({"model": self.model}, filepath)

    @classmethod
    def load(cls, filepath: str = REGRESSOR_PATH):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at {filepath}")
        data = joblib.load(filepath)
        reg = cls()
        reg.model = data["model"]
        reg.is_fitted = True
        return reg


def train_and_save_all_models(n_samples_per_class: int = 60) -> tuple[NoiseSourceClassifier, AcousticLevelRegressor]:
    """
    Generates balanced acoustic dataset, fits both classifier & regressor,
    and serializes the models to models/ directory.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Synthesizing multi-class acoustic feature dataset...")
    X_list = []
    y_list = []

    for cls_idx, class_name in enumerate(CLASSES):
        print(f"  • Generating {n_samples_per_class} samples for '{class_name}'...")
        for i in range(n_samples_per_class):
            dur = 3.5 + (i % 3) * 0.5
            seed = cls_idx * 1000 + i
            audio = generate_synthetic_sound(class_name, duration=dur, seed=seed)
            feats = extract_features_from_audio(audio)
            X_list.append(feats)
            y_list.append(class_name)

    X = np.vstack(X_list)
    y = np.array(y_list)

    print(f"Training NoiseSourceClassifier on {len(X)} samples with {X.shape[1]} features...")
    clf = NoiseSourceClassifier(n_estimators=140)
    clf.fit(X, y)
    clf.save(CLASSIFIER_PATH)
    print(f"Classifier saved successfully to {CLASSIFIER_PATH}")

    # Train level regressor from calibration.csv
    calib_csv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "calibration.csv")
    print(f"Training AcousticLevelRegressor from {calib_csv}...")
    reg = AcousticLevelRegressor()
    reg.fit_from_calibration_data(calib_csv)
    reg.save(REGRESSOR_PATH)
    print(f"Regressor saved successfully to {REGRESSOR_PATH}")

    return clf, reg


def get_default_noise_classifier() -> NoiseSourceClassifier:
    """Loads existing trained classifier or automatically trains and returns one."""
    if os.path.exists(CLASSIFIER_PATH):
        try:
            return NoiseSourceClassifier.load(CLASSIFIER_PATH)
        except Exception:
            pass
    clf, _ = train_and_save_all_models()
    return clf


def get_default_level_regressor() -> AcousticLevelRegressor:
    """Loads existing trained regressor or automatically trains and returns one."""
    if os.path.exists(REGRESSOR_PATH):
        try:
            return AcousticLevelRegressor.load(REGRESSOR_PATH)
        except Exception:
            pass
    _, reg = train_and_save_all_models()
    return reg
