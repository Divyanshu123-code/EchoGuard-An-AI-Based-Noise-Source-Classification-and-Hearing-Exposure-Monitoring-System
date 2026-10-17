import numpy as np
import pytest
from src.audio_level import dba_from_audio, a_weighting
from scipy.signal import lfilter

def test_a_weighting_1khz_and_100hz():
    fs = 44100
    t = np.arange(2 * fs) / fs
    
    # 1 kHz test tone
    tone_1k = np.sin(2 * np.pi * 1000 * t)
    raw_rms_1k = 20 * np.log10(np.sqrt(np.mean(tone_1k ** 2)))
    dba_1k = dba_from_audio(tone_1k, fs, offset_db=0.0)
    # At 1 kHz, A-weighting filter response is approximately 0 dB
    assert abs(dba_1k - raw_rms_1k) < 0.5
    
    # 100 Hz test tone
    tone_100 = np.sin(2 * np.pi * 100 * t)
    raw_rms_100 = 20 * np.log10(np.sqrt(np.mean(tone_100 ** 2)))
    dba_100 = dba_from_audio(tone_100, fs, offset_db=0.0)
    # At 100 Hz, A-weighting attenuation is approximately -19.1 dB
    diff_100 = dba_100 - raw_rms_100
    assert diff_100 == pytest.approx(-19.1, abs=1.0)
