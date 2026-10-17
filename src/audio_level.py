import numpy as np
from scipy.signal import bilinear, lfilter

def a_weighting(fs: int):
    """
    Constructs analog A-weighting filter coefficients and converts them
    to digital domain via bilinear transformation.
    """
    f1, f2, f3, f4, a1000 = 20.598997, 107.65265, 737.86223, 12194.217, 1.9997
    nums = [(2 * np.pi * f4) ** 2 * (10 ** (a1000 / 20)), 0, 0, 0, 0]
    dens = np.polymul([1, 4 * np.pi * f4, (2 * np.pi * f4) ** 2],
                      [1, 4 * np.pi * f1, (2 * np.pi * f1) ** 2])
    dens = np.polymul(np.polymul(dens, [1, 2 * np.pi * f3]), [1, 2 * np.pi * f2])
    return bilinear(nums, dens, fs)

def dba_from_audio(audio: np.ndarray, fs: int, offset_db: float = 0.0) -> float:
    """
    Calculates the A-weighted sound level (dBA) of an audio signal.
    
    Parameters:
        audio: 1D numpy array representing the raw audio waveform
        fs: Sampling rate in Hz
        offset_db: Microphone calibration offset in dB (measured by Role 2)
        
    Returns:
        dBA level as a float
    """
    b, a = a_weighting(fs)
    y = lfilter(b, a, audio)
    rms = np.sqrt(np.mean(y ** 2))
    return float(20 * np.log10(rms + 1e-12) + offset_db)
