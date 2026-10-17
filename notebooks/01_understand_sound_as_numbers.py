"""
Exercise 1: Understand sound as numbers (Step 1.2 in Role 1 Tech Lead Guide).
"""
import numpy as np

fs = 22050
t = np.arange(4 * fs) / fs
x = 0.5 * np.sin(2 * np.pi * 440 * t)  # 4-second 440 Hz tone
print("Signal shape:", x.shape)  # expect (88200,)

def level_db(sig):
    return 20 * np.log10(np.sqrt(np.mean(sig ** 2)))

full_level = level_db(x)
half_level = level_db(x / 2)
diff = half_level - full_level

print(f"Level at full signal: {full_level:.2f} dB")
print(f"Level at half signal: {half_level:.2f} dB")
print(f"Difference: {diff:.2f} dB")
print("Explanation: Halving amplitude drops the RMS by half; 20 * log10(0.5) = -6.02 dB.")
