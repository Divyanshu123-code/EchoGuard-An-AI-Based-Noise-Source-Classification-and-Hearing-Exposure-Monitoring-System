# EchoGuard Calibration Lab Notes

**Role 2: Data and Calibration Lead**  
*Document Version: 2026-10-17 | Last Updated: 2026-10-20*  
*Author / Role Lead:* Data and Calibration Lead (EchoGuard Team)

---

## 1. Physical Environment & Equipment Setup

### 1.1 Room Condition
- **Location:** Dedicated indoor quiet room (closed windows, interior door sealed).
- **HVAC / Environmental:** Fans, air conditioning, and extraneous electronics powered off.
- **Ambient Room Noise Floor:** Measured at **33.8 – 34.2 dBA** (using reference phone sound meter and confirmed via silence check recording).

### 1.2 Hardware & Microphone Configuration
- **External Measurement Microphone:** Dayton Audio iMM-6 calibrated measurement capsule (omnidirectional condenser, 3.5mm TRRS connected via uncompressed USB-C audio interface).
- **Audio Interface / Laptop Settings:**
  - OS Sound Settings: Enhancement features, audio ducking, spatial sound disabled.
  - Noise Suppression: Strictly disabled (**OFF**).
  - Automatic Gain Control (AGC): Strictly disabled (**OFF**).
  - Hardware Gain Knob: Locked at 50% index mark using physical tape. Never touched between calibration sessions.
- **Acoustic Coupler:**
  - 250 ml heavy-stock paper cup acoustic coupler.
  - Earphone sound port fixed directly against the microphone capsule diaphragm.
  - Sealed perimeter with acoustic putty and paper cup shroud to minimize room sound intrusion and standardize volume cavity coupling (~15 cm³ volume).
  - Position marking tape used on coupler rim to ensure repeatable placement.

---

## 2. Microphone Offset Calibration (Step 1.5)

### 2.1 Methodology
1. A secondary smartphone running a calibrated sound-level meter app (*Decibel X*, IEC 61672 Class 2 profile approximation, A-weighting, Slow response) was placed directly adjacent to the external microphone.
2. Steady-state broadband pink noise was played from an external reference loudspeaker at a fixed distance (1.0 meter) and steady SPL.
3. At the exact same time, 10 seconds of audio were recorded through the external microphone.
4. Raw uncalibrated digital level was computed with `dba_from_audio(audio, fs, offset_db=0.0)`.
5. Offset calculated as:
   $$\text{Offset} = \text{SPL}_{\text{app}} - \text{Raw}_{\text{dBA}}$$

### 2.2 Calibration Measurements
| Trial | Reference App Reading ($L_{A,\text{app}}$) | Uncalibrated Mic Reading ($\text{raw\_dba}$) | Calculated Offset ($\Delta\text{dB}$) |
|:---:|:---:|:---:|:---:|
| **Trial 1** | 72.4 dBA | -21.8 dBFS | +94.2 dB |
| **Trial 2** | 72.8 dBA | -21.4 dBFS | +94.2 dB |
| **Trial 3** | 72.1 dBA | -22.1 dBFS | +94.2 dB |
| **Average Offset** | **72.43 dBA** | **-21.77 dBFS** | **+94.2 dB** |

> **Calibration Constant:**  
> $$\text{OFFSET\_DB} = +94.2\text{ dB}$$

> **Honesty & Error Disclosure:**  
> Smartphone sound-meter microphones and commercial measurement capsules carry inherent tolerance limits (capsule frequency response tilt, coupling variations). All reported absolute sound levels represent estimates with an expected uncertainty margin of approximately **$\pm 5.0\text{ dB}$** unless calibrated against a certified Class 1 laboratory calibrator (such as a B&K 4231).

---

## 3. Data Collection Summary (Step 1.6 & Step 2.1)

Two distinct headphone form factors were measured across all 15 discrete system volume steps and 4 stimulus signals (`tone1k`, `pink`, `music1`, `music2`):

1. **Headphone 1 (Over-Ear ANC):** `sony_wh1000xm4`
   - Form factor: Closed-back dynamic over-ear
   - Discrete steps: 1 to 15 ($N = 60$ recordings)
   - Dynamic range: 44.8 dBA (Step 1) to 97.4 dBA (Step 15)
2. **Headphone 2 (In-Ear TWS):** `galaxy_buds_fe`
   - Form factor: Sealed canal in-ear monitor
   - Discrete steps: 1 to 15 ($N = 60$ recordings)
   - Dynamic range: 46.1 dBA (Step 1) to 100.8 dBA (Step 15)

Total baseline calibration measurements logged to [data/calibration.csv](file:///Users/divanshu/advanced%20ai%20project%20/data/calibration.csv): **120 rows**.

---

## 4. Quality Assurance Verifications (Step 1.7)

1. **Silence Check:**
   - 8 seconds of silent room recording (no device playback): **34.1 dBA**.
   - Result: Ambient floor is $> 10\text{ dB}$ below the lowest earphone volume step (44.8 dBA), ensuring clean signal-to-noise ratio.
2. **Clipping Check:**
   - Maximum absolute peak amplitude observed across all volume steps: **$0.86\text{ FS}$** (under $0.95$ safety threshold).
   - Zero clipped samples detected.
3. **Repeat Check:**
   - 5 randomly selected volume steps were re-recorded during Session 1:
     - Step 4 (Pink): 52.8 dBA vs 53.4 dBA ($\Delta = 0.6\text{ dB}$)
     - Step 7 (Tone1k): 64.9 dBA vs 64.2 dBA ($\Delta = 0.7\text{ dB}$)
     - Step 10 (Music1): 74.5 dBA vs 75.6 dBA ($\Delta = 1.1\text{ dB}$)
     - Step 12 (Music2): 84.8 dBA vs 84.1 dBA ($\Delta = 0.7\text{ dB}$)
     - Step 14 (Pink): 92.2 dBA vs 93.3 dBA ($\Delta = 1.1\text{ dB}$)
   - Result: All repeated recordings showed delta within $0.6 - 1.1\text{ dB}$ ($< 2.0\text{ dB}$ acceptance criterion).
4. **Monotonicity Check:**
   - Sound pressure level strictly increases monotonically with volume step index without dips or inversion anomalies.

---

## 5. Physical Repeatability Test (Step 2.2)

To quantify positioning uncertainty introduced by physical user coupling and earphone unseating/reseating, 5 independent trials were conducted at Low (Step 3), Mid (Step 8), and High (Step 13) volume steps using broadband pink noise, removing and replacing the earphone in the paper cup coupler between every trial:

| Volume Step | Benchmark Level | Trial 1 | Trial 2 | Trial 3 | Trial 4 | Trial 5 | Mean Level ($\bar{x}$) | Sample Std Dev ($s$) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Step 3 (Low)** | ~51 dBA | 49.6 | 52.9 | 50.3 | 52.3 | 50.9 | **51.20 dBA** | **1.37 dB** |
| **Step 8 (Mid)** | ~69 dBA | 69.1 | 68.0 | 68.9 | 67.4 | 69.6 | **68.60 dBA** | **0.89 dB** |
| **Step 13 (High)**| ~83 dBA | 82.9 | 83.8 | 83.0 | 84.3 | 83.4 | **83.48 dBA** | **0.58 dB** |

**Repeatability Findings:**
- Acoustic seal repositioning standard deviation ranges from **$0.58\text{ dB}$ to $1.37\text{ dB}$**.
- Low-volume steps demonstrate slightly higher relative variability due to ambient room floor fluctuations.
- The physical repeatability of the paper-cup coupler is well within the project's $\pm 5.0\text{ dB}$ conservative uncertainty budget.
