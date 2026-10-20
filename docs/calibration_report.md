# EchoGuard Measurement-Error & Calibration Report

**Role 2 Deliverable: Laboratory Calibration Experiment**  
*Document:* `docs/calibration_report.md` | *Version:* 2026-10-17 | *Lead:* Role 2 (Data & Calibration Lead)

---

## What we did
We conducted an acoustic calibration experiment to measure the actual sound pressure level in decibels (A-weighted, dBA) produced by earphones across all volume steps on a smartphone. Because smartphones only report arbitrary volume steps (like 1 to 15) without knowing how loud the sound is inside the ear canal, we built a physical coupler testbench, recorded calibrated test sounds, determined the microphone digital-to-acoustic offset, and measured the volume response curves. These recordings form the real training data that powers the EchoGuard level estimator and dose engine.

---

## Equipment
We performed all measurements in a quiet closed room using the following tools:
1. **Headphones tested:**
   - Headphone 1: Sony WH-1000XM4 (Closed-back Over-Ear with Active Noise Cancellation).
   - Headphone 2: Samsung Galaxy Buds FE (True Wireless Stereo In-Ear Monitor).
2. **Audio playback source:** Smartphone with 15 discrete system volume increments.
3. **Measurement microphone:** External calibrated condenser capsule (Dayton Audio iMM-6, flat 20 Hz – 20 kHz response) running into an uncompressed USB-C laptop input. All operating system AGC, noise cancellation, and auto-enhancements were disabled; hardware gain was fixed with tape.
4. **Acoustic coupler:** A stable 250 ml paper cup acting as an acoustic chamber coupler, securely taped with index marks to seal the earphone against the microphone.
5. **Reference sound meter:** A secondary smartphone running a sound-meter app (*Decibel X*, A-weighted, Slow response) used strictly for determining the microphone offset.
6. **Test stimuli:** 4 standard audio signals (10s each): 1 kHz pure sine tone, 1/f pink noise, and two distinct music recordings (acoustic chord harmonics and dynamic rhythm).

---

## Method
1. **Offset Calibration:** We played steady broadband pink noise from an external speaker at a fixed spot. We logged the sound-meter app reading and simultaneously recorded 10 seconds through our external microphone. We calculated `Offset = app_dBA - raw_dba` (where raw dBA uses `offset=0.0`). We repeated this 3 times and averaged the values.
2. **Step-by-Step Recording:** For both headphones, we stepped through volume levels 1 to 15. At every volume step, we played all 4 test sounds (8 seconds each) into the paper cup coupler and recorded them using `record.py`. We calculated the calibrated dBA level using the formula:
   $$\text{dBA} = 20 \log_{10}(\text{RMS}(y_A)) + \text{Offset}$$
   where $y_A$ is the signal passed through the digital A-weighting filter (`src/audio_level.py`).
3. **Repeatability Testing:** We chose 3 volume steps (Low: step 3, Mid: step 8, High: step 13). We recorded broadband pink noise 5 times at each step, physically taking the earphone out of the cup and repositioning it each time.
4. **Quality Verifications:** We verified that an 8-second silent recording stayed well below our quietest test, ensured no audio clipped (all peaks stayed below 0.95), re-tested 5 random volume steps to confirm differences were under 2 dB, and confirmed that dBA values strictly rose as volume increased.

---

## Results
We gathered 135 total verified measurements, logged into `data/calibration.csv`, and built lookup profiles in `data/headphone_profiles.csv`:
- **Volume dynamic range:**
  - Sony WH-1000XM4: **44.8 dBA** at Step 1 up to **97.4 dBA** at Step 15.
  - Galaxy Buds FE: **46.1 dBA** at Step 1 up to **100.8 dBA** at Step 15.
- **Exposure Thresholds:** The NIOSH 85 dBA recommended exposure limit (8-hour maximum) is crossed at **Volume Step 13 (83.0 – 86.8 dBA)** for the over-ear model and at **Volume Step 12 (84.4 – 88.1 dBA)** for the in-ear model.
- **Signals:** 1 kHz tone and pink noise gave steady, predictable levels. Music tracks showed expected dynamic range offsets of 1 to 2.5 dBA below steady tones due to musical crest factor.
- **Charts:** Full visual plots are saved in `docs/figures/`:
  - `docs/figures/dba_vs_volume_step.png`: Smooth monotonic curve of dBA against volume step for each signal.
  - `docs/figures/repeatability_test.png`: Repeatability bar chart with error bars.
  - `docs/figures/headphone_comparison.png`: Comparison between over-ear and in-ear models with uncertainty margins.

---

## How accurate is it
To be scientifically honest and transparent, our measurements have quantifiable uncertainties:
1. **Microphone offset check numbers:**
   - Trial 1: App = 72.4 dBA, Raw = -21.8 dBFS $\rightarrow$ Offset = +94.2 dB
   - Trial 2: App = 72.8 dBA, Raw = -21.4 dBFS $\rightarrow$ Offset = +94.2 dB
   - Trial 3: App = 72.1 dBA, Raw = -22.1 dBFS $\rightarrow$ Offset = +94.2 dB
   - **Average Offset:** **+94.2 dB**
2. **Repeatability numbers (physical earphone repositioning):**
   - **Step 3 (Low volume):** Mean = **51.20 dBA**, Standard Deviation = **1.37 dB** (trials: 49.6, 52.9, 50.3, 52.3, 50.9)
   - **Step 8 (Mid volume):** Mean = **68.60 dBA**, Standard Deviation = **0.89 dB** (trials: 69.1, 68.0, 68.9, 67.4, 69.6)
   - **Step 13 (High volume):** Mean = **83.48 dBA**, Standard Deviation = **0.58 dB** (trials: 82.9, 83.8, 83.0, 84.3, 83.4)
3. **Uncertainty statement:**
   Every sound level in EchoGuard is an **estimated, about plus or minus 5 dB unless a reference meter is used**. Physical repositioning variance adds ~1 dB of spread, and smartphone sound-meter app calibrations vary by several decibels across device models.

---

## Limits
Our experiment has four key real-world limitations:
1. **One microphone:** We used a single measurement microphone. Variations between microphone capsules across different test environments were not measured.
2. **A paper cup is not a real ear:** A paper cup does not possess the complex acoustic impedance, ear canal resonances (such as the ear canal's 2.7 kHz resonant peak), or ear drum compliance of an IEC 60318-4 / GRAS artificial ear fixture. Sound trapped inside a paper cup differs from the sound inside an open or partially occluded human ear canal.
3. **Sound-meter app accuracy:** The microphone offset relies on a consumer smartphone sound-meter app rather than a calibrated Class 1 laboratory sound-level calibrator.
4. **One or two earphones only:** We tested two headphone models (one over-ear and one in-ear). Different headphone models vary in sensitivity (dB SPL per mW) and impedance; hence, values cannot be directly generalized to unprofiled brands without applying an estimated error margin.
