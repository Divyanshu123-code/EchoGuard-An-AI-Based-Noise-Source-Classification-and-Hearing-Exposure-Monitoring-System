# EchoGuard: An AI-Based Noise Source Classification and Hearing Exposure Monitoring System

**Advanced Artificial Intelligence Course Project**  
*Lead Roles: Tech Lead (Role 1), Data & Calibration Lead (Role 2), Machine Learning & AI Lead (Role 3)*

EchoGuard is an intelligent acoustic monitoring system that combines empirical acoustic coupler calibration, digital signal processing (DSP), machine learning noise source classification, and real-time macOS monitoring. It estimates sound pressure level reaching the human ear (dBA), tracks cumulative daily dosage against NIOSH and WHO-ITU exposure limits, and coaches users toward healthier listening habits without making medical claims.

---

## 🌟 Key Features

1. **AI-Based Noise Source Classification (Role 3):**
   - **UrbanSound8K 10-Fold Baseline Evaluation:** Official predefined 10-fold cross-validation (no random splits / no data leakage) on all 10 environmental classes (`air_conditioner`, `car_horn`, `children_playing`, `dog_bark`, `drilling`, `engine_idling`, `gun_shot`, `jackhammer`, `siren`, `street_music`) using 40 MFCC descriptors, fold-by-fold accuracy tracking, and 10x10 cumulative confusion matrix.
   - **Psychoacoustic Feature Extractor:** 49-dimensional spectral & temporal descriptors (MFCCs 1–13 means & stds, spectral centroid, spectral rolloff, spectral bandwidth, zero-crossing rate, chroma vectors, and temporal crest factor).
   - Multi-model evaluation comparing **Random Forest**, **Gradient Boosting**, **Support Vector Machines (SVM)**, and **K-Nearest Neighbors (KNN)**.
   - Real-time classification across 6 core listening & environmental categories with confidence distributions.

2. **Unified Headphone + Ambient Exposure Engine (FR8, FR9, Role 1):**
   - **Combined Daily Sound Dose:** Calculates total cumulative daily exposure across both sources:
     $$D_{\text{total}} = D_{\text{headphone}} + D_{\text{ambient}}$$
   - **Equivalent Continuous Sound Level ($L_{\text{Aeq}}$):** Logarithmic energy accumulation tracking total continuous acoustic power:
     $$L_{\text{Aeq, combined}} = 10 \log_{10} \left( \frac{1}{\sum t} \sum t_i 10^{L_i / 10} \right)$$
   - **Separate vs. Combined Reporting:** Visualized via side-by-side KPI cards, stacked weekly bar charts, and dose alerts at 50%, 80%, and 100% of safe daily boundaries.

3. **Empirical Acoustic Coupler Calibration (Role 2):**
   - Physical acoustic coupler testbench (~15 cm³ sealed paper-cup chamber).
   - Dayton Audio iMM-6 calibrated condenser capsule with locked analog gain.
   - Microphonic offset calibration ($+94.2\text{ dB}$) against certified reference sound-level meters.
   - 135 empirical measurements across 15 discrete system volume increments and 4 standard stimuli (`tone1k`, `pink`, `music1`, `music2`).
   - Quantified physical repeatability repositioning variance ($\sigma = 0.58\text{ dB}$ to $1.37\text{ dB}$).

4. **Core Acoustic DSP & Exposure Dosage (Role 1):**
   - Digital A-weighting filter implementation adhering to IEC 61672-1.
   - Switchable safety exposure standards:
     - **NIOSH:** 85 dBA Recommended Exposure Limit (REL) with 3 dB exchange rate ($T = 8 / 2^{(L - 85)/3}$).
     - **WHO-ITU:** 80 dBA weekly baseline limit (40 hours/week).
   - Continuous dosage accumulator, remaining safe time projection, and adaptive habit coaching.

5. **Live macOS Real-Time Monitor:**
   - Background daemon (`monitor_laptop.py`) tracking active media playback assertions (Spotify, YouTube, Zoom, VLC, Apple Music).
   - Dynamic terminal HUD reporting instantaneous volume step, ear-canal dBA, and safe time remaining.
   - Automatic session logging with persistent synchronization into the dashboard.

6. **Interactive Streamlit Dashboard:**

   - **Tab 1: Today's Exposure & Status:** Real-time gauge, KPI metrics, session table, and smart safety warnings.
   - **Tab 2: Live Volume & Ear Simulator:** Dual-engine estimation (Coupler Lookup vs. Gradient Boosting AI Regressor) with 50 ms waveform preview.
   - **Tab 3: Weekly Trends & Habit Insights:** 7-day dose bar charts, listening duration analytics, and habit coaching.
   - **Tab 4: AI Noise Classifier & Acoustic ML:** Interactive file/audio upload, real-time 49-feature extraction, probability distribution bar chart, confusion matrix, model comparison, and feature importance inspector.
   - **Tab 5: Lab Calibration & Profiles:** Empirical response curves, raw measurement tables, CSV data downloads, and repeatability error bars.
   - **Tab 6: Session Logger & Privacy:** Manual session logging, CSV export, and one-click local data clearing.

---

## 📁 Repository Structure

```text
EchoGuard/
├── app/
│   └── app.py                        # Streamlit 6-tab interactive dashboard
├── data/
│   ├── calibration.csv               # 135 lab coupler recordings (Role 2)
│   ├── headphone_profiles.csv        # Volume step to dBA lookup curves
│   ├── sample_sessions.csv           # Persistent user session history
│   └── demo_sessions.csv             # Pre-configured demo listening data
├── docs/
│   ├── calibration_notes.md          # Full experimental setup & repeatability data
│   ├── calibration_report.md         # Final calibration deliverable report
│   └── figures/                      # High-res publication charts
│       ├── dba_vs_volume_step.png
│       ├── headphone_comparison.png
│       ├── repeatability_test.png
│       ├── ml_model_comparison.png
│       ├── ml_confusion_matrix.png
│       └── ml_feature_importances.png
├── models/                           # Trained ML artifacts
│   ├── noise_classifier_pipeline.joblib
│   ├── acoustic_level_regressor.joblib
│   └── model_metrics.json
├── notebooks/
│   ├── 01_understand_sound_as_numbers.py
│   └── 02_noise_source_classification_and_ml_models.py
├── scripts/
│   ├── build_calibration_dataset.py  # Coupler dataset generator
│   ├── demo_calibration.py           # 60-second live acoustic demo
│   ├── generate_calibration_charts.py
│   ├── generate_test_sounds.py
│   └── train_noise_classifier.py     # End-to-end ML training & benchmarking
├── src/
│   ├── alerts.py                     # Smart safety alerts & coaching logic
│   ├── audio_level.py                # Digital A-weighting & dBA calculations
│   ├── dose_calculator.py            # NIOSH & WHO-ITU dosage models
│   ├── headphone_profiles.py         # Profile lookup tables & interpolation
│   ├── laptop_monitor.py             # macOS CoreAudio & playback assertion tracker
│   ├── level_estimator.py            # Dual-mode level estimator (Lookup + ML)
│   ├── noise_classifier.py           # Audio feature extraction & ML classifier
│   └── session_tracker.py            # Session ingestion & daily aggregation
├── tests/
│   ├── test_audio_level.py
│   ├── test_calibration_data.py
│   ├── test_dose_calculator.py
│   ├── test_level_estimator.py
│   ├── test_noise_classifier.py
│   └── test_session_tracker.py
├── monitor_laptop.py                 # CLI live monitoring utility
├── record.py                         # Acoustic calibration recorder & logger
└── requirements.txt                  # Python dependencies
```

---

## 🚀 Quickstart & Setup

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Divyanshu123-code/EchoGuard-An-AI-Based-Noise-Source-Classification-and-Hearing-Exposure-Monitoring-System.git
cd EchoGuard-An-AI-Based-Noise-Source-Classification-and-Hearing-Exposure-Monitoring-System

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Automated Test Suite (Pytest)

Run all 22 automated unit tests across DSP, calibration, dosage calculations, ML classification, and session tracking:

```bash
pytest
```

### 3. Run the Live Calibration Demo (60-Second Walkthrough)

```bash
python scripts/demo_calibration.py --step 10 --signal tone1k
```

### 4. Train & Benchmark the AI Models

Train the ensemble noise classifier, evaluate 4 machine learning architectures, and generate benchmark charts:

```bash
python scripts/train_noise_classifier.py
```

### 5. Launch the Streamlit Dashboard

```bash
streamlit run app/app.py
```
Open `http://localhost:8501` in your browser.

### 6. Run the Live macOS Headphone Monitor

Track your live listening sessions and volume on macOS in real time:

```bash
python monitor_laptop.py
```

---

## 🔬 Scientific Honesty & Measurement Uncertainty

1. **Uncertainty Bounds:**  
   Every sound level reported by EchoGuard is an estimate carrying an uncertainty margin of approximately **$\pm 5.0\text{ dB}$** unless calibrated against a certified IEC 61672 Class 1 calibrator.
2. **Coupler Physics:**  
   The paper-cup acoustic chamber standardizes volume cavity coupling ($\approx 15\text{ cm}^3$), but does not simulate anatomical ear canal resonances ($2.7\text{ kHz}$ ear canal gain) or eardrum acoustic impedance of an IEC 60318-4 artificial ear.
3. **Non-Diagnostic:**  
   EchoGuard is an exposure awareness and habit coaching system. It is strictly non-diagnostic and makes no clinical audiometry claims.

---

## 📜 Course Deliverables Checklist

- [x] **Role 1 (Tech Lead):** Core DSP pipeline, digital A-weighting, NIOSH/WHO dosage accumulator, full unit tests.
- [x] **Role 2 (Data & Calibration Lead):** Physical coupler experiment, microphone offset ($+94.2\text{ dB}$), 135 calibration points, repeatability testing, lab report and notes.
- [x] **Role 3 (Machine Learning Lead):** 49-feature psychoacoustic extractor, Random Forest ensemble classifier, multi-model benchmarking (RF, GBDT, SVM, KNN), ML acoustic level regressor, evaluation charts.
- [x] **Product Integration:** Real-time macOS monitor daemon, 6-tab Streamlit web application, full pytest coverage.
