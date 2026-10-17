# EchoGuard

**Headphone Hearing-Exposure Estimation and Listening Coach**  
*Advanced Artificial Intelligence Course Project | Version 2026-10-17*

EchoGuard estimates the sound level reaching the ear from volume step information and calibrated headphone profiles, calculates cumulative daily exposure against NIOSH and WHO-ITU standards, and coaches users toward safer listening habits without diagnosing or making medical claims.

---

## Repository Structure

```text
EchoGuard/
├── data/              # Team shared CSV datasets (strictly NO .wav files committed)
│   ├── calibration.csv        # Lab measurement recordings (from Role 2)
│   ├── headphone_profiles.csv # Volume-step to dBA lookup tables
│   └── sample_sessions.csv    # Listening sessions for demo walkthrough
├── src/               # Python core modules
│   ├── audio_level.py         # A-weighting filter & dBA level calculation
│   ├── dose_calculator.py     # NIOSH / WHO-ITU exposure dose accumulator
│   └── level_estimator.py     # Volume step to ear-level dBA estimator
├── tests/             # Automated unit tests (pytest)
│   ├── test_audio_level.py
│   └── test_dose_calculator.py
├── notebooks/         # Exploratory analysis & ML model comparisons
├── app/               # Streamlit dashboard application
└── docs/              # Measurement methodology, notes, and reports
```

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Divyanshu123-code/EchoGuard-An-AI-Based-Noise-Source-Classification-and-Hearing-Exposure-Monitoring-System.git
   cd EchoGuard-An-AI-Based-Noise-Source-Classification-and-Hearing-Exposure-Monitoring-System
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Running Tests

Run the full pytest suite:
```bash
pytest
```

---

## Running the Dashboard

Launch the Streamlit demo application:
```bash
streamlit run app/app.py
```
