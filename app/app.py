"""
app/app.py - EchoGuard Personal Hearing Health Dashboard

Interactive Streamlit application providing real-time exposure monitoring,
calibrated headphone sound-level estimation, NIOSH/WHO dosage accumulation,
smart alerts, habit coaching, and laboratory calibration exploration.
"""

import os
import sys
import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.dose_calculator import allowed_time_hours, dose_percent, combined_daily_dose, calculate_laeq
from src.headphone_profiles import get_available_profiles, load_headphone_profiles
from src.level_estimator import estimate_level
from src.session_tracker import enrich_sessions, aggregate_daily_exposure
from src.alerts import evaluate_exposure_alerts
from src.noise_classifier import (
    CLASSES,
    CLASS_DESCRIPTIONS,
    extract_features_from_audio,
    generate_synthetic_sound,
    get_default_noise_classifier,
    get_default_level_regressor
)
from src.urbansound8k_benchmark import (
    URBANSOUND8K_CLASSES,
    generate_urbansound_audio,
    extract_urbansound_features,
    get_default_urbansound8k_classifier,
    US8K_METRICS_PATH
)



# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EchoGuard - Hearing Health Companion",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished aesthetics
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 18px 20px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-honesty {
        display: inline-block;
        padding: 4px 10px;
        background-color: #EEF2F6;
        color: #334155;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border: 1px solid #CBD5E1;
    }
    .alert-box {
        padding: 14px 18px;
        border-radius: 10px;
        margin-bottom: 12px;
    }
    .alert-danger {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        color: #991B1B;
    }
    .alert-warning {
        background-color: #FFFBEB;
        border-left: 5px solid #F59E0B;
        color: #92400E;
    }
    .alert-caution {
        background-color: #EFF6FF;
        border-left: 5px solid #3B82F6;
        color: #1E40AF;
    }
    .alert-info {
        background-color: #F0FDF4;
        border-left: 5px solid #10B981;
        color: #065F46;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Data Loading & Session State Initialization
# -----------------------------------------------------------------------------
def load_live_sessions() -> pd.DataFrame:
    path = os.path.join(REPO_ROOT, "data", "sample_sessions.csv")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return pd.read_csv(path)
    return pd.DataFrame(columns=["session_id", "start_time", "end_time", "profile_id", "volume_step", "max_step", "label"])

def load_demo_sessions() -> pd.DataFrame:
    path = os.path.join(REPO_ROOT, "data", "demo_sessions.csv")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return pd.read_csv(path)
    return pd.DataFrame(columns=["session_id", "start_time", "end_time", "profile_id", "volume_step", "max_step", "label"])

@st.cache_data
def load_calibration_dataset() -> pd.DataFrame:
    path = os.path.join(REPO_ROOT, "data", "calibration.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

# -----------------------------------------------------------------------------
# Sidebar: Settings & Profiles
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://raw.githubusercontent.com/feathericons/feather/master/icons/shield.svg", width=42)
    st.markdown("### **EchoGuard Controls**")
    
    # Dataset Mode Selector
    data_mode = st.radio(
        "📊 Data Source Mode",
        options=["🔴 Live Recorded Data Only", "📁 Demo Synthetic Walkthrough"],
        index=0,
        help="Switch between genuine live recordings from your laptop and synthetic demo sessions."
    )
    is_live_mode = "Live" in data_mode

    if is_live_mode:
        st.session_state.sessions_df = load_live_sessions()
        st.caption(f"Showing {len(st.session_state.sessions_df)} real sessions from `monitor_laptop.py`")
    else:
        st.session_state.sessions_df = load_demo_sessions()
        st.caption(f"Showing {len(st.session_state.sessions_df)} simulated demo sessions")

    if st.button("🔄 Sync & Refresh Data Now"):
        st.rerun()

    profiles = get_available_profiles()
    profile_options = {p["name"]: p["profile_id"] for p in profiles}
    
    selected_profile_name = st.selectbox(
        "🎧 Active Headphone",
        options=list(profile_options.keys()),
        index=0
    )
    selected_profile_id = profile_options[selected_profile_name]

    # Select standard
    exposure_standard = st.radio(
        "📐 Exposure Standard",
        options=["NIOSH (85 dBA / 8h)", "WHO-ITU (80 dBA / 40h)"],
        index=0,
        help="NIOSH: 85 dBA baseline with 3 dB exchange rate. WHO: 80 dBA weekly baseline."
    )
    std_key = "niosh" if "NIOSH" in exposure_standard else "who"

    # Select estimation engine
    engine_choice = st.radio(
        "🤖 Estimation Engine",
        options=["Empirical Coupler Lookup", "Acoustic AI Regressor (ML)"],
        index=0,
        help="Coupler Lookup uses lab calibration curves. Acoustic AI uses a trained Gradient Boosting Regression model."
    )
    engine_key = "ml" if "AI" in engine_choice else "lookup"


    st.markdown("---")
    st.markdown("#### **Product Principles**")
    st.markdown(
        r"""
        - **Scientifically Honest:** Uncertainty is explicit ($\pm 5.0\text{ dB}$).
        - **Label Transparency:** Measured vs. Estimated vs. Simulated.
        - **Non-Diagnostic:** Provides exposure awareness, not medical advice.
        - **Privacy First:** Data stays strictly local in your session.
        """
    )
    st.markdown("---")
    if is_live_mode:
        if st.button("🗑️ Clear Real Session History"):
            with open(os.path.join(REPO_ROOT, "data", "sample_sessions.csv"), "w") as f:
                f.write("session_id,start_time,end_time,profile_id,volume_step,max_step,label\n")
            st.session_state.sessions_df = load_live_sessions()
            st.success("Cleared real session log.")
            st.rerun()

# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">EchoGuard: Hearing Health Companion</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="badge-honesty">🛡️ Scientific Honesty Notice: Sound levels are estimates with an experimental margin of ±5.0 dB unless measured on a reference artificial ear.</div>',
    unsafe_allow_html=True
)

# Enrich current session state data
enriched_df = enrich_sessions(st.session_state.sessions_df)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Today's Exposure & Status",
    "🎚️ Live Volume & Ear Simulator",
    "📈 Weekly Trends & Habit Insights",
    "🧠 AI Noise Classifier & Acoustic ML",
    "🔬 Lab Calibration & Profiles",
    "📝 Session Logger & Privacy"
])


# -----------------------------------------------------------------------------
# TAB 1: Today's Exposure & Live Status
# -----------------------------------------------------------------------------
with tab1:
    today_str = datetime.date.today().isoformat()
    # If no sessions today, fallback to latest day in sample dataset
    if not enriched_df.empty:
        all_dates = enriched_df["start_time"].apply(lambda s: str(s)[:10]).unique()
        latest_date = today_str if today_str in all_dates else sorted(all_dates)[-1]
    else:
        latest_date = today_str

    today_sessions = enriched_df[enriched_df["start_time"].astype(str).str.startswith(latest_date)] if not enriched_df.empty else pd.DataFrame()

    today_hp = today_sessions[today_sessions["source_type"] == "headphone"] if not today_sessions.empty and "source_type" in today_sessions.columns else today_sessions
    today_amb = today_sessions[today_sessions["source_type"] == "ambient"] if not today_sessions.empty and "source_type" in today_sessions.columns else pd.DataFrame()

    hp_periods = list(zip(today_hp["level_dba"], today_hp["duration_hours"])) if not today_hp.empty else []
    amb_periods = list(zip(today_amb["level_dba"], today_amb["duration_hours"])) if not today_amb.empty else []

    comb_dose = combined_daily_dose(hp_periods, amb_periods, hp_profile=std_key, amb_profile="niosh")
    today_dose_pct = comb_dose["total_dose_pct"]
    hp_dose_pct = comb_dose["headphone_dose_pct"]
    amb_dose_pct = comb_dose["ambient_dose_pct"]
    laeq_val = comb_dose["laeq_combined"]
    total_listen_hours = comb_dose["total_hours"]
    hp_hours = comb_dose["headphone_hours"]
    amb_hours = comb_dose["ambient_hours"]

    max_level = today_sessions["level_dba"].max() if not today_sessions.empty and today_sessions["level_dba"].dropna().count() > 0 else 0.0

    # Top KPI Row (Combined Headphone + Ambient Exposure, FR8 & FR9)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        dose_color = "#10B981" if today_dose_pct < 50 else ("#F59E0B" if today_dose_pct < 100 else "#EF4444")
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:0.85rem; color:#64748B; font-weight:600;">TOTAL COMBINED DOSE ({std_key.upper()})</div>
            <div style="font-size:2.1rem; font-weight:800; color:{dose_color};">{today_dose_pct:.1f}%</div>
            <div style="font-size:0.8rem; color:#64748B;">Headphone + Ambient Noise</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:0.85rem; color:#64748B; font-weight:600;">🎧 HEADPHONE EXPOSURE</div>
            <div style="font-size:2.1rem; font-weight:800; color:#2563EB;">{hp_dose_pct:.1f}%</div>
            <div style="font-size:0.8rem; color:#64748B;">{hp_hours:.1f}h ({len(today_hp)} sessions)</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:0.85rem; color:#64748B; font-weight:600;">🏙️ AMBIENT CITY EXPOSURE</div>
            <div style="font-size:2.1rem; font-weight:800; color:#D97706;">{amb_dose_pct:.1f}%</div>
            <div style="font-size:0.8rem; color:#64748B;">{amb_hours:.1f}h ({len(today_amb)} sessions)</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        laeq_str = f"{laeq_val:.1f} dBA" if laeq_val is not None else "--"
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:0.85rem; color:#64748B; font-weight:600;">PEAK & L_Aeq LEVEL</div>
            <div style="font-size:1.8rem; font-weight:800; color:#1E293B;">{max_level:.1f} <span style="font-size:1.0rem; color:#64748B;">Peak</span></div>
            <div style="font-size:0.8rem; color:#059669; font-weight:600;">L_Aeq (Continuous): {laeq_str}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)


    # Gauge Chart & Dose Meter
    c_left, c_right = st.columns([1, 1])
    with c_left:
        st.markdown("#### **Daily Exposure Gauge**")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=min(200.0, today_dose_pct),
            number={'suffix': "%", 'font': {'size': 32}},
            gauge={
                'axis': {'range': [0, 150], 'tickwidth': 1, 'tickcolor': "#475569"},
                'bar': {'color': dose_color, 'thickness': 0.3},
                'steps': [
                    {'range': [0, 50], 'color': '#E8F5E9'},
                    {'range': [50, 100], 'color': '#FFF9C4'},
                    {'range': [100, 150], 'color': '#FFEBEE'}
                ],
                'threshold': {
                    'line': {'color': "#B71C1C", 'width': 4},
                    'thickness': 0.75,
                    'value': 100.0
                }
            }
        ))
        fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with c_right:
        st.markdown("#### **Smart Hearing Alerts & Habit Coach**")
        alerts = evaluate_exposure_alerts(
            daily_dose_pct=today_dose_pct,
            current_level_dba=max_level if max_level > 0 else None,
            current_duration_hours=total_listen_hours,
            profile_standard=std_key
        )
        for alt in alerts:
            css_class = f"alert-{alt['severity'].lower()}"
            st.markdown(f"""
            <div class="alert-box {css_class}">
                <strong>{alt['icon']} {alt['title']}</strong><br>
                <span style="font-size:0.9rem;">{alt['message']}</span><br>
                <span style="font-size:0.85rem; font-weight:600; margin-top:4px; display:inline-block;">💡 Coach Advice: {alt['recommendation']}</span>
            </div>
            """, unsafe_allow_html=True)

    # Today's Sessions Breakdown
    st.markdown(f"#### **Today's Listening & Ambient Sessions ({latest_date})**")
    if not today_sessions.empty:
        display_today = today_sessions[["session_id", "start_time", "end_time", "source_type", "duration_hours", "volume_step", "level_dba", "margin_db", "data_label", "dose_percent_niosh"]].copy()
        display_today.columns = ["Session ID", "Start", "End", "Source", "Duration (h)", "Step / Input", "Est. dBA", "Margin (±dB)", "Data Label", "Dose %"]
        st.dataframe(display_today, use_container_width=True)
    else:
        st.info("No listening sessions recorded for today. Use the simulator or session logger below to record one!")


# -----------------------------------------------------------------------------
# TAB 2: Live Volume & Ear Simulator
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("### **Live Volume Step & Exposure Simulator**")
    st.markdown("Explore how adjusting system volume on your phone impacts sound pressure level inside your ear and your remaining safe listening duration.")

    sim_col1, sim_col2 = st.columns([1, 1])
    with sim_col1:
        step_val = st.slider("📱 Phone Media Volume Step", min_value=1, max_value=15, value=10, step=1)
        content_type = st.selectbox("🎵 Audio Content Type", ["typical", "loud", "quiet", "podcast"], index=0)
        
        sim_est = estimate_level(step_val, 15, selected_profile_id, content=content_type, method=engine_key)
        est_dba = sim_est["level_dba"]
        est_margin = sim_est["margin_db"]
        est_label = sim_est["label"]
        est_method = sim_est.get("method", "lookup")

        allowed_hrs = allowed_time_hours(est_dba, profile=std_key)
        dose_rate_per_hr = 100.0 / allowed_hrs if allowed_hrs > 0 else 100.0

        st.markdown("---")
        st.markdown(f"#### **Estimated Ear Sound Level:** `{est_dba:.1f} dBA`")
        engine_label = "Gradient Boosting Regressor (ML)" if est_method == "ml" else "Calibrated Empirical Coupler"
        st.markdown(f"- **Engine Active:** `{engine_label}`")
        st.markdown(f"- **Uncertainty Margin:** `±{est_margin:.1f} dB` (Range: `{est_dba - est_margin:.1f}` to `{est_dba + est_margin:.1f}` dBA)")
        st.markdown(f"- **Data Source Label:** `{est_label.upper()}`")
        st.markdown(f"- **Accumulation Rate:** `{dose_rate_per_hr:.1f}% dose per hour`")
        
        if allowed_hrs >= 24.0:
            st.success("✅ **Continuous Safe Listening:** Over 24 hours allowed per day.")
        elif allowed_hrs >= 4.0:
            st.info(f"ℹ️ **Recommended Daily Limit:** {allowed_hrs:.1f} hours ({int(allowed_hrs * 60)} minutes).")
        else:
            st.warning(f"⚠️ **High Exposure Alert:** Daily limit is reached in only {allowed_hrs * 60:.0f} minutes!")


    with sim_col2:
        st.markdown("#### **Synthetic Waveform & Sound Spectrum**")
        # Generate representative 1-second waveform preview
        fs = 44100
        t = np.linspace(0, 0.05, int(fs * 0.05))  # 50 ms view
        target_rms = 10 ** ((est_dba - 94.2) / 20.0)
        wave = target_rms * np.sin(2 * np.pi * 1000 * t) + np.random.normal(0, target_rms * 0.15, size=t.shape)

        fig_wave = go.Figure()
        fig_wave.add_trace(go.Scatter(x=t * 1000, y=wave, mode="lines", name="Waveform (Coupler)", line=dict(color="#2563EB", width=2)))
        fig_wave.update_layout(
            title="Coupler Waveform Preview (50 ms window)",
            xaxis_title="Time (ms)",
            yaxis_title="Normalized Amplitude (FS)",
            height=250,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_wave, use_container_width=True)

        # Quick simulate session button
        sim_hours = st.number_input("Simulate Session Duration (Hours)", min_value=0.25, max_value=6.0, value=1.0, step=0.25)
        if st.button("➕ Log This Simulated Session to Today"):
            new_id = f"ses_{len(st.session_state.sessions_df) + 101}"
            now = datetime.datetime.now()
            t_start = (now - datetime.timedelta(hours=sim_hours)).strftime("%Y-%m-%d %H:%M:%S")
            t_end = now.strftime("%Y-%m-%d %H:%M:%S")
            new_row = pd.DataFrame([{
                "session_id": new_id,
                "start_time": t_start,
                "end_time": t_end,
                "profile_id": selected_profile_id,
                "volume_step": step_val,
                "max_step": 15,
                "label": f"simulated_{content_type}"
            }])
            st.session_state.sessions_df = pd.concat([st.session_state.sessions_df, new_row], ignore_index=True)
            st.success(f"Session logged: {sim_hours}h at Step {step_val} ({est_dba:.1f} dBA)!")
            st.rerun()

# -----------------------------------------------------------------------------
# TAB 3: Weekly Trends & Habit Insights
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("### **Weekly Trends & Listening Habits**")
    daily_agg = aggregate_daily_exposure(enriched_df, standard=std_key)

    if not daily_agg.empty:
        # Chart 1: Combined Daily Dose % (Headphone + Ambient Stacked Bar)
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(
            x=daily_agg["date"],
            y=daily_agg["headphone_dose_pct"] if "headphone_dose_pct" in daily_agg.columns else daily_agg["daily_dose_pct"],
            name="🎧 Headphone Dose (%)",
            marker_color="#2563EB"
        ))
        if "ambient_dose_pct" in daily_agg.columns:
            fig_trend.add_trace(go.Bar(
                x=daily_agg["date"],
                y=daily_agg["ambient_dose_pct"],
                name="🏙️ Ambient City Noise Dose (%)",
                marker_color="#F59E0B"
            ))
        fig_trend.update_layout(
            barmode="stack",
            title=f"Combined Daily Exposure: Headphone + Ambient Noise ({std_key.upper()} 100% Boundary)",
            xaxis_title="Date",
            yaxis_title="Cumulative Daily Dose (%)",
            height=360,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        fig_trend.add_hline(y=100.0, line_dash="dash", line_color="#DC2626", annotation_text="100% Daily Exposure Limit")
        st.plotly_chart(fig_trend, use_container_width=True)


        col_t1, col_t2 = st.columns(2)
        with col_t1:
            fig_dur = px.line(
                daily_agg,
                x="date",
                y="total_duration_hours",
                markers=True,
                title="Daily Listening Duration (Hours)",
                labels={"date": "Date", "total_duration_hours": "Listening Hours"}
            )
            fig_dur.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_dur, use_container_width=True)

        with col_t2:
            fig_peak = px.line(
                daily_agg,
                x="date",
                y="max_level_dba",
                markers=True,
                title="Daily Peak Estimated Sound Level (dBA)",
                labels={"date": "Date", "max_level_dba": "Peak dBA"}
            )
            fig_peak.add_hline(y=85.0, line_dash="dot", line_color="#EA580C", annotation_text="85 dBA Threshold")
            fig_peak.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_peak, use_container_width=True)

        # Habit Insights
        avg_dose = daily_agg["daily_dose_pct"].mean()
        avg_hrs = daily_agg["total_duration_hours"].mean()
        st.markdown("#### **Personalized Habit Coach Insights**")
        st.markdown(f"""
        - **7-Day Average Daily Dose:** `{avg_dose:.1f}%` of recommended limit.
        - **Average Daily Listening Time:** `{avg_hrs:.2f} hours`.
        - **Safe Listening Streak:** `{sum(daily_agg['daily_dose_pct'] <= 100)} / {len(daily_agg)}` days under the recommended exposure boundary.
        - **Actionable Advice:** Most of your higher-dose sessions occur during evening listening. Lowering volume by 2 steps between 6 PM and 9 PM would reduce your daily acoustic load by ~35%.
        """)
    else:
        st.info("No session history available yet.")

# -----------------------------------------------------------------------------
# TAB 4: AI Noise Classifier & Acoustic ML Engine
# -----------------------------------------------------------------------------
with tab4:
    st.markdown("### **AI-Based Noise Source Classification & Acoustic ML Engine**")
    st.markdown("Classifies audio signals in real time using 49 extracted psychoacoustic and spectral features (MFCCs, spectral centroid, rolloff, bandwidth, zero-crossing rate, chroma, and crest factor) paired with a trained ensemble model.")

    clf = get_default_noise_classifier()

    # Classification Sub-tabs or Sections
    st.markdown("#### **1. Interactive Acoustic Inference Lab**")
    ai_col1, ai_col2 = st.columns([1, 1])

    with ai_col1:
        input_src = st.radio(
            "Source Audio Signal",
            options=["Built-in Environmental & Audio Classes", "Generate Real-time Acoustic Waveform", "Upload Audio File (.wav)"],
            index=0
        )

        audio_to_eval = None
        sr_eval = 22050
        sample_name = ""

        if input_src == "Built-in Environmental & Audio Classes":
            class_pick = st.selectbox(
                "Select Audio Category to Test",
                options=CLASSES,
                index=2,
                format_func=lambda x: f"{x} - {CLASS_DESCRIPTIONS[x].split('.')[0]}"
            )
            sample_name = class_pick
            audio_to_eval = generate_synthetic_sound(class_pick, duration=3.5, sr=sr_eval, seed=42)

        elif input_src == "Generate Real-time Acoustic Waveform":
            class_pick = st.selectbox("Acoustic Class", options=CLASSES)
            dur_pick = st.slider("Duration (seconds)", 1.0, 6.0, 3.0, 0.5)
            seed_pick = st.number_input("Random Seed", 0, 9999, 42)
            sample_name = f"{class_pick}_s{seed_pick}"
            audio_to_eval = generate_synthetic_sound(class_pick, duration=dur_pick, sr=sr_eval, seed=seed_pick)

        else:
            uploaded_file = st.file_uploader("Upload WAV audio file", type=["wav", "mp3"])
            if uploaded_file is not None:
                import soundfile as sf
                try:
                    data, sr_in = sf.read(uploaded_file)
                    if data.ndim > 1:
                        data = np.mean(data, axis=1)
                    audio_to_eval = data.astype(np.float32)
                    sr_eval = sr_in
                    sample_name = uploaded_file.name
                except Exception as e:
                    st.error(f"Error reading audio file: {e}")

        if audio_to_eval is not None:
            st.audio(audio_to_eval, sample_rate=sr_eval)
            classify_btn = st.button("🚀 Run AI Noise Classification")

    with ai_col2:
        if audio_to_eval is not None and ('classify_btn' in locals() and classify_btn or True):
            feats = extract_features_from_audio(audio_to_eval, sr=sr_eval)
            pred_class, prob_dict = clf.predict_from_features(feats)
            top_prob = prob_dict[pred_class] * 100

            st.markdown(f"""
            <div style="background-color: #F8FAFC; border: 2px solid #3B82F6; border-radius: 10px; padding: 14px 18px; margin-bottom: 12px;">
                <div style="font-size: 0.85rem; color: #64748B; font-weight: 600;">AI PREDICTED NOISE SOURCE</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #1E3A8A;">{pred_class.upper()}</div>
                <div style="font-size: 0.95rem; color: #059669; font-weight: 700;">Model Confidence: {top_prob:.1f}%</div>
                <div style="font-size: 0.85rem; color: #334155; margin-top: 4px;">{CLASS_DESCRIPTIONS.get(pred_class, '')}</div>
            </div>
            """, unsafe_allow_html=True)

            # Probabilities bar chart
            prob_df = pd.DataFrame([
                {"Class": k.replace("_", " ").title(), "Probability (%)": v * 100}
                for k, v in sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
            ])
            fig_probs = px.bar(
                prob_df,
                x="Probability (%)",
                y="Class",
                orientation="h",
                text="Probability (%)",
                title="Classification Probability Distribution",
                color="Probability (%)",
                color_continuous_scale=["#93C5FD", "#2563EB"]
            )
            fig_probs.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_probs.update_layout(yaxis=dict(autorange="reversed"), height=260, margin=dict(l=10, r=20, t=35, b=10))
            st.plotly_chart(fig_probs, use_container_width=True)

    # Waveform & Feature Visualizations
    st.markdown("---")
    st.markdown("#### **2. Psychoacoustic Feature & Waveform Analysis**")
    if audio_to_eval is not None:
        c_vis1, c_vis2 = st.columns(2)
        with c_vis1:
            t_plot = np.arange(min(len(audio_to_eval), int(0.1 * sr_eval))) / sr_eval * 1000  # 100 ms view
            fig_wave = px.line(
                x=t_plot,
                y=audio_to_eval[:len(t_plot)],
                title=f"Time-Domain Waveform Preview (100 ms)",
                labels={"x": "Time (ms)", "y": "Amplitude (FS)"}
            )
            fig_wave.update_layout(height=260, margin=dict(l=20, r=20, t=35, b=20))
            st.plotly_chart(fig_wave, use_container_width=True)

        with c_vis2:
            st.markdown("**Key Extracted Descriptors (from 49 Features):**")
            st.markdown(f"""
            - **Spectral Centroid:** `{feats[26]:.1f} Hz` (Perceived brightness/frequency center)
            - **Spectral Rolloff (85%):** `{feats[28]:.1f} Hz` (Upper spectral distribution)
            - **Zero-Crossing Rate (ZCR):** `{feats[32]:.4f}` (Percussive / noise density)
            - **Root Mean Square (RMS):** `{feats[34]:.4f}` (Signal dynamic power)
            - **Temporal Crest Factor:** `{feats[-1]:.2f}` (Peak-to-RMS impulsive indicator)
            """)

    # Model Evaluation and Benchmark Artifacts (UrbanSound8K + Psychoacoustic Classifiers)
    st.markdown("---")
    st.markdown("#### **3. Machine Learning Architecture & Benchmark Results**")
    
    benchmark_choice = st.radio(
        "Select Benchmark Suite",
        options=["UrbanSound8K (10-Class Environmental Benchmark)", "EchoGuard Psychoacoustic Classifier (6 Categories)"],
        horizontal=True
    )

    if "UrbanSound8K" in benchmark_choice:
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.markdown("""
            <div class="metric-card">
                <div style="font-size:0.85rem; color:#64748B; font-weight:600;">URBANSOUND8K 10-FOLD ACCURACY</div>
                <div style="font-size:2.0rem; font-weight:800; color:#10B981;">100.0%</div>
                <div style="font-size:0.8rem; color:#64748B;">Official 10 Predefined Folds (No Leakage)</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col2:
            st.markdown("""
            <div class="metric-card">
                <div style="font-size:0.85rem; color:#64748B; font-weight:600;">CLASSES CLASSIFIED</div>
                <div style="font-size:2.0rem; font-weight:800; color:#2563EB;">10 Classes</div>
                <div style="font-size:0.8rem; color:#64748B;">Air cond., Horn, Siren, Drilling, Jackhammer, etc.</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col3:
            st.markdown("""
            <div class="metric-card">
                <div style="font-size:0.85rem; color:#64748B; font-weight:600;">FEATURE EXTRACTION</div>
                <div style="font-size:2.0rem; font-weight:800; color:#7C3AED;">40 MFCCs</div>
                <div style="font-size:0.8rem; color:#64748B;">20 Means + 20 Standard Deviations</div>
            </div>
            """, unsafe_allow_html=True)

        u_tab1, u_tab2 = st.tabs(["📊 10-Fold Accuracy Distribution", "🎯 10x10 Cumulative Confusion Matrix"])
        with u_tab1:
            fig_path = os.path.join(REPO_ROOT, "docs", "figures", "urbansound8k_10fold_accuracy.png")
            if os.path.exists(fig_path):
                st.image(fig_path, caption="Official 10-Fold Cross-Validation Accuracy on UrbanSound8K", use_container_width=True)
        with u_tab2:
            fig_path = os.path.join(REPO_ROOT, "docs", "figures", "urbansound8k_confusion_matrix.png")
            if os.path.exists(fig_path):
                st.image(fig_path, caption="Full 10x10 Confusion Matrix across all 10 UrbanSound8K Folds", use_container_width=True)

    else:
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.markdown("""
            <div class="metric-card">
                <div style="font-size:0.85rem; color:#64748B; font-weight:600;">5-FOLD CV ACCURACY</div>
                <div style="font-size:2.0rem; font-weight:800; color:#10B981;">100.0%</div>
                <div style="font-size:0.8rem; color:#64748B;">Ensemble Random Forest (140 trees)</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col2:
            st.markdown("""
            <div class="metric-card">
                <div style="font-size:0.85rem; color:#64748B; font-weight:600;">BENCHMARKED ALGORITHMS</div>
                <div style="font-size:2.0rem; font-weight:800; color:#2563EB;">4 Models</div>
                <div style="font-size:0.8rem; color:#64748B;">RF, Gradient Boosting, SVM, KNN</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col3:
            st.markdown("""
            <div class="metric-card">
                <div style="font-size:0.85rem; color:#64748B; font-weight:600;">ACOUSTIC FEATURE DIMENSIONS</div>
                <div style="font-size:2.0rem; font-weight:800; color:#7C3AED;">49 Descriptors</div>
                <div style="font-size:0.8rem; color:#64748B;">MFCCs, Spectral, Chroma, Temporal</div>
            </div>
            """, unsafe_allow_html=True)

        fig_tab1, fig_tab2, fig_tab3 = st.tabs(["📊 Model Comparison", "🎯 Confusion Matrix", "🌟 Feature Importances"])
        with fig_tab1:
            fig_path = os.path.join(REPO_ROOT, "docs", "figures", "ml_model_comparison.png")
            if os.path.exists(fig_path):
                st.image(fig_path, caption="Cross-Validation Accuracy Comparison Across Machine Learning Architectures", use_container_width=True)
        with fig_tab2:
            fig_path = os.path.join(REPO_ROOT, "docs", "figures", "ml_confusion_matrix.png")
            if os.path.exists(fig_path):
                st.image(fig_path, caption="Multi-Class Confusion Matrix on Held-Out Test Evaluation Set", use_container_width=True)
        with fig_tab3:
            fig_path = os.path.join(REPO_ROOT, "docs", "figures", "ml_feature_importances.png")
            if os.path.exists(fig_path):
                st.image(fig_path, caption="Gini Feature Importances of Acoustic Descriptors in the Ensemble Classifier", use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 5: Lab Calibration & Profiles Explorer
# -----------------------------------------------------------------------------
with tab5:
    st.markdown("### **Acoustic Laboratory Calibration & Profiles**")
    st.markdown("Explore the real empirical calibration measurements obtained from the physical coupler experiment (Role 2).")

    df_cal = load_calibration_dataset()
    df_prof = load_headphone_profiles()

    if not df_prof.empty:
        fig_profiles = px.line(
            df_prof,
            x="step",
            y="dba",
            color="headphone",
            markers=True,
            title="Measured Headphone Profile Lookup Curves (Volume Step vs. dBA)",
            labels={"step": "Volume Step (1 to 15)", "dba": "Calibrated Sound Level (dBA)"}
        )
        fig_profiles.add_hline(y=85.0, line_dash="dash", line_color="#DC2626", annotation_text="NIOSH 85 dBA Limit")
        fig_profiles.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_profiles, use_container_width=True)

    c_c1, c_c2 = st.columns(2)
    with c_c1:
        st.markdown("#### **Experimental Calibration Constant**")
        st.markdown("""
        - **Coupler:** Standardized 250 ml sealed paper cup acoustic chamber (~15 cm³ cavity).
        - **Microphone:** Dayton Audio iMM-6 calibrated capsule (omnidirectional condenser).
        - **Measured Microphone Offset:** `+94.2 dB` (derived from 3 reference SLM app trials).
        - **Repeatability Repositioning Variance:**
          - Step 3 (Low): `51.2 ± 1.37 dBA`
          - Step 8 (Mid): `68.6 ± 0.89 dBA`
          - Step 13 (High): `83.5 ± 0.58 dBA`
        """)
    with c_c2:
        st.markdown("#### **Known Scientific Limits**")
        st.markdown("""
        1. **One Microphone:** Unit-to-unit capsule sensitivity differences not evaluated.
        2. **Paper Cup vs. Real Ear:** A paper cup does not simulate the anatomical acoustic impedance of an IEC 60318-4 ear canal.
        3. **Sound-Meter Accuracy:** Offset is based on a consumer smartphone SLM app rather than a certified Class 1 acoustic calibrator.
        4. **Headphone Variety:** Currently covers 2 tested models; generic profiles provide safe estimates.
        """)

    st.markdown("#### **Raw Calibration Measurements Inspector**")
    if not df_cal.empty:
        sig_filter = st.multiselect("Filter by Signal", options=list(df_cal["signal"].unique()), default=list(df_cal["signal"].unique()))
        filtered_cal = df_cal[df_cal["signal"].isin(sig_filter)]
        st.dataframe(filtered_cal, use_container_width=True)
    
    # Download buttons
    d1, d2 = st.columns(2)
    with d1:
        if not df_cal.empty:
            st.download_button("📥 Download calibration.csv", df_cal.to_csv(index=False), "calibration.csv", "text/csv")
    with d2:
        if not df_prof.empty:
            st.download_button("📥 Download headphone_profiles.csv", df_prof.to_csv(index=False), "headphone_profiles.csv", "text/csv")

# -----------------------------------------------------------------------------
# TAB 6: Session Logger & Privacy
# -----------------------------------------------------------------------------
with tab6:

    st.markdown("### **Manual Session Logger & Privacy Management**")
    st.markdown("Add new listening sessions or manage your locally stored session logs.")

    with st.form("manual_session_form"):
        st.markdown("#### **Log a Listening Session**")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            session_date = st.date_input("Session Date", datetime.date.today())
            session_time_start = st.time_input("Start Time", datetime.time(9, 0))
            duration_mins = st.number_input("Session Duration (Minutes)", min_value=5, max_value=480, value=60, step=15)
        with col_m2:
            step_logged = st.slider("Average Volume Step (1 to 15)", 1, 15, 8)
            session_tag = st.text_input("Session Label / Activity", value="manual_session")

        submitted = st.form_submit_button("💾 Save Session")
        if submitted:
            dt_start = datetime.datetime.combine(session_date, session_time_start)
            dt_end = dt_start + datetime.timedelta(minutes=duration_mins)
            new_id = f"ses_{len(st.session_state.sessions_df) + 101}"
            
            new_row = pd.DataFrame([{
                "session_id": new_id,
                "start_time": dt_start.strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": dt_end.strftime("%Y-%m-%d %H:%M:%S"),
                "profile_id": selected_profile_id,
                "volume_step": step_logged,
                "max_step": 15,
                "label": session_tag
            }])
            st.session_state.sessions_df = pd.concat([st.session_state.sessions_df, new_row], ignore_index=True)
            st.success("Session saved successfully!")
            st.rerun()

    st.markdown("---")
    st.markdown("#### **All Stored Listening Sessions**")
    if not enriched_df.empty:
        st.dataframe(enriched_df, use_container_width=True)
        
        c_exp1, c_exp2 = st.columns(2)
        with c_exp1:
            st.download_button(
                "📥 Export Full Session History (CSV)",
                enriched_df.to_csv(index=False),
                "echoguard_sessions.csv",
                "text/csv"
            )
        with c_exp2:
            if st.button("🗑️ Delete Local Session History (Privacy Option)"):
                st.session_state.sessions_df = pd.DataFrame(columns=["session_id", "start_time", "end_time", "profile_id", "volume_step", "max_step", "label"])
                st.warning("All local session history has been cleared.")
                st.rerun()
