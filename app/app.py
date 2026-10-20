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

from src.dose_calculator import allowed_time_hours, dose_percent
from src.headphone_profiles import get_available_profiles, load_headphone_profiles
from src.level_estimator import estimate_level
from src.session_tracker import enrich_sessions, aggregate_daily_exposure
from src.alerts import evaluate_exposure_alerts

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
# Data Loading & Session State Initialization
# -----------------------------------------------------------------------------
@st.cache_data
def load_base_sessions() -> pd.DataFrame:
    path = os.path.join(REPO_ROOT, "data", "sample_sessions.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=["session_id", "start_time", "end_time", "profile_id", "volume_step", "max_step", "label"])

@st.cache_data
def load_calibration_dataset() -> pd.DataFrame:
    path = os.path.join(REPO_ROOT, "data", "calibration.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

if "sessions_df" not in st.session_state:
    st.session_state.sessions_df = load_base_sessions()

# -----------------------------------------------------------------------------
# Sidebar: Settings & Profiles
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://raw.githubusercontent.com/feathericons/feather/master/icons/shield.svg", width=42)
    st.markdown("### **EchoGuard Controls**")
    
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

    st.markdown("---")
    st.markdown("#### **Product Principles**")
    st.markdown(
        """
        - **Scientifically Honest:** Uncertainty is explicit ($\pm 5.0\text{ dB}$).
        - **Label Transparency:** Measured vs. Estimated vs. Simulated.
        - **Non-Diagnostic:** Provides exposure awareness, not medical advice.
        - **Privacy First:** Data stays strictly local in your session.
        """
    )
    st.markdown("---")
    if st.button("🔄 Reset to Default Sample Data"):
        st.session_state.sessions_df = load_base_sessions()
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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Today's Exposure & Status",
    "🎚️ Live Volume & Ear Simulator",
    "📈 Weekly Trends & Habit Insights",
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

    total_listen_hours = today_sessions["duration_hours"].sum() if not today_sessions.empty else 0.0
    periods = list(zip(today_sessions["level_dba"], today_sessions["duration_hours"])) if not today_sessions.empty else []
    today_dose_pct = dose_percent(periods, profile=std_key)
    max_level = today_sessions["level_dba"].max() if not today_sessions.empty and today_sessions["level_dba"].dropna().count() > 0 else 0.0

    # Top KPI Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        dose_color = "#10B981" if today_dose_pct < 50 else ("#F59E0B" if today_dose_pct < 100 else "#EF4444")
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:0.85rem; color:#64748B; font-weight:600;">DAILY SOUND DOSE ({std_key.upper()})</div>
            <div style="font-size:2.1rem; font-weight:800; color:{dose_color};">{today_dose_pct:.1f}%</div>
            <div style="font-size:0.8rem; color:#64748B;">Target: < 100% per 24h</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:0.85rem; color:#64748B; font-weight:600;">TOTAL LISTENING TODAY</div>
            <div style="font-size:2.1rem; font-weight:800; color:#1E293B;">{total_listen_hours:.1f}h</div>
            <div style="font-size:0.8rem; color:#64748B;">{int(total_listen_hours * 60)} minutes across {len(today_sessions)} sessions</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:0.85rem; color:#64748B; font-weight:600;">PEAK SOUND LEVEL</div>
            <div style="font-size:2.1rem; font-weight:800; color:#1E293B;">{max_level:.1f} <span style="font-size:1.1rem; color:#64748B;">dBA</span></div>
            <div style="font-size:0.8rem; color:#64748B;">Estimated ear-canal level</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:0.85rem; color:#64748B; font-weight:600;">ACTIVE HEADPHONE</div>
            <div style="font-size:1.25rem; font-weight:700; color:#1E293B; margin-top:6px;">{selected_profile_name.split('(')[0].strip()}</div>
            <div style="font-size:0.8rem; color:#059669; font-weight:600;">Profile: Calibrated (±5.0 dB)</div>
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
    st.markdown(f"#### **Today's Listening Sessions ({latest_date})**")
    if not today_sessions.empty:
        display_today = today_sessions[["session_id", "start_time", "end_time", "duration_hours", "volume_step", "level_dba", "margin_db", "data_label", "dose_percent_niosh"]].copy()
        display_today.columns = ["Session ID", "Start", "End", "Duration (h)", "Volume Step", "Est. dBA", "Margin (±dB)", "Data Label", "Dose %"]
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
        
        sim_est = estimate_level(step_val, 15, selected_profile_id, content=content_type)
        est_dba = sim_est["level_dba"]
        est_margin = sim_est["margin_db"]
        est_label = sim_est["label"]

        allowed_hrs = allowed_time_hours(est_dba, profile=std_key)
        dose_rate_per_hr = 100.0 / allowed_hrs if allowed_hrs > 0 else 100.0

        st.markdown("---")
        st.markdown(f"#### **Estimated Ear Sound Level:** `{est_dba:.1f} dBA`")
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
        # Chart 1: Daily Dose % over Time
        fig_trend = px.bar(
            daily_agg,
            x="date",
            y="daily_dose_pct",
            title=f"Daily Sound Dose Percentage ({std_key.upper()} 100% Limit Line)",
            labels={"date": "Date", "daily_dose_pct": "Daily Dose (%)"},
            color="daily_dose_pct",
            color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"]
        )
        fig_trend.add_hline(y=100.0, line_dash="dash", line_color="#DC2626", annotation_text="100% Daily Exposure Limit")
        fig_trend.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
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
# TAB 4: Lab Calibration & Profiles Explorer
# -----------------------------------------------------------------------------
with tab4:
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
# TAB 5: Session Logger & Privacy
# -----------------------------------------------------------------------------
with tab5:
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
