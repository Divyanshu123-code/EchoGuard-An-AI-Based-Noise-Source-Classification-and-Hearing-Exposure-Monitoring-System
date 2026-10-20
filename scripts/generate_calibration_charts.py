#!/usr/bin/env python3
"""
scripts/generate_calibration_charts.py

Generates publication-quality charts for EchoGuard Role 2 deliverables:
1. docs/figures/dba_vs_volume_step.png: dBA against volume step (one line per signal, for both headphones)
2. docs/figures/repeatability_test.png: Repeatability test with error bars (mean ± std dev across 5 trials)
3. docs/figures/headphone_comparison.png: Over-ear vs in-ear profile comparison with NIOSH 85 dBA line
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "lines.linewidth": 2.2,
    "lines.markersize": 6
})

def generate_charts():
    figures_dir = "docs/figures"
    os.makedirs(figures_dir, exist_ok=True)

    df_cal = pd.read_csv("data/calibration.csv")
    df_prof = pd.read_csv("data/headphone_profiles.csv")

    # Filter out repeatability notes for volume curve plots
    df_main = df_cal[~df_cal["note"].str.startswith("repeatability")]

    # -------------------------------------------------------------------------
    # Chart 1: dBA against Volume Step for each headphone and each signal
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    headphones = [
        ("sony_wh1000xm4", "Sony WH-1000XM4 (Over-Ear ANC)", axes[0]),
        ("galaxy_buds_fe", "Samsung Galaxy Buds FE (In-Ear TWS)", axes[1])
    ]

    signal_styles = {
        "tone1k": {"color": "#1f77b4", "marker": "o", "label": "1 kHz Sine Tone"},
        "pink": {"color": "#d62728", "marker": "s", "label": "Pink Noise (1/f)"},
        "music1": {"color": "#2ca02c", "marker": "^", "label": "Music 1 (Acoustic Harmonics)"},
        "music2": {"color": "#9467bd", "marker": "d", "label": "Music 2 (Dynamic Synth/Bass)"}
    }

    for hp_id, hp_title, ax in headphones:
        hp_df = df_main[df_main["headphone"] == hp_id]
        for sig, style in signal_styles.items():
            sig_df = hp_df[hp_df["signal"] == sig].sort_values("volume_step")
            ax.plot(
                sig_df["volume_step"],
                sig_df["dba"],
                label=style["label"],
                color=style["color"],
                marker=style["marker"]
            )
        
        # Reference threshold lines
        ax.axhline(85, color="#e65100", linestyle="--", linewidth=1.5, alpha=0.85, label="NIOSH REL (85 dBA)")
        ax.axhline(70, color="#2e7d32", linestyle=":", linewidth=1.3, alpha=0.75, label="Safe Baseline (70 dBA)")
        
        ax.set_title(hp_title, fontweight="bold", pad=12)
        ax.set_xlabel("Volume Step (1 to 15)")
        ax.set_xticks(range(1, 16))
        ax.grid(True, linestyle="--", alpha=0.6)

    axes[0].set_ylabel("Measured Sound Level (dBA)")
    axes[0].legend(loc="upper left", framealpha=0.92)
    fig.suptitle("EchoGuard: Calibrated Sound Level vs. Volume Step", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    chart1_path = os.path.join(figures_dir, "dba_vs_volume_step.png")
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    print(f"Generated: {chart1_path}")

    # -------------------------------------------------------------------------
    # Chart 2: Repeatability Test with Error Bars
    # -------------------------------------------------------------------------
    df_rep = df_cal[df_cal["note"].str.startswith("repeatability")]
    rep_summary = df_rep.groupby("volume_step")["dba"].agg(["mean", "std", "count"]).reset_index()

    fig, ax = plt.subplots(figsize=(8, 6))
    x = rep_summary["volume_step"]
    y = rep_summary["mean"]
    yerr = rep_summary["std"]

    # Bar with error bars
    bar_colors = ["#42a5f5", "#26a69a", "#ab47bc"]
    bars = ax.bar(x, y, width=1.4, color=bar_colors, alpha=0.75, edgecolor="#37474f", linewidth=1.2, label="Mean Measured dBA")
    ax.errorbar(
        x, y, yerr=yerr,
        fmt="none", ecolor="#b71c1c", elinewidth=2.2, capsize=8, capthick=2.0,
        label="±1 Std Dev (Repositioning Error)"
    )

    # Annotate values on top of bars
    for idx, row in rep_summary.iterrows():
        step = int(row["volume_step"])
        m = row["mean"]
        s = row["std"]
        step_label = "Low (Step 3)" if step == 3 else ("Mid (Step 8)" if step == 8 else "High (Step 13)")
        ax.text(
            step, m + s + 1.2,
            f"{m:.1f} ± {s:.2f} dBA\n({step_label})",
            ha="center", va="bottom", fontsize=10, fontweight="bold", color="#212121"
        )

    ax.set_title("Repeatability Test: Repositioning Variance (5 Trials per Step)", fontweight="bold", pad=12)
    ax.set_xlabel("Volume Step")
    ax.set_ylabel("Sound Level (dBA)")
    ax.set_xticks([3, 8, 13])
    ax.set_xticklabels(["Step 3 (Low)", "Step 8 (Mid)", "Step 13 (High)"])
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.legend(loc="upper left")

    plt.tight_layout()
    chart2_path = os.path.join(figures_dir, "repeatability_test.png")
    plt.savefig(chart2_path, dpi=300)
    plt.close()
    print(f"Generated: {chart2_path}")

    # -------------------------------------------------------------------------
    # Chart 3: Headphone Profile Comparison
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    for hp_id, grp in df_prof.groupby("headphone"):
        hp_label = "Sony WH-1000XM4 (Over-Ear)" if "sony" in hp_id else "Samsung Galaxy Buds FE (In-Ear)"
        color = "#1565c0" if "sony" in hp_id else "#e91e63"
        marker = "o" if "sony" in hp_id else "s"
        ax.plot(grp["step"], grp["dba"], label=hp_label, color=color, marker=marker, linewidth=2.5)
        # Shaded uncertainty band (±5.0 dB)
        ax.fill_between(grp["step"], grp["dba"] - 5.0, grp["dba"] + 5.0, color=color, alpha=0.15)

    ax.axhline(85, color="#d32f2f", linestyle="--", linewidth=1.5, label="NIOSH 85 dBA Threshold (8h Limit)")
    ax.set_title("Calibrated Profile Lookup Curves with ±5 dB Uncertainty Band", fontweight="bold")
    ax.set_xlabel("Volume Step (1 to 15)")
    ax.set_ylabel("Estimated Ear Sound Level (dBA)")
    ax.set_xticks(range(1, 16))
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")

    plt.tight_layout()
    chart3_path = os.path.join(figures_dir, "headphone_comparison.png")
    plt.savefig(chart3_path, dpi=300)
    plt.close()
    print(f"Generated: {chart3_path}")

if __name__ == "__main__":
    generate_charts()
