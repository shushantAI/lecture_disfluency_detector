"""
visualise.py
------------
Generates two figures:

  1. Word Timeline  — colour-coded word chips on a time axis
  2. Mel Spectrogram + Overlay — spectrogram with disfluency markers
  3. Per-minute trend chart  — FPM / WPM bar chart over time

Colours:
  FLUENT       → green   (#1E8449)
  FILLED_PAUSE → coral   (#E94560)
  FALSE_START  → orange  (#F5A623)
  REPETITION   → purple  (#6C3483)
  LONG_PAUSE   → amber   (#C87D10) (dashed region)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import librosa
import librosa.display
from typing import List, Optional
from classify import TaggedWord, Label
from metrics import FluentyMetrics

# ── Colour map ─────────────────────────────────────────────────────────────────
COLOURS = {
    Label.FLUENT:       "#1E8449",
    Label.FILLED_PAUSE: "#E94560",
    Label.FALSE_START:  "#F5A623",
    Label.REPETITION:   "#6C3483",
    Label.LONG_PAUSE:   "#C87D10",
}

LABEL_NAMES = {
    Label.FLUENT:       "Fluent",
    Label.FILLED_PAUSE: "Filled Pause (um/uh/like)",
    Label.FALSE_START:  "False Start",
    Label.REPETITION:   "Repetition",
    Label.LONG_PAUSE:   "Long Pause (>0.8s)",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ── Figure 1: Word Timeline ────────────────────────────────────────────────────
def plot_word_timeline(tagged: List[TaggedWord],
                       duration: float,
                       output_path: str = "demo/timeline.png",
                       max_words: int = 120):
    """
    Horizontal bar of word chips, colour-coded by disfluency label.
    Shows up to max_words from the beginning.
    """
    display_words = [w for w in tagged if not w.is_synthetic][:max_words]
    pauses = [w for w in tagged if w.label == Label.LONG_PAUSE]

    fig, ax = plt.subplots(figsize=(18, 3.5))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#16213E")

    t_end = display_words[-1].end if display_words else duration
    t_start = display_words[0].start if display_words else 0

    # Draw pause regions
    for p in pauses:
        if p.start > t_end:
            break
        ax.axvspan(p.start, p.end, alpha=0.25,
                   color=COLOURS[Label.LONG_PAUSE], zorder=1)

    # Draw word chips
    for w in display_words:
        colour = COLOURS[w.label]
        dur = max(w.end - w.start, 0.05)
        rect = FancyBboxPatch(
            (w.start, 0.15), dur, 0.7,
            boxstyle="round,pad=0.005",
            linewidth=0,
            facecolor=colour,
            alpha=0.92,
            zorder=2,
        )
        ax.add_patch(rect)

        # Only label if chip is wide enough
        if dur > 0.18:
            fontsize = max(5, min(8, dur * 18))
            ax.text(
                w.start + dur / 2, 0.5, w.text,
                ha="center", va="center",
                fontsize=fontsize, color="white",
                fontweight="bold", zorder=3,
                clip_on=True,
            )

    # Time axis
    ax.set_xlim(t_start, t_end)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time (seconds)", color="#AABBCC", fontsize=10)
    ax.tick_params(colors="#AABBCC", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334466")
    ax.set_yticks([])
    ax.set_title("Word-level Disfluency Timeline",
                 color="white", fontsize=13, pad=10, fontweight="bold")

    # Legend
    legend_patches = [
        mpatches.Patch(color=COLOURS[l], label=LABEL_NAMES[l])
        for l in [Label.FLUENT, Label.FILLED_PAUSE,
                  Label.FALSE_START, Label.REPETITION, Label.LONG_PAUSE]
    ]
    ax.legend(handles=legend_patches, loc="upper right",
              fontsize=8, framealpha=0.2,
              labelcolor="white", facecolor="#1A1A2E",
              edgecolor="#334466")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved timeline → {output_path}")


# ── Figure 2: Mel Spectrogram + Overlay ───────────────────────────────────────
def plot_spectrogram(audio_path: str,
                     tagged: List[TaggedWord],
                     output_path: str = "demo/spectrogram.png"):
    """
    Mel spectrogram with disfluency event markers overlaid.
    """
    import librosa
    import librosa.display

    print("  Computing Mel spectrogram...")
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_db = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=(18, 4))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#0D1117")

    img = librosa.display.specshow(
        S_db, x_axis="time", y_axis="mel",
        sr=sr, fmax=8000, ax=ax,
        cmap="magma"
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB",
                 pad=0.01).ax.yaxis.set_tick_params(color="white")

    # Overlay disfluency markers
    for w in tagged:
        if w.label in (Label.FLUENT, Label.LONG_PAUSE):
            continue
        colour = COLOURS[w.label]
        ax.axvspan(w.start, w.end, alpha=0.35, color=colour, zorder=3)
        ax.axvline(w.start, color=colour, linewidth=0.8, alpha=0.7, zorder=4)

    # Pause regions
    for w in tagged:
        if w.label == Label.LONG_PAUSE:
            ax.axvspan(w.start, w.end, alpha=0.15,
                       color=COLOURS[Label.LONG_PAUSE],
                       linestyle="--", zorder=2)

    ax.set_title("Mel Spectrogram with Disfluency Overlay",
                 color="white", fontsize=13, pad=8, fontweight="bold")
    ax.tick_params(colors="#AABBCC", labelsize=9)
    ax.xaxis.label.set_color("#AABBCC")
    ax.yaxis.label.set_color("#AABBCC")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334466")

    # Mini legend
    legend_patches = [
        mpatches.Patch(color=COLOURS[l], label=LABEL_NAMES[l], alpha=0.7)
        for l in [Label.FILLED_PAUSE, Label.FALSE_START,
                  Label.REPETITION, Label.LONG_PAUSE]
    ]
    ax.legend(handles=legend_patches, loc="upper right",
              fontsize=8, framealpha=0.3,
              labelcolor="white", facecolor="#1A1A2E",
              edgecolor="#334466")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved spectrogram → {output_path}")


# ── Figure 3: Per-minute trend ─────────────────────────────────────────────────
def plot_trend(metrics: FluentyMetrics,
               output_path: str = "demo/trend.png"):
    """
    Side-by-side bar chart of FPM and WPM per minute.
    Helps speakers see exactly where in the lecture fluency dropped.
    """
    pm = metrics.per_minute
    if not pm:
        print("  No per-minute data — skipping trend chart")
        return

    minutes = [f"Min {d['minute']}" for d in pm]
    fpm_vals = [d["fpm"] for d in pm]
    wpm_vals = [d["wpm"] for d in pm]
    x = np.arange(len(minutes))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, len(pm) * 1.4), 6),
                                    sharex=True)
    fig.patch.set_facecolor("#1A1A2E")

    # FPM bars
    ax1.set_facecolor("#16213E")
    bars1 = ax1.bar(x, fpm_vals, color="#E94560", alpha=0.85, width=0.6)
    ax1.axhline(metrics.fpm, color="#F5A623", linewidth=1.5,
                linestyle="--", label=f"Avg FPM: {metrics.fpm:.1f}")
    ax1.set_ylabel("Fillers / min", color="#AABBCC", fontsize=10)
    ax1.set_title("Filler Rate per Minute", color="white",
                  fontsize=12, fontweight="bold")
    ax1.tick_params(colors="#AABBCC")
    ax1.legend(labelcolor="white", facecolor="#1A1A2E",
               edgecolor="#334466", fontsize=9)
    for spine in ax1.spines.values():
        spine.set_edgecolor("#334466")
    # Value labels
    for bar, val in zip(bars1, fpm_vals):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     f"{val:.1f}", ha="center", va="bottom",
                     fontsize=8, color="white")

    # WPM bars
    ax2.set_facecolor("#16213E")
    bars2 = ax2.bar(x, wpm_vals, color="#1E8449", alpha=0.85, width=0.6)
    ax2.axhspan(120, 160, alpha=0.15, color="#27AE60", label="Ideal range")
    ax2.axhline(metrics.wpm, color="#5DADE2", linewidth=1.5,
                linestyle="--", label=f"Avg WPM: {metrics.wpm:.0f}")
    ax2.set_ylabel("Words / min", color="#AABBCC", fontsize=10)
    ax2.set_title("Speaking Rate per Minute", color="white",
                  fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(minutes, color="#AABBCC", fontsize=9)
    ax2.tick_params(colors="#AABBCC")
    ax2.legend(labelcolor="white", facecolor="#1A1A2E",
               edgecolor="#334466", fontsize=9)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#334466")

    plt.tight_layout(pad=1.5)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved trend chart → {output_path}")


def generate_all_visualisations(audio_path: str,
                                 tagged: List[TaggedWord],
                                 metrics: FluentyMetrics,
                                 out_dir: str = "demo"):
    """Generate all three visualisation figures."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    print("\n[Visualise] Generating figures...")
    plot_word_timeline(tagged, metrics.duration_s,
                       f"{out_dir}/timeline.png")
    plot_spectrogram(audio_path, tagged,
                     f"{out_dir}/spectrogram.png")
    plot_trend(metrics, f"{out_dir}/trend.png")
    print("  All figures saved.")


if __name__ == "__main__":
    import json, sys
    from classify import TaggedWord
    from metrics import FluentyMetrics
    import dataclasses

    audio_path = sys.argv[1] if len(sys.argv) > 1 else "demo/sample.wav"

    with open("demo/tagged.json") as f:
        tagged = [TaggedWord(**w) for w in json.load(f)]
    with open("demo/metrics.json") as f:
        m_dict = json.load(f)
    metrics = FluentyMetrics(**m_dict)

    generate_all_visualisations(audio_path, tagged, metrics)
