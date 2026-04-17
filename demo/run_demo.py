"""
demo/run_demo.py
----------------
Runs the full pipeline on SYNTHETIC data — no GPU, no real audio needed.
Simulates a 5-minute lecture with realistic disfluency patterns.

Run from the project root:
  python demo/run_demo.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import numpy as np
import soundfile as sf
import dataclasses

# ── Synthetic lecture transcript ───────────────────────────────────────────────
DEMO_TRANSCRIPT = [
    # (text, duration, pause_before)   pause_before = gap from previous word end
    ("Today",       0.30, 0.00),
    ("we",          0.12, 0.05),
    ("are",         0.15, 0.04),
    ("going",       0.25, 0.05),
    ("to",          0.10, 0.04),
    ("discuss",     0.35, 0.06),
    ("um",          0.30, 0.08),   # filler
    ("the",         0.12, 0.06),
    ("fundamentals",0.55, 0.05),
    ("of",          0.10, 0.04),
    ("machine",     0.35, 0.05),
    ("uh",          0.20, 0.10),   # filler
    ("machine",     0.35, 0.08),   # repetition
    ("learning",    0.40, 0.05),
    ("so",          0.20, 0.07),   # filler
    ("the",         0.12, 0.05),
    ("first",       0.25, 0.05),
    ("con-",        0.10, 0.06),   # false start (truncated)
    ("concept",     0.35, 0.04),
    ("we",          0.12, 0.05),
    ("need",        0.22, 0.05),
    ("to",          0.10, 0.04),
    ("understand",  0.50, 0.05),
    ("is",          0.12, 0.05),
    ("you know",    0.45, 0.09),   # filler
    ("gradient",    0.40, 0.06),
    ("descent",     0.40, 0.05),
    ("basically",   0.40, 0.08),   # filler
    ("it",          0.12, 0.90),   # long pause before this
    ("is",          0.12, 0.05),
    ("an",          0.10, 0.04),
    ("optimisation",0.55, 0.05),
    ("algorithm",   0.50, 0.05),
    ("that",        0.18, 0.05),
    ("um",          0.25, 0.08),   # filler
    ("minimises",   0.45, 0.06),
    ("the",         0.12, 0.05),
    ("loss",        0.25, 0.05),
    ("function",    0.40, 0.05),
    ("right",       0.22, 0.07),   # filler
    ("so",          0.18, 0.08),   # filler
    ("imagine",     0.40, 0.05),
    ("you",         0.12, 0.05),
    ("are",         0.15, 0.04),
    ("standing",    0.38, 0.05),
    ("on",          0.10, 0.04),
    ("a",           0.08, 0.04),
    ("hill",        0.22, 0.05),
    ("and",         0.14, 0.05),
    ("you",         0.12, 0.05),
    ("want",        0.22, 0.05),
    ("to",          0.10, 0.04),
    ("reach",       0.28, 0.05),
    ("the",         0.12, 0.05),
    ("uh",          0.18, 0.09),   # filler
    ("lowest",      0.35, 0.06),
    ("point",       0.28, 0.05),
    ("um",          0.22, 0.95),   # filler after long pause
    ("each",        0.22, 0.06),
    ("step",        0.25, 0.05),
    ("you",         0.12, 0.05),
    ("take",        0.22, 0.05),
    ("is",          0.12, 0.04),
    ("determined",  0.48, 0.05),
    ("by",          0.12, 0.05),
    ("the",         0.12, 0.04),
    ("gradient",    0.40, 0.05),
    ("like",        0.20, 0.08),   # filler
    ("the",         0.12, 0.05),
    ("steepness",   0.45, 0.05),
    ("of",          0.10, 0.04),
    ("the",         0.12, 0.04),
    ("slope",       0.28, 0.05),
    ("at",          0.10, 0.05),
    ("that",        0.18, 0.04),
    ("point",       0.28, 0.05),
    ("so",          0.18, 0.08),   # filler
    ("mathematically",0.65, 0.05),
    ("uh",          0.18, 0.09),   # filler
    ("we",          0.12, 0.06),
    ("update",      0.35, 0.05),
    ("the",         0.12, 0.04),
    ("weights",     0.32, 0.05),
    ("by",          0.12, 0.04),
    ("subtracting", 0.50, 0.05),
    ("the",         0.12, 0.04),
    ("gradient",    0.40, 0.05),
    ("times",       0.25, 0.05),
    ("the",         0.12, 0.04),
    ("learning",    0.40, 0.05),
    ("rate",        0.22, 0.05),
    ("um",          0.25, 1.20),   # long pause then filler
    ("and",         0.14, 0.06),
    ("we",          0.12, 0.05),
    ("repeat",      0.35, 0.05),
    ("this",        0.18, 0.04),
    ("process",     0.38, 0.05),
    ("until",       0.28, 0.05),
    ("basically",   0.40, 0.08),   # filler
    ("convergence", 0.50, 0.05),
    ("thank",       0.28, 0.88),   # long pause
    ("you",         0.20, 0.05),
]


def make_synthetic_audio(words_data, sr=16000, out_path="demo/sample.wav"):
    """Generate synthetic speech audio matching word timings."""
    # Build timeline
    t = 0.0
    chunks = []
    for text, duration, pause in words_data:
        t += pause
        # silence before word
        if pause > 0:
            chunks.append(np.zeros(int(pause * sr)))
        # word = short sine burst (simplified 'speech')
        n = int(duration * sr)
        freq = 200 + hash(text) % 200
        wave = 0.3 * np.sin(2 * np.pi * freq * np.arange(n) / sr)
        # Apply envelope
        env = np.ones(n)
        fade = min(int(0.01 * sr), n // 4)
        env[:fade] = np.linspace(0, 1, fade)
        env[-fade:] = np.linspace(1, 0, fade)
        chunks.append(wave * env)
        t += duration

    audio = np.concatenate(chunks).astype(np.float32)
    sf.write(out_path, audio, sr)
    print(f"  Generated synthetic audio: {len(audio)/sr:.1f}s → {out_path}")
    return audio, sr


def build_word_objects(words_data):
    """Build Word objects from demo transcript."""
    from transcribe import Word
    words = []
    t = 0.0
    for text, duration, pause in words_data:
        t += pause
        words.append(Word(
            text=text,
            start=round(t, 3),
            end=round(t + duration, 3),
            confidence=0.95,
        ))
        t += duration
    return words


def main():
    os.makedirs("demo", exist_ok=True)
    print("\n" + "="*60)
    print("  LECTURE DISFLUENCY DETECTOR — DEMO RUN")
    print("  (Synthetic data — no GPU required)")
    print("="*60)

    # Generate synthetic audio
    print("\n[1/5] Generating synthetic lecture audio...")
    audio, sr = make_synthetic_audio(DEMO_TRANSCRIPT, out_path="demo/sample.wav")

    # Build word objects (skip actual ASR — use demo transcript directly)
    print("\n[2/5] Building word timeline from demo transcript...")
    from transcribe import Word
    words = build_word_objects(DEMO_TRANSCRIPT)
    print(f"  {len(words)} words, duration: {words[-1].end:.1f}s")

    with open("demo/words.json", "w") as f:
        json.dump([vars(w) for w in words], f, indent=2)

    # Classify
    print("\n[3/5] Classifying disfluencies...")
    from classify import classify
    tagged = classify(words, pause_threshold=0.8)

    with open("demo/tagged.json", "w") as f:
        json.dump([vars(tw) for tw in tagged], f, indent=2)

    # Metrics
    print("\n[4/5] Computing metrics...")
    from metrics import compute_metrics
    metrics = compute_metrics(tagged)

    with open("demo/metrics.json", "w") as f:
        json.dump(dataclasses.asdict(metrics), f, indent=2)

    # Visualisations
    print("\n[5/5] Generating visualisations...")
    from visualise import plot_word_timeline, plot_spectrogram, plot_trend
    plot_word_timeline(tagged, metrics.duration_s, "demo/timeline.png")
    plot_spectrogram("demo/sample.wav", tagged, "demo/spectrogram.png")
    plot_trend(metrics, "demo/trend.png")

    # Report card
    print("\n[6/6] Generating PDF report card...")
    from report_card import generate_report_card
    generate_report_card(
        metrics, tagged,
        audio_filename="sample_lecture.wav",
        speaker_name="Demo Speaker",
        out_path="demo/report_card.pdf",
        timeline_img="demo/timeline.png",
        trend_img="demo/trend.png",
        spectrogram_img="demo/spectrogram.png",
    )

    print("\n" + "="*60)
    print("  ✅ DEMO COMPLETE")
    print("="*60)
    print(f"  Score : {metrics.fluency_score}/100  [{metrics.grade}]")
    print(f"  FPM   : {metrics.fpm}")
    print(f"  WPM   : {metrics.wpm}")
    print("\n  Output files in demo/:")
    for f in sorted(os.listdir("demo")):
        size = os.path.getsize(f"demo/{f}")
        print(f"    ✓ demo/{f}  ({size//1024} KB)")
    print("="*60)


if __name__ == "__main__":
    main()
