"""
metrics.py
----------
Computes all fluency metrics from the tagged word list:

  FPM   - Fillers Per Minute
  WPM   - Words Per Minute (fluent words only)
  pauses - count and total duration of long pauses
  fluency_score - composite 0–100 score

Also produces a per-minute breakdown for trend analysis
(useful for showing *where* in the lecture fluency dropped).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import math
from classify import TaggedWord, Label


# ── Result dataclass ───────────────────────────────────────────────────────────
@dataclass
class FluentyMetrics:
    # Core
    duration_s: float          # total speech duration (seconds)
    total_words: int           # all real (non-synthetic) words
    fluent_words: int          # words labelled FLUENT
    filled_pauses: int         # FILLED_PAUSE count
    false_starts: int          # FALSE_START count
    repetitions: int           # REPETITION count
    long_pauses: int           # LONG_PAUSE synthetic token count
    total_pause_time_s: float  # total seconds of long pauses

    # Rates
    wpm: float                 # words per minute (fluent only)
    fpm: float                 # fillers per minute
    pause_freq: float          # long pauses per minute

    # Trend (per-minute breakdown)
    per_minute: List[Dict] = field(default_factory=list)

    # Composite score
    fluency_score: int = 0
    grade: str = ""
    verdict: str = ""

    # Detailed counts by filler text
    filler_breakdown: Dict[str, int] = field(default_factory=dict)


def _rate(count: int, duration_s: float) -> float:
    """Events per minute."""
    if duration_s <= 0:
        return 0.0
    return round(count / (duration_s / 60), 2)


def _compute_fluency_score(fpm: float, wpm: float,
                            pause_freq: float,
                            false_starts: int,
                            duration_s: float) -> int:
    """
    Composite Fluency Score (0–100).

    Scoring breakdown:
      40 pts — Filler rate (FPM)
        40 pts if FPM = 0
         0 pts if FPM >= 10
      30 pts — Speaking rate (WPM)
        ideal range: 120–160 WPM
        penalty for too slow (<100) or too fast (>180)
      20 pts — Long pause frequency
        20 pts if 0 pauses/min
         0 pts if >= 4 pauses/min
      10 pts — False start rate
        10 pts if 0 false starts
         0 pts if >= 5 false starts/min
    """
    # Filler component (40 pts)
    filler_pts = max(0, 40 * (1 - fpm / 10))

    # WPM component (30 pts)
    if 120 <= wpm <= 160:
        wpm_pts = 30
    elif wpm < 120:
        wpm_pts = max(0, 30 * (wpm / 120))
    else:  # > 160
        wpm_pts = max(0, 30 * (1 - (wpm - 160) / 60))

    # Pause component (20 pts)
    pause_pts = max(0, 20 * (1 - pause_freq / 4))

    # False start component (10 pts)
    fs_rate = _rate(false_starts, duration_s)
    fs_pts = max(0, 10 * (1 - fs_rate / 5))

    score = int(round(filler_pts + wpm_pts + pause_pts + fs_pts))
    return min(100, max(0, score))


def _grade(score: int) -> tuple:
    """Return (letter_grade, verdict)."""
    if score >= 90:
        return "A", "Excellent fluency — minimal disfluencies, great pace"
    elif score >= 75:
        return "B", "Good fluency — a few fillers, mostly clear delivery"
    elif score >= 60:
        return "C", "Moderate fluency — noticeable fillers, room to improve"
    elif score >= 45:
        return "D", "Below average — frequent disfluencies affecting clarity"
    else:
        return "F", "Poor fluency — heavy disfluency load, strongly impacts comprehension"


def compute_metrics(tagged: List[TaggedWord]) -> FluentyMetrics:
    """
    Compute all fluency metrics from a list of TaggedWord objects.
    """
    real_words = [tw for tw in tagged if not tw.is_synthetic]
    if not real_words:
        raise ValueError("No words found — check transcription output")

    duration_s = real_words[-1].end - real_words[0].start
    if duration_s <= 0:
        duration_s = 1.0

    # Counts
    fluent      = [w for w in real_words if w.label == Label.FLUENT]
    fp          = [w for w in real_words if w.label == Label.FILLED_PAUSE]
    fs          = [w for w in real_words if w.label == Label.FALSE_START]
    reps        = [w for w in real_words if w.label == Label.REPETITION]
    pauses      = [w for w in tagged    if w.label == Label.LONG_PAUSE]

    total_pause_time = sum(w.end - w.start for w in pauses)

    # Filler breakdown
    filler_breakdown: Dict[str, int] = {}
    for w in fp:
        key = w.text.strip().lower().rstrip(".,!?")
        filler_breakdown[key] = filler_breakdown.get(key, 0) + 1

    # Rates
    wpm        = _rate(len(fluent), duration_s)
    fpm        = _rate(len(fp), duration_s)
    pause_freq = _rate(len(pauses), duration_s)

    # Per-minute breakdown
    per_minute = []
    num_minutes = max(1, math.ceil(duration_s / 60))
    for m in range(num_minutes):
        t_start = real_words[0].start + m * 60
        t_end   = t_start + 60
        minute_words = [w for w in real_words if t_start <= w.start < t_end]
        minute_pauses = [w for w in pauses if t_start <= w.start < t_end]
        minute_dur = min(60, duration_s - m * 60)

        mf = [w for w in minute_words if w.label == Label.FLUENT]
        mfp = [w for w in minute_words if w.label == Label.FILLED_PAUSE]

        per_minute.append({
            "minute": m + 1,
            "t_start": round(t_start, 2),
            "t_end": round(min(t_end, real_words[-1].end), 2),
            "wpm": _rate(len(mf), minute_dur),
            "fpm": _rate(len(mfp), minute_dur),
            "long_pauses": len(minute_pauses),
            "filled_pauses": len(mfp),
            "fluent_words": len(mf),
        })

    # Composite score
    score  = _compute_fluency_score(fpm, wpm, pause_freq, len(fs), duration_s)
    grade, verdict = _grade(score)

    metrics = FluentyMetrics(
        duration_s        = round(duration_s, 2),
        total_words       = len(real_words),
        fluent_words      = len(fluent),
        filled_pauses     = len(fp),
        false_starts      = len(fs),
        repetitions       = len(reps),
        long_pauses       = len(pauses),
        total_pause_time_s= round(total_pause_time, 2),
        wpm               = wpm,
        fpm               = fpm,
        pause_freq        = pause_freq,
        per_minute        = per_minute,
        fluency_score     = score,
        grade             = grade,
        verdict           = verdict,
        filler_breakdown  = filler_breakdown,
    )

    _print_summary(metrics)
    return metrics


def _print_summary(m: FluentyMetrics):
    print("\n" + "="*52)
    print("  FLUENCY METRICS SUMMARY")
    print("="*52)
    print(f"  Duration         : {m.duration_s:.1f}s  ({m.duration_s/60:.1f} min)")
    print(f"  Total words      : {m.total_words}")
    print(f"  Fluent words     : {m.fluent_words}")
    print(f"  Filled pauses    : {m.filled_pauses}  ({m.fpm:.1f}/min)")
    print(f"  False starts     : {m.false_starts}")
    print(f"  Repetitions      : {m.repetitions}")
    print(f"  Long pauses      : {m.long_pauses}  ({m.total_pause_time_s:.1f}s total)")
    print(f"  Speaking rate    : {m.wpm:.0f} WPM")
    print(f"  Filler rate      : {m.fpm:.1f} FPM")
    print("-"*52)
    print(f"  FLUENCY SCORE    : {m.fluency_score}/100  [{m.grade}]")
    print(f"  {m.verdict}")
    print("="*52)
    if m.filler_breakdown:
        print("  Top fillers:")
        for k, v in sorted(m.filler_breakdown.items(),
                            key=lambda x: -x[1])[:5]:
            print(f"    '{k}' × {v}")


if __name__ == "__main__":
    import json, sys
    from classify import TaggedWord

    path = sys.argv[1] if len(sys.argv) > 1 else "demo/tagged.json"
    with open(path) as f:
        raw = json.load(f)
    tagged = [TaggedWord(**w) for w in raw]

    m = compute_metrics(tagged)

    import dataclasses
    out = dataclasses.asdict(m)
    with open("demo/metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved to demo/metrics.json")
