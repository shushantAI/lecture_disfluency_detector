"""
pipeline.py

This is the main script to run the whole disfluency detection system.
You give it an audio file and it runs through transcription, classification,
metrics, visualisation and report card generation in order.
You can also import and call run() directly from another script if needed.
"""

import argparse
import json
import os
import time
import dataclasses
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lecture Speech Disfluency Detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py lecture.wav
  python pipeline.py lecture.wav --speaker "Dr. Sharma" --out results/
  python pipeline.py lecture.wav --no-vad --pause-threshold 1.0
        """
    )
    parser.add_argument("audio", help="Path to audio file (wav, mp3, m4a, flac)")
    parser.add_argument("--speaker", default="Speaker",
                        help="Speaker name for report card (default: Speaker)")
    parser.add_argument("--out", default="output",
                        help="Output directory (default: output/)")
    parser.add_argument("--model", default="nyrahealth/CrisperWhisper",
                        help="ASR model (default: nyrahealth/CrisperWhisper)")
    parser.add_argument("--no-vad", action="store_true",
                        help="Disable VAD pre-filtering")
    parser.add_argument("--pause-threshold", type=float, default=0.8,
                        help="Long pause threshold in seconds (default: 0.8)")
    parser.add_argument("--no-report", action="store_true",
                        help="Skip PDF report card generation")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip visualisation generation")
    return parser.parse_args()


def run(audio_path: str,
        speaker: str = "Speaker",
        out_dir: str = "output",
        model: str = "nyrahealth/CrisperWhisper",
        use_vad: bool = True,
        pause_threshold: float = 0.8,
        generate_report: bool = True,
        generate_viz: bool = True):
    t0 = time.time()

    os.makedirs(out_dir, exist_ok=True)
    stem = Path(audio_path).stem

    print("\n" + "="*60)
    print("  LECTURE SPEECH DISFLUENCY DETECTOR")
    print("="*60)
    print(f"  Audio    : {audio_path}")
    print(f"  Speaker  : {speaker}")
    print(f"  Model    : {model}")
    print(f"  Output   : {out_dir}/")
    print("="*60)


    print("\n📝 STEP 1 — Transcription")
    from transcribe import transcribe
    words = transcribe(audio_path, use_vad=use_vad, model=model)

    if not words:
        print("ERROR: No words transcribed. Check audio file.")
        return None

    words_path = os.path.join(out_dir, f"{stem}_words.json")
    with open(words_path, "w") as f:
        json.dump([dataclasses.asdict(w) if hasattr(w, '__dataclass_fields__')
                   else vars(w) for w in words], f, indent=2)
    print(f"  Saved {len(words)} words → {words_path}")


    print("\n🏷️  STEP 2 — Disfluency Classification")
    from classify import classify
    tagged = classify(words, pause_threshold=pause_threshold)

    tagged_path = os.path.join(out_dir, f"{stem}_tagged.json")
    with open(tagged_path, "w") as f:
        json.dump([vars(tw) for tw in tagged], f, indent=2)
    print(f"  Saved {len(tagged)} tagged tokens → {tagged_path}")


    print("\n📊 STEP 3 — Metrics")
    from metrics import compute_metrics
    metrics = compute_metrics(tagged)

    metrics_path = os.path.join(out_dir, f"{stem}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(dataclasses.asdict(metrics), f, indent=2)
    print(f"  Saved metrics → {metrics_path}")

    timeline_img    = os.path.join(out_dir, f"{stem}_timeline.png")
    spectrogram_img = os.path.join(out_dir, f"{stem}_spectrogram.png")
    trend_img       = os.path.join(out_dir, f"{stem}_trend.png")

    if generate_viz:
        print("\n🎨 STEP 4 — Visualisations")
        from visualise import (plot_word_timeline, plot_spectrogram,
                               plot_trend)
        plot_word_timeline(tagged, metrics.duration_s, timeline_img)
        plot_spectrogram(audio_path, tagged, spectrogram_img)
        plot_trend(metrics, trend_img)

    report_path = os.path.join(out_dir, f"{stem}_report_card.pdf")

    if generate_report:
        print("\n📄 STEP 5 — Report Card")
        from report_card import generate_report_card
        generate_report_card(
            metrics, tagged,
            audio_filename=os.path.basename(audio_path),
            speaker_name=speaker,
            out_path=report_path,
            timeline_img=timeline_img,
            trend_img=trend_img,
            spectrogram_img=spectrogram_img,
        )


    elapsed = time.time() - t0
    print("\n" + "="*60)
    print(f"  ✅ DONE in {elapsed:.1f}s")
    print("="*60)
    print(f"  Fluency Score : {metrics.fluency_score}/100  [{metrics.grade}]")
    print(f"  FPM           : {metrics.fpm:.1f}")
    print(f"  WPM           : {metrics.wpm:.0f}")
    print(f"  Long pauses   : {metrics.long_pauses}")
    print(f"\n  Output files:")
    for p in [words_path, tagged_path, metrics_path,
              timeline_img, spectrogram_img, trend_img, report_path]:
        if os.path.exists(p):
            size_kb = os.path.getsize(p) / 1024
            print(f"    ✓ {p}  ({size_kb:.0f} KB)")
    print("="*60)

    return {
        "metrics": metrics,
        "tagged": tagged,
        "words": words,
        "paths": {
            "words":       words_path,
            "tagged":      tagged_path,
            "metrics":     metrics_path,
            "timeline":    timeline_img,
            "spectrogram": spectrogram_img,
            "trend":       trend_img,
            "report_card": report_path,
        }
    }


if __name__ == "__main__":
    args = parse_args()
    run(
        audio_path=args.audio,
        speaker=args.speaker,
        out_dir=args.out,
        model=args.model,
        use_vad=not args.no_vad,
        pause_threshold=args.pause_threshold,
        generate_report=not args.no_report,
        generate_viz=not args.no_viz,
    )
