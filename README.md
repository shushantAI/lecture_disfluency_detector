# 🎙️ Lecture Speech Disfluency Detector

> Automatically detects filler words, hesitations & false starts in lecture speech — with a **Speaker Fluency Report Card**

**Speech Understanding Course Project**


## 👨‍💻 Team Details

| Name | Roll Number |
|------|------------|
| Shushant Kumar Tiwari | M25DE1071 |
| Aniket Srivastava | M25DE1051 |
| Akshay Kumar | M25DE1028 |

---

## What It Does

Takes a raw lecture audio file and produces:

| Output | Description |
|---|---|
| **Word Timeline** | Colour-coded visual showing every word labelled by type |
| **Mel Spectrogram** | Spectrogram with disfluency overlays |
| **Trend Chart** | FPM + WPM per minute across the lecture |
| **Fluency Report Card** | PDF with score, metrics, recommendations |
| **JSON exports** | Machine-readable words, tags, metrics |

---

## Disfluency Types Detected

| Label | Examples | Colour |
|---|---|---|
| `FILLED_PAUSE` | um, uh, like, you know, basically | 🔴 Coral |
| `FALSE_START` | "ex— example", truncated words | 🟠 Amber |
| `REPETITION` | "the the", "I I think" | 🟣 Purple |
| `LONG_PAUSE` | silence > 0.8s between words | 🟡 Gold |
| `FLUENT` | normal speech | 🟢 Green |

---

## Fluency Score Formula

```
Score (0–100) =
  40 pts  — Filler Rate   (40 if FPM=0 → 0 if FPM≥10)
  30 pts  — Speaking Rate (ideal: 120–160 WPM)
  20 pts  — Pause Freq    (20 if 0 pauses/min → 0 if ≥4/min)
  10 pts  — False Starts  (10 if 0/min → 0 if ≥5/min)
```

| Grade | Score | Verdict |
|---|---|---|
| A | 90–100 | Excellent fluency |
| B | 75–89  | Good fluency |
| C | 60–74  | Moderate — room to improve |
| D | 45–59  | Below average |
| F | 0–44   | Heavy disfluency load |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the demo (no GPU, no real audio needed)

```bash
python demo/run_demo.py
```

Output files appear in `demo/`:
- `demo/timeline.png`
- `demo/spectrogram.png`
- `demo/trend.png`
- `demo/report_card.pdf`  ← main deliverable

### 3. Run on your own lecture

```bash
python pipeline.py lecture.wav --speaker "Dr. Sharma" --out results/
```

### 4. All options

```
python pipeline.py <audio> [options]

Options:
  --speaker NAME        Speaker name for report (default: Speaker)
  --out DIR             Output directory (default: output/)
  --model MODEL         ASR model (default: nyrahealth/CrisperWhisper)
  --no-vad              Disable Silero VAD pre-filtering
  --pause-threshold N   Long pause cutoff in seconds (default: 0.8)
  --no-report           Skip PDF report card
  --no-viz              Skip visualisation figures
```

---

## Pipeline Architecture

```
Lecture Audio (WAV/MP3)
        │
        ▼
  [Silero VAD]          ← strips silence segments
        │
        ▼
  [CrisperWhisper]      ← verbatim ASR, keeps um/uh, word timestamps
        │
        ▼
  [Disfluency Classifier]
    • Lexical matching  → FILLED_PAUSE
    • Sequence analysis → REPETITION, FALSE_START
    • Gap analysis      → LONG_PAUSE
        │
        ▼
  [Metrics Engine]      ← FPM, WPM, pause count, Fluency Score
        │
        ├──► [Timeline Viz]     → timeline.png
        ├──► [Spectrogram Viz]  → spectrogram.png
        ├──► [Trend Chart]      → trend.png
        └──► [Report Card]      → report_card.pdf
```

---

## Module Reference

| File | Purpose |
|---|---|
| `pipeline.py` | Main entry point — runs full pipeline |
| `transcribe.py` | CrisperWhisper verbatim ASR + VAD |
| `classify.py` | Disfluency type tagging |
| `metrics.py` | FPM, WPM, Fluency Score computation |
| `visualise.py` | Timeline, spectrogram, trend figures |
| `report_card.py` | PDF report card generator |
| `demo/run_demo.py` | Self-contained demo, no GPU needed |

---

## Datasets Used for Evaluation

| Dataset | Size | Use |
|---|---|---|
| TED-LIUM 3 | ~450 hrs | Primary F1 evaluation |
| Podcast-Fillers Corpus | ~5,000 clips | Filler detection testing |
| Self-recorded lectures | 4–6 sessions | Indian-English evaluation |
| Switchboard (subset) | — | Baseline model comparison |

---

## Responsible AI

**Fairness:** We evaluate FPM and F1 separately for native English, Indian-English, and other non-native speakers. Any disparity > ΔF1 0.05 is flagged.

**Transparency:** Every metric is formula-derived and shown explicitly in the report card. All source code is open.

---

## References

1. Wagner, L., Thallinger, B., & Zusag, M. (2024). CrisperWhisper. *INTERSPEECH 2024.*
2. Jamshid Lou, P. et al. (2018). Disfluency Detection using Auto-Correlational Neural Networks. *EMNLP 2018.*
3. Jamshid Lou, P. & Johnson, M. (2019). Improving Disfluency Detection by Self-Training. *NAACL 2019.*
4. Radford, A. et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision. *arXiv:2212.04356.*
5. Hernandez, F. et al. (2018). TED-LIUM 3. *SPECOM 2018.*
6. Christenfeld, N. (1995). Does it hurt to say um? *J. Nonverbal Behavior, 19(3).*
