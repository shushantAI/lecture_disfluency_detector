"""
evaluate.py

This script evaluates how well the classifier does against ground truth labels.
I compute precision, recall and F1 for each disfluency type, then average them.
There's also a fairness check that breaks down F1 by speaker accent group
to see if the model performs worse for non-native speakers.
"""

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from classify import TaggedWord, Label


@dataclass
class ClassMetrics:
    label: str
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


@dataclass
class EvalResult:
    per_class: Dict[str, ClassMetrics] = field(default_factory=dict)
    macro_f1: float = 0.0
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    accuracy: float = 0.0
    n_tokens: int = 0

    accent_f1: Dict[str, float] = field(default_factory=dict)
    fairness_gap: float = 0.0


EVAL_LABELS = [
    Label.FILLED_PAUSE,
    Label.FALSE_START,
    Label.REPETITION,
]


def _align_by_time(pred: List[TaggedWord],
                   gt: List[TaggedWord],
                   tol: float = 0.05) -> List[Tuple[TaggedWord, TaggedWord]]:
    pairs = []
    j = 0
    for p in pred:
        if p.is_synthetic:
            continue
        best_gt = None
        best_overlap = tol
        while j < len(gt) and gt[j].end < p.start - tol:
            j += 1
        k = j
        while k < len(gt) and gt[k].start < p.end + tol:
            overlap = min(p.end, gt[k].end) - max(p.start, gt[k].start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_gt = gt[k]
            k += 1
        if best_gt is not None:
            pairs.append((p, best_gt))
    return pairs


def evaluate_predictions(pred: List[TaggedWord],
                          gt: List[TaggedWord]) -> EvalResult:
    result = EvalResult()
    pairs = _align_by_time(pred, gt)
    result.n_tokens = len(pairs)

    if not pairs:
        print("  WARNING: No aligned pairs found — check timestamp format")
        return result

    class_metrics = {l: ClassMetrics(label=l) for l in EVAL_LABELS}
    correct = 0

    for p_word, g_word in pairs:
        p_label = p_word.label
        g_label = g_word.label

        if p_label == g_label:
            correct += 1
            if p_label in class_metrics:
                class_metrics[p_label].tp += 1
        else:
            if p_label in class_metrics:
                class_metrics[p_label].fp += 1
            if g_label in class_metrics:
                class_metrics[g_label].fn += 1

    result.per_class = class_metrics
    result.accuracy = correct / len(pairs) if pairs else 0.0
    result.macro_f1 = sum(m.f1 for m in class_metrics.values()) / len(class_metrics)
    result.macro_precision = sum(m.precision for m in class_metrics.values()) / len(class_metrics)
    result.macro_recall = sum(m.recall for m in class_metrics.values()) / len(class_metrics)

    return result


def print_eval_result(result: EvalResult, title: str = "Evaluation Results"):
    print(f"\n{'='*58}")
    print(f"  {title}")
    print(f"{'='*58}")
    print(f"  Tokens evaluated : {result.n_tokens}")
    print(f"  Accuracy         : {result.accuracy*100:.1f}%")
    print(f"  Macro F1         : {result.macro_f1*100:.1f}%")
    print(f"  Macro Precision  : {result.macro_precision*100:.1f}%")
    print(f"  Macro Recall     : {result.macro_recall*100:.1f}%")
    print(f"  {'-'*54}")
    print(f"  {'Label':<22} {'P':>6} {'R':>6} {'F1':>6} {'TP':>5} {'FP':>5} {'FN':>5}")
    print(f"  {'-'*54}")
    for label, m in result.per_class.items():
        print(f"  {label:<22} {m.precision*100:>5.1f}% {m.recall*100:>5.1f}% "
              f"{m.f1*100:>5.1f}% {m.tp:>5} {m.fp:>5} {m.fn:>5}")
    if result.accent_f1:
        print(f"  {'-'*54}")
        print(f"  Fairness breakdown:")
        for accent, f1 in sorted(result.accent_f1.items()):
            flag = "  ⚠️  GAP > 0.05" if abs(f1 - result.macro_f1) > 0.05 else ""
            print(f"    {accent:<24} F1={f1*100:.1f}%{flag}")
        print(f"  Fairness gap (max ΔF1): {result.fairness_gap*100:.1f}%")
    print(f"{'='*58}")



def load_tagged_json(path: str) -> List[TaggedWord]:
    with open(path) as f:
        data = json.load(f)
    return [TaggedWord(**d) for d in data]


def load_tedlium_gt(stm_path: str) -> List[TaggedWord]:
    words = []
    with open(stm_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";;"):
                continue
            parts = line.split(None, 6)
            if len(parts) < 7:
                continue
            try:
                seg_start = float(parts[3])
                seg_end   = float(parts[4])
                text      = parts[6]
            except (ValueError, IndexError):
                continue

            tokens = text.split()
            n = len(tokens)
            if n == 0:
                continue
            step = (seg_end - seg_start) / n

            for i, tok in enumerate(tokens):
                t_start = seg_start + i * step
                t_end   = t_start + step * 0.9

                if tok.startswith("<") and tok.endswith(">"):
                    inner = tok[1:-1].lower()
                    if inner in ("uh", "um", "hm", "ah", "er"):
                        label = Label.FILLED_PAUSE
                    else:
                        continue
                    clean_text = inner
                else:
                    label = Label.FLUENT
                    clean_text = tok.strip(".,!?;:")

                words.append(TaggedWord(
                    text=clean_text,
                    start=round(t_start, 3),
                    end=round(t_end, 3),
                    confidence=1.0,
                    label=label,
                ))
    return words


SPEAKER_ACCENT_MAP = {
    "AlGore_2006":          "native_en",
    "BillGates_2010":       "native_en",
    "DanDennett_2009":      "native_en",
    "RichardDawkins_2009":  "native_en",
    "SunilLalvani_2009":    "indian_en",
    "PranavMistry_2009":    "indian_en",
    "HanRosling_2006":      "non_native_other",
    "RicardoSemler_2014":   "non_native_other",
}


def fairness_report(results_by_speaker: Dict[str, EvalResult]) -> Dict[str, float]:
    group_f1s: Dict[str, List[float]] = defaultdict(list)

    for speaker_id, result in results_by_speaker.items():
        group = SPEAKER_ACCENT_MAP.get(speaker_id, "unknown")
        group_f1s[group].append(result.macro_f1)

    mean_f1 = {g: sum(vs)/len(vs) for g, vs in group_f1s.items() if vs}
    return mean_f1


def evaluate_tedlium(tedlium_root: str,
                     max_files: int = 20) -> EvalResult:
    from transcribe import transcribe
    from classify import classify

    stm_dir = os.path.join(tedlium_root, "test", "stm")
    wav_dir = os.path.join(tedlium_root, "test", "sph")

    if not os.path.isdir(stm_dir):
        print(f"  TED-LIUM STM directory not found: {stm_dir}")
        print("  Download from: https://huggingface.co/datasets/LIUM/tedlium")
        return EvalResult()

    stm_files = sorted(f for f in os.listdir(stm_dir) if f.endswith(".stm"))[:max_files]
    print(f"  Evaluating on {len(stm_files)} TED-LIUM files...")

    all_pred, all_gt = [], []
    per_speaker: Dict[str, EvalResult] = {}

    for stm_file in stm_files:
        speaker = stm_file.replace(".stm", "")
        stm_path = os.path.join(stm_dir, stm_file)

        wav_path = None
        for ext in [".wav", ".sph", ".mp3"]:
            p = os.path.join(wav_dir, speaker + ext)
            if os.path.exists(p):
                wav_path = p
                break

        if wav_path is None:
            print(f"  Skipping {speaker} — audio not found")
            continue

        print(f"  Processing {speaker}...")
        try:
            words = transcribe(wav_path)
            tagged_pred = classify(words)
            tagged_gt   = load_tedlium_gt(stm_path)

            result = evaluate_predictions(tagged_pred, tagged_gt)
            per_speaker[speaker] = result
            all_pred.extend(tagged_pred)
            all_gt.extend(tagged_gt)
        except Exception as e:
            print(f"  Error on {speaker}: {e}")
            continue

    agg = evaluate_predictions(all_pred, all_gt)

    accent_f1 = fairness_report(per_speaker)
    agg.accent_f1 = accent_f1
    if accent_f1:
        f1_vals = list(accent_f1.values())
        agg.fairness_gap = max(f1_vals) - min(f1_vals) if len(f1_vals) > 1 else 0.0

    return agg


def demo_evaluation():
    import random
    random.seed(42)

    gt_path = "demo/tagged.json"
    if not os.path.exists(gt_path):
        print("Run demo/run_demo.py first to generate demo data.")
        return

    gt = load_tagged_json(gt_path)

    # Simulate model predictions: copy GT but introduce ~15% label errors
    pred = []
    for tw in gt:
        import copy
        p = copy.copy(tw)
        if not tw.is_synthetic and random.random() < 0.15:
            alt = [l for l in [Label.FILLED_PAUSE, Label.FALSE_START,
                                Label.REPETITION, Label.FLUENT]
                   if l != tw.label]
            p.label = random.choice(alt)
        pred.append(p)

    result = evaluate_predictions(pred, gt)

    # Add synthetic fairness data
    result.accent_f1 = {
        "native_en":        result.macro_f1 + 0.03,
        "indian_en":        result.macro_f1 - 0.04,
        "non_native_other": result.macro_f1 - 0.07,
    }
    result.accent_f1 = {k: min(1.0, max(0.0, v))
                        for k, v in result.accent_f1.items()}
    f1_vals = list(result.accent_f1.values())
    result.fairness_gap = max(f1_vals) - min(f1_vals)

    print_eval_result(result, "Demo Evaluation (15% simulated noise)")

    import dataclasses
    out = {
        "macro_f1":        result.macro_f1,
        "macro_precision": result.macro_precision,
        "macro_recall":    result.macro_recall,
        "accuracy":        result.accuracy,
        "n_tokens":        result.n_tokens,
        "per_class": {
            l: {"precision": m.precision, "recall": m.recall, "f1": m.f1}
            for l, m in result.per_class.items()
        },
        "accent_f1":     result.accent_f1,
        "fairness_gap":  result.fairness_gap,
    }
    with open("demo/eval_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n  Saved → demo/eval_results.json")
    return result



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Disfluency Detector Evaluation")
    parser.add_argument("--gt",       help="Ground truth tagged JSON")
    parser.add_argument("--pred",     help="Predicted tagged JSON")
    parser.add_argument("--tedlium",  help="Path to TED-LIUM 3 root directory")
    parser.add_argument("--max-files",type=int, default=20,
                        help="Max TED-LIUM files to evaluate (default: 20)")
    parser.add_argument("--demo",     action="store_true",
                        help="Run demo evaluation on synthetic data")
    args = parser.parse_args()

    if args.demo:
        demo_evaluation()
    elif args.gt and args.pred:
        gt   = load_tagged_json(args.gt)
        pred = load_tagged_json(args.pred)
        result = evaluate_predictions(pred, gt)
        print_eval_result(result)
    elif args.tedlium:
        result = evaluate_tedlium(args.tedlium, max_files=args.max_files)
        print_eval_result(result, "TED-LIUM 3 Evaluation")
    else:
        parser.print_help()
