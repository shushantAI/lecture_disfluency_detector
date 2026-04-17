"""
classify.py
-----------
Classifies each word token as one of:
  FLUENT       - normal speech
  FILLED_PAUSE - um, uh, er, hmm, like (discourse), you know, basically, right
  FALSE_START  - word followed immediately by self-correction / restart
  REPETITION   - same word or phrase repeated consecutively
  LONG_PAUSE   - silence gap > threshold between words (not a word itself,
                 but inserted as a synthetic token)

Two-pass approach:
  Pass 1: Lexical matching for filled pauses
  Pass 2: Sequence analysis for repetitions and false starts
  Pass 3: Gap analysis for long pauses between words
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional
from transcribe import Word


# ── Label enum ────────────────────────────────────────────────────────────────
class Label:
    FLUENT       = "FLUENT"
    FILLED_PAUSE = "FILLED_PAUSE"
    FALSE_START  = "FALSE_START"
    REPETITION   = "REPETITION"
    LONG_PAUSE   = "LONG_PAUSE"   # synthetic — inserted between words


@dataclass
class TaggedWord:
    text: str
    start: float
    end: float
    confidence: float
    label: str = Label.FLUENT
    pause_before: float = 0.0    # silence gap before this word (seconds)
    is_synthetic: bool = False   # True for LONG_PAUSE tokens


# ── Filled pause lexicon ───────────────────────────────────────────────────────
FILLED_PAUSE_PATTERNS = [
    r"^u+m+$",           # um, umm, ummm
    r"^u+h+$",           # uh, uhh
    r"^e+r+$",           # er, err
    r"^h+m+$",           # hm, hmm, hmmm
    r"^mhm+$",           # mhm
    r"^ah+$",            # ah
    r"^oh+$",            # oh (hesitation use)
    r"^like$",           # discourse 'like'
    r"^basically$",
    r"^literally$",
    r"^actually$",
    r"^right\??$",       # 'right?' as filler
    r"^so+$",            # elongated 'so'
    r"^well$",
    r"^i mean$",
    r"^you know$",
    r"^you know what i mean$",
    r"^kind of$",
    r"^sort of$",
]

FILLED_PAUSE_EXACT = {
    "um", "uh", "er", "hmm", "hm", "mhm", "ah",
    "like", "basically", "literally", "actually",
    "right", "so", "well",
}

_FP_COMPILED = [re.compile(p, re.IGNORECASE) for p in FILLED_PAUSE_PATTERNS]


def _is_filled_pause(text: str) -> bool:
    t = text.strip().lower().rstrip(".,!?")
    if t in FILLED_PAUSE_EXACT:
        return True
    return any(pat.match(t) for pat in _FP_COMPILED)


def _clean(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.strip().lower())


# ── Pass 1: Filled pause lexical tagging ──────────────────────────────────────
def _tag_filled_pauses(words: List[Word]) -> List[TaggedWord]:
    tagged = []
    for w in words:
        tw = TaggedWord(
            text=w.text,
            start=w.start,
            end=w.end,
            confidence=w.confidence,
        )
        if _is_filled_pause(w.text):
            tw.label = Label.FILLED_PAUSE
        tagged.append(tw)
    return tagged


# ── Pass 2: Repetition detection ──────────────────────────────────────────────
def _tag_repetitions(tagged: List[TaggedWord],
                     window: int = 4) -> List[TaggedWord]:
    """
    Mark a word as REPETITION if the same clean token appears
    in the previous `window` fluent/filled-pause words.
    Only marks the duplicate, not the original.
    """
    for i, tw in enumerate(tagged):
        if tw.label != Label.FLUENT:
            continue
        current = _clean(tw.text)
        if not current:
            continue
        lookback = tagged[max(0, i - window): i]
        prev_tokens = [_clean(x.text) for x in lookback
                       if x.label in (Label.FLUENT, Label.REPETITION)]
        if current in prev_tokens:
            tw.label = Label.REPETITION
    return tagged


# ── Pass 3: False start detection ─────────────────────────────────────────────
# Short function words that are naturally brief — never flag these as false starts
_SHORT_FUNCTION_WORDS = {
    "a", "an", "the", "to", "of", "in", "on", "at", "by", "for",
    "is", "it", "i", "we", "he", "she", "be", "as", "or", "so",
    "do", "go", "my", "no", "up", "am", "us", "an", "if",
}

def _tag_false_starts(tagged: List[TaggedWord],
                      max_gap: float = 0.2) -> List[TaggedWord]:
    """
    Heuristic: a word is a false start if:
    - It explicitly ends with a hyphen/dash (truncated word), OR
    - It is very short duration (<0.09s), is NOT a common function word,
      AND is followed quickly by a different word within max_gap seconds.

    Deliberately conservative — better to under-detect than over-detect.
    """
    for i, tw in enumerate(tagged):
        if tw.label != Label.FLUENT:
            continue

        # Explicit truncation marker
        if tw.text.endswith("-") or tw.text.endswith("–"):
            tw.label = Label.FALSE_START
            continue

        # Skip common short function words — they are naturally brief
        clean = _clean(tw.text)
        if clean in _SHORT_FUNCTION_WORDS:
            continue

        duration = tw.end - tw.start

        # Only flag if very short AND not a known short word AND immediately
        # followed by a different word (gap < max_gap)
        if duration < 0.09 and i + 1 < len(tagged):
            next_tw = tagged[i + 1]
            gap = next_tw.start - tw.end
            next_clean = _clean(next_tw.text)
            if gap < max_gap and next_clean != clean:
                tw.label = Label.FALSE_START

    return tagged


# ── Pass 4: Long pause insertion ──────────────────────────────────────────────
def _insert_long_pauses(tagged: List[TaggedWord],
                        threshold: float = 0.8) -> List[TaggedWord]:
    """
    Insert synthetic LONG_PAUSE tokens between words where the
    silence gap exceeds `threshold` seconds.
    """
    result = []
    for i, tw in enumerate(tagged):
        if i > 0:
            gap = tw.start - tagged[i - 1].end
            tw.pause_before = gap
            if gap >= threshold:
                pause_token = TaggedWord(
                    text=f"[PAUSE {gap:.1f}s]",
                    start=tagged[i - 1].end,
                    end=tw.start,
                    confidence=1.0,
                    label=Label.LONG_PAUSE,
                    pause_before=gap,
                    is_synthetic=True,
                )
                result.append(pause_token)
        result.append(tw)
    return result


# ── Master classify function ───────────────────────────────────────────────────
def classify(words: List[Word],
             pause_threshold: float = 0.8) -> List[TaggedWord]:
    """
    Run the full disfluency classification pipeline on a list of Words.

    Args:
        words:           Flat list of Word objects from transcribe.py
        pause_threshold: Silence gap (seconds) to count as a long pause

    Returns:
        List of TaggedWord objects with .label set.
    """
    print(f"\n[Classifier] Tagging {len(words)} words...")

    tagged = _tag_filled_pauses(words)
    tagged = _tag_repetitions(tagged)
    tagged = _tag_false_starts(tagged)
    tagged = _insert_long_pauses(tagged, threshold=pause_threshold)

    # Summary
    counts = {}
    for tw in tagged:
        counts[tw.label] = counts.get(tw.label, 0) + 1

    print("  Label breakdown:")
    for label, n in sorted(counts.items()):
        print(f"    {label:<20} {n:>4}")

    return tagged


if __name__ == "__main__":
    import json, sys

    path = sys.argv[1] if len(sys.argv) > 1 else "demo/words.json"
    with open(path) as f:
        raw = json.load(f)

    words = [Word(**w) for w in raw]
    tagged = classify(words)

    out = [vars(tw) for tw in tagged]
    with open("demo/tagged.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nFirst 30 tagged words:")
    for tw in tagged[:30]:
        marker = "" if tw.label == Label.FLUENT else f"  ← {tw.label}"
        print(f"  [{tw.start:.2f}→{tw.end:.2f}] {tw.text!r}{marker}")

    print("\nSaved to demo/tagged.json")
