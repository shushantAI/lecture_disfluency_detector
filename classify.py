"""
classify.py

This file handles tagging each word from the transcript with one of
five labels: FLUENT, FILLED_PAUSE, FALSE_START, REPETITION, or LONG_PAUSE.
I used a multi-pass approach - first match filler words by their text,
then look for repeated words, then catch false starts, and finally insert
long pause tokens wherever there's a big silence gap between words.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional
from transcribe import Word


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
    pause_before: float = 0.0
    is_synthetic: bool = False


FILLED_PAUSE_PATTERNS = [
    r"^u+m+$",
    r"^u+h+$",
    r"^e+r+$",
    r"^h+m+$",
    r"^mhm+$",
    r"^ah+$",
    r"^oh+$",
    r"^like$",
    r"^basically$",
    r"^literally$",
    r"^actually$",
    r"^right\??$",
    r"^so+$",
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


def _tag_repetitions(tagged: List[TaggedWord],
                     window: int = 4) -> List[TaggedWord]:
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


_SHORT_FUNCTION_WORDS = {
    "a", "an", "the", "to", "of", "in", "on", "at", "by", "for",
    "is", "it", "i", "we", "he", "she", "be", "as", "or", "so",
    "do", "go", "my", "no", "up", "am", "us", "an", "if",
}

def _tag_false_starts(tagged: List[TaggedWord],
                      max_gap: float = 0.2) -> List[TaggedWord]:
    for i, tw in enumerate(tagged):
        if tw.label != Label.FLUENT:
            continue

        if tw.text.endswith("-") or tw.text.endswith("–"):
            tw.label = Label.FALSE_START
            continue

        clean = _clean(tw.text)
        if clean in _SHORT_FUNCTION_WORDS:
            continue

        duration = tw.end - tw.start

        if duration < 0.09 and i + 1 < len(tagged):
            next_tw = tagged[i + 1]
            gap = next_tw.start - tw.end
            next_clean = _clean(next_tw.text)
            if gap < max_gap and next_clean != clean:
                tw.label = Label.FALSE_START

    return tagged


def _insert_long_pauses(tagged: List[TaggedWord],
                        threshold: float = 0.8) -> List[TaggedWord]:
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


def classify(words: List[Word],
             pause_threshold: float = 0.8) -> List[TaggedWord]:
    print(f"\n[Classifier] Tagging {len(words)} words...")

    tagged = _tag_filled_pauses(words)
    tagged = _tag_repetitions(tagged)
    tagged = _tag_false_starts(tagged)
    tagged = _insert_long_pauses(tagged, threshold=pause_threshold)

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
