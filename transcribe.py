"""
transcribe.py
-------------
Verbatim transcription using CrisperWhisper.
Preserves every filler word (um, uh, like, you know) with precise
word-level timestamps.

Falls back to faster-whisper with an initial_prompt trick if
CrisperWhisper is not available (e.g. limited VRAM).
"""

import os
import re
import numpy as np
import soundfile as sf
import torch
from dataclasses import dataclass, field
from typing import List, Optional
from tqdm import tqdm


@dataclass
class Word:
    text: str
    start: float        # seconds
    end: float          # seconds
    confidence: float = 1.0
    speaker: Optional[str] = None


@dataclass
class Segment:
    text: str
    start: float
    end: float
    words: List[Word] = field(default_factory=list)


def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    """Load audio file and resample to target sample rate."""
    import librosa
    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    print(f"  Loaded: {path} | duration={audio.shape[0]/target_sr:.1f}s | sr={sr}Hz")
    return audio


def apply_vad(audio: np.ndarray, sr: int = 16000,
              threshold: float = 0.4,
              min_speech_duration: float = 0.25,
              min_silence_duration: float = 0.8) -> List[dict]:
    """
    Apply Silero VAD to detect speech segments.
    Returns list of {start, end} dicts (in seconds).
    """
    print("  Running VAD (Silero)...")
    try:
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False,
            verbose=False
        )
        get_speech_ts = utils[0]

        audio_tensor = torch.from_numpy(audio).float()
        speech_timestamps = get_speech_ts(
            audio_tensor, model,
            threshold=threshold,
            min_speech_duration_ms=int(min_speech_duration * 1000),
            min_silence_duration_ms=int(min_silence_duration * 1000),
            sampling_rate=sr,
            return_seconds=True
        )
        print(f"  VAD found {len(speech_timestamps)} speech segment(s)")
        return speech_timestamps

    except Exception as e:
        print(f"  VAD failed ({e}), treating entire audio as speech")
        duration = len(audio) / sr
        return [{"start": 0.0, "end": duration}]


def transcribe_crisperwhisper(audio: np.ndarray,
                               sr: int = 16000,
                               model_size: str = "nyrahealth/CrisperWhisper") -> List[Segment]:
    """
    Transcribe using CrisperWhisper for verbatim output.
    Keeps filler words (um, uh, like, you know) in transcript.
    """
    print(f"  Loading CrisperWhisper ({model_size})...")
    try:
        from transformers import pipeline as hf_pipeline
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Using device: {device}")

        asr = hf_pipeline(
            "automatic-speech-recognition",
            model=model_size,
            device=device,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            return_timestamps="word",
            chunk_length_s=30,
            stride_length_s=5,
        )

        print("  Transcribing (verbatim mode)...")
        result = asr(
            {"array": audio, "sampling_rate": sr},
            generate_kwargs={"language": "english"},
        )

        segments = _parse_hf_output(result)
        return segments

    except Exception as e:
        print(f"  CrisperWhisper failed: {e}")
        print("  Falling back to faster-whisper with filler prompt...")
        return transcribe_fallback(audio, sr)


def transcribe_fallback(audio: np.ndarray, sr: int = 16000) -> List[Segment]:
    """
    Fallback: faster-whisper with an initial_prompt that biases
    the model toward retaining filler words.
    """
    from faster_whisper import WhisperModel

    FILLER_PROMPT = (
        "Umm, let me think. Uh, so basically, um, you know, like, "
        "uh, the idea is, um, yeah. Uh, okay so."
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute = "float16" if device == "cuda" else "int8"

    print(f"  Loading faster-whisper (base) on {device}...")
    model = WhisperModel("base", device=device, compute_type=compute)

    # Write audio to a temp file (faster-whisper needs a file path)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, sr)
        tmp_path = f.name

    segments_raw, _ = model.transcribe(
        tmp_path,
        language="en",
        word_timestamps=True,
        initial_prompt=FILLER_PROMPT,
        vad_filter=False,  # VAD already applied
    )
    os.unlink(tmp_path)

    segments = []
    for seg in segments_raw:
        words = [
            Word(
                text=w.word.strip(),
                start=w.start,
                end=w.end,
                confidence=w.probability,
            )
            for w in (seg.words or [])
        ]
        segments.append(Segment(
            text=seg.text.strip(),
            start=seg.start,
            end=seg.end,
            words=words,
        ))
    return segments


def _parse_hf_output(result: dict) -> List[Segment]:
    """Parse HuggingFace pipeline output into Segment/Word objects."""
    chunks = result.get("chunks", [])
    if not chunks:
        # No word-level timestamps — wrap full text in one segment
        return [Segment(
            text=result.get("text", ""),
            start=0.0,
            end=0.0,
            words=[]
        )]

    words = []
    for chunk in chunks:
        ts = chunk.get("timestamp", (0, 0))
        words.append(Word(
            text=chunk["text"].strip(),
            start=ts[0] or 0.0,
            end=ts[1] or 0.0,
            confidence=1.0,
        ))

    # Group words into ~30s segments
    segments = []
    current_words = []
    current_start = words[0].start if words else 0.0
    SEGMENT_DURATION = 30.0

    for w in words:
        current_words.append(w)
        if w.end - current_start >= SEGMENT_DURATION:
            segments.append(Segment(
                text=" ".join(x.text for x in current_words),
                start=current_start,
                end=w.end,
                words=current_words,
            ))
            current_words = []
            current_start = w.end

    if current_words:
        segments.append(Segment(
            text=" ".join(x.text for x in current_words),
            start=current_start,
            end=current_words[-1].end,
            words=current_words,
        ))

    return segments


def transcribe(audio_path: str,
               use_vad: bool = True,
               model: str = "nyrahealth/CrisperWhisper") -> List[Word]:
    """
    Full transcription pipeline.
    Returns a flat list of Word objects covering the entire audio.
    """
    print("\n[1/2] Loading audio...")
    audio = load_audio(audio_path)
    sr = 16000

    if use_vad:
        print("\n[2/2] Transcribing with VAD-filtered segments...")
        speech_segs = apply_vad(audio, sr)
        all_words: List[Word] = []
        for seg in tqdm(speech_segs, desc="  Segments"):
            start_sample = int(seg["start"] * sr)
            end_sample = int(seg["end"] * sr)
            chunk = audio[start_sample:end_sample]
            if len(chunk) < sr * 0.1:  # skip very short chunks
                continue
            segs = transcribe_crisperwhisper(chunk, sr, model)
            for s in segs:
                for w in s.words:
                    # Shift timestamps back to global time
                    all_words.append(Word(
                        text=w.text,
                        start=w.start + seg["start"],
                        end=w.end + seg["start"],
                        confidence=w.confidence,
                    ))
        return all_words
    else:
        print("\n[2/2] Transcribing (no VAD)...")
        segs = transcribe_crisperwhisper(audio, sr, model)
        return [w for s in segs for w in s.words]


if __name__ == "__main__":
    import sys, json
    path = sys.argv[1] if len(sys.argv) > 1 else "demo/sample.wav"
    words = transcribe(path)
    print(f"\nTranscribed {len(words)} words:")
    for w in words[:20]:
        print(f"  [{w.start:.2f}→{w.end:.2f}] {w.text!r}")
    with open("demo/words.json", "w") as f:
        json.dump([vars(w) for w in words], f, indent=2)
    print("\nSaved to demo/words.json")
