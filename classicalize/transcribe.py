"""Note-for-note transcription of every part of the clip.

* pitched parts (vocals, bass, guitars/synths/keys) -> Spotify's open-source
  **Basic Pitch** model (polyphonic, Apache-2.0), run through ONNX Runtime
* drums -> onset detection split into kick / snare / cymbal bands

These notes are what lets the "original" end of the slider play *exactly* the
original song on classical instruments.
"""
from __future__ import annotations

import os
import sys
import tempfile
import types
from dataclasses import dataclass

import numpy as np

SR = 22050


@dataclass
class TNote:
    start: float      # seconds
    end: float
    pitch: int
    amp: float        # 0..1
    role: str         # vocals | bass | other


def _stub(name):
    """Basic Pitch imports two packages it only needs for sonification; stub them
    if absent so we don't pull in numba/mir_eval."""
    if name not in sys.modules:
        try:
            __import__(name)
        except Exception:  # noqa: BLE001
            m = types.ModuleType(name)
            if name == "mir_eval":
                m.sonify = types.SimpleNamespace()
            sys.modules[name] = m


def _prepare_basic_pitch():
    """must run before *any* basic_pitch import (its note_creation module imports
    mir_eval and resampy at the top, which we install without)."""
    _stub("resampy")
    _stub("mir_eval")


_prepare_basic_pitch()
_MODEL = {}


def _quiet_import():
    import logging
    _prepare_basic_pitch()
    logging.disable(logging.WARNING)       # basic_pitch warns about every backend it doesn't find
    try:
        import basic_pitch  # noqa: F401
        import basic_pitch.inference  # noqa: F401
    finally:
        logging.disable(logging.NOTSET)


def basic_pitch_available():
    try:
        _stub("resampy")
        _stub("mir_eval")
        _quiet_import()
        import onnxruntime  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _model():
    if "m" not in _MODEL:
        _stub("resampy")
        _stub("mir_eval")
        _quiet_import()
        import pathlib

        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import Model
        p = pathlib.Path(ICASSP_2022_MODEL_PATH)
        onnx = p.parent / "nmp.onnx" if p.suffix != ".onnx" else p
        _MODEL["m"] = Model(onnx if onnx.exists() else p)
    return _MODEL["m"]


def warm_up():
    if basic_pitch_available():
        try:
            _model()
        except Exception:  # noqa: BLE001
            pass


def basic_pitch_notes(y, role, fmin=None, fmax=None, onset=0.5, frame=0.3, min_ms=90, adaptive=True):
    """Polyphonic transcription of a mono 22.05 kHz signal -> [TNote].
    adaptive: in dense mixes Basic Pitch's activations for upper parts come out weak;
    each octave band is rescaled to its own loudness so quieter parts are kept."""
    import contextlib
    import io

    import soundfile as sf
    _prepare_basic_pitch()
    _quiet_import()
    from basic_pitch import note_creation as infer
    from basic_pitch.constants import AUDIO_SAMPLE_RATE, FFT_HOP
    from basic_pitch.inference import run_inference
    if np.abs(y).max() < 1e-4:
        return []
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.wav")
        sf.write(p, (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), SR)
        with contextlib.redirect_stdout(io.StringIO()):
            out = run_inference(p, _model())
    if adaptive:
        note, ons = out["note"].copy(), out["onset"].copy()
        top = float(np.percentile(note.max(axis=1), 98))
        for o in range(0, note.shape[1], 12):
            band = note[:, o:o + 12]
            ref = float(np.percentile(band.max(axis=1), 98))
            # only lift bands that hold a real (quieter) part; empty bands contain just
            # leakage/harmonics of the loud part and must not be amplified into ghost notes
            if 0.2 < ref < 0.7 and ref > 0.45 * top:
                g = min(2.0, 0.7 / ref)
                note[:, o:o + 12] = np.clip(band * g, 0, 1)
                ons[:, o:o + 12] = np.clip(ons[:, o:o + 12] * g, 0, 1)
        out = dict(out, note=note, onset=ons)
    min_len = int(np.round(min_ms / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP)))
    _, events = infer.model_output_to_notes(out, onset_thresh=onset, frame_thresh=frame, min_note_len=min_len,
                                            min_freq=fmin, max_freq=fmax, include_pitch_bends=False,
                                            melodia_trick=True)
    notes = [TNote(float(s_), float(e), int(p_), float(a), role) for s_, e, p_, a, *_ in events]
    return clean(notes)


def clean(notes, max_poly=6):
    """remove octave/12th 'ghost' harmonics and cap polyphony (keeps the loudest)."""
    notes = sorted(notes, key=lambda n: (n.start, -n.amp))
    keep = []
    for n in notes:
        ghost = False
        for m in keep[-24:]:
            overlap = min(n.end, m.end) - max(n.start, m.start)
            if overlap > 0.5 * (n.end - n.start) and n.pitch - m.pitch in (12, 19, 24) and n.amp < 0.85 * m.amp:
                ghost = True
                break
        if not ghost:
            keep.append(n)
    out = []
    for n in keep:
        sounding = [m for m in out if m.start <= n.start < m.end]
        if len(sounding) >= max_poly:
            weakest = min(sounding, key=lambda m: m.amp)
            if weakest.amp < n.amp:
                out.remove(weakest)
            else:
                continue
        out.append(n)
    return out


def melody_from_yin(melody_notes, beat_times):
    """convert the analysis' (beat-based) vocal notes back to seconds."""
    bt = np.asarray(beat_times)
    idx = np.arange(len(bt), dtype=float)
    per = float(np.median(np.diff(bt))) if len(bt) > 1 else 0.5

    def b2s(b):
        if b <= 0:
            return bt[0] + b * per
        if b >= len(bt) - 1:
            return bt[-1] + (b - len(bt) + 1) * per
        return float(np.interp(b, idx, bt))
    return [TNote(b2s(n.start), b2s(n.start + n.dur), n.pitch, n.vel / 110, "vocals") for n in melody_notes]


def drum_hits(y_drums):
    """-> list of (time, kind, strength) with kind in kick/snare/cymbal."""
    import librosa
    if y_drums is None or np.abs(y_drums).max() < 1e-4:
        return []
    S = np.abs(librosa.stft(y_drums, n_fft=1024, hop_length=256))
    f = librosa.fft_frequencies(sr=SR, n_fft=1024)
    bands = {"kick": (f < 150), "snare": (f >= 180) & (f < 4000), "cymbal": f >= 6000}
    hits = []
    for kind, m in bands.items():
        env = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S[m] + 1e-9), sr=SR, hop_length=256)
        env = env / (env.max() + 1e-9)
        delta = {"kick": 0.25, "snare": 0.3, "cymbal": 0.45}[kind]
        on = librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=256, delta=delta, wait=6,
                                        units="frames")
        for fr in on:
            hits.append((float(librosa.frames_to_time(fr, sr=SR, hop_length=256)), kind, float(env[fr])))
    # a snare onset that coincides with a kick and is weaker is usually kick spill
    hits.sort()
    out = []
    for t, k, s in hits:
        if k == "snare" and any(abs(t - t2) < 0.03 and k2 == "kick" and s2 > s for t2, k2, s2 in hits):
            continue
        out.append((t, k, s))
    return out


def transcribe(stems: dict, song, separated: bool):
    """All the notes of the clip, by role."""
    notes = []
    if separated:
        notes += melody_from_yin(song.melody, song.beat_times)
        notes += basic_pitch_notes(stems["bass"], "bass", fmin=30, fmax=400, onset=0.45, adaptive=False)
        # keep the bass line monophonic (lowest note wins)
        notes = _mono_bass(notes)
        notes += basic_pitch_notes(stems["other"], "other", fmin=80)
    else:
        # no separation available: transcribe the whole (harmonic) mix and split by register
        allm = basic_pitch_notes(stems["other"] + stems["vocals"] + stems["bass"], "other", fmin=30)
        notes += _split_mix(allm)
    hits = drum_hits(stems.get("drums"))
    return notes, hits


def _mono_bass(notes):
    bass = sorted([n for n in notes if n.role == "bass"], key=lambda n: n.start)
    rest = [n for n in notes if n.role != "bass"]
    out = []
    for n in bass:
        if out and n.start < out[-1].end - 0.02:
            if n.pitch < out[-1].pitch:
                out[-1].end = n.start
                out.append(n)
            continue
        out.append(n)
    return rest + [n for n in out if n.end - n.start > 0.06]


def _split_mix(notes):
    """lowest sounding note below E3 -> bass; highest line above C4 -> melody; rest -> other."""
    notes = sorted(notes, key=lambda n: n.start)
    for n in notes:
        sounding = [m for m in notes if m.start <= n.start + 0.02 and m.end > n.start + 0.02]
        if n.pitch <= 52 and n.pitch == min(m.pitch for m in sounding):
            n.role = "bass"
        elif n.pitch >= 60 and n.pitch == max(m.pitch for m in sounding) and n.amp > 0.35:
            n.role = "vocals"
    return notes
