"""Render an Arrangement to audio with sampled classical instruments.

Fast path: ONE FluidSynth instance (SoundFont loaded once and kept warm
between requests), every part on its own MIDI channel with stage panning,
then a synthetic concert-hall reverb, then MP3 straight from memory via ffmpeg.
A 3–4 minute orchestral song renders in a few seconds.
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys

import numpy as np
import pretty_midi
from scipy.signal import fftconvolve

from .arrange import Arrangement

SR = 44100
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PAN = {"violin": -0.5, "violins": -0.55, "violins2": -0.2, "viola": 0.2, "violas": 0.2, "cello": 0.45,
       "cellos": 0.45, "contrabass": 0.6, "flute": -0.15, "oboe": 0.1, "clarinet": 0.15, "bassoon": 0.3,
       "horn": -0.35, "trumpet": 0.35, "timpani": 0.0, "harp": -0.6, "piano_rh": -0.05, "piano_lh": 0.05,
       "harpsichord_rh": -0.05, "harpsichord_lh": 0.05, "organ": 0.0, "pizzicato": -0.25, "perc": 0.1}
SUSTAINED = {"violin", "violins", "violins2", "viola", "violas", "cello", "cellos", "contrabass", "flute",
             "oboe", "clarinet", "bassoon", "horn", "trumpet"}

if sys.platform == "darwin":   # let ctypes find libfluidsynth (conda env first, then Homebrew)
    os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(
        [os.path.join(sys.prefix, "lib"), "/opt/homebrew/lib", "/usr/local/lib", "/usr/lib"]
        + [p for p in os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "").split(":") if p])


def find_soundfont():
    env = os.environ.get("CLASSICALIZE_SOUNDFONT")
    if env and os.path.exists(env):
        return env
    local = sorted(glob.glob(os.path.join(ROOT, "soundfonts", "*.sf[23]")),
                   key=lambda p: ("musescore" not in p.lower(), -os.path.getsize(p)))
    if local:
        return local[0]
    for p in ["/usr/share/sounds/sf2/FluidR3_GM.sf2", "/usr/share/sounds/sf3/MuseScore_General.sf3",
              "/opt/homebrew/share/soundfonts/default.sf2", "/usr/share/sounds/sf2/default-GM.sf2",
              "/usr/share/sounds/sf2/TimGM6mb.sf2"]:
        if os.path.exists(p):
            return p
    return os.path.join(os.path.dirname(pretty_midi.__file__), "TimGM6mb.sf2")


# ------------------------------------------------------------------ MIDI
def to_midi(arr: Arrangement, humanize=True, seed=0) -> pretty_midi.PrettyMIDI:
    rng = np.random.default_rng(seed)
    pm = pretty_midi.PrettyMIDI(initial_tempo=60.0 / max(1e-3, arr.beat_sec[1] - arr.beat_sec[0]))
    for p in arr.parts:
        inst = pretty_midi.Instrument(program=p.program, name=f"{p.role}:{p.name}", is_drum=(p.name == "perc"))
        for st, d, q, v in p.notes:
            t0 = arr.beat_to_sec(st)
            t1 = arr.beat_to_sec(st + d)
            if humanize:
                t0 = max(0.0, t0 + rng.normal(0, 0.006))
                v = int(np.clip(v + rng.normal(0, 3), 20, 124))
            if t1 - t0 < 0.03:
                t1 = t0 + 0.03
            inst.notes.append(pretty_midi.Note(velocity=int(np.clip(v, 1, 127)), pitch=int(q), start=t0, end=t1))
            if p.name in SUSTAINED and d >= 1.5:     # bowed / breathed swell on long notes
                mid = arr.beat_to_sec(st + d * 0.45)
                inst.control_changes += [pretty_midi.ControlChange(11, 84, t0),
                                         pretty_midi.ControlChange(11, 112, mid),
                                         pretty_midi.ControlChange(11, 96, max(t0, t1 - 0.05))]
        for b, c, val in p.cc:
            inst.control_changes.append(pretty_midi.ControlChange(int(c), int(val), arr.beat_to_sec(b)))
        if p.name not in SUSTAINED:
            inst.control_changes.insert(0, pretty_midi.ControlChange(11, 110, 0.0))
        inst.control_changes.append(pretty_midi.ControlChange(10, int(64 + 63 * PAN.get(p.name, 0)), 0.0))
        inst.control_changes.sort(key=lambda c: c.time)
        inst.notes.sort(key=lambda n: n.start)
        pm.instruments.append(inst)
    return pm


# ------------------------------------------------------------------ synth (kept warm)
_SYNTH = {}


def _synth(sf2):
    if sf2 not in _SYNTH:
        import fluidsynth
        fs = fluidsynth.Synth(gain=0.35, samplerate=float(SR), **{"synth.polyphony": 512,
                                                                     "synth.reverb.active": 0,
                                                                     "synth.chorus.active": 0,
                                                                     "synth.cpu-cores": max(1, min(4, os.cpu_count() or 1))})
        sfid = fs.sfload(sf2)
        _SYNTH[sf2] = (fs, sfid)
    return _SYNTH[sf2]


def synthesize(pm: pretty_midi.PrettyMIDI, sf2: str) -> np.ndarray:
    """Single-pass render of all instruments -> float32 stereo (n, 2)."""
    fs, sfid = _synth(sf2)
    chans = [c for c in range(16) if c != 9]
    events = []
    k = 0
    for inst in pm.instruments:
        if inst.is_drum:
            ch = 9
            # orchestral percussion kit (GM bank 128, program 48); fall back to the standard kit
            if fs.program_select(9, sfid, 128, 48) != 0:
                fs.program_select(9, sfid, 128, 0)
        else:
            if k >= len(chans):
                continue
            ch = chans[k]
            k += 1
            fs.program_select(ch, sfid, 0, inst.program)
        fs.cc(ch, 7, 100)
        fs.cc(ch, 64, 0)
        for cc in inst.control_changes:
            events.append((cc.time, 1, "cc", ch, cc.number, cc.value))
        for n in inst.notes:
            events.append((n.start, 2, "on", ch, n.pitch, n.velocity))
            events.append((n.end, 0, "off", ch, n.pitch, 0))
    events.sort(key=lambda e: (e[0], e[1]))
    end = (events[-1][0] if events else 0) + 2.5
    out = np.zeros((int(end * SR) + SR, 2), np.float32)
    pos = 0
    for t, _, kind, ch, a, b in events:
        target = int(t * SR)
        if target > pos:
            buf = fs.get_samples(target - pos)
            out[pos:target] = np.asarray(buf, np.float32).reshape(-1, 2) / 32768.0
            pos = target
        if kind == "on":
            fs.noteon(ch, a, b)
        elif kind == "off":
            fs.noteoff(ch, a)
        else:
            fs.cc(ch, a, b)
    tail = len(out) - pos
    if tail > 0:
        out[pos:] = np.asarray(fs.get_samples(tail), np.float32).reshape(-1, 2) / 32768.0
    for ch in chans:
        fs.all_notes_off(ch)
    fs.get_samples(SR // 2)   # flush release tails so the next song starts clean
    return out


# ------------------------------------------------------------------ hall reverb
def hall_ir(rt60=1.8, sr=SR, seed=1):
    rng = np.random.default_rng(seed)
    n = int(sr * rt60 * 1.2)
    t = np.arange(n) / sr
    env = np.exp(-6.91 * t / rt60)
    ir = np.zeros((n, 2))
    from scipy.signal import lfilter
    for ch in range(2):
        noise = rng.normal(0, 1, n)
        # darker tail: low-pass the second half more strongly
        a1 = lfilter([0.65], [1, -0.35], noise)
        a2 = lfilter([0.15], [1, -0.85], noise)
        mix = np.linspace(0, 1, n)
        ir[:, ch] = ((1 - mix) * a1 + mix * a2) * env
        for d, g in ((0.011, 0.6), (0.019, 0.45), (0.027, 0.35), (0.041, 0.3), (0.053, 0.22)):
            ir[int((d + 0.003 * ch) * sr), ch] += g
    ir /= np.sqrt((ir ** 2).sum(axis=0, keepdims=True))
    return ir.astype(np.float32)


_IR = {}


def _master(mix, era, ensemble, reverb=None, trim=True):
    if reverb is None:
        reverb = 0.22 if ensemble in ("piano", "string_quartet") else 0.3
    rt = {"baroque": 1.9, "classical": 1.5, "romantic": 2.2}.get(era, 1.8) * (1.15 if ensemble == "orchestra" else 1.0)
    key = round(rt, 2)
    if key not in _IR:
        _IR[key] = hall_ir(rt)
    ir = _IR[key]
    n = len(mix)
    wet = np.stack([fftconvolve(mix[:, c], ir[:, c])[:n] for c in range(2)], axis=1)
    mix = (1 - reverb) * mix + reverb * 1.6 * wet
    rms = np.sqrt(np.mean(mix ** 2) + 1e-12)
    mix *= 0.12 / rms
    peak = np.abs(mix).max()
    if peak > 0.97:
        mix = np.tanh(mix / peak * 1.2) / np.tanh(1.2) * 0.97
    if trim:
        env = np.abs(mix).max(axis=1)
        last = np.where(env > 1e-3)[0]
        if len(last):
            mix = mix[:min(len(mix), last[-1] + int(0.3 * SR))]
    return mix.astype(np.float32)


def write_mp3(mix, path, bitrate="192k"):
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-",
                        "-b:a", bitrate, path], input=np.ascontiguousarray(mix).tobytes(), check=True)
        return path
    import soundfile as sf
    wav = os.path.splitext(path)[0] + ".wav"
    sf.write(wav, mix, SR, subtype="PCM_16")
    return wav


def render(arr: Arrangement, out_path: str, sf2: str | None = None, reverb: float | None = None,
           write_midi: bool = True, duration: float | None = None) -> str:
    """Arrangement -> MP3 (or WAV if out_path ends with .wav). Returns the audio path."""
    sf2 = sf2 or find_soundfont()
    pm = to_midi(arr, seed=arr.params.get("seed", 0))
    base = os.path.splitext(out_path)[0]
    if write_midi:
        pm.write(base + ".mid")
    mix = synthesize(pm, sf2)
    mix = _master(mix, arr.params.get("era", "classical"), arr.params.get("ensemble", "string_quartet"), reverb,
                  trim=duration is None)
    if duration is not None:          # exactly as long as the original clip
        n = int(round(duration * SR))
        if len(mix) < n:
            mix = np.concatenate([mix, np.zeros((n - len(mix), 2), np.float32)])
        mix = mix[:n]
        f = min(n, int(0.35 * SR))
        mix[-f:] *= np.linspace(1, 0, f)[:, None] ** 2
    if out_path.endswith(".wav"):
        import soundfile as sf
        sf.write(out_path, mix, SR, subtype="PCM_16")
        return out_path
    return write_mp3(mix, out_path)
