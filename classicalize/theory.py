"""Music-theory primitives shared by analysis, arranging and judging.

The same chord vocabulary + template matcher is used for
  * detecting chords in pop audio (from chroma), and
  * labelling chords in the classical training corpus (from scores),
so the era harmony models learned from classical scores speak the same
"language" as the analysed pop songs.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import numpy as np

PC_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

# chord quality -> intervals above root
QUALITIES = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "sus4": (0, 5, 7),
    "dom7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
    "hdim7": (0, 3, 6, 10),
    "dim7": (0, 3, 6, 9),
}
# qualities a pop audio detector is allowed to emit (more = less robust)
AUDIO_QUALITIES = ["maj", "min", "dim", "sus4", "dom7", "min7", "maj7"]
TRIAD_OF = {"maj": "maj", "min": "min", "dim": "dim", "aug": "aug", "sus4": "maj",
            "dom7": "maj", "maj7": "maj", "min7": "min", "hdim7": "dim", "dim7": "dim"}

MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10)          # natural; harmonic adds 11
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

ROMAN = ["I", "bII", "II", "bIII", "III", "IV", "#IV", "V", "bVI", "VI", "bVII", "VII"]


@dataclass(frozen=True)
class Chord:
    root: int          # pitch class 0..11
    qual: str          # key of QUALITIES

    @property
    def pcs(self):
        return tuple((self.root + i) % 12 for i in QUALITIES[self.qual])

    def name(self):
        suffix = {"maj": "", "min": "m", "dim": "dim", "aug": "+", "sus4": "sus4", "dom7": "7",
                  "maj7": "maj7", "min7": "m7", "hdim7": "m7b5", "dim7": "dim7"}[self.qual]
        return PC_NAMES[self.root] + suffix

    def token(self, tonic: int) -> str:
        """Key-relative token, e.g. '7:dom7' == V7 in any key."""
        return f"{(self.root - tonic) % 12}:{self.qual}"

    @staticmethod
    def from_token(tok: str, tonic: int) -> "Chord":
        iv, q = tok.split(":")
        return Chord((int(iv) + tonic) % 12, q)


def roman(tok: str) -> str:
    iv, q = tok.split(":")
    r = ROMAN[int(iv)]
    if q in ("min", "min7", "dim", "hdim7", "dim7"):
        r = r.lower()
    return r + {"maj": "", "min": "", "dim": "°", "aug": "+", "sus4": "sus4", "dom7": "7",
                "maj7": "M7", "min7": "7", "hdim7": "ø7", "dim7": "°7"}[q]


# ----------------------------------------------------------------- templates
def _templates(qualities):
    names, mats = [], []
    for q in qualities:
        for r in range(12):
            v = np.zeros(12)
            ivs = QUALITIES[q]
            for k, i in enumerate(ivs):
                v[(r + i) % 12] = 1.0 if k < 3 else 0.7   # 7ths count a bit less
            v[r] += 0.25                                    # root emphasis
            names.append(Chord(r, q))
            mats.append(v / np.linalg.norm(v))
    return names, np.array(mats)


_TPL_ALL = _templates(list(QUALITIES))
_TPL_AUDIO = _templates(AUDIO_QUALITIES)


def match_chord(pc_weights, bass_pc=None, audio=False, return_scores=False):
    """Template-match a 12-d pitch-class weight vector to a chord."""
    names, mats = _TPL_AUDIO if audio else _TPL_ALL
    v = np.asarray(pc_weights, float)
    if v.sum() <= 1e-9:
        return (None, None) if return_scores else None
    v = v / (np.linalg.norm(v) + 1e-9)
    s = mats @ v
    if bass_pc is not None:
        # a chord whose root is in the bass gets a small boost
        s = s + np.array([0.06 if c.root == bass_pc else 0.0 for c in names])
    # mild prior towards simple triads (Occam)
    s = s + np.array([0.03 if c.qual in ("maj", "min") else 0.0 for c in names])
    i = int(np.argmax(s))
    if return_scores:
        return names, s
    return names[i]


def estimate_key(pc_hist):
    """Krumhansl–Schmuckler key estimate -> (tonic_pc, 'major'|'minor', confidence)."""
    h = np.asarray(pc_hist, float)
    if h.sum() <= 0:
        return 0, "major", 0.0
    best = (-2, 0, "major")
    scores = []
    for t in range(12):
        for mode, prof in (("major", KK_MAJOR), ("minor", KK_MINOR)):
            r = np.corrcoef(np.roll(prof, t), h)[0, 1]
            scores.append(r)
            if r > best[0]:
                best = (r, t, mode)
    scores = sorted(scores, reverse=True)
    conf = float(scores[0] - scores[1])
    return best[1], best[2], conf


def scale_pcs(tonic, mode):
    sc = MAJOR_SCALE if mode == "major" else MINOR_SCALE
    s = {(tonic + i) % 12 for i in sc}
    if mode == "minor":
        s.add((tonic + 11) % 12)  # raised leading tone
    return s


def diatonic_chords(tonic, mode):
    """Era-neutral diatonic chord set (+ the few chromatic chords every era uses)."""
    if mode == "major":
        toks = ["0:maj", "2:min", "4:min", "5:maj", "7:maj", "9:min", "11:dim",
                "7:dom7", "2:min7", "11:hdim7", "5:maj7", "0:maj7", "9:min7"]
    else:
        toks = ["0:min", "2:dim", "3:maj", "5:min", "7:maj", "7:dom7", "8:maj", "10:maj",
                "11:dim7", "2:hdim7", "5:min7", "0:min7"]
    return [Chord.from_token(t, tonic) for t in toks]


# ----------------------------------------------------------- era LM wrapper
_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


class EraHarmonyModel:
    """Bigram language model over key-relative chord tokens, learned from
    classical scores (see scripts/build_era_models.py)."""

    def __init__(self, data: dict):
        self.d = data
        self.vocab = data["vocab"]
        self.k = data.get("smoothing", 0.1)
        self._cache = {}

    def logp(self, era: str, mode: str, prev: str | None, cur: str) -> float:
        key = (era, mode, prev, cur)
        c = self._cache.get(key)
        if c is None:
            c = self._cache[key] = self._logp(era, mode, prev, cur)
        return c

    def _logp(self, era, mode, prev, cur):
        m = self.d["eras"][era][mode]
        uni = m["unigram"]
        V = len(self.vocab)
        if prev is None or prev not in m["bigram"]:
            return math.log((uni.get(cur, 0) + self.k) / (m["unigram_total"] + self.k * V))
        row = m["bigram"][prev]
        tot = m["bigram_total"][prev]
        # interpolate bigram with unigram (Jelinek–Mercer)
        pb = (row.get(cur, 0) + self.k) / (tot + self.k * V)
        pu = (uni.get(cur, 0) + self.k) / (m["unigram_total"] + self.k * V)
        return math.log(0.75 * pb + 0.25 * pu)

    def seq_logp(self, era, mode, toks):
        lp, prev = 0.0, None
        for t in toks:
            lp += self.logp(era, mode, prev, t)
            prev = t
        return lp / max(1, len(toks))

    def stat(self, era, name, default=None):
        return self.d["eras"][era].get("stats", {}).get(name, default)

    def eras(self):
        return list(self.d["eras"])


_ERA_MODEL = None
_POP_MODEL = None


def pop_model():
    """bigram model of pop chord progressions (built by train.py from the
    training songs); None until trained."""
    global _POP_MODEL
    if _POP_MODEL is None:
        p = os.path.join(_MODEL_DIR, "pop_harmony.json")
        if os.path.exists(p):
            with open(p) as f:
                _POP_MODEL = EraHarmonyModel(json.load(f))
    return _POP_MODEL


def build_bigram(seqs_by_mode, vocab, smoothing=0.05):
    """seqs_by_mode: {"major": [[tok..]..], "minor": [...]} -> model dict (era 'pop')."""
    from collections import Counter, defaultdict
    em = {}
    for mode in ("major", "minor"):
        uni, bi = Counter(), defaultdict(Counter)
        for t in seqs_by_mode.get(mode, []):
            uni.update(t)
            for a, b in zip(t, t[1:]):
                bi[a][b] += 1
        em[mode] = {"unigram": dict(uni), "unigram_total": sum(uni.values()),
                    "bigram": {a: dict(c) for a, c in bi.items()},
                    "bigram_total": {a: sum(c.values()) for a, c in bi.items()}}
    return {"vocab": vocab, "smoothing": smoothing, "eras": {"pop": em}}


def harmony_llr(toks, era, mode):
    """mean per-chord log-likelihood ratio  log P_era / P_pop  (None if no pop model)."""
    pm = pop_model()
    if pm is None or len(toks) < 2:
        return None
    M = era_model()
    return M.seq_logp(era, mode, toks) - pm.seq_logp("pop", mode, toks)


def era_model() -> EraHarmonyModel:
    global _ERA_MODEL
    if _ERA_MODEL is None:
        with open(os.path.join(_MODEL_DIR, "era_models.json")) as f:
            _ERA_MODEL = EraHarmonyModel(json.load(f))
    return _ERA_MODEL
