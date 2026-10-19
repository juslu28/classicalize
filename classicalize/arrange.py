"""Step 4 – write a classical arrangement of the analysed pop song.

Everything is symbolic (notes on a beat grid) so it can be scored and
re-rendered quickly inside the optimisation loop.

Pipeline
  1. chord slots on a classical harmonic rhythm
  2. re-harmonisation: Viterbi over candidate chords, trading fidelity to the
     detected pop chords against the era harmony model learned from real scores
     (+ cadence bonuses at phrase ends)
  3. 4-part voice leading (DP over voicings, penalising parallel 5ths/8ves,
     crossings, bad spacing, doubled leading tones)
  4. melody: range fitting, de-syncopation ("straighten"), era ornaments
  5. era-specific accompaniment textures for the chosen ensemble
     Baroque  – walking/continuo bass, contrapuntal inner line, harpsichord
     Classical– Alberti bass / murmuring 8ths, clear cadences, echoes
     Romantic – wide arpeggios, singing cello counter-melody, rubato, swells
  6. dynamics & tempo map
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field

import numpy as np

from .analysis import Note, SongAnalysis
from .theory import QUALITIES, Chord, diatonic_chords, era_model, scale_pcs

ERAS = ("baroque", "classical", "romantic")
ENSEMBLES = ("piano", "string_quartet", "chamber", "orchestra")


@dataclass
class ArrangeParams:
    era: str = "classical"
    ensemble: str = "string_quartet"
    reharm: float = 0.5          # 0 = keep pop chords, 1 = follow era harmony model
    ornament: float = 0.4        # trills, turns, appoggiaturas, diminutions
    straighten: float = 0.5      # remove pop syncopation from the melody
    density: float = 0.5         # accompaniment rhythmic activity
    tempo_factor: float = 1.0
    harmonic_rhythm: int = 2     # beats per chord slot
    dynamics: float = 0.6        # dynamic range
    counterpoint: float = 0.5    # independence of inner voices
    melody_octave: int = 0       # extra octave shift of the melody
    add_ending: bool = True      # final classical cadence
    rubato: float = 0.0          # tempo flexibility at phrase ends (Chopin high, Bach none)
    develop: float = 0.0         # 0 = keep the tune's notes; 1 = keep only its skeleton, re-compose the rest
    exact_timing: bool = True    # play on the ORIGINAL beat times (output length == input length)
    composer: str = ""           # style preset this arrangement imitates (see composers.py)
    seed: int = 0

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------- instruments
# name: (GM program, low, high, comfortable centre)
INSTR = {
    "violin": (40, 55, 100, 76), "violins": (48, 55, 96, 76), "violins2": (49, 55, 88, 69),
    "viola": (41, 48, 84, 62), "violas": (49, 48, 81, 62),
    "cello": (42, 36, 76, 52), "cellos": (48, 36, 72, 50), "contrabass": (43, 28, 60, 40),
    "piano_rh": (0, 55, 96, 74), "piano_lh": (0, 28, 72, 48),
    "harpsichord_rh": (6, 53, 89, 72), "harpsichord_lh": (6, 29, 65, 48),
    "flute": (73, 60, 96, 79), "oboe": (68, 58, 91, 74), "clarinet": (71, 50, 91, 70),
    "bassoon": (70, 34, 72, 50), "horn": (60, 41, 77, 58), "trumpet": (56, 58, 84, 72),
    "timpani": (47, 40, 57, 45), "harp": (46, 24, 103, 60), "organ": (19, 36, 84, 60),
    "pizzicato": (45, 36, 96, 60), "perc": (0, 27, 87, 40),       # perc = orchestral percussion kit
}

ERA_ANACHRONISMS = {
    "baroque": {"clarinet", "piano_rh", "piano_lh", "harp", "contrabass_only"},
    "classical": {"harpsichord_rh", "harpsichord_lh", "harp"},
    "romantic": {"harpsichord_rh", "harpsichord_lh"},
}


def ensemble_layout(era: str, ens: str) -> dict:
    """role -> list of instrument names (first is primary)."""
    kb_rh, kb_lh = ("harpsichord_rh", "harpsichord_lh") if era == "baroque" else ("piano_rh", "piano_lh")
    if ens == "piano":
        return {"melody": [kb_rh], "keys_lh": [kb_lh]}
    if ens == "string_quartet":
        return {"melody": ["violin"], "inner_hi": ["violin"], "inner_lo": ["viola"],
                "bass": ["cello"]}
    if ens == "chamber":
        if era == "baroque":
            return {"melody": ["flute"], "inner_hi": ["violin"], "bass": ["cello"],
                    "continuo": ["harpsichord_rh", "harpsichord_lh"]}
        if era == "classical":
            return {"melody": ["flute"], "inner_hi": ["violin"], "inner_lo": ["viola"], "bass": ["cello"]}
        return {"melody": ["clarinet"], "inner_hi": ["violin"], "inner_lo": ["viola"], "bass": ["cello"],
                "keys_lh": ["piano_lh"]}
    # orchestra
    if era == "baroque":
        return {"melody": ["violins", "oboe"], "inner_hi": ["violins2"], "inner_lo": ["violas"],
                "bass": ["cellos", "bassoon"], "continuo": ["harpsichord_rh", "harpsichord_lh"],
                "brass": ["trumpet"], "timpani": ["timpani"]}
    if era == "classical":
        return {"melody": ["violins", "flute"], "inner_hi": ["violins2"], "inner_lo": ["violas"],
                "bass": ["cellos", "contrabass"], "winds": ["oboe", "bassoon"], "brass": ["horn"],
                "timpani": ["timpani"]}
    return {"melody": ["violins", "flute"], "inner_hi": ["violins2"], "inner_lo": ["violas"],
            "counter": ["cellos"], "bass": ["contrabass", "bassoon"], "winds": ["clarinet", "oboe"],
            "brass": ["horn"], "harp": ["harp"], "timpani": ["timpani"]}


@dataclass
class Part:
    name: str
    program: int
    role: str
    notes: list = field(default_factory=list)       # (start, dur, pitch, vel)
    cc: list = field(default_factory=list)          # (beat, ctrl, value)

    def add(self, start, dur, pitch, vel=70):
        lo, hi = INSTR[self.name][1], INSTR[self.name][2]
        while pitch < lo:
            pitch += 12
        while pitch > hi:
            pitch -= 12
        if dur > 1e-3:
            self.notes.append([float(start), float(dur), int(pitch), int(vel)])


@dataclass
class Arrangement:
    params: dict
    parts: list
    beat_sec: list                 # time (s) of each integer beat 0..N (incl. ending)
    slots: list                    # [{start, end, chord, token, voicing, phrase_end}]
    melody: list                   # final melody notes (start, dur, pitch)
    n_beats: int                   # length of the song part (without added ending)
    tonic: int
    mode: str
    beats_per_bar: int
    sections: list
    melody_plain: list = field(default_factory=list)   # melody before ornaments (rhythm judged on this)

    def beat_to_sec(self, b):
        bs = self.beat_sec
        if b <= 0:
            return b * (bs[1] - bs[0])
        i = int(b)
        if i >= len(bs) - 1:
            return bs[-1] + (b - (len(bs) - 1)) * (bs[-1] - bs[-2])
        return bs[i] + (b - i) * (bs[i + 1] - bs[i])

    def all_notes(self):
        for p in self.parts:
            for n in p.notes:
                yield p, n


# ================================================================ helpers
def _scale_step(p, direction, scale):
    q = p + direction
    while q % 12 not in scale:
        q += direction
    return q


def _nearest_pc(p, pc, below=False):
    """nearest pitch to p having pitch class pc (optionally <= p)."""
    c = p - ((p - pc) % 12)
    if below:
        return c
    return c if (p - c) <= 6 else c + 12


def _slot_bounds(n, downbeat, L):
    b = [0]
    if downbeat > 0:
        b.append(downbeat)
    x = downbeat + L
    while x < n:
        b.append(x)
        x += L
    b.append(n)
    return sorted(set(b))


def _metric_w(beat, bpb):
    f = beat % bpb
    if abs(f) < 1e-6:
        return 1.0
    if abs(f - bpb / 2) < 1e-6 and bpb == 4:
        return 0.7
    if abs(f - round(f)) < 1e-6:
        return 0.5
    return 0.25


# ================================================================ 1+2 harmony
def _candidates(tonic, mode, era, detected):
    cs = list(diatonic_chords(tonic, mode))
    roots = {c.root for c in cs if c.qual in ("maj", "min")}
    for r in roots:                                            # secondary dominants
        if r != tonic:
            cs.append(Chord((r + 7) % 12, "dom7"))
    if era != "baroque":
        if mode == "major":
            cs += [Chord((tonic + 5) % 12, "min"), Chord((tonic + 8) % 12, "maj"),   # iv, bVI
                   Chord((tonic + 6) % 12, "dim7")]                                  # vii°7/V
        cs.append(Chord((tonic + 1) % 12, "maj"))                                   # Neapolitan
    if detected is not None:
        cs.append(detected)
        if detected.qual in ("sus4", "maj7"):
            cs.append(Chord(detected.root, "maj"))
        if detected.qual == "min7":
            cs.append(Chord(detected.root, "min"))
    out, seen = [], set()
    for c in cs:
        if (c.root, c.qual) not in seen:
            seen.add((c.root, c.qual))
            out.append(c)
    return out


def _tpl(c: Chord):
    v = np.zeros(12)
    for k, i in enumerate(QUALITIES[c.qual]):
        v[(c.root + i) % 12] = 1.0 if k < 3 else 0.6
    return v / np.linalg.norm(v)


def harmonise(song: SongAnalysis, p: ArrangeParams):
    n = song.n_beats
    bpb = song.beats_per_bar
    L = p.harmonic_rhythm if bpb == 4 else (3 if p.harmonic_rhythm >= 2 else 1)
    bounds = _slot_bounds(n, song.downbeat % bpb if bpb else 0, L)
    chords = song.chord_objs()
    chroma = np.asarray(song.beat_chroma) if song.beat_chroma else np.zeros((n, 12))
    M = era_model()
    sec_starts = sorted({int(s["start"]) for s in song.sections} | {0})
    slots = []
    last = None
    for a, b in zip(bounds[:-1], bounds[1:]):
        votes = {}
        for i in range(a, b):
            c = chords[i] if i < len(chords) else None
            if c is not None:
                votes[c] = votes.get(c, 0) + _metric_w(i - song.downbeat, bpb) + 0.2
        det = max(votes, key=votes.get) if votes else last
        last = det or last
        mel = [m for m in song.melody if m.start < b and m.start + m.dur > a]
        cvec = chroma[a:b].mean(axis=0) if b > a and len(chroma) >= b else np.zeros(12)
        # phrase position (bars since section start)
        sec0 = max([s for s in sec_starts if s <= a] or [0])
        bar_end = (b - max(sec0, song.downbeat)) / bpb
        phrase_end = (abs(bar_end - round(bar_end)) < 1e-6 and round(bar_end) > 0 and round(bar_end) % 4 == 0)
        big_end = phrase_end and round(bar_end) % 8 == 0
        nxt_sec = any(s == b for s in sec_starts) or b == n
        slots.append({"start": a, "end": b, "detected": det, "mel": mel, "chroma": cvec,
                      "phrase_end": bool(phrase_end or nxt_sec), "big_end": bool(big_end or nxt_sec)})

    r = p.reharm
    tonic, mode = song.tonic, song.mode
    S = len(slots)
    cand = [_candidates(tonic, mode, p.era, s["detected"]) for s in slots]
    emis = []
    for s, cs in zip(slots, cand):
        cv = s["chroma"]
        cvn = cv / (np.linalg.norm(cv) + 1e-9)
        row = []
        mel_tot = sum(m.dur for m in s["mel"]) + 1e-9
        for c in cs:
            fid = float(_tpl(c) @ cvn) if cv.sum() > 0 else 0.5
            det = 0.0
            if s["detected"] is not None:
                d = s["detected"]
                if c == d:
                    det = 1.0
                elif c.root == d.root and {"maj": 0, "min": 1}.get(c.qual[:3]) == {"maj": 0, "min": 1}.get(d.qual[:3]):
                    det = 0.7
                elif set(c.pcs[:3]) & set(d.pcs[:3]) and len(set(c.pcs[:3]) & set(d.pcs[:3])) >= 2:
                    det = 0.35
            mel = 0.0
            for m in s["mel"]:
                ov = min(m.start + m.dur, s["end"]) - max(m.start, s["start"])
                w = _metric_w(m.start - song.downbeat, song.beats_per_bar)
                mel += ov * (1.0 if m.pitch % 12 in c.pcs else (-0.8 * w))
            mel /= mel_tot
            uni = M.logp(p.era, mode, None, c.token(tonic))
            e = (1 - r) * (1.6 * fid + 2.4 * det) + r * (0.9 * fid + 0.6 * det) + 1.4 * mel + r * 0.5 * uni
            row.append(e)
        emis.append(row)

    # Viterbi
    dp = list(emis[0])
    back = []
    for t in range(1, S):
        new, bp = [], []
        for j, c in enumerate(cand[t]):
            best, arg = -1e9, 0
            tok = c.token(tonic)
            for i, pc in enumerate(cand[t - 1]):
                ptok = pc.token(tonic)
                tr = (0.35 + 1.1 * r) * M.logp(p.era, mode, ptok, tok)
                if slots[t]["phrase_end"] and r > 0:
                    if tok in ("0:maj", "0:min") and ptok in ("7:maj", "7:dom7"):
                        tr += (3.0 if slots[t]["big_end"] else 1.5) * r
                    elif tok in ("7:maj", "7:dom7") and not slots[t]["big_end"]:
                        tr += 0.8 * r
                if pc == c and slots[t]["detected"] != slots[t - 1]["detected"]:
                    tr -= 0.4 * (1 - r)
                v = dp[i] + tr
                if v > best:
                    best, arg = v, i
            new.append(best + emis[t][j])
            bp.append(arg)
        dp = new
        back.append(bp)
    k = int(np.argmax(dp))
    path = [k]
    for bp in reversed(back):
        k = bp[k]
        path.append(k)
    path = path[::-1]
    for s, cs, k in zip(slots, cand, path):
        s["chord"] = cs[k]
        s["token"] = cs[k].token(tonic)
    return slots


# ================================================================ 3 voicing
_VL_CACHE = {}


def voice_lead(slots, era, rng):
    """4-part voicing by dynamic programming (memoised: the optimiser often
    revisits the same chord sequence with different texture settings)."""
    key = (era, tuple((s["start"], s["end"], s["token"], s["mel"][0].pitch if s["mel"] else None,
                       min([m.pitch for m in s["mel"]], default=None)) for s in slots))
    if key in _VL_CACHE:
        for s, v in zip(slots, _VL_CACHE[key]):
            s["voicing"] = list(v)
        return
    _voice_lead(slots, era, rng)
    if len(_VL_CACHE) > 512:
        _VL_CACHE.clear()
    _VL_CACHE[key] = [tuple(s["voicing"]) for s in slots]


def _voice_lead(slots, era, rng):
    inv_cost = {"baroque": 0.6, "classical": 1.2, "romantic": 0.9}[era]
    cands_all = []
    for s in slots:
        c = s["chord"]
        pcs = c.pcs
        third = pcs[1]
        mel_lo = min([m.pitch for m in s["mel"]], default=79)
        mel_lo = max(mel_lo, 64)
        lt = None
        cand = []
        bass_pcs = [(c.root, 0.0), (third, inv_cost)] if c.qual not in ("sus4",) else [(c.root, 0.0)]
        for bpc, bc in bass_pcs:
            for bp in range(36, 56):
                if bp % 12 != bpc:
                    continue
                for tp in range(max(bp + 3, 48), 68):
                    if tp % 12 not in pcs:
                        continue
                    for ap in range(max(tp + 1, 55), 76):
                        if ap % 12 not in pcs:
                            continue
                        have = {bp % 12, tp % 12, ap % 12} | {m.pitch % 12 for m in s["mel"][:1]}
                        need = {pcs[0], pcs[1]} | ({pcs[3]} if len(pcs) > 3 else set())
                        if not need <= have:
                            continue
                        cost = bc
                        if ap - tp > 12:
                            cost += 2
                        if tp - bp > 19:
                            cost += 1
                        if ap >= mel_lo - 1:
                            cost += 4 + (ap - mel_lo)
                        if pcs[2] not in have:
                            cost += 0.3
                        if c.qual == "maj" and [bp % 12, tp % 12, ap % 12].count(third) > 1:
                            cost += 1
                        cand.append(((bp, tp, ap), cost))
        cand.sort(key=lambda x: x[1])
        cands_all.append(cand[:36] or [((48, 55, 60), 5)])
        _ = lt

    def trans_matrix(V1, V2, s1, s2):
        """vectorised transition costs between all voicing pairs (K1 x K2)."""
        A, B = V1[:, None, :], V2[None, :, :]
        db = np.abs(B[..., 0] - A[..., 0])
        cost = 0.35 * db + 0.8 * np.abs(B[..., 1] - A[..., 1]) + 0.8 * np.abs(B[..., 2] - A[..., 2]) + (db > 7)
        shape = cost.shape
        v1 = [np.broadcast_to(A[..., k], shape) for k in range(3)]
        v2 = [np.broadcast_to(B[..., k], shape) for k in range(3)]
        if s1 is not None and s2 is not None:
            v1.append(np.full(shape, s1))
            v2.append(np.full(shape, s2))
        last = len(v1) - 1
        for i in range(len(v1)):
            for j in range(i + 1, len(v1)):
                x1, y1, x2, y2 = v1[i], v1[j], v2[i], v2[j]
                i1, i2 = np.abs(y1 - x1) % 12, np.abs(y2 - x2) % 12
                m = (x1 != x2) & (y1 != y2) & ((x2 - x1) * (y2 - y1) > 0) & ((i1 == 0) | (i1 == 7)) & (i1 == i2)
                cost = cost + m * (10 if (i == 0 and j == last) else 6)
        return cost

    sop = [s["mel"][0].pitch if s["mel"] else None for s in slots]
    S = len(slots)
    Vs = [np.array([v for v, _ in c], float) for c in cands_all]
    Cs = [np.array([c for _, c in cc], float) for cc in cands_all]
    dp = Cs[0]
    back = []
    for t in range(1, S):
        tot = dp[:, None] + trans_matrix(Vs[t - 1], Vs[t], sop[t - 1], sop[t])
        arg = np.argmin(tot, axis=0)
        dp = tot[arg, np.arange(tot.shape[1])] + Cs[t]
        back.append(arg)
    k = int(np.argmin(dp))
    path = [k]
    for bp in reversed(back):
        k = int(bp[k])
        path.append(k)
    path = path[::-1]
    for s, cs, k in zip(slots, cands_all, path):
        s["voicing"] = list(cs[k][0])


# ================================================================ 4 melody
_LAST_PLAIN = []


def prepare_melody(song: SongAnalysis, p: ArrangeParams, slots, center, rng):
    mel = [Note(n.start, n.dur, n.pitch, n.vel) for n in song.melody]
    if not mel:
        return []
    # --- octave placement
    med = float(np.median([n.pitch for n in mel]))
    shift = 12 * round((center - med) / 12) + 12 * p.melody_octave
    for n in mel:
        n.pitch += shift
    # --- straighten syncopation
    s = p.straighten
    if s > 0:
        out = []
        for n in mel:
            frac = n.start % 1
            end = n.start + n.dur
            if s >= 0.33 and (abs(frac - 0.25) < 1e-6 or abs(frac - 0.75) < 1e-6):
                # 16th anticipation: move to nearest 8th (or beat if strongly straightening)
                grid = 1.0 if s >= 0.75 else 0.5
                ns = round(n.start / grid) * grid
                if ns > n.start and ns < end:   # anticipation of next beat -> land on it
                    n.start = ns
                elif s >= 0.6:
                    n.start = ns
            if s >= 0.5 and abs((n.start % 1) - 0.5) < 1e-6 and n.dur >= 1.0:
                # eighth-note syncopation tied over the beat -> delay to the beat
                if rng.random() < s:
                    n.start = math.ceil(n.start)
            n.dur = max(0.25, end - n.start)
            out.append(n)
        out.sort(key=lambda x: x.start)
        # merge rapid repeated notes (pop/rap "patter") into longer tones
        merged = []
        for n in out:
            if (merged and n.pitch == merged[-1].pitch and n.dur <= 0.5 and merged[-1].dur <= 0.5
                    and n.start - (merged[-1].start + merged[-1].dur) < 0.26 and rng.random() < s):
                merged[-1].dur = n.start + n.dur - merged[-1].start
                continue
            merged.append(n)
        # fix overlaps
        for i in range(len(merged) - 1):
            if merged[i].start + merged[i].dur > merged[i + 1].start:
                merged[i].dur = max(0.125, merged[i + 1].start - merged[i].start)
        mel = [n for n in merged if n.dur > 0.1]
    # --- legato (fill small gaps) – classical lines breathe at phrase ends only
    for i in range(len(mel) - 1):
        gap = mel[i + 1].start - (mel[i].start + mel[i].dur)
        if 0 < gap <= 0.5:
            mel[i].dur += gap
    # --- develop: keep the tune's skeleton (strong-beat notes), re-compose the rest
    if p.develop > 0 and len(mel) > 2:
        scale = scale_pcs(song.tonic, song.mode)
        skel = [n for n in mel if _metric_w(n.start - song.downbeat, song.beats_per_bar) >= 0.7 or n.dur >= 1.5]
        if len(skel) >= 2:
            out = []
            for a, b in zip(skel, skel[1:] + [None]):
                inner = [n for n in mel if a.start < n.start < (b.start if b else 1e9)]
                if b is None or rng.random() > p.develop or b.start - a.start < 1.0:
                    out.append(a)
                    out += inner
                    continue
                gap = b.start - a.start
                head = min(1.0, gap / 2)
                out.append(Note(a.start, head, a.pitch, a.vel))
                t, q = a.start + head, a.pitch
                direction = 1 if b.pitch > a.pitch else -1
                while t < b.start - 1e-6:
                    q = _scale_step(q, direction, scale)
                    if (direction > 0 and q > b.pitch + 2) or (direction < 0 and q < b.pitch - 2):
                        direction *= -1
                        q = _scale_step(q, direction, scale)
                    out.append(Note(t, min(0.5, b.start - t), q, a.vel - 6))
                    t += 0.5
            mel = sorted(out, key=lambda n: n.start)
    # --- ornaments
    global _LAST_PLAIN
    _LAST_PLAIN = [(n.start, n.dur, n.pitch) for n in mel]
    o = p.ornament
    if o > 0:
        scale = scale_pcs(song.tonic, song.mode)
        orn = []
        slot_of = lambda t: next((sl for sl in slots if sl["start"] <= t < sl["end"]), slots[-1])  # noqa: E731
        for i, n in enumerate(mel):
            sl = slot_of(n.start)
            nxt = mel[i + 1] if i + 1 < len(mel) else None
            cad = sl["phrase_end"] or (nxt and slot_of(nxt.start)["phrase_end"] and sl is not slot_of(nxt.start))
            dominant = sl["token"] in ("7:maj", "7:dom7")
            up = _scale_step(n.pitch, +1, scale)
            lo = _scale_step(n.pitch, -1, scale)
            if cad and dominant and n.dur >= 1.0 and p.era != "romantic" and rng.random() < 0.4 + 0.6 * o:
                # cadential trill (start on upper note in Baroque) + termination
                step = 0.125 if n.dur <= 2 else 0.25 / 2
                t, alt = n.start, (p.era == "baroque")
                end = n.start + n.dur - 0.25
                while t < end - 1e-6:
                    orn.append(Note(t, step, up if alt else n.pitch, n.vel - 8))
                    alt = not alt
                    t += step
                orn.append(Note(end, 0.125, lo, n.vel - 8))
                orn.append(Note(end + 0.125, 0.125, n.pitch, n.vel - 8))
                continue
            if n.dur >= 2.0 and p.era == "baroque" and rng.random() < 0.5 * o:
                # diminution: turn figure on the first beat
                for k, q in enumerate([n.pitch, up, n.pitch, lo]):
                    orn.append(Note(n.start + 0.25 * k, 0.25, q, n.vel - 5))
                orn.append(Note(n.start + 1, n.dur - 1, n.pitch, n.vel))
                continue
            strong = _metric_w(n.start - song.downbeat, song.beats_per_bar) >= 0.7
            if (strong and n.dur >= 1.0 and p.era in ("classical", "romantic") and nxt is not None
                    and rng.random() < 0.35 * o and n.pitch % 12 in sl["chord"].pcs):
                # appoggiatura from above, resolving by step
                a = min(0.5, n.dur / 2)
                orn.append(Note(n.start, a, up, n.vel + 6))
                orn.append(Note(n.start + a, n.dur - a, n.pitch, n.vel - 6))
                continue
            if (p.era == "baroque" and strong and 0.5 <= n.dur < 2 and rng.random() < 0.25 * o):
                orn.append(Note(n.start, 0.125, n.pitch, n.vel))       # mordent
                orn.append(Note(n.start + 0.125, 0.125, lo, n.vel - 10))
                orn.append(Note(n.start + 0.25, n.dur - 0.25, n.pitch, n.vel))
                continue
            if p.era == "romantic" and nxt and n.dur >= 1.5 and abs(nxt.pitch - n.pitch) >= 3 and rng.random() < 0.3 * o:
                # turn into the next note
                tail = n.start + n.dur - 0.5
                orn.append(Note(n.start, n.dur - 0.5, n.pitch, n.vel))
                for k, q in enumerate([up, n.pitch, lo, n.pitch]):
                    orn.append(Note(tail + 0.125 * k, 0.125, q, n.vel - 10))
                continue
            orn.append(n)
        mel = orn
    return mel


# ================================================================ 5 textures
def _sec_energy(song, beat):
    for s in song.sections:
        if s["start"] <= beat < s["end"]:
            return float(s["energy"])
    return 0.6


def _vel_for(song, beat, p, base=74):
    """terraced dynamics from section energy (0..1, relative to the loudest
    section of the whole song) scaled by the dynamic-range parameter."""
    e = _sec_energy(song, beat)
    return int(base + p.dynamics * 40 * (e - 0.65))


def _bass_line(slots, era, p, rng, scale, bpb):
    """returns list of (start, dur, pitch)"""
    out = []
    unit = 1.0 if p.density < 0.6 else 0.5
    for i, s in enumerate(slots):
        b0 = s["voicing"][0]
        a, e = s["start"], s["end"]
        nxt = slots[i + 1]["voicing"][0] if i + 1 < len(slots) else b0
        if era == "baroque":
            # walking continuo bass: chord root then stepwise/arpeggio approach to next bass
            n_steps = int(round((e - a) / unit))
            line = [b0]
            for k in range(1, n_steps):
                remaining = n_steps - k
                target = nxt
                cur = line[-1]
                if abs(target - cur) <= remaining * 2 and remaining <= 2 and target != cur:
                    d = 1 if target > cur else -1
                    line.append(_scale_step(cur, d, scale))
                else:
                    # chord-tone arpeggiation, alternating octave
                    pcs = s["chord"].pcs
                    opts = [q for q in range(cur - 7, cur + 8) if q % 12 in pcs and q != cur and 36 <= q <= 60]
                    line.append(rng.choice(opts) if opts else cur)
            for k, q in enumerate(line):
                out.append((a + k * unit, unit * 0.92, q))
        elif era == "classical":
            if p.density < 0.35:
                out.append((a, (e - a) * 0.9, b0))
            elif p.density < 0.7:
                for t in np.arange(a, e, 1.0):
                    if (t - a) % 2 == 0 or p.density > 0.5:
                        out.append((t, 0.8, b0))
            else:   # "drum bass": repeated 8ths
                for t in np.arange(a, e, 0.5):
                    out.append((t, 0.4, b0))
        else:   # romantic: long sustained bass, octave on downbeat
            out.append((a, (e - a), b0))
    return out


def _is_parallel(a1, a2, b1, b2):
    if None in (a1, a2, b1, b2) or a1 == a2 or b1 == b2:
        return False
    if (a2 - a1) * (b2 - b1) <= 0:
        return False
    i1, i2 = abs(a1 - b1) % 12, abs(a2 - b2) % 12
    return i1 == i2 and i1 in (0, 7)


def _fix_parallels(line, slots, ref_fn, lo=28, hi=72):
    """repair pass: re-pick notes of `line` that make parallel 5ths/8ves with
    the reference voice (usually the melody)."""
    out = list(line)

    def slot_at(t):
        for s in slots:
            if s["start"] <= t < s["end"]:
                return s
        return slots[-1]
    for i in range(1, len(out)):
        t0, _, a1 = out[i - 1]
        t1, d1, a2 = out[i]
        m1, m2 = ref_fn(t0), ref_fn(t1)
        if _is_parallel(a1, a2, m1, m2):
            pcs = slot_at(t1)["chord"].pcs
            alts = [q for q in range(a1 - 7, a1 + 8) if q % 12 in pcs and lo <= q <= hi and q != a2
                    and not _is_parallel(a1, q, m1, m2)]
            if alts:
                out[i] = (t1, d1, min(alts, key=lambda q: abs(q - a2)))
    return out


def _sounding(part, t):
    best = None
    for st, d, q, v in part.notes:
        if st <= t + 1e-6 < st + d:
            best = q if best is None else max(best, q)
    return best


def _repair_voices(parts, slots, bpb):
    """Global voice-leading clean-up over the independent voices: any note that
    forms parallel 5ths/8ves with an already-fixed voice is re-picked among the
    chord tones (melody first, then bass, then inner voices)."""
    order = ["melody", "bass", "keys_lh", "counter", "inner_hi", "inner_lo"]
    voices, seen = [], set()
    for role in order:
        for pt in parts:
            if pt.role == role and role not in seen:
                voices.append(pt)
                seen.add(role)
                break

    def slot_at(t):
        for sl in slots:
            if sl["start"] <= t < sl["end"]:
                return sl
        return slots[-1]
    fixed = voices[:1]
    for pt in voices[1:]:
        if pt.role == "keys_lh":           # arpeggio figures: leave the pattern intact
            fixed.append(pt)
            continue
        lo, hi = INSTR[pt.name][1], INSTR[pt.name][2]
        notes = sorted(pt.notes, key=lambda n: n[0])
        for i in range(1, len(notes)):
            t0, t1 = notes[i - 1][0], notes[i][0]
            a1, a2 = notes[i - 1][2], notes[i][2]
            refs = [(_sounding(f, t0), _sounding(f, t1)) for f in fixed]
            if not any(_is_parallel(a1, a2, b1, b2) for b1, b2 in refs):
                continue
            pcs = slot_at(t1)["chord"].pcs
            alts = [q for q in range(a1 - 7, a1 + 8) if q % 12 in pcs and lo <= q <= hi and q != a2
                    and not any(_is_parallel(a1, q, b1, b2) for b1, b2 in refs)]
            if not alts:   # oblique motion (repeat the previous note) is always safe
                alts = [a1]
            notes[i][2] = min(alts, key=lambda q: abs(q - a2))
        pt.notes = notes
        fixed.append(pt)


def _arp_pattern(era, p, bpb, voicing, chord, slot_len, tempo, top=72):
    """keyboard left hand figuration -> list of (offset, dur, pitch); stays below `top`"""
    b, t, a = voicing
    if era == "classical":   # Alberti bass
        lo = b if b >= 43 else b + 12
        mid = _nearest_pc(lo + 7, t % 12)
        if mid <= lo:
            mid += 12
        hi = _nearest_pc(mid + 4, a % 12)
        if hi <= mid:
            hi += 12
        while hi >= top and lo - 12 >= 36:
            lo, mid, hi = lo - 12, mid - 12, hi - 12
        unit = 0.5 if p.density < 0.7 else 0.25
        pat = [lo, hi, mid, hi]
        out, k = [], 0
        for o in np.arange(0, slot_len, unit):
            out.append((o, unit * 0.95, pat[k % 4]))
            k += 1
        return out
    if era == "romantic":    # wide arpeggio: root, 5th, octave, 10th, octave, 5th
        r = b if b < 48 else b - 12
        fifth = r + 7 if (r + 7) % 12 in chord.pcs else _nearest_pc(r + 7, chord.pcs[2])
        third = _nearest_pc(r + 16, chord.pcs[1])
        seq = [r, fifth, r + 12, third, r + 12, fifth]
        while max(seq) >= top and min(seq) - 12 >= 28:
            seq = [x - 12 for x in seq]
        triplets = tempo < 96 and p.density >= 0.35
        unit = (1 / 3 if triplets else 0.5) if p.density < 0.8 else (0.25 if not triplets else 1 / 3)
        out, k = [], 0
        for o in np.arange(0, slot_len - 1e-6, unit):
            out.append((o, unit * 1.6, seq[k % len(seq)]))   # overlapping (pedal)
            k += 1
        return out
    # baroque keyboard LH: bass in 8ths with octave leaps
    unit = 0.5 if p.density >= 0.4 else 1.0
    out = []
    for k, o in enumerate(np.arange(0, slot_len, unit)):
        out.append((o, unit * 0.9, b if k % 2 == 0 else b + 12))
    return out


def _counter_line(slots, mel, hi_bound_fn, scale, era, p, rng, reg_lo, reg_hi, start_voice=2,
                  avoid_voices=(0, 1)):
    """Contrapuntal inner line in complementary rhythm to the melody.
    On strong beats it takes chord tones that are NOT already sounding in the
    other voices or the melody, which keeps it independent (no parallel
    octaves/unisons with the other parts)."""
    out = []
    prev = None
    prev_v = None
    for s in slots:
        a, e = s["start"], s["end"]
        pcs = s["chord"].pcs
        busy_pcs = {s["voicing"][i] % 12 for i in avoid_voices}

        def free(t, cands):
            m = hi_bound_fn(t)
            bad = busy_pcs | ({int(m) % 12} if m is not None else set())
            f = [q for q in cands if q % 12 not in bad]
            return f or cands
        target = s["voicing"][start_voice]
        while target < reg_lo:
            target += 12
        while target > reg_hi:
            target -= 12
        base = prev if prev is not None else target
        near = [q for q in range(base - 9, base + 10) if q % 12 in pcs and reg_lo <= q <= reg_hi] or [target]
        if prev is not None and prev_v is not None:
            # hard rule: no parallel 8ves/5ths with any chord voice
            def parallel(q):
                for i in range(3):
                    x1, x2 = prev_v[i], s["voicing"][i]
                    if x1 == x2 or q == prev:
                        continue
                    if (x2 - x1) * (q - prev) > 0 and abs(prev - x1) % 12 in (0, 7) and \
                            abs(q - x2) % 12 == abs(prev - x1) % 12:
                        return True
                return False
            near = [q for q in near if not parallel(q)] or [prev]
        near = free(a, near)
        cur = min(near, key=lambda q: abs(q - (prev if prev is not None else target)))
        # melody activity in this slot
        onsets = [m for m in mel if a <= m.start < e]
        busy = len(onsets) / max(1.0, (e - a)) > 1.6
        unit = 1.0 if busy else (0.5 if p.counterpoint > 0.3 else 1.0)
        if era == "romantic" and not busy:
            unit = 1.0 if p.counterpoint < 0.7 else 0.5
        t = a
        direction = -1 if (onsets and len(onsets) > 1 and onsets[-1].pitch > onsets[0].pitch) else 1
        while t < e - 1e-6:
            mlim = hi_bound_fn(t) - 2
            q = cur
            while q >= mlim and q - 12 >= reg_lo:
                q -= 12
            dur = min(unit, e - t)
            out.append((t, dur * 0.95, q))
            # next pitch: chord tone on strong beats, passing tones otherwise
            nt = t + unit
            if abs((nt - a) % 1) < 1e-6:
                cands = [x for x in range(q - 5, q + 6) if x % 12 in pcs and reg_lo <= x <= reg_hi]
                cands = free(nt, cands)
                cur = rng.choice(cands) if cands and rng.random() < p.counterpoint else (
                    cands[0] if cands and q % 12 in busy_pcs else q)
            else:
                cur = _scale_step(q, direction, scale)
                if cur > reg_hi or cur < reg_lo:
                    direction *= -1
                    cur = _scale_step(q, direction, scale)
            t = nt
        prev = out[-1][2] if out else prev
        prev_v = s["voicing"]
    return out


def build_parts(song, p, slots, mel, rng):
    era = p.era
    layout = ensemble_layout(era, p.ensemble)
    scale = scale_pcs(song.tonic, song.mode)
    bpb = song.beats_per_bar
    tempo = song.tempo * p.tempo_factor
    parts = {}

    def part(role, idx=0):
        name = layout[role][idx]
        key = f"{role}:{name}"
        if key not in parts:
            parts[key] = Part(name, INSTR[name][0], role)
        return parts[key]

    def loud(beat):
        return _sec_energy(song, beat)

    # melody top-limit for accompaniment (keep accompaniment under the tune)
    mel_sorted = sorted(mel, key=lambda n: n.start)

    def mel_at(t):
        best = None
        for n in mel_sorted:
            if n.start <= t + 1e-6 and n.start + n.dur > t - 1.0:
                best = n.pitch
            if n.start > t + 1:
                break
        return best if best is not None else 79

    def mel_pitch(t):
        for n in mel_sorted:
            if n.start <= t + 1e-6 < n.start + n.dur:
                return n.pitch
            if n.start > t + 1e-6:
                break
        return None

    # ---------------- melody (+ doublings)
    m_lead = part("melody", 0)
    for n in mel:
        v = _vel_for(song, n.start, p, 84) + int(6 * (n.vel - 80) / 40)
        art = {"baroque": 0.9, "classical": 0.95, "romantic": 1.03}[era]
        m_lead.add(n.start, n.dur * art, n.pitch, v)
    if len(layout["melody"]) > 1:
        dbl = part("melody", 1)
        for n in mel:
            if loud(n.start) >= 0.6 or era == "baroque":
                octave = 12 if dbl.name == "flute" else 0
                dbl.add(n.start, n.dur * 0.95, n.pitch + octave, _vel_for(song, n.start, p, 70))

    # gap filler: if no melody for >= 1 bar, lead plays a chord-tone line
    covered = np.zeros(song.n_beats + 1, bool)
    for n in mel:
        covered[int(max(0, n.start)):int(min(song.n_beats, math.ceil(n.start + n.dur)))] = True
    prev = INSTR[m_lead.name][3]
    for s in slots:
        a, e = s["start"], s["end"]
        if covered[a:e].any() or (e - a) < 1:
            continue
        pcs = s["chord"].pcs
        q = min([x for x in range(prev - 6, prev + 7) if x % 12 in pcs] or [prev], key=lambda x: abs(x - prev))
        rh = [(0, e - a)] if (e - a) <= 2 or era == "romantic" else [(0, 1.5), (1.5, 0.5)] + (
            [(2, e - a - 2)] if e - a > 2 else [])
        for k, (o, d) in enumerate(rh):
            pitch = q if k != 1 else _scale_step(q, 1, scale)
            m_lead.add(a + o, d * 0.95, pitch, _vel_for(song, a, p, 62))
        prev = q

    # ---------------- bass
    if "bass" in layout:
        line = _bass_line(slots, era, p, rng, scale, bpb)
        line = _fix_parallels(line, slots, lambda t: mel_pitch(t), 33, 62)
        bp = part("bass", 0)
        for st, d, q in line:
            bp.add(st, d, q, _vel_for(song, st, p, 70))
        if len(layout["bass"]) > 1:
            b2 = part("bass", 1)
            for st, d, q in line:
                if b2.name == "contrabass":
                    b2.add(st, d, q - 12, _vel_for(song, st, p, 64))
                elif loud(st) >= 0.4:
                    b2.add(st, d, q, _vel_for(song, st, p, 58))

    # ---------------- keyboard / continuo
    if "keys_lh" in layout:
        lh = part("keys_lh")
        for s in slots:
            top = min(mel_at(s["start"] + k) for k in range(max(1, s["end"] - s["start"]))) - 2
            for o, d, q in _arp_pattern(era, p, bpb, s["voicing"], s["chord"], s["end"] - s["start"], tempo, top):
                lh.add(s["start"] + o, d, q, _vel_for(song, s["start"], p, 62))
            if era == "romantic" and lh.name.startswith("piano"):
                lh.cc.append((s["start"] + 0.02, 64, 127))
                lh.cc.append((s["end"] - 0.05, 64, 0))
        if p.ensemble == "piano":
            # RH adds chord tones under the melody on strong beats in loud sections
            rh = part("melody")
            for s in slots:
                if loud(s["start"]) >= 0.5 or era == "baroque":
                    top = mel_at(s["start"]) - 3
                    for q in s["voicing"][1:]:
                        while q >= top:
                            q -= 12
                        if q >= 52:
                            rh.add(s["start"], min(1.0, s["end"] - s["start"]) * 0.9, q,
                                   _vel_for(song, s["start"], p, 56))
    if "continuo" in layout:
        crh, clh = part("continuo", 0), part("continuo", 1)
        for s in slots:
            v = _vel_for(song, s["start"], p, 54)
            for t in np.arange(s["start"], s["end"], 1.0):
                top = mel_at(t) - 2
                for q in s["voicing"][1:]:
                    while q >= top:
                        q -= 12
                    crh.add(t, 0.8, max(q, 53), v)
            clh.add(s["start"], (s["end"] - s["start"]) * 0.9, s["voicing"][0], v)

    # ---------------- inner strings
    if "inner_hi" in layout:
        ih = part("inner_hi")
        lo_r, hi_r = INSTR[ih.name][1], 84
        if era == "baroque" or (era == "romantic" and p.counterpoint >= 0.6):
            line = _counter_line(slots, mel, mel_at, scale, era, p, rng, lo_r, hi_r, avoid_voices=(0, 1))
            for st, d, q in _fix_parallels(line, slots, mel_pitch, lo_r, hi_r):
                ih.add(st, d, q, _vel_for(song, st, p, 64))
        elif era == "classical":
            unit = 0.5 if p.density >= 0.35 else 1.0
            for s in slots:
                for t in np.arange(s["start"], s["end"], unit):
                    q = s["voicing"][2]
                    while q >= mel_at(t) - 2:
                        q -= 12
                    ih.add(t, unit * 0.8, max(q, 55), _vel_for(song, t, p, 58))
        elif p.density >= 0.5:  # romantic pulsing chords (Tchaikovsky-style), swelling
            unit = 0.5 if p.density < 0.85 else 1 / 3
            for s in slots:
                for t in np.arange(s["start"], s["end"] - 1e-6, unit):
                    q = s["voicing"][2]
                    while q >= mel_at(t) - 2:
                        q -= 12
                    ph = (t - s["start"]) / max(1e-6, s["end"] - s["start"])
                    ih.add(t, unit * 0.9, max(q, 55), _vel_for(song, t, p, 54) + int(10 * math.sin(math.pi * ph)))
        else:  # romantic sustained, swell
            for s in slots:
                q = s["voicing"][2]
                while q >= mel_at(s["start"]) - 2:
                    q -= 12
                ih.add(s["start"], s["end"] - s["start"], max(q, 55), _vel_for(song, s["start"], p, 58))
                ih.cc += [(s["start"], 11, 70), (s["start"] + (s["end"] - s["start"]) / 2, 11, 110),
                          (s["end"] - 0.1, 11, 80)]
    if "inner_lo" in layout:
        il = part("inner_lo")
        for s in slots:
            q = s["voicing"][1]
            if era == "classical" and p.density >= 0.35:
                for t in np.arange(s["start"], s["end"], 0.5):
                    il.add(t, 0.4, q, _vel_for(song, t, p, 55))
            elif era == "baroque":
                for t in np.arange(s["start"], s["end"], 1.0):
                    il.add(t, 0.9, q, _vel_for(song, t, p, 56))
            else:
                il.add(s["start"], s["end"] - s["start"], q, _vel_for(song, s["start"], p, 55))
                il.cc += [(s["start"], 11, 70), (s["start"] + (s["end"] - s["start"]) / 2, 11, 105),
                          (s["end"] - 0.1, 11, 80)]

    # ---------------- romantic cello counter-melody
    if "counter" in layout:
        cp = part("counter")
        for st, d, q in _counter_line(slots, mel, mel_at, scale, era,
                                      ArrangeParams(**{**p.to_dict(), "counterpoint": max(0.4, p.counterpoint)}),
                                      rng, 43, 67, start_voice=1, avoid_voices=(0, 2)):
            cp.add(st, d * 1.0, q, _vel_for(song, st, p, 66))

    # ---------------- winds / brass / harp / timpani (orchestra colour)
    if "winds" in layout:
        for k, nm in enumerate(layout["winds"]):
            w = part("winds", k)
            for s in slots:
                if loud(s["start"]) < 0.45:
                    continue
                q = s["voicing"][2 - k] if k < 2 else s["voicing"][1]
                while q >= mel_at(s["start"]) - 2:
                    q -= 12
                w.add(s["start"], s["end"] - s["start"], q, _vel_for(song, s["start"], p, 54))
    if "brass" in layout:
        br = part("brass")
        for s in slots:
            if loud(s["start"]) < 0.7:
                continue
            q = s["voicing"][1]
            if br.name == "trumpet":
                q = _nearest_pc(70, s["chord"].root)
            br.add(s["start"], (s["end"] - s["start"]) * 0.9, q, _vel_for(song, s["start"], p, 56))
    if "harp" in layout:
        hp = part("harp")
        for s in slots:
            if loud(s["start"]) < 0.3:
                continue
            seq = sorted(set([s["voicing"][0] + 12, s["voicing"][1] + 12, s["voicing"][2],
                              s["voicing"][2] + 12 if s["voicing"][2] + 12 < mel_at(s["start"]) else s["voicing"][1]]))
            unit = (s["end"] - s["start"]) / max(4, 2 * len(seq))
            for k in range(int(round((s["end"] - s["start"]) / unit))):
                hp.add(s["start"] + k * unit, unit * 3, seq[k % len(seq)], _vel_for(song, s["start"], p, 50))
    if "timpani" in layout:
        tp = part("timpani")
        tpc = {song.tonic % 12, (song.tonic + 7) % 12}
        for s in slots:
            if s["phrase_end"] and loud(s["start"]) >= 0.6 and s["chord"].root in tpc:
                q = _nearest_pc(45, s["chord"].root)
                if era == "romantic":
                    for k in range(8):
                        tp.add(s["start"] + k * 0.125, 0.12, q, 50 + 4 * k)
                else:
                    tp.add(s["start"], 0.5, q, 70)
                    tp.add(s["start"] + 1, 0.5, q, 62)
    _repair_voices(list(parts.values()), slots, bpb)
    return list(parts.values())


# ================================================================ 6 tempo map & ending
def tempo_map(song, p, n_total):
    base = 60.0 / max(40.0, song.tempo * p.tempo_factor)
    mult = np.ones(n_total)
    bpb = song.beats_per_bar
    r = p.rubato if p.era != "baroque" else 0.0
    if r > 0:
        for s in song.sections:
            e = int(s["end"])
            for k in range(1, 3):
                if 0 <= e - k < n_total:
                    mult[e - k] *= 1 + 0.09 * k * r          # breathe at section ends
        # phrase-level ebb and flow (push forward, then relax into the cadence)
        ph = np.arange(n_total) % (4 * bpb) / (4 * bpb)
        mult *= 1 + 0.045 * r * np.cos(2 * np.pi * ph)
    # final ritardando
    for k in range(1, min(n_total, 2 * bpb) + 1):
        mult[n_total - k] *= 1 + 0.18 * (1 - (k - 1) / (2 * bpb))
    bs = np.concatenate([[0.0], np.cumsum(base * mult)])
    return bs.tolist()


def exact_beat_map(song, n_total):
    """the original performance's beat times (relative to the clip start), so the
    arrangement lines up with, and lasts exactly as long as, the original."""
    bt = np.asarray(song.beat_times, float)
    bt = bt - bt[0]
    per = float(np.median(np.diff(bt))) if len(bt) > 1 else 0.5
    extra = np.arange(1, max(0, n_total + 1 - len(bt)) + 1) * per + bt[-1]
    return np.concatenate([bt, extra])[:n_total + 1].tolist()


def add_cadence(song, p, slots, parts, layout_parts, start):
    """Append a V7 – I perfect authentic cadence (classical endings, pop fades)."""
    tonic, mode = song.tonic, song.mode
    bpb = song.beats_per_bar
    V = Chord((tonic + 7) % 12, "dom7")
    I = Chord(tonic, "maj" if mode == "major" else "min")
    last_v = slots[-1]["voicing"]
    pseudo = [{"start": start, "end": start + bpb, "chord": V, "mel": [], "token": V.token(tonic)},
              {"start": start + bpb, "end": start + 2 * bpb, "chord": I, "mel": [], "token": I.token(tonic)}]
    for s in pseudo:
        b, t, a = last_v
        s["voicing"] = [_nearest_pc(b, s["chord"].root), _nearest_pc(t, s["chord"].pcs[1]),
                        _nearest_pc(a, s["chord"].pcs[2 if s["chord"] is I else 3])]
        last_v = s["voicing"]
    vel = 80
    for pt in parts:
        lo, hi, ctr = INSTR[pt.name][1], INSTR[pt.name][2], INSTR[pt.name][3]
        for k, s in enumerate(pseudo):
            dur = bpb if k == 1 else bpb
            if pt.role == "melody":
                q = _nearest_pc(ctr, s["chord"].pcs[1] if k == 0 else s["chord"].root)   # leading tone -> tonic
                if k == 0:
                    pt.add(s["start"], dur * 0.5, _nearest_pc(ctr, s["chord"].root), vel)
                    pt.add(s["start"] + dur * 0.5, dur * 0.5, q, vel)
                else:
                    pt.add(s["start"], dur * 1.0, q, vel - 6)
            elif pt.role in ("bass", "keys_lh", "counter") or pt.name.endswith("_lh"):
                pt.add(s["start"], dur, s["voicing"][0], vel - 10)
            elif pt.role == "timpani":
                pt.add(s["start"], 0.5, _nearest_pc(45, s["chord"].root), 70)
            else:
                pick = s["voicing"][2] if pt.role in ("inner_hi", "winds", "harp", "continuo", "brass") else s["voicing"][1]
                pt.add(s["start"], dur, pick, vel - 14)
    return pseudo


# ================================================================ entry point
def arrange(song: SongAnalysis, p: ArrangeParams) -> Arrangement:
    rng = random.Random(p.seed)
    slots = harmonise(song, p)
    voice_lead(slots, p.era, rng)
    layout = ensemble_layout(p.era, p.ensemble)
    center = INSTR[layout["melody"][0]][3]
    global _LAST_PLAIN
    _LAST_PLAIN = []
    mel = prepare_melody(song, p, slots, center, rng)
    plain = list(_LAST_PLAIN)
    parts = build_parts(song, p, slots, mel, rng)
    n_total = song.n_beats
    if p.add_ending and slots:
        add_cadence(song, p, slots, parts, layout, song.n_beats)
        n_total += 2 * song.beats_per_bar
    if p.exact_timing and song.beat_times:
        bs = exact_beat_map(song, n_total)
    else:
        bs = tempo_map(song, p, n_total)
    out_slots = [{"start": s["start"], "end": s["end"], "chord": s["chord"].name(), "token": s["token"],
                  "orig": s["detected"].name() if s["detected"] is not None else None,
                  "voicing": s["voicing"], "phrase_end": s["phrase_end"]} for s in slots]
    return Arrangement(params=p.to_dict(), parts=parts, beat_sec=bs, slots=out_slots,
                       melody=[(n.start, n.dur, n.pitch) for n in mel], n_beats=song.n_beats,
                       melody_plain=plain,
                       tonic=song.tonic, mode=song.mode, beats_per_bar=song.beats_per_bar,
                       sections=song.sections)
