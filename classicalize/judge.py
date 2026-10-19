"""The two-part loss.

  score = w * CLASSICALNESS + (1 - w) * RETENTION          (loss = 1 - score)

CLASSICALNESS – "how would a listener from the period hear this?"
  The *period listener* combines
    * era_harmony   – how idiomatic the chord progression is under the era's
                      harmony model learned from real Bach / Haydn / Mozart /
                      Beethoven / Schumann... scores
    * voice_leading – parallel 5ths/8ves between independent voices (vs Bach's rate)
    * cadences      – phrase ends closing with authentic / half cadences
    * rhythm        – melodic syncopation vs the era's 90th percentile
    * melody_style  – stepwise motion, few machine-gun repeated notes
    * form          – not stuck in one 4-chord loop for the whole piece
    * texture       – contrary/oblique motion between melody and bass
    * instruments   – period-appropriate ensemble, no drum kit
RETENTION – "is it still recognisably the song?"
    * melody     – the audible top line (skyline of all parts) vs the original
                   vocal melody: pitch-class agreement + onset F1 (chorus x2)
    * harmony    – beat-synchronous chroma of the rendered audio vs the
                   original's harmonic stems (or symbolic fallback)
    * structure  – bar-level self-similarity of output vs original
    * tempo      – penalty for drifting from the original tempo
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from .analysis import SongAnalysis
from .arrange import ERA_ANACHRONISMS, Arrangement
from .theory import era_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIB_PATH = os.path.join(ROOT, "models", "calibration.json")

DEFAULT_CALIB = {
    "pop_lm_logp": {"baroque": -4.2, "classical": -4.3, "romantic": -4.0},
    "c_weights": {"era_harmony": 0.16, "voice_leading": 0.12, "cadences": 0.1, "rhythm": 0.12,
                  "melody_style": 0.08, "form": 0.08, "texture": 0.06, "instruments": 0.08},
    "r_weights": {"melody": 0.5, "harmony": 0.25, "structure": 0.12, "tempo": 0.13},
}


def load_calibration():
    c = json.loads(json.dumps(DEFAULT_CALIB))
    if os.path.exists(CALIB_PATH):
        with open(CALIB_PATH) as f:
            c.update(json.load(f))
    return c


def _clip01(x):
    return float(min(1.0, max(0.0, x)))


# ================================================================ symbolic helpers
def melody_syncopation(notes):
    """fraction of notes that start off the beat and are held across the next beat
    (same definition used on the classical corpus)."""
    if not notes:
        return 0.0
    c = 0
    for st, d, *_ in notes:
        if (st % 1) > 1e-3 and d >= 0.5 and math.floor(st + d - 1e-6) > math.floor(st):
            c += 1
    return c / len(notes)


VOICE_ROLES = ("melody", "inner_hi", "inner_lo", "counter", "bass", "keys_lh")


def _independent_parts(arr: Arrangement):
    """the real contrapuntal voices. Orchestral colour parts (winds/brass that
    double the strings, continuo realisation, harp, timpani) and second
    instruments of a role (octave/unison doublings) are deliberate doublings,
    which period practice allows, so they are not checked for parallels."""
    keep, seen = [], set()
    for p in arr.parts:
        if p.role not in VOICE_ROLES or p.role in seen:
            continue
        seen.add(p.role)
        keep.append(p)
    return keep


def _sample_top(part, grid):
    starts = np.array([n[0] for n in part.notes]) if part.notes else np.zeros(0)
    ends = np.array([n[0] + n[1] for n in part.notes]) if part.notes else np.zeros(0)
    pitches = np.array([n[2] for n in part.notes]) if part.notes else np.zeros(0)
    out = np.full(len(grid), np.nan)
    for i, g in enumerate(grid):
        on = (starts <= g + 1e-6) & (ends > g + 1e-6)
        if on.any():
            out[i] = pitches[on].max()
    return out


def parallel_rate(arr: Arrangement):
    parts = _independent_parts(arr)
    grid = np.arange(0, arr.n_beats, 0.5)
    V = [_sample_top(p, grid) for p in parts]
    par = trans = 0
    for i in range(len(V)):
        for j in range(i + 1, len(V)):
            a, b = V[i], V[j]
            ok = ~(np.isnan(a[1:]) | np.isnan(a[:-1]) | np.isnan(b[1:]) | np.isnan(b[:-1]))
            ma = (a[1:] != a[:-1]) & (b[1:] != b[:-1]) & ok
            trans += int(ma.sum())
            i1 = np.abs(a[:-1] - b[:-1]) % 12
            i2 = np.abs(a[1:] - b[1:]) % 12
            same_dir = np.sign(a[1:] - a[:-1]) == np.sign(b[1:] - b[:-1])
            par += int((ma & same_dir & (i1 == i2) & ((i1 == 7) | (i1 == 0))).sum())
    bars = max(1.0, arr.n_beats / max(1, arr.beats_per_bar))
    return par / bars          # parallel perfect intervals per bar


def _merged_tokens(arr):
    t = [s["token"] for s in arr.slots]
    return [x for i, x in enumerate(t) if i == 0 or x != t[i - 1]]


def loop_index(tokens):
    if len(tokens) < 8:
        return 0.0
    grams = [tuple(tokens[i:i + 4]) for i in range(len(tokens) - 3)]
    # rotation-invariant: a loop I-V-vi-IV and V-vi-IV-I are the same loop
    canon = [min(tuple(g[k:] + g[:k]) for k in range(4)) for g in grams]
    best = max(canon.count(c) for c in set(canon))
    return best / len(canon)


def cadence_rate(arr):
    ends = [i for i, s in enumerate(arr.slots) if s["phrase_end"] and i > 0]
    if not ends:
        return 0.5
    good = 0
    for i in ends:
        cur, prev = arr.slots[i]["token"], arr.slots[i - 1]["token"]
        if cur in ("0:maj", "0:min") and prev in ("7:maj", "7:dom7", "11:dim", "11:dim7"):
            good += 1
        elif cur in ("7:maj", "7:dom7"):
            good += 0.7
        elif cur in ("0:maj", "0:min") and prev in ("5:maj", "5:min"):
            good += 0.5      # plagal
    return good / len(ends)


def contrary_motion(arr):
    mel = [p for p in arr.parts if p.role == "melody"]
    bas = [p for p in arr.parts if p.role in ("bass", "keys_lh") or p.name.endswith("_lh")]
    if not mel or not bas:
        return 0.5
    grid = np.arange(0, arr.n_beats, 1.0)
    a, b = _sample_top(mel[0], grid), _sample_top(bas[0], grid)
    da, db = np.diff(a), np.diff(b)
    ok = ~(np.isnan(da) | np.isnan(db)) & ((da != 0) | (db != 0))
    if ok.sum() < 4:
        return 0.5
    good = ((np.sign(da) != np.sign(db)) | (da == 0) | (db == 0))[ok]
    return float(good.mean())


# ================================================================ audio CLAP
# ================================================================ classicalness
def classicalness(arr: Arrangement, audio_path=None, use_clap=False, calib=None):
    calib = calib or load_calibration()
    era = arr.params["era"]
    M = era_model()
    comps, notes = {}, {}

    toks = _merged_tokens(arr)
    from .theory import harmony_llr
    llr = harmony_llr(toks, era, arr.mode)
    if llr is not None and "llr_pop" in calib:
        lo, hi = calib["llr_pop"][era], calib["llr_classical"][era]
        comps["era_harmony"] = _clip01((llr - lo) / (hi - lo + 1e-9))
        notes["harmony_llr"] = llr
    else:   # untrained fallback: absolute likelihood under the era model
        lp = M.seq_logp(era, arr.mode, toks) if toks else -5
        hi = M.stat(era, "lm_logp_median", -2.8)
        lo = calib["pop_lm_logp"].get(era, -4.3)
        comps["era_harmony"] = _clip01((lp - lo) / (hi - lo + 1e-9))
        notes["lm_logp"] = lp

    pr = parallel_rate(arr)
    comps["voice_leading"] = _clip01(math.exp(-pr / 0.25))   # 1 per 4 bars -> 0.37
    notes["parallel_rate"] = pr

    comps["cadences"] = _clip01(cadence_rate(arr) / 0.6)

    mel = arr.melody
    sync = melody_syncopation(arr.melody_plain or mel)     # ornaments are not syncopation
    # (Bach chorales dominate the Baroque corpus and almost never syncopate; floor the norm)
    p90 = max(0.12, M.stat(era, "sync_p90", 0.12))
    comps["rhythm"] = _clip01(1 - (sync - p90) / 0.25)
    notes["syncopation"] = sync

    if len(mel) > 3:
        ivs = np.abs(np.diff([m[2] for m in mel]))
        steps = float(np.mean((ivs > 0) & (ivs <= 2)))
        reps = float(np.mean(ivs == 0))
        big = float(np.mean(ivs > 7))
        comps["melody_style"] = _clip01(0.4 + steps - 1.2 * max(0, reps - 0.12) - 1.0 * max(0, big - 0.08))
        notes.update(step_frac=steps, repeat_frac=reps)
    else:
        comps["melody_style"] = 0.5

    li = loop_index(toks)
    comps["form"] = _clip01(1 - (li - 0.25) / 0.5)
    notes["loop_index"] = li

    comps["texture"] = _clip01((contrary_motion(arr) - 0.3) / 0.45)

    names = {p.name for p in arr.parts}
    anach = names & ERA_ANACHRONISMS.get(era, set())
    comps["instruments"] = 1.0 - 0.5 * min(1, len(anach))
    notes["anachronisms"] = sorted(anach)

    w = calib["c_weights"]
    tot = sum(w[k] for k in comps)
    C = sum(w[k] * v for k, v in comps.items()) / tot
    return {"score": float(C), "components": comps, "details": notes}


# ================================================================ retention
def skyline(arr: Arrangement, grid):
    """the pitch a listener hears on top at each grid point (melody-weighted)."""
    out = np.full(len(grid), np.nan)
    for p in arr.parts:
        if p.role in ("timpani", "harp"):
            continue
        boost = 0 if p.role == "melody" else -3       # loud melody is perceptually on top
        top = _sample_top(p, grid)
        cand = top + boost
        better = np.isnan(out) | (~np.isnan(cand) & (cand > out))
        out = np.where(better & ~np.isnan(cand), cand, out)
    # undo boost for pitch classes: re-sample real pitch of the winning part
    return out


def melody_retention(song: SongAnalysis, arr: Arrangement):
    if not song.melody:
        return 0.5, {}
    grid = np.arange(0, song.n_beats, 0.25)
    orig = np.full(len(grid), np.nan)
    on_o = []
    for n in song.melody:
        i0, i1 = int(round(n.start * 4)), int(round((n.start + n.dur) * 4))
        orig[max(0, i0):max(0, min(len(grid), i1))] = n.pitch
        on_o.append(n.start)
    top = skyline(arr, grid)
    has = ~np.isnan(orig)
    w = np.ones(len(grid))
    for s in song.sections:            # the hook (most energetic section) counts double
        if s["energy"] >= max(x["energy"] for x in song.sections) - 1e-9:
            w[int(s["start"] * 4):int(s["end"] * 4)] = 2.0
    ok = has & ~np.isnan(top)
    pc_match = ((np.mod(orig - np.nan_to_num(top), 12) == 0) & ok)
    pc = float((pc_match * w).sum() / max(1e-9, (has * w).sum()))
    # onset F1 (melody instrument only), tolerance 1/4 beat
    on_a = [m[0] for m in arr.melody if m[0] < song.n_beats]
    on_a = np.array(sorted(on_a))
    hit = 0
    for t in on_o:
        if len(on_a) and np.min(np.abs(on_a - t)) <= 0.26:
            hit += 1
    rec = hit / max(1, len(on_o))
    prec_hits = sum(1 for t in on_a if np.min(np.abs(np.array(on_o) - t)) <= 0.26) if on_o else 0
    prec = prec_hits / max(1, len(on_a))
    f1 = 2 * rec * prec / max(1e-9, rec + prec)
    return 0.65 * pc + 0.35 * f1, {"pitch_match": pc, "onset_f1": f1}


def _audio_beat_chroma(audio_path, arr: Arrangement):
    import librosa
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    C = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512, bins_per_octave=36)
    bt = np.array([arr.beat_to_sec(b) for b in range(arr.n_beats + 1)])
    fr = librosa.time_to_frames(bt, sr=sr, hop_length=512)
    fr = np.clip(fr, 0, C.shape[1] - 1)
    out = np.stack([C[:, fr[i]:max(fr[i] + 1, fr[i + 1])].mean(axis=1) for i in range(arr.n_beats)])
    return out


def _symbolic_beat_chroma(arr: Arrangement):
    out = np.zeros((arr.n_beats, 12))
    for p, (st, d, q, v) in arr.all_notes():
        if p.role == "timpani":
            continue
        a, b = int(max(0, math.floor(st))), int(min(arr.n_beats, math.ceil(st + d)))
        for i in range(a, b):
            ov = min(i + 1, st + d) - max(i, st)
            if ov > 0:
                out[i, q % 12] += ov * v / 100
    return out


def _ssm(X):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return Xn @ Xn.T


def retention(song: SongAnalysis, arr: Arrangement, audio_path=None, calib=None):
    calib = calib or load_calibration()
    comps, det = {}, {}
    comps["melody"], det["melody"] = melody_retention(song, arr)
    orig = np.asarray(song.beat_chroma, float)[:arr.n_beats].copy()
    if len(orig):
        # the listener hears the tune too: add the original melody to the reference chroma
        mc = np.zeros_like(orig)
        for m in song.melody:
            a0, b0 = int(max(0, math.floor(m.start))), int(min(len(orig), math.ceil(m.start + m.dur)))
            for i in range(a0, b0):
                mc[i, m.pitch % 12] += min(i + 1, m.start + m.dur) - max(i, m.start)
        orig = orig / (orig.max(axis=1, keepdims=True) + 1e-9) + 0.6 * mc
    if audio_path:
        try:
            outc = _audio_beat_chroma(audio_path, arr)
        except Exception:  # noqa: BLE001
            outc = _symbolic_beat_chroma(arr)
    else:
        outc = _symbolic_beat_chroma(arr)
    n = min(len(orig), len(outc))
    if n >= 4:
        o, a = orig[:n], outc[:n]
        on = o / (np.linalg.norm(o, axis=1, keepdims=True) + 1e-9)
        an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
        valid = (o.sum(axis=1) > 0) & (a.sum(axis=1) > 0)
        cos = float((on * an).sum(axis=1)[valid].mean()) if valid.any() else 0.5
        comps["harmony"] = _clip01((cos - 0.45) / 0.45)
        det["chroma_cos"] = cos
        bpb = song.beats_per_bar
        nb = n // bpb
        if nb >= 4:
            ob = o[:nb * bpb].reshape(nb, bpb, 12).mean(1)
            ab = a[:nb * bpb].reshape(nb, bpb, 12).mean(1)
            s1, s2 = _ssm(ob), _ssm(ab)
            iu = np.triu_indices(nb, 1)
            r = np.corrcoef(s1[iu], s2[iu])[0, 1] if np.std(s1[iu]) > 1e-6 and np.std(s2[iu]) > 1e-6 else 0.5
            comps["structure"] = _clip01((r + 0.1) / 0.9)
        else:
            comps["structure"] = 0.5
    else:
        comps["harmony"] = comps["structure"] = 0.5
    tf = arr.params.get("tempo_factor", 1.0)
    comps["tempo"] = float(math.exp(-4 * abs(math.log(tf))))
    w = calib["r_weights"]
    R = sum(w[k] * comps[k] for k in comps) / sum(w[k] for k in comps)
    return {"score": float(R), "components": comps, "details": det}


def total(C, R, w=0.5):
    s = w * C["score"] + (1 - w) * R["score"]
    return {"score": float(s), "loss": float(1 - s), "w": w}


# ================================================================ the period listener
PERSONAS = {
    "baroque": ("Kantor Gottfried", "a church musician in Leipzig, 1735"),
    "classical": ("Frau von Arnstein", "a Viennese concert-goer, 1788"),
    "romantic": ("M. Delacroix", "a Parisian salon critic, 1842"),
}


def period_review(C, R, arr: Arrangement):
    """A short written verdict from an imagined listener of the period, derived
    from the scores ."""
    era = arr.params["era"]
    who, where = PERSONAS[era]
    c = C["components"]
    d = C["details"]
    good, bad = [], []
    if c.get("cadences", 0) > 0.7:
        good.append("the phrases close properly, with honest cadences")
    else:
        bad.append("too many phrases simply stop instead of cadencing")
    if c.get("voice_leading", 0) > 0.8:
        good.append("the voices move with propriety")
    else:
        bad.append(f"I count parallel fifths and octaves (about {d.get('parallel_rate', 0):.2f} per bar)")
    if c.get("rhythm", 0) > 0.7:
        good.append("the melody sits squarely on the beat")
    else:
        bad.append("the tune stumbles off the beat in a most peculiar, restless way")
    if c.get("form", 0) < 0.5:
        bad.append("the same four harmonies go round and round like a mill wheel")
    if c.get("era_harmony", 0) > 0.7:
        good.append("the harmonies are such as any good master would write")
    elif c.get("era_harmony", 0) < 0.4:
        bad.append("some chord successions are foreign to my ear")
    if d.get("anachronisms"):
        bad.append(f"I do not know what a {d['anachronisms'][0].split('_')[0]} is doing here")
    m = R["components"]["melody"]
    tail = ("The melody is quite memorable – I find myself humming it."
            if m > 0.7 else "I struggled to follow a clear tune.")
    text = f"{who} ({where}): "
    if good:
        text += "I am pleased that " + "; ".join(good) + ". "
    if bad:
        text += "However, " + "; ".join(bad) + ". "
    text += tail
    return text
