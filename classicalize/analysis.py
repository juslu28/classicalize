"""Step 3 – musical analysis of the pop recording.

From the separated stems we extract everything the arranger needs, on a *beat*
time axis (so later comparisons are tempo-independent):

  tempo + beat grid + downbeats + meter
  key / mode
  lead melody (pYIN on the vocal stem -> quantised notes)
  chord per beat (CQT chroma of bass+other -> template match -> HMM smoothing)
  sections (verse/chorus...) + per-section energy
  beat-synchronous chroma & self-similarity (used by the retention loss)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import librosa
import numpy as np

from .theory import Chord, estimate_key, match_chord

SR = 22050
HOP = 512


@dataclass
class Note:
    start: float      # beats
    dur: float        # beats
    pitch: int        # MIDI
    vel: int = 80


@dataclass
class SongAnalysis:
    tempo: float
    beat_times: list
    beats_per_bar: int
    downbeat: int                   # index of first downbeat in beat_times
    tonic: int
    mode: str
    key_conf: float
    melody: list                    # [Note]
    chords: list                    # per beat: (root, qual) or None
    sections: list                  # [{start, end, label, energy}] in beats (bar aligned)
    beat_chroma: list               # n_beats x 12  (harmonic content)
    beat_energy: list               # n_beats
    duration: float
    title: str = ""
    meta: dict = field(default_factory=dict)

    # ---------- helpers
    @property
    def n_beats(self):
        return len(self.beat_times)

    def chord_objs(self):
        return [Chord(*c) if c else None for c in self.chords]

    def to_json(self):
        d = asdict(self)
        return json.dumps(d)

    @staticmethod
    def from_json(s):
        d = json.loads(s)
        d["melody"] = [Note(**n) for n in d["melody"]]
        d["chords"] = [tuple(c) if c else None for c in d["chords"]]
        return SongAnalysis(**d)

    def excerpt(self, start_beat: int, n_beats: int) -> "SongAnalysis":
        """Sub-analysis (used for fast optimisation on e.g. the first chorus)."""
        a, b = start_beat, min(self.n_beats, start_beat + n_beats)
        mel = [Note(n.start - a, min(n.dur, b - n.start), n.pitch, n.vel)
               for n in self.melody if a <= n.start < b]
        secs = []
        for s in self.sections:
            s0, s1 = max(a, s["start"]), min(b, s["end"])
            if s1 > s0:
                secs.append(dict(s, start=s0 - a, end=s1 - a))
        bt = self.beat_times[a:b]
        return SongAnalysis(self.tempo, [t - bt[0] for t in bt], self.beats_per_bar, 0,
                            self.tonic, self.mode, self.key_conf, mel, self.chords[a:b], secs,
                            self.beat_chroma[a:b], self.beat_energy[a:b], bt[-1] - bt[0], self.title,
                            dict(self.meta, excerpt=[a, b]))

    def chorus_start(self, n_beats):
        """Bar-aligned start of the most energetic section (≈ chorus)."""
        if not self.sections:
            return 0
        best = max(self.sections, key=lambda s: s["energy"])
        st = int(best["start"])
        return max(0, min(st, self.n_beats - n_beats))


# ----------------------------------------------------------------- loading
def _load(src, sr=SR):
    """path -> mono float32 at SR; arrays (already mono @ SR) pass straight through."""
    if isinstance(src, np.ndarray):
        return src.astype(np.float32)
    y, _ = librosa.load(src, sr=sr, mono=True, res_type="soxr_hq")
    return y


def _beat_track(y_mix, y_drums=None):
    src = y_drums if (y_drums is not None and np.abs(y_drums).mean() > 1e-4) else y_mix
    oenv = librosa.onset.onset_strength(y=src, sr=SR, hop_length=HOP, aggregate=np.median)
    tempo, beats = librosa.beat.beat_track(onset_envelope=oenv, sr=SR, hop_length=HOP,
                                           start_bpm=110, tightness=120, units="frames")
    tempo = float(np.atleast_1d(tempo)[0])
    bt = librosa.frames_to_time(beats, sr=SR, hop_length=HOP)
    # fold tempo octave into a musical range
    if tempo > 165 and len(bt) > 4:
        bt, tempo = bt[::2], tempo / 2
    if tempo < 62 and len(bt) > 4:
        mids = (bt[:-1] + bt[1:]) / 2
        bt = np.sort(np.concatenate([bt, mids]))
        tempo *= 2
    # extend the grid to cover the whole track (extrapolate with median period)
    dur = len(y_mix) / SR
    if len(bt) < 8:
        period = 60.0 / max(60.0, tempo or 100.0)
        bt = np.arange(0, dur, period)
    else:
        period = float(np.median(np.diff(bt)))
        head = np.arange(bt[0] - period, -1e-6, -period)[::-1]
        tail = np.arange(bt[-1] + period, dur, period)
        bt = np.concatenate([head, bt, tail])
    return 60.0 / float(np.median(np.diff(bt))), bt, oenv


def _beat_sync(feat, beat_frames, agg=np.median):
    """one column per beat: column i summarises [beat i, beat i+1)."""
    n = feat.shape[1]
    f = np.clip(np.asarray(beat_frames, int), 0, n - 1)
    bounds = list(f) + [n]
    cols = [agg(feat[:, a:max(a + 1, b)], axis=1) for a, b in zip(bounds[:-1], bounds[1:])]
    return np.stack(cols, axis=1)


# ----------------------------------------------------------------- melody
def extract_melody(y_voc, beat_times, energy_gate=0.02):
    # YIN (~10x faster than pYIN). On a separated vocal stem, voicing is decided
    # by loudness + pitch stability instead of pYIN's HMM.
    f0 = librosa.yin(y_voc, fmin=80, fmax=1100, sr=SR, frame_length=2048, hop_length=HOP)
    rms = librosa.feature.rms(y=y_voc, frame_length=2048, hop_length=HOP)[0]
    n = min(len(f0), len(rms))
    f0, rms = f0[:n], rms[:n]
    gate = rms > max(energy_gate * 2.5, 0.05) * (np.percentile(rms, 98) + 1e-9)
    m0 = librosa.hz_to_midi(f0)
    stable = np.zeros(n, bool)
    stable[1:-1] = (np.abs(m0[1:-1] - m0[:-2]) < 0.8) | (np.abs(m0[1:-1] - m0[2:]) < 0.8)
    voiced = gate & stable & (f0 > 82) & (f0 < 1090)
    t = librosa.times_like(f0, sr=SR, hop_length=HOP)
    midi = librosa.hz_to_midi(f0)
    ok = voiced & gate & ~np.isnan(midi)
    if ok.sum() < 10:
        return [], 0.0
    tuning = float(np.median((midi[ok] - np.round(midi[ok]))))
    midi = midi - tuning
    # median smooth inside voiced runs
    sm = midi.copy()
    for i in np.where(ok)[0]:
        lo, hi = max(0, i - 2), min(len(midi), i + 3)
        w = midi[lo:hi][ok[lo:hi]]
        sm[i] = np.median(w)
    notes = []
    cur, s0, vals = None, None, []
    for i in range(len(sm)):
        p = int(round(sm[i])) if ok[i] else None
        if cur is not None and p is not None and abs(sm[i] - cur) < 0.65:
            vals.append(i)
            continue
        if cur is not None and len(vals) * HOP / SR >= 0.07:
            notes.append((t[vals[0]], t[vals[-1]] + HOP / SR, int(round(np.median(sm[vals]))),
                          float(np.mean(rms[vals]))))
        cur, vals = (p if p is not None else None), ([i] if p is not None else [])
    if cur is not None and len(vals) * HOP / SR >= 0.07:
        notes.append((t[vals[0]], t[vals[-1]] + HOP / SR, int(round(np.median(sm[vals]))),
                      float(np.mean(rms[vals]))))
    if not notes:
        return [], tuning
    # seconds -> beats
    bt = np.asarray(beat_times)
    idx = np.arange(len(bt), dtype=float)
    period = float(np.median(np.diff(bt)))

    def s2b(x):
        if x <= bt[0]:
            return (x - bt[0]) / period
        if x >= bt[-1]:
            return len(bt) - 1 + (x - bt[-1]) / period
        return float(np.interp(x, bt, idx))

    rmax = max(n[3] for n in notes) + 1e-9
    out = []
    for a, b, p, r in notes:
        sb, eb = s2b(a), s2b(b)
        q0 = round(sb * 4) / 4
        q1 = max(q0 + 0.25, round(eb * 4) / 4)
        out.append(Note(q0, q1 - q0, p, int(50 + 60 * min(1.0, r / rmax))))
    # merge same-pitch fragments / resolve overlaps from quantisation
    merged = []
    for n in sorted(out, key=lambda n: n.start):
        if merged and n.start < merged[-1].start + merged[-1].dur - 1e-6:
            if n.pitch == merged[-1].pitch:
                m = merged[-1]
                m.dur = max(m.dur, n.start + n.dur - m.start)
                continue
            merged[-1].dur = max(0.25, n.start - merged[-1].start)
            if n.start <= merged[-1].start:
                continue
        elif merged and n.pitch == merged[-1].pitch and n.start - (merged[-1].start + merged[-1].dur) < 0.13:
            merged[-1].dur = n.start + n.dur - merged[-1].start
            continue
        merged.append(n)
    # octave-error repair
    ps = np.array([n.pitch for n in merged], float)
    for i, n in enumerate(merged):
        ctx = np.median(ps[max(0, i - 6):i + 7])
        while n.pitch - ctx > 9:
            n.pitch -= 12
        while ctx - n.pitch > 9:
            n.pitch += 12
    return [n for n in merged if n.start >= 0], tuning


# ----------------------------------------------------------------- chords
def _viterbi(emis, beat_strength, stay_hi=0.92, stay_lo=0.55):
    """emis: T x K log-emissions. Chord changes are cheaper on strong beats."""
    T, K = emis.shape
    dp = emis[0].copy()
    bp = np.zeros((T, K), int)
    for t in range(1, T):
        st = stay_lo if beat_strength[t] >= 2 else (0.75 if beat_strength[t] == 1 else stay_hi)
        ls, lc = np.log(st), np.log((1 - st) / (K - 1))
        best_prev = int(np.argmax(dp))
        cand_change = dp[best_prev] + lc
        cand_stay = dp + ls
        choose_stay = cand_stay >= cand_change
        bp[t] = np.where(choose_stay, np.arange(K), best_prev)
        dp = np.where(choose_stay, cand_stay, cand_change) + emis[t]
    path = [int(np.argmax(dp))]
    for t in range(T - 1, 0, -1):
        path.append(bp[t, path[-1]])
    return path[::-1]


def harmonic_chroma(y_harm, y_bass):
    C = librosa.feature.chroma_cqt(y=y_harm, sr=SR, hop_length=HOP, bins_per_octave=36)
    Cb = librosa.feature.chroma_cqt(y=y_bass, sr=SR, hop_length=HOP, fmin=librosa.note_to_hz("C1"),
                                    n_octaves=3) if y_bass is not None else None
    if Cb is not None:
        k = min(C.shape[1], Cb.shape[1])
        C, Cb = C[:, :k], Cb[:, :k]
    return C, Cb


def extract_chords(C, Cb, beat_frames, beat_strength):
    bc = _beat_sync(C, beat_frames)
    bb = _beat_sync(Cb, beat_frames) if Cb is not None else None
    T = bc.shape[1]
    names = None
    S = []
    for t in range(T):
        w = bc[:, t].copy()
        bass = None
        if bb is not None and bb[:, t].max() > 0.3:
            bass = int(np.argmax(bb[:, t]))
            w = w + 0.5 * bb[:, t]
        names, s = match_chord(w, bass_pc=bass, audio=True, return_scores=True)
        S.append(s)
    S = np.array(S)
    emis = 12.0 * (S - S.max(axis=1, keepdims=True))
    bs = np.zeros(T)
    k = min(T, len(beat_strength))
    bs[:k] = np.asarray(beat_strength)[:k]
    path = _viterbi(emis, bs)
    energy = bc.sum(axis=0)
    chords = []
    thr = 0.05 * np.percentile(energy, 95)
    for t, k in enumerate(path):
        chords.append((names[k].root, names[k].qual) if energy[t] > thr else None)
    return chords, bc


# ----------------------------------------------------------------- meter / sections
def _downbeat_and_meter(beat_frames, oenv_low, chords, bchroma=None):
    best = (-1e9, 4, 0)
    n = len(beat_frames) - 1
    if bchroma is not None:
        X = bchroma[:, :n].T
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        nov = np.r_[0.0, 1 - (X[1:] * X[:-1]).sum(axis=1)]
        changes = nov / (nov.max() + 1e-9)
    else:
        changes = np.array([1.0 if i > 0 and chords[i] != chords[i - 1] else 0.0 for i in range(n)])
    low = np.array([oenv_low[min(len(oenv_low) - 1, f)] for f in beat_frames[:n]])
    low = low / (low.max() + 1e-9)
    for bpb in (4, 3):
        for ph in range(bpb):
            sel = np.arange(ph, n, bpb)
            sc = changes[sel].mean() + 0.5 * low[sel].mean()
            sc -= (changes.mean() + 0.5 * low.mean())
            if bpb == 3:
                sc -= 0.08      # prior: most pop is 4/4
            if sc > best[0]:
                best = (sc, bpb, ph)
    return best[1], best[2]


def _sections(beat_feat, bpb, downbeat, energy):
    n = beat_feat.shape[1]
    bars = [(s, min(n, s + bpb)) for s in range(downbeat, n, bpb)]
    if downbeat > 0:
        bars = [(0, downbeat)] + bars
    if len(bars) < 4:
        return [{"start": 0, "end": n, "label": "A", "energy": float(np.mean(energy))}]
    bar_feat = np.stack([beat_feat[:, a:b].mean(axis=1) for a, b in bars], axis=1)
    k = int(np.clip(round(len(bars) / 8), 2, 12))
    bounds = librosa.segment.agglomerative(bar_feat, k)
    bounds = sorted(set([0] + list(bounds) + [len(bars)]))
    segs = []
    for i in range(len(bounds) - 1):
        a, b = bars[bounds[i]][0], bars[bounds[i + 1] - 1][1]
        if b > a:
            segs.append([a, b, bar_feat[:, bounds[i]:bounds[i + 1]].mean(axis=1)])
    # label by similarity clustering
    labels, protos = [], []
    for s in segs:
        v = s[2] / (np.linalg.norm(s[2]) + 1e-9)
        sims = [float(v @ p) for p in protos]
        if sims and max(sims) > 0.97:
            labels.append(chr(65 + int(np.argmax(sims))))
        else:
            protos.append(v)
            labels.append(chr(65 + len(protos) - 1))
    return [{"start": int(a), "end": int(b), "label": lab, "energy": float(np.mean(energy[a:b]))}
            for (a, b, _), lab in zip(segs, labels)]


# ----------------------------------------------------------------- main entry
def analyse(stems: dict, mix, title: str = "", progress=lambda *_: None, beat_times=None) -> SongAnalysis:
    """stems: {vocals,bass,other,drums} -> path or mono array @22.05 kHz; mix: path or array."""
    progress("Loading stems")
    y_mix = _load(mix)
    y_voc = _load(stems["vocals"])
    y_bass = _load(stems["bass"])
    y_oth = _load(stems["other"])
    y_drm = _load(stems["drums"])
    n = min(map(len, (y_mix, y_voc, y_bass, y_oth, y_drm)))
    y_mix, y_voc, y_bass, y_oth, y_drm = (x[:n] for x in (y_mix, y_voc, y_bass, y_oth, y_drm))

    progress("Tracking beats")
    if beat_times is not None and len(beat_times) > 8:
        bt = np.asarray([b for b in beat_times if b < n / SR], float)   # from the full-song beat tracker
        tempo = 60.0 / float(np.median(np.diff(bt)))
    else:
        tempo, bt, _ = _beat_track(y_mix, y_drm)
    beat_frames = librosa.time_to_frames(bt, sr=SR, hop_length=HOP)
    low = librosa.onset.onset_strength(y=y_drm + y_bass, sr=SR, hop_length=HOP, fmax=200)

    progress("Detecting chords")
    prelim = np.ones(len(bt))
    C, Cb = harmonic_chroma(y_oth + 0.7 * y_bass, y_bass)
    ch0, bc0 = extract_chords(C, Cb, beat_frames, prelim)
    bpb, db = _downbeat_and_meter(beat_frames, low, ch0, bc0)
    strength = np.array([2 if (i - db) % bpb == 0 else (1 if (i - db) % bpb == bpb // 2 else 0)
                         for i in range(len(bt) + 1)])
    chords, bchroma = extract_chords(C, Cb, beat_frames, strength)
    chords = chords[:len(bt)]

    progress("Transcribing the vocal melody")
    melody, tuning = extract_melody(y_voc, bt)

    progress("Finding key & sections")
    pc = bchroma.sum(axis=1)
    for nn in melody:
        pc[nn.pitch % 12] += 0.15 * nn.dur * bchroma.sum() / max(1, len(bt)) / 12
    tonic, mode, conf = estimate_key(pc)
    # relative major/minor tie-break: which tonic chord actually opens bars / sections?
    rel = ((tonic + 9) % 12, "minor") if mode == "major" else ((tonic + 3) % 12, "major")
    def _tonic_weight(t, m):
        q = "maj" if m == "major" else "min"
        w = 0.0
        for i, c in enumerate(chords):
            if c and c[0] == t and c[1].startswith(q[:3]):
                w += 2.0 if (i - db) % bpb == 0 else 0.5
        for sec_i in range(0, len(chords), bpb * 8):
            c = chords[sec_i] if sec_i < len(chords) else None
            if c and c[0] == t:
                w += 3.0
        return w
    if _tonic_weight(*rel) > 1.6 * _tonic_weight(tonic, mode):
        tonic, mode = rel
    rms = librosa.feature.rms(y=y_mix, hop_length=HOP)[0]
    benergy = _beat_sync(rms[None, :], beat_frames, agg=np.mean)[0][:len(bt)]
    mfcc = librosa.feature.mfcc(y=y_mix, sr=SR, hop_length=HOP, n_mfcc=13)
    feat = np.vstack([_beat_sync(librosa.util.normalize(
        C + 0.5 * Cb if Cb is not None else C, axis=0), beat_frames),
        0.3 * librosa.util.normalize(_beat_sync(mfcc, beat_frames), axis=1)])[:, :len(bt)]
    sections = _sections(feat, bpb, db, benergy)
    emax = max([x["energy"] for x in sections] or [1.0]) + 1e-9
    for x in sections:
        x["energy"] = round(x["energy"] / emax, 4)      # 1.0 = loudest section (≈ chorus)
    bchroma = bchroma[:, :len(bt)]
    return SongAnalysis(
        tempo=float(tempo), beat_times=[float(x) for x in bt], beats_per_bar=int(bpb), downbeat=int(db),
        tonic=int(tonic), mode=mode, key_conf=float(conf), melody=melody,
        chords=[tuple(c) if c else None for c in chords], sections=sections,
        beat_chroma=(bchroma.T / (bchroma.T.max(axis=1, keepdims=True) + 1e-9)).round(4).tolist(),
        beat_energy=(benergy / (benergy.max() + 1e-9)).round(4).tolist(),
        duration=n / SR, title=title, meta={"tuning": tuning},
    )
