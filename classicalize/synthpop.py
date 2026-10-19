"""Synthetic pop-song generator used for offline testing / training when no
internet is available. It writes songs in many modern genres from *real-world
common chord progressions* (the I–V–vi–IV "axis", vi–IV–I–V, Andalusian
cadence, 12-bar blues, J/K-pop "royal road", trap minor loops...) with
genre-typical, syncopated vocal-like melodies, verse/chorus forms and drums.

It returns both the ground-truth SongAnalysis and (optionally) rendered stems,
so the full audio pipeline can be tested *and* its analysis accuracy measured.
"""
from __future__ import annotations

import os
import random

import numpy as np

from .analysis import Note, SongAnalysis
from .theory import Chord, scale_pcs

PROGRESSIONS = {   # tokens relative to tonic, one per bar (or half bar if listed in pairs)
    "axis": ("major", ["0:maj", "7:maj", "9:min", "5:maj"]),
    "sensitive": ("major", ["9:min", "5:maj", "0:maj", "7:maj"]),
    "fifties": ("major", ["0:maj", "9:min", "5:maj", "7:maj"]),
    "royal_road": ("major", ["5:maj7", "7:maj", "4:min7", "9:min"]),
    "mixolydian": ("major", ["0:maj", "10:maj", "5:maj", "0:maj"]),
    "country": ("major", ["0:maj", "5:maj", "0:maj", "7:maj"]),
    "creep": ("major", ["0:maj", "4:maj", "5:maj", "5:min"]),
    "rnb": ("major", ["5:maj7", "4:min7", "2:min7", "0:maj7"]),
    "jazzpop": ("major", ["2:min7", "7:dom7", "0:maj7", "9:min7"]),
    "pachelbel": ("major", ["0:maj", "7:maj", "9:min", "4:min", "5:maj", "0:maj", "5:maj", "7:maj"]),
    "blues": ("major", ["0:dom7"] * 4 + ["5:dom7"] * 2 + ["0:dom7"] * 2 + ["7:dom7", "5:dom7", "0:dom7", "7:dom7"]),
    "minor_pop": ("minor", ["0:min", "8:maj", "3:maj", "10:maj"]),
    "andalusian": ("minor", ["0:min", "10:maj", "8:maj", "7:maj"]),
    "trap": ("minor", ["0:min", "5:min", "8:maj", "7:maj"]),
    "reggaeton": ("minor", ["0:min", "8:maj", "3:maj", "10:maj"]),
    "latin": ("minor", ["0:min", "5:min", "7:dom7", "0:min"]),
    "sad_ballad": ("major", ["0:maj", "4:min", "9:min", "5:maj", "2:min", "7:sus4", "7:maj", "7:maj"]),
    "edm": ("minor", ["0:min", "8:maj", "10:maj", "10:maj"]),
    "kpop": ("minor", ["8:maj", "10:maj", "0:min", "0:min"]),
    "doo_wop_minor": ("minor", ["0:min", "3:maj", "10:maj", "8:maj"]),
}
GENRES = {   # tempo range, syncopation, rep-note prob, progressions, notes per beat
    "pop": ((96, 128), 0.45, 0.25, ["axis", "sensitive", "fifties", "creep", "pachelbel"], 1.6),
    "ballad": ((66, 84), 0.25, 0.15, ["axis", "sad_ballad", "pachelbel", "fifties"], 1.0),
    "rock": ((110, 150), 0.35, 0.2, ["mixolydian", "axis", "country", "creep"], 1.4),
    "edm": ((122, 128), 0.5, 0.3, ["edm", "minor_pop", "sensitive"], 1.6),
    "hiphop": ((80, 98), 0.6, 0.55, ["trap", "minor_pop", "rnb"], 2.4),
    "rnb": ((68, 96), 0.55, 0.2, ["rnb", "jazzpop", "royal_road"], 1.6),
    "country": ((90, 120), 0.3, 0.25, ["country", "axis", "fifties"], 1.4),
    "kpop": ((100, 130), 0.5, 0.25, ["kpop", "royal_road", "minor_pop"], 1.8),
    "latin": ((90, 100), 0.6, 0.3, ["reggaeton", "latin", "andalusian"], 1.8),
    "blues": ((80, 120), 0.4, 0.2, ["blues"], 1.3),
    "indie": ((100, 140), 0.35, 0.2, ["creep", "mixolydian", "andalusian", "doo_wop_minor"], 1.3),
}

FORM = [("intro", 4, 0.35, False), ("verse", 8, 0.5, True), ("pre", 4, 0.65, True),
        ("chorus", 8, 1.0, True), ("verse", 8, 0.55, True), ("chorus", 8, 1.0, True), ("outro", 4, 0.4, False)]


def _melody_phrase(rng, chords_by_beat, scale, start, n_beats, lo, hi, sync, rep, density, hook=None):
    """contour-driven vocal-style phrase: chord tones on strong beats near a
    target contour (arch / descent / rise / wave), steps in between."""
    notes = []
    shape = rng.choice(["arch", "descend", "rise", "wave"])
    mid, amp = (lo + hi) / 2, (hi - lo) / 2

    def target(x):
        u = (x - start) / n_beats
        return mid + amp * {"arch": np.sin(np.pi * u) * 1.4 - 0.5, "descend": 0.8 - 1.4 * u,
                            "rise": -0.7 + 1.3 * u, "wave": 0.7 * np.sin(2 * np.pi * u * 1.5)}[shape]
    t = start
    p = int(target(start))
    while t < start + n_beats - 0.5:
        if notes and rng.random() < 0.10:            # breath
            t += rng.choice([0.5, 1.0])
            continue
        r = rng.random()
        dur = 0.25 if r < 0.22 * density else (0.5 if r < 0.72 else (1.0 if r < 0.92 else 2.0))
        if density < 1.2 and rng.random() < 0.5:
            dur *= 2
        onset = t
        if rng.random() < sync and dur >= 0.5:
            onset = max(start, t - 0.25 if rng.random() < 0.5 else t + 0.5)
        ch = chords_by_beat[min(len(chords_by_beat) - 1, int(onset))]
        strong = abs(onset % 1) < 1e-6
        tg = target(onset)
        if rng.random() < rep and notes:
            q = notes[-1].pitch
        elif strong or rng.random() < 0.4:
            cands = [x for x in range(lo, hi + 1) if x % 12 in ch.pcs]
            q = min(cands, key=lambda x: abs(x - tg) * 0.7 + abs(x - p) * 0.3 + rng.random() * 3)
        else:
            d = 1 if tg > p else -1
            q = p + d * rng.choice([1, 2, 2, 3])
            while q % 12 not in scale:
                q += d
        q = int(np.clip(q, lo, hi))
        notes.append(Note(onset, dur * 0.95, q, 80))
        p = q
        t = onset + dur
    return notes


def make_song(seed: int, genre: str | None = None):
    rng = random.Random(seed)
    genre = genre or rng.choice(list(GENRES))
    (t0, t1), sync, rep, progs, dens = GENRES[genre]
    prog_name = rng.choice(progs)
    mode, prog = PROGRESSIONS[prog_name]
    tonic = rng.randrange(12)
    tempo = rng.uniform(t0, t1)
    bpb = 4
    scale = scale_pcs(tonic, mode)
    # second (chorus) progression sometimes differs
    chorus_prog = prog if rng.random() < 0.6 else PROGRESSIONS[rng.choice(progs)][1]
    if PROGRESSIONS.get(prog_name)[0] != mode:
        chorus_prog = prog
    beats_chords, sections, melody = [], [], []
    hook_notes = None
    lo, hi = (55, 74) if rng.random() < 0.5 else (60, 79)
    beat = 0
    for name, bars, energy, sing in FORM:
        pr = chorus_prog if name == "chorus" else prog
        chunk = []
        for b in range(bars):
            tok = pr[b % len(pr)]
            c = Chord.from_token(tok, tonic)
            chunk += [c] * bpb
        beats_chords += chunk
        sections.append({"start": beat, "end": beat + bars * bpb, "label": name, "energy": energy})
        if sing:
            if name == "chorus" and hook_notes is not None:
                for n in hook_notes:
                    melody.append(Note(n.start + beat, n.dur, n.pitch, n.vel))
            else:
                ph = []
                for k in range(0, bars, 4):          # 4-bar phrases, 2nd one varies the 1st
                    sub = _melody_phrase(rng, beats_chords, scale, beat + k * bpb, 4 * bpb,
                                         lo + (5 if name == "chorus" else 0), hi + (3 if name == "chorus" else 0),
                                         sync, rep, dens)
                    ph += sub
                melody += ph
                if name == "chorus":
                    hook_notes = [Note(n.start - beat, n.dur, n.pitch, n.vel) for n in ph]
        beat += bars * bpb
    n_beats = beat
    period = 60.0 / tempo
    beat_times = [i * period for i in range(n_beats)]
    chroma = []
    for c in beats_chords:
        v = np.zeros(12)
        for k, i in enumerate(c.pcs):
            v[i] = 1.0 if k < 3 else 0.6
        v[c.root] += 0.3
        chroma.append((v / v.max()).round(3).tolist())
    energy = []
    for s in sections:
        energy += [s["energy"]] * (s["end"] - s["start"])
    song = SongAnalysis(tempo=tempo, beat_times=beat_times, beats_per_bar=bpb, downbeat=0, tonic=tonic, mode=mode,
                        key_conf=1.0, melody=sorted(melody, key=lambda n: n.start),
                        chords=[(c.root, c.qual) for c in beats_chords], sections=sections, beat_chroma=chroma,
                        beat_energy=energy, duration=n_beats * period, title=f"synthetic {genre} #{seed} ({prog_name})",
                        meta={"genre": genre, "progression": prog_name, "synthetic": True})
    return song


# --------------------------------------------------------------- pop audio render (stems)
def render_pop_stems(song: SongAnalysis, out_dir: str, sf2: str | None = None, seed=0):
    """Render the synthetic song as a pop production: vocal-ish lead, synth bass,
    e-piano/pad chords, drum kit. Returns (mix_path, stems dict)."""
    import pretty_midi
    import soundfile as sf

    from .render import find_soundfont
    sf2 = sf2 or find_soundfont()
    rng = random.Random(seed)
    os.makedirs(out_dir, exist_ok=True)
    period = 60 / song.tempo
    T = lambda b: b * period  # noqa: E731
    stems = {}
    genre = song.meta.get("genre", "pop")
    # vocals
    voc = pretty_midi.Instrument(program=rng.choice([53, 54, 85, 82]))
    for n in song.melody:
        voc.notes.append(pretty_midi.Note(95, n.pitch, T(n.start), T(n.start + n.dur)))
    # bass
    bas = pretty_midi.Instrument(program=rng.choice([38, 33, 39, 34]))
    oth = pretty_midi.Instrument(program=rng.choice([4, 5, 89, 90, 0, 29, 25]))
    for i, ch in enumerate(song.chords):
        c = Chord(*ch)
        root = 36 + (c.root - 36) % 12
        if genre in ("hiphop", "edm", "latin"):
            for k in (0, 0.75, 2, 2.5) if i % 4 == 0 else ():
                bas.notes.append(pretty_midi.Note(100, root, T(i + k), T(i + k + 0.4)))
        else:
            for k in (0, 0.5):
                bas.notes.append(pretty_midi.Note(95, root, T(i + k), T(i + k + 0.45)))
        if i % 2 == 0:
            for q in c.pcs:
                pitch = 60 + (q - 60) % 12
                oth.notes.append(pretty_midi.Note(70, pitch, T(i), T(i + 1.9)))
    drm = pretty_midi.Instrument(program=0, is_drum=True)
    for s in song.sections:
        if s["label"] == "intro":
            continue
        for b in range(s["start"], s["end"]):
            if genre == "latin":
                pat = [(0, 36), (0.75, 38), (1.5, 38)] if b % 2 == 0 else [(0, 36), (0.5, 38)]
            elif genre == "hiphop":
                pat = [(0, 36)] if b % 4 in (0,) else ([(0, 38)] if b % 4 == 2 else [(0.5, 36)])
            else:
                pat = [(0, 36 if b % 2 == 0 else 38)]
            for o, k in pat + [(0, 42), (0.5, 42)]:
                drm.notes.append(pretty_midi.Note(100 if k != 42 else 60, k, T(b + o), T(b + o + 0.1)))
    total = int((song.n_beats * period + 2) * 44100)
    mix = np.zeros(total)
    for name, inst, g in (("vocals", voc, 0.9), ("bass", bas, 0.8), ("other", oth, 0.6), ("drums", drm, 0.7)):
        y = inst.fluidsynth(fs=44100, sf2_path=sf2) if inst.notes else np.zeros(1000)
        y = y[:total] * g / (np.abs(y).max() + 1e-9)
        buf = np.zeros(total)
        buf[:len(y)] = y
        p = os.path.join(out_dir, f"{name}.wav")
        sf.write(p, buf.astype(np.float32), 44100)
        stems[name] = p
        mix += buf
    mix /= np.abs(mix).max() + 1e-9
    mp = os.path.join(out_dir, "original.wav")
    sf.write(mp, (0.9 * mix).astype(np.float32), 44100)
    return mp, stems
