"""The Original ↔ Classical slider.

s = 0   the original clip, note for note, played by classical instruments:
        every transcribed vocal, bass, synth/guitar/keys note at its original
        time, and the drums turned into orchestral percussion.
s = 1   a classical piece in the chosen composer's style that keeps only a few
        tidbits of the song: the skeleton of the tune, re-harmonised, with
        period accompaniment and ornaments.

In between, each layer switches over from the original transcription to the
classical arrangement at its own point, so the change is gradual:

  percussion            full orchestral kit  -> timpani on downbeats -> none     (0.25, 0.5)
  accompaniment         original parts       -> period textures                  (0.4)
  bass line             original bass        -> period bass line                 (0.4)
  melody                original timing      -> on the beat, ornamented          (0.3)
                                             -> skeleton + new passagework       (0.75+)
  harmony / ornaments / counterpoint scale up continuously with s.
Timing always follows the original beats, so the output is exactly as long
as the original clip.
"""
from __future__ import annotations

import numpy as np

from .arrange import INSTR, Arrangement, ArrangeParams, Part, arrange, ensemble_layout
from .composers import COMPOSERS

T_MELODY, T_ACCOMP, T_BASS, T_PERC_FULL, T_PERC_SOME = 0.3, 0.4, 0.4, 0.25, 0.5


def slider_params(p: ArrangeParams, s: float) -> ArrangeParams:
    """Scale a composer preset by the slider: the closer to 'original', the
    fewer classical liberties (re-harmonisation, ornaments, rhythm changes...)."""
    d = p.to_dict()
    f = min(1.0, max(0.0, (s - 0.3) / 0.7))
    d["reharm"] = min(1.0, d["reharm"] * 1.5 * f)
    d["ornament"] = d["ornament"] * f
    d["straighten"] = min(1.0, d["straighten"] * min(1.0, s * 1.6))
    d["counterpoint"] = d["counterpoint"] * max(0.25, f)
    d["develop"] = min(1.0, max(0.0, (s - 0.72) / 0.28))
    d["tempo_factor"] = 1.0
    d["rubato"] = 0.0
    d["add_ending"] = False
    d["exact_timing"] = True
    return ArrangeParams(**d)


def _s2b_factory(beat_times):
    bt = np.asarray(beat_times, float)
    bt = bt - bt[0]
    idx = np.arange(len(bt), dtype=float)
    per = float(np.median(np.diff(bt))) if len(bt) > 1 else 0.5

    def s2b(t):
        if t <= 0:
            return t / per
        if t >= bt[-1]:
            return len(bt) - 1 + (t - bt[-1]) / per
        return float(np.interp(t, bt, idx))
    return s2b


def _vel(a, base=45, span=70):
    return int(np.clip(base + span * a, 25, 118))


def _octave_shift(pitches, centre):
    if not pitches:
        return 0
    return int(12 * round((centre - float(np.median(pitches))) / 12))


def _other_target(ens, era, pitch, dur_s):
    """which classical instrument plays a transcribed accompaniment note."""
    if ens == "piano":
        return ("melody", "harpsichord_rh" if era == "baroque" else "piano_rh") if pitch >= 60 else \
               ("keys_lh", "harpsichord_lh" if era == "baroque" else "piano_lh")
    if ens == "string_quartet":
        if dur_s < 0.16:
            return ("inner_lo", "pizzicato")
        return ("inner_hi", "violin") if pitch >= 67 else (("inner_lo", "viola") if pitch >= 52 else ("bass", "cello"))
    if ens == "chamber":
        if dur_s < 0.16:
            return ("continuo", "harpsichord_rh" if era == "baroque" else "piano_rh") if era != "romantic" else \
                   ("keys_lh", "piano_lh")
        return ("inner_hi", "violin") if pitch >= 67 else (("inner_lo", "viola") if pitch >= 52 else ("bass", "cello"))
    # orchestra
    if dur_s < 0.16:
        return ("harp", "harp") if era == "romantic" else ("inner_lo", "pizzicato")
    if dur_s >= 1.2 and 53 <= pitch < 67:
        return ("brass", "horn")
    if pitch >= 72:
        return ("inner_hi", "violins2")
    if pitch >= 64 and dur_s >= 0.8:
        return ("winds", "clarinet" if era != "baroque" else "oboe")
    if pitch >= 55:
        return ("inner_lo", "violas")
    return ("counter", "cellos")


def build(song, tnotes, hits, p: ArrangeParams, s: float, progress=lambda *_: None):
    """-> (Arrangement, info dict describing what the slider did)."""
    pb = slider_params(p, s)
    B = arrange(song, pb)                               # the classical arrangement (exact timing)
    era, ens = pb.era, pb.ensemble
    layout = ensemble_layout(era, ens)
    s2b = _s2b_factory(song.beat_times)
    parts = {}

    def part(role, name):
        k = f"{role}:{name}"
        if k not in parts:
            parts[k] = Part(name, INSTR[name][0], role)
        return parts[k]

    used = {"melody": "classical", "bass": "classical", "accompaniment": "classical", "percussion": "none"}

    # ---------------- melody
    if s < T_MELODY:
        voc = [n for n in tnotes if n.role == "vocals"]
        lead = layout["melody"][0]
        sh = _octave_shift([n.pitch for n in voc], INSTR[lead][3])
        for n in voc:
            b0, b1 = s2b(n.start), s2b(n.end)
            part("melody", lead).add(b0, max(0.1, b1 - b0), n.pitch + sh, _vel(n.amp, 60, 50))
        melody_beats = [(s2b(n.start), s2b(n.end) - s2b(n.start), n.pitch + sh) for n in voc]
        used["melody"] = "original"
    else:
        for pt in B.parts:
            if pt.role == "melody":
                parts[f"melody:{pt.name}"] = pt
        melody_beats = B.melody
        used["melody"] = "developed" if pb.develop > 0 else "classical"

    # ---------------- bass
    if s < T_BASS:
        bass = [n for n in tnotes if n.role == "bass"]
        targets = {"piano": ["harpsichord_lh" if era == "baroque" else "piano_lh"], "string_quartet": ["cello"],
                   "chamber": ["cello"], "orchestra": ["contrabass", "cellos"]}[ens]
        role = "keys_lh" if ens == "piano" else "bass"
        for n in bass:
            b0, b1 = s2b(n.start), s2b(n.end)
            for k, name in enumerate(targets):
                q = n.pitch + (12 if (k == 1 or name == "cello") and n.pitch < 43 else 0)
                part(role, name).add(b0, max(0.1, b1 - b0), q, _vel(n.amp, 55, 45) - 6 * k)
        used["bass"] = "original"
    else:
        for pt in B.parts:
            if pt.role == "bass" or (pt.role == "keys_lh" and ens == "piano"):
                parts[f"{pt.role}:{pt.name}"] = pt

    # ---------------- accompaniment (everything else that is pitched)
    if s < T_ACCOMP:
        q16 = s >= 0.2           # tidy the timing a little as we move away from the original
        for n in tnotes:
            if n.role != "other":
                continue
            b0, b1 = s2b(n.start), s2b(n.end)
            if q16:
                b0 = round(b0 * 4) / 4
                b1 = max(b0 + 0.25, round(b1 * 4) / 4)
            role, name = _other_target(ens, era, n.pitch, n.end - n.start)
            part(role, name).add(b0, max(0.1, b1 - b0), n.pitch, _vel(n.amp, 40, 55))
        used["accompaniment"] = "original"
    else:
        for pt in B.parts:
            if pt.role in ("melody", "bass", "timpani") or (pt.role == "keys_lh" and ens == "piano"):
                continue
            parts[f"{pt.role}:{pt.name}"] = pt

    # ---------------- percussion (orchestra only)
    if ens == "orchestra" and s < T_PERC_SOME:
        tonic = song.tonic
        tim = part("timpani", "timpani")
        slots = B.slots
        for t, kind, strength in hits:
            b = s2b(t)
            if s >= T_PERC_FULL and not (kind == "kick" and abs(b - round(b)) < 0.1 and
                                         (round(b) - song.downbeat) % song.beats_per_bar == 0):
                continue
            if kind == "kick":
                root = next((sl for sl in slots if sl["start"] <= b < sl["end"]), slots[0] if slots else None)
                pc = tonic
                if root:
                    from .theory import Chord
                    r = Chord.from_token(root["token"], tonic).root
                    pc = r if r in (tonic, (tonic + 7) % 12) else tonic
                q = 40 + (pc - 40) % 12
                tim.add(b, 0.4, q, _vel(strength, 55, 50))
            elif s < T_PERC_FULL:
                kit = part("perc", "perc")
                if kind == "snare":
                    kit.add(b, 0.2, 38, _vel(strength, 40, 50))      # concert snare drum
                elif strength > 0.6:
                    kit.add(b, 1.0, 49, _vel(strength, 30, 40))      # suspended cymbal (strong hits only)
        used["percussion"] = "full" if s < T_PERC_FULL else "timpani"
    elif ens == "orchestra" and s >= T_PERC_SOME:
        for pt in B.parts:
            if pt.role == "timpani":
                parts["timpani:timpani"] = pt
        used["percussion"] = "cadences" if any(pt.role == "timpani" for pt in B.parts) else "none"

    arr = Arrangement(params={**pb.to_dict(), "slider": s}, parts=list(parts.values()), beat_sec=B.beat_sec,
                      slots=B.slots, melody=melody_beats, n_beats=B.n_beats, tonic=B.tonic, mode=B.mode,
                      beats_per_bar=B.beats_per_bar, sections=B.sections,
                      melody_plain=B.melody_plain if s >= T_MELODY else melody_beats)
    info = {"slider": s, "layers": used, "params": pb.to_dict(),
            "thresholds": {"melody": T_MELODY, "accompaniment": T_ACCOMP, "bass": T_BASS,
                           "percussion_full": T_PERC_FULL, "percussion_some": T_PERC_SOME},
            "n_original_notes": len(tnotes), "n_drum_hits": len(hits)}
    _ = COMPOSERS
    return arr, info
