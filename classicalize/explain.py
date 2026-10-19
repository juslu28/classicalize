"""Build the step-by-step "Details" walkthrough for a finished arrangement."""
from __future__ import annotations

from collections import Counter

import numpy as np

from . import judge
from .analysis import SongAnalysis
from .arrange import Arrangement
from .composers import COMPOSERS, ENSEMBLES, ERAS, INSTRUMENT_NAMES
from .theory import PC_NAMES, Chord, roman

NOTE = lambda m: f"{PC_NAMES[int(m) % 12]}{int(m) // 12 - 1}"  # noqa: E731
ROLE_ORDER = ["melody", "inner_hi", "inner_lo", "counter", "bass", "keys_lh", "continuo", "winds", "brass",
              "harp", "timpani"]


def _loop(song: SongAnalysis):
    toks = []
    for c in song.chord_objs():
        if c is None:
            continue
        t = c.token(song.tonic)
        if not toks or toks[-1] != t:
            toks.append(t)
    if len(toks) < 4:
        return None
    grams = Counter(tuple(toks[i:i + 4]) for i in range(len(toks) - 3))
    g, n = grams.most_common(1)[0]
    return {"roman": [roman(t) for t in g], "names": [Chord.from_token(t, song.tonic).name() for t in g],
            "share": n / max(1, len(toks) - 3)}


def _sec(song, beat):
    bt = song.beat_times
    if not bt:
        return 0.0
    b = int(min(max(beat, 0), len(bt) - 1))
    return float(bt[b])


LAYER_TEXT = {
    "melody": {"original": "The vocal line is played note for note, with the singer's own timing and phrasing.",
               "classical": "The vocal line is kept, but placed squarely on the beat and given period ornaments.",
               "developed": "Only the skeleton of the tune is kept (its strong-beat notes). The notes in between "
                            "are re-composed as classical scale passages."},
    "bass": {"original": "The original bass line is played exactly, by the double basses and cellos.",
             "classical": "The bass is re-written as a period bass line (walking, Alberti or sustained)."},
    "accompaniment": {"original": "Every other part (synths, guitars, keys, pads) is transcribed note for note "
                                  "and given to classical instruments by register: violins, violas, cellos, horns, "
                                  "clarinets, plus pizzicato or harp for short plucked notes.",
                      "classical": "The original backing is replaced by the composer's own accompaniment "
                                   "textures over (re-)harmonised chords."},
    "percussion": {"full": "The drum kit becomes orchestral percussion: kick → timpani (tuned to the chord), "
                           "snare → concert snare drum, crashes → suspended cymbal.",
                   "timpani": "Only the timpani remain, on the downbeat kicks.",
                   "cadences": "No drums. Timpani only mark the big cadences, as in a real score.",
                   "none": "No drums. The rhythm comes from the accompaniment, as in real classical music."},
}


def slider_step(binfo, ens, forced):
    if not binfo:
        return None
    s = binfo["slider"]
    lay = binfo["layers"]
    items = [{"layer": k, "state": v, "text": LAYER_TEXT[k].get(v, "")} for k, v in lay.items()]
    pos = ("played exactly as the original" if s < 0.2 else "close to the original" if s < 0.45 else
           "half way" if s < 0.6 else "mostly classical" if s < 0.8 else "a classical piece with tidbits of the song")
    p = binfo["params"]
    return {"value": s, "label": pos, "layers": items, "forced_orchestra": forced,
            "knobs": {k: round(p[k], 2) for k in ("reharm", "ornament", "straighten", "counterpoint", "develop")},
            "n_notes": binfo.get("n_original_notes"), "n_hits": binfo.get("n_drum_hits"),
            "thresholds": binfo.get("thresholds")}


def build(song: SongAnalysis, arr: Arrangement, info: dict, C: dict, R: dict, total: dict, review: str,
          meta: dict, blend_info=None) -> dict:
    p = arr.params
    era, cid, ens = p["era"], p["composer"], p["ensemble"]
    comp = COMPOSERS[cid]
    f = info["features"]
    uc = info.get("user_choice", {})
    bpb = song.beats_per_bar

    # ------------------------------------------------ 1. the original song
    labels = {}
    form = []
    emax = max([s["energy"] for s in song.sections] or [1])
    for s in song.sections:
        lab = s["label"]
        if lab not in labels:
            labels[lab] = len(labels)
        form.append({"label": lab, "start": round(_sec(song, s["start"]), 1), "end": round(_sec(song, s["end"] - 1), 1),
                     "energy": round(s["energy"] / emax, 2)})
    chorus = max(song.sections, key=lambda s: s["energy"])["label"] if song.sections else None
    pitches = [n.pitch for n in song.melody]
    loop = _loop(song)
    song_step = {
        "title": meta.get("title") or song.title, "url": meta.get("url"), "views": meta.get("views"),
        "key": f"{PC_NAMES[song.tonic]} {song.mode}", "tempo": round(song.tempo), "meter": f"{bpb}/4",
        "duration": round(song.duration), "form": form, "chorus_label": chorus, "loop": loop,
        "melody": {"low": NOTE(min(pitches)) if pitches else None, "high": NOTE(max(pitches)) if pitches else None,
                   "range": int(f["range"]), "notes_per_beat": round(f["notes_per_beat"], 2),
                   "syncopation": round(100 * f["syncopation"]), "n_notes": len(pitches)},
    }

    # ------------------------------------------------ 2. the era
    per_era = info.get("per_era", {})
    ranked_eras = sorted(per_era.items(), key=lambda kv: -kv[1]["adj"])
    why_era = []
    if uc.get("era", "auto") != "auto":
        why_era.append(f"You chose the {ERAS[era]['name']} era, so only {ERAS[era]['name']} composers were tried.")
    elif uc.get("composer", "auto") != "auto":
        why_era.append(f"You chose {comp['short']}, who belongs to the {ERAS[era]['name']} era.")
    else:
        if len(ranked_eras) > 1:
            parts = ", ".join(f"{ERAS[e]['name']} {v['adj']:.3f}" for e, v in ranked_eras)
            why_era.append(f"All three eras were tried. {ERAS[era]['name']} scored best ({parts}). The score combines "
                           f"how classical it sounds, how much of the song survives, and how well the song suits the composer.")
    sync = f["syncopation"]
    if era == "baroque":
        if f["notes_per_beat"] > 0.9:
            why_era.append("The melody is busy and runs in steady note values, which suits Baroque 'motor' rhythm "
                           "and running counterpoint.")
        if sync < 0.15:
            why_era.append("The vocal line mostly sits on the beat, so it survives Baroque's strict pulse with "
                           "little change.")
        if f["energy_var"] < 0.15:
            why_era.append("The energy stays fairly even from verse to chorus, matching Baroque's one-mood-per-piece style.")
    elif era == "classical":
        why_era.append("The song is built from regular 4- and 8-bar phrases. That matches the Classical era's "
                       "balanced question-and-answer phrasing, so cadences fit naturally at phrase ends.")
        if f["loop"] > 0.5:
            why_era.append("Its repeating chord loop maps neatly onto Classical tonic-dominant harmony "
                           "(I, IV, V, vi), so few chords need changing.")
    else:
        if sync > 0.15:
            why_era.append(f"The vocal line is quite syncopated ({sync:.0%} of notes push across the beat). "
                           "Romantic rubato and freer phrasing absorb that better than a strict Baroque or Classical pulse.")
        if f["range"] >= 12:
            why_era.append("The melody spans a wide range, like the long, singing Romantic melodies.")
        if f["energy_var"] >= 0.15:
            why_era.append("There's a big jump in energy into the chorus, and Romantic dynamics (pp to ff) can "
                           "follow that swell.")
        if f["non_diatonic"] > 0.1:
            why_era.append("Some chords fall outside the plain key, which Romantic harmony welcomes.")
    era_step = {"id": era, **ERAS[era], "why": why_era,
                "scores": [{"era": e, "name": ERAS[e]["name"], "adj": v["adj"], "C": v["C"], "R": v["R"],
                            "composer": COMPOSERS[v["composer"]]["short"]} for e, v in ranked_eras]}

    # ------------------------------------------------ 3. the composer
    rank = info.get("composer_rank", [])
    me = next((r for r in rank if r["id"] == cid), rank[0] if rank else {"prob": 1, "reasons": [], "fit": 0.5})
    applied = []
    k = p
    if era == "baroque":
        applied.append("A walking continuo bass moves under every beat" if k["density"] >= 0.6 else
                       "The bass moves in steady quarter notes, like a Baroque continuo")
    elif era == "classical":
        if ens == "piano":
            applied.append("Alberti bass in the left hand: the low–high–middle–high broken chord heard all through Mozart")
        else:
            applied.append("Inner strings murmur repeated eighth notes under the tune, the classic Viennese accompaniment")
    else:
        if ens in ("piano", "chamber"):
            applied.append("Wide left-hand arpeggios spanning two octaves, held by the sustain pedal")
        if ens == "orchestra":
            applied.append("Pulsing string chords, horns and harp swell under the melody in the loud sections")
    if k["counterpoint"] >= 0.6:
        applied.append(f"An independent counter-melody weaves around the tune (counterpoint {k['counterpoint']:.0%})")
    if k["ornament"] >= 0.45:
        orn = {"baroque": "cadential trills, mordents and decorated long notes",
               "classical": "graceful appoggiaturas and cadential trills",
               "romantic": "ornamental turns leading into big melody notes"}[era]
        applied.append(f"Ornaments in the style of the period: {orn}")
    if k["rubato"] >= 0.3 and era != "baroque":
        applied.append(f"Rubato: the tempo relaxes at the end of each phrase (rubato {k['rubato']:.0%})")
    if k["dynamics"] >= 0.8:
        applied.append("A very wide dynamic range: quiet verses and full-force choruses")
    elif era == "baroque":
        applied.append("Terraced dynamics: whole sections are loud or soft, Baroque-style")
    if k["reharm"] >= 0.55:
        applied.append("Re-harmonised more freely toward the composer's harmonic language")
    composer_step = {
        "id": cid, **{x: comp[x] for x in ("name", "short", "years", "place", "works", "bio", "signature")},
        "prob": me.get("prob", 1.0), "fit": me.get("fit", 0.5),
        "reasons": me.get("reasons", []) or ["its overall character scored highest for this style"],
        "applied": applied, "chosen_by_user": uc.get("composer", "auto") != "auto",
        "top": [{"id": r["id"], "name": COMPOSERS[r["id"]]["short"], "era": COMPOSERS[r["id"]]["era"],
                 "prob": r["prob"], "fit": r["fit"], "reasons": r["reasons"][:2]} for r in rank[:5]],
    }

    # ------------------------------------------------ 4. the instruments
    parts = []
    for role in ROLE_ORDER:
        for pt in arr.parts:
            if pt.role != role or not pt.notes:
                continue
            ps = [n[2] for n in pt.notes]
            doubling = any(q["role"] == role for q in parts)
            parts.append({"instrument": INSTRUMENT_NAMES.get(pt.name, pt.name), "role": role,
                          "what": ("Doubles the line above for extra weight and colour" if doubling else
                                   ENSEMBLES[ens]["parts"].get(role, "")),
                          "range": f"{NOTE(min(ps))}–{NOTE(max(ps))}", "notes": len(ps)})
    per_ens = info.get("per_ensemble", {})
    alt = sorted(({"id": e, "name": ENSEMBLES[e]["name"], **v} for e, v in per_ens.items()),
                 key=lambda x: -x["adj"])
    why_ens = []
    if uc.get("ensemble", "auto") != "auto":
        why_ens.append(f"You chose the {ENSEMBLES[ens]['name'].lower()}.")
    elif info.get("forced_orchestra"):
        why_ens.append("Near the 'original' end of the slider the full orchestra is used. It is the only classical "
                       "ensemble with enough colours to cover the vocals, synths, bass and drums at once.")
    else:
        pref = comp["ensembles"].get(ens, 0.5)
        fav = max(comp["ensembles"], key=comp["ensembles"].get)
        if ens == fav:
            why_ens.append(f"It is {comp['short']}'s signature setting.")
        elif pref >= 0.8:
            why_ens.append(f"It is one of {comp['short']}'s favourite settings.")
        if len(alt) > 1:
            why_ens.append("In this composer's style it scored best of the four ensembles: " +
                           ", ".join(f"{a['name']} {a['adj']:.3f}" for a in alt) + ".")
    why_ens.append(f"Why it works: {ENSEMBLES[ens]['why']}.")
    lead = next((x for x in parts if x["role"] == "melody"), None)
    if lead:
        why_ens.append(f"The {lead['instrument'].lower()} carries the vocal melody ({lead['range']}). It was moved "
                       f"by octaves into the instrument's most singing register, with the key unchanged.")
    ens_step = {"id": ens, **{x: ENSEMBLES[ens][x] for x in ("name", "summary")}, "why": why_ens,
                "parts": parts, "alternatives": alt}

    # ------------------------------------------------ 5. what changed
    slots = arr.slots
    kept = [s for s in slots if s.get("orig") and s["orig"] == s["chord"]]
    changed = [(s["orig"], s["chord"], round(_sec(song, s["start"]), 1)) for s in slots
               if s.get("orig") and s["orig"] != s["chord"]]
    cads = 0
    for i, s in enumerate(slots[1:], 1):
        if s["phrase_end"] and s["token"] in ("0:maj", "0:min") and slots[i - 1]["token"] in ("7:maj", "7:dom7"):
            cads += 1
    sync_after = judge.melody_syncopation(arr.melody_plain or arr.melody)
    if blend_info and blend_info["layers"].get("accompaniment") == "original":
        kept, changed = [s_ for s_ in slots if s_.get("orig")], []
    changes = {
        "chords_total": len(slots), "chords_kept": len(kept),
        "kept_pct": round(100 * len(kept) / max(1, len([s for s in slots if s.get("orig")]))),
        "examples": [{"from": a, "to": b, "at": t} for a, b, t in changed[:8]],
        "cadences": cads, "sync_before": round(100 * f["syncopation"]), "sync_after": round(100 * sync_after),
        "melody_notes_before": len(song.melody), "melody_notes_after": len(arr.melody),
        "tempo_before": round(song.tempo), "tempo_after": round(song.tempo),
        "ending": f"It follows the original beat by beat and lasts exactly as long as the clip ({song.duration:.1f}s).",
        "dropped": "The drum kit, synths and electric bass are gone. Their rhythmic drive now comes "
                   "from the accompaniment patterns.",
    }

    win = meta.get("window") or {}
    song_step["clip"] = {"start": round(win.get("start", 0), 1), "end": round(win.get("end", song.duration), 1),
                         "reason": win.get("reason", ""), "length": round(song.duration, 1),
                         "song_duration": round(win.get("song_duration") or 0)}
    return {"slider": slider_step(blend_info, ens, info.get("forced_orchestra", False)),
            "song": song_step, "era": era_step, "composer": composer_step, "ensemble": ens_step,
            "changes": changes,
            "scores": {"total": total, "classicalness": C, "retention": R, "w": total.get("w", 0.5)},
            "review": review, "search": {"n_trials": info.get("n_trials"), "seconds": round(info.get("seconds", 0), 1)}}


_ = np
