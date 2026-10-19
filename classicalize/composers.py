"""Catalogue of eras, composers and ensembles + song-to-composer matching.

Each composer is a *style preset* for the arranger (era, favourite ensembles,
texture knobs) plus a *fit profile* describing which kinds of songs suit them.
The same data feeds the website's Guide and Details pages.
"""
from __future__ import annotations

import math

import numpy as np

ERAS = {
    "baroque": {
        "name": "Baroque", "years": "c. 1600–1750",
        "summary": "Music built from interweaving lines (counterpoint) over a constantly moving bass, "
                   "with ornamented melodies and steady, motor-like rhythm.",
        "traits": ["Counterpoint: several independent melodic lines at once",
                   "Basso continuo: a bass line plus harpsichord filling in the chords",
                   "Terraced dynamics: whole sections loud or soft, rarely gradual",
                   "Ornaments: trills, mordents and decorated long notes",
                   "One steady pulse and mood per movement"],
        "instruments": "Strings, harpsichord, organ, recorder/flute, oboe, bassoon, natural trumpet, timpani",
    },
    "classical": {
        "name": "Classical", "years": "c. 1750–1820",
        "summary": "Clear, balanced phrases with a singable tune on top and light, transparent "
                   "accompaniment, where every phrase ends in a satisfying cadence.",
        "traits": ["Homophony: one main melody with accompaniment beneath",
                   "Alberti bass and 'murmuring' repeated-note accompaniments",
                   "Question-and-answer 4- and 8-bar phrases",
                   "Crescendos and contrasts of mood within a movement",
                   "The string quartet and the symphony orchestra take their classic shape"],
        "instruments": "String quartet, fortepiano, flute, oboe, clarinet, bassoon, horns, trumpets, timpani",
    },
    "romantic": {
        "name": "Romantic", "years": "c. 1820–1900",
        "summary": "Emotion first: long singing melodies, rich and colourful harmony, flexible "
                   "tempo (rubato), big dynamic swells and larger orchestras.",
        "traits": ["Rubato: tempo breathes, slowing at phrase ends",
                   "Wide piano arpeggios held by the sustain pedal",
                   "Richer chords: borrowed chords, diminished sevenths, chromatic colour",
                   "Singing counter-melodies (often in the cellos)",
                   "Huge dynamic range, from whisper to full orchestra"],
        "instruments": "Grand piano, full string sections, clarinets, horns, harp, expanded brass and percussion",
    },
}

ENSEMBLES = {
    "piano": {
        "name": "Solo keyboard",
        "summary": "One player. Harpsichord for Baroque styles, piano for Classical and Romantic.",
        "parts": {"melody": "Right hand plays the tune (plus chord tones beneath it in loud passages)",
                  "keys_lh": "Left hand plays the bass and the accompaniment figure (walking bass, "
                             "Alberti bass or wide pedalled arpeggios)"},
        "why": "the most intimate setting: one instrument carries both melody and harmony, which keeps "
               "the tune very clear",
    },
    "string_quartet": {
        "name": "String quartet",
        "summary": "Two violins, viola and cello. The Classical era's favourite chamber group.",
        "parts": {"melody": "First violin sings the vocal line",
                  "inner_hi": "Second violin fills in harmony or plays a counter-line",
                  "inner_lo": "Viola holds the middle of the chord",
                  "bass": "Cello plays the bass line"},
        "why": "four equal voices give clean voice-leading and a warm, blended, unmistakably "
               "classical sound",
    },
    "chamber": {
        "name": "Chamber ensemble",
        "summary": "A wind soloist with strings (and keyboard): flute for Baroque and Classical "
                   "styles, clarinet for Romantic ones.",
        "parts": {"melody": "A wind soloist (flute or clarinet) carries the tune",
                  "inner_hi": "Violin plays harmony or counter-melody",
                  "inner_lo": "Viola fills in the chord",
                  "bass": "Cello plays the bass",
                  "continuo": "Harpsichord realises the chords (Baroque)",
                  "keys_lh": "Piano left hand adds arpeggios (Romantic)"},
        "why": "a contrasting wind colour on the melody makes the tune stand out from the strings",
    },
    "orchestra": {
        "name": "Orchestra",
        "summary": "String sections plus woodwinds, horns or trumpets, timpani (and harp in Romantic style).",
        "parts": {"melody": "Violin section on the tune, doubled by flute or oboe in loud sections",
                  "inner_hi": "Second violins: harmony / pulsing chords",
                  "inner_lo": "Violas: inner harmony",
                  "counter": "Cellos: a singing counter-melody (Romantic)",
                  "bass": "Cellos / double basses / bassoon on the bass line",
                  "winds": "Oboes, clarinets, bassoons sustain harmony in louder sections",
                  "brass": "Horns or trumpets reinforce climaxes",
                  "harp": "Harp arpeggios (Romantic)",
                  "timpani": "Timpani mark cadences in the loudest sections",
                  "continuo": "Harpsichord continuo (Baroque)"},
        "why": "the full colour palette: the song's loud choruses can swell into the whole orchestra",
    },
}

INSTRUMENT_NAMES = {
    "violin": "Violin", "violins": "Violins I", "violins2": "Violins II", "viola": "Viola", "violas": "Violas",
    "cello": "Cello", "cellos": "Cellos", "contrabass": "Double bass", "piano_rh": "Piano (right hand)",
    "piano_lh": "Piano (left hand)", "harpsichord_rh": "Harpsichord (right hand)",
    "harpsichord_lh": "Harpsichord (left hand)", "flute": "Flute", "oboe": "Oboe", "clarinet": "Clarinet",
    "bassoon": "Bassoon", "horn": "Horn", "trumpet": "Trumpet", "timpani": "Timpani", "harp": "Harp",
    "organ": "Organ",
}

# knobs: reharm, ornament, straighten, density, counterpoint, dynamics, harmonic_rhythm, rubato, tempo_factor
# fit: tempo (centre, width), minor (-1 major … +1 minor), drama 0..1 (verse/chorus contrast),
#      busy 0..1 (melody notes per beat), lyric 0..1 (melodic range), rich 0..1 (chromatic harmony),
#      waltz bonus for 3/4
COMPOSERS = {
    "bach": {
        "name": "Johann Sebastian Bach", "short": "Bach", "years": "1685–1750", "era": "baroque",
        "place": "Leipzig", "works": ["Brandenburg Concertos", "Goldberg Variations", "Cello Suites", "Air on the G String"],
        "bio": "The great master of counterpoint: several melodies woven together over a walking bass, "
               "every line meaningful.",
        "signature": ["contrapuntal inner lines", "walking continuo bass", "cadential trills", "steady motor rhythm"],
        "ensembles": {"piano": 1.0, "chamber": 0.9, "string_quartet": 0.8, "orchestra": 0.7},
        "knobs": dict(reharm=0.6, ornament=0.55, straighten=0.75, density=0.7, counterpoint=0.9, dynamics=0.3,
                      harmonic_rhythm=2, rubato=0.0, tempo_factor=1.0),
        "fit": dict(tempo=(100, 25), minor=0.15, drama=0.2, busy=0.65, lyric=0.5, rich=0.35, waltz=0.0),
    },
    "vivaldi": {
        "name": "Antonio Vivaldi", "short": "Vivaldi", "years": "1678–1741", "era": "baroque",
        "place": "Venice", "works": ["The Four Seasons", "Gloria", "Concerto for Two Violins in A minor"],
        "bio": "The 'Red Priest' of Venice: energetic string concertos built on driving repeated-note "
               "basses and catchy, repeating patterns.",
        "signature": ["driving repeated-note bass", "bright string orchestra", "echo (loud/soft) dynamics",
                      "repeating sequences"],
        "ensembles": {"orchestra": 1.0, "chamber": 0.8, "string_quartet": 0.7, "piano": 0.4},
        "knobs": dict(reharm=0.45, ornament=0.4, straighten=0.6, density=0.95, counterpoint=0.45, dynamics=0.6,
                      harmonic_rhythm=2, rubato=0.0, tempo_factor=1.05),
        "fit": dict(tempo=(130, 18), minor=0.3, drama=0.5, busy=0.7, lyric=0.4, rich=0.2, waltz=0.0),
    },
    "handel": {
        "name": "George Frideric Handel", "short": "Handel", "years": "1685–1759", "era": "baroque",
        "place": "London", "works": ["Messiah ('Hallelujah')", "Water Music", "Music for the Royal Fireworks"],
        "bio": "Grand, ceremonial music for London audiences: broad harmonies, trumpets and timpani, "
               "and big choruses everyone could follow.",
        "signature": ["broad, slow-moving harmony", "trumpets and timpani at climaxes", "majestic major keys"],
        "ensembles": {"orchestra": 1.0, "chamber": 0.6, "string_quartet": 0.5, "piano": 0.6},
        "knobs": dict(reharm=0.55, ornament=0.3, straighten=0.8, density=0.5, counterpoint=0.5, dynamics=0.7,
                      harmonic_rhythm=4, rubato=0.0, tempo_factor=0.95),
        "fit": dict(tempo=(88, 18), minor=-0.6, drama=0.7, busy=0.35, lyric=0.5, rich=0.2, waltz=0.0),
    },
    "haydn": {
        "name": "Joseph Haydn", "short": "Haydn", "years": "1732–1809", "era": "classical",
        "place": "Vienna / Esterháza", "works": ["'Emperor' Quartet", "'Surprise' Symphony", "Trumpet Concerto"],
        "bio": "'Father of the string quartet and the symphony': witty, cheerful, perfectly balanced "
               "phrases with the odd surprise.",
        "signature": ["four equal string voices", "clear question-and-answer phrases", "playful rhythm"],
        "ensembles": {"string_quartet": 1.0, "orchestra": 0.8, "piano": 0.6, "chamber": 0.6},
        "knobs": dict(reharm=0.5, ornament=0.35, straighten=0.6, density=0.5, counterpoint=0.5, dynamics=0.55,
                      harmonic_rhythm=2, rubato=0.0, tempo_factor=1.0),
        "fit": dict(tempo=(118, 18), minor=-0.5, drama=0.35, busy=0.5, lyric=0.45, rich=0.25, waltz=0.3),
    },
    "mozart": {
        "name": "Wolfgang Amadeus Mozart", "short": "Mozart", "years": "1756–1791", "era": "classical",
        "place": "Salzburg / Vienna", "works": ["Eine kleine Nachtmusik", "Symphony No. 40", "Piano Sonata K. 545",
                                                 "Clarinet Concerto"],
        "bio": "Effortless, song-like melodies over light Alberti-bass accompaniment, with graceful "
               "appoggiaturas and flawless proportions.",
        "signature": ["Alberti bass", "graceful appoggiaturas", "singing melody on top", "elegant cadences"],
        "ensembles": {"piano": 1.0, "string_quartet": 0.95, "chamber": 0.9, "orchestra": 0.85},
        "knobs": dict(reharm=0.5, ornament=0.5, straighten=0.65, density=0.55, counterpoint=0.35, dynamics=0.5,
                      harmonic_rhythm=2, rubato=0.05, tempo_factor=1.0),
        "fit": dict(tempo=(110, 22), minor=-0.3, drama=0.35, busy=0.5, lyric=0.65, rich=0.25, waltz=0.2),
    },
    "beethoven": {
        "name": "Ludwig van Beethoven", "short": "Beethoven", "years": "1770–1827", "era": "classical",
        "place": "Vienna", "works": ["Symphony No. 5", "'Moonlight' Sonata", "'Pathétique' Sonata", "Für Elise"],
        "bio": "The bridge to Romanticism: dramatic contrasts, explosive dynamics, driving rhythm and "
               "a struggle-to-triumph arc.",
        "signature": ["extreme loud/soft contrasts", "driving repeated chords", "minor-key drama"],
        "ensembles": {"orchestra": 1.0, "piano": 0.95, "string_quartet": 0.8, "chamber": 0.6},
        "knobs": dict(reharm=0.55, ornament=0.2, straighten=0.6, density=0.7, counterpoint=0.5, dynamics=1.0,
                      harmonic_rhythm=2, rubato=0.15, tempo_factor=1.0),
        "fit": dict(tempo=(105, 25), minor=0.6, drama=0.9, busy=0.5, lyric=0.5, rich=0.4, waltz=0.0),
    },
    "schubert": {
        "name": "Franz Schubert", "short": "Schubert", "years": "1797–1828", "era": "romantic",
        "place": "Vienna", "works": ["Ave Maria", "Ständchen (Serenade)", "'Trout' Quintet", "Erlkönig"],
        "bio": "Master of the art song (Lied): a voice-like melody with an expressive piano accompaniment "
               "that paints the mood.",
        "signature": ["vocal, singable melody", "repeated-chord piano accompaniment", "sudden major/minor shifts"],
        "ensembles": {"piano": 1.0, "string_quartet": 0.85, "chamber": 0.8, "orchestra": 0.6},
        "knobs": dict(reharm=0.45, ornament=0.3, straighten=0.5, density=0.5, counterpoint=0.3, dynamics=0.6,
                      harmonic_rhythm=2, rubato=0.3, tempo_factor=1.0),
        "fit": dict(tempo=(88, 20), minor=0.1, drama=0.45, busy=0.55, lyric=0.75, rich=0.45, waltz=0.4),
    },
    "mendelssohn": {
        "name": "Felix Mendelssohn", "short": "Mendelssohn", "years": "1809–1847", "era": "romantic",
        "place": "Leipzig / Berlin", "works": ["Violin Concerto in E minor", "A Midsummer Night's Dream",
                                                "Songs Without Words"],
        "bio": "Light, quicksilver Romanticism: sparkling fast strings, clear forms, elegant lyricism.",
        "signature": ["quick, light string textures", "clear forms", "lyrical 'songs without words'"],
        "ensembles": {"orchestra": 1.0, "string_quartet": 0.9, "piano": 0.8, "chamber": 0.6},
        "knobs": dict(reharm=0.45, ornament=0.3, straighten=0.55, density=0.85, counterpoint=0.5, dynamics=0.6,
                      harmonic_rhythm=2, rubato=0.2, tempo_factor=1.03),
        "fit": dict(tempo=(128, 18), minor=-0.1, drama=0.4, busy=0.65, lyric=0.55, rich=0.35, waltz=0.0),
    },
    "chopin": {
        "name": "Frédéric Chopin", "short": "Chopin", "years": "1810–1849", "era": "romantic",
        "place": "Warsaw / Paris", "works": ["Nocturne Op. 9 No. 2", "Ballade No. 1", "Minute Waltz",
                                              "Raindrop Prelude"],
        "bio": "The poet of the piano: singing nocturne melodies with ornamental turns over wide, "
               "pedalled left-hand arpeggios, and a constantly breathing rubato.",
        "signature": ["wide pedalled arpeggios", "rubato", "ornamental turns", "solo piano"],
        "ensembles": {"piano": 1.0, "chamber": 0.4, "string_quartet": 0.3, "orchestra": 0.3},
        "knobs": dict(reharm=0.5, ornament=0.6, straighten=0.5, density=0.6, counterpoint=0.2, dynamics=0.75,
                      harmonic_rhythm=2, rubato=0.85, tempo_factor=0.95),
        "fit": dict(tempo=(76, 18), minor=0.25, drama=0.5, busy=0.45, lyric=0.8, rich=0.55, waltz=0.9),
    },
    "liszt": {
        "name": "Franz Liszt", "short": "Liszt", "years": "1811–1886", "era": "romantic",
        "place": "Weimar / Paris", "works": ["Liebestraum No. 3", "Hungarian Rhapsody No. 2", "La Campanella"],
        "bio": "The first superstar pianist: thundering, virtuosic textures, cascading arpeggios and "
               "huge dynamic climaxes.",
        "signature": ["dense, virtuosic arpeggios", "extreme dynamics", "theatrical rubato"],
        "ensembles": {"piano": 1.0, "orchestra": 0.6, "chamber": 0.3, "string_quartet": 0.3},
        "knobs": dict(reharm=0.55, ornament=0.5, straighten=0.5, density=1.0, counterpoint=0.3, dynamics=1.0,
                      harmonic_rhythm=2, rubato=0.65, tempo_factor=1.0),
        "fit": dict(tempo=(100, 25), minor=0.2, drama=0.95, busy=0.6, lyric=0.65, rich=0.6, waltz=0.0),
    },
    "brahms": {
        "name": "Johannes Brahms", "short": "Brahms", "years": "1833–1897", "era": "romantic",
        "place": "Hamburg / Vienna", "works": ["Hungarian Dance No. 5", "Clarinet Quintet", "Lullaby",
                                                "Symphony No. 3"],
        "bio": "Romantic warmth with Baroque craft: rich, autumnal textures, inner counter-melodies and "
               "the dark glow of clarinet and viola.",
        "signature": ["rich inner counter-melodies", "clarinet and strings", "warm, dark colours"],
        "ensembles": {"chamber": 1.0, "orchestra": 0.9, "string_quartet": 0.85, "piano": 0.7},
        "knobs": dict(reharm=0.55, ornament=0.25, straighten=0.55, density=0.6, counterpoint=0.75, dynamics=0.7,
                      harmonic_rhythm=2, rubato=0.3, tempo_factor=0.97),
        "fit": dict(tempo=(90, 20), minor=0.4, drama=0.55, busy=0.45, lyric=0.6, rich=0.6, waltz=0.5),
    },
    "tchaikovsky": {
        "name": "Pyotr Ilyich Tchaikovsky", "short": "Tchaikovsky", "years": "1840–1893", "era": "romantic",
        "place": "St Petersburg / Moscow", "works": ["Swan Lake", "The Nutcracker", "Piano Concerto No. 1",
                                                      "Symphony No. 6"],
        "bio": "Big-hearted melody and orchestral colour: soaring tunes over pulsing strings, horns, harp "
               "and climaxes that sweep everything up.",
        "signature": ["soaring melody over pulsing strings", "horns and harp", "huge emotional climaxes"],
        "ensembles": {"orchestra": 1.0, "piano": 0.6, "string_quartet": 0.6, "chamber": 0.5},
        "knobs": dict(reharm=0.5, ornament=0.25, straighten=0.5, density=0.7, counterpoint=0.6, dynamics=1.0,
                      harmonic_rhythm=2, rubato=0.5, tempo_factor=0.97),
        "fit": dict(tempo=(92, 22), minor=0.35, drama=0.9, busy=0.4, lyric=0.75, rich=0.5, waltz=0.8),
    },
}


# ---------------------------------------------------------------- song features
def song_features(song) -> dict:
    """Musical features of the analysed pop song (beat-based, tempo independent)."""
    from .judge import loop_index, melody_syncopation
    from .theory import diatonic_chords
    mel = song.melody
    pitches = [n.pitch for n in mel]
    ivs = np.abs(np.diff(pitches)) if len(pitches) > 2 else np.array([1])
    toks = []
    for c in song.chord_objs():
        if c is None:
            continue
        t = c.token(song.tonic)
        if not toks or toks[-1] != t:
            toks.append(t)
    dia = {d.token(song.tonic) for d in diatonic_chords(song.tonic, song.mode)}
    en = [s["energy"] for s in song.sections] or [1.0]
    return {
        "tempo": float(song.tempo),
        "minor": 1.0 if song.mode == "minor" else 0.0,
        "syncopation": melody_syncopation([(n.start, n.dur) for n in mel]),
        "notes_per_beat": len(mel) / max(1, song.n_beats),
        "range": float(np.percentile(pitches, 95) - np.percentile(pitches, 5)) if pitches else 12.0,
        "repeat_frac": float(np.mean(ivs == 0)),
        "loop": loop_index(toks),
        "non_diatonic": float(np.mean([t not in dia for t in toks])) if toks else 0.0,
        "energy_var": float(np.std(en)),
        "triple": 1.0 if song.beats_per_bar == 3 else 0.0,
    }


def _norm(f):
    return {
        "drama": min(1.0, f["energy_var"] / 0.4),
        "busy": min(1.0, f["notes_per_beat"] / 1.6),
        "lyric": min(1.0, max(0.0, (f["range"] - 5) / 14)),
        "rich": min(1.0, f["non_diatonic"] / 0.35),
    }


def composer_fit(features: dict, cid: str):
    """How well a song suits a composer's style: (score 0..1, list of reasons)."""
    c = COMPOSERS[cid]
    fp = c["fit"]
    n = _norm(features)
    tc, tw = fp["tempo"]
    parts = {
        "tempo": math.exp(-0.5 * ((features["tempo"] - tc) / tw) ** 2),
        "mode": 0.5 + 0.5 * fp["minor"] * (1 if features["minor"] else -1),
        "drama": 1 - abs(fp["drama"] - n["drama"]),
        "busy": 1 - abs(fp["busy"] - n["busy"]),
        "lyric": 1 - abs(fp["lyric"] - n["lyric"]),
        "rich": 1 - abs(fp["rich"] - n["rich"]),
    }
    w = {"tempo": 0.28, "mode": 0.16, "drama": 0.2, "busy": 0.12, "lyric": 0.14, "rich": 0.1}
    score = sum(w[k] * v for k, v in parts.items())
    if features["triple"]:
        score += 0.1 * fp["waltz"]
    reasons = []
    t = features["tempo"]
    if parts["tempo"] > 0.7:
        pace = "slow" if t < 85 else ("moderate" if t < 115 else "quick")
        reasons.append(f"its {pace} tempo ({t:.0f} bpm) is right in {c['short']}'s comfort zone")
    if parts["mode"] > 0.7:
        reasons.append(f"it is in a {'minor' if features['minor'] else 'major'} key, which {c['short']} "
                       f"{'favoured for dramatic or tender music' if features['minor'] else 'used for bright, festive music'}")
    if parts["drama"] > 0.75 and fp["drama"] >= 0.6:
        reasons.append("the big jump in energy from verse to chorus suits a composer famous for dramatic contrast")
    elif parts["drama"] > 0.75 and fp["drama"] < 0.4:
        reasons.append("its fairly even energy suits a steady, unbroken style")
    if parts["lyric"] > 0.75 and fp["lyric"] >= 0.6:
        reasons.append("the wide-ranging vocal melody suits a composer known for long, singing lines")
    if parts["busy"] > 0.75 and fp["busy"] >= 0.6:
        reasons.append("the busy, many-note melody suits a style of running lines")
    if parts["rich"] > 0.75 and fp["rich"] >= 0.5:
        reasons.append("its colourful chords (outside the plain key) suit richer Romantic harmony")
    if features["triple"] and fp["waltz"] >= 0.5:
        reasons.append(f"it is in 3/4 time, and {c['short']} wrote famous waltzes")
    return float(min(1.0, score)), reasons, parts


def rank_composers(features, allowed=None):
    allowed = allowed or list(COMPOSERS)
    out = []
    for cid in allowed:
        s, r, p = composer_fit(features, cid)
        out.append({"id": cid, "fit": s, "reasons": r, "parts": p})
    return sorted(out, key=lambda x: -x["fit"])


def catalog():
    """Public data for the Guide page."""
    comps = []
    for cid, c in COMPOSERS.items():
        comps.append({"id": cid, **{k: c[k] for k in ("name", "short", "years", "era", "place", "works", "bio",
                                                       "signature")},
                      "ensembles": sorted(c["ensembles"], key=lambda e: -c["ensembles"][e])})
    return {"eras": ERAS, "composers": comps, "ensembles": ENSEMBLES, "instrument_names": INSTRUMENT_NAMES}
