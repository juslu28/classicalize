"""Fast per-song search over composer styles.

Every trial is scored *symbolically* (notes, chords and voices, no audio
rendering) on a 16-bar excerpt around the chorus, so a trial costs ~0.05–0.1 s
and the whole search takes a few seconds:

  stage 1  every allowed composer preset × its favourite ensembles
  stage 2  local refinement of the 3 best (composer, ensemble) pairs
  final    score_adj = w·classicalness + (1-w)·retention
                       + 0.35 · composer–song fit + 0.03 · ensemble preference
"""
from __future__ import annotations

import math
import random
import time

from . import judge
from .analysis import SongAnalysis
from .arrange import ENSEMBLES, ArrangeParams, arrange
from .composers import COMPOSERS, composer_fit, song_features

FIT_W, ENS_W = 0.35, 0.03
KNOBS = ("reharm", "ornament", "straighten", "density", "counterpoint", "dynamics", "rubato")


def evaluate(song: SongAnalysis, p: ArrangeParams, w=0.5):
    arr = arrange(song, p)
    C = judge.classicalness(arr)
    R = judge.retention(song, arr)
    return {"params": p.to_dict(), "C": C, "R": R, "total": judge.total(C, R, w)}


def preset(cid, ensemble, seed=0, **over):
    c = COMPOSERS[cid]
    k = dict(c["knobs"])
    k.update(over)
    return ArrangeParams(era=c["era"], ensemble=ensemble, composer=cid, add_ending=False, seed=seed, **k)


def _perturb(p: ArrangeParams, rng: random.Random, scale=0.18):
    d = p.to_dict()
    for k in KNOBS:
        d[k] = min(1.0, max(0.0, d[k] + rng.gauss(0, scale)))
    d["tempo_factor"] = min(1.1, max(0.88, d["tempo_factor"] + rng.gauss(0, 0.03)))
    if rng.random() < 0.25:
        d["harmonic_rhythm"] = rng.choice([1, 2, 4])
    d["seed"] = rng.randrange(10 ** 6)
    return ArrangeParams(**d)


def search(song: SongAnalysis, w=0.5, era="auto", composer="auto", ensemble="auto", excerpt_beats=64,
           budget_s=4.0, progress=lambda *_: None, slider=None):
    """slider (0 = original … 1 = classical) scales every preset (see blend.slider_params);
    near the original end the full orchestra is used, since it is the only classical
    ensemble with enough colours for vocals, synths, bass and drums at once."""
    from .blend import T_ACCOMP, slider_params
    t0 = time.time()
    user_ensemble = ensemble
    if slider is not None and ensemble == "auto" and slider < T_ACCOMP:
        ensemble = "orchestra"
    rng = random.Random(f"{song.title}-{song.n_beats}")
    feats = song_features(song)
    ex = song.excerpt(song.chorus_start(excerpt_beats), excerpt_beats)

    if composer != "auto":
        comps = [composer]
    elif era != "auto":
        comps = [c for c, v in COMPOSERS.items() if v["era"] == era]
    else:
        comps = list(COMPOSERS)
    fits = {c: composer_fit(feats, c) for c in comps}

    def ens_for(c, k=2):
        if ensemble != "auto":
            return [ensemble]
        prefs = COMPOSERS[c]["ensembles"]
        return sorted(prefs, key=lambda e: -prefs[e])[:k]

    trials = []

    def run(p):
        raw = p                      # keep the unscaled preset; blend.build applies the slider again
        if slider is not None:
            p = slider_params(p, slider)
        r = evaluate(ex, p, w)
        c = p.composer
        adj = r["total"]["score"] + FIT_W * fits[c][0] + ENS_W * COMPOSERS[c]["ensembles"].get(p.ensemble, 0.5)
        t = {"params": raw.to_dict(), "C": r["C"]["score"], "R": r["R"]["score"], "score": r["total"]["score"],
             "adj": adj, "C_parts": r["C"]["components"], "R_parts": r["R"]["components"]}
        trials.append(t)
        return t

    # ---- stage 1: presets
    for c in comps:
        for e in ens_for(c):
            run(preset(c, e, seed=len(trials)))
    progress(f"Tried {len(trials)} composer/ensemble styles", None)

    # ---- stage 2: refine the best few
    def best_by_pair():
        bp = {}
        for t in trials:
            key = (t["params"]["composer"], t["params"]["ensemble"])
            if key not in bp or t["adj"] > bp[key]["adj"]:
                bp[key] = t
        return sorted(bp.values(), key=lambda t: -t["adj"])
    leaders = best_by_pair()[:3]
    rounds = 0
    while time.time() - t0 < budget_s and rounds < 10:
        for lead in leaders:
            run(_perturb(ArrangeParams(**lead["params"]), rng, scale=0.2 * (0.85 ** rounds)))
        leaders = best_by_pair()[:3]
        rounds += 1
    best = max(trials, key=lambda t: t["adj"])
    bc = best["params"]["composer"]

    # ---- for the Details page: how every ensemble would do in the winning composer's style
    if ensemble == "auto" or (user_ensemble == "auto" and slider is not None):
        done = {t["params"]["ensemble"] for t in trials if t["params"]["composer"] == bc}
        for e in ENSEMBLES:
            if e not in done:
                run(preset(bc, e, seed=len(trials)))
    progress(f"Searched {len(trials)} arrangements in {time.time() - t0:.1f}s", None)

    # ---- summaries
    per_comp = {}
    for t in trials:
        c = t["params"]["composer"]
        if c not in per_comp or t["adj"] > per_comp[c]["adj"]:
            per_comp[c] = t
    ranked = sorted(per_comp.values(), key=lambda t: -t["adj"])
    top = ranked[0]["adj"]
    ex_w = [math.exp((t["adj"] - top) / 0.02) for t in ranked]
    Z = sum(ex_w)
    composer_rank = [{"id": t["params"]["composer"], "prob": e / Z, "score": t["score"], "adj": t["adj"],
                      "fit": fits[t["params"]["composer"]][0], "reasons": fits[t["params"]["composer"]][1],
                      "ensemble": t["params"]["ensemble"], "C": t["C"], "R": t["R"]}
                     for t, e in zip(ranked, ex_w)]
    per_era = {}
    for t in trials:
        e = t["params"]["era"]
        if e not in per_era or t["adj"] > per_era[e]["adj"]:
            per_era[e] = {"adj": t["adj"], "score": t["score"], "C": t["C"], "R": t["R"],
                          "composer": t["params"]["composer"], "ensemble": t["params"]["ensemble"]}
    per_ens = {}
    for t in trials:
        if t["params"]["composer"] != bc:
            continue
        e = t["params"]["ensemble"]
        if e not in per_ens or t["adj"] > per_ens[e]["adj"]:
            per_ens[e] = {"adj": t["adj"], "score": t["score"], "C": t["C"], "R": t["R"]}
    bp = ArrangeParams(**{**best["params"], "add_ending": False})
    info = {"n_trials": len(trials), "seconds": time.time() - t0, "features": feats,
            "composer_rank": composer_rank, "per_era": per_era, "per_ensemble": per_ens,
            "excerpt_start": song.chorus_start(excerpt_beats), "excerpt_beats": excerpt_beats,
            "user_choice": {"era": era, "composer": composer, "ensemble": user_ensemble},
            "forced_orchestra": user_ensemble == "auto" and ensemble == "orchestra" and slider is not None,
            "fit_parts": fits[bc][2]}
    return bp, trials, info
