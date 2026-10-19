#!/usr/bin/env python3
"""Train / calibrate Classicalize on hundreds of songs.

1. Learn a POP harmony model from the training songs' chord progressions and
   calibrate the classical-vs-pop harmony scale used by the period listener
   (classical side: 483 public-domain scores, see scripts/build_era_models.py).
2. Run the fast composer-style search on every song and record which era,
   composer and ensemble won, with the classicalness / retention it reached
   → runs/report.json (shown on the website's Guide page).

Usage
  python train.py --synthetic 300                 # offline, generated pop songs
  python train.py --songs datasets/pop_songs.txt  # real recordings from YouTube (slow: downloads)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from classicalize import judge, theory  # noqa: E402
from classicalize.optimize import search  # noqa: E402

RUNS = os.path.join(ROOT, "runs")


def load_song(entry):
    kind, x = entry
    if kind == "synthetic":
        from classicalize.synthpop import make_song
        s = make_song(int(x))
        return s, {"title": s.title, "genre": s.meta.get("genre")}
    from classicalize.pipeline import prepare_song
    s, _, meta, _ = prepare_song(x)
    return s, {"title": meta.get("title", x), "genre": "real"}


def tokens(song):
    t = []
    for c in song.chord_objs():
        if c is not None:
            k = c.token(song.tonic)
            if not t or t[-1] != k:
                t.append(k)
    return t


def calibrate(entries):
    print("phase 1: learning the pop harmony model + calibrating the period listener …")
    seqs = {"major": [], "minor": []}
    for e in entries:
        try:
            s, _ = load_song(e)
        except Exception as ex:  # noqa: BLE001
            print("  ! skip", e, ex)
            continue
        t = tokens(s)
        if len(t) >= 4:
            seqs[s.mode].append(t)
    M = theory.era_model()
    with open(os.path.join(ROOT, "models", "pop_harmony.json"), "w") as f:
        json.dump(theory.build_bigram(seqs, M.vocab), f)
    theory._POP_MODEL = None
    lp, lc = {}, {}
    for era in M.eras():
        v = [theory.harmony_llr(t, era, m) for m in seqs for t in seqs[m]]
        lp[era] = float(np.median([x for x in v if x is not None]))
        v = [theory.harmony_llr(p["tokens"], era, p["mode"]) for p in M.d.get("pieces", []) if p["era"] == era]
        lc[era] = float(np.median([x for x in v if x is not None]))
    calib = {"llr_pop": lp, "llr_classical": lc, "trained_on": sum(len(v) for v in seqs.values())}
    with open(judge.CALIB_PATH, "w") as f:
        json.dump(calib, f, indent=1)
    print("  harmony log-likelihood ratio  pop:", {k: round(v, 2) for k, v in lp.items()},
          " classical:", {k: round(v, 2) for k, v in lc.items()})


def run_one(args):
    entry, w = args
    try:
        s, info = load_song(entry)
        t0 = time.time()
        best, trials, inf = search(s, w=w)
        return {**info, "w": w, "composer": best.composer, "era": best.era, "ensemble": best.ensemble,
                "C": max(trials, key=lambda t: t["adj"])["C"], "R": max(trials, key=lambda t: t["adj"])["R"],
                "score": max(trials, key=lambda t: t["adj"])["score"], "seconds": time.time() - t0,
                "n_trials": inf["n_trials"], "top3": [c["id"] for c in inf["composer_rank"][:3]]}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e), "entry": entry}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs")
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--weights", default="0.3,0.5,0.7")
    a = ap.parse_args()
    entries = [("synthetic", 1000 + i) for i in range(a.synthetic)]
    if a.songs:
        entries += [("youtube", ln.strip()) for ln in open(a.songs) if ln.strip() and not ln.startswith("#")]
    if a.limit:
        entries = entries[:a.limit]
    if not entries:
        ap.error("give --synthetic N and/or --songs FILE")
    random.Random(0).shuffle(entries)
    os.makedirs(RUNS, exist_ok=True)
    calibrate(entries)

    print(f"phase 2: composer-style search on {len(entries)} songs …")
    ws = [float(x) for x in a.weights.split(",")]
    jobs = [(e, ws[i % len(ws)]) for i, e in enumerate(entries)]
    recs = []
    with Pool(a.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(run_one, jobs)):
            if "error" in r:
                print("  !", r)
                continue
            recs.append(r)
            if (k + 1) % 25 == 0:
                print(f"  {k + 1}/{len(jobs)}  last: {r['title'][:40]} → {r['composer']} / {r['ensemble']} "
                      f"(C {r['C']:.2f}, R {r['R']:.2f}, {r['seconds']:.1f}s)")
    with open(os.path.join(RUNS, "train.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    by_genre = defaultdict(Counter)
    for r in recs:
        by_genre[r["genre"]][r["composer"]] += 1
    rep = {"n": len(recs), "mean_C": float(np.mean([r["C"] for r in recs])),
           "mean_R": float(np.mean([r["R"] for r in recs])),
           "mean_search_seconds": float(np.mean([r["seconds"] for r in recs])),
           "composer_counts": Counter(r["composer"] for r in recs).most_common(),
           "era_counts": Counter(r["era"] for r in recs).most_common(),
           "ensemble_counts": Counter(r["ensemble"] for r in recs).most_common(),
           "by_genre": {g: c.most_common(3) for g, c in sorted(by_genre.items())},
           "by_w": {str(w): {"C": float(np.mean([r["C"] for r in recs if r["w"] == w])),
                             "R": float(np.mean([r["R"] for r in recs if r["w"] == w]))} for w in ws}}
    with open(os.path.join(RUNS, "report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
