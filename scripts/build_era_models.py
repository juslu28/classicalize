"""Learn per-era harmony language models + style statistics from real classical
scores in the music21 corpus (public domain).

  Baroque   : J.S. Bach (chorales, inventions...), Handel, Corelli, C.P.E. Bach
  Classical : Haydn, Mozart, Beethoven (quartets, sonatas)
  Romantic  : Chopin, R. & C. Schumann, Schubert, Verdi, Weber
              (small corpus -> smoothed toward the Classical model + extra
               chromatic vocabulary observed in the pieces)

Output: models/era_models.json
"""
import json
import math
import os
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classicalize.theory import QUALITIES, match_chord  # noqa: E402

ERA_COMPOSERS = {
    "baroque": ["bach", "handel", "corelli", "cpebach"],
    "classical": ["haydn", "mozart", "beethoven"],
    "romantic": ["chopin", "schumann_robert", "schumann_clara", "schubert", "verdi", "weber"],
}
MAX_BEATS = 480


def analyse_piece(args):
    era, path = args
    from music21 import corpus
    try:
        s = corpus.parse(path)
        if hasattr(s, "scores"):          # opus
            s = s.scores[0]
        k = s.analyze("key")
        tonic, mode = k.tonic.pitchClass, k.mode
        if mode not in ("major", "minor"):
            return None
        notes = []      # (start, end, midi, part_idx)
        parts = list(s.parts) if s.parts else [s]
        for pi, p in enumerate(parts):
            for n in p.flatten().notes:
                st = float(n.offset)
                en = st + float(n.quarterLength)
                if st > MAX_BEATS:
                    continue
                for pt in n.pitches:
                    notes.append((st, en, pt.midi, pi))
        if len(notes) < 20:
            return None
        end = min(MAX_BEATS, max(n[1] for n in notes))
        nb = int(math.floor(end))
        notes_arr = np.array([(a, b, m, p) for a, b, m, p in notes], float)

        # ---- chord per beat
        toks = []
        bass_root = []
        for b in range(nb):
            ov = np.clip(np.minimum(notes_arr[:, 1], b + 1) - np.maximum(notes_arr[:, 0], b), 0, None)
            idx = ov > 0
            if not idx.any():
                toks.append(None)
                continue
            w = np.zeros(12)
            for m, o in zip(notes_arr[idx, 2], ov[idx]):
                w[int(m) % 12] += o
            bass = int(notes_arr[idx, 2].min()) % 12
            c = match_chord(w, bass_pc=bass)
            toks.append(c.token(tonic))
            bass_root.append(bass == c.root)
        seq = [t for t in toks if t is not None]
        merged = [t for i, t in enumerate(seq) if i == 0 or t != seq[i - 1]]

        # ---- voice-pair parallels + top-voice stats (sampled on beats & half beats)
        par, trans = 0, 0
        if len(parts) >= 2:
            grid = np.arange(0, nb, 0.5)
            voice_at = []
            for pi in range(min(4, len(parts))):
                pn = notes_arr[notes_arr[:, 3] == pi]
                row = []
                for g in grid:
                    on = pn[(pn[:, 0] <= g) & (pn[:, 1] > g)]
                    row.append(on[:, 2].max() if len(on) else np.nan)
                voice_at.append(row)
            V = np.array(voice_at)
            for i in range(len(V)):
                for j in range(i + 1, len(V)):
                    a, b = V[i], V[j]
                    for t in range(1, len(grid)):
                        if np.isnan([a[t - 1], a[t], b[t - 1], b[t]]).any():
                            continue
                        if a[t] == a[t - 1] or b[t] == b[t - 1]:
                            continue
                        trans += 1
                        i1, i2 = int(abs(a[t - 1] - b[t - 1])) % 12, int(abs(a[t] - b[t])) % 12
                        if i1 == i2 and i1 in (0, 7):
                            par += 1
        # top voice = part 0 melodic stats
        p0 = sorted([n for n in notes if n[3] == 0], key=lambda x: x[0])
        mel = []
        for n in p0:
            if mel and abs(mel[-1][0] - n[0]) < 1e-6:
                if n[2] > mel[-1][2]:
                    mel[-1] = n
            else:
                mel.append(n)
        ivs = [abs(mel[i][2] - mel[i - 1][2]) for i in range(1, len(mel))]
        sync = sum(1 for n in mel if (n[0] % 1) > 1e-3 and (n[1] - n[0]) >= 0.5 and
                   math.floor(n[1] - 1e-6) > math.floor(n[0])) / max(1, len(mel))
        return {
            "era": era, "mode": mode, "tokens": merged,
            "chg_per_beat": len(merged) / max(1, len(seq)),
            "root_pos": float(np.mean(bass_root)) if bass_root else 0.5,
            "parallels": par, "transitions": trans,
            "step_frac": float(np.mean([i <= 2 for i in ivs])) if ivs else 0.5,
            "mean_leap": float(np.mean(ivs)) if ivs else 2.0,
            "sync": sync,
            "path": str(path),
        }
    except Exception as e:  # noqa: BLE001
        return {"error": f"{path}: {e}"}


def main():
    from music21 import corpus
    jobs = []
    for era, comps in ERA_COMPOSERS.items():
        for c in comps:
            for p in corpus.getComposer(c):
                if str(p).endswith((".mxl", ".xml", ".musicxml", ".krn", ".abc", ".mid", ".md")):
                    jobs.append((era, p))
    print("pieces queued:", Counter(e for e, _ in jobs))
    with Pool(max(1, os.cpu_count())) as pool:
        res = [r for r in pool.imap_unordered(analyse_piece, jobs, chunksize=4) if r]
    errs = [r for r in res if "error" in r]
    res = [r for r in res if "error" not in r]
    print("analysed:", Counter(r["era"] for r in res), "errors:", len(errs))

    vocab = sorted({f"{i}:{q}" for i in range(12) for q in QUALITIES})
    out = {"vocab": vocab, "smoothing": 0.05, "eras": {}, "n_pieces": {}}
    for era in ERA_COMPOSERS:
        rs = [r for r in res if r["era"] == era]
        out["n_pieces"][era] = len(rs)
        em = {}
        for mode in ("major", "minor"):
            uni, bi = Counter(), defaultdict(Counter)
            for r in rs:
                if r["mode"] != mode:
                    continue
                t = r["tokens"]
                uni.update(t)
                for a, b in zip(t, t[1:]):
                    bi[a][b] += 1
            em[mode] = {"unigram": dict(uni), "unigram_total": sum(uni.values()),
                        "bigram": {a: dict(c) for a, c in bi.items()},
                        "bigram_total": {a: sum(c.values()) for a, c in bi.items()}}
        par = sum(r["parallels"] for r in rs) / max(1, sum(r["transitions"] for r in rs))
        em["stats"] = {
            "parallel_rate": par,
            "chg_per_beat": float(np.median([r["chg_per_beat"] for r in rs])),
            "root_pos": float(np.median([r["root_pos"] for r in rs])),
            "step_frac": float(np.median([r["step_frac"] for r in rs])),
            "mean_leap": float(np.median([r["mean_leap"] for r in rs])),
            "sync": float(np.median([r["sync"] for r in rs])),
            "sync_p90": float(np.percentile([r["sync"] for r in rs], 90)),
        }
        out["eras"][era] = em

    # Romantic: small corpus -> back-off mixture with Classical counts (weight 0.5)
    rom, cla = out["eras"]["romantic"], out["eras"]["classical"]
    for mode in ("major", "minor"):
        R, C = rom[mode], cla[mode]
        scale = max(1, R["unigram_total"]) / max(1, C["unigram_total"])
        for t, n in C["unigram"].items():
            R["unigram"][t] = R["unigram"].get(t, 0) + 0.5 * n * scale
        for a, row in C["bigram"].items():
            rr = R["bigram"].setdefault(a, {})
            for b, n in row.items():
                rr[b] = rr.get(b, 0) + 0.5 * n * scale
        R["unigram_total"] = sum(R["unigram"].values())
        R["bigram_total"] = {a: sum(r.values()) for a, r in R["bigram"].items()}

    # per-piece (held-out style) log-prob of real pieces under each model -> calibration
    from classicalize.theory import EraHarmonyModel
    M = EraHarmonyModel(out)
    calib = defaultdict(list)
    for r in res:
        if len(r["tokens"]) >= 8:
            calib[r["era"]].append(M.seq_logp(r["era"], r["mode"], r["tokens"]))
    for era, v in calib.items():
        out["eras"][era]["stats"]["lm_logp_median"] = float(np.median(v))
        out["eras"][era]["stats"]["lm_logp_p10"] = float(np.percentile(v, 10))

    # keep the per-piece chord sequences: train.py uses them to calibrate the
    # classical-vs-pop likelihood ratio
    out["pieces"] = [{"era": r["era"], "mode": r["mode"], "tokens": r["tokens"][:200]}
                     for r in res if len(r["tokens"]) >= 8]
    os.makedirs("models", exist_ok=True)
    with open("models/era_models.json", "w") as f:
        json.dump(out, f)
    print(json.dumps({e: out["eras"][e]["stats"] for e in out["eras"]}, indent=1))
    print("n_pieces", out["n_pieces"])
    for era in out["eras"]:
        top = Counter(out["eras"][era]["major"]["unigram"]).most_common(10)
        print(era, "major top:", [(t, round(n)) for t, n in top])


if __name__ == "__main__":
    main()
