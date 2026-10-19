#!/usr/bin/env python3
"""Accuracy of the audio analysis on generated songs with known ground truth:
tempo, key, chord roots per beat, and melody pitch per 16th note."""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classicalize.analysis import analyse  # noqa: E402
from classicalize.synthpop import GENRES, make_song, render_pop_stems  # noqa: E402
from classicalize.theory import Chord  # noqa: E402


def bench(n=30):
    rows = []
    genres = list(GENRES)
    for i in range(n):
        g = genres[i % len(genres)]
        truth = make_song(5000 + i, g)
        d = tempfile.mkdtemp(prefix="bench_")
        mix, stems = render_pop_stems(truth, d)
        a = analyse(stems, mix)
        per = 60 / truth.tempo
        acc = []
        for k, bt in enumerate(a.beat_times):
            tb = int(round(bt / per))
            if 0 <= tb < truth.n_beats and a.chords[k]:
                acc.append(Chord(*a.chords[k]).root == truth.chords[tb][0])

        def roll(notes, bts=None):
            out = {}
            for nn in notes:
                for k in np.arange(nn.start, nn.start + nn.dur, 0.25):
                    t = k * per if bts is None else np.interp(k, np.arange(len(bts)), bts)
                    out[round(t / (per / 4))] = nn.pitch % 12
            return out
        rt, ra = roll(truth.melody), roll(a.melody, a.beat_times)
        common = set(rt) & set(ra)
        rows.append({"genre": g, "tempo_ok": abs(a.tempo / truth.tempo - 1) < 0.04,
                     "key_ok": (a.tonic, a.mode) == (truth.tonic, truth.mode),
                     "key_or_relative": (a.tonic, a.mode) == (truth.tonic, truth.mode) or
                     (a.tonic == (truth.tonic + (9 if truth.mode == "major" else 3)) % 12),
                     "chord_root_acc": float(np.mean(acc)) if acc else 0.0,
                     "melody_pc_acc": float(np.mean([rt[k] == ra[k] for k in common])) if common else 0.0,
                     "melody_recall": len(common) / max(1, len(rt))})
        print(i, rows[-1])
    keys = [k for k in rows[0] if k != "genre"]
    summ = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    print("\nSUMMARY", json.dumps(summ, indent=1))
    os.makedirs("runs", exist_ok=True)
    json.dump({"summary": summ, "rows": rows}, open("runs/bench_analysis.json", "w"), indent=1)


if __name__ == "__main__":
    bench(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
