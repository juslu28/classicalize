"""Pick the most popular ~20 seconds of the song and cut it out.

1st choice: YouTube's own "Most replayed" heat-map (what listeners actually
            replay), when the video has one.
fallback  : the loudest, most-repeated passage (almost always the chorus).
The window is snapped to whole bars so the music starts and ends cleanly.
"""
from __future__ import annotations

import numpy as np

SR = 22050


def _beats(y):
    import librosa
    oenv = librosa.onset.onset_strength(y=y, sr=SR, hop_length=512, aggregate=np.median)
    tempo, beats = librosa.beat.beat_track(onset_envelope=oenv, sr=SR, hop_length=512, start_bpm=110,
                                           tightness=120, units="time")
    tempo = float(np.atleast_1d(tempo)[0])
    if tempo > 165 and len(beats) > 8:
        beats, tempo = beats[::2], tempo / 2
    if tempo < 62 and len(beats) > 8:
        beats = np.sort(np.concatenate([beats, (beats[:-1] + beats[1:]) / 2]))
        tempo *= 2
    return np.asarray(beats), tempo, oenv


def choose_window(wav_path, heatmap=None, target=20.0):
    """-> dict(start, end, beats (absolute times inside the window), reason)."""
    import librosa
    y, _ = librosa.load(wav_path, sr=SR, mono=True, res_type="soxr_hq")
    dur = len(y) / SR
    beats, tempo, oenv = _beats(y)
    if len(beats) < 16:
        s = max(0.0, min(dur - target, dur * 0.4))
        return {"start": s, "end": min(dur, s + target), "beats": [], "reason": "middle of the song",
                "tempo": tempo}
    period = float(np.median(np.diff(beats)))
    n_beats = max(16, int(round(target / period / 4)) * 4)          # whole 4/4 bars ≈ 20 s
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    t_rms = librosa.times_like(rms, sr=SR, hop_length=512)

    hm = None
    if heatmap:
        hm = [(float(h["start_time"]), float(h["end_time"]), float(h.get("value", 0))) for h in heatmap]

    # downbeat phase: beat index with the strongest low-end onsets, modulo 4
    low = librosa.onset.onset_strength(y=y, sr=SR, hop_length=512, fmax=200)
    fr = np.clip(librosa.time_to_frames(beats, sr=SR, hop_length=512), 0, len(low) - 1)
    phase = int(np.argmax([low[fr[k::4]].mean() for k in range(4)]))

    C = librosa.feature.chroma_stft(y=y, sr=SR, hop_length=2048)
    tc = librosa.times_like(C, sr=SR, hop_length=2048)

    def rep_score(a, b):
        """how much this passage recurs elsewhere in the song (choruses repeat)."""
        i0, i1 = np.searchsorted(tc, a), np.searchsorted(tc, b)
        seg = C[:, i0:i1]
        if seg.shape[1] < 4:
            return 0.0
        L = seg.shape[1]
        best = 0.0
        sn = seg / (np.linalg.norm(seg, axis=0, keepdims=True) + 1e-9)
        for j in range(0, C.shape[1] - L, 4):
            if abs(j - i0) < L:
                continue
            o = C[:, j:j + L]
            on = o / (np.linalg.norm(o, axis=0, keepdims=True) + 1e-9)
            best = max(best, float((sn * on).sum(axis=0).mean()))
        return best

    cands = []
    for k in range(phase, len(beats) - n_beats, 4):
        a, b = beats[k], beats[k + n_beats]
        if b > dur:
            break
        loud = float(rms[(t_rms >= a) & (t_rms < b)].mean())
        cands.append((k, a, b, loud))
    if not cands:
        return {"start": 0.0, "end": min(dur, target), "beats": [], "reason": "start of the song", "tempo": tempo}
    lmax = max(c[3] for c in cands) + 1e-9
    scored = []
    for k, a, b, loud in cands:
        if hm:
            v = [val for s, e, val in hm if e > a and s < b]
            score = float(np.mean(v)) if v else 0.0
        else:
            score = loud / lmax
        scored.append((score, k, a, b, loud))
    scored.sort(reverse=True)
    if not hm:   # refine the top few by how often they repeat (chorus test)
        top = scored[:6]
        scored = sorted(((0.6 * s + 0.4 * rep_score(a, b), k, a, b, l) for s, k, a, b, l in top), reverse=True)
    _, k, a, b, _ = scored[0]
    return {"start": float(a), "end": float(b), "beats": [float(x) for x in beats[k:k + n_beats + 1]],
            "reason": "YouTube's “most replayed” data" if hm else "the loudest, most-repeated passage (the chorus)",
            "tempo": tempo, "song_duration": dur}


def cut(wav_path, start, end, out_path, fade=0.02):
    import soundfile as sf
    info = sf.info(wav_path)
    sr = info.samplerate
    y, _ = sf.read(wav_path, start=int(start * sr), stop=int(end * sr), dtype="float32", always_2d=True)
    n = int(fade * sr)
    if len(y) > 2 * n:
        ramp = np.linspace(0, 1, n)[:, None]
        y[:n] *= ramp
        y[-n:] *= ramp[::-1]
    sf.write(out_path, y, sr, subtype="PCM_16")
    return out_path
