"""End-to-end: song request -> the song's most popular ~20 s, played classically.

  search + download (yt-dlp, with YouTube's "most replayed" data)
  pick the most popular ~20 s, snapped to whole bars, and cut it out
  separate the clip (Demucs) and transcribe every part (Basic Pitch + YIN + drum onsets)
  composer / ensemble search (a few seconds)
  blend original transcription ↔ classical arrangement by the slider
  record with sampled instruments, exactly as long as the clip
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import asdict

from . import blend, explain, judge
from .analysis import SongAnalysis, analyse
from .clip import choose_window, cut
from .fetch import download_audio, safe_name, search
from .optimize import search as style_search
from .render import render
from .separate import demucs_available, separate
from .transcribe import TNote, transcribe

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT = os.path.join(ROOT, "output")
CLIP_SECONDS = 20.0


class Timer:
    def __init__(self, progress):
        self.t0 = self.last = time.time()
        self.steps, self.progress, self._cur = {}, progress, None

    def step(self, name, msg=None):
        self.done()
        if msg:
            self.progress(msg)
        self._cur, self.last = name, time.time()

    def done(self):
        if self._cur:
            self.steps[self._cur] = round(self.steps.get(self._cur, 0) + time.time() - self.last, 2)
        self._cur = None

    def total(self):
        return round(time.time() - self.t0, 1)


def prepare(query=None, video_url=None, local_wav=None, progress=lambda *_: None, timer=None):
    """-> (song analysis of the clip, transcribed notes, drum hits, out_dir, meta). Cached."""
    T = timer or Timer(progress)
    if local_wav:
        name = os.path.basename(local_wav).split("_", 1)[-1].rsplit(".", 1)[0]
        out_dir = os.path.join(OUT_ROOT, safe_name(query or name) + "_upload")
        os.makedirs(out_dir, exist_ok=True)
        wav = os.path.join(out_dir, "original.wav")
        if not os.path.exists(wav):
            shutil.copy(local_wav, wav)
        meta = {"title": query or name, "source": "uploaded file"}
        with open(os.path.join(out_dir, "source.json"), "w") as f:
            json.dump(meta, f)
    else:
        meta = {}
        if not video_url:
            T.step("search", "Searching YouTube for the most popular recording…")
            res = [r for r in search(query) if r["score"] > 0] or search(query)
            if not res:
                raise RuntimeError("No YouTube results")
            meta = dict(res[0])
            video_url = meta["url"]
        else:
            meta = {"url": video_url, "title": query or video_url}
        vid = video_url.split("v=")[-1][:11]
        out_dir = os.path.join(OUT_ROOT, f"{safe_name(meta.get('title') or query)}_{vid}")
        os.makedirs(out_dir, exist_ok=True)
        wav = os.path.join(out_dir, "original.wav")
        mp = os.path.join(out_dir, "source.json")
        if not os.path.exists(wav):
            T.step("download", f"Downloading “{meta.get('title')}”")
            _, extra = download_audio(video_url, out_dir)
            meta.update({k: v for k, v in extra.items() if v})
            with open(mp, "w") as f:
                json.dump(meta, f)
        elif os.path.exists(mp):
            meta = {**json.load(open(mp)), **{k: v for k, v in meta.items() if v}}
    cache = os.path.join(out_dir, "clip_analysis.json")
    if os.path.exists(cache):
        progress("Using the cached analysis of this clip")
        d = json.load(open(cache))
        song = SongAnalysis.from_json(json.dumps(d["song"]))
        notes = [TNote(**n) for n in d["notes"]]
        hits = [tuple(h) for h in d["hits"]]
        meta["window"] = d["window"]
        T.done()
        return song, notes, hits, out_dir, meta
    T.step("pick", "Finding the most popular 20 seconds…")
    win = choose_window(wav, meta.get("heatmap"), CLIP_SECONDS)
    clip = os.path.join(out_dir, "clip.wav")
    cut(wav, win["start"], win["end"], clip)
    meta["window"] = {k: win[k] for k in ("start", "end", "reason", "tempo")} | {
        "song_duration": win.get("song_duration")}
    T.step("separate", "Separating vocals, bass, drums and other instruments…")
    stems, mix = separate(clip)
    T.step("analyse", "Finding the key, chords and melody…")
    rel_beats = [b - win["start"] for b in win["beats"]] if win["beats"] else None
    song = analyse(stems, mix, title=meta.get("title", ""), beat_times=rel_beats)
    T.step("transcribe", "Transcribing every note (Basic Pitch)…")
    notes, hits = transcribe(stems, song, separated=demucs_available())
    if not demucs_available():
        # no separated vocal stem: take the tune from the transcription's top line instead
        from .analysis import Note
        from .blend import _s2b_factory
        s2b = _s2b_factory(song.beat_times)
        song.melody = [Note(round(s2b(n.start) * 4) / 4, max(0.25, round((s2b(n.end) - s2b(n.start)) * 4) / 4),
                            n.pitch, int(50 + 60 * n.amp)) for n in notes if n.role == "vocals"]
    T.done()
    with open(cache, "w") as f:
        json.dump({"song": json.loads(song.to_json()), "notes": [asdict(n) for n in notes],
                   "hits": hits, "window": meta["window"]}, f)
    return song, notes, hits, out_dir, meta


def _mp3_later(wav, mp3):
    def job():
        if shutil.which("ffmpeg"):
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-b:a", "192k", mp3])
    threading.Thread(target=job, daemon=True).start()


def make(song, notes, hits, out_dir, meta, s=0.5, era="auto", composer="auto", ensemble="auto",
         progress=lambda *_: None, timer=None):
    T = timer or Timer(progress)
    T.step("styles", "Choosing the composer and instruments…")
    best, trials, info = style_search(song, w=s, era=era, composer=composer, ensemble=ensemble,
                                      slider=s, budget_s=2.5)
    T.step("arrange", f"Arranging in the style of {best.composer.title()} ({best.ensemble.replace('_', ' ')})")
    arr, binfo = blend.build(song, notes, hits, best, s)
    T.step("render", "Recording it with classical instruments…")
    tag = f"{int(round(s * 100)):03d}"
    wav = os.path.join(out_dir, f"classical_{tag}.wav")
    render(arr, wav, duration=song.duration)
    _mp3_later(wav, wav[:-4] + ".mp3")
    T.done()
    C = judge.classicalness(arr)
    R = judge.retention(song, arr)
    tot = judge.total(C, R, s)
    review = judge.period_review(C, R, arr)
    details = explain.build(song, arr, info, C, R, tot, review, meta, blend_info=binfo)
    result = {
        "title": meta.get("title") or song.title, "source": {k: v for k, v in meta.items() if k != "heatmap"},
        "params": arr.params, "slider": s, "classicalness": C, "retention": R, "total": tot,
        "review": review, "details": details, "window": meta.get("window"),
        "files": {"classical": os.path.basename(wav), "mp3": os.path.basename(wav)[:-4] + ".mp3",
                  "midi": os.path.basename(wav)[:-4] + ".mid", "original": "clip.wav"},
        "durations": {"original": round(song.duration, 2), "classical": round(song.duration, 2)},
        "timings": dict(T.steps), "seconds": T.total(),
    }
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=1, default=str)
    return result


def remake(dirname, s=0.5, era="auto", composer="auto", ensemble="auto", progress=lambda *_: None):
    """re-arrange an already analysed clip with new settings (a few seconds)."""
    out_dir = os.path.join(OUT_ROOT, os.path.basename(dirname))
    d = json.load(open(os.path.join(out_dir, "clip_analysis.json")))
    song = SongAnalysis.from_json(json.dumps(d["song"]))
    notes = [TNote(**n) for n in d["notes"]]
    hits = [tuple(h) for h in d["hits"]]
    mp = os.path.join(out_dir, "source.json")
    meta = json.load(open(mp)) if os.path.exists(mp) else {"title": song.title}
    meta["window"] = d["window"]
    T = Timer(progress)
    res = make(song, notes, hits, out_dir, meta, s, era, composer, ensemble, progress, T)
    res["out_dir"] = out_dir
    return res


def run(query, s=0.5, era="auto", composer="auto", ensemble="auto", progress=lambda *_: None, video_url=None,
        local_wav=None, w=None):
    if w is not None:
        s = w
    T = Timer(progress)
    song, notes, hits, out_dir, meta = prepare(query, video_url, local_wav, progress, T)
    res = make(song, notes, hits, out_dir, meta, s, era, composer, ensemble, progress, T)
    res["out_dir"] = out_dir
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    return res
