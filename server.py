#!/usr/bin/env python3
"""Classicalize – local web app.   python server.py  ->  http://localhost:8765"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
import uuid

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from classicalize import pipeline  # noqa: E402
from classicalize.composers import catalog  # noqa: E402

app = FastAPI(title="Classicalize")
JOBS: dict[str, dict] = {}
Q: "queue.Queue[str]" = queue.Queue()
UPLOADS = os.path.join(ROOT, "output", "_uploads")


class JobReq(BaseModel):
    query: str = ""
    video_url: str | None = None
    upload_id: str | None = None
    dir: str | None = None            # re-make an already analysed clip
    s: float = 0.5                    # slider: 0 = original … 1 = classical
    era: str = "auto"
    composer: str = "auto"
    ensemble: str = "auto"


class SearchReq(BaseModel):
    query: str


def warm_up():
    """Load the heavy things once, in the background, so the first song is fast too."""
    try:
        from classicalize import separate
        separate.warm_up()
        from classicalize.render import _synth, find_soundfont
        _synth(find_soundfont())
        from classicalize.theory import era_model, pop_model
        era_model(), pop_model()
        from classicalize import transcribe
        transcribe.warm_up()
    except Exception:  # noqa: BLE001
        traceback.print_exc()


def worker():
    while True:
        jid = Q.get()
        job = JOBS[jid]
        job["status"] = "running"
        req = job["req"]

        def progress(msg, *_):
            job["log"].append({"t": round(time.time() - job["created"], 1), "msg": msg})

        try:
            local = os.path.join(UPLOADS, req["upload_id"]) if req.get("upload_id") else None
            if req.get("dir"):
                res = pipeline.remake(req["dir"], s=req["s"], era=req["era"], composer=req["composer"],
                                      ensemble=req["ensemble"], progress=progress)
            else:
                res = pipeline.run(req["query"], s=req["s"], era=req["era"], composer=req["composer"],
                                   ensemble=req["ensemble"], progress=progress, video_url=req.get("video_url"),
                                   local_wav=local)
            job["dir"] = os.path.basename(res["out_dir"])
            job["status"] = "done"
            progress(f"Done in {res['seconds']:.0f}s ✓")
        except Exception as e:  # noqa: BLE001
            job["status"] = "error"
            job["error"] = f"{e}"
            traceback.print_exc()
            progress(f"Error: {e}")
        Q.task_done()


threading.Thread(target=worker, daemon=True).start()
threading.Thread(target=warm_up, daemon=True).start()


@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "static", "index.html"))


@app.get("/api/status")
def status():
    from classicalize.render import find_soundfont
    from classicalize.separate import demucs_available
    from classicalize.transcribe import basic_pitch_available
    return {"demucs": demucs_available(), "basic_pitch": basic_pitch_available(),
            "soundfont": os.path.basename(find_soundfont())}


@app.get("/api/catalog")
def api_catalog():
    return catalog()


@app.get("/api/training")
def training():
    p = os.path.join(ROOT, "runs", "report.json")
    return json.load(open(p)) if os.path.exists(p) else {}


@app.post("/api/search")
def api_search(r: SearchReq):
    from classicalize.fetch import search
    return {"results": search(r.query)[:6]}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    os.makedirs(UPLOADS, exist_ok=True)
    base = os.path.splitext(os.path.basename(file.filename or "song"))[0]
    uid = uuid.uuid4().hex[:8] + "_" + "".join(ch if ch.isalnum() else "_" for ch in base)[:50]
    raw = os.path.join(UPLOADS, uid + ".src")
    with open(raw, "wb") as f:
        f.write(await file.read())
    wav = os.path.join(UPLOADS, uid + ".wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-t", "360", "-ar", "44100", "-ac", "2", wav],
                   check=True)
    os.remove(raw)
    return {"upload_id": uid + ".wav", "name": file.filename}


@app.post("/api/jobs")
def create_job(r: JobReq):
    if not (r.query.strip() or r.video_url or r.upload_id or r.dir):
        raise HTTPException(400, "Type a song name")
    jid = uuid.uuid4().hex[:10]
    JOBS[jid] = {"id": jid, "req": r.model_dump(), "status": "queued", "log": [], "created": time.time()}
    Q.put(jid)
    return {"id": jid, "position": Q.qsize()}


@app.get("/api/jobs/{jid}")
def get_job(jid: str, since: int = 0):
    j = JOBS.get(jid)
    if not j:
        raise HTTPException(404)
    return JSONResponse({"id": jid, "status": j["status"], "log": j["log"][since:], "n_log": len(j["log"]),
                         "dir": j.get("dir"), "error": j.get("error"),
                         "elapsed": round(time.time() - j["created"], 1)})


@app.get("/api/library")
def library():
    out = []
    base = pipeline.OUT_ROOT
    if os.path.isdir(base):
        for d in sorted(os.listdir(base), key=lambda x: -os.path.getmtime(os.path.join(base, x))):
            rp = os.path.join(base, d, "result.json")
            if os.path.exists(rp):
                try:
                    r = json.load(open(rp))
                    out.append({"dir": d, "title": r.get("title"), "era": r["params"]["era"], "slider": r.get("slider"),
                                "composer": r["params"].get("composer"), "ensemble": r["params"]["ensemble"],
                                "score": r["total"]["score"], "seconds": r.get("seconds")})
                except Exception:  # noqa: BLE001
                    pass
    return {"items": out[:60]}


@app.get("/api/result/{d}")
def result(d: str):
    p = os.path.join(pipeline.OUT_ROOT, os.path.basename(d), "result.json")
    if not os.path.exists(p):
        raise HTTPException(404)
    r = json.load(open(p))
    r["dir"] = os.path.basename(d)
    r.pop("out_dir", None)
    r["mp3_ready"] = os.path.exists(os.path.join(pipeline.OUT_ROOT, os.path.basename(d),
                                                 r.get("files", {}).get("mp3", "classical.mp3")))
    return r


@app.get("/files/{d}/{name}")
def files(d: str, name: str):
    p = os.path.join(pipeline.OUT_ROOT, os.path.basename(d), os.path.basename(name))
    if not os.path.exists(p):
        raise HTTPException(404)
    return FileResponse(p)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8765))
    print(f"\n  Classicalize running →  http://localhost:{port}\n")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
