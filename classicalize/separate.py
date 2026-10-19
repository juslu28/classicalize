"""Split the recording into vocals / bass / drums / other with Demucs (htdemucs).

Speed choices (this is the slowest step of the whole pipeline):
  * model loaded once and kept in memory between requests
  * Apple-Silicon GPU (MPS) when available, CPU fallback
  * shifts=0 and overlap=0.1 (vs 0.25): ~20% faster with negligible quality loss
  * stems are returned as arrays (no WAV files written/re-read)
  * songs are capped at 6 minutes
Without Demucs installed it falls back to a rough HPSS split.
"""
from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
STEMS = ("vocals", "bass", "drums", "other")
ANALYSIS_SR = 22050
MAX_SECONDS = 360
_MODEL = {}


def demucs_available():
    try:
        import demucs  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _model():
    if "m" not in _MODEL:
        import torch
        from demucs.pretrained import get_model
        m = get_model("htdemucs")
        m.eval()
        dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        _MODEL["m"], _MODEL["dev"] = m.to(dev), dev
    return _MODEL["m"], _MODEL["dev"]


def warm_up():
    """Load the model in the background when the server starts."""
    if demucs_available():
        try:
            _model()
        except Exception:  # noqa: BLE001
            pass


def _resample(y, sr):
    import librosa
    return librosa.resample(y, orig_sr=sr, target_sr=ANALYSIS_SR, res_type="soxr_hq").astype(np.float32)


def separate(wav_path: str):
    """Returns (stems dict of mono float32 arrays @22.05 kHz, mix array @22.05 kHz)."""
    import soundfile as sf
    y, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    y = y[: int(MAX_SECONDS * sr)]
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)
    mix = _resample(y.mean(axis=1), sr)
    if not demucs_available():
        return _hpss_fallback(mix), mix
    import torch
    from demucs.apply import apply_model
    model, dev = _model()
    x = torch.from_numpy(np.ascontiguousarray(y.T))
    ref = x.mean(0)
    mu, sd = ref.mean(), ref.std() + 1e-8
    x = (x - mu) / sd
    for d in dict.fromkeys([dev, "cpu"]):
        try:
            with torch.no_grad():
                out = apply_model(model.to(d), x[None], shifts=0, split=True, overlap=0.1, device=d,
                                  progress=False)[0]
            if d != dev:
                _MODEL["dev"] = d       # remember that the GPU path failed
            break
        except Exception:  # noqa: BLE001
            if d == "cpu":
                raise
    out = (out * sd + mu).cpu().numpy()          # (sources, 2, n)
    stems = {name: _resample(out[i].mean(axis=0), sr) for i, name in enumerate(model.sources)}
    return {k: stems[k] for k in STEMS}, mix


def _hpss_fallback(y):
    import librosa
    H, P = librosa.effects.hpss(y)
    S = librosa.stft(H)
    f = librosa.fft_frequencies(sr=ANALYSIS_SR)
    lo = librosa.istft(S * (f[:, None] < 200), length=len(y))
    mid = librosa.istft(S * ((f[:, None] >= 200) & (f[:, None] < 3000)), length=len(y))
    return {"vocals": mid.astype(np.float32), "bass": lo.astype(np.float32), "other": H.astype(np.float32),
            "drums": P.astype(np.float32)}
