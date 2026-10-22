#!/usr/bin/env python3
"""
Listening POC v0.2 — the classical ear, now in stereo.

Everything v0.1 did (per-second mel + chroma + centroid + RMS, tempo, key,
novelty sections) computed the same way for comparability, PLUS:

  - Mid/Side stereo analysis: per-second stereo width (Side/Mid energy, dB),
    inter-channel correlation, and width by frequency band (low/mid/high).
    v0.1 folded stereo to mono, which made it blind to stereo-splitter and
    widening effects (the Weight first-chorus blind spot). v0.2 measures the
    channels as delivered.
  - Per-section key estimates (Krumhansl per section), so modulations are
    read section by section instead of smeared into one whole-track answer.

Outputs (unchanged shape, extended):
  - <name>-features.npy   per-second matrix: 64 log-mel + 12 chroma +
                          centroid + rms + width_db + corr + 3 band widths
                          (83 dims; seconds with no usable signal carry NaN
                          in the stereo columns only)
  - <name>-summary.json    v0.1 fields + "stereo" block + per-section keys

Usage: python3 listen.py <wav_path> <out_dir> <name>
"""
import sys, os, json
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, resample_poly
from math import gcd

WAV, OUT, NAME = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)

TARGET_SR = 22050

# ---------- load (keep the channels this time) ----------
sr, data = wavfile.read(WAV)
if data.dtype == np.int16:
    raw = data.astype(np.float32) / 32768.0
elif data.dtype == np.int32:
    raw = data.astype(np.float32) / 2147483648.0
else:
    raw = data.astype(np.float32)
is_stereo = raw.ndim == 2 and raw.shape[1] >= 2

def to_target(x, sr_in):
    if sr_in == TARGET_SR:
        return x
    g = gcd(sr_in, TARGET_SR)
    return resample_poly(x, TARGET_SR // g, sr_in // g).astype(np.float32)

if is_stereo:
    L = to_target(raw[:, 0], sr)
    R = to_target(raw[:, 1], sr)
    n = min(len(L), len(R))
    L, R = L[:n], R[:n]
    y = (L + R) / 2.0          # the fold v0.1 heard — kept for continuity
    mid, side = (L + R) / 2.0, (L - R) / 2.0
else:
    y = to_target(raw if raw.ndim == 1 else raw[:, 0], sr)
    L = R = mid = side = None
sr = TARGET_SR
duration_s = len(y) / sr

# ---------- STFT (mono fold, same as v0.1) ----------
NFFT, HOP = 2048, 512
freqs, times, Z = stft(y, fs=sr, nperseg=NFFT, noverlap=NFFT - HOP, boundary=None)
S = np.abs(Z)
fps = sr / HOP

# ---------- mel filterbank (64 bands) ----------
def mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
def inv_mel(m): return 700.0 * (10 ** (m / 2595.0) - 1.0)
mel_pts = np.linspace(mel(20), mel(sr / 2), 66)
hz_pts = inv_mel(mel_pts)
bin_pts = np.floor((NFFT + 1) * hz_pts / sr).astype(int)
fb = np.zeros((64, len(freqs)))
for i in range(64):
    l, c, r = bin_pts[i], bin_pts[i + 1], bin_pts[i + 2]
    if c > l: fb[i, l:c] = (np.arange(l, c) - l) / (c - l)
    if r > c: fb[i, c:r] = (r - np.arange(c, r)) / (r - c)
mel_spec = fb @ S

# ---------- chroma ----------
with np.errstate(divide="ignore"):
    midi = 69 + 12 * np.log2(np.where(freqs > 0, freqs, 1.0) / 440.0)
pc = np.round(midi).astype(int) % 12
chroma = np.zeros((12, S.shape[1]))
for p in range(12):
    rows = np.where(pc == p)[0]
    rows = rows[freqs[rows] > 30]
    if len(rows): chroma[p] = S[rows].sum(axis=0)
chroma /= (chroma.sum(axis=0, keepdims=True) + 1e-9)

# ---------- spectral centroid ----------
centroid = (freqs[:, None] * S).sum(axis=0) / (S.sum(axis=0) + 1e-9)

# ---------- per-second aggregation (v0.1 columns) ----------
n_sec = int(duration_s)
frame_sec = np.floor(times).astype(int)
feat = np.zeros((n_sec, 64 + 12 + 2), dtype=np.float32)
rms_sec = np.zeros(n_sec, dtype=np.float32)
for sIdx in range(n_sec):
    m = frame_sec == sIdx
    if m.any():
        feat[sIdx, :64] = np.log1p(mel_spec[:, m]).mean(axis=1)
        feat[sIdx, 64:76] = chroma[:, m].mean(axis=1)
        feat[sIdx, 76] = centroid[m].mean()
    seg = y[sIdx * sr:(sIdx + 1) * sr]
    rms_sec[sIdx] = np.sqrt(np.mean(seg ** 2)) if len(seg) else 0.0
feat[:, 77] = rms_sec

# ---------- stereo features (v0.2) ----------
STEREO_DIMS = 5  # width_db, corr, band_low_db, band_mid_db, band_high_db
stereo_feat = np.full((n_sec, STEREO_DIMS), np.nan, dtype=np.float32)
if is_stereo:
    # band energies from mid/side spectra
    _, _, Zm = stft(mid, fs=sr, nperseg=NFFT, noverlap=NFFT - HOP, boundary=None)
    _, _, Zs = stft(side, fs=sr, nperseg=NFFT, noverlap=NFFT - HOP, boundary=None)
    Sm, Ss = np.abs(Zm) ** 2, np.abs(Zs) ** 2
    band_masks = {
        "low": freqs < 250,
        "mid": (freqs >= 250) & (freqs < 4000),
        "high": freqs >= 4000,
    }
    band_ratio = {}
    for bname, mask in band_masks.items():
        em = Sm[mask].sum(axis=0) + 1e-12
        es = Ss[mask].sum(axis=0) + 1e-12
        band_ratio[bname] = 10 * np.log10(es / em)  # per frame, dB
    for sIdx in range(n_sec):
        segL = L[sIdx * sr:(sIdx + 1) * sr]
        segR = R[sIdx * sr:(sIdx + 1) * sr]
        if not len(segL):
            continue
        segM = (segL + segR) / 2.0
        segS = (segL - segR) / 2.0
        rms_m = np.sqrt(np.mean(segM ** 2)) + 1e-9
        rms_s = np.sqrt(np.mean(segS ** 2)) + 1e-9
        if rms_m < 1e-4:  # ~-80 dBFS: no usable signal this second
            continue
        stereo_feat[sIdx, 0] = 20 * np.log10(rms_s / rms_m)
        if segL.std() > 1e-6 and segR.std() > 1e-6:
            stereo_feat[sIdx, 1] = np.corrcoef(segL, segR)[0, 1]
        m = frame_sec == sIdx
        if m.any():
            stereo_feat[sIdx, 2] = np.mean(band_ratio["low"][m])
            stereo_feat[sIdx, 3] = np.mean(band_ratio["mid"][m])
            stereo_feat[sIdx, 4] = np.mean(band_ratio["high"][m])
feat = np.hstack([feat, stereo_feat]).astype(np.float32)

# ---------- tempo (unchanged) ----------
flux = np.maximum(0, np.diff(S, axis=1)).sum(axis=0)
flux = flux - flux.mean()
ac = np.correlate(flux, flux, mode="full")[len(flux) - 1:]
lag_min = int(60 / 200 * fps); lag_max = int(60 / 40 * fps)
seg_ac = ac[lag_min:lag_max]
best_lag = lag_min + int(np.argmax(seg_ac))
tempo = 60.0 / (best_lag / fps)
top = np.argsort(seg_ac)[::-1][:4]
tempo_candidates = sorted({round(60.0 / ((lag_min + t) / fps), 1) for t in top})

# ---------- key (Krumhansl-Schmuckler; whole-track + per section) ----------
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

def estimate_key(mean_chroma):
    best = (-2, None, None)
    for root in range(12):
        for prof, mode in ((MAJ, "major"), (MIN, "minor")):
            c = np.corrcoef(np.roll(prof, root), mean_chroma)[0, 1]
            if c > best[0]: best = (float(c), NOTES[root], mode)
    return best

key_conf, key_root, key_mode = estimate_key(feat[:, 64:76].mean(axis=0))

# ---------- sections (novelty, unchanged basis: first 76 dims) ----------
f_norm = feat[:, :76].copy()
f_norm = (f_norm - f_norm.mean(axis=0)) / (f_norm.std(axis=0) + 1e-9)
W = 5
nov = np.zeros(n_sec)
for t in range(W, n_sec - W):
    a = f_norm[t - W:t].mean(axis=0); b = f_norm[t:t + W].mean(axis=0)
    nov[t] = 1 - float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
thr = nov.mean() + 0.5 * nov.std()
bounds = [0]
last = -12
for t in range(W, n_sec - W):
    if nov[t] > thr and nov[t] == nov[max(0, t - 3):t + 4].max() and t - last >= 10:
        bounds.append(int(t)); last = t
bounds.append(n_sec)
rms_db = 20 * np.log10(rms_sec + 1e-6)

def nanmean_or_none(x):
    x = np.asarray(x, dtype=np.float64)
    if np.all(np.isnan(x)):
        return None
    return round(float(np.nanmean(x)), 2)

sections = []
for i in range(len(bounds) - 1):
    a, b = bounds[i], bounds[i + 1]
    sec_chroma = feat[a:b, 64:76].mean(axis=0)
    s_conf, s_root, s_mode = estimate_key(sec_chroma)
    sec = {
        "start_s": a, "end_s": b,
        "energy_db": round(float(rms_db[a:b].mean()), 1),
        "dominant_note": NOTES[int(np.argmax(sec_chroma))],
        "key": f"{s_root} {s_mode}", "key_correlation": round(s_conf, 3),
    }
    if is_stereo:
        sec["width_db"] = nanmean_or_none(feat[a:b, 78])
        sec["corr"] = nanmean_or_none(feat[a:b, 79])
    sections.append(sec)

# ---------- energy arc ----------
arc = []
for t in range(0, n_sec, 10):
    arc.append({"t": t, "rms_db": round(float(rms_db[t:t + 10].mean()), 1)})

# ---------- stereo summary ----------
stereo_summary = {"is_stereo": bool(is_stereo)}
if is_stereo:
    width = feat[:, 78]; corr = feat[:, 79]
    med_width = float(np.nanmedian(width)) if not np.all(np.isnan(width)) else None
    med_corr = float(np.nanmedian(corr)) if not np.all(np.isnan(corr)) else None
    stereo_summary.update({
        "mean_width_db": nanmean_or_none(width),
        "median_width_db": round(med_width, 2) if med_width is not None else None,
        "mean_corr": nanmean_or_none(corr),
        "min_corr": (round(float(np.nanmin(corr)), 2)
                     if not np.all(np.isnan(corr)) else None),
        "band_width_db": {
            "low": nanmean_or_none(feat[:, 80]),
            "mid": nanmean_or_none(feat[:, 81]),
            "high": nanmean_or_none(feat[:, 82]),
        },
        "width_arc_10s": [
            {"t": t, "width_db": nanmean_or_none(width[t:t + 10])}
            for t in range(0, n_sec, 10)
        ],
    })
    flags = []
    for i, sec in enumerate(sections):
        reasons = []
        if sec.get("width_db") is not None and med_width is not None \
                and abs(sec["width_db"] - med_width) >= 4.0:
            reasons.append(f"width {sec['width_db']} dB vs track median {round(med_width, 1)} dB")
        if sec.get("corr") is not None and med_corr is not None \
                and sec["corr"] <= med_corr - 0.35:
            reasons.append(f"correlation {sec['corr']} vs track median {round(med_corr, 2)}")
        if reasons:
            flags.append({"section": i, "start_s": sec["start_s"],
                          "end_s": sec["end_s"], "reasons": reasons})
    stereo_summary["anomalies"] = flags

# ---------- save ----------
def sanitize(obj):
    if isinstance(obj, dict): return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list): return [sanitize(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj): return None
    return obj

npy_path = os.path.join(OUT, f"{NAME}-features.npy")
np.save(npy_path, feat)
summary = {
    "track": NAME, "source": os.path.basename(WAV),
    "ear_version": "0.2",
    "duration_s": round(float(duration_s), 2),
    "tempo_bpm_top": round(float(tempo), 1),
    "tempo_candidates_bpm": tempo_candidates,
    "key": f"{key_root} {key_mode}", "key_correlation": round(key_conf, 3),
    "mean_rms_db": round(float(rms_db.mean()), 1),
    "dynamic_range_db": round(float(rms_db.max() - rms_db.min()), 1),
    "mean_centroid_hz": round(float(feat[:, 76].mean()), 0),
    "sections": sections,
    "energy_arc_10s": arc,
    "stereo": stereo_summary,
    "feature_matrix": {"shape": list(feat.shape),
                       "dims": "64 log-mel + 12 chroma + centroid + rms + width_db + corr + band widths (low/mid/high), 1 row/sec"},
    "sizes_bytes": {
        "wav": os.path.getsize(WAV),
        "embedding_npy": os.path.getsize(npy_path),
    },
}
summary = sanitize(summary)
with open(os.path.join(OUT, f"{NAME}-summary.json"), "w") as f:
    json.dump(summary, f, indent=1)
print(json.dumps(summary, indent=1))
