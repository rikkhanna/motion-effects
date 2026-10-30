"""Reference video -> effect spec.

Measures, per frame, the signals that motion-graphics "effects" leave behind and
turns them into the keyframed tracks the engine can replay on any other video:

  zoom / shift / rotate   dominant global motion (feature tracking + RANSAC similarity fit)
  brightness              luma deviation from a rolling-median baseline (flashes, dips)
  contrast / saturation   same idea on luma std and HSV saturation
  blur                    Laplacian sharpness loss, calibrated by re-blurring the sharpest frame
  rgb_split               red/blue channel misregistration (phase correlation)
  glitch                  rows that jump horizontally relative to the previous frame
  vignette, grain         whole-clip statistics (content averages out)

Everything is a heuristic: a reference video mixes the effect with its own
footage, so we use robust statistics and dead-bands to avoid learning content.
"""
import math
from collections import defaultdict

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter, percentile_filter, uniform_filter1d

AW = 320  # analysis width


def _odd(n):
    n = int(max(3, n))
    return n if n % 2 else n + 1


def deadband(x, neutral, eps):
    """Soft threshold: values within eps of neutral snap to neutral, no jump at the edge."""
    d = np.asarray(x, float) - neutral
    return neutral + np.sign(d) * np.maximum(np.abs(d) - eps, 0.0)


def highpass(x, win):
    """Subtract a rolling median: short pulses survive, slow drift and estimator bias go."""
    return x - median_filter(x, size=_odd(win), mode="nearest")


def rdp(ts, vs, eps):
    """Ramer-Douglas-Peucker on a (t, v) polyline; returns indices to keep."""
    n = len(ts)
    keep = {0, n - 1}
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        seg_t = ts[i + 1:j]
        interp = vs[i] + (vs[j] - vs[i]) * (seg_t - ts[i]) / (ts[j] - ts[i])
        err = np.abs(vs[i + 1:j] - interp)
        k = int(err.argmax())
        if err[k] > eps:
            idx = i + 1 + k
            keep.add(idx)
            stack += [(i, idx), (idx, j)]
    return sorted(keep)


def glitch_rows(cur, prev, smax=24):
    """Rows whose content moved horizontally differently from the rest of the frame."""
    a, b = cur, prev
    costs = np.stack([np.abs(a - np.roll(b, s, axis=1))[:, smax:-smax].mean(axis=1)
                      for s in range(-smax, smax + 1)], axis=1)
    c0 = costs[:, smax]
    s = costs.argmin(1) - smax
    cmin = costs.min(1)
    rel = s - np.median(s)  # ignore camera motion that moves every row together
    bad = (np.abs(rel) >= 3) & (cmin < 0.6 * c0) & (c0 > 6)
    return float(bad.mean()), np.abs(rel[bad])


def noise_sigma(prev, cur):
    d = cur - prev
    d = d[np.abs(d) < 25]
    if d.size < 500:
        return float("nan")
    return 1.4826 * np.median(np.abs(d - np.median(d))) / math.sqrt(2)


def analyze(path, name, max_seconds=20.0, motion="shake", grade=False, baseline_s=1.5, log=print):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W0, H0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = max(1, int(round(fps / 30.0)))
    afps = fps / stride
    AH = int(round(H0 * AW / W0 / 2) * 2)
    hann = cv2.createHanningWindow((AW, AH), cv2.CV_32F)

    yy, xx = np.mgrid[0:AH, 0:AW].astype(np.float32)
    r = np.hypot((xx - AW / 2) / (AW / 2), (yy - AH / 2) / (AH / 2)) / math.sqrt(2)
    ring, centre = (r > 0.7) & (r < 0.9), r < 0.25

    rec = defaultdict(list)
    prev_g8 = prev_gf = prev_crop = None
    frames_sharp = []
    grade_frames = []
    cx, cy = AW / 2, AH / 2
    max_frames = int(max_seconds * fps)

    i = -1
    while i + 1 < max_frames:
        ok, f = cap.read()
        i += 1
        if not ok:
            break
        if i % stride:
            continue
        small = cv2.resize(f, (AW, AH), interpolation=cv2.INTER_AREA)
        g8 = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gf = g8.astype(np.float32)

        rec["L"].append(gf.mean() / 255.0)
        rec["std"].append(gf.std() / 255.0)
        rec["sat"].append(cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[..., 1].mean() / 255.0)
        lap = float(cv2.Laplacian(gf, cv2.CV_32F).var())
        rec["sharp"].append(lap)
        frames_sharp.append(gf.copy() if len(frames_sharp) < 600 else None)
        c_mean = gf[centre].mean()
        rec["vig"].append((1.0 - gf[ring].mean() / max(c_mean, 1.0)) / 0.64)
        gm = lambda c: cv2.magnitude(cv2.Sobel(c, cv2.CV_32F, 1, 0), cv2.Sobel(c, cv2.CV_32F, 0, 1))
        (dx, dy), resp = cv2.phaseCorrelate(gm(small[..., 0].astype(np.float32)), gm(small[..., 2].astype(np.float32)), hann)
        ok_pc = resp > 0.15 and abs(dx) < 12 and abs(dy) < 12
        rec["dx"].append(dx if ok_pc else np.nan)
        rec["dy"].append(dy if ok_pc else np.nan)
        if grade and len(rec["L"]) % max(1, int(afps)) == 0:
            grade_frames.append(f)

        # grain from a native-resolution crop
        gray_full = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        ch, cw = gray_full.shape
        crop = gray_full[ch // 2 - min(128, ch // 2):ch // 2 + min(128, ch // 2),
                         cw // 2 - min(128, cw // 2):cw // 2 + min(128, cw // 2)]

        if prev_g8 is None:
            for k in ("zoom", "rot", "tx", "ty", "glitch", "gamp", "grain"):
                rec[k].append(0.0 if k != "zoom" else 1.0)
        else:
            # --- global motion: similarity transform prev -> cur
            s, th, tau = 1.0, 0.0, np.zeros(2)
            p0 = cv2.goodFeaturesToTrack(prev_g8, 300, 0.01, 7)
            if p0 is not None and len(p0) >= 12:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_g8, g8, p0, None)
                st = st.ravel() == 1
                if st.sum() >= 12:
                    A, inl = cv2.estimateAffinePartial2D(p0[st], p1[st], method=cv2.RANSAC, ransacReprojThreshold=2.0)
                    if A is not None and inl.sum() >= 10 and inl.mean() >= 0.4:
                        a, b = A[0, 0], A[1, 0]
                        s, th = math.hypot(a, b), math.atan2(b, a)
                        Rm = np.array([[a, -b], [b, a]])
                        tau = A[:, 2] + Rm @ np.array([cx, cy]) - np.array([cx, cy])  # offset about frame centre
                        if not (0.8 < s < 1.25):
                            s, th, tau = 1.0, 0.0, np.zeros(2)
            rec["zoom"].append(s)
            rec["rot"].append(th)
            rec["tx"].append(tau[0] / AW)
            rec["ty"].append(tau[1] / AH)
            # --- glitch rows
            frac, mags = glitch_rows(gf, prev_gf)
            rec["glitch"].append(frac)
            if mags.size:
                rec["gamp"].append(float(np.median(mags)) / AW)
            # --- grain
            rec["grain"].append(noise_sigma(prev_crop, crop) if prev_crop.shape == crop.shape else float("nan"))
        prev_g8, prev_gf, prev_crop = g8, gf, crop

    N = len(rec["L"])
    if N < 8:
        raise SystemExit("reference video is too short to analyse (need at least ~8 frames)")
    t = np.arange(N) / afps
    dur = N / afps
    log(f"analysed {N} frames @ {afps:.1f} fps ({dur:.1f}s) from {W0}x{H0} source")

    arr = {k: np.asarray(v, float) for k, v in rec.items()}
    base_k = _odd(baseline_s * afps)
    tracks, notes = {}, {}

    def emit(key, vals, neutral, rdp_eps, floor):
        vals = np.asarray(vals, float)
        if np.max(np.abs(vals - neutral)) < floor:
            return
        keep = rdp(t, vals, rdp_eps)
        tracks[key] = [[round(float(t[j]), 3), round(float(vals[j]), 5)] for j in keep]

    # --- motion
    win_xy = max(3, int(0.5 * afps))
    win_z = max(3, int(2.0 * afps))
    logz = np.cumsum(np.log(arr["zoom"]))
    rot = np.degrees(np.cumsum(arr["rot"]))
    tx, ty = np.cumsum(arr["tx"]), np.cumsum(arr["ty"])
    if motion == "shake":  # drop slow drift: it is footage camera motion or estimator bias, not the effect
        logz, rot, tx, ty = highpass(logz, win_z), highpass(rot, win_xy), highpass(tx, win_xy), highpass(ty, win_xy)
    zoom = np.exp(deadband(logz, 0.0, 0.008))
    emit("zoom", np.clip(zoom, 0.5, 3.0), 1.0, 0.004, 0.01)
    emit("shift_x", deadband(tx, 0.0, 0.0015), 0.0, 0.0008, 0.002)
    emit("shift_y", deadband(ty, 0.0, 0.0015), 0.0, 0.0008, 0.002)
    emit("rotate", deadband(rot, 0.0, 0.12), 0.0, 0.05, 0.15)

    # --- photometric (deviation from rolling median: a flash is short, a cut is a step the median keeps)
    sm = lambda x: gaussian_filter1d(x, 0.6)
    L, S, Sat = arr["L"], arr["std"], arr["sat"]
    bright = deadband(sm(L - median_filter(L, base_k, mode="nearest")), 0.0, 0.015)
    emit("brightness", np.clip(bright, -0.8, 0.8), 0.0, 0.01, 0.03)
    busy = np.abs(bright) > 0.02
    busy = uniform_filter1d(busy.astype(float), 5) > 0
    contrast = deadband(sm(S / np.maximum(median_filter(S, base_k, mode="nearest"), 1e-3)), 1.0, 0.06)
    blur_busy = uniform_filter1d((arr["sharp"] < 0.85 * percentile_filter(arr["sharp"], 95, size=base_k, mode="nearest")).astype(float), 5) > 0
    busy = busy | blur_busy
    contrast = np.where(busy, 1.0, contrast)
    emit("contrast", np.clip(contrast, 0.3, 3.0), 1.0, 0.03, 0.08)
    sat = deadband(sm(Sat / np.maximum(median_filter(Sat, base_k, mode="nearest"), 1e-3)), 1.0, 0.08)
    sat = np.where(busy, 1.0, sat)
    emit("saturation", np.clip(sat, 0.0, 3.0), 1.0, 0.04, 0.1)

    # --- blur: calibrate sharpness-ratio -> sigma on the sharpest frame itself
    order = np.argsort(arr["sharp"])
    best = frames_sharp[int(order[int(0.9 * (len(order) - 1))])]
    sigmas = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, 8], float)
    curve = np.array([cv2.Laplacian(cv2.GaussianBlur(best, (0, 0), s) if s else best, cv2.CV_32F).var() for s in sigmas])
    curve = curve / curve[0]
    base = percentile_filter(arr["sharp"], 95, size=base_k, mode="nearest")
    ratio = np.clip(arr["sharp"] / np.maximum(base, 1e-6), 0, 1)
    blur_px = np.where(ratio > 0.85, 0.0, np.interp(ratio, curve[::-1], sigmas[::-1]))
    emit("blur", deadband(sm(blur_px), 0.0, 0.6) / AW, 0.0, 0.0005, 0.0015)

    # --- rgb split: R-vs-B offset, minus the clip's own median (lens fringing already in the footage)
    dxs = np.nan_to_num(arr["dx"] - np.nanmedian(arr["dx"])) if np.isfinite(arr["dx"]).any() else np.zeros(N)
    dys = np.nan_to_num(arr["dy"] - np.nanmedian(arr["dy"])) if np.isfinite(arr["dy"]).any() else np.zeros(N)
    emit("rgb_split", deadband(sm(dxs) / AW / 2, 0.0, 0.0008), 0.0, 0.0004, 0.0012)
    emit("rgb_split_y", deadband(sm(dys) / AH / 2, 0.0, 0.0008), 0.0, 0.0004, 0.0012)

    # --- glitch
    g = np.clip((arr["glitch"] - 0.02) * 5.0, 0, 1)
    emit("glitch", g, 0.0, 0.05, 0.1)
    spec_extra = {}
    if "glitch" in tracks and len(arr.get("gamp", [])):
        spec_extra["glitch_amp"] = round(float(np.clip(np.median(arr["gamp"]) * 1.5, 0.01, 0.2)), 4)

    # --- static looks
    vig = float(np.median(arr["vig"]))
    if vig > 0.15:
        tracks["vignette"] = round(min(vig, 0.9), 3)
    gr = arr["grain"][~np.isnan(arr["grain"])]
    if gr.size:
        gs = max(0.0, float(np.median(gr)) - 1.5) / 255.0
        if gs > 0.004:
            tracks["grain"] = round(gs, 4)

    spec = {"name": name, "version": 1, "source": str(path).split("/")[-1],
            "duration": round(dur, 3), "fps": round(afps, 2), "tracks": tracks, **spec_extra}
    if grade and grade_frames:
        from engine import lab_stats
        m, s = lab_stats(grade_frames)
        spec["grade"] = {"mean": [round(float(x), 3) for x in m], "std": [round(float(x), 3) for x in s], "strength": 0.6}
    return spec


def summarize(spec):
    """Human/LLM readable table of what was learned."""
    rows = []
    for k, tr in spec.get("tracks", {}).items():
        if isinstance(tr, (int, float)):
            rows.append((k, "constant", f"{tr:g}", "100%"))
            continue
        ts = np.array([p[0] for p in tr])
        vs = np.array([p[1] for p in tr])
        from engine import NEUTRAL
        n = NEUTRAL[k]
        dense = np.interp(np.linspace(ts[0], ts[-1], 400), ts, vs)
        active = float(np.mean(np.abs(dense - n) > 0.2 * max(np.abs(dense - n).max(), 1e-9)))
        peak = vs[np.argmax(np.abs(vs - n))]
        rows.append((k, f"{len(tr)} keys", f"peak {peak:+.4g}", f"{active:.0%} of time"))
    if spec.get("grade"):
        rows.append(("grade", "LAB transfer", f"strength {spec['grade']['strength']}", "always"))
    return rows
