"""Effect engine: an effect spec (keyframed parameter tracks) -> per-frame image operators.

Spec format (JSON, see SKILL.md):
    {
      "name": "...", "description": "...",
      "duration": 2.0,                    # length of the effect's own timeline, seconds
      "timing": "stretch" | "loop" | "once",
      "tracks": {"zoom": [[t, v], ...], "grain": 0.02, ...},   # list = keyframes, number = constant
      "glitch_amp": 0.06,                 # max horizontal tear, fraction of width
      "grade": {"mean": [L,a,b], "std": [L,a,b], "strength": 0.6},   # optional LAB look transfer
      "plugin": "name.py",                # optional python overlay layer, see SKILL.md
      "params": {...}                     # free-form settings handed to the plugin as info["params"]
    }

All track units are resolution independent so an effect learned on a 480p clip
looks the same on 4K footage.
"""
import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

# name -> (neutral value, unit)
TRACKS = {
    "zoom":        (1.0, "scale factor about the frame centre"),
    "shift_x":     (0.0, "fraction of width"),
    "shift_y":     (0.0, "fraction of height"),
    "rotate":      (0.0, "degrees, clockwise positive"),
    "blur":        (0.0, "gaussian sigma, fraction of width"),
    "rgb_split":   (0.0, "red/blue horizontal offset each way, fraction of width"),
    "rgb_split_y": (0.0, "red/blue vertical offset each way, fraction of height"),
    "glitch":      (0.0, "0..1 tear intensity"),
    "brightness":  (0.0, "additive, fraction of full range (flash > 0, dip to black < 0)"),
    "contrast":    (1.0, "multiplier around mid grey"),
    "saturation":  (1.0, "multiplier"),
    "vignette":    (0.0, "0..1 corner darkening"),
    "grain":       (0.0, "gaussian noise sigma, fraction of full range"),
}
NEUTRAL = {k: v[0] for k, v in TRACKS.items()}


# ----------------------------------------------------------------------------- spec

def _interp(track):
    if isinstance(track, (int, float)):
        c = float(track)
        return lambda t: c
    a = np.asarray(track, dtype=float)
    ts, vs = a[:, 0], a[:, 1]
    return lambda t: float(np.interp(t, ts, vs))


class Effect:
    def __init__(self, spec, intensity=1.0, timing=None, start=0.0, seed=None, effects_dir=None):
        self.spec = spec
        self.intensity = float(intensity)
        self.timing = timing or spec.get("timing", "stretch")
        self.start = float(start)
        self.seed = int(spec.get("seed", 7) if seed is None else seed)
        self.duration = float(spec.get("duration", 1.0)) or 1.0
        self.glitch_amp = float(spec.get("glitch_amp", 0.06))
        self.tracks = {}
        for name, tr in spec.get("tracks", {}).items():
            if name not in NEUTRAL:
                print(f"warning: unknown track '{name}' ignored", file=sys.stderr)
                continue
            self.tracks[name] = (_interp(tr), not isinstance(tr, (int, float)))
        self.plugin = None
        if spec.get("plugin"):
            p = Path(effects_dir or ".") / spec["plugin"]
            m_spec = importlib.util.spec_from_file_location(f"effect_plugin_{p.stem}", p)
            self.plugin = importlib.util.module_from_spec(m_spec)
            m_spec.loader.exec_module(self.plugin)
        self.grade = spec.get("grade")
        self._grade_src = None  # target video LAB stats, filled by set_target_stats

    def params(self, t, total):
        """Parameter values at video time t (seconds), after timing + intensity."""
        D = self.duration
        inside = True
        if self.timing == "loop":
            te = t % D
        elif self.timing == "once":
            te = t - self.start
            inside = 0.0 <= te <= D
        else:  # stretch
            te = (t / total * D) if total > 0 else 0.0
        p = dict(NEUTRAL)
        for name, (fn, keyed) in self.tracks.items():
            if keyed and not inside:
                continue
            n = NEUTRAL[name]
            p[name] = n + (fn(te) - n) * self.intensity
        p["zoom"] = max(p["zoom"], 0.3)
        p["contrast"] = max(p["contrast"], 0.0)
        p["saturation"] = max(p["saturation"], 0.0)
        return p

    def set_target_stats(self, mean, std):
        self._grade_src = (np.asarray(mean, np.float32), np.asarray(std, np.float32))

    def apply_frame(self, img, t, idx, total, info):
        p = self.params(t, total)
        h, w = img.shape[:2]
        rng = np.random.default_rng(self.seed * 100003 + idx)

        if abs(p["zoom"] - 1) > 1e-4 or abs(p["shift_x"]) > 1e-5 or abs(p["shift_y"]) > 1e-5 or abs(p["rotate"]) > 1e-3:
            img = op_geometry(img, p["zoom"], p["shift_x"], p["shift_y"], p["rotate"])
        if p["blur"] * w > 0.3:
            img = cv2.GaussianBlur(img, (0, 0), p["blur"] * w)
        if abs(p["rgb_split"]) * w > 0.2 or abs(p["rgb_split_y"]) * h > 0.2:
            img = op_rgb_split(img, p["rgb_split"] * w, p["rgb_split_y"] * h)
        if p["glitch"] > 0.02:
            img = op_glitch(img, p["glitch"], self.glitch_amp, rng)
        if self.grade and self._grade_src is not None and self.grade.get("strength", 0) > 0:
            img = op_grade(img, self.grade, self._grade_src)
        if abs(p["brightness"]) > 1e-4 or abs(p["contrast"] - 1) > 1e-3 or abs(p["saturation"] - 1) > 1e-3:
            img = op_color(img, p["brightness"], p["contrast"], p["saturation"])
        if p["vignette"] > 1e-3:
            img = op_vignette(img, p["vignette"])
        if p["grain"] * 255 > 0.3:
            img = op_grain(img, p["grain"], rng)
        if self.plugin is not None and hasattr(self.plugin, "apply"):
            img = self.plugin.apply(img, t, dict(info, frame_index=idx, effect_t=self._effect_time(t, total),
                                                 effect_duration=self.duration))
        return img

    def _effect_time(self, t, total):
        if self.timing == "loop":
            return t % self.duration
        if self.timing == "once":
            return t - self.start
        return t / total * self.duration if total > 0 else 0.0


# ----------------------------------------------------------------------------- operators

def op_geometry(img, zoom, sx, sy, rot_deg):
    h, w = img.shape[:2]
    th = math.radians(rot_deg)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]]) * zoom
    c = np.array([w / 2.0, h / 2.0])
    t = c - R @ c + np.array([sx * w, sy * h])
    M = np.hstack([R, t[:, None]])
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)


def op_rgb_split(img, dx, dy):
    h, w = img.shape[:2]
    out = img.copy()
    for ch, sgn in ((2, 1), (0, -1)):  # BGR: red goes one way, blue the other
        M = np.float32([[1, 0, sgn * dx], [0, 1, sgn * dy]])
        out[..., ch] = cv2.warpAffine(img[..., ch], M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return out


def op_glitch(img, g, amp, rng):
    h, w = img.shape[:2]
    out = img.copy()
    for _ in range(1 + int(g * 7)):
        bh = max(2, int(h * rng.uniform(0.01, 0.10) * (0.5 + g)))
        y = int(rng.integers(0, max(1, h - bh)))
        dx = int(rng.choice([-1, 1]) * rng.uniform(0.3, 1.0) * amp * w * (0.5 + g))
        out[y:y + bh] = np.roll(img[y:y + bh], dx, axis=1)
        if rng.random() < 0.5:  # colour-channel tear inside the band
            ch = int(rng.integers(0, 3))
            out[y:y + bh, :, ch] = np.roll(img[y:y + bh, :, ch], dx // 2 + 3, axis=1)
    return out


def op_color(img, b, c, s):
    f = img.astype(np.float32) / 255.0
    if abs(b) > 1e-4:
        f += b
    if abs(c - 1) > 1e-3:
        f = (f - 0.5) * c + 0.5
    if abs(s - 1) > 1e-3:
        gray = (f @ np.float32([0.114, 0.587, 0.299]))[..., None]
        f = gray + (f - gray) * s
    return (np.clip(f, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def op_vignette(img, v):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2)) / math.sqrt(2)
    mask = 1.0 - v * np.clip(r, 0, 1) ** 2
    return (img.astype(np.float32) * mask[..., None]).clip(0, 255).astype(np.uint8)


def op_grain(img, sigma, rng):
    h, w = img.shape[:2]
    n = rng.normal(0.0, sigma * 255.0, (h, w, 1)).astype(np.float32)
    return (img.astype(np.float32) + n).clip(0, 255).astype(np.uint8)


def op_grade(img, grade, src):
    """Reinhard-style LAB mean/std transfer from the target video's look to the learned look."""
    ref_m = np.asarray(grade["mean"], np.float32)
    ref_s = np.asarray(grade["std"], np.float32)
    src_m, src_s = src
    k = float(grade.get("strength", 0.6))
    lab = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    ratio = np.clip(ref_s / np.maximum(src_s, 1e-3), 0.5, 2.0)
    new = (lab - src_m) * ratio + ref_m
    lab = lab * (1 - k) + new * k
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return (np.clip(out, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def lab_stats(frames_bgr):
    labs = [cv2.cvtColor(cv2.resize(f, (160, 90)).astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB) for f in frames_bgr]
    a = np.concatenate([l.reshape(-1, 3) for l in labs])
    return a.mean(0), a.std(0)


# ----------------------------------------------------------------------------- rendering

def render(effect, src, dst, seconds=None, crf=18, log=lambda *_: None):
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    n_out = min(n_total, int(seconds * fps)) if seconds else n_total
    total = n_out / fps if n_out else 0.0

    if effect.grade:  # sample the target's look once so the grade is stable across frames
        samples = []
        for k in np.linspace(0, max(n_total - 1, 0), 16).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
            ok, f = cap.read()
            if ok:
                samples.append(f)
        if samples:
            effect.set_target_stats(*lab_stats(samples))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ok, frame = cap.read()
    if not ok:
        raise SystemExit("video has no frames")
    h, w = frame.shape[:2]
    info = {"w": w, "h": h, "fps": fps, "duration": total, "n_frames": n_out,
            "src": str(src), "params": effect.spec.get("params", {}),
            "effect_duration": effect.duration, "timing": effect.timing, "start": effect.start}
    if effect.plugin is not None and hasattr(effect.plugin, "setup"):
        effect.plugin.setup(info)

    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
           "-i", str(src), "-map", "0:v:0", "-map", "1:a:0?",
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
           "-c:v", "libx264", "-crf", str(crf), "-preset", "medium", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart"]
    if seconds:
        cmd += ["-t", f"{n_out / fps:.3f}"]
    else:
        cmd += ["-shortest"]
    cmd.append(str(dst))
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    i = 0
    try:
        while ok and (not n_out or i < n_out):
            out = effect.apply_frame(frame, i / fps, i, total, info)
            proc.stdin.write(np.ascontiguousarray(out).tobytes())
            i += 1
            if n_out and i % max(1, n_out // 10) == 0:
                log(f"  {100 * i // n_out}%")
            ok, frame = cap.read()
    finally:
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace")
        proc.wait()
    if proc.returncode != 0:
        raise SystemExit(f"ffmpeg failed:\n{err}")
    return i, fps
