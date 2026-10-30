#!/usr/bin/env python3
"""effect.py - learn motion-graphics effects from a reference video and apply them to other videos.

  effect.py list
  effect.py show <name>
  effect.py apply <name> [video.mp4] [-o out.mp4] [--intensity 1.0] [--timing loop|stretch|once]
                          [--start SEC] [--seconds N] [--seed N] [--set KEY=VALUE ...]
  effect.py learn <name> <reference.mp4> [--description TEXT] [--timing auto|loop|stretch|once]
                          [--max-seconds 20] [--motion shake|full] [--grade]
  effect.py inspect <video> [-n 12] [-o sheet.png]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EFFECTS_DIR = HERE.parent / "effects"

from engine import Effect, TRACKS, render  # noqa: E402
from presets import BUILTIN  # noqa: E402


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def load_spec(name):
    p = EFFECTS_DIR / f"{name}.json"
    if p.exists():
        spec = json.loads(p.read_text())
        return spec, ("learned" if spec.get("source") else "custom")
    if name in BUILTIN:
        return {"name": name, **BUILTIN[name]}, "built-in"
    avail = ", ".join(sorted(all_names()))
    raise SystemExit(f"unknown effect '{name}'. Available: {avail or '(none)'}")


def all_names():
    names = set(BUILTIN)
    names |= {p.stem for p in EFFECTS_DIR.glob("*.json")}
    return names


def contact_sheet(video, n=12, cols=4, width=1600):
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    tiles = []
    tw = width // cols
    for k in np.linspace(0, max(total - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
        ok, f = cap.read()
        if not ok:
            continue
        th = int(f.shape[0] * tw / f.shape[1])
        f = cv2.resize(f, (tw, th), interpolation=cv2.INTER_AREA)
        cv2.putText(f, f"{k / fps:.2f}s", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(f, f"{k / fps:.2f}s", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        tiles.append(f)
    if not tiles:
        raise SystemExit("could not read any frames")
    th = tiles[0].shape[0]
    rows = []
    for r in range(0, len(tiles), cols):
        row = tiles[r:r + cols]
        row += [np.zeros_like(tiles[0])] * (cols - len(row))
        rows.append(np.hstack(row))
    return np.vstack(rows)


# ----------------------------------------------------------------------------- commands

def cmd_list(_):
    for n in sorted(all_names()):
        spec, kind = load_spec(n)
        print(f"{n:<16} [{kind:<8}] {spec.get('description', '')}")


def cmd_show(a):
    from analyze import summarize
    spec, kind = load_spec(a.name)
    print(f"{a.name} [{kind}] - {spec.get('description', '(no description)')}")
    print(f"timeline {spec.get('duration')}s, timing={spec.get('timing', 'stretch')}"
          + (f", plugin={spec['plugin']}" if spec.get("plugin") else ""))
    for k, how, peak, act in summarize(spec):
        print(f"  {k:<12} {how:<14} {peak:<16} {act}")


def cmd_apply(a):
    spec, _ = load_spec(a.name)
    src = Path(a.video)
    if not src.exists():
        raise SystemExit(f"video not found: {src}")
    dst = Path(a.output) if a.output else src.with_name(f"{src.stem}_{a.name}.mp4")
    for kv in a.set or []:  # --set key=value overrides a plugin param for this run (value parsed as JSON if possible)
        k, _, v = kv.partition("=")
        try:
            v = json.loads(v)
        except ValueError:
            pass
        spec = {**spec, "params": {**spec.get("params", {}), k: v}}
    fx = Effect(spec, intensity=a.intensity, timing=a.timing, start=a.start, seed=a.seed, effects_dir=EFFECTS_DIR)
    log(f"applying '{a.name}' (timing={fx.timing}, intensity={fx.intensity:g}) to {src}")
    n, fps = render(fx, src, dst, seconds=a.seconds, log=log)
    print(f"wrote {dst}  ({n} frames @ {fps:g} fps)")


def cmd_learn(a):
    from analyze import analyze, summarize
    ref = Path(a.reference)
    if not ref.exists():
        raise SystemExit(f"reference video not found: {ref}")
    spec = analyze(ref, a.name, max_seconds=a.max_seconds, motion=a.motion, grade=a.grade, log=log)
    spec["description"] = a.description or f"Learned from {ref.name}"
    timing = a.timing
    if timing == "auto":
        timing = "loop" if spec["duration"] <= 4.0 else "stretch"
    spec["timing"] = timing
    if not spec["tracks"] and not spec.get("grade"):
        log("note: no measurable effect found - the reference may rely on overlays/text/particles "
            "(see SKILL.md 'plugin layer'). Spec saved anyway.")
    EFFECTS_DIR.mkdir(exist_ok=True)
    out = EFFECTS_DIR / f"{a.name}.json"
    out.write_text(json.dumps(spec, indent=1))
    sheet = EFFECTS_DIR / f"{a.name}.reference.png"
    cv2.imwrite(str(sheet), contact_sheet(ref))
    print(f"saved {out}")
    print(f"reference contact sheet: {sheet}")
    print(f"timing={timing}  timeline={spec['duration']}s")
    for k, how, peak, act in summarize(spec):
        print(f"  {k:<12} {how:<14} {peak:<16} {act}")


def cmd_inspect(a):
    out = Path(a.output or "contact_sheet.png")
    cv2.imwrite(str(out), contact_sheet(a.video, n=a.n))
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)

    p = sub.add_parser("show"); p.add_argument("name"); p.set_defaults(fn=cmd_show)

    p = sub.add_parser("apply")
    p.add_argument("name"); p.add_argument("video", nargs="?", default="video.mp4")
    p.add_argument("-o", "--output")
    p.add_argument("--intensity", type=float, default=1.0, help="scale every track's deviation from neutral")
    p.add_argument("--timing", choices=["loop", "stretch", "once"], help="override the effect's own timing")
    p.add_argument("--start", type=float, default=0.0, help="for timing=once: when the effect begins (s)")
    p.add_argument("--seconds", type=float, help="render only the first N seconds (quick preview)")
    p.add_argument("--seed", type=int)
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a plugin param, e.g. --set mode=around_subject")
    p.set_defaults(fn=cmd_apply)

    p = sub.add_parser("learn")
    p.add_argument("name"); p.add_argument("reference")
    p.add_argument("--description")
    p.add_argument("--timing", choices=["auto", "loop", "stretch", "once"], default="auto")
    p.add_argument("--max-seconds", type=float, default=20.0)
    p.add_argument("--motion", choices=["shake", "full"], default="shake",
                   help="shake: high-pass motion (drops slow drift). full: keep accumulated pans/zooms")
    p.add_argument("--grade", action="store_true", help="also learn the colour look (content dependent)")
    p.set_defaults(fn=cmd_learn)

    p = sub.add_parser("inspect")
    p.add_argument("video"); p.add_argument("-n", type=int, default=12); p.add_argument("-o", "--output")
    p.set_defaults(fn=cmd_inspect)

    argv = sys.argv[1:]
    if argv and argv[0] not in sub.choices and not argv[0].startswith("-"):
        argv = ["apply"] + argv  # `effect.py <name> [video]` is shorthand for apply
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
