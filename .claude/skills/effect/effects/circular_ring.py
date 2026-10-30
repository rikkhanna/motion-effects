"""circular_ring: a 3D carousel of upright, double-sided cards, rotating forever.

Ported from a CSS 3D recipe to a software renderer (numpy + OpenCV), keeping its geometry in the
recipe's native 1920x1080 pixel units and scaling to the video at the end.

  * Each card is positioned by  rotateY(i*360/N) translateZ(RADIUS + 44*hl).  The card itself never
    rotates about X or Z; only the parent ring turns (rotateY(-spin)), so text is always upright.
  * Cards are double sided with the SAME content.  Seen from inside the ring the back face is drawn
    horizontally flipped relative to the front, which is exactly what rotateY(180deg) on a
    backface-hidden layer does, so nothing is ever mirrored.
  * turn = 0.5*u + 0.5*stepPart(u): half a constant-speed drift plus half a stepped "push and dock"
    (8 frames hold + 13 frames eased move per 21-frame step).  8 steps = 168 frames = exactly one turn,
    so the loop closes pixel for pixel.
  * Camera is fixed: perspective 2200, ring container translateZ(-R) rotateX(-9deg).
  * The card facing the camera lights up (graphite -> lime) inside a +-26 degree window and slides
    44px outward.  Back cards fade to the background (air perspective).
  * Ground disc + 96-tick scale ring (turns with the ring) + a faded mirror reflection of the cards.

Card faces are stills sampled evenly across the input video by default. With  "faces": "devops"  (or a list of
names such as ["docker", "kubernetes", "aws"]) they are logos from effects/icons/<name>.png on a clean tile,
titled with the tool name and a small category tag; "faces": "stills" brings the video stills back.
Settings come from "params" in circular_ring.json (all optional):
  cards (8), radius (700), tilt (-9), bg_video (0.15: how much blurred live video shows behind),
  reflection (true), names (list of card titles).
Odd/short cards counts work, but keep radius/card-width >= ~2 or neighbours overlap.

Card look (params): card_style "full" (default: the face fills the whole card, rounded corners, lime rim on the lit
card) or "classic" (framed panel with numbers and labels); corner_radius (0.12 of the short side), card_size
([360, 360] full / [340, 460] classic, in the recipe's 1920x1080 units), logo_scale (0.78, logo size inside its tile).

mode "around_subject" (talking heads): the live video stays, the ring orbits the person. The far half of
the ring is composited under a person matte and the near half over it, the ring follows the head, and the
card faces are portrait crops of the subject. Extra params: ring_to_face (2.1, ring radius as a multiple of
face width; the ring must clear the shoulders or the far cards hide behind the torso), scale (overrides
ring_to_face), card_scale (0.72, card size relative to the ring), cards (10 for around_subject or one per logo, else 8), ring_y (0.1, ring centre below
the chin as a fraction of face height), dim (0, darken the room so cards pop), tilt (-26), ground (true,
orbit tick ring), subtitle (false).
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np

STEP_FRAMES, HOLD_FRAMES = 21, 8
CARD_W, CARD_H = 340.0, 460.0
PERSP = 2200.0
LIFT = 44.0
HL_WINDOW = 26.0  # degrees either side of dead-ahead
REFLECT_OPACITY, REFLECT_FADE = 0.10, 0.62

BG = np.array([22, 22, 19], np.float32)       # BGR
PANEL = np.array([50, 49, 46], np.float32)
GRAPHITE = np.array([105, 105, 98], np.float32)
LIME = np.array([55, 238, 190], np.float32)

STATE = {}
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from cardkit import (DEVOPS, blit as _blit, fit as _fit, logo_face as _logo_face, pad_tex as _pad,  # noqa: E402
                     portrait as _portrait, rounded_masks as _rounded_masks)


def _ss(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def _wrap(deg):
    return (deg + 180.0) % 360.0 - 180.0


def _mix(a, b, t):
    return a + (b - a) * t


# ----------------------------------------------------------------------------- projection

def _cam(st, P, spin):
    """ring space -> camera space: ring rotateY(-spin), container rotateX(tilt), translateZ(-R)."""
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    c, s = math.cos(-spin), math.sin(-spin)
    x2, z2 = x * c + z * s, -x * s + z * c
    ct, sn = math.cos(st["tilt"]), math.sin(st["tilt"])
    return x2, y * ct - z2 * sn, y * sn + z2 * ct - st["R"]


def _proj(st, x, y, z):
    k = PERSP / np.maximum(PERSP - z, 60.0)
    return st["cx"] + x * k * st["S"], st["cy"] + y * k * st["S"]


def _screen(st, P, spin):
    x, y, z = _cam(st, P, spin)
    sx, sy = _proj(st, x, y, z)
    return np.stack([sx, sy], 1), z


def _card_pts(st, i, hl, spin, mirror):
    th = math.radians(i * 360.0 / st["n"])
    lx = np.array([-1, 1, 1, -1]) * st["cw"] / 2
    ly = np.array([-1, -1, 1, 1]) * st["ch"] / 2
    z0 = st["R"] + LIFT * st["card_scale"] * hl  # lift is a pure translation along the card normal, never a rotation
    x = lx * math.cos(th) + z0 * math.sin(th)
    z = -lx * math.sin(th) + z0 * math.cos(th)
    y = st["ch"] - ly if mirror else ly  # mirror about the ground plane y = card height / 2
    return _screen(st, np.stack([x, y, z], 1), spin)


def _poly16(pts):
    return np.round(pts * 16).astype(np.int32)


# ----------------------------------------------------------------------------- setup















def setup(info):
    p = info.get("params", {})
    w, h = info["w"], info["h"]
    base_S = min(w / 1920.0, h / 1080.0)
    mode, subj = p.get("mode", "gallery"), None
    faces0 = p.get("faces", "stills")
    default_n = len(DEVOPS) if faces0 == "devops" else len(faces0) if isinstance(faces0, (list, tuple)) else (10 if mode == "around_subject" else 8)
    n = max(3, int(p.get("cards", default_n)))
    R = float(p.get("radius", 700))
    if mode == "around_subject":
        from subject import track_subject
        subj = track_subject(info["src"], fps=info.get("fps") or 25.0, limit=info.get("n_frames") or None) if info.get("src") else None
        if subj is None:
            print("circular_ring: no subject found, falling back to gallery mode", file=sys.stderr)
            mode = "gallery"
    if subj is not None:
        fw_px = float(np.median(subj["face"][:, 2])) * w
        S = base_S * float(p["scale"]) if "scale" in p else float(p.get("ring_to_face", 2.1)) * fw_px / R
    else:
        S = base_S
    st = dict(w=w, h=h, S=S, n=n, R=R, mode=mode, subj=subj,
              tilt=math.radians(float(p.get("tilt", -26.0 if subj is not None else -9.0))),
              cx=w / 2.0, cy=h / 2.0 - 90 * S, bg_video=float(p.get("bg_video", 0.15)),
              reflection=bool(p.get("reflection", subj is None)),
              ground=bool(p.get("ground", True)), subtitle=bool(p.get("subtitle", subj is None)),
              dim=float(p.get("dim", 0.0)), ring_y=float(p.get("ring_y", 0.1)))
    cs = float(p.get("card_scale", 0.72 if subj is not None else 1.0))
    full = p.get("card_style", "full") == "full"
    card_w, card_h = (float(v) for v in p.get("card_size", [360, 360] if full else [CARD_W, CARD_H]))
    st["style"], st["card_scale"], st["cw"], st["ch"], st["tS"] = ("full" if full else "classic"), cs, card_w * cs, card_h * cs, S * cs
    st["tw"], st["th"] = max(40, int(round(card_w * S * cs))), max(54, int(round(card_h * S * cs)))
    tw, th = st["tw"], st["th"]
    sc = lambda v: int(round(v * S * cs))
    if full:  # the face is the whole card: rounded corners, no frame, labels or margins
        st["win"] = (0, 0, tw, th)
        radius = int(round(float(p.get("corner_radius", 0.12)) * min(tw, th)))
        st["mask"], st["rim"] = _rounded_masks(tw, th, radius, max(2, int(round(5 * S * cs))))
    else:
        st["win"] = (sc(20), sc(52), tw - sc(20), sc(372))  # still window inside the card, x0 y0 x1 y1
        st["mask"], st["rim"] = np.ones((th, tw), np.float32), None
    ww, wh = st["win"][2] - st["win"][0], st["win"][3] - st["win"][1]

    faces = p.get("faces", "stills")
    keys = list(DEVOPS) if faces == "devops" else (list(faces) if isinstance(faces, (list, tuple)) else None)
    st["gray_gain"] = (0.7 if full else 0.55) if keys else 0.85  # unlit logo tiles sit back as graphite; photos keep more light

    # card faces: logos, or stills spread across the source video
    fps = info.get("fps") or 30.0
    stills, times = [], []
    cap = cv2.VideoCapture(info["src"]) if info.get("src") and not keys else None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap is not None and cap.isOpened() else 0
    for i in range(n if not keys else 0):
        k = int((i + 0.5) / n * max(total - 1, 1))
        f = None
        if total:
            cap.set(cv2.CAP_PROP_POS_FRAMES, k)
            ok, f = cap.read()
            f = f if ok else None
        if f is None:  # no source available: neutral gradient so the effect still renders
            f = np.tile(np.linspace(40, 140, 128, dtype=np.uint8)[None, :, None], (72, 1, 3))
            stills.append(_fit(f, ww, wh))
        elif subj is not None:
            stills.append(_portrait(f, subj["face"][min(k, len(subj["face"]) - 1)], ww, wh))
        else:
            stills.append(_fit(f, ww, wh))
        times.append(k / fps)
    names = p.get("names")
    if keys:
        ks = [keys[i % len(keys)] for i in range(n)]
        stills = [_logo_face(k, ww, wh, (float(p.get("logo_scale", 0.78)),) * 2 if full else (0.88, 0.62)) for k in ks]
        st["names"] = [(names[i] if names and i < len(names) else DEVOPS.get(k, (k.upper(), ""))[0]) for i, k in enumerate(ks)]
        st["stamps"] = [DEVOPS.get(k, ("", ""))[1] for k in ks]
    else:
        st["names"] = [(names[i] if names and i < len(names) else f"SHOT {i + 1:02d}") for i in range(n)]
        st["stamps"] = [f"{int(t // 60):02d}:{t % 60:04.1f}" for t in times]
    st["stills"] = stills
    st["gray"] = [cv2.cvtColor(cv2.cvtColor(s_, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) for s_ in stills]

    STATE.clear()
    STATE.update(st)
    STATE["ground"] = _build_ground(STATE) if subj is None else None


def _build_ground(st):
    """Static part of the scene: dark field, soft ground disc (rotationally symmetric), vignette."""
    w, h, S, R = st["w"], st["h"], st["S"], st["R"]
    img = np.empty((h, w, 3), np.uint8)
    img[:] = BG.astype(np.uint8)
    a = np.linspace(0, 2 * math.pi, 120, endpoint=False)
    rmax = 2.7 * R
    for r in np.linspace(rmax, 0.1 * R, 14):
        P = np.stack([r * np.sin(a), np.full_like(a, st["ch"] / 2), r * np.cos(a)], 1)
        pts, _ = _screen(st, P, 0.0)
        col = BG + 15.0 * (1 - r / rmax) ** 1.3
        cv2.fillPoly(img, [np.round(pts).astype(np.int32)], tuple(float(c) for c in col))
    img = cv2.GaussianBlur(img, (0, 0), max(1.0, 14 * S))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2)) / math.sqrt(2)
    vig = (1 - 0.55 * np.clip(r, 0, 1) ** 2)[..., None]
    return (img.astype(np.float32) * vig).astype(np.uint8)


# ----------------------------------------------------------------------------- drawing

def _tex(st, i, hl, air):
    """Card face as float BGR, tw x th. Colour, border and labels follow the light window `hl`."""
    if st["style"] == "full":
        k = 0.3 + 0.7 * hl  # graphite-gray face -> full colour when lit
        face = st["gray"][i].astype(np.float32) * (1 - k) * st["gray_gain"] + st["stills"][i].astype(np.float32) * k
        rim = (st["rim"] * hl)[..., None]  # lime rim only while lit
        return (face * (1 - rim) + LIME * rim) * (1 - air) + BG * air
    S, tw, th = st["tS"], st["tw"], st["th"]
    x0, y0, x1, y1 = st["win"]
    sc = lambda v: int(round(v * S))
    tex = np.empty((th, tw, 3), np.float32)
    tex[:] = PANEL
    k = 0.25 + 0.75 * hl  # graphite-gray still -> full colour when lit
    still = st["gray"][i].astype(np.float32) * (1 - k) * st["gray_gain"] + st["stills"][i].astype(np.float32) * k
    tex[y0:y1, x0:x1] = still
    accent = _mix(GRAPHITE, LIME, hl)
    t8 = np.clip(tex, 0, 255).astype(np.uint8)
    col = tuple(float(c) for c in accent)
    thick = max(1, sc(2))
    cv2.rectangle(t8, (0, 0), (tw - 1, th - 1), col, max(1, sc(3)), cv2.LINE_AA)
    cv2.rectangle(t8, (x0, y0), (x1 - 1, y1 - 1), col, thick, cv2.LINE_AA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(t8, f"{i + 1:02d}/{st['n']:02d}", (x0, sc(38)), font, 0.62 * S, col, thick, cv2.LINE_AA)
    tag_w = cv2.getTextSize(st["stamps"][i], font, 0.5 * S, thick)[0][0]
    cv2.putText(t8, st["stamps"][i], (tw - x0 - tag_w, sc(38)), font, 0.5 * S, col, thick, cv2.LINE_AA)
    cv2.putText(t8, f"{i + 1:02d}", (x0, sc(436)), cv2.FONT_HERSHEY_DUPLEX, 1.7 * S, col, max(1, sc(3)), cv2.LINE_AA)
    cv2.putText(t8, st["names"][i], (x0 + sc(112), sc(436)), font, 0.62 * S, (235, 235, 235) if hl > 0.5 else col, thick, cv2.LINE_AA)
    tex = t8.astype(np.float32)
    return tex * (1 - air) + BG * air






def _ground_marks(canvas, st, spin_deg, side="all"):
    """96-tick scale ring on the ground; it turns with the ring and warms up near the front."""
    R, gy = st["R"], st["ch"] / 2
    r0 = 1.2 * R
    ang = np.arange(96) * 3.75
    long_ = (np.arange(96) % 8) == 0
    rad = np.radians(ang)
    inner = np.stack([r0 * np.sin(rad), np.full(96, gy), r0 * np.cos(rad)], 1)
    outer_r = r0 + np.where(long_, 38.0, 17.0)
    outer = np.stack([outer_r * np.sin(rad), np.full(96, gy), outer_r * np.cos(rad)], 1)
    spin = math.radians(spin_deg)
    a_pts, za = _screen(st, inner, spin)
    b_pts, _ = _screen(st, outer, spin)
    for k in range(96):
        if side != "all" and (za[k] < -R) != (side == "back"):  # behind / in front of the subject plane
            continue
        near = _ss(1 - abs(_wrap(ang[k] - spin_deg)) / (HL_WINDOW * 1.8))
        col = _mix(GRAPHITE * 0.5, LIME * 0.85, near)
        cv2.line(canvas, tuple(int(v) for v in np.round(a_pts[k] * 16)), tuple(int(v) for v in np.round(b_pts[k] * 16)),
                 tuple(float(c) for c in col), 1, cv2.LINE_AA, 4)
    t = np.linspace(0, 2 * math.pi, 180, endpoint=False)
    for rr in (r0, r0 + 44):
        P = np.stack([rr * np.sin(t), np.full_like(t, gy), rr * np.cos(t)], 1)
        pts, zz = _screen(st, P, 0.0)
        if side == "all":
            cv2.polylines(canvas, [_poly16(pts)], True, tuple(float(c) for c in GRAPHITE * 0.45), 1, cv2.LINE_AA, 4)
        else:  # draw only the half on the requested side of the subject
            keep = (zz < -R) if side == "back" else (zz >= -R)
            for seg in np.split(np.arange(len(t)), np.where(np.diff(keep.astype(int)) != 0)[0] + 1):
                if keep[seg[0]] and len(seg) > 1:
                    cv2.polylines(canvas, [_poly16(pts[seg])], False, tuple(float(c) for c in GRAPHITE * 0.45), 1, cv2.LINE_AA, 4)


def _subtitle(canvas, st, cur, nxt, e):
    """Mono number + big name, rolling through a clipping window while the ring steps, frozen when docked."""
    S, w, h = st["S"], st["w"], st["h"]
    Hs = max(40, int(round(150 * S)))
    y0 = h - Hs - int(round(18 * S))
    layer = np.zeros((Hs, w, 3), np.uint8)

    def draw(idx, off):
        num = f"{idx + 1:02d} / {st['n']:02d}"
        for text, base, scale, col, th_ in ((num, 34 * S, 0.8 * S, LIME, max(1, int(round(S)))),
                                            (st["names"][idx], 120 * S, 2.3 * S, (238, 238, 238), max(2, int(round(3 * S))))):
            font = cv2.FONT_HERSHEY_SIMPLEX if text is num else cv2.FONT_HERSHEY_DUPLEX
            tw_ = cv2.getTextSize(text, font, scale, th_)[0][0]
            cv2.putText(layer, text, (int((w - tw_) / 2), int(base + off)), font, scale,
                        tuple(float(c) for c in col), th_, cv2.LINE_AA)

    if e <= 0:
        draw(cur, 0)
    else:
        draw(cur, -e * Hs)
        draw(nxt, (1 - e) * Hs)
    roi = canvas[y0:y0 + Hs]
    cv2.add(roi, layer, dst=roi)


# ----------------------------------------------------------------------------- per frame

def apply(frame, t, info):
    if not STATE or STATE["w"] != frame.shape[1] or STATE["h"] != frame.shape[0]:
        setup(info)
    st = STATE
    n, w, h = st["n"], st["w"], st["h"]

    # timeline: turn = 0.5*uniform + 0.5*stepped; n steps of 21 frames, 8 held + 13 eased
    u = (info["effect_t"] / info["effect_duration"]) % 1.0
    L = u * n * STEP_FRAMES
    s = min(int(L // STEP_FRAMES), n - 1)
    e = _ss((L - s * STEP_FRAMES - HOLD_FRAMES) / (STEP_FRAMES - HOLD_FRAMES))
    turn = 0.5 * u + 0.5 * (s + e) / n
    spin_deg = turn * 360.0
    spin = math.radians(spin_deg)

    if st["subj"] is not None:
        return _apply_subject(frame, st, info, spin, spin_deg)

    canvas = st["ground"].copy()
    k = st["bg_video"]
    if k > 0:  # blurred, dimmed live video keeps the original moving underneath
        small = cv2.resize(frame, (max(8, w // 24), max(8, h // 24)), interpolation=cv2.INTER_AREA)
        vid = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        canvas = cv2.addWeighted(canvas, 1 - k, vid, k * 0.5, 0)

    _ground_marks(canvas, st, spin_deg)
    ready = cards_for(st, spin, spin_deg)

    if st["reflection"]:  # every reflection goes under every card
        rows = np.clip(1 - (st["th"] - 1 - np.arange(st["th"])) / (REFLECT_FADE * st["th"]), 0, 1)
        alpha = np.tile((REFLECT_OPACITY * rows)[:, None], (1, st["tw"])) * st["mask"]
        for c in ready:
            rpts, _ = _card_pts(st, c["i"], c["hl"], spin, True)
            _blit(canvas, _pad(c["tex"], alpha), rpts[c["order"]])
    solid = st["mask"]
    for c in ready:
        _blit(canvas, _pad(c["tex"], solid), c["pts"][c["order"]])

    if st["subtitle"]:
        _subtitle(canvas, st, s, (s + 1) % n, e)
    return canvas


def cards_for(st, spin, spin_deg):
    """Visible cards for this frame, far to near: geometry, face texture, and which way each is facing."""
    n = st["n"]
    cards = []
    for i in range(n):
        eff = _wrap(i * 360.0 / n - spin_deg)
        hl = _ss(1 - abs(eff) / HL_WINDOW)
        pts, z = _card_pts(st, i, hl, spin, False)
        cards.append((float(z.mean()), i, eff, hl, pts))
    cards.sort(key=lambda c: c[0])
    ready = []
    for zc, i, eff, hl, pts in cards:
        area = 0.5 * float(np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1]))
        if abs(area) < 4:
            continue
        order = [0, 1, 2, 3] if area > 0 else [1, 0, 3, 2]  # back face: same content, flipped so it reads upright
        air = 0.7 * (1 - math.cos(math.radians(eff))) / 2
        ready.append({"i": i, "hl": hl, "pts": pts, "order": order, "z": zc, "tex": _tex(st, i, hl, air)})
    return ready


def _apply_subject(frame, st, info, spin, spin_deg):
    """Ring around a talking head: far half under the person matte, near half over it."""
    w, h = st["w"], st["h"]
    sj = st["subj"]
    idx = min(int(info.get("frame_index", 0)), len(sj["face"]) - 1)
    fx, fy, fw, fh = sj["face"][idx] * np.array([w, h, w, h])
    st["cx"], st["cy"] = fx + fw / 2, fy + fh * (1.0 + st["ring_y"])  # ring centre: just under the chin

    ready = cards_for(st, spin, spin_deg)  # after the centre moved, so geometry follows the head
    room = (frame.astype(np.float32) * (1 - st["dim"])).astype(np.uint8)
    canvas = room.copy()
    solid = st["mask"]
    behind = [c for c in ready if c["z"] < -st["R"]]
    front = [c for c in ready if c["z"] >= -st["R"]]

    if st["ground"]:
        _ground_marks(canvas, st, spin_deg, "back")
    for c in behind:
        _blit(canvas, _pad(c["tex"], solid), c["pts"][c["order"]])

    m = cv2.resize(sj["matte"][idx].astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)[..., None]
    canvas = (canvas.astype(np.float32) * (1 - m) + frame.astype(np.float32) * m + 0.5).astype(np.uint8)

    if st["ground"]:
        _ground_marks(canvas, st, spin_deg, "front")
    for c in front:
        _blit(canvas, _pad(c["tex"], solid), c["pts"][c["order"]])
    return canvas
