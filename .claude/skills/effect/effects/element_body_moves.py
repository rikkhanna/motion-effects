"""element_body_moves: cards that leave the table. "Contact-shadow-lift" roll call.

Ported from a motion recipe (150f / 5s, low-medium energy). Most motion cards change an element's *position*;
this one changes its *body*: when a card is lifted, the shadow left on the table under it grows, softens, lightens
and slides to the backlit side, which is the evidence that the paper really left the table.

  * Scene: oblique overhead camera on a table plane (rotateX 24deg -> 20deg over the shot, perspective 2400).
    A lift is a translation along the table normal (translateZ 96px) plus scale 1.05, so the card rises about
    40-50px on screen. Lifts under 12 visible px read as a mistake, so the lift is clamped up to that.
  * Shadows are separate entities lying on the table (z=0), never part of the card: a contact core
    (alpha .32 -> .14, blur 3 -> 25px) and an ambient shadow (alpha .09 -> .22, blur 14 -> 54px, offset to the lower
    right). Both enlarge with the lift.
  * Roll call: ONE card at a time (two lifted at once and nobody can tell who was called). A call = rise,
    hold, fall back (8f in-cubic) + 2f micro-press (scale .99) + 5f rebound to 1; the micro-press is what gives
    the card weight. The last card springs up with a little overshoot, gets its accent outline and check badge, and
    hovers (~50f) in the air. timing "once" keeps it there; timing "loop" lets it fall back like the others so the
    roll call repeats.
  * Layout and card shape follow the frame: a row of tall cards in 16:9, a stack of wide cards in 9:16, a grid for
    many cards. Everything is measured in units where the frame's short side is 1080, so it scales to any size.

Card content comes from params (all optional):
  items            list of {title, value, text, icon, chips, chips_on}; icon is a logo key in effects/icons/
  cards            how many cards (items cycle if fewer)
  order            call order as card indexes, default left to right / top to bottom
  layout           auto | row | column | grid (+ cols); anchor center | top | bottom; region [x0,y0,x1,y1] fractions
  view             "2.5d" (default) or "2d" (flat: -28px and scale 1.08 instead of translateZ)
  lift_z, lift_scale, tilt [24, 20], persp, speed
  shadow_scale     multiplies shadow strength (default: 1.0 on desk, 1.4 on dark, 1.8 over video)
  background       video (default: the input video is the backdrop) | desk | dark;  bg_blur, bg_dim, veil
  theme            light | dark;  accent "#5B5BD6";  select_last true;  corner_radius 0.09;  gap 0.035
  title, caption   optional headline above the cards;  intro true;  outro seconds (fade at the end of the video)
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import cardkit as ck  # noqa: E402

FPS_FX = 30.0  # the recipe's frames are 30 fps frames, whatever the video's own rate
STATE = {}

DEFAULT_ITEMS = [
    {"title": "Containers", "value": "Docker", "text": "Package the app once and run it the same way everywhere.", "icon": "docker"},
    {"title": "Orchestration", "value": "Kubernetes", "text": "Schedule, scale and heal services across the cluster.", "icon": "kubernetes"},
    {"title": "Infra as code", "value": "Terraform", "text": "Provision cloud resources from versioned, reviewable code.", "icon": "terraform"},
]
DEFAULTS = dict(
    items=None, cards=None, order=None, layout="auto", cols=None, anchor="center", region=None, gap=0.035,
    view="2.5d", lift_z=96.0, lift_scale=1.05, lift_y=28.0, lift_scale_2d=1.08, tilt=[24.0, 20.0], persp=2400.0,
    speed=1.0, shadow_scale=None, background="video", bg_blur=0.0, bg_dim=0.0, veil=0.0,
    theme="light", accent="#5B5BD6", select_last=True, corner_radius=0.09, tex_scale=1.25,
    title=None, caption=None, intro=True, outro=0.4,
)

# ----------------------------------------------------------------------------- easing

_c = lambda x: min(max(x, 0.0), 1.0)
ease_out = lambda x: 1 - (1 - _c(x)) ** 3
ease_in = lambda x: _c(x) ** 3
ease_back = lambda x, s=2.2: 1 + (s + 1) * (_c(x) - 1) ** 3 + s * (_c(x) - 1) ** 2  # spring: overshoots then settles


def _smooth(x):
    x = _c(x)
    return x * x * (3 - 2 * x)


# ----------------------------------------------------------------------------- layout

def _region(p, portrait):
    if p["region"]:
        return [float(v) for v in p["region"]]
    a = p["anchor"]
    if portrait:
        return {"top": [0.06, 0.04, 0.94, 0.5], "bottom": [0.06, 0.5, 0.94, 0.9]}.get(a, [0.06, 0.22, 0.94, 0.86])
    return {"top": [0.06, 0.03, 0.94, 0.5], "bottom": [0.06, 0.48, 0.94, 0.9]}.get(a, [0.06, 0.2, 0.94, 0.88])


def _layout(n, Wu, Hu, portrait, p):
    """-> (cell centres, card w, card h, region centre), all in units from the frame centre (y down)."""
    short = 1080.0
    rx0, ry0, rx1, ry1 = _region(p, portrait)
    X0, X1, Y0, Y1 = (rx0 - 0.5) * Wu, (rx1 - 0.5) * Wu, (ry0 - 0.5) * Hu, (ry1 - 0.5) * Hu
    Rw, Rh, g = X1 - X0, Y1 - Y0, p["gap"] * short
    lay = p["layout"]
    if lay == "auto":
        lay = ("column" if n <= 4 else "grid") if portrait else ("row" if n <= 4 else "grid")
    if lay == "row":
        cols = n
    elif lay == "column":
        cols = 1
    else:
        cols = int(p["cols"] or (2 if portrait else math.ceil(n / 2)))
    cols = max(1, min(cols, n))
    rows = math.ceil(n / cols)
    cw, ch = (Rw - (cols - 1) * g) / cols, (Rh - (rows - 1) * g) / rows
    if ch / cw > 1.4:        # very tall cell: keep a card-like shape
        ch = 1.4 * cw
    elif ch / cw < 0.38:     # very flat cell
        cw = ch / 0.38
    cw = min(cw, 0.84 * short) if cols == 1 else min(cw, 0.42 * short)   # wide cards leave room for the lift (scale 1.05 x perspective ~1.10)
    if cols > 1:
        ch = min(ch, 1.4 * cw)
    tot_w, tot_h = cols * cw + (cols - 1) * g, rows * ch + (rows - 1) * g
    cx0, cy0 = (X0 + X1) / 2 - tot_w / 2 + cw / 2, (Y0 + Y1) / 2 - tot_h / 2 + ch / 2
    cells = [(cx0 + (i % cols) * (cw + g), cy0 + (i // cols) * (ch + g)) for i in range(n)]
    return cells, cw, ch, ((X0 + X1) / 2, (Y0 + Y1) / 2), (Y0, tot_h)


# ----------------------------------------------------------------------------- timeline (frames at 30 fps)

def _timeline(n, order, T, loop):
    """One call = rise 10 + hold 5 + fall 8 + press 2 + rebound 5 frames; the last card rises 12 with a spring and
    hovers. If the shot is too short for that many cards, every phase shrinks by the same factor."""
    R, H, F, P, B, RL, LEAD = 10.0, 5.0, 8.0, 2.0, 5.0, 12.0, 14.0
    slot, tail = R + H + F + P + B, F + P + B
    need = (n - 1) * slot + RL + (tail if loop else 0.0)
    c = min(1.0, max(0.35, (T - LEAD - (30.0 if loop else 20.0)) / max(need, 1.0)))
    ev = []
    for k, idx in enumerate(order):
        s = LEAD + k * slot * c
        if k < n - 1:
            ev.append(dict(i=idx, s=s, r=R * c, hold=H * c, fall=F * c, press=P * c, bounce=B * c, spring=False))
        else:
            hold = None
            if loop:
                hold = max(T - (s + RL * c + tail * c), 20.0)
            ev.append(dict(i=idx, s=s, r=RL * c, hold=hold, fall=F * c, press=P * c, bounce=B * c, spring=True))
    return ev


def _call_state(f, ev):
    """-> (lift 0..~1.1, press scale) of one card at frame f."""
    x = f - ev["s"]
    if x < 0:
        return 0.0, 1.0
    if x < ev["r"]:
        q = x / ev["r"]
        return (ease_back(q) if ev["spring"] else ease_out(q)), 1.0
    x -= ev["r"]
    if ev["hold"] is None or x < ev["hold"]:
        return 1.0, 1.0
    x -= ev["hold"]
    if x < ev["fall"]:
        return 1.0 - ease_in(x / ev["fall"]), 1.0   # 8f in-cubic: it drops, it doesn't float down
    x -= ev["fall"]
    if x < ev["press"]:
        return 0.0, 0.99                             # micro-press: the paper hits the table
    x -= ev["press"]
    if x < ev["bounce"]:
        return 0.0, 0.99 + 0.01 * ease_out(x / ev["bounce"])
    return 0.0, 1.0


# ----------------------------------------------------------------------------- setup

def setup(info):
    p = {**DEFAULTS, **(info.get("params") or {})}
    w, h = info["w"], info["h"]
    u = min(w, h) / 1080.0
    Wu, Hu = w / u, h / u
    portrait = h > w
    items = list(p["items"] or DEFAULT_ITEMS)
    n = int(p["cards"] or len(items))
    items = [items[i % len(items)] for i in range(n)]
    cells, cw, ch, pivot, (ry_top, tot_h) = _layout(n, Wu, Hu, portrait, p)
    order = [int(i) for i in (p["order"] or range(n)) if 0 <= int(i) < n]
    flat = p["view"] == "2d"
    tilt = [0.0, 0.0] if flat else [float(v) for v in p["tilt"]]
    lift_z = float(p["lift_z"])
    if not flat:  # a lift that moves the card less than 12 visible px reads as a mistake, not as paper leaving the table
        vis = lift_z * math.sin(math.radians(min(tilt)))
        if vis < 12:
            lift_z = 12 / max(math.sin(math.radians(min(tilt))), 1e-3)
    # perspective enlarges the card nearest the camera (and a lifted one more); shrink the layout until the biggest
    # case still sits inside the frame
    rx, ry = pivot
    ext_x = max(abs(c[0] - rx) for c in cells) + cw / 2
    ext_y = max(abs(c[1] - ry) for c in cells) + ch / 2
    if flat:
        k_max, lift_grow = 1.0, float(p["lift_scale_2d"])
    else:
        z1 = ext_y * math.sin(math.radians(max(tilt))) + lift_z * math.cos(math.radians(max(tilt)))
        k_max, lift_grow = float(p["persp"]) / max(float(p["persp"]) - z1, 50.0), float(p["lift_scale"])
    fit = min(1.0, (0.5 * Wu * 0.97 - abs(rx)) / (ext_x * k_max * lift_grow), (0.5 * Hu * 0.97 - abs(ry)) / (ext_y * k_max * lift_grow))
    if fit < 1.0:
        cells = [(rx + (c[0] - rx) * fit, ry + (c[1] - ry) * fit) for c in cells]
        cw, ch = cw * fit, ch * fit
    T = float(info.get("effect_duration", 5.0)) * FPS_FX / max(float(p["speed"]), 1e-3)

    tw, th = max(32, int(round(cw * u * p["tex_scale"]))), max(32, int(round(ch * u * p["tex_scale"])))
    accent = ck.hex_bgr(p["accent"])
    radius = int(round(float(p["corner_radius"]) * min(tw, th)))
    mask, rim = ck.rounded_masks(tw, th, radius, max(2, int(round(0.012 * min(tw, th)))))
    faces = [ck.text_card(it, tw, th, p["theme"]) for it in items]
    badge = ck.check_badge(tw, th, accent, 0.055 * min(tw, th))

    bg_kind = p["background"]
    desk = None
    if bg_kind in ("desk", "dark"):
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2)) / math.sqrt(2)
        base = np.array((236, 232, 232) if bg_kind == "desk" else (30, 28, 28), np.float32)
        k = (-14.0 if bg_kind == "desk" else 10.0) * np.clip(r, 0, 1) ** 2   # a soft falloff toward the corners
        dither = np.random.default_rng(3).uniform(-0.5, 0.5, (h, w, 1)).astype(np.float32)  # no 8-bit banding rings
        desk = np.clip(base + k[..., None] + dither + 0.5, 0, 255).astype(np.uint8)

    # the recipe's alphas were tuned on a light grey desk; over real footage the shadow needs more punch to read
    shadow_scale = float(p["shadow_scale"]) if p["shadow_scale"] is not None else {"desk": 1.0, "dark": 1.4}.get(bg_kind, 1.8)

    STATE.clear()
    STATE.update(dict(
        shadow_scale=shadow_scale, p=p, w=w, h=h, u=u, n=n, cells=cells, cw=cw, ch=ch, pivot=pivot, order=order, flat=flat, tilt=tilt,
        lift_z=lift_z, T=T, loop=info.get("timing") == "loop", events=_timeline(n, order, T, info.get("timing") == "loop"),
        tw=tw, th=th, mask=mask, rim=rim, faces=faces, badge=badge, accent=np.array(accent, np.float32), desk=desk,
        last=order[-1] if order else None, dur=float(info.get("duration") or 0.0), overlay=_title_overlay(p, w, h, u, ry_top, bg_kind, p["theme"]),
    ))


def _title_overlay(p, w, h, u, ry_top, bg_kind, theme):
    """Optional headline + caption drawn once as premultiplied-free BGR + alpha."""
    if not (p["title"] or p["caption"]):
        return None
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    light_bg = bg_kind == "desk"
    ink = (30, 31, 48, 255) if light_bg else (255, 255, 255, 255)
    soft = (110, 112, 130, 255) if light_bg else (235, 235, 235, 255)
    y_top = (ry_top + h / (2 * u)) * u  # region top in px
    y = y_top - 0.045 * 1080 * u
    if p["title"]:
        f = ck.fit_font("inter", str(p["title"]), 0.86 * w, 0.075 * 1080 * u, 800)
        if not light_bg:
            d.text((w / 2 + 0.003 * w, y + 0.003 * w), str(p["title"]), font=f, fill=(0, 0, 0, 110), anchor="ms")
        d.text((w / 2, y), str(p["title"]), font=f, fill=ink, anchor="ms")
        y -= f.size * 1.05
    if p["caption"]:
        fc = ck.font("mono", 0.0165 * 1080 * u)
        text = str(p["caption"]).upper()
        total = sum(fc.getlength(c) + 0.2 * fc.size for c in text)
        ck.tracked(d, (w / 2 - total / 2, y), text, fc, soft, 0.2 * fc.size)
    a = np.asarray(img)
    return a[..., :3][..., ::-1].astype(np.float32), a[..., 3].astype(np.float32) / 255.0


# ----------------------------------------------------------------------------- per frame

def _project(st, x, y, z, tilt_deg):
    """Table coords (units, relative to the frame centre) -> screen px. Tilt is rotateX about the region centre."""
    rx, ry = st["pivot"]
    a = math.radians(tilt_deg)
    ca, sa = math.cos(a), math.sin(a)
    dy = y - ry
    y1, z1 = dy * ca - z * sa, dy * sa + z * ca
    k = st["p"]["persp"] / np.maximum(st["p"]["persp"] - z1, 50.0) if not st["flat"] else 1.0
    return (st["w"] / 2 + (rx + (x - rx) * k) * st["u"], st["h"] / 2 + (ry + y1 * k) * st["u"])


def _quad(st, xc, yc, cw, ch, s, z, tilt):
    sx = np.array([-1, 1, 1, -1]) * cw * s / 2
    sy = np.array([-1, -1, 1, 1]) * ch * s / 2
    X, Y = _project(st, xc + sx, yc + sy, z, tilt)
    return np.stack([X, Y], 1).astype(np.float32)


def apply(frame, t, info):
    if not STATE or STATE["w"] != frame.shape[1] or STATE["h"] != frame.shape[0]:
        setup(info)
    st, p = STATE, STATE["p"]
    f = info["effect_t"] * FPS_FX * float(p["speed"])
    if f < 0:  # timing "once" with --start: nothing before the shot begins
        return frame
    T = st["T"]
    tilt = (p["tilt"][0] + (p["tilt"][1] - p["tilt"][0]) * _c(f / T)) if not st["loop"] else \
           (sum(p["tilt"]) / 2 + (p["tilt"][0] - p["tilt"][1]) / 2 * math.cos(2 * math.pi * f / T))
    tilt = 0.0 if st["flat"] else tilt

    # global fades: intro (staggered drop-in) at the start of the video, outro at its end
    t_abs = float(t)
    g = 1.0
    if st["dur"] > 0 and p["outro"] > 0:
        g = 1.0 - _smooth((t_abs - (st["dur"] - float(p["outro"]))) / float(p["outro"]))
    fi = (t_abs * FPS_FX) if st["loop"] else f

    canvas = _backdrop(frame, st, p)
    ov = st["overlay"]
    if ov is not None:
        a = (ov[1] * g * (_smooth(fi / 10.0) if p["intro"] else 1.0))[..., None]
        canvas[:] = (canvas * (1 - a) + ov[0] * a + 0.5).astype(np.uint8)

    cards = []
    ev_by = {e["i"]: e for e in st["events"]}
    s_sh = st["shadow_scale"]
    for i, (xc, yc) in enumerate(st["cells"]):
        lift, press = _call_state(f, ev_by[i]) if i in ev_by else (0.0, 1.0)
        a_in = _smooth((fi - 3.0 * i) / 8.0) if p["intro"] else 1.0
        sel = _c(lift) if (p["select_last"] and i == st["last"]) else 0.0
        cards.append((lift, i, xc, yc, press, a_in * g, sel, a_in))

    # 1) shadows first: they lie on the table, under every card
    for lift, i, xc, yc, press, alpha, sel, a_in in cards:
        if alpha <= 0.003:
            continue
        L = min(lift, 1.0)
        for (a0, a1, b0, b1, ox, oy, ss, size) in (
                (0.09, 0.22, 14.0, 54.0, 6 + 22 * L, 10 + 36 * L, 1.0, 0.08),     # ambient: lighter at rest, bigger, to the lower right
                (0.32, 0.14, 3.0, 25.0, 2 + 6 * L, 3 + 11 * L, 0.0, 0.03)):       # contact core: dark and tight at rest, wider in the air
            sc = (1.0 + size * L) * press
            quad = _quad(st, xc + ox, yc + oy, st["cw"], st["ch"], sc, 0.0, tilt)
            sigma = (b0 + (b1 - b0) * L) * 0.5 * st["u"]
            ck.shadow_blit(canvas, st["mask"], quad, sigma, (a0 + (a1 - a0) * L) * s_sh * alpha, (36, 26, 20))

    # 2) cards, lowest first so the lifted one is on top
    for lift, i, xc, yc, press, alpha, sel, a_in in sorted(cards, key=lambda c: c[0]):
        if alpha <= 0.003:
            continue
        L = lift
        drop = (1.0 - a_in) * 26.0 if p["intro"] else 0.0
        if st["flat"]:
            s = (1 + (p["lift_scale_2d"] - 1) * L) * press
            quad = _quad(st, xc, yc - p["lift_y"] * L + drop, st["cw"], st["ch"], s, 0.0, 0.0)
        else:
            s = (1 + (p["lift_scale"] - 1) * L) * press
            quad = _quad(st, xc, yc + drop, st["cw"], st["ch"], s, st["lift_z"] * L, tilt)
        tex = st["faces"][i].astype(np.float32)
        if sel > 0.002:
            r = (st["rim"] * sel)[..., None]
            tex = tex * (1 - r) + st["accent"] * r
            ba = (st["badge"][1] * sel)[..., None]
            tex = tex * (1 - ba) + st["badge"][0] * ba
        ck.blit(canvas, ck.pad_tex(tex, st["mask"] * alpha), quad)
    return canvas


def _backdrop(frame, st, p):
    if st["desk"] is not None:
        return st["desk"].copy()
    out = frame
    if p["bg_blur"] > 0:
        out = cv2.GaussianBlur(out, (0, 0), float(p["bg_blur"]) * st["u"])
    if p["bg_dim"] > 0 or p["veil"] > 0:
        f = out.astype(np.float32) * (1 - float(p["bg_dim"]))
        f = f * (1 - float(p["veil"])) + 255.0 * float(p["veil"])
        return np.clip(f, 0, 255).astype(np.uint8)
    return out.copy()
