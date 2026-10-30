"""Shared helpers for card-style plugin effects (circular_ring, element_body_moves, ...).

  * geometry/compositing: anti-aliased rounded-rectangle masks, perspective blit of a BGRA texture onto a quad
  * card faces: logos on a tile, video stills / portrait crops, and PIL-rendered "text cards"
    (tracked mono label, big value, wrapped description, number chips, check badge)

Fonts live in ../fonts (Inter + Space Mono, OFL), logos in ../effects/icons (see NOTICE.md in each).
Text cards need Pillow; everything else needs only OpenCV + numpy.
"""
import math
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ICON_DIR = ROOT / "effects" / "icons"
FONT_DIR = ROOT / "fonts"

# key -> (card title, small tag). Order interleaves the clouds with the tooling so neighbours differ.
DEVOPS = {
    "docker": ("DOCKER", "CONTAINERS"), "aws": ("AWS", "CLOUD"), "kubernetes": ("KUBERNETES", "ORCHESTRATE"),
    "azure": ("AZURE", "CLOUD"), "terraform": ("TERRAFORM", "IaC"), "gcp": ("GCP", "CLOUD"),
    "github": ("GITHUB", "SOURCE"), "packer": ("PACKER", "IMAGES"), "opa": ("OPA", "POLICY"),
    "terragrunt": ("TERRAGRUNT", "IaC"),
}


# ----------------------------------------------------------------------------- masks

def rrect_fill(img, x0, y0, x1, y1, r):
    r = int(max(0, min(r, (x1 - x0) // 2, (y1 - y0) // 2)))
    cv2.rectangle(img, (x0 + r, y0), (x1 - r - 1, y1 - 1), 255, -1)
    cv2.rectangle(img, (x0, y0 + r), (x1 - 1, y1 - r - 1), 255, -1)
    for cx, cy in ((x0 + r, y0 + r), (x1 - r - 1, y0 + r), (x0 + r, y1 - r - 1), (x1 - r - 1, y1 - r - 1)):
        cv2.circle(img, (cx, cy), r, 255, -1, cv2.LINE_8)


def rounded_masks(tw, th, radius, rim):
    """Anti-aliased rounded-rectangle coverage and a rim band just inside its edge (4x supersampled)."""
    ss = 4
    big = np.zeros((th * ss, tw * ss), np.uint8)
    rrect_fill(big, 0, 0, tw * ss, th * ss, radius * ss)
    inner = np.zeros_like(big)
    rrect_fill(inner, rim * ss, rim * ss, (tw - rim) * ss, (th - rim) * ss, max(0, radius - rim) * ss)
    down = lambda m: cv2.resize(m, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    mask, inn = down(big), down(inner)
    return mask, np.clip(mask - inn, 0, 1)


# ----------------------------------------------------------------------------- faces from images

def cover(img, w, h):
    s = max(w / img.shape[1], h / img.shape[0])
    r = cv2.resize(img, (max(w, int(round(img.shape[1] * s))), max(h, int(round(img.shape[0] * s)))), interpolation=cv2.INTER_AREA)
    y0, x0 = (r.shape[0] - h) // 2, (r.shape[1] - w) // 2
    return r[y0:y0 + h, x0:x0 + w]


def fit(img, w, h):
    """Whole frame visible (contain) over a dark blurred fill of itself, so a 16:9 still isn't cropped to a sliver."""
    back = cv2.GaussianBlur(cover(img, w, h), (0, 0), max(2.0, w / 25))
    back = (back.astype(np.float32) * 0.45).astype(np.uint8)
    s = min(w / img.shape[1], h / img.shape[0])
    fw, fh = max(1, int(round(img.shape[1] * s))), max(1, int(round(img.shape[0] * s)))
    front = cv2.resize(img, (fw, fh), interpolation=cv2.INTER_AREA)
    y0, x0 = (h - fh) // 2, (w - fw) // 2
    back[y0:y0 + fh, x0:x0 + fw] = front
    return back


def portrait(f, face, ww, wh):
    """Crop around the head (plus some shoulders) to the card window's aspect."""
    H, W = f.shape[:2]
    fx, fy, fw, fh = face * np.array([W, H, W, H])
    ch = fh * 2.6
    cw = ch * ww / wh
    cx, cy = fx + fw / 2, fy + fh * 0.95
    pad = max(W, H) // 2
    big = cv2.copyMakeBorder(f, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    x0, y0 = int(round(cx - cw / 2)) + pad, int(round(cy - ch / 2)) + pad
    crop = big[max(y0, 0):max(y0, 0) + int(ch), max(x0, 0):max(x0, 0) + int(cw)]
    return cv2.resize(crop, (ww, wh), interpolation=cv2.INTER_AREA)


def load_logo(key):
    """BGRA logo from effects/icons/<key>.png (or an absolute / relative file path)."""
    p = Path(key)
    p = p if p.suffix and p.exists() else ICON_DIR / f"{key}.png"
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(f"no logo for '{key}': put {key}.png in {ICON_DIR}")
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGRA)
    elif im.shape[2] == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
    return im


def logo_face(key, ww, wh, fill=(0.88, 0.62)):
    """A logo centred on a clean tile. Light tile for dark logos, dark tile for white-on-transparent ones (GitHub)."""
    im = load_logo(key)
    a = im[..., 3:4].astype(np.float32) / 255.0
    lum = float((cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2GRAY).astype(np.float32)[..., None] * a).sum() / max(a.sum(), 1e-6))
    dark_tile = lum > 185 and float(a.mean()) < 0.9
    tile = np.empty((wh, ww, 3), np.float32)
    tile[:] = (34, 33, 31) if dark_tile else (246, 246, 243)
    sc = min(fill[0] * ww / im.shape[1], fill[1] * wh / im.shape[0])
    lw, lh = max(1, int(round(im.shape[1] * sc))), max(1, int(round(im.shape[0] * sc)))
    logo = cv2.resize(im, (lw, lh), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC)
    la = np.clip(logo[..., 3:4].astype(np.float32) / 255.0, 0, 1)
    y0, x0 = (wh - lh) // 2, (ww - lw) // 2
    roi = tile[y0:y0 + lh, x0:x0 + lw]
    roi[:] = roi * (1 - la) + logo[..., :3].astype(np.float32) * la
    return np.clip(tile, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------- perspective blit

def pad_tex(bgr, alpha):
    """BGRA texture with a 1px transparent border so warped edges anti-alias."""
    color = cv2.copyMakeBorder(np.clip(bgr, 0, 255).astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    a = cv2.copyMakeBorder(np.clip(alpha * 255, 0, 255).astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    return np.dstack([color, a])


def blit(canvas, tex, dst):
    """Warp a padded BGRA texture onto the quad `dst` (TL, TR, BR, BL in screen px) and alpha-composite it."""
    th, tw = tex.shape[0] - 2, tex.shape[1] - 2
    src = np.float32([[1, 1], [tw + 1, 1], [tw + 1, th + 1], [1, th + 1]])
    h, w = canvas.shape[:2]
    x0, y0 = max(int(math.floor(dst[:, 0].min())) - 1, 0), max(int(math.floor(dst[:, 1].min())) - 1, 0)
    x1, y1 = min(int(math.ceil(dst[:, 0].max())) + 2, w), min(int(math.ceil(dst[:, 1].max())) + 2, h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    M = cv2.getPerspectiveTransform(src, (dst - [x0, y0]).astype(np.float32))
    warped = cv2.warpPerspective(tex, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    a = warped[..., 3:4].astype(np.float32) / 255.0
    roi = canvas[y0:y1, x0:x1]
    roi[:] = (roi * (1 - a) + warped[..., :3] * a + 0.5).astype(np.uint8)


def shadow_blit(canvas, mask, dst, sigma, alpha, tint):
    """Soft shadow: warp a coverage mask (float 0..1) onto `dst`, blur it in screen space, darken the canvas."""
    th, tw = mask.shape
    m = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    src = np.float32([[1, 1], [tw + 1, 1], [tw + 1, th + 1], [1, th + 1]])
    h, w = canvas.shape[:2]
    pad = int(3 * sigma) + 2
    x0, y0 = max(int(math.floor(dst[:, 0].min())) - pad, 0), max(int(math.floor(dst[:, 1].min())) - pad, 0)
    x1, y1 = min(int(math.ceil(dst[:, 0].max())) + pad, w), min(int(math.ceil(dst[:, 1].max())) + pad, h)
    if x1 - x0 < 2 or y1 - y0 < 2 or alpha <= 0.002:
        return
    M = cv2.getPerspectiveTransform(src, (dst - [x0, y0]).astype(np.float32))
    wm = cv2.warpPerspective(m, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if sigma > 0.4:
        wm = cv2.GaussianBlur(wm, (0, 0), sigma)
    a = (wm * alpha)[..., None]
    roi = canvas[y0:y1, x0:x1]
    roi[:] = (roi * (1 - a) + np.asarray(tint, np.float32) * a + 0.5).astype(np.uint8)


# ----------------------------------------------------------------------------- text cards (Pillow)

THEMES = {  # BGR
    "light": dict(card=(252, 252, 252), ink=(48, 31, 30), label=(150, 130, 128), body=(128, 112, 110),
                  chip_on=(64, 46, 45), chip_on_ink=(255, 255, 255), chip_off_ink=(190, 178, 176), pill=(235, 232, 232)),
    "dark": dict(card=(36, 34, 32), ink=(245, 245, 245), label=(150, 150, 150), body=(185, 185, 185),
                 chip_on=(235, 235, 235), chip_on_ink=(30, 30, 30), chip_off_ink=(110, 110, 110), pill=(240, 240, 240)),
}
_FONTS = {}


def hex_bgr(s, default=(214, 91, 91)):
    if isinstance(s, (list, tuple)) and len(s) == 3:
        return tuple(int(v) for v in s)
    try:
        s = str(s).lstrip("#")
        return (int(s[4:6], 16), int(s[2:4], 16), int(s[0:2], 16))
    except ValueError:
        return default


def _pil():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as e:
        raise SystemExit("text cards need Pillow: pip install pillow") from e
    return Image, ImageDraw, ImageFont


def font(kind, px, weight=700):
    """kind 'inter' (variable weight) or 'mono' (Space Mono Bold). Falls back to Helvetica if the bundled font is missing."""
    px = max(6, int(round(px)))
    key = (kind, px, weight)
    if key not in _FONTS:
        _, _, ImageFont = _pil()
        try:
            if kind == "mono":
                f = ImageFont.truetype(str(FONT_DIR / "SpaceMono-Bold.ttf"), px)
            else:
                f = ImageFont.truetype(str(FONT_DIR / "Inter.ttf"), px)
                f.set_variation_by_axes([min(max(px, 14), 32), weight])
        except OSError:
            f = ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", px, index=1 if weight >= 600 else 0)
        _FONTS[key] = f
    return _FONTS[key]


def _wrap(text, f, width):
    lines, cur = [], ""
    for word in str(text).split():
        trial = f"{cur} {word}".strip()
        if cur and f.getlength(trial) > width:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    return lines + ([cur] if cur else [])


def tracked(draw, xy, text, f, fill, tracking):
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=f, fill=fill, anchor="ls")
        x += f.getlength(ch) + tracking
    return x


def fit_font(kind, text, max_w, px, weight):
    while px > 8 and font(kind, px, weight).getlength(text) > max_w:
        px -= 1
    return font(kind, px, weight)


def text_card(item, tw, th, theme="light"):
    """One UI card face as BGR uint8. item keys: title, value, text, icon (logo key), chips, chips_on."""
    Image, ImageDraw, _ = _pil()
    T = THEMES.get(theme, THEMES["light"])
    rgb = lambda c: (c[2], c[1], c[0])
    img = Image.new("RGB", (tw, th), rgb(T["card"]))
    d = ImageDraw.Draw(img)
    tall = th >= 0.7 * tw
    title, value, text = str(item.get("title", "")).upper(), str(item.get("value", "")), item.get("text", "")
    logo = load_logo(item["icon"]) if item.get("icon") else None

    def paste_logo(x, y, max_w, max_h):
        if logo is None:
            return 0
        s = min(max_w / logo.shape[1], max_h / logo.shape[0])
        lw, lh = max(1, int(logo.shape[1] * s)), max(1, int(logo.shape[0] * s))
        l = cv2.resize(logo, (lw, lh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        rgba = Image.fromarray(cv2.cvtColor(l, cv2.COLOR_BGRA2RGBA))
        if theme == "dark":  # logos are drawn for light backgrounds: give them a light pill
            d.rounded_rectangle((x - lh * 0.25, y - lh * 0.2, x + lw + lh * 0.25, y + lh * 1.2), radius=lh * 0.45, fill=rgb(T["pill"]))
        img.paste(rgba, (int(x), int(y)), rgba)
        return lh

    def chips(x, y, size, gap):
        n = int(item.get("chips", 0))
        on = set(item.get("chips_on", []))
        f = font("inter", size * 0.55, 700)
        for i in range(n):
            x0 = x + i * (size + gap)
            if (i + 1) in on:
                d.rounded_rectangle((x0, y - size, x0 + size, y), radius=size * 0.28, fill=rgb(T["chip_on"]))
            d.text((x0 + size / 2, y - size / 2), str(i + 1), font=f, fill=rgb(T["chip_on_ink"] if (i + 1) in on else T["chip_off_ink"]), anchor="mm")

    if tall:
        # flow layout in units of b (the card width, or less if the card is squat) so it holds from tall to nearly square
        b = min(tw, 0.78 * th) if th < tw else tw
        pad = 0.085 * b
        lh = paste_logo(pad, pad, 0.62 * tw, 0.11 * b)
        f_label = fit_font("mono", title, tw - 2 * pad - 0.18 * 0.043 * b * len(title), 0.043 * b, 700)
        y_l = pad + 0.11 * b + 0.085 * b + f_label.size
        tracked(d, (pad, y_l), title, f_label, rgb(T["label"]), 0.18 * f_label.size)
        f_val = fit_font("inter", value, tw - 2 * pad, 0.21 * b, 800)
        y_v = y_l + 0.05 * b + 0.74 * f_val.size
        d.text((pad, y_v), value, font=f_val, fill=rgb(T["ink"]), anchor="ls")
        f_body = font("inter", 0.062 * b, 450)
        has_chips = int(item.get("chips", 0)) > 0
        step = 1.32 * f_body.size
        floor_y = (0.93 * th - 0.075 * b - 0.04 * b) if has_chips else 0.93 * th   # lowest allowed text baseline
        first_min = y_v + 0.09 * b + f_body.size                                      # never closer to the value than this
        room = max(1, int((floor_y - first_min) / step) + 1)
        lines = _wrap(text, f_body, tw - 2 * pad)[:min(3, room)]
        y0 = first_min if has_chips else max(first_min, floor_y - (len(lines) - 1) * step)  # without chips the copy hangs from the bottom
        for i, line in enumerate(lines):
            d.text((pad, y0 + i * step), line, font=f_body, fill=rgb(T["body"]), anchor="ls")
        chips(pad, 0.93 * th, 0.075 * b, 0.02 * b)
    else:  # wide card: identity on the left, description on the right
        pad = 0.14 * th
        paste_logo(pad, pad, 0.4 * tw, 0.2 * th)
        left_w = 0.5 * tw - pad
        f_label = fit_font("mono", title, left_w - 0.18 * 0.085 * th * len(title), 0.085 * th, 700)
        tracked(d, (pad, 0.46 * th), title, f_label, rgb(T["label"]), 0.18 * f_label.size)
        f_val = fit_font("inter", value, left_w, 0.3 * th, 800)
        d.text((pad, 0.8 * th), value, font=f_val, fill=rgb(T["ink"]), anchor="ls")
        x2 = 0.58 * tw
        f_body = font("inter", 0.105 * th, 450)
        step = 1.32 * f_body.size
        n_chips = int(item.get("chips", 0))
        chip = min(0.12 * th, (tw - x2 - pad) / (1.25 * n_chips)) if n_chips else 0.0   # chips shrink to fit the right column
        lines = _wrap(text, f_body, tw - x2 - pad)
        if n_chips:  # copy hangs from the top of the column, above the chip row
            y2 = 0.2 * th + f_body.size
            lines = lines[:max(1, int((0.86 * th - 1.35 * chip - y2) / step) + 1)]
        else:
            lines = lines[:4]
            y2 = 0.5 * th - (len(lines) - 1) * 0.66 * f_body.size
        for i, line in enumerate(lines):
            d.text((x2, y2 + i * step), line, font=f_body, fill=rgb(T["body"]), anchor="ls")
        chips(x2, 0.88 * th, chip, 0.25 * chip)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def check_badge(tw, th, accent, r):
    """Round accent badge with a white check, top-right of the card. -> (BGR float, alpha float) full card size."""
    ss = 4
    big = np.zeros((th * ss, tw * ss, 3), np.uint8)
    cx, cy, R = int((tw - 1.7 * r) * ss), int(1.7 * r * ss), int(r * ss)
    cv2.circle(big, (cx, cy), R, accent, -1, cv2.LINE_AA)
    pts = np.array([[cx - 0.42 * R, cy + 0.02 * R], [cx - 0.1 * R, cy + 0.36 * R], [cx + 0.46 * R, cy - 0.34 * R]], np.int32)
    cv2.polylines(big, [pts], False, (255, 255, 255), max(2, int(0.2 * R)), cv2.LINE_AA)
    cov = np.zeros((th * ss, tw * ss), np.uint8)
    cv2.circle(cov, (cx, cy), R, 255, -1, cv2.LINE_AA)
    bgr = cv2.resize(big, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32)
    a = cv2.resize(cov, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return bgr / np.maximum(a[..., None], 1e-3), a
