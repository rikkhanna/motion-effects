"""Built-in effects so /effect works before anything has been learned.

These are ordinary specs, the same shape `learn` writes, so they double as
examples of how to hand-author or tweak an effect.
"""
import numpy as np


def _noise(dur, fps, amp, seed, smooth=1):
    n = int(dur * fps)
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n + 2 * smooth)
    if smooth > 1:
        x = np.convolve(x, np.ones(smooth) / smooth, mode="same")
    x = x[smooth:smooth + n]
    x[-1] = x[0]  # loop seamlessly
    return [[round(i / fps, 4), round(float(v) * amp, 5)] for i, v in enumerate(x)]


BUILTIN = {
    "zoom_punch": {
        "description": "Beat-style punch-in with a white flash and RGB fringing, repeats every second.",
        "duration": 1.0, "timing": "loop",
        "tracks": {
            "zoom": [[0, 1.0], [0.1, 1.22], [0.6, 1.0], [1.0, 1.0]],
            "brightness": [[0, 0], [0.05, 0.22], [0.3, 0], [1.0, 0]],
            "rgb_split": [[0, 0], [0.1, 0.006], [0.5, 0], [1.0, 0]],
        },
    },
    "shake": {
        "description": "Handheld camera shake with slight roll; edges are mirrored so no black borders.",
        "duration": 2.0, "timing": "loop",
        "tracks": {
            "zoom": 1.04,
            "shift_x": _noise(2.0, 30, 0.010, 1, 3),
            "shift_y": _noise(2.0, 30, 0.010, 2, 3),
            "rotate": _noise(2.0, 30, 0.5, 3, 4),
        },
    },
    "glitch": {
        "description": "Digital glitch bursts: torn horizontal bands with RGB split, twice every three seconds.",
        "duration": 3.0, "timing": "loop", "glitch_amp": 0.07,
        "tracks": {
            "glitch": [[0, 0], [0.5, 0], [0.55, 0.9], [0.8, 0.35], [0.85, 0], [1.8, 0], [1.85, 1.0], [2.1, 0.25], [2.15, 0], [3.0, 0]],
            "rgb_split": [[0, 0], [0.5, 0], [0.55, 0.012], [0.85, 0.002], [0.9, 0], [1.8, 0], [1.85, 0.014], [2.15, 0], [3.0, 0]],
        },
    },
    "vhs": {
        "description": "Worn VHS look: soft, fringed, grainy, slightly oversaturated with a gentle vignette and wobble.",
        "duration": 4.0, "timing": "loop",
        "tracks": {
            "blur": 0.0012, "rgb_split": 0.003, "grain": 0.03, "vignette": 0.35,
            "saturation": 1.15, "contrast": 1.1,
            "shift_x": _noise(4.0, 30, 0.0015, 5, 5),
        },
    },
    "flash_pulse": {
        "description": "Rhythmic white flash with a saturation kick, twice a second.",
        "duration": 0.5, "timing": "loop",
        "tracks": {
            "brightness": [[0, 0], [0.04, 0.45], [0.25, 0], [0.5, 0]],
            "saturation": [[0, 1], [0.04, 1.4], [0.3, 1], [0.5, 1]],
        },
    },
}
