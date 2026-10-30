"""Example plugin layer: a progress bar and a pulsing corner ring drawn over the video.

A plugin is a python file next to the effect JSON (effects/<name>.py) that the spec
points at with  "plugin": "<name>.py".  The built-in tracks run first, then apply().

    setup(info)            optional, called once. info = {w, h, fps, duration, n_frames}
    apply(frame, t, info)  frame is a BGR uint8 ndarray; return the new frame.
                           info also has frame_index and effect_t (time on the effect's own timeline).

Keep it deterministic: derive everything from t / frame_index so re-renders are identical.
"""
import math

import cv2


def apply(frame, t, info):
    h, w = frame.shape[:2]
    out = frame.copy()
    # progress bar along the bottom
    prog = min(1.0, t / max(info["duration"], 1e-6))
    cv2.rectangle(out, (0, h - max(4, h // 90)), (int(w * prog), h), (255, 255, 255), -1)
    # pulsing ring, sized relative to the frame so it scales with resolution
    pulse = 0.5 + 0.5 * math.sin(2 * math.pi * 2.0 * info["effect_t"])
    cv2.circle(out, (int(w * 0.9), int(h * 0.12)), int(h * (0.04 + 0.02 * pulse)), (0, 200, 255), max(2, h // 180), cv2.LINE_AA)
    return out
