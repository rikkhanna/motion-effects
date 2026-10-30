"""Talking-head subject tracking: a soft person matte plus the head position, per frame.

  * Person matte: MediaPipe Selfie Segmenter (models/selfie_segmenter.tflite, runs on CPU, offline).
  * Head position: YuNet face detector via OpenCV (models/yunet.onnx); gaps are interpolated and the
    box smoothed so anything anchored to the head doesn't jitter.
  * The matte gets a contrast curve (the raw confidence is soft) and a 3-tap temporal blend against shimmer.

Both models are tiny and ship in models/ (see models/NOTICE.md for licences). Edges are those of a
256px segmenter: good for a person against a room, soft on fine hair.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

MODELS = Path(__file__).resolve().parent.parent / "models"


def _log(*a):
    print(*a, file=sys.stderr, flush=True)


def _fill_and_smooth(boxes, fps):
    n = len(boxes)
    idx = np.array([i for i, b in enumerate(boxes) if b is not None])
    if idx.size == 0:
        return None
    arr = np.array([boxes[i] for i in idx], float)
    out = np.stack([np.interp(np.arange(n), idx, arr[:, k]) for k in range(4)], 1)
    out = median_filter(out, size=(5, 1), mode="nearest")
    return gaussian_filter1d(out, max(1.0, fps * 0.12), axis=0, mode="nearest")


def track_subject(src, fps=25.0, width=640, limit=None, log=_log):
    """-> dict(matte=(N,h,w) float16 0..1, face=(N,4) x,y,w,h as fractions of the frame, size=(h,w)) or None."""
    import mediapipe as mp  # imported here so effects that don't need a subject never pay for it

    V = mp.tasks.vision
    seg = V.ImageSegmenter.create_from_options(V.ImageSegmenterOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODELS / "selfie_segmenter.tflite")),
        running_mode=V.RunningMode.IMAGE, output_confidence_masks=True, output_category_mask=False))

    cap = cv2.VideoCapture(str(src))
    faces, mattes, size, det = [], [], None, None
    log("subject: segmenting the person and tracking the face...")
    while limit is None or len(mattes) < limit:
        ok, f = cap.read()
        if not ok:
            break
        h = int(round(f.shape[0] * width / f.shape[1]))
        f = cv2.resize(f, (width, h), interpolation=cv2.INTER_AREA)
        if det is None:
            size = (h, width)
            det = cv2.FaceDetectorYN.create(str(MODELS / "yunet.onnx"), "", (width, h), 0.6)
        _, found = det.detect(f)
        faces.append(None if found is None or not len(found) else tuple(max(found, key=lambda r: r[2] * r[3])[:4]))
        res = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
        c = np.array(res.confidence_masks[0].numpy_view(), np.float32).reshape(h, width)
        t = np.clip((c - 0.5) / 0.3, 0, 1)  # soft confidence -> usable matte, biased tight so no wall halo
        mattes.append(t * t * (3 - 2 * t))
        if len(mattes) % 100 == 0:
            log(f"  {len(mattes)} frames")
    if not mattes:
        return None
    sm = _fill_and_smooth(faces, fps)
    if sm is None:
        log("subject: no face found in the video")
        return None
    log(f"subject: face seen in {sum(b is not None for b in faces)}/{len(faces)} frames")
    m = np.stack(mattes)
    if len(m) > 2:
        p = np.concatenate([m[:1], m, m[-1:]])
        m = 0.5 * p[1:-1] + 0.25 * p[:-2] + 0.25 * p[2:]
    h, w = size
    face = np.stack([sm[:, 0] / w, sm[:, 1] / h, sm[:, 2] / w, sm[:, 3] / h], 1)
    return {"matte": m.astype(np.float16), "face": face, "size": size}
