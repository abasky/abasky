# -*- coding: utf-8 -*-
"""
IQ AI — Abasky
Offline AI:
1. MODNet: Portrait matting / background removal.
2. U2-Net: Background removal fallback.
3. YuNet: Face detection.

All processing runs locally through OpenCV DNN.
No internet connection is required during inference.
"""

import os
import sys
import threading

import cv2
import numpy as np


# ---------------------------------------------------------------------
# Model names
# ---------------------------------------------------------------------

MODNET_MODEL = "modnet-photographic-portrait-matting.onnx"

BG_MODELS = (
    "u2net_human_seg.onnx",
    "u2netp.onnx",
)

FACE_MODELS = (
    "face_detection_yunet_2023mar.onnx",
    "face_detection_yunet_2022mar.onnx",
)


# ---------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------

_modnet_net = None
_modnet_name = None

_bg_net = None
_bg_name = None

_face_det = None
_face_name = None

_loaded = False

_model_lock = threading.RLock()


# ---------------------------------------------------------------------
# Model directories
# ---------------------------------------------------------------------

def models_dirs():
    base = os.path.dirname(
        os.path.abspath(
            sys.executable if getattr(sys, "frozen", False) else __file__
        )
    )

    dirs = [
        os.path.join(base, "models")
    ]

    appdata = os.environ.get("APPDATA")

    if appdata:
        dirs.append(
            os.path.join(appdata, "AbaskyIQ", "models")
        )

    return dirs


def find_model(names):
    for directory in models_dirs():

        for name in names:

            path = os.path.join(directory, name)

            if (
                os.path.isfile(path)
                and os.path.getsize(path) > 100000
            ):
                return path

    return None


# ---------------------------------------------------------------------
# ONNX loader
# ---------------------------------------------------------------------

def _read_bytes_net(path):
    """
    Load ONNX from memory.
    This avoids Windows path problems with Arabic characters.
    """

    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8)

    return cv2.dnn.readNetFromONNX(data)


# ---------------------------------------------------------------------
# AI initialization
# ---------------------------------------------------------------------

def load():
    """
    Load available AI models.

    Priority:
        MODNet
        U2-Net
        YuNet

    Returns:
        (success, status_message)
    """

    global _modnet_net
    global _modnet_name

    global _bg_net
    global _bg_name

    global _face_det
    global _face_name

    global _loaded

    with _model_lock:

        if _loaded:

            ready = (
                _modnet_net is not None
                or _bg_net is not None
                or _face_det is not None
            )

            return ready, status_text()

        if not hasattr(cv2, "dnn"):
            return False, "نسخة OpenCV لا تدعم DNN."

        # -------------------------------------------------------------
        # MODNet
        # -------------------------------------------------------------

        modnet_path = find_model((MODNET_MODEL,))

        if modnet_path:

            try:

                _modnet_net = _read_bytes_net(modnet_path)

                _modnet_name = MODNET_MODEL

            except Exception:

                _modnet_net = None
                _modnet_name = None

        # -------------------------------------------------------------
        # U2-Net fallback
        # -------------------------------------------------------------

        for name in BG_MODELS:

            path = find_model((name,))

            if path is None:
                continue

            try:

                _bg_net = _read_bytes_net(path)

                _bg_name = name

                break

            except Exception:

                _bg_net = None
                _bg_name = None

        # -------------------------------------------------------------
        # YuNet face detector
        # -------------------------------------------------------------

        if hasattr(cv2, "FaceDetectorYN"):

            for name in FACE_MODELS:

                path = find_model((name,))

                if path is None:
                    continue

                try:

                    with open(path, "rb") as f:
                        buf = np.frombuffer(f.read(), np.uint8)

                    _face_det = cv2.FaceDetectorYN.create(
                        "onnx",
                        buf,
                        np.empty(0, np.uint8),
                        (320, 320),
                        0.6,
                        0.3,
                        1000
                    )

                    _face_name = name

                    break

                except Exception:

                    _face_det = None
                    _face_name = None

        _loaded = True

        ready = (
            _modnet_net is not None
            or _bg_net is not None
            or _face_det is not None
        )

        return ready, status_text()


# ---------------------------------------------------------------------
# Model status
# ---------------------------------------------------------------------

def status_text():

    parts = []

    parts.append(
        "MODNet: %s" % (
            _modnet_name
            if _modnet_net is not None
            else "غير متوفر"
        )
    )

    parts.append(
        "U2-Net: %s" % (
            _bg_name
            if _bg_net is not None
            else "غير متوفر"
        )
    )

    parts.append(
        "الوجه: %s" % (
            _face_name
            if _face_det is not None
            else "غير متوفر (الطريقة العادية)"
        )
    )

    return " | ".join(parts)


def missing_models_text():

    need = []

    if find_model(BG_MODELS) is None:
        need.append("u2netp.onnx")

    if find_model(FACE_MODELS) is None:
        need.append("face_detection_yunet_2023mar.onnx")

    return need


# ---------------------------------------------------------------------
# MODNet inference
# ---------------------------------------------------------------------

def _modnet_mask(bgr, size=512):
    """
    MODNet portrait matting.

    Expected input:
        1x3x512x512
        BGR
        Mean = 127.5
        Scale = 1/127.5

    Returns:
        Alpha mask in range 0..1.
    """

    if _modnet_net is None:
        return None

    if bgr is None or bgr.size == 0:
        return None

    h, w = bgr.shape[:2]

    if h < 2 or w < 2:
        return None

    resized = cv2.resize(
        bgr,
        (size, size),
        interpolation=cv2.INTER_AREA
    )

    blob = cv2.dnn.blobFromImage(
        resized,
        scalefactor=1.0 / 127.5,
        size=(size, size),
        mean=(127.5, 127.5, 127.5),
        swapRB=False,
        crop=False
    )

    with _model_lock:

        _modnet_net.setInput(blob)

        output = _modnet_net.forward()

    if isinstance(output, (tuple, list)):

        if not output:
            return None

        output = output[0]

    matte = np.asarray(
        output,
        dtype=np.float32
    ).squeeze()

    if matte.ndim != 2:
        return None

    if not np.isfinite(matte).all():
        return None

    matte = np.clip(matte, 0.0, 1.0)

    if matte.shape != (h, w):

        matte = cv2.resize(
            matte,
            (w, h),
            interpolation=cv2.INTER_CUBIC
        )

    matte = np.clip(
        matte,
        0.0,
        1.0
    )

    return matte.astype(np.float32)


# ---------------------------------------------------------------------
# U2-Net inference
# ---------------------------------------------------------------------

def _u2_mask(bgr, size=320):

    if _bg_net is None:
        return None

    if bgr is None or bgr.size == 0:
        return None

    rgb = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2RGB
    )

    x = cv2.resize(
        rgb,
        (size, size),
        interpolation=cv2.INTER_AREA
    ).astype(np.float32)

    x = x / max(float(x.max()), 1e-6)

    mean = np.array(
        [0.485, 0.456, 0.406],
        np.float32
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        np.float32
    )

    x = (x - mean) / std

    blob = np.ascontiguousarray(
        x.transpose(2, 0, 1)[None]
    )

    with _model_lock:

        _bg_net.setInput(blob)

        output = _bg_net.forward()

    m = output[0, 0].astype(np.float32)

    lo = float(m.min())
    hi = float(m.max())

    m = (m - lo) / (hi - lo + 1e-9)

    return m


# ---------------------------------------------------------------------
# Face detection
# ---------------------------------------------------------------------

def face_box(bgr):
    """
    Returns:
        (x, y, width, height)
    or None.
    """

    if _face_det is None:
        return None

    if bgr is None or bgr.size == 0:
        return None

    h, w = bgr.shape[:2]

    scale = min(
        1.0,
        800.0 / max(h, w)
    )

    if scale < 1.0:

        small = cv2.resize(
            bgr,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )

    else:

        small = bgr

    sh, sw = small.shape[:2]

    with _model_lock:

        _face_det.setInputSize((sw, sh))

        _ret, faces = _face_det.detect(small)

    if faces is None or len(faces) == 0:
        return None

    best = max(
        faces,
        key=lambda r: float(r[2]) * float(r[3]) * float(r[-1])
    )

    x, y, fw, fh = [
        float(v) / scale
        for v in best[:4]
    ]

    return (x, y, fw, fh)


# ---------------------------------------------------------------------
# MODNet alpha processing
# ---------------------------------------------------------------------

def _modnet_alpha(bgr, face=None):
    """
    Process MODNet matte and validate its result.
    """

    try:

        alpha = _modnet_mask(bgr)

        if alpha is None:
            return None

        if not np.isfinite(alpha).all():
            return None

        h, w = bgr.shape[:2]

        if alpha.shape != (h, w):

            alpha = cv2.resize(
                alpha,
                (w, h),
                interpolation=cv2.INTER_CUBIC
            )

        alpha = np.clip(
            alpha,
            0.0,
            1.0
        )

        mean_value = float(alpha.mean())

        # Reject obviously invalid masks.
        if mean_value < 0.02 or mean_value > 0.98:
            return None

        # Light smoothing to reduce small edge noise.
        alpha = cv2.GaussianBlur(
            alpha,
            (0, 0),
            0.7
        )

        alpha = np.clip(
            alpha,
            0.0,
            1.0
        )

        return alpha.astype(np.float32)

    except Exception:

        return None


# ---------------------------------------------------------------------
# U2-Net alpha processing
# ---------------------------------------------------------------------

def _u2_alpha(bgr, face=None):
    """
    Existing U2-Net processing.
    Kept as fallback for MODNet.
    """

    if _bg_net is None:
        return None

    h, w = bgr.shape[:2]

    try:

        m = _u2_mask(bgr)

        if m is None:
            return None

    except Exception:

        return None

    m = cv2.resize(
        m,
        (w, h),
        interpolation=cv2.INTER_CUBIC
    )

    a = np.clip(
        (m - 0.30) / 0.40,
        0.0,
        1.0
    )

    hard = (a > 0.5).astype(np.uint8)

    n, lab, stats, _c = cv2.connectedComponentsWithStats(
        hard,
        connectivity=8
    )

    if n <= 1:
        return None

    keep = None

    if face is not None:

        fx, fy, fw, fh = face

        cx = int(
            min(max(fx + fw / 2, 0), w - 1)
        )

        cy = int(
            min(max(fy + fh / 2, 0), h - 1)
        )

        if lab[cy, cx] > 0:
            keep = int(lab[cy, cx])

    if keep is None:

        keep = 1 + int(
            np.argmax(
                stats[1:, cv2.CC_STAT_AREA]
            )
        )

    comp = (lab == keep).astype(np.uint8)

    comp = cv2.dilate(
        comp,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5)
        )
    )

    # Fill internal holes.
    inv = (
        (1 - comp).astype(np.uint8) * 255
    )

    ff = inv.copy()

    cv2.floodFill(
        ff,
        np.zeros((h + 2, w + 2), np.uint8),
        (0, 0),
        0
    )

    comp = np.maximum(
        comp,
        (ff > 0).astype(np.uint8)
    )

    alpha = (
        a
        * cv2.GaussianBlur(
            comp.astype(np.float32),
            (0, 0),
            1.5
        )
    )

    alpha = np.where(
        comp > 0,
        np.maximum(alpha, 0.0),
        0.0
    )

    core = cv2.erode(
        comp,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (9, 9)
        )
    )

    alpha = np.maximum(
        alpha,
        core.astype(np.float32)
    )

    frac = float(alpha.mean())

    if frac < 0.06 or frac > 0.95:
        return None

    return alpha.astype(np.float32)


# ---------------------------------------------------------------------
# Public background removal interface
# ---------------------------------------------------------------------

def person_alpha(bgr, face=None):
    """
    Main background-removal interface.

    Priority:
        1. MODNet
        2. U2-Net
        3. None (caller uses its normal fallback)
    """

    if bgr is None or bgr.size == 0:
        return None

    # MODNet first.
    if _modnet_net is not None:

        alpha = _modnet_alpha(
            bgr,
            face=face
        )

        if alpha is not None:
            return alpha

    # U2-Net fallback.
    if _bg_net is not None:

        alpha = _u2_alpha(
            bgr,
            face=face
        )

        if alpha is not None:
            return alpha

    return None


# ---------------------------------------------------------------------
# Salient masks
# ---------------------------------------------------------------------

def salient_masks(bgr):
    """
    Binary masks used by the existing smart ID detection system.

    This keeps the same interface expected by iq_core.py.
    """

    if _bg_net is None:
        return []

    if bgr is None or bgr.size == 0:
        return []

    h, w = bgr.shape[:2]

    try:

        m = _u2_mask(bgr)

        if m is None:
            return []

    except Exception:

        return []

    m = cv2.resize(
        m,
        (w, h),
        interpolation=cv2.INTER_CUBIC
    )

    return [
        (m > threshold).astype(np.uint8) * 255
        for threshold in (0.25, 0.45, 0.65)
    ]


# ---------------------------------------------------------------------
# Self test
# ---------------------------------------------------------------------

def selftest():
    """
    Basic model self-test.

    Intended to run in the isolated process used by the application.
    """

    ok, _msg = load()

    if not ok:
        return False

    img = np.random.RandomState(1).randint(
        0,
        255,
        (200, 160, 3)
    ).astype(np.uint8)

    if _modnet_net is not None:

        _modnet_mask(img)

    if _bg_net is not None:

        _u2_mask(img)

    if _face_det is not None:

        face_box(img)

    return True
