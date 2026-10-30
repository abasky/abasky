# -*- coding: utf-8 -*-
"""
IQ AI — Abasky

Offline models:
1. MODNet: ONNX Runtime portrait matting.
2. U2-Net: OpenCV DNN background-removal fallback.
3. YuNet: OpenCV face detection.

MODNet is optional. If it fails, U2-Net and YuNet remain available.
"""

import os
import sys
import threading

import cv2
import numpy as np

try:
    import onnxruntime as ort
    _ORT_IMPORT_ERROR = None
except Exception as exc:
    ort = None
    _ORT_IMPORT_ERROR = str(exc)


# ---------------------------------------------------------------------
# Model names
# ---------------------------------------------------------------------

MODNET_MODELS = (
    "modnet-photographic-portrait-matting.onnx",
    "modnet_photographic.onnx",
)

BG_MODELS = (
    "u2net_human_seg.onnx",
    "u2netp.onnx",
)

FACE_MODELS = (
    "face_detection_yunet_2023mar.onnx",
    "face_detection_yunet_2022mar.onnx",
)


# ---------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------

_modnet_session = None
_modnet_input = None
_modnet_output = None
_modnet_name = None
_modnet_error = None

_bg_net = None
_bg_name = None
_bg_error = None

_face_det = None
_face_name = None
_face_error = None

_loaded = False
_model_lock = threading.RLock()


# ---------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------

def models_dirs():
    if getattr(sys, "frozen", False):
        base = os.path.dirname(os.path.abspath(sys.executable))
    else:
        base = os.path.dirname(os.path.abspath(__file__))

    dirs = [os.path.join(base, "models")]

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

            try:
                if (
                    os.path.isfile(path)
                    and os.path.getsize(path) > 100000
                ):
                    return path
            except OSError:
                pass

    return None


def _read_bytes_net(path):
    """OpenCV ONNX loader; keeps compatibility with Arabic paths."""
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8)

    return cv2.dnn.readNetFromONNX(data)


# ---------------------------------------------------------------------
# MODNet loader — ONNX Runtime
# ---------------------------------------------------------------------

def _load_modnet():
    global _modnet_session
    global _modnet_input
    global _modnet_output
    global _modnet_name
    global _modnet_error

    path = find_model(MODNET_MODELS)

    if path is None:
        _modnet_error = "ملف MODNet غير موجود داخل مجلد models."
        return False

    if ort is None:
        _modnet_error = (
            "مكتبة onnxruntime غير مثبتة أو تعذر تحميلها: "
            + str(_ORT_IMPORT_ERROR)
        )
        return False

    try:
        options = ort.SessionOptions()

        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        # ---------------------------------------------------------
        # CPU performance tuning
        # ---------------------------------------------------------

        # لا نسمح للموديل باستهلاك جميع أنوية المعالج.
        # نترك موارد لبقية البرنامج وعمليات OpenCV.
        cpu_count = os.cpu_count() or 2

        options.intra_op_num_threads = max(
            1,
            min(4, cpu_count),
        )

        # تنفيذ العمليات المتوازية بين العقد غير مطلوب هنا.
        options.inter_op_num_threads = 1

        # يقلل التنافس بين مجموعات خيوط ONNX.
        options.execution_mode = (
            ort.ExecutionMode.ORT_SEQUENTIAL
        )

        options.enable_cpu_mem_arena = True
        options.enable_mem_pattern = True

        # CPU provider: يعمل دون الحاجة إلى CUDA.
        session = ort.InferenceSession(
            path,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

        inputs = session.get_inputs()
        outputs = session.get_outputs()

        if not inputs or not outputs:
            raise RuntimeError("الموديل لا يحتوي على مدخلات أو مخرجات.")

        input_info = inputs[0]
        output_info = outputs[0]

        input_shape = input_info.shape

        # MODNet photographic expects NCHW with 3 channels.
        if len(input_shape) != 4:
            raise RuntimeError(
                "شكل إدخال الموديل غير متوقع: %s" % (input_shape,)
            )

        if isinstance(input_shape[1], int) and input_shape[1] != 3:
            raise RuntimeError(
                "الموديل لا يستقبل 3 قنوات: %s" % (input_shape,)
            )

        _modnet_session = session
        _modnet_input = input_info.name
        _modnet_output = output_info.name
        _modnet_name = os.path.basename(path)
        _modnet_error = None

        return True

    except Exception as exc:
        _modnet_session = None
        _modnet_input = None
        _modnet_output = None
        _modnet_name = None
        _modnet_error = "%s: %s" % (
            type(exc).__name__,
            str(exc),
        )
        return False


# ---------------------------------------------------------------------
# U2-Net loader — OpenCV DNN
# ---------------------------------------------------------------------

def _load_u2net():
    global _bg_net
    global _bg_name
    global _bg_error

    for name in BG_MODELS:
        path = find_model((name,))

        if path is None:
            continue

        try:
            net = _read_bytes_net(path)

            _bg_net = net
            _bg_name = name
            _bg_error = None

            return True

        except Exception as exc:
            _bg_net = None
            _bg_name = None
            _bg_error = "%s: %s" % (
                type(exc).__name__,
                str(exc),
            )

    if _bg_error is None:
        _bg_error = "ملف U2-Net غير موجود."

    return False


# ---------------------------------------------------------------------
# YuNet loader
# ---------------------------------------------------------------------

def _load_yunet():
    global _face_det
    global _face_name
    global _face_error

    if not hasattr(cv2, "FaceDetectorYN"):
        _face_error = "نسخة OpenCV لا توفر FaceDetectorYN."
        return False

    for name in FACE_MODELS:
        path = find_model((name,))

        if path is None:
            continue

        try:
            with open(path, "rb") as f:
                buf = np.frombuffer(f.read(), np.uint8)

            detector = cv2.FaceDetectorYN.create(
                "onnx",
                buf,
                np.empty(0, np.uint8),
                (320, 320),
                0.6,
                0.3,
                1000,
            )

            _face_det = detector
            _face_name = name
            _face_error = None

            return True

        except Exception as exc:
            _face_det = None
            _face_name = None
            _face_error = "%s: %s" % (
                type(exc).__name__,
                str(exc),
            )

    if _face_error is None:
        _face_error = "ملف YuNet غير موجود."

    return False


# ---------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------

def load():
    """
    Loads all available models independently.

    MODNet failure does not disable U2-Net or YuNet.
    Returns (success, status_message).
    """

    global _loaded

    with _model_lock:
        if _loaded:
            ready = any((
                _modnet_session is not None,
                _bg_net is not None,
                _face_det is not None,
            ))
            return ready, status_text()

        if not hasattr(cv2, "dnn"):
            return False, "نسخة OpenCV لا تدعم DNN."

        _load_modnet()
        _load_u2net()
        _load_yunet()

        _loaded = True

        ready = any((
            _modnet_session is not None,
            _bg_net is not None,
            _face_det is not None,
        ))

        return ready, status_text()


# ---------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------

def status_text():
    modnet_status = (
        _modnet_name
        if _modnet_session is not None
        else "غير متوفر"
    )

    u2_status = (
        _bg_name
        if _bg_net is not None
        else "غير متوفر"
    )

    face_status = (
        _face_name
        if _face_det is not None
        else "غير متوفر"
    )

    parts = [
        "MODNet: %s" % modnet_status,
        "U2-Net: %s" % u2_status,
        "YuNet: %s" % face_status,
    ]

    if _modnet_session is None and _modnet_error:
        parts.append("سبب MODNet: %s" % _modnet_error)

    if _bg_net is None and _bg_error:
        parts.append("سبب U2-Net: %s" % _bg_error)

    return " | ".join(parts)


def missing_models_text():
    need = []

    if find_model(BG_MODELS) is None:
        need.append("u2netp.onnx")

    if find_model(FACE_MODELS) is None:
        need.append("face_detection_yunet_2023mar.onnx")

    return need


# ---------------------------------------------------------------------
# MODNet preprocessing and inference
# ---------------------------------------------------------------------

def _modnet_prepare(bgr, input_size=512):
    """
    Official yakhyo MODNet preprocessing:
    BGR -> RGB -> preserve aspect ratio -> multiples of 32
    -> float32 [-1, 1] -> NCHW.
    """

    if bgr is None or bgr.size == 0:
        raise ValueError("الصورة فارغة.")

    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("الصورة يجب أن تحتوي على ثلاث قنوات.")

    original_h, original_w = bgr.shape[:2]

    if original_h < 2 or original_w < 2:
        raise ValueError("أبعاد الصورة غير صالحة.")

    rgb = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2RGB,
    )

    if (
        max(original_h, original_w) < input_size
        or min(original_h, original_w) > input_size
    ):
        if original_w >= original_h:
            new_h = input_size
            new_w = int(
                original_w / original_h * input_size
            )
        else:
            new_w = input_size
            new_h = int(
                original_h / original_w * input_size
            )
    else:
        new_h = original_h
        new_w = original_w

    new_h = max(32, new_h - (new_h % 32))
    new_w = max(32, new_w - (new_w % 32))

    resized = cv2.resize(
        rgb,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )

    x = resized.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5

    x = np.transpose(x, (2, 0, 1))
    tensor = np.expand_dims(x, axis=0).astype(np.float32)

    return tensor, original_h, original_w


def _modnet_mask(bgr):
    """
    Returns full-resolution float32 matte in range 0..1.
    """

    if _modnet_session is None:
        return None

    tensor, original_h, original_w = _modnet_prepare(bgr)

    with _model_lock:
        outputs = _modnet_session.run(
            [_modnet_output],
            {_modnet_input: tensor},
        )

    if not outputs:
        raise RuntimeError("MODNet لم يرجع أي مخرجات.")

    raw = np.asarray(
        outputs[0],
        dtype=np.float32,
    )

    if raw.ndim != 4 or raw.shape[0] != 1 or raw.shape[1] != 1:
        raise RuntimeError(
            "شكل خرج MODNet غير متوقع: %s" % (raw.shape,)
        )

    matte = raw[0, 0]

    if not np.isfinite(matte).all():
        raise RuntimeError("خرج MODNet يحتوي قيماً غير صالحة.")

    matte = cv2.resize(
        matte,
        (original_w, original_h),
        interpolation=cv2.INTER_AREA,
    )

    matte = np.clip(matte, 0.0, 1.0)

    return matte.astype(np.float32)


def _modnet_alpha(bgr, face=None):
    try:
        alpha = _modnet_mask(bgr)

        if alpha is None:
            return None

        mean_value = float(alpha.mean())

        if mean_value < 0.005 or mean_value > 0.995:
            return None

        alpha = cv2.GaussianBlur(
            alpha,
            (0, 0),
            0.45,
        )

        return np.clip(
            alpha,
            0.0,
            1.0,
        ).astype(np.float32)

    except Exception as exc:
        global _modnet_error
        _modnet_error = "%s: %s" % (
            type(exc).__name__,
            str(exc),
        )
        return None


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
        cv2.COLOR_BGR2RGB,
    )

    x = cv2.resize(
        rgb,
        (size, size),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)

    x = x / max(float(x.max()), 1e-6)

    mean = np.array(
        [0.485, 0.456, 0.406],
        np.float32,
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        np.float32,
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

    return (m - lo) / (hi - lo + 1e-9)


def _u2_alpha(bgr, face=None):
    """
    Original U2-Net alpha processing retained as fallback.
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
        interpolation=cv2.INTER_CUBIC,
    )

    a = np.clip(
        (m - 0.30) / 0.40,
        0.0,
        1.0,
    )

    hard = (a > 0.5).astype(np.uint8)

    n, lab, stats, _c = cv2.connectedComponentsWithStats(
        hard,
        connectivity=8,
    )

    if n <= 1:
        return None

    keep = None

    if face is not None:
        fx, fy, fw, fh = face

        cx = int(min(max(fx + fw / 2, 0), w - 1))
        cy = int(min(max(fy + fh / 2, 0), h - 1))

        if lab[cy, cx] > 0:
            keep = int(lab[cy, cx])

    if keep is None:
        keep = 1 + int(
            np.argmax(stats[1:, cv2.CC_STAT_AREA])
        )

    comp = (lab == keep).astype(np.uint8)

    comp = cv2.dilate(
        comp,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5),
        ),
    )

    inv = (1 - comp).astype(np.uint8) * 255
    ff = inv.copy()

    cv2.floodFill(
        ff,
        np.zeros((h + 2, w + 2), np.uint8),
        (0, 0),
        0,
    )

    comp = np.maximum(
        comp,
        (ff > 0).astype(np.uint8),
    )

    alpha = a * cv2.GaussianBlur(
        comp.astype(np.float32),
        (0, 0),
        1.5,
    )

    alpha = np.where(
        comp > 0,
        np.maximum(alpha, 0.0),
        0.0,
    )

    core = cv2.erode(
        comp,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (9, 9),
        ),
    )

    alpha = np.maximum(
        alpha,
        core.astype(np.float32),
    )

    frac = float(alpha.mean())

    if frac < 0.06 or frac > 0.95:
        return None

    return alpha.astype(np.float32)


# ---------------------------------------------------------------------
# Face detection
# ---------------------------------------------------------------------

def face_box(bgr):
    """Returns (x, y, width, height), or None."""

    if _face_det is None:
        return None

    if bgr is None or bgr.size == 0:
        return None

    h, w = bgr.shape[:2]

    scale = min(
        1.0,
        800.0 / max(h, w),
    )

    if scale < 1.0:
        small = cv2.resize(
            bgr,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
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
        key=lambda r: float(r[2]) * float(r[3]) * float(r[-1]),
    )

    x, y, fw, fh = [
        float(v) / scale
        for v in best[:4]
    ]

    return (x, y, fw, fh)


# ---------------------------------------------------------------------
# Public background-removal interface
# ---------------------------------------------------------------------

def person_alpha(bgr, face=None):
    """
    Priority:
        1. MODNet
        2. U2-Net
        3. None -> caller fallback
    """

    if bgr is None or bgr.size == 0:
        return None

    if _modnet_session is not None:
        alpha = _modnet_alpha(bgr, face)

        if alpha is not None:
            return alpha

    if _bg_net is not None:
        return _u2_alpha(bgr, face)

    return None


# ---------------------------------------------------------------------
# Salient masks — U2-Net only
# ---------------------------------------------------------------------

def salient_masks(bgr):
    """
    Existing interface used by iq_core.py.
    MODNet is not used for ID-card boundary detection.
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
        interpolation=cv2.INTER_CUBIC,
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
    Test models independently.
    A MODNet failure must not disable U2-Net or YuNet.
    """

    ok, _message = load()

    if not ok:
        return False

    img = np.random.RandomState(1).randint(
        0,
        255,
        (200, 160, 3),
    ).astype(np.uint8)

    global _modnet_session
    global _modnet_error
    global _bg_net
    global _bg_error
    global _face_det
    global _face_error

    if _modnet_session is not None:
        try:
            _modnet_mask(img)
        except Exception as exc:
            _modnet_session = None
            _modnet_error = "فشل الفحص الذاتي: %s" % exc

    if _bg_net is not None:
        try:
            _u2_mask(img)
        except Exception as exc:
            _bg_net = None
            _bg_error = "فشل الفحص الذاتي: %s" % exc

    if _face_det is not None:
        try:
            face_box(img)
        except Exception as exc:
            _face_det = None
            _face_error = "فشل الفحص الذاتي: %s" % exc

    return any((
        _modnet_session is not None,
        _bg_net is not None,
        _face_det is not None,
    ))
