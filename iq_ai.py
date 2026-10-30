# -*- coding: utf-8 -*-
"""
IQ AI — ميزة اختيارية: تعرّف أدق على الوجه (YuNet) + إزالة خلفية أنظف (U2-Net).
تعمل بالكامل على الجهاز عبر cv2.dnn (بدون إنترنت وبدون مكتبات إضافية).
لا تُحمَّل إلا إذا فعّل المستخدم الميزة، وأي فشل يرجع للطريقة العادية.
"""
import os
import sys

import cv2
import numpy as np

BG_MODELS = ("u2net_human_seg.onnx", "u2netp.onnx")      # الأول أدق للأشخاص، الثاني صغير وسريع
FACE_MODELS = ("face_detection_yunet_2023mar.onnx", "face_detection_yunet_2022mar.onnx")

_bg_net = None
_bg_name = None
_face_det = None
_face_name = None
_loaded = False


def models_dirs():
    base = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))
    dirs = [os.path.join(base, "models")]
    appdata = os.environ.get("APPDATA")
    if appdata:
        dirs.append(os.path.join(appdata, "AbaskyIQ", "models"))
    return dirs


def find_model(names):
    for d in models_dirs():
        for n in names:
            p = os.path.join(d, n)
            if os.path.isfile(p) and os.path.getsize(p) > 100000:
                return p
    return None


def _read_bytes_net(path):
    # قراءة الملف من بايثون ثم تحميله من الذاكرة (يتجنب مشاكل المسارات العربية بويندوز)
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8)
    return cv2.dnn.readNetFromONNX(data)


def load():
    """يحمّل النماذج المتوفرة. يرجع (ok, رسالة). ok=True إذا نموذج الخلفية على الأقل جاهز."""
    global _bg_net, _bg_name, _face_det, _face_name, _loaded
    if _loaded:
        return (_bg_net is not None), status_text()
    if not hasattr(cv2, "dnn"):
        return False, "نسخة OpenCV ما تدعم وحدة الشبكات العصبية."
    for name in BG_MODELS:
        p = find_model((name,))
        if p is None:
            continue
        try:
            _bg_net = _read_bytes_net(p)
            _bg_name = name
            break
        except Exception:  # noqa: BLE001
            _bg_net = None
    if hasattr(cv2, "FaceDetectorYN"):
        for name in FACE_MODELS:
            p = find_model((name,))
            if p is None:
                continue
            try:
                with open(p, "rb") as f:
                    buf = np.frombuffer(f.read(), np.uint8)
                _face_det = cv2.FaceDetectorYN.create("onnx", buf, np.empty(0, np.uint8), (320, 320), 0.6, 0.3, 1000)
                _face_name = name
                break
            except Exception:  # noqa: BLE001
                _face_det = None
    _loaded = True
    return (_bg_net is not None or _face_det is not None), status_text()


def status_text():
    parts = []
    parts.append("إزالة الخلفية: %s" % (_bg_name if _bg_net is not None else "غير متوفر"))
    parts.append("الوجه: %s" % (_face_name if _face_det is not None else "غير متوفر (يستعمل الطريقة العادية)"))
    return " | ".join(parts)


def missing_models_text():
    need = []
    if find_model(BG_MODELS) is None:
        need.append("u2netp.onnx")
    if find_model(FACE_MODELS) is None:
        need.append("face_detection_yunet_2023mar.onnx")
    return need


def selftest():
    """فحص ذاتي (يُشغَّل بعملية منفصلة حتى لو انهار المعالج ما يتأثر البرنامج الرئيسي)."""
    ok, _msg = load()
    if not ok:
        return False
    img = np.random.RandomState(1).randint(0, 255, (200, 160, 3)).astype(np.uint8)
    if _bg_net is not None:
        _u2_mask(img)
    if _face_det is not None:
        face_box(img)
    return True


# ----------------------------------------------------------------------------
def face_box(bgr):
    """(x, y, w, h) لأكبر وجه، أو None."""
    if _face_det is None:
        return None
    h, w = bgr.shape[:2]
    s = min(1.0, 800.0 / max(h, w))
    small = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1.0 else bgr
    sh, sw = small.shape[:2]
    _face_det.setInputSize((sw, sh))
    _ret, faces = _face_det.detect(small)
    if faces is None or len(faces) == 0:
        return None
    best = max(faces, key=lambda r: float(r[2]) * float(r[3]) * float(r[-1]))
    x, y, fw, fh = [float(v) / s for v in best[:4]]
    return (x, y, fw, fh)


def _u2_mask(bgr, size=320):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    x = x / max(float(x.max()), 1e-6)
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    _bg_net.setInput(np.ascontiguousarray(x.transpose(2, 0, 1)[None]))
    out = _bg_net.forward()
    m = out[0, 0].astype(np.float32)
    lo, hi = float(m.min()), float(m.max())
    return (m - lo) / (hi - lo + 1e-9)


def person_alpha(bgr, face=None):
    """قناع شفافية (0..1) بحجم الصورة للشخص، أو None إذا فشل/غير موثوق."""
    if _bg_net is None:
        return None
    h, w = bgr.shape[:2]
    try:
        m = _u2_mask(bgr)
    except Exception:  # noqa: BLE001
        return None
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_CUBIC)
    a = np.clip((m - 0.30) / 0.40, 0.0, 1.0)             # يشد الحواف قليلاً
    hard = (a > 0.5).astype(np.uint8)
    n, lab, stats, _c = cv2.connectedComponentsWithStats(hard, connectivity=8)
    if n <= 1:
        return None
    keep = None
    if face is not None:                                  # المكوّن الذي يحوي الوجه
        fx, fy, fw, fh = face
        cx, cy = int(min(max(fx + fw / 2, 0), w - 1)), int(min(max(fy + fh / 2, 0), h - 1))
        if lab[cy, cx] > 0:
            keep = int(lab[cy, cx])
    if keep is None:
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = (lab == keep).astype(np.uint8)
    comp = cv2.dilate(comp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    # ثقوب داخلية
    inv = (1 - comp).astype(np.uint8) * 255
    ff = inv.copy()
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 0)
    comp = np.maximum(comp, (ff > 0).astype(np.uint8))
    alpha = a * cv2.GaussianBlur(comp.astype(np.float32), (0, 0), 1.5)
    alpha = np.where(comp > 0, np.maximum(alpha, 0.0), 0.0)
    # داخل الجسم الصلب نخليه 1 (حتى ما تبان فجوات)
    core = cv2.erode(comp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    alpha = np.maximum(alpha, core.astype(np.float32))
    frac = float(alpha.mean())
    if frac < 0.06 or frac > 0.95:
        return None
    return alpha.astype(np.float32)


def salient_masks(bgr):
    """أقنعة ثنائية للجسم البارز (تساعد القص الذكي للهويات): عتبات متعددة بحجم الصورة."""
    if _bg_net is None:
        return []
    h, w = bgr.shape[:2]
    try:
        m = _u2_mask(bgr)
    except Exception:  # noqa: BLE001
        return []
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_CUBIC)
    return [((m > t).astype(np.uint8) * 255) for t in (0.25, 0.45, 0.65)]
