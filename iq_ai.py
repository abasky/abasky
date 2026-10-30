# -*- coding: utf-8 -*-
"""
IQ core — منظومة الهويات والمستمسكات
معالجة الصور، ضبط الألوان، التعرف الذكي، تخطيط الصفحات، كتابة PDF، حفظ المشاريع.
هذا الملف لا يعتمد على Qt إطلاقاً (لذلك يمكن اختباره بسهولة).
متوافق مع Python 3.8 (ويندوز 7 / 32-bit).
"""
import copy
import io
import itertools
import json
import math
import os
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

VERSION = "2.2"
MAX_WORK_SIDE = 4000      # أكبر ضلع للصورة الأصلية المحفوظة
RESULT_MAX_SIDE = 3000    # أكبر ضلع لنتيجة القص النهائية (يكفي لجواز بدقة 600 dpi)

_uid_counter = itertools.count(1)


# ----------------------------------------------------------------------------
# قراءة وكتابة الصور
# ----------------------------------------------------------------------------
def limit_size(img, max_side=MAX_WORK_SIDE):
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    s = float(max_side) / m
    return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)


def _pil_to_bgr_limited(im):
    im = ImageOps.exif_transpose(im).convert("RGB")
    arr = np.ascontiguousarray(np.asarray(im)[:, :, ::-1])
    return limit_size(arr)


def load_image(path):
    """يقرأ الصورة (يدعم المسارات العربية ودوران EXIF) ويرجعها BGR."""
    with Image.open(path) as im:
        return _pil_to_bgr_limited(im)


def decode_image(data):
    with Image.open(io.BytesIO(data)) as im:
        return _pil_to_bgr_limited(im)


def encode_jpeg(bgr, quality=96):
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError("JPEG encode failed")
    return buf.tobytes()


def bgr_to_pil(bgr):
    return Image.fromarray(np.ascontiguousarray(bgr[:, :, ::-1]))


def pil_to_bgr(im):
    return np.ascontiguousarray(np.asarray(im.convert("RGB"))[:, :, ::-1])


# ----------------------------------------------------------------------------
# ضبط الألوان والإضاءة
# ----------------------------------------------------------------------------
DEFAULT_ADJ = {
    "brightness": 0,    # -100..100
    "contrast": 0,      # -100..100
    "saturation": 0,    # -100..100
    "warmth": 0,        # -100..100  (سالب = أبرد/أزرق، موجب = أدفأ/أصفر)
    "sharpness": 0,     # 0..100
    "auto": False,      # تحسين تلقائي
    "flatten": False,   # معادلة الإضاءة (إزالة الظلال والتدرج)
    "bw": False,        # أبيض وأسود
    "whitebg": False,   # إزالة الخلفية ووضع أبيض (للصور الشخصية)
}


def adj_is_default(adj):
    a = dict(DEFAULT_ADJ)
    a.update(adj or {})
    return all(a[k] == DEFAULT_ADJ[k] for k in DEFAULT_ADJ)


def apply_adjust(bgr, adj):
    """يطبق ضبط الألوان. الترتيب: معادلة الإضاءة ← تحسين تلقائي ← سطوع/تباين ← تشبع ← دفء ← أبيض وأسود ← حدة."""
    a = dict(DEFAULT_ADJ)
    a.update(adj or {})
    if adj_is_default(a):
        return bgr
    img = bgr
    h, w = img.shape[:2]

    if a["flatten"]:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L = lab[..., 0].astype(np.float32)
        s = 128.0 / max(h, w) if max(h, w) > 128 else 1.0
        small = cv2.resize(L, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        sigma = max(3.0, 0.22 * max(small.shape[:2]))
        bg = cv2.GaussianBlur(small, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
        bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
        factor = np.clip(float(bg.mean()) / np.maximum(bg, 1.0), 0.75, 1.45)
        lab[..., 0] = np.clip(L * (1.0 + 0.85 * (factor - 1.0)), 0, 255).astype(np.uint8)
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if a["auto"]:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L = lab[..., 0]
        lo, hi = np.percentile(L, (0.7, 99.3))
        if hi - lo > 20:
            L = np.clip((L.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
        lab[..., 0] = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(6, 6)).apply(L)
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    f = img.astype(np.float32)

    b = float(a["brightness"])
    if b:
        f += b * 1.2

    c = float(a["contrast"])
    if c:
        factor = 1.0 + (c / 100.0 if c >= 0 else c / 125.0)
        pivot = 0.5 * 128.0 + 0.5 * float(f.mean())
        f = (f - pivot) * factor + pivot

    s_ = float(a["saturation"])
    if s_:
        gray = 0.114 * f[..., 0] + 0.587 * f[..., 1] + 0.299 * f[..., 2]
        f = gray[..., None] + (f - gray[..., None]) * (1.0 + s_ / 100.0)

    wm = float(a["warmth"])
    if wm:
        f[..., 2] += wm * 0.3
        f[..., 0] -= wm * 0.3

    if a["bw"]:
        gray = 0.114 * f[..., 0] + 0.587 * f[..., 1] + 0.299 * f[..., 2]
        f = np.repeat(gray[..., None], 3, axis=2)

    sh = float(a["sharpness"])
    if sh > 0:
        f = np.clip(f, 0, 255)
        sigma = max(0.8, max(h, w) / 1500.0)
        blur = cv2.GaussianBlur(f, (0, 0), sigma)
        f = f + (sh / 100.0 * 1.6) * (f - blur)

    return np.clip(f, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# حالة التعديل غير المدمرة (إطار القص + تدوير النتيجة + ضبط الألوان)
# ----------------------------------------------------------------------------
FULL_QUAD = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]


def default_state():
    return {"rot": 0, "quad": None, "trim": 0.0, "adj": dict(DEFAULT_ADJ)}


def normalize_state(state):
    s = default_state()
    if state:
        s["rot"] = int(state.get("rot", 0)) % 4
        q = state.get("quad")
        s["quad"] = None if q is None else [[float(x), float(y)] for x, y in q]
        s["trim"] = float(state.get("trim", 0.0))
        s["adj"].update(state.get("adj") or {})
    return s


def rotate_k(img, k):
    k = int(k) % 4
    if k == 0:
        return img
    code = (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE)[k - 1]
    return cv2.rotate(img, code)


def is_full_quad(qn, tol=0.003):
    if qn is None:
        return True
    return bool(np.allclose(np.asarray(qn, np.float32), np.asarray(FULL_QUAD, np.float32), atol=tol))


def render_side(orig_bgr, state, max_side=RESULT_MAX_SIDE):
    """
    يطبق القص (منظور) ثم التدوير ثم ضبط الألوان على الصورة الأصلية ويرجع BGR.
    التدوير rot (بخطوات 90° مع عقارب الساعة) يطبَّق على النتيجة: نقوم بتدوير ترتيب
    رؤوس الإطار فتنقلب البطاقة بدون أي فقدان بالجودة.
    """
    st = normalize_state(state)
    img = orig_bgr
    if st["adj"].get("whitebg"):
        img = remove_background_white(img)   # على الصورة الكاملة (أدق: الشعر والكتفين داخل الإطار)
    h, w = img.shape[:2]
    rot = st["rot"]
    qn = st["quad"]
    trim = float(st["trim"])
    if is_full_quad(qn) and trim <= 0:
        out = limit_size(rotate_k(img, rot), max_side)
    else:
        q = np.asarray(qn if qn is not None else FULL_QUAD, np.float32) * np.array([w - 1, h - 1], np.float32)
        if trim > 0:
            c = q.mean(axis=0)
            q = c + (q - c) * max(0.2, 1.0 - 2.0 * trim / 100.0)
        q = np.roll(q, rot, axis=0)
        out = warp_quad(img, q, max_side)
    return apply_adjust(out, st["adj"])


def auto_state(orig_bgr, seed=None):
    """تعرف ذكي: يرجع (state, score). إذا ما لقى هوية يرجع state بدون قص و score = 0."""
    h, w = orig_bgr.shape[:2]
    q, score = detect_card_quad_scored(orig_bgr, seed)
    st = default_state()
    if q is not None:
        qn = np.asarray(q, np.float32) / np.array([max(w - 1, 1), max(h - 1, 1)], np.float32)
        qn = np.clip(qn, 0.0, 1.0)
        st["quad"] = qn.tolist()
    return st, float(score)


# ----------------------------------------------------------------------------
# التعرف الذكي على الهوية (مختبَر على صور حقيقية: خلفية خشبية، ظلال، ميلان)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# الصور الشخصية: تعرّف على الوجه + قص تلقائي بنسبة المقاس + إزالة الخلفية (أبيض)
# ----------------------------------------------------------------------------
import base64
import gzip
import hashlib

_face_cascades = None
_ai = {"on": False, "mod": None}      # ميزة الذكاء الاصطناعي الاختيارية (iq_ai)


def set_ai(mod):
    """mod = وحدة iq_ai بعد نجاح الفحص، أو None لإطفاء الميزة."""
    _ai["mod"] = mod
    _ai["on"] = mod is not None
    _bg_cache.clear()


def ai_enabled():
    return bool(_ai["on"])


def _cascade_from_xml_bytes(data):
    """يحمّل المصنف من الذاكرة (يتجنب مشاكل المسارات العربية بويندوز)."""
    try:
        fs = cv2.FileStorage(data.decode("utf-8", "ignore"), cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
        c = cv2.CascadeClassifier()
        if c.read(fs.getFirstTopLevelNode()) and not c.empty():
            return c
    except Exception:  # noqa: BLE001
        pass
    return None


def _load_cascades():
    global _face_cascades
    if _face_cascades is not None:
        return _face_cascades
    out = []
    if hasattr(cv2, "CascadeClassifier"):
        try:
            d = cv2.data.haarcascades
            for n in ("haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml"):
                p = os.path.join(d, n)
                if not os.path.exists(p):
                    continue
                with open(p, "rb") as f:
                    c = _cascade_from_xml_bytes(f.read())
                if c is None:
                    c = cv2.CascadeClassifier(p)
                    c = c if not c.empty() else None
                if c is not None:
                    out.append(c)
        except Exception:  # noqa: BLE001
            pass
        try:
            c = _cascade_from_xml_bytes(gzip.decompress(base64.b64decode(_LBP_B64)))
            if c is not None:
                out.append(c)
        except Exception:  # noqa: BLE001
            pass
    _face_cascades = out
    return out


def find_face(bgr):
    """يرجع (x, y, w, h) لأكبر وجه بالصورة بإحداثياتها، أو None."""
    if _ai["on"]:
        try:
            f = _ai["mod"].face_box(bgr)
            if f is not None:
                return f
        except Exception:  # noqa: BLE001
            pass
    cascades = _load_cascades()
    if not cascades:
        return None
    h, w = bgr.shape[:2]
    s = min(1.0, 640.0 / max(h, w))
    small = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1.0 else bgr
    gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    min_side = int(max(24, 0.10 * min(gray.shape[:2])))
    for neighbors in (5, 3):
        for c in cascades:
            try:
                faces = c.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=neighbors,
                                           minSize=(min_side, min_side))
            except Exception:  # noqa: BLE001
                continue
            if len(faces):
                x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])
                return (float(x) / s, float(y) / s, float(fw) / s, float(fh) / s)
    return None


def _rect_quad_norm(x0, y0, x1, y1, w, h):
    q = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32) / np.array([max(w - 1, 1), max(h - 1, 1)], np.float32)
    return np.clip(q, 0.0, 1.0).tolist()


def fit_rect(cx, cy, rw, rh, w, h):
    """يقصّ مستطيلاً (مركز + أبعاد) ليدخل بالصورة مع الحفاظ على النسبة ثم يزيحه للداخل."""
    k = min(1.0, float(w) / rw, float(h) / rh)
    rw, rh = rw * k, rh * k
    x0 = min(max(cx - rw / 2.0, 0.0), w - rw)
    y0 = min(max(cy - rh / 2.0, 0.0), h - rh)
    return x0, y0, x0 + rw, y0 + rh


def default_photo_quad(w, h, aspect):
    """إطار افتراضي بنسبة المقاس بوسط الصورة (مائل قليلاً للأعلى حيث يكون الوجه عادة)."""
    rh = h * 0.92
    rw = rh * aspect
    if rw > w * 0.92:
        rw = w * 0.92
        rh = rw / aspect
    return _rect_quad_norm(*fit_rect(w / 2.0, h * 0.46, rw, rh, w, h), w, h)


def photo_state(bgr, aspect, face=None):
    """قص تلقائي للصورة الشخصية: يرجع (state, found). aspect = العرض/الارتفاع للمقاس."""
    h, w = bgr.shape[:2]
    st = default_state()
    if face is None:
        face = find_face(bgr)
    if face is None:
        st["quad"] = default_photo_quad(w, h, aspect)
        return st, False
    fx, fy, fw, fh = face
    head_h = 1.30 * fh                      # من قمة الرأس إلى الذقن تقريباً
    rh = head_h / 0.70                      # الرأس ~70% من ارتفاع الصورة
    rw = rh * aspect
    crown = fy - 0.28 * fh
    top = crown - 0.10 * rh
    x0, y0, x1, y1 = fit_rect(fx + fw / 2.0, top + rh / 2.0, rw, rh, w, h)
    st["quad"] = _rect_quad_norm(x0, y0, x1, y1, w, h)
    return st, True


_bg_cache = {}


def _bg_key(bgr):
    t = cv2.resize(bgr, (48, 48), interpolation=cv2.INTER_AREA)
    return (bgr.shape[:2], bool(_ai["on"]), hashlib.md5(t.tobytes()).hexdigest())


def remove_background_white(bgr):
    """يفصل الشخص عن الخلفية (GrabCut موجّه بموضع الوجه) ويضع خلفية بيضاء. إذا فشل يرجع الصورة كما هي."""
    key = _bg_key(bgr)
    hit = _bg_cache.get(key)
    if hit is not None:
        return hit
    out = _remove_bg(bgr)
    if len(_bg_cache) > 6:
        _bg_cache.pop(next(iter(_bg_cache)))
    _bg_cache[key] = out
    return out


def _remove_bg_ai(bgr):
    """نسخة الذكاء الاصطناعي (U2-Net). ترجع None إذا ما نجحت فنرجع لـ GrabCut."""
    try:
        face = find_face(bgr)
        alpha = _ai["mod"].person_alpha(bgr, face)
    except Exception:  # noqa: BLE001
        return None
    if alpha is None:
        return None
    a = alpha[..., None]
    return np.clip(bgr.astype(np.float32) * a + 255.0 * (1.0 - a), 0, 255).astype(np.uint8)


def _remove_bg(bgr):
    if _ai["on"]:
        out = _remove_bg_ai(bgr)
        if out is not None:
            return out
    h, w = bgr.shape[:2]
    s = min(1.0, 560.0 / max(h, w))
    sm = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1.0 else bgr.copy()
    sh, sw = sm.shape[:2]
    face = find_face(sm)
    BGD, FGD, PBGD, PFGD = cv2.GC_BGD, cv2.GC_FGD, cv2.GC_PR_BGD, cv2.GC_PR_FGD
    mask = np.full((sh, sw), PBGD, np.uint8)
    b = max(2, int(0.025 * min(sh, sw)))
    mask[:b, :] = BGD
    mask[:, :b] = BGD
    mask[:, sw - b:] = BGD
    if face is not None:
        fx, fy, fw, fh = face
        cx = fx + fw / 2.0
    else:                                     # لا وجه: نفترض شخصاً بالوسط
        fw, fh = 0.34 * sw, 0.34 * sw
        cx, fx, fy = sw / 2.0, sw / 2.0 - 0.17 * sw, 0.16 * sh
    cv2.ellipse(mask, (int(cx), int(fy + fh * 0.30)), (int(fw * 0.98), int(fh * 1.30)), 0, 0, 360, PFGD, -1)
    body = np.array([[cx - 0.55 * fw, fy + 0.95 * fh], [cx + 0.55 * fw, fy + 0.95 * fh],
                     [cx + 1.35 * fw, fy + 1.45 * fh], [cx + 2.0 * fw, sh - 1],
                     [cx - 2.0 * fw, sh - 1], [cx - 1.35 * fw, fy + 1.45 * fh]], np.int32)
    cv2.fillPoly(mask, [body], PFGD)
    cv2.ellipse(mask, (int(cx), int(fy + fh * 0.38)), (max(2, int(fw * 0.52)), max(2, int(fh * 0.62))), 0, 0, 360, FGD, -1)
    inner = np.array([[cx - 0.6 * fw, fy + 1.7 * fh], [cx + 0.6 * fw, fy + 1.7 * fh],
                      [cx + 0.8 * fw, sh - 1], [cx - 0.8 * fw, sh - 1]], np.int32)
    cv2.fillPoly(mask, [inner], FGD)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(sm, mask, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return bgr
    fg = np.where((mask == FGD) | (mask == PFGD), 255, 0).astype(np.uint8)
    n, lab, stats, _c = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n <= 1:
        return bgr
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    fg = np.where(lab == keep, 255, 0).astype(np.uint8)
    k = max(3, int(0.012 * max(sh, sw)) | 1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    inv = 255 - fg                              # سدّ الثقوب الداخلية
    ff = inv.copy()
    seed_pt = None
    cv2.floodFill(ff, np.zeros((sh + 2, sw + 2), np.uint8), (0, 0), 0)
    fg = cv2.bitwise_or(fg, ff)
    frac = float(np.count_nonzero(fg)) / fg.size
    if frac < 0.08 or frac > 0.93:
        return bgr                              # الفصل غير موثوق — لا نخرّب الصورة
    fg = cv2.erode(fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    alpha = cv2.GaussianBlur(fg.astype(np.float32) / 255.0, (0, 0), 1.6)
    alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)[..., None]
    out = bgr.astype(np.float32) * alpha + 255.0 * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)

# مصنف الوجه الاحتياطي (LBP) من OpenCV — رخصة BSD — مضغوط ومشفّر
_LBP_B64 = (
    "H4sIACeWxmoC/8Wd3a8c13Ld3w34f5jcd4r7+wOgZfg6SPIQ2EZy4TwKx9LRFRFJFEjqXue/z/qtPfyaoYPT3UxTEKTDw+mZnt17"
    "V62qWrXqxd//+y8/X/7y+PrNy1e//t0f4jfhD3//7d/+zYv/9OzZ3/7Nr7//8m+Pry+vfrz89urNy7cv//J4efPwy28/P7655BDC"
    "xy/49fHPD5+8IFZe8OwZb/bqt8dfv//Ld2/evnr98OdHfvP9w5vvH354vLz9P789fvfyh7/7w3rJs+vvn33/88ObNy9/fPn4+g96"
    "+eXy4s1bXfknvfrbP/7zP//PP714/uEX/vsfHx/e/v56/eK///FfXjz/+Bd+xU+PL//809tvU3nx/Pqjf/3Xlz+8/cm/XT99+LR/"
    "eXj98Msb/0K/+rdXr9689bv913/444vnH/54/ftfXv76316+/R8Pbx+/Dd/MqW+vf0pvI/fYXjz/6K/fXfDw7//l4ec3j//w88Pr"
    "X3TNuuLDP7rmk1dcL/ur7/1Pr1/+8u6z6u11Ny/58Hn/+fE3fcPod14/f/i7//X48L//8dXvv779Nl4/+8Nvrqv90YKs9f7p9c0i"
    "6ap/fHi7Lkq1+W3e/+L6TD69aC31P/3+y7cpXD+GP3z4q/fvrR158W8u4eJt5d9+d/3h9kvkm6/w/lVr3+gm3vz06ucfvn0Wvuk1"
    "hTFTSCWUkFJ6t7fev+b9tX/l7d7vzDfv/+J6d29fP35yczc36D++/PXt4+tfH37+p1c/PL755K8uXBovpV2etR5z6GFenqVYm/4X"
    "S2oxhRCzfk69xqR7rv7rUT99k2ex5VEvdfRY6pi8Rwmhvnj+H3/yi58fH37814eff7+7I61OqyWnGGIZI8ZcwiV8M9Y/s9YYW+pa"
    "sI/e4MXz7z67LvHguvirt6yv3puWqLdZZ+516LchpB68Wi3pS1+e1VhK7aOW26WZWRflqreqqaTByu5emN7zTLGxdcpotWtheuqj"
    "ZR343LWP0nzSwqSDC6Prc8sztz715zFzCrl50+hOZhkh8rLcZuhj3qxIDGP0kXpuvKSk2HticUuspcymZxbrDHqf/asU2qgtR92T"
    "HlZJWqXWW6x8WAlx9l7vVgkL9ulB+7B2H5mB+AQzUJ5uBsroSUuUZLsTR+brmoFRLlqgXkZupV/iLDr/2rFdOze3Idug8+AX1DD6"
    "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"
)
def order_points(pts):
    """ترتيب: أعلى-يسار، أعلى-يمين، أسفل-يمين، أسفل-يسار"""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def make_landscape(q):
    tl, tr, br, bl = q
    if np.linalg.norm(tr - tl) < np.linalg.norm(bl - tl):
        q = np.array([tr, br, bl, tl], dtype=np.float32)
    return q


def _side_contrast(lab_blur, quad, offs=(3, 5, 7)):
    """
    تباين الحدود: لكل ضلع يقارن لون داخل الضلع بلون خارجه.
    الهوية الحقيقية لها تباين واضح على الأضلاع الأربعة، أما الخلفية المزخرفة
    أو إطار الصورة الكامل فتباينها ضعيف. يرجع قيمة تجمع أضعف ضلعين.
    """
    h, w = lab_blur.shape[:2]
    cen = quad.mean(axis=0)
    vals = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        L = float(np.linalg.norm(b - a))
        if L < 10:
            return 0.0
        d = (b - a) / L
        nrm = np.array([-d[1], d[0]], np.float32)
        if float(np.dot(nrm, cen - (a + b) / 2)) < 0:
            nrm = -nrm  # يشير للداخل
        t = np.linspace(0.12, 0.88, max(int(L / 5), 8))[:, None]
        p = a + (b - a) * t

        def sample(sign):
            acc = 0
            for o in offs:
                q = p + nrm * (sign * o)
                xi = np.clip(np.round(q[:, 0]).astype(int), 0, w - 1)
                yi = np.clip(np.round(q[:, 1]).astype(int), 0, h - 1)
                acc = acc + lab_blur[yi, xi].astype(np.float32)
            return acc / len(offs)

        de = np.linalg.norm(sample(+1) - sample(-1), axis=1)
        vals.append(float(np.median(de)))
    vals.sort()
    return 0.6 * vals[0] + 0.4 * vals[1]


def _refine_quad_by_lines(contour, quad):
    """يدقق الزوايا: يفرد خط لكل ضلع من نقاط الحدود ثم يقاطع الخطوط المتجاورة."""
    pts = contour.reshape(-1, 2).astype(np.float32)
    lines = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        L = float(np.linalg.norm(b - a))
        if L < 10:
            return quad
        d = (b - a) / L
        nrm = np.array([-d[1], d[0]], np.float32)
        rel = pts - a
        t = rel @ d / L
        dist = np.abs(rel @ nrm)
        sel = pts[(t > 0.12) & (t < 0.88) & (dist < 0.04 * L)]
        if len(sel) < 8:
            return quad
        vx, vy, x0, y0 = cv2.fitLine(sel, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0], np.float32), np.array([vx, vy], np.float32)))
    tol = 0.06 * max(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[3] - quad[0]))
    out = []
    for i in range(4):
        p1, d1 = lines[i - 1]
        p2, d2 = lines[i]
        den = float(d1[0] * d2[1] - d1[1] * d2[0])
        if abs(den) < 1e-6:
            return quad
        s = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / den
        c = p1 + d1 * s
        if np.linalg.norm(c - quad[i]) > tol:
            return quad
        out.append(c)
    return np.array(out, np.float32)


def _snap_quad_to_edges(img, quad):
    """
    تثبيت الأضلاع على حواف الهوية بالدقة الكاملة: لكل ضلع يبحث عن أقوى حافة
    عمودية عليه ضمن نطاق صغير ثم يفرد خطاً قوياً (يتجاهل النقاط الشاذة).
    إذا الضلع ما اعتمد عليه بثقة يبقي الضلع الأصلي.
    """
    h, w = img.shape[:2]
    lab = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2LAB), (0, 0), 1.0).astype(np.float32)
    cen = quad.mean(axis=0)
    lines, radii = [], []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        L = float(np.linalg.norm(b - a))
        d = (b - a) / L
        nrm = np.array([-d[1], d[0]], np.float32)
        if float(np.dot(nrm, cen - (a + b) / 2)) < 0:
            nrm = -nrm
        r = int(max(6, min(0.03 * L, 40)))
        radii.append(r)
        offs = np.arange(-r, r + 1, dtype=np.float32)
        ts = np.linspace(0.1, 0.9, max(int(L / 12), 12))
        snapped = []
        for t in ts:
            p = a + (b - a) * t
            q = p[None, :] + offs[:, None] * nrm[None, :]
            xi = np.clip(np.round(q[:, 0]).astype(int), 0, w - 1)
            yi = np.clip(np.round(q[:, 1]).astype(int), 0, h - 1)
            prof = lab[yi, xi]
            g = np.linalg.norm(prof[2:] - prof[:-2], axis=1)
            k = int(np.argmax(g))
            if g[k] < 8:
                continue
            snapped.append(p + offs[1:-1][k] * nrm)
        line = None
        if len(snapped) >= max(6, 0.4 * len(ts)):
            pts = np.array(snapped, np.float32)
            vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            dd = np.array([vx, vy], np.float32)
            pp = np.array([x0, y0], np.float32)
            res = np.abs((pts - pp) @ np.array([-dd[1], dd[0]], np.float32))
            keep = res < 2.5
            if keep.sum() >= max(6, 0.5 * len(pts)):
                vx, vy, x0, y0 = cv2.fitLine(pts[keep], cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
                line = (np.array([x0, y0], np.float32), np.array([vx, vy], np.float32))
        if line is None:
            line = (a.astype(np.float32), d.astype(np.float32))
        lines.append(line)
    out = []
    for i in range(4):
        p1, d1 = lines[i - 1]
        p2, d2 = lines[i]
        den = float(d1[0] * d2[1] - d1[1] * d2[0])
        c = None
        if abs(den) > 1e-6:
            s = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / den
            c = p1 + d1 * s
        if c is None or np.linalg.norm(c - quad[i]) > 1.8 * max(radii[i], radii[i - 1]):
            c = quad[i]
        out.append(c)
    return np.array(out, np.float32)


def _grabcut_masks(small, seed=None):
    """
    فصل الهوية عن الخلفية بـ GrabCut (يتعامل مع الخلفيات المزخرفة مثل الخشب).
    دائماً: تهيئات مركزية. وإذا المستخدم ضغط نقطة (seed): تهيئات إضافية تبدأ منها.
    """
    h, w = small.shape[:2]
    s = 320.0 / max(h, w)
    g = cv2.resize(small, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else small.copy()
    gh, gw = g.shape[:2]
    m_ = min(gh, gw)
    b = max(3, int(0.025 * m_))
    inits = []
    for (x0, y0, x1, y1) in ((0.2, 0.2, 0.8, 0.8), (0.1, 0.1, 0.9, 0.9), (0.3, 0.3, 0.7, 0.7)):
        mk = np.full((gh, gw), cv2.GC_PR_BGD, np.uint8)
        mk[int(y0 * gh):int(y1 * gh), int(x0 * gw):int(x1 * gw)] = cv2.GC_PR_FGD
        inits.append(mk)
    if seed is not None:
        sx, sy = int(seed[0] * s), int(seed[1] * s)
        for r2f in (0.14, 0.26, 0.40):
            mk = np.full((gh, gw), cv2.GC_PR_BGD, np.uint8)
            cv2.circle(mk, (sx, sy), max(4, int(r2f * m_)), int(cv2.GC_PR_FGD), -1)
            cv2.circle(mk, (sx, sy), max(3, int(0.03 * m_)), int(cv2.GC_FGD), -1)
            inits.append(mk)
        mk = np.full((gh, gw), cv2.GC_PR_BGD, np.uint8)
        x0, x1 = max(0, int(sx - 0.30 * gw)), min(gw, int(sx + 0.30 * gw))
        y0, y1 = max(0, int(sy - 0.20 * gh)), min(gh, int(sy + 0.20 * gh))
        mk[y0:y1, x0:x1] = cv2.GC_PR_FGD
        cv2.circle(mk, (sx, sy), max(3, int(0.03 * m_)), int(cv2.GC_FGD), -1)
        inits.append(mk)
    out = []
    for mk in inits:
        mk[:b] = cv2.GC_BGD
        mk[-b:] = cv2.GC_BGD
        mk[:, :b] = cv2.GC_BGD
        mk[:, -b:] = cv2.GC_BGD
        try:
            bgd = np.zeros((1, 65), np.float64)
            fgd = np.zeros((1, 65), np.float64)
            cv2.grabCut(g, mk, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            continue
        fg = ((mk == cv2.GC_FGD) | (mk == cv2.GC_PR_FGD)).astype(np.uint8) * 255
        fg = cv2.resize(fg, (w, h), interpolation=cv2.INTER_NEAREST)
        out.append(fg)
    return out


def detect_card_quad_scored(img, seed=None):
    """
    تعرف ذكي على الهوية (يعمل حتى لو الخلفية مزخرفة والهوية صغيرة):
    1) يولّد مرشحات كثيرة من عتبات متعددة على قنوات (رمادي/تشبع/Lab) مع كل الحدود الداخلية
    2) يقيّم كل مرشح بالشكل (مستطيل ونسبة ~1.586) وبتباين حدوده مع ما حوله
    3) يدقق الزوايا بفرد خط لكل ضلع
    يرجع 4 نقاط أو None.
    """
    h, w = img.shape[:2]
    scale = 800.0 / max(h, w) if max(h, w) > 800 else 1.0
    small = cv2.resize(img, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA) if scale != 1.0 else img.copy()
    sh, sw = small.shape[:2]
    img_area = float(sh * sw)
    min_area = 0.03 * img_area

    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    chans = [cv2.GaussianBlur(c, (5, 5), 0) for c in (gray, hsv[..., 1], lab[..., 1], lab[..., 2])]
    k9 = np.ones((9, 9), np.uint8)
    k5 = np.ones((5, 5), np.uint8)

    masks = []
    for ch in chans:
        levels = {int(cv2.threshold(ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])}
        levels.update(int(v) for v in np.percentile(ch, (20, 40, 60, 80)))
        for t in levels:
            _, m = cv2.threshold(ch, t, 255, cv2.THRESH_BINARY)
            masks.append(m)
            masks.append(255 - m)

    edges = cv2.Canny(chans[0], 30, 100)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    masks.append(edges)

    border = np.concatenate([small[0:6].reshape(-1, 3), small[-6:].reshape(-1, 3),
                             small[:, 0:6].reshape(-1, 3), small[:, -6:].reshape(-1, 3)])
    bg = np.median(border, axis=0).astype(np.float32)
    diff = np.linalg.norm(small.astype(np.float32) - bg, axis=2)
    d8 = cv2.GaussianBlur(np.clip(diff, 0, 255).astype(np.uint8), (5, 5), 0)
    masks.append(cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])

    seed_s = None if seed is None else (float(seed[0]) * scale, float(seed[1]) * scale)
    try:
        gc_masks = _grabcut_masks(small, seed_s)
    except Exception:  # noqa: BLE001
        gc_masks = []
    masks.extend(gc_masks)
    if _ai["on"]:                      # أقنعة الجسم البارز من الذكاء الاصطناعي كمرشحات إضافية
        try:
            masks.extend(_ai["mod"].salient_masks(small))
        except Exception:  # noqa: BLE001
            pass

    cands = {}
    for m in masks:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k9)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k5)
        cnts, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for c in cnts:
            if len(c) < 40:
                continue
            area = cv2.contourArea(c)
            if area < min_area or area > 0.95 * img_area:
                continue
            hull = cv2.convexHull(c)
            peri = cv2.arcLength(hull, True)
            quad = None
            for eps in (0.015, 0.025, 0.04, 0.06):
                ap = cv2.approxPolyDP(hull, eps * peri, True)
                if len(ap) == 4:
                    quad = ap.reshape(4, 2).astype(np.float32)
                    break
            penalty = 1.0
            if quad is None:
                quad = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
                penalty = 0.8
            q = order_points(quad)
            qa = cv2.contourArea(q)
            if qa < 1:
                continue
            if seed_s is not None and cv2.pointPolygonTest(q.reshape(-1, 1, 2), seed_s, False) < 0:
                continue
            ha = cv2.contourArea(hull)
            rect = min(ha, qa) / max(ha, qa)
            if rect < 0.88:
                continue
            wd = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
            ht = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
            aspect = max(wd, ht) / max(1.0, min(wd, ht))
            aspect_score = math.exp(-((aspect - 1.586) / 0.35) ** 2)
            if aspect_score < 0.15:
                continue
            geo = math.sqrt(qa / img_area) * rect * aspect_score * penalty
            cx, cy = q.mean(axis=0)
            key = (round(cx / (0.02 * sw)), round(cy / (0.02 * sh)), round(math.sqrt(qa) / (0.02 * sw)))
            if key not in cands or geo > cands[key][0]:
                cands[key] = (geo, q, c)

    if not cands:
        return None, 0.0

    lab_blur = cv2.GaussianBlur(lab, (0, 0), 1.2)
    best, best_score = None, 0.0
    for geo, q, c in sorted(cands.values(), key=lambda t: -t[0])[:300]:
        contrast = _side_contrast(lab_blur, q)
        cs = min(1.0, max(0.0, (contrast - 10.0) / 30.0))
        score = geo * cs
        if score > best_score:
            best_score, best = score, (q, c)

    if best is None or best_score < 0.02:
        return None, 0.0
    q, c = best
    q = _refine_quad_by_lines(c, q)
    q = q / scale
    q = _snap_quad_to_edges(img, q)
    q[:, 0] = np.clip(q[:, 0], 0, w - 1)
    q[:, 1] = np.clip(q[:, 1], 0, h - 1)
    return make_landscape(order_points(q)), best_score


def detect_card_quad(img, seed=None):
    return detect_card_quad_scored(img, seed)[0]


def warp_quad(img, quad, max_side=3200):
    quad = np.asarray(quad, dtype=np.float32)
    tl, tr, br, bl = quad
    W = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    H = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    W, H = max(W, 20), max(H, 20)
    s = min(1.0, max_side / max(W, H))
    W, H = max(int(W * s), 20), max(int(H * s), 20)
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)




# ----------------------------------------------------------------------------
# نموذج البيانات: وجه/ظهر + بطاقة
# ----------------------------------------------------------------------------
class Side(object):
    """صورة واحدة (وجه أو ظهر): الأصل محفوظ مضغوطاً + حالة التعديل + النتيجة الجاهزة للطباعة."""

    def __init__(self):
        self.orig_bytes = None
        self.state = None
        self.result = None   # PIL.Image RGB
        self.uid = 0

    def has_image(self):
        return self.result is not None

    def orig(self):
        return decode_image(self.orig_bytes)

    def set_image(self, orig_bgr, state):
        self.orig_bytes = encode_jpeg(orig_bgr)
        self.state = normalize_state(state)
        self.rerender()   # من البيانات المخزنة، حتى تكون النتيجة نفسها بعد إعادة فتح المشروع

    def rerender(self, orig=None):
        if self.orig_bytes is None:
            return
        if orig is None:
            orig = self.orig()
        self.result = bgr_to_pil(render_side(orig, self.state))
        self.uid = next(_uid_counter)

    def rotate_quick(self, delta):
        """تدوير سريع 90° مع الحفاظ على إطار القص."""
        if not self.has_image():
            return
        st = normalize_state(self.state)
        st["rot"] = (st["rot"] + int(delta)) % 4
        self.state = st
        self.rerender()

    def clear(self):
        self.orig_bytes = None
        self.state = None
        self.result = None
        self.uid = 0


def is_photo_name(kind):
    k = str(kind or "")
    return k.startswith("صورة شخصية") or k.startswith("صورة جواز")


class Card(object):
    def __init__(self, kind, w, h, copies=1, single=None):
        self.kind = kind
        self.single = is_photo_name(kind) if single is None else bool(single)   # صورة شخصية: وجه فقط
        self.w = float(w)   # سم
        self.h = float(h)   # سم
        self.copies = int(copies)
        self.face = Side()
        self.back = Side()

    def side(self, name):
        return self.face if name == "face" else self.back

    def has_any(self):
        return self.face.has_image() or (not self.single and self.back.has_image())

    def swap_sides(self):
        if not self.single:
            self.face, self.back = self.back, self.face


# ----------------------------------------------------------------------------
# قوالب الأحجام الجاهزة (الاسم, العرض سم, الارتفاع سم)
# ----------------------------------------------------------------------------
PRESETS = [
    ("بطاقة وطنية أو سكن", 8.6, 5.6),
    ("بطاقة بنكية / CR80", 8.56, 5.4),
    ("رخصة سوق", 8.6, 5.4),
    ("صفحة جواز", 12.5, 8.8),
    ("صورة شخصية 4×6", 4.0, 6.0),
    ("صورة جواز 3.5×4.5", 3.5, 4.5),
]

PAPERS = [("A4", 210.0, 297.0), ("A5", 148.0, 210.0), ("Letter", 215.9, 279.4), ("10x15", 100.0, 150.0)]
FIT_MODES = ("stretch", "cover", "contain")
BORDER_MODES = ("none", "line", "marks")

DEFAULT_LAYOUT = {
    "paper": "A4",
    "margin": 10.0,      # مم
    "gap_v": 4.0,        # مم (أعلى وأسفل)
    "gap_h": 4.0,        # مم (الجوانب)
    "pair_rows": True,   # كل بطاقة (وجه+ظهر) تبدأ بسطر جديد
    "rtl": False,        # ترتيب من اليمين لليسار
    "corner": 3.0,       # نصف قطر الزوايا المدورة (مم)
    "border": "none",    # none | line | marks
    "fit": "stretch",    # stretch | cover | contain
    "photo_per_row": 6,  # أقصى عدد صور شخصية بالصف (0 = بدون حد، فقط ما يتسع)
}


def paper_size(name):
    for n, w, h in PAPERS:
        if n == name:
            return w, h
    return PAPERS[0][1], PAPERS[0][2]


def merged_layout(opts):
    o = dict(DEFAULT_LAYOUT)
    if opts:
        o.update(opts)
    return o


# ----------------------------------------------------------------------------
# محرك التخطيط: توزيع الصور على الصفحات بالرفوف
# ----------------------------------------------------------------------------
class Item(object):
    __slots__ = ("w", "h", "img", "uid", "newrow", "card", "side", "single")


def build_items(cards, opts, include_empty=False):
    items = []
    prev_single = False
    for ci, card in enumerate(cards):
        sides = [("face", card.face)] if card.single else [("face", card.face), ("back", card.back)]
        filled = [(n, s) for n, s in sides if s.has_image()]
        if filled:
            use = filled
        elif include_empty:
            use = sides
        else:
            use = []
        for _ in range(max(1, int(card.copies))):
            for k, (n, s) in enumerate(use):
                it = Item()
                it.w = card.w * 10.0
                it.h = card.h * 10.0
                it.img = s.result
                it.uid = s.uid
                if card.single:
                    # الصور الشخصية تُرتَّب أفقياً بالصف؛ أول صورة بعد بطاقات أخرى تبدأ سطراً جديداً
                    it.newrow = bool(items) and not prev_single
                else:
                    it.newrow = bool(opts["pair_rows"] and k == 0)
                it.card = ci
                it.side = n
                it.single = bool(card.single)
                items.append(it)
                prev_single = bool(card.single)
    return items


def layout_pages(items, paper_w, paper_h, opts):
    """يرجع قائمة صفحات؛ كل صفحة قائمة (x, y, w, h, item) بالمليمتر."""
    m = float(opts["margin"])
    gh = float(opts["gap_h"])
    gv = float(opts["gap_v"])
    uw = max(10.0, paper_w - 2 * m)
    uh = max(10.0, paper_h - 2 * m)

    pages = []
    cur_rows = []
    state = {"used": 0.0}
    row = []
    row_w = 0.0
    row_h = 0.0

    def commit():
        nonlocal row, row_w, row_h, cur_rows
        if not row:
            return
        need = row_h + (gv if cur_rows else 0.0)
        if cur_rows and state["used"] + need > uh + 1e-6:
            pages.append(cur_rows)
            cur_rows = []
            state["used"] = 0.0
            need = row_h
        cur_rows.append((row, row_w, row_h))
        state["used"] += need
        row = []
        row_w = 0.0
        row_h = 0.0

    cap = int(opts.get("photo_per_row", 0) or 0)
    for it in items:
        w = min(it.w, uw)
        h = min(it.h, uh)
        if row and (it.newrow or row_w + gh + w > uw + 1e-6
                    or (cap and it.single and len(row) >= cap)):
            commit()
        row_w += (gh if row else 0.0) + w
        row.append((it, w, h))
        row_h = max(row_h, h)
    commit()
    if cur_rows:
        pages.append(cur_rows)
    if not pages:
        pages = [[]]

    out = []
    for rows in pages:
        placements = []
        y = m
        for ritems, rw, rh in rows:
            x = m + (uw - rw) / 2.0
            seq = list(reversed(ritems)) if opts["rtl"] else ritems
            for it, w, h in seq:
                placements.append((x, y, w, h, it))
                x += w + gh
            y += rh + gv
        out.append(placements)
    return out


# ----------------------------------------------------------------------------
# رسم الصفحات
# ----------------------------------------------------------------------------
_resize_cache = {}
_mask_cache = {}


def _fit_to_box(img, bw, bh, fit, preview, uid):
    key = (uid, bw, bh, fit) if (preview and uid) else None
    if key is not None and key in _resize_cache:
        return _resize_cache[key]
    method = Image.BILINEAR if preview else Image.LANCZOS
    if fit == "cover":
        r = ImageOps.fit(img, (bw, bh), method=method, centering=(0.5, 0.5))
    elif fit == "contain":
        r0 = ImageOps.contain(img, (bw, bh), method=method)
        r = Image.new("RGB", (bw, bh), "white")
        r.paste(r0, ((bw - r0.width) // 2, (bh - r0.height) // 2))
    elif preview:
        r = img.resize((bw, bh), method, reducing_gap=2.0)
    else:
        r = img.resize((bw, bh), method)
    if key is not None:
        if len(_resize_cache) > 120:
            _resize_cache.clear()
        _resize_cache[key] = r
    return r


def _rounded_mask(w, h, radius, ss):
    key = (w, h, radius, ss)
    m = _mask_cache.get(key)
    if m is not None:
        return m
    radius = max(0, min(radius, min(w, h) // 2))
    if ss > 1:
        big = Image.new("L", (w * ss, h * ss), 0)
        ImageDraw.Draw(big).rounded_rectangle([0, 0, w * ss - 1, h * ss - 1], radius=radius * ss, fill=255)
        m = big.resize((w, h), Image.LANCZOS)
    else:
        m = Image.new("L", (w, h), 0)
        ImageDraw.Draw(m).rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, fill=255)
    if len(_mask_cache) > 40:
        _mask_cache.clear()
    _mask_cache[key] = m
    return m


def render_page(placements, paper_w, paper_h, dpi, opts, preview=False):
    def px(mm):
        return int(round(mm / 25.4 * dpi))

    W, H = px(paper_w), px(paper_h)
    page = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(page)
    corner_px = px(opts["corner"]) if opts["corner"] > 0 else 0
    ss = 2 if dpi < 200 else 1

    for x, y, w, h, it in placements:
        x0, y0 = px(x), px(y)
        bw, bh = max(px(w), 1), max(px(h), 1)
        if it.img is None:
            if preview:
                draw.rectangle([x0, y0, x0 + bw, y0 + bh], outline=(185, 185, 185), width=1)
            continue
        r = _fit_to_box(it.img, bw, bh, opts["fit"], preview, it.uid)
        if corner_px > 0:
            page.paste(r, (x0, y0), _rounded_mask(bw, bh, corner_px, ss))
        else:
            page.paste(r, (x0, y0))
        if opts["border"] == "line":
            draw.rounded_rectangle([x0, y0, x0 + bw - 1, y0 + bh - 1], radius=corner_px,
                                   outline=(150, 150, 150), width=max(1, px(0.12)))
        elif opts["border"] == "marks":
            ln, off, wd = px(2.5), px(1.0), max(1, px(0.15))
            x1, y1 = x0 + bw - 1, y0 + bh - 1
            for cx, cy, sx, sy in ((x0, y0, -1, -1), (x1, y0, 1, -1), (x0, y1, -1, 1), (x1, y1, 1, 1)):
                draw.line([(cx + sx * off, cy), (cx + sx * (off + ln), cy)], fill=(0, 0, 0), width=wd)
                draw.line([(cx, cy + sy * off), (cx, cy + sy * (off + ln))], fill=(0, 0, 0), width=wd)

    if preview:
        draw.rectangle([0, 0, W - 1, H - 1], outline=(40, 40, 40), width=2)
    return page


def compose(cards, opts, dpi, preview=False, progress=None):
    """يرجع قائمة صفحات PIL جاهزة للمعاينة أو التصدير."""
    o = merged_layout(opts)
    pw, ph = paper_size(o["paper"])
    items = build_items(cards, o, include_empty=preview)
    pages = layout_pages(items, pw, ph, o)
    out = []
    for i, placements in enumerate(pages):
        if progress:
            progress(i, len(pages))
        out.append(render_page(placements, pw, ph, dpi, o, preview))
    return out


# ----------------------------------------------------------------------------
# التصدير: صور + PDF (كاتب PDF بسيط يدمج JPEG مباشرة بدون فقدان إضافي)
# ----------------------------------------------------------------------------
def jpeg_bytes(pil_img, quality=95, dpi=300):
    buf = io.BytesIO()
    pil_img.save(buf, "JPEG", quality=int(quality), dpi=(int(dpi), int(dpi)), subsampling=0)
    return buf.getvalue()


def export_images(pages, path, fmt, dpi, quality=95):
    """يحفظ كل صفحة كصورة. إذا أكثر من صفحة يضيف _1 _2 لاسم الملف. يرجع قائمة المسارات."""
    fmt = fmt.upper()
    ext = ".png" if fmt == "PNG" else ".jpg"
    base, e = os.path.splitext(path)
    if e.lower() not in (".png", ".jpg", ".jpeg"):
        base, e = path, ext
    written = []
    for i, pg in enumerate(pages, 1):
        out = base + e if len(pages) == 1 else "%s_%d%s" % (base, i, e)
        if fmt == "PNG":
            pg.save(out, "PNG", dpi=(int(dpi), int(dpi)))
        else:
            pg.save(out, "JPEG", quality=int(quality), dpi=(int(dpi), int(dpi)), subsampling=0)
        written.append(out)
    return written


def write_pdf(path, pages, paper_w_mm, paper_h_mm, dpi=300, quality=95, title="IQ Cards"):
    """PDF متعدد الصفحات؛ كل صفحة صورة JPEG بحجم الورقة بالضبط."""
    pt = lambda mm: mm / 25.4 * 72.0  # noqa: E731
    W, H = pt(paper_w_mm), pt(paper_h_mm)
    n = len(pages)
    offsets = {}
    total_objs = 2 + 3 * n + 1
    info_num = total_objs
    safe_title = "".join(ch for ch in title if 32 <= ord(ch) < 127 and ch not in "()\\") or "IQ"
    with open(path, "wb") as f:
        f.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

        def obj(num, body):
            offsets[num] = f.tell()
            f.write(("%d 0 obj\n" % num).encode("ascii"))
            f.write(body)
            f.write(b"\nendobj\n")

        obj(1, b"<< /Type /Catalog /Pages 2 0 R >>")
        kids = " ".join("%d 0 R" % (3 + 3 * k) for k in range(n))
        obj(2, ("<< /Type /Pages /Kids [%s] /Count %d >>" % (kids, n)).encode("ascii"))
        for k, pg in enumerate(pages):
            pnum, cnum, inum = 3 + 3 * k, 4 + 3 * k, 5 + 3 * k
            jb = jpeg_bytes(pg, quality, dpi)
            obj(pnum, ("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %.4f %.4f] "
                       "/Resources << /XObject << /Im0 %d 0 R >> >> /Contents %d 0 R >>"
                       % (W, H, inum, cnum)).encode("ascii"))
            content = ("q %.4f 0 0 %.4f 0 0 cm /Im0 Do Q" % (W, H)).encode("ascii")
            obj(cnum, ("<< /Length %d >>\nstream\n" % len(content)).encode("ascii") + content + b"\nendstream")
            head = ("<< /Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace /DeviceRGB "
                    "/BitsPerComponent 8 /Filter /DCTDecode /Length %d >>\nstream\n"
                    % (pg.width, pg.height, len(jb))).encode("ascii")
            obj(inum, head + jb + b"\nendstream")
        obj(info_num, ("<< /Title (%s) /Producer (Abasky IQ %s) >>" % (safe_title, VERSION)).encode("ascii"))
        xref_pos = f.tell()
        f.write(("xref\n0 %d\n" % (total_objs + 1)).encode("ascii"))
        f.write(b"0000000000 65535 f \n")
        for num in range(1, total_objs + 1):
            f.write(("%010d 00000 n \n" % offsets[num]).encode("ascii"))
        f.write(("trailer\n<< /Size %d /Root 1 0 R /Info %d 0 R >>\nstartxref\n%d\n%%%%EOF\n"
                 % (total_objs + 1, info_num, xref_pos)).encode("ascii"))


# ----------------------------------------------------------------------------
# المشاريع (.iqp): ملف zip فيه الصور الأصلية + حالة التعديل + إعدادات الصفحة
# ----------------------------------------------------------------------------
PROJECT_EXT = ".iqp"


def save_project(path, cards, layout):
    meta = {"app": "IQ", "version": VERSION, "layout": layout, "cards": []}
    tmp = path + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        for i, c in enumerate(cards):
            cd = {"kind": c.kind, "w": c.w, "h": c.h, "copies": c.copies, "single": bool(c.single)}
            for name in (("face",) if c.single else ("face", "back")):
                s = c.side(name)
                if s.has_image():
                    arc = "img/%d_%s.jpg" % (i, name)
                    z.writestr(arc, s.orig_bytes)
                    cd[name] = {"img": arc, "state": s.state}
                else:
                    cd[name] = None
            meta["cards"].append(cd)
        z.writestr("project.json", json.dumps(meta, ensure_ascii=False, indent=1))
    os.replace(tmp, path)


def load_project(path):
    cards = []
    with zipfile.ZipFile(path) as z:
        meta = json.loads(z.read("project.json").decode("utf-8"))
        for cd in meta.get("cards", []):
            card = Card(cd.get("kind", ""), cd.get("w", 8.6), cd.get("h", 5.6), cd.get("copies", 1), cd.get("single"))
            for name in ("face", "back"):
                info = cd.get(name)
                if info:
                    s = card.side(name)
                    s.orig_bytes = z.read(info["img"])
                    s.state = normalize_state(info.get("state"))
                    s.rerender()
            cards.append(card)
    return cards, meta.get("layout", {})

