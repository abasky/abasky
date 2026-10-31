# -*- coding: utf-8 -*-
"""
Abasky-IQ — منظومة الهويات والمستمسكات (واجهة البرنامج)
قص ذكي + ضبط ألوان + تركيب تلقائي على الورق + تصدير PDF/صور + طباعة.
يعمل مع PySide6 (ويندوز 10/11) أو PyQt5 (ويندوز 7 / 32-bit).
"""
import json
import os
import sys

if sys.maxsize <= 2 ** 32:   # بايثون 32-bit (ويندوز 7 / معالجات قديمة): نمنع اختيار تعليمات غير مدعومة بالمعالج
    os.environ.setdefault("OPENBLAS_CORETYPE", "Core2")
import tempfile
import time
import traceback

import cv2
import numpy as np

import iq_core as core

try:  # ويندوز 10/11 (64-bit)
    from PySide6.QtCore import (QObject, QPointF, QRectF, QSettings, QSizeF, Qt, QThread,
                                QTimer, QUrl, Signal)
    from PySide6.QtGui import (QAction, QColor, QDesktopServices, QFont, QIcon, QImage,
                               QKeySequence, QLinearGradient, QPageSize, QPainter,
                               QPainterPath, QPen, QPixmap, QPolygonF)
    from PySide6.QtPrintSupport import QPrintDialog, QPrinter
    from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
                                   QDoubleSpinBox, QFileDialog, QFrame, QGridLayout,
                                   QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMenu,
                                   QMessageBox, QProgressDialog, QPushButton,
                                   QScrollArea, QSizePolicy, QSlider, QSpinBox,
                                   QTabWidget, QVBoxLayout, QWidget)
    QT_API = "PySide6"
except ImportError:  # ويندوز 7 / 32-bit
    from PyQt5.QtCore import (QObject, QPointF, QRectF, QSettings, QSizeF, Qt, QThread,
                              QTimer, QUrl, pyqtSignal as Signal)
    from PyQt5.QtGui import (QColor, QDesktopServices, QFont, QIcon, QImage,
                             QKeySequence, QLinearGradient, QPageSize, QPainter,
                             QPainterPath, QPen, QPixmap, QPolygonF)
    from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
    from PyQt5.QtWidgets import (QAction, QApplication, QCheckBox, QComboBox, QDialog,
                                 QDoubleSpinBox, QFileDialog, QFrame, QGridLayout,
                                 QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMenu,
                                 QMessageBox, QProgressDialog, QPushButton,
                                 QScrollArea, QSizePolicy, QSlider, QSpinBox,
                                 QTabWidget, QVBoxLayout, QWidget)
    QT_API = "PyQt5"

APP_NAME = "Abasky-IQ"
APP_TITLE = "%s %s — منظومة الهويات والمستمسكات" % (APP_NAME, core.VERSION)
APPDATA = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~"), "AbaskyIQ")
LOG_PATH = os.path.join(APPDATA, "log.txt")
AUTOSAVE_PATH = os.path.join(APPDATA, "autosave" + core.PROJECT_EXT)

PREVIEW_DPI = 110
DEFAULT_DPI = 600 if sys.maxsize > 2 ** 32 else 300   # بايثون 32-bit: ذاكرة محدودة
QUICK_THRESHOLD = 0.25                                 # حد الثقة للقص التلقائي السريع
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
IMAGE_FILTER = "الصور (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff);;كل الملفات (*)"
PROJECT_FILTER = "مشروع IQ (*%s)" % core.PROJECT_EXT


# ----------------------------------------------------------------------------
# ملاءمة النوافذ لحجم الشاشة الفعلي (مهم مع تقريب DPI المختلف بين ويندوز 7/10/11
# ومع الشاشات الصغيرة — بدون هذا ممكن تفتح نافذة أكبر من الشاشة المرئية)
# ----------------------------------------------------------------------------
def available_screen_geo(widget=None):
    scr = None
    try:
        if widget is not None:
            scr = widget.screen()
    except Exception:  # noqa: BLE001
        scr = None
    if scr is None:
        try:
            scr = QApplication.primaryScreen()
        except Exception:  # noqa: BLE001
            scr = None
    try:
        geo = scr.availableGeometry() if scr is not None else None
        int(geo.width())  # يتأكد أنها أبعاد حقيقية (قيم صحيحة) لا كائن وهمي
        int(geo.height())
        return geo
    except Exception:  # noqa: BLE001
        return None


def fit_window(widget, want_w, want_h, margin=48, center=True):
    geo = available_screen_geo(widget)
    if geo is not None:
        w = max(360, min(want_w, geo.width() - margin))
        h = max(320, min(want_h, geo.height() - margin))
    else:
        w, h = want_w, want_h
    widget.resize(w, h)
    if center and geo is not None:
        x = geo.x() + max(0, (geo.width() - w) // 2)
        y = geo.y() + max(0, (geo.height() - h) // 2)
        widget.move(x, y)

BORDER_LABELS = [("none", "بدون"), ("line", "خط رفيع حول البطاقة"), ("marks", "علامات قص بالزوايا")]
FIT_LABELS = [("stretch", "تمديد لملء المقاس"), ("cover", "ملء المقاس مع قص بسيط"), ("contain", "إظهار كامل الصورة")]

ACCENT = "#1a56c8"
STYLE = """
* { font-family: "Segoe UI", "Tahoma", "Arial"; font-size: 10pt; }
QMainWindow, QDialog { background: #f3f5f9; }
QLabel { color: #1f2937; }
QMenuBar { background: #ffffff; border-bottom: 1px solid #dfe4ea; }
QMenuBar::item { padding: 6px 12px; background: transparent; }
QMenuBar::item:selected { background: #e8effc; }
QMenu { background: #ffffff; border: 1px solid #cfd6df; }
QMenu::item { padding: 6px 28px; }
QMenu::item:selected { background: #e8effc; color: #12306e; }
QPushButton { background: #ffffff; border: 1px solid #cfd6df; border-radius: 6px; padding: 6px 12px; }
QPushButton:hover { background: #eef3fb; border-color: #aab9d6; }
QPushButton:pressed { background: #dfe8f8; }
QPushButton:disabled { color: #9aa4b2; background: #f0f2f5; }
QPushButton#primary { background: %(a)s; color: white; border: none; font-weight: bold; padding: 9px 14px; }
QPushButton#primary:hover { background: #2a66d8; }
QPushButton#primary:pressed { background: #123f9c; }
QPushButton#danger { background: #ffffff; color: #c62839; border: 1px solid #efc2c8; }
QPushButton#danger:hover { background: #fdecee; }
QPushButton#mini { padding: 2px 6px; min-width: 22px; border-radius: 5px; }
QFrame#card { background: #ffffff; border: 1px solid #dfe4ea; border-radius: 10px; }
QFrame#panel { background: #ffffff; border: 1px solid #dfe4ea; border-radius: 10px; }
QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit { background: #fff; border: 1px solid #cfd6df; border-radius: 5px; padding: 3px 6px; min-height: 22px; }
QDoubleSpinBox, QSpinBox { padding-right: 18px; min-height: 26px; }
QTabWidget::pane { border: 1px solid #dfe4ea; border-radius: 8px; background: #ffffff; top: -1px; }
QTabBar::tab { background: #e9edf3; border: 1px solid #dfe4ea; padding: 7px 16px; margin: 0 2px; border-top-left-radius: 7px; border-top-right-radius: 7px; }
QTabBar::tab:selected { background: #ffffff; color: %(a)s; font-weight: bold; border-bottom-color: #ffffff; }
QSlider::groove:horizontal { height: 5px; background: #d5dbe5; border-radius: 2px; }
QSlider::sub-page:horizontal { background: %(a)s; border-radius: 2px; }
QSlider::handle:horizontal { background: #ffffff; border: 2px solid %(a)s; width: 14px; height: 14px; margin: -6px 0; border-radius: 8px; }
QCheckBox { spacing: 6px; }
QStatusBar { background: #ffffff; border-top: 1px solid #dfe4ea; }
QScrollArea { border: none; background: transparent; }
""" % {"a": ACCENT}


# ----------------------------------------------------------------------------
# أدوات عامة
# ----------------------------------------------------------------------------
def ensure_appdata():
    try:
        os.makedirs(APPDATA, exist_ok=True)
    except OSError:
        pass


def log_error(text):
    try:
        ensure_appdata()
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write("\n[%s] %s\n%s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), APP_TITLE, text))
    except OSError:
        pass


def _event_pos(e):
    """موضع الماوس: PySide6 / Qt6 يستخدم position() و Qt5 يستخدم localPos()."""
    if hasattr(e, "position"):
        return e.position()
    return e.localPos()


def _is_accepted(result):
    return bool(result)


class Prefs(object):
    """إعدادات البرنامج (تُحفظ تلقائياً بين مرات التشغيل)."""

    def __init__(self):
        self.q = QSettings("AbaskyIQ", "IQ")

    def get(self, key, default):
        v = self.q.value(key, None)
        if v is None:
            return default
        try:
            if isinstance(default, bool):
                return str(v).lower() in ("1", "true", "yes")
            if isinstance(default, int):
                return int(float(v))
            if isinstance(default, float):
                return float(v)
            return str(v)
        except (TypeError, ValueError):
            return default

    def set(self, key, value):
        if isinstance(value, bool):
            value = "true" if value else "false"
        self.q.setValue(key, value)


def bgr_to_qimage(bgr):
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def pil_to_qimage(im):
    im = im.convert("RGB")
    data = im.tobytes("raw", "RGB")
    return QImage(data, im.width, im.height, im.width * 3, QImage.Format.Format_RGB888).copy()


def pil_thumb_pixmap(im, w, h):
    t = im.copy()
    t.thumbnail((w, h))
    return QPixmap.fromImage(pil_to_qimage(t))


def make_app_icon():
    pm = QPixmap(256, 256)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    grad = QLinearGradient(0, 0, 0, 256)
    grad.setColorAt(0, QColor("#2f6bdc"))
    grad.setColorAt(1, QColor("#123f9c"))
    p.setBrush(grad)
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(8, 8, 240, 240, 52, 52)
    p.setBrush(QColor("#ffffff"))
    p.drawRoundedRect(36, 64, 184, 116, 16, 16)
    p.setBrush(QColor("#9db8ee"))
    p.drawRoundedRect(52, 82, 44, 56, 6, 6)
    p.setBrush(QColor("#cfd9ee"))
    p.drawRoundedRect(110, 86, 92, 10, 4, 4)
    p.drawRoundedRect(110, 106, 92, 10, 4, 4)
    p.drawRoundedRect(110, 126, 60, 10, 4, 4)
    f = QFont("Segoe UI", 38)
    f.setBold(True)
    p.setFont(f)
    p.setPen(QColor("#ffffff"))
    p.drawText(QRectF(0, 178, 256, 66), Qt.AlignmentFlag.AlignCenter, "IQ")
    p.end()
    return QIcon(pm)


def qimage_file_from_clipboard():
    """يحفظ صورة الحافظة (إن وجدت) كملف مؤقت ويرجع مساره، أو None."""
    cb = QApplication.clipboard()
    md = cb.mimeData()
    if md is None or not md.hasImage():
        return None
    img = cb.image()
    if img is None or img.isNull():
        return None
    fd, path = tempfile.mkstemp(suffix=".png", prefix="iq_clip_")
    os.close(fd)
    if not img.save(path, "PNG"):
        return None
    return path


def urls_to_image_paths(mime):
    paths = []
    if mime is not None and mime.hasUrls():
        for u in mime.urls():
            p = u.toLocalFile()
            if p and p.lower().endswith(IMAGE_EXTS) and os.path.isfile(p):
                paths.append(p)
    return paths


# ----------------------------------------------------------------------------
# لوحة القص
# ----------------------------------------------------------------------------
class CropCanvas(QWidget):
    HANDLE_R = 8
    picked = Signal(float, float)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(560, 380)
        self.setMouseTracking(True)
        self.qimg = None
        self._buf = None
        self.img_w = self.img_h = 0
        self.quad = None          # 4×2 بكسل بإحداثيات الصورة المعروضة
        self.editable = True
        self.top_edge = 0         # أي ضلع سيصبح أعلى البطاقة بالنتيجة
        self.trim_pct = 0.0
        self.scale = 1.0
        self.offset = QPointF(0, 0)
        self.drag = None
        self.pan_last = None
        self.move_last = None
        self.aspect = None        # إذا حُددت: إطار القص مستطيل بنسبة ثابتة (الصور الشخصية)
        self.pick_mode = False
        self._shown = False

    # --- الصورة
    def set_image(self, bgr, keep_view=False):
        h, w = bgr.shape[:2]
        self._buf = np.ascontiguousarray(bgr[:, :, ::-1])
        self.qimg = QImage(self._buf.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.img_w, self.img_h = w, h
        if not keep_view:
            self.fit()
        self.update()

    def fit(self):
        if self.qimg is None:
            return
        pad = 24
        s = min((self.width() - pad) / float(self.img_w), (self.height() - pad) / float(self.img_h))
        self.scale = max(s, 0.01)
        self.offset = QPointF((self.width() - self.img_w * self.scale) / 2.0,
                              (self.height() - self.img_h * self.scale) / 2.0)
        self.update()

    def showEvent(self, e):
        super().showEvent(e)
        if not self._shown:
            self._shown = True
            self.fit()

    def to_view(self, p):
        return QPointF(p[0] * self.scale + self.offset.x(), p[1] * self.scale + self.offset.y())

    def to_img(self, pt):
        return ((pt.x() - self.offset.x()) / self.scale, (pt.y() - self.offset.y()) / self.scale)

    def set_aspect(self, aspect):
        """يثبّت نسبة الإطار (عرض/ارتفاع) ويعيد ضبط الإطار الحالي عليها حول مركزه."""
        self.aspect = aspect
        if aspect and self.quad is not None and self.qimg is not None:
            q = np.asarray(self.quad, np.float32)
            rh = float(q[:, 1].max() - q[:, 1].min())
            x0, y0, x1, y1 = core.fit_rect(float(q[:, 0].mean()), float(q[:, 1].mean()),
                                           rh * aspect, rh, self.img_w - 1, self.img_h - 1)
            self.quad = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
            self.update()

    def _drag_corner_locked(self, x, y):
        q = np.asarray(self.quad, np.float32)
        i = self.drag
        ox, oy = float(q[(i + 2) % 4][0]), float(q[(i + 2) % 4][1])
        dx, dy = x - ox, y - oy
        sx = 1.0 if dx >= 0 else -1.0
        sy = 1.0 if dy >= 0 else -1.0
        wd = max(abs(dx), abs(dy) * self.aspect, 24.0)
        max_w = (self.img_w - 1 - ox) if sx > 0 else ox
        max_h = (self.img_h - 1 - oy) if sy > 0 else oy
        wd = min(wd, max_w, max_h * self.aspect)
        if wd < 8:
            return
        hd = wd / self.aspect
        nx, ny = ox + sx * wd, oy + sy * hd
        x0, x1 = min(ox, nx), max(ox, nx)
        y0, y1 = min(oy, ny), max(oy, ny)
        self.quad = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
        right, bottom = nx >= ox, ny >= oy
        self.drag = 2 if (right and bottom) else 1 if right else 3 if bottom else 0
        self.update()

    def _inside_quad(self, ix, iy):
        q = np.asarray(self.quad, np.float32)
        return q[:, 0].min() <= ix <= q[:, 0].max() and q[:, 1].min() <= iy <= q[:, 1].max()

    def begin_pick(self):
        self.pick_mode = True
        self.setCursor(Qt.CursorShape.CrossCursor)

    # --- الرسم
    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor("#14212e"))
        if self.qimg is None:
            return
        target = QRectF(self.offset.x(), self.offset.y(), self.img_w * self.scale, self.img_h * self.scale)
        p.drawImage(target, self.qimg)
        if self.quad is None or not self.editable:
            return
        pts = [self.to_view(pt) for pt in self.quad]
        poly = QPolygonF(pts)
        outer = QPainterPath()
        outer.addRect(target)
        inner = QPainterPath()
        inner.addPolygon(poly)
        inner.closeSubpath()
        p.fillPath(outer.subtracted(inner), QColor(0, 0, 0, 110))
        p.setPen(QPen(QColor("#2ee66b"), 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPolygon(poly)
        if self.trim_pct > 0:
            c = np.asarray(self.quad, np.float32).mean(axis=0)
            k = max(0.2, 1.0 - 2.0 * self.trim_pct / 100.0)
            q2 = c + (np.asarray(self.quad, np.float32) - c) * k
            pen = QPen(QColor("#ffd23f"), 1.5)
            pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.drawPolygon(QPolygonF([self.to_view(pt) for pt in q2]))
        # ضلع "أعلى البطاقة"
        a = pts[self.top_edge % 4]
        b = pts[(self.top_edge + 1) % 4]
        p.setPen(QPen(QColor("#ff9f1c"), 5))
        p.drawLine(a, b)
        mid = QPointF((a.x() + b.x()) / 2.0, (a.y() + b.y()) / 2.0)
        cx = sum(q.x() for q in pts) / 4.0
        cy = sum(q.y() for q in pts) / 4.0
        dx, dy = mid.x() - cx, mid.y() - cy
        n = max((dx * dx + dy * dy) ** 0.5, 1.0)
        tp = QPointF(mid.x() + dx / n * 16 - 28, mid.y() + dy / n * 16 + 4)
        p.setPen(QColor("#ff9f1c"))
        f = QFont("Segoe UI", 9)
        f.setBold(True)
        p.setFont(f)
        p.drawText(tp, "أعلى البطاقة")
        for q in pts:
            p.setPen(QPen(QColor("#1ec8ff"), 2))
            p.setBrush(QColor("white"))
            p.drawEllipse(q, self.HANDLE_R, self.HANDLE_R)

    # --- الماوس
    def mousePressEvent(self, e):
        pos = _event_pos(e)
        if self.pick_mode and self.qimg is not None and e.button() == Qt.MouseButton.LeftButton:
            x, y = self.to_img(pos)
            x = min(max(x, 0), self.img_w - 1)
            y = min(max(y, 0), self.img_h - 1)
            self.pick_mode = False
            self.unsetCursor()
            self.picked.emit(float(x), float(y))
            return
        if e.button() == Qt.MouseButton.LeftButton and self.quad is not None and self.editable:
            best, bd = None, 26.0
            for i, pt in enumerate(self.quad):
                c = self.to_view(pt)
                d = ((c.x() - pos.x()) ** 2 + (c.y() - pos.y()) ** 2) ** 0.5
                if d < bd:
                    best, bd = i, d
            if best is not None:
                self.drag = best
                return
            if self.aspect:
                ix, iy = self.to_img(pos)
                if self._inside_quad(ix, iy):
                    self.move_last = (ix, iy)
                    return
        self.pan_last = pos

    def mouseMoveEvent(self, e):
        pos = _event_pos(e)
        if self.drag is not None and self.quad is not None:
            x, y = self.to_img(pos)
            if self.aspect:
                self._drag_corner_locked(x, y)
            else:
                self.quad[self.drag] = [min(max(x, 0), self.img_w - 1), min(max(y, 0), self.img_h - 1)]
                self.update()
        elif self.move_last is not None and self.quad is not None:
            x, y = self.to_img(pos)
            q = np.asarray(self.quad, np.float32)
            dx = min(max(x - self.move_last[0], -float(q[:, 0].min())), self.img_w - 1 - float(q[:, 0].max()))
            dy = min(max(y - self.move_last[1], -float(q[:, 1].min())), self.img_h - 1 - float(q[:, 1].max()))
            self.quad = (q + np.array([dx, dy], np.float32)).astype(np.float32)
            self.move_last = (self.move_last[0] + dx, self.move_last[1] + dy)
            self.update()
        elif self.pan_last is not None:
            self.offset += pos - self.pan_last
            self.pan_last = pos
            self.update()

    def mouseReleaseEvent(self, e):
        self.drag = None
        self.pan_last = None
        self.move_last = None

    def mouseDoubleClickEvent(self, e):
        self.fit()

    def wheelEvent(self, e):
        if self.qimg is None:
            return
        f = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        new_scale = min(max(self.scale * f, 0.03), 25.0)
        f = new_scale / self.scale
        pos = _event_pos(e)
        self.offset = QPointF(pos.x() - (pos.x() - self.offset.x()) * f,
                              pos.y() - (pos.y() - self.offset.y()) * f)
        self.scale = new_scale
        self.update()


# ----------------------------------------------------------------------------
# مربع الصورة (وجه / ظهر) داخل البطاقة
# ----------------------------------------------------------------------------
class ImageSlot(QLabel):
    clicked = Signal()
    menu_requested = Signal(object)
    files_dropped = Signal(object)

    def __init__(self, caption, portrait=False):
        super().__init__()
        self.W, self.H = (104, 134) if portrait else (152, 98)
        self.caption = caption
        self.setFixedSize(self.W, self.H)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.set_image(None)

    def set_image(self, pil_img):
        if pil_img is None:
            self.setPixmap(QPixmap())
            self.setText("%s\nاضغط أو اسحب صورة" % self.caption)
            self.setToolTip("اضغط لاختيار صورة %s، أو اسحبها وأفلتها هنا" % self.caption)
            self.setStyleSheet("QLabel { background:#f7f9fc; border:2px dashed #b9c4d6; border-radius:8px; color:#6b7686; }"
                               "QLabel:hover { border-color: %s; background:#eef3fb; }" % ACCENT)
        else:
            self.setText("")
            self.setPixmap(pil_thumb_pixmap(pil_img, self.W - 8, self.H - 8))
            self.setToolTip("اضغط لتعديل القص والألوان — زر الماوس الأيمن لخيارات أكثر")
            self.setStyleSheet("QLabel { background:#ffffff; border:2px solid #cfd6df; border-radius:8px; }"
                               "QLabel:hover { border-color: %s; }" % ACCENT)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()

    def contextMenuEvent(self, e):
        self.menu_requested.emit(e.globalPos())

    def dragEnterEvent(self, e):
        if urls_to_image_paths(e.mimeData()):
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = urls_to_image_paths(e.mimeData())
        if paths:
            e.acceptProposedAction()
            self.files_dropped.emit(paths)


class CardRow(QFrame):
    changed = Signal()
    delete_requested = Signal(object)
    move_requested = Signal(object, int)
    slot_clicked = Signal(object, str)
    slot_menu = Signal(object, str, object)
    slot_dropped = Signal(object, str, object)
    rotate_requested = Signal(object, str, int)
    whitebg_requested = Signal(object)

    def __init__(self, card, index):
        super().__init__()
        self.setObjectName("card")
        self.card = card
        lay = QGridLayout(self)
        lay.setContentsMargins(10, 8, 10, 10)
        lay.setHorizontalSpacing(8)
        lay.setVerticalSpacing(6)

        head = QHBoxLayout()
        self.title = QLabel()
        self.title.setStyleSheet("font-weight:bold; color:#12306e;")
        head.addWidget(self.title, 1)
        self.btn_up = QPushButton("▲")
        self.btn_down = QPushButton("▼")
        for b, tip in ((self.btn_up, "نقل البطاقة للأعلى"), (self.btn_down, "نقل البطاقة للأسفل")):
            b.setObjectName("mini")
            b.setToolTip(tip)
            head.addWidget(b)
        self.btn_del = QPushButton("حذف")
        self.btn_del.setObjectName("danger")
        self.btn_del.setToolTip("حذف هذه البطاقة")
        head.addWidget(self.btn_del)
        lay.addLayout(head, 0, 0, 1, 2)

        self.slots = {}
        self.holders = {}
        single = bool(card.single)
        for col, (name, cap) in enumerate((("face", "الصورة" if single else "الوجه"), ("back", "الظهر"))):
            holder = QWidget()
            self.holders[name] = holder
            box = QVBoxLayout(holder)
            box.setContentsMargins(0, 0, 0, 0)
            box.setSpacing(3)
            slot = ImageSlot(cap, portrait=single)
            self.slots[name] = slot
            box.addWidget(slot, 0, Qt.AlignmentFlag.AlignHCenter)
            rr = QHBoxLayout()
            rr.addStretch(1)
            for txt, d, tip in (("↺", -1, "تدوير 90° لليسار"), ("↻", 1, "تدوير 90° لليمين")):
                b = QPushButton(txt)
                b.setObjectName("mini")
                b.setToolTip(tip)
                b.clicked.connect(lambda _=False, n=name, dd=d: self.rotate_requested.emit(self, n, dd))
                rr.addWidget(b)
            rr.addStretch(1)
            box.addLayout(rr)
            if single and name == "face":
                lay.addWidget(holder, 1, 0, 1, 2, Qt.AlignmentFlag.AlignHCenter)
            else:
                lay.addWidget(holder, 1, col)
            if single and name == "back":
                holder.hide()
            slot.clicked.connect(lambda n=name: self.slot_clicked.emit(self, n))
            slot.menu_requested.connect(lambda pos, n=name: self.slot_menu.emit(self, n, pos))
            slot.files_dropped.connect(lambda paths, n=name: self.slot_dropped.emit(self, n, paths))

        self.spin_w = self._spin(card.w)
        self.spin_h = self._spin(card.h)
        self.spin_c = QSpinBox()
        self.spin_c.setRange(1, 50)
        self.spin_c.setValue(card.copies)
        self.spin_c.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.spin_c.setToolTip("عدد النسخ المطلوبة من هذه البطاقة")
        sizes = QGridLayout()                      # عمودياً (تسمية فوق الحقل) حتى لا يتجاوز عرض اللوحة الجانبية
        sizes.setHorizontalSpacing(6)
        sizes.setVerticalSpacing(1)
        for col, (txt, sp) in enumerate((("العرض (سم)", self.spin_w), ("الارتفاع (سم)", self.spin_h), ("النسخ", self.spin_c))):
            sp.setMinimumWidth(0)
            sp.setMaximumWidth(96)
            lbl = QLabel(txt)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            sizes.addWidget(lbl, 0, col)
            sizes.addWidget(sp, 1, col, Qt.AlignmentFlag.AlignHCenter)
        lay.addLayout(sizes, 2, 0, 1, 2)
        self.btn_white = QPushButton()
        self.btn_white.setToolTip("يفصل الشخص عن الخلفية ويضع خلفية بيضاء (تقدر ترجع الخلفية الأصلية بأي وقت)")
        self.btn_white.clicked.connect(lambda: self.whitebg_requested.emit(self))
        lay.addWidget(self.btn_white, 3, 0, 1, 2)
        if not single:
            self.btn_white.hide()

        self.spin_w.valueChanged.connect(self._size_changed)
        self.spin_h.valueChanged.connect(self._size_changed)
        self.spin_c.valueChanged.connect(self._size_changed)
        self.btn_del.clicked.connect(lambda: self.delete_requested.emit(self))
        self.btn_up.clicked.connect(lambda: self.move_requested.emit(self, -1))
        self.btn_down.clicked.connect(lambda: self.move_requested.emit(self, 1))
        self.set_index(index)
        self.refresh()

    @staticmethod
    def _spin(v):
        s = QDoubleSpinBox()
        s.setRange(1.0, 29.0)
        s.setDecimals(2)
        s.setSingleStep(0.1)
        s.setValue(v)
        s.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        return s

    def _size_changed(self, *_):
        self.card.w = self.spin_w.value()
        self.card.h = self.spin_h.value()
        self.card.copies = self.spin_c.value()
        self.changed.emit()

    def set_index(self, i):
        self.title.setText("البطاقة %d — %s" % (i, self.card.kind))

    def refresh(self):
        if self.card.single:
            st = self.card.face.state or {}
            on = bool((st.get("adj") or {}).get("whitebg"))
            self.btn_white.setText("إرجاع الخلفية الأصلية" if on else "إزالة الخلفية (أبيض)")
            self.btn_white.setEnabled(self.card.face.has_image())
        for name, slot in self.slots.items():
            s = self.card.side(name)
            slot.set_image(s.result if s.has_image() else None)


class PreviewLabel(QLabel):
    swap_requested = Signal(object, object)   # (item_a, item_b) — عنصرا iq_core.Item المطلوب تبديل مواضعهما

    def __init__(self):
        super().__init__()
        self._pm = None
        self.placements = []            # [(x, y, w, h, Item), ...] بالمليمتر — لصفحة المعاينة الحالية
        self.paper_wh = (210.0, 297.0)  # أبعاد الورقة الحالية (مم)
        self._drag_item = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setMinimumSize(240, 320)
        self.setStyleSheet("background:#e4e9f0; border:1px solid #cfd6df; border-radius:8px;")
        self.setToolTip("اسحب أي صورة هنا وأفلتها على مكان صورة أخرى لتبديل مواضعهما (الأحجام والترتيب العام لا يتغيران)")

    def set_source(self, pm):
        self._pm = pm
        self._rescale()

    def set_layout(self, paper_w, paper_h, placements):
        """يحفظ هندسة الصفحة الحالية (بالمليمتر) حتى يقدر السحب والإفلات يحدد الصورة تحت المؤشر."""
        self.paper_wh = (float(paper_w), float(paper_h))
        self.placements = placements

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._rescale()

    def _rescale(self):
        if self._pm is not None:
            self.setPixmap(self._pm.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                           Qt.TransformationMode.SmoothTransformation))

    def _item_at(self, pos):
        """يرجع عنصر iq_core.Item تحت نقطة pos (إحداثيات داخل هذا الودجت)، أو None."""
        pm = self.pixmap()
        if pm is None or pm.isNull() or not self.placements:
            return None
        ox = (self.width() - pm.width()) / 2.0
        oy = (self.height() - pm.height()) / 2.0
        px, py = pos.x() - ox, pos.y() - oy
        if px < 0 or py < 0 or px >= pm.width() or py >= pm.height():
            return None
        pw_mm, ph_mm = self.paper_wh
        if pm.width() <= 0 or pm.height() <= 0:
            return None
        mx = px / pm.width() * pw_mm
        my = py / pm.height() * ph_mm
        for x, y, w, h, it in self.placements:
            if x <= mx <= x + w and y <= my <= y + h:
                return it
        return None

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag_item = self._item_at(_event_pos(e))
            if self._drag_item is not None:
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        if self._drag_item is not None:
            src = self._drag_item
            self._drag_item = None
            self.unsetCursor()
            target = self._item_at(_event_pos(e))
            if target is not None and target is not src:
                self.swap_requested.emit(src, target)
        super().mouseReleaseEvent(e)


# ----------------------------------------------------------------------------
# رسائل بأزرار عربية
# ----------------------------------------------------------------------------
def ask_buttons(parent, title, text, labels, icon="question"):
    """يعرض رسالة بأزرار عربية. يرجع رقم الزر المضغوط (حسب ترتيب labels) أو -1."""
    box = QMessageBox(parent)
    box.setWindowTitle(title)
    box.setText(text)
    box.setIcon(QMessageBox.Icon.Warning if icon == "warning" else QMessageBox.Icon.Question)
    box.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
    btns = []
    for i, lab in enumerate(labels):
        role = QMessageBox.ButtonRole.AcceptRole if i == 0 else (
            QMessageBox.ButtonRole.RejectRole if i == len(labels) - 1 else QMessageBox.ButtonRole.DestructiveRole)
        btns.append(box.addButton(lab, role))
    box.exec()
    clicked = box.clickedButton()
    for i, b in enumerate(btns):
        if clicked is b:
            return i
    return -1


def info_box(parent, title, text, warning=False):
    box = QMessageBox(parent)
    box.setWindowTitle(title)
    box.setText(text)
    box.setIcon(QMessageBox.Icon.Warning if warning else QMessageBox.Icon.Information)
    box.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
    box.addButton("حسناً", QMessageBox.ButtonRole.AcceptRole)
    box.exec()


# ----------------------------------------------------------------------------
# نافذة تحرير الصورة: قص + تدوير + ألوان وإضاءة (غير مدمّر — تقدر ترجع تعدّل بأي وقت)
# ----------------------------------------------------------------------------
class CropDialog(QDialog):
    ADJ_SLIDERS = (
        ("brightness", "السطوع", -100, 100),
        ("contrast", "التباين", -100, 100),
        ("saturation", "التشبع", -100, 100),
        ("warmth", "حرارة اللون", -100, 100),
        ("sharpness", "الحدة", 0, 100),
    )
    ADJ_CHECKS = (
        ("auto", "تحسين تلقائي"),
        ("flatten", "معادلة الإضاءة (تخفيف الظلال)"),
        ("bw", "أبيض وأسود"),
    )

    def __init__(self, orig_bgr, state, parent=None, score=None, aspect=None):
        super().__init__(parent)
        self.aspect = aspect      # عرض/ارتفاع مقاس الصورة الشخصية، أو None للهويات
        self.setWindowTitle("تحرير الصورة — قص وتدوير وألوان")
        self.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.orig = orig_bgr
        self.small = core.limit_size(orig_bgr, 2400)
        self.st = core.normalize_state(state)
        self.result_state = None
        self.sliders = {}
        self.checks = {}
        self._loading = False
        self._mode = "crop"
        self._prev_shape = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._compute_preview)

        root = QHBoxLayout(self)
        left = QVBoxLayout()
        self.canvas = CropCanvas()
        self.canvas.picked.connect(self.on_picked)
        left.addWidget(self.canvas, 1)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color:#44546a; padding:2px;")
        left.addWidget(self.status)
        root.addLayout(left, 1)

        side = QVBoxLayout()
        self.tabs = QTabWidget()
        self.tabs.setMinimumWidth(340)
        self.tabs.setMaximumWidth(400)
        self.tabs.addTab(self._build_crop_tab(), "القص والتدوير")
        self.tabs.addTab(self._build_color_tab(), "الألوان والإضاءة")
        self.tabs.currentChanged.connect(self._on_tab)
        side.addWidget(self.tabs, 1)
        row = QHBoxLayout()
        ok = QPushButton("اعتماد وتثبيت")
        ok.setObjectName("primary")
        ok.setDefault(True)
        ok.clicked.connect(self.accept_result)
        cancel = QPushButton("إلغاء")
        cancel.clicked.connect(self.reject)
        row.addWidget(ok, 2)
        row.addWidget(cancel, 1)
        side.addLayout(row)
        root.addLayout(side)
        self.setMinimumSize(640, 460)
        fit_window(self, 1180, 760, margin=60)

        q = self.st["quad"]
        self.quad_norm = [list(p) for p in q] if q is not None else self._inset_quad()
        self._load_ui_from_state()
        self._show_crop_view()
        if self.aspect:
            if score is not None and score <= 0:
                self.status.setText("ما كدرت أتعرف على الوجه تلقائياً — حرّك الإطار أو كبّره من الزوايا (النسبة ثابتة) ثم اضغط «اعتماد وتثبيت».")
            else:
                self.status.setText("راجع الإطار: اسحب الزوايا للتكبير والتصغير، واسحب من الداخل للتحريك. زر «إزالة الخلفية» يضع خلفية بيضاء.")
        elif q is None:
            self.status.setText("ما كدرت أحدد الهوية تلقائياً — اضغط «حدد الهوية بالنقر» ثم انقر على الهوية، أو حرّك الزوايا يدوياً.")
        elif score is not None and score < 0.08:
            self.status.setText("الثقة بالحدود ضعيفة — راجعها، أو جرّب «حدد الهوية بالنقر».")
        else:
            self.status.setText("راجع الإطار الأخضر ثم اضغط «اعتماد وتثبيت». الضلع البرتقالي = أعلى البطاقة.")

    # --- بناء التبويبات
    def _build_crop_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        hint = QLabel("اسحب الزوايا الأربع حول الهوية. عجلة الماوس للتقريب، وضغطتين لملاءمة الصورة. "
                      "الضلع البرتقالي هو أعلى البطاقة بالنتيجة.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#44546a;")
        lay.addWidget(hint)
        grid = QGridLayout()
        if self.aspect:
            items = (
                ("تعرّف على الوجه وقص", self.smart_detect), ("إزالة الخلفية (أبيض)", self.toggle_white),
                ("ملاءمة الصورة", self.canvas_fit), ("استعادة الأصل", self.reset_all),
                ("تدوير ↺ 90°", lambda: self.rotate(-1)), ("تدوير ↻ 90°", lambda: self.rotate(1)),
            )
        else:
            items = (
                ("تعرّف ذكي على الهوية", self.smart_detect), ("حدد الهوية بالنقر", self.start_pick),
                ("قص بدون (الصورة كاملة)", self.no_crop), ("ملاءمة الصورة", self.canvas_fit),
                ("تدوير ↺ 90°", lambda: self.rotate(-1)), ("تدوير ↻ 90°", lambda: self.rotate(1)),
                ("استعادة الأصل", self.reset_all),
            )
        for i, (txt, fn) in enumerate(items):
            b = QPushButton(txt)
            b.clicked.connect(lambda _=False, f=fn: f())
            grid.addWidget(b, i // 2, i % 2)
        lay.addLayout(grid)
        trow = QHBoxLayout()
        trow.addWidget(QLabel("تضييق الحواف"))
        self.trim_slider = QSlider(Qt.Orientation.Horizontal)
        self.trim_slider.setRange(0, 80)
        self.trim_slider.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.trim_slider.setToolTip("يقص من أطراف البطاقة قليلاً لإزالة حافة الخلفية")
        self.trim_label = QLabel("0.0%")
        self.trim_label.setMinimumWidth(42)
        self.trim_slider.valueChanged.connect(self._trim_changed)
        trow.addWidget(self.trim_slider, 1)
        trow.addWidget(self.trim_label)
        lay.addLayout(trow)
        lay.addStretch(1)
        return w

    def _build_color_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        hint = QLabel("حرّك المنزلقات وشوف النتيجة مباشرة. هذي التعديلات تنطبق على الصورة بعد القص.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#44546a;")
        lay.addWidget(hint)
        for key, text, lo, hi in self.ADJ_SLIDERS:
            row = QHBoxLayout()
            lbl = QLabel(text)
            lbl.setMinimumWidth(78)
            sl = QSlider(Qt.Orientation.Horizontal)
            sl.setRange(lo, hi)
            sl.setValue(0)
            sl.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
            val = QLabel("0")
            val.setMinimumWidth(32)
            val.setAlignment(Qt.AlignmentFlag.AlignCenter)
            sl.valueChanged.connect(lambda v, k=key, vl=val: self._adj_slider_changed(k, v, vl))
            row.addWidget(lbl)
            row.addWidget(sl, 1)
            row.addWidget(val)
            lay.addLayout(row)
            self.sliders[key] = (sl, val)
        checks = self.ADJ_CHECKS
        if self.aspect:
            checks = (("whitebg", "خلفية بيضاء (إزالة الخلفية)"),) + tuple(checks)
        for key, text in checks:
            cb = QCheckBox(text)
            cb.toggled.connect(lambda checked, k=key: self._adj_check_changed(k, checked))
            lay.addWidget(cb)
            self.checks[key] = cb
        self.chk_compare = QCheckBox("عرض الصورة قبل التعديل (للمقارنة)")
        self.chk_compare.toggled.connect(lambda _c: self._compute_preview())
        lay.addWidget(self.chk_compare)
        reset = QPushButton("إعادة ضبط الألوان")
        reset.clicked.connect(self.reset_colors)
        lay.addWidget(reset)
        lay.addStretch(1)
        return w

    # --- الحالة
    def _eff_aspect(self):
        return self.aspect if self.st["rot"] % 2 == 0 else 1.0 / self.aspect

    def _inset_quad(self):
        if self.aspect:
            return core.default_photo_quad(self.small.shape[1], self.small.shape[0], self._eff_aspect())
        return [[0.08, 0.08], [0.92, 0.08], [0.92, 0.92], [0.08, 0.92]]

    def _load_ui_from_state(self):
        self._loading = True
        adj = self.st["adj"]
        for key, (sl, val) in self.sliders.items():
            sl.setValue(int(adj.get(key, 0)))
            val.setText(str(int(adj.get(key, 0))))
        for key, cb in self.checks.items():
            cb.setChecked(bool(adj.get(key, False)))
        self.trim_slider.setValue(int(round(float(self.st["trim"]) * 10)))
        self.trim_label.setText("%.1f%%" % float(self.st["trim"]))
        self._loading = False

    def _sync_quad_from_canvas(self):
        if self._mode == "crop" and self.canvas.quad is not None and self.canvas.img_w > 1:
            dims = np.array([max(self.canvas.img_w - 1, 1), max(self.canvas.img_h - 1, 1)], np.float32)
            a = np.clip(np.asarray(self.canvas.quad, np.float32) / dims, 0.0, 1.0)
            self.quad_norm = a.tolist()

    def _state_now(self):
        return {"rot": self.st["rot"], "quad": [list(p) for p in self.quad_norm],
                "trim": float(self.st["trim"]), "adj": dict(self.st["adj"])}

    # --- العرض
    def _show_crop_view(self):
        self._mode = "crop"
        self.canvas.editable = True
        self.canvas.set_image(self.small)
        dims = np.array([self.canvas.img_w - 1, self.canvas.img_h - 1], np.float32)
        self.canvas.quad = (np.asarray(self.quad_norm, np.float32) * dims).astype(np.float32)
        self.canvas.top_edge = (4 - self.st["rot"]) % 4
        self.canvas.trim_pct = float(self.st["trim"])
        self.canvas.aspect = self._eff_aspect() if self.aspect else None
        self.canvas.update()

    def _show_preview(self):
        self._sync_quad_from_canvas()
        self._mode = "preview"
        self.canvas.editable = False
        self._prev_shape = None
        self._compute_preview()

    def _compute_preview(self):
        if self._mode != "preview":
            return
        st = self._state_now()
        if self.chk_compare.isChecked():
            st["adj"] = dict(core.DEFAULT_ADJ)
        out = core.render_side(self.small, st, 1100)
        changed = out.shape[:2] != self._prev_shape
        self._prev_shape = out.shape[:2]
        self.canvas.set_image(out, keep_view=not changed)

    def _schedule_preview(self):
        if self._mode == "preview":
            self._timer.start(40)

    def _on_tab(self, idx):
        if idx == 1:
            self._show_preview()
            self.status.setText("هذي المعاينة بعد القص والتعديل.")
        else:
            self._sync_quad_from_canvas() if self._mode == "crop" else None
            self._show_crop_view()
            self.status.setText("اسحب الزوايا لضبط القص.")

    # --- أوامر القص
    def canvas_fit(self):
        self.canvas.fit()

    def _detect(self, seed=None):
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            return core.detect_card_quad_scored(self.small, seed)
        finally:
            QApplication.restoreOverrideCursor()

    def toggle_white(self):
        cb = self.checks.get("whitebg")
        if cb is None:
            return
        self._sync_quad_from_canvas()
        cb.setChecked(not cb.isChecked())
        self.tabs.setCurrentIndex(1)

    def _detect_face(self):
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            st, found = core.photo_state(self.small, self._eff_aspect())
        finally:
            QApplication.restoreOverrideCursor()
        dims = np.array([max(self.canvas.img_w - 1, 1), max(self.canvas.img_h - 1, 1)], np.float32)
        self.canvas.quad = (np.asarray(st["quad"], np.float32) * dims).astype(np.float32)
        self.canvas.update()
        self.status.setText("تم قص الوجه تلقائياً. راجع الإطار ثم اعتمد." if found
                            else "ما لقيت وجه — حرّك الإطار يدوياً.")

    def smart_detect(self):
        if self.aspect:
            self._detect_face()
            return
        q, score = self._detect()
        if q is None:
            self.status.setText("ما كدرت أتعرف على الهوية — جرّب «حدد الهوية بالنقر» أو حرّك الزوايا يدوياً.")
            return
        self.canvas.quad = np.asarray(q, np.float32)
        self.canvas.update()
        self.status.setText("تم اقتراح الحدود. راجع الزوايا ثم اعتمد." if score >= 0.08
                            else "الثقة ضعيفة بالحدود — راجعها أو جرّب «حدد الهوية بالنقر».")

    def start_pick(self):
        self.canvas.begin_pick()
        self.status.setText("انقر بالماوس على أي نقطة داخل الهوية...")

    def on_picked(self, x, y):
        q, score = self._detect((x, y))
        if q is None:
            self.status.setText("ما كدرت أحدد الهوية من هذي النقطة — جرّب نقطة ثانية أو حرّك الزوايا يدوياً.")
            return
        self.canvas.quad = np.asarray(q, np.float32)
        self.canvas.update()
        self.status.setText("تم تحديد الهوية. راجع الزوايا ثم اعتمد.")

    def no_crop(self):
        w, h = self.canvas.img_w, self.canvas.img_h
        self.canvas.quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
        self.canvas.update()
        self.status.setText("بدون قص — تُستعمل الصورة كاملة.")

    def rotate(self, d):
        self.st["rot"] = (self.st["rot"] + d) % 4
        self.canvas.top_edge = (4 - self.st["rot"]) % 4
        if self.aspect:
            self.canvas.set_aspect(self._eff_aspect())
        self.canvas.update()
        self.status.setText("تم تدوير اتجاه البطاقة — الضلع البرتقالي هو الأعلى.")

    def _trim_changed(self, v):
        t = v / 10.0
        self.trim_label.setText("%.1f%%" % t)
        if self._loading:
            return
        self.st["trim"] = t
        self.canvas.trim_pct = t
        self.canvas.update()
        self._schedule_preview()

    def reset_all(self):
        self.st = core.default_state()
        if self.aspect:
            self.quad_norm = core.photo_state(self.small, self._eff_aspect())[0]["quad"]
            q = None
        else:
            q, score = self._detect()
            self.quad_norm = self._inset_quad()
        if q is not None:
            dims = np.array([max(self.small.shape[1] - 1, 1), max(self.small.shape[0] - 1, 1)], np.float32)
            self.quad_norm = np.clip(np.asarray(q, np.float32) / dims, 0, 1).tolist()
        self._load_ui_from_state()
        if self.tabs.currentIndex() != 0:
            self.tabs.setCurrentIndex(0)
        else:
            self._show_crop_view()
        self.status.setText("تمت استعادة الأصل.")

    # --- أوامر الألوان
    def _adj_slider_changed(self, key, value, lbl):
        lbl.setText(str(value))
        if self._loading:
            return
        self.st["adj"][key] = int(value)
        self._schedule_preview()

    def _adj_check_changed(self, key, checked):
        if self._loading:
            return
        self.st["adj"][key] = bool(checked)
        self._schedule_preview()

    def reset_colors(self):
        self.st["adj"] = dict(core.DEFAULT_ADJ)
        self._load_ui_from_state()
        self._schedule_preview()
        self._compute_preview()

    # --- النتيجة
    def accept_result(self):
        self._sync_quad_from_canvas()
        st = self._state_now()
        if core.is_full_quad(st["quad"]):
            st["quad"] = [list(p) for p in core.FULL_QUAD]
        self.result_state = core.normalize_state(st)
        self.accept()


# ----------------------------------------------------------------------------
# نوافذ صغيرة: قالب جديد + إعدادات
# ----------------------------------------------------------------------------
class TemplateDialog(QDialog):
    def __init__(self, parent=None, name="", w=8.6, h=5.6, single=False):
        super().__init__(parent)
        self.setWindowTitle("حفظ مقاس كقالب")
        self.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        lay = QGridLayout(self)
        self.edit_name = QLineEdit(name)
        self.spin_w = QDoubleSpinBox()
        self.spin_h = QDoubleSpinBox()
        for s, v in ((self.spin_w, w), (self.spin_h, h)):
            s.setRange(1.0, 29.0)
            s.setDecimals(2)
            s.setSingleStep(0.1)
            s.setValue(v)
            s.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        lay.addWidget(QLabel("اسم القالب"), 0, 0)
        lay.addWidget(self.edit_name, 0, 1)
        lay.addWidget(QLabel("العرض (سم)"), 1, 0)
        lay.addWidget(self.spin_w, 1, 1)
        lay.addWidget(QLabel("الارتفاع (سم)"), 2, 0)
        lay.addWidget(self.spin_h, 2, 1)
        self.chk_single = QCheckBox("صورة شخصية (وجه فقط، ترتيب أفقي، قص على الوجه)")
        self.chk_single.setChecked(single)
        lay.addWidget(self.chk_single, 3, 0, 1, 2)
        row = QHBoxLayout()
        ok = QPushButton("حفظ")
        ok.setObjectName("primary")
        ok.clicked.connect(self.accept)
        cancel = QPushButton("إلغاء")
        cancel.clicked.connect(self.reject)
        row.addWidget(ok)
        row.addWidget(cancel)
        lay.addLayout(row, 4, 0, 1, 2)
        self.setMinimumWidth(320)

    def values(self):
        return self.edit_name.text().strip(), self.spin_w.value(), self.spin_h.value(), self.chk_single.isChecked()


class SettingsDialog(QDialog):
    def __init__(self, prefs, parent=None):
        super().__init__(parent)
        self.prefs = prefs
        self.setWindowTitle("الإعدادات")
        self.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        lay = QVBoxLayout(self)
        self.chk_autosave = QCheckBox("حفظ الجلسة تلقائياً عند الإغلاق واسترجاعها عند الفتح")
        self.chk_autosave.setChecked(prefs.get("autosave", False))
        note = QLabel("تنبيه: هذا الخيار يحفظ صور الهويات داخل مجلد البرنامج على هذا الجهاز. "
                      "لا تفعّله على جهاز مشترك.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#8a5a00;")
        self.chk_open = QCheckBox("فتح الملف تلقائياً بعد التصدير")
        self.chk_open.setChecked(prefs.get("open_after", False))
        form = QGridLayout()
        self.combo_q = QComboBox()
        for q in (85, 90, 95, 100):
            self.combo_q.addItem("%d%%" % q, q)
        self.combo_q.setCurrentIndex(max(0, self.combo_q.findData(prefs.get("jpg_q", 95))))
        form.addWidget(QLabel("جودة JPEG عند التصدير"), 0, 0)
        form.addWidget(self.combo_q, 0, 1)
        lay.addWidget(self.chk_autosave)
        lay.addWidget(note)
        lay.addWidget(self.chk_open)
        lay.addLayout(form)
        self.btn_clear = QPushButton("حذف الجلسة المحفوظة تلقائياً الآن")
        self.btn_clear.clicked.connect(self._clear_autosave)
        lay.addWidget(self.btn_clear)
        row = QHBoxLayout()
        ok = QPushButton("موافق")
        ok.setObjectName("primary")
        ok.clicked.connect(self._apply)
        cancel = QPushButton("إلغاء")
        cancel.clicked.connect(self.reject)
        row.addWidget(ok)
        row.addWidget(cancel)
        lay.addLayout(row)
        self.setMinimumWidth(420)

    def _clear_autosave(self):
        try:
            if os.path.exists(AUTOSAVE_PATH):
                os.remove(AUTOSAVE_PATH)
            info_box(self, "تم", "تم حذف الجلسة المحفوظة.")
        except OSError as e:
            info_box(self, "خطأ", "تعذر الحذف:\n%s" % e, warning=True)

    def _apply(self):
        self.prefs.set("autosave", self.chk_autosave.isChecked())
        self.prefs.set("open_after", self.chk_open.isChecked())
        self.prefs.set("jpg_q", int(self.combo_q.currentData()))
        self.accept()


# ----------------------------------------------------------------------------
# النافذة الرئيسية
# ----------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self, prefs):
        super().__init__()
        self.prefs = prefs
        self.setWindowTitle(APP_TITLE)
        self.setWindowIcon(make_app_icon())
        self.setAcceptDrops(True)
        self.cards, self.rows = [], []
        self.pages, self.page_idx = [], 0
        self.project_path = None
        self.dirty = False
        self._building = True
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._render_preview)

        self._build_ui()
        self._build_menu()
        self._load_prefs()
        self._building = False
        self.add_card()
        self.statusBar().showMessage("جاهز — اسحب صور الهويات وأفلتها هنا، أو اضغط على مربع الوجه/الظهر.")
        try:
            geo = prefs.q.value("geometry")
            if geo:
                self.restoreGeometry(geo)
                self._fit_to_current_screen()
            else:
                fit_window(self, 1280, 860)
        except Exception:  # noqa: BLE001
            fit_window(self, 1280, 860)

    def _fit_to_current_screen(self):
        """إذا الحجم/الموقع المحفوظ من جلسة سابقة صار أكبر من الشاشة الحالية أو خارجها
        (مثلاً المشروع انتقل لجهاز بشاشة أصغر، أو تغيّر تقريب DPI)، نعيد ضبطه."""
        geo = available_screen_geo(self)
        if geo is None:
            return
        fr = self.frameGeometry()
        if fr.width() > geo.width() or fr.height() > geo.height() or not geo.intersects(fr):
            fit_window(self, min(fr.width() or 1280, 1280), min(fr.height() or 860, 860))

    # ------------------------------------------------------------------ بناء الواجهة
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        v = QVBoxLayout(root)
        v.setContentsMargins(10, 8, 10, 8)

        main = QHBoxLayout()
        v.addLayout(main, 1)

        # --- المعاينة
        pv = QVBoxLayout()
        nav = QWidget()
        nav.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        nl = QHBoxLayout(nav)
        nl.setContentsMargins(0, 0, 0, 0)
        self.btn_prev = QPushButton("السابق")
        self.btn_next = QPushButton("التالي")
        self.lbl_page = QLabel()
        self.lbl_page.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nl.addWidget(self.btn_prev)
        nl.addWidget(self.lbl_page, 1)
        nl.addWidget(self.btn_next)
        pv.addWidget(nav)
        self.preview = PreviewLabel()
        self.preview.swap_requested.connect(self._swap_preview_items)
        pv.addWidget(self.preview, 1)
        self.lbl_info = QLabel()
        self.lbl_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_info.setStyleSheet("color:#556275;")
        pv.addWidget(self.lbl_info)
        main.addLayout(pv, 1)

        # --- لوحة التحكم (تبويبان)
        self.ctrl_tabs = QTabWidget()
        self.ctrl_tabs.setMinimumWidth(410)
        self.ctrl_tabs.setMaximumWidth(450)
        self.ctrl_tabs.addTab(self._build_cards_tab(), "البطاقات")
        self.ctrl_tabs.addTab(self._build_layout_tab(), "الصفحة والتخطيط")
        main.addWidget(self.ctrl_tabs)

        # --- أزرار الأسفل
        bottom = QHBoxLayout()
        self.combo_fmt = QComboBox()
        self.combo_fmt.addItems(["JPG", "PNG"])
        self.combo_dpi = QComboBox()
        for label, d in (("150 dpi (مسودة)", 150), ("300 dpi (عادي)", 300), ("600 dpi (عالي)", 600)):
            self.combo_dpi.addItem(label, d)
        self.btn_img = QPushButton("تصدير صورة")
        self.btn_pdf = QPushButton("تصدير PDF")
        self.btn_print = QPushButton("طباعة مباشرة")
        for b in (self.btn_img, self.btn_pdf, self.btn_print):
            b.setObjectName("primary")
        bottom.addWidget(self.combo_fmt)
        bottom.addWidget(self.combo_dpi)
        bottom.addWidget(self.btn_img, 1)
        bottom.addWidget(self.btn_pdf, 1)
        bottom.addWidget(self.btn_print, 1)
        v.addLayout(bottom)

        self.btn_prev.clicked.connect(lambda: self.go_page(-1))
        self.btn_next.clicked.connect(lambda: self.go_page(1))
        self.btn_img.clicked.connect(self.export_image)
        self.btn_pdf.clicked.connect(self.export_pdf)
        self.btn_print.clicked.connect(self.print_direct)
        self.combo_dpi.currentIndexChanged.connect(self._export_opts_changed)
        self.combo_fmt.currentIndexChanged.connect(self._export_opts_changed)

    def _build_cards_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)

        srow = QHBoxLayout()
        lbl = QLabel("المقاس")
        lbl.setStyleSheet("font-weight:bold;")
        self.combo_style = QComboBox()
        self.btn_tpl_save = QPushButton("قالب جديد")
        self.btn_tpl_del = QPushButton("حذف القالب")
        self.btn_tpl_save.setToolTip("حفظ مقاس مخصص كقالب جاهز")
        self.btn_tpl_del.setToolTip("حذف القالب المخصص المحدد")
        srow.addWidget(lbl)
        srow.addWidget(self.combo_style, 1)
        lay.addLayout(srow)
        trow = QHBoxLayout()
        trow.addWidget(self.btn_tpl_save)
        trow.addWidget(self.btn_tpl_del)
        lay.addLayout(trow)

        self.btn_add = QPushButton("＋ إضافة بطاقة")
        self.btn_add.setObjectName("primary")
        self.btn_add_files = QPushButton("إضافة عدة صور دفعة واحدة...")
        lay.addWidget(self.btn_add)
        lay.addWidget(self.btn_add_files)

        self.chk_quick = QCheckBox("قص تلقائي سريع (بدون نافذة التحرير إذا الثقة عالية)")
        self.chk_quick.setToolTip("تُقص الصورة تلقائياً وتدخل الصفحة مباشرة. تقدر تضغط عليها لاحقاً لتعديلها.")
        lay.addWidget(self.chk_quick)
        self.chk_ai = QCheckBox("الذكاء الاصطناعي (اختياري): قص وتعرّف وإزالة خلفية أدق")
        self.chk_ai.setToolTip("يستخدم نماذج مجانية تعمل على جهازك بدون إنترنت. إذا ما اشتغلت بجهازك يرجع البرنامج للطريقة العادية.")
        self.lbl_ai = QLabel("")
        self.lbl_ai.setWordWrap(True)
        self.lbl_ai.setStyleSheet("color:#6b7686; font-size:9pt;")
        lay.addWidget(self.chk_ai)
        lay.addWidget(self.lbl_ai)
        self.chk_white = QCheckBox("الصور الشخصية: خلفية بيضاء تلقائياً")
        self.chk_white.setToolTip("عند إضافة صورة شخصية تُزال خلفيتها وتصير بيضاء. تقدر تغيّرها لكل صورة لاحقاً.")
        lay.addWidget(self.chk_white)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        holder = QWidget()
        self.cards_lay = QVBoxLayout(holder)
        self.cards_lay.setContentsMargins(0, 0, 4, 0)
        self.cards_lay.setSpacing(8)
        self.cards_lay.addStretch(1)
        scroll.setWidget(holder)
        lay.addWidget(scroll, 1)

        tip = QLabel("نصيحة: اسحب صور الهويات من المجلد وأفلتها على أي مكان بالنافذة، "
                     "أو انسخ صورة والصقها بـ Ctrl+V. اضغط على أي صورة لتعديلها.")
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#6b7686; font-size:9pt;")
        lay.addWidget(tip)

        self.btn_add.clicked.connect(lambda: self.add_card())
        self.btn_add_files.clicked.connect(self.add_files_dialog)
        self.btn_tpl_save.clicked.connect(self.save_template)
        self.btn_tpl_del.clicked.connect(self.delete_template)
        self.chk_quick.toggled.connect(lambda c: self.prefs.set("quick", bool(c)))
        self.chk_white.toggled.connect(lambda c: self.prefs.set("auto_white", bool(c)))
        self.chk_ai.toggled.connect(self._ai_toggled)
        self.combo_style.currentIndexChanged.connect(self._style_changed)
        return w

    def _build_layout_tab(self):
        w = QWidget()
        g = QGridLayout(w)
        g.setContentsMargins(10, 10, 10, 10)
        g.setVerticalSpacing(10)
        r = 0

        self.combo_paper = QComboBox()
        for n, pw, ph in core.PAPERS:
            self.combo_paper.addItem("%s  (%g × %g مم)" % (n, pw, ph), n)
        g.addWidget(QLabel("حجم الورق"), r, 0)
        g.addWidget(self.combo_paper, r, 1, 1, 2)
        r += 1

        self.spin_margin = QDoubleSpinBox()
        self.spin_margin.setRange(0.0, 40.0)
        self.spin_margin.setDecimals(1)
        self.spin_margin.setSuffix(" مم")
        self.spin_margin.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        g.addWidget(QLabel("هامش الصفحة"), r, 0)
        g.addWidget(self.spin_margin, r, 1, 1, 2)
        r += 1

        self.sl_v, self.lbl_v = self._make_gap_slider(g, r, "تباعد أعلى وأسفل")
        r += 1
        self.sl_h, self.lbl_h = self._make_gap_slider(g, r, "تباعد الجوانب")
        r += 1

        self.spin_corner = QDoubleSpinBox()
        self.spin_corner.setRange(0.0, 10.0)
        self.spin_corner.setDecimals(1)
        self.spin_corner.setSingleStep(0.5)
        self.spin_corner.setSuffix(" مم")
        self.spin_corner.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.spin_corner.setToolTip("تدوير زوايا البطاقة لإخفاء بقايا الخلفية (0 = زوايا حادة)")
        g.addWidget(QLabel("تدوير الزوايا"), r, 0)
        g.addWidget(self.spin_corner, r, 1, 1, 2)
        r += 1

        self.combo_border = QComboBox()
        for k, label in BORDER_LABELS:
            self.combo_border.addItem(label, k)
        g.addWidget(QLabel("حدود القص"), r, 0)
        g.addWidget(self.combo_border, r, 1, 1, 2)
        r += 1

        self.combo_fit = QComboBox()
        for k, label in FIT_LABELS:
            self.combo_fit.addItem(label, k)
        g.addWidget(QLabel("ملاءمة الصورة بالمقاس"), r, 0)
        g.addWidget(self.combo_fit, r, 1, 1, 2)
        r += 1

        self.spin_ppr = QSpinBox()
        self.spin_ppr.setRange(0, 12)
        self.spin_ppr.setSpecialValueText("بدون حد")
        self.spin_ppr.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.spin_ppr.setToolTip("أقصى عدد صور شخصية بالصف الواحد (إذا ما اتسع أكثر من هالعدد ينزل للسطر الجديد تلقائياً)")
        g.addWidget(QLabel("الصور الشخصية بالصف"), r, 0)
        g.addWidget(self.spin_ppr, r, 1, 1, 2)
        r += 1

        self.chk_pair = QCheckBox("كل بطاقة (وجه + ظهر) تبدأ بسطر جديد")
        self.chk_rtl = QCheckBox("الترتيب من اليمين لليسار (الوجه جهة اليمين)")
        g.addWidget(self.chk_pair, r, 0, 1, 3)
        r += 1
        g.addWidget(self.chk_rtl, r, 0, 1, 3)
        r += 1
        g.setRowStretch(r, 1)

        for sig in (self.combo_paper.currentIndexChanged, self.combo_border.currentIndexChanged,
                    self.combo_fit.currentIndexChanged, self.spin_margin.valueChanged,
                    self.spin_corner.valueChanged, self.spin_ppr.valueChanged, self.sl_v.valueChanged, self.sl_h.valueChanged,
                    self.chk_pair.toggled, self.chk_rtl.toggled):
            sig.connect(self._layout_changed)
        return w

    def _make_gap_slider(self, grid, row, label):
        sl = QSlider(Qt.Orientation.Horizontal)
        sl.setRange(0, 200)
        sl.setValue(40)
        sl.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        lbl = QLabel("4.0 مم")
        lbl.setMinimumWidth(56)
        lbl.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        grid.addWidget(QLabel(label), row, 0)
        grid.addWidget(sl, row, 1)
        grid.addWidget(lbl, row, 2)
        return sl, lbl

    def _act(self, menu, text, fn, shortcut=None):
        a = QAction(text, self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        a.triggered.connect(lambda checked=False, f=fn: f())
        menu.addAction(a)
        return a

    def _build_menu(self):
        mb = self.menuBar()
        m = mb.addMenu("ملف")
        self._act(m, "مشروع جديد", self.new_project, "Ctrl+N")
        self._act(m, "فتح مشروع...", self.open_project_dialog, "Ctrl+O")
        self._act(m, "حفظ المشروع", self.save_project, "Ctrl+S")
        self._act(m, "حفظ المشروع باسم...", self.save_project_as, "Ctrl+Shift+S")
        m.addSeparator()
        self._act(m, "إضافة صور...", self.add_files_dialog, "Ctrl+I")
        self._act(m, "لصق صورة من الحافظة", lambda: self.paste_image(None), "Ctrl+V")
        m.addSeparator()
        self._act(m, "تصدير صورة...", self.export_image, "Ctrl+E")
        self._act(m, "تصدير PDF...", self.export_pdf, "Ctrl+Shift+E")
        self._act(m, "طباعة مباشرة...", self.print_direct, "Ctrl+P")
        m.addSeparator()
        self._act(m, "خروج", self.close, "Ctrl+Q")
        t = mb.addMenu("أدوات")
        self._act(t, "الإعدادات...", self.open_settings)
        h = mb.addMenu("مساعدة")
        self._act(h, "حول البرنامج", self.show_about)
        self._act(h, "فتح مجلد سجل الأخطاء", self.open_log_folder)

    # ------------------------------------------------------------------ الإعدادات
    def _load_prefs(self):
        p = self.prefs
        self._building = True
        self._reload_styles()
        self._set_layout_widgets({
            "paper": p.get("paper", "A4"), "margin": p.get("margin", 10.0),
            "gap_v": p.get("gap_v", 4.0), "gap_h": p.get("gap_h", 4.0),
            "corner": p.get("corner", 3.0), "border": p.get("border", "none"),
            "fit": p.get("fit", "stretch"), "photo_per_row": p.get("photo_per_row", 6), "pair_rows": p.get("pair_rows", True),
            "rtl": p.get("rtl", False)})
        self.chk_quick.setChecked(p.get("quick", True))
        self.chk_white.setChecked(p.get("auto_white", False))
        self.chk_ai.blockSignals(True)
        self.chk_ai.setChecked(False)
        self.chk_ai.blockSignals(False)
        i = self.combo_fmt.findText(p.get("fmt", "JPG"))
        self.combo_fmt.setCurrentIndex(max(0, i))
        j = self.combo_dpi.findData(p.get("dpi", DEFAULT_DPI))
        self.combo_dpi.setCurrentIndex(j if j >= 0 else 1)
        self._building = False
        self._update_info_label()

    def _set_layout_widgets(self, d):
        for w_ in (self.combo_paper, self.combo_border, self.combo_fit, self.spin_margin, self.spin_corner, self.spin_ppr,
                   self.sl_v, self.sl_h, self.chk_pair, self.chk_rtl):
            w_.blockSignals(True)
        o = core.merged_layout(d)
        self.combo_paper.setCurrentIndex(max(0, self.combo_paper.findData(o["paper"])))
        self.combo_border.setCurrentIndex(max(0, self.combo_border.findData(o["border"])))
        self.combo_fit.setCurrentIndex(max(0, self.combo_fit.findData(o["fit"])))
        self.spin_margin.setValue(float(o["margin"]))
        self.spin_ppr.setValue(int(o["photo_per_row"]))
        self.spin_corner.setValue(float(o["corner"]))
        self.sl_v.setValue(int(round(float(o["gap_v"]) * 10)))
        self.sl_h.setValue(int(round(float(o["gap_h"]) * 10)))
        self.chk_pair.setChecked(bool(o["pair_rows"]))
        self.chk_rtl.setChecked(bool(o["rtl"]))
        for w_ in (self.combo_paper, self.combo_border, self.combo_fit, self.spin_margin, self.spin_corner, self.spin_ppr,
                   self.sl_v, self.sl_h, self.chk_pair, self.chk_rtl):
            w_.blockSignals(False)
        self.lbl_v.setText("%.1f مم" % (self.sl_v.value() / 10.0))
        self.lbl_h.setText("%.1f مم" % (self.sl_h.value() / 10.0))

    def layout_opts(self):
        return {
            "paper": self.combo_paper.currentData() or "A4",
            "margin": self.spin_margin.value(),
            "gap_v": self.sl_v.value() / 10.0,
            "gap_h": self.sl_h.value() / 10.0,
            "corner": self.spin_corner.value(),
            "photo_per_row": self.spin_ppr.value(),
            "border": self.combo_border.currentData() or "none",
            "fit": self.combo_fit.currentData() or "stretch",
            "pair_rows": self.chk_pair.isChecked(),
            "rtl": self.chk_rtl.isChecked(),
        }

    def _layout_changed(self, *_):
        self.lbl_v.setText("%.1f مم" % (self.sl_v.value() / 10.0))
        self.lbl_h.setText("%.1f مم" % (self.sl_h.value() / 10.0))
        if self._building:
            return
        for k, v in self.layout_opts().items():
            self.prefs.set(k, v)
        self._update_info_label()
        self.refresh_preview()

    def _export_opts_changed(self, *_):
        if self._building:
            return
        self.prefs.set("fmt", self.combo_fmt.currentText())
        self.prefs.set("dpi", int(self.combo_dpi.currentData() or DEFAULT_DPI))
        self._update_info_label()

    def _update_info_label(self):
        pw, ph = core.paper_size(self.combo_paper.currentData() or "A4")
        self.lbl_info.setText("معاينة %s  •  %g × %g مم  •  جودة التصدير %s dpi"
                              % (self.combo_paper.currentData(), pw, ph, self.combo_dpi.currentData()))

    # ------------------------------------------------------------------ القوالب
    def _custom_templates(self):
        try:
            data = json.loads(self.prefs.get("templates", "[]"))
            return [(str(t[0]), float(t[1]), float(t[2]), bool(t[3]) if len(t) > 3 else False) for t in data]
        except (ValueError, TypeError, IndexError):
            return []

    def _reload_styles(self, select=None):
        self.combo_style.blockSignals(True)
        self.combo_style.clear()
        for name, w, h in core.PRESETS:
            self.combo_style.addItem("%s — %g×%g" % (name, w, h), (name, w, h, False, core.is_photo_name(name)))
        for name, w, h, single in self._custom_templates():
            self.combo_style.addItem("★ %s — %g×%g" % (name, w, h), (name, w, h, True, single))
        if select is not None:
            idx = self.combo_style.findText(select)
            if idx >= 0:
                self.combo_style.setCurrentIndex(idx)
        self.combo_style.blockSignals(False)

    def save_template(self):
        cur = self.combo_style.currentData()
        dlg = TemplateDialog(self, "", cur[1] if cur else 8.6, cur[2] if cur else 5.6, bool(cur[4]) if cur else False)
        if not _is_accepted(dlg.exec()):
            return
        name, w, h, single = dlg.values()
        if not name:
            info_box(self, "تنبيه", "اكتب اسماً للقالب.", warning=True)
            return
        items = [t for t in self._custom_templates() if t[0] != name]
        items.append((name, w, h, bool(single)))
        self.prefs.set("templates", json.dumps(items, ensure_ascii=False))
        self._reload_styles("★ %s — %g×%g" % (name, w, h))
        self.statusBar().showMessage("تم حفظ القالب «%s»" % name)

    def delete_template(self):
        cur = self.combo_style.currentData()
        if not cur or not cur[3]:
            info_box(self, "تنبيه", "اختر قالباً مخصصاً (عليه ★) لحذفه. القوالب الجاهزة ما تنحذف.")
            return
        items = [t for t in self._custom_templates() if t[0] != cur[0]]
        self.prefs.set("templates", json.dumps(items, ensure_ascii=False))
        self._reload_styles()
        self.statusBar().showMessage("تم حذف القالب «%s»" % cur[0])

    # ------------------------------------------------------------------ البطاقات
    def _mark_dirty(self):
        self.dirty = True

    def _on_changed(self):
        self._mark_dirty()
        self.refresh_preview()

    def add_card(self, card=None):
        if card is None:
            cur = self.combo_style.currentData() or (core.PRESETS[0][0], core.PRESETS[0][1], core.PRESETS[0][2], False, False)
            card = core.Card(cur[0], cur[1], cur[2], single=cur[4])
        row = CardRow(card, len(self.cards) + 1)
        row.changed.connect(self._on_changed)
        row.delete_requested.connect(self.delete_card)
        row.move_requested.connect(self.move_card)
        row.slot_clicked.connect(self.slot_clicked)
        row.slot_menu.connect(self.slot_menu)
        row.slot_dropped.connect(self.slot_dropped)
        row.rotate_requested.connect(self.rotate_side)
        row.whitebg_requested.connect(self.toggle_whitebg)
        self.cards.append(card)
        self.rows.append(row)
        self.cards_lay.insertWidget(self.cards_lay.count() - 1, row)
        if not self._building:
            self._on_changed()
        return row

    def _renumber(self):
        for n, r in enumerate(self.rows, 1):
            r.set_index(n)

    def delete_card(self, row):
        if row.card.has_any():
            if ask_buttons(self, "حذف البطاقة", "هذي البطاقة فيها صور. تريد تحذفها؟", ["حذف", "إلغاء"], "warning") != 0:
                return
        i = self.rows.index(row)
        self.rows.pop(i)
        self.cards.pop(i)
        row.setParent(None)
        row.deleteLater()
        if not self.rows:
            self.add_card()
        self._renumber()
        self._on_changed()

    def move_card(self, row, delta):
        i = self.rows.index(row)
        j = i + delta
        if j < 0 or j >= len(self.rows):
            return
        self.rows[i], self.rows[j] = self.rows[j], self.rows[i]
        self.cards[i], self.cards[j] = self.cards[j], self.cards[i]
        self.cards_lay.removeWidget(row)
        self.cards_lay.insertWidget(j, row)
        self._renumber()
        self._on_changed()

    # ------------------------------------------------------------------ الذكاء الاصطناعي (اختياري)
    def _ai_set_label(self, text):
        self.lbl_ai.setText(text)

    def _ai_toggled(self, on):
        if on:
            if getattr(self, "_ai_busy", False):
                return
            self._ai_cancel = False
            self._start_enable_ai(quiet=False)
        else:
            if getattr(self, "_ai_busy", False):
                # إلغاء التفعيل أثناء التحميل: نترك الخيط يكمّل ثم نطفئ عند الانتهاء
                self._ai_cancel = True
                self.prefs.set("ai", False)
                return
            core.set_ai(None)
            self.prefs.set("ai", False)
            self._ai_set_label("")
            self.statusBar().showMessage("تم إطفاء الذكاء الاصطناعي — الطريقة العادية.")

    def _start_enable_ai(self, quiet=False):
        """يشغّل enable_ai في خيط خلفي حتى لا تتجمّد الواجهة."""
        if getattr(self, "_ai_busy", False):
            return
        self._ai_busy = True
        self._ai_quiet = quiet
        self._ai_cancel = False
        self._ai_set_label("جاري تشغيل الذكاء الاصطناعي...")
        if not quiet:
            self.statusBar().showMessage("جاري تشغيل الذكاء الاصطناعي في الخلفية...")

        self._ai_thread = QThread(self)
        self._ai_worker = _AiEnableWorker()
        self._ai_worker.moveToThread(self._ai_thread)
        self._ai_thread.started.connect(self._ai_worker.run)
        self._ai_worker.finished.connect(self._on_ai_enable_done)
        self._ai_worker.finished.connect(self._ai_thread.quit)
        self._ai_worker.finished.connect(self._ai_worker.deleteLater)
        self._ai_thread.finished.connect(self._ai_thread.deleteLater)
        self._ai_thread.start()

    def _on_ai_enable_done(self, ok, msg):
        """يُستدعى على الخيط الرئيسي بعد انتهاء التحميل في الخلفية."""
        self._ai_busy = False
        quiet = getattr(self, "_ai_quiet", True)
        cancelled = getattr(self, "_ai_cancel", False)

        # المستخدم ألغى التفعيل أثناء التحميل
        if cancelled or (not quiet and not self.chk_ai.isChecked() and not self.prefs.get("ai", False)):
            if ok:
                core.set_ai(None)
            self.prefs.set("ai", False)
            self.chk_ai.blockSignals(True)
            self.chk_ai.setChecked(False)
            self.chk_ai.blockSignals(False)
            self._ai_set_label("")
            self.statusBar().showMessage("تم إطفاء الذكاء الاصطناعي — الطريقة العادية.")
            return

        if ok:
            self.prefs.set("ai", True)
            self.chk_ai.blockSignals(True)
            self.chk_ai.setChecked(True)
            self.chk_ai.blockSignals(False)
            self._ai_set_label("يعمل — " + msg)
            if not quiet:
                self.statusBar().showMessage(
                    "تم تشغيل الذكاء الاصطناعي. الصور الجديدة تستخدمه، وللقديمة اضغط عليها وأعد الاعتماد.")
        else:
            self.prefs.set("ai", False)
            self.chk_ai.blockSignals(True)
            self.chk_ai.setChecked(False)
            self.chk_ai.blockSignals(False)
            if quiet:
                self._ai_set_label("تعذر تشغيل الذكاء الاصطناعي: " + msg)
            else:
                self._ai_set_label("غير مفعّل — " + msg)
                info_box(self, "الذكاء الاصطناعي", msg, warning=True)

    def restore_ai(self):
        """عند فتح البرنامج: إذا كان مفعّلاً سابقاً نحاول تشغيله في الخلفية (بدون تجميد الواجهة)."""
        if not self.prefs.get("ai", False):
            return
        self._start_enable_ai(quiet=True)

    def _style_changed(self, *_):
        """إذا آخر بطاقة فارغة نبدّلها فوراً للمقاس المختار (حتى تنزل الصور الجديدة بالقالب الصحيح)."""
        if self._building or not self.rows or self.cards[-1].has_any():
            return
        row = self.rows.pop()
        self.cards.pop()
        row.setParent(None)
        row.deleteLater()
        self.add_card()
        self._renumber()

    def _clear_all(self):
        for r in self.rows:
            r.setParent(None)
            r.deleteLater()
        self.rows, self.cards = [], []

    # ------------------------------------------------------------------ إضافة وتعديل الصور
    def slot_clicked(self, row, side):
        if row.card.side(side).has_image():
            self.edit_side(row, side)
        else:
            self.pick_files(row, side)

    def slot_dropped(self, row, side, paths):
        self.add_images(paths, start=(row, side))

    def slot_menu(self, row, side, pos):
        s = row.card.side(side)
        menu = QMenu(self)
        if s.has_image():
            self._act(menu, "تعديل القص والألوان", lambda: self.edit_side(row, side))
        self._act(menu, "اختيار صورة من ملف...", lambda: self.pick_files(row, side))
        self._act(menu, "لصق صورة من الحافظة هنا", lambda: self.paste_image((row, side)))
        if s.has_image():
            menu.addSeparator()
            self._act(menu, "تدوير ↻ 90°", lambda: self.rotate_side(row, side, 1))
            self._act(menu, "تدوير ↺ 90°", lambda: self.rotate_side(row, side, -1))
            if row.card.single:
                self._act(menu, "إزالة الخلفية (أبيض) / إرجاعها", lambda: self.toggle_whitebg(row))
            self._act(menu, "إزالة الصورة", lambda: self.remove_side(row, side))
        if row.card.has_any() and not row.card.single:
            self._act(menu, "تبديل الوجه والظهر", lambda: self.swap_card(row))
        menu.exec(pos)

    def pick_files(self, row=None, side="face"):
        start = self.prefs.get("last_dir", "")
        paths, _f = QFileDialog.getOpenFileNames(self, "اختر صور الهوية", start, IMAGE_FILTER)
        if not paths:
            return
        self.prefs.set("last_dir", os.path.dirname(paths[0]))
        self.add_images(list(paths), start=(row, side) if row is not None else None)

    def add_files_dialog(self):
        self.pick_files(None)

    def paste_image(self, target):
        path = qimage_file_from_clipboard()
        paths = [path] if path else urls_to_image_paths(QApplication.clipboard().mimeData())
        if not paths:
            self.statusBar().showMessage("الحافظة ما فيها صورة.")
            return
        try:
            self.add_images(paths, start=target)
        finally:
            if path:   # ملف مؤقت أنشأناه لصورة الحافظة (add_images قرأت بياناته لذاكرتها أعلاه، ما عاد يحتاجه)
                try:
                    os.remove(path)
                except OSError:
                    pass

    def _next_empty_slot(self):
        cur = self.combo_style.currentData()
        kind = cur[0] if cur else None
        for row in self.rows:
            if kind is not None and row.card.kind != kind:
                continue
            for side in (("face",) if row.card.single else ("face", "back")):
                if not row.card.side(side).has_image():
                    return row, side, False
        row = self.add_card()
        return row, "face", True

    def add_images(self, paths, start=None):
        paths = [p for p in paths if p.lower().endswith(IMAGE_EXTS)]
        if not paths:
            self.statusBar().showMessage("ما لقيت صور صالحة.")
            return
        prog = None
        if len(paths) > 1:
            prog = QProgressDialog("جاري إضافة الصور...", "إلغاء", 0, len(paths), self)
            prog.setWindowTitle("إضافة صور")
            prog.setWindowModality(Qt.WindowModality.WindowModal)
            prog.setMinimumDuration(0)
            prog.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        added = 0
        target = start
        for i, p in enumerate(paths):
            if prog is not None:
                prog.setValue(i)
                prog.setLabelText("%s  (%d/%d)" % (os.path.basename(p), i + 1, len(paths)))
                QApplication.processEvents()
                if prog.wasCanceled():
                    break
            try:
                bgr = core.load_image(p)
            except Exception as e:  # noqa: BLE001
                info_box(self, "خطأ", "تعذر فتح الصورة:\n%s\n\n%s" % (p, e), warning=True)
                continue
            created = False
            if target is not None:
                row, side = target
                target = None
            else:
                row, side, created = self._next_empty_slot()
            if self._process_new(row, side, bgr):
                added += 1
            elif created and not row.card.has_any() and len(self.rows) > 1:
                self.delete_card(row)
        if prog is not None:
            prog.setValue(len(paths))
        if added:
            self.statusBar().showMessage("تمت إضافة %d صورة." % added)

    def _process_photo(self, row, side, bgr):
        card = row.card
        aspect = card.w / card.h
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            st, found = core.photo_state(bgr, aspect)
        finally:
            QApplication.restoreOverrideCursor()
        if self.chk_white.isChecked():
            st["adj"]["whitebg"] = True
        if self.chk_quick.isChecked() and found:
            state = st
        else:
            dlg = CropDialog(bgr, st, self, score=1.0 if found else 0.0, aspect=aspect)
            if not _is_accepted(dlg.exec()):
                return False
            state = dlg.result_state
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            card.side(side).set_image(bgr, state)
        finally:
            QApplication.restoreOverrideCursor()
        row.refresh()
        self._on_changed()
        return True

    def toggle_whitebg(self, row):
        s = row.card.face
        if not s.has_image():
            return
        st = core.normalize_state(s.state)
        st["adj"]["whitebg"] = not st["adj"].get("whitebg", False)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            s.state = st
            s.rerender()
        finally:
            QApplication.restoreOverrideCursor()
        row.refresh()
        self._on_changed()

    def _process_new(self, row, side, bgr):
        if row.card.single:
            return self._process_photo(row, side, bgr)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            st, score = core.auto_state(bgr)
        finally:
            QApplication.restoreOverrideCursor()
        if self.chk_quick.isChecked() and st["quad"] is not None and score >= QUICK_THRESHOLD:
            state = st
        else:
            dlg = CropDialog(bgr, st, self, score=score)
            if not _is_accepted(dlg.exec()):
                return False
            state = dlg.result_state
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            row.card.side(side).set_image(bgr, state)
        finally:
            QApplication.restoreOverrideCursor()
        row.refresh()
        self._on_changed()
        return True

    def edit_side(self, row, side):
        s = row.card.side(side)
        if not s.has_image():
            return
        orig = s.orig()
        dlg = CropDialog(orig, s.state, self, aspect=(row.card.w / row.card.h) if row.card.single else None)
        if not _is_accepted(dlg.exec()):
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            s.state = dlg.result_state
            s.rerender()
        finally:
            QApplication.restoreOverrideCursor()
        row.refresh()
        self._on_changed()

    def rotate_side(self, row, side, delta):
        s = row.card.side(side)
        if not s.has_image():
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            s.rotate_quick(delta)
        finally:
            QApplication.restoreOverrideCursor()
        row.refresh()
        self._on_changed()

    def remove_side(self, row, side):
        row.card.side(side).clear()
        row.refresh()
        self._on_changed()

    def swap_card(self, row):
        row.card.swap_sides()
        row.refresh()
        self._on_changed()

    # ------------------------------------------------------------------ سحب وإفلات
    def dragEnterEvent(self, e):
        md = e.mimeData()
        if md.hasUrls() and any(u.toLocalFile().lower().endswith(IMAGE_EXTS + (core.PROJECT_EXT,)) for u in md.urls()):
            e.acceptProposedAction()

    def dropEvent(self, e):
        md = e.mimeData()
        files = [u.toLocalFile() for u in md.urls()] if md.hasUrls() else []
        proj = [f for f in files if f.lower().endswith(core.PROJECT_EXT)]
        e.acceptProposedAction()
        if proj:
            self.open_project(proj[0])
        else:
            self.add_images(files)

    # ------------------------------------------------------------------ المعاينة
    def refresh_preview(self):
        self._preview_timer.start(25)

    def _render_preview(self):
        try:
            self.pages = core.compose(self.cards, self.layout_opts(), PREVIEW_DPI, preview=True)
        except Exception:  # noqa: BLE001
            log_error(traceback.format_exc())
            return
        self.page_idx = min(self.page_idx, len(self.pages) - 1)
        self.show_page()

    def show_page(self):
        n = len(self.pages)
        if n == 0:
            return
        self.lbl_page.setText("الصفحة %d / %d" % (self.page_idx + 1, n))
        self.btn_prev.setEnabled(self.page_idx > 0)
        self.btn_next.setEnabled(self.page_idx < n - 1)
        self.preview.set_source(QPixmap.fromImage(pil_to_qimage(self.pages[self.page_idx])))
        pw, ph, placements = self._current_placements()
        self.preview.set_layout(pw, ph, placements)

    def _current_placements(self):
        """هندسة عناصر صفحة المعاينة الحالية (بالمليمتر) — بنفس حساب core.compose تماماً،
        حتى تتطابق مواضع السحب والإفلات مع الصورة المعروضة فعلياً."""
        o = core.merged_layout(self.layout_opts())
        pw, ph = core.paper_size(o["paper"])
        items = core.build_items(self.cards, o, include_empty=True)
        pages = core.layout_pages(items, pw, ph, o)
        if not pages:
            return pw, ph, []
        idx = min(self.page_idx, len(pages) - 1)
        return pw, ph, pages[idx]

    def _swap_preview_items(self, item_a, item_b):
        """يبدّل محتوى صورتين (وجه/ظهر) ضمن المعاينة — بدون أي مساس بمقاس أي بطاقة أو ترتيب البطاقات نفسها."""
        if item_a is item_b or item_a.card >= len(self.cards) or item_b.card >= len(self.cards):
            return
        side_a = self.cards[item_a.card].side(item_a.side)
        side_b = self.cards[item_b.card].side(item_b.side)
        (side_a.orig_bytes, side_a.state, side_a.result, side_a.uid,
         side_b.orig_bytes, side_b.state, side_b.result, side_b.uid) = (
         side_b.orig_bytes, side_b.state, side_b.result, side_b.uid,
         side_a.orig_bytes, side_a.state, side_a.result, side_a.uid)
        if item_a.card < len(self.rows):
            self.rows[item_a.card].refresh()
        if item_b.card < len(self.rows):
            self.rows[item_b.card].refresh()
        self.dirty = True
        self.refresh_preview()

    def go_page(self, d):
        if not self.pages:
            return
        self.page_idx = min(max(self.page_idx + d, 0), len(self.pages) - 1)
        self.show_page()

    # ------------------------------------------------------------------ المشاريع
    def _has_images(self):
        return any(c.has_any() for c in self.cards)

    def _confirm_discard(self):
        if not (self._has_images() and self.dirty):
            return True
        r = ask_buttons(self, "تغييرات غير محفوظة", "عندك تغييرات ما انحفظت. شتريد؟",
                        ["حفظ المشروع", "متابعة بدون حفظ", "إلغاء"])
        if r == 0:
            return self.save_project()
        return r == 1

    def new_project(self):
        if not self._confirm_discard():
            return
        self._clear_all()
        self.project_path = None
        self.add_card()
        self.dirty = False
        self._on_changed()
        self.dirty = False

    def open_project_dialog(self):
        path, _f = QFileDialog.getOpenFileName(self, "فتح مشروع", self.prefs.get("last_dir", ""), PROJECT_FILTER)
        if path:
            self.open_project(path)

    def open_project(self, path):
        if not self._confirm_discard():
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            cards, layout = core.load_project(path)
        except Exception as e:  # noqa: BLE001
            QApplication.restoreOverrideCursor()
            log_error(traceback.format_exc())
            info_box(self, "خطأ", "تعذر فتح المشروع:\n%s" % e, warning=True)
            return
        QApplication.restoreOverrideCursor()
        self._set_cards(cards, layout)
        self.project_path = path
        self.prefs.set("last_dir", os.path.dirname(path))
        self.statusBar().showMessage("تم فتح المشروع: %s" % path)

    def _set_cards(self, cards, layout):
        self._building = True
        self._clear_all()
        if layout:
            self._set_layout_widgets(layout)
        for c in cards:
            self.add_card(card=c)
        if not self.rows:
            self.add_card()
        self._building = False
        self._update_info_label()
        self._on_changed()
        self.dirty = False

    def save_project(self):
        if not self.project_path:
            return self.save_project_as()
        return self._write_project(self.project_path)

    def save_project_as(self):
        default = os.path.join(self.prefs.get("last_dir", ""), "IQ_%s%s" % (time.strftime("%Y%m%d_%H%M"), core.PROJECT_EXT))
        path, _f = QFileDialog.getSaveFileName(self, "حفظ المشروع", default, PROJECT_FILTER)
        if not path:
            return False
        if not path.lower().endswith(core.PROJECT_EXT):
            path += core.PROJECT_EXT
        return self._write_project(path)

    def _write_project(self, path):
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            core.save_project(path, self.cards, self.layout_opts())
        except Exception as e:  # noqa: BLE001
            QApplication.restoreOverrideCursor()
            log_error(traceback.format_exc())
            info_box(self, "خطأ", "تعذر حفظ المشروع:\n%s" % e, warning=True)
            return False
        QApplication.restoreOverrideCursor()
        self.project_path = path
        self.dirty = False
        self.prefs.set("last_dir", os.path.dirname(path))
        self.statusBar().showMessage("تم حفظ المشروع: %s" % path)
        return True

    # ------------------------------------------------------------------ التصدير والطباعة
    def _final_pages(self):
        if not self._has_images():
            info_box(self, "تنبيه", "ما اكو أي صورة مضافة للبطاقات بعد.")
            return None
        dpi = int(self.combo_dpi.currentData() or DEFAULT_DPI)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            return core.compose(self.cards, self.layout_opts(), dpi, preview=False)
        except MemoryError:
            QApplication.restoreOverrideCursor()
            info_box(self, "الذاكرة ما تكفي", "اخفض الدقة إلى 300 أو 150 dpi وجرّب مرة ثانية.", warning=True)
            return None
        finally:
            QApplication.restoreOverrideCursor()

    def _after_export(self, paths):
        self.dirty = False
        self.statusBar().showMessage("تم التصدير: %s" % paths[0])
        self.prefs.set("last_dir", os.path.dirname(paths[0]))
        if self.prefs.get("open_after", False):
            QDesktopServices.openUrl(QUrl.fromLocalFile(paths[0]))

    def _default_name(self, ext):
        return os.path.join(self.prefs.get("last_dir", ""), "IQ_%s%s" % (time.strftime("%Y%m%d_%H%M%S"), ext))

    def export_image(self):
        pages = self._final_pages()
        if not pages:
            return
        fmt = self.combo_fmt.currentText()
        ext = ".png" if fmt == "PNG" else ".jpg"
        path, _f = QFileDialog.getSaveFileName(self, "تصدير صورة", self._default_name(ext), "%s (*%s)" % (fmt, ext))
        if not path:
            return
        dpi = int(self.combo_dpi.currentData() or DEFAULT_DPI)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            written = core.export_images(pages, path, fmt, dpi, self.prefs.get("jpg_q", 95))
        except Exception as e:  # noqa: BLE001
            QApplication.restoreOverrideCursor()
            log_error(traceback.format_exc())
            info_box(self, "خطأ", "تعذر التصدير:\n%s" % e, warning=True)
            return
        QApplication.restoreOverrideCursor()
        self._after_export(written)

    def export_pdf(self):
        pages = self._final_pages()
        if not pages:
            return
        path, _f = QFileDialog.getSaveFileName(self, "تصدير PDF", self._default_name(".pdf"), "PDF (*.pdf)")
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        pw, ph = core.paper_size(self.combo_paper.currentData() or "A4")
        dpi = int(self.combo_dpi.currentData() or DEFAULT_DPI)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            core.write_pdf(path, pages, pw, ph, dpi, self.prefs.get("jpg_q", 95), "IQ Cards")
        except Exception as e:  # noqa: BLE001
            QApplication.restoreOverrideCursor()
            log_error(traceback.format_exc())
            info_box(self, "خطأ", "تعذر التصدير:\n%s" % e, warning=True)
            return
        QApplication.restoreOverrideCursor()
        self._after_export([path])

    def _qpagesize(self):
        name = self.combo_paper.currentData() or "A4"
        ids = {"A4": QPageSize.PageSizeId.A4, "A5": QPageSize.PageSizeId.A5, "Letter": QPageSize.PageSizeId.Letter}
        if name in ids:
            return QPageSize(ids[name])
        w, h = core.paper_size(name)
        return QPageSize(QSizeF(w, h), QPageSize.Unit.Millimeter)

    def print_direct(self):
        pages = self._final_pages()
        if not pages:
            return
        printer = QPrinter(QPrinter.PrinterMode.HighResolution)
        printer.setPageSize(self._qpagesize())
        printer.setFullPage(True)
        dlg = QPrintDialog(printer, self)
        if not _is_accepted(dlg.exec()):
            return
        printer.setFullPage(True)
        painter = QPainter()
        if not painter.begin(printer):
            info_box(self, "خطأ", "تعذر بدء الطباعة.", warning=True)
            return
        try:
            for i, pg in enumerate(pages):
                if i:
                    printer.newPage()
                painter.drawImage(painter.viewport(), pil_to_qimage(pg))
        finally:
            painter.end()
        self.dirty = False
        self.statusBar().showMessage("تم إرسال الطباعة")

    # ------------------------------------------------------------------ أخرى
    def open_settings(self):
        SettingsDialog(self.prefs, self).exec()

    def show_about(self):
        info_box(self, "حول البرنامج",
                 "%s\nالإصدار %s\n\nقص الهويات والمستمسكات وتركيبها تلقائياً على الورق.\n"
                 "المعالجة تتم بالكامل على جهازك — لا يتم رفع أي صورة لأي مكان.\n\n"
                 "واجهة: %s" % (APP_NAME, core.VERSION, QT_API))

    def open_log_folder(self):
        ensure_appdata()
        QDesktopServices.openUrl(QUrl.fromLocalFile(APPDATA))

    def maybe_restore_autosave(self):
        if not self.prefs.get("autosave", False) or not os.path.exists(AUTOSAVE_PATH):
            return
        if ask_buttons(self, "استرجاع الجلسة", "لقيت جلسة محفوظة من المرة السابقة. تريد تسترجعها؟", ["استرجاع", "لا"]) != 0:
            return
        try:
            cards, layout = core.load_project(AUTOSAVE_PATH)
        except Exception:  # noqa: BLE001
            log_error(traceback.format_exc())
            return
        self._set_cards(cards, layout)

    def closeEvent(self, e):
        autosave = self.prefs.get("autosave", False)
        if autosave:
            ensure_appdata()
            try:
                if self._has_images():
                    core.save_project(AUTOSAVE_PATH, self.cards, self.layout_opts())
                elif os.path.exists(AUTOSAVE_PATH):
                    os.remove(AUTOSAVE_PATH)
            except Exception:  # noqa: BLE001
                log_error(traceback.format_exc())
        elif self._has_images() and self.dirty:
            r = ask_buttons(self, "قبل الخروج", "عندك صور ما انحفظت بمشروع. شتريد؟",
                            ["حفظ المشروع", "خروج بدون حفظ", "إلغاء"])
            if r == 0:
                if not self.save_project():
                    e.ignore()
                    return
            elif r != 1:
                e.ignore()
                return
        try:
            self.prefs.q.setValue("geometry", self.saveGeometry())
        except Exception:  # noqa: BLE001
            pass
        e.accept()


# ----------------------------------------------------------------------------
# التشغيل
# ----------------------------------------------------------------------------
def install_excepthook():
    def hook(exc_type, exc, tb):
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        log_error(text)
        try:
            if QApplication.instance() is not None:
                info_box(None, "خطأ غير متوقع",
                         "حدث خطأ غير متوقع:\n%s\n\nتم حفظ التفاصيل بالملف:\n%s" % (exc, LOG_PATH), warning=True)
        except Exception:  # noqa: BLE001
            pass
    sys.excepthook = hook


def _ai_selftest_main():
    """وضع الفحص الذاتي: يُشغَّل بعملية منفصلة. رمز الخروج 0 = المعالج والنماذج تعمل."""
    try:
        import iq_ai
        sys.exit(0 if iq_ai.selftest() else 3)
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        sys.exit(4)


class _AiEnableWorker(QObject):
    """يشغّل enable_ai في خيط خلفي ويرسل النتيجة عبر إشارة (آمن للواجهة)."""
    finished = Signal(bool, str)

    def run(self):
        try:
            ok, msg = enable_ai(None, quiet=True)
        except Exception as e:  # noqa: BLE001
            log_error(traceback.format_exc())
            ok, msg = False, str(e)
        self.finished.emit(ok, msg)


def enable_ai(parent=None, quiet=False):
    """يشغّل ميزة الذكاء الاصطناعي بعد فحص ذاتي بعملية منفصلة. يرجع (ok, رسالة).
    quiet=True: بدون مؤشر انتظار (مناسب للاستدعاء من خيط خلفي أو عند الإقلاع)."""
    import subprocess
    try:
        import iq_ai
    except Exception as e:  # noqa: BLE001
        return False, "وحدة الذكاء الاصطناعي غير موجودة (%s)." % e
    missing = iq_ai.missing_models_text()
    if iq_ai.find_model(iq_ai.BG_MODELS) is None:
        return False, ("ملفات النماذج غير موجودة. شغّل get_models.bat (يحمّلها مرة وحدة) أو ضع الملفات داخل مجلد models بجانب البرنامج.\n"
                       "المطلوب: " + "، ".join(missing))
    cmd = [sys.executable, "--ai-selftest"] if getattr(sys, "frozen", False) else [sys.executable, os.path.abspath(__file__), "--ai-selftest"]
    if not quiet:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        rc = subprocess.call(cmd, timeout=120, creationflags=flags)
    except Exception as e:  # noqa: BLE001
        rc = -1
        log_error("ai selftest failed to run: %s" % e)
    finally:
        if not quiet:
            QApplication.restoreOverrideCursor()
    if rc != 0:
        log_error("ai selftest rc=%s" % rc)
        return False, ("هذا الجهاز ما يدعم تشغيل الذكاء الاصطناعي (فشل الفحص الذاتي، الرمز %s). "
                       "البرنامج يكمل بالطريقة العادية بدون مشكلة." % rc)
    ok, msg = iq_ai.load()
    if not ok:
        return False, "تعذر تحميل النماذج: " + msg
    core.set_ai(iq_ai)
    extra = ""
    if missing:
        extra = "\n(ينقصك: %s — التعرف على الوجه يستعمل الطريقة العادية لحد ما تنزّله)" % "، ".join(missing)
    return True, msg + extra


def main():
    if "--ai-selftest" in sys.argv:
        _ai_selftest_main()
    ensure_appdata()
    install_excepthook()
    if QT_API == "PyQt5":
        try:
            QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
        except Exception:  # noqa: BLE001
            pass
    try:   # نفس سياسة تقريب DPI على PySide6 وPyQt5 حتى يتطابق حجم الواجهة على كل النسخ
        QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    except Exception:  # noqa: BLE001
        pass
    app = QApplication(sys.argv)
    app.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(STYLE)
    app.setWindowIcon(make_app_icon())
    prefs = Prefs()
    win = MainWindow(prefs)
    win.show()
    win.restore_ai()
    win.maybe_restore_autosave()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
