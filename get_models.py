# -*- coding: utf-8 -*-
# Downloads the optional AI models into the "models" folder next to this file (needs internet once).
# Usage:  python get_models.py            -> small set (recommended)
#         python get_models.py --best     -> also the larger, more accurate person model (176 MB)
import os, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, "models")
GH = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/"
ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
FILES = [
    ("u2netp.onnx", [GH + "u2netp.onnx"], True),
    ("face_detection_yunet_2023mar.onnx", [ZOO + "face_detection_yunet_2023mar.onnx"], False),
    ("face_detection_yunet_2022mar.onnx", [ZOO + "face_detection_yunet_2022mar.onnx"], False),
]
if "--best" in sys.argv:
    FILES.append(("u2net_human_seg.onnx", [GH + "u2net_human_seg.onnx"], False))

os.makedirs(DEST, exist_ok=True)
for name, urls, required in FILES:
    path = os.path.join(DEST, name)
    if os.path.isfile(path) and os.path.getsize(path) > 100000:
        print("already have", name); continue
    ok = False
    for u in urls:
        try:
            print("downloading", name, "...")
            req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(path + ".part", "wb") as f:
                while True:
                    b = r.read(1 << 16)
                    if not b: break
                    f.write(b)
            if os.path.getsize(path + ".part") < 100000:
                raise IOError("file too small")
            os.replace(path + ".part", path); ok = True; print("  ok", os.path.getsize(path) // 1024, "KB"); break
        except Exception as e:
            print("  failed:", e)
    if not ok and required:
        print("ERROR: could not download the required model", name)
print("\nDone. Models folder:", DEST)
