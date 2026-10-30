# -*- coding: utf-8 -*-

"""
Abasky model downloader.

Usage:
    python get_models.py
    python get_models.py --best
"""

import os
import sys
import urllib.request


HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, "models")

GH_REMBG = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/"
GH_MODNET = "https://github.com/yakhyo/modnet/releases/download/weights/"
GH_YUNET = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"

FILES = [
    (
        "modnet_photographic.onnx",
        [GH_MODNET + "modnet_photographic.onnx"],
        False,
    ),
    (
        "u2netp.onnx",
        [GH_REMBG + "u2netp.onnx"],
        True,
    ),
    (
        "face_detection_yunet_2023mar.onnx",
        [GH_YUNET + "face_detection_yunet_2023mar.onnx"],
        False,
    ),
    (
        "face_detection_yunet_2022mar.onnx",
        [GH_YUNET + "face_detection_yunet_2022mar.onnx"],
        False,
    ),
]

if "--best" in sys.argv:
    FILES.append(
        (
            "u2net_human_seg.onnx",
            [GH_REMBG + "u2net_human_seg.onnx"],
            False,
        )
    )


def download_file(name, urls, required):
    path = os.path.join(DEST, name)
    temp_path = path + ".part"

    if os.path.isfile(path) and os.path.getsize(path) > 100000:
        print("[OK] موجود مسبقاً:", name)
        return True

    for url in urls:
        try:
            print("[DOWNLOAD]", name)

            request = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 Abasky"},
            )

            with urllib.request.urlopen(request, timeout=180) as response:
                with open(temp_path, "wb") as output:
                    while True:
                        block = response.read(256 * 1024)

                        if not block:
                            break

                        output.write(block)

            size = os.path.getsize(temp_path)

            if size < 100000:
                raise IOError("حجم الملف أصغر من المتوقع.")

            os.replace(temp_path, path)

            print(
                "[OK] %s — %.2f MB"
                % (name, size / (1024 * 1024))
            )

            return True

        except Exception as exc:
            print("[FAILED]", name, "-", exc)

            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError:
                pass

    if required:
        print("[ERROR] فشل تنزيل النموذج المطلوب:", name)

    return False


def main():
    os.makedirs(DEST, exist_ok=True)

    print("=" * 55)
    print("ABASKY OFFLINE AI MODEL DOWNLOADER")
    print("=" * 55)
    print("Models folder:", DEST)
    print()

    failed = []

    for name, urls, required in FILES:
        if not download_file(name, urls, required):
            failed.append(name)

    print()
    print("=" * 55)

    if failed:
        print("بعض الملفات لم يتم تنزيلها:")
        for name in failed:
            print(" -", name)
    else:
        print("اكتمل تنزيل النماذج.")

    print()
    print("MODNet يحتاج onnxruntime مثبتاً.")
    print("نفّذ: pip install onnxruntime")
    print("=" * 55)


if __name__ == "__main__":
    main()
