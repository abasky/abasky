# -*- coding: utf-8 -*-
# Finds which library crashes on this PC. Each import runs in its own process so a crash cannot hide the others.
import subprocess, sys, platform, struct
out = []
def log(s):
    print(s); out.append(s)
log("Python %s %d-bit | %s" % (sys.version.split()[0], struct.calcsize("P") * 8, platform.platform()))
try:
    log("CPU: " + subprocess.check_output("wmic cpu get name", shell=True).decode("utf-8", "ignore").split("\n")[1].strip())
except Exception as e:
    log("CPU: ?")
tests = [
    ("numpy", "import numpy; print(numpy.__version__)"),
    ("numpy math", "import numpy as n; a=n.arange(1000.0).reshape(10,100); print((a@a.T).sum(), n.fft.fft(a).shape)"),
    ("PIL", "import PIL; from PIL import Image; print(PIL.__version__)"),
    ("cv2", "import cv2; print(cv2.__version__)"),
    ("cv2 work", "import cv2,numpy as n; i=n.zeros((200,200,3),'uint8'); cv2.GaussianBlur(i,(5,5),0); cv2.cvtColor(i,cv2.COLOR_BGR2LAB); cv2.grabCut(i,n.zeros((200,200),'uint8'),(10,10,100,100),n.zeros((1,65)),n.zeros((1,65)),1,cv2.GC_INIT_WITH_RECT); print('ok')"),
    ("PyQt5", "from PyQt5 import QtCore, QtGui, QtWidgets; print(QtCore.QT_VERSION_STR)"),
]
for name, code in tests:
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    o = p.communicate()[0].decode("utf-8", "ignore").strip().splitlines()
    rc = p.returncode
    tag = "OK   " if rc == 0 else "FAIL "
    log("%s%-10s rc=%s  %s" % (tag, name, hex(rc & 0xFFFFFFFF), (o[-1] if o else "")))
open("diag_result.txt", "w").write("\n".join(out))
log("\nSaved to diag_result.txt - send me this file or a photo of this screen.")
