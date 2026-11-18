#!/usr/bin/env python3
"""Cut realistic plates out of the design reference, remove baked-in text, upscale."""
import cv2, numpy as np, os
os.makedirs("plates", exist_ok=True)
ref = cv2.imread("ref.png")

def text_mask(img, boxes, sat_purple=True):
    """Mask of bright-white / saturated-purple text pixels inside boxes, dilated."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = np.zeros(img.shape[:2], np.uint8)
    for x0, y0, x1, y1 in boxes:
        sub = hsv[y0:y1, x0:x1]
        h, s, v = sub[..., 0], sub[..., 1], sub[..., 2]
        white = (v > 175) & (s < 70)
        purple = (h > 120) & (h < 160) & (s > 90) & (v > 90)
        mm = (white | purple).astype(np.uint8) * 255
        m[y0:y1, x0:x1] = np.maximum(m[y0:y1, x0:x1], mm)
    m = cv2.dilate(m, np.ones((7, 7), np.uint8), iterations=2)
    return m

def up(img, k=4):
    big = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_LANCZOS4)
    # gentle detail recovery
    blur = cv2.GaussianBlur(big, (0, 0), 2.2)
    big = cv2.addWeighted(big, 1.55, blur, -0.55, 0)
    big = cv2.bilateralFilter(big, 5, 25, 5)
    return big

# Panel 1 – drifting white BMW, remove title
p1 = ref[0:352, 0:833].copy()
m1 = text_mask(p1, [(255, 35, 790, 205)])
p1 = cv2.inpaint(p1, m1, 9, cv2.INPAINT_TELEA)
cv2.imwrite("plates/p1.png", up(p1))

# Panel 4 – autosalon with AMG / Charger / VAZ, remove title + caption
p4 = ref[358:665, 838:1672].copy()
m4 = text_mask(p4, [(70, 8, 475, 122), (225, 242, 620, 295)])
p4 = cv2.inpaint(p4, m4, 9, cv2.INPAINT_TELEA)
cv2.imwrite("plates/p4.png", up(p4))

# Panel 5 – convoy, left part without the logo
p5 = ref[670:941, 0:640].copy()
cv2.imwrite("plates/p5.png", up(p5))

# Panel 2 – winter city + tree background for the advent scene
p2 = ref[0:250, 1330:1672].copy()
m2 = text_mask(p2, [(0, 0, 40, 250)])
cv2.imwrite("plates/p2.png", up(p2, 3))

# Panel 3 – BMW thumbnail is too small; reuse other plates for cards.
print("plates:", {f: cv2.imread("plates/" + f).shape for f in os.listdir("plates")})
