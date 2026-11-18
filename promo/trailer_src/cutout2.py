#!/usr/bin/env python3
"""Hand-traced car outlines -> RGBA car layers + clean backgrounds with cars removed."""
import cv2, numpy as np, json, os
os.makedirs("cars", exist_ok=True)
S = 0.6
# outlines traced on the gridded crops (crop-image coords), crop origin in plate coords
OUT = {
    "amg": ("p4", 100, 460, [(40,170),(70,105),(110,95),(120,60),(180,38),(330,30),(450,42),(515,105),(600,140),(660,170),(675,215),(682,300),(665,330),(600,350),(440,345),(330,358),(200,365),(185,390),(140,390),(95,370),(45,330),(30,250)]),
    "charger": ("p4", 1100, 500, [(80,120),(95,90),(120,60),(180,40),(330,32),(420,40),(470,70),(560,100),(640,130),(660,160),(655,220),(640,270),(560,275),(420,270),(310,290),(250,300),(200,300),(150,265),(110,230),(90,180)]),
    "vaz": ("p4", 2150, 540, [(20,180),(30,90),(70,60),(150,40),(330,35),(390,75),(395,120),(520,125),(565,130),(570,185),(560,200),(500,215),(330,215),(310,240),(265,250),(230,225),(60,225),(40,245),(30,240)]),
    "bmw": ("p1", 260, 540, [(30,240),(30,170),(60,110),(130,60),(200,35),(260,30),(380,40),(440,95),(470,140),(560,170),(700,190),(850,210),(915,235),(935,270),(940,330),(915,370),(830,385),(640,400),(560,395),(450,370),(330,350),(300,390),(260,395),(240,350),(150,300),(60,290)]),
    "bmwr": ("p5", 260, 250, [(25,300),(40,250),(70,200),(150,135),(230,110),(350,40),(450,30),(700,28),(830,45),(880,70),(950,140),(1000,165),(1040,200),(1050,250),(1055,350),(1040,430),(1000,500),(40,500),(20,400)]),
}
meta = {}
masks = {}
for name, (pl, x0, y0, pts) in OUT.items():
    img = cv2.imread(f"plates/{pl}.png"); H, W = img.shape[:2]
    poly = np.array([[x / S + x0, y / S + y0] for x, y in pts], np.int32)
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [poly], 255, cv2.LINE_AA)
    m = cv2.GaussianBlur(m, (0, 0), 3.0)
    masks.setdefault(pl, []).append(m)
    ys, xs = np.where(m > 4)
    bx0, by0, bx1, by1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    cv2.imwrite(f"cars/{name}.png", np.dstack([img[by0:by1, bx0:bx1], m[by0:by1, bx0:bx1]]))
    meta[name] = {"plate": pl, "box": [bx0, by0, bx1, by1]}
for pl, ms in masks.items():
    img = cv2.imread(f"plates/{pl}.png"); H, W = img.shape[:2]
    allm = np.max(np.stack(ms), 0)
    dm = cv2.dilate((allm > 10).astype(np.uint8) * 255, np.ones((31, 31), np.uint8))
    s_img = cv2.resize(img, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    s_m = cv2.resize(dm, (W // 4, H // 4), interpolation=cv2.INTER_NEAREST)
    s_bg = cv2.inpaint(s_img, s_m, 9, cv2.INPAINT_TELEA)
    s_bg = cv2.GaussianBlur(s_bg, (0, 0), 1.2)
    bg = cv2.resize(s_bg, (W, H), interpolation=cv2.INTER_CUBIC)
    a = (cv2.GaussianBlur(dm, (0, 0), 8).astype(np.float32) / 255)[..., None]
    cv2.imwrite(f"cars/{pl}_bg.png", (img * (1 - a) + bg * a).astype(np.uint8))
json.dump(meta, open("cars/meta.json", "w"), indent=1)
print(meta)
