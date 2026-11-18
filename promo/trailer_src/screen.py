#!/usr/bin/env python3
"""Landscape 'desktop player' screen content (1728x1080) built around the vertical trailer, + per-frame avg colour."""
import os, sys, json, math, subprocess, wave
import numpy as np, cv2
from multiprocessing import Pool
from PIL import Image, ImageDraw, ImageFont, ImageFilter

SW, SH = 1728, 1080
FPS = 30
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "screen"), exist_ok=True)
FD = os.path.join(HERE, "fonts")
def F(n, s): return ImageFont.truetype(os.path.join(FD, n), s)
P = (140, 56, 255); PL = (196, 140, 255)

# design-time knots (same as trailer_music)
_BT = np.load(os.path.join(HERE, "beats.npy")); _i0 = int(np.argmin(np.abs(_BT - 44.54))) - 2
REL = _BT[_i0:_i0 + 60] - _BT[_i0]
KR = [float(REL[k]) for k in [0, 2, 6, 14, 20, 22, 24, 26, 30, 36]]
KD = [0.0, 0.45, 3.0, 7.0, 11.0, 12.0, 13.0, 14.0, 16.0, 20.0]
def dtime(rt): return float(np.interp(rt, KR, KD))

SCENES = [(0.0, "01", "ЗИМА НА", "STATE RP", "Обновление 7 декабря"),
          (3.0, "02", "АДВЕНТ-", "КАЛЕНДАРЬ", "24 дня наград"),
          (7.0, "03", "НОВЫЙ", "БАТЛПАСС", "Качай уровни и забирай награды"),
          (11.0, "04", "НОВЫЕ", "ТАЧКИ", "ВАЗ · AMG · Charger"),
          (16.0, "05", "ЗАЛЕТАЙ", "НА СЕРВЕР", "Зимнее обновление · 7 декабря")]

# audio spectrum envelope per frame (bass-reactive visualiser)
def audio_bands():
    with wave.open(os.path.join(HERE, "audio_music.wav")) as w:
        sr = w.getframerate(); a = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, 2).mean(1) / 32768
    nf = int(len(a) / sr * FPS); hop = sr // FPS; nb = 24
    out = np.zeros((nf, nb))
    edges = np.geomspace(30, 12000, nb + 1)
    for i in range(nf):
        seg = a[max(0, i * hop - 1024): i * hop + 1024]
        if len(seg) < 256: continue
        sp = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        fr = np.fft.rfftfreq(len(seg), 1 / sr)
        for b in range(nb):
            m = (fr >= edges[b]) & (fr < edges[b + 1])
            out[i, b] = sp[m].mean() if m.any() else 0
    out = np.log1p(out * 4)
    lo = np.percentile(out, 20, axis=0, keepdims=True); hi = np.percentile(out, 99, axis=0, keepdims=True)
    out = np.clip((out - lo) / (hi - lo + 1e-6), 0, 1) ** 1.6
    sm = out.copy()
    for i in range(1, nf): sm[i] = np.maximum(out[i], sm[i - 1] * 0.7)
    return np.clip(sm, 0, 1)
BANDS = audio_bands()

cap = None
def frames():
    c = cv2.VideoCapture(os.path.join(HERE, "trailer_music_noaudio.mp4"))
    i = 0
    while True:
        ok, fr = c.read()
        if not ok: break
        yield i, fr
        i += 1

def ease(x): x = max(0, min(1, x)); return 1 - (1 - x) ** 3

def build(args):
    i, fr = args
    rt = i / FPS; td = dtime(rt)
    vid = fr[:, :, ::-1].astype(np.float32) / 255
    # background: blurred cover of the trailer
    bg = cv2.resize(vid, (SW // 8, int(SW // 8 * 1920 / 1080)), interpolation=cv2.INTER_AREA)
    oy = (bg.shape[0] - SH // 8) // 2
    bg = bg[oy:oy + SH // 8]
    bg = cv2.GaussianBlur(bg, (0, 0), 6)
    bg = cv2.resize(bg, (SW, SH)) * 0.38 + np.array([0.03, 0.0, 0.06], np.float32)
    img = Image.fromarray((np.clip(bg, 0, 1) * 255).astype(np.uint8)).convert("RGBA")
    d = ImageDraw.Draw(img)
    # centre player with glow
    ph = 980; pw = int(ph * 1080 / 1920)
    px, py = SW // 2 - pw // 2, 60 + (SH - 60 - ph) // 2
    glow = Image.new("RGBA", (SW, SH), (0, 0, 0, 0))
    ImageDraw.Draw(glow).rounded_rectangle([px - 6, py - 6, px + pw + 6, py + ph + 6], 26, outline=P + (255,), width=14)
    img = Image.alpha_composite(img, glow.filter(ImageFilter.GaussianBlur(22)))
    v = Image.fromarray(fr[:, :, ::-1]).resize((pw, ph), Image.LANCZOS)
    m = Image.new("L", (pw, ph), 0); ImageDraw.Draw(m).rounded_rectangle([0, 0, pw - 1, ph - 1], 20, fill=255)
    img.paste(v, (px, py), m)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([px, py, px + pw, py + ph], 20, outline=PL + (200,), width=2)
    # window/title bar (generic)
    d.rectangle([0, 0, SW, 44], fill=(18, 14, 26, 255))
    for k, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([20 + k * 30, 14, 36 + k * 30, 30], fill=c + (255,))
    d.text((SW / 2, 22), "STATE RP — Зимнее обновление.mp4", font=F("mont600.ttf", 22), fill=(220, 215, 235, 255), anchor="mm")
    # left: scene typography
    si = max(k for k in range(5) if td >= SCENES[k][0])
    t0, num, a, b, sub = SCENES[si]
    k = ease((td - t0) / 0.5)
    lx = 70 - (1 - k) * 80
    al = int(255 * k)
    d.text((lx, 300), num, font=F("mont900i.ttf", 120), fill=P + (int(al * 0.9),))
    d.text((lx, 450), a, font=F("mont900i.ttf", 64), fill=(255, 255, 255, al))
    d.text((lx, 525), b, font=F("mont900i.ttf", 64), fill=PL + (al,))
    # wrap subtitle
    words, lines, cur = sub.split(), [], ""
    for w_ in words:
        if len(cur + " " + w_) > 18: lines.append(cur); cur = w_
        else: cur = (cur + " " + w_).strip()
    lines.append(cur)
    for li, ln in enumerate(lines):
        d.text((lx, 620 + li * 40), ln, font=F("mont600.ttf", 30), fill=(210, 200, 230, al))
    # right: chapter list + visualiser
    rx = px + pw + 70
    d.text((rx, 200), "В ОБНОВЛЕНИИ", font=F("mont800.ttf", 26), fill=(200, 190, 225, 255))
    items = ["Зима и снег", "Адвент-календарь", "Новый батлпасс", "Новые тачки", "7 декабря"]
    for j, it in enumerate(items):
        y = 260 + j * 64
        on = j == si; done = j < si
        col = (255, 255, 255, 255) if on else ((170, 150, 210, 255) if done else (110, 100, 135, 255))
        d.ellipse([rx, y + 8, rx + 22, y + 30], fill=(P + (255,)) if on or done else (60, 50, 80, 255))
        d.text((rx + 40, y + 19), it, font=F("mont800.ttf" if on else "mont600.ttf", 32), fill=col, anchor="lm")
    bands = BANDS[min(i, len(BANDS) - 1)]
    bx0, by = rx, 840; bw = 14
    for j, val in enumerate(bands[:20]):
        hgt = 8 + val * 170
        x = bx0 + j * (bw + 7)
        d.rounded_rectangle([x, by - hgt, x + bw, by], 6, fill=PL + (230,))
    # progress bar
    prog = rt / 20.0
    d.rounded_rectangle([px, SH - 14, px + pw, SH - 8], 3, fill=(60, 50, 80, 255))
    d.rounded_rectangle([px, SH - 14, px + int(pw * prog), SH - 8], 3, fill=P + (255,))
    out = np.asarray(img.convert("RGB"))
    cv2.imwrite(os.path.join(HERE, "screen", f"s_{i:04d}.jpg"), out[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
    small = cv2.resize(out, (16, 10), interpolation=cv2.INTER_AREA).reshape(-1, 3).mean(0) / 255
    return i, small.tolist()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        for i, fr in frames():
            if i in (40, 200, 330, 420, 560): build((i, fr))
        sys.exit()
    with Pool(4) as p:
        res = dict(p.imap(build, frames(), chunksize=4))
    json.dump([res[k] for k in sorted(res)], open(os.path.join(HERE, "screen_colors.json"), "w"))
    print("done", len(res))
