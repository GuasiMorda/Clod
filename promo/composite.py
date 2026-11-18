#!/usr/bin/env python3
"""Composite 3D frames (three.js) with titles, UI overlays, beat-synced camera FX and subtitles."""
import math, random, subprocess, sys
from multiprocessing import Pool
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops
import render as R
from render import W, H, FPS, NF, S, SLOWMO, clamp, ease_out, lerp, beat_pulse, font, text_center, punch_text

FR = sys.argv[2] if len(sys.argv) > 2 else "3d/frames"
OUT = sys.argv[1] if len(sys.argv) > 1 else "video3d_noaudio.mp4"

def ui_hook(t, ui, glow):
    punch_text(ui, glow, t, 0.5, "7 ДЕКАБРЯ", 330, 150, glow_col=(150, 210, 255))
    punch_text(ui, glow, t, 1.0, "STATE RP", 510, 190, color=(140, 220, 255), glow_col=(60, 160, 255))
    if t > 1.5:
        a = ease_out((t - 1.5) * 3)
        text_center(ImageDraw.Draw(ui), W / 2, 650, "ЗИМНЕЕ ОБНОВЛЕНИЕ", int(70 * a) + 1, sw=6)

def ui_advent(t, ui, glow):
    lt = t - S[1]
    d = ImageDraw.Draw(ui)
    for t0, s in [(0.6, "+50 000 $ на карту"), (1.6, "Новая тачка в гараже!"), (2.6, "Эксклюзивная окраска")]:
        if t0 < lt < t0 + 1.0:
            a = ease_out((lt - t0) * 4)
            x = lerp(W + 50, W - 80, a); y = 1430
            d.rounded_rectangle([x - 700, y - 60, x, y + 60], 30, fill=(30, 160, 80, 235), outline=(255, 255, 255), width=3)
            d.text((x - 350, y), s, font=font(46, False), fill=(255, 255, 255), anchor="mm")
    punch_text(ui, glow, t, S[1] + 0.05, "АДВЕНТ-КАЛЕНДАРЬ", 200, 78, color=(255, 230, 120), glow_col=(255, 180, 60))
    punch_text(ui, glow, t, S[1] + 0.55, "24 ДНЯ НАГРАД", 320, 110, glow_col=(255, 200, 90))

def ui_pass(t, ui, glow):
    lt = t - S[2]
    d = ImageDraw.Draw(ui)
    prog = clamp(lt / 3.8)
    d.rounded_rectangle([80, 1380, W - 80, 1430], 25, fill=(30, 20, 50, 220), outline=(170, 140, 255), width=3)
    d.rounded_rectangle([84, 1384, 84 + (W - 168) * prog, 1426], 22, fill=(255, 200, 60))
    d.text((W / 2, 1480), f"УРОВЕНЬ {int(1 + prog * 74)} / 75", font=font(46), fill=(255, 255, 255), anchor="mm",
           stroke_width=4, stroke_fill=(0, 0, 0))
    punch_text(ui, glow, t, S[2] + 0.05, "НОВЫЙ", 230, 120, color=(255, 215, 80), glow_col=(255, 170, 40))
    punch_text(ui, glow, t, S[2] + 0.5, "БАТЛПАСС", 380, 150, glow_col=(190, 140, 255))

def ui_cars(t, ui, glow):
    lt = t - S[3]
    d = ImageDraw.Draw(ui)
    if lt < 2.5:
        idx = 0 if lt < 1.0 else (1 if lt < 2.0 else 2)
        label = ["ВАЗ", "AMG", "CHARGER"][idx]
        st = lt - idx
        sz = int(130 * (1 + 0.4 * math.exp(-st * 12)))
        d.text((W / 2, 640), label, font=font(sz), fill=(255, 255, 255), anchor="mm", stroke_width=8, stroke_fill=(200, 20, 40))
    if SLOWMO[0] <= t < SLOWMO[1]:
        d.rectangle([0, 0, W, 130], fill=(0, 0, 0, 255))
        d.rectangle([0, H - 130, W, H], fill=(0, 0, 0, 255))
        d.text((70, 1480), "SLOW-MO", font=font(52), fill=(255, 255, 255), anchor="lm", stroke_width=4, stroke_fill=(0, 0, 0))
    punch_text(ui, glow, t, S[3] + 0.05, "МНОГО", 270, 130, color=(255, 80, 90), glow_col=(255, 40, 60))
    punch_text(ui, glow, t, S[3] + 0.5, "НОВЫХ ТАЧЕК", 410, 108, glow_col=(255, 90, 90))

def ui_cta(t, ui, glow):
    lt = t - S[4]
    du = ImageDraw.Draw(ui)
    g = ImageDraw.Draw(glow)
    if lt > 0.3:
        a = ease_out((lt - 0.3) * 2.5)
        uy = lerp(200, 400, a)
        sz = int(200 * (0.6 + 0.4 * a))
        text_center(du, W / 2, uy, "STATE RP", sz, fill=(150, 225, 255), stroke=(10, 30, 80), sw=10)
        g.text((W / 8, uy / 4), "STATE RP", font=font(sz // 4), fill=(60, 160, 255), anchor="mm")
        text_center(du, W / 2, uy + 160, "ЗИМНЕЕ ОБНОВЛЕНИЕ · 7 ДЕКАБРЯ", 50, sw=5)
    if lt > 1.5:
        a = ease_out((lt - 1.5) * 3)
        bob = math.sin(lt * 10) * 20
        cy = 1150 + bob
        du.rounded_rectangle([W / 2 - 330 * a, cy - 70, W / 2 + 330 * a, cy + 70], 70, fill=(230, 30, 40), outline=(255, 255, 255), width=5)
        if a > 0.6:
            du.text((W / 2, cy), "ПОДПИШИСЬ", font=font(76), fill=(255, 255, 255), anchor="mm")
        ay = cy + 110 + bob * 0.5
        du.polygon([(W / 2 - 60, ay), (W / 2 + 60, ay), (W / 2 + 60, ay + 60), (W / 2 + 110, ay + 60), (W / 2, ay + 160),
                    (W / 2 - 110, ay + 60), (W / 2 - 60, ay + 60)], fill=(255, 255, 255), outline=(230, 30, 40))

UIS = [ui_hook, ui_advent, ui_pass, ui_cars, ui_cta]

def frame(fi):
    t = fi / FPS
    si = max(i for i in range(5) if t >= S[i])
    img = Image.open(f"{FR}/f_{fi:04d}.jpg").convert("RGB")
    if img.size != (W, H):
        img = img.resize((W, H), Image.LANCZOS)
    glow = Image.new("RGB", (W // 4, H // 4), (0, 0, 0))
    ui = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    UIS[si](t, ui, glow)
    gl = glow.filter(ImageFilter.GaussianBlur(10)).resize((W, H), Image.BILINEAR)
    img = ImageChops.screen(img, gl)
    img = Image.alpha_composite(img.convert("RGBA"), ui).convert("RGB")
    slow = SLOWMO[0] <= t < SLOWMO[1]
    bp = beat_pulse(t, 9.0) * (0.25 if slow else 1.0)
    since_cut = t - S[si]
    cutp = math.exp(-since_cut * 10)
    zoom = 1.0 + 0.04 * bp + 0.10 * cutp
    r = random.Random(fi)
    amp = 18 * bp + 34 * cutp
    if abs(t - 14.1) < 0.5:
        amp += 45 * math.exp(-abs(t - 14.1) * 6)
    ox, oy = (r.random() - 0.5) * amp, (r.random() - 0.5) * amp
    rot = (r.random() - 0.5) * 0.8 * bp
    img = img.rotate(rot, resample=Image.BILINEAR, center=(W / 2 + ox, H / 2 + oy))
    cw, ch = W / zoom, H / zoom
    x0, y0 = (W - cw) / 2 + ox, (H - ch) / 2 + oy
    img = img.transform((W, H), Image.EXTENT, (x0, y0, x0 + cw, y0 + ch), Image.BILINEAR)
    shift = int(8 * bp + 16 * cutp)
    if shift > 1:
        rch, gch, bch = img.split()
        img = Image.merge("RGB", (ImageChops.offset(rch, shift, 0), gch, ImageChops.offset(bch, -shift, 0)))
    if since_cut < 0.12 and si > 0:
        img = Image.blend(img, Image.new("RGB", (W, H), (255, 255, 255)), 0.65 * (1 - since_cut / 0.12))
    arr = np.asarray(img).astype(np.float32)
    arr *= (0.25 + 0.75 * R.VIG)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    for a, b, s in R.SUBS:
        if a <= t < b:
            for li, ln in enumerate(s.split("|")):
                text_center(d, W / 2, 1600 + li * 72, ln, 58, fill=(255, 255, 120), sw=6, italic=False)
    return img.tobytes()

if __name__ == "__main__":
    if len(sys.argv) > 3:
        for tt in sys.argv[3].split(","):
            fi = int(float(tt) * FPS)
            Image.frombytes("RGB", (W, H), frame(fi)).resize((360, 640)).save(f"c_{fi:04d}.png")
        sys.exit()
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                           "-pix_fmt", "yuv420p", OUT], stdin=subprocess.PIPE)
    with Pool(4) as p:
        for i, b in enumerate(p.imap(frame, range(NF), chunksize=4)):
            ff.stdin.write(b)
    ff.stdin.close(); ff.wait()
    print("done", OUT)
