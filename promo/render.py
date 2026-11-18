#!/usr/bin/env python3
"""Procedural promo video: State RP winter update (vertical 1080x1920, 20s, 30fps)."""
import math, random, subprocess, sys, os
from multiprocessing import Pool
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops

W, H, FPS, DUR = 1080, 1920, 30, 20.0
BPM = 120.0
BEAT = 60.0 / BPM  # 0.5s
NF = int(DUR * FPS)
OUT = sys.argv[1] if len(sys.argv) > 1 else "video_noaudio.mp4"

FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FBI = "/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf"
if not os.path.exists(FBI):
    FBI = FB
_fc = {}
def font(size, italic=True):
    k = (size, italic)
    if k not in _fc:
        _fc[k] = ImageFont.truetype(FBI if italic else FB, size)
    return _fc[k]

# scene boundaries (seconds) — aligned to the beat grid
S = [0.0, 3.0, 7.0, 11.0, 16.0, 20.0]

def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def ease_out(x): x = clamp(x); return 1 - (1 - x) ** 3
def ease_in_out(x): x = clamp(x); return 3 * x * x - 2 * x * x * x
def lerp(a, b, x): return a + (b - a) * x

def beat_pulse(t, decay=8.0):
    """1.0 right on a beat, decaying exponentially afterwards."""
    ph = (t % BEAT)
    return math.exp(-ph * decay)

# ---------- snow ----------
rng = random.Random(7)
FLAKES = [(rng.random(), rng.random(), 0.3 + rng.random() * 1.2, rng.random() * 6.28, 2 + rng.random() * 5)
          for _ in range(420)]

def draw_snow(d, t, speed=1.0, wind=0.15, alpha=220):
    for fx, fy, sp, ph, sz in FLAKES:
        y = (fy + t * sp * 0.25 * speed) % 1.0
        x = (fx + t * wind * sp * 0.3 + 0.02 * math.sin(t * 2 + ph)) % 1.0
        s = sz * (0.6 + sp * 0.5)
        a = int(alpha * (0.4 + 0.6 * sp / 1.5))
        X, Y = x * W, y * H
        d.ellipse([X - s, Y - s, X + s, Y + s], fill=(235, 242, 255, a))

def vgrad(top, bot):
    a = np.linspace(0, 1, H)[:, None, None]
    arr = (np.array(top)[None, None, :] * (1 - a) + np.array(bot)[None, None, :] * a)
    arr = np.repeat(arr, W, axis=1).astype(np.uint8)
    return Image.fromarray(arr, "RGB")

BG_NIGHT = vgrad((6, 10, 28), (30, 44, 80))
BG_UI = vgrad((18, 20, 30), (10, 11, 18))
BG_CITY = vgrad((4, 6, 20), (40, 34, 70))

# ---------- text helpers ----------
def text_center(d, cx, cy, s, size, fill=(255, 255, 255), stroke=(0, 0, 0), sw=8, italic=True, anchor="mm"):
    d.text((cx, cy), s, font=font(size, italic), fill=fill, stroke_width=sw, stroke_fill=stroke, anchor=anchor)

def punch_text(layer, glow, t, t0, s, cy, size, color=(255, 255, 255), glow_col=(120, 200, 255)):
    """Text that slams in at t0 (scale overshoot) and stays."""
    if t < t0: return
    k = t - t0
    sc = 1.0 + 0.6 * math.exp(-k * 14) * math.cos(k * 30)
    sz = max(10, int(size * sc))
    d = ImageDraw.Draw(layer)
    text_center(d, W / 2, cy, s, sz, fill=color, sw=max(4, sz // 12))
    g = ImageDraw.Draw(glow)
    g.text((W / 8, cy / 4), s, font=font(max(4, sz // 4)), fill=glow_col, anchor="mm")

# ---------- car shapes (generic, original silhouettes) ----------
CARS = {
    # side profile, normalized: x 0..1 (rear->front), y 0..1 (top->ground). wheels: (x, r)
    "box": dict(body=[(0.02, 0.62), (0.03, 0.40), (0.20, 0.38), (0.30, 0.08), (0.70, 0.08), (0.80, 0.36),
                      (0.98, 0.40), (0.99, 0.66), (0.02, 0.70)],
                win=[(0.32, 0.14), (0.68, 0.14), (0.76, 0.36), (0.24, 0.37)],
                wheels=[(0.20, 0.15), (0.80, 0.15)], h=0.55),
    "coupe": dict(body=[(0.0, 0.60), (0.02, 0.42), (0.18, 0.36), (0.40, 0.10), (0.62, 0.10), (0.80, 0.34),
                        (0.99, 0.44), (1.0, 0.62), (0.0, 0.70)],
                  win=[(0.42, 0.15), (0.61, 0.15), (0.74, 0.34), (0.26, 0.36)],
                  wheels=[(0.18, 0.15), (0.82, 0.15)], h=0.42),
    "muscle": dict(body=[(0.0, 0.58), (0.01, 0.36), (0.22, 0.32), (0.36, 0.06), (0.60, 0.06), (0.70, 0.30),
                         (0.99, 0.34), (1.0, 0.64), (0.0, 0.70)],
                   win=[(0.38, 0.11), (0.58, 0.11), (0.66, 0.30), (0.28, 0.32)],
                   wheels=[(0.17, 0.16), (0.83, 0.16)], h=0.40),
}

def draw_car_side(d, g, cx, cy, length, kind, color, crush=0.0, t=0.0, facing=1, lights=True):
    c = CARS[kind]
    hh = length * c["h"]
    rot = globals().get("_DRIFT", 0.0)
    def P(x, y):
        # crush pushes the front (x>0.75) backwards & crumples it
        if crush > 0 and x > 0.7:
            k = (x - 0.7) / 0.3
            x = x - crush * 0.30 * k
            y = y + crush * 0.14 * math.sin(k * 9 + y * 7) * k
        X = (x - 0.5) * length * facing
        Y = -hh + y * hh
        return (cx + X * math.cos(rot) - Y * math.sin(rot), cy + X * math.sin(rot) + Y * math.cos(rot))
    body = [P(x, y) for x, y in c["body"]]
    d.polygon(body, fill=color, outline=(255, 255, 255))
    # highlight line
    d.line([P(0.04, 0.45), P(0.96, 0.47)], fill=tuple(min(255, v + 70) for v in color), width=max(2, int(length / 160)))
    d.polygon([P(x, y) for x, y in c["win"]], fill=(20, 30, 55))
    for wx, wr in c["wheels"]:
        X, Y = P(wx, 0.70)
        r = wr * hh * 1.6 * 0.6
        spin = t * 25
        d.ellipse([X - r, Y - r, X + r, Y + r], fill=(15, 15, 18))
        d.ellipse([X - r * 0.6, Y - r * 0.6, X + r * 0.6, Y + r * 0.6], fill=(170, 175, 185))
        for i in range(5):
            a = spin + i * 1.2566
            d.line([X, Y, X + math.cos(a) * r * 0.55, Y + math.sin(a) * r * 0.55], fill=(60, 60, 70), width=4)
    if lights:
        fx, fy = P(0.985, 0.46)
        rx, ry = P(0.02, 0.45)
        d.ellipse([fx - 10, fy - 7, fx + 10, fy + 7], fill=(255, 255, 230))
        d.ellipse([rx - 9, ry - 6, rx + 9, ry + 6], fill=(255, 40, 40))
        g.ellipse([fx / 4 - 14, fy / 4 - 10, fx / 4 + 14, fy / 4 + 10], fill=(255, 250, 200))
        g.ellipse([rx / 4 - 8, ry / 4 - 6, rx / 4 + 8, ry / 4 + 6], fill=(255, 30, 30))
        # headlight beam
        bx = fx / 4
        g.polygon([(bx, fy / 4), (bx + 120 * facing, fy / 4 - 25), (bx + 120 * facing, fy / 4 + 30)], fill=(90, 90, 70))

def spray(d, x0, y0, t, n=70, spread=1.0, dirx=-1, seed=1):
    r = random.Random(seed)
    for i in range(n):
        life = (t * 2.2 + r.random()) % 1.0
        ang = math.pi + dirx * (0.2 + r.random() * 0.9) if dirx < 0 else -(0.2 + r.random() * 0.9)
        dist = life * (200 + r.random() * 300) * spread
        x = x0 + math.cos(ang) * dist * (1 if dirx > 0 else 1)
        y = y0 - abs(math.sin(ang)) * dist * 0.6 + life * life * 120
        s = 6 + life * 40 * spread
        a = int(170 * (1 - life))
        d.ellipse([x - s, y - s, x + s, y + s], fill=(230, 238, 250, a))

# ---------- scenes ----------
def scene_hook(t, img, glow, ui):
    """0-3s: night, car drifting toward camera, headlights blasting."""
    d = ImageDraw.Draw(img, "RGBA")
    hz = 820
    # ground snow + road
    d.rectangle([0, hz, W, H], fill=(70, 82, 110))
    d.polygon([(W / 2 - 30, hz), (W / 2 + 30, hz), (W + 500, H), (-500, H)], fill=(38, 42, 56))
    for i in range(12):  # road dashes rushing
        z = ((i / 12) + t * 0.9) % 1.0
        y = hz + (H - hz) * z ** 2.2
        w = 4 + 30 * z ** 2.2
        hgt = 10 + 90 * z ** 2.2
        d.rectangle([W / 2 - w / 2, y, W / 2 + w / 2, y + hgt], fill=(220, 220, 200))
    # treeline silhouette
    r = random.Random(3)
    for i in range(40):
        x = i * 30 + r.random() * 20
        h = 60 + r.random() * 110
        d.polygon([(x - 30, hz), (x, hz - h), (x + 30, hz)], fill=(10, 16, 30))
    # approaching car (front view), sliding sideways = drift
    k = clamp(t / 3.0)
    scale = lerp(0.25, 1.35, k ** 1.6)
    cx = W / 2 + math.sin(t * 2.2) * 160 * (1 - k * 0.6)
    cy = lerp(hz + 40, 1500, k ** 1.6)
    tilt = math.sin(t * 2.2 + 0.6) * 0.12
    bw, bh = 520 * scale, 230 * scale
    def R(px, py):
        return (cx + px * math.cos(tilt) - py * math.sin(tilt), cy + px * math.sin(tilt) + py * math.cos(tilt))
    # snow spray from the rear wheels
    sd = ImageDraw.Draw(img, "RGBA")
    spray(sd, cx - bw * 0.4, cy, t, n=60, spread=scale, dirx=-1, seed=2)
    spray(sd, cx + bw * 0.4, cy, t + 0.3, n=60, spread=scale, dirx=1, seed=5)
    body = [R(-bw / 2, 0), R(-bw / 2, -bh * 0.55), R(-bw * 0.32, -bh * 0.62), R(-bw * 0.25, -bh), R(bw * 0.25, -bh),
            R(bw * 0.32, -bh * 0.62), R(bw / 2, -bh * 0.55), R(bw / 2, 0)]
    d.polygon(body, fill=(22, 24, 32))
    d.polygon([R(-bw * 0.22, -bh * 0.95), R(bw * 0.22, -bh * 0.95), R(bw * 0.28, -bh * 0.66), R(-bw * 0.28, -bh * 0.66)],
              fill=(40, 60, 90))
    d.rectangle([R(-bw * 0.15, -bh * 0.35)[0], R(0, -bh * 0.35)[1], R(bw * 0.15, -bh * 0.2)[0], R(0, -bh * 0.2)[1]], fill=(10, 10, 12))
    g = ImageDraw.Draw(glow)
    for sx in (-1, 1):
        hx, hy = R(sx * bw * 0.36, -bh * 0.4)
        rr = 22 * scale
        d.ellipse([hx - rr * 1.4, hy - rr * 0.8, hx + rr * 1.4, hy + rr * 0.8], fill=(255, 255, 245))
        gr = rr * (3.5 + 1.5 * beat_pulse(t))
        g.ellipse([(hx - gr * 2) / 4, (hy - gr) / 4, (hx + gr * 2) / 4, (hy + gr) / 4], fill=(255, 250, 225))
    # beams to camera
    hx, hy = R(0, -bh * 0.4)
    g.polygon([(hx / 4 - 40 * scale, hy / 4), (hx / 4 + 40 * scale, hy / 4), (W / 4 + 200, H / 4 + 60), (-200, H / 4 + 60)],
              fill=(70, 70, 60))
    # big text
    punch_text(ui, glow, t, 0.5, "7 ДЕКАБРЯ", 380, 150, glow_col=(150, 210, 255))
    punch_text(ui, glow, t, 1.0, "STATE RP", 560, 190, color=(140, 220, 255), glow_col=(60, 160, 255))
    if t > 1.5:
        dd = ImageDraw.Draw(ui)
        a = ease_out((t - 1.5) * 3)
        text_center(dd, W / 2, 700, "ЗИМНЕЕ ОБНОВЛЕНИЕ", int(70 * a) + 1, fill=(255, 255, 255), sw=6)

def scene_advent(t, img, glow, ui):
    """3-7s: advent calendar UI, cells opening on the beat, money & cars dropping."""
    lt = t - S[1]
    img.paste(BG_UI)
    d = ImageDraw.Draw(img, "RGBA")
    # generic chat-app style panel
    px0, py0, px1, py1 = 70, 500, W - 70, 1420
    d.rounded_rectangle([px0, py0, px1, py1], 36, fill=(40, 43, 58), outline=(80, 90, 130), width=3)
    d.rounded_rectangle([px0, py0, px1, py0 + 110], 36, fill=(52, 56, 76))
    d.text((px0 + 40, py0 + 55), "#  адвент-календарь", font=font(44, False), fill=(200, 205, 230), anchor="lm")
    cols, rows = 4, 6
    gx0, gy0 = px0 + 40, py0 + 150
    cw = (px1 - px0 - 80 - (cols - 1) * 18) / cols
    ch = (py1 - gy0 - 40 - (rows - 1) * 18) / rows
    # number of opened cells grows faster over time (acceleration)
    opened_f = 24 * (clamp(lt / 3.6) ** 1.7)
    rewards = ["$", "car", "*", "$", "car", "$"]
    g = ImageDraw.Draw(glow)
    for i in range(24):
        c, r = i % cols, i // cols
        x = gx0 + c * (cw + 18)
        y = gy0 + r * (ch + 18)
        o = clamp(opened_f - i)
        if o <= 0:
            d.rounded_rectangle([x, y, x + cw, y + ch], 20, fill=(28, 70, 120), outline=(120, 190, 255), width=3)
            # ribbon
            d.rectangle([x + cw / 2 - 8, y, x + cw / 2 + 8, y + ch], fill=(200, 40, 60))
            d.rectangle([x, y + ch / 2 - 8, x + cw, y + ch / 2 + 8], fill=(200, 40, 60))
            d.text((x + cw / 2, y + ch / 2), str(i + 1), font=font(50, False), fill=(255, 255, 255), anchor="mm",
                   stroke_width=5, stroke_fill=(20, 30, 50))
        else:
            pop = 1 + 0.25 * math.exp(-(opened_f - i) * 3) * math.sin((opened_f - i) * 12)
            mx, my = x + cw / 2, y + ch / 2
            ww, hh = cw / 2 * pop, ch / 2 * pop
            d.rounded_rectangle([mx - ww, my - hh, mx + ww, my + hh], 20, fill=(255, 210, 90), outline=(255, 255, 255), width=3)
            kind = rewards[i % len(rewards)]
            if kind == "$":
                d.text((mx, my), "$", font=font(int(80 * pop), False), fill=(30, 120, 40), anchor="mm")
            elif kind == "car":
                draw_car_side(d, ImageDraw.Draw(Image.new("RGB", (4, 4))), mx, my + hh * 0.45, cw * 0.8, "coupe",
                              (200, 40, 50), lights=False)
            else:
                d.text((mx, my), "★", font=font(int(80 * pop), False), fill=(140, 60, 200), anchor="mm")
            if o < 1:
                g.ellipse([(mx - 140) / 4, (my - 140) / 4, (mx + 140) / 4, (my + 140) / 4], fill=(int(255 * (1 - o)), int(200 * (1 - o)), 60))
    # notifications sliding in
    notes = [(0.6, "+50 000 $ на карту"), (1.6, "Новая тачка в гараже!"), (2.6, "Эксклюзивная окраска")]
    for k, (t0, s) in enumerate(notes):
        if lt > t0:
            a = ease_out((lt - t0) * 4)
            x = lerp(W + 50, W - 80, a)
            y = 1475
            if lt > t0 + 1.0: continue
            d.rounded_rectangle([x - 700, y - 60, x, y + 60], 30, fill=(30, 160, 80, 240), outline=(255, 255, 255), width=3)
            d.text((x - 350, y), s, font=font(46, False), fill=(255, 255, 255), anchor="mm")
    punch_text(ui, glow, t, S[1] + 0.05, "АДВЕНТ-КАЛЕНДАРЬ", 260, 78, color=(255, 230, 120), glow_col=(255, 180, 60))
    punch_text(ui, glow, t, S[1] + 0.55, "24 ДНЯ НАГРАД", 390, 110, glow_col=(255, 200, 90))

def scene_pass(t, img, glow, ui):
    """7-11s: battle pass levels scrolling, rewards in between."""
    lt = t - S[2]
    img.paste(vgrad((40, 10, 70), (10, 20, 60)) if not hasattr(scene_pass, "bg") else scene_pass.bg)
    d = ImageDraw.Draw(img, "RGBA")
    # diagonal speed stripes
    for i in range(14):
        x = ((i * 160 - lt * 900) % (W + 600)) - 300
        d.polygon([(x, 0), (x + 60, 0), (x - 500, H), (x - 560, H)], fill=(255, 255, 255, 12))
    scroll = ease_in_out(lt / 3.8) * 5600
    cardw, cardh, gap = 420, 620, 50
    y0 = 930
    g = ImageDraw.Draw(glow)
    items = ["car", "paint", "plate", "$", "car", "paint", "plate", "car", "$", "paint", "car", "plate", "car", "car", "paint"]
    colors = [(220, 40, 50), (40, 140, 255), (250, 200, 40), (60, 200, 120), (240, 240, 245)]
    for i, it in enumerate(items):
        x = W / 2 - cardw / 2 + i * (cardw + gap) - scroll + 300
        if x > W or x + cardw < 0: continue
        center = 1 - min(1, abs((x + cardw / 2) - W / 2) / 700)
        sc = 0.85 + 0.25 * center
        cw, chh = cardw * sc, cardh * sc
        cx = x + cardw / 2
        top = y0 - chh / 2
        lvl = (i + 1) * 5
        prem = i % 3 == 2
        d.rounded_rectangle([cx - cw / 2, top, cx + cw / 2, top + chh], 30,
                            fill=(70, 40, 120) if not prem else (120, 80, 20),
                            outline=(255, 215, 80) if prem else (170, 140, 255), width=6)
        d.text((cx, top + 60 * sc), f"УР. {lvl}", font=font(int(56 * sc)), fill=(255, 255, 255), anchor="mm",
               stroke_width=4, stroke_fill=(0, 0, 0))
        my = top + chh * 0.55
        if it == "car":
            draw_car_side(d, g, cx, my + 70 * sc, cw * 0.85, ["coupe", "muscle", "box"][i % 3], colors[i % 5], t=t)
        elif it == "paint":
            for k in range(4):
                d.pieslice([cx - 110 * sc, my - 110 * sc, cx + 110 * sc, my + 110 * sc], k * 90 + lt * 200, k * 90 + 90 + lt * 200,
                           fill=colors[k])
            d.text((cx, top + chh - 70 * sc), "ОКРАСКА", font=font(int(40 * sc)), fill=(255, 255, 255), anchor="mm")
        elif it == "plate":
            d.rounded_rectangle([cx - 160 * sc, my - 50 * sc, cx + 160 * sc, my + 50 * sc], 10, fill=(250, 250, 250), outline=(0, 0, 0), width=4)
            d.text((cx, my), "А777АА", font=font(int(62 * sc), False), fill=(10, 10, 10), anchor="mm")
            d.text((cx, top + chh - 70 * sc), "НОМЕРА", font=font(int(40 * sc)), fill=(255, 255, 255), anchor="mm")
        else:
            d.text((cx, my), "$", font=font(int(200 * sc), False), fill=(90, 230, 120), anchor="mm", stroke_width=6, stroke_fill=(0, 60, 0))
        if prem:
            d.text((cx, top + chh - 20 * sc), "ПРЕМИУМ", font=font(int(30 * sc)), fill=(255, 215, 80), anchor="mm")
    # progress bar
    prog = clamp(lt / 3.8)
    d.rounded_rectangle([80, 1320, W - 80, 1370], 25, fill=(30, 20, 50), outline=(170, 140, 255), width=3)
    d.rounded_rectangle([84, 1324, 84 + (W - 168) * prog, 1366], 22, fill=(255, 200, 60))
    lv = int(1 + prog * 74)
    d.text((W / 2, 1430), f"УРОВЕНЬ {lv} / 75", font=font(48), fill=(255, 255, 255), anchor="mm", stroke_width=4, stroke_fill=(0, 0, 0))
    g.rectangle([84 / 4, 1324 / 4, (84 + (W - 168) * prog) / 4, 1366 / 4], fill=(200, 140, 30))
    punch_text(ui, glow, t, S[2] + 0.05, "НОВЫЙ", 300, 120, color=(255, 215, 80), glow_col=(255, 170, 40))
    punch_text(ui, glow, t, S[2] + 0.5, "БАТЛПАСС", 450, 150, glow_col=(190, 140, 255))

SLOWMO = (13.5, 15.5)

def scene_cars(t, img, glow, ui):
    """11-16s: new cars — showroom spins, launches, drifting on snow, slow-mo crash deformation."""
    lt = t - S[3]
    d = ImageDraw.Draw(img, "RGBA")
    g = ImageDraw.Draw(glow)
    shots = [(0.0, "box", (200, 30, 30), "ВАЗ"), (1.0, "coupe", (230, 230, 235), "AMG"), (2.0, "muscle", (20, 20, 25), "CHARGER")]
    if lt < 2.5:
        # showroom: spotlight floor, car rolls into frame
        idx = 0 if lt < 1.0 else (1 if lt < 2.0 else 2)
        t0, kind, col, label = shots[idx]
        st = lt - t0
        d.rectangle([0, 0, W, H], fill=(12, 12, 20))
        d.ellipse([W / 2 - 600, 1150, W / 2 + 600, 1450], fill=(40, 40, 60))
        for k in range(3):
            g.polygon([(W / 8 + (k - 1) * 70, 0), (W / 8 + (k - 1) * 70 + 15, 0), (W / 8 + (k - 1) * 120 + 120, 360), (W / 8 + (k - 1) * 120 - 120, 360)],
                      fill=(55, 55, 75))
        x = lerp(-600, W / 2, ease_out(st * 2.2)) + st * 60
        draw_car_side(d, g, x, 1300, 860, kind, col, t=t)
        d.text((W / 2, 760), label, font=font(130), fill=(255, 255, 255), anchor="mm", stroke_width=8, stroke_fill=(200, 20, 40))
        g.text((W / 8, 760 / 4), label, font=font(34), fill=(255, 60, 80), anchor="mm")
    else:
        # snowy track, drifting car with speed, slow-mo crash
        slow = SLOWMO[0] <= t < SLOWMO[1]
        # local "game time": runs 4x slower inside slow-mo
        gt = lt if t < SLOWMO[0] else (SLOWMO[0] - S[3]) + (min(t, SLOWMO[1]) - SLOWMO[0]) * 0.25 + max(0, t - SLOWMO[1])
        hz = 900
        d.rectangle([0, 0, W, hz], fill=(8, 12, 30))
        d.rectangle([0, hz, W, H], fill=(190, 200, 220))
        d.rectangle([0, hz + 260, W, hz + 560], fill=(60, 66, 84))
        # passing lamp posts
        for i in range(6):
            x = ((i * 380 - gt * 2600) % (W + 800)) - 400
            d.rectangle([x, hz - 380, x + 14, hz + 260], fill=(30, 30, 40))
            g.ellipse([(x - 30) / 4, (hz - 410) / 4, (x + 50) / 4, (hz - 350) / 4], fill=(255, 200, 120))
        # speed lines
        for i in range(30):
            r = random.Random(i)
            y = hz + 260 + r.random() * 300
            x = ((r.random() * W - gt * 3800) % (W + 400)) - 200
            d.line([x, y, x + 200, y], fill=(255, 255, 255, 60), width=3)
        crush = 0.0
        ang_shake = 0
        if t >= SLOWMO[0] + 0.6:
            crush = ease_out((t - SLOWMO[0] - 0.6) / 1.0)
        cx = W / 2 + 60 * math.sin(gt * 3)
        cy = hz + 470
        globals()["_DRIFT"] = -0.10 * math.sin(gt * 3) - 0.04
        draw_car_side(d, g, cx, cy, 900, "muscle", (230, 120, 20), crush=crush, t=gt)
        globals()["_DRIFT"] = 0.0
        spray(d, cx - 450, cy - 10, gt, n=90, spread=1.3, dirx=-1, seed=9)
        if crush > 0:
            # barrier the car hits + debris
            bx = cx + 900 * 0.5 - crush * 160 + 40
            d.rectangle([bx, cy - 280, bx + 80, cy + 10], fill=(200, 30, 30), outline=(255, 255, 255), width=4)
            rr = random.Random(4)
            for i in range(40):
                a = rr.random() * 6.28
                dist = crush * (80 + rr.random() * 380)
                x, y = bx + math.cos(a) * dist, cy - 150 + math.sin(a) * dist * 0.7
                d.polygon([(x, y), (x + 14, y + 4), (x + 4, y + 16)], fill=(220, 230, 255, 220))
        if slow:
            d.text((80, 220), "SLOW-MO", font=font(56), fill=(255, 255, 255), anchor="lm", stroke_width=4, stroke_fill=(0, 0, 0))
            # letterbox
            d.rectangle([0, 0, W, 140], fill=(0, 0, 0))
            d.rectangle([0, H - 140, W, H], fill=(0, 0, 0))
    punch_text(ui, glow, t, S[3] + 0.05, "МНОГО", 330, 130, color=(255, 80, 90), glow_col=(255, 40, 60))
    punch_text(ui, glow, t, S[3] + 0.5, "НОВЫХ ТАЧЕК", 470, 108, glow_col=(255, 90, 90))

def scene_cta(t, img, glow, ui):
    """16-20s: convoy through snowy night city, camera rises, logo + subscribe arrow."""
    lt = t - S[4]
    img.paste(BG_CITY)
    d = ImageDraw.Draw(img, "RGBA")
    g = ImageDraw.Draw(glow)
    rise = ease_in_out(lt / 3.5)
    hz = lerp(900, 1350, rise)  # camera going up -> horizon moves down, view steeper
    vx = W / 2
    # buildings in perspective on both sides
    r = random.Random(11)
    for side in (-1, 1):
        for i in range(10):
            z = 1 - i / 10
            bw = 220 * z + 40
            bh = (500 + r.random() * 700) * z + 150
            x = vx + side * (90 + 520 * z ** 1.3)
            xb = x - bw / 2
            top = hz - bh
            d.rectangle([xb, top, xb + bw, hz + 40 * z], fill=(14 + i, 16 + i, 34 + i))
            for wy in range(int(top + 20), int(hz), int(28 * z + 10)):
                for wx in range(int(xb + 10), int(xb + bw - 10), int(26 * z + 10)):
                    if r.random() < 0.35:
                        d.rectangle([wx, wy, wx + 8 * z + 3, wy + 10 * z + 3], fill=(255, 200, 110))
    # street
    d.polygon([(vx - 40, hz), (vx + 40, hz), (W + 300, H), (-300, H)], fill=(170, 180, 205))
    d.polygon([(vx - 20, hz), (vx + 20, hz), (W / 2 + 340, H), (W / 2 - 340, H)], fill=(60, 64, 82))
    # convoy: pairs of tail lights receding
    for i in range(9):
        z = ((i / 9) + lt * 0.12) % 1.0
        depth = 1 - z
        y = hz + (H - hz) * depth ** 2.0
        s = depth ** 2.0
        if s < 0.02: continue
        cw = 300 * s
        x = vx + math.sin(i * 1.3) * 60 * s
        d.rounded_rectangle([x - cw / 2, y - cw * 0.45, x + cw / 2, y], max(1, int(12 * s)), fill=(18, 18, 26))
        for sx in (-1, 1):
            lx = x + sx * cw * 0.38
            ly = y - cw * 0.25
            rr = max(2, 16 * s)
            d.ellipse([lx - rr * 1.5, ly - rr * 0.6, lx + rr * 1.5, ly + rr * 0.6], fill=(255, 40, 40))
            g.ellipse([(lx - rr * 4) / 4, (ly - rr * 2) / 4, (lx + rr * 4) / 4, (ly + rr * 2) / 4], fill=(255, 20, 30))
    # logo
    if lt > 0.3:
        a = ease_out((lt - 0.3) * 2.5)
        uy = lerp(200, 430, a)
        du = ImageDraw.Draw(ui)
        sz = int(200 * (0.6 + 0.4 * a))
        text_center(du, W / 2, uy, "STATE RP", sz, fill=(150, 225, 255), stroke=(10, 30, 80), sw=10)
        g.text((W / 8, uy / 4), "STATE RP", font=font(sz // 4), fill=(60, 160, 255), anchor="mm")
        text_center(du, W / 2, uy + 170, "ЗИМНЕЕ ОБНОВЛЕНИЕ · 7 ДЕКАБРЯ", 52, sw=5)
    if lt > 1.5:
        du = ImageDraw.Draw(ui)
        a = ease_out((lt - 1.5) * 3)
        bob = math.sin(lt * 10) * 20
        cy = 1180 + bob
        du.rounded_rectangle([W / 2 - 330 * a, cy - 70, W / 2 + 330 * a, cy + 70], 70, fill=(230, 30, 40), outline=(255, 255, 255), width=5)
        if a > 0.6:
            du.text((W / 2, cy), "ПОДПИШИСЬ", font=font(76), fill=(255, 255, 255), anchor="mm")
        # arrow
        ay = cy + 120 + bob * 0.5
        du.polygon([(W / 2 - 60, ay), (W / 2 + 60, ay), (W / 2 + 60, ay + 60), (W / 2 + 110, ay + 60), (W / 2, ay + 160),
                    (W / 2 - 110, ay + 60), (W / 2 - 60, ay + 60)], fill=(255, 255, 255), outline=(230, 30, 40))

SCENES = [scene_hook, scene_advent, scene_pass, scene_cars, scene_cta]

SUBS = [
    (0.15, 2.95, "Зима на State RP!|7 декабря — обнова!"),
    (3.1, 6.9, "Деньги, тачки, эксклюзив —|заходи каждый день!"),
    (7.1, 10.9, "Качай уровни —|забирай топ-награды!"),
    (11.1, 15.9, "От ВАЗ до AMG и Charger —|забирай мечту!"),
    (16.1, 19.9, "Лайк, сохраняй —|и залетай на сервер!"),
]

def render(fi):
    t = fi / FPS
    si = max(i for i in range(5) if t >= S[i])
    img = BG_NIGHT.copy()
    glow = Image.new("RGB", (W // 4, H // 4), (0, 0, 0))
    ui = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    SCENES[si](t, img, glow, ui)
    # snow on top of the world (not in UI scenes)
    if si in (0, 3, 4):
        sl = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        slow = SLOWMO[0] <= t < SLOWMO[1]
        draw_snow(ImageDraw.Draw(sl), t if not slow else SLOWMO[0] + (t - SLOWMO[0]) * 0.25,
                  speed=1.8 if si == 0 else 1.0, wind=0.6 if si == 3 else 0.15)
        img = Image.alpha_composite(img.convert("RGBA"), sl).convert("RGB")
    # glow (bloom)
    gl = glow.filter(ImageFilter.GaussianBlur(10)).resize((W, H), Image.BILINEAR)
    img = ImageChops.screen(img, gl)
    img = Image.alpha_composite(img.convert("RGBA"), ui).convert("RGB")
    # ---- camera: zoom punch + shake on beats, extra hit at scene cuts ----
    slow = SLOWMO[0] <= t < SLOWMO[1]
    bp = beat_pulse(t, 9.0) * (0.25 if slow else 1.0)
    since_cut = t - S[si]
    cutp = math.exp(-since_cut * 10)
    zoom = 1.0 + 0.045 * bp + 0.12 * cutp
    if si == 3 and slow:
        zoom += 0.10 * ease_in_out((t - SLOWMO[0]) / 2.0)
    r = random.Random(fi)
    amp = 22 * bp + 40 * cutp
    if si == 3 and t > SLOWMO[0] + 0.6 and t < SLOWMO[0] + 1.2:
        amp += 50
    ox, oy = (r.random() - 0.5) * amp, (r.random() - 0.5) * amp
    rot = (r.random() - 0.5) * 0.9 * bp
    img = img.rotate(rot, resample=Image.BILINEAR, center=(W / 2 + ox, H / 2 + oy))
    cw, ch = W / zoom, H / zoom
    x0, y0 = (W - cw) / 2 + ox, (H - ch) / 2 + oy
    img = img.transform((W, H), Image.EXTENT, (x0, y0, x0 + cw, y0 + ch), Image.BILINEAR)
    # chromatic aberration on hits
    shift = int(10 * bp + 18 * cutp)
    if shift > 1:
        rch, gch, bch = img.split()
        rch = ImageChops.offset(rch, shift, 0)
        bch = ImageChops.offset(bch, -shift, 0)
        img = Image.merge("RGB", (rch, gch, bch))
    # white flash at cuts
    if since_cut < 0.12 and si > 0:
        a = 1 - since_cut / 0.12
        img = Image.blend(img, Image.new("RGB", (W, H), (255, 255, 255)), 0.7 * a)
    # cold grade + vignette
    arr = np.asarray(img).astype(np.float32)
    arr[..., 2] = np.minimum(255, arr[..., 2] * 1.06 + 4)
    arr *= VIG
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    # subtitles (not shaken)
    d = ImageDraw.Draw(img)
    for a, b, s in SUBS:
        if a <= t < b:
            for li, ln in enumerate(s.split("|")):
                text_center(d, W / 2, 1600 + li * 72, ln, 58, fill=(255, 255, 120), sw=6, italic=False)
    return img.tobytes()

yy, xx = np.mgrid[0:H, 0:W]
_v = ((xx - W / 2) / (W * 0.75)) ** 2 + ((yy - H / 2) / (H * 0.75)) ** 2
VIG = np.clip(1.15 - _v * 0.55, 0.45, 1.0)[..., None].astype(np.float32)

if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "stills":
        for tt in [0.4, 1.8, 2.8, 4.5, 6.5, 8.5, 10.0, 11.5, 12.5, 13.3, 14.6, 15.8, 17.0, 19.0]:
            Image.frombytes("RGB", (W, H), render(int(tt * FPS))).resize((360, 640)).save(f"stills/s_{tt:05.2f}.png")
        sys.exit()
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "19",
                           "-pix_fmt", "yuv420p", OUT], stdin=subprocess.PIPE)
    with Pool(4) as p:
        for i, b in enumerate(p.imap(render, range(NF), chunksize=4)):
            ff.stdin.write(b)
            if i % 60 == 0: print("frame", i, flush=True)
    ff.stdin.close(); ff.wait()
    print("done", OUT)
