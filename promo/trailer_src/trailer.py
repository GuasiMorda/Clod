#!/usr/bin/env python3
"""STATE RP — winter update trailer. 1080x1920, 30 fps, 20 s.
Realistic plates (from the design reference) + camera moves, motion blur, particles,
light flares, neon UI motion graphics. Output: raw frames piped to ffmpeg."""
import math, random, subprocess, sys, os
from functools import lru_cache
from multiprocessing import Pool
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H, FPS, DUR = 1080, 1920, 30, 20.0
NF = int(DUR * FPS)
BEAT = 0.5
CUTS = [0.0, 3.0, 7.0, 11.0, 16.0, 20.0]
SLOW = (14.0, 16.0)

PURPLE = np.array([0.55, 0.22, 1.0], np.float32)      # neon purple (RGB 0..1)
PURPLE_L = np.array([0.78, 0.55, 1.0], np.float32)
P_RGB = (140, 56, 255)
P_RGB_L = (196, 140, 255)

HERE = os.path.dirname(os.path.abspath(__file__))
FD = os.path.join(HERE, "fonts")
F_TITLE = os.path.join(FD, "mont900i.ttf")
F_BOLD = os.path.join(FD, "mont800.ttf")
F_BOLDI = os.path.join(FD, "mont800i.ttf")
F_UI = os.path.join(FD, "mont600.ttf")
F_SYM = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def lerp(a, b, x): return a + (b - a) * x
def eo(x): x = clamp(x); return 1 - (1 - x) ** 3          # ease out cubic
def eo5(x): x = clamp(x); return 1 - (1 - x) ** 5
def eio(x): x = clamp(x); return x * x * (3 - 2 * x)
def back(x, s=1.7):
    x = clamp(x); x -= 1; return x * x * ((s + 1) * x + s) + 1
def pulse(t, decay=9.0):
    return math.exp(-(t % BEAT) * decay)

# ---------------------------------------------------------------- plates
def load_plate(name):
    im = cv2.imread(os.path.join(HERE, "plates", name + ".png"))[:, :, ::-1].astype(np.float32) / 255.0
    h, w = im.shape[:2]
    a = np.ones((h, w), np.float32)
    f = 90
    ramp = np.linspace(0, 1, f, dtype=np.float32)
    a[:f, :] *= ramp[:, None]; a[-f:, :] *= ramp[::-1][:, None]
    a[:, :f] *= ramp[None, :]; a[:, -f:] *= ramp[::-1][None, :]
    return im, a
PL = {k: load_plate(k) for k in ("p1", "p2", "p4", "p5")}

def cam_matrix(cx, cy, z, rot=0.0, sx=W / 2, sy=H / 2):
    c, s = math.cos(rot) * z, math.sin(rot) * z
    return np.array([[c, -s, sx - (c * cx - s * cy)], [s, c, sy - (s * cx + c * cy)]], np.float32)

def map_pt(M, x, y):
    return (M[0, 0] * x + M[0, 1] * y + M[0, 2], M[1, 0] * x + M[1, 1] * y + M[1, 2])

def render_plate(name, cams, bg_boost=1.18, bg_dark=0.45):
    """cams: list of (cx,cy,z,rot) sub-samples (motion blur). Returns frame + matrix of the last sample."""
    img, alpha = PL[name]
    h, w = img.shape[:2]
    acc = np.zeros((H, W, 3), np.float32)
    for cx, cy, z, rot in cams:
        M = cam_matrix(cx, cy, z, rot)
        fg = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        a = cv2.warpAffine(alpha, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        if a.min() < 0.999:
            cover = max(H / h, W / w) * bg_boost
            zb = max(z * 1.05, cover)
            Mb = cam_matrix(cx, cy, zb / 6, rot, W / 12, H / 12)
            bg = cv2.warpAffine(img, Mb, (W // 6, H // 6), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT)
            bg = cv2.GaussianBlur(bg, (0, 0), 5)
            bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_CUBIC) * bg_dark
            bg = bg * 0.75 + bg.mean(axis=2, keepdims=True) * 0.25 * PURPLE * 1.4
            fg = fg * a[..., None] + bg * (1 - a[..., None])
        acc += fg
    return acc / len(cams), cam_matrix(*cams[0])

# ---------------------------------------------------------------- sprites / text
def to_sprite(pil_rgba):
    a = np.asarray(pil_rgba).astype(np.float32) / 255.0
    rgb = a[..., :3] * a[..., 3:4]
    return np.dstack([rgb, a[..., 3]])      # premultiplied RGBA

@lru_cache(maxsize=None)
def font(path, size): return ImageFont.truetype(path, size)

@lru_cache(maxsize=256)
def text_sprite(text, size, fill=(255, 255, 255), path=F_TITLE, stroke=0, stroke_fill=(0, 0, 0), glow=None, shadow=True, pad=60):
    f = font(path, size)
    bbox = f.getbbox(text, stroke_width=stroke)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    im = Image.new("RGBA", (tw + pad * 2, th + pad * 2), (0, 0, 0, 0))
    if shadow:
        sh = Image.new("RGBA", im.size, (0, 0, 0, 0))
        ImageDraw.Draw(sh).text((pad - bbox[0] + 6, pad - bbox[1] + 10), text, font=f, fill=(0, 0, 0, 170), stroke_width=stroke + 2, stroke_fill=(0, 0, 0, 170))
        im = Image.alpha_composite(im, sh.filter(ImageFilter.GaussianBlur(10)))
    if glow:
        gl = Image.new("RGBA", im.size, (0, 0, 0, 0))
        ImageDraw.Draw(gl).text((pad - bbox[0], pad - bbox[1]), text, font=f, fill=glow + (255,), stroke_width=stroke + 4, stroke_fill=glow + (255,))
        im = Image.alpha_composite(im, gl.filter(ImageFilter.GaussianBlur(18)))
    ImageDraw.Draw(im).text((pad - bbox[0], pad - bbox[1]), text, font=f, fill=fill + (255,), stroke_width=stroke, stroke_fill=stroke_fill + (255,))
    return to_sprite(im)

@lru_cache(maxsize=64)
def brush_sprite(w, h, seed=1, color=P_RGB, skew=0.18):
    """Procedural paint-brush stroke (like the purple strokes in the design)."""
    r = random.Random(seed)
    S = 2
    im = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    ww, hh = w * S, h * S
    pts_top, pts_bot = [], []
    n = 28
    for i in range(n + 1):
        x = ww * (0.04 + 0.92 * i / n)
        pts_top.append((x + skew * hh * 0.5 + r.uniform(-8, 8) * S, hh * 0.16 + r.uniform(-1, 1) * hh * 0.07))
        pts_bot.append((x - skew * hh * 0.5 + r.uniform(-8, 8) * S, hh * 0.86 + r.uniform(-1, 1) * hh * 0.07))
    # ragged ends
    left = [(ww * 0.04 - skew * hh * 0.5 + r.uniform(-30, 10) * S, hh * (0.2 + 0.6 * k / 6)) for k in range(7)]
    right = [(ww * 0.96 + skew * hh * 0.5 + r.uniform(-10, 40) * S, hh * (0.8 - 0.6 * k / 6)) for k in range(7)]
    poly = pts_top + right + pts_bot[::-1] + left[::-1]
    d.polygon(poly, fill=color + (255,))
    # dry-brush streaks
    for i in range(70):
        y = r.uniform(0.1, 0.9) * hh
        x0 = r.uniform(-0.05, 0.3) * ww
        x1 = x0 + r.uniform(0.3, 0.9) * ww
        lw = int(r.uniform(1, 6) * S)
        c = tuple(int(v * r.uniform(0.75, 1.25)) for v in color)
        c = tuple(min(255, v) for v in c)
        d.line([(x0, y), (x1, y + r.uniform(-6, 6) * S)], fill=c + (int(r.uniform(90, 255)),), width=lw)
    # splatter
    for i in range(50):
        x, y = r.uniform(-0.02, 1.02) * ww, r.uniform(0.0, 1.0) * hh
        s = r.uniform(1, 5) * S
        d.ellipse([x - s, y - s, x + s, y + s], fill=color + (int(r.uniform(120, 255)),))
    # carve gaps for a dry texture
    for i in range(40):
        y = r.uniform(0.15, 0.85) * hh
        x0 = r.uniform(0.0, 1.0) * ww
        d.line([(x0, y), (x0 + r.uniform(0.05, 0.25) * ww, y)], fill=(0, 0, 0, 0), width=int(r.uniform(1, 3) * S))
    im = im.resize((w, h), Image.LANCZOS)
    return to_sprite(im)

def blit(frame, spr, cx, cy, scale=1.0, alpha=1.0, blur_x=0, blur_y=0, add=False):
    """Composite a premultiplied sprite centred at (cx,cy)."""
    if alpha <= 0.003 or scale <= 0.01: return
    sh, sw = spr.shape[:2]
    if abs(scale - 1) > 1e-3:
        nw, nh = max(1, int(sw * scale)), max(1, int(sh * scale))
        spr = cv2.resize(spr, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    if blur_x > 1 or blur_y > 1:
        spr = cv2.blur(spr, (max(1, int(blur_x)), max(1, int(blur_y))))
    sh, sw = spr.shape[:2]
    x0, y0 = int(round(cx - sw / 2)), int(round(cy - sh / 2))
    fx0, fy0, fx1, fy1 = max(0, x0), max(0, y0), min(W, x0 + sw), min(H, y0 + sh)
    if fx1 <= fx0 or fy1 <= fy0: return
    s = spr[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0] * alpha
    roi = frame[fy0:fy1, fx0:fx1]
    if add:
        roi += s[..., :3]
    else:
        roi *= (1 - s[..., 3:4]); roi += s[..., :3]

def slam(frame, spr, cx, cy, lt, alpha=1.0, base_scale=1.0, from_scale=2.4, dur=0.28):
    """Big title slam-in: scales down from from_scale with motion blur."""
    if lt < 0: return
    k = eo5(lt / dur)
    sc = lerp(from_scale, 1.0, k) * base_scale
    bl = (1 - k) * 40
    blit(frame, spr, cx, cy, sc, alpha * clamp(lt / 0.06), blur_x=bl, blur_y=bl * 0.3)

# ---------------------------------------------------------------- glow sprites
def _gauss(n, sig):
    x = np.linspace(-1, 1, n, dtype=np.float32)
    return np.exp(-(x / sig) ** 2)
FLARE = (_gauss(256, 0.35)[:, None] * _gauss(256, 0.35)[None, :])[..., None] * np.ones(3, np.float32)
CORE = (_gauss(128, 0.18)[:, None] * _gauss(128, 0.18)[None, :])[..., None] * np.ones(3, np.float32)
STREAK = (_gauss(48, 0.25)[:, None] * _gauss(1400, 0.45)[None, :] ** 2)[..., None] * np.ones(3, np.float32)

def add_glow(frame, sprite, cx, cy, sx, sy, color, inten):
    if inten <= 0.005: return
    nw, nh = max(2, int(sprite.shape[1] * sx)), max(2, int(sprite.shape[0] * sy))
    g = cv2.resize(sprite, (nw, nh), interpolation=cv2.INTER_LINEAR) * (np.asarray(color, np.float32) * inten)
    x0, y0 = int(cx - nw / 2), int(cy - nh / 2)
    fx0, fy0, fx1, fy1 = max(0, x0), max(0, y0), min(W, x0 + nw), min(H, y0 + nh)
    if fx1 <= fx0 or fy1 <= fy0: return
    frame[fy0:fy1, fx0:fx1] += g[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]

def headlight(frame, x, y, scale, inten, col=(1.0, 0.97, 0.9), streak=True, streak_col=(0.75, 0.6, 1.0)):
    add_glow(frame, FLARE, x, y, 1.6 * scale, 1.6 * scale, col, 0.9 * inten)
    add_glow(frame, CORE, x, y, 0.9 * scale, 0.9 * scale, (1, 1, 1), 1.2 * inten)
    if streak:
        add_glow(frame, STREAK, x, y, 1.1 * scale, 1.2 * scale, streak_col, 0.7 * inten)

def wet_reflection(frame, x, y, length, width, col, inten):
    """vertical smeared reflection on wet asphalt under a light."""
    g = (_gauss(64, 0.5)[None, :] * np.linspace(1, 0, 256, dtype=np.float32)[:, None] ** 1.5)[..., None] * np.ones(3, np.float32)
    add_glow(frame, g, x, y + length / 2, width / 64, length / 256, col, inten)

# ---------------------------------------------------------------- snow
_rs = np.random.default_rng(5)
SNOW_FAR = _rs.random((420, 4)).astype(np.float32)
SNOW_NEAR = _rs.random((26, 4)).astype(np.float32)

def snow(frame, t, speed=1.0, wind=0.25, streak=(0.0, 0.0), amount=1.0, tt=None):
    tt = t if tt is None else tt
    lay = np.zeros((H // 2, W // 2), np.float32)
    for fx, fy, sp, ph in SNOW_FAR[: int(420 * amount)]:
        v = 0.35 + sp
        y = (fy + tt * 0.11 * v * speed) % 1.0
        x = (fx + tt * 0.035 * v * wind + 0.01 * math.sin(tt * 1.7 + ph * 6)) % 1.0
        X, Y = x * W / 2, y * H / 2
        r = 0.6 + sp * 1.6
        dx, dy = streak[0] * v, streak[1] * v
        if abs(dx) + abs(dy) > 1.5:
            cv2.line(lay, (int(X), int(Y)), (int(X + dx), int(Y + dy)), float(0.55 + 0.45 * sp), max(1, int(r)), cv2.LINE_AA)
        else:
            cv2.circle(lay, (int(X), int(Y)), max(1, int(round(r))), float(0.5 + 0.5 * sp), -1, cv2.LINE_AA)
    lay = cv2.GaussianBlur(lay, (0, 0), 0.8)
    lay = cv2.resize(lay, (W, H), interpolation=cv2.INTER_LINEAR)
    frame += lay[..., None] * np.array([0.85, 0.87, 1.0], np.float32) * 0.8
    # near, out-of-focus flakes (bokeh)
    near = np.zeros((H // 4, W // 4), np.float32)
    for fx, fy, sp, ph in SNOW_NEAR[: int(26 * amount)]:
        v = 0.6 + sp
        y = (fy + tt * 0.22 * v * speed) % 1.0
        x = (fx + tt * 0.06 * v * wind) % 1.0
        cv2.circle(near, (int(x * W / 4), int(y * H / 4)), int(3 + sp * 6), float(0.25 + 0.2 * sp), -1, cv2.LINE_AA)
    near = cv2.GaussianBlur(near, (0, 0), 2.5)
    frame += cv2.resize(near, (W, H))[..., None] * np.array([0.9, 0.88, 1.0], np.float32)

def spray(frame, M, src_x, src_y, t, n=60, seed=1, dirx=-1.0, power=1.0, scale_px=1.0):
    """snow spray from a wheel (in plate coordinates), mapped through the camera."""
    r = random.Random(seed)
    lay = np.zeros((H // 2, W // 2), np.float32)
    for i in range(n):
        ph = r.random(); life = 0.7 + 0.5 * r.random()
        age = ((t + ph * life) % life) / life
        vx = dirx * (250 + 600 * r.random()) * power
        vy = -(150 + 350 * r.random()) * power
        px = src_x + vx * age
        py = src_y + vy * age + 500 * age * age
        X, Y = map_pt(M, px, py)
        rad = (6 + 40 * age) * scale_px * (0.5 + r.random())
        cv2.circle(lay, (int(X / 2), int(Y / 2)), max(1, int(rad / 2)), float(0.35 * (1 - age) ** 1.3), -1, cv2.LINE_AA)
    lay = cv2.GaussianBlur(lay, (0, 0), 3)
    frame += cv2.resize(lay, (W, H))[..., None] * np.array([0.92, 0.92, 1.0], np.float32)

# ---------------------------------------------------------------- UI pieces
def rrect(d, box, r, **kw): d.rounded_rectangle(box, r, **kw)

@lru_cache(maxsize=None)
def glass_panel(w, h, radius=38, border=P_RGB_L, glow=True):
    pad = 50
    im = Image.new("RGBA", (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
    if glow:
        g = Image.new("RGBA", im.size, (0, 0, 0, 0))
        rrect(ImageDraw.Draw(g), [pad, pad, pad + w, pad + h], radius, outline=P_RGB + (220,), width=10)
        im = Image.alpha_composite(im, g.filter(ImageFilter.GaussianBlur(16)))
    d = ImageDraw.Draw(im)
    rrect(d, [pad, pad, pad + w, pad + h], radius, fill=(14, 10, 24, 200), outline=border + (170,), width=3)
    # top sheen
    sh = Image.new("RGBA", im.size, (0, 0, 0, 0))
    rrect(ImageDraw.Draw(sh), [pad + 3, pad + 3, pad + w - 3, pad + 90], radius, fill=(255, 255, 255, 14))
    im = Image.alpha_composite(im, sh)
    return to_sprite(im)

def frost_behind(frame, cx, cy, w, h, radius=38, amount=1.0):
    """blur what is behind a glass panel."""
    x0, y0, x1, y1 = int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 - x0 < 4 or y1 - y0 < 4 or amount <= 0: return
    roi = frame[y0:y1, x0:x1]
    small = cv2.resize(roi, ((x1 - x0) // 8 + 1, (y1 - y0) // 8 + 1), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 2.5)
    bl = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    frame[y0:y1, x0:x1] = roi * (1 - amount) + bl * amount

def icon_gift(size, col=P_RGB_L):
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0)); d = ImageDraw.Draw(im); s = size
    rrect(d, [s * .12, s * .38, s * .88, s * .9], s * .06, fill=col + (255,))
    rrect(d, [s * .06, s * .26, s * .94, s * .42], s * .05, fill=tuple(min(255, int(c * 1.15)) for c in col) + (255,))
    d.rectangle([s * .44, s * .26, s * .56, s * .9], fill=(255, 220, 120, 255))
    d.ellipse([s * .22, s * .06, s * .5, s * .3], outline=(255, 220, 120, 255), width=max(2, int(s * .06)))
    d.ellipse([s * .5, s * .06, s * .78, s * .3], outline=(255, 220, 120, 255), width=max(2, int(s * .06)))
    return im

def icon_crown(size, col=(255, 200, 70)):
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0)); d = ImageDraw.Draw(im); s = size
    d.polygon([(s * .08, s * .78), (s * .12, s * .3), (s * .33, s * .52), (s * .5, s * .18), (s * .67, s * .52), (s * .88, s * .3), (s * .92, s * .78)], fill=col + (255,))
    d.rectangle([s * .08, s * .8, s * .92, s * .9], fill=col + (255,))
    for x, y in [(.12, .28), (.5, .16), (.88, .28)]:
        d.ellipse([s * x - s * .06, s * y - s * .06, s * x + s * .06, s * y + s * .06], fill=col + (255,))
    return im

def icon_cash(size):
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0)); d = ImageDraw.Draw(im); s = size
    for k in range(3):
        o = k * s * .07
        rrect(d, [s * .08 + o, s * .3 - o, s * .82 + o, s * .72 - o], s * .05, fill=(40 + 30 * k, 150 + 30 * k, 70 + 10 * k, 255), outline=(20, 90, 40, 255), width=2)
    d.ellipse([s * .38, s * .36, s * .66, s * .58], outline=(200, 255, 200, 255), width=max(2, int(s * .04)))
    return im

def icon_like(size, col=(255, 255, 255)):
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0)); d = ImageDraw.Draw(im); s = size
    rrect(d, [s * .08, s * .45, s * .28, s * .92], s * .04, fill=col + (255,))
    d.polygon([(s * .32, s * .45), (s * .5, s * .1), (s * .6, s * .14), (s * .56, s * .4), (s * .9, s * .42), (s * .92, s * .52),
               (s * .82, s * .9), (s * .32, s * .92)], fill=col + (255,))
    return im

def icon_bookmark(size, col=(255, 255, 255)):
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0)); d = ImageDraw.Draw(im); s = size
    d.polygon([(s * .22, s * .08), (s * .78, s * .08), (s * .78, s * .92), (s * .5, s * .68), (s * .22, s * .92)], fill=col + (255,))
    return im

def icon_play(size, col=(255, 255, 255)):
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0)); d = ImageDraw.Draw(im); s = size
    rrect(d, [s * .06, s * .2, s * .94, s * .8], s * .16, fill=col + (255,))
    d.polygon([(s * .42, s * .36), (s * .66, s * .5), (s * .42, s * .64)], fill=P_RGB + (255,))
    return im

def car_thumb(plate, box, w, h):
    img = PL[plate][0]
    x0, y0, x1, y1 = box
    crop = img[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    s = max(w / cw, h / ch)
    crop = cv2.resize(crop, (int(cw * s), int(ch * s)), interpolation=cv2.INTER_AREA)
    oy, ox = (crop.shape[0] - h) // 2, (crop.shape[1] - w) // 2
    crop = crop[oy:oy + h, ox:ox + w]
    return Image.fromarray((np.clip(crop, 0, 1) * 255).astype(np.uint8))

CAR_BOX = {"bmw": ("p1", (240, 520, 1880, 1280)), "amg": ("p4", (110, 470, 1290, 1100)),
           "charger": ("p4", (1120, 500, 2240, 1010)), "vaz": ("p4", (2160, 540, 3140, 960))}

def plate_number(w, h, text="А777МР", region="77"):
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    rrect(d, [0, 0, w - 1, h - 1], int(h * .12), fill=(245, 245, 245, 255), outline=(20, 20, 20, 255), width=max(2, int(h * .05)))
    d.line([(w * .72, h * .08), (w * .72, h * .92)], fill=(20, 20, 20, 255), width=max(2, int(h * .04)))
    f = font(F_BOLD, int(h * .62))
    d.text((w * .36, h * .53), text, font=f, fill=(10, 10, 10, 255), anchor="mm")
    d.text((w * .86, h * .4), region, font=font(F_BOLD, int(h * .42)), fill=(10, 10, 10, 255), anchor="mm")
    fy = h * .72
    for k, c in enumerate([(255, 255, 255), (30, 60, 200), (220, 30, 40)]):
        d.rectangle([w * .9, fy + k * h * .05, w * .96, fy + (k + 1) * h * .05], fill=c + (255,))
    d.text((w * .82, fy + h * .07), "RUS", font=font(F_BOLD, int(h * .13)), fill=(10, 10, 10, 255), anchor="mm")
    return im

@lru_cache(maxsize=None)
def reward_card(kind, w=300, h=360, glow_col=P_RGB, gold=False):
    pad = 40
    im = Image.new("RGBA", (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
    gc = (255, 190, 60) if gold else glow_col
    g = Image.new("RGBA", im.size, (0, 0, 0, 0))
    rrect(ImageDraw.Draw(g), [pad, pad, pad + w, pad + h], 26, outline=gc + (255,), width=12)
    im = Image.alpha_composite(im, g.filter(ImageFilter.GaussianBlur(14)))
    d = ImageDraw.Draw(im)
    rrect(d, [pad, pad, pad + w, pad + h], 26, fill=(36, 18, 64, 255) if not gold else (60, 36, 10, 255),
          outline=tuple(min(255, int(c * 1.2)) for c in gc) + (255,), width=4)
    inner = (pad + 14, pad + 14, pad + w - 14, pad + h - 80)
    iw, ih = inner[2] - inner[0], inner[3] - inner[1]
    label = ""
    if kind in CAR_BOX:
        pl, box = CAR_BOX[kind]
        th = car_thumb(pl, box, iw, ih).convert("RGBA")
        m = Image.new("L", (iw, ih), 0); rrect(ImageDraw.Draw(m), [0, 0, iw - 1, ih - 1], 18, fill=255)
        im.paste(th, inner[:2], m)
        label = {"bmw": "BMW E38", "amg": "MERCEDES-AMG", "charger": "CHARGER", "vaz": "ВАЗ КЛАССИКА"}[kind]
    elif kind == "plate":
        pn = plate_number(int(iw * .92), int(iw * .92 * 0.24))
        im.alpha_composite(pn, (int(inner[0] + iw * .04), int(inner[1] + ih / 2 - pn.size[1] / 2)))
        label = "НОМЕРА"
    elif kind == "skin":
        br = Image.fromarray((np.clip(brush_sprite(iw, ih // 2, 7)[..., :4], 0, 1) * 255).astype(np.uint8), "RGBA")
        for k in range(3):
            im.alpha_composite(br.rotate(-25 + k * 6, expand=False), (inner[0], inner[1] + k * ih // 5))
        label = "СКИН"
    elif kind == "crown":
        c = icon_crown(int(ih * .8)); im.alpha_composite(c, (int(inner[0] + iw / 2 - c.size[0] / 2), int(inner[1] + ih * .1)))
        label = "ЭКСКЛЮЗИВ"
    elif kind == "cash":
        c = icon_cash(int(ih * .9)); im.alpha_composite(c, (int(inner[0] + iw / 2 - c.size[0] / 2), int(inner[1] + ih * .05)))
        label = "100 000 ₽"
    f = font(F_BOLD, 30)
    d.text((pad + w / 2, pad + h - 42), label.replace("₽", ""), font=f, fill=(255, 255, 255, 255), anchor="mm")
    if "₽" in label:
        tw = f.getlength(label.replace("₽", ""))
        d.text((pad + w / 2 + tw / 2 + 4, pad + h - 42), "₽", font=font(F_SYM, 30), fill=(255, 255, 255, 255), anchor="lm")
    return to_sprite(im)

# ---------------------------------------------------------------- light streak intro
_rl = random.Random(3)
STREAKS = [(_rl.uniform(-0.6, 0.6), _rl.uniform(0.0, 1.0), _rl.uniform(0.6, 1.4), _rl.random() < 0.5) for _ in range(16)]

def light_streaks(frame, t, t0, t1, inten=1.0):
    """comet-like light streaks converging to centre (opening, like the reference)."""
    if not (t0 <= t < t1): return
    k = (t - t0) / (t1 - t0)
    lay = np.zeros((H // 2, W // 2, 3), np.float32)
    for ang, ph, sp, white in STREAKS:
        u = (k * sp + ph) % 1.0
        a = ang * math.pi + (math.pi if ph > 0.5 else 0)
        dist = (1 - u) ** 2 * 900
        x, y = W / 4 + math.cos(a) * dist, H / 4 + math.sin(a) * dist * 1.6
        tail = 140 * (0.5 + u)
        x2, y2 = x + math.cos(a) * tail, y + math.sin(a) * tail * 1.6
        col = (1.0, 1.0, 1.0) if white else tuple(PURPLE_L.tolist())
        cv2.line(lay, (int(x), int(y)), (int(x2), int(y2)), col, 3, cv2.LINE_AA)
    glow = cv2.GaussianBlur(lay, (0, 0), 6)
    lay = cv2.resize(lay + glow * 2.5, (W, H))
    frame += lay * inten

# ---------------------------------------------------------------- transitions
def whip(frame, amount, horizontal=True):
    if amount <= 0.02: return frame
    k = int(4 + amount * 160)
    return cv2.blur(frame, (k, 1) if horizontal else (1, k))

def zoom_blur(frame, amount, cx=W / 2, cy=H / 2):
    if amount <= 0.02: return frame
    acc = frame.copy(); n = 6
    for i in range(1, n):
        s = 1 + amount * 0.06 * i
        M = np.array([[s, 0, cx - s * cx], [0, s, cy - s * cy]], np.float32)
        acc += cv2.warpAffine(frame, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    return acc / n

# ================================================================ SCENES
def scene1(t):
    """0–3 s: light streaks → white BMW drifting, headlights into camera, titles."""
    f = np.zeros((H, W, 3), np.float32)
    if t < 0.45:
        f[:] = np.array([0.02, 0.01, 0.04], np.float32)
        light_streaks(f, t, 0.0, 0.5, 1.0)
        return f, 0.0
    lt = t - 0.45
    def cam(tt):
        u = tt - 0.45
        zk = eo5(u / 0.55)
        z2 = lerp(1.25, 0.9, eio(u / 2.4))
        z = lerp(3.2, z2, zk)
        cx = lerp(1420, 1080, zk) + 60 * u
        cy = lerp(950, 820, zk) - 10 * u
        rot = math.radians(lerp(-6, 0, zk) + 1.6 * math.sin(u * 2.4))
        return cx, cy, z, rot
    n = 5 if lt < 0.6 else 2
    cams = [cam(t - i * 0.012) for i in range(n)]
    f, M = render_plate("p1", cams)
    z = cams[0][2]
    pb = pulse(t)
    # headlights + wet reflections
    for (hx, hy), w8 in zip([(1065, 921), (1191, 936), (1680, 969), (1749, 975)], [1, 1, .9, .9]):
        X, Y = map_pt(M, hx, hy)
        inten = (0.55 + 0.45 * pb + 1.4 * math.exp(-lt * 4)) * w8
        headlight(f, X, Y, 0.45 * z / 1.2, inten)
        wet_reflection(f, X, Y + 40 * z, 520 * z / 1.38, 90 * z / 1.38, (1.0, 0.92, 0.85), 0.25 * inten)
    # snow spray from wheels (drift)
    spray(f, M, 760, 1090, t, n=70, seed=2, dirx=-1.0, power=1.3, scale_px=z / 1.38)
    spray(f, M, 420, 950, t + 0.3, n=50, seed=5, dirx=-0.6, power=1.0, scale_px=z / 1.38)
    snow(f, t, speed=1.6, wind=-1.2, streak=(-6, 10), amount=1.0)
    # titles
    s1 = text_sprite("7 ДЕКАБРЯ", 140, (255, 255, 255), F_TITLE)
    s2 = text_sprite("STATE RP", 175, (255, 255, 255), F_TITLE, glow=P_RGB)
    br = brush_sprite(900, 200, 11)
    slam(f, s1, W / 2, 330, t - 0.55)
    if t > 0.95:
        k = eo5((t - 0.95) / 0.25)
        blit(f, br, W / 2 - (1 - k) * 600, 540, 1.0, k, blur_x=(1 - k) * 60)
    slam(f, s2, W / 2 + 10, 530, t - 1.0)
    if t > 1.0:
        txt = text_sprite("ЗИМНЕЕ ОБНОВЛЕНИЕ", 54, (255, 255, 255), F_BOLDI)
        a = eo((t - 1.45) / 0.3)
        blit(f, txt, W / 2, 700 + (1 - a) * 30, 1.0, a)
    hit = max(math.exp(-(t - 0.45) * 8), 0.8 * math.exp(-max(0, t - 0.55) * 10) if t > 0.55 else 0,
              0.8 * math.exp(-max(0, t - 1.0) * 10) if t > 1.0 else 0)
    return f, hit

def scene2(t):
    """3–7 s: advent calendar UI over the winter city."""
    lt = t - 3.0
    cams = [(2250 + 40 * lt, 420 - 10 * lt, 1.9 + 0.05 * lt, 0.0)]
    f, _ = render_plate("p5", cams, bg_dark=0.4)
    f = cv2.GaussianBlur(f, (0, 0), 7) * 0.7          # deep bokeh background
    f += pulse(t) * 0.04 * PURPLE
    snow(f, t, speed=0.8, wind=0.4, amount=0.8)
    # titles
    slam(f, text_sprite("АДВЕНТ-КАЛЕНДАРЬ", 76, (255, 255, 255), F_TITLE), W / 2, 215, lt - 0.05, from_scale=1.8)
    if lt > 0.35:
        k = eo5((lt - 0.35) / 0.25)
        blit(f, brush_sprite(900, 170, 21), W / 2 + (1 - k) * 500, 352, 1.0, k, blur_x=(1 - k) * 60)
    slam(f, text_sprite("24 ДНЯ НАГРАД!", 92, (255, 255, 255), F_TITLE), W / 2, 350, lt - 0.38, from_scale=1.8)
    # panel slides in
    pk = eo5((lt - 0.1) / 0.45)
    pw, ph = 980, 1000
    pcx, pcy = W / 2, 1000 + (1 - pk) * 900
    frost_behind(f, pcx, pcy, pw, ph, amount=pk)
    blit(f, glass_panel(pw, ph), pcx, pcy, 1.0, pk, blur_y=(1 - pk) * 80)
    if pk > 0.05:
        top = pcy - ph / 2
        gi = to_sprite(icon_gift(84))
        blit(f, gi, pcx - pw / 2 + 80, top + 80, 1.0, pk)
        blit(f, text_sprite("Адвент-календарь", 46, (255, 255, 255), F_BOLD, shadow=False, pad=10), pcx - 90, top + 62, 1.0, pk)
        blit(f, text_sprite("24 дня наград!", 34, P_RGB_L, F_UI, shadow=False, pad=10), pcx - 165, top + 112, 1.0, pk)
        # grid 4 x 6
        cw, chh, gap = 196, 112, 18
        gx0 = pcx - (4 * cw + 3 * gap) / 2 + cw / 2
        gy0 = top + 200 + chh / 2
        clicks = [(0.55, 0), (1.55, 9), (2.55, 18)]
        cascade = clamp((lt - 3.15) / 0.7)
        opened = {i for (tc, i) in clicks if lt > tc}
        for i in range(24):
            c, r = i % 4, i // 4
            x, y = gx0 + c * (cw + gap), gy0 + r * (chh + gap)
            appear = eo((lt - 0.25 - i * 0.012) / 0.25) * pk
            is_open = i in opened or cascade * 24 > i
            blit(f, cell_sprite(i + 1, is_open), x, y, 0.9 + 0.1 * appear, appear)
        # cursor
        tgt = None
        for k_, (tc, i) in enumerate(clicks):
            if lt < tc + 0.5: tgt = (k_, tc, i); break
        if tgt and lt > 0.3:
            k_, tc, i = tgt
            prev = clicks[k_ - 1][1] if k_ > 0 else 23
            px, py = gx0 + (prev % 4) * (cw + gap) + 40, gy0 + (prev // 4) * (chh + gap) + 30
            qx, qy = gx0 + (i % 4) * (cw + gap) + 40, gy0 + (i // 4) * (chh + gap) + 30
            mk = eio((lt - (tc - 0.45)) / 0.4)
            cx_, cy_ = lerp(px, qx, mk), lerp(py, qy, mk)
            press = 0.85 if 0 < lt - tc < 0.12 else 1.0
            blit(f, CURSOR, cx_ + 18, cy_ + 24, press)
            if 0 < lt - tc < 0.35:
                rr = (lt - tc) / 0.35
                ring = ring_sprite()
                blit(f, ring, cx_, cy_, 0.4 + rr * 1.6, (1 - rr) * 0.9, add=True)
    # reward pop + notifications
    rewards = [(0.6, "cash", "+100 000 ₽"), (1.6, "car", "Новая машина!"), (2.6, "crown", "Эксклюзивный скин!")]
    for k_, (tc, kind, label) in enumerate(rewards):
        if lt < tc: continue
        u = lt - tc
        # big pop card in the middle (0..0.55s), then shrinks to notification
        if u < 0.7:
            s = back(u / 0.25) if u < 0.45 else lerp(1.0, 0.0, eo((u - 0.45) / 0.25))
            add_glow(f, FLARE, W / 2, 1000, 3.0 * s, 3.0 * s, tuple(PURPLE_L.tolist()), 0.6 * s)
            blit(f, pop_card(kind, label), W / 2, 1000, max(0.01, s), clamp(s * 1.5))
        nk = eo5((u - 0.55) / 0.3)
        if nk > 0:
            ny = 1580 + k_ * 112
            blit(f, notif_sprite(kind, label), W / 2 + (1 - nk) * 700, ny, 1.0, nk, blur_x=(1 - nk) * 60)
    hit = 0.9 * math.exp(-lt * 10)
    return f, hit

@lru_cache(maxsize=None)
def cell_sprite(n, is_open):
    w, h = 196, 112
    pad = 26
    im = Image.new("RGBA", (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
    if is_open:
        g = Image.new("RGBA", im.size, (0, 0, 0, 0))
        rrect(ImageDraw.Draw(g), [pad, pad, pad + w, pad + h], 20, fill=P_RGB + (255,))
        im = Image.alpha_composite(im, g.filter(ImageFilter.GaussianBlur(12)))
    d = ImageDraw.Draw(im)
    rrect(d, [pad, pad, pad + w, pad + h], 20, fill=(120, 50, 230, 255) if is_open else (30, 26, 44, 235),
          outline=(210, 170, 255, 255) if is_open else (70, 60, 100, 255), width=3)
    if is_open:
        gi = icon_gift(80, col=(255, 236, 250)); im.alpha_composite(gi, (int(pad + w / 2 - 40), int(pad + h / 2 - 42)))
    else:
        d.text((pad + w / 2, pad + h / 2), str(n), font=font(F_BOLD, 52), fill=(235, 230, 245, 255), anchor="mm")
    return to_sprite(im)

def _cursor():
    im = Image.new("RGBA", (60, 80), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    pts = [(4, 4), (4, 62), (18, 48), (30, 74), (40, 69), (28, 44), (48, 44)]
    d.polygon(pts, fill=(255, 255, 255, 255), outline=(0, 0, 0, 255))
    return to_sprite(im)
CURSOR = _cursor()

@lru_cache(maxsize=None)
def ring_sprite():
    im = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
    ImageDraw.Draw(im).ellipse([10, 10, 150, 150], outline=P_RGB_L + (255,), width=8)
    return to_sprite(im.filter(ImageFilter.GaussianBlur(2)))

@lru_cache(maxsize=None)
def pop_card(kind, label):
    w, h = 620, 470
    pad = 50
    im = Image.new("RGBA", (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
    g = Image.new("RGBA", im.size, (0, 0, 0, 0))
    rrect(ImageDraw.Draw(g), [pad, pad, pad + w, pad + h], 40, outline=P_RGB + (255,), width=16)
    im = Image.alpha_composite(im, g.filter(ImageFilter.GaussianBlur(20)))
    d = ImageDraw.Draw(im)
    rrect(d, [pad, pad, pad + w, pad + h], 40, fill=(22, 12, 40, 245), outline=P_RGB_L + (255,), width=4)
    d.text((pad + w / 2, pad + 48), "НАГРАДА ПОЛУЧЕНА", font=font(F_BOLD, 32), fill=P_RGB_L + (255,), anchor="mm")
    area = (pad + 40, pad + 90, pad + w - 40, pad + h - 110)
    aw, ah = area[2] - area[0], area[3] - area[1]
    if kind == "car":
        th = car_thumb(*CAR_BOX["amg"], aw, ah).convert("RGBA")
        m = Image.new("L", (aw, ah), 0); rrect(ImageDraw.Draw(m), [0, 0, aw - 1, ah - 1], 24, fill=255)
        im.paste(th, area[:2], m)
    elif kind == "cash":
        c = icon_cash(ah); im.alpha_composite(c, (int(area[0] + aw / 2 - ah / 2), area[1]))
    else:
        c = icon_crown(ah, col=(200, 120, 255)); im.alpha_composite(c, (int(area[0] + aw / 2 - ah / 2), area[1]))
    f = font(F_TITLE, 52)
    base = label.replace("₽", "").strip()
    tw = f.getlength(base) + (40 if "₽" in label else 0)
    d.text((pad + w / 2 - tw / 2, pad + h - 55), base, font=f, fill=(255, 255, 255, 255), anchor="lm")
    if "₽" in label:
        d.text((pad + w / 2 + tw / 2 - 30, pad + h - 55), "₽", font=font(F_SYM, 52), fill=(255, 255, 255, 255), anchor="lm")
    return to_sprite(im)

@lru_cache(maxsize=None)
def notif_sprite(kind, label):
    w, h = 760, 96
    pad = 30
    im = Image.new("RGBA", (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
    g = Image.new("RGBA", im.size, (0, 0, 0, 0))
    rrect(ImageDraw.Draw(g), [pad, pad, pad + w, pad + h], 24, outline=P_RGB + (255,), width=8)
    im = Image.alpha_composite(im, g.filter(ImageFilter.GaussianBlur(10)))
    d = ImageDraw.Draw(im)
    rrect(d, [pad, pad, pad + w, pad + h], 24, fill=(26, 14, 46, 240), outline=P_RGB_L + (230,), width=3)
    if kind == "car":
        th = car_thumb(*CAR_BOX["amg"], 120, 74).convert("RGBA"); im.paste(th, (pad + 16, pad + 11))
    elif kind == "cash":
        im.alpha_composite(icon_cash(84), (pad + 30, pad + 6))
    else:
        im.alpha_composite(icon_crown(80, col=(200, 120, 255)), (pad + 32, pad + 8))
    base = label.replace("₽", "").strip()
    f = font(F_BOLD, 40)
    d.text((pad + 170, pad + h / 2), base, font=f, fill=(255, 255, 255, 255), anchor="lm")
    if "₽" in label:
        d.text((pad + 170 + f.getlength(base) + 10, pad + h / 2), "₽", font=font(F_SYM, 40), fill=(255, 255, 255, 255), anchor="lm")
    return to_sprite(im)

BP_CARDS = ["bmw", "plate", "skin", "crown", "amg", "cash", "charger", "skin", "vaz", "crown"]

def scene3(t):
    """7–11 s: battle pass — giant level counter, XP bar, reward card carousel."""
    lt = t - 7.0
    f = np.zeros((H, W, 3), np.float32)
    # background: dark purple with animated brush strokes + crown watermark
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    f += (np.array([0.05, 0.02, 0.10], np.float32) * (1 - yy) + np.array([0.02, 0.01, 0.04], np.float32) * yy)
    for k, (y, sd, sp) in enumerate([(300, 31, 1.0), (1650, 32, -0.8), (1850, 33, 1.3)]):
        br = brush_sprite(1400, 260, sd, color=(90, 30, 180))
        blit(f, br, W / 2 + sp * (lt * 120 - 200), y, 1.0, 0.55)
    blit(f, crown_wm(), W / 2 + 260, 560, 1.0, 0.18)
    snow(f, t, speed=0.6, wind=0.3, amount=0.5)
    # titles
    slam(f, text_sprite("НОВЫЙ БАТЛПАСС", 86, (255, 255, 255), F_TITLE), W / 2, 200, lt - 0.05, from_scale=1.8)
    a = eo((lt - 0.4) / 0.3)
    blit(f, text_sprite("КАЧАЙ УРОВНИ", 64, P_RGB_L, F_TITLE, glow=P_RGB), W / 2 - (1 - a) * 300, 330, 1.0, a, blur_x=(1 - a) * 50)
    a2 = eo((lt - 0.55) / 0.3)
    blit(f, text_sprite("И ЗАБИРАЙ НАГРАДЫ!", 64, P_RGB_L, F_TITLE, glow=P_RGB), W / 2 + (1 - a2) * 300, 420, 1.0, a2, blur_x=(1 - a2) * 50)
    # panel
    pk = eo5((lt - 0.15) / 0.45)
    pw, ph = 1000, 1180
    pcx, pcy = W / 2, 1150 + (1 - pk) * 900
    frost_behind(f, pcx, pcy, pw, ph, amount=pk * 0.5)
    blit(f, glass_panel(pw, ph), pcx, pcy, 1.0, pk, blur_y=(1 - pk) * 80)
    if pk < 0.05: return f, 0.9 * math.exp(-lt * 10)
    top = pcy - ph / 2
    blit(f, to_sprite(icon_crown(90, col=P_RGB_L)), pcx - pw / 2 + 90, top + 85, 1.0, pk)
    blit(f, text_sprite("Батлпасс", 50, (255, 255, 255), F_BOLD, shadow=False, pad=10), pcx - pw / 2 + 290, top + 65, 1.0, pk)
    blit(f, text_sprite("Новый сезон", 34, (190, 180, 210), F_UI, shadow=False, pad=10), pcx - pw / 2 + 268, top + 115, 1.0, pk)
    # level counter (big number counting up, like the reference counters)
    prog = eio((lt - 0.5) / 3.2)
    level = 1 + int(prog * 49)
    blit(f, text_sprite("УРОВЕНЬ", 40, (200, 190, 230), F_BOLD, shadow=False, pad=10), pcx, top + 215, 1.0, pk)
    num = text_sprite(str(level), 210, (255, 255, 255), F_TITLE, glow=P_RGB, shadow=False)
    lvl_pop = pulse(t, 12) if lt > 0.5 else 0
    blit(f, num, pcx, top + 360, 1.0 + 0.06 * lvl_pop, pk)
    # XP bar
    xp_k = (prog * 49) % 1.0 if prog < 1 else 1.0
    bx0, bx1, by = pcx - 400, pcx + 400, top + 520
    bar = np.zeros((H, W, 3), np.float32)
    cv2.rectangle(f, (int(bx0), int(by - 14)), (int(bx1), int(by + 14)), (0.10, 0.07, 0.16), -1, cv2.LINE_AA)
    fill_x = int(bx0 + (bx1 - bx0) * xp_k)
    cv2.rectangle(f, (int(bx0), int(by - 14)), (fill_x, int(by + 14)), tuple((PURPLE_L * 1.1).tolist()), -1, cv2.LINE_AA)
    add_glow(f, FLARE, fill_x, by, 0.6, 0.35, tuple(PURPLE_L.tolist()), 0.9)
    blit(f, text_sprite(f"{int(xp_k * 1000)} / 1000 XP", 30, (220, 210, 240), F_BOLD, shadow=False, pad=8), bx1 - 110, by + 46, 1.0, pk)
    # carousel of reward cards
    cwid, gap = 300, 40
    scroll = eio((lt - 0.4) / 3.4) * (len(BP_CARDS) - 3) * (cwid + gap)
    cy = top + 820
    for i, kind in enumerate(BP_CARDS):
        x = pcx - (cwid + gap) + i * (cwid + gap) - scroll
        if x < -300 or x > W + 300: continue
        center = 1 - min(1, abs(x - pcx) / 600)
        unlocked = (x - pcx) < 40
        sc = 0.82 + 0.2 * center
        spr = reward_card(kind, gold=(kind == "crown"))
        blit(f, spr, x, cy, sc, pk * (0.55 + 0.45 * center))
        if unlocked and center > 0.6:
            add_glow(f, FLARE, x, cy, 1.4, 1.6, tuple(PURPLE.tolist()), 0.25 * center)
    # level-up toast
    if lt > 1.0:
        u = (lt - 1.0) % 1.0
        if u < 0.6:
            a = eo(u / 0.15) * (1 - eo((u - 0.4) / 0.2))
            blit(f, text_sprite("НОВЫЙ УРОВЕНЬ!", 46, (255, 230, 120), F_TITLE, glow=(255, 160, 40), shadow=False), pcx, top + 640 - u * 30, 1.0, a)
    return f, 0.9 * math.exp(-lt * 10)

@lru_cache(maxsize=None)
def crown_wm():
    return to_sprite(icon_crown(520, col=(120, 70, 200)))

CARS4 = [  # (t0, t1, plate, cam_from, cam_to, lights, name, stat_label, stat_value, unit)
    (11.0, 12.0, "p4", (760, 800, 1.75, -0.03), (700, 780, 2.25, 0.0), [(555, 795), (1170, 786)], "MERCEDES-AMG", "0–100 КМ/Ч", 3.2, "С"),
    (12.0, 13.0, "p4", (1820, 760, 1.85, 0.02), (1600, 760, 2.2, 0.0), [(1725, 768), (2160, 768)], "DODGE CHARGER", "МОЩНОСТЬ", 717, "Л.С."),
    (13.0, 14.0, "p4", (2560, 760, 2.3, 0.0), (2720, 740, 2.0, 0.02), [(2790, 774), (3045, 768)], "ВАЗ КЛАССИКА", "ЛЕГЕНДА", 1976, "Г."),
]

def scene4(t):
    """11–16 s: car showcase — fast push-ins with spec counters, then slow-mo wide shot."""
    lt = t - 11.0
    if t < 14.0:
        for (t0, t1, pl, c0, c1, lights, name, sl, sv, su) in CARS4:
            if t0 <= t < t1:
                u = (t - t0) / (t1 - t0)
                def cam(tt):
                    uu = clamp((tt - t0) / (t1 - t0))
                    e = eo(uu * 1.4)
                    return tuple(lerp(a, b, e) for a, b in zip(c0, c1))
                n = 5 if u < 0.25 else 2
                f, M = render_plate(pl, [cam(t - i * 0.014) for i in range(n)])
                z = cam(t)[2]
                cols = [(0.75, 0.85, 1.0)] * 2 if "CHARGER" in name or "AMG" in name else [(1, 0.95, 0.85)] * 2
                for (hx, hy), col in zip(lights, cols):
                    X, Y = map_pt(M, hx, hy)
                    inten = 0.45 + 0.4 * pulse(t) + 0.8 * math.exp(-u * 8)
                    headlight(f, X, Y, 0.38 * z / 2, inten, col=col)
                    wet_reflection(f, X, Y + 60, 500, 80, col, 0.18 * inten)
                # light sweep across car body
                sx = lerp(-200, W + 200, eio(u))
                add_glow(f, STREAK, sx, H * 0.5, 0.4, 25, (0.6, 0.45, 1.0), 0.12)
                snow(f, t, speed=1.2, wind=0.6, streak=(8, 6))
                # lower-third spec card (like reference counters)
                a = eo5((u - 0.1) / 0.25)
                blit(f, glass_panel(860, 230, 30), W / 2, 1530 + (1 - a) * 80, 1.0, a)
                blit(f, text_sprite(name, 62, (255, 255, 255), F_TITLE, shadow=False, pad=12), W / 2, 1470 + (1 - a) * 80, 1.0, a)
                val = sv * eo((u - 0.15) / 0.6)
                vs = (f"{val:.1f}".replace(".", ",") if isinstance(sv, float) else f"{int(val)}") + " " + su
                blit(f, text_sprite(sl, 32, (200, 180, 240), F_BOLD, shadow=False, pad=8), W / 2 - 230, 1575 + (1 - a) * 80, 1.0, a)
                blit(f, text_sprite(vs, 64, P_RGB_L, F_TITLE, glow=P_RGB, shadow=False, pad=12), W / 2 + 170, 1575 + (1 - a) * 80, 1.0, a)
                # header
                blit(f, text_sprite("НОВИНКА В АВТОСАЛОНЕ", 40, (255, 255, 255), F_BOLDI), W / 2, 230, 1.0, eo(u / 0.2))
                hit = 0.7 * math.exp(-u * 12)
                return f, hit
    # 14–16: slow motion wide shot of the showroom, all three cars
    u = (t - 14.0) / 2.0
    def cam(tt):
        uu = clamp((tt - 14.0) / 2.0)
        return (lerp(800, 2650, eio(uu)), lerp(760, 720, uu), lerp(1.3, 1.45, eio(uu)), math.radians(lerp(-1.5, 1.5, uu)))
    f, M = render_plate("p4", [cam(t)])
    z = cam(t)[2]
    for (hx, hy), col in [((555, 795), (0.8, 0.88, 1)), ((1170, 786), (0.8, 0.88, 1)), ((1725, 768), (0.6, 0.75, 1)),
                          ((2160, 768), (0.6, 0.75, 1)), ((2790, 774), (1, .95, .85)), ((3045, 768), (1, .95, .85))]:
        X, Y = map_pt(M, hx, hy)
        headlight(f, X, Y, 0.3 * z / 1.3, 0.5 + 0.15 * math.sin(t * 3 + hx), col=col)
        wet_reflection(f, X, Y + 50, 420, 70, col, 0.12)
    sx, sy = map_pt(M, 2550, 300)
    add_glow(f, FLARE, sx, sy, 3.5, 1.2, (0.35, 0.35, 1.0), 0.35)           # АВТОСАЛОН neon glow
    snow(f, t, speed=0.25, wind=0.1, amount=1.0, tt=14.0 + (t - 14.0) * 0.25)
    br = brush_sprite(1000, 200, 41)
    slam(f, text_sprite("МНОГО", 132, (255, 255, 255), F_TITLE), W / 2 - 60, 300, t - 14.0, from_scale=2.0)
    if t > 14.2:
        k = eo5((t - 14.2) / 0.3)
        blit(f, br, W / 2 + (1 - k) * 600, 480, 1.0, k, blur_x=(1 - k) * 60)
    slam(f, text_sprite("НОВЫХ ТАЧЕК", 112, (255, 255, 255), F_TITLE), W / 2 + 20, 470, t - 14.25, from_scale=2.0)
    a = eo((t - 14.7) / 0.4)
    blit(f, text_sprite("ОТ КЛАССИКИ ВАЗ ДО AMG И CHARGER", 38, (255, 255, 255), F_BOLDI), W / 2, 1560, 1.0, a)
    # SLOW-MO letterbox
    lb = int(110 * eo((t - 14.0) / 0.3))
    f[:lb] *= 0.0; f[H - lb:] *= 0.0
    hit = 1.0 * math.exp(-(t - 14.0) * 6)
    return f, hit

def scene5(t):
    """16–20 s: convoy in the snowy night city, camera rises; STATE RP logo + CTA."""
    lt = t - 16.0
    def cam(tt):
        u = clamp((tt - 16.0) / 3.6)
        e = eio(u)
        return (lerp(1150, 1300, e), lerp(720, 380, e) , lerp(2.1, 1.0, eo(u * 1.2)), math.radians(lerp(2.0, -1.0, e)))
    n = 3 if lt < 0.5 else 2
    f, M = render_plate("p5", [cam(t - i * 0.015) for i in range(n)], bg_dark=0.5)
    z = cam(t)[2]
    for (hx, hy) in [(495, 645), (1425, 645), (2115, 570), (2355, 570)]:
        X, Y = map_pt(M, hx, hy)
        inten = 0.6 + 0.4 * pulse(t)
        headlight(f, X, Y, 0.35 * z / 2.0, inten, col=(1.0, 0.12, 0.08), streak_col=(1.0, 0.2, 0.25))
        wet_reflection(f, X, Y + 40, 600 * z / 2, 70 * z / 2, (1.0, 0.1, 0.08), 0.35 * inten)
    snow(f, t, speed=1.3, wind=0.5, streak=(0, 8))
    # depth of field + darken for logo legibility
    dk = eo((lt - 0.4) / 0.8)
    if dk > 0.01:
        sm = cv2.resize(f, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        bl = cv2.resize(cv2.GaussianBlur(sm, (0, 0), 3.5 * dk), (W, H))
        f = f * (1 - dk) + bl * dk
    f *= (1 - 0.5 * dk)
    # logo
    lk = lt - 0.5
    if lk > 0:
        sky = skyline_sprite()
        rev = eo(lk / 0.6)
        sh, sw = sky.shape[:2]
        part = sky.copy(); part[:, int(sw * rev):] = 0
        blit(f, part, W / 2, 610, 1.0, 1.0)
        logo = text_sprite("STATE RP", 185, (255, 255, 255), F_TITLE, glow=P_RGB, stroke=0)
        slam(f, logo, W / 2, 770, lk - 0.1, from_scale=2.2, dur=0.3)
        # light sweep across the logo (like the reference logo reveal)
        if 0.6 < lk < 1.4:
            sx = lerp(-200, W + 200, (lk - 0.6) / 0.8)
            add_glow(f, STREAK, sx, 770, 0.5, 6, (1, 1, 1), 0.35)
        if lk > 0.45:
            k = eo5((lk - 0.45) / 0.3)
            blit(f, brush_sprite(980, 330, 51), W / 2 - (1 - k) * 600, 1010, 1.0, k, blur_x=(1 - k) * 60)
            blit(f, text_sprite("ЗИМНЕЕ ОБНОВЛЕНИЕ", 60, (255, 255, 255), F_TITLE, shadow=False), W / 2, 955, 1.0, eo((lk - 0.55) / 0.25))
            slam(f, text_sprite("7 ДЕКАБРЯ", 104, (12, 6, 24), F_TITLE, shadow=False), W / 2, 1055, lk - 0.6, from_scale=1.8)
    # CTA row
    ck = lt - 1.6
    if ck > 0:
        items = [(icon_like, "ЛАЙК"), (icon_bookmark, "СОХРАНЯЙ"), (icon_play, "ЗАЛЕТАЙ НА СЕРВЕР")]
        for i, (ic, label) in enumerate(items):
            a = eo5((ck - i * 0.15) / 0.3)
            y = 1340 + i * 130
            blit(f, cta_sprite(i), W / 2 + (1 - a) * 500, y, 1.0, a, blur_x=(1 - a) * 50)
    # final fade
    if t > 19.75:
        f *= clamp((20.0 - t) / 0.25)
    hit = max(0.8 * math.exp(-lt * 10), 0.8 * math.exp(-max(0, lt - 0.6) * 8) if lt > 0.6 else 0)
    return f, hit

@lru_cache(maxsize=None)
def cta_sprite(i):
    icons = [icon_like, icon_bookmark, icon_play]
    labels = ["ЛАЙК", "СОХРАНЯЙ", "ЗАЛЕТАЙ НА СЕРВЕР"]
    w, h = 820, 110
    pad = 20
    im = Image.new("RGBA", (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
    ic = icons[i](86, col=P_RGB_L)
    gl = Image.new("RGBA", im.size, (0, 0, 0, 0)); gl.alpha_composite(icons[i](86, col=P_RGB), (pad + 20, pad + 12))
    im = Image.alpha_composite(im, gl.filter(ImageFilter.GaussianBlur(10)))
    im.alpha_composite(ic, (pad + 20, pad + 12))
    d = ImageDraw.Draw(im)
    d.text((pad + 140, pad + h / 2), labels[i], font=font(F_TITLE, 58), fill=(255, 255, 255, 255), anchor="lm",
           stroke_width=2, stroke_fill=(20, 8, 40, 255))
    return to_sprite(im)

@lru_cache(maxsize=None)
def skyline_sprite():
    w, h = 760, 140
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    r = random.Random(4)
    x = 0; pts = [(0, h)]
    while x < w:
        bw = r.randint(18, 50); bh = r.randint(20, 120) * (1 - abs(x / w - 0.5) * 1.2)
        bh = max(12, bh)
        pts += [(x, h - bh), (x + bw, h - bh)]
        x += bw
    pts += [(w, h)]
    d.line(pts, fill=(255, 255, 255, 255), width=5)
    for k in range(60):
        px, py = r.randint(10, w - 10), r.randint(h - 100, h - 10)
        d.rectangle([px, py, px + 4, py + 5], fill=(255, 255, 255, 120))
    g = im.filter(ImageFilter.GaussianBlur(6))
    out = Image.alpha_composite(g, im)
    return to_sprite(out)

SCENES = [scene1, scene2, scene3, scene4, scene5]

# ================================================================ frame
_yy, _xx = np.mgrid[0:H, 0:W].astype(np.float32)
VIG = np.clip(1.12 - (((_xx - W / 2) / (W * 0.8)) ** 2 + ((_yy - H / 2) / (H * 0.75)) ** 2) * 0.6, 0.35, 1.0)[..., None]

def frame(fi):
    t = fi / FPS
    si = max(i for i in range(5) if t >= CUTS[i])
    f, hit = SCENES[si](t)
    # ---- beat camera: punch zoom + shake (not during slow-mo) ----
    slow = SLOW[0] <= t < SLOW[1]
    bp = pulse(t, 10) * (0.2 if slow else 1.0) if t > 0.45 else 0
    zoom = 1.0 + 0.035 * bp + 0.08 * hit
    rr = random.Random(fi)
    amp = 16 * bp + 34 * hit
    ox, oy = (rr.random() - 0.5) * amp, (rr.random() - 0.5) * amp
    rot = (rr.random() - 0.5) * 0.012 * (bp + hit)
    if zoom > 1.001 or amp > 0.5:
        M = cam_matrix(W / 2 + ox, H / 2 + oy, zoom, rot)
        f = cv2.warpAffine(f, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    # ---- transitions: whip / zoom blur around cuts ----
    for c in CUTS[1:5]:
        d = t - c
        if -0.1 <= d < 0.13:
            a = 1 - abs(d) / 0.13
            f = whip(f, a * 0.9, horizontal=(c != 16.0)) if c in (3.0, 7.0) else zoom_blur(f, a * 2.0)
    if hit > 0.2:
        f = zoom_blur(f, (hit - 0.2) * 0.8)
    # ---- flash on cuts ----
    for c in CUTS[1:5] + [0.45]:
        d = t - c
        if 0 <= d < 0.1:
            f = f + (1 - d / 0.1) * 0.6
    # ---- chromatic aberration on hits ----
    sh = int(5 * bp + 14 * hit)
    if sh > 1:
        f[..., 0] = np.roll(f[..., 0], sh, axis=1)
        f[..., 2] = np.roll(f[..., 2], -sh, axis=1)
    # ---- grade: purple shadows, contrast, vignette, bloom-ish, grain ----
    lum = f.mean(axis=2, keepdims=True)
    f = f + (1 - np.clip(lum * 2, 0, 1)) * np.array([0.02, 0.0, 0.045], np.float32)
    bright = np.clip(f - 0.75, 0, None)
    small = cv2.resize(bright, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
    f += cv2.resize(cv2.GaussianBlur(small, (0, 0), 4), (W, H)) * 0.9
    f = f * VIG
    f = f / (1 + 0.25 * f)            # soft highlight roll-off
    f = np.clip((f - 0.5) * 1.08 + 0.5 + 0.01, 0, 1)
    g = np.random.default_rng(fi).normal(0, 0.018, (H // 2, W // 2)).astype(np.float32)
    f += cv2.resize(g, (W, H), interpolation=cv2.INTER_NEAREST)[..., None]
    return (np.clip(f, 0, 1) * 255).astype(np.uint8).tobytes()

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "trailer_noaudio.mp4"
    if len(sys.argv) > 2:
        os.makedirs("stills", exist_ok=True)
        for tt in sys.argv[2].split(","):
            fi = int(round(float(tt) * FPS))
            im = np.frombuffer(frame(fi), np.uint8).reshape(H, W, 3)
            cv2.imwrite(f"stills/s_{float(tt):05.2f}.jpg", cv2.resize(im, (360, 640), interpolation=cv2.INTER_AREA)[:, :, ::-1])
        sys.exit()
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "17",
                           "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    with Pool(4) as p:
        for i, b in enumerate(p.imap(frame, range(NF), chunksize=2)):
            ff.stdin.write(b)
            if i % 60 == 0: print("frame", i, flush=True)
    ff.stdin.close(); ff.wait()
    print("done", out)
