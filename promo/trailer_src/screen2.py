#!/usr/bin/env python3
"""Screen content v2 (1728x1080): smooth motion-design player UI around the vertical trailer."""
import os, sys, json, math, wave
from functools import lru_cache
import numpy as np, cv2
from multiprocessing import Pool
from PIL import Image, ImageDraw, ImageFont, ImageFilter

SW, SH, FPS = 1728, 1080, 30
HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, "screen2"); os.makedirs(OUTD, exist_ok=True)
FD = os.path.join(HERE, "fonts")
@lru_cache(maxsize=None)
def F(n, s): return ImageFont.truetype(os.path.join(FD, n), s)
P = np.array([0.55, 0.22, 1.0], np.float32); PL = np.array([0.78, 0.58, 1.0], np.float32)
P8 = (140, 56, 255); PL8 = (198, 148, 255)

_BT = np.load(os.path.join(HERE, "beats.npy")); _i0 = int(np.argmin(np.abs(_BT - 44.54))) - 2
REL = _BT[_i0:_i0 + 60] - _BT[_i0]
KR = [float(REL[k]) for k in [0, 2, 6, 14, 20, 22, 24, 26, 30, 36]]
KD = [0.0, 0.45, 3.0, 7.0, 11.0, 12.0, 13.0, 14.0, 16.0, 20.0]
def dtime(rt): return float(np.interp(rt, KR, KD))
DROP = KR[1]
CUTS_R = [KR[1], KR[2], KR[3], KR[4], KR[8]]          # real-time scene starts
END = KR[-1]

def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def eo(x): x = clamp(x); return 1 - (1 - x) ** 3
def eo5(x): x = clamp(x); return 1 - (1 - x) ** 5
def eio(x): x = clamp(x); return x * x * (3 - 2 * x)
def lerp(a, b, x): return a + (b - a) * x
def beat_pulse(rt, d=8.0):
    prev = REL[REL <= rt + 1e-6]
    return math.exp(-(rt - prev[-1]) * d) if len(prev) else 0.0

SCENES = [("01", "ЗИМА НА", "STATE RP", "Обновление 7 декабря"),
          ("02", "АДВЕНТ-", "КАЛЕНДАРЬ", "24 дня наград"),
          ("03", "НОВЫЙ", "БАТЛПАСС", "Качай уровни и забирай награды"),
          ("04", "НОВЫЕ", "ТАЧКИ", "ВАЗ · AMG · Charger"),
          ("05", "ЗАЛЕТАЙ", "НА СЕРВЕР", "Зимнее обновление · 7 декабря")]
ITEMS = ["Зима и снег", "Адвент-календарь", "Новый батлпасс", "Новые тачки", "7 декабря"]

# ---------------- audio bands ----------------
def audio_bands():
    with wave.open(os.path.join(HERE, "audio_music.wav")) as w:
        sr = w.getframerate(); a = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, 2).mean(1) / 32768
    nf = int(len(a) / sr * FPS) + 1; hop = sr // FPS; nb = 28
    out = np.zeros((nf, nb)); edges = np.geomspace(35, 14000, nb + 1)
    for i in range(nf):
        seg = a[max(0, i * hop - 1024): i * hop + 1024]
        if len(seg) < 512: continue
        sp = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))); fr = np.fft.rfftfreq(len(seg), 1 / sr)
        for b in range(nb):
            m = (fr >= edges[b]) & (fr < edges[b + 1]); out[i, b] = sp[m].mean() if m.any() else 0
    out = np.log1p(out * 4)
    lo = np.percentile(out, 25, 0, keepdims=True); hi = np.percentile(out, 99.5, 0, keepdims=True)
    out = np.clip((out - lo) / (hi - lo + 1e-6), 0, 1) ** 1.5
    sm = out.copy()
    for i in range(1, nf):
        up = out[i] > sm[i - 1]
        sm[i] = np.where(up, sm[i - 1] + (out[i] - sm[i - 1]) * 0.7, sm[i - 1] * 0.78)
    return sm
BANDS = audio_bands()
BASS = BANDS[:, :4].mean(1)

# ---------------- compositing helpers ----------------
def to_spr(im):
    a = np.asarray(im).astype(np.float32) / 255
    return np.dstack([a[..., :3] * a[..., 3:4], a[..., 3]])

def blit(f, spr, x, y, alpha=1.0, scale=1.0, blur=0.0, anchor="lt", add=False):
    if alpha <= 0.003: return
    if abs(scale - 1) > 1e-3:
        spr = cv2.resize(spr, (max(1, int(spr.shape[1] * scale)), max(1, int(spr.shape[0] * scale))), interpolation=cv2.INTER_LINEAR)
    if blur > 0.3: spr = cv2.GaussianBlur(spr, (0, 0), blur)
    h, w = spr.shape[:2]
    if anchor == "c": x, y = x - w / 2, y - h / 2
    elif anchor == "lc": y = y - h / 2
    x0, y0 = int(round(x)), int(round(y))
    fx0, fy0, fx1, fy1 = max(0, x0), max(0, y0), min(SW, x0 + w), min(SH, y0 + h)
    if fx1 <= fx0 or fy1 <= fy0: return
    s = spr[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0] * alpha
    r = f[fy0:fy1, fx0:fx1]
    if add: r[..., :3] += s[..., :3]
    elif r.shape[2] == 4: r *= (1 - s[..., 3:4]); r += s
    else: r *= (1 - s[..., 3:4]); r += s[..., :3]

@lru_cache(maxsize=4096)
def glyph(ch, font, size, col, glow=None, stroke=0):
    f = F(font, size); pad = 30 if glow else 8
    bb = f.getbbox("Hg" + ch)
    w = int(f.getlength(ch)) + pad * 2 + stroke * 2; h = bb[3] + pad * 2 + 10
    im = Image.new("RGBA", (max(2, w), h), (0, 0, 0, 0))
    if glow:
        g = Image.new("RGBA", im.size, (0, 0, 0, 0))
        ImageDraw.Draw(g).text((pad + stroke, pad), ch, font=f, fill=glow + (255,), stroke_width=stroke + 3, stroke_fill=glow + (255,))
        im = Image.alpha_composite(im, g.filter(ImageFilter.GaussianBlur(12)))
    if stroke and col is None:
        ImageDraw.Draw(im).text((pad + stroke, pad), ch, font=f, fill=(0, 0, 0, 0), stroke_width=stroke, stroke_fill=PL8 + (255,))
    else:
        ImageDraw.Draw(im).text((pad + stroke, pad), ch, font=f, fill=col + (255,), stroke_width=stroke, stroke_fill=(col or (0, 0, 0)) + (255,))
    return to_spr(im), pad

def stagger_text(f, text, font, size, col, x, y, t_in, t_out, rt, glow=None, step=0.028, rise=34, stroke=0):
    """per-character reveal (rise + fade + blur) and staggered exit upwards."""
    fo = F(font, size); cx = x
    for k, ch in enumerate(text):
        adv = fo.getlength(ch)
        if ch != " ":
            a_in = eo5((rt - t_in - k * step) / 0.32)
            a_out = eio((rt - t_out - k * step * 0.6) / 0.22) if t_out is not None else 0
            a = a_in * (1 - a_out)
            if a > 0.003:
                spr, pad = glyph(ch, font, size, col, glow, stroke)
                dy = (1 - a_in) * rise - a_out * rise * 1.2
                blit(f, spr, cx - pad, y - pad + dy, alpha=a, blur=(1 - a_in) * 5 + a_out * 5)
        cx += adv

@lru_cache(maxsize=None)
def text_spr(text, font, size, col, glow=None):
    f = F(font, size); pad = 30 if glow else 6
    bb = f.getbbox(text)
    im = Image.new("RGBA", (bb[2] + pad * 2, bb[3] + pad * 2), (0, 0, 0, 0))
    if glow:
        g = Image.new("RGBA", im.size, (0, 0, 0, 0)); ImageDraw.Draw(g).text((pad, pad), text, font=f, fill=glow + (255,))
        im = Image.alpha_composite(im, g.filter(ImageFilter.GaussianBlur(14)))
    ImageDraw.Draw(im).text((pad, pad), text, font=f, fill=col + (255,))
    return to_spr(im)

@lru_cache(maxsize=None)
def rrect_spr(w, h, r, fill, outline=None, width=2, glow=None, glow_r=18):
    pad = glow_r * 2 if glow else 2
    im = Image.new("RGBA", (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
    if glow:
        g = Image.new("RGBA", im.size, (0, 0, 0, 0))
        ImageDraw.Draw(g).rounded_rectangle([pad, pad, pad + w, pad + h], r, outline=glow + (255,), width=max(6, width * 4))
        im = Image.alpha_composite(im, g.filter(ImageFilter.GaussianBlur(glow_r)))
    ImageDraw.Draw(im).rounded_rectangle([pad, pad, pad + w, pad + h], r, fill=fill, outline=outline, width=width)
    return to_spr(im), pad

def gauss_blob(n=256):
    x = np.linspace(-1, 1, n, dtype=np.float32); g = np.exp(-(x[:, None] ** 2 + x[None, :] ** 2) * 3.0)
    return g[..., None]
BLOB = gauss_blob()

_rp = np.random.default_rng(3)
PARTS = _rp.random((90, 4)).astype(np.float32)

# ---------------- trailer source ----------------
VID = os.path.join(HERE, "trailer_music_noaudio.mp4")
_cap = {}
def trailer_frame(i):
    if "c" not in _cap:
        _cap["c"] = cv2.VideoCapture(VID); _cap["i"] = -1
    c = _cap["c"]
    if i != _cap["i"] + 1:
        c.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, fr = c.read(); _cap["i"] = i
    if not ok: fr = np.zeros((1920, 1080, 3), np.uint8)
    return fr[:, :, ::-1].astype(np.float32) / 255

# ---------------- light-curve logo intro ----------------
def intro(f, rt):
    """light curves sweep in and converge into the STATE RP logo (pre-drop)."""
    k = clamp(rt / DROP)
    lay = np.zeros((SH // 2, SW // 2, 3), np.float32)
    cx, cy = SW / 4, SH / 4
    for j, (col, side) in enumerate([((1, 1, 1), -1), (tuple(PL.tolist()), 1), ((1.0, 0.35, 0.8), -1), ((0.6, 0.6, 1.0), 1)]):
        pts = []
        u_end = eo(k * 1.15 - j * 0.06)
        for s in np.linspace(max(0, u_end - 0.35), u_end, 40):
            ang = side * (1 - s) * 2.6 + j * 0.4
            rad = (1 - s) * 360 + 10
            pts.append((cx + math.cos(ang) * rad * 1.4, cy + math.sin(ang) * rad * 0.7 - (1 - s) * 60 * side))
        pts = np.array(pts, np.int32)
        if len(pts) > 1:
            cv2.polylines(lay, [pts], False, col, 3, cv2.LINE_AA)
            cv2.circle(lay, tuple(pts[-1]), 5, col, -1, cv2.LINE_AA)
    glow = cv2.GaussianBlur(lay, (0, 0), 8)
    f += cv2.resize(lay + glow * 2.0, (SW, SH))
    # logo forming near the end of the intro
    la = eo((rt - DROP * 0.55) / (DROP * 0.4))
    if la > 0:
        spr = text_spr("STATE RP", "mont900i.ttf", 150, (255, 255, 255), glow=P8)
        blit(f, spr, SW / 2, SH / 2, alpha=la, scale=lerp(0.85, 1.0, la), blur=(1 - la) * 8, anchor="c")
        add_flare(f, SW / 2, SH / 2, 2.4 * la, 0.25 * la)

def add_flare(f, x, y, s, inten, col=None):
    col = PL if col is None else col
    n = max(4, int(256 * s)); g = cv2.resize(BLOB, (n * 3, n))[..., None] if BLOB.ndim == 2 else cv2.resize(BLOB[..., 0], (n * 3, n))[..., None]
    g = g * col * inten
    blit_add(f, g, x - g.shape[1] / 2, y - g.shape[0] / 2)

def blit_add(f, g, x, y):
    h, w = g.shape[:2]; x0, y0 = int(x), int(y)
    fx0, fy0, fx1, fy1 = max(0, x0), max(0, y0), min(SW, x0 + w), min(SH, y0 + h)
    if fx1 > fx0 and fy1 > fy0: f[fy0:fy1, fx0:fx1] += g[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]

# ---------------- main frame ----------------
YY = np.linspace(0, 1, SH, dtype=np.float32)[:, None, None]
BG0 = (np.array([0.045, 0.02, 0.085], np.float32) * (1 - YY) + np.array([0.02, 0.01, 0.04], np.float32) * YY) * np.ones((1, SW, 1), np.float32)

def build(i):
    rt = i / FPS; td = dtime(rt)
    bass = float(BASS[min(i, len(BASS) - 1)]); bp = beat_pulse(rt)
    f = BG0.copy()
    # drifting neon blobs (bass reactive)
    small = np.zeros((SH // 8, SW // 8, 3), np.float32)
    for j, (col, sp, ph, rad) in enumerate([((0.55, 0.15, 1.0), 0.13, 0.0, 70), ((1.0, 0.2, 0.7), 0.09, 2.0, 55), ((0.25, 0.3, 1.0), 0.11, 4.0, 60)]):
        bx = (0.5 + 0.38 * math.sin(rt * sp * 2 * math.pi + ph)) * SW / 8
        by = (0.5 + 0.30 * math.cos(rt * sp * 1.6 * math.pi + ph * 1.3)) * SH / 8
        cv2.circle(small, (int(bx), int(by)), rad, col, -1, cv2.LINE_AA)
    small = cv2.GaussianBlur(small, (0, 0), 22)
    f += cv2.resize(small, (SW, SH)) * (0.16 + 0.12 * bass)
    # particles
    for px, py, sp, sz in PARTS:
        x = (px * SW + rt * 18 * (sp - 0.5)) % SW; y = (py * SH - rt * 25 * (0.3 + sp)) % SH
        cv2.circle(f, (int(x), int(y)), int(1 + sz * 2), tuple((PL * (0.25 + 0.3 * sz)).tolist()), -1, cv2.LINE_AA)

    if rt < DROP:
        intro(f, rt)
        return finish(f, i)

    # ---- layout reveal at the drop ----
    u = rt - DROP
    open_k = eo5(u / 0.45)
    side_k = [eo5((u - 0.25 - 0.08 * k) / 0.5) for k in range(3)]
    end_k = eio((rt - (END - 1.2)) / 0.9)            # outro: side panels leave, player grows
    # logo flash dissolving
    if u < 0.4:
        spr = text_spr("STATE RP", "mont900i.ttf", 150, (255, 255, 255), glow=P8)
        blit(f, spr, SW / 2, SH / 2, alpha=1 - u / 0.4, scale=1 + u * 1.2, blur=u * 30, anchor="c")
    # ---- player ----
    ph_full = 980; pw = int(ph_full * 1080 / 1920)
    sc = lerp(1.0, 1.06, end_k) * (1 + 0.012 * bp)
    ph = int(max(4, ph_full * open_k) * sc); pwc = int(pw * sc)
    cx, cy = SW / 2, 44 + (SH - 44) / 2
    glow_spr, gp = rrect_spr(pwc, max(ph, 4), 22, (0, 0, 0, 0), outline=PL8 + (220,), width=2, glow=P8, glow_r=22)
    blit(f, glow_spr, cx - pwc / 2 - gp, cy - ph / 2 - gp, alpha=0.6 + 0.4 * bp + 0.3 * end_k, add=False)
    if ph > 6:
        vid = trailer_frame(i)
        v = cv2.resize(vid, (pwc, int(ph_full * sc)), interpolation=cv2.INTER_AREA)
        oy = (v.shape[0] - ph) // 2
        v = v[oy:oy + ph]
        m = np.zeros((ph, pwc), np.float32); cv2.rectangle(m, (0, 0), (pwc - 1, ph - 1), 1.0, -1)
        rr = 20
        for (ccx, ccy) in [(rr, rr), (pwc - rr - 1, rr), (rr, ph - rr - 1), (pwc - rr - 1, ph - rr - 1)]:
            m[max(0, ccy - rr):ccy + rr + 1, max(0, ccx - rr):ccx + rr + 1] = 0
            cv2.circle(m, (ccx, ccy), rr, 1.0, -1, cv2.LINE_AA)
        vb = 1 - open_k
        if vb > 0.02: v = cv2.GaussianBlur(v, (0, 0), vb * 12)
        spr = np.dstack([v * m[..., None], m]) * (open_k ** 2)
        blit(f, spr, cx - pwc / 2, cy - ph / 2)
        # glass sheen sweeping on scene cuts
        for c in CUTS_R:
            d = rt - c
            if 0 <= d < 0.5:
                sx = cx - pwc / 2 + lerp(-200, pwc + 200, eio(d / 0.5))
                band = np.zeros((ph, pwc, 4), np.float32)
                xs = np.arange(pwc, dtype=np.float32)[None, :] - (sx - (cx - pwc / 2)) - np.arange(ph, dtype=np.float32)[:, None] * 0.3
                a = np.exp(-(xs / 60) ** 2) * 0.12 * m
                band[..., :3] = a[..., None]; band[..., 3] = 0
                blit(f, band, cx - pwc / 2, cy - ph / 2, add=True)

    # ---- left: scene typography ----
    si = max(k for k in range(5) if rt >= CUTS_R[k] - 1e-6)
    lx = 90 - 180 * end_k
    la = (1 - end_k)
    if la > 0.01:
        num, a_, b_, sub = SCENES[si]
        t_in = CUTS_R[si] + (0.15 if si == 0 else 0.05)
        t_out = CUTS_R[si + 1] - 0.32 if si + 1 < len(CUTS_R) else None
        # rolling scene number (outline)
        prev_num = SCENES[si - 1][0] if si > 0 else None
        nk = eo5((rt - CUTS_R[si]) / 0.4)
        spr_new = text_spr(num, "mont900i.ttf", 130, PL8, glow=P8)
        tmp = np.zeros((SH, SW, 4), np.float32)
        blit(tmp, spr_new, lx - 30, 250 + (1 - nk) * 130 - 30, alpha=nk * la)
        if prev_num and nk < 1:
            blit(tmp, text_spr(prev_num, "mont900i.ttf", 130, PL8, glow=P8), lx - 30, 250 - nk * 130 - 30, alpha=(1 - nk) * la)
        win = np.zeros((SH, SW, 1), np.float32); win[230:430] = 1
        # composite the rolling number only inside its window (mask reveal)
        a_t = tmp[..., 3:4] * win
        f = f * (1 - a_t) + tmp[..., :3] * win
        stagger_text(f, a_, "mont900i.ttf", 68, (255, 255, 255), lx, 430, t_in + 0.08, t_out, rt)
        stagger_text(f, b_, "mont900i.ttf", 68, PL8, lx, 510, t_in + 0.18, t_out, rt, glow=P8)
        # subtitle line + underline growing
        sa = eo((rt - t_in - 0.4) / 0.35) * (1 - (eio((rt - t_out) / 0.2) if t_out else 0)) * la
        words, lines, cur = sub.split(), [], ""
        for w_ in words:
            if F("mont600.ttf", 30).getlength((cur + " " + w_).strip()) > 330: lines.append(cur); cur = w_
            else: cur = (cur + " " + w_).strip()
        lines.append(cur)
        for li, ln in enumerate(lines):
            blit(f, text_spr(ln, "mont600.ttf", 30, (215, 205, 235)), lx - 6, 615 + li * 40 + (1 - sa) * 14, alpha=sa)
        ul = eo((rt - t_in - 0.3) / 0.5) * la
        cv2.line(f, (int(lx), 605), (int(lx + 380 * ul), 605), tuple((P * 1.2).tolist()), 3, cv2.LINE_AA)

    # ---- right: chapters with sliding highlight ----
    rx = cx + pw / 2 + 80 + 220 * end_k
    ra = side_k[1] * (1 - end_k)
    if ra > 0.01:
        blit(f, text_spr("В ОБНОВЛЕНИИ", "mont800.ttf", 24, (190, 180, 215)), rx - 6, 186, alpha=ra)
        # highlight pill position eased between items
        pos = float(si)
        for k in range(1, 5):
            pos = pos if rt >= CUTS_R[k] + 0.4 else pos
        prev_si = max(0, si - 1)
        mv = eo5((rt - CUTS_R[si]) / 0.4)
        pill_y = lerp(prev_si, si, mv) if si > 0 else 0
        pill, pp = rrect_spr(330, 54, 27, (60, 26, 120, 230), outline=PL8 + (180,), width=2, glow=P8, glow_r=12)
        blit(f, pill, rx - 18 - pp, 248 + pill_y * 64 - pp, alpha=ra)
        # progress line
        cv2.line(f, (int(rx + 10), 275), (int(rx + 10), 275 + 4 * 64), (0.22, 0.18, 0.3), 2, cv2.LINE_AA)
        prog_y = 275 + pill_y * 64
        cv2.line(f, (int(rx + 10), 275), (int(rx + 10), int(prog_y)), tuple((PL * 1.1).tolist()), 3, cv2.LINE_AA)
        for k, it in enumerate(ITEMS):
            y = 248 + k * 64
            on = k == si; done = k < si
            dot = PL * 1.1 if (on or done) else np.array([0.3, 0.25, 0.4], np.float32)
            cv2.circle(f, (int(rx + 10), int(y + 27)), 9 if on else 7, tuple(dot.tolist()), -1, cv2.LINE_AA)
            col = (255, 255, 255) if on else ((185, 165, 225) if done else (120, 110, 145))
            ia = eo5((u - 0.35 - 0.06 * k) / 0.4) * ra
            blit(f, text_spr(it, "mont800.ttf" if on else "mont600.ttf", 30, col), rx + 34, y + 27 + (1 - ia) * 20, alpha=ia, anchor="lc")
    # ---- visualiser (mirrored bars with glow) ----
    va = side_k[2] * (1 - end_k)
    if va > 0.01:
        bands = BANDS[min(i, len(BANDS) - 1)]
        vx0, vy = rx - 6, 820
        lay = np.zeros((SH, SW, 3), np.float32)
        for j, val in enumerate(bands[:24]):
            hgt = 4 + val * 95 * va
            x = int(vx0 + j * 15)
            c = tuple((PL * (0.6 + 0.6 * val) * va).tolist())
            cv2.rectangle(lay, (x, int(vy - hgt)), (x + 8, int(vy + hgt * 0.55)), c, -1, cv2.LINE_AA)
        f += lay + cv2.GaussianBlur(lay, (0, 0), 10) * 1.2
        # timecode
        tc = f"0:{int(rt):02d} / 0:20"
        blit(f, text_spr(tc, "mont600.ttf", 26, (200, 190, 225)), vx0, 900, alpha=va)
    # ---- left bottom: scrub line ----
    sa = side_k[0] * (1 - end_k)
    if sa > 0.01:
        x0, x1, y = 90, int(cx - pw / 2 - 90), 900
        cv2.line(f, (x0, y), (x1, y), (0.22, 0.18, 0.3), 3, cv2.LINE_AA)
        px = int(lerp(x0, x1, rt / END))
        cv2.line(f, (x0, y), (px, y), tuple((PL * sa).tolist()), 3, cv2.LINE_AA)
        cv2.circle(f, (px, y), 8, tuple((np.array([1, 1, 1], np.float32) * sa).tolist()), -1, cv2.LINE_AA)
        add_flare(f, px, y, 0.25, 0.4 * sa)
    # outro fade
    if rt > END - 0.35:
        f *= clamp((END - rt) / 0.35)
    return finish(f, i)

def finish(f, i):
    out = (np.clip(f, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(OUTD, f"s_{i:04d}.jpg"), out[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 93])
    small = cv2.resize(out, (16, 10), interpolation=cv2.INTER_AREA).reshape(-1, 3).mean(0) / 255
    return i, small.tolist()

NF = int(round(END * FPS))
def run(ch): return [build(j) for j in ch]
if __name__ == "__main__":
    if len(sys.argv) > 1:
        for s in sys.argv[1].split(","): build(int(s))
        sys.exit()
    chunks = [list(range(k, min(NF, k + 50))) for k in range(0, NF, 50)]
    with Pool(4) as p:
        res = {}
        for lst in p.imap(run, chunks):
            res.update(dict(lst))
    json.dump([res[k] for k in sorted(res)], open(os.path.join(HERE, "screen2_colors.json"), "w"))
    print("done", len(res))
