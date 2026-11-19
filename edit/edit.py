#!/usr/bin/env python3
"""STATE RP — Dark Phonk car edit. 1080x1920, 60 fps, exactly 20.000 s, beat-synced to music/track.mp3.

Usage:
  python3 edit.py                       # full render -> out/edit_staterp.mp4
  python3 edit.py --stills 3.1,7.5,16.8 # preview frames -> out/stills/
Shots are defined in TIMELINE below (beat-indexed). Kind "clip" accepts real BeamNG recordings from clips/.
"""
import os, sys, math, random, json, subprocess, wave
import numpy as np, cv2
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E                      # plates, car layers, lights, snow, spray, glow helpers

W, H = E.W, E.H
FPS = 60
DUR = 20.0
NF = int(DUR * FPS)                     # 1200 frames
TRACK = os.path.join(HERE, "music", "track.mp3")
SONG_START = 4.20                       # track second that maps to edit 0.0 (drop lands at edit 3.0)
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)

# use the showroom plate with the neon lettering removed (no text in frame)
E.PL["p4"] = E.load_plate("p4ns")
_b = cv2.imread(os.path.join(HERE, "cars", "p4ns_bg.png"))[:, :, ::-1].astype(np.float32) / 255.0
E.PL["p4c"] = (_b, E.PL["p4"][1])

clamp, lerp, eo, eo5, eio = E.clamp, E.lerp, E.eo, E.eo5, E.eio

# hide the inpainting smudge in the BMW E38 plate sky (where the reference title used to be)
def _darken_sky(img):
    h, w = img.shape[:2]
    m = np.zeros((h, w), np.float32)
    cv2.ellipse(m, (2050, 330), (1150, 480), 0, 0, 360, 1.0, -1)
    m = cv2.GaussianBlur(m, (0, 0), 120)[..., None]
    sky = cv2.GaussianBlur(img, (0, 0), 25) * 0.45
    return img * (1 - m) + sky * m
E.PL["p1"] = (_darken_sky(E.PL["p1"][0]), E.PL["p1"][1])
E.PL["p1c"] = (_darken_sky(E.PL["p1c"][0]), E.PL["p1c"][1])

# ------------------------------------------------------------------ beats
_BT = np.load(os.path.join(HERE, "music", "beats.npy"))
BEATS = (_BT - SONG_START)
BEATS = BEATS[(BEATS > -1) & (BEATS < DUR + 1)]
I0 = int(np.argmin(np.abs(BEATS - 3.0)))           # drop beat
def beat(k):
    """time (edit seconds) of beat k relative to the drop; fractional k interpolates."""
    i = I0 + k
    lo = int(math.floor(i)); fr = i - lo
    lo = max(0, min(len(BEATS) - 2, lo))
    return float(BEATS[lo] + (BEATS[lo + 1] - BEATS[lo]) * fr)

def beat_pulse(t, d=9.0):
    prev = BEATS[BEATS <= t + 1e-6]
    return math.exp(-(t - prev[-1]) * d) if len(prev) else 0.0

def bass_env():
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", f"{SONG_START}", "-t", f"{DUR}", "-i", TRACK,
                          "-ac", "1", "-ar", "12000", "-af", "lowpass=f=140", "-f", "f32le", "-"], capture_output=True, check=True).stdout
    a = np.frombuffer(raw, np.float32)
    hop = 12000 // FPS
    e = np.array([np.sqrt(np.mean(a[i * hop:(i + 1) * hop] ** 2)) if i * hop < len(a) else 0 for i in range(NF)])
    e = e / (np.percentile(e, 98) + 1e-9)
    return np.clip(e, 0, 1.3)
BASS = bass_env()

# ------------------------------------------------------------------ car data
LIGHTS = {
    "bmw": [((1065, 921), (1.0, .97, .9)), ((1191, 936), (1.0, .97, .9)), ((1680, 969), (1, .97, .9)), ((1749, 975), (1, .97, .9))],
    "amg": [((555, 795), (.78, .88, 1.0)), ((1170, 786), (.78, .88, 1.0))],
    "charger": [((1725, 768), (.6, .78, 1.0)), ((2160, 768), (.6, .78, 1.0))],
    "vaz": [((2790, 774), (1, .93, .8)), ((3045, 768), (1, .93, .8))],
    "bmwr": [((495, 645), (1.0, .12, .08)), ((1425, 645), (1.0, .12, .08))],
}
WHEELS = {"bmw": [(760, 1090), (420, 950)], "amg": [(330, 1060), (1050, 1060)], "charger": [(1330, 990)],
          "vaz": [(2200, 930), (2640, 905)], "bmwr": [(450, 1040), (1750, 1040)]}
PLATE_OF = {"bmw": "p1", "amg": "p4", "charger": "p4", "vaz": "p4", "bmwr": "p5"}
STATIC_P4 = ("amg", "charger", "vaz")

def motion(kind, u, d=1.0):
    """car layer motion (dx, dy, rot_deg, scale) for normalised shot time u."""
    if kind == "static": return (0.0, 0.0, 0.0, 1.0)
    if kind == "creep": return (70 * u * d, 0.0, 0.0, 1.0 + 0.03 * u)
    if kind == "slide_in":
        k = eo(u * 1.7); land = max(0.0, u - 0.58)
        return (-d * 1500 * (1 - k), 0.0, -d * 1.6 * (1 - k) + d * 0.9 * math.exp(-land * 7) * math.sin(land * 28) * (u > 0.58), 1.0)
    if kind == "drift_in":
        k = eo(u * 1.6); land = max(0.0, u - 0.62)
        return (d * 1350 * (1 - k), -30 * (1 - k), d * 8.5 * (1 - k) ** 1.4 + 0.8 * math.exp(-land * 7) * math.sin(land * 26) * (u > 0.62), 1.0)
    if kind == "launch":
        k = eo5(clamp((u - 0.1) / 0.6)); squat = math.sin(math.pi * clamp((u - 0.08) / 0.45))
        return (d * 240 * k, 60 * k, -1.8 * squat, 1.0 + 0.17 * k)
    if kind == "driveby":
        return (d * lerp(-1900, 1900, u), 10 * math.sin(u * 9), d * 1.2 * math.sin(u * 3), 1.0)
    if kind == "drift_osc":
        return (120 * math.sin(u * 4.2 * d) - 40 * u, 16 * math.sin(u * 8.4), 2.8 * math.sin(u * 4.2 * d + 0.7) - 1.0, 1.0)
    if kind == "away":
        return (20 * u, -110 * eio(u) + 5 * math.sin(u * 22), 0.4 * math.sin(u * 4), 1.0 - 0.18 * eio(u))
    raise ValueError(kind)

def ramp(kind, u, k=0.75):
    """speed-ramp / easing of shot time."""
    u = clamp(u)
    if kind == "lin": return u
    if kind == "eo": return eo(u)
    if kind == "eio": return eio(u)
    if kind == "fsf": return u + k * math.sin(2 * math.pi * u) / (2 * math.pi)       # fast-slow-fast speed ramp
    if kind == "sff": return u - k * math.sin(2 * math.pi * u) / (2 * math.pi)       # slow-fast-slow
    if kind == "slow": return 0.12 * u                                                 # slow motion
    return u

# ------------------------------------------------------------------ shot renderers
def render_shot(sh, t):
    t0, t1 = sh["t0"], sh["t1"]
    u_lin = clamp((t - t0) / max(1e-6, t1 - t0))
    cams = []
    nsub = sh.get("blur", 3)
    dt_sub = 1.0 / FPS / max(1, nsub) * 1.6
    lay = []
    for i in range(nsub):
        tt = t - i * dt_sub
        u = ramp(sh.get("ramp", "eo"), clamp((tt - t0) / max(1e-6, t1 - t0)))
        c0, c1 = sh["cam"]
        cam = [lerp(a, b, u) for a, b in zip(c0, c1)]
        cam[3] = math.radians(cam[3])
        cars = {}
        hero = sh.get("car")
        if hero:
            mu = ramp(sh.get("mramp", "lin"), clamp((tt - t0) / max(1e-6, t1 - t0))) * sh.get("mspeed", 1.0)
            m = motion(sh.get("move", "static"), mu, sh.get("dir", 1.0))
            cam[0] += m[0] * sh.get("follow", 0.0); cam[1] += m[1] * sh.get("follow", 0.0)
            if PLATE_OF[hero] == "p4":
                for c in STATIC_P4:
                    if c != hero: cars[c] = (0.0, 0.0, 0.0, 1.0)
            cars[hero] = m
        lay.append((tuple(cam), cars))
    plate = sh["plate"]
    if any(c for _, c in lay):
        f, M = E.render_cars(plate, lay)
        cars_now = lay[0][1]
    else:
        f, M = E.render_plate(plate, [c for c, _ in lay])
        cars_now = {}
    z = lay[0][0][2]
    pb = beat_pulse(t)
    # lights (headlights / tail lights) following the car layers
    owners = [sh["car"]] if sh.get("car") else []
    owners += [c for c in sh.get("lights", []) if c not in owners]
    for c in owners:
        if PLATE_OF[c] != plate: continue
        prm = cars_now.get(c, (0.0, 0.0, 0.0, 1.0))
        for (hx, hy), col in LIGHTS[c]:
            X, Y = E.map_pt(M, *E.car_pt(c, hx, hy, prm) if c in cars_now else (hx, hy))
            inten = sh.get("glow", 0.7) * (0.6 + 0.4 * pb) + sh.get("ignite", 0) * math.exp(-max(0, t - t0) * 6)
            red = col[1] < 0.3
            E.headlight(f, X, Y, 0.34 * z / 1.6, inten, col=col, streak_col=(1.0, 0.2, 0.25) if red else (0.7, 0.6, 1.0))
            E.wet_reflection(f, X, Y + 40 * z / 1.6, 520 * z / 1.6, 80 * z / 1.6, col, 0.2 * inten)
    # snow spray from moving wheels
    if sh.get("spray") and sh.get("car"):
        c = sh["car"]; prm = cars_now.get(c, (0, 0, 0, 1))
        st = t if sh.get("ramp") != "slow" else t0 + (t - t0) * 0.15
        for k, (wx, wy) in enumerate(WHEELS[c]):
            E.spray(f, M, *E.car_pt(c, wx, wy, prm), st + k * 0.3, n=70, seed=11 + k, dirx=sh.get("spray"), power=1.3, scale_px=z / 1.4)
    # falling snow (slowed in slow motion)
    tt = t if sh.get("ramp") != "slow" else t0 + (t - t0) * 0.15
    E.snow(f, t, speed=sh.get("snow", 1.0), wind=0.5, streak=sh.get("streak", (0, 0)), amount=0.9, tt=tt)
    if sh.get("expo"):
        a, b = sh["expo"]; f *= lerp(a, b, eio(u_lin))
    if sh.get("dolly"):          # forward rush feel
        f = E.zoom_blur(f, sh["dolly"] * (0.5 + 0.5 * math.sin(math.pi * u_lin)))
    if sh.get("flip"): f = f[:, ::-1].copy()
    return f

def render_clip(sh, t):
    """real gameplay clip (BeamNG recording) — cover-cropped to 9:16 with speed ramp."""
    cap = cv2.VideoCapture(os.path.join(HERE, "clips", sh["file"]))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60
    u = ramp(sh.get("ramp", "lin"), clamp((t - sh["t0"]) / (sh["t1"] - sh["t0"])))
    src_t = sh.get("in", 0.0) + u * (sh["t1"] - sh["t0"]) * sh.get("speed", 1.0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(src_t * fps)); ok, fr = cap.read()
    if not ok: return np.zeros((H, W, 3), np.float32)
    fr = fr[:, :, ::-1].astype(np.float32) / 255
    h, w = fr.shape[:2]; s = max(W / w, H / h) * sh.get("zoom", 1.0)
    fr = cv2.resize(fr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    oy, ox = (fr.shape[0] - H) // 2, (fr.shape[1] - W) // 2
    return fr[oy:oy + H, ox:ox + W]

# ------------------------------------------------------------------ TIMELINE (beat indices relative to the drop)
# cam = ((cx, cy, zoom, rot_deg) start, (...) end) in plate pixels.
CLOSE = {  # close-up framings
    "amg_lights": ("p4", "amg", ((860, 810, 2.5, -3), (880, 800, 2.9, 0))),
    "bmw_lights": ("p1", "bmw", ((1420, 1010, 2.5, -5), (1380, 990, 2.9, -1))),
    "chg_drl": ("p4", "charger", ((1980, 780, 2.4, 2), (1900, 775, 2.8, 0))),
    "f10_tails": ("p5", "bmwr", ((960, 660, 2.0, 0), (980, 650, 2.4, 2))),
    "vaz_lights": ("p4", "vaz", ((2910, 790, 2.5, 0), (2880, 780, 2.9, -2))),
    "amg_wheel": ("p4", "amg", ((390, 1000, 2.6, 4), (420, 990, 3.0, 0))),
    "bmw_wheel": ("p1", "bmw", ((780, 1060, 2.4, -6), (760, 1050, 2.8, -2))),
}
def close(name, b0, b1, **kw):
    pl, car, cam = CLOSE[name]
    d = dict(plate=pl, car=car, cam=cam, b0=b0, b1=b1, move="creep", glow=0.9, ramp="eo", blur=3)
    d.update(kw); return d

TIMELINE = [
    # INTRO 0–3 s: dark F10 rear, slow push, lights breathe up; pre-drop micro cuts accelerate
    dict(plate="p5", car="bmwr", move="static", t0=0.0, b1=-1, cam=((1150, 640, 1.05, 1.5), (1160, 650, 1.45, 0)),
         ramp="eio", expo=(0.25, 1.0), glow=0.6, blur=2, snow=0.5),
    close("bmw_lights", -1, -0.5, ignite=1.5, trans="flash"),
    close("amg_lights", -0.5, 0, ignite=1.5, trans="flash"),
    # DROP 3–6 s: one close-up per beat
    close("amg_lights", 0, 1, ignite=2.0, trans="flash", hit=1.0),
    close("bmw_lights", 1, 2, trans="flash"),
    close("chg_drl", 2, 3, trans="zoom"),
    close("f10_tails", 3, 4, trans="flash"),
    close("vaz_lights", 4, 5, trans="zoom"),
    close("amg_wheel", 5, 6, trans="flash"),
    close("bmw_wheel", 6, 7, move="drift_osc", spray=-1.0, trans="whip"),
    # SPEED 6–10 s: drifts, drive-bys, speed ramps
    dict(plate="p1", car="bmw", move="drift_osc", b0=7, b1=9, cam=((1150, 860, 1.2, -3), (1050, 830, 1.0, 1)), follow=0.55,
         ramp="fsf", spray=-1.0, blur=4, streak=(-8, 8), trans="flash", hit=0.6),
    dict(plate="p5", car="bmwr", move="away", b0=9, b1=11, cam=((1150, 700, 1.0, 0), (1180, 600, 1.75, -2)), ramp="fsf",
         dolly=0.9, blur=3, streak=(0, 14), trans="zoom"),
    dict(plate="p4", car="amg", move="driveby", b0=11, b1=13, cam=((700, 800, 1.5, 0), (800, 800, 1.6, 0)), follow=0.45,
         spray=-1.0, blur=5, trans="whip"),
    dict(plate="p4", car="vaz", move="drift_in", dir=1.0, b0=13, b1=15, cam=((2650, 760, 1.9, 0), (2720, 750, 2.0, 1)), follow=0.35,
         spray=1.0, blur=4, trans="flash"),
    dict(plate="p4", car="charger", move="launch", b0=15, b1=16, cam=((1720, 790, 1.8, 1), (1800, 800, 2.0, 0)), follow=0.6,
         spray=-1.0, blur=4, trans="zoom", hit=0.6),
    # MAIN 10–15 s: cuts every 1–2 beats
    dict(plate="p1", car="bmw", move="drift_osc", b0=16, b1=17, cam=((1250, 900, 1.7, -4), (1200, 880, 1.9, 0)), follow=0.6,
         spray=-1.0, flip=True, blur=4, trans="flash", hit=0.6),
    dict(plate="p4", car="amg", move="slide_in", dir=1.0, b0=17, b1=18, cam=((760, 800, 1.9, -2), (700, 790, 2.2, 0)), follow=0.35,
         flip=True, spray=-1.0, blur=4, trans="whip"),
    dict(plate="p4", car="charger", move="creep", b0=18, b1=20, cam=((900, 760, 1.35, -1.5), (2500, 720, 1.5, 1.5)), ramp="sff",
         lights=["amg", "vaz"], blur=3, trans="flash"),
    close("f10_tails", 20, 21, move="away", trans="zoom"),
    close("chg_drl", 21, 22, move="launch", trans="flash"),
    dict(plate="p1", car="bmw", move="drift_osc", b0=22, b1=24, cam=((1000, 800, 0.95, 2), (1120, 840, 1.2, -2)), follow=0.5,
         ramp="fsf", spray=-1.0, blur=4, streak=(-8, 8), trans="whip", hit=0.6),
    close("vaz_lights", 24, 25, trans="flash"),
    dict(plate="p4", car="amg", move="driveby", dir=-1.0, b0=25, b1=26, cam=((800, 800, 1.6, 0), (700, 800, 1.7, 0)), follow=0.45,
         flip=True, spray=1.0, blur=5, trans="zoom"),
    dict(plate="p5", car="bmwr", move="away", b0=26, b1=27, cam=((1200, 760, 1.6, 2), (1220, 700, 1.9, 0)), dolly=0.6, trans="flash"),
    # CLIMAX 15–18 s
    dict(plate="p4", car="charger", move="launch", dir=1.0, b0=27, b1=28, cam=((1700, 790, 1.7, 0), (1800, 800, 1.9, 0)), follow=0.6,
         flip=True, spray=-1.0, blur=4, trans="flash", hit=0.8),
    close("bmw_lights", 28, 29, trans="zoom", hit=0.6),
    close("amg_lights", 29, "STOP", trans="flash", hit=0.6),
    # music stops -> slow motion drift, then the bass returns
    dict(plate="p1", car="bmw", move="drift_osc", b0="STOP", b1="HIT", cam=((1180, 880, 1.5, -2), (1160, 870, 1.65, 0)), follow=0.5,
         ramp="slow", mramp="lin", mspeed=0.35, spray=-1.0, blur=1, snow=0.3, trans="flash", glow=1.0),
    close("vaz_lights", "HIT", "HIT+0.5", trans="flash", hit=1.0, rgb=1.0),
    close("chg_drl", "HIT+0.5", "HIT+1", trans="flash", rgb=1.0),
    close("bmw_lights", "HIT+1", "HIT+1.5", trans="flash", rgb=1.0),
    close("amg_lights", "HIT+1.5", "HIT+2", trans="flash", rgb=1.0),
    # OUTRO 18–20 s: one car, camera pulls back, fade out ending exactly at 20.000
    dict(plate="p1", car="bmw", move="drift_osc", b0="HIT+2", t1=DUR, cam=((1300, 930, 2.2, -3), (1080, 820, 0.95, 0)), follow=0.4,
         ramp="eo", spray=-1.0, blur=2, glow=0.9, trans="flash", fade=1.1),
]

def resolve_times():
    """turn beat references into seconds; STOP/HIT = music stop & return found from the track."""
    stop = 20.53 - SONG_START; hit = 21.43 - SONG_START
    def bt(v):
        if isinstance(v, (int, float)): return beat(v)
        if v == "STOP": return stop
        if v.startswith("HIT"):
            off = float(v[3:]) if len(v) > 3 else 0.0
            return hit + off * (beat(1) - beat(0))
        raise ValueError(v)
    for sh in TIMELINE:
        if "t0" not in sh: sh["t0"] = bt(sh["b0"])
        if "t1" not in sh: sh["t1"] = bt(sh["b1"])
    for a, b in zip(TIMELINE, TIMELINE[1:]):
        b["t0"] = a["t1"]                         # no gaps
    TIMELINE[-1]["t1"] = DUR
resolve_times()
CUTS = [sh["t0"] for sh in TIMELINE[1:]]

# ------------------------------------------------------------------ frame
_yy, _xx = np.mgrid[0:H, 0:W].astype(np.float32)
VIG = np.clip(1.15 - (((_xx - W / 2) / (W * 0.75)) ** 2 + ((_yy - H / 2) / (H * 0.72)) ** 2) * 0.75, 0.25, 1.0)[..., None]

def shot_at(t):
    for k, sh in enumerate(TIMELINE):
        if sh["t0"] <= t < sh["t1"]: return k, sh
    return len(TIMELINE) - 1, TIMELINE[-1]

def frame(fi):
    t = fi / FPS
    k, sh = shot_at(t)
    f = render_clip(sh, t) if sh.get("kind") == "clip" else render_shot(sh, t)
    since = t - sh["t0"]
    bass = float(BASS[fi])
    section_k = 0.35 if t < 3.0 else (0.25 if sh.get("ramp") == "slow" else 1.0)
    bp = beat_pulse(t, 10) * section_k
    hit = sh.get("hit", 0.0) * math.exp(-since * 9)
    # beat zoom + camera shake
    zoom = 1.0 + 0.045 * bp + 0.03 * bass * section_k + 0.09 * hit
    rr = random.Random(fi)
    amp = (14 * bp + 26 * hit + 6 * bass * section_k)
    ox, oy = (rr.random() - 0.5) * amp, (rr.random() - 0.5) * amp
    rot = (rr.random() - 0.5) * 0.014 * (bp + hit)
    if zoom > 1.001 or amp > 0.5:
        M = E.cam_matrix(W / 2 + ox, H / 2 + oy, zoom, rot)
        f = cv2.warpAffine(f, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    # transitions around cuts (incoming side)
    tr = sh.get("trans", "cut")
    if since < 0.09 and k > 0:
        a = 1 - since / 0.09
        if tr == "zoom": f = E.zoom_blur(f, a * 2.2)
        elif tr == "whip": f = cv2.blur(f, (int(6 + a * 180), 1))
        if tr in ("flash",): f = f + a * 0.75
        else: f = f + a * 0.25
    # outgoing side: accelerate into the next cut
    nxt = TIMELINE[k + 1]["trans"] if k + 1 < len(TIMELINE) else None
    rem = sh["t1"] - t
    if nxt in ("zoom", "whip") and rem < 0.06:
        a = 1 - rem / 0.06
        f = E.zoom_blur(f, a * 1.5) if nxt == "zoom" else cv2.blur(f, (int(6 + a * 140), 1))
    # RGB split
    sh_px = int(4 * bp + 14 * hit + 18 * sh.get("rgb", 0) * math.exp(-since * 6))
    if sh_px > 1:
        f[..., 0] = np.roll(f[..., 0], sh_px, axis=1)
        f[..., 2] = np.roll(f[..., 2], -sh_px, axis=1)
    # ---- dark phonk grade ----
    lum = f.mean(axis=2, keepdims=True)
    f = lum + (f - lum) * 0.85                                         # slightly desaturated mids
    f = f + (1 - np.clip(lum * 2.2, 0, 1)) * np.array([0.008, 0.0, 0.03], np.float32)   # cold purple shadows
    bright = np.clip(f - 0.7, 0, None)                                  # glow / bloom
    small = cv2.resize(bright, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
    f += cv2.resize(cv2.GaussianBlur(small, (0, 0), 5), (W, H)) * 1.1
    f = f * VIG
    f = f / (1 + 0.3 * f)
    f = np.clip((f - 0.5) * 1.18 + 0.47, 0, 1)                          # contrast, crushed blacks
    g = np.random.default_rng(fi).normal(0, 0.022, (H // 2, W // 2)).astype(np.float32)
    f += cv2.resize(g, (W, H), interpolation=cv2.INTER_NEAREST)[..., None]
    if sh.get("fade"):
        f *= clamp((DUR - t) / sh["fade"]) ** 1.3
    return (np.clip(f, 0, 1) * 255).astype(np.uint8).tobytes()

if __name__ == "__main__":
    if "--stills" in sys.argv:
        d = os.path.join(OUT, "stills"); os.makedirs(d, exist_ok=True)
        for s in sys.argv[sys.argv.index("--stills") + 1].split(","):
            fi = int(float(s) * FPS)
            im = np.frombuffer(frame(fi), np.uint8).reshape(H, W, 3)
            cv2.imwrite(os.path.join(d, f"s_{float(s):05.2f}.jpg"), cv2.resize(im, (360, 640))[:, :, ::-1])
        sys.exit()
    json.dump([{k: v for k, v in sh.items() if k in ("t0", "t1", "plate", "car", "move", "trans")} for sh in TIMELINE],
              open(os.path.join(OUT, "timeline_resolved.json"), "w"), indent=1, ensure_ascii=False)
    video = os.path.join(OUT, "video_noaudio.mp4")
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                           "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", video], stdin=subprocess.PIPE)
    with Pool(4) as p:
        for i, b in enumerate(p.imap(frame, range(NF), chunksize=3)):
            ff.stdin.write(b)
            if i % 120 == 0: print("frame", i, flush=True)
    ff.stdin.close(); ff.wait()
    final = os.path.join(OUT, "edit_staterp.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-ss", f"{SONG_START}", "-t", f"{DUR}", "-i", TRACK,
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-af", f"afade=t=in:d=0.03,afade=t=out:st={DUR - 0.8}:d=0.8,loudnorm=I=-9:TP=-1.0:LRA=7",
                    "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-t", f"{DUR}", "-movflags", "+faststart", final], check=True)
    print("done", final)
