#!/usr/bin/env python3
"""Soundtrack for the STATE RP trailer: original drift-phonk beat + engine/turbo/tyre/UI/winter SFX, synced to cuts."""
import numpy as np, wave
from scipy.signal import butter, sosfilt

SR = 48000
DUR = 20.0
N = int(SR * DUR)
BEAT = 0.5
SLOW = (14.0, 16.0)
rng = np.random.default_rng(7)
T = np.arange(N) / SR

def bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, hi], btype="band", fs=SR, output="sos"), x)
def lp(x, fc, order=2):
    return sosfilt(butter(order, fc, btype="low", fs=SR, output="sos"), x)
def hp(x, fc, order=2):
    return sosfilt(butter(order, fc, btype="high", fs=SR, output="sos"), x)
def env_exp(n, d): return np.exp(-np.arange(n) / SR * d)
def add(buf, sig, t, gain=1.0):
    i = int(t * SR)
    if i >= len(buf) or i + len(sig) <= 0: return
    s0 = max(0, -i); i = max(0, i)
    j = min(len(buf), i + len(sig) - s0)
    buf[i:j] += sig[s0:s0 + j - i] * gain
def adsr(n, a=0.01, r=0.1):
    e = np.ones(n); na, nr = int(a * SR), int(r * SR)
    if na: e[:na] = np.linspace(0, 1, na)
    if nr: e[-nr:] *= np.linspace(1, 0, nr)
    return e

# ------------------------------------------------------------------ music (phonk, 120 BPM)
def kick():
    n = int(0.5 * SR); t = np.arange(n) / SR
    f = 42 + 120 * np.exp(-t * 30)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * env_exp(n, 6)
    s[:240] += rng.uniform(-1, 1, 240) * np.linspace(0.5, 0, 240)
    return np.tanh(s * 2.5)
def clap():
    n = int(0.3 * SR); e = env_exp(n, 16)
    for off in (0, 0.011, 0.022): k = int(off * SR); e[k:k + 300] += 0.7
    return bp(rng.uniform(-1, 1, n), 900, 5000) * e * 1.4
def hat(open_=False):
    n = int((0.2 if open_ else 0.05) * SR)
    return hp(rng.uniform(-1, 1, n), 7000) * env_exp(n, 11 if open_ else 70) * 0.5
def cowbell(semi):
    n = int(0.25 * SR); t = np.arange(n) / SR; k = 2 ** (semi / 12)
    s = np.sign(np.sin(2 * np.pi * 587 * k * t)) + np.sign(np.sin(2 * np.pi * 845 * k * t))
    return bp(s, 400, 3000) * env_exp(n, 14) * 0.35
def b808(freq, length, glide=None):
    n = int(length * SR); t = np.arange(n) / SR
    f = np.full(n, freq) if glide is None else freq + (glide - freq) * np.exp(-t * 16)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * env_exp(n, 1.3) * adsr(n, 0.004, 0.03)
    return np.tanh(s * 3.0) * 0.8

drums = np.zeros(N); bass = np.zeros(N); lead = np.zeros(N)
K, C = kick(), clap()
for b in range(40):
    t = b * BEAT
    if t < 0.45 or t >= 19.5: continue
    if SLOW[0] <= t < SLOW[1]:
        if b % 2 == 0: add(drums, K, t, 0.9)
        continue
    add(drums, K, t)
    if b % 2 == 1: add(drums, C, t, 0.8)
    for s in range(4):
        if s % 2 == 0 or b % 4 == 3:
            add(drums, hat(open_=(s == 2 and b % 4 == 1)), t + s * BEAT / 4, 0.8)
E1 = 41.2
prog = [0, 0, -4, -2]
mel = [{0: 12, 3: 12, 6: 15, 8: 19, 10: 17, 12: 15, 14: 14}, {0: 12, 3: 12, 6: 15, 8: 12, 9: 10, 11: 12, 13: 15, 15: 17}]
for bar in range(10):
    t0 = bar * 2.0
    root = E1 * 2 ** (prog[bar % 4] / 12)
    st = max(t0, 0.45)
    if t0 >= 19.5: break
    add(bass, b808(root, 1.3, glide=root * 1.5 if bar % 2 else None), st)
    add(bass, b808(root * 2 ** (3 / 12), 0.7), t0 + 1.25)
    if SLOW[0] - 0.1 <= t0 < SLOW[1]: continue
    for step, semi in mel[bar % 2].items():
        tt = t0 + step * BEAT / 4
        if tt >= 0.45: add(lead, cowbell(semi + prog[bar % 4] - 12), tt)
# sidechain
duck = np.ones(N)
for b in range(40):
    i = int(b * BEAT * SR); n = int(0.2 * SR)
    duck[i:i + n] = np.minimum(duck[i:i + n], 1 - 0.6 * np.exp(-np.arange(n) / SR * 16)[:len(duck[i:i + n])])
music = drums * 0.9 + (bass * 0.85 + lead * 0.8) * duck
# slow-mo: muffle the music
sm = (T >= SLOW[0]) & (T < SLOW[1])
music_lp = lp(music, 400, 4)
music = np.where(sm, music_lp * 1.6, music)

# ------------------------------------------------------------------ SFX
sfx = np.zeros(N)

def engine(t0, dur, rpm_curve, cyl=8, gain=1.0, rough=0.25, bright=1.0, seed=0):
    """Additive engine: firing harmonics + combustion noise, rpm_curve(u)->rpm."""
    r = np.random.default_rng(seed)
    n = int(dur * SR); t = np.arange(n) / SR; u = t / dur
    rpm = np.array([rpm_curve(x) for x in u[::200]]); rpm = np.interp(np.arange(n), np.arange(0, n, 200)[:len(rpm)], rpm)
    ff = rpm / 60 * cyl / 2
    ph = 2 * np.pi * np.cumsum(ff) / SR
    s = np.zeros(n)
    for k in range(1, 14):
        s += np.sin(k * ph + r.uniform(0, 6.28)) / k ** (1.1 - 0.3 * bright)
    crank = 0.6 + 0.4 * np.sin(ph / (cyl / 2) * 1.0)           # lumpy cam
    noise = bp(r.normal(0, 1, n), 80, 2500) * (0.5 + 0.5 * np.sin(ph)) * rough
    s = (s * crank + noise * 3)
    s = np.tanh(s * 0.9)
    s = lp(s, 2500 * bright + 600, 2)
    return s * adsr(n, 0.05, 0.25) * gain

def turbo(t0, dur, f0=2500, f1=7000, gain=0.12):
    n = int(dur * SR); t = np.arange(n) / SR; u = t / dur
    f = f0 + (f1 - f0) * u ** 0.7
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * u ** 1.5 + bp(rng.normal(0, 1, n), 3000, 9000) * 0.3 * u
    return s * adsr(n, 0.05, 0.05) * gain

def blowoff(gain=0.5):
    n = int(0.45 * SR)
    s = bp(rng.normal(0, 1, n), 1500, 7000) * env_exp(n, 7) * adsr(n, 0.005, 0.1)
    return s * gain

def screech(dur, gain=0.3, f0=1700):
    n = int(dur * SR); t = np.arange(n) / SR
    f = f0 + 250 * np.sin(2 * np.pi * 5.5 * t) + 120 * np.sin(2 * np.pi * 13 * t)
    tone = np.sin(2 * np.pi * np.cumsum(f) / SR) + 0.5 * np.sin(2 * np.pi * np.cumsum(f * 2.01) / SR)
    nz = bp(rng.normal(0, 1, n), 1200, 4500)
    am = 0.6 + 0.4 * np.sin(2 * np.pi * 3 * t + 1)
    return (tone * 0.5 + nz * 0.8) * am * adsr(n, 0.08, 0.3) * gain

def whoosh(dur=0.5, rev=False, gain=0.6, lo=300, hi=6000):
    n = int(dur * SR); t = np.arange(n) / SR; u = t / dur
    nz = rng.normal(0, 1, n)
    # sweeping band via short blocks
    out = np.zeros(n); blk = 1024
    for i in range(0, n, blk):
        uu = i / n
        c = lo * (hi / lo) ** (uu if not rev else 1 - uu)
        seg = nz[max(0, i - 2048):i + blk]
        y = bp(seg, max(40, c * 0.6), min(SR / 2 - 100, c * 1.6))[-min(blk, n - i):]
        out[i:i + len(y)] = y
    e = np.sin(np.pi * u) ** 1.5 if not rev else u ** 2.5
    return out * e * gain

def impact(gain=1.0):
    n = int(1.8 * SR); t = np.arange(n) / SR
    f = 28 + 90 * np.exp(-t * 7)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * env_exp(n, 2.0)
    nz = lp(rng.normal(0, 1, n), 3000) * env_exp(n, 9) * 0.5
    return np.tanh((s + nz) * 2.2) * gain

def riser(dur, gain=0.35):
    n = int(dur * SR); u = np.arange(n) / n
    return hp(rng.normal(0, 1, n), 1500) * u ** 2.2 * gain

def click(gain=0.4):
    n = int(0.05 * SR)
    return (bp(rng.normal(0, 1, n), 2000, 9000) * env_exp(n, 120) + np.sin(2 * np.pi * 1800 * np.arange(n) / SR) * env_exp(n, 80) * 0.5) * gain

def chime(f0=1046.5, gain=0.25, dur=0.9):
    n = int(dur * SR); t = np.arange(n) / SR
    s = sum(np.sin(2 * np.pi * f0 * m * t) * a for m, a in [(1, 1), (2.0, .4), (3.01, .25), (4.2, .12)])
    s2 = sum(np.sin(2 * np.pi * f0 * 1.5 * m * t) * a for m, a in [(1, .7), (2, .25)])
    s = s * env_exp(n, 5) + np.concatenate([np.zeros(int(0.08 * SR)), s2[:n - int(0.08 * SR)] * env_exp(n - int(0.08 * SR), 5)])
    return s * gain

def tick(f=2600, gain=0.12):
    n = int(0.025 * SR)
    return np.sin(2 * np.pi * f * np.arange(n) / SR) * env_exp(n, 160) * gain

# --- scene 1: intro streaks + drift
add(sfx, riser(0.45, 0.4), 0.0)
add(sfx, whoosh(0.45, rev=True, gain=0.7), 0.0)
add(sfx, impact(1.1), 0.45)
add(sfx, engine(0.45, 2.6, lambda u: 3500 + 3800 * np.sin(np.pi * min(1, u * 1.6)) ** 0.6 - 1200 * u, cyl=6, gain=0.55, bright=1.2, seed=1), 0.45)
add(sfx, turbo(0.5, 1.0), 0.5)
add(sfx, blowoff(0.45), 1.5)
add(sfx, blowoff(0.35), 2.6)
add(sfx, screech(2.2, 0.22), 0.65)
# --- cut whooshes + impacts
for c in (3.0, 7.0, 11.0, 16.0):
    add(sfx, whoosh(0.5, gain=0.6), c - 0.3)
    add(sfx, impact(0.6), c)
# --- scene 2: advent UI
add(sfx, whoosh(0.4, gain=0.35, lo=800, hi=8000), 3.1)
for tc in (3.55, 4.55, 5.55):
    add(sfx, click(0.5), tc)
    add(sfx, chime(1046.5 if tc < 5 else 1318.5, 0.22), tc + 0.06)
    add(sfx, whoosh(0.35, gain=0.25, lo=1500, hi=9000), tc + 0.55)
for i in range(14):
    add(sfx, tick(2000 + i * 120, 0.1), 6.15 + i * 0.05)
add(sfx, chime(1568, 0.2, 1.2), 6.85)
# --- scene 3: battle pass level ticks + level-up stingers
def eio(x): x = min(1, max(0, x)); return x * x * (3 - 2 * x)
last = 1
for k in range(int(4.0 * 400)):
    t = 7.0 + k / 400
    lvl = 1 + int(eio((t - 7.5) / 3.2) * 49)
    if lvl != last:
        add(sfx, tick(1800 + lvl * 25, 0.07), t); last = lvl
for tc in (8.0, 9.0, 10.0):
    add(sfx, chime(1174.7, 0.16, 0.7), tc)
    add(sfx, whoosh(0.3, gain=0.2, lo=2000, hi=9000), tc - 0.1)
# --- scene 4: three cars revving, then slow-mo boom
add(sfx, engine(11.0, 1.05, lambda u: 1200 + 5200 * np.sin(np.pi * min(1, u * 1.3)) ** 2, cyl=8, gain=0.75, rough=0.4, bright=0.8, seed=2), 11.0)   # AMG V8
add(sfx, engine(12.0, 1.05, lambda u: 1000 + 4600 * np.sin(np.pi * min(1, u * 1.2)) ** 2, cyl=8, gain=0.8, rough=0.55, bright=0.6, seed=3), 12.0)  # Charger
add(sfx, turbo(12.1, 0.8, 4000, 9000, 0.06), 12.1)                                                                                                        # supercharger whine
add(sfx, engine(13.0, 1.0, lambda u: 1500 + 4500 * np.sin(np.pi * min(1, u * 1.3)) ** 2, cyl=4, gain=0.55, rough=0.6, bright=1.4, seed=4), 13.0)   # VAZ
add(sfx, screech(0.6, 0.15, 2000), 13.4)
add(sfx, whoosh(0.5, rev=True, gain=0.6), 13.5)
add(sfx, impact(1.3), 14.0)
n = int(2.0 * SR); tt = np.arange(n) / SR
drone = (np.sin(2 * np.pi * 36 * tt) + 0.5 * np.sin(2 * np.pi * 54 * tt)) * adsr(n, 0.2, 0.5) * 0.25
add(sfx, drone, 14.0)
add(sfx, engine(14.0, 2.0, lambda u: 900 + 900 * np.sin(np.pi * u), cyl=8, gain=0.35, rough=0.6, bright=0.3, seed=5), 14.0)  # slowed growl
# --- scene 5: convoy pass-bys + final hit
for i, (t0, cyl, sd) in enumerate([(16.0, 8, 6), (16.6, 6, 7), (17.3, 8, 8)]):
    e = engine(t0, 2.2, lambda u: 2600 + 1800 * np.sin(np.pi * u), cyl=cyl, gain=0.4, rough=0.35, bright=0.8, seed=sd)
    n_ = len(e); u = np.arange(n_) / n_
    add(sfx, e * (np.sin(np.pi * u) ** 2), t0)
add(sfx, chime(784, 0.15, 1.5), 16.6)
add(sfx, impact(1.2), 16.5)
add(sfx, whoosh(0.6, gain=0.35, lo=500, hi=9000), 17.1)
add(sfx, impact(1.0), 19.5)
# --- winter ambience (wind)
wind = lp(rng.normal(0, 1, N), 700, 2) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.13 * T + 1) ** 2)
wind = hp(wind, 120)
amb_gain = np.where(T < 3, 0.5, np.where(T > 16, 0.5, 0.3))
sfx += wind * 0.12 * amb_gain

# ------------------------------------------------------------------ mix
fade = np.clip((DUR - T) / 0.3, 0, 1)
mix = (music * 0.75 + sfx * 0.85) * fade
mix = np.tanh(mix * 1.15)
mix = mix / np.max(np.abs(mix)) * 0.97
# slight stereo width: delay sfx/ambience component
L = mix
R = np.roll(mix, int(0.0007 * SR)) * 0.98
st = np.stack([L, R], axis=1)
with wave.open("audio3.wav", "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((st * 32767).astype(np.int16).tobytes())
print("ok")
