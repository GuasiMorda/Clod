#!/usr/bin/env python3
"""Original drift-phonk style beat (120 BPM) + voiceover mix -> audio.wav"""
import numpy as np, subprocess, wave

SR = 44100
DUR = 20.0
BEAT = 0.5
N = int(SR * DUR)
rng = np.random.default_rng(1)

def env(n, decay):
    return np.exp(-np.arange(n) / SR * decay)

def add(buf, sig, t):
    i = int(t * SR)
    if i >= len(buf): return
    j = min(len(buf), i + len(sig))
    buf[i:j] += sig[:j - i]

def onepole_lp(x, fc):
    a = np.exp(-2 * np.pi * fc / SR)
    y = np.zeros_like(x); acc = 0.0
    # vectorised via lfilter-like loop is slow; use cumulative trick in chunks
    for k in range(len(x)):
        acc = (1 - a) * x[k] + a * acc
        y[k] = acc
    return y

def hp(x):
    return np.diff(x, prepend=0.0)

def kick():
    n = int(0.45 * SR); tt = np.arange(n) / SR
    f = 45 + 110 * np.exp(-tt * 28)
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = np.sin(ph) * env(n, 7)
    s[:200] += rng.uniform(-1, 1, 200) * np.linspace(0.6, 0, 200)
    return np.tanh(s * 2.2) * 0.9

def clap():
    n = int(0.25 * SR)
    nz = rng.uniform(-1, 1, n)
    nz = hp(hp(nz)) * 0.5
    e = env(n, 18)
    for off in (0, 0.012, 0.024):
        k = int(off * SR); e[k:k + 300] += 0.6
    return nz * e * 0.55

def hat(open_=False):
    n = int((0.18 if open_ else 0.05) * SR)
    nz = hp(hp(rng.uniform(-1, 1, n)))
    return nz * env(n, 12 if open_ else 70) * 0.18

def cowbell(semi):
    n = int(0.22 * SR); tt = np.arange(n) / SR
    k = 2 ** (semi / 12)
    s = np.sign(np.sin(2 * np.pi * 587 * k * tt)) + np.sign(np.sin(2 * np.pi * 845 * k * tt))
    s = hp(s) * 0.5 + s * 0.15
    return s * env(n, 16) * 0.16

def bass808(freq, length, glide_from=None):
    n = int(length * SR); tt = np.arange(n) / SR
    f = np.full(n, freq)
    if glide_from:
        f = freq + (glide_from - freq) * np.exp(-tt * 18)
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = np.sin(ph) * env(n, 1.6)
    s *= np.minimum(1, tt / 0.004)
    rel = np.minimum(1, (length - tt) / 0.03)
    return np.tanh(s * 3.5) * 0.5 * rel

def impact():
    n = int(1.6 * SR); tt = np.arange(n) / SR
    f = 30 + 90 * np.exp(-tt * 6)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 2.2)
    nz = rng.uniform(-1, 1, n) * env(n, 6) * 0.4
    return np.tanh((s + nz) * 2) * 0.9

def riser(length):
    n = int(length * SR); tt = np.arange(n) / SR
    nz = hp(rng.uniform(-1, 1, n))
    return nz * (tt / length) ** 2 * 0.35

drums = np.zeros(N); bass = np.zeros(N); lead = np.zeros(N); fx = np.zeros(N)
K, C = kick(), clap()

E1 = 41.2
prog = [0, 0, -4, -2]          # bar roots (semitones from E): E E C D
mel_a = {0: 12, 3: 12, 6: 15, 8: 19, 10: 17, 12: 15, 14: 14}
mel_b = {0: 12, 3: 12, 6: 15, 8: 12, 9: 10, 11: 12, 13: 15, 15: 17}

SLOW = (13.5, 15.5)
for b in range(40):
    t = b * BEAT
    in_slow = SLOW[0] <= t < SLOW[1]
    if b == 0 or t >= 19.5:
        continue
    if not in_slow:
        add(drums, K, t)
        if b % 2 == 1:
            add(drums, C, t)
        for s in range(4):
            if s % 2 == 0 or (b % 8 == 7):
                add(drums, hat(open_=(s == 2 and b % 4 == 3)), t + s * BEAT / 4)
    else:
        if b % 2 == 0:
            add(drums, K * 0.8, t)

for bar in range(10):
    t0 = bar * 4 * BEAT
    if t0 >= 19.5: break
    root = E1 * 2 ** (prog[bar % 4] / 12)
    in_slow = SLOW[0] <= t0 + 1 and t0 < SLOW[1]
    add(bass, bass808(root, 1.2, glide_from=root * 1.5 if bar % 2 else None), t0 + (0.0 if bar else 0.5))
    add(bass, bass808(root * (2 ** (3 / 12)), 0.7), t0 + 1.25)
    if not in_slow:
        mel = mel_a if bar % 2 == 0 else mel_b
        for step, semi in mel.items():
            add(lead, cowbell(semi + prog[bar % 4] - 12), t0 + step * BEAT / 4)

# FX: impacts at the start, on scene cuts, at the crash, risers before cuts
add(fx, impact(), 0.0)
for c in (3.0, 7.0, 11.0, 16.0):
    add(fx, impact() * 0.6, c)
    add(fx, riser(1.0), c - 1.0)
add(fx, impact() * 1.0, 14.1)
add(fx, impact() * 0.8, 19.5)

# slow-mo: muffle drums+lead
tt = np.arange(N) / SR
slowmask = ((tt >= SLOW[0]) & (tt < SLOW[1])).astype(float)
music = drums + bass * 0.9 + lead + fx
# sidechain bass/lead to kicks
duck = np.ones(N)
for b in range(40):
    i = int(b * BEAT * SR); n = int(0.18 * SR)
    duck[i:i + n] = np.minimum(duck[i:i + n], 1 - 0.55 * np.exp(-np.arange(n) / SR * 18)[:len(duck[i:i + n])])
music = drums + fx + (bass * 0.9 + lead) * duck
# cheap lowpass for slow-mo region via moving average
k = 60
lp = np.convolve(music, np.ones(k) / k, mode="same")
music = music * (1 - slowmask) + lp * 1.4 * slowmask
music *= np.minimum(1, (DUR - tt) / 0.4)  # fade tail

# ---- voice ----
def load(path, tempo):
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-af",
                          f"atempo={tempo:.3f},highpass=f=90,equalizer=f=3000:t=q:w=1.2:g=4,equalizer=f=200:t=q:w=1:g=2,"
                          "acompressor=threshold=-20dB:ratio=3:attack=5:release=80:makeup=3", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).astype(np.float64)

def dur(path):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                                capture_output=True, text=True).stdout)

slots = [(0.15, 2.85), (3.1, 6.85), (7.15, 10.85), (11.1, 15.6), (16.15, 19.6)]
voice = np.zeros(N)
for i, (a, b) in enumerate(slots, 1):
    p = f"vo3/t{i}.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"vo3/w{i}.wav", "-af",
                    "silenceremove=start_periods=1:start_threshold=-45dB,areverse,silenceremove=start_periods=1:start_threshold=-45dB,areverse", p], check=True)
    tempo = max(1.0, dur(p) / (b - a))
    print(i, round(dur(p),2), round(tempo,2))
    v = load(p, tempo)
    v = v / (np.abs(v).max() + 1e-9) * 0.9
    add(voice, v, a)

venv = np.convolve(np.abs(voice), np.ones(int(0.08 * SR)) / int(0.08 * SR), mode="same")
duckv = 1 - 0.6 * np.clip(venv * 8, 0, 1)
mix = music * duckv * 0.7 + voice * 1.15
mix = np.tanh(mix * 1.1)
mix = mix / np.abs(mix).max() * 0.95
st = np.stack([mix, mix], axis=1)
with wave.open("audio.wav", "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((st * 32767).astype(np.int16).tobytes())
# also music-only stem
m = music / np.abs(music).max() * 0.95
with wave.open("music_only.wav", "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((np.stack([m, m], 1) * 32767).astype(np.int16).tobytes())
print("ok")
