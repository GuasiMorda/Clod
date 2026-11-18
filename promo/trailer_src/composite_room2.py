#!/usr/bin/env python3
"""Put the landscape screen content into the 3D laptop, add glow/glare, caption and phone-camera look."""
import os, sys, json, glob, subprocess
import numpy as np, cv2
from multiprocessing import Pool
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H, FPS = 1080, 1920, 30
HERE = os.path.dirname(os.path.abspath(__file__))
RF = os.path.join(HERE, "..", "3d", "rframes2")
CORN = {}
for f in glob.glob(os.path.join(RF, "corners_*.json")):
    CORN.update({int(k): v for k, v in json.load(open(f)).items()})
SCW, SCH = 1728, 1080
SRC = np.float32([[0, 0], [SCW, 0], [SCW, SCH], [0, SCH]])

def caption():
    f = ImageFont.truetype(os.path.join(HERE, "fonts", "mont800.ttf"), 54)
    text = "Claude Code + STATE RP"
    im = Image.new("RGBA", (W, 200), (0, 0, 0, 0))
    sh = Image.new("RGBA", im.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).text((W / 2, 100), text, font=f, fill=(0, 0, 0, 200), anchor="mm", stroke_width=3, stroke_fill=(0, 0, 0, 200))
    im = Image.alpha_composite(im, sh.filter(ImageFilter.GaussianBlur(6)))
    ImageDraw.Draw(im).text((W / 2, 100), text, font=f, fill=(255, 255, 255, 255), anchor="mm")
    a = np.asarray(im).astype(np.float32) / 255
    return np.dstack([a[..., :3] * a[..., 3:4], a[..., 3]])
CAP = caption()

def frame(i):
    room = cv2.imread(os.path.join(RF, f"r_{i:04d}.jpg"))[:, :, ::-1].astype(np.float32) / 255
    scr = cv2.imread(os.path.join(HERE, "screen2", f"s_{i:04d}.jpg"))[:, :, ::-1].astype(np.float32) / 255
    q = np.float32(CORN[i])
    M = cv2.getPerspectiveTransform(SRC, q)
    warped = cv2.warpPerspective(scr, M, (W, H), flags=cv2.INTER_AREA)
    mask = cv2.warpPerspective(np.ones((SCH, SCW), np.float32), M, (W, H), flags=cv2.INTER_LINEAR)
    # screen is emissive: slightly lifted blacks (LCD), small softness like a phone filming a screen
    warped = cv2.GaussianBlur(warped, (0, 0), 0.6) * 0.95 + 0.012
    # glare: soft diagonal reflection of the room lights on the glass
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    g = np.exp(-(((xx - 0.85 * W) * 0.7 + (yy - 0.36 * H)) / 260) ** 2) * 0.05
    glare = g[..., None] * np.array([1.0, 0.75, 1.0], np.float32)
    out = room * (1 - mask[..., None]) + (warped + glare) * mask[..., None]
    # screen light spill / bloom around the panel
    sm = cv2.resize(warped * mask[..., None], (W // 8, H // 8), interpolation=cv2.INTER_AREA)
    sm = cv2.GaussianBlur(sm, (0, 0), 6)
    out += cv2.resize(sm, (W, H)) * 0.55
    # camera motion blur from screen-corner velocity (pan) and scale change (dolly)
    if i > 0 and (i - 1) in CORN:
        q0 = np.float32(CORN[i - 1]); v = (q - q0).mean(0)
        sp = float(np.hypot(*v))
        if sp > 2.5:
            L = int(min(70, sp * 0.9)) | 1
            k = np.zeros((L, L), np.float32); c = L // 2
            dx, dy = v / (sp + 1e-6)
            cv2.line(k, (int(c - dx * c), int(c - dy * c)), (int(c + dx * c), int(c + dy * c)), 1.0, 1)
            out = cv2.filter2D(out, -1, k / k.sum())
        per = lambda Q: float(sum(np.hypot(*(Q[j] - Q[j - 1])) for j in range(4)))
        sc = per(q) / per(q0)
        if abs(sc - 1) > 0.012:
            ccx, ccy = q.mean(0); acc = out.copy(); n = 5
            for j in range(1, n):
                s_ = 1 + (sc - 1) * 0.9 * j / n
                Mz = np.float32([[s_, 0, ccx - s_ * ccx], [0, s_, ccy - s_ * ccy]])
                acc += cv2.warpAffine(out, Mz, (W, H), borderMode=cv2.BORDER_REFLECT)
            out = acc / n
    # phone camera look: auto-exposure breathing, slight softness, noise, vignette
    ae = 1 + 0.025 * np.sin(i / FPS * 1.3)
    out = out * ae
    out = cv2.GaussianBlur(out, (0, 0), 0.5)
    blur = cv2.GaussianBlur(out, (0, 0), 2)
    out = np.clip(out * 1.25 - blur * 0.25, 0, None)          # phone sharpening halo
    rng = np.random.default_rng(i)
    n = rng.normal(0, 0.014, (H // 2, W // 2)).astype(np.float32)
    out += cv2.resize(n, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
    v = 1 - 0.25 * (((xx - W / 2) / (W * 0.9)) ** 2 + ((yy - H / 2) / (H * 0.9)) ** 2)
    out *= v[..., None]
    return (np.clip(out, 0, 1) * 255).astype(np.uint8).tobytes()

if __name__ == "__main__":
    if len(sys.argv) > 2:
        for s in sys.argv[2].split(","):
            i = int(s); im = np.frombuffer(frame(i), np.uint8).reshape(H, W, 3)
            cv2.imwrite(f"cr_{i:04d}.jpg", im[:, :, ::-1])
        sys.exit()
    out = sys.argv[1]
    nf = len(CORN)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                           "-i", "-", "-i", os.path.join(HERE, "audio_music.wav"), "-c:v", "libx264", "-preset", "slow", "-crf", "20",
                           "-pix_fmt", "yuv420p", "-af", "loudnorm=I=-10:TP=-1.0:LRA=8", "-c:a", "aac", "-b:a", "256k", "-shortest",
                           "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    with Pool(4) as p:
        for b in p.imap(frame, range(nf), chunksize=4):
            ff.stdin.write(b)
    ff.stdin.close(); ff.wait()
    print("done", out)
