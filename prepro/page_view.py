"""Round 4 page view: pages (raw h >= 250) get a SQUARE grey canvas instead of the v2 160x640 letterbox.
eda/21: in v2 a page fills ~1/4 of the canvas (glyph ~3.5px); eda/22 frozen probe: resolution is the lever
(held-out page acc v2 .60 -> sq448 .69), colour adds nothing (rgb .68 vs gray .69), v2 crop/stretch costs ~2pts.
eda/23: test pages carry (almost) none of the synthetic line corruptions -> mild, photo-like augmentation only.
Everything else (line / block / glyph) keeps the exact v2 pipeline.
"""
import io
import random

import numpy as np
from PIL import Image, ImageFilter, ImageOps

PAGE_MIN_H = 250


def is_page(size):
    """size = (w, h) of the ORIGINAL file (before any cache thumbnail)."""
    return size[1] >= PAGE_MIN_H


def letterbox(img, S):
    """grey L image -> SxS, aspect kept, white padding."""
    s = S / max(img.size)
    img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))),
                     Image.Resampling.LANCZOS if s < 1 else Image.Resampling.BICUBIC)
    c = Image.new("L", (S, S), 255); c.paste(img, ((S - img.width) // 2, (S - img.height) // 2)); return c


def page_eval(rgb, S):
    return letterbox(rgb.convert("L"), S)


def page_train(rgb, S, rng=random):
    """mild photo-like augmentation (eda/23: test pages are clean web photos, not synthetically corrupted).
    Strengths calibrated in prepro/08 so that train-aug canvases stay indistinguishable from test pages
    (first try with blur p=.5 / rotate-expand p=.4 / gamma .75-1.33 opened a gap: domain AUC .45 -> .67)."""
    w, h = rgb.size
    if rng.random() < 0.6:  # re-framing: keep 85-100% of each side (layout / genre cues live at the borders too)
        cw, ch = int(w * rng.uniform(.85, 1)), int(h * rng.uniform(.85, 1))
        x, y = rng.randint(0, w - cw), rng.randint(0, h - ch)
        rgb = rgb.crop((x, y, x + cw, y + ch))
    g = rgb.convert("L")
    if rng.random() < 0.25:  # camera tilt, same canvas size; corners get the border median (no black wedge)
        a = np.asarray(g)
        fill = int(np.median(np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])))
        g = g.rotate(rng.uniform(-5, 5), resample=Image.Resampling.BICUBIC, expand=False, fillcolor=fill)
    if rng.random() < 0.4:  # exposure: gamma + contrast (grey-level ColorJitter)
        a = np.asarray(g, dtype=np.float32) / 255
        a = a ** rng.uniform(.85, 1.18)
        m = a.mean(); a = np.clip((a - m) * rng.uniform(.85, 1.15) + m, 0, 1)
        g = Image.fromarray((a * 255).astype(np.uint8))
    if rng.random() < 0.15:  # low-res web copy
        f = rng.uniform(.5, .9)
        small = g.resize((max(8, int(g.width * f)), max(8, int(g.height * f))), Image.Resampling.BILINEAR)
        g = small.resize(g.size, Image.Resampling.BICUBIC)
    if rng.random() < 0.3:
        b = io.BytesIO(); g.save(b, "JPEG", quality=rng.randint(50, 95)); g = Image.open(io.BytesIO(b.getvalue())).convert("L")
    return letterbox(g, S)


if __name__ == "__main__":
    img = Image.new("RGB", (900, 600), (230, 220, 190))
    assert page_eval(img, 320).size == (320, 320)
    r = random.Random(0)
    for _ in range(50):
        v = page_train(img, 320, r); assert v.size == (320, 320) and v.mode == "L"
        assert np.asarray(v).min() > 60, "no black wedge from rotation"
    assert is_page((100, 250)) and not is_page((900, 249))
    print("page_view self-check OK")
