"""Preprocessing + augmentation for INFEST aksara classification. Import this from the training notebook.

Order matters:   raw RGB --(train only) corrupt()--> preprocess() --> to_canvas()
corrupt() BEFORE preprocess() because the test corruptions were applied to raw images; the network must
see train images that went through exactly the same normalisation the corrupted test images go through.

Every number below comes from eda/outputs (see eda/README.md):
  rotation +-15 deg          07: test |skew| p95=10, p99=13; 30% of test lines >=3 deg (train 4%)
  tint p=.45                 03: bg_sat>12 in 44% test vs 3% train
  blur/pixelate p=.55        03: sharp<0.6 in 69% test vs 12% train
  occluder p=.12, dashes .15 03 flat_gray 6.6% test (detector under-counts, seen in 02 zoom)
  black bar p=.07            03: black border 7.3% test vs 1.2% train
  jpeg p=.3                  01: JPEG 21% test vs 4.6% train
  canvas H=160 fixed height  07: letterbox 160x640 leaves bali/lontara at ~2 patches of ink
"""
import io
import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from scipy import ndimage as ndi


# ----------------------------------------------------------------------------- load
def load_rgb(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "P", "PA"):
            im = im.convert("RGBA")
            im = Image.alpha_composite(Image.new("RGBA", im.size, "white"), im)
        return im.convert("RGB")


# ----------------------------------------------------------------------------- deterministic (train + test)
def _otsu(g):
    hist = np.bincount(g.ravel(), minlength=256).astype(float)
    p = hist / hist.sum(); w = np.cumsum(p); mu = np.cumsum(p * np.arange(256))
    return int(np.argmax((mu[-1] * w - mu) ** 2 / (w * (1 - w) + 1e-12)))


def remove_black_fill(g, dark=40, min_contact=0.3, min_solidity=0.9, max_holes=0.03):
    """Solid black regions glued to the border (shift / rotate fill, black bars) -> white.
    A border-touching pure-black component is fill when it has no holes (a real dark background has
    light text inside it) AND either (a) runs along >=min_contact of a side or (b) is near-convex
    (rotation corner triangle). Glyph-sized images (both sides <64px) only lose full-side bars.
    Only the thick core (survives 2px erosion) is removed, so thin strokes glued to the fill stay."""
    blk = g < dark
    if not blk.any():
        return g
    lab, n = ndi.label(blk)
    h, w = g.shape
    border = np.zeros_like(blk); border[0] = border[-1] = border[:, 0] = border[:, -1] = True
    tiny = max(h, w) < 64  # glyph-sized crops (e.g. a 20x20 bold numeral): only full-side bars count
    contact = 0.9 if tiny else min_contact
    objs = ndi.find_objects(lab)
    kill = set()
    for i in set(np.unique(lab[border & blk]).tolist()) - {0}:
        sl = objs[i - 1]
        comp = lab[sl] == i
        area = comp.sum()
        filled = ndi.binary_fill_holes(comp).sum()
        if (filled - area) / filled > max_holes or np.median(g[sl][comp]) > 15:
            continue
        run = max((lab[0] == i).sum() / w, (lab[-1] == i).sum() / w, (lab[:, 0] == i).sum() / h, (lab[:, -1] == i).sum() / h)
        if run >= contact:
            kill.add(i); continue
        if tiny or area < 0.005 * g.size:
            continue
        ys, xs = np.nonzero(comp)
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(np.c_[np.r_[ys, ys + 1, ys, ys + 1], np.r_[xs, xs, xs + 1, xs + 1]]).volume
        except Exception:
            continue
        if area / hull >= min_solidity:
            kill.add(i)
    if not kill:
        return g
    mask = np.isin(lab, list(kill))
    mask &= ndi.binary_dilation(ndi.binary_erosion(mask, iterations=2, border_value=1), iterations=3)
    out = g.copy()
    out[mask] = 255
    return out


def preprocess(rgb):
    """RGB PIL -> L PIL: gray, black fill removed, dark-ink-on-white, contrast normalised, cropped to ink.
    Fill removal runs BEFORE the polarity test: black rotation corners on a thin line can cover >50% of
    the frame and would otherwise flip the whole image to white-on-black (seen in debug_black_rotation.png)."""
    g = remove_black_fill(np.asarray(rgb.convert("L")))
    t = _otsu(g)
    if (g > t).mean() < 0.5:  # background is the majority class; if it is the dark side, invert
        g = 255 - g
    t = _otsu(g)
    ink, bg = g[g <= t], g[g > t]
    if ink.size and bg.size:
        lo, hi = np.percentile(ink, 10), np.median(bg)
        if hi - lo >= 20:  # skip blank / single-tone images
            g = np.clip((g.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    return Image.fromarray(crop_to_ink(g))


def crop_to_ink(g, thr=128, trim=0.005, margin=0.08):
    ink = g < thr
    if ink.sum() < 20:
        return g
    def span(profile):
        c = np.cumsum(profile) / profile.sum()
        return int(np.searchsorted(c, trim)), int(np.searchsorted(c, 1 - trim))
    y0, y1 = span(ink.sum(1)); x0, x1 = span(ink.sum(0))
    if ink[y0:y1 + 1, x0:x1 + 1].mean() > 0.6:  # crop would be one solid blob (a black dash on a sparse page) -> keep all
        return g
    m = int(round(margin * max(y1 - y0, 8)))
    return g[max(0, y0 - m):y1 + m + 1, max(0, x0 - m):x1 + m + 1]


def to_canvas(img, H=160, W=640, train=False, rng=random):
    """Line-like crops (w/h>=2): resize to fixed height H, then random window (train) or tiles (eval).
    Others (posters, pages, square patches): letterbox. Returns list of HxW L images (len 1 when train)."""
    w, h = img.size
    if w / h >= 2:
        nw = max(1, round(w * H / h))
        img = img.resize((nw, H), Image.Resampling.BICUBIC if nw > w else Image.Resampling.LANCZOS)
        if nw <= W:
            c = Image.new("L", (W, H), 255); c.paste(img, ((W - nw) // 2, 0)); return [c]
        if train:
            x = rng.randint(0, nw - W); return [img.crop((x, 0, x + W, H))]
        n = math.ceil((nw - W) / (W // 2)) + 1
        xs = np.linspace(0, nw - W, n).round().astype(int)
        return [img.crop((x, 0, x + W, H)) for x in xs]
    s = min(H / h, W / w)
    img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.Resampling.BICUBIC if s > 1 else Image.Resampling.LANCZOS)
    c = Image.new("L", (W, H), 255); c.paste(img, ((W - img.width) // 2, (H - img.height) // 2)); return [c]


# ----------------------------------------------------------------------------- train-only test-like corruption
PARCHMENT = [(236, 224, 196), (222, 205, 170), (245, 238, 214), (210, 196, 160), (196, 186, 160), (250, 246, 230), (170, 170, 170), (230, 230, 230)]
INKS = [(20, 20, 20), (90, 20, 30), (30, 70, 40), (40, 40, 90), (70, 50, 30), (60, 30, 80)]


def rotate(img, angle, fill="edge"):
    """RandomRotation that mimics the test set: frame size kept, corners filled by edge-replicate / white / black."""
    if fill == "edge":
        a = np.asarray(img); pad = max(a.shape[:2]) // 2
        big = Image.fromarray(np.pad(a, ((pad, pad), (pad, pad), (0, 0)), mode="edge"))
        big = big.rotate(angle, resample=Image.Resampling.BICUBIC)
        return big.crop((pad, pad, pad + img.width, pad + img.height))
    color = (255, 255, 255) if fill == "white" else (0, 0, 0)
    return img.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=color)


def _ink_mask(img):
    g = np.asarray(img.convert("L")); return g < min(_otsu(g), 128)


def ink_retained(img, angle):
    """Share of the ORIGINAL ink still inside the frame after rotation (fill can't fake it)."""
    m = Image.fromarray(_ink_mask(img).astype(np.uint8) * 255)
    return (np.asarray(m.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)) > 0).sum() / max((np.asarray(m) > 0).sum(), 1)


def corrupt(img, rng=random):
    """RGB PIL -> RGB PIL with test-like corruptions. Rotation angle is re-drawn while <60% of the ink stays in frame."""
    w, h = img.size
    if rng.random() < 0.5:
        for _ in range(4):
            angle = rng.uniform(-15, 15)
            if ink_retained(img, angle) >= 0.6:
                img = rotate(img, angle, rng.choice(["edge"] * 5 + ["white"] * 2 + ["black"])); break
    if rng.random() < 0.25 and w > 60:  # partial crop along the line
        cw, ch = int(w * rng.uniform(0.6, 1.0)), int(h * rng.uniform(0.85, 1.0))
        x, y = rng.randint(0, w - cw), rng.randint(0, h - ch)
        img = img.crop((x, y, x + cw, y + ch)); w, h = img.size
    if rng.random() < 0.45:  # tint: re-colour ink and paper
        g = np.asarray(img.convert("L"), dtype=np.float32)[..., None] / 255
        bg = np.array(rng.choice(PARCHMENT), dtype=np.float32) * rng.uniform(0.85, 1.05)
        ink = np.array(rng.choice(INKS), dtype=np.float32) + rng.uniform(0, 60)
        img = Image.fromarray(np.clip(ink * (1 - g) + bg * g, 0, 255).astype(np.uint8))
    if rng.random() < 0.45:
        if rng.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2) * max(h, 32) / 64))
        else:
            f = rng.uniform(0.35, 0.8)
            small = img.resize((max(4, int(w * f)), max(4, int(h * f))), Image.Resampling.BILINEAR)
            img = small.resize((w, h), rng.choice([Image.Resampling.NEAREST, Image.Resampling.BILINEAR, Image.Resampling.BICUBIC]))
    if rng.random() < 0.08:  # semi-transparent gray polygons
        over = Image.new("RGBA", img.size, (0, 0, 0, 0)); d = ImageDraw.Draw(over)
        for _ in range(rng.randint(1, 3)):
            v = rng.randint(60, 150)
            d.polygon([(rng.uniform(-.1, 1.1) * w, rng.uniform(-.1, 1.1) * h) for _ in range(3)], fill=(v, v, v, rng.randint(120, 230)))
        img = Image.alpha_composite(img.convert("RGBA"), over).convert("RGB")
    if rng.random() < 0.15:  # small black dashes / cutout
        d = ImageDraw.Draw(img)
        for _ in range(rng.randint(1, 4)):
            rw, rh = rng.uniform(.03, .1) * w, rng.uniform(.03, .1) * h + 1
            x, y = rng.uniform(0, w - rw), rng.uniform(0, h - rh)
            d.rectangle([x, y, x + rw, y + rh], fill=(0, 0, 0))
    if rng.random() < 0.07:  # black bar from a vertical shift
        a = np.asarray(img).copy(); k = int(h * rng.uniform(0.05, 0.25)) + 1
        if rng.random() < 0.5:
            a[:-k] = a[k:]; a[-k:] = 0
        else:
            a[k:] = a[:-k]; a[:k] = 0
        img = Image.fromarray(a)
    if rng.random() < 0.15:
        a = np.asarray(img, dtype=np.float32) + np.random.default_rng(rng.randint(0, 2**31)).normal(0, rng.uniform(5, 25), (h, w, 1))
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if rng.random() < 0.3:
        buf = io.BytesIO(); img.save(buf, "JPEG", quality=rng.randint(20, 75)); buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img


def train_view(path, rng=random, H=160, W=640):
    return to_canvas(preprocess(corrupt(load_rgb(path), rng)), H, W, train=True, rng=rng)[0]


def eval_views(path, H=160, W=640):
    return to_canvas(preprocess(load_rgb(path)), H, W, train=False)


if __name__ == "__main__":  # self-check on synthetic images
    img = Image.new("RGB", (400, 60), "white"); ImageDraw.Draw(img).text((10, 20), "abc def ghi jkl", fill="black")
    p = preprocess(img)
    assert p.mode == "L" and p.width < 400 and np.asarray(p).min() == 0
    inv = preprocess(ImageOps.invert(img))  # white-on-black must come out black-on-white
    assert np.asarray(inv).mean() > 128
    barred = np.asarray(img).copy(); barred[-15:] = 0
    g = remove_black_fill(np.asarray(Image.fromarray(barred).convert("L")))
    assert (g[-15:] == 255).all() and (g < 128).sum() > 0, "bar removed, text kept"
    page = np.full((800, 600), 255, np.uint8); page[::40, ::40] = 180; page[400:410, 300:360] = 0
    assert crop_to_ink(page).shape == page.shape, "sparse page + black dash must not collapse to the dash"
    views = to_canvas(Image.new("L", (3000, 80), 255), train=False)
    assert all(v.size == (640, 160) for v in views) and len(views) > 1
    r = random.Random(0)
    for _ in range(50):
        c = corrupt(img, r); assert c.mode == "RGB"
    print("pipeline self-check OK")
