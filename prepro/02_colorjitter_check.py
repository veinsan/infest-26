"""02 - "Theory says ColorJitter fixes the colour gap" - does it survive the pipeline?
Compares, on the FINAL model input (160x640 canvas):
  none         preprocess only
  cj_before    torchvision ColorJitter on raw RGB, then preprocess     (theory: simulates tinted test)
  cj_after     ColorJitter on the preprocessed canvas (what v1 did)
  tint         pipeline.corrupt-style tint only, then preprocess
Metrics: pixel change vs 'none' (0 = the aug is a no-op after preprocess), background / ink level of the
canvas vs the TEST canvas distribution (an aug that creates levels test never has = new train/test gap)."""
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms import ColorJitter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eda"))
from common import load_frames, pmap  # noqa: E402
from pipeline import INKS, PARCHMENT, corrupt, load_rgb, preprocess, to_canvas  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs" / "02_colorjitter_check"
OUT.mkdir(parents=True, exist_ok=True)
CJ = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)


def canvas(img):
    return to_canvas(img, train=False)[0]


def tint(img, rng):
    g = np.asarray(img.convert("L"), dtype=np.float32)[..., None] / 255
    bg = np.array(rng.choice(PARCHMENT), dtype=np.float32) * rng.uniform(0.85, 1.05)
    ink = np.array(rng.choice(INKS), dtype=np.float32) + rng.uniform(0, 60)
    return Image.fromarray(np.clip(ink * (1 - g) + bg * g, 0, 255).astype(np.uint8))


def variants(path, seed=0):
    import torch
    rng = random.Random(seed); torch.manual_seed(seed)
    raw = load_rgb(path)
    none = canvas(preprocess(raw))
    return {
        "none": none,
        "cj_before": canvas(preprocess(CJ(raw))),
        "cj_after": CJ(none.convert("RGB")).convert("L"),
        "tint": canvas(preprocess(tint(raw, rng))),
        "full_corrupt": canvas(preprocess(corrupt(raw, rng))),
    }


def levels(a):
    a = np.asarray(a, dtype=np.float32)
    return float(np.percentile(a, 90)), float(np.percentile(a, 5))  # background level, ink level


def measure(path):
    v = variants(path)
    base = np.asarray(v["none"], dtype=np.float32)
    out = {}
    for k, im in v.items():
        a = np.asarray(im, dtype=np.float32)
        out[f"{k}_diff"] = float(np.abs(a - base).mean())
        out[f"{k}_bg"], out[f"{k}_ink"] = levels(im)
    return out


def test_levels(path):
    return levels(canvas(preprocess(load_rgb(path))))


if __name__ == "__main__":
    df = load_frames()
    tr = df[df.split == "train"].sample(800, random_state=0)
    te = df[df.split == "test"]
    m = pd.DataFrame(pmap(measure, list(tr.path)))
    tl = np.array(pmap(test_levels, list(te.path)))
    print("== mean |pixel change| vs preprocess-only (0 = aug erased by preprocess)")
    print(m.filter(like="_diff").describe(percentiles=[.5, .9]).T[["mean", "50%", "90%"]].round(2))
    print("\n== canvas background (p90) and ink (p5) levels")
    rows = []
    for k in ["none", "cj_before", "cj_after", "tint", "full_corrupt"]:
        rows.append(dict(variant=k, bg_median=m[f"{k}_bg"].median(), bg_share_below_235=(m[f"{k}_bg"] < 235).mean() * 100,
                         ink_median=m[f"{k}_ink"].median(), ink_share_above_60=(m[f"{k}_ink"] > 60).mean() * 100))
    rows.append(dict(variant="TEST", bg_median=np.median(tl[:, 0]), bg_share_below_235=(tl[:, 0] < 235).mean() * 100,
                     ink_median=np.median(tl[:, 1]), ink_share_above_60=(tl[:, 1] > 60).mean() * 100))
    print(pd.DataFrame(rows).round(1).to_string(index=False))

    sub = tr.sample(6, random_state=4)
    fig, axes = plt.subplots(6, 6, figsize=(36, 10))
    for r, row in enumerate(sub.itertuples()):
        raw = load_rgb(row.path)
        axes[r, 0].imshow(CJ(raw)); axes[r, 0].set_title("ColorJitter on raw (before preprocess)", fontsize=8)
        for c, (k, im) in enumerate(variants(row.path, r).items(), 1):
            axes[r, c].imshow(im, cmap="gray", vmin=0, vmax=255); axes[r, c].set_title(f"canvas: {k}", fontsize=8)
        for a in axes[r]:
            a.axis("off")
    plt.tight_layout(); plt.savefig(OUT / "colorjitter_variants.png", dpi=55); plt.close()
    print("->", OUT)
