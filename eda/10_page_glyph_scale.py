"""10 - Page images (25% of test, v2 OOF acc 0.74): how big is a glyph, and how big does v2 make it?
Glyph height = median height of text-like connected components on the preprocessed image
(components 3px..25% of image height, not line-shaped debris).
  * train LINE canvases (what the model is good at): glyph height in canvas px  -> the target scale
  * PAGE images: raw glyph height, glyph height after v2 letterbox 160x640, and in ViT patches
  * glyph count per page (how many tiles a page yields at target scale)
Theory: a ViT needs a glyph to span >=2-3 patches (32-48px) to resolve strokes/diacritics.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi

from common import ROOT, out_dir, pmap

sys.path.insert(0, str(ROOT / "prepro"))
import pipeline as P  # noqa: E402

OUT = out_dir("10_page_glyph_scale")
H, W = 160, 640


def glyph_stats(g):
    ink = np.asarray(g) < 128
    lab, n = ndi.label(ink)
    if n == 0:
        return np.nan, 0
    sl = ndi.find_objects(lab)
    hs = np.array([s[0].stop - s[0].start for s in sl]); ws = np.array([s[1].stop - s[1].start for s in sl])
    area = np.bincount(lab.ravel())[1:]
    ok = (hs >= 3) & (hs <= 0.25 * ink.shape[0] if ink.shape[0] > 200 else hs >= 3) & (area >= 8) & (ws < 8 * hs + 10)
    if ok.sum() < 3:
        return np.nan, int(ok.sum())
    hs = hs[ok]
    return float(np.median(hs[hs >= np.percentile(hs, 40)])), int(ok.sum())  # upper part: letters, not dots


def line_canvas_glyph(path):
    c = P.to_canvas(P.preprocess(P.load_rgb(path)), H, W, train=False)
    return glyph_stats(c[len(c) // 2])[0]


def page_row(path):
    pp = P.preprocess(P.load_rgb(path))
    raw_gh, n = glyph_stats(pp)
    s = min(H / pp.height, W / pp.width)
    return dict(pp_w=pp.width, pp_h=pp.height, raw_glyph_h=raw_gh, n_glyphs=n, v2_scale=s, v2_glyph_h=raw_gh * s)


if __name__ == "__main__":
    tr = pd.read_csv(out_dir("08_v2_gap_decomposition") / "train_types_oof.csv")
    te = pd.read_csv(out_dir("08_v2_gap_decomposition") / "test_types_pred.csv")
    tr["path"] = [str(ROOT / "data/images/train" / i) for i in tr.image_id]
    te["path"] = [str(ROOT / "data/images/test" / i) for i in te.image_id]
    lines = tr[tr.ctype == "line"].sample(800, random_state=0)
    lg = np.array(pmap(line_canvas_glyph, list(lines.path)), dtype=float)
    print(f"== train LINE canvas glyph height (px): p25={np.nanpercentile(lg, 25):.0f} median={np.nanmedian(lg):.0f} p75={np.nanpercentile(lg, 75):.0f} -> {np.nanmedian(lg) / 16:.1f} patches")

    pages = pd.concat([tr[tr.ctype == "page"].assign(split="train"), te[te.ctype == "page"].assign(split="test")], ignore_index=True)
    pages = pd.concat([pages, pd.DataFrame(pmap(page_row, list(pages.path)))], axis=1)
    pd.set_option("display.width", 220)
    print("\n== PAGE images")
    print(pages.groupby("split")[["pp_w", "pp_h", "raw_glyph_h", "n_glyphs", "v2_glyph_h"]].median().round(1))
    for s in ["train", "test"]:
        v = pages[pages.split == s].v2_glyph_h
        print(f"  {s}: glyph after v2 canvas <8px (half a patch): {(v < 8).mean():.0%} | <16px: {(v < 16).mean():.0%} | median {v.median():.1f}px = {v.median() / 16:.2f} patches")
    target = np.nanmedian(lg)
    pages["zoom_to_line_scale"] = target / pages.raw_glyph_h
    print(f"\n== to match line scale ({target:.0f}px glyph) pages need zoom (median) {pages.zoom_to_line_scale.median():.2f}x of the preprocessed page;"
          f" v2 used {pages.v2_scale.median():.3f}x -> v2 renders page glyphs {target / pages.v2_glyph_h.median():.0f}x smaller than line glyphs")
    tiles = (pages.pp_w * pages.zoom_to_line_scale.clip(upper=4)) * (pages.pp_h * pages.zoom_to_line_scale.clip(upper=4)) / (H * W)
    print(f"   area in 160x640 tiles at that scale: median {tiles.median():.0f}, p90 {tiles.quantile(.9):.0f}  -> too many; use a mid scale + keep ink-rich tiles")
    for z in [0.5, 0.35, 0.25]:
        gh = target * z
        t = (pages.pp_w * gh / pages.raw_glyph_h) * (pages.pp_h * gh / pages.raw_glyph_h) / (H * W)
        print(f"   glyph {gh:.0f}px ({gh / 16:.1f} patches): tiles median {t.median():.0f}, p90 {t.quantile(.9):.0f}")
    pages.drop(columns="path").to_csv(OUT / "pages.csv", index=False)
    print(f"\n== OOF acc on train pages by v2 glyph size: small(<8px) {(pages[(pages.split == 'train') & (pages.v2_glyph_h < 8)].pipe(lambda d: (d.pred_x == d.label.map({l: i for i, l in enumerate(sorted(set(tr.label)))})).mean())):.3f}"
          f" | larger {(pages[(pages.split == 'train') & (pages.v2_glyph_h >= 8)].pipe(lambda d: (d.pred_x == d.label.map({l: i for i, l in enumerate(sorted(set(tr.label)))})).mean())):.3f}")

    sub = pages[pages.split == "test"].dropna(subset=["raw_glyph_h"]).sample(6, random_state=2)
    fig, axes = plt.subplots(6, 3, figsize=(30, 16))
    for r, x in enumerate(sub.itertuples()):
        pp = P.preprocess(P.load_rgb(x.path))
        axes[r, 0].imshow(P.load_rgb(x.path)); axes[r, 0].set_title(f"test page raw, glyph {x.raw_glyph_h:.0f}px", fontsize=9)
        axes[r, 1].imshow(P.to_canvas(pp, H, W)[0], cmap="gray"); axes[r, 1].set_title(f"v2 canvas: glyph {x.v2_glyph_h:.1f}px", fontsize=9)
        z = min(4.0, 0.35 * target / x.raw_glyph_h)
        big = pp.resize((max(1, int(pp.width * z)), max(1, int(pp.height * z))), Image.Resampling.LANCZOS if z < 1 else Image.Resampling.BICUBIC)
        a = np.asarray(big); ys, xs = np.nonzero(a < 128)
        cy, cx = (int(np.median(ys)), int(np.median(xs))) if len(ys) else (0, 0)
        crop = big.crop((max(0, cx - W // 2), max(0, cy - H // 2), max(0, cx - W // 2) + W, max(0, cy - H // 2) + H))
        axes[r, 2].imshow(crop, cmap="gray"); axes[r, 2].set_title(f"one 160x640 tile at glyph {0.35 * target:.0f}px (zoom {z:.2f}x)", fontsize=9)
        for a_ in axes[r]: a_.axis("off")
    plt.tight_layout(); plt.savefig(OUT / "page_v2_vs_tile.png", dpi=55); plt.close()
    print("->", OUT)
