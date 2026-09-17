"""07 - Resolution, stroke width and skew: the numbers that fix canvas size and rotation range.
  * effective text height after the v1 letterbox (160x640) vs alternatives -> is detail destroyed?
  * stroke width (distance transform) in px and relative to ink height -> blur / upscale in test
  * skew angle (projection-profile search) train vs test -> RandomRotation range from data, not a guess
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from common import LABELS, load_frames, open_rgb, out_dir, pmap

OUT = out_dir("07_resolution_skew")
PATCH = 16


def analyse(path):
    im = open_rgb(path).convert("L")
    w, h = im.size
    g = np.asarray(im, dtype=np.float32)
    lo, hi = np.percentile(g, [2, 98]); g = np.clip((g - lo) / max(hi - lo, 1), 0, 1)
    if g.mean() < .5:
        g = 1 - g
    ink = g < .5
    ys, xs = np.where(ink)
    ink_h = (ys.max() - ys.min() + 1) if len(ys) else h
    dt = ndi.distance_transform_edt(ink)
    ridge = dt[(dt >= ndi.maximum_filter(dt, 3)) & ink]
    stroke = float(2 * np.median(ridge)) if ridge.size else np.nan
    # skew: only for line-like crops, on a <=400px-wide copy
    angle = np.nan
    if w / h >= 3 and len(ys) > 50:
        sc = min(1.0, 400 / w)
        small = ndi.zoom(ink.astype(np.float32), sc, order=1) > .3 if sc < 1 else ink
        pad = np.pad(small, ((small.shape[1] // 3, small.shape[1] // 3), (0, 0)))
        best, angle = -1, 0.0
        for a in np.arange(-20, 20.5, 1.0):
            prof = ndi.rotate(pad.astype(np.float32), a, reshape=False, order=0).sum(1)
            v = prof.var()
            if v > best:
                best, angle = v, a
    return dict(w=w, h=h, ink_h=int(ink_h), stroke=stroke, angle=angle)


def eff_text_h(r, H, W):
    s = min(H / r.h, W / r.w)
    return r.ink_h * s


if __name__ == "__main__":
    df = load_frames()
    df = pd.concat([df, pd.DataFrame(pmap(analyse, list(df.path)))], axis=1)
    df["stroke_rel"] = df.stroke / df.ink_h
    df.drop(columns="path").to_csv(OUT / "res_skew.csv", index=False)
    lines = df[(df.w / df.h >= 2) & (df.h < 200)]  # line crops only (not posters)

    pd.set_option("display.width", 200)
    print("== text-line crops: ink height (px) quantiles")
    print(lines.groupby("split").ink_h.quantile([.05, .25, .5, .75, .95]).unstack().round(1))
    print("\n== effective ink height after resize, in ViT patches (16px). <1.5 patches = glyphs smeared")
    for H, W in [(160, 640), (224, 224), (224, 896), (128, 1024)]:
        e = lines.apply(lambda r: eff_text_h(r, H, W), axis=1) / PATCH
        print(f"  letterbox {H}x{W}: median={e.median():.2f} patches, <1.5 patches: "
              f"train {(e[lines.split == 'train'] < 1.5).mean() * 100:.1f}%  test {(e[lines.split == 'test'] < 1.5).mean() * 100:.1f}%")
        if (H, W) == (160, 640):
            print("   per class (train) median patches:", (e[lines.split == "train"].groupby(lines.label).median()).round(2).to_dict())
    print("  fixed-height 160 + width crop/tiling (no shrink): ink height =", round(160 * lines.ink_h.div(lines.h).median() / PATCH, 2), "patches for every line")

    print("\n== stroke width (px) and relative to ink height")
    print(df.groupby("split")[["stroke", "stroke_rel"]].quantile([.1, .5, .9]).unstack().round(3))
    print(df[df.split == "train"].groupby("label")[["stroke", "stroke_rel"]].median().round(3))

    print("\n== skew angle (deg) of line crops, |angle| quantiles")
    a = df.dropna(subset=["angle"])
    print(a.groupby("split").angle.apply(lambda s: s.abs().quantile([.5, .75, .9, .95, .99])).unstack().round(1))
    for s in ["train", "test"]:
        v = a[a.split == s].angle.abs()
        print(f"  {s}: |angle|>=3: {(v >= 3).mean() * 100:.1f}%  >=5: {(v >= 5).mean() * 100:.1f}%  >=10: {(v >= 10).mean() * 100:.1f}%  (n={len(v)})")

    fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
    for s, c in [("train", "#4C72B0"), ("test", "#DD8452")]:
        ax[0].hist(np.log2(lines[lines.split == s].ink_h), 40, alpha=.55, density=True, color=c, label=s)
        ax[1].hist(df[df.split == s].stroke_rel.clip(0, .5), 50, alpha=.55, density=True, color=c, label=s)
        ax[2].hist(a[a.split == s].angle, np.arange(-20.5, 21, 1), alpha=.55, density=True, color=c, label=s)
    ax[0].set_title("log2 ink height (line crops)"); ax[1].set_title("stroke width / ink height"); ax[2].set_title("skew angle (deg)")
    for x in ax: x.legend()
    plt.tight_layout(); plt.savefig(OUT / "res_stroke_skew.png", dpi=90); plt.close()

    te = a[(a.split == "test")].copy(); te["abs"] = te.angle.abs()
    sub = te[te["abs"] >= 4].sample(min(12, (te["abs"] >= 4).sum()), random_state=0)
    fig, axes = plt.subplots(4, 3, figsize=(18, 7))
    for ax_, r in zip(axes.ravel(), sub.itertuples()):
        ax_.imshow(open_rgb(r.path)); ax_.set_title(f"test angle={r.angle:+.0f}", fontsize=9)
    for ax_ in axes.ravel(): ax_.axis("off")
    plt.tight_layout(); plt.savefig(OUT / "skewed_test_examples.png", dpi=70); plt.close()
    print("->", OUT)
