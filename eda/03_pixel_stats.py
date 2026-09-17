"""03 - Pixel-level statistics that measure each corruption seen in 02, train vs test and per class.
Every feature is scale-free so image size does not leak into it.
  colorfulness   : Hasler-Suesstrunk metric (0 = gray)          -> colour tint / coloured ink
  bg_lum, bg_sat : luminance / chroma of the background mode     -> sepia / parchment paper
  ink_frac       : fraction of pixels darker than Otsu threshold  -> content density / polarity
  midtone_frac   : pixels between 15% and 85% gray                -> binarised (train) vs photo/blur
  n_levels       : distinct gray levels                           -> binarised vs natural
  sharp          : mean |laplacian| at edges / contrast           -> blur / upscale
  noise          : MAD of high-pass inside background             -> speckle / grain
  black_border   : share of the 4 borders that are solid black    -> shift / rotate with black fill
  flat_gray_frac : big flat mid-gray blobs                        -> polygon occluder
  dark_bg        : background darker than ink                     -> inverted polarity
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from common import LABELS, load_frames, open_rgb, out_dir, pmap

OUT = out_dir("03_pixel_stats")
META = pd.read_csv(out_dir("01_profile") / "meta.csv")


def otsu(g):
    hist = np.bincount(g.ravel(), minlength=256).astype(float)
    p = hist / hist.sum(); w = np.cumsum(p); mu = np.cumsum(p * np.arange(256))
    between = (mu[-1] * w - mu) ** 2 / (w * (1 - w) + 1e-12)
    return int(np.argmax(between))


def feats(path):
    im = open_rgb(path)
    if max(im.size) > 768:
        im.thumbnail((768, 768))
    rgb = np.asarray(im).astype(np.float32)
    g = np.asarray(im.convert("L"))
    gf = g.astype(np.float32)
    rg, yb = rgb[..., 0] - rgb[..., 1], 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
    colorful = np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())
    t = otsu(g)
    dark = g <= t
    dark_bg = dark.mean() > 0.5  # background = majority class
    bg = ~dark if not dark_bg else dark
    bg_rgb = rgb[bg].mean(0) if bg.any() else rgb.reshape(-1, 3).mean(0)
    bg_lum = float(gf[bg].mean()) if bg.any() else float(gf.mean())
    bg_sat = float(bg_rgb.max() - bg_rgb.min())
    lap = np.abs(ndi.laplace(gf))
    edges = ndi.binary_dilation(dark ^ ndi.binary_erosion(dark))
    contrast = abs(gf[~dark].mean() - gf[dark].mean()) + 1e-3 if dark.any() and (~dark).any() else 1.0
    sharp = float(lap[edges].mean() / contrast) if edges.any() else 0.0
    hp = gf - ndi.median_filter(gf, 3)
    inner_bg = ndi.binary_erosion(bg, iterations=2)
    noise = float(np.median(np.abs(hp[inner_bg]))) if inner_bg.sum() > 50 else 0.0
    b = max(1, min(g.shape) // 40)
    borders = [g[:b], g[-b:], g[:, :b], g[:, -b:]]
    black_border = float(np.mean([(x < 30).mean() > 0.9 for x in borders]))
    # flat mid-gray blob: low local std & mid luminance, largest component share
    local_std = ndi.uniform_filter(gf ** 2, 5) - ndi.uniform_filter(gf, 5) ** 2
    flat_mid = (local_std < 4) & (gf > 70) & (gf < 180) & (np.abs(rg) < 12) & (np.abs(yb) < 12)
    lab, n = ndi.label(flat_mid)
    flat_gray = float(np.bincount(lab.ravel())[1:].max() / g.size) if n else 0.0
    return dict(colorfulness=float(colorful), bg_lum=bg_lum, bg_sat=bg_sat, ink_frac=float((~bg).mean()),
                midtone_frac=float(((g > 38) & (g < 217)).mean()), n_levels=int(len(np.unique(g))),
                sharp=sharp, noise=noise, black_border=black_border, flat_gray_frac=flat_gray,
                dark_bg=bool(dark_bg), otsu=t)


if __name__ == "__main__":
    df = load_frames()
    df = pd.concat([df, pd.DataFrame(pmap(feats, list(df.path)))], axis=1)
    df = df.merge(META[["image_id", "split", "fmt", "w", "h", "ratio", "bpp"]], on=["image_id", "split"])
    df.drop(columns="path").to_csv(OUT / "pixel_feats.csv", index=False)
    F = ["colorfulness", "bg_lum", "bg_sat", "ink_frac", "midtone_frac", "n_levels", "sharp", "noise",
         "black_border", "flat_gray_frac", "dark_bg"]
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print("== median per split")
    print(df.groupby("split")[F].median().T.round(3))
    print("\n== median per class (train) + test")
    g = df.assign(grp=df.label.fillna("TEST")).groupby("grp")[F].median().round(3)
    print(g)

    # corruption flags -> prevalence (these thresholds come from eyeballing 02 + the histograms below)
    flags = pd.DataFrame({
        "colour_tint(bg_sat>12)": df.bg_sat > 12,
        "colourful(>15)": df.colorfulness > 15,
        "not_binary(midtone>0.15)": df.midtone_frac > 0.15,
        "dark_background": df.dark_bg,
        "black_border>=1side": df.black_border > 0,
        "gray_occluder(flat>2%)": df.flat_gray_frac > 0.02,
        "noisy(noise>=2)": df.noise >= 2,
        "blurry(sharp<0.6)": df.sharp < 0.6,
        "bg_not_white(bg_lum<225)": df.bg_lum < 225,
    })
    prev = (flags.groupby(df.split).mean() * 100).T.round(1)
    prev["test/train"] = (prev.test / prev.train.clip(lower=0.1)).round(1)
    print("\n== corruption prevalence (% of split)")
    print(prev)
    print("\n== corruption prevalence per class (train, %)")
    print((flags[df.split == "train"].groupby(df.label[df.split == "train"]).mean() * 100).round(1).T)
    prev.to_csv(OUT / "corruption_prevalence.csv")

    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    for ax, f in zip(axes.ravel(), F + ["bpp"]):
        x = df[f].astype(float)
        lo, hi = np.nanpercentile(x, [0.5, 99.5])
        bins = np.linspace(lo, hi + 1e-6, 50)
        for s, c in [("train", "#4C72B0"), ("test", "#DD8452")]:
            ax.hist(x[df.split == s].clip(lo, hi), bins=bins, alpha=.55, density=True, color=c, label=s)
        ax.set_title(f); ax.legend()
    plt.tight_layout(); plt.savefig(OUT / "feat_hist_train_vs_test.png", dpi=90); plt.close()

    # show the most extreme test image for each flag - sanity check the metric measures what it claims
    fig, axes = plt.subplots(len(flags.columns), 4, figsize=(18, 3 * len(flags.columns)))
    te = df[df.split == "test"]
    order = {"colour_tint(bg_sat>12)": "bg_sat", "colourful(>15)": "colorfulness", "not_binary(midtone>0.15)": "midtone_frac",
             "dark_background": "ink_frac", "black_border>=1side": "black_border", "gray_occluder(flat>2%)": "flat_gray_frac",
             "noisy(noise>=2)": "noise", "blurry(sharp<0.6)": "sharp", "bg_not_white(bg_lum<225)": "bg_lum"}
    for i, (flag, col) in enumerate(order.items()):
        sub = te[flags.loc[te.index, flag]]
        sub = sub.sample(min(4, len(sub)), random_state=0)
        for j in range(4):
            axes[i, j].axis("off")
        for j, r in enumerate(sub.itertuples()):
            axes[i, j].imshow(open_rgb(r.path)); axes[i, j].set_title(f"{flag} {col}={getattr(r, col):.2f}", fontsize=8)
    plt.tight_layout(); plt.savefig(OUT / "flag_examples_test.png", dpi=70); plt.close()
    print("figures ->", OUT)
