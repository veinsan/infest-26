"""01 - Look at what the pipeline actually produces, per class, and at test. Then numeric sanity checks:
  * rotation: ink retained & fake ink added per fill mode (the "theoretically right, broken in practice" check)
  * remove_black_fill false positives on clean train (must remove ~0 ink)
  * background / ink level after preprocess: train vs test should match
"""
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eda"))
from common import LABELS, load_frames, pmap  # noqa: E402
from pipeline import _otsu, corrupt, ink_retained, load_rgb, preprocess, remove_black_fill, rotate, to_canvas  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs" / "01_visual_check"
OUT.mkdir(parents=True, exist_ok=True)


def show(ax, img, title, gray=False):
    ax.imshow(img, cmap="gray" if gray else None, vmin=0, vmax=255, interpolation="nearest"); ax.set_title(title, fontsize=8); ax.axis("off")


def ink_mask(img):
    g = np.asarray(img.convert("L")); return g < min(_otsu(g), 128)


def rot_check(path, seed):
    rng = random.Random(seed)
    img = load_rgb(path)
    base = ink_mask(img).sum()
    out = {}
    for fill in ["edge", "white", "black"]:
        a = rng.uniform(-15, 15)
        r = rotate(img, a, fill)
        raw_ink = ink_mask(r).sum()
        pp = np.asarray(preprocess(r)) < 128
        out[f"{fill}_raw_ink_ratio"] = raw_ink / max(base, 1)
        out[f"{fill}_true_retained"] = ink_retained(img, a)
        out[f"{fill}_after_prepro_ink_ratio"] = pp.sum() / max((np.asarray(preprocess(img)) < 128).sum(), 1)
    return out


def bf_check(path):
    g = np.asarray(load_rgb(path).convert("L"))  # same order as preprocess(): fill removal on raw gray
    before = (g < 128).sum(); after = (remove_black_fill(g) < 128).sum()
    return 1 - after / max(before, 1)


def levels(path):
    g = np.asarray(preprocess(load_rgb(path)))
    t = 128
    return dict(bg=float(np.median(g[g > t])) if (g > t).any() else np.nan, ink=float(np.median(g[g <= t])) if (g <= t).any() else np.nan,
                mid=float(((g > 40) & (g < 215)).mean()), out_h=g.shape[0], out_w=g.shape[1])


if __name__ == "__main__":
    df = load_frames()
    tr, te = df[df.split == "train"], df[df.split == "test"]
    rng = random.Random(0)

    for lab in LABELS:
        sub = tr[tr.label == lab].sample(5, random_state=1)
        fig, axes = plt.subplots(5, 5, figsize=(26, 9))
        for r, row in enumerate(sub.itertuples()):
            raw = load_rgb(row.path)
            pp = preprocess(raw)
            cor = corrupt(raw, rng)
            ppc = preprocess(cor)
            can = to_canvas(ppc, train=True, rng=rng)[0]
            show(axes[r, 0], raw, f"raw {raw.size}")
            show(axes[r, 1], pp, f"preprocess {pp.size}", True)
            show(axes[r, 2], cor, "corrupt (train aug)")
            show(axes[r, 3], ppc, "preprocess(corrupt)", True)
            show(axes[r, 4], can, "canvas 160x640 (model input)", True)
        fig.suptitle(f"train / {lab}"); plt.tight_layout(); plt.savefig(OUT / f"train_{lab}.png", dpi=60); plt.close()

    sub = te.sample(24, random_state=3)
    fig, axes = plt.subplots(12, 6, figsize=(30, 22))
    for k, row in enumerate(sub.itertuples()):
        r, c = divmod(k, 2)
        raw = load_rgb(row.path); pp = preprocess(raw); views = to_canvas(pp)
        show(axes[r, 3 * c], raw, f"test raw {raw.size}")
        show(axes[r, 3 * c + 1], pp, "preprocess", True)
        show(axes[r, 3 * c + 2], views[0], f"canvas (1/{len(views)} tiles)", True)
    plt.tight_layout(); plt.savefig(OUT / "test_samples.png", dpi=55); plt.close()

    # ---- numbers
    samp = tr.sample(600, random_state=0)
    from multiprocessing import Pool
    with Pool(20) as pool:
        rc = pd.DataFrame(pool.starmap(rot_check, zip(samp.path, range(len(samp)))))
    print("== RandomRotation +-15 on 600 train images: ink ratio vs original (1.0 = unchanged)")
    print(rc.describe(percentiles=[.05, .5, .95]).T[["5%", "50%", "95%"]].round(2))
    print("  share with >30% extra 'ink' (fill counted as ink):", (rc.filter(like="raw").gt(1.3).mean() * 100).round(1).to_dict())
    print("  share losing >50% ink:", (rc.filter(like="raw").lt(.5).mean() * 100).round(1).to_dict())
    print("  TRUE retention (mask rotated) <0.6 -> corrupt() re-draws the angle:", (rc.filter(like="true").lt(.6).mean() * 100).round(1).to_dict())
    # does preprocess undo black-fill fake ink? rotate with black fill then preprocess, vs white fill then preprocess
    print("  after preprocess, black-fill vs white-fill ink ratio median:",
          round(float((rc.black_after_prepro_ink_ratio / rc.white_after_prepro_ink_ratio.clip(lower=1e-3)).median()), 3))

    bf = np.array(pmap(bf_check, list(tr.path)))
    print(f"\n== remove_black_fill on ALL clean train: ink removed median={np.median(bf):.4f}, "
          f"images losing >5% ink: {(bf > .05).sum()} ({(bf > .05).mean() * 100:.1f}%)")
    worst = tr.assign(loss=bf).nlargest(8, "loss")
    print(worst[["image_id", "label", "loss"]].to_string())
    fig, axes = plt.subplots(8, 2, figsize=(14, 14))
    for r, row in enumerate(worst.itertuples()):
        g = np.asarray(load_rgb(row.path).convert("L"))
        show(axes[r, 0], g, f"{row.label} {row.image_id} before", True); show(axes[r, 1], remove_black_fill(g), f"after (lost {row.loss:.0%} ink)", True)
    plt.tight_layout(); plt.savefig(OUT / "black_fill_worst_train.png", dpi=60); plt.close()

    lv = pd.DataFrame(pmap(levels, list(df.path))); lv["split"] = df.split.values
    print("\n== after preprocess: medians (bg should be ~255, ink ~0-40, train ~ test)")
    print(lv.groupby("split")[["bg", "ink", "mid"]].quantile([.1, .5, .9]).unstack().round(1))
    print("->", OUT)
