"""21 - Round 4 starts from the images themselves: what IS a page, per class, in train vs test, and what does v2 see?
  * size / aspect / colour / JPEG stats of page images, train vs test
  * galleries: train pages per class (raw), test pages grouped by v2 prediction (raw + v2 conf)
  * 'what the model sees': raw page next to its v2 160x640 canvas
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from common import LABELS, ROOT, load_frames, open_rgb, out_dir, pmap

sys.path.insert(0, str(ROOT / "prepro" / "outputs"))
import pipeline_v2_snapshot as V2  # noqa: E402

OUT = out_dir("21_page_gallery")


def stats(path):
    with Image.open(path) as im:
        fmt, (w, h) = im.format, im.size
    a = np.asarray(open_rgb(path).resize((128, 128)), dtype=np.float32)
    g = a.mean(2)
    return dict(w=w, h=h, fmt=fmt, colourful=float(np.abs(a[..., 0] - a[..., 2]).mean()), bg=float(np.median(g)),
                dark_frac=float((g < 100).mean()), sat=float((a.max(2) - a.min(2)).mean()))


def grid(rows, fname, title, ncol=6, canvas=False):
    n = len(rows); nr = int(np.ceil(n / ncol)) or 1
    fig, axes = plt.subplots(nr * (2 if canvas else 1), ncol, figsize=(3.2 * ncol, (2.6 if not canvas else 3.4) * nr), squeeze=False)
    for a in axes.ravel(): a.axis("off")
    for k, r in enumerate(rows):
        i, j = divmod(k, ncol)
        ax = axes[i * (2 if canvas else 1), j]
        ax.imshow(open_rgb(r["path"])); ax.set_title(r["title"], fontsize=7)
        if canvas:
            axes[i * 2 + 1, j].imshow(V2.eval_views(r["path"])[0], cmap="gray", vmin=0, vmax=255)
    fig.suptitle(title); plt.tight_layout(); plt.savefig(OUT / fname, dpi=60); plt.close()


if __name__ == "__main__":
    df = load_frames()
    tr = pd.read_csv(out_dir("08_v2_gap_decomposition") / "train_types_oof.csv")
    te = pd.read_csv(out_dir("08_v2_gap_decomposition") / "test_types_pred.csv")
    ct = pd.concat([tr[["image_id", "split", "ctype"]], te[["image_id", "split", "ctype"]]])
    df = df.merge(ct, on=["image_id", "split"])
    v2 = pd.read_csv(ROOT / "results/v2/main/submission.csv").set_index("image_id").label
    df["v2"] = np.where(df.split == "test", v2.reindex(df.image_id).to_numpy(dtype=object), None)
    conf = te.set_index("image_id").conf
    df["conf"] = conf.reindex(df.image_id).to_numpy()
    pg = df[df.ctype == "page"].reset_index(drop=True)
    pg = pd.concat([pg, pd.DataFrame(pmap(stats, list(pg.path)))], axis=1)
    pg["aspect"] = pg.w / pg.h
    pg["key"] = np.where(pg.split == "train", pg.label, pg.v2)
    print(f"== pages: train {(pg.split == 'train').sum()} | test {(pg.split == 'test').sum()}")
    print(pg.groupby("split")[["w", "h", "aspect", "colourful", "sat", "bg", "dark_frac"]].median().round(2).to_string())
    print("JPEG share:", pg.groupby("split").fmt.apply(lambda s: round((s == "JPEG").mean(), 3)).to_dict())
    print("\n== per class (train label / test v2 pred): n, median w,h,aspect, sat")
    print(pg.groupby(["key", "split"]).agg(n=("w", "size"), w=("w", "median"), h=("h", "median"), aspect=("aspect", "median"),
                                           sat=("sat", "median"), jpeg=("fmt", lambda s: (s == "JPEG").mean())).round(2).unstack().to_string())
    print("\n== test page v2 confidence by predicted class")
    t = pg[pg.split == "test"]
    print(t.groupby("v2").conf.describe()[["count", "mean", "25%", "50%"]].round(2).to_string())
    print(f"   test pages conf<0.6: {(t.conf < .6).mean():.1%} (all test non-page: {(te[te.ctype != 'page'].conf < .6).mean():.1%})")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for s, c in [("train", "tab:blue"), ("test", "tab:red")]:
        x = pg[pg.split == s]
        axes[0].hist(np.log2(x.aspect), bins=30, alpha=.5, color=c, density=True, label=s)
        axes[1].hist(np.log10(x.w * x.h), bins=30, alpha=.5, color=c, density=True, label=s)
        axes[2].hist(x.sat, bins=30, alpha=.5, color=c, density=True, label=s)
    for a, t_ in zip(axes, ["log2 aspect (w/h)", "log10 pixels", "saturation"]): a.set_title(t_); a.legend()
    plt.tight_layout(); plt.savefig(OUT / "page_stats_train_vs_test.png", dpi=70); plt.close()

    rng = np.random.default_rng(0)
    for lab in LABELS:
        x = pg[(pg.split == "train") & (pg.label == lab)]
        rows = [dict(path=r.path, title=f"{r.image_id} {r.w}x{r.h}") for r in x.itertuples()]
        grid(rows[:36], f"train_pages_{lab}.png", f"TRAIN pages label={lab} (n={len(x)})")
        y = pg[(pg.split == "test") & (pg.v2 == lab)].sort_values("conf")
        rows = [dict(path=r.path, title=f"{r.image_id} conf={r.conf:.2f}") for r in y.itertuples()]
        grid(rows[:48], f"test_pages_pred_{lab}.png", f"TEST pages v2 pred={lab} (n={len(y)}) sorted by conf")
    sel = pg[pg.split == "test"].sample(12, random_state=1)
    grid([dict(path=r.path, title=f"test v2={r.v2} {r.w}x{r.h}") for r in sel.itertuples()], "what_v2_sees_test_pages.png",
         "raw test page (top) vs v2 canvas 160x640 (bottom)", ncol=4, canvas=True)
    pg.drop(columns="path").to_csv(OUT / "pages.csv", index=False)
    print("->", OUT)
