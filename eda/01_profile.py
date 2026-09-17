"""01 - File-level profile: class balance, formats, colour modes, resolution, aspect ratio, bytes.
Compares train vs test on every axis, because holdout F1 0.976 vs public LB 0.863 smells like shift."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from common import LABELS, load_frames, out_dir, pmap

OUT = out_dir("01_profile")


def meta(path):
    with Image.open(path) as im:
        w, h = im.size
        return dict(fmt=im.format, mode=im.mode, w=w, h=h, bytes=__import__("os").path.getsize(path),
                    has_alpha=im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info,
                    exif_orient=(im.getexif() or {}).get(274, 1), dpi=str(im.info.get("dpi", "")))


if __name__ == "__main__":
    df = load_frames()
    df = pd.concat([df, pd.DataFrame(pmap(meta, list(df.path)))], axis=1)
    df["ratio"] = df.w / df.h
    df["megapix"] = df.w * df.h / 1e6
    df["bpp"] = df.bytes * 8 / (df.w * df.h)  # bits/pixel: rendered PNG << photo JPEG
    df.drop(columns="path").to_csv(OUT / "meta.csv", index=False)

    tr, te = df[df.split == "train"], df[df.split == "test"]
    pd.set_option("display.width", 200)
    print("== class balance (train)")
    vc = tr.label.value_counts()
    print(pd.DataFrame({"n": vc, "pct": (vc / len(tr) * 100).round(1)}))
    print("imbalance max/min =", round(vc.max() / vc.min(), 2))

    print("\n== format x mode share (%)")
    print((pd.crosstab(df.fmt + "/" + df["mode"], df.split, normalize="columns") * 100).round(2))
    print("\nalpha:", df.groupby("split").has_alpha.mean().round(4).to_dict(),
          " exif-rotated:", df.groupby("split").exif_orient.apply(lambda s: (s != 1).sum()).to_dict())

    q = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    for col in ["w", "h", "ratio", "megapix", "bytes", "bpp"]:
        print(f"\n== {col} quantiles")
        print(df.groupby("split")[col].quantile(q).unstack().round(3))

    print("\n== per-class median (train)")
    print(tr.groupby("label")[["w", "h", "ratio", "bytes", "bpp"]].median().round(2))
    print("\n== per-class JPEG share (train, %)")
    print((tr.assign(jpg=tr.fmt == "JPEG").groupby("label").jpg.mean() * 100).round(1))

    # height buckets: line-level (<=100px) vs block / photo
    bins = [0, 32, 64, 100, 200, 500, 1e9]
    print("\n== height buckets (% of split)")
    print((pd.crosstab(pd.cut(df.h, bins), df.split, normalize="columns") * 100).round(1))
    print("\n== height buckets per class (train, %)")
    print((pd.crosstab(tr.label, pd.cut(tr.h, bins), normalize="index") * 100).round(1))
    print("\n== aspect buckets (% of split)")
    rb = [0, 0.8, 1.25, 2, 5, 10, 20, 1e9]
    print((pd.crosstab(pd.cut(df.ratio, rb), df.split, normalize="columns") * 100).round(1))

    summary = {
        "n_train": len(tr), "n_test": len(te),
        "test_jpeg_pct": round((te.fmt == "JPEG").mean() * 100, 1),
        "train_jpeg_pct": round((tr.fmt == "JPEG").mean() * 100, 1),
        "test_h_gt100_pct": round((te.h > 100).mean() * 100, 1),
        "train_h_gt100_pct": round((tr.h > 100).mean() * 100, 1),
    }
    print("\n", json.dumps(summary))

    # ---- figures
    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    vc.reindex(LABELS).plot.bar(ax=ax[0, 0], color="#4C72B0", title="train class counts")
    for col, a, log in [("w", ax[0, 1], True), ("h", ax[0, 2], True), ("ratio", ax[1, 0], True), ("bpp", ax[1, 1], True)]:
        bins_ = np.logspace(np.log10(max(df[col].min(), 1e-2)), np.log10(df[col].max()), 60)
        for s, c in [("train", "#4C72B0"), ("test", "#DD8452")]:
            a.hist(df[df.split == s][col], bins=bins_, alpha=.55, density=True, label=s, color=c)
        a.set_xscale("log"); a.set_title(f"{col} (density)"); a.legend()
    a = ax[1, 2]
    for s, c in [("train", "#4C72B0"), ("test", "#DD8452")]:
        d = df[df.split == s]
        a.scatter(d.w, d.h, s=4, alpha=.4, c=c, label=s)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("w"); a.set_ylabel("h"); a.legend(); a.set_title("w vs h")
    plt.tight_layout(); plt.savefig(OUT / "dist_train_vs_test.png", dpi=110); plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for i, col in enumerate(["h", "ratio", "bpp"]):
        data = [np.log10(tr[tr.label == l][col]) for l in LABELS] + [np.log10(te[col])]
        ax[i].boxplot(data, tick_labels=LABELS + ["TEST"], showfliers=False)
        ax[i].set_title(f"log10({col}) per class vs test"); ax[i].tick_params(axis="x", rotation=45)
    plt.tight_layout(); plt.savefig(OUT / "per_class_box.png", dpi=110); plt.close()
    print("figures ->", OUT)
