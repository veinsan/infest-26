"""02 - Look at the pixels. Grids per class (train), test sliced by the shift axes found in 01
(bpp = bits/pixel, height, aspect ratio) so we see WHAT the out-of-distribution test images are."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import LABELS, load_frames, open_rgb, out_dir

OUT = out_dir("02_visual_samples")
META = pd.read_csv(out_dir("01_profile") / "meta.csv")


def grid(rows, title, fname, ncol=4, seed=0):
    rows = rows.sample(min(len(rows), ncol * 5), random_state=seed) if len(rows) > ncol * 5 else rows
    n = len(rows)
    if n == 0:
        return
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 4.5, nrow * 2.2), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, r in zip(axes.ravel(), rows.itertuples()):
        ax.imshow(open_rgb(r.path))
        lab = r.label if isinstance(r.label, str) else "?"
        ax.set_title(f"{lab} {r.w}x{r.h} {r.fmt} bpp={r.bpp:.1f}", fontsize=8)
    fig.suptitle(title); plt.tight_layout(); plt.savefig(OUT / fname, dpi=90); plt.close()


if __name__ == "__main__":
    df = load_frames().merge(META[["image_id", "split", "fmt", "w", "h", "ratio", "bpp"]], on=["image_id", "split"])
    tr, te = df[df.split == "train"], df[df.split == "test"]
    for lab in LABELS:
        grid(tr[tr.label == lab], f"train / {lab} random", f"train_{lab}.png")
        grid(tr[(tr.label == lab) & (tr.h > 150)], f"train / {lab} h>150", f"train_{lab}_tall.png", seed=1)
    grid(te, "test random", "test_random.png")
    slices = {
        "test_bpp_low": te[te.bpp < 0.4], "test_bpp_high": te[te.bpp > 4], "test_tall": te[te.h > 300],
        "test_square": te[(te.ratio > 0.8) & (te.ratio < 1.3)], "test_jpeg": te[te.fmt == "JPEG"],
        "train_bpp_high": tr[tr.bpp > 4], "train_jpeg": tr[tr.fmt == "JPEG"], "train_square": tr[(tr.ratio > 0.8) & (tr.ratio < 1.3)],
    }
    for k, v in slices.items():
        print(f"{k:16s} n={len(v):4d}  ({len(v) / (len(te) if k.startswith('test') else len(tr)) * 100:.1f}% of split)")
        grid(v, k, f"{k}.png")
    print("\nlabel mix of the train slices that resemble test:")
    for k in ["train_bpp_high", "train_jpeg", "train_square"]:
        print(k, slices[k].label.value_counts().to_dict())
    print("figures ->", OUT)
