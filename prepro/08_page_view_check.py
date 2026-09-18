"""08 - Visual + numeric check of the round-4 page view (prepro/page_view.py) before it goes into a notebook.
  * per class: raw | v2 canvas (what v2 saw) | page_eval S=640 | 3x page_train
  * gap check (lesson from ColorJitter in prepro/02): grey-level stats of train-aug canvases vs TEST page canvases
    and a quick domain classifier on those stats (clean train vs test, aug train vs test). Aug must not open a gap.
  * pixel budget: fraction of the canvas covered by the page and px per page-height, v2 vs square view
"""
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eda"))
from common import LABELS, ROOT, load_frames, open_rgb, out_dir, pmap  # noqa: E402
sys.path[:0] = [str(ROOT / "prepro"), str(ROOT / "prepro" / "outputs")]
import page_view as PV  # noqa: E402
import pipeline_v2_snapshot as V2  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs" / "08_page_view_check"
OUT.mkdir(parents=True, exist_ok=True)
S = 640


def gstats(a):
    a = np.asarray(a, dtype=np.float32)
    content = a[a < 254] if (a < 254).any() else a.ravel()
    return dict(mean=a.mean(), std=a.std(), bg=np.median(content), p5=np.percentile(content, 5), dark=(a < 100).mean(),
                edge=np.abs(np.diff(a, axis=1)).mean(), pad=(a >= 254).mean())


def job(args):
    path, aug, seed = args
    rgb = open_rgb(path)
    v = PV.page_train(rgb, S, random.Random(seed)) if aug else PV.page_eval(rgb, S)
    return gstats(v)


if __name__ == "__main__":
    df = load_frames()
    ct = pd.concat([pd.read_csv(out_dir("08_v2_gap_decomposition") / f)[["image_id", "split", "ctype"]] for f in ["train_types_oof.csv", "test_types_pred.csv"]])
    df = df.merge(ct, on=["image_id", "split"])
    pg = df[df.ctype == "page"].reset_index(drop=True)

    fig, axes = plt.subplots(len(LABELS) * 2, 6, figsize=(30, 5.2 * len(LABELS)))
    for r, lab in enumerate(LABELS):
        for k, row in enumerate(pg[(pg.split == "train") & (pg.label == lab)].sample(2, random_state=r).itertuples()):
            ax = axes[2 * r + k]; rgb = open_rgb(row.path)
            ax[0].imshow(rgb); ax[0].set_title(f"{lab} raw {row.image_id} {rgb.size}", fontsize=9)
            ax[1].imshow(V2.eval_views(row.path)[0], cmap="gray", vmin=0, vmax=255); ax[1].set_title("v2 canvas 160x640", fontsize=9)
            ax[2].imshow(PV.page_eval(rgb, S), cmap="gray", vmin=0, vmax=255); ax[2].set_title(f"page_eval {S}", fontsize=9)
            for c in range(3):
                ax[3 + c].imshow(PV.page_train(rgb, S, random.Random(c)), cmap="gray", vmin=0, vmax=255); ax[3 + c].set_title(f"page_train seed {c}", fontsize=9)
            for a in ax: a.axis("off")
    plt.tight_layout(); plt.savefig(OUT / "page_views_per_class.png", dpi=45); plt.close()

    tr, te = pg[pg.split == "train"], pg[pg.split == "test"]
    A = pd.DataFrame(pmap(job, [(p, False, 0) for p in tr.path])).assign(set="train_clean")
    # ONE aug copy per train page: 3 copies + non-grouped CV let the classifier match copies across folds (AUC .67, a leak not a gap)
    B = pd.DataFrame(pmap(job, [(p, True, i) for i, p in enumerate(tr.path)])).assign(set="train_aug")
    T = pd.DataFrame(pmap(job, [(p, False, 0) for p in te.path])).assign(set="test")
    allst = pd.concat([A, B, T])
    print("== canvas grey stats (median), train clean / train aug / test pages")
    print(allst.groupby("set").median().round(3).to_string())
    cols = [c for c in A.columns if c != "set"]
    for name, X in [("train_clean", A), ("train_aug", B)]:
        Z = pd.concat([X, T]); yy = (Z.set == "test").to_numpy()
        auc = np.mean([cross_val_score(GradientBoostingClassifier(random_state=s_), Z[cols], yy, cv=StratifiedKFold(5, shuffle=True, random_state=s_), scoring="roc_auc").mean() for s_ in range(5)])
        print(f"   domain AUC {name} vs test pages: {auc:.3f}   (0.5 = indistinguishable)")
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 3))
    for a, c in zip(axes, cols):
        for s_ in ["train_clean", "train_aug", "test"]:
            a.hist(allst[allst.set == s_][c], bins=30, alpha=.45, density=True, label=s_)
        a.set_title(c)
    axes[0].legend(); plt.tight_layout(); plt.savefig(OUT / "canvas_stats.png", dpi=60); plt.close()

    sz = np.array([open_rgb(p).size for p in te.path])
    s_v2 = np.minimum(160 / sz[:, 1], 640 / sz[:, 0]); s_sq = S / sz.max(1)
    print(f"\n== test pages pixel budget: page height in canvas px median v2 {np.median(sz[:, 1] * s_v2):.0f} -> square {np.median(sz[:, 1] * s_sq):.0f}"
          f" | canvas covered v2 {np.median(sz.prod(1) * s_v2 ** 2 / (160 * 640)):.2f} -> square {np.median(sz.prod(1) * s_sq ** 2 / S ** 2):.2f}"
          f" | page area in px x{np.median((s_sq / s_v2) ** 2):.1f}")
    print("->", OUT)
