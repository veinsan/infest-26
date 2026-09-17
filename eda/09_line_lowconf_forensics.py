"""09 - Test LINE crops: v2 is unsure on 7% of them vs 0.7% in OOF (10x). What do they have that corrupt() lacks?
Per image (raw, before preprocess) detectors:
  content_h_frac  vertical extent of ink / image height   (content shifted out of frame?)
  bar_rows        share of rows that are near-uniform dark (black bar, incl. noisy bars)
  bar_noisy       dark bar rows with high pixel noise (bar + noise -> not removed by remove_black_fill)
  streak_frac     columns constant across >=30% height near top/bottom (edge-replicate streaks)
  ink_frac_pp     ink share after preprocess (how much text survives)
Compared for: test lines conf<0.6, test lines conf>=0.9, corrupt(train lines) 1 draw.
"""
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ROOT, open_rgb, out_dir, pmap

sys.path.insert(0, str(ROOT / "prepro"))
import pipeline as P  # noqa: E402

OUT = out_dir("09_line_lowconf_forensics")


def detect(img):
    g = np.asarray(img.convert("L"), dtype=np.float32)
    h, w = g.shape
    t = P._otsu(g.astype(np.uint8))
    ink = g <= t
    rows = np.where(ink.mean(1) > 0.01)[0]
    row_mean, row_std = g.mean(1), g.std(1)
    bar = (row_mean < 60) & (ink.mean(1) > 0.9)
    noisy_bar = bar & (row_std > 12)
    # edge-replicate streak: column pixels identical over a long vertical run touching top or bottom
    dv = np.abs(np.diff(g, axis=0)) < 1.5
    run_top = np.argmin(np.vstack([dv, np.zeros((1, w), bool)]), axis=0)
    run_bot = np.argmin(np.vstack([dv[::-1], np.zeros((1, w), bool)]), axis=0)
    colvar = g.std(0) > 10
    streak = ((np.maximum(run_top, run_bot) >= 0.3 * h) & colvar).mean()
    pp = np.asarray(P.preprocess(img))
    return dict(h=h, w=w, content_h_frac=(rows.max() - rows.min() + 1) / h if len(rows) else 0.0,
                bar_rows=bar.mean(), bar_noisy=noisy_bar.mean(), streak_frac=float(streak),
                ink_frac_pp=float((pp < 128).mean()), pp_aspect=pp.shape[1] / pp.shape[0])


def test_row(path):
    return detect(P.load_rgb(path))


def train_row(args):
    path, seed = args
    return detect(P.corrupt(P.load_rgb(path), random.Random(seed)))


if __name__ == "__main__":
    t = pd.read_csv(out_dir("08_v2_gap_decomposition") / "test_types_pred.csv")
    t["path"] = [str(ROOT / "data/images/test" / i) for i in t.image_id]
    tr = pd.read_csv(out_dir("08_v2_gap_decomposition") / "train_types_oof.csv")
    tr["path"] = [str(ROOT / "data/images/train" / i) for i in tr.image_id]
    tl = t[t.ctype == "line"].reset_index(drop=True)
    tl = pd.concat([tl, pd.DataFrame(pmap(test_row, list(tl.path)))], axis=1)
    trl = tr[tr.ctype == "line"].sample(1200, random_state=0).reset_index(drop=True)
    trl = pd.concat([trl, pd.DataFrame(pmap(train_row, list(zip(trl.path, range(len(trl))))))], axis=1)

    F = ["content_h_frac", "bar_rows", "bar_noisy", "streak_frac", "ink_frac_pp", "pp_aspect"]
    groups = {"test conf<0.6": tl[tl.conf < .6], "test 0.6-0.9": tl[(tl.conf >= .6) & (tl.conf < .9)], "test conf>=0.9": tl[tl.conf >= .9], "corrupt(train)": trl}
    pd.set_option("display.width", 220)
    print("== medians / prevalence")
    rows = []
    for k, d in groups.items():
        rows.append(dict(group=k, n=len(d), content_h_med=d.content_h_frac.median(), content_h_lt50=(d.content_h_frac < .5).mean(),
                         has_bar=(d.bar_rows > .05).mean(), has_noisy_bar=(d.bar_noisy > .03).mean(), has_streak=(d.streak_frac > .05).mean(),
                         ink_pp_med=d.ink_frac_pp.median(), ink_pp_lt3=(d.ink_frac_pp < .03).mean(), pp_aspect_lt2=(d.pp_aspect < 2).mean()))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("\n== low-confidence test lines: predicted class mix", tl[tl.conf < .6].pred.value_counts().to_dict())

    for k in ["test conf<0.6", "corrupt(train)"]:
        d = groups[k].sample(min(20, len(groups[k])), random_state=0)
        fig, axes = plt.subplots(10, 2, figsize=(24, 14))
        for a, r in zip(axes.ravel(), d.itertuples()):
            img = P.load_rgb(r.path) if k.startswith("test") else P.corrupt(P.load_rgb(r.path), random.Random(int(r.Index)))
            a.imshow(img, interpolation="nearest"); a.set_title(f"conf={getattr(r, 'conf', 1):.2f} bar={r.bar_rows:.2f} noisybar={r.bar_noisy:.2f} streak={r.streak_frac:.2f} content_h={r.content_h_frac:.2f}", fontsize=8)
        for a in axes.ravel(): a.axis("off")
        plt.tight_layout(); plt.savefig(OUT / f"{k.split()[0]}_{'lowconf' if 'test' in k else 'corrupt'}.png", dpi=60); plt.close()
    tl.drop(columns="path").to_csv(OUT / "test_lines_forensics.csv", index=False)
    print("->", OUT)
