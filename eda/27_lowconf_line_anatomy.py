"""27 - Anatomy of B's low-confidence test LINES (eda/26: ~40 expected errors, as many as pages).
For each of the 40 least confident lines: raw (upscaled), v2 canvas (what B saw), plus measurements:
  content band = rows that hold real ink after removing bars/streaks, its height in px and share of the image,
  bar share (dark near-uniform rows), streak share (rows that are copies of the row above = edge-replicate fill).
Compare with the same numbers on train lines and on confident test lines.
Theory: if the text band is ~5-15 px inside a 60 px image, v2 resizes the WHOLE image to 160 high, so the band gets
13-40 px and the rest of the canvas is fill -> fix = detect the band and give it the canvas height.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ROOT, open_rgb, out_dir, pmap

sys.path.insert(0, str(ROOT / "prepro" / "outputs"))
import pipeline_v2_snapshot as V2  # noqa: E402

OUT = out_dir("27_lowconf_line_anatomy")


def rowstats(path):
    g = np.asarray(open_rgb(path).convert("L"), dtype=np.float32)
    h, w = g.shape
    rstd = g.std(1); rmean = g.mean(1)
    bar = (rmean < 70) & (rstd < 25)                               # dark flat row
    rep = np.r_[False, np.abs(np.diff(g, axis=0)).mean(1) < 1.0]  # row == row above (edge replicate / flat fill)
    flat = rstd < 6                                               # blank row
    ink = ~(bar | rep | flat)
    # longest run of 'ink' rows = content band
    best = cur = 0
    for v in ink:
        cur = cur + 1 if v else 0; best = max(best, cur)
    pp = V2.preprocess(V2.load_rgb(path))
    return dict(h=h, w=w, band_px=best, band_frac=best / h, bar_frac=bar.mean(), rep_frac=rep.mean(), pp_h=pp.height, pp_w=pp.width,
                pp_aspect=pp.width / pp.height)


if __name__ == "__main__":
    u = pd.read_csv(out_dir("26_uncertainty_map_B") / "uncertainty.csv")
    tl = u[u.ctype == "line"].sort_values("conf")
    low, hi = tl.head(40), tl[tl.conf > .9].sample(120, random_state=0)
    tr = pd.read_csv(out_dir("08_v2_gap_decomposition") / "train_types_oof.csv")
    trl = tr[tr.ctype == "line"].sample(200, random_state=0)
    paths = [str(ROOT / "data/images/test" / i) for i in pd.concat([low, hi]).image_id] + [str(ROOT / "data/images/train" / i) for i in trl.image_id]
    S = pd.DataFrame(pmap(rowstats, paths))
    S["set"] = ["test_lowconf"] * len(low) + ["test_conf>0.9"] * len(hi) + ["train_line"] * len(trl)
    print("== medians")
    print(S.groupby("set")[["h", "w", "band_px", "band_frac", "bar_frac", "rep_frac", "pp_h", "pp_aspect"]].median().round(3).to_string())
    print("\n== share with band_frac<0.5 | bar_frac>0.2 | rep_frac>0.2 | pp_aspect>12")
    print(S.groupby("set").apply(lambda x: pd.Series({"band<.5": (x.band_frac < .5).mean(), "bar>.2": (x.bar_frac > .2).mean(), "rep>.2": (x.rep_frac > .2).mean(),
                                                      "pp_aspect>12": (x.pp_aspect > 12).mean()})).round(3).to_string())
    fig, axes = plt.subplots(20, 2, figsize=(26, 44))
    for k, r in enumerate(low.head(20).itertuples()):
        p = ROOT / "data/images/test" / r.image_id
        axes[k, 0].imshow(open_rgb(p), aspect="auto"); axes[k, 0].set_title(f"RAW {r.image_id} {S.h[k]}x{S.w[k]} band={S.band_px[k]}px B={r.B}({r.conf:.2f})", fontsize=11)
        axes[k, 1].imshow(V2.eval_views(str(p))[0], cmap="gray", vmin=0, vmax=255, aspect="auto"); axes[k, 1].set_title("v2 canvas 160x640 (tile 0)", fontsize=11)
        for a in axes[k]: a.axis("off")
    plt.tight_layout(); plt.savefig(OUT / "lowconf_lines_raw_vs_canvas.png", dpi=40); plt.close()
    S.to_csv(OUT / "rowstats.csv", index=False)
    print("->", OUT)
