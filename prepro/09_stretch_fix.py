"""09 - Fix for eda/27-28: v2's contrast stretch erases grey text when a dark bar / fill is present.
Mechanism: Otsu splits BAR vs everything else -> text lands on the 'bg' side -> stretched to light grey -> crop_to_ink
then crops to the bar. Label-free metrics on the image BEFORE crop (crop is geometry only):
  survival = share of raw stroke pixels (>40 darker than the local background = smoothed 31px max filter, outside fill) that
             are <128 after preprocess                      (text kept: higher = better)
  junk     = share of NON-stroke pixels that are <128 after preprocess (bars/fill kept as ink: lower = better)
  erased   = image with survival < 0.3 (text effectively gone)
Candidates:
  v2       : remove_black_fill -> polarity -> stretch(otsu on all pixels)
  r2       : + remove_dark_bars (round-2 pipeline, used in v3)
  v6       : r2 + stretch statistics from CONTENT pixels only (rows/cols that are not flat fill), + Otsu recomputed there
Run on all test lines, 600 train lines, and the 50 least confident test lines of B (eda/26).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eda"))
from common import ROOT, out_dir, pmap  # noqa: E402
sys.path[:0] = [str(ROOT / "prepro"), str(ROOT / "prepro" / "outputs")]
import pipeline as R2  # noqa: E402  (round-2: has remove_dark_bars)
import pipeline_v2_snapshot as V2  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs" / "09_stretch_fix"
OUT.mkdir(parents=True, exist_ok=True)


def _stretch(g, mask=None):
    """v2 stretch; statistics (otsu, ink p10, bg median) taken on mask pixels only when mask is given."""
    src = g if mask is None or mask.sum() < 50 else g[mask]
    t = V2._otsu(src.astype(np.uint8))
    if (src > t).mean() < 0.5:
        g, src = 255 - g, 255 - src
        t = V2._otsu(src.astype(np.uint8))
    ink, bg = src[src <= t], src[src > t]
    if ink.size and bg.size:
        lo, hi = np.percentile(ink, 10), np.median(bg)
        if hi - lo >= 20:
            g = np.clip((g.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    return g


def fill_mask(g):
    """pixels that belong to fill, not content: dark flat rows/cols (bars, incl. salt noise via median),
    edge-replicate streak rows (row == row above), pure-black fill already whitened (255 flat)."""
    m = ndi.median_filter(g, 3).astype(np.float32)
    h, w = g.shape
    row_dark = (m < 90).mean(1) > .85
    col_dark = (m < 90).mean(0) > .85
    rep_row = np.r_[False, np.abs(np.diff(m, axis=0)).mean(1) < 1.0]
    fill = np.zeros_like(g, bool)
    fill[row_dark | rep_row] = True
    fill[:, col_dark] = True
    return fill


def pre_v2(g):
    return _stretch(V2.remove_black_fill(g))


def pre_r2(g):
    return _stretch(R2.remove_black_fill(R2.remove_dark_bars(g)))


def pre_v6(g):
    g = R2.remove_black_fill(R2.remove_dark_bars(g))
    return _stretch(g, ~fill_mask(g))


CANDS = {"v2": pre_v2, "r2": pre_r2, "v6": pre_v6}


def strokes(g):
    """stroke = clearly darker than the LOCAL BACKGROUND (grey closing = max filter, then smoothed).
    First try used a 15px local median: inside thick strokes the median is dark itself -> train survival .04 (broken metric)."""
    gf = g.astype(np.float32)
    bg = ndi.uniform_filter(ndi.maximum_filter(gf, 31), 15)
    return (gf < bg - 40) & ~fill_mask(g)


def job(path):
    g = np.asarray(V2.load_rgb(path).convert("L"))
    # polarity-agnostic stroke detection: strokes are the minority side of the local contrast
    # polarity: background = majority tone of the non-fill pixels (a 'minority side' rule failed on bold text: train survival .17)
    nf = ~fill_mask(g)
    st = strokes(g) if np.median(g[nf] if nf.any() else g) > 127 else strokes(255 - g)
    out = {}
    for k, f in CANDS.items():
        p = f(g) < 128
        out[f"surv_{k}"] = float(p[st].mean()) if st.sum() > 20 else np.nan
        out[f"junk_{k}"] = float(p[~st].mean())
    return out


if __name__ == "__main__":
    u = pd.read_csv(out_dir("26_uncertainty_map_B") / "uncertainty.csv")
    tl = u[u.ctype == "line"]
    tr = pd.read_csv(out_dir("08_v2_gap_decomposition") / "train_types_oof.csv"); tr = tr[tr.ctype == "line"].sample(600, random_state=0)
    sets = {"test_line": [str(ROOT / "data/images/test" / i) for i in tl.image_id],
            "test_lowconf": [str(ROOT / "data/images/test" / i) for i in tl.sort_values("conf").head(50).image_id],
            "train_line": [str(ROOT / "data/images/train" / i) for i in tr.image_id]}
    tall = pd.read_csv(out_dir("08_v2_gap_decomposition") / "train_types_oof.csv")
    sets["test_block_glyph"] = [str(ROOT / "data/images/test" / i) for i in u[u.ctype.isin(["block", "glyph"])].image_id]
    sets["train_block_glyph"] = [str(ROOT / "data/images/train" / i) for i in tall[tall.ctype.isin(["block", "glyph"])].sample(400, random_state=0).image_id]
    rows = []
    for name, paths in sets.items():
        D = pd.DataFrame(pmap(job, paths))
        for k in CANDS:
            rows.append(dict(set=name, cand=k, survival=D[f"surv_{k}"].median(), survival_mean=D[f"surv_{k}"].mean(), erased=(D[f"surv_{k}"] < .3).mean(),
                             junk=D[f"junk_{k}"].median()))
        D.to_csv(OUT / f"metrics_{name}.csv", index=False)
    r = pd.DataFrame(rows)
    print(r.round(3).to_string(index=False))

    low = tl.sort_values("conf").head(12)
    fig, axes = plt.subplots(12, 4, figsize=(40, 22))
    for i, x in enumerate(low.itertuples()):
        p = ROOT / "data/images/test" / x.image_id
        g = np.asarray(V2.load_rgb(p).convert("L"))
        axes[i, 0].imshow(V2.load_rgb(p), aspect="auto"); axes[i, 0].set_title(f"raw {x.image_id}", fontsize=12)
        for j, (k, f) in enumerate(CANDS.items()):
            axes[i, j + 1].imshow(f(g), cmap="gray", vmin=0, vmax=255, aspect="auto"); axes[i, j + 1].set_title(k, fontsize=12)
        for a in axes[i]: a.axis("off")
    plt.tight_layout(); plt.savefig(OUT / "lowconf_candidates.png", dpi=35); plt.close()
    print("->", OUT)
