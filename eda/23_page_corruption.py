"""23 - Were TEST PAGES synthetically corrupted like test lines? (decides page augmentation)
eda/21: page stats train ~= test (JPEG 79% vs 80%, sat 17.7 vs 20.3) while lines differ a lot (eda/03).
Detectors (eda/09 detect + flat grey polygon + black dash) on train/test x page/line, and
full-resolution crops of random test pages to look for triangles / dashes / edge streaks by eye.
If pages are clean -> page aug should be mild (JPEG/scale/crop), not the line corruption recipe.
"""
import importlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from common import load_frames, open_rgb, out_dir, pmap

detect = importlib.import_module("09_line_lowconf_forensics").detect
OUT = out_dir("23_page_corruption")


def feats(path):
    img = open_rgb(path)
    if max(img.size) > 1200:
        img.thumbnail((1200, 1200))
    d = detect(img)
    g = np.asarray(img.convert("L"), dtype=np.float32)
    # flat grey semi-transparent polygon: large region of near-zero local variance at mid-grey
    loc_std = ndi.generic_filter(g[::4, ::4], np.std, size=5)
    flat_mid = (loc_std < 1.0) & (g[::4, ::4] > 70) & (g[::4, ::4] < 190)
    lab, n = ndi.label(flat_mid)
    d["flat_grey"] = float(np.bincount(lab.ravel())[1:].max() / lab.size) if n else 0.0
    # black dash: small solid black blobs, elongated, very dark
    blk = g < 25
    lab, n = ndi.label(blk)
    dash = 0
    for sl in ndi.find_objects(lab)[:500]:
        hh, ww = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if 3 <= min(hh, ww) <= 12 and max(hh, ww) >= 3 * min(hh, ww):
            dash += 1
    d["dashes"] = dash
    lap = np.abs(ndi.laplace(g)); d["sharp"] = float(np.percentile(lap, 99) / (g.std() + 1))
    return d


if __name__ == "__main__":
    df = load_frames()
    ct = pd.concat([pd.read_csv(out_dir("08_v2_gap_decomposition") / f)[["image_id", "split", "ctype"]] for f in ["train_types_oof.csv", "test_types_pred.csv"]])
    df = df.merge(ct, on=["image_id", "split"])
    rng = np.random.default_rng(0)
    parts = [df[(df.ctype == "page")]] + [df[(df.ctype == "line") & (df.split == s)].sample(200, random_state=0) for s in ["train", "test"]]
    d = pd.concat(parts).reset_index(drop=True)
    d = pd.concat([d, pd.DataFrame(pmap(feats, list(d.path)))], axis=1)
    flags = pd.DataFrame({"streak>5%": d.streak_frac > .05, "bar>2%": d.bar_rows > .02, "flat_grey>2%": d.flat_grey > .02,
                          "dashes>=3": d.dashes >= 3, "sharp<p25(train)": d.sharp < d[d.split == "train"].sharp.quantile(.25)})
    print("== % flagged by split x type (test lines are known-corrupted: that is the reference signature)")
    print((flags.groupby([d.ctype, d.split]).mean() * 100).round(1).to_string())
    print("\n== medians"); print(d.groupby(["ctype", "split"])[["streak_frac", "bar_rows", "flat_grey", "dashes", "sharp"]].median().round(3).to_string())
    tp = d[(d.ctype == "page") & (d.split == "test")].sample(12, random_state=3)
    fig, axes = plt.subplots(3, 4, figsize=(24, 15))
    for a, r in zip(axes.ravel(), tp.itertuples()):
        img = open_rgb(r.path); w, h = img.size; s = min(w, h, 400)
        x0, y0 = rng.integers(0, w - s + 1), rng.integers(0, h - s + 1)
        a.imshow(img.crop((x0, y0, x0 + s, y0 + s))); a.set_title(f"{r.image_id} crop {s}px streak={r.streak_frac:.2f} flat={r.flat_grey:.3f}", fontsize=9); a.axis("off")
    plt.tight_layout(); plt.savefig(OUT / "test_page_fullres_crops.png", dpi=55); plt.close()
    d.drop(columns="path").to_csv(OUT / "feats.csv", index=False)
    print("->", OUT)
