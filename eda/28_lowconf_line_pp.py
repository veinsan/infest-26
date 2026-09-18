"""28 - Follow-up of eda/27: is the text lost in PREPROCESS (contrast stretch / fill removal) or only diluted by TILING?
For the 20 least confident test lines: raw | full v2 preprocessed image | ink-share of every 640-px tile.
Numbers over all test lines vs train lines: #tiles, share of tiles with <1% 'text-like' ink, text survival after preprocess
(ink pixels in pp that come from dark, non-flat regions of the raw).
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from common import ROOT, open_rgb, out_dir, pmap

sys.path.insert(0, str(ROOT / "prepro" / "outputs"))
import pipeline_v2_snapshot as V2  # noqa: E402

OUT = out_dir("28_lowconf_line_pp")


def tile_ink(path):
    pp = V2.preprocess(V2.load_rgb(path))
    tiles = V2.to_canvas(pp, train=False)
    a = [np.asarray(t) for t in tiles]
    ink = [float((t < 128).mean()) for t in a]
    # 'text-like' ink: dark pixels NOT in full-width dark rows (bars) - bars span the whole tile
    txt = []
    for t in a:
        d = t < 128; rows = d.mean(1) > .9
        txt.append(float(d[~rows].mean()) if (~rows).any() else 0.0)
    return dict(n_tiles=len(a), ink_min=min(ink), ink_max=max(ink), txt_min=min(txt), txt_med=float(np.median(txt)), pp_mean=float(np.asarray(pp).mean()))


if __name__ == "__main__":
    u = pd.read_csv(out_dir("26_uncertainty_map_B") / "uncertainty.csv")
    tl = u[u.ctype == "line"].copy()
    tl["path"] = [str(ROOT / "data/images/test" / i) for i in tl.image_id]
    tr = pd.read_csv(out_dir("08_v2_gap_decomposition") / "train_types_oof.csv"); tr = tr[tr.ctype == "line"].sample(400, random_state=0)
    T = pd.DataFrame(pmap(tile_ink, list(tl.path))); T.index = tl.index
    R = pd.DataFrame(pmap(tile_ink, [str(ROOT / "data/images/train" / i) for i in tr.image_id]))
    tl = pd.concat([tl, T], axis=1)
    tl["grp"] = pd.cut(tl.conf, [0, .6, .9, 1.01], labels=["conf<.6", ".6-.9", ">.9"])
    tl["has_emptyish_tile"] = tl.txt_min < .01
    print("== test lines by confidence group vs train lines")
    agg = lambda x: pd.Series({"n": len(x), "tiles_med": x.n_tiles.median(), "multi_tile": (x.n_tiles > 1).mean(), "tile_txt<1%": (x.txt_min < .01).mean(),
                               "txt_med": x.txt_med.median(), "pp_mean": x.pp_mean.median()})
    print(pd.concat([tl.groupby("grp", observed=True).apply(agg), agg(R).to_frame("train").T]).round(3).to_string())
    print("\n== P(conf<.6) given a near-empty tile vs not (test lines):", tl.groupby("has_emptyish_tile").conf.apply(lambda s: round((s < .6).mean(), 3)).to_dict())
    low = tl.sort_values("conf").head(16)
    fig, axes = plt.subplots(16, 2, figsize=(30, 30), gridspec_kw={"width_ratios": [1, 2]})
    for k, r in enumerate(low.itertuples()):
        axes[k, 0].imshow(open_rgb(r.path), aspect="auto"); axes[k, 0].set_title(f"RAW {r.image_id} B={r.B}({r.conf:.2f})", fontsize=11)
        pp = V2.preprocess(V2.load_rgb(r.path)); pp = pp.resize((int(pp.width * 160 / pp.height), 160))
        axes[k, 1].imshow(pp, cmap="gray", vmin=0, vmax=255, aspect="auto"); axes[k, 1].set_title(f"v2 preprocess (full, {r.n_tiles} tiles, text-ink per tile min {r.txt_min:.3f})", fontsize=11)
        for a in axes[k]: a.axis("off")
    plt.tight_layout(); plt.savefig(OUT / "lowconf_raw_vs_full_pp.png", dpi=40); plt.close()
    tl.drop(columns="path").to_csv(OUT / "test_lines_tiles.csv", index=False)
    print("->", OUT)
