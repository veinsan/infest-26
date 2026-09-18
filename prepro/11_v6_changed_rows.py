"""11 - Which test rows does preprocess v6 actually change vs v2 (full preprocess incl. crop)? Those are the only rows a
v6 model can legitimately change for the preprocessing reason -> a 'targeted hybrid' submission uses the v6 model there and
B everywhere else (isolates the factor, keeps seed noise out). Also: how B's confidence looks on them, grid for visual check.
"""
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eda"))
from common import ROOT, out_dir, pmap  # noqa: E402

spec = importlib.util.spec_from_file_location("m09", Path(__file__).with_name("09_stretch_fix.py")); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
OUT = Path(__file__).resolve().parent / "outputs" / "11_v6_changed_rows"; OUT.mkdir(parents=True, exist_ok=True)


def full(g, f):
    return M.V2.crop_to_ink(f(g))


def job(path):
    g = np.asarray(M.V2.load_rgb(path).convert("L"))
    a, b = full(g, M.pre_v2), full(g, M.pre_v6)
    if a.shape != b.shape:
        return dict(changed=True, diff=1.0, shape_change=True)
    d = float((np.abs(a.astype(int) - b.astype(int)) > 40).mean())
    return dict(changed=d > .02, diff=d, shape_change=False)


if __name__ == "__main__":
    u = pd.read_csv(out_dir("26_uncertainty_map_B") / "uncertainty.csv")
    tr = pd.read_csv(out_dir("08_v2_gap_decomposition") / "train_types_oof.csv")
    D = pd.DataFrame(pmap(job, [str(ROOT / "data/images/test" / i) for i in u.image_id])); u = pd.concat([u, D], axis=1)
    np_ = u.ctype != "page"
    print(f"== test non-page rows whose v6 preprocess differs from v2: {u[np_].changed.sum()} / {np_.sum()} (shape change {u[np_].shape_change.sum()})")
    print("   by type:", u[np_ & u.changed].ctype.value_counts().to_dict())
    print(f"   B conf on changed rows: median {u[np_ & u.changed].conf.median():.2f} | unchanged {u[np_ & ~u.changed].conf.median():.2f}")
    print(f"   share of changed rows with conf<.6: {(u[np_ & u.changed].conf < .6).mean():.2f} (unchanged {(u[np_ & ~u.changed].conf < .6).mean():.2f})")
    T = pd.DataFrame(pmap(job, [str(ROOT / "data/images/train" / i) for i in tr[tr.ctype != "page"].image_id]))
    print(f"   clean train non-page changed: {T.changed.mean():.3%}")
    ch = u[np_ & u.changed].sort_values("conf").head(24)
    fig, axes = plt.subplots(24, 3, figsize=(36, 40))
    for k, r in enumerate(ch.itertuples()):
        p = ROOT / "data/images/test" / r.image_id; g = np.asarray(M.V2.load_rgb(p).convert("L"))
        axes[k, 0].imshow(M.V2.load_rgb(p), aspect="auto"); axes[k, 0].set_title(f"raw {r.image_id} {r.ctype} B={r.B}({r.conf:.2f})", fontsize=12)
        axes[k, 1].imshow(full(g, M.pre_v2), cmap="gray", vmin=0, vmax=255, aspect="auto"); axes[k, 1].set_title("v2 preprocess", fontsize=12)
        axes[k, 2].imshow(full(g, M.pre_v6), cmap="gray", vmin=0, vmax=255, aspect="auto"); axes[k, 2].set_title("v6 preprocess", fontsize=12)
        for a in axes[k]: a.axis("off")
    plt.tight_layout(); plt.savefig(OUT / "changed_rows_v2_vs_v6.png", dpi=30); plt.close()
    u.to_csv(OUT / "test_v6_changed.csv", index=False)
    print("->", OUT)
