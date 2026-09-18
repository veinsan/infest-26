"""10 - Is the train corruption calibrated on the text-survival axis (prepro/09 metric)?
test lines survival .77 / erased 3.7% (v6) vs clean train .96 / 1%. The model trains on corrupt(train): it should look like test.
Also: share of corrupted-train lines where v2 erased the text but v6 keeps it (= label noise v2 was training on).
"""
import importlib.util
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eda"))
from common import ROOT, out_dir, pmap  # noqa: E402

spec = importlib.util.spec_from_file_location("m09", Path(__file__).with_name("09_stretch_fix.py")); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)


def job(args):
    path, seed = args
    img = M.V2.corrupt(M.V2.load_rgb(path), random.Random(seed))
    g = np.asarray(img.convert("L"))
    nf = ~M.fill_mask(g)
    st = M.strokes(g) if np.median(g[nf] if nf.any() else g) > 127 else M.strokes(255 - g)
    out = {}
    for k, f in M.CANDS.items():
        p = f(g) < 128
        out[k] = float(p[st].mean()) if st.sum() > 20 else np.nan
    return out


if __name__ == "__main__":
    tr = pd.read_csv(out_dir("08_v2_gap_decomposition") / "train_types_oof.csv"); tr = tr[tr.ctype == "line"].sample(800, random_state=1)
    D = pd.DataFrame(pmap(job, [(str(ROOT / "data/images/train" / i), k) for k, i in enumerate(tr.image_id)]))
    print("== corrupted TRAIN lines (what the model trains on)")
    for k in M.CANDS:
        print(f"  {k}: survival median {D[k].median():.3f} mean {D[k].mean():.3f} erased {(D[k] < .3).mean():.3f}")
    print(f"  erased by v2 but kept by v6: {((D.v2 < .3) & (D.v6 >= .3)).mean():.3f} | kept by v2 but erased by v6: {((D.v2 >= .3) & (D.v6 < .3)).mean():.3f}")
    print("  reference test lines (prepro/09): v2 .772 / erased .046 | v6 .776 / erased .037")
