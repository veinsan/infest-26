"""03 - Calibrate corrupt() against test on the FINAL model input.
Features per 160x640 canvas: ink level, sharpness, mid-tone share, ink share, speckle, occluder-ish gray share.
  * per-feature quantiles + KS distance: clean train vs test, corrupted train vs test
  * domain classifier AUC (train-view vs test-view, HGB on those features, 5-fold):
      0.5 = model input distribution indistinguishable from test.
Run with a strength argument to compare settings:  python 03_calibrate.py [name]"""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eda"))
from common import load_frames, pmap  # noqa: E402
import pipeline as P  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs" / "03_calibrate"
OUT.mkdir(parents=True, exist_ok=True)


def feats(v):
    a = np.asarray(v, dtype=np.float32)
    ink = a < 128
    lap = np.abs(ndi.laplace(a))
    edge = ndi.binary_dilation(ink ^ ndi.binary_erosion(ink))
    hp = np.abs(a - ndi.median_filter(a, 3))
    bgm = ndi.binary_erosion(~ink, iterations=2)
    return dict(ink_p5=np.percentile(a, 5), ink_med=float(np.median(a[ink])) if ink.any() else 255.0,
                sharp=float(lap[edge].mean()) if edge.any() else 0.0, midtone=float(((a > 40) & (a < 215)).mean()),
                ink_frac=float(ink.mean()), speckle=float(hp[bgm].mean()) if bgm.sum() > 50 else 0.0,
                gray_flat=float(((a > 70) & (a < 190) & (ndi.uniform_filter(a, 7) - a < 3) & (a - ndi.uniform_filter(a, 7) < 3)).mean()))


def view(args):
    path, seed, mode = args
    rng = random.Random(seed)
    img = P.load_rgb(path)
    if mode == "corrupt":
        img = P.corrupt(img, rng)
    pp = P.preprocess(img)
    return dict(feats(P.to_canvas(pp, train=mode == "corrupt", rng=rng)[0]), line=pp.width / pp.height >= 2)


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "current"
    df = load_frames()
    tr = df[df.split == "train"].sample(1500, random_state=0)
    te = df[df.split == "test"]
    test_f = pd.DataFrame(pmap(view, [(p, 0, "clean") for p in te.path]))
    clean_f = pd.DataFrame(pmap(view, [(p, i, "clean") for i, p in enumerate(tr.path)]))
    cor_f = pd.DataFrame(pmap(view, [(p, i, "corrupt") for i, p in enumerate(tr.path)]))
    cols = [c for c in test_f.columns if c != "line"]
    rows = []
    for c in cols:
        rows.append(dict(feature=c, test_med=test_f[c].median(), clean_med=clean_f[c].median(), corrupt_med=cor_f[c].median(),
                         test_p90=test_f[c].quantile(.9), clean_p90=clean_f[c].quantile(.9), corrupt_p90=cor_f[c].quantile(.9),
                         KS_clean=ks_2samp(clean_f[c], test_f[c]).statistic, KS_corrupt=ks_2samp(cor_f[c], test_f[c]).statistic))
    t = pd.DataFrame(rows).round(3)
    pd.set_option("display.width", 220)
    print(f"== [{name}] canvas feature distributions (KS: 0 = identical to test)")
    print(t.to_string(index=False))

    def auc(a, only_lines=False):
        b = test_f[test_f.line] if only_lines else test_f
        a = a[a.line] if only_lines else a
        X = pd.concat([a, b])[cols]; y = np.r_[np.zeros(len(a)), np.ones(len(b))]
        p = cross_val_predict(HistGradientBoostingClassifier(max_iter=200, random_state=0), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
        return roc_auc_score(y, p)
    a_clean, a_cor = auc(clean_f), auc(cor_f)
    print(f"\n== [{name}] domain AUC on canvas features: clean-train vs test = {a_clean:.3f} | corrupted-train vs test = {a_cor:.3f}")
    print(f"   line crops only (posters/pages excluded): clean = {auc(clean_f, True):.3f} | corrupted = {auc(cor_f, True):.3f}"
          f"   | non-line share: train {(~clean_f.line).mean():.1%} test {(~test_f.line).mean():.1%}")
    t.assign(setting=name, auc_clean=a_clean, auc_corrupt=a_cor).to_csv(OUT / f"calib_{name}.csv", index=False)
