"""24 - WHERE does the higher-resolution page view help? Per-class confusion of the frozen probe (eda/22 features),
v2 view vs sq640_gray, held-out train pages. Focus: jawi<->pegon (genre) and bali<->jawa (glyph detail).
Also on TEST pages: transitions v2-submission -> sq640 probe, and a grid of the pages where they disagree
(visual sanity check only - these are not labels).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from common import LABELS, ROOT, load_frames, open_rgb, out_dir

OUT = out_dir("24_page_confusion")

if __name__ == "__main__":
    df = load_frames()
    ct = pd.concat([pd.read_csv(out_dir("08_v2_gap_decomposition") / f)[["image_id", "split", "ctype"]] for f in ["train_types_oof.csv", "test_types_pred.csv"]])
    df = df.merge(ct, on=["image_id", "split"])
    d = df[((df.split == "train") & (df.ctype != "line")) | ((df.split == "test") & (df.ctype == "page"))].reset_index(drop=True)
    d["fold"] = pd.read_csv(ROOT / "data/folds.csv").set_index("image_id").fold.reindex(d.image_id).to_numpy()
    F = dict(np.load(out_dir("22_page_view_probe") / "feats_small.npz"))
    y = d.label.map({l: i for i, l in enumerate(LABELS)}).fillna(-1).astype(int).to_numpy()
    is_tr = (d.split == "train").to_numpy(); pg = is_tr & (d.ctype == "page").to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    preds = {}
    for a, k in zip(axes, ["v2", "sq640_gray"]):
        X = F[k] / np.linalg.norm(F[k], axis=1, keepdims=True)
        p = np.full(len(d), -1)
        for fo in range(5):
            trn, val = is_tr & (d.fold != fo).to_numpy(), is_tr & (d.fold == fo).to_numpy()
            p[val] = LogisticRegression(C=4, max_iter=3000, class_weight="balanced").fit(X[trn], y[trn]).predict(X[val])
        preds[k] = p
        cm = confusion_matrix(y[pg], p[pg], labels=range(7))
        a.imshow(cm, cmap="Blues"); a.set_xticks(range(7), LABELS, rotation=45); a.set_yticks(range(7), LABELS)
        for i in range(7):
            for j in range(7):
                a.text(j, i, cm[i, j], ha="center", va="center", fontsize=9)
        a.set_title(f"{k}: held-out train pages acc {np.mean(p[pg] == y[pg]):.3f}"); a.set_xlabel("pred"); a.set_ylabel("true")
        print(f"== {k}: per-class page recall", {LABELS[c]: round(float(np.mean(p[pg][y[pg] == c] == c)), 2) for c in range(7)})
    plt.tight_layout(); plt.savefig(OUT / "page_confusion_v2_vs_sq640.png", dpi=70); plt.close()
    fixed = pg & (preds["sq640_gray"] == y) & (preds["v2"] != y); broke = pg & (preds["sq640_gray"] != y) & (preds["v2"] == y)
    print(f"   sq640 fixes {fixed.sum()} pages, breaks {broke.sum()}")
    print("   fixed (true: v2 pred):", pd.Series([f"{LABELS[y[i]]}:{LABELS[preds['v2'][i]]}" for i in np.where(fixed)[0]]).value_counts().to_dict())

    t = pd.read_csv(out_dir("22_page_view_probe") / "test_page_preds_small.csv")
    t["v2"] = pd.read_csv(ROOT / "results/v2/main/submission.csv").set_index("image_id").label.reindex(t.image_id).values  # submitted labels
    ch = t[t.v2 != t.sq640_gray]
    print(f"\n== test pages: sq640 probe differs from v2 submission on {len(ch)}/{len(t)}")
    print("   transitions v2 -> sq640:", pd.Series([f"{a}->{b}" for a, b in zip(ch.v2, ch.sq640_gray)]).value_counts().head(12).to_dict())
    ch = ch.sample(min(32, len(ch)), random_state=0)
    fig, axes = plt.subplots(4, 8, figsize=(32, 16))
    for a in axes.ravel(): a.axis("off")
    for a, r in zip(axes.ravel(), ch.itertuples()):
        a.imshow(open_rgb(ROOT / "data/images/test" / r.image_id)); a.set_title(f"v2={r.v2} probe640={r.sq640_gray}", fontsize=11)
    plt.tight_layout(); plt.savefig(OUT / "test_disagreements.png", dpi=45); plt.close()
    print("->", OUT)
