"""26 - Where are B's (LB 0.91820) remaining errors likely to be? No labels: use B's probabilities (v2 for non-page,
v5 for page) and agreement between independent runs (v1, v2, v3, v4-A, v5-A) as an error proxy.
eda/25: one fixed row ~ +0.0008; v3's non-shared changes (jawi<->pegon, lontara->jawi on lines) were almost all wrong.
  * per content type x confusion pair (top1/top2): how many low-confidence rows
  * OOF of the same models: expected error count per type at the test mix (calibrated by confidence bins)
  * grids of the low-confidence test rows for the biggest pairs
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import LABELS, ROOT, open_rgb, out_dir

OUT = out_dir("26_uncertainty_map_B")
L = np.array(LABELS)


def sm(x):
    e = np.exp(x - x.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)


if __name__ == "__main__":
    tv = pd.read_csv(ROOT / "analysis/outputs/v5_submission_check/test_versions.csv")
    test = pd.read_csv(ROOT / "data/test.csv"); assert (tv.image_id == test.image_id).all()
    pg = (tv.ctype == "page").to_numpy()
    p2 = np.mean([sm(np.load(ROOT / f"results/v2/main/fold{k}_test_logits.npy")) for k in range(5)], 0)
    p5 = np.mean([sm(np.load(ROOT / f"results/v5/main/fold{k}_test_logits.npy")) for k in range(5)], 0)
    pB = np.where(pg[:, None], p5, p2)
    v4 = pd.read_csv(ROOT / "results/v4/submission_A_cv.csv").set_index("image_id").label.reindex(test.image_id).values
    o = np.argsort(-pB, 1); top1, top2 = L[o[:, 0]], L[o[:, 1]]
    conf = pB.max(1)
    votes = np.stack([np.asarray(v, dtype=object) for v in [tv.v1, tv.v2, tv.v3, v4, tv.A]]); agree = (votes == tv.B.to_numpy(dtype=object)[None]).mean(0)
    d = pd.DataFrame({"image_id": tv.image_id, "ctype": tv.ctype, "B": tv.B, "top2": top2, "conf": conf, "agree": agree,
                      "pair": [" / ".join(sorted([a, b])) for a, b in zip(top1, top2)]})

    # OOF calibration: error rate per confidence bin and type (v2 OOF for non-page corrupted, v5 OOF for page clean)
    train = pd.read_csv(ROOT / "data/train.csv"); y = train.label.map({l: i for i, l in enumerate(LABELS)}).to_numpy()
    tt = pd.read_csv(out_dir("08_v2_gap_decomposition") / "train_types_oof.csv").set_index("image_id").ctype.reindex(train.image_id).to_numpy()
    def oof(dr, kind):
        idx = np.concatenate([np.load(dr / f"fold{k}_val_idx.npy") for k in range(5)])
        p = sm(np.concatenate([np.load(dr / f"fold{k}_oof_{kind}_logits.npy") for k in range(5)])); r = np.empty_like(p); r[idx] = p; return r
    o2, o5 = oof(ROOT / "results/v2/main", "corrupt"), oof(ROOT / "results/v5/main", "clean")
    oB = np.where((tt == "page")[:, None], o5, o2)
    bins = [0, .5, .6, .7, .8, .9, 1.01]
    cal = pd.DataFrame({"ctype": tt, "bin": pd.cut(oB.max(1), bins), "err": oB.argmax(1) != y}).groupby(["ctype", "bin"], observed=False).err.mean()
    d["bin"] = pd.cut(d.conf, bins)
    d["p_err_oof"] = [cal.get((c, b), np.nan) for c, b in zip(d.ctype, d.bin)]
    d["p_err_conf"] = 1 - d.conf
    print("== expected #errors in test (two estimates: OOF-calibrated by conf bin | 1-conf)")
    e = d.groupby("ctype").agg(n=("conf", "size"), conf_lt_06=("conf", lambda s: int((s < .6).sum())), err_oof=("p_err_oof", "sum"), err_1mconf=("p_err_conf", "sum"),
                               disagree_any=("agree", lambda s: int((s < 1).sum())))
    print(e.round(1).to_string()); print("   total:", e[["err_oof", "err_1mconf"]].sum().round(1).to_dict())
    print("   (OOF calibration underestimates non-page: OOF is corrupted TRAIN; LB 0.918 implies ~90-100 wrong rows)")

    print("\n== low-confidence (<0.7) rows by type x top-2 pair")
    lc = d[d.conf < .7]
    tab = pd.crosstab(lc.pair, lc.ctype, margins=True).sort_values("All", ascending=False)
    print(tab.head(16).to_string())
    print("\n== rows where the 6 runs disagree (agree<1) by type x pair")
    dg = d[d.agree < 1]
    print(pd.crosstab(dg.pair, dg.ctype, margins=True).sort_values("All", ascending=False).head(12).to_string())

    for t in ["line", "block", "glyph", "page"]:
        x = d[(d.ctype == t)].sort_values("conf").head(30)
        fig, axes = plt.subplots(6, 5, figsize=(30, 16 if t == "line" else 26))
        for a in axes.ravel(): a.axis("off")
        for a, r in zip(axes.ravel(), x.itertuples()):
            a.imshow(open_rgb(ROOT / "data/images/test" / r.image_id)); a.set_title(f"{r.image_id} B={r.B} ({r.conf:.2f}) 2nd={r.top2} agree={r.agree:.2f}", fontsize=12)
        plt.tight_layout(); plt.savefig(OUT / f"lowconf_{t}.png", dpi=40); plt.close()
    d.to_csv(OUT / "uncertainty.csv", index=False)
    print("->", OUT)
