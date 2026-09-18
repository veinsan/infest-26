"""14 - Why did train-page OOF say v3 > v2 (0.864 vs 0.744) while the LB says v3 < v2?
  * paired v2 vs v3 OOF on the same 199 train pages: wins/losses per true class (Arabic-script vs others)
  * per-epoch page accuracy spread (how noisy is a 40-page val fold?) and bootstrap CI of the page gain
  * v3 OOF errors on Arabic-script pages: what does v3 predict?
"""
import numpy as np
import pandas as pd

from common import ROOT, out_dir

OUT = out_dir("14_oof_v2_vs_v3_pages")

if __name__ == "__main__":
    o2 = pd.read_csv(ROOT / "results/v2/main/oof_predictions.csv").set_index("image_id")
    o3 = pd.read_csv(ROOT / "results/v3/main/oof_predictions.csv").set_index("image_id")
    d = o3.join(o2[["pred_clean", "pred_corrupt"]].rename(columns={"pred_clean": "v2_clean", "pred_corrupt": "v2_corrupt"}))
    p = d[d.ctype == "page"].copy()
    p["v2_ok"], p["v3_ok"] = p.v2_clean == p.label, p.pred_testlike == p.label   # v3 page val = clean view (test-like = clean for pages)
    print(f"== train pages n={len(p)}: v2 acc {p.v2_ok.mean():.3f} | v3 acc {p.v3_ok.mean():.3f}")
    tab = p.groupby("label").agg(n=("v2_ok", "size"), v2=("v2_ok", "mean"), v3=("v3_ok", "mean"),
                                 v3_wins=("v3_ok", lambda s: int((s & ~p.loc[s.index, "v2_ok"]).sum())),
                                 v2_wins=("v2_ok", lambda s: int((s & ~p.loc[s.index, "v3_ok"]).sum()))).round(3)
    print(tab)
    arab = p.label.isin(["jawi", "pegon"])
    print(f"   Arabic-script pages (jawi+pegon) n={arab.sum()}: v2 {p.v2_ok[arab].mean():.3f} v3 {p.v3_ok[arab].mean():.3f} | other scripts: v2 {p.v2_ok[~arab].mean():.3f} v3 {p.v3_ok[~arab].mean():.3f}")
    rng = np.random.default_rng(0)
    diff = (p.v3_ok.values.astype(int) - p.v2_ok.values.astype(int))
    boots = [rng.choice(diff, len(diff)).mean() for _ in range(5000)]
    print(f"   page gain v3-v2 = {diff.mean():+.3f}, bootstrap 95% CI [{np.percentile(boots, 2.5):+.3f}, {np.percentile(boots, 97.5):+.3f}]")
    print("\n== v3 predictions on Arabic-script train pages it got wrong:", p[arab & ~p.v3_ok].pred_testlike.value_counts().to_dict())
    print("   v2 predictions on Arabic-script train pages it got wrong:", p[arab & ~p.v2_ok].v2_clean.value_counts().to_dict())

    h = pd.concat([pd.read_csv(ROOT / f"results/v3/main/fold{k}_history.csv") for k in range(5)])
    late = h[h.epoch >= 5].groupby("fold").page_acc.agg(["min", "max", "std"])
    print("\n== v3 page_acc spread across epochs>=5 per fold (40-page val):")
    print(late.round(3))
    print(f"   typical within-fold epoch-to-epoch swing: {(late['max'] - late['min']).mean():.3f} -> single-number page val is ~±{late['std'].mean():.3f} noise")
    p.to_csv(OUT / "pages_v2_v3.csv")
    print("->", OUT)

    # recompute the round-2 'composition' claim with the RIGHT view per type (pages clean, others test-like)
    from sklearn.metrics import f1_score
    tr = pd.read_csv(out_dir("08_v2_gap_decomposition") / "train_types_oof.csv")[["image_id", "ctype"]]
    te = pd.read_csv(out_dir("08_v2_gap_decomposition") / "test_types_pred.csv")[["image_id", "ctype"]]
    wmap = (te.ctype.value_counts(normalize=True) / tr.ctype.value_counts(normalize=True)).to_dict()
    e = d.copy()
    e["v2_tl"] = np.where(e.ctype == "page", e.v2_clean, e.v2_corrupt)
    w = e.ctype.map(wmap).values
    print("\n== corrected LB proxy (pages clean, others test-like), reweighted to test type mix")
    for name, col in [("v2", "v2_tl"), ("v3", "pred_testlike")]:
        print(f"   {name}: plain {f1_score(e.label, e[col], average='macro'):.4f} | reweighted {f1_score(e.label, e[col], average='macro', sample_weight=w):.4f}")
    print("   public LB: v2 0.91514 | v3 0.89077")
