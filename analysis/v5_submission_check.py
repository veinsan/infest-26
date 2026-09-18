"""Is v5 (page view 640x640, rest = v2) a safe first submission of the day? A = v5 all, B = hybrid (non-page v2, page v5).
  1. format + local recompute of A; build B
  2. OOF paired v5 vs v2 per content type (same folds.csv): page clean acc, bootstrap CI over dup groups, fixes/breaks
  3. test: rows A/B change vs v2, per type, transitions; v5 confidence on changed pages
  4. risk: overlap with v3's page changes (v3 LB 0.891) and v1; agreement with the frozen 640 probe (eda/22)
  5. grid of changed pages (visual check only)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
V2, V5 = ROOT / "results/v2/main", ROOT / "results/v5/main"
OUT = ROOT / "analysis/outputs/v5_submission_check"
OUT.mkdir(parents=True, exist_ok=True)
LABELS = ["bali", "jawa", "jawi", "lampung", "lontara", "pegon", "sunda"]
L = np.array(LABELS)


def sm(x):
    e = np.exp(x - x.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)


def ctype_of(path):
    with Image.open(path) as im:
        w, h = im.size
    return "page" if h >= 250 else ("line" if w / h >= 2 else ("block" if w >= 80 else "glyph"))


def oof(d):
    idx = np.concatenate([np.load(d / f"fold{k}_val_idx.npy") for k in range(5)])
    return idx, sm(np.concatenate([np.load(d / f"fold{k}_oof_clean_logits.npy") for k in range(5)])), sm(np.concatenate([np.load(d / f"fold{k}_oof_corrupt_logits.npy") for k in range(5)]))


if __name__ == "__main__":
    test, sample = pd.read_csv(ROOT / "data/test.csv"), pd.read_csv(ROOT / "data/sample_submission.csv")
    train = pd.read_csv(ROOT / "data/train.csv"); y = train.label.map({l: i for i, l in enumerate(LABELS)}).to_numpy()
    grp = pd.read_csv(ROOT / "data/folds.csv").set_index("image_id").dup_group.reindex(train.image_id).to_numpy()
    tct = np.array([ctype_of(ROOT / "data/images/test" / i) for i in test.image_id])
    trct = np.array([ctype_of(ROOT / "data/images/train" / i) for i in train.image_id])
    twins = pd.read_csv(ROOT / "data/test_train_twins.csv").set_index("image_id").train_label
    m_tw = test.image_id.isin(twins.index).to_numpy()

    p5 = np.mean([sm(np.load(V5 / f"fold{k}_test_logits.npy")) for k in range(5)], 0)
    p2 = np.mean([sm(np.load(V2 / f"fold{k}_test_logits.npy")) for k in range(5)], 0)
    pg = tct == "page"
    P = {"A": p5, "B": np.where(pg[:, None], p5, p2)}
    pred = {}
    for k, p in P.items():
        o = L[p.argmax(1)].astype(object); o[m_tw] = twins.reindex(test.image_id[m_tw]).to_numpy(dtype=object); pred[k] = o
    subs = {v: pd.read_csv(ROOT / f"results/{v}{'/main' if v != 'v1' else ''}/submission.csv").set_index("image_id").label.reindex(test.image_id).to_numpy(dtype=object) for v in ["v1", "v2", "v3"]}

    print("== 1. format")
    a = pd.read_csv(V5 / "submission_A_v5.csv")
    ok = list(a.columns) == ["image_id", "label"] and (a.image_id.values == sample.image_id.values).all() and a.label.isin(LABELS).all()
    print(f"  submission_A_v5.csv valid={ok} | identical to local recompute={(a.set_index('image_id').label.reindex(test.image_id).to_numpy(dtype=object) == pred['A']).all()}")
    b = sample[["image_id"]].merge(pd.DataFrame({"image_id": test.image_id, "label": pred["B"]}), on="image_id")
    b.to_csv(OUT / "submission_B_hybrid.csv", index=False); print(f"  wrote {OUT / 'submission_B_hybrid.csv'}")

    print("\n== 2. OOF paired v5 vs v2 (same folds). page = clean view (test pages are clean), others = corrupted")
    i2, c2, x2 = oof(V2); i5, c5, x5 = oof(V5)
    assert (np.sort(i2) == np.sort(i5)).all()
    o2 = np.argsort(i2); o5 = np.argsort(i5)
    c2, x2, c5, x5 = c2[o2], x2[o2], c5[o5], x5[o5]
    tl2 = np.where((trct == "page")[:, None], c2, x2); tl5 = np.where((trct == "page")[:, None], c5, x5)
    r2, r5 = tl2.argmax(1) == y, tl5.argmax(1) == y
    rows = []
    for t in ["page", "line", "block", "glyph", "all"]:
        m = np.ones(len(y), bool) if t == "all" else trct == t
        rows.append(dict(type=t, n=m.sum(), v2=r2[m].mean(), v5=r5[m].mean(), fixes=int((r5 & ~r2 & m).sum()), breaks=int((r2 & ~r5 & m).sum()),
                         f1_v2=f1_score(y[m], tl2.argmax(1)[m], average="macro"), f1_v5=f1_score(y[m], tl5.argmax(1)[m], average="macro")))
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    m = trct == "page"; g = grp[m]; d = (r5[m].astype(float) - r2[m])
    ug = np.unique(g); rng = np.random.default_rng(0); bs = []
    for _ in range(4000):
        s = rng.choice(ug, len(ug)); bs.append(np.concatenate([d[g == k] for k in s]).mean())
    print(f"  page acc diff v5-v2 = {d.mean():+.3f}, 95% CI (group bootstrap) [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")
    print("  page per-class recall v2:", {LABELS[c]: round(float(r2[m][y[m] == c].mean()), 2) for c in range(7)})
    print("  page per-class recall v5:", {LABELS[c]: round(float(r5[m][y[m] == c].mean()), 2) for c in range(7)})
    print("  (note eda/14: v3 page OOF also beat v2 (+.055) and lost on LB)")

    print("\n== 3. test: rows changed vs v2 (LB 0.91514)")
    for k in ["A", "B"]:
        dd = pred[k] != subs["v2"]
        print(f"  {k}: {dd.sum():3d} rows | by type {pd.Series(tct[dd]).value_counts().to_dict()}")
        print(f"     transitions:", pd.Series([f"{u}->{v}" for u, v in zip(subs['v2'][dd], pred[k][dd])]).value_counts().head(12).to_dict())
    dA = pred["A"] != subs["v2"]
    print(f"  A non-page changes: {(dA & ~pg).sum()} | page changes {(dA & pg).sum()} / {pg.sum()}")
    print(f"  v5 conf on changed pages: median {np.median(p5[dA & pg].max(1)):.2f} | v2 conf on those: {np.median(p2[dA & pg].max(1)):.2f}")
    print(f"  test page conf<0.6: v2 {np.mean(p2[pg].max(1) < .6):.1%} -> v5 {np.mean(p5[pg].max(1) < .6):.1%}")
    print("  predicted class share on pages (%):")
    print(pd.DataFrame({"v2": pd.Series(subs["v2"][pg]).value_counts(normalize=True), "v5": pd.Series(pred["A"][pg]).value_counts(normalize=True)}).reindex(LABELS).mul(100).round(1).T.to_string())

    print("\n== 4. risk: overlap with v3 / v1 / frozen probe")
    ch3 = subs["v3"] != subs["v2"]
    for k in ["A", "B"]:
        dd = pred[k] != subs["v2"]
        print(f"  {k}: changed {dd.sum()} | = v3 label {(dd & (pred[k] == subs['v3'])).sum()} | = v1 label {(dd & (pred[k] == subs['v1'])).sum()}"
              f" | against v1+v2 agreement {(dd & (subs['v1'] == subs['v2'])).sum()} | of v3's {ch3.sum()} changes it copies {(ch3 & (pred[k] == subs['v3'])).sum()}")
    pr = pd.read_csv(ROOT / "eda/outputs/22_page_view_probe/test_page_preds_small.csv").set_index("image_id").sq640_gray.reindex(test.image_id[pg]).to_numpy(dtype=object)
    dp = dA[pg]
    print(f"  frozen 640 probe (independent model, same view): agrees with v5 on changed pages {(pr[dp] == pred['A'][pg][dp]).mean():.1%}, with v2 {(pr[dp] == subs['v2'][pg][dp]).mean():.1%}")

    tab = pd.DataFrame({"image_id": test.image_id, "ctype": tct, "v1": subs["v1"], "v2": subs["v2"], "v3": subs["v3"], "A": pred["A"], "B": pred["B"],
                        "conf_v5": p5.max(1), "conf_v2": p2.max(1)})
    tab.to_csv(OUT / "test_versions.csv", index=False)
    ch = tab[tab.A != tab.v2].sort_values(["ctype", "v2"])
    n = len(ch); r = int(np.ceil(n / 6)) or 1
    fig, axes = plt.subplots(r, 6, figsize=(36, 4.2 * r), squeeze=False)
    for ax in axes.ravel(): ax.axis("off")
    for ax, x in zip(axes.ravel(), ch.itertuples()):
        with Image.open(ROOT / "data/images/test" / x.image_id) as im:
            ax.imshow(ImageOps.exif_transpose(im).convert("RGB"))
        ax.set_title(f"{x.image_id} {x.ctype}\nv2={x.v2}({x.conf_v2:.2f}) -> v5={x.A}({x.conf_v5:.2f}) v3={x.v3}", fontsize=11)
    plt.tight_layout(); plt.savefig(OUT / "A_vs_v2_changed.png", dpi=40); plt.close()
    print("->", OUT)
