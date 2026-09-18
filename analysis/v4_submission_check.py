"""Is a v4 file (A: 10 CV models, B: +3 full-data, C: B + v2's 5 folds) a safe first submission of the day?
  1. format / consistency; build C locally (v2 logits are here, not on vast.ai)
  2. OOF: v2 seed42 alone vs new seeds vs all three seeds (same folds -> honest seed ensemble)
  3. test: rows each file changes vs v2, by content type; member agreement; confidence
  4. risk check with LB forensics (eda/12): do v4's changes move TOWARD v3's labels (v3 was right on ~20-30% of its changes)
     or toward v1? rows where v4 sides with v3 against v2+v1 are the risky ones
  5. grid of changed rows
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
V2, V4 = ROOT / "results/v2/main", ROOT / "results/v4"
OUT = ROOT / "analysis/outputs/v4_submission_check"
OUT.mkdir(parents=True, exist_ok=True)
LABELS = ["bali", "jawa", "jawi", "lampung", "lontara", "pegon", "sunda"]
L = np.array(LABELS)
pd.set_option("display.width", 220)


def sm(x):
    e = np.exp(x - x.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)


def ctype_of(path):
    with Image.open(path) as im:
        w, h = im.size
    return "page" if h >= 250 else ("line" if w / h >= 2 else ("block" if w >= 80 else "glyph"))


if __name__ == "__main__":
    test = pd.read_csv(ROOT / "data/test.csv"); sample = pd.read_csv(ROOT / "data/sample_submission.csv")
    train = pd.read_csv(ROOT / "data/train.csv"); y = train.label.map({l: i for i, l in enumerate(LABELS)}).to_numpy()
    tct = np.array([ctype_of(ROOT / "data/images/test" / i) for i in test.image_id])
    trct = np.array([ctype_of(ROOT / "data/images/train" / i) for i in train.image_id])
    twins = pd.read_csv(ROOT / "data/test_train_twins.csv").set_index("image_id").train_label
    m_tw = test.image_id.isin(twins.index).to_numpy()

    cv = [sm(np.load(V4 / f"cv_s{s}_f{k}_test_logits.npy")) for s in [7, 2026] for k in range(5)]
    full = [sm(np.load(V4 / f"full_s{s}_test_logits.npy")) for s in [11, 23, 37]]
    v2m = [sm(np.load(V2 / f"fold{k}_test_logits.npy")) for k in range(5)]
    P = {"A": np.mean(cv, 0), "B": np.mean(cv + full, 0), "C": np.mean(cv + full + v2m, 0), "v2": np.mean(v2m, 0)}
    pred = {}
    for k, p in P.items():
        o = L[p.argmax(1)].astype(object); o[m_tw] = twins.reindex(test.image_id[m_tw]).to_numpy(dtype=object); pred[k] = o
    subs = {v: pd.read_csv(ROOT / f"results/{v}{'/main' if v != 'v1' else ''}/submission.csv").set_index("image_id").label.reindex(test.image_id).to_numpy(dtype=object) for v in ["v1", "v2", "v3"]}

    print("== 1. format")
    for name, key in [("submission_A_cv.csv", "A"), ("submission_B_cv_full.csv", "B")]:
        s = pd.read_csv(V4 / name)
        ok = list(s.columns) == ["image_id", "label"] and (s.image_id.values == sample.image_id.values).all() and s.label.isin(LABELS).all()
        same = (s.set_index("image_id").label.reindex(test.image_id).to_numpy(dtype=object) == pred[key]).all()
        print(f"  {name}: valid={ok} | identical to local recompute={same}")
    print(f"  local v2 recompute == v2 submission: {(pred['v2'] == subs['v2']).all()}")
    c = sample[["image_id"]].merge(pd.DataFrame({"image_id": test.image_id, "label": pred["C"]}), on="image_id")
    c.to_csv(OUT / "submission_C_plus_v2.csv", index=False)
    print(f"  wrote {OUT / 'submission_C_plus_v2.csv'}")

    print("\n== 2. OOF (same folds.csv for all seeds)")
    def oof(prefix_fn):
        idx = np.concatenate([np.load(prefix_fn(k, "val_idx")) for k in range(5)])
        pc = sm(np.concatenate([np.load(prefix_fn(k, "oof_clean_logits")) for k in range(5)]))
        px = sm(np.concatenate([np.load(prefix_fn(k, "oof_corrupt_logits")) for k in range(5)]))
        return idx, pc, px
    seeds = {"v2 s42": oof(lambda k, n: V2 / f"fold{k}_{n}.npy"), "s7": oof(lambda k, n: V4 / f"cv_s7_f{k}_{n}.npy"), "s2026": oof(lambda k, n: V4 / f"cv_s2026_f{k}_{n}.npy")}
    idx = seeds["v2 s42"][0]
    for s in seeds.values():
        assert (s[0] == idx).all()
    rows = []
    for names in [["v2 s42"], ["s7"], ["s2026"], ["s7", "s2026"], ["v2 s42", "s7", "s2026"]]:
        pc = np.mean([seeds[n][1] for n in names], 0); px = np.mean([seeds[n][2] for n in names], 0)
        yy, ct = y[idx], trct[idx]; tl = np.where((ct == "page")[:, None], pc, px)
        rows.append(dict(members=" + ".join(names), f1_clean=f1_score(yy, pc.argmax(1), average="macro"), f1_corrupt=f1_score(yy, px.argmax(1), average="macro"),
                         f1_testlike=f1_score(yy, tl.argmax(1), average="macro"), page_clean=(pc.argmax(1)[ct == "page"] == yy[ct == "page"]).mean(),
                         line_corrupt=(px.argmax(1)[ct == "line"] == yy[ct == "line"]).mean(), block_corrupt=(px.argmax(1)[ct == "block"] == yy[ct == "block"]).mean()))
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    print("\n== 3. test: rows changed vs v2 (LB 0.91514)")
    allm = np.stack(cv + full + v2m); arg = allm.argmax(2)
    for k in ["A", "B", "C"]:
        d = pred[k] != subs["v2"]
        print(f"  {k}: {d.sum():3d} rows | by type {pd.Series(tct[d]).value_counts().to_dict()}")
    agree = (arg == P["C"].argmax(1)).mean(0)
    print(f"  18-member agreement with C: all agree {np.mean(agree == 1):.1%} | <=60% {np.mean(agree <= .6):.1%}")
    print(f"  confidence<0.6: v2 {np.mean(P['v2'].max(1) < .6):.1%} | B {np.mean(P['B'].max(1) < .6):.1%} | C {np.mean(P['C'].max(1) < .6):.1%}")

    print("\n== 4. risk check with v1 / v3 (LB 0.863 / 0.891)")
    for k in ["A", "B", "C"]:
        d = pred[k] != subs["v2"]
        toward_v3 = d & (pred[k] == subs["v3"]); toward_v1 = d & (pred[k] == subs["v1"])
        alone_vs_v1v2 = d & (subs["v1"] == subs["v2"])
        print(f"  {k}: changed {d.sum()} | = v3's label {toward_v3.sum()} | = v1's label {toward_v1.sum()} | against v1+v2 agreement {alone_vs_v1v2.sum()}"
              f" | of v3's 59 changes, {k} copies {((subs['v3'] != subs['v2']) & (pred[k] == subs['v3'])).sum()}")
    d = pred["B"] != subs["v2"]
    print("  B transitions v2 -> B:", pd.Series([f"{a}->{b}" for a, b in zip(subs["v2"][d], pred["B"][d])]).value_counts().to_dict())

    tab = pd.DataFrame({"image_id": test.image_id, "ctype": tct, "v1": subs["v1"], "v2": subs["v2"], "v3": subs["v3"], "A": pred["A"], "B": pred["B"], "C": pred["C"],
                        "conf_B": P["B"].max(1), "agree": agree})
    tab.to_csv(OUT / "test_versions.csv", index=False)
    ch = tab[tab.B != tab.v2].sort_values("ctype")
    n = len(ch); r = int(np.ceil(n / 4)) or 1
    fig, axes = plt.subplots(r, 4, figsize=(24, 2.8 * r), squeeze=False)
    for a in axes.ravel(): a.axis("off")
    for a, x in zip(axes.ravel(), ch.itertuples()):
        with Image.open(ROOT / "data/images/test" / x.image_id) as im:
            a.imshow(ImageOps.exif_transpose(im).convert("RGB"))
        a.set_title(f"{x.ctype} v1={x.v1} v2={x.v2} v3={x.v3} B={x.B} conf={x.conf_B:.2f}", fontsize=8)
    plt.tight_layout(); plt.savefig(OUT / "B_vs_v2_changed.png", dpi=55); plt.close()
    print("->", OUT)
