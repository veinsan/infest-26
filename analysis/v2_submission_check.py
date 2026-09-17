"""Is results/v2/main/submission.csv safe as the first submission of the day?
  1. format / consistency (sample order, labels, twins, probs == submission argmax)
  2. fold agreement + confidence on test vs OOF-corrupt  -> how far test is outside what CV measured
  3. expected accuracy on test from OOF-corrupt calibration (optimistic under shift)
  4. v1 vs v2 disagreement matrix + where v2 prediction share deviates (pegon surge)
  5. which test subsets drive the deviation (non-line, JPEG, colourful, black canvas)
Figures -> analysis/outputs/v2_submission_check/"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eda")); sys.path.insert(0, str(ROOT / "prepro"))
from common import LABELS, load_frames, out_dir, pmap  # noqa: E402
import pipeline as P  # noqa: E402

RUN = ROOT / "results/v2/main"
OUT = ROOT / "analysis/outputs/v2_submission_check"
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 220)


def softmax(x):
    return torch.softmax(torch.from_numpy(x), 1).numpy()


def canvas_black(path):
    pp = P.preprocess(P.load_rgb(path))
    tiles = P.to_canvas(pp, train=False)
    return float(np.mean([(np.asarray(t) < 128).mean() for t in tiles])), pp.width / pp.height < 2


if __name__ == "__main__":
    df = load_frames()
    tr = df[df.split == "train"].reset_index(drop=True)
    te = df[df.split == "test"].reset_index(drop=True)
    sample = pd.read_csv(ROOT / "data/sample_submission.csv")
    sub = pd.read_csv(RUN / "submission.csv")
    prob = pd.read_csv(RUN / "test_probabilities.csv")
    y = tr.label.map({l: i for i, l in enumerate(LABELS)}).values

    print("== 1. format")
    assert list(sub.columns) == ["image_id", "label"] and len(sub) == len(sample) and (sub.image_id.values == sample.image_id.values).all()
    assert sub.label.isin(LABELS).all() and sub.image_id.is_unique
    prob = prob.set_index("image_id").loc[sub.image_id]
    argmax = np.array(LABELS)[prob[LABELS].values.argmax(1)]
    twins = pd.read_csv(ROOT / "data/test_train_twins.csv").set_index("image_id").train_label
    diff = sub.label.values != argmax
    print(f"  rows={len(sub)} OK | submission != prob argmax: {diff.sum()} (twin overrides: {sub.image_id[diff].isin(twins.index).sum()})")
    tw = sub.set_index("image_id").label.reindex(twins.index)
    print(f"  twins agree with train label: {(tw == twins).sum()}/{len(twins)}")

    print("\n== 2. fold agreement & confidence")
    fold_logits = np.stack([np.load(RUN / f"fold{k}_test_logits.npy") for k in range(5)])  # test.csv order
    order = te.image_id.values
    assert (order == prob.index.values).all() or set(order) == set(prob.index)
    fold_pred = fold_logits.argmax(2)
    ens = softmax(fold_logits.transpose(1, 0, 2).reshape(-1, 7)).reshape(len(te), 5, 7).mean(1)
    agree = (fold_pred == ens.argmax(1)).sum(0)
    idx = np.concatenate([np.load(RUN / f"fold{k}_val_idx.npy") for k in range(5)])
    oof = softmax(np.concatenate([np.load(RUN / f"fold{k}_oof_corrupt_logits.npy") for k in range(5)]))
    oof_y = y[idx]
    oof_conf, te_conf = oof.max(1), ens.max(1)
    print(f"  test: all 5 folds agree {np.mean(agree == 5):.1%} | <=3 agree {np.mean(agree <= 3):.1%}")
    print(f"  confidence median  OOF-corrupt={np.median(oof_conf):.3f}  test={np.median(te_conf):.3f}")
    print(f"  share conf<0.6     OOF-corrupt={np.mean(oof_conf < .6):.1%}  test={np.mean(te_conf < .6):.1%}")

    print("\n== 3. expected test accuracy from OOF calibration (per confidence bin)")
    bins = [0, .4, .5, .6, .7, .8, .9, 1.01]
    ob = np.digitize(oof_conf, bins); tb = np.digitize(te_conf, bins)
    acc_bin = {b: (oof.argmax(1)[ob == b] == oof_y[ob == b]).mean() for b in np.unique(ob)}
    rows = [dict(bin=f"[{bins[b - 1]:.1f},{bins[b]:.1f})", oof_n=int((ob == b).sum()), oof_acc=acc_bin.get(b, np.nan), test_n=int((tb == b).sum())) for b in range(1, len(bins))]
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    exp_acc = np.nansum([acc_bin.get(b, 0.5) * (tb == b).sum() for b in range(1, len(bins))]) / len(te)
    print(f"  OOF-corrupt accuracy={np.mean(oof.argmax(1) == oof_y):.4f} | calibrated expected TEST accuracy ~ {exp_acc:.4f} (upper bound: shift makes it optimistic)")

    print("\n== 4. v1 vs v2")
    v1 = pd.read_csv(ROOT / "results/v1/submission.csv").set_index("image_id").label.reindex(sub.image_id).values
    v2 = sub.label.values
    print(f"  disagreement: {np.mean(v1 != v2):.1%} ({(v1 != v2).sum()} rows)")
    print(pd.crosstab(pd.Series(v1, name="v1"), pd.Series(v2, name="v2")))
    share = pd.DataFrame({"train_%": tr.label.value_counts(normalize=True) * 100, "v1_%": pd.Series(v1).value_counts(normalize=True) * 100,
                          "v2_%": pd.Series(v2).value_counts(normalize=True) * 100}).round(1)
    share["oof_pred_%"] = (pd.Series(np.array(LABELS)[oof.argmax(1)]).value_counts(normalize=True) * 100).round(1)
    print(share)

    print("\n== 5. where do test predictions come from")
    meta = pd.read_csv(out_dir("01_profile") / "meta.csv")
    px = pd.read_csv(out_dir("03_pixel_stats") / "pixel_feats.csv")
    t = te[["image_id", "path"]].merge(meta[meta.split == "test"][["image_id", "fmt", "ratio"]], on="image_id") \
        .merge(px[px.split == "test"][["image_id", "colorfulness"]], on="image_id")
    t = t.set_index("image_id").loc[te.image_id].reset_index()
    cb = pmap(canvas_black, list(t.path))
    t["black"], t["nonline"] = [c[0] for c in cb], [c[1] for c in cb]
    t["v1"], t["v2"] = pd.Series(v1, index=sub.image_id).reindex(t.image_id).values, pd.Series(v2, index=sub.image_id).reindex(t.image_id).values
    t["conf"], t["agree"] = te_conf, agree
    for name, m in [("all", np.ones(len(t), bool)), ("non-line", t.nonline.values), ("line", ~t.nonline.values), ("JPEG", (t.fmt == "JPEG").values),
                    ("colourful>15", (t.colorfulness > 15).values), ("canvas>50% black", (t.black > .5).values), ("conf<0.6", (t.conf < .6).values)]:
        s = t[m]
        print(f"  {name:17s} n={m.sum():4d} conf_med={s.conf.median():.2f} v1!=v2={np.mean(s.v1 != s.v2):5.1%} pegon% v2={np.mean(s.v2 == 'pegon'):5.1%} v1={np.mean(s.v1 == 'pegon'):5.1%} jawi% v2={np.mean(s.v2 == 'jawi'):5.1%}")
    t.drop(columns="path").to_csv(OUT / "test_diagnostics.csv", index=False)

    def grid(sel, fname, title):
        sel = sel.head(20)
        if not len(sel):
            return
        fig, axes = plt.subplots(int(np.ceil(len(sel) / 4)), 4, figsize=(24, 2.6 * np.ceil(len(sel) / 4)), squeeze=False)
        for a in axes.ravel(): a.axis("off")
        for a, r in zip(axes.ravel(), sel.itertuples()):
            a.imshow(P.load_rgb(r.path)); a.set_title(f"v1={r.v1} v2={r.v2} conf={r.conf:.2f} agree={r.agree}/5", fontsize=9)
        fig.suptitle(title); plt.tight_layout(); plt.savefig(OUT / fname, dpi=60); plt.close()
    grid(t[(t.v1 == "jawi") & (t.v2 == "pegon")].sample(frac=1, random_state=0), "v1_jawi_to_v2_pegon.png", "v1 jawi -> v2 pegon")
    grid(t[(t.v1 != t.v2) & (t.v2 != "pegon")].sample(frac=1, random_state=0), "v1_v2_other_changes.png", "other v1 -> v2 changes")
    grid(t[t.black > .5], "black_canvas_test.png", "test canvas >50% black")
    grid(t.nsmallest(20, "conf"), "lowest_confidence.png", "lowest v2 confidence")
    grid(t[t.v2 == "pegon"].sample(20, random_state=1), "v2_pegon_random.png", "random v2 = pegon")
    print("->", OUT)
