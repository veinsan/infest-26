"""Is results/v3/main/submission.csv (or submission_prior.csv) a safe second submission after v2 (LB 0.91514)?
  1. format + consistency (probs argmax, twins)
  2. optimism of the OOF numbers: epoch picked ON the val metric -> compare best vs mean of later epochs
  3. v2 vs v3 on test: disagreement per content type, class transitions, fold agreement / confidence
  4. v3 vs v3-prior
  5. grids of changed images (pages, lines) for eyeball verification
Figures -> analysis/outputs/v3_submission_check/"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
V2, V3 = ROOT / "results/v2/main", ROOT / "results/v3/main"
OUT = ROOT / "analysis/outputs/v3_submission_check"
OUT.mkdir(parents=True, exist_ok=True)
LABELS = ["bali", "jawa", "jawi", "lampung", "lontara", "pegon", "sunda"]
pd.set_option("display.width", 220)


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)


def show(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


if __name__ == "__main__":
    test = pd.read_csv(ROOT / "data/test.csv")
    sample = pd.read_csv(ROOT / "data/sample_submission.csv")
    types = pd.read_csv(ROOT / "eda/outputs/08_v2_gap_decomposition/test_types_pred.csv")[["image_id", "ctype"]]
    test = test.merge(types, on="image_id", how="left")
    s2 = pd.read_csv(V2 / "submission.csv").set_index("image_id").label.reindex(test.image_id).values
    s3 = pd.read_csv(V3 / "submission.csv").set_index("image_id").label.reindex(test.image_id).values
    s3p = pd.read_csv(V3 / "submission_prior.csv").set_index("image_id").label.reindex(test.image_id).values

    print("== 1. format")
    for name in ["submission.csv", "submission_prior.csv"]:
        s = pd.read_csv(V3 / name)
        ok = list(s.columns) == ["image_id", "label"] and len(s) == len(sample) and (s.image_id.values == sample.image_id.values).all() \
            and s.label.isin(LABELS).all() and s.image_id.is_unique
        print(f"  {name}: rows={len(s)} valid={ok}")
    fl = np.stack([np.load(V3 / f"fold{k}_test_logprob.npy") for k in range(5)])  # test.csv order, log-probs
    probs = softmax(fl).mean(0)
    arg = np.array(LABELS)[probs.argmax(1)]
    twins = pd.read_csv(ROOT / "data/test_train_twins.csv").set_index("image_id").train_label
    m_tw = test.image_id.isin(twins.index).values
    print(f"  submission == prob argmax except twins: {(arg[~m_tw] == s3[~m_tw]).all()} | twin overrides that changed argmax: {(arg[m_tw] != s3[m_tw]).sum()}")

    print("\n== 2. selection optimism (best epoch picked on the same val metric)")
    rows = []
    for k in range(5):
        h = pd.read_csv(V3 / f"fold{k}_history.csv")
        b = h.loc[h.f1_testlike_reweighted.idxmax()]
        late = h[h.epoch >= 5]
        rows.append(dict(fold=k, best_epoch=int(b.epoch), best_reweighted=b.f1_testlike_reweighted, late_mean_reweighted=late.f1_testlike_reweighted.mean(),
                         best_page=b.page_acc, late_mean_page=late.page_acc.mean(), best_line=b.line_acc, late_mean_line=late.line_acc.mean()))
    opt = pd.DataFrame(rows)
    print(opt.round(4).to_string(index=False))
    print(f"  mean best reweighted {opt.best_reweighted.mean():.4f} vs late-epoch mean {opt.late_mean_reweighted.mean():.4f} "
          f"(optimism ~{opt.best_reweighted.mean() - opt.late_mean_reweighted.mean():.4f}); page {opt.best_page.mean():.3f} vs {opt.late_mean_page.mean():.3f}")
    v2h = pd.concat([pd.read_csv(V2 / f"fold{k}_history.csv") for k in range(5)])
    v2b = v2h.loc[v2h.groupby("fold").f1_corrupt.idxmax()]
    print(f"  v2 same check (f1_corrupt): best {v2b.f1_corrupt.mean():.4f} vs late mean {v2h[v2h.epoch >= 5].groupby('fold').f1_corrupt.mean().mean():.4f}")

    print("\n== 3. v2 vs v3 on test")
    agree = (softmax(fl).argmax(2) == probs.argmax(1)).sum(0)
    conf = probs.max(1)
    d = test.assign(v2=s2, v3=s3, v3p=s3p, conf=conf, agree=agree)
    tab = d.groupby("ctype").apply(lambda g: pd.Series(dict(n=len(g), v2_ne_v3=(g.v2 != g.v3).mean(), v3_conf_med=g.conf.median(),
                                                           v3_conf_lt06=(g.conf < .6).mean(), all5agree=(g.agree == 5).mean())), include_groups=False)
    print(tab.round(3))
    print(f"  overall v2!=v3: {(s2 != s3).mean():.1%} ({(s2 != s3).sum()} rows) | all 5 folds agree {np.mean(agree == 5):.1%} | conf<0.6 {np.mean(conf < .6):.1%} (v2 was 8.0%)")
    print("  transitions v2 -> v3 (changed rows):")
    print(pd.crosstab(pd.Series(s2[s2 != s3], name="v2"), pd.Series(s3[s2 != s3], name="v3")))
    dist = pd.DataFrame({"v2_%": pd.Series(s2).value_counts(normalize=True) * 100, "v3_%": pd.Series(s3).value_counts(normalize=True) * 100,
                         "v3prior_%": pd.Series(s3p).value_counts(normalize=True) * 100}).reindex(LABELS).round(1)
    print(dist)
    print("  page predictions v2 vs v3 (%):")
    pg = d[d.ctype == "page"]
    print(pd.DataFrame({"v2": pg.v2.value_counts(normalize=True) * 100, "v3": pg.v3.value_counts(normalize=True) * 100}).reindex(LABELS).round(1))

    print("\n== 4. v3 vs v3-prior")
    ch = s3 != s3p
    print(f"  changed {ch.sum()} rows; by ctype {d[ch].ctype.value_counts().to_dict()}; transitions {pd.Series([f'{a}->{b}' for a, b in zip(s3[ch], s3p[ch])]).value_counts().head(6).to_dict()}")
    print(f"  of those, v2 agreed with v3: {(s2[ch] == s3[ch]).mean():.0%}, with prior: {(s2[ch] == s3p[ch]).mean():.0%}")

    def grid(sub, fname, title, n=24):
        sub = sub.head(n)
        if not len(sub):
            return
        r = int(np.ceil(len(sub) / 4))
        fig, axes = plt.subplots(r, 4, figsize=(24, 2.8 * r), squeeze=False)
        for a in axes.ravel(): a.axis("off")
        for a, x in zip(axes.ravel(), sub.itertuples()):
            a.imshow(show(ROOT / "data/images/test" / x.image_id)); a.set_title(f"v2={x.v2} v3={x.v3} conf={x.conf:.2f} agree={x.agree}/5", fontsize=9)
        fig.suptitle(title); plt.tight_layout(); plt.savefig(OUT / fname, dpi=55); plt.close()
    grid(d[(d.v2 != d.v3) & (d.ctype == "page")].sample(frac=1, random_state=0), "changed_pages.png", "v2 -> v3 changed PAGES")
    grid(d[(d.v2 != d.v3) & (d.ctype != "page")].sample(frac=1, random_state=0), "changed_nonpages.png", "v2 -> v3 changed line/block/glyph")
    grid(d[ch].sample(frac=1, random_state=0), "prior_changes.png", "v3 -> v3prior changes")
    d.to_csv(OUT / "test_v2_v3.csv", index=False)
    print("->", OUT)
