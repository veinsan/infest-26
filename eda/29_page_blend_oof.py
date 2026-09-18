"""29 - eda/25: B was right on only ~50% of its 37 page changes -> v5 page predictions are not strictly better than v2's.
Do v2 (160x640 view, sees layout small) and v5 (640 view) make DIFFERENT page errors? If yes, a blend beats both.
OOF (same folds), clean page view for both: acc / macro-F1 for w*v5 + (1-w)*v2, w in 0..1; group-bootstrap CI vs v5.
Then on test pages: how many rows each blend changes vs B, and vs v2.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from common import LABELS, ROOT, out_dir

OUT = out_dir("29_page_blend_oof")


def sm(x):
    e = np.exp(x - x.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)


def oof(d):
    idx = np.concatenate([np.load(d / f"fold{k}_val_idx.npy") for k in range(5)])
    p = sm(np.concatenate([np.load(d / f"fold{k}_oof_clean_logits.npy") for k in range(5)])); r = np.empty_like(p); r[idx] = p; return r


if __name__ == "__main__":
    train = pd.read_csv(ROOT / "data/train.csv"); y = train.label.map({l: i for i, l in enumerate(LABELS)}).to_numpy()
    ct = pd.read_csv(out_dir("08_v2_gap_decomposition") / "train_types_oof.csv").set_index("image_id").ctype.reindex(train.image_id).to_numpy()
    grp = pd.read_csv(ROOT / "data/folds.csv").set_index("image_id").dup_group.reindex(train.image_id).to_numpy()
    m = ct == "page"
    o2, o5 = oof(ROOT / "results/v2/main")[m], oof(ROOT / "results/v5/main")[m]; yy, g = y[m], grp[m]
    r2, r5 = o2.argmax(1) == yy, o5.argmax(1) == yy
    print(f"== OOF pages n={m.sum()}: v2 right {r2.mean():.3f} | v5 right {r5.mean():.3f} | both {np.mean(r2 & r5):.3f} | only v2 {np.mean(r2 & ~r5):.3f} | only v5 {np.mean(~r2 & r5):.3f} | neither {np.mean(~r2 & ~r5):.3f}")
    rows = []
    ug = np.unique(g); rng = np.random.default_rng(0); boots = [rng.choice(ug, len(ug)) for _ in range(2000)]
    idx_of = {k: np.where(g == k)[0] for k in ug}
    for w in np.arange(0, 1.01, .1):
        p = w * o5 + (1 - w) * o2; r = p.argmax(1) == yy
        d = r.astype(float) - r5
        bs = [np.concatenate([d[idx_of[k]] for k in b]).mean() for b in boots[:500]]
        rows.append(dict(w_v5=round(w, 1), acc=r.mean(), f1=f1_score(yy, p.argmax(1), average="macro"), vs_v5=d.mean(), ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5)))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    # geometric (log) mean as alternative
    pl = np.exp(.5 * np.log(o5 + 1e-9) + .5 * np.log(o2 + 1e-9)); print(f"   log-mean 0.5: acc {(pl.argmax(1) == yy).mean():.3f}")

    test = pd.read_csv(ROOT / "data/test.csv")
    tv = pd.read_csv(ROOT / "analysis/outputs/v5_submission_check/test_versions.csv")
    pg = (tv.ctype == "page").to_numpy()
    t2 = np.mean([sm(np.load(ROOT / f"results/v2/main/fold{k}_test_logits.npy")) for k in range(5)], 0)
    t5 = np.mean([sm(np.load(ROOT / f"results/v5/main/fold{k}_test_logits.npy")) for k in range(5)], 0)
    L = np.array(LABELS)
    print("\n== test pages: rows changed vs B / vs v2 per blend weight")
    for w in [0.5, 0.6, 0.7]:
        lab = L[(w * t5 + (1 - w) * t2).argmax(1)][pg]
        print(f"   w={w}: vs B {int((lab != tv.B.values[pg]).sum())} | vs v2 {int((lab != tv.v2.values[pg]).sum())}")
    pd.DataFrame(rows).to_csv(OUT / "blend_oof.csv", index=False)
    print("->", OUT)
