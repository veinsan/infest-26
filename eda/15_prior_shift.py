"""15 - Label (prior) shift: does the test set contain far more pegon than train, as EM suggested (eda/11)?
~75 of v2's ~100 LB errors must sit in rows where v1, v2, v3 all agree -> a systematic bias, and all models
share the TRAIN prior. Two independent estimators on v2 outputs:
  EM   (Saerens 2002) on temperature-calibrated probabilities (T fitted on OOF)
  BBSE (Lipton 2018): solve C w = q, C = OOF confusion p(pred|true), q = test prediction distribution
OOF protocol per type: pages clean view, others corrupted view (the corrected proxy from eda/14).
Bootstrap over test rows for CIs; per content type; what a correction would change.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, nnls

from common import LABELS, ROOT, out_dir

OUT = out_dir("15_prior_shift")
RUN = ROOT / "results/v2/main"


def softmax(x, T=1.0):
    z = x / T; e = np.exp(z - z.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)


def em_prior(p, src, it=200):
    pri = src.copy()
    for _ in range(it):
        a = p * (pri / src); a /= a.sum(1, keepdims=True)
        new = a.mean(0)
        if np.abs(new - pri).max() < 1e-7:
            break
        pri = new
    return pri, a


def bbse(C, q):
    """C[j, i] = p(pred=j | true=i); q[j] = test p(pred=j). Solve C @ pi = q with pi >= 0, sum 1."""
    pi, _ = nnls(C, q)
    return pi / pi.sum()


if __name__ == "__main__":
    tr = pd.read_csv(ROOT / "data/train.csv")
    y = tr.label.map({l: i for i, l in enumerate(LABELS)}).values
    ctype_tr = pd.read_csv(out_dir("08_v2_gap_decomposition") / "train_types_oof.csv").set_index("image_id").ctype.reindex(tr.image_id).values
    idx = np.concatenate([np.load(RUN / f"fold{k}_val_idx.npy") for k in range(5)])
    lc = np.concatenate([np.load(RUN / f"fold{k}_oof_clean_logits.npy") for k in range(5)])
    lx = np.concatenate([np.load(RUN / f"fold{k}_oof_corrupt_logits.npy") for k in range(5)])
    oof_logit = np.where((ctype_tr[idx] == "page")[:, None], lc, lx); yo = y[idx]

    nll = lambda T: -np.log(softmax(oof_logit, T)[np.arange(len(yo)), yo] + 1e-12).mean()
    T = minimize_scalar(nll, bounds=(0.3, 5), method="bounded").x
    print(f"== temperature on OOF: T={T:.2f} (NLL {nll(1):.4f} -> {nll(T):.4f})")

    te = pd.read_csv(ROOT / "data/test.csv")
    ctype_te = pd.read_csv(out_dir("08_v2_gap_decomposition") / "test_types_pred.csv").set_index("image_id").ctype.reindex(te.image_id).values
    tl = np.stack([np.load(RUN / f"fold{k}_test_logits.npy") for k in range(5)])
    p_raw = softmax(tl).mean(0); p_cal = softmax(tl, T).mean(0)
    po = softmax(oof_logit, T)
    src = po.mean(0)
    C = np.zeros((7, 7))
    for i in range(7):
        m = yo == i
        C[:, i] = np.bincount(po[m].argmax(1), minlength=7) / m.sum()
    q = np.bincount(p_cal.argmax(1), minlength=7) / len(te)

    pri_em, adj = em_prior(p_cal, src)
    pri_bb = bbse(C, q)
    rng = np.random.default_rng(0)
    B = [[], []]
    for _ in range(500):
        b = rng.integers(0, len(te), len(te))
        B[0].append(em_prior(p_cal[b], src)[0]); B[1].append(bbse(C, np.bincount(p_cal[b].argmax(1), minlength=7) / len(te)))
    ci = lambda a: [f"{lo * 100:.1f}-{hi * 100:.1f}" for lo, hi in zip(np.percentile(a, 2.5, 0), np.percentile(a, 97.5, 0))]
    tab = pd.DataFrame({"train_%": np.bincount(y, minlength=7) / len(y) * 100, "v2_pred_%": np.bincount(p_raw.argmax(1), minlength=7) / len(te) * 100,
                        "EM_%": pri_em * 100, "EM_CI": ci(np.array(B[0])), "BBSE_%": pri_bb * 100, "BBSE_CI": ci(np.array(B[1]))}, index=LABELS)
    pd.set_option("display.width", 220)
    print("\n== test class prior estimates")
    print(tab.round(1))

    print("\n== per content type (EM on calibrated probs; BBSE with type-specific OOF confusion)")
    rows = []
    for t in ["line", "block", "glyph", "page"]:
        m, mo = ctype_te == t, ctype_tr[idx] == t
        Ct = np.zeros((7, 7))
        for i in range(7):
            mm = mo & (yo == i)
            Ct[:, i] = np.bincount(po[mm].argmax(1), minlength=7) / mm.sum() if mm.sum() >= 5 else C[:, i]
        e_ = em_prior(p_cal[m], po[mo].mean(0))[0]
        b_ = bbse(Ct, np.bincount(p_cal[m].argmax(1), minlength=7) / m.sum())
        for i, l in enumerate(LABELS):
            rows.append(dict(ctype=t, cls=l, train_type_share=(y[ctype_tr == t] == i).mean() * 100, v2_pred=(p_raw[m].argmax(1) == i).mean() * 100, EM=e_[i] * 100, BBSE=b_[i] * 100))
    pt = pd.DataFrame(rows).round(1)
    print(pt[pt.cls.isin(["jawi", "pegon"])].to_string(index=False))

    ch = adj.argmax(1) != p_raw.argmax(1)
    tr_ = pd.Series([f"{LABELS[a]}->{LABELS[b]}" for a, b in zip(p_raw.argmax(1)[ch], adj.argmax(1)[ch])])
    print(f"\n== EM correction (calibrated) would change {ch.sum()} rows: {tr_.value_counts().head(8).to_dict()} | by type {pd.Series(ctype_te[ch]).value_counts().to_dict()}")
    # BBSE-weighted correction: reweight calibrated probs by w = pi_test / pi_src
    w = pri_bb / src
    pb = p_cal * w; pb /= pb.sum(1, keepdims=True)
    chb = pb.argmax(1) != p_raw.argmax(1)
    print(f"   BBSE correction would change {chb.sum()} rows: {pd.Series([f'{LABELS[a]}->{LABELS[b]}' for a, b in zip(p_raw.argmax(1)[chb], pb.argmax(1)[chb])]).value_counts().head(8).to_dict()}")
    subs = pd.read_csv(RUN / "submission.csv").set_index("image_id").label
    for name, p in [("v2_prior_em", adj), ("v2_prior_bbse", pb)]:
        pred = pd.Series(np.array(LABELS)[p.argmax(1)], index=te.image_id)
        tw = pd.read_csv(ROOT / "data/test_train_twins.csv").set_index("image_id").train_label
        pred.loc[pred.index.isin(tw.index)] = tw.reindex(pred.index[pred.index.isin(tw.index)]).values
        s = pd.read_csv(ROOT / "data/sample_submission.csv")[["image_id"]].assign(label=lambda d: d.image_id.map(pred))
        assert s.label.notna().all()
        s.to_csv(OUT / f"{name}.csv", index=False)
        print(f"   wrote {name}.csv: differs from v2 submission on {(s.set_index('image_id').label != subs.reindex(s.image_id)).sum()} rows")
    tab.to_csv(OUT / "prior_estimates.csv"); pt.to_csv(OUT / "prior_by_type.csv", index=False)
    print("->", OUT)
