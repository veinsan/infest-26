"""25 - LB forensics after v5-hybrid (B): v2 0.91514 -> B 0.91820 (+0.00306), B differs from v2 on 37 PAGE rows only.
  1. same Monte-Carlo as eda/12: q = share of the 37 changed rows where B is right, per public fraction
  2. re-read v3 (0.89077): 23 of v3's 59 changes are shared with B. With q_B known, how right must v3's OTHER 36 rows
     have been to produce -0.0244? -> tells whether the round-3 conclusion ("page tiles hurt") was about pages at all
  3. per-row value of one fixed row (how many rows = +0.0018 to reach 0.92)
"""
import importlib

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from common import LABELS, ROOT, out_dir

F = importlib.import_module("12_lb_forensics")
OUT = out_dir("25_lb_forensics_v5")
LB = {"v1": 0.86289, "v2": 0.91514, "v3": 0.89077, "B": 0.91820}
L = {l: i for i, l in enumerate(LABELS)}


def simulate_multi(pa, pb, second, q_of, a, frac, n_sim=200, seed=0):
    """like eda/12 simulate, but q may differ per changed row (array)."""
    rng = np.random.default_rng(seed); n = len(pa); ch = pa != pb; out = []
    for _ in range(n_sim):
        truth = np.where(rng.random(n) < a, pa, second)
        uc = rng.random(n)
        t_ch = np.where(uc < q_of, pb, np.where(uc < q_of + (1 - q_of) * 0.9, pa, second))
        truth = np.where(ch, t_ch, truth)
        bad = ch & (uc >= q_of + (1 - q_of) * 0.9) & ((truth == pa) | (truth == pb))
        truth[bad] = rng.integers(0, 7, bad.sum())
        pub = rng.random(n) < frac
        out.append(f1_score(truth[pub], pb[pub], average="macro", labels=range(7)) - f1_score(truth[pub], pa[pub], average="macro", labels=range(7)))
    return np.mean(out), np.std(out)


if __name__ == "__main__":
    test = pd.read_csv(ROOT / "data/test.csv")
    rd = lambda p: pd.read_csv(p).set_index("image_id").label.reindex(test.image_id).map(L).values
    subs = {v: rd(ROOT / f"results/{v}{'/main' if v != 'v1' else ''}/submission.csv") for v in ["v1", "v2", "v3"]}
    subs["B"] = rd(ROOT / "analysis/outputs/v5_submission_check/submission_B_hybrid.csv")
    tv = pd.read_csv(ROOT / "analysis/outputs/v5_submission_check/test_versions.csv").set_index("image_id").reindex(test.image_id)
    p2 = F.softmax(np.stack([np.load(ROOT / f"results/v2/main/fold{k}_test_logits.npy") for k in range(5)])).mean(0)
    second = np.argsort(-p2, 1)[:, 1]
    a = 0.93
    rng = np.random.default_rng(1)
    for _ in range(8):
        truth = np.where(rng.random(len(p2)) < a, subs["v2"], second)
        a += (LB["v2"] - f1_score(truth, subs["v2"], average="macro")) * 0.9
    chB = subs["B"] != subs["v2"]; ch3 = subs["v3"] != subs["v2"]
    shared = ch3 & chB & (subs["v3"] == subs["B"])
    print(f"== B vs v2: {chB.sum()} rows changed; v3 vs v2: {ch3.sum()}; v3 changes identical to B: {shared.sum()}; v3-only changes: {(ch3 & ~shared).sum()}"
          f" (types {tv.ctype.values[ch3 & ~shared].tolist().count('page')} page / {(tv.ctype.values[ch3 & ~shared] != 'page').sum()} non-page)")

    rows = []
    for frac in [0.3, 0.5, 1.0]:
        for q in np.arange(0, 1.01, 0.1):
            m, s = F.simulate(subs["v2"], subs["B"], second, q, a, frac, n_sim=200, seed=int(q * 100) + int(frac * 10))
            rows.append(dict(public_frac=frac, q=round(q, 1), mean=m, sd=s))
    r = pd.DataFrame(rows); obs = LB["B"] - LB["v2"]; r["z"] = (obs - r["mean"]) / r.sd
    print(f"\n== v2 -> B: simulated delta by q (share of the {chB.sum()} changed rows where B is right); observed {obs:+.5f}")
    print(r.pivot(index="q", columns="public_frac", values="mean").round(4).to_string())
    print("   q consistent (|z|<=1):", r[r.z.abs() <= 1].groupby("public_frac").q.agg(lambda s: f"{s.min():.1f}-{s.max():.1f}").to_dict())
    print("   q consistent (|z|<=2):", r[r.z.abs() <= 2].groupby("public_frac").q.agg(lambda s: f"{s.min():.1f}-{s.max():.1f}").to_dict())
    best = r.loc[(r["mean"] - obs).abs().groupby(r.public_frac).idxmin()]
    print("   best-fitting q per frac:", dict(zip(best.public_frac, best.q)))

    print("\n== v3 re-read: shared rows get q_B, v3-only rows get q_rest; which q_rest reproduces v3's -0.0244?")
    rows = []
    for frac in [0.3, 0.5, 1.0]:
        qB = float(best[best.public_frac == frac].q.iloc[0])
        for q_rest in np.arange(0, 0.61, 0.1):
            q_of = np.where(shared, qB, q_rest)
            m, s = simulate_multi(subs["v2"], subs["v3"], second, q_of, a, frac, seed=int(q_rest * 100))
            rows.append(dict(public_frac=frac, qB=qB, q_rest=round(q_rest, 1), mean=m, sd=s))
    r3 = pd.DataFrame(rows); r3["z"] = (LB["v3"] - LB["v2"] - r3["mean"]) / r3.sd
    print(r3.round(4).to_string(index=False))
    ex = pd.DataFrame({"ctype": tv.ctype.values, "v2": np.array(LABELS)[subs["v2"]], "v3": np.array(LABELS)[subs["v3"]]})[ch3 & ~shared]
    print("   v3-only changes by type:", ex.ctype.value_counts().to_dict())
    print("   v3-only transitions:", pd.Series([f"{u}->{v}" for u, v in zip(ex.v2, ex.v3)]).value_counts().head(10).to_dict())

    print("\n== value of one row: simulate B with k extra rows fixed (random rows among B's current errors)")
    for frac in [0.3, 0.5, 1.0]:
        gains = []
        for k in [5, 10, 20]:
            g = []
            for s_ in range(100):
                rg = np.random.default_rng(s_)
                truth = np.where(rg.random(len(p2)) < a, subs["B"], second)
                wrong = np.where(truth != subs["B"])[0]
                fix = rg.choice(wrong, k, replace=False); pb = subs["B"].copy(); pb[fix] = truth[fix]
                pub = rg.random(len(p2)) < frac
                g.append(f1_score(truth[pub], pb[pub], average="macro") - f1_score(truth[pub], subs["B"][pub], average="macro"))
            gains.append(f"{k} rows: {np.mean(g):+.4f}")
        print(f"   frac {frac}: " + " | ".join(gains))
    r.to_csv(OUT / "v2_to_B.csv", index=False); r3.to_csv(OUT / "v3_reread.csv", index=False)
    print("->", OUT)
