"""12 - LB forensics: what do 3 public scores say about the rows where submissions disagree? (no test labels used)
  v1 0.86289 | v2 0.91514 | v3 0.89077
v3 differs from v2 on only 59 rows, so the -0.0244 drop must come from those rows.
Monte-Carlo: sample a plausible 'truth' per row, a random public subset, compute macro-F1 of v2 and v3.
  unchanged rows: truth = shared prediction with prob a (a fitted so macro-F1(v2) ~ 0.915), else v2's 2nd choice
  changed rows  : truth = v3 pred with prob q, v2 pred with prob (1-q)*0.9, else other
Find q (share of changed rows where v3 is right) consistent with the observed delta, for public fractions 30/50/100%.
Same trick for v1 -> v2 (99 changed rows) as a sanity check of the method.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from common import LABELS, ROOT, out_dir

OUT = out_dir("12_lb_forensics")
LB = {"v1": 0.86289, "v2": 0.91514, "v3": 0.89077}
L = {l: i for i, l in enumerate(LABELS)}


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)


def simulate(pa, pb, second, q, a, frac, n_sim=400, seed=0):
    """macro-F1(pb) - macro-F1(pa) where changed rows are right for pb with prob q."""
    rng = np.random.default_rng(seed)
    n = len(pa); ch = pa != pb
    out = []
    for _ in range(n_sim):
        u = rng.random(n)
        truth = np.where(u < a, pa, second)
        uc = rng.random(n)
        t_ch = np.where(uc < q, pb, np.where(uc < q + (1 - q) * 0.9, pa, second))
        truth = np.where(ch, t_ch, truth)
        # 'second' may equal pa on changed rows: fall back to a random other class
        bad = ch & (uc >= q + (1 - q) * 0.9) & ((truth == pa) | (truth == pb))
        truth[bad] = rng.integers(0, 7, bad.sum())
        pub = rng.random(n) < frac
        out.append(f1_score(truth[pub], pb[pub], average="macro", labels=range(7)) - f1_score(truth[pub], pa[pub], average="macro", labels=range(7)))
    return np.mean(out), np.std(out)


if __name__ == "__main__":
    test = pd.read_csv(ROOT / "data/test.csv")
    types = pd.read_csv(out_dir("08_v2_gap_decomposition") / "test_types_pred.csv").set_index("image_id").ctype
    subs = {v: pd.read_csv(ROOT / f"results/{v}{'/main' if v != 'v1' else ''}/submission.csv").set_index("image_id").label.reindex(test.image_id).map(L).values for v in LB}
    p2 = softmax(np.stack([np.load(ROOT / f"results/v2/main/fold{k}_test_logits.npy") for k in range(5)])).mean(0)
    second = np.argsort(-p2, 1)[:, 1]
    ctype = types.reindex(test.image_id).values

    print("== disagreement between submissions")
    for a_, b_ in [("v1", "v2"), ("v2", "v3"), ("v1", "v3")]:
        ch = subs[a_] != subs[b_]
        print(f"  {a_} vs {b_}: {ch.sum():3d} rows ({ch.mean():.1%}) | by type {pd.Series(ctype[ch]).value_counts().to_dict()} | LB delta {LB[b_] - LB[a_]:+.4f}")
    print("  naive accuracy reading: delta*1220 = net rows gained ->",
          {f"{a_}->{b_}": round((LB[b_] - LB[a_]) * 1220) for a_, b_ in [("v1", "v2"), ("v2", "v3")]})

    # fit a: accuracy on shared rows such that simulated F1(v2) ~ LB(v2)
    rng = np.random.default_rng(1)
    a = 0.93
    for _ in range(8):
        u = rng.random(len(p2)); truth = np.where(u < a, subs["v2"], second)
        f = f1_score(truth, subs["v2"], average="macro")
        a += (LB["v2"] - f) * 0.9
    print(f"\n== fitted per-row agreement of v2 with truth on shared rows: a={a:.3f}")

    rows = []
    for (pa, pb, name) in [(subs["v2"], subs["v3"], "v2->v3"), (subs["v1"], subs["v2"], "v1->v2")]:
        obs = LB["v3"] - LB["v2"] if name == "v2->v3" else LB["v2"] - LB["v1"]
        for frac in [0.3, 0.5, 1.0]:
            for q in np.arange(0.0, 1.01, 0.1):
                m, s = simulate(pa, pb, second, q, a, frac, n_sim=150, seed=int(q * 100) + int(frac * 10))
                rows.append(dict(pair=name, public_frac=frac, q=round(q, 1), mean_delta=m, sd=s, observed=obs))
    r = pd.DataFrame(rows)
    r["z"] = (r.observed - r.mean_delta) / r.sd
    pd.set_option("display.width", 200)
    for name in ["v2->v3", "v1->v2"]:
        t = r[r.pair == name].pivot(index="q", columns="public_frac", values="mean_delta").round(4)
        print(f"\n== {name}: simulated macro-F1 delta by q (share of changed rows where the NEW submission is right); observed {r[r.pair == name].observed.iloc[0]:+.4f}")
        print(t)
        ok = r[(r.pair == name) & (r.z.abs() <= 2)]
        print("   q consistent with observed (|z|<=2):", ok.groupby("public_frac").q.agg(lambda s: f"{s.min():.1f}-{s.max():.1f}").to_dict())
    r.to_csv(OUT / "lb_forensics.csv", index=False)

    # which changed rows did v1 side with?
    d = pd.DataFrame({"image_id": test.image_id, "ctype": ctype, "v1": subs["v1"], "v2": subs["v2"], "v3": subs["v3"]})
    ch = d.v2 != d.v3
    pat = np.where(d.v1 == d.v2, "v1=v2 (v3 alone)", np.where(d.v1 == d.v3, "v1=v3 (v2 alone)", "all differ"))
    print("\n== v2!=v3 rows: which side does the independent v1 take?")
    print(pd.crosstab(pd.Series(pat[ch], name="pattern"), pd.Series(ctype[ch], name="ctype"), margins=True))
    d.assign(pattern=pat).to_csv(OUT / "patterns.csv", index=False)
    print("->", OUT)
