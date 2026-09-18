"""18 - Score no-retrain options on the silver set (eda/17, ranks v1/v2/v3 like the LB).
Options: v2, v3, v2+prior(EM/BBSE, eda/15), prob-average v2/v3 at several weights, v2 on pages + v3 elsewhere.
Silver thresholds 0.8 (n~347, precision .981) and 0.85 (n~199, precision .988). Paired bootstrap vs v2.
Also per content type on the silver set (where do v2 and v3 differ?).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from common import LABELS, ROOT, out_dir

OUT = out_dir("18_silver_eval_options")


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)


if __name__ == "__main__":
    import importlib
    s17 = importlib.import_module("17_silver_set")  # noqa: F841 (documents the dependency)
    test = pd.read_csv(ROOT / "data/test.csv")
    ctype = pd.read_csv(out_dir("08_v2_gap_decomposition") / "test_types_pred.csv").set_index("image_id").ctype.reindex(test.image_id).to_numpy()
    p2 = softmax(np.stack([np.load(ROOT / f"results/v2/main/fold{k}_test_logits.npy") for k in range(5)])).mean(0)
    p3 = np.exp(np.stack([np.load(ROOT / f"results/v3/main/fold{k}_test_logprob.npy") for k in range(5)])).mean(0)
    L = np.array(LABELS)
    opts = {"v2": L[p2.argmax(1)], "v3": L[p3.argmax(1)]}
    for w in [0.25, 0.5, 0.75]:
        opts[f"avg w_v3={w}"] = L[((1 - w) * p2 + w * p3).argmax(1)]
    opts["v2 pages + v3 rest"] = np.where(ctype == "page", opts["v2"], opts["v3"])
    opts["v3 pages + v2 rest"] = np.where(ctype == "page", opts["v3"], opts["v2"])
    for name in ["v2_prior_em", "v2_prior_bbse"]:
        opts[name] = pd.read_csv(out_dir("15_prior_shift") / f"{name}.csv").set_index("image_id").label.reindex(test.image_id).to_numpy(dtype=object)
    tw = pd.read_csv(ROOT / "data/test_train_twins.csv").set_index("image_id").train_label
    m_tw = test.image_id.isin(tw.index).to_numpy()
    for k in list(opts):
        o = np.array(opts[k], dtype=object); o[m_tw] = tw.reindex(test.image_id[m_tw]).to_numpy(dtype=object); opts[k] = o

    # rebuild silver sets at two thresholds from 17's saved logic
    df_sil = {}
    from common import load_frames
    df = load_frames(); is_tr = (df.split == "train").to_numpy()
    Es = []
    for p in [out_dir("05_embed_duplicates") / "emb.npy", out_dir("06_shortcuts_adversarial") / "emb_raw.npy"]:
        E = np.load(p); Ec = E - E[is_tr].mean(0); Es.append(Ec / np.linalg.norm(Ec, axis=1, keepdims=True))
    tr_i, te_i = np.where(is_tr)[0], np.where(~is_tr)[0]
    S = np.minimum(Es[0][te_i] @ Es[0][tr_i].T, Es[1][te_i] @ Es[1][tr_i].T)
    lab = df.label.to_numpy(dtype=object)[tr_i][S.argmax(1)]; sim = S.max(1)
    assert (df.image_id.to_numpy()[te_i] == test.image_id.to_numpy()).all()

    rng = np.random.default_rng(0)
    pd.set_option("display.width", 220)
    for th in [0.8, 0.85]:
        m = sim >= th
        y = lab[m]
        base = opts["v2"][m] == y
        rows = []
        for k, o in opts.items():
            ok = o[m] == y
            d = ok.astype(int) - base.astype(int)
            bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
            rows.append(dict(option=k, acc=ok.mean(), macroF1=f1_score(y, o[m], average="macro", labels=LABELS, zero_division=0),
                             rows_diff_vs_v2=int((o[m] != opts["v2"][m]).sum()), net_vs_v2=int(d.sum()),
                             ci95=f"[{np.percentile(bs, 2.5):+.3f},{np.percentile(bs, 97.5):+.3f}]"))
        print(f"\n== silver th={th} n={m.sum()} (by type {pd.Series(ctype[m]).value_counts().to_dict()})")
        print(pd.DataFrame(rows).round(4).to_string(index=False))
    m = sim >= 0.8
    print("\n== silver th=0.8 per type: acc v2 / v3")
    for t in ["line", "block", "glyph", "page"]:
        mm = m & (ctype == t)
        if mm.sum():
            print(f"   {t:6s} n={mm.sum():3d}  v2 {np.mean(opts['v2'][mm] == lab[mm]):.3f}  v3 {np.mean(opts['v3'][mm] == lab[mm]):.3f}")
    print("->", OUT)
