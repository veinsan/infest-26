"""17 - A labelled slice of the TEST distribution without touching test labels: 'silver' labels from train twins.
Candidates: test images whose nearest train image (05 normalised-view embedding, centred) is close.
Label precision of a threshold is measured on TRAIN-TRAIN pairs at the same similarity (fraction same label).
Then v1 / v2 / v3 are scored on the silver set: if the ranking matches the public LB (v2 > v3 > v1)
the silver set is a usable offline proxy for decisions the train-CV could not rank (eda/14).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from common import LABELS, ROOT, load_frames, open_rgb, out_dir

OUT = out_dir("17_silver_set")
LB = {"v1": 0.86289, "v2": 0.91514, "v3": 0.89077}

if __name__ == "__main__":
    df = load_frames()
    is_tr = (df.split == "train").values
    sets = {}
    for name, path in [("normview", out_dir("05_embed_duplicates") / "emb.npy"), ("raw", out_dir("06_shortcuts_adversarial") / "emb_raw.npy")]:
        E = np.load(path); Ec = E - E[is_tr].mean(0); Ec /= np.linalg.norm(Ec, axis=1, keepdims=True)
        sets[name] = Ec
    tr_i, te_i = np.where(is_tr)[0], np.where(~is_tr)[0]
    lab = df.label.to_numpy(dtype=object)
    grp = pd.read_csv(out_dir("05_embed_duplicates") / "groups.csv").set_index(["image_id", "split"]).dup_group
    g = grp.reindex(list(zip(df.image_id, df.split))).to_numpy()
    # combined similarity: min of the two views (both must agree the images are the same thing)
    Str = np.minimum(sets["normview"][tr_i] @ sets["normview"][tr_i].T, sets["raw"][tr_i] @ sets["raw"][tr_i].T)
    np.fill_diagonal(Str, -2)
    Ste = np.minimum(sets["normview"][te_i] @ sets["normview"][tr_i].T, sets["raw"][te_i] @ sets["raw"][tr_i].T)
    nn_tr, s_tr = Str.argmax(1), Str.max(1)
    same = lab[tr_i][nn_tr] == lab[tr_i]
    print("== label precision of 'nearest train image' by combined similarity (train-train, leave-one-out)")
    rows = []
    for th in [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        m = s_tr >= th
        mt = Ste.max(1) >= th
        rows.append(dict(threshold=th, train_pairs=int(m.sum()), label_precision=same[m].mean() if m.sum() else np.nan, test_covered=int(mt.sum())))
    r = pd.DataFrame(rows); print(r.round(3).to_string(index=False))

    subs = {v: pd.read_csv(ROOT / f"results/{v}{'/main' if v != 'v1' else ''}/submission.csv").set_index("image_id").label for v in LB}
    ids = df.image_id.to_numpy()[te_i]
    print("\n== v1/v2/v3 on silver sets (silver label = nearest train label)")
    for th in [0.8, 0.85, 0.9]:
        m = Ste.max(1) >= th
        silver = lab[tr_i][Ste.argmax(1)][m]; sid = ids[m]
        res = {v: (np.mean(subs[v].reindex(sid).to_numpy(dtype=object) == silver),
                   f1_score(silver, subs[v].reindex(sid).to_numpy(dtype=object), average="macro", labels=LABELS, zero_division=0)) for v in LB}
        print(f"   th={th}: n={m.sum():3d} | " + " | ".join(f"{v} acc {a:.3f} F1 {f:.3f}" for v, (a, f) in res.items())
              + f" | LB order v2>v3>v1: {res['v2'][0] >= res['v3'][0] >= res['v1'][0]}")
    th = 0.85
    m = Ste.max(1) >= th
    silver = pd.DataFrame({"image_id": ids[m], "silver_label": lab[tr_i][Ste.argmax(1)][m], "sim": Ste.max(1)[m],
                           "train_twin": df.image_id.to_numpy()[tr_i][Ste.argmax(1)][m]})
    for v in LB:
        silver[v] = subs[v].reindex(silver.image_id).values
    silver.to_csv(OUT / "silver_set_085.csv", index=False)
    wrong = silver[(silver.v2 != silver.silver_label) | (silver.v3 != silver.silver_label)].head(12)
    fig, axes = plt.subplots(max(1, len(wrong)), 2, figsize=(14, 2.5 * max(1, len(wrong))), squeeze=False)
    for r_, x in enumerate(wrong.itertuples()):
        axes[r_, 0].imshow(open_rgb(ROOT / "data/images/test" / x.image_id)); axes[r_, 0].set_title(f"TEST v1={x.v1} v2={x.v2} v3={x.v3}", fontsize=9)
        axes[r_, 1].imshow(open_rgb(ROOT / "data/images/train" / x.train_twin)); axes[r_, 1].set_title(f"train twin: {x.silver_label} sim={x.sim:.2f}", fontsize=9)
        for a in axes[r_]: a.axis("off")
    plt.tight_layout(); plt.savefig(OUT / "silver_disagreements.png", dpi=55); plt.close()
    print("->", OUT)

    # per-type label precision (train-train), excluding same dup_group so near-copies don't flatter it
    ct = pd.read_csv(out_dir("08_v2_gap_decomposition") / "train_types_oof.csv").set_index("image_id").ctype
    ctr = ct.reindex(df.image_id.to_numpy()[tr_i]).to_numpy()
    S2 = Str.copy(); S2[g[tr_i][:, None] == g[tr_i][None, :]] = -2
    nn2, s2 = S2.argmax(1), S2.max(1); same2 = lab[tr_i][nn2] == lab[tr_i]
    print("\n== silver label precision per type (train-train, other dup groups only)")
    for t in ["line", "block", "glyph", "page"]:
        for th_ in [0.8, 0.85, 0.9]:
            mm = (ctr == t) & (s2 >= th_)
            print(f"   {t:6s} th={th_}: pairs={mm.sum():4d} precision={same2[mm].mean() if mm.sum() else float('nan'):.3f}")
