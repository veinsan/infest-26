"""13 - Why did page tiles (v3) lose on the LB while improving train-page OOF?
Hypothesis: test pages come from the same web sources as train pages (same charts/posters/books);
the GLOBAL view recognises the source layout, tiles throw that away.
Using DINOv3-S embeddings of the raw letterboxed image (eda/06 emb_raw.npy), mean-centred:
  * train pages: 1-NN label accuracy vs similarity (leave-one-dup-group-out)   -> how predictive is 'source'?
  * test pages: nearest TRAIN neighbour similarity vs train-page self-similarity
  * on test pages where v2 != v3: whose prediction does the train nearest neighbour support?
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import LABELS, ROOT, load_frames, open_rgb, out_dir

OUT = out_dir("13_page_source_overlap")


if __name__ == "__main__":
    df = load_frames()
    E = np.load(out_dir("06_shortcuts_adversarial") / "emb_raw.npy")
    is_tr = (df.split == "train").values
    Ec = E - E[is_tr].mean(0); Ec /= np.linalg.norm(Ec, axis=1, keepdims=True)
    meta = pd.read_csv(out_dir("01_profile") / "meta.csv")[["image_id", "split", "h"]]
    grp = pd.read_csv(out_dir("05_embed_duplicates") / "groups.csv")[["image_id", "split", "dup_group"]]
    df = df.merge(meta, on=["image_id", "split"]).merge(grp, on=["image_id", "split"])
    page = (df.h >= 250).values
    tr_idx = np.where(is_tr)[0]; te_idx = np.where(~is_tr)[0]
    lab = df.label.values

    # train pages: NN among ALL train images (excluding own dup group)
    S = Ec @ Ec[tr_idx].T
    same_grp = df.dup_group.values[:, None] == df.dup_group.values[tr_idx][None, :]
    S[same_grp] = -2
    nn = tr_idx[S.argmax(1)]; nn_sim = S.max(1)
    trp = np.where(is_tr & page)[0]
    ok = lab[nn[trp]] == lab[trp]
    print(f"== train pages (n={len(trp)}): 1-NN (other groups) label accuracy = {ok.mean():.3f}")
    for lo, hi in [(-1, .5), (.5, .6), (.6, .7), (.7, .8), (.8, 1.01)]:
        m = (nn_sim[trp] >= lo) & (nn_sim[trp] < hi)
        if m.sum():
            print(f"   sim [{lo:.1f},{hi:.1f}) n={m.sum():3d} acc={ok[m].mean():.3f}")

    tep = np.where(~is_tr & page)[0]
    print(f"\n== nearest-train similarity: train pages median {np.median(nn_sim[trp]):.3f} | test pages median {np.median(nn_sim[tep]):.3f}")
    print(f"   test pages with nn sim >= 0.7: {(nn_sim[tep] >= .7).mean():.1%} (train pages {(nn_sim[trp] >= .7).mean():.1%})")
    print(f"   nn train neighbour of a test page is itself a PAGE: {page[nn[tep]].mean():.1%}")

    pats = pd.read_csv(out_dir("12_lb_forensics") / "patterns.csv")
    t = df.iloc[te_idx][["image_id"]].assign(nn_label=lab[nn[te_idx]], nn_sim=nn_sim[te_idx], page=page[te_idx]).merge(pats, on="image_id")
    for v in ["v1", "v2", "v3"]:
        t[v] = t[v].map(dict(enumerate(LABELS)))
    tp = t[t.page]
    print(f"\n== test pages: agreement with train-NN label: v1 {np.mean(tp.v1 == tp.nn_label):.3f} | v2 {np.mean(tp.v2 == tp.nn_label):.3f} | v3 {np.mean(tp.v3 == tp.nn_label):.3f}")
    ch = tp[tp.v2 != tp.v3]
    print(f"   on the {len(ch)} pages where v2 != v3: NN sides with v2 {np.mean(ch.nn_label == ch.v2):.2f} | v3 {np.mean(ch.nn_label == ch.v3):.2f} | neither {np.mean((ch.nn_label != ch.v2) & (ch.nn_label != ch.v3)):.2f}")
    hi = ch[ch.nn_sim >= .6]
    print(f"   same, only when NN sim>=0.6 (n={len(hi)}): v2 {np.mean(hi.nn_label == hi.v2):.2f} | v3 {np.mean(hi.nn_label == hi.v3):.2f}")
    tl = t[~t.page & (t.v2 != t.v3)]
    print(f"   non-page changed rows (n={len(tl)}): NN sides with v2 {np.mean(tl.nn_label == tl.v2):.2f} | v3 {np.mean(tl.nn_label == tl.v3):.2f}")

    sel = ch.sort_values("nn_sim", ascending=False).head(10)
    fig, axes = plt.subplots(len(sel), 2, figsize=(14, 2.8 * len(sel)), squeeze=False)
    for r, x in enumerate(sel.itertuples()):
        i = df.index[(df.image_id == x.image_id) & ~is_tr][0]
        axes[r, 0].imshow(open_rgb(df.path[i])); axes[r, 0].set_title(f"TEST v1={x.v1} v2={x.v2} v3={x.v3}", fontsize=9)
        axes[r, 1].imshow(open_rgb(df.path[nn[i]])); axes[r, 1].set_title(f"nearest TRAIN: {lab[nn[i]]} sim={x.nn_sim:.2f}", fontsize=9)
        for a in axes[r]: a.axis("off")
    plt.tight_layout(); plt.savefig(OUT / "changed_pages_vs_nn_train.png", dpi=55); plt.close()
    t.to_csv(OUT / "test_nn.csv", index=False)
    print("->", OUT)
