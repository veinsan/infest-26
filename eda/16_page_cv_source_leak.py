"""16 - Did page CV lie because of SOURCE leakage (same book / same website, different photo, other fold)?
dup_group only merged near-copies. For each train page: max similarity (DINOv3-S raw view, centred) to a
SAME-CLASS train page in ANOTHER fold ('sibling'). If v3's page gain over v2 lives only on pages that have a
close sibling, the tile model memorised the source, which test pages (new sources) don't share.
Also: test pages' similarity to their nearest train page vs train pages' sibling similarity.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ROOT, load_frames, open_rgb, out_dir

OUT = out_dir("16_page_cv_source_leak")

if __name__ == "__main__":
    df = load_frames()
    E = np.load(out_dir("06_shortcuts_adversarial") / "emb_raw.npy")
    is_tr = (df.split == "train").values
    Ec = E - E[is_tr].mean(0); Ec /= np.linalg.norm(Ec, axis=1, keepdims=True)
    folds = pd.read_csv(ROOT / "data/folds.csv").set_index("image_id").fold
    pv = pd.read_csv(out_dir("14_oof_v2_vs_v3_pages") / "pages_v2_v3.csv").set_index("image_id")
    pos = {i: k for k, i in enumerate(df.image_id.where(is_tr))}
    ids = pv.index.tolist(); P = np.array([pos[i] for i in ids])
    S = Ec[P] @ Ec[P].T
    f = folds.reindex(ids).to_numpy(); lab = pv.label.to_numpy(dtype=object)
    other_fold_same = (f[:, None] != f[None, :]) & (lab[:, None] == lab[None, :])
    other_fold_diff = (f[:, None] != f[None, :]) & (lab[:, None] != lab[None, :])
    pv["sib_same"] = np.where(other_fold_same, S, -2).max(1)
    pv["sib_diff"] = np.where(other_fold_diff, S, -2).max(1)
    pv["margin"] = pv.sib_same - pv.sib_diff  # >0: closest other-fold page has the SAME label (source sibling)
    pv["gain"] = pv.v3_ok.astype(int) - pv.v2_ok.astype(int)
    bins = [-1, 0, 0.05, 0.1, 1]
    pv["margin_bin"] = pd.cut(pv.margin, bins)
    t = pv.groupby("margin_bin", observed=True).agg(n=("gain", "size"), v2_acc=("v2_ok", "mean"), v3_acc=("v3_ok", "mean"), v3_minus_v2=("gain", "mean")).round(3)
    print("== train pages by 'sibling margin' (same-label other-fold sim minus diff-label other-fold sim)")
    print(t)
    hi, lo = pv.margin > 0.05, pv.margin <= 0
    print(f"   v3 gain on pages WITH a clear same-label sibling in another fold: {pv.gain[hi].mean():+.3f} (n={hi.sum()})"
          f" | WITHOUT (margin<=0): {pv.gain[lo].mean():+.3f} (n={lo.sum()})")

    te_ids = df.image_id[~is_tr & (pd.read_csv(out_dir("01_profile") / "meta.csv").set_index(["image_id", "split"]).h.reindex(list(zip(df.image_id, df.split))).values >= 250)]
    T = Ec[te_ids.index] @ Ec[P].T
    print(f"\n== nearest train-page similarity: train pages to other-fold pages median {np.median(np.where(f[:, None] != f[None, :], S, -2).max(1)):.3f}"
          f" | test pages to train pages median {np.median(T.max(1)):.3f}")
    print(f"   share with a very close (>0.85) neighbour: train(other fold) {(np.where(f[:, None] != f[None, :], S, -2).max(1) > .85).mean():.1%} | test {(T.max(1) > .85).mean():.1%}")

    sel = pv[hi & (pv.gain > 0)].head(6)
    fig, axes = plt.subplots(max(1, len(sel)), 2, figsize=(12, 2.8 * max(1, len(sel))), squeeze=False)
    for r, (iid, x) in enumerate(sel.iterrows()):
        j = np.argmax(np.where(other_fold_same[ids.index(iid)], S[ids.index(iid)], -2))
        axes[r, 0].imshow(open_rgb(ROOT / "data/images/train" / iid)); axes[r, 0].set_title(f"{x.label}: v2 wrong, v3 right (fold {f[ids.index(iid)]})", fontsize=9)
        axes[r, 1].imshow(open_rgb(ROOT / "data/images/train" / ids[j])); axes[r, 1].set_title(f"same-label sibling fold {f[j]} sim={S[ids.index(iid), j]:.2f}", fontsize=9)
        for a in axes[r]: a.axis("off")
    plt.tight_layout(); plt.savefig(OUT / "v3_wins_with_siblings.png", dpi=55); plt.close()
    pv.to_csv(OUT / "pages_sibling.csv")
    print("->", OUT)
