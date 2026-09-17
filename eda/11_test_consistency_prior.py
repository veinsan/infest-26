"""11 - Two cheap, label-free checks on the test set using v2 outputs.
  a) test-test near-duplicate groups (05 dup_group): does v2 predict the same class inside a group?
     If not, averaging probabilities inside the group is free accuracy.
  b) test class prior: EM (Saerens et al. 2002) on v2 fold-averaged probabilities, using the OOF-corrupt
     prior as the source prior. Tells whether the pegon 17% is a real prior shift and whether a
     prior correction would move many predictions.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from common import LABELS, ROOT, open_rgb, out_dir

OUT = out_dir("11_test_consistency_prior")
RUN = ROOT / "results/v2/main"


def softmax(x):
    return torch.softmax(torch.from_numpy(x), -1).numpy()


if __name__ == "__main__":
    te = pd.read_csv(ROOT / "data/test.csv")
    fl = np.stack([np.load(RUN / f"fold{k}_test_logits.npy") for k in range(5)])
    prob = softmax(fl).mean(0)
    te["pred"] = np.array(LABELS)[prob.argmax(1)]; te["conf"] = prob.max(1)
    g = pd.read_csv(out_dir("05_embed_duplicates") / "groups.csv")
    te = te.merge(g[g.split == "test"][["image_id", "dup_group"]], on="image_id")
    train_groups = set(g[g.split == "train"].dup_group)
    multi = te[te.groupby("dup_group").image_id.transform("size") > 1]
    incons = multi.groupby("dup_group").pred.nunique()
    print(f"== a) test-test dup groups: {multi.dup_group.nunique()} groups / {len(multi)} images; "
          f"inconsistent predictions: {(incons > 1).sum()} groups; groups also containing train: {multi.dup_group.isin(train_groups).sum()} images")
    bad = multi[multi.dup_group.isin(incons[incons > 1].index)].sort_values("dup_group")
    print(bad[["image_id", "dup_group", "pred", "conf"]].to_string(index=False))
    if len(bad):
        n = len(bad); fig, axes = plt.subplots(int(np.ceil(n / 2)), 2, figsize=(18, 2.4 * np.ceil(n / 2)), squeeze=False)
        for a in axes.ravel(): a.axis("off")
        for a, r in zip(axes.ravel(), bad.itertuples()):
            a.imshow(open_rgb(ROOT / "data/images/test" / r.image_id)); a.set_title(f"g{r.dup_group} pred={r.pred} conf={r.conf:.2f}", fontsize=9)
        plt.tight_layout(); plt.savefig(OUT / "inconsistent_test_dups.png", dpi=60); plt.close()

    # b) EM prior estimation
    idx = np.concatenate([np.load(RUN / f"fold{k}_val_idx.npy") for k in range(5)])
    oof = softmax(np.concatenate([np.load(RUN / f"fold{k}_oof_corrupt_logits.npy") for k in range(5)]))
    src = oof.mean(0)
    pri = src.copy()
    for it in range(100):
        adj = prob * (pri / src); adj /= adj.sum(1, keepdims=True)
        new = adj.mean(0)
        if np.abs(new - pri).max() < 1e-6:
            break
        pri = new
    tab = pd.DataFrame({"train_label_%": pd.read_csv(ROOT / "data/train.csv").label.value_counts(normalize=True).reindex(LABELS) * 100,
                        "v2_argmax_%": te.pred.value_counts(normalize=True).reindex(LABELS).fillna(0) * 100,
                        "EM_test_prior_%": pri * 100}).round(1)
    print(f"\n== b) EM prior ({it + 1} iters)")
    print(tab)
    moved = (adj.argmax(1) != prob.argmax(1))
    print(f"   predictions changed by prior correction: {moved.sum()} ({moved.mean():.1%}); changes: "
          f"{pd.Series([f'{LABELS[a]}->{LABELS[b]}' for a, b in zip(prob.argmax(1)[moved], adj.argmax(1)[moved])]).value_counts().head(8).to_dict()}")
    tab.to_csv(OUT / "test_prior.csv")
    print("->", OUT)
