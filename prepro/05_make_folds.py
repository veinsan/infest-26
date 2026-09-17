"""05 - Leakage-safe CV folds + duplicate handling, from eda/outputs/05_embed_duplicates/groups.csv.
  folds.csv          image_id, label, dup_group, fold (StratifiedGroupKFold 5), weight
                     weight = 1/sqrt(group size): a 42-copy cluster counts ~6.5 images, not 42
  test_train_twins.csv  test images that are near-copies of a train image (+ that train label)
Checks: no dup_group spans two folds; per-fold class share close to global."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eda"))
from common import out_dir  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"

if __name__ == "__main__":
    g = pd.read_csv(out_dir("05_embed_duplicates") / "groups.csv")
    tr = g[g.split == "train"].reset_index(drop=True)
    tr["fold"] = -1
    for k, (_, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(tr, tr.label, tr.dup_group)):
        tr.loc[va, "fold"] = k
    size = tr.groupby("dup_group").image_id.transform("size")
    tr["weight"] = 1 / np.sqrt(size)
    assert (tr.groupby("dup_group").fold.nunique() == 1).all(), "a duplicate group leaks across folds"
    assert (tr.fold >= 0).all()
    tr[["image_id", "label", "dup_group", "fold", "weight"]].to_csv(OUT / "folds.csv", index=False)

    share = pd.crosstab(tr.fold, tr.label, normalize="index")
    glob = tr.label.value_counts(normalize=True)
    print("== fold sizes:", tr.fold.value_counts().sort_index().to_dict())
    print("== max |class share - global| per fold (pp):", ((share - glob).abs().max(axis=1) * 100).round(2).to_dict())
    print(f"== weights: {len(tr)} images -> effective {tr.weight.sum():.0f}; images down-weighted: {(tr.weight < 1).sum()}")

    te = g[g.split == "test"]
    tw = te[te.dup_group.isin(set(tr.dup_group))].merge(tr[["dup_group", "label", "image_id"]].rename(columns={"image_id": "train_image_id", "label": "train_label"}), on="dup_group")
    tw = tw.groupby("image_id").agg(train_label=("train_label", lambda s: s.mode()[0]), n_labels=("train_label", "nunique"), train_image_id=("train_image_id", "first")).reset_index()
    tw.to_csv(OUT / "test_train_twins.csv", index=False)
    print(f"== test images with a train twin: {len(tw)} (label-ambiguous: {(tw.n_labels > 1).sum()})")
    print(tw.train_label.value_counts().to_dict())
