"""04 - Duplicates & near-duplicates: within train (label conflicts, split leakage) and train<->test.
Hashes are computed on a *normalised* view (gray, background-polarity fixed, contrast-stretched,
cropped to ink bbox) so colour tint / blur / padding in test do not hide a duplicate.
Outputs groups.csv (group_id per train image) to be used as the StratifiedGroupKFold group."""
import hashlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn

from common import load_frames, open_rgb, out_dir, pmap

OUT = out_dir("04_duplicates")


def norm_view(path):
    g = np.asarray(open_rgb(path).convert("L"), dtype=np.float32)
    lo, hi = np.percentile(g, [2, 98])
    g = np.clip((g - lo) / max(hi - lo, 1), 0, 1)
    if g.mean() < 0.5:  # dark background -> invert so ink is dark
        g = 1 - g
    ink = g < 0.5
    ys, xs = np.where(ink)
    if len(ys) > 10:
        g = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return g


def phash(g, size=8, hi=32):
    im = Image.fromarray((g * 255).astype(np.uint8)).resize((hi, hi), Image.Resampling.BILINEAR)
    d = dctn(np.asarray(im, dtype=np.float32), norm="ortho")[:size, :size]
    return (d > np.median(d[1:, 1:] if size > 1 else d)).ravel()


def dhash(g, size=8):
    im = np.asarray(Image.fromarray((g * 255).astype(np.uint8)).resize((size + 1, size), Image.Resampling.BILINEAR), dtype=np.int16)
    return (im[:, 1:] > im[:, :-1]).ravel()


def hashes(path):
    raw = hashlib.md5(open(path, "rb").read()).hexdigest()
    im = open_rgb(path)
    pix = hashlib.md5(np.asarray(im.convert("L")).tobytes() + str(im.size).encode()).hexdigest()
    g = norm_view(path)
    return raw, pix, np.concatenate([phash(g), dhash(g)]), g.shape[1] / max(g.shape[0], 1)


class UF:
    def __init__(self, n): self.p = list(range(n))
    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a
    def union(self, a, b): self.p[self.find(a)] = self.find(b)


def show_pairs(pairs, df, title, fname, k=12):
    pairs = pairs[:k]
    if not pairs:
        return
    fig, axes = plt.subplots(len(pairs), 2, figsize=(12, 1.8 * len(pairs)), squeeze=False)
    for row, (i, j, d) in zip(axes, pairs):
        for ax, idx in zip(row, (i, j)):
            r = df.iloc[idx]
            ax.imshow(open_rgb(r.path)); ax.axis("off")
            ax.set_title(f"{r.split}/{r.label if isinstance(r.label, str) else '?'} {r.image_id} ham={d}", fontsize=8)
    fig.suptitle(title); plt.tight_layout(); plt.savefig(OUT / fname, dpi=80); plt.close()


if __name__ == "__main__":
    df = load_frames()
    res = pmap(hashes, list(df.path))
    df["md5"], df["pix_md5"] = [r[0] for r in res], [r[1] for r in res]
    H = np.stack([r[2] for r in res]).astype(np.uint8)
    ar = np.array([r[3] for r in res])
    tr_idx = np.where(df.split == "train")[0]; te_idx = np.where(df.split == "test")[0]

    print("== exact duplicates (file md5 / decoded-pixel md5)")
    for key in ["md5", "pix_md5"]:
        t = df[df.split == "train"]
        grp = t.groupby(key)
        dup_groups = grp.filter(lambda x: len(x) > 1)
        conflicts = grp.label.nunique()
        print(f"  {key}: train images in dup groups={len(dup_groups)}, groups={dup_groups[key].nunique()}, "
              f"label-conflict groups={(conflicts > 1).sum()}")
        cross = set(df[df.split == "train"][key]) & set(df[df.split == "test"][key])
        print(f"  {key}: test images identical to a train image = {df[(df.split == 'test') & df[key].isin(cross)].shape[0]}")
    t = df[df.split == "train"]
    conf = t.groupby("pix_md5").filter(lambda x: x.label.nunique() > 1)
    if len(conf):
        print("  conflicting exact dups:\n", conf[["image_id", "label", "pix_md5"]].sort_values("pix_md5").to_string())

    # Hamming distance matrix on 128-bit (64 pHash + 64 dHash)
    n = len(H)
    near = []
    TH = 10  # of 128 bits; chosen from the distance histogram below (clear valley)
    all_min = np.full(n, 999)
    hist = np.zeros(129, dtype=np.int64)
    for s in range(0, n, 512):
        blk = (H[s:s + 512, None, :] != H[None, :, :]).sum(-1)
        # aspect ratio gate: near dups must have similar crop shape (log-ratio within 0.25)
        gate = np.abs(np.log(ar[s:s + 512, None] / ar[None, :])) < 0.25
        for a in range(blk.shape[0]):
            blk[a, s + a] = 999
        blk = np.where(gate, blk, 999)
        m = blk.min(1); all_min[s:s + 512] = m
        hist += np.bincount(np.clip(blk[blk < 999], 0, 128), minlength=129)
        ii, jj = np.where(blk <= TH)
        near += [(s + i, j, int(blk[i, j])) for i, j in zip(ii, jj) if s + i < j]
    print(f"\n== near duplicates (hamming<= {TH}/128, aspect-gated): {len(near)} pairs")
    print("  nearest-neighbour distance quantiles (all images):", np.percentile(all_min, [1, 5, 10, 25, 50]).round(1).tolist())

    uf = UF(n)
    for i, j, _ in near:
        uf.union(i, j)
    for key in ["pix_md5"]:
        for _, idx in df.groupby(key).groups.items():
            idx = list(idx)
            for a in idx[1:]:
                uf.union(idx[0], a)
    df["group"] = [uf.find(i) for i in range(n)]
    gsize = df.groupby("group").size()
    df["group_size"] = df.group.map(gsize)

    kinds = {"train-train": [], "train-test": [], "test-test": []}
    for i, j, d in near:
        si, sj = df.split.iat[i], df.split.iat[j]
        k = "train-train" if si == sj == "train" else "test-test" if si == sj == "test" else "train-test"
        kinds[k].append((i, j, d) if si == "train" or sj == "test" else (j, i, d))
    for k, v in kinds.items():
        print(f"  {k}: {len(v)} pairs")
    tt = kinds["train-train"]
    conflicts = [(i, j, d) for i, j, d in tt if df.label.iat[i] != df.label.iat[j]]
    print(f"  train-train pairs with DIFFERENT labels: {len(conflicts)}")

    trg = df[df.split == "train"]
    multi = trg[trg.group_size > 1]
    print(f"\n  train images inside a multi-member group: {len(multi)} ({len(multi) / len(trg) * 100:.1f}%)")
    print("  per class (% of class in dup groups):")
    print((trg.assign(d=trg.group_size > 1).groupby("label").d.mean() * 100).round(1).to_string())
    print("  largest groups:", gsize.sort_values(ascending=False).head(8).tolist())

    test_with_train_twin = df[(df.split == "test") & df.group.isin(set(trg.group[trg.group_size > 1]) | set(trg.group))]
    twin = test_with_train_twin[test_with_train_twin.group_size > 1]
    print(f"\n  test images with a near-identical TRAIN image: {len(twin)} ({len(twin) / len(te_idx) * 100:.1f}%)")

    # How much would a random 80/20 split leak? (what v1 did)
    from sklearn.model_selection import train_test_split
    tri, vai = train_test_split(trg.index, test_size=0.2, random_state=42, stratify=trg.label)
    leak = trg.loc[vai].group.isin(set(trg.loc[tri].group)).mean()
    print(f"  v1-style random 80/20 split: {leak * 100:.1f}% of validation images have a near-dup in train")

    plt.figure(figsize=(8, 4)); plt.bar(range(40), hist[:40]); plt.axvline(TH + .5, c="r")
    plt.yscale("log"); plt.title("pairwise hamming distance (aspect-gated), first 40 bins"); plt.savefig(OUT / "hamming_hist.png", dpi=90); plt.close()
    rng = np.random.default_rng(0)
    for k, v in kinds.items():
        v = sorted(v, key=lambda x: -x[2])  # hardest (largest distance) first -> checks threshold
        show_pairs(v, df, f"{k} near-dups (largest distance first)", f"pairs_{k}.png")
    show_pairs(conflicts, df, "train near-dups with CONFLICTING labels", "pairs_label_conflict.png")
    # just above threshold: are they really different?
    above = []
    for s in range(0, n, 512):
        blk = (H[s:s + 512, None, :] != H[None, :, :]).sum(-1)
        ii, jj = np.where((blk > TH) & (blk <= TH + 4))
        above += [(s + i, j, int(blk[i, j])) for i, j in zip(ii, jj) if s + i < j and abs(np.log(ar[s + i] / ar[j])) < .25]
        if len(above) > 200:
            break
    above = [above[i] for i in rng.choice(len(above), min(12, len(above)), replace=False)] if above else []
    show_pairs(above, df, f"just ABOVE threshold ({TH + 1}-{TH + 4}) - should be different", "pairs_above_threshold.png")

    df[["image_id", "split", "label", "md5", "pix_md5", "group", "group_size"]].to_csv(OUT / "groups.csv", index=False)
    print("figures + groups.csv ->", OUT)
