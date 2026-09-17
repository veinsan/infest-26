"""05 - Near-duplicates the hashes in 04 missed (re-rendered / re-scanned / tinted copies).
Step 1  DINOv3-S embedding of the normalised view (gray, polarity fixed, ink-cropped, letterboxed).
        Raw cosine is useless here (any two text lines of a script score ~0.97, see
        pairs_by_similarity_band_RAW.png), so embeddings are mean-centred first.
Step 2  top-K embedding neighbours are only *candidates*; a pair is a duplicate when the
        pixel correlation of the normalised views (aspect-gated, +-shift) is high.
Thresholds are picked by eye from pairs_by_pixcorr_band.png.
Saves emb.npy + groups.csv (dup_group for StratifiedGroupKFold)."""
import importlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image

from common import load_frames, open_rgb, out_dir, pmap

OUT = out_dir("05_embed_duplicates")
norm_view = importlib.import_module("04_duplicates").norm_view
S, K = 224, 10
PH, PW = 32, 160  # pixel-verification canvas


def views(path):
    g = norm_view(path)
    u8 = Image.fromarray((g * 255).astype(np.uint8))
    lb = u8.copy(); lb.thumbnail((S, S), Image.Resampling.BICUBIC)
    c = Image.new("L", (S, S), 255); c.paste(lb, ((S - lb.width) // 2, (S - lb.height) // 2))
    small = np.asarray(u8.resize((PW, PH), Image.Resampling.BOX), dtype=np.float32)
    return np.asarray(c), small, g.shape[1] / max(g.shape[0], 1)


def pixcorr(a, b, shift=3):
    best = -1.0
    for dx in range(-shift, shift + 1):
        x, y = (a[:, dx:], b[:, :PW - dx]) if dx >= 0 else (a[:, :PW + dx], b[:, -dx:])
        x = x - x.mean(); y = y - y.mean()
        best = max(best, float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-6)))
    return best


class UF:
    def __init__(self, n): self.p = np.arange(n)
    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a
    def union(self, a, b): self.p[self.find(a)] = self.find(b)


def pair_grid(pairs, df, fname, bands, key):
    fig, axes = plt.subplots(len(bands), 4, figsize=(20, 2.3 * len(bands)))
    rng = np.random.default_rng(0)
    for r, (lo, hi) in enumerate(bands):
        sub = pairs[(pairs[key] >= lo) & (pairs[key] < hi)]
        pick = sub.sample(min(2, len(sub)), random_state=0) if len(sub) else sub
        for c in range(4): axes[r, c].axis("off")
        axes[r, 0].text(-.05, .5, f"{key}\n[{lo},{hi})\nn={len(sub)}", transform=axes[r, 0].transAxes, ha="right", fontsize=10)
        for k, p in enumerate(pick.itertuples()):
            for t, a in enumerate((p.i, p.j)):
                ax = axes[r, 2 * k + t]
                ax.imshow(open_rgb(df.path[a]))
                ax.set_title(f"{df.split[a]}/{df.label[a] if isinstance(df.label[a], str) else '?'} emb={p.emb:.2f} pix={p.pix:.2f}", fontsize=8)
    plt.tight_layout(); plt.savefig(OUT / fname, dpi=65); plt.close()


if __name__ == "__main__":
    df = load_frames()
    V = pmap(views, list(df.path))
    small = np.stack([v[1] for v in V]); ar = np.array([v[2] for v in V])
    emb_path = OUT / "emb.npy"
    if emb_path.exists():
        E = np.load(emb_path)
    else:
        X = np.stack([v[0] for v in V])
        model = timm.create_model("vit_small_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0).eval()
        torch.set_num_threads(20)
        mean, std = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1), torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
        out = []
        with torch.inference_mode():
            for s in range(0, len(X), 64):
                x = torch.from_numpy(X[s:s + 64]).float().div(255).unsqueeze(1).repeat(1, 3, 1, 1)
                out.append(model((x - mean) / std))
        E = torch.nn.functional.normalize(torch.cat(out), dim=1).numpy()
        np.save(emb_path, E)
    is_tr = (df.split == "train").values
    Ec = E - E[is_tr].mean(0); Ec /= np.linalg.norm(Ec, axis=1, keepdims=True)
    sim = Ec @ Ec.T; np.fill_diagonal(sim, -1)

    # candidates: top-K neighbours, aspect-gated, then pixel correlation
    rows = []
    topk = np.argsort(-sim, axis=1)[:, :K]
    for i in range(len(df)):
        for j in topk[i]:
            if i < j or i not in topk[j]:
                a, b = (i, j) if i < j else (j, i)
                if abs(np.log(ar[a] / ar[b])) < 0.3:
                    rows.append((a, b, float(sim[a, b])))
    pairs = pd.DataFrame(rows, columns=["i", "j", "emb"]).drop_duplicates(["i", "j"])
    pairs["pix"] = [pixcorr(small[a], small[b]) for a, b in zip(pairs.i, pairs.j)]
    pairs["kind"] = [("train" if is_tr[a] else "test") + "-" + ("train" if is_tr[b] else "test") for a, b in zip(pairs.i, pairs.j)]
    pairs["kind"] = pairs.kind.replace({"test-train": "train-test"})
    pairs["same_label"] = [df.label[a] == df.label[b] if is_tr[a] and is_tr[b] else np.nan for a, b in zip(pairs.i, pairs.j)]
    print(f"== candidate pairs: {len(pairs)}")
    bands = [(.95, 1.01), (.9, .95), (.85, .9), (.8, .85), (.75, .8), (.7, .75), (.6, .7)]
    print("   pixcorr band   n   train-train same-label%   kinds")
    for lo, hi in bands:
        s = pairs[(pairs.pix >= lo) & (pairs.pix < hi)]
        print(f"   [{lo:.2f},{hi:.2f})  {len(s):5d}  {s.same_label.dropna().mean() * 100:5.1f}%   {s.kind.value_counts().to_dict()}")
    pair_grid(pairs, df, "pairs_by_pixcorr_band.png", bands, "pix")

    # train_pairs_pix_050_070.png: even pix 0.5-0.7 = same glyph / same book line (99% same label).
    # For CV grouping over-grouping is safe, under-grouping leaks -> loose 0.5 on train-train.
    # Strict 0.8 only for "is this test image a copy of a train image".
    PIX_TH, GROUP_TH = 0.80, 0.50
    dup = pairs[((pairs.kind == "train-train") & (pairs.pix >= GROUP_TH)) | (pairs.pix >= PIX_TH)]
    uf = UF(len(df))
    for a, b in zip(dup.i, dup.j):
        uf.union(a, b)
    # merge exact / hash dups from 04 too
    g04 = pd.read_csv(out_dir("04_duplicates") / "groups.csv")
    for _, idx in g04.groupby("group").groups.items():
        idx = list(idx)
        for a in idx[1:]:
            uf.union(idx[0], a)
    df["dup_group"] = [uf.find(i) for i in range(len(df))]
    df["dup_group_size"] = df.dup_group.map(df.dup_group.value_counts())
    tr = df[is_tr]
    print(f"\n== duplicate groups (train-train pix>={GROUP_TH}, other pix>={PIX_TH}, + 04 hashes)")
    print(f"  train imgs in multi-member groups: {(tr.dup_group_size > 1).sum()} ({(tr.dup_group_size > 1).mean() * 100:.1f}%)")
    print("  per class %:", (tr.assign(d=tr.dup_group_size > 1).groupby("label").d.mean() * 100).round(1).to_dict())
    print("  largest group sizes:", df.dup_group.value_counts().head(10).tolist())
    mixed = tr.groupby("dup_group").label.nunique()
    print("  mixed-label groups:", int((mixed > 1).sum()))
    te_grp = set(df[~is_tr].dup_group)
    twin_te = df[~is_tr & df.dup_group.isin(set(tr.dup_group))]
    print(f"  test imgs with a train duplicate: {len(twin_te)} ({len(twin_te) / (~is_tr).sum() * 100:.1f}%)")
    from sklearn.model_selection import train_test_split
    tri, vai = train_test_split(tr.index, test_size=.2, random_state=42, stratify=tr.label)
    print(f"  v1 random 80/20: {tr.loc[vai].dup_group.isin(set(tr.loc[tri].dup_group)).mean() * 100:.1f}% of val imgs have a dup in train")

    # label transfer: does the duplicate label agree? -> tells if kNN-on-dups is safe for test
    tt = dup[dup.kind == "train-train"]
    print(f"  train-train dup pairs: {len(tt)}, same label: {tt.same_label.mean() * 100:.1f}%")
    bad = tt[tt.same_label == False]
    if len(bad):
        fig, axes = plt.subplots(min(len(bad), 10), 2, figsize=(12, 2 * min(len(bad), 10)), squeeze=False)
        for row, p in zip(axes, bad.head(10).itertuples()):
            for ax, a in zip(row, (p.i, p.j)):
                ax.imshow(open_rgb(df.path[a])); ax.axis("off"); ax.set_title(f"{df.label[a]} {df.image_id[a]} pix={p.pix:.2f}", fontsize=8)
        plt.tight_layout(); plt.savefig(OUT / "dup_label_conflicts.png", dpi=70); plt.close()
        print(bad.assign(a=df.image_id[bad.i].values, la=df.label[bad.i].values, b=df.image_id[bad.j].values, lb=df.label[bad.j].values)[["a", "la", "b", "lb", "pix"]].to_string())
    big = df.dup_group.value_counts().index[:4]
    fig, axes = plt.subplots(4, 5, figsize=(22, 9))
    for r, gid in enumerate(big):
        mem = df[df.dup_group == gid].head(5)
        for c in range(5): axes[r, c].axis("off")
        for c, m in enumerate(mem.itertuples()):
            axes[r, c].imshow(open_rgb(m.path)); axes[r, c].set_title(f"g{gid} n={m.dup_group_size} {m.split}/{m.label}", fontsize=8)
    plt.tight_layout(); plt.savefig(OUT / "largest_groups.png", dpi=65); plt.close()
    pairs.to_csv(OUT / "candidate_pairs.csv", index=False)
    df.drop(columns="path").to_csv(OUT / "groups.csv", index=False)
    print("->", OUT)
