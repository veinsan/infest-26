"""20 - Silver labels, done right: measure nearest-train label precision with CORRUPTED queries.
eda/19 showed silver 'errors' where a corrupted test lontara/lampung line matched a noisy halftone JAWA train line:
train-train precision (clean queries) does not transfer to corrupted test queries.
Here: 1500 non-page train images are corrupted (pipeline.corrupt), embedded in both views (normalised view,
raw letterbox), and matched against the clean train set (own dup group excluded).
  * precision vs combined-similarity threshold AND vs the similarity margin to the best other-label neighbour
  * pick a rule with >=98% precision under corruption, rebuild silver non-page, rescore v1/v2/v3
"""
import importlib
import random
import sys

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image

from common import ROOT, load_frames, out_dir, pmap

sys.path.insert(0, str(ROOT / "prepro"))
import pipeline as P  # noqa: E402

OUT = out_dir("20_silver_corrupt_precision")
S = 224


def norm_view_img(img):
    g = np.asarray(img.convert("L"), dtype=np.float32)
    lo, hi = np.percentile(g, [2, 98]); g = np.clip((g - lo) / max(hi - lo, 1), 0, 1)
    if g.mean() < .5:
        g = 1 - g
    ys, xs = np.where(g < .5)
    if len(ys) > 10:
        g = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return Image.fromarray((g * 255).astype(np.uint8))


def boxes(img):
    nv = norm_view_img(img); nv.thumbnail((S, S), Image.Resampling.BICUBIC)
    a = Image.new("L", (S, S), 255); a.paste(nv, ((S - nv.width) // 2, (S - nv.height) // 2))
    rw = img.copy(); rw.thumbnail((S, S), Image.Resampling.BICUBIC)
    b = Image.new("RGB", (S, S), (255, 255, 255)); b.paste(rw, ((S - rw.width) // 2, (S - rw.height) // 2))
    return np.asarray(a), np.asarray(b)


def job(args):
    path, seed = args
    return boxes(P.corrupt(P.load_rgb(path), random.Random(seed)))


def embed(X, gray):
    m = timm.create_model("vit_small_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0).eval()
    torch.set_num_threads(20)
    mean, std = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1), torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
    out = []
    with torch.inference_mode():
        for s in range(0, len(X), 64):
            x = torch.from_numpy(X[s:s + 64]).float().div(255)
            x = x.unsqueeze(1).repeat(1, 3, 1, 1) if gray else x.permute(0, 3, 1, 2)
            out.append(m((x - mean) / std))
    return torch.nn.functional.normalize(torch.cat(out), dim=1).numpy()


if __name__ == "__main__":
    df = load_frames(); is_tr = (df.split == "train").to_numpy()
    tr_i, te_i = np.where(is_tr)[0], np.where(~is_tr)[0]
    ct_tr = pd.read_csv(out_dir("08_v2_gap_decomposition") / "train_types_oof.csv").set_index("image_id").ctype.reindex(df.image_id.to_numpy()[tr_i]).to_numpy()
    ct_te = pd.read_csv(out_dir("08_v2_gap_decomposition") / "test_types_pred.csv").set_index("image_id").ctype.reindex(df.image_id.to_numpy()[te_i]).to_numpy()
    lab = df.label.to_numpy(dtype=object)[tr_i]
    g = pd.read_csv(out_dir("05_embed_duplicates") / "groups.csv").set_index(["image_id", "split"]).dup_group.reindex(list(zip(df.image_id, df.split))).to_numpy()[tr_i]
    q = np.random.default_rng(0).choice(np.where(ct_tr != "page")[0], 1500, replace=False)
    f = OUT / "corrupt_query_emb.npz"
    if f.exists():
        z = np.load(f); Qn, Qr = z["n"], z["r"]
    else:
        B = pmap(job, [(df.path.to_numpy()[tr_i][i], int(i)) for i in q])
        Qn = embed(np.stack([b[0] for b in B]), True); Qr = embed(np.stack([b[1] for b in B]), False)
        np.savez(f, n=Qn, r=Qr)
    base = {}
    for name, path, Q in [("n", out_dir("05_embed_duplicates") / "emb.npy", Qn), ("r", out_dir("06_shortcuts_adversarial") / "emb_raw.npy", Qr)]:
        E = np.load(path); mu = E[is_tr].mean(0)
        Ec = E - mu; Ec /= np.linalg.norm(Ec, axis=1, keepdims=True)
        Qc = Q - mu; Qc /= np.linalg.norm(Qc, axis=1, keepdims=True)
        base[name] = (Ec, Qc)
    Sq = np.minimum(base["n"][1] @ base["n"][0][tr_i].T, base["r"][1] @ base["r"][0][tr_i].T)
    Sq[g[q][:, None] == g[None, :]] = -2          # exclude own group (incl. itself)
    nn = Sq.argmax(1); s1 = Sq.max(1)
    other = np.where(lab[None, :] != lab[nn][:, None], Sq, -2).max(1)
    margin = s1 - other
    ok = lab[nn] == lab[q]
    print(f"== corrupted train queries (n={len(q)}): overall nn precision {ok.mean():.3f}")
    rows = []
    for th in [0.6, 0.7, 0.8, 0.85, 0.9]:
        for mg in [0.0, 0.05, 0.1, 0.15]:
            m = (s1 >= th) & (margin >= mg)
            rows.append(dict(sim_th=th, margin_th=mg, coverage=m.mean(), precision=ok[m].mean() if m.sum() else np.nan))
    r = pd.DataFrame(rows)
    print(r.pivot(index="sim_th", columns="margin_th", values="precision").round(3).to_string())
    print("coverage:\n", r.pivot(index="sim_th", columns="margin_th", values="coverage").round(3).to_string())
    good = r[r.precision >= 0.98].sort_values("coverage", ascending=False)
    rule = good.iloc[0] if len(good) else r.sort_values("precision", ascending=False).iloc[0]
    print(f"   -> rule: sim>={rule.sim_th}, margin>={rule.margin_th} (precision {rule.precision:.3f}, coverage {rule.coverage:.1%} of corrupted queries)")

    St = np.minimum(base["n"][0][te_i] @ base["n"][0][tr_i].T, base["r"][0][te_i] @ base["r"][0][tr_i].T)
    nn_t = St.argmax(1); s_t = St.max(1)
    mg_t = s_t - np.where(lab[None, :] != lab[nn_t][:, None], St, -2).max(1)
    m = (ct_te != "page") & (s_t >= rule.sim_th) & (mg_t >= rule.margin_th)
    sil = pd.DataFrame({"image_id": df.image_id.to_numpy()[te_i][m], "ctype": ct_te[m], "silver": lab[nn_t][m], "sim": s_t[m], "margin": mg_t[m],
                        "twin": df.image_id.to_numpy()[tr_i][nn_t][m]})
    for v in ["v1", "v2", "v3"]:
        sil[v] = pd.read_csv(ROOT / f"results/{v}{'/main' if v != 'v1' else ''}/submission.csv").set_index("image_id").label.reindex(sil.image_id).values
    print(f"\n== silver non-page (corruption-aware rule) n={len(sil)} {sil.ctype.value_counts().to_dict()} | label mix {sil.silver.value_counts().to_dict()}")
    for v in ["v1", "v2", "v3"]:
        print(f"   {v}: acc {np.mean(sil[v] == sil.silver):.4f} errors {(sil[v] != sil.silver).sum()}")
    e = sil[sil.v2 != sil.silver]
    print("   v2 errors:", pd.Series([f"{a}->{b}" for a, b in zip(e.silver, e.v2)]).value_counts().to_dict())
    sil.to_csv(OUT / "silver_nonpage_v2rule.csv", index=False)
    import matplotlib.pyplot as plt
    from common import open_rgb
    if len(e):
        fig, axes = plt.subplots(len(e), 2, figsize=(16, 2.2 * len(e)), squeeze=False)
        for k, x in enumerate(e.itertuples()):
            axes[k, 0].imshow(open_rgb(ROOT / "data/images/test" / x.image_id)); axes[k, 0].set_title(f"TEST v1={x.v1} v2={x.v2} v3={x.v3}", fontsize=9)
            axes[k, 1].imshow(open_rgb(ROOT / "data/images/train" / x.twin)); axes[k, 1].set_title(f"twin={x.silver} sim={x.sim:.2f} margin={x.margin:.2f}", fontsize=9)
            for a in axes[k]: a.axis("off")
        plt.tight_layout(); plt.savefig(OUT / "v2_errors_silver_v2rule.png", dpi=55); plt.close()
    print("->", OUT)
