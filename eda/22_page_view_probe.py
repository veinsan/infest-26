"""22 - Which INPUT VIEW lets a frozen DINOv3 tell page classes apart? (eda/21: v2 shrinks a page to ~1/4 of a grey
160x640 canvas; bali<->jawa pages need glyph detail, jawi<->pegon pages differ by genre: printed vs brown manuscript)
Views (non-line train images = page + block + glyph, and test pages):
  v2        : v2 eval canvas (grey, preprocessed, 160x640 letterbox)          <- baseline
  sq224_rgb : raw RGB, letterbox 224x224 (colour + layout, low res)
  sq448_rgb : raw RGB, letterbox 448x448 (colour + layout + 2x res)
  sq448_gray: raw grey, letterbox 448x448 (isolates colour)
  sq448_pre : v2 preprocess (grey, crop-to-ink, stretch) then letterbox 448   (isolates v2 preprocessing)
Probe: logistic regression on [CLS, mean patch] trained on non-line images of the other folds (data/folds.csv),
scored on held-out PAGES and BLOCKS. 5 folds x 3 C values; mean +- sd over folds.
Also: on test pages, how many predictions each view changes vs v2's submitted labels, and which transitions.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from common import LABELS, ROOT, load_frames, open_rgb, out_dir, pmap

sys.path.insert(0, str(ROOT / "prepro" / "outputs"))
import pipeline_v2_snapshot as V2  # noqa: E402

OUT = out_dir("22_page_view_probe")
MODEL = sys.argv[1] if len(sys.argv) > 1 else "vit_small_patch16_dinov3.lvd1689m"
TAG = MODEL.split("_")[1]


def letterbox(img, S, fill):
    img = img.copy(); img.thumbnail((S, S), Image.Resampling.LANCZOS) if max(img.size) > S else None
    if max(img.size) < S:
        s = S / max(img.size); img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.Resampling.BICUBIC)
    c = Image.new(img.mode, (S, S), fill); c.paste(img, ((S - img.width) // 2, (S - img.height) // 2)); return c


def views(path):
    rgb = open_rgb(path)
    return {"v2": np.asarray(V2.eval_views(path)[0].convert("RGB")),
            "sq224_rgb": np.asarray(letterbox(rgb, 224, (255, 255, 255))),
            "sq448_rgb": np.asarray(letterbox(rgb, 448, (255, 255, 255))),
            "sq448_gray": np.asarray(letterbox(rgb.convert("L"), 448, 255).convert("RGB")),
            "sq448_pre": np.asarray(letterbox(V2.preprocess(V2.load_rgb(path)), 448, 255).convert("RGB")),
            "sq640_gray": np.asarray(letterbox(rgb.convert("L"), 640, 255).convert("RGB"))}


_M = None


def embed(arrs):
    global _M
    if _M is None:
        _M = timm.create_model(MODEL, pretrained=True, num_classes=0, dynamic_img_size=True).eval(); torch.set_num_threads(20)
    mean, std = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1), torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
    out = []
    with torch.inference_mode():
        for s in range(0, len(arrs), 16):
            x = (torch.from_numpy(np.stack(arrs[s:s + 16])).permute(0, 3, 1, 2).float().div(255) - mean) / std
            t = _M.forward_features(x); npt = _M.num_prefix_tokens
            f = torch.cat([t[:, 0], t[:, npt:].mean(1)], 1)
            out.append(f)
    return torch.cat(out).numpy()


if __name__ == "__main__":
    df = load_frames()
    tr = pd.read_csv(out_dir("08_v2_gap_decomposition") / "train_types_oof.csv")[["image_id", "ctype"]].assign(split="train")
    te = pd.read_csv(out_dir("08_v2_gap_decomposition") / "test_types_pred.csv")[["image_id", "ctype"]].assign(split="test")
    df = df.merge(pd.concat([tr, te]), on=["image_id", "split"])
    folds = pd.read_csv(ROOT / "data/folds.csv").set_index("image_id").fold
    d = df[((df.split == "train") & (df.ctype != "line")) | ((df.split == "test") & (df.ctype == "page"))].reset_index(drop=True)
    d["fold"] = folds.reindex(d.image_id).to_numpy()
    f = OUT / f"feats_{TAG}.npz"
    F = dict(np.load(f)) if f.exists() else {}
    if "sq640_gray" not in F and F:  # round-2 view added later: embed only the missing one
        V = pmap(views, list(d.path), procs=16)
        F["sq640_gray"] = embed([v["sq640_gray"] for v in V]); np.savez(f, **F)
    if not F:
        V = pmap(views, list(d.path), procs=16)
        F = {k: embed([v[k] for v in V]) for k in V[0]}
        np.savez(f, **F)
        ex = d[(d.split == "train") & (d.ctype == "page")].sample(4, random_state=0).index
        fig, axes = plt.subplots(len(ex), len(V[0]), figsize=(22, 4 * len(ex)))
        for r, i in enumerate(ex):
            for c, k in enumerate(V[0]):
                axes[r, c].imshow(V[i][k]); axes[r, c].set_title(f"{k} {d.label[i]}"); axes[r, c].axis("off")
        plt.tight_layout(); plt.savefig(OUT / "views_example.png", dpi=55); plt.close()
    y = d.label.map({l: i for i, l in enumerate(LABELS)}).fillna(-1).astype(int).to_numpy()
    is_tr = (d.split == "train").to_numpy(); ct = d.ctype.to_numpy()
    rows = []
    for k, X in F.items():
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        for C in [0.5, 2, 8]:
            pg_acc, pg_f1, bl_acc = [], [], []
            pred = np.full(len(d), -1)
            for fo in range(5):
                trn = is_tr & (d.fold != fo).to_numpy(); val = is_tr & (d.fold == fo).to_numpy()
                m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(X[trn], y[trn])
                p = m.predict(X[val]); pred[val] = p
                v = ct[val] == "page"; pg_acc.append((p[v] == y[val][v]).mean()); bl_acc.append((p[ct[val] == "block"] == y[val][ct[val] == "block"]).mean())
            pv = is_tr & (ct == "page")
            rows.append(dict(view=k, C=C, page_acc=np.mean(pg_acc), page_sd=np.std(pg_acc), page_f1=f1_score(y[pv], pred[pv], average="macro"),
                             block_acc=np.mean(bl_acc)))
    r = pd.DataFrame(rows)
    print(f"== frozen {TAG} probe, held-out train pages (n={(is_tr & (ct == 'page')).sum()}) / blocks")
    print(r.round(3).to_string(index=False))
    best = r.loc[r.groupby("view").page_acc.idxmax()].set_index("view")

    v2s = pd.read_csv(ROOT / "results/v2/main/submission.csv").set_index("image_id").label
    tp = ~is_tr
    v2lab = v2s.reindex(d.image_id[tp]).to_numpy(dtype=object)
    print("\n== test pages: probe (fit on all non-line train) vs v2 submission")
    out = pd.DataFrame({"image_id": d.image_id[tp].values, "v2_sub": v2lab})  # probe columns are named by view
    for k, X in F.items():
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        m = LogisticRegression(C=best.loc[k, "C"], max_iter=3000, class_weight="balanced").fit(X[is_tr], y[is_tr])
        out[k] = np.array(LABELS)[m.predict(X[tp])]
        print(f"   {k:10s}: agrees with v2 {np.mean(out[k] == v2lab):.3f}")
    out.to_csv(OUT / f"test_page_preds_{TAG}.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(best.index, best.page_acc, yerr=best.page_sd); ax.set_ylabel("held-out train page acc"); ax.set_ylim(0, 1)
    plt.tight_layout(); plt.savefig(OUT / f"page_acc_by_view_{TAG}.png", dpi=70); plt.close()
    print("->", OUT)
