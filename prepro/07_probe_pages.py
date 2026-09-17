"""07 - Does page handling actually fix pages? Frozen DINOv3-S + logistic regression, group 5-fold (data/folds.csv).
Evaluated on the 199 TRAIN pages (fold-k pages, probe trained on other folds):
  e0  probe A (v2 data: clean+corrupt canvases)      page -> global letterbox only        (= v2 behaviour)
  e1  probe A                                         page -> mean logits over page_views
  e2  probe B = A + stack_view synthetic page tiles   page -> mean logits
  e3  probe C = B + page_views tiles of other-fold pages
  e4  probe C, confidence-weighted mean (w = maxprob^2)
  e5  probe C, tiles only (global view dropped)
Sanity: line accuracy of probe C vs A on fold-k clean lines must not drop.
Also: test pages, share of v2-view vs page_views predictions that change.
"""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eda"))
from common import LABELS, ROOT, out_dir, pmap  # noqa: E402
import pipeline as P  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs" / "07_probe_pages"
OUT.mkdir(parents=True, exist_ok=True)
EMB04 = Path(__file__).resolve().parent / "outputs" / "04_embedding_eval"
_MODEL = None


def embed(arrs):
    global _MODEL
    if _MODEL is None:
        _MODEL = timm.create_model("vit_small_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0, dynamic_img_size=True).eval()
        torch.set_num_threads(20)
    mean, std = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1), torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
    out = []
    with torch.inference_mode():
        for s in range(0, len(arrs), 32):
            x = torch.from_numpy(np.stack(arrs[s:s + 32])).float().div(255).unsqueeze(1).repeat(1, 3, 1, 1)
            out.append(_MODEL((x - mean) / std))
    return torch.nn.functional.normalize(torch.cat(out), dim=1).numpy()


def cached(name, fn):
    f = OUT / f"{name}.npz"
    if f.exists():
        return dict(np.load(f, allow_pickle=True))
    d = fn(); np.savez(f, **d); print("  built", name, {k: v.shape for k, v in d.items()}, flush=True)
    return d


TARGETS = tuple(int(t) for t in (sys.argv[1] if len(sys.argv) > 1 else "24,44").split(","))
TAG = "_".join(map(str, TARGETS))


def page_view_arrays(path):
    return [np.asarray(v, dtype=np.uint8) for v in P.page_views(P.preprocess(P.load_rgb(path)), targets=TARGETS)]


def build_stack(args):
    paths, seed = args
    return np.asarray(P.stack_view([P.preprocess(P.load_rgb(p)) for p in paths], random.Random(seed)), dtype=np.uint8)


def views_of(paths):
    vs = pmap(page_view_arrays, list(paths))
    owner = np.concatenate([[i] * len(v) for i, v in enumerate(vs)])
    is_global = np.concatenate([[True] + [False] * (len(v) - 1) for v in vs])
    arr = [a for v in vs for a in v]
    return dict(emb=embed(arr), owner=owner, is_global=is_global)


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def aggregate(prob, owner, n, weight=None):
    out = np.zeros((n, prob.shape[1])); ws = np.zeros((n, 1))
    w = np.ones(len(prob)) if weight is None else weight
    np.add.at(out, owner, np.log(prob + 1e-9) * w[:, None]); np.add.at(ws, owner, w[:, None])
    return out / np.maximum(ws, 1e-9)


if __name__ == "__main__":
    folds = pd.read_csv(ROOT / "data/folds.csv")
    tr = pd.read_csv(out_dir("08_v2_gap_decomposition") / "train_types_oof.csv").merge(folds[["image_id", "fold"]], on="image_id")
    order = pd.read_csv(ROOT / "data/train.csv").image_id  # 04 embeddings follow train.csv order
    tr = tr.set_index("image_id").loc[order].reset_index()
    tr["path"] = [str(ROOT / "data/images/train" / i) for i in tr.image_id]
    y = tr.label.map({l: i for i, l in enumerate(LABELS)}).values
    Ec, Ex = np.load(EMB04 / "emb_new_train_clean.npy"), np.load(EMB04 / "emb_new_train_corrupt.npy")
    pages = np.where(tr.ctype == "page")[0]
    lines = np.where(tr.ctype == "line")[0]

    PV = cached("train_page_views" + ("" if TAG == "24_44" else "_" + TAG), lambda: views_of(tr.path.values[pages]))
    rng = random.Random(0)
    stack_meta = []
    for lab_i, lab in enumerate(LABELS):
        pool = [i for i in lines if y[i] == lab_i]
        for s in range(200):
            pick = rng.sample(pool, 4); stack_meta.append((pick, lab_i))
    SV = cached("stack_views", lambda: dict(emb=embed(pmap(build_stack, [(tr.path.values[p].tolist(), i) for i, (p, _) in enumerate(stack_meta)])),
                                            y=np.array([l for _, l in stack_meta]), fold=np.array([tr.fold.values[p[0]] for p, _ in stack_meta])))

    res = {k: np.zeros((len(pages), 7)) for k in ["e0", "e1", "e2", "e3", "e4", "e5"]}
    line_acc = {"A": [], "C": []}
    for k in range(5):
        trn = tr.fold.values != k
        XA = np.r_[Ec[trn], Ex[trn]]; yA = np.r_[y[trn], y[trn]]
        sv = SV["fold"] != k
        XB = np.r_[XA, SV["emb"][sv]]; yB = np.r_[yA, SV["y"][sv]]
        pg_tr = tr.fold.values[pages] != k
        tile_tr = pg_tr[PV["owner"]] & ~PV["is_global"]
        XC = np.r_[XB, PV["emb"][tile_tr]]; yC = np.r_[yB, y[pages][PV["owner"][tile_tr]]]
        A = LogisticRegression(max_iter=3000, C=2.0).fit(XA, yA)
        B = LogisticRegression(max_iter=3000, C=2.0).fit(XB, yB)
        Cm = LogisticRegression(max_iter=3000, C=2.0).fit(XC, yC)
        te_pages = np.where(~pg_tr)[0]
        vm = np.isin(PV["owner"], te_pages)
        own = PV["owner"][vm]; E = PV["emb"][vm]; glob = PV["is_global"][vm]
        remap = {p: i for i, p in enumerate(te_pages)}; o = np.array([remap[x] for x in own])
        pA, pB, pC = A.predict_proba(E), B.predict_proba(E), Cm.predict_proba(E)
        n = len(te_pages)
        res["e0"][te_pages] = np.log(pA[glob] + 1e-9)[np.argsort(o[glob])]
        res["e1"][te_pages] = aggregate(pA, o, n)
        res["e2"][te_pages] = aggregate(pB, o, n)
        res["e3"][te_pages] = aggregate(pC, o, n)
        res["e4"][te_pages] = aggregate(pC, o, n, pC.max(1) ** 2)
        tiles_only = ~glob
        has_tiles = np.bincount(o[tiles_only], minlength=n) > 0
        e5 = aggregate(pC[tiles_only], o[tiles_only], n)
        e5[~has_tiles] = np.log(pC[glob] + 1e-9)[np.argsort(o[glob])][~has_tiles]
        res["e5"][te_pages] = e5
        fl = lines[tr.fold.values[lines] == k]
        line_acc["A"].append((A.predict(Ec[fl]) == y[fl]).mean()); line_acc["C"].append((Cm.predict(Ec[fl]) == y[fl]).mean())
        print(f"  fold {k} done", flush=True)

    yp = y[pages]
    rows = []
    names = {"e0": "A global view (v2)", "e1": "A page_views mean", "e2": "B (+stack) page_views", "e3": "C (+stack+page tiles) page_views",
             "e4": "C conf-weighted", "e5": "C tiles only"}
    for k, v in res.items():
        rows.append(dict(setting=names[k], page_acc=(v.argmax(1) == yp).mean(), page_macroF1=f1_score(yp, v.argmax(1), average="macro")))
    r = pd.DataFrame(rows).round(3)
    print(f"\n== targets={TARGETS} train pages (n=199), frozen DINOv3-S probe, group 5-fold")
    print(r.to_string(index=False))
    print(f"   line accuracy (clean, fold-k): probe A {np.mean(line_acc['A']):.4f} | probe C {np.mean(line_acc['C']):.4f}")
    r.to_csv(OUT / f"page_probe_{TAG}.csv", index=False)
