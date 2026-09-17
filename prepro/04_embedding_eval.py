"""04 - Does the new preprocessing help a real backbone? (CPU, frozen DINOv3-S, no fine-tuning)
Pipelines on the model input canvas 160x640:
  v1   : gray on white, thumbnail letterbox (notebooks/v1.ipynb)
  new  : pipeline.preprocess + to_canvas (middle tile for long lines)
For each pipeline, embeddings of: clean train, corrupted train (pipeline.corrupt, 1 draw), test.
Metrics
  adversarial AUC  train-view vs test (logreg, 5-fold)   -> lower = model input looks like test
  probe macro-F1   logreg, StratifiedGroupKFold on dup_group (05):
        clean->clean      what a random holdout reports
        clean->corrupt    trained without aug, evaluated on test-like inputs  (the v1 LB drop)
        corrupt->corrupt  trained with aug, evaluated on test-like inputs
"""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eda"))
from common import LABELS, load_frames, out_dir, pmap  # noqa: E402
import pipeline as P  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs" / "04_embedding_eval"
OUT.mkdir(parents=True, exist_ok=True)
H, W = 160, 640


def v1_canvas(img):
    g = img.convert("L")
    if g.width > W or g.height > H:
        g.thumbnail((W, H), Image.Resampling.BICUBIC)
    c = Image.new("L", (W, H), 255); c.paste(g, ((W - g.width) // 2, (H - g.height) // 2))
    return c


def new_canvas(img):
    tiles = P.to_canvas(P.preprocess(img), H, W, train=False)
    return tiles[len(tiles) // 2]


def build(args):
    path, pipe, corrupt, seed = args
    img = P.load_rgb(path)
    if corrupt:
        img = P.corrupt(img, random.Random(seed))
    return np.asarray(v1_canvas(img) if pipe == "v1" else new_canvas(img))


def embed(name, paths, pipe, corrupt):
    f = OUT / f"emb_{name}.npy"
    if f.exists():
        return np.load(f)
    X = np.stack(pmap(build, [(p, pipe, corrupt, i) for i, p in enumerate(paths)]))
    model = timm.create_model("vit_small_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0, dynamic_img_size=True).eval()
    torch.set_num_threads(20)
    mean, std = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1), torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
    out = []
    with torch.inference_mode():
        for s in range(0, len(X), 32):
            x = torch.from_numpy(X[s:s + 32]).float().div(255).unsqueeze(1).repeat(1, 3, 1, 1)
            out.append(model((x - mean) / std))
    E = torch.nn.functional.normalize(torch.cat(out), dim=1).numpy()
    np.save(f, E)
    print(f"  embedded {name}: {E.shape}", flush=True)
    return E


if __name__ == "__main__":
    df = load_frames()
    grp = pd.read_csv(out_dir("05_embed_duplicates") / "groups.csv")
    tr = df[df.split == "train"].merge(grp[["image_id", "split", "dup_group"]], on=["image_id", "split"]).reset_index(drop=True)
    te = df[df.split == "test"].reset_index(drop=True)
    y = tr.label.map({l: i for i, l in enumerate(LABELS)}).values
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(tr, y, tr.dup_group))
    rows = []
    for pipe in ["v1", "new"]:
        Ec = embed(f"{pipe}_train_clean", tr.path, pipe, False)
        Ex = embed(f"{pipe}_train_corrupt", tr.path, pipe, True)
        Et = embed(f"{pipe}_test", te.path, pipe, False)
        res = dict(pipeline=pipe)
        for nm, A in [("clean", Ec), ("corrupt", Ex)]:
            X = np.r_[A, Et]; ya = np.r_[np.zeros(len(A)), np.ones(len(Et))]
            p = cross_val_predict(LogisticRegression(max_iter=3000), X, ya, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
            res[f"adv_auc_{nm}_train_vs_test"] = roc_auc_score(ya, p)
        for trn, evl in [("clean", "clean"), ("clean", "corrupt"), ("corrupt", "corrupt"), ("corrupt", "clean")]:
            A, B = (Ec if trn == "clean" else Ex), (Ec if evl == "clean" else Ex)
            pred = np.zeros(len(y), int)
            for a, b in folds:
                pred[b] = LogisticRegression(max_iter=3000, C=2.0).fit(A[a], y[a]).predict(B[b])
            res[f"F1_{trn}->{evl}"] = f1_score(y, pred, average="macro")
        rows.append(res)
        print(pd.Series(res).to_string(), flush=True)
    r = pd.DataFrame(rows).set_index("pipeline").T.round(3)
    print("\n== summary (frozen DINOv3-S + logistic regression, group 5-fold)")
    print(r.to_string())
    r.to_csv(OUT / "summary.csv")
