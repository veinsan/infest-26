"""06 - (a) Shortcut test: can the label be predicted WITHOUT reading the script?
         (metadata / low-level pixel stats only, group-aware CV)
     (b) Adversarial validation: how separable is train from test?
         on the same features and on DINOv3-S embeddings of the raw image.
Adversarial AUC ~0.5 = same distribution; ~1.0 = the holdout score says nothing about the LB."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.inspection import permutation_importance

from common import LABELS, load_frames, open_rgb, out_dir, pmap

OUT = out_dir("06_shortcuts_adversarial")
S = 224


def raw_letterbox(path):
    im = open_rgb(path)
    im.thumbnail((S, S), Image.Resampling.BICUBIC)
    c = Image.new("RGB", (S, S), (255, 255, 255))
    c.paste(im, ((S - im.width) // 2, (S - im.height) // 2))
    return np.asarray(c)


def embed(arrs):
    model = timm.create_model("vit_small_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0).eval()
    torch.set_num_threads(20)
    mean, std = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1), torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
    out = []
    with torch.inference_mode():
        for s in range(0, len(arrs), 64):
            x = torch.from_numpy(np.stack(arrs[s:s + 64])).float().div(255).permute(0, 3, 1, 2)
            out.append(model((x - mean) / std))
    return torch.nn.functional.normalize(torch.cat(out), dim=1).numpy()


def adv_auc(X, y, model):
    p = cross_val_predict(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    return roc_auc_score(y, p), p


if __name__ == "__main__":
    df = load_frames()
    px = pd.read_csv(out_dir("03_pixel_stats") / "pixel_feats.csv")
    grp = pd.read_csv(out_dir("05_embed_duplicates") / "groups.csv")
    df = df.merge(px.drop(columns=["label"]), on=["image_id", "split"]).merge(grp[["image_id", "split", "dup_group"]], on=["image_id", "split"])
    df["is_jpeg"] = (df.fmt == "JPEG").astype(int)
    df["log_w"], df["log_h"], df["log_ratio"] = np.log(df.w), np.log(df.h), np.log(df.ratio)
    df["dark_bg"] = df.dark_bg.astype(int)
    META = ["log_w", "log_h", "log_ratio", "bpp", "is_jpeg"]
    PIX = ["colorfulness", "bg_lum", "bg_sat", "ink_frac", "midtone_frac", "n_levels", "sharp", "noise", "black_border", "flat_gray_frac", "dark_bg"]
    tr = df[df.split == "train"].reset_index(drop=True)
    y = tr.label.map({l: i for i, l in enumerate(LABELS)}).values
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=42)

    print("== (a) shortcut test: macro-F1 predicting the SCRIPT from non-script features (group 5-fold)")
    print("   chance macro-F1 ~", round(1 / 7, 3))
    res = {}
    for name, cols in [("shape only (w,h,ratio)", META[:3]), ("file meta (+bpp,jpeg)", META), ("pixel stats", PIX), ("meta + pixel", META + PIX), ("n_levels only", ["n_levels"])]:
        pred = cross_val_predict(HistGradientBoostingClassifier(max_iter=300, random_state=0), tr[cols], y, cv=cv, groups=tr.dup_group)
        res[name] = f1_score(y, pred, average="macro")
        print(f"   {name:26s} macro-F1 = {res[name]:.3f}")
        if name == "meta + pixel":
            from sklearn.metrics import classification_report
            print(classification_report(y, pred, target_names=LABELS, digits=3))
    print("   per-class share of binarised images (n_levels<=16):")
    print((tr.assign(b=tr.n_levels <= 16).groupby("label").b.mean() * 100).round(1).to_string())

    print("\n== (b) adversarial validation train(0) vs test(1)")
    yadv = (df.split == "test").astype(int).values
    hgb = HistGradientBoostingClassifier(max_iter=300, random_state=0)
    for name, cols in [("file meta", META), ("pixel stats", PIX), ("meta + pixel", META + PIX)]:
        auc, _ = adv_auc(df[cols], yadv, hgb)
        print(f"   {name:14s} AUC = {auc:.3f}")
    full = hgb.fit(df[META + PIX], yadv)
    imp = permutation_importance(full, df[META + PIX], yadv, n_repeats=3, random_state=0, scoring="roc_auc")
    order = np.argsort(-imp.importances_mean)
    print("   top drivers of the shift (permutation importance, in-sample):")
    for k in order[:8]:
        print(f"     {(META + PIX)[k]:15s} {imp.importances_mean[k]:.3f}")

    emb_path = OUT / "emb_raw.npy"
    if emb_path.exists():
        E = np.load(emb_path)
    else:
        E = embed(pmap(raw_letterbox, list(df.path)))
        np.save(emb_path, E)
    En = np.load(out_dir("05_embed_duplicates") / "emb.npy")  # normalised view (gray, polarity, ink-crop)
    lr = LogisticRegression(C=1.0, max_iter=3000)
    for name, X in [("DINOv3 raw image", E), ("DINOv3 normalised view (04 norm_view)", En)]:
        auc, p = adv_auc(X, yadv, lr)
        is_tr = yadv == 0
        pc = cross_val_predict(LogisticRegression(C=1.0, max_iter=3000), X[is_tr], y, cv=cv, groups=tr.dup_group)
        print(f"   {name:40s} adversarial AUC = {auc:.3f} | linear-probe script macro-F1 (group CV) = {f1_score(y, pc, average='macro'):.3f}")
        if name.startswith("DINOv3 raw"):
            df["p_test_like"] = p
    tr = tr.merge(df[["image_id", "split", "p_test_like"]], on=["image_id", "split"])
    te = df[df.split == "test"]
    print("\n   train images that look most like test (top 5% p_test_like), class mix:")
    print("  ", tr.nlargest(int(len(tr) * .05), "p_test_like").label.value_counts().to_dict())
    print(f"   test images that look like train (p<0.5): {(te.p_test_like < .5).mean() * 100:.1f}%")

    fig, axes = plt.subplots(3, 6, figsize=(24, 8))
    for r, (title, sub) in enumerate([("most test-like TRAIN", tr.nlargest(6, "p_test_like")),
                                      ("most train-like TEST", te.nsmallest(6, "p_test_like")),
                                      ("most test-like TEST", te.nlargest(6, "p_test_like"))]):
        for ax, x in zip(axes[r], sub.itertuples()):
            ax.imshow(open_rgb(x.path)); ax.axis("off"); ax.set_title(f"{title}\n{x.label if isinstance(x.label, str) else '?'} p={x.p_test_like:.2f}", fontsize=8)
    plt.tight_layout(); plt.savefig(OUT / "adversarial_examples.png", dpi=65); plt.close()
    plt.figure(figsize=(7, 4))
    plt.hist(df.p_test_like[yadv == 0], 50, alpha=.6, label="train", density=True); plt.hist(df.p_test_like[yadv == 1], 50, alpha=.6, label="test", density=True)
    plt.legend(); plt.title("adversarial p(test) - DINOv3 raw"); plt.savefig(OUT / "adversarial_hist.png", dpi=90); plt.close()
    print("->", OUT)
