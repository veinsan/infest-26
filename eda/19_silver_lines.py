"""19 - Silver set restricted to NON-page test images (label precision >=0.99 for line/block, eda/17).
These are test images with real test corruptions whose label we know from a train twin.
  * choose the lowest threshold whose train-train precision (other dup groups) stays >= 0.99 for line/block
  * v1/v2/v3 accuracy on it (paired), v2 errors listed + grid with the train twin
  * corruption detectors (eda/09) on silver errors vs silver correct -> which test corruption breaks v2
Saves silver_nonpage.csv for later offline evaluation of new preprocessing.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import LABELS, ROOT, load_frames, open_rgb, out_dir, pmap

sys.path.insert(0, str(ROOT / "prepro"))
import pipeline as P  # noqa: E402
import importlib
detect = importlib.import_module("09_line_lowconf_forensics").detect

OUT = out_dir("19_silver_lines")


def feats(path):
    img = P.load_rgb(path)
    d = detect(img)
    g = np.asarray(img.convert("L"), dtype=np.float32)
    rgb = np.asarray(img, dtype=np.float32)
    d["colourful"] = float(np.abs(rgb[..., 0] - rgb[..., 2]).mean())
    from scipy import ndimage as ndi
    lap = np.abs(ndi.laplace(g)); d["sharp"] = float(np.percentile(lap, 99) / (g.std() + 1))
    return d


if __name__ == "__main__":
    df = load_frames(); is_tr = (df.split == "train").to_numpy()
    Es = []
    for p in [out_dir("05_embed_duplicates") / "emb.npy", out_dir("06_shortcuts_adversarial") / "emb_raw.npy"]:
        E = np.load(p); Ec = E - E[is_tr].mean(0); Es.append(Ec / np.linalg.norm(Ec, axis=1, keepdims=True))
    tr_i, te_i = np.where(is_tr)[0], np.where(~is_tr)[0]
    lab = df.label.to_numpy(dtype=object)
    g = pd.read_csv(out_dir("05_embed_duplicates") / "groups.csv").set_index(["image_id", "split"]).dup_group.reindex(list(zip(df.image_id, df.split))).to_numpy()
    ct_tr = pd.read_csv(out_dir("08_v2_gap_decomposition") / "train_types_oof.csv").set_index("image_id").ctype.reindex(df.image_id.to_numpy()[tr_i]).to_numpy()
    ct_te = pd.read_csv(out_dir("08_v2_gap_decomposition") / "test_types_pred.csv").set_index("image_id").ctype.reindex(df.image_id.to_numpy()[te_i]).to_numpy()
    Str = np.minimum(Es[0][tr_i] @ Es[0][tr_i].T, Es[1][tr_i] @ Es[1][tr_i].T)
    Str[g[tr_i][:, None] == g[tr_i][None, :]] = -2
    same = lab[tr_i][Str.argmax(1)] == lab[tr_i]; s_tr = Str.max(1)
    Ste = np.minimum(Es[0][te_i] @ Es[0][tr_i].T, Es[1][te_i] @ Es[1][tr_i].T)
    s_te = Ste.max(1)
    nonpage_tr, nonpage_te = ct_tr != "page", ct_te != "page"
    print("== non-page precision (train-train, other dup groups) vs test coverage")
    th_pick = None
    for th in [0.6, 0.65, 0.7, 0.75, 0.8]:
        m = nonpage_tr & (s_tr >= th)
        prec = same[m].mean()
        cov = int((nonpage_te & (s_te >= th)).sum())
        print(f"   th={th}: precision {prec:.4f} (pairs {m.sum()}) | test covered {cov}")
        if th_pick is None and prec >= 0.985:  # 0.99 is never reached (glyphs); ~1.5% silver noise accepted
            th_pick = th
    print(f"   -> threshold {th_pick}")
    m = nonpage_te & (s_te >= th_pick)
    sil = pd.DataFrame({"image_id": df.image_id.to_numpy()[te_i][m], "ctype": ct_te[m], "silver": lab[tr_i][Ste.argmax(1)][m],
                        "sim": s_te[m], "twin": df.image_id.to_numpy()[tr_i][Ste.argmax(1)][m]})
    for v in ["v1", "v2", "v3"]:
        sil[v] = pd.read_csv(ROOT / f"results/{v}{'/main' if v != 'v1' else ''}/submission.csv").set_index("image_id").label.reindex(sil.image_id).values
    print(f"\n== silver non-page n={len(sil)} by type {sil.ctype.value_counts().to_dict()}")
    for v in ["v1", "v2", "v3"]:
        print(f"   {v}: acc {np.mean(sil[v] == sil.silver):.4f} | errors {(sil[v] != sil.silver).sum()}")
    print("   v2 per class recall on silver:", (sil.v2 == sil.silver).groupby(sil.silver).mean().round(3).to_dict())
    print("   errors shared by v1, v2 AND v3:", int(((sil.v1 != sil.silver) & (sil.v2 != sil.silver) & (sil.v3 != sil.silver)).sum()))
    err = sil[sil.v2 != sil.silver]
    print("   v2 error transitions:", pd.Series([f"{a}->{b}" for a, b in zip(err.silver, err.v2)]).value_counts().to_dict())

    sil["path"] = [str(ROOT / "data/images/test" / i) for i in sil.image_id]
    f = pd.DataFrame(pmap(feats, list(sil.path)))
    sil = pd.concat([sil.reset_index(drop=True), f], axis=1)
    sil["v2_ok"] = sil.v2 == sil.silver
    cols = ["content_h_frac", "bar_rows", "streak_frac", "ink_frac_pp", "colourful", "sharp", "h", "w"]
    print("\n== corruption features: v2 wrong vs right (medians) + prevalence of flags")
    print(sil.groupby("v2_ok")[cols].median().round(3))
    flags = pd.DataFrame({"bar>5%": sil.bar_rows > .05, "ink_pp<3%": sil.ink_frac_pp < .03, "h<40": sil.h < 40, "blurry(sharp<p25)": sil.sharp < sil.sharp.quantile(.25)})
    print((flags.groupby(sil.v2_ok).mean() * 100).round(1).T.rename(columns={False: "v2 wrong %", True: "v2 right %"}))
    fig, axes = plt.subplots(max(1, len(err)), 2, figsize=(16, 2.2 * max(1, len(err))), squeeze=False)
    for r, x in enumerate(err.itertuples()):
        axes[r, 0].imshow(open_rgb(ROOT / "data/images/test" / x.image_id)); axes[r, 0].set_title(f"TEST {x.ctype} v2={x.v2} (v1={x.v1}, v3={x.v3})", fontsize=9)
        axes[r, 1].imshow(open_rgb(ROOT / "data/images/train" / x.twin)); axes[r, 1].set_title(f"train twin = {x.silver} sim={x.sim:.2f}", fontsize=9)
        for a in axes[r]: a.axis("off")
    plt.tight_layout(); plt.savefig(OUT / "v2_errors_on_silver.png", dpi=55); plt.close()
    sil.drop(columns="path").to_csv(OUT / "silver_nonpage.csv", index=False)
    print("->", OUT)
