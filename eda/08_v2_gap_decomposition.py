"""08 - Where does the gap OOF-corrupt 0.9735 -> public LB 0.915 come from?
Content types from RAW image geometry (corruption can't change raw w/h much):
  line        w/h >= 2 and h < 250        text-line crops
  block       w/h < 2, h < 250, w >= 80    square-ish text patches (pegon/jawi blocks)
  glyph       w/h < 2, h < 250, w < 80     single glyph / number / short word
  page        h >= 250                     posters, tables, book photos, pages
Then:
  * type share train vs test
  * v2 OOF macro-F1 (clean / corrupt) per type
  * importance-weighted OOF F1 (weights = test share / train share per type): if ~0.915 the gap is COMPOSITION
  * test-side proxy per type: v2 confidence, 5-fold agreement
  * label-noise suspects: wrong on clean AND corrupt with high confidence
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from common import LABELS, ROOT, load_frames, open_rgb, out_dir

OUT = out_dir("08_v2_gap_decomposition")
RUN = ROOT / "results/v2/main"
LB_V2 = 0.91514


def content_type(w, h):
    r = w / h
    if h >= 250:
        return "page"
    if r >= 2:
        return "line"
    return "block" if w >= 80 else "glyph"


def softmax(x):
    return torch.softmax(torch.from_numpy(x), 1).numpy()


if __name__ == "__main__":
    df = load_frames()
    meta = pd.read_csv(out_dir("01_profile") / "meta.csv")
    df = df.merge(meta[["image_id", "split", "w", "h", "fmt"]], on=["image_id", "split"])
    df["ctype"] = [content_type(w, h) for w, h in zip(df.w, df.h)]
    tr = df[df.split == "train"].reset_index(drop=True)
    te = df[df.split == "test"].reset_index(drop=True)
    y = tr.label.map({l: i for i, l in enumerate(LABELS)}).values

    idx = np.concatenate([np.load(RUN / f"fold{k}_val_idx.npy") for k in range(5)])
    pc = np.zeros((len(tr), 7)); px = np.zeros((len(tr), 7))
    pc[idx] = softmax(np.concatenate([np.load(RUN / f"fold{k}_oof_clean_logits.npy") for k in range(5)]))
    px[idx] = softmax(np.concatenate([np.load(RUN / f"fold{k}_oof_corrupt_logits.npy") for k in range(5)]))
    tr["pred_c"], tr["pred_x"] = pc.argmax(1), px.argmax(1)
    tr["conf_x"] = px.max(1)
    fl = np.stack([np.load(RUN / f"fold{k}_test_logits.npy") for k in range(5)])
    tp = softmax(fl.reshape(-1, 7)).reshape(5, len(te), 7).mean(0)
    te["pred"], te["conf"] = tp.argmax(1), tp.max(1)
    te["agree"] = (fl.argmax(2) == te.pred.values).sum(0)

    pd.set_option("display.width", 220)
    share = pd.DataFrame({"train_%": tr.ctype.value_counts(normalize=True) * 100, "test_%": te.ctype.value_counts(normalize=True) * 100}).round(1)
    share["test/train"] = (share["test_%"] / share["train_%"]).round(2)
    print("== content type share")
    print(share)
    print("\n== type x class (train, %) — which classes live in which type")
    print((pd.crosstab(tr.ctype, tr.label, normalize="index") * 100).round(1))
    print("\n== type x v2-predicted class (test, %)")
    print((pd.crosstab(te.ctype, te.pred.map(dict(enumerate(LABELS))), normalize="index") * 100).round(1))

    rows = []
    for t in ["line", "block", "glyph", "page"]:
        m = (tr.ctype == t).values; mt = (te.ctype == t).values
        rows.append(dict(ctype=t, n_train=int(m.sum()), n_test=int(mt.sum()),
                         oof_acc_clean=(tr.pred_c[m] == y[m]).mean(), oof_acc_corrupt=(tr.pred_x[m] == y[m]).mean(),
                         oof_f1_corrupt=f1_score(y[m], tr.pred_x[m], average="macro", labels=range(7), zero_division=0),
                         test_conf_med=te.conf[mt].median(), test_conf_lt06=(te.conf[mt] < .6).mean(),
                         test_all5agree=(te.agree[mt] == 5).mean(), oof_conf_lt06=(tr.conf_x[m] < .6).mean()))
    per = pd.DataFrame(rows).round(3)
    print("\n== per type: OOF (train) vs test-side proxies")
    print(per.to_string(index=False))

    wmap = (share["test_%"] / share["train_%"]).to_dict()
    w = tr.ctype.map(wmap).values
    f_plain = f1_score(y, tr.pred_x, average="macro")
    f_w = f1_score(y, tr.pred_x, average="macro", sample_weight=w)
    print(f"\n== OOF-corrupt macro-F1 plain={f_plain:.4f} | reweighted to TEST type mix={f_w:.4f} | LB={LB_V2}")
    print(f"   composition explains {(f_plain - f_w) / (f_plain - LB_V2) * 100:.0f}% of the gap; the rest = corruption/source mismatch")

    # expected LB from type-wise accuracy + test-side error inflation (conf<0.6 ratio)
    infl = (per.test_conf_lt06 / per.oof_conf_lt06.clip(lower=1e-3)).round(2)
    print("   test/OOF low-confidence ratio per type:", dict(zip(per.ctype, infl)))

    # label-noise suspects
    sus = tr[(tr.pred_c != y) & (tr.pred_x != y) & (pc.max(1) > .8)].copy()
    sus["pred"] = sus.pred_c.map(dict(enumerate(LABELS)))
    print(f"\n== label-noise suspects (wrong clean+corrupt, conf>0.8): {len(sus)}")
    print(sus[["image_id", "label", "pred", "ctype", "w", "h"]].to_string(index=False))
    sus[["image_id", "label", "pred", "ctype"]].to_csv(OUT / "label_noise_suspects.csv", index=False)

    def grid(sub, fname, title, ttl):
        sub = sub.head(24)
        if not len(sub):
            return
        n = int(np.ceil(len(sub) / 4))
        fig, axes = plt.subplots(n, 4, figsize=(24, 2.8 * n), squeeze=False)
        for a in axes.ravel(): a.axis("off")
        for a, r in zip(axes.ravel(), sub.itertuples()):
            a.imshow(open_rgb(r.path)); a.set_title(ttl(r), fontsize=9)
        fig.suptitle(title); plt.tight_layout(); plt.savefig(OUT / fname, dpi=60); plt.close()
    grid(sus, "label_noise_suspects.png", "wrong on clean AND corrupt, conf>0.8", lambda r: f"true={r.label} pred={r.pred} {r.image_id}")
    err = tr[tr.pred_x != y].assign(pred=lambda d: d.pred_x.map(dict(enumerate(LABELS))))
    for t in ["line", "block", "glyph", "page"]:
        grid(err[err.ctype == t].sample(frac=1, random_state=0), f"oof_errors_{t}.png", f"OOF corrupt errors - {t}", lambda r: f"true={r.label} pred={r.pred} conf={r.conf_x:.2f}")
    for t in ["block", "glyph", "page"]:
        grid(te[te.ctype == t].sample(frac=1, random_state=0), f"test_{t}_random.png", f"test {t}", lambda r: f"v2={LABELS[r.pred]} conf={r.conf:.2f} agree={r.agree}")
    tr.drop(columns="path").to_csv(OUT / "train_types_oof.csv", index=False)
    te.drop(columns="path").to_csv(OUT / "test_types_pred.csv", index=False)
    print("->", OUT)
