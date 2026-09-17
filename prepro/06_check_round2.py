"""06 - Round-2 preprocessing check (after eda/08-10).
  a) remove_dark_bars: bars left on test lines before/after, ink lost on clean train (false positives)
  b) new vertical-shift corruption: what the model will see
  c) page_views: glyph estimate failures, #views, tiles per scale; grids for train (per class) and test
  d) stack_view: synthetic page tiles per class
"""
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eda"))
from common import LABELS, ROOT, out_dir, pmap  # noqa: E402
import pipeline as P  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs" / "06_check_round2"
OUT.mkdir(parents=True, exist_ok=True)


def bar_share(g):
    d = ndi.median_filter(np.asarray(g), 3) < 90
    rows = d.mean(1) >= .85
    return float(rows.mean())


def canvas_bar(path):
    c = P.to_canvas(P.preprocess(P.load_rgb(path)))
    return float(np.mean([bar_share(v) for v in c]))


def canvas_bar_old(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("old", Path(__file__).parent / "outputs/pipeline_v2_snapshot.py")
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    c = old.to_canvas(old.preprocess(old.load_rgb(path)))
    return float(np.mean([bar_share(v) for v in c]))


def fp_loss(path):
    g = np.asarray(P.load_rgb(path).convert("L"))
    before = (g < 128).sum()
    return 1 - (P.remove_dark_bars(g) < 128).sum() / max(before, 1)


def page_stats(path):
    pp = P.preprocess(P.load_rgb(path))
    gh = P.glyph_height(np.asarray(pp))
    v = P.page_views(pp)
    return dict(glyph_h=gh, n_views=len(v), ink_tiles=[float((np.asarray(x) < 128).mean()) for x in v[1:]])


def show(ax, im, title, gray=True):
    ax.imshow(im, cmap="gray" if gray else None, vmin=0, vmax=255, interpolation="nearest"); ax.set_title(title, fontsize=8); ax.axis("off")


if __name__ == "__main__":
    te = pd.read_csv(out_dir("09_line_lowconf_forensics") / "test_lines_forensics.csv")
    te["path"] = [str(ROOT / "data/images/test" / i) for i in te.image_id]
    tt = pd.read_csv(out_dir("08_v2_gap_decomposition") / "test_types_pred.csv"); tt["path"] = [str(ROOT / "data/images/test" / i) for i in tt.image_id]
    tr = pd.read_csv(out_dir("08_v2_gap_decomposition") / "train_types_oof.csv"); tr["path"] = [str(ROOT / "data/images/train" / i) for i in tr.image_id]

    # a) bars
    barred = te[te.bar_rows > .05]
    old = np.array(pmap(canvas_bar_old, list(barred.path))); new = np.array(pmap(canvas_bar, list(barred.path)))
    print(f"== a) test lines with a bar (n={len(barred)}): canvas rows still dark-bar  old median={np.median(old):.3f} (>10%: {(old > .1).mean():.0%})"
          f"  new median={np.median(new):.3f} (>10%: {(new > .1).mean():.0%})")
    fp = np.array(pmap(fp_loss, list(tr.path)))
    print(f"   remove_dark_bars on ALL clean train: images losing >2% ink: {(fp > .02).sum()} ({(fp > .02).mean():.1%}), max loss {fp.max():.2f}")
    print("   by content type (>2% loss):", tr.assign(loss=fp)[fp > .02].ctype.value_counts().to_dict())
    worst = tr.assign(loss=fp).nlargest(6, "loss")
    print(worst[["image_id", "label", "ctype", "loss"]].to_string(index=False))
    fig, axes = plt.subplots(6, 3, figsize=(27, 11))
    for r, x in enumerate(barred.sort_values("conf").head(6).itertuples()):
        raw = P.load_rgb(x.path)
        show(axes[r, 0], raw, f"test raw conf={x.conf:.2f}", False)
        spec_old = canvas_bar_old  # just for title symmetry
        show(axes[r, 1], P.to_canvas(P.preprocess(raw))[0], f"NEW canvas")
        show(axes[r, 2], np.asarray(P.remove_dark_bars(np.asarray(raw.convert('L')))), "after remove_dark_bars (gray)")
    plt.tight_layout(); plt.savefig(OUT / "a_test_bars_new.png", dpi=60); plt.close()
    fig, axes = plt.subplots(6, 2, figsize=(20, 11))
    for r, x in enumerate(worst.itertuples()):
        g = np.asarray(P.load_rgb(x.path).convert("L"))
        show(axes[r, 0], g, f"{x.label} {x.image_id} before"); show(axes[r, 1], P.remove_dark_bars(g), f"after (ink lost {x.loss:.0%})")
    plt.tight_layout(); plt.savefig(OUT / "a_train_bar_false_positives.png", dpi=60); plt.close()

    # b) shift corruption: force the shift branch by sampling many seeds and keeping ones with a bar
    lines = tr[tr.ctype == "line"].sample(60, random_state=1)
    fig, axes = plt.subplots(8, 3, figsize=(27, 13)); r = 0
    n_black = 0
    for i, x in enumerate(tr[tr.ctype == "line"].sample(1500, random_state=2).itertuples()):
        c = P.corrupt(P.load_rgb(x.path), random.Random(1000 + i))
        n_black += (np.asarray(P.to_canvas(P.preprocess(c), train=True, rng=random.Random(i))[0]) < 128).mean() > .6
    print(f"== b) corrupted train canvases >60% black (degenerate): {n_black}/1500")
    for i, x in enumerate(lines.itertuples()):
        if r == 8:
            break
        c = P.corrupt(P.load_rgb(x.path), random.Random(1000 + i))
        if bar_share(c.convert("L")) < .15:
            continue
        pp = P.preprocess(c)
        show(axes[r, 0], c, f"{x.label} corrupt (bar)", False); show(axes[r, 1], pp, "preprocess"); show(axes[r, 2], P.to_canvas(pp, train=True, rng=random.Random(i))[0], "train canvas"); r += 1
    plt.tight_layout(); plt.savefig(OUT / "b_shift_corruption.png", dpi=60); plt.close()

    # c) pages
    pages = pd.concat([tr[tr.ctype == "page"].assign(split="train"), tt[tt.ctype == "page"].assign(split="test")], ignore_index=True)
    ps = pd.DataFrame(pmap(page_stats, list(pages.path)))
    pages = pd.concat([pages, ps], axis=1)
    for s in ["train", "test"]:
        d = pages[pages.split == s]
        inks = np.concatenate([np.array(v) for v in d.ink_tiles if len(v)]) if any(len(v) for v in d.ink_tiles) else np.array([np.nan])
        print(f"== c) {s} pages n={len(d)}: glyph estimate NaN {d.glyph_h.isna().mean():.1%} | views median {d.n_views.median():.0f} max {d.n_views.max()} | only global view {(d.n_views == 1).mean():.1%} | tile ink median {np.median(inks):.3f}")
    for s, n_cls in [("train", True), ("test", False)]:
        d = pages[pages.split == s]
        sel = d.groupby("label").head(1).head(7) if n_cls else d.sample(7, random_state=5)
        fig, axes = plt.subplots(len(sel), 5, figsize=(40, 2.6 * len(sel)))
        for r, x in enumerate(sel.itertuples()):
            raw = P.load_rgb(x.path); v = P.page_views(P.preprocess(raw))
            show(axes[r, 0], raw, f"{s} {x.label if s == 'train' else ''} glyph={x.glyph_h:.0f}px views={len(v)}", False)
            for c in range(1, 5):
                if c - 1 < len(v):
                    show(axes[r, c], v[[0, 1, len(v) // 2, len(v) - 1][c - 1]], ["global (v2)", "tile #1", "tile mid", "tile last"][c - 1])
                else:
                    axes[r, c].axis("off")
        plt.tight_layout(); plt.savefig(OUT / f"c_page_views_{s}.png", dpi=50); plt.close()
    pages.drop(columns=["path", "ink_tiles"]).to_csv(OUT / "page_views_stats.csv", index=False)

    # d) stack views per class
    fig, axes = plt.subplots(7, 3, figsize=(30, 11))
    for r, lab in enumerate(LABELS):
        pool = tr[(tr.ctype == "line") & (tr.label == lab)].sample(12, random_state=r)
        pps = [P.preprocess(P.load_rgb(p)) for p in pool.path]
        for c in range(3):
            show(axes[r, c], P.stack_view(pps[c * 4:c * 4 + 4], random.Random(r * 10 + c)), f"stack_view {lab}")
    plt.tight_layout(); plt.savefig(OUT / "d_stack_views.png", dpi=55); plt.close()
    print("->", OUT)
