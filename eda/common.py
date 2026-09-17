"""Shared paths + loaders for every EDA / preprocessing script."""
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LABELS = ["bali", "jawa", "jawi", "lampung", "lontara", "pegon", "sunda"]


def out_dir(name):
    d = ROOT / "eda" / "outputs" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_frames():
    """train + test as one frame with columns image_id, label (NaN on test), split, path."""
    tr = pd.read_csv(DATA / "train.csv").assign(split="train")
    te = pd.read_csv(DATA / "test.csv").assign(split="test")
    df = pd.concat([tr, te], ignore_index=True)
    df["path"] = [str(DATA / "images" / s / i) for s, i in zip(df.split, df.image_id)]
    return df


def open_rgb(path):
    """EXIF-aware, alpha/palette flattened onto white, always RGB."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "P", "PA"):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, "white")
            im = Image.alpha_composite(bg, im)
        return im.convert("RGB")


def open_gray_np(path, max_side=None):
    im = open_rgb(path).convert("L")
    if max_side and max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return np.asarray(im)


def pmap(fn, items, procs=20):
    with Pool(procs) as p:
        return p.map(fn, items, chunksize=16)
