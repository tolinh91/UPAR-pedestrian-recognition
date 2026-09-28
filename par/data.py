"""UPAR data: annotation loading, encoded-image cache, dataset, transforms."""
import io
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
import torchvision.transforms as T

DATA = Path(__file__).resolve().parents[1] / "UPAR-Challenge-2027" / "data"
CACHE = Path(__file__).resolve().parents[1] / "cache"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def load_split(split):
    df = pd.read_csv(DATA / f"annotations/task1/{split}/gt.csv")
    df.columns = ["image"] + list(df.columns[1:])
    df["ds"] = df.image.str.split("/").str[0]
    return df


def attr_names():
    return list(load_split("val").columns[1:41])


def invalid_mask(df_train):
    """(n_ds, A) bool: attribute never positive in a dataset's train labels -> treated as unannotated."""
    attrs = attr_names()
    return {ds: (g[attrs].values.sum(0) == 0) for ds, g in df_train.groupby("ds")}


def image_cache(split):
    """All encoded image bytes of a split in one uint8 array + offsets (fork/COW friendly)."""
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{split}_bytes.npz"
    if not f.exists():
        df = load_split(split)
        blobs = [(DATA / p).read_bytes() for p in df.image]
        off = np.cumsum([0] + [len(b) for b in blobs]).astype(np.int64)
        np.savez(f, buf=np.frombuffer(b"".join(blobs), np.uint8), off=off)
    z = np.load(f)
    return z["buf"], z["off"]


class RandomLowRes:
    """Simulate low-resolution / far-distance crops: downscale then upscale."""

    def __init__(self, p=0.3, min_scale=0.3):
        self.p, self.min_scale = p, min_scale

    def __call__(self, img):
        if random.random() > self.p:
            return img
        s = random.uniform(self.min_scale, 0.8)
        w, h = img.size
        small = img.resize((max(8, int(w * s)), max(16, int(h * s))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)


class RandomBlur:
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.5))) if random.random() < self.p else img


def build_transforms(size, train):
    h, w = size
    if not train:
        return T.Compose([T.Resize((h, w)), T.ToTensor(), T.Normalize(MEAN, STD)])
    return T.Compose([
        RandomLowRes(),
        T.Resize((h, w)),
        T.RandomHorizontalFlip(),
        T.Pad(10, padding_mode="edge"),
        T.RandomCrop((h, w)),
        # no hue jitter: colour is a label
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        RandomBlur(),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
        T.RandomErasing(p=0.3, scale=(0.02, 0.15), value="random"),
    ])


class UPARDataset(Dataset):
    def __init__(self, split, transform, invalid=None, rows=None):
        df = load_split(split)
        self.buf, self.off = image_cache(split)
        self.idx = np.arange(len(df)) if rows is None else np.asarray(rows)
        attrs = attr_names()
        self.labels = df[attrs].values.astype(np.float32)
        self.ds = df.ds.values
        self.ds_names = sorted(set(self.ds))
        valid = np.ones_like(self.labels)
        if invalid:
            for ds, m in invalid.items():
                valid[self.ds == ds] = ~m
        self.valid = valid.astype(np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        img = Image.open(io.BytesIO(self.buf[self.off[j]:self.off[j + 1]].tobytes())).convert("RGB")
        return self.transform(img), torch.from_numpy(self.labels[j]), torch.from_numpy(self.valid[j]), j

    def sample_weights(self, alpha=0.5):
        """Per-sample weights so dataset d is drawn ~ N_d^alpha (alpha=0: equal per dataset)."""
        ds = self.ds[self.idx]
        names, counts = np.unique(ds, return_counts=True)
        per = {n: c ** alpha / c for n, c in zip(names, counts)}
        return np.array([per[d] for d in ds])
