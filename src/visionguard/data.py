"""GTSRB traffic-sign images: safe extraction, ROI crop, 32x32 arrays.

German Traffic Sign Recognition Benchmark (Stallkamp et al., 2012): 43 classes, 39,209 training and
12,630 test images of real signs photographed from a moving vehicle, each with an annotated sign region.
Archives come from the benchmark's public mirror (sid.erda.dk).
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 32


def _safe_members(z: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    """Reject absolute paths and '..' components (zip-slip) in downloaded archives."""
    out = []
    for m in z.infolist():
        p = Path(m.filename)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError(f"unsafe path in archive: {m.filename}")
        out.append(m)
    return out


def _crop(img_bytes: bytes, roi: tuple[int, int, int, int]) -> np.ndarray:
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.asarray(im.crop((roi[0], roi[1], roi[2] + 1, roi[3] + 1)).resize((SIZE, SIZE), Image.BILINEAR), np.uint8)


def _read_gt(text: str) -> list[tuple[str, tuple[int, int, int, int], int]]:
    rows = []
    for line in text.strip().splitlines()[1:]:
        f = line.strip().split(";")
        rows.append((f[0], (int(f[3]), int(f[4]), int(f[5]), int(f[6])), int(f[7])))
    return rows


def build_arrays(raw_dir: str | Path, out_path: str | Path) -> Path:
    """Read the three GTSRB archives (without extracting to disk) and save ``x_train, y_train, x_test, y_test``."""
    raw = Path(raw_dir)
    xs, ys = [], []
    with zipfile.ZipFile(raw / "GTSRB-Training_fixed.zip") as z:
        members = {m.filename: m for m in _safe_members(z)}
        for name in sorted(n for n in members if n.endswith(".csv")):
            folder = name.rsplit("/", 1)[0]
            for fname, roi, cls in _read_gt(z.read(members[name]).decode("utf-8")):
                xs.append(_crop(z.read(members[f"{folder}/{fname}"]), roi))
                ys.append(cls)
    xt, yt = [], []
    with zipfile.ZipFile(raw / "GTSRB_Final_Test_GT.zip") as z:
        gt = _read_gt(z.read(next(m for m in _safe_members(z) if m.filename.endswith(".csv"))).decode("utf-8"))
    with zipfile.ZipFile(raw / "GTSRB_Final_Test_Images.zip") as z:
        members = {Path(m.filename).name: m for m in _safe_members(z) if m.filename.endswith(".ppm")}
        for fname, roi, cls in gt:
            xt.append(_crop(z.read(members[fname]), roi))
            yt.append(cls)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, x_train=np.stack(xs), y_train=np.array(ys), x_test=np.stack(xt), y_test=np.array(yt))
    return out


def load(npz: str | Path) -> dict[str, np.ndarray]:
    d = np.load(npz)
    return {k: d[k] for k in ("x_train", "y_train", "x_test", "y_test")}
