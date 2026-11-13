from __future__ import annotations

import base64
import io
import zipfile

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from visionguard.api.app import create_app
from visionguard.core import (SignNet, dirty_lens, integrity_score, jpeg, lowpass, median3, noise, pgd, place_patch,
                              predict, squeeze, stickers, to_tensor, train)
from visionguard.data import _safe_members


def _toy(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two separable 'sign' classes: a red disc and a blue square on grey."""
    rng = np.random.default_rng(seed)
    x = np.full((n, 32, 32, 3), 120, np.uint8)
    y = rng.integers(0, 2, n)
    yy, xx = np.mgrid[:32, :32]
    for i in range(n):
        if y[i] == 0:
            x[i][(yy - 16) ** 2 + (xx - 16) ** 2 < 90] = (200, 30, 30)
        else:
            x[i, 8:24, 8:24] = (30, 30, 200)
    return np.clip(x + rng.normal(0, 8, x.shape), 0, 255).astype(np.uint8), y


def test_zip_slip_is_rejected() -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("../evil.txt", "x")
    with zipfile.ZipFile(buf) as z, pytest.raises(ValueError):
        _safe_members(z)


def test_transforms_keep_shape_and_range() -> None:
    rng = np.random.default_rng(0)
    x = torch.rand(4, 3, 32, 32)
    for f in (median3, jpeg, lowpass, squeeze, lambda t: noise(t, rng), lambda t: dirty_lens(t, rng), lambda t: stickers(t, rng)):
        y = f(x)
        assert y.shape == x.shape and float(y.min()) >= 0 and float(y.max()) <= 1
    flat = torch.full((1, 3, 32, 32), 0.5)
    assert torch.allclose(median3(flat), flat) and torch.allclose(lowpass(flat), flat, atol=1e-5)
    spike = flat.clone()
    spike[0, :, 10, 10] = 1.0
    assert torch.allclose(median3(spike), flat)  # isolated pixel removed
    out = place_patch(torch.zeros(1, 3, 32, 32), torch.ones(3, 7, 7), np.array([5]), np.array([6]))
    assert float(out.sum()) == 3 * 49 and float(out[0, :, 5:12, 6:13].min()) == 1.0


def test_training_pgd_and_integrity_on_toy_data() -> None:
    x, y = _toy()
    model = train(x[:300], y[:300], epochs=6, batch=32)
    xt, yt = to_tensor(x[300:]), torch.tensor(y[300:])
    assert float((predict(model, xt).argmax(1) == yt).float().mean()) > 0.95
    adv = pgd(model, xt, yt, 64 / 255, steps=20)  # toy classes are very separable: use a large budget
    assert float((adv - xt).abs().max()) <= 64 / 255 + 1e-6
    assert float((predict(model, adv).argmax(1) == yt).float().mean()) < float((predict(model, xt).argmax(1) == yt).float().mean())
    assert integrity_score(model, xt).shape == (100,)


def test_api_classify_and_dashboard() -> None:
    model = SignNet()
    cl = TestClient(create_app(bundle={"state": model.state_dict(), "integrity_threshold": 0.5}))
    buf = io.BytesIO()
    Image.fromarray(_toy(1)[0][0]).resize((48, 48)).save(buf, format="PNG")
    r = cl.post("/v1/classify", json={"image_b64": base64.b64encode(buf.getvalue()).decode()})
    j = r.json()
    assert r.status_code == 200 and 0 <= j["class"] < 43 and len(j["input_sha256"]) == 64 and set(j["images"]) == {
        "raw", "defended", "fmap_raw", "fmap_defended"}
    assert cl.post("/v1/classify", json={"image_b64": base64.b64encode(b"not an image").decode()}).status_code == 422
    assert "Traffic-sign integrity" in cl.get("/dashboard").text
