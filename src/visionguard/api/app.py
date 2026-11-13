"""Classification API with provenance, and a small dashboard.

``POST /v1/classify``  base64 image (any size; the sign crop) -> class and confidence on the raw and on the
                       median-filtered input, the integrity score and flag, the input's SHA-256 and the
                       model version, so every decision can be traced to its exact input.
``GET  /dashboard``    upload an image; see raw vs defended input, conv-1 feature maps and the score.
``GET  /health``
"""

from __future__ import annotations

import base64
import hashlib
import io
import os

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from .. import __version__
from ..core import SignNet, integrity_score, median3, predict, to_tensor


class ImageIn(BaseModel):
    image_b64: str = Field(min_length=8, max_length=4_000_000)


def _png(arr: np.ndarray, cmap: bool = False) -> str:
    if cmap:
        a = (arr - arr.min()) / (np.ptp(arr) + 1e-9)
        arr = np.stack([a, a**0.5, 1 - a], -1)
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).resize((128, 128), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def create_app(model_path: str | None = None, bundle: dict | None = None) -> FastAPI:
    if bundle is None:
        bundle = torch.load(model_path or os.environ.get("VISIONGUARD_MODEL", "models/signnet.pt"), weights_only=False)
    model = SignNet()
    model.load_state_dict(bundle["state"])
    model.eval()
    thr = float(bundle["integrity_threshold"])
    version = hashlib.sha256(b"".join(v.numpy().tobytes() for v in bundle["state"].values())).hexdigest()[:12]
    app = FastAPI(title="Traffic-sign integrity", version=__version__)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__, "model": version}

    @app.post("/v1/classify")
    def classify(req: ImageIn) -> dict[str, object]:
        try:
            raw = base64.b64decode(req.image_b64, validate=True)
            img = Image.open(io.BytesIO(raw)).convert("RGB").resize((32, 32), Image.BILINEAR)
        except (ValueError, UnidentifiedImageError) as exc:
            raise HTTPException(422, f"not a readable image: {exc}") from exc
        x = to_tensor(np.asarray(img, np.uint8)[None])
        xd = median3(x)
        p, pd_ = predict(model, x)[0], predict(model, xd)[0]
        score = float(integrity_score(model, x)[0])
        with torch.no_grad():
            fm, fmd = model.features[0][:3](x)[0, :16].mean(0).numpy(), model.features[0][:3](xd)[0, :16].mean(0).numpy()
        return {"class": int(p.argmax()), "confidence": round(float(p.max()), 3),
                "defended_class": int(pd_.argmax()), "defended_confidence": round(float(pd_.max()), 3),
                "integrity_score": round(score, 3), "flagged": score > thr, "threshold": round(thr, 3),
                "input_sha256": hashlib.sha256(raw).hexdigest(), "model": version,
                "images": {"raw": _png(x[0].permute(1, 2, 0).numpy()), "defended": _png(xd[0].permute(1, 2, 0).numpy()),
                           "fmap_raw": _png(fm, True), "fmap_defended": _png(fmd, True)}}

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard() -> str:
        return DASHBOARD

    return app


DASHBOARD = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sign integrity</title><style>
body{font-family:system-ui,sans-serif;margin:24px;background:#fafafa;color:#222}
.row{display:flex;gap:16px;flex-wrap:wrap}.card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px;text-align:center}
img{image-rendering:pixelated;width:128px;height:128px}.flag{color:#b71c1c;font-weight:600}.ok{color:#1b5e20;font-weight:600}
</style></head><body><h2>Traffic-sign integrity check</h2>
<input type="file" id="f" accept="image/*"><p id="out"></p><div class="row" id="imgs"></div>
<script>
document.getElementById('f').onchange = async e => {
  const file = e.target.files[0]; if (!file) return;
  const b64 = await new Promise(r => { const fr = new FileReader(); fr.onload = () => r(fr.result.split(',')[1]); fr.readAsDataURL(file); });
  const res = await fetch('/v1/classify', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({image_b64: b64})});
  const j = await res.json(); if (!res.ok) { document.getElementById('out').textContent = j.detail; return; }
  document.getElementById('out').innerHTML = `raw: class ${j.class} (${j.confidence}) &middot; defended: class ${j.defended_class} (${j.defended_confidence}) &middot; ` +
    `integrity score ${j.integrity_score} / ${j.threshold}: <span class="${j.flagged ? 'flag' : 'ok'}">${j.flagged ? 'FLAGGED' : 'consistent'}</span><br><small>input sha256 ${j.input_sha256} &middot; model ${j.model}</small>`;
  const t = {raw: 'raw input', defended: 'median-filtered', fmap_raw: 'conv-1 maps (raw)', fmap_defended: 'conv-1 maps (defended)'};
  document.getElementById('imgs').innerHTML = Object.entries(t).map(([k, v]) => `<div class="card"><img src="data:image/png;base64,${j.images[k]}"><div>${v}</div></div>`).join('');
};
</script></body></html>"""
