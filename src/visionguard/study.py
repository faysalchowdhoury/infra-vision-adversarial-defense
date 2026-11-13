"""Experiment: train on GTSRB, attack the test set, compare defences and the integrity score."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from .core import (DEFENCES, dirty_lens, integrity_score, median3, noise, pgd, place_patch, predict, stickers,  # noqa: E402
                   to_tensor, train, train_patch)
from .data import load  # noqa: E402

THREATS = ("clean", "noise", "dirty_lens", "stickers", "adv_patch", "pgd4", "pgd8", "pgd8_adaptive")


def run(npz: str, out_dir: str | Path, *, epochs: int = 8, n_eval: int = 3000, seed: int = 0,
        save_model: str | Path | None = None) -> dict[str, Any]:
    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    d = load(npz)
    perm = rng.permutation(len(d["x_train"]))
    n_val = len(perm) // 10
    va, tr = perm[:n_val], perm[n_val:]
    t0 = time.perf_counter()
    model = train(d["x_train"][tr], d["y_train"][tr], epochs=epochs, seed=seed)
    fit_s = time.perf_counter() - t0
    xv, yv = to_tensor(d["x_train"][va]), torch.tensor(d["y_train"][va])
    thr = float(torch.quantile(integrity_score(model, xv), 0.95))
    x_all, y_all = to_tensor(d["x_test"]), torch.tensor(d["y_test"])
    clean_acc_full = float((predict(model, x_all).argmax(1) == y_all).float().mean())
    idx = rng.choice(len(x_all), n_eval, replace=False)
    x, y = x_all[idx], y_all[idx]
    patch = train_patch(model, to_tensor(d["x_train"][tr[:5000]]), torch.tensor(d["y_train"][tr[:5000]]), seed=seed)
    if save_model:
        Path(save_model).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": model.state_dict(), "integrity_threshold": thr}, save_model)

    pr, pc = rng.integers(4, 21, n_eval), rng.integers(4, 21, n_eval)
    inputs = {"clean": x, "noise": noise(x, rng), "dirty_lens": dirty_lens(x, rng), "stickers": stickers(x, rng),
              "adv_patch": place_patch(x, patch, pr, pc), "pgd4": pgd(model, x, y, 4 / 255), "pgd8": pgd(model, x, y, 8 / 255),
              "pgd8_adaptive": pgd(model, x, y, 8 / 255, through=median3)}
    acc: dict[str, dict[str, float]] = {}
    flagged: dict[str, float] = {}
    for th, xi in inputs.items():
        acc[th] = {}
        for name, f in DEFENCES.items():
            if th == "pgd8_adaptive" and name != "median":
                continue  # the adaptive attack is built against the median filter
            acc[th][name] = float((predict(model, f(xi)).argmax(1) == y).float().mean())
        flagged[th] = float((integrity_score(model, xi) > thr).float().mean())
    t1 = time.perf_counter()
    predict(model, x[:1000])
    infer_ms = (time.perf_counter() - t1)  # per 1000 images -> ms per image
    summary = {"train_images": int(len(tr)), "val_images": int(n_val), "test_images": int(len(x_all)), "eval_images": n_eval,
               "fit_min": fit_s / 60, "clean_accuracy_full_test": clean_acc_full, "integrity_threshold": thr,
               "accuracy": acc, "integrity_flag_rate": flagged, "inference_ms_per_image": infer_ms}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot(model, inputs, y, out / "figures")
    write_report(summary, out / "RESULTS.md")
    return summary


def _plot(model, inputs: dict[str, torch.Tensor], y: torch.Tensor, fig_dir: Path) -> None:  # type: ignore[no-untyped-def]
    k = int(torch.nonzero(y == 14)[0]) if (y == 14).any() else 0  # a stop sign if one was sampled
    names = list(inputs)
    fig, axes = plt.subplots(3, len(names), figsize=(2.0 * len(names), 6.4))
    for j, n in enumerate(names):
        img = inputs[n][k:k + 1]
        for row, (lbl, im) in enumerate((("input", img), ("median-filtered", median3(img)))):
            p = predict(model, im)[0]
            axes[row, j].imshow(im[0].permute(1, 2, 0).numpy())
            axes[row, j].set_title(f"{n if row == 0 else lbl}\nclass {int(p.argmax())} ({float(p.max()):.2f})", fontsize=8)
            axes[row, j].axis("off")
        with torch.no_grad():
            fm = model.features[0][:3](img)[0]
        axes[2, j].imshow(fm[:16].mean(0).numpy(), cmap="magma")
        axes[2, j].set_title("conv-1 feature map", fontsize=8)
        axes[2, j].axis("off")
    fig.suptitle(f"One test sign (true class {int(y[k])}) under each threat", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "threats.png", dpi=150)
    plt.close(fig)


def write_report(s: dict[str, Any], path: Path) -> Path:
    lines = ["# Traffic-sign classifier under corruption and attack (GTSRB)", "",
             f"Trained on {s['train_images']:,} images ({s['fit_min']:.1f} min, CPU); clean accuracy on all {s['test_images']:,} test images: "
             f"**{s['clean_accuracy_full_test']:.1%}**. Threats evaluated on {s['eval_images']:,} random test images.", "",
             "## Accuracy by threat and defence", "", "| Threat | none | median | jpeg | lowpass | Flagged by integrity score |",
             "|---|---|---|---|---|---|"]
    for th, row in s["accuracy"].items():
        cells = [f"{row[d]:.1%}" if d in row else "-" for d in ("none", "median", "jpeg", "lowpass")]
        lines.append(f"| {th} | " + " | ".join(cells) + f" | {s['integrity_flag_rate'][th]:.1%} |")
    lines += ["", f"Integrity threshold {s['integrity_threshold']:.3f} (5% false alarms on clean validation images). "
              f"Inference {s['inference_ms_per_image']:.2f} ms per image on CPU.", "", "![Threats](figures/threats.png)", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
