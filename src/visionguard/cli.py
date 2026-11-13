"""Command line: ``visionguard prepare`` (archives -> arrays), ``visionguard study``, ``visionguard serve``."""

from __future__ import annotations

import argparse
import json


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="visionguard")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("prepare", help="read the GTSRB archives into a 32x32 array file")
    pr.add_argument("--raw", required=True, help="directory with the three GTSRB zip archives")
    pr.add_argument("--out", default="data/gtsrb32.npz")
    st = sub.add_parser("study", help="train, attack, defend, report")
    st.add_argument("--data", default="data/gtsrb32.npz")
    st.add_argument("--out", default="results/gtsrb")
    st.add_argument("--epochs", type=int, default=8)
    st.add_argument("--seed", type=int, default=0)
    st.add_argument("--save-model", default="models/signnet.pt")
    sv = sub.add_parser("serve", help="classification API and dashboard")
    sv.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        from .data import build_arrays

        print(build_arrays(a.raw, a.out))
        return 0
    if a.cmd == "study":
        from .study import run

        s = run(a.data, a.out, epochs=a.epochs, seed=a.seed, save_model=a.save_model)
        print(json.dumps(s["accuracy"], indent=2))
        return 0
    import uvicorn

    uvicorn.run("visionguard.api.app:create_app", factory=True, host="0.0.0.0", port=a.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
