"""Turn each DIII-D parquet shot into a small npz: inputs on efit_times, plus targets if present.

    python prep.py diii_d_train diii_d_public_test

Features per EFIT time (built with the starter kit's own loaders, Ip time fix included):
    mag  (T, 21)   coil currents + Ip interpolated onto efit_times
    th   (T, 128)  Thomson core Te, core ne (padded to 54 channels), edge Te, edge ne (10 each),
                   linear in time, 0 outside
Targets (train only): psi (T, 65, 65) float32, q95 (T,), betaN (T,).
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))
from experiments import (interpolate_magnetics_to_efit, interpolate_thomson_to_efit,  # noqa: E402
                         load_shot_from_hf_row)

HF = ROOT / "data" / "hf_dataset" / "data"
OUT = ROOT / "data" / "prep"


def one(args):
    config, f = args
    out = OUT / config / (f.stem + ".npz")
    if out.exists():
        return
    row = pd.read_parquet(f).iloc[0]
    has_t = "efit_psirz" in row.index and row["efit_psirz"] is not None
    if not has_t:  # test rows: give the loader an empty ψ so it does not trip
        row = row.copy(); row["efit_psirz"] = np.zeros((len(row["efit_times"]), 65, 65), np.float32)
    shot = load_shot_from_hf_row(row)
    mag = interpolate_magnetics_to_efit(shot)
    th = interpolate_thomson_to_efit(shot)
    T = len(shot["efit_times"])
    # core has 44 or 54 channels depending on the shot, edge 10: pad core to 54 -> 128 columns
    nc = shot["thomson_core"]["Te"].shape[1] if "thomson_core" in shot else 0
    blocks = np.zeros((T, 128), np.float32)
    if th is not None and th.shape[1] == 2 * nc + 20:
        blocks[:, 0:nc] = th[:, 0:nc]; blocks[:, 54:54 + nc] = th[:, nc:2 * nc]
        blocks[:, 108:128] = th[:, 2 * nc:]
    th = blocks
    d = {"mag": mag.astype(np.float32), "th": th.astype(np.float32), "t": shot["efit_times"]}
    if has_t:
        d["psi"] = shot["efit_psirz"].astype(np.float32)
        d["q95"] = np.asarray(row["efit_q95"], np.float64)
        d["betaN"] = np.asarray(row["efit_beta_n"], np.float64)
    np.savez(out, **d)


def main():
    jobs = []
    for config in sys.argv[1:]:
        (OUT / config).mkdir(parents=True, exist_ok=True)
        jobs += [(config, f) for f in sorted((HF / config).glob("*.parquet"))]
    with Pool(10) as p:
        for i, _ in enumerate(p.imap_unordered(one, jobs, chunksize=4)):
            if i % 200 == 0:
                print(i, "/", len(jobs), flush=True)


if __name__ == "__main__":
    main()
