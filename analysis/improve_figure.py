"""Four deploy heatmaps in a row, one per Astra policy version (v1 to v4), from the improve loop.

    python -m analysis.improve_figure

Each panel is the 2-D success grid of analysis.coverage.heatmap, cropped at the gap after its colourbar,
titled with the version and its deployment success rate (pass == 1 rows, api_error rows dropped).
Writes results/improve_v1_v4.png and prints the deploy and local rates and the deploy medians of
n_decisions and t_wall per version.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from analysis.coverage import _drop_infra, _success, first_pass, heatmap
from runner.batch import RESULTS, ROOT

OUT = RESULTS / "improve_v1_v4.png"
VERSIONS = [("v1", "results/deploy.csv", "results/local.csv")] + [
    (f"v{k}", f"results/mujoco_astra-script-v{k}_deploy.csv", f"results/mujoco_astra-script-v{k}_local.csv")
    for k in (2, 3, 4)]


def _rows(path: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / path)
    df["failure_mode"] = df["failure_mode"].fillna("")
    return _drop_infra(first_pass(df))


def _crop_grid(png: Path) -> Image.Image:
    """The 2-D grid panel: cut at the widest background-only column run between 35% and 50% of the width."""
    im = np.asarray(Image.open(png).convert("RGB"))
    blank = (im == im[2, 2]).all(axis=2).all(axis=0)
    w = im.shape[1]
    runs, start = [], None
    for x in range(int(0.35 * w), int(0.5 * w)):
        if blank[x] and start is None:
            start = x
        if not blank[x] and start is not None:
            runs.append((start, x))
            start = None
    cut = max(runs, key=lambda r: r[1] - r[0])[1]
    return Image.fromarray(im[:, :cut])


def main() -> None:
    panels, rows = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for version, deploy_csv, local_csv in VERSIONS:
            deploy, local = _rows(deploy_csv), _rows(local_csv)
            rate = float(_success(deploy).mean())
            png = heatmap(pd.read_csv(ROOT / deploy_csv), path=f"{tmp}/{version}.png", policy=f"{version} {rate:.1%}")
            panels.append(_crop_grid(Path(png)))
            rows.append({"version": version, "deploy": rate, "local": float(_success(local).mean()),
                         "n_deploy": len(deploy), "n_local": len(local),
                         "median n_decisions": float(deploy["n_decisions"].median()),
                         "median t_wall": float(deploy["t_wall"].median())})
    h = panels[0].height
    strip = Image.new("RGB", (sum(p.width for p in panels), h), tuple(np.asarray(panels[0])[2, 2]))
    x = 0
    for p in panels:
        strip.paste(p, (x, 0))
        x += p.width
    strip.save(OUT)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"wrote {OUT.relative_to(ROOT)} ({strip.width}x{strip.height})")


if __name__ == "__main__":
    main()
