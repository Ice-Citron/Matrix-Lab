"""Real-arm validation of the Isaac SO-101 deploy heatmap.

Bins the sim csv into 4 cm (dx, dy) cells, picks the three best and three
worst cells, and walks the operator through one real trial per cell. The
answer goes to results/real_so101_validation.csv with the sim columns plus
predicted_success (the cell's sim success rate).

    python -m hardware.validate                 # prompt the operator
    python -m hardware.validate --dry-run       # answer the prompts itself
    python -m hardware.validate --grid          # write reference/grid.pdf
    python -m hardware.validate --live          # run the real arm (stub)
"""
import argparse
import csv
from pathlib import Path

import matplotlib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SIM_CSV = ROOT / "results" / "isaac_astra-live-so101_deploy.csv"
OUT_CSV = ROOT / "results" / "real_so101_validation.csv"
GRID_PDF = ROOT / "hardware" / "reference" / "grid.pdf"
COLUMNS = ["dx", "dy", "yaw", "seed", "success", "failure_mode",
           "n_decisions", "t_wall", "clip", "transcript", "pose_error_mm",
           "pass", "predicted_success"]
CELL_CM = 4
MIN_TRIALS = 3


def pick_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Three cells with success >= 0.8 and three with <= 0.2, yaw folded.

    Each row carries the cell centre in cm, the trial count, the sim
    success rate and the yaw of the sim trial nearest the centre.
    """
    df = df.copy()
    df["cx"] = (df["dx"] * 100 // CELL_CM) * CELL_CM + CELL_CM / 2
    df["cy"] = (df["dy"] * 100 // CELL_CM) * CELL_CM + CELL_CM / 2
    df["r"] = ((df["dx"] * 100 - df["cx"]) ** 2
               + (df["dy"] * 100 - df["cy"]) ** 2)
    near = df.sort_values("r").groupby(["cx", "cy"])["yaw"].first()
    cells = df.groupby(["cx", "cy"])["success"].agg(["count", "mean"])
    cells = cells[cells["count"] >= MIN_TRIALS].join(near).reset_index()
    cells = cells.sort_values("mean", ascending=False)
    good = cells[cells["mean"] >= 0.8].head(3)
    bad = cells[cells["mean"] <= 0.2].tail(3)
    if len(good) < 3 or len(bad) < 3:
        print(f"only {len(good)} cells >= 0.8 and {len(bad)} cells <= 0.2 "
              f"with {MIN_TRIALS}+ trials; taking the best/worst available")
        good, bad = cells.head(3), cells.tail(3)
    out = pd.concat([good, bad]).drop_duplicates(subset=["cx", "cy"])
    out["yaw"] = (out["yaw"] / 5).round() * 5
    return out.reset_index(drop=True)


def print_table(cells: pd.DataFrame) -> None:
    print("cell  dx_cm  dy_cm  yaw_deg  trials  sim_success")
    for i, c in cells.iterrows():
        print(f"{i:>4} {c.cx:+6.0f} {c.cy:+6.0f} {c.yaw:+8.0f} "
              f"{c['count']:>7.0f} {c['mean']:>12.2f}")


def run_live(dx_m: float, dy_m: float, yaw_deg: float) -> bool:
    """STUB: run the real SO-101 trial. See hardware/so101_astra_loop.py."""
    raise NotImplementedError(
        "wire hardware/so101_astra_loop.run() here and return the success flag")


def run_trials(cells: pd.DataFrame, out: Path, live: bool, dry: bool) -> None:
    ask = (lambda msg: "") if dry else input
    new = not out.exists()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(COLUMNS)
        for i, c in cells.iterrows():
            ask(f"Place the orange at ({c.cx:+.0f} cm, {c.cy:+.0f} cm), "
                f"yaw {c.yaw:.0f} deg. Press Enter to run.")
            if live:
                ok = run_live(c.cx / 100, c.cy / 100, c.yaw)
            elif dry:
                ok = c["mean"] >= 0.5
            else:
                ok = ask("pass or fail? ").strip().lower().startswith("p")
            w.writerow([c.cx / 100, c.cy / 100, c.yaw, i, ok,
                        "" if ok else "manual", "", "", "", "", "", 1,
                        round(c["mean"], 3)])
            print(f"  recorded {'pass' if ok else 'fail'} "
                  f"(sim predicted {c['mean']:.2f})")


def write_grid(cells: pd.DataFrame, pdf: Path) -> None:
    """A3 landscape sheet at true scale: 2 cm squares over +/-14 cm."""
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(42 / 2.54, 29.7 / 2.54))
    ax = fig.add_axes([1.0 / 42, 0.85 / 29.7, 28 / 42, 28 / 29.7])
    ax.set_xlim(-14, 14)
    ax.set_ylim(-14, 14)
    ax.set_xticks(range(-14, 15, 2))
    ax.set_yticks(range(-14, 15, 2))
    ax.tick_params(labelsize=7, length=2, pad=1)
    ax.grid(True, color="0.6", linewidth=0.5)
    ax.set_xlabel("x (cm) -> plate 29 cm this way", fontsize=8, labelpad=0)
    ax.set_ylabel("y (cm)   robot base 30 cm below", fontsize=8, labelpad=0)
    ax.plot(0, 0, "k+", markersize=16, markeredgewidth=1.2)
    ax.add_patch(plt.Circle((0, 0), 3.5, fill=False, linestyle="--"))
    ax.annotate("nominal orange", (0.4, 0.4), fontsize=8)
    for _, c in cells.iterrows():
        colour = "tab:green" if c["mean"] >= 0.5 else "tab:red"
        ax.add_patch(plt.Rectangle((c.cx - 2, c.cy - 2), 4, 4, alpha=0.3,
                                   color=colour))
        label = (f"({c.cx:+.0f}, {c.cy:+.0f})\nyaw {c.yaw:.0f}\n"
                 f"sim {c['mean']:.2f}")
        ax.text(c.cx, c.cy, label, ha="center", va="center", fontsize=7)
    fig.text(0.72, 0.9, "SO-101 orange placement grid\n"
             "A3 landscape, print at 100 % (no fit to page)\n"
             "Squares are 2 cm. Origin = nominal Orange001.\n"
             "Robot base: 30 cm below the origin (-y).\n"
             "Plate: 29 cm to the right (+x).\n"
             "Green = sim success >= 0.8, red = <= 0.2.", fontsize=10,
             va="top")
    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf)
    fig.savefig(pdf.with_suffix(".png"), dpi=60)
    print(f"wrote {pdf} and {pdf.with_suffix('.png')}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--csv", default=str(SIM_CSV))
    p.add_argument("--out", default=str(OUT_CSV))
    p.add_argument("--grid", action="store_true", help=f"write {GRID_PDF}")
    p.add_argument("--live", action="store_true", help="run the real arm")
    p.add_argument("--dry-run", action="store_true", help="auto-answer")
    args = p.parse_args()
    cells = pick_cells(pd.read_csv(args.csv))
    print_table(cells)
    if args.grid:
        write_grid(cells, GRID_PDF)
        return
    run_trials(cells, Path(args.out), args.live, args.dry_run)


if __name__ == "__main__":
    main()
