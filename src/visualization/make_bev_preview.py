"""Render a compact BEV preview from the public point-cloud sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("examples/points_sample.txt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/bev-preview.png"),
    )
    args = parser.parse_args()

    points = np.loadtxt(args.input, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("Expected a text file with at least X, Y, Z columns")

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    fig, ax = plt.subplots(figsize=(10, 7.5))
    scatter = ax.scatter(
        x,
        y,
        c=z,
        s=4,
        cmap="viridis",
        alpha=0.82,
        linewidths=0,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("4D radar point cloud - bird's-eye view", fontsize=15, fontweight="bold")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, alpha=0.22)
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label("Elevation / Z (m)")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
