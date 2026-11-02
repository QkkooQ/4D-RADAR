"""Shared defaults for the 4D radar perception demo."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(
    os.environ.get("DRAD_DATASET_ROOT", PROJECT_ROOT / "data" / "DRadDataset")
)

CLASS_NAMES = ["Car", "Cyclist", "Truck"]
CLASS_COLORS = {
    "Car": (255, 0, 0),
    "Cyclist": (0, 0, 255),
    "Truck": (0, 255, 0),
}
