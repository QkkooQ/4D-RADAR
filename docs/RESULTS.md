# Results and available evidence

## What is available

The original local working project contains several historical experiment snapshots,
model checkpoints, and per-frame prediction folders. Those large artifacts are not
included in this public repository.

The project was developed for the 2025 AIC 4D millimeter-wave radar and monocular-camera
fusion challenge. The work covered point-cloud loading, calibration, a PointPillars-style
baseline, camera fusion experiments, prediction export, and visualization.

## What is not claimed

No official competition score or reproducible benchmark table is included in the source
snapshot. Earlier output folders used different thresholds and post-processing settings,
so their raw detection counts are not directly comparable.

Accordingly, this repository does **not** claim a specific mAP, precision, recall, or
leaderboard rank.

## Follow-up experiments

When the underlying dataset and permissions are available, add:

- LiDAR-only, camera-only, and fusion baselines;
- AP and mAP@0.5 for Car, Cyclist, and Truck;
- latency and parameter counts;
- fixed seeds and a single canonical post-processing configuration;
- a dedicated error analysis for small and distant objects.
