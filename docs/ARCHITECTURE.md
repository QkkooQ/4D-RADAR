# Architecture and data flow

## Data assumptions

The cleaned repository expects a DRad/KITTI-style dataset with this structure:

```text
DRadDataset/
├── training/
│   ├── calib/
│   ├── image_2/
│   ├── label_2/
│   └── velodyne/
├── testing/
│   ├── calib/
│   ├── image_2/
│   └── velodyne/
└── ImageSets/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

Each radar `.bin` record is interpreted as an eight-channel point sample. The first three
columns are mapped into camera coordinates, transformed into a common sensor frame with
`Tr_cam_to_velo`, and combined with Doppler and power features for the detector.

## LiDAR-only branch

`src/models/pointpillars.py` contains a compact PointPillars-style implementation:

1. Voxelize points into pillars.
2. Encode each pillar with a PointNet-like feature extractor.
3. Scatter pillar features into a BEV map.
4. Apply a small 2D CNN backbone.
5. Predict class logits and seven box parameters.

## Camera-fusion branch

`src/models/fusion.py` combines the pillar branch with image features:

- ResNet18 extracts image features.
- A lightweight depth head predicts a discrete depth distribution.
- Image features are weighted by depth probabilities and reduced to a BEV-shaped feature
  map.
- The camera and radar BEV features are concatenated before the detection head.

This implementation is useful as an educational fusion baseline, but it omits the full
camera-ray geometry, splatting operator, and depth supervision used by a complete LSS
system.

## Evaluation output

`src/pipeline/evaluate.py` converts predicted boxes from the sensor frame into the camera
frame and writes KITTI-style text predictions. The output format is intended for
task-specific offline evaluation.
