# Tennis Positioning Project — Overview

## Goal
Detect and track player/ball positions in tennis footage and map them onto a top-down view of the court, using real detections (not simulated data). First sport in a broader "sport positioning" series — football/soccer and 3D field homography via camera pose estimation are planned next.

## Pipeline

1. **Court keypoint detection**
   - Model: ResNet50 backbone, custom `fc` head outputting 14 keypoints (x, y pairs) — court corners, service line intersections, center marks, etc.
   - Trained separately, loaded from a `.pth` checkpoint.
   - Input: 224x224 resized frame, ImageNet-normalized.
   - Run infrequently (court geometry is stable if the camera is static), not every frame.

2. **Ball / player detection**
   - Model: YOLOv5su (`best.pt`), via `ultralytics.YOLO`.
   - Currently detecting the ball; player detection is part of the plan but not yet the focus.
   - Foot/ball position taken from bounding box (bottom-center for players once added).

3. **Visualization**
   - Top-down court + position dots rendered in pygame, fed by the real detection pipeline (not fake/simulated positions).

## Current status
- Court keypoint model and ball detection model both running on CUDA, real-time (~20-40ms combined per frame after fixing a first-frame CUDA warm-up cost and adding frame-skipping for both models).
- Pygame visualization not yet built — next step.
- Decided against ONNX conversion for now — PyTorch models are already fast enough; not worth the added complexity.

## Next steps
1. Add player detection (separate model or class) and extract foot position (bottom-center of bbox).
2. Build the pygame top-down view fed by live detections.
3. After tennis: move to football/soccer positioning, then 3D field homography via camera pose estimation.