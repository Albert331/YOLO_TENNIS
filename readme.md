# Tennis Positioning Project — Overview

## Goal
Detect and track player/ball positions in tennis footage and map them onto a top-down view of the court, using real detections (not simulated data). First sport in a broader "sport positioning" series — football/soccer and 3D field homography via camera pose estimation are planned next.

## Pipeline

1. **Court keypoint detection**
   - Model: ResNet50 backbone, custom `fc` head outputting 14 keypoints (x, y pairs) — court corners, service line intersections, center marks, etc.
   - Trained separately, loaded from a `.pth` checkpoint.
   - Input: 224x224 resized frame, ImageNet-normalized.
   - Run infrequently (court geometry is stable if the camera is static), not every frame.

2. **Homography**
   - Maps detected court keypoints to known real-world tennis court coordinates using `cv2.findHomography()`.
   - Only needs 4 corner points in principle, though the current model predicts 14 — using a subset for homography is an open option.

3. **Ball / player detection**
   - Model: YOLOv5su (`best.pt`), via `ultralytics.YOLO`.
   - Currently detecting the ball; player detection is part of the plan but not yet the focus.
   - Foot/ball position taken from bounding box (bottom-center for players once added).

4. **Projection**
   - Detected positions warped through the homography matrix into top-down court coordinates.

5. **Visualization**
   - Top-down court + position dots rendered in pygame, fed by the real detection pipeline (not fake/simulated positions).

## Current status
- Court keypoint model and ball detection model both running on CUDA, real-time (~20-40ms combined per frame after fixing a first-frame CUDA warm-up cost and adding frame-skipping for both models).
- Homography step and pygame visualization not yet built — next steps.
- Decided against ONNX conversion for now — PyTorch models are already fast enough; not worth the added complexity.

## Next steps
1. Decide how to get 4 clean corner points from the 14-keypoint model output (either just use 4 of the 14, or retrain with a 4-point head).
2. Implement `cv2.findHomography()` using those corners against known real-world court dimensions.
3. Add player detection (separate model or class) and extract foot position (bottom-center of bbox).
4. Warp ball/player positions through the homography matrix.
5. Build the pygame top-down view fed by live detections.
6. After tennis: move to football/soccer positioning, then 3D field homography via camera pose estimation.