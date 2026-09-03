# Tennis Positioning Pipeline — Debug Notes

## Setup
- **Ball detection**: YOLOv5su (`best.pt`), run via `ultralytics.YOLO`
- **Court keypoints**: ResNet50 backbone, custom `fc` head outputting `14*2` (14 keypoint x,y pairs), loaded from `keypoints_model.pth`
- Both models run on CUDA

## Bugs fixed
1. `cv2.release()` → should be `cap.release()` (typo, wrong object)
2. `if cv2.waitKey(1) and 0xFF == ord('q'):` → should be `cv2.waitKey(1) & 0xFF == ord('q')` (bitwise `&`, not logical `and` — operator precedence bug meant quit key never worked reliably)
3. YOLO model wasn't actually on CUDA — fixed by not passing `device=device` redundantly on every call once `ball.to(device)` (or passing `device=0` per-call, which also works fine for `.pt` models)

## Performance debugging (the "slo-mo" saga)
Symptom: per-frame processing time looked like it was climbing every loop iteration.

Root causes found, in order of discovery:
1. **False leak**: `torch.cuda.memory_allocated()` checked across frames — it jumped once (232MB → 266MB) then stayed flat. Not a memory leak.
2. **First-frame CUDA warm-up cost**: first call to any CUDA model pays a one-time cost (kernel compilation, cuDNN algorithm selection, memory allocation). This showed up as ~0.5–0.6s on frame 1, then dropped to ~20–40ms on every frame after. Fixed by adding a dummy warm-up inference call before the main loop starts, e.g.:
   ```python
   _ = ball(np.zeros((1080, 1920, 3), dtype=np.uint8), device=0, verbose=False)
   ```
3. Once warm-up was added and timing was isolated per stage, both models settled to a healthy ~20–30ms combined per frame — comfortably real-time.
4. Frame-skipping was added for both models to reduce compute further:
   - Keypoint model: recompute every 30 frames (court geometry barely changes if camera is static), reuse cached keypoints in between.
   - YOLO ball detection: recompute every 10 frames (tunable — lower for smoother ball tracking, higher for more speed).

## Decision: no ONNX
- Attempted exporting YOLO to ONNX (`model.export(format='onnx')`) to see if it would speed things up further.
- Hit `onnxruntime-gpu` / CUDA version mismatch (`onnxruntime-gpu` wanted CUDA 13 + cuDNN 9, but installed CUDA toolchain was different — likely CUDA 12.x, matching the working PyTorch install).
- Attempted reinstalling `onnxruntime-gpu` at a version matching CUDA 12, hit a broken/conflicting install (`onnxruntime` + `onnxruntime-gpu` coexisting badly), causing `ImportError` on `onnxruntime.capi._pybind_state`.
- **Decision: dropped ONNX entirely.** The PyTorch `.pt` YOLO model was already running at ~20–40ms per frame after the warm-up fix and frame-skipping — well within real-time — so the ONNX conversion wasn't solving an actual problem. Not worth the environment/dependency fight for the marginal gain.
- Sticking with `best.pt` via `ultralytics.YOLO` going forward.

## Open question / next step
- Considered reducing keypoint model output from `14*2` to `4*2` (just the 4 court corners) since that's likely all that's needed for `cv2.findHomography()`.
- **Important**: cannot just change the `fc` layer size on an already-trained model — the saved `state_dict` has weights shaped for 14 keypoints, so `load_state_dict()` will throw a shape mismatch if the head is resized without retraining.
- Options if only 4 corners are needed:
  1. Keep the model as-is, just use the first 4 of the 14 predicted points at inference time (no retraining) — assuming those 4 correspond to the outer corners.
  2. Retrain/finetune with a 4-keypoint head — needs relabeled data with only 4 points annotated.
- Not yet decided which route to take.