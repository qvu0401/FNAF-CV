# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FNAF-CV is a real-time hand gesture recognition system (ACM AI Project Winter '26). It uses a ResNet18 classifier fine-tuned on the HaGRID dataset, with MediaPipe handling hand detection and landmark extraction from a webcam feed.

## Environment Setup

Requires Python 3.12 (PyTorch does not support 3.13+) with pinned dependencies:

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install numpy==1.26.4
pip install torch torchvision torchaudio
pip install opencv-python==4.10.0.84
pip install mediapipe pillow
```

## Running the Demo

```bash
source venv/bin/activate
python webcam_gesture_demo.py
# Press 'q' to quit
```

The model weights file `fnaf_hgr_final.pth` must be in the same directory as the script.

## Architecture

The inference pipeline in `webcam_gesture_demo.py`:

1. **Hand detection** — MediaPipe `Hands` extracts 21 landmarks per hand (up to 2 hands)
2. **Bounding box** — `get_hand_bbox()` converts normalized landmarks to pixel coords with 25% padding (`CROP_PADDING = 0.25`)
3. **Classification** — The cropped hand region is resized to 224×224, normalized with ImageNet stats, and passed through ResNet18
4. **Confidence filtering** — Predictions below `CONFIDENCE_THRESHOLD = 0.8` display as `?`
5. **Directional sub-classification** — `get_finger_direction()` uses wrist→middle-finger-tip vector to split `two_sideways` into `two_sideways_left` / `two_sideways_right`

**Model checkpoint format** (`fnaf_hgr_final.pth`):
- `state_dict` — ResNet18 weights with replaced `fc` layer
- `classes` — ordered list of gesture class names
- `img_size` — input size (224)

**Gesture classes**: `palm`, `fist`, `stop`, `stop_inverted`, `two_sideways`

## Training

Training is done in Google Colab via `FNAF_CV_testing.ipynb`. The notebook:
- Downloads the HaGRID 500k dataset from Hugging Face (`cj-mills/hagrid-sample-500k-384p`)
- Reads `annotations_df.parquet`, filters to target classes, and explodes multi-label rows to one crop per row
- Fine-tunes ResNet18 (pretrained ImageNet weights) with SGD, 5 epochs, achieving ~99.5% val accuracy
- Saves checkpoint as `gesture_resnet18.pt`

## Two-Computer Setup

The system runs across two machines:

- **Computer 1 (CV machine)** — runs `webcam_gesture_demo.py`, sends recognized gestures over UDP to Computer 2
- **Computer 2 (game machine)** — runs `binds_fnaf2.py`, receives UDP payloads and executes FNAF 2 actions via PyAutoGUI

**UDP protocol**: Computer 1 sends `"<Hand>:<gesture>"` strings (e.g., `"Left:palm"`, `"Right:ok"`) to `UDP_IP:5005`. `COOLDOWN_SECONDS = 1.0` throttles sends. Only gestures in `TARGET_GESTURES` are transmitted.

To use, update `UDP_IP` in `webcam_gesture_demo.py` to Computer 2's IPv4 address. `binds_fnaf2.py` also exposes keyboard bindings (keys 1–8) as a fallback without the CV pipeline.

**Additional dependency for `binds_fnaf2.py`**:
```bash
pip install pyautogui keyboard
```

## Known Limitations

- Accuracy drops when the hand is rotated from upright; fix: augment training data with rotations
- No `no_gesture` class, so the model tries to classify any visible hand; fix: add `no_gesture` from HaGRID
- Model does gesture classification only, not detection; MediaPipe handles localization as a pre-step
