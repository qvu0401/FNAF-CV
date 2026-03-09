# FNAF-CV
ACM AI Project Winter '26

2/6/26 DEMO

https://github.com/user-attachments/assets/76abd7fb-8fa1-4911-a3e6-510ae64169b9

Trained with downscaled HaGRID dataset: https://huggingface.co/datasets/cj-mills/hagrid-sample-500k-384p

Uses ResNet18 fine-tuned for gesture classification. Hand detection and localization is handled by MediaPipe — the model only classifies the cropped hand region.

Classes: `palm`, `mute`, `ok`, `two_up`, `two_up_inverted`. Last 2 was augmented by rotating 0, 90, 180, 270 degrees and relabelling as `two_up`, `two_right`, `two_left`, `two_down`. Shows as unknown if confidence < 55%.

## Setup

Requires Python 3.12 (PyTorch does not support 3.13+).

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```bash
python webcam_gesture_demo.py
```

Press `q` to quit.

## Limitations

- No `no_gesture` class, so the model always tries to classify a visible hand as one of the gestures. Fix: add the `no_gesture` class from HaGRID.
- Model does classification only, not detection. MediaPipe handles localization as a pre-step. A unified detection+classification model (e.g. YOLO) might improve accuracy.
