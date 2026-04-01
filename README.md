# FNAF-CV
ACM AI Project Winter '26

Fine-tuned ResNet18 classifier on the [HaGRID 500k dataset](https://huggingface.co/datasets/cj-mills/hagrid-sample-500k-384p) for real-time hand gesture recognition. Hand detection and localization is handled by MediaPipe — the model only classifies the cropped hand region. `two_sideways` is further split into `two_sideways_left` / `two_sideways_right` at inference time using wrist-to-fingertip landmark geometry. Predictions below 80% confidence display as `?`.

**Recognized gestures:** `palm`, `fist`, `ok`, `mute`, `two_up`, `two_down`, `two_sideways_left`, `two_sideways_right`

## Architecture

The system runs across two machines:

- **CV machine** (`webcam_gesture_demo.py`) — captures webcam input, runs inference, and sends recognized gestures over UDP
- **Game machine** (`binds_fnaf2.py`) — receives UDP payloads and executes system-level mouse/keyboard actions via PyAutoGUI; also supports keyboard bindings (keys 1–8) as a fallback

UDP payload format: `"<Hand>:<gesture>"` (e.g. `"Left:palm"`, `"Right:ok"`) sent to port `5005` with a 1-second cooldown.

## Setup

Requires Python 3.12 (PyTorch does not support 3.13+).

**CV machine:**
```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Game machine (additional dependencies):**
```bash
pip install pyautogui keyboard
```

Update `UDP_IP` in `webcam_gesture_demo.py` to the game machine's IPv4 address before running.

## Run

**CV machine:**
```bash
source venv/bin/activate
python webcam_gesture_demo.py
# Press 'q' to quit
```

**Game machine:**
```bash
python binds_fnaf2.py
# Press 'ESC' to exit
```

The model weights file `fnaf_hgr_final.pth` must be in the same directory as `webcam_gesture_demo.py`.

## Limitations

- No `no_gesture` class — the model always tries to classify any visible hand. Suggested fix: add the `no_gesture` class from HaGRID.
- Accuracy drops when the hand is rotated from upright. Fix: augment training data with rotation variants.
- Model does classification only, not detection. MediaPipe handles localization as a pre-step.
