# FNAF-CV
ACM AI Project Winter '26

2/6/26 DEMO

Ran in venv with python 3.12 (python 3.13+ doesnt have pytorch support), numpy==1.26.4, opencv-python==4.10.0.84

trained with downscaled hagrid dataset https://huggingface.co/datasets/cj-mills/hagrid-sample-500k-384p

Uses resnet18 as pre-trained full frame classifier (NOTE this model only does gesture classification not detection)

classes used: palm, stop, stop_inverted, fist; shows as unknown if confidence < 0.8

to run demo in virtual environment:

python3.12 -m venv venv

source venv/bin/activate

pip install --upgrade pip

pip install numpy==1.26.4

pip install torch torchvision torchaudio

pip install opencv-python==4.10.0.84

pip install mediapipe pillow

python webcam_gesture_demo.py

Limitations:

Can't detect gestures with high confidence when hand is rotated from upright position. To fix, we could augment the dataset first (rotated 90, 180, mirrored, etc...)

Model also tries to detect a hand not doing any gesture as one of the gesture. To fix, we could add the no_gesture class to the dataset, which might help? It contains natural hand postures. However, no_gesture only works if we use a gesture detection model (YOLO)

Model right now (resnet18) only does the gesture recognition, not detection (detecting where the hand is + recognizing gesture). The python program detects the hand for the model using MediaPipe, crop it, then use the model for recognition. Maybe it'd be more accurate if the model does both with YOLO? I'm not sure, but it'd be more to train.
