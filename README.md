# Vision-test

Real-time human pose action recognition using **MediaPipe Pose Landmarker** + **TensorFlow LSTM**.

## Features

- Pose keypoint extraction (33 landmarks × 4 values = 132 features/frame)
- Sequence dataset recording from webcam
- LSTM training pipeline for action classification
- Real-time webcam inference with on-frame prediction display

## Repository layout

- [rec.py](rec.py): real-time recognition script
- [src/Collection.py](src/Collection.py): dataset recording script
- [src/detection_model.py](src/detection_model.py): model training script
- [models/model_67.keras](models/model_67.keras): trained model output
- [data/](data): pose sequence dataset
- [pose_landmarker_lite.task](pose_landmarker_lite.task): MediaPipe pose model asset

## Current classes

The project is currently configured for:

- `67` ( for real ? yes, the gen-alpha trend ... )
- `idle`

## Data format

Each sample is saved as a NumPy file in:

`data/<action>/<sequence>/<frame>.npy`

- `no_sequences = 20`
- `sequence_length = 60`
- Feature vector per frame: 132 values (`33 × (x, y, z, visibility)`)

## Workflow

### 1) Collect data

Run:

```bash
python src/Collection.py
```

Controls:

- Press `s` to start recording a sequence
- Press `q` to quit

### 2) Train model

Run:

```bash
python src/detection_model.py
```

Output model is saved to:

- [models/model_67.keras](models/model_67.keras)

TensorBoard logs:

- [logs/train](logs/train)

### 3) Run real-time recognition

Run:

```bash
python rec.py
```

Controls:

- Press `q` to quit the detection window

## Notes

- Camera index is set to `1` in both [src/Collection.py](src/Collection.py) and [rec.py](rec.py).  
  If needed, change to `0` depending on your system.
- Inference threshold is currently `0.4` in [rec.py](rec.py).
- Inference is throttled for performance in [rec.py](rec.py).
