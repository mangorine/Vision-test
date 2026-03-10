import numpy as np
import os
import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model

DATA_PATH = os.path.join("data")
actions = np.array(["67", "idle"])
no_sequences = 20
sequence_length = 60


POSE_CONNECTIONS = [
    # Visage
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # Tronc
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Bras gauche
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # Bras droit
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # Jambe gauche
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # Jambe droite
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
]


def draw_pose_landmarks(image, pose_result):
    """Dessine manuellement les points et les lignes avec OpenCV"""
    h, w, _ = image.shape

    if pose_result and pose_result.pose_landmarks:
        # Dans la nouvelle API, pose_landmarks est une liste de listes (une par personne)
        for pose_landmarks in pose_result.pose_landmarks:
            pixel_pts = [(int(lm.x * w), int(lm.y * h)) for lm in pose_landmarks]

            for connection in POSE_CONNECTIONS:
                pt1, pt2 = pixel_pts[connection[0]], pixel_pts[connection[1]]
                cv2.line(image, pt1, pt2, (0, 255, 0), 2)

            for pt in pixel_pts:
                cv2.circle(image, pt, 4, (0, 0, 255), -1)


def extract_pose_landmarks(pose_result):
    """Extrait les 33 points du corps (x, y, z, visibilité)"""
    if pose_result and pose_result.pose_landmarks:
        res = pose_result.pose_landmarks[0]
        # La nouvelle API inclut la propriété 'visibility' pour chaque point
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in res]).flatten()
        return pose
    else:
        return np.zeros(33 * 4)


base_options = python.BaseOptions(model_asset_path="pose_landmarker_lite.task")

pose_options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1
)

pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

sequence = []
threshold = 0.4
capture = cv2.VideoCapture(1)

frame_counter = 0  # pour la performance
current_action = "Mouvement inconnu"
current_prob = 0.0

model = load_model('models/model_67.keras')
print("Modèle chargé")

with vision.PoseLandmarker.create_from_options(
    pose_options
) as pose_landmarker:
    while capture.isOpened():
        ret, frame = capture.read()

        frame_counter += 1

        timestamp_ms = int(time.time() * 1000)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

        draw_pose_landmarks(frame, pose_result)

        # Detection
        pose = extract_pose_landmarks(pose_result)
        sequence.append(pose)
        sequence = sequence[-sequence_length:]

        if len(sequence) == sequence_length and frame_counter % 60 == 0:  # Inference tous les 60 frames pour la perf
            res = model(np.expand_dims(sequence, axis=0), training=False)[0].numpy()

            current_action = actions[np.argmax(res)]
            current_prob = res[np.argmax(res)]

        if current_prob > threshold:
            cv2.putText(frame, f"{current_action.upper()} {current_prob:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow('REAL-TIME DETECTION', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


capture.release()
cv2.destroyAllWindows()
