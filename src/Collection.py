import numpy as np
import os
import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

DATA_PATH = os.path.join("data")
actions = np.array(["67", "idle"])
no_sequences = 30
sequence_length = 30

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # Pouce
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # Index
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # Majeur
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # Annulaire
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # Auriculaire
]


def draw_custom_landmarks(image, face_result, hand_result):
    """Dessine manuellement les points et les lignes avec OpenCV"""
    h, w, _ = image.shape

    if hand_result.hand_landmarks:
        for hand_landmarks in hand_result.hand_landmarks:
            pixel_pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

            for connection in HAND_CONNECTIONS:
                pt1, pt2 = pixel_pts[connection[0]], pixel_pts[connection[1]]
                cv2.line(image, pt1, pt2, (0, 255, 0), 2)

            for pt in pixel_pts:
                cv2.circle(image, pt, 4, (0, 0, 255), -1)

    if face_result.face_landmarks:
        for face_landmarks in face_result.face_landmarks:
            for lm in face_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(image, (cx, cy), 1, (255, 255, 255), -1)


def extract_landmarks(hand_result):
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)

    if hand_result.hand_landmarks:
        for i, res in enumerate(hand_result.hand_landmarks):
            label = hand_result.handedness[i][0].category_name
            coords = np.array([[lm.x, lm.y, lm.z] for lm in res]).flatten()
            if label == "Left":
                lh = coords
            else:
                rh = coords
    return np.concatenate([lh, rh])


for action in actions:
    for sequence in range(no_sequences):
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)

base_options_face = python.BaseOptions(model_asset_path="face_landmarker.task")
base_options_hand = python.BaseOptions(model_asset_path="hand_landmarker.task")

face_options = vision.FaceLandmarkerOptions(
    base_options=base_options_face,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)

hand_options = vision.HandLandmarkerOptions(
    base_options=base_options_hand,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2
)

face_landmarker = vision.FaceLandmarker.create_from_options(face_options)
hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

capture = cv2.VideoCapture(0)

for action in actions:
    sequence = 0
    while sequence < no_sequences:
        ret, frame = capture.read()
        cv2.putText(frame, f"ACTION: {action} | Sequence: {sequence}/{no_sequences}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Appuyez sur 'S' pour ENREGISTRER", (100, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Data Collection", frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            sequence = no_sequences
            break

        if key & 0xFF == ord('s'):
            for i in range(2, 0, -1):
                ret, frame = capture.read()
                cv2.putText(frame, f"DEBUT DANS {i}...", (150, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
                cv2.imshow("Data Collection", frame)
                cv2.waitKey(1000)

            for frame_num in range(sequence_length):
                ret, frame = capture.read()

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )
                timestamp = int(time.time() * 1000)

                h_res = hand_landmarker.detect_for_video(mp_image, timestamp)
                draw_custom_landmarks(frame, None, h_res)
                keypoints = extract_landmarks(h_res)

                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                cv2.putText(frame, f"ENREGISTREMENT: {action} ({frame_num})", (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Data Collection", frame)
                cv2.waitKey(1)

            sequence += 1

capture.release()
cv2.destroyAllWindows()
