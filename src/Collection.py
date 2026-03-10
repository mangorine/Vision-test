import numpy as np
import os
import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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


for action in actions:
    for sequence in range(no_sequences):
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)

base_options = python.BaseOptions(model_asset_path="pose_landmarker_lite.task")

pose_options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1
)

pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)


capture = cv2.VideoCapture(1)

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
            # Compte à rebours
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

                timestamp = int(time.time() * 1000) if frame_num == 0 else timestamp + 33

                p_res = pose_landmarker.detect_for_video(mp_image, timestamp)

                draw_pose_landmarks(frame, p_res)
                keypoints = extract_pose_landmarks(p_res)

                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                cv2.putText(frame, f"ENREGISTREMENT: {action} ({frame_num}/{sequence_length})", (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Data Collection", frame)

                cv2.waitKey(33)

            sequence += 1

capture.release()
cv2.destroyAllWindows()
print("Collection terminée")
