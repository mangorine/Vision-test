from argparse import OPTIONAL
from token import OP

import cv2
import urllib.request
import os
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

face_options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
)

hand_options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
)

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


capture = cv2.VideoCapture(0)

with vision.FaceLandmarker.create_from_options(
    face_options
) as face_landmarker, vision.HandLandmarker.create_from_options(
    hand_options
) as hand_landmarker:

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        timestamp_ms = int(time.time() * 1000)

        face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
        hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        draw_custom_landmarks(frame, face_result, hand_result)

        cv2.imshow("custom drawn lm", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

capture.release()
cv2.destroyAllWindows()
