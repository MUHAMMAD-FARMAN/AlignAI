import cv2
import mediapipe as mp
import time
import numpy as np


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


# Define the message and time to display it
message = "Press ESC key to stop exercise monitoring"
display_time = 15  # seconds


def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
        np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def start_exercise_monitor():

    start_time = time.time()

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if time.time() - start_time < display_time:
            cv2.putText(
                image,
                message,
                (10, 30),  # Position (x, y)
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,  # Font scale
                (0, 255, 0),  # Color (B, G, R)
                1,  # Thickness
                cv2.LINE_AA
            )

        cv2.imshow('MediaPipe Pose', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def start_bridging_exercise_monitor(video_path=None):
    counter = 0
    stage = "down"
    user_side = None
    if video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(0)
    start_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Error: Unable to capture video")
            continue

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if time.time() - start_time < display_time:
            cv2.putText(
                image,
                message,
                (10, 30),  # Position (x, y)
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,  # Font scale
                (0, 255, 0),  # Color (B, G, R)
                1,  # Thickness
                cv2.LINE_AA
            )

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # extract landmarks
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].visibility]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].visibility]
            # Determine which side is closer to the camera
            shoulder = None
            hip = None
            if (left_shoulder[2] > right_shoulder[2]):
                shoulder = left_shoulder
            else:
                shoulder = right_shoulder

            if (left_hip[2] > right_hip[2]):
                hip = left_hip
            else:
                hip = right_hip

            if shoulder[0] < hip[0]:
                # User is lying on their left side
                user_side = "left"
                relevant_shoulder = left_shoulder
                relevant_hip = left_hip
                relevant_knee = left_knee
                relevant_ankle = left_ankle
            else:
                # User is lying on their right side
                user_side = "Right"
                relevant_shoulder = right_shoulder
                relevant_hip = right_hip
                relevant_knee = right_knee
                relevant_ankle = right_ankle

            # Calculate normalized hip raise distance
            hip_raise = relevant_shoulder[1] - relevant_hip[1]
            height_reference = np.linalg.norm(np.array(
                [relevant_knee[0], relevant_knee[1]]) - np.array([relevant_ankle[0], relevant_ankle[1]]))
            normalized_hip_raise = hip_raise / height_reference

            # Monitor the exercise based on normalized hip raise
            if stage == 'up' and normalized_hip_raise < 0.1:
                stage = 'down'
            elif stage == 'down' and normalized_hip_raise > 0.4:
                stage = 'up'
                counter += 1

            # Render text
            cv2.putText(image, f'REPS: {counter}', (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, f'STAGE: {stage}', (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            if time.time() - start_time < display_time:
                cv2.putText(
                    image,
                    "User is lying on their " + user_side + " side",
                    (10, 60),  # Position (x, y)
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,  # Font scale
                    (0, 255, 0),  # Color (B, G, R)
                    1,  # Thickness
                    cv2.LINE_AA
                )

        except Exception as e:
            print(f"Error: {e}")

        cv2.imshow('MediaPipe Pose', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
