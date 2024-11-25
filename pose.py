import cv2
import mediapipe as mp
import numpy as np
import time
import requests

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# Function to send FCM notification
def send_notification(Lcounter, Rcounter):
    url = 'https://fcm.googleapis.com/fcm/send'
    headers = {
        'Authorization': 'key=YOUR_SERVER_KEY',  # Replace with your server key from Firebase
        'Content-Type': 'application/json'
    }
    payload = {
        'to': '/topics/all',  # Change this to the appropriate topic or token
        'notification': {
            'title': 'Workout Completed',
            'body': f'Total Left Arm Reps: {Lcounter}, Total Right Arm Reps: {Rcounter}'
        }
    }
    response = requests.post(url, headers=headers, json=payload)
    print(response.status_code, response.json())


cap = cv2.VideoCapture(0)

# Curl counter variables
Lcounter = 0
Lstage = None
Rcounter = 0
Rstage = None

# Timer variables
last_movement_time = time.time()
movement_threshold = 5  # seconds

# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            Lshoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            Lelbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            Lwrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            Rshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            Relbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            Rwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Calculate angle
            Langle = calculate_angle(Lshoulder, Lelbow, Lwrist)
            Rangle = calculate_angle(Rshoulder, Relbow, Rwrist)

            # Visualize angle
            cv2.putText(image, str(Langle),
                        tuple(np.multiply(Lelbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, str(Rangle),
                        tuple(np.multiply(Relbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # Curl counter logic
            if Langle > 160:
                Lstage = "down"
            if Langle < 30 and Lstage == 'down':
                Lstage = "up"
                Lcounter += 1
                last_movement_time = time.time()  # Reset the timer on movement
                print(f"Left Counter: {Lcounter}")

            if Rangle > 160:
                Rstage = "down"
            if Rangle < 30 and Rstage == 'down':
                Rstage = "up"
                Rcounter += 1
                last_movement_time = time.time()  # Reset the timer on movement
                print(f"Right Counter: {Rcounter}")
        except Exception as e:
            print(f"Error: {e}")
            pass

        # Check for inactivity
        if time.time() - last_movement_time > movement_threshold:
            print(f"No movement detected for {movement_threshold} seconds.")
            print(f"Total Left Counter: {Lcounter}")
            print(f"Total Right Counter: {Rcounter}")
            send_notification(Lcounter, Rcounter)  # Send notification
            break

        # Render curl counter
        # Setup status box
        cv2.rectangle(image, (0, 0), (180, 73), (245, 117, 16), -1)
        # Rep data
        cv2.putText(image, 'REPS', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(Lcounter),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255), 2, cv2.LINE_AA)

        # Stage data
        cv2.putText(image, 'STATE', (65, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, Lstage,
                    (60, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255), 2, cv2.LINE_AA)

        # Right arm counter
        cv2.rectangle(image, (460, 0), (640, 73), (0, 0, 255), -1)
        # Rep data
        cv2.putText(image, 'REPS', (470, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(Rcounter),
                    (470, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255), 2, cv2.LINE_AA)

        # Stage data
        cv2.putText(image, 'STATE', (520, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, Rstage,
                    (520, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255), 2, cv2.LINE_AA)

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

print(f"Final Left Counter: {Lcounter}")
print(f"Final Right Counter: {Rcounter}")
send_notification(Lcounter, Rcounter)  # Send final notification
