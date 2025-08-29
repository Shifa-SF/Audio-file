import cv2
import mediapipe as mp
import pyttsx3
import time
import math
import threading

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Initialize Text-to-Speech
engine = pyttsx3.init()

# Start webcam
cap = cv2.VideoCapture(0)


# Set frame size (increase resolution)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)   # You can change to 1920
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)   # You can change to 1080
last_gesture = None
last_time = 0
cooldown = 1.5  # seconds

def speak(text):
    def _speak():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=_speak, daemon=True).start()

def get_gesture(hand_landmarks):
    # Landmarks for fingers
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]
    index_tip = hand_landmarks.landmark[8]
    index_pip = hand_landmarks.landmark[6]
    middle_tip = hand_landmarks.landmark[12]
    middle_pip = hand_landmarks.landmark[10]
    ring_tip = hand_landmarks.landmark[16]
    ring_pip = hand_landmarks.landmark[14]
    pinky_tip = hand_landmarks.landmark[20]
    pinky_pip = hand_landmarks.landmark[18]

    # Check fingers (up = True)
    index_up = index_tip.y < index_pip.y
    middle_up = middle_tip.y < middle_pip.y
    ring_up = ring_tip.y < ring_pip.y
    pinky_up = pinky_tip.y < pinky_pip.y
    thumb_up = thumb_tip.x < thumb_ip.x  # for "call me" (sideways)

    # Distance for OK (index tip close to thumb tip)
    dist_ok = math.dist((thumb_tip.x, thumb_tip.y), (index_tip.x, index_tip.y))

    # Gesture detection
    if index_up and middle_up and not ring_up and not pinky_up :
        return "victory"  # index + middle up
    elif index_up and not middle_up and not ring_up and not pinky_up:
        return "One"  # ☝ single finger
    elif not thumb_up and not index_up and not middle_up and not ring_up and thumb_tip.y > thumb_ip.y:
        return "Bad"           # 👎 thumb down
    elif not index_up and not middle_up and not ring_up and not pinky_up and not thumb_up:
        return "Power"  # ✊ closed fist
    elif thumb_up and not index_up and not middle_up and not ring_up and pinky_up:
        return "Call me"  # 🤙 thumb + pinky
    elif dist_ok < 0.05 and middle_up and ring_up and pinky_up:
        return "Okay"  # 👌 thumb touching index
    elif thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
        return "Good job"  # 👍 thumbs up
    elif index_up and not middle_up and not ring_up and pinky_up:
        return "Rock on"  # 🤘 index + pinky
    else:
        return "Gesture unknown maybe it's an Alien Language !"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = get_gesture(hand_landmarks)

            if gesture:
                cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Speak only if cooldown passed or gesture changed
                current_time = time.time()
                if gesture != last_gesture or (current_time - last_time) > cooldown:
                    print("Speaking:", gesture)
                    speak(gesture)
                    last_gesture = gesture
                    last_time = current_time

    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
