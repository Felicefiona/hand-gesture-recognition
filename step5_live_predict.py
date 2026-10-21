import cv2
import numpy as np
import joblib
import mediapipe as mp

model = joblib.load("gesture_model.pkl")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    text = "No hand"
    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        # Same preparation as when we collected data
        wrist = hand.landmark[0]
        row = []
        for p in hand.landmark:
            row.append(p.x - wrist.x)
            row.append(p.y - wrist.y)
        biggest = max(abs(v) for v in row)
        row = [v / biggest for v in row]

        # Ask the AI
        probs = model.predict_proba([row])[0]
        best = np.argmax(probs)
        gesture = model.classes_[best]
        confidence = probs[best] * 100
        text = f"{gesture} ({confidence:.0f}%)"

    cv2.putText(frame, text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
    cv2.imshow("Live Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()