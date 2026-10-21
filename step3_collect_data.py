import cv2
import csv
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

labels = {'1': 'fist', '2': 'palm', '3': 'peace', '4': 'thumbs_up'}
counts = {name: 0 for name in labels.values()}

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    row = None
    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        # Make numbers relative to the wrist (point 0)
        wrist = hand.landmark[0]
        row = []
        for p in hand.landmark:
            row.append(p.x - wrist.x)
            row.append(p.y - wrist.y)

        # Scale so hand size does not matter
        biggest = max(abs(v) for v in row)
        row = [v / biggest for v in row]

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    if row is not None and chr(key) in labels:
        name = labels[chr(key)]
        with open("gesture_data.csv", "a", newline="") as f:
            csv.writer(f).writerow([name] + row)
        counts[name] += 1

    # Show how many samples we saved
    y = 30
    for name, c in counts.items():
        cv2.putText(frame, f"{name}: {c}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y += 30

    cv2.imshow("Collect Data", frame)

cap.release()
cv2.destroyAllWindows()