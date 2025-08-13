import cv2
import mediapipe as mp
import numpy as np  # Added for creating black canvas

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

# Create a canvas to draw on
canvas = None

# For tracking previous point
prev_x, prev_y = 0, 0

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)

    # Initialize empty black canvas for drawing only
    if canvas is None:
        canvas = np.zeros_like(img)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lm_list = []

            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            # Drawing with index finger
            if lm_list:
                x1, y1 = lm_list[8]  # Index fingertip

                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x1, y1

                # Draw line on transparent canvas only
                cv2.line(canvas, (prev_x, prev_y), (x1, y1), (255, 0, 0), 8)

                prev_x, prev_y = x1, y1
    else:
        # Reset previous points if hand not detected
        prev_x, prev_y = 0, 0

    # Merge without ghost effect
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY)
    inv_mask = cv2.bitwise_not(mask)

    img_bg = cv2.bitwise_and(img, img, mask=inv_mask)
    draw_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
    img = cv2.add(img_bg, draw_fg)

    cv2.imshow("AI Drawing - Hand Gesture", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
