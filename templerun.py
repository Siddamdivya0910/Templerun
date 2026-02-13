import cv2
import mediapipe as mp
import pyautogui

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

screen_w = 640
screen_h = 480

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    h, w, _ = img.shape

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(img, handLms,
                                   mp_hands.HAND_CONNECTIONS)

            # Palm center approx landmark 9
            cx = int(handLms.landmark[9].x * w)
            cy = int(handLms.landmark[9].y * h)

            cv2.circle(img, (cx, cy), 10,
                       (0,255,0), cv2.FILLED)

            # Gesture zones
            if cx < w//3:
                pyautogui.press('left')

            elif cx > 2*w//3:
                pyautogui.press('right')

            elif cy < h//3:
                pyautogui.press('up')

            elif cy > 2*h//3:
                pyautogui.press('down')

    cv2.imshow("Temple Run Hand Control", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()