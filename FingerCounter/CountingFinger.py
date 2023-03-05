import os
import time

import HandTracking.HandTrackingModule as htm
import cv2

##################################
webcamp_height, webcamp_weight = 640, 460
hand_detector = htm.handDetector()
##################################
cap = cv2.VideoCapture(0)

# 3 for height and 4 for weight
cap.set(3, webcamp_height)
cap.set(4, webcamp_weight)
pTime = 0

# how to overlay our images on the webcam
folder_path = 'Finger'
my_image_list = os.listdir(folder_path)
# print(my_image_list)
overlay_list = []
for image_paht in my_image_list:
    image = cv2.imread(f'{folder_path}/{image_paht}')
    # print(f'{folder_path}/{image_paht}')
    overlay_list.append(image)
# print(len(overlay_list))

tipIds = [4, 8, 12, 16, 20]
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty frame")
        # if loading video then used break instead of continue
        continue
    image = hand_detector.findHands(image)
    lmList = hand_detector.findPosition(image, draw=False)
    if len(lmList) != 0:
        fingers = []
        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 Finger
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        totalFinger = fingers.count(1)
        print(totalFinger)
        h, w, c = overlay_list[totalFinger - 1].shape
        image[0:h, 0:w] = overlay_list[totalFinger - 1]
        cv2.rectangle(image, (20, 2550), (175, 245), (200, 100, 0), cv2.FILLED)
        cv2.putText(image, str(totalFinger), (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 10,
                    (200, 255, 200), 25)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(image, f'FPS: {str(int(fps))}', (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (200, 255, 200), 3)
    cv2.imshow("FingerCounting: ", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
