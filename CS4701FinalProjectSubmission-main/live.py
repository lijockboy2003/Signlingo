from ultralytics import YOLO
import cv2
import math
import random
import time

model = YOLO("runs/detect/train3/weights/best.pt")

classNames = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

cap = cv2.VideoCapture(0)

def show_correct_sign(letter):
    img_path = f"sign_images/{letter}.png" 
    sign_img = cv2.imread(img_path)
    if sign_img is not None:
        cv2.imshow('Correct Sign', sign_img)
    else:
        print(f"Image for letter {letter} not found!")

def choose_random_letter():
    letter = random.choice(classNames)
    start_time = time.time()
    return letter, start_time

current_letter, start_time = choose_random_letter()
feedback_message = ""
feedback_color = (0, 0, 0)  
feedback_time = 0  
correct_sign_shown = False  

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    letter_detected = False

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

            confidence = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            org = (x1, y1)
            font = cv2.FONT_HERSHEY_TRIPLEX
            color = (255, 0, 0)

            cv2.putText(img, classNames[cls], org, font, fontScale=1, color=color, thickness=4)

            if classNames[cls] == current_letter:
                letter_detected = True
                break
        if letter_detected:
            break

    prompt_text = f"Sign this letter: {current_letter}"
    cv2.putText(img, prompt_text, (10, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    if letter_detected:
        feedback_message = "Correct!"
        feedback_color = (0, 255, 0)  
        feedback_time = time.time()
        if correct_sign_shown:
            cv2.destroyWindow('Correct Sign')
            correct_sign_shown = False
        current_letter, start_time = choose_random_letter()
    elif time.time() - start_time > 10:
        feedback_time = time.time()
        if not correct_sign_shown:
            show_correct_sign(current_letter)
            correct_sign_shown = True
        start_time = time.time()
    
    if time.time() - feedback_time < 2:  
        cv2.putText(img, feedback_message, (10, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, feedback_color, 2, cv2.LINE_AA)
    else:
        feedback_message = ""  

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
