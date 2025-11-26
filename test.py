import matplotlib
matplotlib.use("TkAgg")

from retinaface import RetinaFace
import cv2
import matplotlib.pyplot as plt



cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Cannot open camera")
    exit()


while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Failed to read frame.")
        break



    try:
        resp=RetinaFace.detect_faces(frame)
        if not resp:
            print("Error: Failed to detect faces.")
    except Exception as e:
        print("Detection error:", e)
        resp = None

    if isinstance(resp, dict):
        for key in resp:
            face = resp[key]
            x1, y1, x2, y2 = face["facial_area"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('Webcam Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cam.release()
cv2.destroyAllWindows()