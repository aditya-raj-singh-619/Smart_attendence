from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

def speak(str1):
    speak = Dispatch(("SAPI.SpVoice"))
    speak.Speak(str1)

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

# Fix mismatch
min_len = min(FACES.shape[0], len(LABELS))
FACES = FACES[:min_len]
LABELS = LABELS[:min_len]

print("Training KNN...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

imgBackground = cv2.imread("background.png")
COL_NAMES = ['NAME', 'TIME']

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)

        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")
        file_path = "Attendance/Attendance_" + date + ".csv"
        exist = os.path.isfile(file_path)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output[0]), (x, y-15),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

        attendance = [str(output[0]), str(timestamp)]

    frame_resized = cv2.resize(frame, (640, 351))
    imgBackground[162:162+351, 55:55+640] = frame_resized
    cv2.imshow("Frame", imgBackground)
    k = cv2.waitKey(1)

    if k == ord('o'):  # Press 'o' to take attendance
        speak("Attendance Taken..")
        time.sleep(2)

        # Prevent duplicate entries
        if exist:
            with open(file_path, "r") as csvfile:
                existing_names = [row[0] for row in csv.reader(csvfile)]
        else:
            existing_names = []

        if str(output[0]) not in existing_names:
            with open(file_path, "+a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                if not exist:
                    writer.writerow(COL_NAMES)
                writer.writerow(attendance)

    if k == ord('q'):  # for clossing the camera
        break

video.release()
cv2.destroyAllWindows()
