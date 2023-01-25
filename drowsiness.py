import cv2  # to open Camara
import os  # to interact with os
from keras.models import load_model  # to load model
import numpy as np  # for numarical operations used
# import playsound
import pygame  # for beep alert without hang the video

cascPath = "haarcascade_frontalface_default.xml"  # face features detection
face = cv2.CascadeClassifier(cv2.data.haarcascades + cascPath)

cascPath = "haarcascade_lefteye_2splits.xml"
leye = cv2.CascadeClassifier(cv2.data.haarcascades + cascPath)

cascPath = "haarcascade_righteye_2splits.xml"
reye = cv2.CascadeClassifier(cv2.data.haarcascades + cascPath)

lbl = ['Close', 'Open']

model = load_model('models/cnncat2.h5')  # model loaded
path = os.getcwd()  # return current directory of process
cap = cv2.VideoCapture(0)  # video streaming on
font = cv2.FONT_HERSHEY_COMPLEX_SMALL  # font style
count = 0
score = 0
thicc = 2
rpred = [99]
lpred = [99]
opt = 1

while (True):

    key = cv2.waitKey(10)
    if (key == 113):
        break  ##toggle key
    elif (key == 116):
        opt = opt * -1
    if (opt == -1):
        continue

    ret, frame = cap.read()
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y + h, x:x + w]
        count = count + 1
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24))
        r_eye = r_eye / 255
        r_eye = r_eye.reshape(24, 24, -1)
        r_eye = np.expand_dims(r_eye, axis=0)
        rpred = model.predict_classes(r_eye)
        if (rpred[0] == 1):
            lbl = 'Open'
        if (rpred[0] == 0):
            lbl = 'Closed'
        break

    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y + h, x:x + w]
        count = count + 1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye / 255
        l_eye = l_eye.reshape(24, 24, -1)
        l_eye = np.expand_dims(l_eye, axis=0)
        lpred = model.predict_classes(l_eye)
        if (lpred[0] == 1):
            lbl = 'Open'
        if (lpred[0] == 0):
            lbl = 'Closed'
        break

    if (rpred[0] == 0 and lpred[0] == 0):
        score = score + 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):
    else:
        score = 0
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # if(score<0):
    # score=0
    # cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    if (score > 3):
        # person is feeling sleepy so we beep the alarm
        # cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        try:
            # playsound.playsound('audio.wav')#sound playing
            pygame.mixer.init()
            pygame.mixer.music.load("audio.wav")
            pygame.mixer.music.play()
            cv2.putText(frame, "Beep Alert ON!", (50, 50), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
        except:  # isplaying = False
            pass

        if (thicc < 11):
            thicc = thicc + 2
        else:
            thicc = thicc - 2
            if (thicc < 2):
                thicc = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
    cv2.imshow('frame', frame)

cap.release()
cv2.destroyAllWindows()