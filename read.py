import cv2 as cv
import numpy as np


def rescaleFrame(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] *scale)
    dim = (width, height)

    return cv.resize(frame, dim, interpolation=cv.INTER_AREA)

# Allows program to begin Webcam Capture
webcam = cv.VideoCapture(0) 

while True:
    isTrue, frame = webcam.read()
    frame_shrink = rescaleFrame(frame)
   
    # Face Detection 
    haar_face_cascade = cv.CascadeClassifier('haar_face.xml')
    face_target = haar_face_cascade.detectMultiScale(frame_shrink, scaleFactor=1.1, minNeighbors=3)
    haar_glasses_cascade = cv.CascadeClassifier('haar_glasses.xml')
    glasses_target = haar_glasses_cascade.detectMultiScale(frame_shrink, scaleFactor=1.3, minNeighbors= 5)

    # Test Parameters for glasses detection
    #print(f'Number of faces found = {len(face_target)}')
    #print(f'Number of glasses found = {len(glasses_target)}')

   
    for (x, y, w, h) in face_target:
        cv.rectangle(frame_shrink, (x, y), (x + w, y+ h), (0, 255, 0), thickness=1)
        if len(glasses_target) < 1:
            cv.putText(frame_shrink, 'You are a Nerd!', (x, y + 225), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0,255,0), 2)
        else:
            cv.putText(frame_shrink, 'You are Cool!', (x, y + 225), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0,255,0), 2)

    cv.imshow('Webcam Capture', frame_shrink)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

webcam.release()
cv.destroyAllWindows()