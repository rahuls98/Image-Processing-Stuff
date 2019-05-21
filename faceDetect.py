#IMPORT libraries
import cv2
import numpy as np
import dlib
import time

#LOAD camera frame
cap = cv2.VideoCapture(0) #index for multiple webcams
time.sleep(2)

#LOAD detector
detector = dlib.get_frontal_face_detector()

while True:
    #Get the frame
    _, frame = cap.read() 
    #Convert to gray scale (Reduces computation required for processing)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Use detector on the gray frame 
    faces = detector(gray) 

    for face in faces:
        #print(face) #Returns top-left and right-bottom points
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        #Draw rectangle around face -> (window, positions, color, width)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 3)

    #show the window named 'frame'
    cv2.imshow("Frame",frame) 
    
    #Exit
    key=cv2.waitKey(1)
    if key==27:
        break