#!/usr/bin/python

import numpy as np

import cv2

def nothing(x):
 pass
cap=cv2.VideoCapture(0)
bats=cv2.VideoCapture('flying_bats.mp4')
menu=False
cv2.namedWindow('frame')
cv2.createTrackbar('batman','frame',0,1,nothing)
bats_counter=0
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('haarcascade_eye.xml')
while True:
 ret,frame=cap.read()
 ret2,back_bats=bats.read()
 bats_counter+=1
 original=frame
 cv2.imshow('frame',frame)
 b=cv2.getTrackbarPos('batman','frame')
 gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
 face=face_cascade.detectMultiScale(gray,1.3,5)

 if b==1:
  batman=cv2.imread('/home/shivam/opencv/facemasks/masks/batman.jpg')
  batman=np.array(batman)
  for (x,y,w,h) in face:
   roi_frame=np.array(frame[y-70:y+h-70,x:x+w])
   (p,q,l)=roi_frame.shape
   try:
    batman=cv2.resize(batman,(p,q))
    batman=cv2.bitwise_and(batman,roi_frame)
    frame[y-70:y+h-70,x:x+w]=batman
    if bats_counter==bats.get(cv2.CAP_PROP_FRAME_COUNT):
     bats_counter=0
     bats.set(cv2.CAP_PROP_POS_FRAMES,0)
    frame=cv2.bitwise_and(frame,back_bats)
   except Exception as e:
    cv2.imshow('frame',frame)
    continue
   cv2.imshow('frame',frame)
 else:
  frame[:]=original
 
 k=cv2.waitKey(10) & 0xFF 
 if k==ord('q'):
  break

cap.release()
cv2.destroyAllWindows()
