# Importing all required packages
import cv2
import numpy as np
import matplotlib.pyplot as plt
 
 
# Read in the cascade classifiers for face and eyes
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')


# create a function to detect full body
def adjusted_detect_full_body(img):
     
    body_img = img.copy()
     
    body_rect = body_cascade.detectMultiScale(body_img,
                                              scaleFactor = 1.2, 
                                              minNeighbors = 5)
     
    for (x, y, w, h) in body_rect:
        cv2.rectangle(body_img, (x, y), 
                      (x + w, y + h), (255, 255, 255), 10)\
         
    return body_img

# create a function to detect face
def adjusted_detect_face(img):
     
    face_img = img.copy()
     
    face_rect = face_cascade.detectMultiScale(face_img, 
                                              scaleFactor = 1.2, 
                                              minNeighbors = 5)
     
    for (x, y, w, h) in face_rect:
        cv2.rectangle(face_img, (x, y), 
                      (x + w, y + h), (255, 255, 255), 10)\
         
    return face_img
 
 
# create a function to detect eyes
def detect_eyes(img):
     
    eye_img = img.copy()    
    eye_rect = eye_cascade.detectMultiScale(eye_img, 
                                            scaleFactor = 1.2, 
                                            minNeighbors = 5)    
    for (x, y, w, h) in eye_rect:
        cv2.rectangle(eye_img, (x, y), 
                      (x + w, y + h), (255, 255, 255), 10)        
    return eye_img
 
# Reading in the image and creating copies
img = cv2.imread('Leo_Messi_v_Almeria_020314_.jpg')
# img_copy1 = img.copy()
# img_copy2 = img.copy()
# img_copy3 = img.copy()
 
# Detecting the face 
face = adjusted_detect_face(img)
plt.imshow(face)
plt.show()
# Saving the image
cv2.imwrite('face_2.jpg', face)

# Detecting the body
body = adjusted_detect_full_body(img)
plt.imshow(body)
plt.show()
# Saving the image
cv2.imwrite('body_2.jpg', body)