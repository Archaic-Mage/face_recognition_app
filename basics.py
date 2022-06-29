import cv2
import numpy as np
import face_recognition

#load image file and convert to rgb

img_elon = face_recognition.load_image_file('./pictureTraining/Elon-Musk.jpg')
img_elon = cv2.cvtColor(img_elon, cv2.COLOR_BGR2RGB)

#test image

img_elon_test = face_recognition.load_image_file('./pictureTraining/Elon_test.jpg')
img_elon_test = cv2.cvtColor(img_elon_test, cv2.COLOR_BGR2RGB)

#finding face locations - encoding (for matching)

face_loc = face_recognition.face_locations(img_elon)[0]
face_enc = face_recognition.face_encodings(img_elon)[0]

#drawing box to locate the face

cv2.rectangle(img_elon,(face_loc[3],face_loc[0]),(face_loc[1],face_loc[2]),(255,0,255),2)

#encoding for the test image

enc_test = face_recognition.face_encodings(img_elon_test)[0]


#finding results and mismatch ratings

results = face_recognition.compare_faces([face_enc], enc_test)
face_dis = face_recognition.face_distance([face_enc], enc_test)

#printing results on the image

cv2.putText(img_elon_test,f'{results} {round(face_dis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)


#showing the image and test image results

cv2.imshow('Elon Musk', img_elon)
cv2.imshow('Elon Musk T', img_elon_test)
cv2.waitKey(0)