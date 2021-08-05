# import the necessary packages
from __future__ import print_function
import numpy as np
import imutils
import cv2
import os

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture('eth.mp4') 

imageNum = 0
path = 'C:/Users/manas/Desktop/unet/Pytorch-UNet-1.0/YOOO/imgs'

while True:

    imageNum += 1
    _, image = cap.read()

    image = imutils.resize(image, width=min(640, image.shape[1]))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if (imageNum < 10):
        cv2.imwrite(os.path.join(path , '00000'+ str(imageNum) + '.jpg'), image)
    elif (imageNum >= 10 and imageNum < 100):
        cv2.imwrite(os.path.join(path , '0000'+ str(imageNum) + '.jpg'), image)
    elif (imageNum >= 100 and imageNum < 1000):
        cv2.imwrite(os.path.join(path , '000'+ str(imageNum) + '.jpg'), image)
    elif (imageNum >= 1000 and imageNum < 10000):
        cv2.imwrite(os.path.join(path , '00'+ str(imageNum) + '.jpg'), image)
    else:
        cv2.imwrite(os.path.join(path , '0'+ str(imageNum) + '.jpg'), image)
        
    k = cv2.waitKey(30) & 0xff  
    if k==27:
        break  

cap.release()  
