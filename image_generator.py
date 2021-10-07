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
path1 = 'C:/Users/manas/Desktop/unet/Pytorch-UNet-1.0/YOOO/imgs1'

def imgGen(i, img):

    if (i < 10):
        for k in range(0,10):
            for j in range(0,10):
                cv2.imwrite(os.path.join(path1 , '000'+ str(k) + str(j) + str(i) + '.jpg'), img)
    elif (i == 10):
        for k in range(0,10):
            for j in range(0,10):
                if(not(k == 0 and j == 0)):
                    cv2.imwrite(os.path.join(path1 , '00'+ str(k) + str(j) + '0.jpg'), img)

        cv2.imwrite(os.path.join(path1 , '001000.jpg'), img)
        
    # elif (i >= 10 and i < 100):
    #     cv2.imwrite(os.path.join(path1 , '0000'+ str(i) + '.jpg'), img)
    # elif (i >= 100 and i < 1000):
    #     cv2.imwrite(os.path.join(path1 , '000'+ str(i) + '.jpg'), img)
    # elif (i >= 1000 and i < 10000):
    #     cv2.imwrite(os.path.join(path1 , '00'+ str(i) + '.jpg'), img)
    # else:
    #     cv2.imwrite(os.path.join(path1 , '0'+ str(i) + '.jpg'), img)

    return

while True:

    imageNum += 1
    _, image = cap.read()

    image = imutils.resize(image, width=min(640, image.shape[1]))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    imgGen(imageNum, image)

    # if(imageNum == 1):
    #     imgGen(1,1001,image)
    # elif(imageNum == 51):
    #     imgGen(51,101,image)
    # elif(imageNum == 101):
    #     imgGen(101,151,image)
    # elif(imageNum == 151):
    #     imgGen(151,201,image)
    # elif(imageNum == 201):
    #     imgGen(201,251,image)
    # elif(imageNum == 251):
    #     imgGen(251,301,image)
    # elif(imageNum == 301):
    #     imgGen(301,351,image)
    # elif(imageNum == 351):
    #     imgGen(351,401,image)
    # elif(imageNum == 401):
    #     imgGen(401,451,image)
    # elif(imageNum == 451):
    #     imgGen(451,501,image)
    # elif(imageNum == 501):
    #     imgGen(501,551,image)
    # elif(imageNum == 551):
    #     imgGen(551,601,image)
    # elif(imageNum == 601):
    #     imgGen(601,651,image)
    # elif(imageNum == 651):
    #     imgGen(651,701,image)
    # elif(imageNum == 701):
    #     imgGen(701,751,image)
    # elif(imageNum == 751):
    #     imgGen(751,801,image)
    # elif(imageNum == 801):
    #     imgGen(801,851,image)
    # elif(imageNum == 851):
    #     imgGen(851,901,image)
    # elif(imageNum == 901):
    #     imgGen(901,951,image)
    # elif(imageNum == 951):
    #     imgGen(951,1001,image)


    k = cv2.waitKey(30) & 0xff  
    if k==27:
        break  

cap.release()




