# import the necessary packages
from __future__ import print_function
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import os
from PIL import Image

imageNum = 0

tree = ET.parse('data-tud/gt/ETH-Person/bahnhof_assc_gt.xml')
root = tree.getroot()

path = 'C:/Users/manas/Desktop/unet/Pytorch-UNet-1.0/YOOO/masks'
path1 = 'C:/Users/manas/Desktop/unet/Pytorch-UNet-1.0/YOOO/masks1'

def imgGen(min, max, img):

    for i in range(min,max):

            if (i < 10):
                cv2.imwrite(os.path.join(path1 , '00000'+ str(i) + '.jpg'), img)
            elif (i >= 10 and i < 100):
                cv2.imwrite(os.path.join(path1 , '0000'+ str(i) + '.jpg'), img)
            elif (i >= 100 and i < 1000):
                cv2.imwrite(os.path.join(path1 , '000'+ str(i) + '.jpg'), img)
            elif (i >= 1000 and i < 10000):
                cv2.imwrite(os.path.join(path1 , '00'+ str(i) + '.jpg'), img)
            else:
                cv2.imwrite(os.path.join(path1 , '0'+ str(i) + '.jpg'), img)

    return

for frame in root:
    for objectlist in frame:

        imageNum += 1

        img = np.zeros([480,640],dtype=np.float32)
        img.fill(0.0) # or img[:] = 255

        for object in objectlist:
            for box in object:
                print(box.tag, box.attrib)
                xc = int(float(box.get("xc")))
                w = int(float(box.get("w"))/2)
                yc = int(float(box.get("yc")))
                h = int(float(box.get("h"))/2)

                for i in range (yc - h, yc + h):
                    for j in range (xc - w, xc + w):
                        img[i][j] = 1.0

        if(imageNum == 1):
            imgGen(1,1001,img)
        # elif(imageNum == 51):
        #     imgGen(51,101,img)
        # elif(imageNum == 101):
        #     imgGen(101,151,img)
        # elif(imageNum == 151):
        #     imgGen(151,201,img)
        # elif(imageNum == 201):
        #     imgGen(201,251,img)
        # elif(imageNum == 251):
        #     imgGen(251,301,img)
        # elif(imageNum == 301):
        #     imgGen(301,351,img)
        # elif(imageNum == 351):
        #     imgGen(351,401,img)
        # elif(imageNum == 401):
        #     imgGen(401,451,img)
        # elif(imageNum == 451):
        #     imgGen(451,501,img)
        # elif(imageNum == 501):
        #     imgGen(501,551,img)
        # elif(imageNum == 551):
        #     imgGen(551,601,img)
        # elif(imageNum == 601):
        #     imgGen(601,651,img)
        # elif(imageNum == 651):
        #     imgGen(651,701,img)
        # elif(imageNum == 701):
        #     imgGen(701,751,img)
        # elif(imageNum == 751):
        #     imgGen(751,801,img)
        # elif(imageNum == 801):
        #     imgGen(801,851,img)
        # elif(imageNum == 851):
        #     imgGen(851,901,img)
        # elif(imageNum == 901):
        #     imgGen(901,951,img)
        # elif(imageNum == 951):
        #     imgGen(951,1001,img)

    



                

