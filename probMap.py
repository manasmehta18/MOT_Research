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

for frame in root:
    for objectlist in frame:

        imageNum += 1

        img = np.zeros([480,640],dtype=np.uint8)
        img.fill(0) # or img[:] = 255

        for object in objectlist:
            for box in object:
                print(box.tag, box.attrib)
                xc = int(float(box.get("xc")))
                w = int(float(box.get("w"))/2)
                yc = int(float(box.get("yc")))
                h = int(float(box.get("h"))/2)

                for i in range (yc - h, yc + h):
                    for j in range (xc - w, xc + w):
                        img[i][j] = 255

        im = Image.fromarray(img)
        im.save(os.path.join(path , '00000'+ str(imageNum) + '_mask.gif'))

        cv2.imshow("yee", img)
        if (imageNum < 10):
            im.save(os.path.join(path , '00000'+ str(imageNum) + '_mask.gif'))
        elif (imageNum >= 10 and imageNum < 100):
            im.save(os.path.join(path , '0000'+ str(imageNum) + '_mask.gif'))
        elif (imageNum >= 100 and imageNum < 1000):
            im.save(os.path.join(path , '000'+ str(imageNum) + '_mask.gif'))
        elif (imageNum >= 1000 and imageNum < 10000):
            im.save(os.path.join(path , '00'+ str(imageNum) + '_mask.gif'))
        else:
            im.save(os.path.join(path , '0'+ str(imageNum) + '_mask.gif'))



                

