# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
import numpy as np
import imutils
import cv2

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture('vid.mp4') 

imageNum = 0

while True:
	imageNum += 1

	_, image = cap.read()  

	image = imutils.resize(image, width=min(800, image.shape[1]))
	orig = image.copy()

	# detect people in the image
	(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
		padding=(8, 8), scale=1.05)

	# draw the original bounding boxes
	for (x, y, w, h) in rects:
		cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

	# draw the final bounding boxes
	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
		print(str(imageNum) + ",-1" + str(xA) + "," + str(yA) + "," + str(xB) + "," + str(yB) + "," + "-1,-1,-1\n")

	# print some info on the bounding boxes
	print("[INFO]: {} people detected in the frame".format(len(pick)))
	
	# show the output images
	cv2.imshow("peeps detected", image)

	k = cv2.waitKey(30) & 0xff  
	if k==27:
		break  

cap.release()  
