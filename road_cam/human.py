# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
 
# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--images", required=True, help="path to images directory")
#args = vars(ap.parse_args())
 
camera = cv2.VideoCapture(0)

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

counter = 0
	
# loop over the image paths
while True:

	# grab the current frame
	(grabbed, frame) = camera.read()

	if not grabbed:
		print("No input image")
		break
	
	image = frame.copy()
	# load the image and resize it to (1) reduce detection time
	# and (2) improve detection accuracy
	#image = cv2.imread(imagePath)
	image = imutils.resize(image, width=min(400, image.shape[1]))
	orig = image.copy()
 
	# detect people in the image
	(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
		padding=(8, 8), scale=1.05)

	#global counter
	if type(rects) == np.ndarray:
		counter += 1

	if(counter == 30):
		print("Human Detected")
		counter = 0
 
	# draw the original bounding boxes
	#for (x, y, w, h) in rects:
		#cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
 
	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
 
	# draw the final bounding boxes
	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
 
	# show some information on the number of bounding boxes
	#filename = imagePath[imagePath.rfind("/") + 1:]
	#print("[INFO] {}: {} original boxes, {} after suppression".format(filename, len(rects), len(pick)))
 
	# show the output images
	cv2.imshow("After NMS", image)


	# if the `q` key was pressed, break from the loop
	if cv2.waitKey(1) & 0xFF is ord('q'):
		cv2.destroyAllWindows()
		print("Stop programm and close all windows")
		break
