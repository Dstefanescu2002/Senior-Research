import cv2
import numpy as np
from Pre_Processing import kernel, colorThresholding, contouring, findFieldContours, edgeDetection, crop
from Player_Detection import findEdgeCoordinates
import os

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged

#Establish what image to use and read it in
imageURL = "images/mid.jpg"
image = cv2.imread(imageURL)
#image = cv2.resize(image, (900, 600))
cv2.imshow("origional", image)


#cv2.imshow("Contoured kerneled image", contouredImage)
fieldMask = findFieldContours(image,3)
#cv2.imshow("first field mask", fieldMask[1])
croppedImage = crop(image, fieldMask[0], 15)
#cv2.imshow("Second field finder",findFieldContours(croppedImage,2)[1])
#cv2.imshow("Second field finder 2",findFieldContours(croppedImage,3)[1])
#cv2.imshow("cropped image", croppedImage)
edgeImage = edgeDetection(croppedImage)
cv2.imshow("Final Image", edgeImage)

temp = cv2.cvtColor(fieldMask[2], cv2.COLOR_BGR2HSV)
cv2.imshow("Temp", temp)

median = cv2.medianBlur(fieldMask[2],5)
cv2.imshow("Median", median)

autoCannyTest = auto_canny(fieldMask[2])
#autoCannyTestFinal = autoCannyTest ^ fieldMask[3]
cv2.imshow("Auto Canny Test", autoCannyTest)

edgeTest = edgeDetection(fieldMask[2])
edgeTestFinal = edgeTest ^ fieldMask[3]
cv2.imshow("Final edge test 2", edgeTestFinal)
#contouredImage = contouring(croppedImage)
#cv2.imshow("Final Contoured Image", contouredImage[1])
#print (contouredImage[0])
#print (findEdgeCoordinates(edgeImage))

#Closing the windows upon key press
cv2.waitKey(0)
cv2.destroyAllWindows()