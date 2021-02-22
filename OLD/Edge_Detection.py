#USING EDGE DETECTION

import cv2
import numpy as np
from matplotlib import pyplot as plt

#Establish what image to use and read it in
imageURL = "images/center.jpg"
image = cv2.imread(imageURL)

#Grayscaling and thresholding
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  #GRAYSCALE the origional image
ret, thresh = cv2.threshold(gray, 150, 255, 0)  #THRESHOLD the grayscale image

#Contouring on thresholded image
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #Find contours
contourImage = cv2.drawContours(image, contours, -1, (0,0,255), 3) #Draws them on origional image
cv2.imshow("origional image with contours with contours from threshold", contourImage) #Display image

#Canny edge detection on the origional image
edges = cv2.Canny(image,100,200)
cv2.imshow("Edges on origional", np.hstack([edges]))

#Canny edge detection on grayscaled image
edgesBW = cv2.Canny(gray, 100, 200)
cv2.imshow("Edges on grayscale", np.hstack([edgesBW]))

#Canny edge detection on threshold image
edgesTH = cv2.Canny(thresh, 100, 200)
cv2.imshow("Edges on threshold", np.hstack([edgesTH]))

#Closing the windows upon key press
cv2.waitKey(0)
cv2.destroyAllWindows()
