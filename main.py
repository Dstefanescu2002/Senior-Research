import cv2
import numpy as np
from Color_Filtering import kernel, colorThresholding, contouring, findFieldContours, edgeDetection, crop

#Establish what image to use and read it in
imageURL = "images/mid.jpg"
image = cv2.imread(imageURL)
image = cv2.resize(image, (900, 600))
cv2.imshow("origional", image)


#cv2.imshow("Contoured kerneled image", contouredImage)
fieldMask = findFieldContours(image,3)
cv2.imshow("first field mask", fieldMask[1])
croppedImage = crop(image, fieldMask[0], 15)
cv2.imshow("Second field finder",findFieldContours(croppedImage,2)[1])
cv2.imshow("Second field finder 2",findFieldContours(croppedImage,3)[1])
cv2.imshow("cropped image", croppedImage)
edgeImage = edgeDetection(croppedImage)
cv2.imshow("Final Image", edgeImage)

#Closing the windows upon key press
cv2.waitKey(0)
cv2.destroyAllWindows()