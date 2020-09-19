import cv2

imageURL = "images/center.jpg"
image = cv2.imread(imageURL)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #GRAYSCALE
ret, thresh = cv2.threshold(gray, 127, 255, 0) #THRESH HOLD
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
image = cv2.drawContours(image, contours, -1, (0,0,255), 3)

cv2.imshow("image", image)
cv2.imshow("thresh", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
