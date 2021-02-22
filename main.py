import cv2
import numpy as np
import Pre_Processing
import YOLO.yolo
import Line_Finding
import os
import matplotlib.pyplot as plt

#Establish what image to use and read it in
imageURL = "images/Capture2.jpg"
image = cv2.imread(imageURL)
image = cv2.resize(image, (1024,768))

preprocessedImages = Pre_Processing.preprocess(image,3)
playerImage = preprocessedImages[0]
lineImage = preprocessedImages[1]
playerImage = playerImage[4:764, 4:1020]
#cv2.imshow("Player Image", playerImage)
lineImage = lineImage[4:764, 4:1020]
#cv2.imshow("Line Image", lineImage)

lines = Line_Finding.findLines(playerImage)
cv2.imshow("Lines", lines)
#circles = Line_Finding.findCircles(lineImage)
#cv2.imshow("Circles", circles)

players = YOLO.yolo.findPlayers(playerImage)
cv2.imshow("Players", players)

#Closing the windows upon key press
cv2.waitKey(0)
cv2.destroyAllWindows()

import os, signal
os.kill (os.getpid(), signal.SIGTERM)