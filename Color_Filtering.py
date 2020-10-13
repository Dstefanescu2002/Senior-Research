#USING COLOR THRESHOLDING TO CROP OUT THE STANDS

import cv2
import numpy as np
from matplotlib import pyplot as plt
from Dominant_Color import kmeans

def colorThresholding(image, numCenters):
    """Detects the field of a image of a soccer game using k-means and color thresholding

    Args:
        image (Image): Image of soccer game 
        numCenters (Integer): Number of centers desired for k-means algorithm

    Returns:
        Image: Black and white mask for where the field was found
    """
    kmeansImage, centers = kmeans(image,numCenters)
    #cv2.imshow("Kmeans image", kmeansImage)
    tempValues = [x[1]-x[0]-x[2] for x in centers]
    gp = centers[tempValues.index(max(tempValues))]

    lower_color_bounds = (int(gp[0])-1, int(gp[1])-1, int(gp[2])-1)
    upper_color_bounds = (int(gp[0])+1, int(gp[1])+1, int(gp[2])+1)

    blur = cv2.GaussianBlur(kmeansImage, (15, 15), 10)
    mask = cv2.inRange(blur, lower_color_bounds, upper_color_bounds)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return (mask_rgb)
    

def kernel(mask_rgb, image):
    """Kerneling algorithm for reducing noise caused by stands

    Args:
        mask_rgb (Image): Black and white mask
        image (Image): Origional image of soccer game

    Returns:
        Tuple: (Black and white mask after kerneling, RGB mask after kerneling using mask and image)
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)) #Learn this too
    opened_mask = cv2.morphologyEx(mask_rgb, cv2.MORPH_OPEN, kernel) #IDK learn what this does
    thresh_color = image & opened_mask #Runs a bitwise and
    return (opened_mask, thresh_color)
    

def contouring(opened_mask):
    """Runs a contouring algorithm on image

    Args:
        opened_mask (Image): RGB image mask

    Returns:
        Tuple: (countour array, Image with contours)
    """
    thresh = cv2.cvtColor(opened_mask, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #Find contours
    contourImage = cv2.drawContours(thresh, contours, -1, (0,0,255), 3) #Draws them on origional image
    return (contours, contourImage)

def crop(image, contours, cushion):
    """Crops image according to highest and lowest contours

    Args:
        image (Image): Image to be cropped
        contours (List): List of contours 
        cushion (Integer): Desired cushion around 

    Returns:
        Image: Cropped image
    """
    xVals = []
    yVals = []
    for cont in contours:
        for point in cont:
            xVals.append(point[0][0])
            yVals.append(point[0][1])
    xMax = min([max(xVals) + cushion, image.shape[1]])
    yMax = min([max(yVals) + cushion, image.shape[0]])
    xMin = max([min(xVals) - cushion, 0])
    yMin = max([min(yVals) - cushion, 0])       
    croppedImage = image[yMin:yMax, xMin:xMax]
    return (croppedImage)

def edgeDetection(image):
    """Runs canny edge detection on image after grayscaling it

    Args:
        image (Image): Image to run edge detection on

    Returns:
        Image: Black and white edges image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    edgesBW = cv2.Canny(gray, 100, 200)
    return (edgesBW)

def findFieldContours(image, numCenters):
    """Finds field using colorThresholding, kernel, and contouring

    Args:
        image (Image): Image to be processed
        numCenters (Integer): Number of desired centers for k-means

    Returns:
        Tuple: (Contours giving location of field, RGB mask of field)
    """
    #cv2.imshow("Origional", image)
    CTImage = colorThresholding(image,numCenters)
    #cv2.imshow("Color Thresholded Image", CTImage)
    openedMask, kernalColorImage = kernel(CTImage, image)
    #cv2.imshow("Black and White mask", openedMask)
    #cv2.imshow("Colored image after kerneling", kernalColorImage)
    contours, contouredImage = contouring(openedMask)
    return (contours, kernalColorImage)
