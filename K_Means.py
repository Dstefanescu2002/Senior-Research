#USING DOMINANT COLORS TO CROP OUT THE STANDS

import cv2
import numpy as np
import matplotlib.pyplot as plt

def kmeans(image, k):
    """Runs k-means on image

    Args:
        image (Image): Image to run k-means on
        k (Integer): number of centers for k-means

    Returns:
        Tuple: (Image after k-means, centers from k-means algorithm)
    """
    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # define stopping criteria for 100 repeats or epsilon of .2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    #Run k-means 
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # convert back to 8 bit values (No longer floats)
    centers = np.uint8(centers)
    # flatten the labels array (from [[0],[0]...] to [0,0...])
    labels = labels.flatten()
    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]
    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(image.shape)
    return (segmented_image, centers)

def main():
    # read the image
    image = cv2.imread("images/center.jpg")
    # show the image
    #cv2.imshow("k-means", kmeans(image))
    cv2.waitKey(0)
    cv2.destroyAllWindows()