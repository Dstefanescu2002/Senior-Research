import cv2
import numpy as np

def findEdgeCoordinates(edges):
    indices = np.where(edges != [0])
    coordinates = list(zip(indices[0], indices[1]))
    
    return (coordinates)