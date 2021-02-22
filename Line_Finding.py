import cv2
import math
import numpy as np
import statistics
import matplotlib.pyplot as plt

def trial(img1):
    #img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)         # queryImage
    img2 = cv2.imread('images/field.jpg',cv2.IMREAD_GRAYSCALE) # trainImage
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = cv2.DrawMatchesFlags_DEFAULT)
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    plt.imshow(img3,),plt.show()

def trial2(img1):
    #img1 = cv.imread('box.png',cv.IMREAD_GRAYSCALE)          # queryImage
    img2 = cv2.imread('images/field.jpg',cv2.IMREAD_GRAYSCALE) # trainImage
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3),plt.show()

def trial3(img1):
    #img1 = cv.imread('box.png',cv.IMREAD_GRAYSCALE)          # queryImage
    img2 = cv2.imread('images/field.jpg',cv2.IMREAD_GRAYSCALE) # trainImage
    # Initiate ORB detector
    orb = cv2.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3),plt.show()

def findLines(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,200,apertureSize = 3)
    tempImage = image.copy()
    
    lines = cv2.HoughLines(edges,1,np.pi/180,150)
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(tempImage,pt1,pt2,(0,0,255),2)
    
    newLines = segment_by_angle_kmeans(lines)
    for i in range(0, len(newLines[0])):
        rho = newLines[0][i][0][0]
        theta = newLines[0][i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(tempImage,pt1,pt2,(0,255,0),2)

    temp1 = [list(x[0]) for x in newLines[0]]
    temp2 = [list(x[0]) for x in newLines[1]]

    av1 = (statistics.median([x[0] for x in temp1]), statistics.median([x[1] for x in temp1]))
    av2 = (statistics.median([x[0] for x in temp2]), statistics.median([x[1] for x in temp2]))

    rho = av1[0]
    theta = av1[1]
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    cv2.line(tempImage,pt1,pt2,(255,255,0),2)

    rho = av2[0]
    theta = av2[1]
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    cv2.line(tempImage,pt1,pt2,(255,255,0),2)

    points1 = [intersection([av1], [[av2[0]+20,av2[1]]]), 
        intersection([av1], [[av2[0]-20,av2[1]]]), 
        intersection([av2], [[av1[0]+20,av1[1]]]),
        intersection([av2], [[av1[0]-20,av1[1]]])]
    points1 = order_points(np.asarray([(x[0][0], x[0][1]) for x in points1], dtype=np.float32))
    points2 = np.asarray([(512,364), (532, 384), (512, 404), (492, 384)], dtype=np.float32)

    print (points1)
    M = cv2.getPerspectiveTransform(points1, points2)
    image = findFinalLines(image, M)

    intersections = segmented_intersections(newLines)
    for i in intersections:
        cv2.circle(tempImage, (i[0][0],i[0][1]), radius=3, color=(255, 0, 0), thickness=3)

    cv2.imshow("Gray", tempImage)
    return (image)

def findFinalLines(img, M):
    image = img.copy()
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,200,apertureSize = 3)
    cv2.imshow("edges", edges)
    #lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, None, 100, 50)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,threshold = 100,minLineLength = 100,maxLineGap = 10)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
    cv2.imshow("FinalLines", image)

    image = cv2.warpPerspective(image, M, (1024, 768))
    thresh = cv2.inRange(image, (255,0,0), (255,0,0))
    thresh = cv2.bitwise_not(thresh)
    thresh = cv2.medianBlur(thresh, 7)
    cv2.imshow("FinalLines", thresh)
    trial3(thresh)
    return (image)


def findCircles(image):
    # Convert to gray-scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Blur the image to reduce noise
    image_blur = cv2.medianBlur(gray, 5)
    cv2.imshow("asd asd ", image_blur)
    # Apply hough transform on the image
    circles = cv2.HoughCircles(image_blur, cv2.HOUGH_GRADIENT, 1, image.shape[0]/64, param1=200, param2=10, minRadius=5, maxRadius=30)
    # Draw detected circles
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw outer circle
            cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw inner circle
            cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
    return (image)

from collections import defaultdict
def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on the unit circle 
    to segment `k` angles inside `lines`.
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in zip(range(len(lines)), lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented

def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]


def segmented_intersections(lines):
    """Finds the intersections between groups of lines."""

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2)) 

    return intersections

def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	return rect