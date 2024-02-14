import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import random

print("Reading image")
img = cv.imread('img4.jpeg', cv.IMREAD_GRAYSCALE)
print("Image width and height ",img.shape)

width = img.shape[1]
height = img.shape[0]
nh = 0
nw = 0
if height%100 >=40:
    nh += int(height//100)
    nh += 1
    height = 100 * (nh)

if width%100 >=40:
    nw += int(width//100)
    print("nw now is ",nw)
    nw += 1
    width = 100 * (nw)

print("Width and height after change is ",width, "   ", height)

print("Image read from hard disk")
assert img is not None, "file could not be read, check with os.path.exists()"

edges = cv.Canny(img,50,800)
print("Edge detection algorithm done")

contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

contours_poly = [None]*len(contours)
boundRect = [None]*len(contours)
centers = [None]*len(contours)
radius = [None]*len(contours)
drawn_rectangles = []

for i,c in enumerate(contours):
    contours_poly[i] = cv.approxPolyDP(c, 3, True)
    boundRect[i] = cv.boundingRect(contours_poly[i])
    centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])

drawing = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)

print("Len of contours ",len(contours))

for i in range(len(contours)):
    # color = (random.randint(0,256), random.randint(0, 256), random.randint(0,256))
    color = (252, 186, 3)
    cv.drawContours(drawing, contours_poly, i, color)
    # print(f"Bounding rectangle parameters at index {i} :  {boundRect[i]}")
    if (int(boundRect[i][0]) < 500) and (int(boundRect[i][2]) >= 50 or int(boundRect[i][3]) >= 50):
    # if int(boundRect[i][2]) >= 50 or int(boundRect[i][3]) >= 50:
        cv.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
                        (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color=(255,255,255), thickness=2)
        drawn_rectangles.append(boundRect[i])
    #cv.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)
    # cv.imshow('Contours', drawing)
    # cv.waitKey(0)

# print("\nRectangles drawn are ",drawn_rectangles)

rectangles_height = []

for rectangle in drawn_rectangles:
    h = ((rectangle[3] - rectangle[1])**2)**0.5
    rectangles_height.append(h)

print("\nHeight of rectangles --- ",rectangles_height)
print("\nHeight of object ",sum(rectangles_height))

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([i * 100 for i in range(nw)]), plt.yticks([j * 100 for j in range(nh)])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([i * 100 for i in range(nw)]), plt.yticks([j * 100 for j in range(nh)])
#plt.subplot(123),plt.imshow(drawing,cmap='gray')
#plt.title('Contours'), plt.xticks([i * 100 for i in range(nw)]), plt.yticks([j * 100 for j in range(nh)])
plt.show()

cv.imshow('Contours', drawing)
cv.waitKey()