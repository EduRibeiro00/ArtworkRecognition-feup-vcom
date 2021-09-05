import numpy as np
import cv2 as cv
import glob
import math
import random
from matplotlib import pyplot as plt
# from scipy.optimize import leastsq
from skspatial.objects import Plane, Points
from skspatial.plotting import plot_3d

#-------------------------# 
#   HOUGH LINES BUNDLER   #
#-------------------------#
class HoughBundler:
    # Based on: https://stackoverflow.com/questions/45531074/how-to-merge-lines-after-houghlinesp

    def get_slope(self, line):
        # https://en.wikipedia.org/wiki/Atan2
        slope = math.atan2(abs((line[0] - line[2])), abs((line[1] - line[3])))
        return math.degrees(slope)

    def is_different_line(self, line_new, groups, min_distance_to_merge, min_angle_to_merge):
        for group in groups:
            for line_old in group:
                if self.get_distance_between_lines(line_old, line_new) < min_distance_to_merge:
                    slope_new = self.get_slope(line_new)
                    slope_old = self.get_slope(line_old)
                    if abs(slope_new - slope_old) < min_angle_to_merge:
                        group.append(line_new)
                        return False
        return True

    def point_to_line_dist(self, point, line):
        # http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/source.vba
        px, py = point
        x1, y1, x2, y2 = line

        def get_line_length(x1, y1, x2, y2):
            line_length = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
            return line_length

        line_length = get_line_length(x1, y1, x2, y2)
        if line_length < 0.00000001:
            point_to_line_dist = 9999
            return point_to_line_dist

        u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
        u = u1 / (line_length * line_length)

        if (u < 0.00001) or (u > 1):
            # closest point does not fall within the line segment, take the shorter distance to an endpoint
            ix = get_line_length(px, py, x1, y1)
            iy = get_line_length(px, py, x2, y2)
            if ix > iy:
                point_to_line_dist = iy
            else:
                point_to_line_dist = ix
        else:
            # intersecting point is on the line, use the formula
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            point_to_line_dist = get_line_length(px, py, ix, iy)

        return point_to_line_dist

    def get_distance_between_lines(self, line1, line2):
        dist1 = self.point_to_line_dist(line1[:2], line2)
        dist2 = self.point_to_line_dist(line1[2:], line2)
        dist3 = self.point_to_line_dist(line2[:2], line1)
        dist4 = self.point_to_line_dist(line2[2:], line1)

        return min(dist1, dist2, dist3, dist4)

    def merge_lines_pipeline(self, lines):
        groups = []  # all lines groups are here

        # Parameters to play with
        min_distance_to_merge = 30
        min_angle_to_merge = 30

        # first line will create new group every time
        groups.append([lines[0]])

        # if line is different from existing gropus, create a new group
        for line_new in lines[1:]:
            if self.is_different_line(line_new, groups, min_distance_to_merge, min_angle_to_merge):
                groups.append([line_new])

        return groups

    def merge_lines_segments(self, lines):
        # Sort lines cluster and return first and last coordinates
        slope = self.get_slope(lines[0])

        # special case
        if(len(lines) == 1):
            return [lines[0][:2], lines[0][2:]]

        # [[1,2,3,4],[]] to [[1,2],[3,4],[],[]]
        points = []
        for line in lines:
            points.append(line[:2])
            points.append(line[2:])
        # if vertical
        if 45 < slope < 135:
            #sort by y
            points = sorted(points, key=lambda point: point[1])
        else:
            #sort by x
            points = sorted(points, key=lambda point: point[0])

        # return first and last point in sorted group
        # [[x,y],[x,y]]
        return [points[0], points[-1]]

    def process_lines(self, lines):
        lines_x = []
        lines_y = []
        # for every line of cv.HoughLinesP()
        for line_i in [l[0] for l in lines]:
                slope = self.get_slope(line_i)
                # if vertical
                if 45 < slope < 135:
                    lines_y.append(line_i)
                else:
                    lines_x.append(line_i)

        lines_y = sorted(lines_y, key=lambda line: line[1])
        lines_x = sorted(lines_x, key=lambda line: line[0])
        merged_lines_all = []

        # for each cluster in vertical and horizontal lines leave only one line
        for i in [lines_x, lines_y]:
                if len(i) > 0:
                    groups = self.merge_lines_pipeline(i)
                    merged_lines = []
                    for group in groups:
                        merged_lines.append(self.merge_lines_segments(group))

                    merged_lines_all.extend(merged_lines)

        return merged_lines_all


#-------------------------# 
#        EXTRINSIC        #
#-------------------------#

# pattern size
pattern = ((9,6))

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((pattern[0]*pattern[1],3), np.float32)
objp[:,:2] = np.mgrid[0:pattern[0],0:pattern[1]].T.reshape(-1,2) * 2.5
axis = np.float32([[2.5,0,0], [0,6.5,0], [0,0,-2.5]]).reshape(-1,3)

# load camera intrinsic parameters
with np.load('intrinsicParams.npz') as intrinsicParams:
    mtx, dist = [intrinsicParams[i] for i in ('camera_matrix', 'dist_coeffs')]

# find chessboard pose using solvePnP
extrinsicCalibImage = cv.imread('images/small_res/chess3.jpg')
grayextrinsicCalibImage = cv.cvtColor(extrinsicCalibImage, cv.COLOR_BGR2GRAY)
ret, corners = cv.findChessboardCorners(grayextrinsicCalibImage, pattern, None)
if ret != True:
    print("Failed to calibrate extrinsic parameters")
    exit()
corners2 = cv.cornerSubPix(grayextrinsicCalibImage, corners, (11,11), (-1,-1), criteria)  
ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)

# load xiaomi box img
img = cv.imread('images/small_res/test36.jpg')

# undistort
# h, w = img.shape[:2]
# newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
# img = cv.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
# x, y, w, h = roi
# img = img[y:y+h, x:x+w]

# subtract images
# img_bg = cv.imread('images/small_res/test37.jpg')
# sub = cv.subtract(img_bg,img)
#cv.imshow('Subtraction',cv.resize(sub, (800, 800)))
#cv.waitKey()

# detect edges
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
filtered = cv.bilateralFilter(gray, 30, 50, 50)
blur = cv.blur(filtered, (3,3))
edges = cv.Canny(blur, 20, 30)

# line enhancement
kernel = np.ones((3,3), np.uint8)
closing = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel, iterations=15)

### Hough detection ###
rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 100  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 150  # minimum number of pixels making up a line
max_line_gap = 100  # maximum gap in pixels between connectable line segments

# "lines" is an array containing endpoints of detected line segments
lines = cv.HoughLinesP(closing, rho, theta, threshold, np.array([]),
    min_line_length, max_line_gap)

# filter unwanted lines
new_lines = []
for line in lines:
    for x1,y1,x2,y2 in line:
        if not y1 > 500 and abs(y1-y2) < 200:
            new_lines.append(line)
lines = new_lines

# bundle repeated and/or staggered lines
line_bundler = HoughBundler()
bundled_lines = line_bundler.process_lines(lines)

# draw resulting lines
imgBundled = np.copy(img) * 0
for line in bundled_lines:
    x1, y1, x2, y2 = line[0][0], line[0][1], line[1][0], line[1][1]
    cv.line(imgBundled, (x1, y1), (x2, y2), (random.random() * 255,random.random() * 255,random.random() * 255),2)

### TEMPORARY ###
def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
img = draw(img,corners2,imgpts)
### /TEMPORARY ###

### Display all results ###
#cv.imshow('', np.hstack([cv.resize(edges, (1000, 1000)), cv.resize(closing, (1000, 1000))]))
cv.imshow('Edge detection', np.hstack([cv.resize(img, (800, 800)), cv.resize(imgBundled, (800, 800))]))
cv.waitKey()


#-------------------------# 
#     3D CALCULATION      #
#-------------------------#

# calculate projection matrix (projMtx) -> 3D to 2D
rmtx, _ = cv.Rodrigues(rvecs)
rotTransMtx = np.zeros((3,4))
rotTransMtx[:,:-1] = rmtx
rotTransMtx[:,-1:] = tvecs
projMtx = np.dot(mtx, rotTransMtx)

# 3D -> 2D
def world_to_image(x, y, z):
    res = np.dot(projMtx, [[x],[y],[z],[1]])
    return res/res[2]

# 2D -> 3D
def image_to_world(matrix, i, j):
    res = np.linalg.solve(matrix, [[i], [j], [1], [0]])
    [[x], [y], [z], [_]] = res/res[3]
    return [x, y, z]


###############
# TODO: assumming the hough lines returns 3 lines:
#       left (on the floor)
#       middle (on the object)
#       right (on the floor)

def get_pnts_from_line(line):
    [[x1, y1, x2, y2]] = line
    slope = (y2 - y1) / (x2 - x1)
    b = y1 - x1 * slope
    pnts = []
    for i in range(x1, x2 + 1):
        j = i * slope + b
        pnts.append([i, j])
    return pnts


# calculate inverse matrix (inverseMtx) -> 2D to 3D
OBJ_HEIGHT = 4
# initialize list for least squares
obj_points_shadow = []

# we know lines on the left and right are on the plane z = 0´
inverseMtxFloor = np.zeros((4,4))
inverseMtxFloor[:-1,:] = projMtx
inverseMtxFloor[-1:,:] = [[0, 0, 1, 0]]

# left line
# points_line_0 = get_pnts_from_line(lines[0])
# for i, j in points_line_0:
#     obj_points_shadow.append(image_to_world(inverseMtxFloor, i, j))

[[i1, j1, i2, j2]] = lines[0]
obj_points_shadow.append(image_to_world(inverseMtxFloor, i1, j1))
obj_points_shadow.append(image_to_world(inverseMtxFloor, i2, j2))

# right line
# points_line_2 = get_pnts_from_line(lines[2])
# for i, j in points_line_2:
#     obj_points_shadow.append(image_to_world(inverseMtxFloor, i, j))


[[i1, j1, i2, j2]] = lines[2]
obj_points_shadow.append(image_to_world(inverseMtxFloor, i1, j1))
obj_points_shadow.append(image_to_world(inverseMtxFloor, i2, j2))

# we know the center line is on the plane z = OBJ_HEIGHT
inverseMtxObj = np.zeros((4,4))
inverseMtxObj[:-1,:] = projMtx
inverseMtxObj[-1:,:] = [[0, 0, 1, OBJ_HEIGHT]]

# points_line_1 = get_pnts_from_line(lines[1])
# for i, j in points_line_1:
#     obj_points_shadow.append(image_to_world(inverseMtxObj, i, j))

[[i1, j1, i2, j2]] = lines[1]
obj_points_shadow.append(image_to_world(inverseMtxObj, i1, j1))
obj_points_shadow.append(image_to_world(inverseMtxObj, i2, j2))


############################
# least_squares method (start)
############################

# initial guess
# initial_plane = [0, 1, 0, -6.5]

# # with the points, use least_squares to calculate plane equation
# def f_min(X, p):
#     plane_xyz = p[0:3]
#     distance = (plane_xyz * X).sum(axis=1) + p[3]
#     return distance / np.linalg.norm(plane_xyz)

# def residuals(params, signal, X):
#     return f_min(X, params)

# final_shadow_plane = leastsq(residuals, initial_plane, args=(None, np.array(obj_points_shadow)))[0]

############################
# least_squares method (end)
############################

############################
# ransac method (start)
############################

plane_points = Points(obj_points_shadow)
plane = Plane.best_fit(plane_points)

# plot_3d(
#     plane_points.plotter(c='k', s=50, depthshade=False),
#     plane.plotter(alpha=0.2, lims_x=(-5, 5), lims_y=(-5, 5)),
# )
# cv.waitKey()
# exit(0)

final_shadow_plane = plane.cartesian()

print(f'A: {final_shadow_plane[0]}')
print(f'B: {final_shadow_plane[1]}')
print(f'C: {final_shadow_plane[2]}')
print(f'D: {final_shadow_plane[3]}')

############################
# ransac method (end)
############################

# calculate final matrix for 2D -> 3D conversion, now with the shadow plane's equations
inverseMtxFinal = np.zeros((4,4))
inverseMtxFinal[:-1,:] = projMtx
inverseMtxFinal[-1:,:] = [final_shadow_plane]

[[_, _, IMAGE_I, IMAGE_J]] = lines[1]
# IMAGE_I = 773
# IMAGE_J = 376

[ex_x, ex_y, ex_z] = image_to_world(inverseMtxObj, IMAGE_I, IMAGE_J)

[x, y, z] = image_to_world(inverseMtxFinal, IMAGE_I, IMAGE_J)
print(f'Image: ({IMAGE_I}, {IMAGE_J})')
print(f'Expected: ({ex_x}, {ex_y}, {ex_z})')
print(f'Calculated: ({x}, {y}, {z})')

#-------------------------# 
#      PLOT RESULTS       #
#-------------------------#

# scatterplot to display height of all pixels on the line
# EDU
pixel_hor = []
pnt_height = []
for line in lines:
    [[x1, y1, x2, y2]] = line
    slope = (y2 - y1) / (x2 - x1)
    b = y1 - x1 * slope
    for i in range(x1, x2 + 1):
        j = i * slope + b
        [[_], [_], [h], [_]] = image_to_world(inverseMtxFinal, i, j)
        pixel_hor.append(i)
        pnt_height.append(-h)

# HENRIQUE
# pixel_hor = []
# pnt_height = []
# for i in range (0, 6000, 50):
#     height = None
#     for line in lines:
#         points = line[0]
#         if points[0] < i and points[2] > i:
#             slope = (points[3]-points[1])/(points[2]-points[0])
#             b = points[3]-slope*points[2]
#             j = slope*i+b
#             [[_], [_], [height], [_]] = image_to_world(i, j)
#             break
#     if height != None:
#         pixel_hor.append(i)
#         pnt_height.append(height)

plt.scatter(pixel_hor, pnt_height)
plt.show()
cv.waitKey()
