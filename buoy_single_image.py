#!/usr/bin/env python3
# coding: utf-8

# Call with buoy_single_image.sh

import numpy as np
import matplotlib.pyplot as plt
import re
import cv2
import sys
import imageio
from PIL import Image
from skimage import data, color, transform
from skimage.io import imread
from skimage.util import img_as_ubyte
from skimage.draw import circle_perimeter
from skimage.feature import peak_local_max, canny
from skimage.transform import hough_circle, hough_circle_peaks

# ARGUMENTS FROM THE USER
images_dir = sys.argv[1]
min_circle_size = int(sys.argv[2])
max_circle_size = int(sys.argv[3])
ref_coords = sys.argv[4]
colour = sys.argv[5]
circle_sens = float(sys.argv[6])
name_scale = (sys.argv[7])
waterline_threshold_min = int(sys.argv[8])
enhance = sys.argv[9]
filename = sys.argv[10]
name = sys.argv[11]

# read reference coordinates
csv = np.genfromtxt (ref_coords, delimiter=",")
fixed = csv[:,0:2].copy()
no_circles_ref = len(fixed)
no_rows_ref = int(no_circles_ref/3)

# Load picture and detect edges
if enhance == 't':
    image_clr = imread(images_dir + filename)
else:
    image_clr = imread(images_dir + filename)[..., 0]

print('extracting edges')
edges = canny(image_clr, sigma=2, low_threshold=5, high_threshold=50)

# plot edges
plt.title('Edges - ' + filename)
fig=plt.imshow(edges)
fig.figure.savefig(images_dir + name + '_edges.png', bbox_inches='tight', dpi=1000)

# Detect circles
print('detecting circles')
hough_radii = np.arange(min_circle_size, max_circle_size)
hough_res = hough_circle(edges, hough_radii)
accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, min_xdistance=min_circle_size, min_ydistance=min_circle_size, total_num_peaks=80, normalize=False)

# Print scatterplot of detected radii
plt.close()
colour_map = plt.get_cmap('rainbow_r')
fig = plt.scatter(cx, -(cy), c=accums, cmap=colour_map)
plt.gca().set_aspect('equal')
cbar = plt.colorbar()
cbar.set_label('goodness of circle fit')
plt.title('min radius: %d, max radius: %d\n single best fit radius: %d' %(np.nanmin(radii), np.nanmax(radii), radii[0]))
plt.tight_layout()
# plt.show()
fig.figure.savefig(images_dir + name + '_radii.png', bbox_inches='tight')

# delete circles that are not the scale by
#   1) delete all below 'circle fit' sensitivity threshold
#   2) delete all with radius more than +- 2 pixels of median 5 best fit circles
#   3) delete all more than 20 mm in x from median of remaining circles
#   4) delete all more than 80 mm in y from median of remaining circles

radii = radii[accums > circle_sens]
cx = cx[accums > circle_sens]
cy = cy[accums > circle_sens]

radii_median = np.median(radii[0:18])
inside = np.logical_and(radii < (radii_median + 5), radii > (radii_median - 5))
index = np.where(inside)[0]
radii = radii[index]
cx = cx[index]
cy = cy[index]

# find pixel size in mm (radius = 2.5 mm)
pix_size = radii_median / 2.5

cx_median = np.median(cx[0:18])
inside = np.logical_and(cx < (cx_median + pix_size * 15), cx > (cx_median - pix_size * 15))
index = np.where(inside)[0]
radii = radii[index]
cx = cx[index]
cy = cy[index]

cy_median = np.median(cy)
inside = np.logical_and(cy < (cy_median + pix_size * 80), cy > (cy_median - pix_size * 80))
index = np.where(inside)[0]
radii = radii[index]
cx = cx[index]
cy = cy[index]

# highest circle might be the screw... remove it if it is
# find indices of highest two cy values
check_screw = cy.argsort()[0:2]
# if the difference between the two is greater than 10 mm, remove the highest one
if (cy[check_screw[1]] - cy[check_screw[0]]) > (pix_size * 10):
    print("removing highest circle")
    radii = radii[cy.argsort()[1:]]
    cx = cx[cy.argsort()[1:]]
    cy = cy[cy.argsort()[1:]]

# remove lowest circles if the number of circles is not a multiple of 3
if len(radii) % 3 == 2:
    radii = radii[cy.argsort()][0:-2]
    cx = cx[cy.argsort()][0:-2]
    cy = cy[cy.argsort()][0:-2]

if len(radii) % 3 == 1:
    radii = radii[cy.argsort()][0:-1]
    cx = cx[cy.argsort()][0:-1]
    cy = cy[cy.argsort()][0:-1]

# Print detected circles on grey image
orig_image = re.sub('\_' + colour + '.png$', '', filename)
# print(orig_image)

image_plot = color.gray2rgb(imread(images_dir + orig_image + '.png')[..., 0])

for center_y, center_x, radius in zip(cy, cx, radii):
    circy, circx = circle_perimeter(center_y, center_x, radius, shape=image_plot.shape)
    image_plot[circy, circx] = (220, 20, 20)

plt.close()
plt.title('Circle Detection - ' + filename)
fig = plt.imshow(image_plot)
# plt.show()
fig.figure.savefig(images_dir + orig_image + '_circles.png', bbox_inches='tight', dpi=1000)

# load grey image
image_grey = color.gray2rgb(imread(images_dir + orig_image + '.png')[..., 0])

print('defining transformation')
# determine our transformation to rectify the image
tform_resids_stdev = 10
src_points = np.vstack((cx, cy, radii)).T
cut = 0

while tform_resids_stdev > 3:
    no_circles = int(len(src_points))
    no_rows = int(no_circles / 3)
    # print('no rows',  no_rows, 'type:', type(no_rows))
    print("number of circles detected:", no_circles)
    fixed = csv[:, 0:2].copy()
    first_col = fixed[0:no_rows, 0:2].copy()
    second_col = fixed[no_rows_ref * 1:no_rows_ref * 1 + no_rows, 0:2].copy()
    third_col = fixed[no_rows_ref * 2:no_rows_ref * 2 + no_rows, 0:2].copy()
    fixed = np.vstack((first_col, second_col, third_col))

    # create array of centre points and sort by y then x
    col_1 = src_points[src_points[:, 0].argsort()][0:no_rows, :]
    col_1 = col_1[col_1[:, 1].argsort()]
    col_2 = src_points[src_points[:, 0].argsort()][no_rows: no_rows * 2, :]
    col_2 = col_2[col_2[:, 1].argsort()]
    col_3 = src_points[src_points[:, 0].argsort()][no_rows * 2: no_rows * 3, :]
    col_3 = col_3[col_3[:, 1].argsort()]

    sorted_all = np.vstack((col_1, col_2, col_3))
    src_points = sorted_all[:, 0:2]
    dst_points = fixed * 10

    tform = transform.ProjectiveTransform()
    tform.estimate(dst_points, src_points)

    tform_resids = ((tform.inverse(src_points)) - dst_points)
    tform_resids_stdev = np.std(tform_resids)

    print('tform_resid', tform_resids_stdev)

    # remove the bottom three circles
    src_points = src_points[src_points[:, 1].argsort()]
    src_points = src_points[0:-3]

    cut = cut + 1

    # if we have less than 21 circles, give up
    if no_circles < 21:
        print ('not enough circles matched')
        10 / 0


# rectify the image
print("rectifying image")
img_rectified = transform.warp(image_grey, tform, output_shape=(3000,3000))
img_rectified = img_as_ubyte(img_rectified)
imageio.imwrite(images_dir + orig_image + '_rectified.png', img_rectified)

print('extracting default ROI')
# open the image we just saved
img = Image.open(images_dir + orig_image + '_rectified.png')
# CROPPING ROI FROM IMAGE
left = int(np.min(dst_points[:, 0]) - 200)
top = 1750
right = int(np.max(dst_points[:, 0]) + 200)
bottom = 2500
ROI = img.crop((left, top, right, bottom))
# print("ROI: ", top, bottom)

print("Enter new ROI top (waterline minus ~50), then close image:")

# plot the ROI
plt.imshow(ROI, extent=[left,right,bottom,top])
plt.xlabel('micrometer')
plt.ylabel('micrometer')
plt.title('Region of interest')
plt.show()

user_input = input()

try:
    top = int(user_input)
except ValueError:
    print("Invalid input")

print("Now enter new ROI left (choose the left side of the clearest waterline section), then close image:")

# plot the ROI
plt.imshow(ROI, extent=[left,right,bottom,top])
plt.xlabel('micrometer')
plt.ylabel('micrometer')
plt.title('Region of interest')
plt.show()

user_input = input()

try:
    left = int(user_input)
    right = left + 50
    bottom = top + 100
except ValueError:
    print("Invalid input")

ROI = img.crop((left, top, right, bottom))

# plot the ROI
plt.imshow(ROI, extent=[left,right,bottom,top])
plt.xlabel('micrometer')
plt.ylabel('micrometer')
plt.title('Region of interest')
plt.show()

# fig.figure.savefig(images_dir + orig_image + '_ROI_with_scale.png', bbox_inches='tight', dpi=300)


imageio.imwrite(images_dir + orig_image + '_ROI.png', ROI)

# read ROI image
ROI = cv2.imread(images_dir + orig_image + '_ROI.png')
ROI_grey = cv2.split(ROI)[0]

print('extracting waterline')
# set a start value for the edge detection sensitivity of the waterline
waterline_threshold=40
contours = []

while waterline_threshold > waterline_threshold_min:
    edges = canny(ROI_grey, sigma=2, low_threshold=5, high_threshold=waterline_threshold)
    edges_print = color.gray2rgb(img_as_ubyte(edges))

    cv2.imwrite(images_dir + orig_image + '_ROI_edges.png', edges_print)
    edges_import = cv2.imread(images_dir + orig_image + '_ROI_edges.png', 0)

    contours, hier = cv2.findContours(edges_import, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE )

    waterline = []
    waterline_x = []
    waterline_y = []
    longest = 0
    highest = 0

    # find the longest line
    for i in range(len(contours)):
        c = contours[i]
        c = np.vstack(c).squeeze()
        if longest < len(c):
            waterline = c
            longest = len(c)
        #if highest < np.median(c):
        #    waterline = c


    if len(waterline) != 0:
        waterline_x = (waterline[:,0])
        waterline_y = (waterline[:,1])
        break

    waterline_threshold = waterline_threshold - 5

    print ('trying waterline threshold: %d' % (waterline_threshold))


print('waterline detected!')

plt.close()
final_img = img.crop((750, 600, 1700, 2600))
waterline_x_plot = waterline_x + left - 750
waterline_y_plot = waterline_y + top - 600
fig = plt.imshow(final_img)
plt.scatter(waterline_x_plot, waterline_y_plot, s=0.5)
plt.title('Waterline Detection - ' + orig_image)
plt.xlabel('micrometre')
plt.ylabel('micrometre')
# plt.show()
fig.figure.savefig(images_dir + orig_image + '_waterline.png', bbox_inches='tight', dpi=300)

waterline_RL = (np.median(waterline_y + top)/10) - 167.75
print('waterline_RL: ', waterline_RL)



