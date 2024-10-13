#!/usr/bin/env python3
# coding: utf-8

# run using buoy_float_level.sh

import numpy as np
import matplotlib.pyplot as plt
import os
import re
import cv2
import sys
import imageio
import shutil
import statistics as stats
import pandas as pd
from PIL import Image
from skimage import data, color, transform
from skimage.io import imread
from skimage.util import img_as_ubyte
from skimage.draw import circle_perimeter
from skimage.feature import peak_local_max, canny
from skimage.transform import hough_circle, hough_circle_peaks

print("starting python")

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
ROI_top = sys.argv[10]
ROI_left = sys.argv[11]

#read reference coordinates
csv = np.genfromtxt (ref_coords, delimiter=",")
fixed = csv[:,0:2].copy()

no_circles_ref = len(fixed)
no_rows_ref = int(no_circles_ref/3)

# Load picture and detect edges
for file in sorted(os.listdir(images_dir + 'output/'+ colour + '/')):
    try:
        print("processing photo:", file)
        filename = os.fsdecode(file)
        name, ext = os.path.splitext(filename)

        #print(name)
        #print(filename)
        if enhance == 't':
            image_clr = imread(images_dir + 'output/'+ colour + '/' + filename)
        else:
            image_clr = imread(images_dir + 'output/'+ colour + '/' + filename)[..., 0]
        edges = canny(image_clr, sigma=2, low_threshold=5, high_threshold=50)

        plt.title('Edges - ' + filename)
        fig=plt.imshow(edges)
        fig.figure.savefig(images_dir + 'output/edges/' + name + '_edges.png', bbox_inches='tight', dpi=1000)

        # Detect circles
        hough_radii = np.arange(min_circle_size, max_circle_size)
        hough_res = hough_circle(edges, hough_radii)
        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, min_xdistance=min_circle_size, min_ydistance=min_circle_size, total_num_peaks=80, normalize=False)


        # Print scatterplot of detected radii
        plt.close()
        colour_map = plt.get_cmap('rainbow_r')
        fig = plt.scatter(cx, -(cy), c=accums, cmap=colour_map)
        plt.gca().set_aspect('equal')
        plt.colorbar()
        fig.figure.savefig(images_dir + 'output/circle_detection/' + name + '_radii.png', bbox_inches='tight')

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
        pix_size = radii_median/2.5

        cx_median = np.median(cx[0:18])
        inside = np.logical_and(cx < (cx_median + pix_size*15), cx > (cx_median - pix_size*15))
        index = np.where(inside)[0]
        radii = radii[index]
        cx = cx[index]
        cy = cy[index]

        cy_median = np.median(cy)
        inside = np.logical_and(cy < (cy_median + pix_size*80), cy > (cy_median - pix_size*80))
        index = np.where(inside)[0]
        radii = radii[index]
        cx = cx[index]
        cy = cy[index]

        # highest circle might be the screw... remove it if it is
        # find indices of highest two cy values
        check_screw = cy.argsort()[0:2]
        # if the difference between the two is greater than 10 mm, remove the highest one
        if (cy[check_screw[1]] - cy[check_screw[0]]) > (pix_size*10):
            print("removing highest circle")
            radii = radii[cy.argsort()[1:]]
            cx = cx[cy.argsort()[1:]]
            cy = cy[cy.argsort()[1:]]

        # remove lowest circles if the number of circles is not a multiple of 3
        if len(radii)%3==2:
            radii = radii[cy.argsort()][0:-2]
            cx = cx[cy.argsort()][0:-2]
            cy = cy[cy.argsort()][0:-2]

        if len(radii)%3==1:
            radii = radii[cy.argsort()][0:-1]
            cx = cx[cy.argsort()][0:-1]
            cy = cy[cy.argsort()][0:-1]

        # Print detected circles on grey image
        orig_image = re.sub('\_' + colour + '.png$', '', filename)
        #print(orig_image)

        image_plot = color.gray2rgb(imread(images_dir + orig_image + '.png')[..., 0])

        for center_y, center_x, radius in zip(cy, cx, radii):
            circy, circx = circle_perimeter(center_y, center_x, radius, shape=image_plot.shape)
            image_plot[circy, circx] = (220, 20, 20)

        plt.close()
        plt.title('Circle Detection - ' + filename)
        fig=plt.imshow(image_plot)
        fig.figure.savefig(images_dir + 'output/circle_detection/' + orig_image + '_circles.png', bbox_inches='tight', dpi=1000)

        image_grey = color.gray2rgb(imread(images_dir + orig_image + '.png')[..., 0])

        tform_resids_stdev = 10
        src_points = np.vstack((cx, cy, radii)).T
        cut = 0

        while tform_resids_stdev > 3:
            no_circles = int(len(src_points))
            no_rows = int(no_circles/3)
            #print('no rows',  no_rows, 'type:', type(no_rows))
            print("number of circles detected:", no_circles)
            fixed = csv[:,0:2].copy()
            first_col = fixed[0:no_rows,0:2].copy()
            second_col = fixed[no_rows_ref*1:no_rows_ref*1+no_rows,0:2].copy()
            third_col = fixed[no_rows_ref*2:no_rows_ref*2+no_rows,0:2].copy()
            fixed = np.vstack((first_col, second_col, third_col))

            #create array of centre points and sort by y then x
            col_1 = src_points[src_points[:, 0].argsort()][0:no_rows, :]
            col_1 = col_1[col_1[:,1].argsort()]
            col_2 = src_points[src_points[:, 0].argsort()][no_rows: no_rows*2, :]
            col_2 = col_2[col_2[:,1].argsort()]
            col_3 = src_points[src_points[:, 0].argsort()][no_rows*2: no_rows*3, :]
            col_3 = col_3[col_3[:,1].argsort()]

            sorted_all = np.vstack((col_1, col_2, col_3))
            src_points = sorted_all[:, 0:2]
            dst_points = fixed*10

            tform = transform.ProjectiveTransform()
            tform.estimate(dst_points, src_points)

            tform_resids = ((tform.inverse(src_points)) - dst_points)
            tform_resids_stdev = np.std(tform_resids)

            print('tform_resid', tform_resids_stdev)
            src_points = src_points[src_points[:,1].argsort()]
            src_points = src_points[0:-3]

            cut = cut + 1

            if no_circles < 21:
                10/0



        img_rectified = transform.warp(image_grey, tform, output_shape=(3000,3000))
        img_rectified = img_as_ubyte(img_rectified)
        imageio.imwrite(images_dir + 'output/rectified_images/' + orig_image + '_rectified.png', img_rectified)

        print("image rectified")

        img = Image.open(images_dir + 'output/rectified_images/' + orig_image + '_rectified.png')
        # CROPPING ROI FROM IMAGE
        #left = int(np.max(dst_points[:, 0]) + 50)
        left = int(ROI_left)
        #top = 1750
        #top = int(np.max(dst_points[:, 1]) - 50)
        top = int(ROI_top)
        right = int(left + 50)
        bottom = int(top + 100)

        ROI = img.crop((left, top, right, bottom))

        imageio.imwrite(images_dir + 'output/ROI/' + orig_image + '_ROI.png', ROI)
        ROI = cv2.imread(images_dir + 'output/ROI/' + orig_image + '_ROI.png')
        ROI_grey = cv2.split(ROI)[0]

        waterline_threshold=40
        contours = []

        while waterline_threshold > waterline_threshold_min:
            edges = canny(ROI_grey, sigma=2, low_threshold=5, high_threshold=waterline_threshold)
            edges_print = color.gray2rgb(img_as_ubyte(edges))

            cv2.imwrite(images_dir + 'output/ROI/' + orig_image + '_ROI_edges.png', edges_print)
            edges_import = cv2.imread(images_dir + 'output/ROI/' + orig_image + '_ROI_edges.png', 0)

            contours, hier = cv2.findContours(edges_import, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE )

            waterline = []
            waterline_x = []
            waterline_y = []
            longest = 0
            highest = 0

            for i in range(len(contours)):
                c = contours[i]
                c = np.vstack(c).squeeze()
                if longest < len(c):
                    waterline = c
                    longest = len(c)


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
        fig.figure.savefig(images_dir + 'output/waterline/' + orig_image + '_waterline.png', bbox_inches='tight', dpi=300)

        waterline_RL = (np.median(waterline_y + top)/10) - 167.75
        print('waterline_RL: ', waterline_RL)
        print('orig_image: ', orig_image)
        print('name_scale: ', name_scale)
        print('image number: ', orig_image.replace(name_scale, ''))

        results = np.vstack((int(orig_image.replace(name_scale, '')),np.median(waterline_RL))).T


        print("waterline detected")

        print("**RESULTS**")
        print(results)
        print("\n")

        results_file = open(images_dir + 'output/' + name_scale + '_results.csv', 'ab')
        np.savetxt(results_file,results, delimiter=',', fmt='%.2f')
        results_file.close()

    except:
        shutil.copy(images_dir + orig_image + '.png', images_dir + 'output/failed/' + orig_image + '.png')
        results = np.vstack((int(orig_image.replace(name_scale, '')),999)).T
        results_file = open(images_dir + 'output/' + name_scale + '_results.csv', 'ab')
        np.savetxt(results_file,results, delimiter=',', fmt='%.2f')
        results_file.close()
        print("FAILED\n")


df = pd.read_csv(images_dir + 'output/' + name_scale +  '_results.csv')

df.replace(999, np.nan, inplace=True)
df_data = df.dropna()

stdev = stats.stdev(df_data['waterline'])
mean = np.mean(df_data['waterline'])
median = np.median(df_data['waterline'])
no_waterlines = len(df_data['waterline'])
no_images_all = len(df['waterline'])
no_images_failed = no_images_all - no_waterlines
std_err = stdev/(np.sqrt(no_waterlines))
q75, q25 = np.percentile(df_data['waterline'], [75, 25])
IQR = q75-q25
text = '\n'.join((
    'mean = %.1f mm' % (mean, ),
    'median = %.1f mm' % (median, ),
    'std. dev. = %.1f mm' % (stdev, ),
    'std. error = %.1f mm \n' % (std_err, ),
    'IQR = %.1f mm' % (IQR, ),
    'Q1 = %.1f mm' % (q25),
    'Q3 = %.1f mm \n' % (q75),
    'no. images used = %d' % (no_waterlines),
    'no. images failed = %d' % (no_images_failed)))


plt.close()

plt.scatter(df['image_no'], df['waterline'], color='C0', s=3)
plt.axhline(y=median, color='C0', linestyle='-')
plt.axhline(y=(q25-1.5*IQR), color='red', linestyle='--')
plt.axhline(y=(q75+1.5*IQR), color='red', linestyle='--')
plt.axhline(y=q25, color='C1', linestyle='--')
plt.axhline(y=q75, color='C1', linestyle='--')
plt.text(0.98, 0.5, text, fontsize=10, transform=plt.gcf().transFigure)
plt.text(np.max(df['image_no']), median, 'median', color='C0')
plt.text(np.max(df['image_no']), q25, 'Q1', color='C1')
plt.text(np.max(df['image_no']), q75, 'Q3', color='C1')
plt.text(np.max(df['image_no']), mean-stdev, 'std. dev.', color='C0')
plt.text(np.max(df['image_no']), (q25-1.5*IQR), 'Q1-1.5*IQR', color='red')
plt.xlim(np.min(df['image_no']), np.max(df['image_no']))

plt.xlabel('Images')
plt.ylabel('Waterline level (mm)')
plt.fill_between(df['image_no'], mean-stdev, mean+stdev, alpha=0.2)
fig = plt.title(name_scale)

fig.figure.savefig(images_dir + 'output/' + name_scale + '_results_with_outliers.png', bbox_inches='tight', dpi=300)

#trim outliers (more than 1.5 * IQR from mean)
inliers = df_data[df_data['waterline'] > (q25-1.5*IQR)]
inliers = inliers[inliers['waterline'] < (q75+1.5*IQR)]
y = inliers['waterline']
x = inliers['image_no']

no_waterlines = len(y)
no_images_failed = no_images_all - no_waterlines
q75, q25 = np.percentile(y, [75, 25])
IQR = q75-q25
stdev = stats.stdev(y)
mean = np.mean(y)
median = np.median(y)
no_images = len(y)
std_err = stdev/(np.sqrt(no_waterlines))

text = '\n'.join((
        'mean = %.1f mm' % (mean, ),
        'median = %.1f mm' % (median, ),
        'std. dev. = %.1f mm' % (stdev, ),
        'std. error = %.1f mm \n' % (std_err, ),
        'IQR = %.1f mm' % (IQR, ),
        'Q1 = %.1f mm' % (q25),
        'Q3 = %.1f mm \n' % (q75),
        'no. images used = %d' % (no_waterlines),
        'no. images failed = %d' % (no_images_failed)))


plt.close()

plt.scatter(x, y, color='C0', s=3)
plt.axhline(y=median, color='C0', linestyle='-')
plt.axhline(y=q25, color='C1', linestyle='--')
plt.axhline(y=q75, color='C1', linestyle='--')
plt.text(0.98, 0.5, text, fontsize=10, transform=plt.gcf().transFigure)
plt.text(np.max(df['image_no']), median, 'median', color='C0')
plt.text(np.max(df['image_no']), q25, 'Q1', color='C1')
plt.text(np.max(df['image_no']), q75, 'Q3', color='C1')
plt.text(np.max(df['image_no']), mean-stdev, 'std. dev.', color='C0')
plt.xlabel('Images')
plt.ylabel('Waterline level (mm)')
plt.xlim(np.min(df['image_no']), np.max(df['image_no']))
plt.fill_between(df['image_no'], mean-stdev, mean+stdev, alpha=0.2)
fig = plt.title(name_scale)

fig.figure.savefig(images_dir + 'output/' + name_scale + '_results.png', bbox_inches='tight', dpi=300)

#plot histogram
plt.close()

plt.hist(y, 25)
plt.axvline(x=median, color='red', linestyle='--')
plt.xlabel('waterline level (mm)')
plt.ylabel('frequency')
fig = plt.title(name_scale)

fig.figure.savefig(images_dir + 'output/' + name_scale + '_histogram.png', bbox_inches='tight', dpi=300)

final_stats = np.vstack((mean,median,stdev,std_err,q25,q75,IQR,no_waterlines,no_images_all)).T
final_stats_file = open(images_dir + 'output/' + name_scale + '_final_stats.csv', 'ab')
np.savetxt(final_stats_file,final_stats, delimiter=',', fmt='%.2f')

images = []
for file in sorted(os.listdir(images_dir + 'output/waterline/')):
    images.append(imageio.imread(images_dir + 'output/waterline/' + file))

imageio.mimsave(images_dir + 'output/' + name_scale + '_waterline.gif', images)

