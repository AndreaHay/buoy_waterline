#!/usr/bin/env python3
# coding: utf-8

# Called from buoy_float_level.sh

import numpy as np
import matplotlib.pyplot as plt
import sys
import statistics as stats
import pandas as pd

images_dir = sys.argv[1]
name_scale = (sys.argv[2])

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


# trim outliers (more than 1.5 * IQR from mean)
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
#plt.text(1, np.max(y), text, fontsize=10, verticalalignment='top')
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


# plot histogram
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


