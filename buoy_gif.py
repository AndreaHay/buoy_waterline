#!/usr/bin/env python3
# coding: utf-8

# Called by buoy_float_level.sh

import os
import sys
import imageio

print("starting python")

#ARGUMENTS FROM THE USER
images_dir = sys.argv[1]
name_scale = (sys.argv[2])

images = []
for file in sorted(os.listdir(images_dir + 'output/waterline/')):
    images.append(imageio.imread(images_dir + 'output/waterline/' + file))
    
imageio.mimsave(images_dir + 'output/' + name_scale + '_waterline.gif', images)

