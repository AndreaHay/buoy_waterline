#!/usr/bin/env python3
# coding: utf-8

import sys
from PIL import Image, ImageEnhance

# read file and contrast value
file = sys.argv[1]
contrast_value = float(sys.argv[2])

# open file, convert to greyscale, increase contrast, and overwite original file
img = Image.open(file)
grey = ImageEnhance.Color(img)
grey_img = grey.enhance(0)
contrast = ImageEnhance.Contrast(grey_img)
contrast_img = contrast.enhance(contrast_value)
contrast_img.save(file)

