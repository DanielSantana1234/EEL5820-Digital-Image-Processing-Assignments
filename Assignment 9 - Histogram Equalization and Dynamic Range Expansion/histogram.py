from PIL import Image
import numpy as np
import pandas as pd
import math

image_path = "Images/Natural/256_lena_gray.bmp"
img = Image.open(image_path).convert('L')
width, height = img.size
histogram_image = Image.new("L", img.size, 0xffffff)

def histogram(image, sigma):
    img_array = np.array(image)
    h, w = img_array.shape
    total_pixels = h * w

    hist = np.zeros(256, dtype=int)
    for pixel in img_array.flatten():
        hist[pixel] += 1
    
    if sigma == 0:
        cdf = np.cumsum(hist)
        cdf_min = cdf[cdf > 0].min()
        lookup_table = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            if cdf[i] > 0:
                lookup_table[i] = round(((cdf[i] - cdf_min) / (total_pixels - cdf_min)) * 255)
        result_array = lookup_table[img_array]
    else:
        min_val = img_array.min()
        max_val = img_array.max()
        if max_val == min_val:
            return image
        result_array = (((img_array - min_val) / (max_val - min_val)) * 255).astype(np.uint8)
    
    return Image.fromarray(result_array)

equalized_img = histogram(img, 0)
equalized_img.save("Assignment 9 - Histogram Equalization and Dynamic Range Expansion/histogram_equalized.bmp")

expanded_img = histogram(img, 1)
expanded_img.save("Assignment 9 - Histogram Equalization and Dynamic Range Expansion/dynamic_range_expanded.bmp")