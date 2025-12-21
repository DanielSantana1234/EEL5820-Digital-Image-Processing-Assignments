from PIL import Image
import numpy as np
import pandas as pd
import math

image_path = "Images/Synthetic/256_synthetic.bmp"
img = Image.open(image_path).convert('L')
width, height = img.size
filtering_image = Image.new("L", img.size, 0xffffff)

F = np.zeros((height, width), dtype = complex)
f_transform = np.fft.fft2(img)
F = np.fft.fftshift(f_transform)
filter_L = np.zeros((height, width), dtype = np.float32)
filter_H = np.ones((height, width), dtype = np.float32)
cutoff_frequency = 25

def ideal_low_pass_filtering(height, width):
    for h in range(height):
        for w in range(width):
            if math.sqrt((h - height / 2) ** 2 + (w - width / 2) ** 2) <= cutoff_frequency:
                filter_L[h][w] = 1
    return filter_L


def ideal_high_pass_filtering(height, width):
    for h in range(height):
        for w in range(width):
            if math.sqrt((h - height / 2) ** 2 + (w - width / 2) ** 2) <= cutoff_frequency:
                filter_H[h][w] = 0
    return filter_H

def butterworth_low_pass_filtering(height, width):
    for h in range(height):
        for w in range(width):
            if math.sqrt((h - height / 2) ** 2 + (w - width / 2) ** 2) <= cutoff_frequency:
                filter_L[h][w] = 1 / (1 + (math.sqrt((h - height / 2) ** 2 + (w - width / 2) ** 2) / cutoff_frequency) ** 2)
    return filter_L

def butterworth_high_pass_filtering(height, width):
    for h in range(height):
        for w in range(width):
            if math.sqrt((h - height / 2) ** 2 + (w - width / 2) ** 2) <= cutoff_frequency:
                filter_H[h][w] = 1 - (1 / (1 + (math.sqrt((h - height / 2) ** 2 + (w - width / 2) ** 2) / cutoff_frequency) ** 2))
    return filter_H

print("What type of filter would you like to apply? (Enter the number corresponding to the filter)")
print("1. Ideal Low Pass Filtering")
print("2. Ideal High Pass Filtering")
print("3. Butterworth Low Pass Filtering")
print("4. Butterworth High Pass Filtering")
input = int(input())

if(input == 1):
    H = ideal_low_pass_filtering(height, width)
    filtered_f = F * H
elif(input == 2):
    H = ideal_high_pass_filtering(height, width)
    filtered_f = F * H
elif(input == 3):
    H = butterworth_low_pass_filtering(height, width)
    filtered_f = F * H
elif(input == 4):
    H = butterworth_high_pass_filtering(height, width)
    filtered_f = F * H
else:
    print("Invalid Input")

inv_F = np.fft.ifftshift(filtered_f)
inv_f_transform = np.fft.ifft2(inv_F)
inv_f = np.abs(inv_f_transform).astype(np.uint8)

magnitude = np.abs(inv_f_transform)
log_magnitude = np.log1p(magnitude)
mn, mx = log_magnitude.min(), log_magnitude.max()
eps = 1e-8
norm = ((log_magnitude - mn) / (mx - mn + eps) * 255.0).astype(np.uint8)

out = Image.fromarray(norm, mode='L')
out.save('Assignment 6 - Image Filtering/filter_lena_gray.bmp')