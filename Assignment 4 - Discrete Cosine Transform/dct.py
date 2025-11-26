from PIL import Image
import numpy as np
import pandas as pd
import math

image_path = "Assignment 4 - Discrete Cosine Transform/128_lena_gray.bmp"
img = Image.open(image_path).convert('L')
width, height = img.size
dct_image = Image.new("L", img.size, 0xffffff)

def dct(width, height):
    F = np.zeros((height, width), dtype = float)
    for u in range(width):
        alpha_u = math.sqrt(1.0/width) if u == 0 else math.sqrt(2.0/width)
        for v in range(height):
            alpha_v = math.sqrt(1.0/height) if v == 0 else math.sqrt(2.0/height)
            current = 0
            for x in range(width):
                cos_x = (math.cos(((2 * x + 1) * u * math.pi)/(2 * width)))
                for y in range(height):
                    current_pixels = float(img.getpixel((x, y)))
                    cos_y = (math.cos(((2 * y + 1) * v * math.pi)/ (2 * height)))
                    term = current_pixels * cos_x * cos_y
                    current = current + term
            F[v, u] = current * alpha_u * alpha_v
    # For display: take magnitude (absolute) and normalize to 0-255
    magnitude = np.abs(F)
    log_magnitude = np.log1p(magnitude)
    mn, mx = log_magnitude.min(), log_magnitude.max()
    eps = 1e-8
    norm = ((log_magnitude - mn) / (mx - mn + eps) * 255.0).astype(np.uint8)

    out = Image.fromarray(norm, mode='L')
    out.save('Assignment 4 - Discrete Cosine Transform/dct_lena_gray.bmp')

dct(width, height)