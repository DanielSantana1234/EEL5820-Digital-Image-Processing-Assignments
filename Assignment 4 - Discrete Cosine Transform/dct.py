from PIL import Image
import numpy as np
import pandas as pd
import math

image_path = "Assignment 3 - Discrete Fourier Transform/128_lena_gray.bmp"
img = Image.open(image_path)
width, height = img.size
fourier_image = Image.new("L", img.size, 0xffffff)