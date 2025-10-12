from PIL import Image
import numpy as np
import pandas as pd
import math

image_path = "Assignment 3 - Discrete Fourier Transform/128_lena_gray.bmp"
img = Image.open(image_path)
width, height = img.size
fourier_image = Image.new("L", img.size, 0xffffff)

# Transform the image to the Fourier domain 
# Time the amount of time it takes to transfer 
# Just center the image 
# Be able to display it 
# Then inverse the transform and display it back again 

def dft(width, height):
    F = np.zeros((height, width), dtype=np.complex128)
    for u in range(width):
        for v in range(height):
            current = complex(0, 0)
            for x in range(width):
                for y in range(height):
                    # The Fourier Transform
                    current_pixels = float(img.getpixel((x, y))) * ((-1) ** (x + y))
                    theta = (-2 * math.pi) * (((u*x)/width) + ((v*y)/height))
                    euler_exponential = complex(math.cos(theta), math.sin(theta))

                    term = current_pixels * euler_exponential
                    current = current + term
    # Trying to fix being able to put "current" into the .putpixel() method to fix
            F[v, u] = current
    magnitude_spectrum = np.abs(F)
    log_magnitude = np.log1p(magnitude_spectrum)

    minimum = np.min(log_magnitude)
    maximum = np.max(log_magnitude)

    epsilon = 1e-8
    normalized_magnitude = (log_magnitude - minimum) / (maximum - minimum + epsilon)
            
    scale_to_8_bit = np.clip(normalized_magnitude * 255, 0, 255).astype(np.uint8)

    fourier_image = Image.fromarray(scale_to_8_bit, mode='L')
    fourier_image.save('Assignment 3 - Discrete Fourier Transform/fourier_lena_gray.bmp')

dft(width, height)