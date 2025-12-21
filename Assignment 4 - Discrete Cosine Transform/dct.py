from PIL import Image
import numpy as np
import pandas as pd
import math

image_path = "Images/Synthetic/128_synthetic.bmp"
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
    
    magnitude = np.abs(F)
    log_magnitude = np.log1p(magnitude)
    mn, mx = log_magnitude.min(), log_magnitude.max()
    eps = 1e-8
    norm = ((log_magnitude - mn) / (mx - mn + eps) * 255.0).astype(np.uint8)

    out = Image.fromarray(norm, mode='L')
    out.save('Assignment 4 - Discrete Cosine Transform/dct.bmp')

def walsh(width, height):
    F = np.zeros((height, width), dtype = float)

    def build_hadamard(n):
        if n == 1:
            return np.array([[1.0]])
        H_half = build_hadamard(n // 2)
        return np.block([[H_half, H_half], [H_half, -H_half]])
    
    H = build_hadamard(width)
    sequency = [sum(H[i, j] * H[i, j+1] < 0 for j in range(width-1)) for i in range(width)]
    order = sorted(range(width), key=lambda i: sequency[i])
    W = H[order, :]
    
    f = np.array([[float(img.getpixel((x, y))) for x in range(width)] for y in range(height)])
    
    F = (1.0 / width) * W @ f @ W.T

    magnitude = np.abs(F)
    log_magnitude = np.log1p(magnitude)
    mn, mx = log_magnitude.min(), log_magnitude.max()
    eps = 1e-8
    norm = ((log_magnitude - mn) / (mx - mn + eps) * 255.0).astype(np.uint8)

    out = Image.fromarray(norm, mode='L')
    out.save('Assignment 4 - Discrete Cosine Transform/walsh.bmp')

def hadamard(width, height):
    F = np.zeros((height, width), dtype = float)

    def build_hadamard(n):
        if n == 1:
            return np.array([[1.0]])
        H_half = build_hadamard(n // 2)
        return np.block([[H_half, H_half], [H_half, -H_half]])
    
    H = build_hadamard(width)
    
    f = np.array([[float(img.getpixel((x, y))) for x in range(width)] for y in range(height)])
    
    F = (1.0 / width) * H @ f @ H.T


    magnitude = np.abs(F)
    log_magnitude = np.log1p(magnitude)
    mn, mx = log_magnitude.min(), log_magnitude.max()
    eps = 1e-8
    norm = ((log_magnitude - mn) / (mx - mn + eps) * 255.0).astype(np.uint8)

    out = Image.fromarray(norm, mode='L')
    out.save('Assignment 4 - Discrete Cosine Transform/hadamard.bmp')

def haar(width, height):
    F = np.zeros((height, width), dtype = float)

    H = np.zeros((width, width))
    H[0, :] = 1.0
    row = 1
    for level in range(int(math.log2(width))):
        num_funcs = 2 ** level
        block_width = width // (2 ** (level + 1))
        amplitude = math.sqrt(2 ** level)
        for k in range(num_funcs):
            if row >= width:
                break
            start = k * 2 * block_width
            H[row, start:start + block_width] = amplitude
            H[row, start + block_width:start + 2 * block_width] = -amplitude
            row += 1
    H = H / math.sqrt(width)
    
    f = np.array([[float(img.getpixel((x, y))) for x in range(width)] for y in range(height)])
    
    F = H @ f @ H.T

    magnitude = np.abs(F)
    log_magnitude = np.log1p(magnitude)
    mn, mx = log_magnitude.min(), log_magnitude.max()
    eps = 1e-8
    norm = ((log_magnitude - mn) / (mx - mn + eps) * 255.0).astype(np.uint8)

    out = Image.fromarray(norm, mode='L')
    out.save('Assignment 4 - Discrete Cosine Transform/haar.bmp')

dct(width, height)
hadamard(width, height)
walsh(width, height)
haar(width, height)