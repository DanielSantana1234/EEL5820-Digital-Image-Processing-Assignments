from PIL import Image
import numpy as np
import pandas as pd
import math

image_path = "Images/Natural/512_lena_gray.bmp"
img = Image.open(image_path).convert('L')
width, height = img.size
fft_image = Image.new("L", img.size, 0xffffff)

def fft(width, height):
    F = np.zeros((height, width), dtype=complex)

    img_array = np.array(img, dtype=float)

    def fft_1d(x):
        N = len(x)
        if N <= 1:
            return x
        if N & (N - 1): 
            raise ValueError("Size must be power of 2")
        
        X = np.array(x, dtype=complex)

        j = 0
        for i in range(N):
            if i < j:
                X[i], X[j] = X[j], X[i]
            m = N // 2
            while m >= 1 and j >= m:
                j -= m
                m //= 2
            j += m

        size = 2
        while size <= N:
            half = size // 2
            step = N // size
            for i in range(0, N, size):
                k = 0
                for j in range(i, i + half):
                    w = np.exp(-2j * np.pi * k / size)
                    t = w * X[j + half]
                    X[j + half] = X[j] - t
                    X[j] = X[j] + t
                    k += step
            size *= 2
        return X

    for i in range(height):
        F[i, :] = fft_1d(img_array[i, :])

    for j in range(width):
        F[:, j] = fft_1d(F[:, j])

    M2, N2 = height // 2, width // 2
    F_centered = np.zeros_like(F)
    F_centered[0:M2, 0:N2] = F[M2:height, N2:width]
    F_centered[0:M2, N2:width] = F[M2:height, 0:N2]
    F_centered[M2:height, 0:N2] = F[0:M2, N2:width]
    F_centered[M2:height, N2:width] = F[0:M2, 0:N2]
    F = F_centered

    magnitude = np.abs(F)
    log_magnitude = np.log1p(magnitude)
    mn, mx = log_magnitude.min(), log_magnitude.max()
    eps = 1e-8
    norm = ((log_magnitude - mn) / (mx - mn + eps) * 255.0).astype(np.uint8)

    out = Image.fromarray(norm, mode='L')
    out.save('Assignment 5 - Fast Fourier Transform FFT/fft_lena_gray.bmp')


fft(width, height)