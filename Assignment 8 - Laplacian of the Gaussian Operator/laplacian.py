from PIL import Image
import numpy as np
import pandas as pd
import math
import os

image_path = "Images/Natural/256_lena_gray.bmp"
img = Image.open(image_path).convert('L')
width, height = img.size
laplacian_image = Image.new("L", img.size, 0xffffff)

def create_log_kernel(sigma, size=None):
    if size is None:
        size = int(6 * sigma + 1)
        if size % 2 == 0:
            size += 1
    
    half = size // 2
    kernel = np.zeros((size, size), dtype=np.float64)
    
    for i in range(size):
        for j in range(size):
            x = i - half
            y = j - half
            
            r_squared = x**2 + y**2
            sigma_squared = sigma**2
            
            term1 = -1 / (np.pi * sigma**4)
            term2 = 1 - (r_squared / (2 * sigma_squared))
            term3 = np.exp(-r_squared / (2 * sigma_squared))
            
            kernel[i, j] = term1 * term2 * term3
    
    kernel = kernel - kernel.mean()
    
    return kernel


def convolve2d(image_array, kernel):
    img_height, img_width = image_array.shape
    k_height, k_width = kernel.shape
    
    pad_h = k_height // 2
    pad_w = k_width // 2
    
    padded = np.pad(image_array, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    
    output = np.zeros_like(image_array, dtype=np.float64)
    
    for i in range(img_height):
        for j in range(img_width):
            region = padded[i:i+k_height, j:j+k_width]
            output[i, j] = np.sum(region * kernel)
    
    return output


def laplacian_of_gaussian(image, sigma):
    img_array = np.array(image, dtype=np.float64)
    
    kernel = create_log_kernel(sigma)
    
    result = convolve2d(img_array, kernel)
    
    result = np.abs(result)

    if result.max() > result.min():
        result = (result - result.min()) / (result.max() - result.min()) * 255
    
    result = result.astype(np.uint8)
    
    return Image.fromarray(result)


def zero_crossing_detection(log_result):
    img_array = np.array(log_result, dtype=np.float64)
    height, width = img_array.shape
    edges = np.zeros_like(img_array)
    
    for i in range(1, height-1):
        for j in range(1, width-1):
            neighbors = [
                img_array[i-1, j], img_array[i+1, j],
                img_array[i, j-1], img_array[i, j+1]
            ]
            center = img_array[i, j]
            
            for n in neighbors:
                if (center > 0 and n < 0) or (center < 0 and n > 0):
                    edges[i, j] = 255
                    break
    
    return Image.fromarray(edges.astype(np.uint8))


def analyze_log_effects():
    sigmas = [0.5, 1.0, 2.0, 3.0, 5.0]
    
    results = {}
    
    for sigma in sigmas:
        
        log_result = laplacian_of_gaussian(img, sigma)
        
        result_array = np.array(log_result)
        
        results[sigma] = {
            'image': log_result,
            'mean': result_array.mean(),
            'std': result_array.std(),
            'edge_pixels': np.sum(result_array > 128),
        }
        
        output_path = f"Assignment 8 - Laplacian of the Gaussian Operator/log_sigma_{sigma}.png"
        log_result.save(output_path)
    
    img.save("Assignment 8 - Laplacian of the Gaussian Operator/original.png")
    
    return results, sigmas

def create_comparison_image(original, results, sigmas):
    width, height = original.size
    
    n_images = len(sigmas) + 1
    comparison = Image.new('L', (width * n_images, height + 30), 255)
    
    comparison.paste(original, (0, 30))
    
    for idx, sigma in enumerate(sigmas):
        comparison.paste(results[sigma]['image'], ((idx + 1) * width, 30))
    
    return comparison

results, sigmas = analyze_log_effects()


