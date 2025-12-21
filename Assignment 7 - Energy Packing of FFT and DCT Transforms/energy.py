from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt

image_path = "Images/Natural/32_lena_gray.bmp"
img = Image.open(image_path).convert('L')
width, height = img.size

new_size = min(2 ** int(np.floor(np.log2(min(width, height)))), 256)
if width != new_size or height != new_size:
    img = img.resize((new_size, new_size), Image.Resampling.LANCZOS)
    width, height = new_size, new_size

image_array = np.array(img, dtype=float)

def dft(width, height):
    F = np.zeros((height, width), dtype=np.complex128)
    for u in range(width):
        for v in range(height):
            current = complex(0, 0)
            for x in range(width):
                for y in range(height):
                    current_pixels = float(img.getpixel((x, y))) * ((-1) ** (x + y))
                    theta = (-2 * math.pi) * (((u*x)/width) + ((v*y)/height))
                    euler_exponential = complex(math.cos(theta), math.sin(theta))
                    current += current_pixels * euler_exponential
            F[v, u] = current
    return F

def idft(F, width, height):
    f = np.zeros((height, width), dtype=float)
    for x in range(width):
        for y in range(height):
            current = complex(0, 0)
            for u in range(width):
                for v in range(height):
                    theta = (2 * math.pi) * (((u*x)/width) + ((v*y)/height))
                    euler_exponential = complex(math.cos(theta), math.sin(theta))
                    current += F[v, u] * euler_exponential
            f[y, x] = (current.real / (width * height)) * ((-1) ** (x + y))
    return f

def dct(width, height):
    F = np.zeros((height, width), dtype=float)
    for u in range(width):
        alpha_u = math.sqrt(1.0/width) if u == 0 else math.sqrt(2.0/width)
        for v in range(height):
            alpha_v = math.sqrt(1.0/height) if v == 0 else math.sqrt(2.0/height)
            current = 0
            for x in range(width):
                cos_x = math.cos(((2 * x + 1) * u * math.pi)/(2 * width))
                for y in range(height):
                    current_pixels = float(img.getpixel((x, y)))
                    cos_y = math.cos(((2 * y + 1) * v * math.pi)/(2 * height))
                    current += current_pixels * cos_x * cos_y
            F[v, u] = current * alpha_u * alpha_v
    return F

def idct(F, width, height):
    f = np.zeros((height, width), dtype=float)
    for x in range(width):
        for y in range(height):
            current = 0
            for u in range(width):
                alpha_u = math.sqrt(1.0/width) if u == 0 else math.sqrt(2.0/width)
                cos_x = math.cos(((2 * x + 1) * u * math.pi)/(2 * width))
                for v in range(height):
                    alpha_v = math.sqrt(1.0/height) if v == 0 else math.sqrt(2.0/height)
                    cos_y = math.cos(((2 * y + 1) * v * math.pi)/(2 * height))
                    current += alpha_u * alpha_v * F[v, u] * cos_x * cos_y
            f[y, x] = current
    return f

def walsh(width, height):
    def build_hadamard(n):
        if n == 1:
            return np.array([[1.0]])
        H_half = build_hadamard(n // 2)
        return np.block([[H_half, H_half], [H_half, -H_half]])
    
    H = build_hadamard(width)
    sequency = [sum(H[i, j] * H[i, j+1] < 0 for j in range(width-1)) for i in range(width)]
    order = sorted(range(width), key=lambda i: sequency[i])
    W = H[order, :] / np.sqrt(width)
    f = np.array([[float(img.getpixel((x, y))) for x in range(width)] for y in range(height)])
    return W @ f @ W.T, W

def iwalsh(F, W):
    return W.T @ F @ W

def keep_top_k(coeffs, k):
    result = coeffs.copy()
    mags = np.abs(result).flatten()
    if k >= len(mags):
        return result
    threshold = np.partition(mags, -k)[-k]
    return np.where(np.abs(result) >= threshold, result, 0)

def compute_mse(original, reconstructed):
    return np.mean((original - np.clip(reconstructed, 0, 255)) ** 2)

def energy_packing_analysis():
    dft_coeffs = dft(width, height)
    dct_coeffs = dct(width, height)
    walsh_coeffs, W = walsh(width, height)
    
    percentages = [1, 5, 10, 25, 50]
    total_coeffs = width * height
    results = {'DFT': [], 'DCT': [], 'Walsh': []}
    
    for pct in percentages:
        k = max(1, int(pct * total_coeffs / 100))
        
        recon_dft = idft(keep_top_k(dft_coeffs, k), width, height)
        mse_dft = compute_mse(image_array, recon_dft)
        results['DFT'].append((pct, mse_dft, recon_dft))
        print(f"{'DFT':<10} {pct:<10} {k:<12} {mse_dft:<15.2f}")
        
        recon_dct = idct(keep_top_k(dct_coeffs, k), width, height)
        mse_dct = compute_mse(image_array, recon_dct)
        results['DCT'].append((pct, mse_dct, recon_dct))
        print(f"{'DCT':<10} {pct:<10} {k:<12} {mse_dct:<15.2f}")
        
        recon_walsh = iwalsh(keep_top_k(walsh_coeffs, k), W)
        mse_walsh = compute_mse(image_array, recon_walsh)
        results['Walsh'].append((pct, mse_walsh, recon_walsh))
        print(f"{'Walsh':<10} {pct:<10} {k:<12} {mse_walsh:<15.2f}\n")

    for i, pct in enumerate(percentages):
        best = min(['DFT', 'DCT', 'Walsh'], key=lambda x: results[x][i][1])
    
    plt.figure(figsize=(10, 6))
    for name in ['DFT', 'DCT', 'Walsh']:
        plt.plot(percentages, [r[1] for r in results[name]], 'o-', label=name, linewidth=2)
    plt.xlabel('% of Coefficients Kept')
    plt.ylabel('Mean Square Error')
    plt.title('Energy Packing: MSE vs Coefficients Kept')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig('Assignment 7 - Energy Packing of FFT and DCT Transforms/mse_comparison.png', dpi=150)
    plt.close()
    
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes[0, 0].imshow(image_array, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    
    idx = percentages.index(10)
    for i, name in enumerate(['DFT', 'DCT', 'Walsh']):
        pct, mse, recon = results[name][idx]
        diff = np.abs(image_array - np.clip(recon, 0, 255))
        axes[0, i+1].imshow(np.clip(recon, 0, 255), cmap='gray')
        axes[0, i+1].set_title(f'{name} @ 10%')
        axes[0, i+1].axis('off')
        axes[1, i+1].imshow(diff, cmap='hot', vmin=0, vmax=50)
        axes[1, i+1].set_title(f'Diff (MSE={mse:.2f})')
        axes[1, i+1].axis('off')
    
    plt.suptitle('Pixel-by-Pixel Difference at 10% Coefficients')
    plt.tight_layout()
    plt.savefig('Assignment 7 - Energy Packing of FFT and DCT Transforms/difference_images.png', dpi=150)
    plt.close()

energy_packing_analysis()