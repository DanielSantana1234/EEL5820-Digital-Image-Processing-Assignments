from PIL import Image
import numpy as np
import math

image_path = "Images/Natural/512_natural.bmp"

# Standard JPEG luminance quantization matrix
QUANTIZATION_MATRIX = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=float)

# Zigzag scan order for 8x8 block
ZIGZAG_ORDER = [
    (0,0), (0,1), (1,0), (2,0), (1,1), (0,2), (0,3), (1,2),
    (2,1), (3,0), (4,0), (3,1), (2,2), (1,3), (0,4), (0,5),
    (1,4), (2,3), (3,2), (4,1), (5,0), (6,0), (5,1), (4,2),
    (3,3), (2,4), (1,5), (0,6), (0,7), (1,6), (2,5), (3,4),
    (4,3), (5,2), (6,1), (7,0), (7,1), (6,2), (5,3), (4,4),
    (3,5), (2,6), (1,7), (2,7), (3,6), (4,5), (5,4), (6,3),
    (7,2), (7,3), (6,4), (5,5), (4,6), (3,7), (4,7), (5,6),
    (6,5), (7,4), (7,5), (6,6), (5,7), (6,7), (7,6), (7,7)
]


def dct_2d(block):
    N = 8
    F = np.zeros((N, N), dtype=float)
    for u in range(N):
        alpha_u = math.sqrt(1.0/N) if u == 0 else math.sqrt(2.0/N)
        for v in range(N):
            alpha_v = math.sqrt(1.0/N) if v == 0 else math.sqrt(2.0/N)
            total = 0.0
            for x in range(N):
                cos_x = math.cos(((2*x + 1) * u * math.pi) / (2*N))
                for y in range(N):
                    cos_y = math.cos(((2*y + 1) * v * math.pi) / (2*N))
                    total += block[x, y] * cos_x * cos_y
            F[u, v] = alpha_u * alpha_v * total
    return F


def idct_2d(F):
    N = 8
    block = np.zeros((N, N), dtype=float)
    for x in range(N):
        for y in range(N):
            total = 0.0
            for u in range(N):
                alpha_u = math.sqrt(1.0/N) if u == 0 else math.sqrt(2.0/N)
                cos_x = math.cos(((2*x + 1) * u * math.pi) / (2*N))
                for v in range(N):
                    alpha_v = math.sqrt(1.0/N) if v == 0 else math.sqrt(2.0/N)
                    cos_y = math.cos(((2*y + 1) * v * math.pi) / (2*N))
                    total += alpha_u * alpha_v * F[u, v] * cos_x * cos_y
            block[x, y] = total
    return block


def quantize(dct_block, Q):
    return np.round(dct_block / Q).astype(int)


def dequantize(quantized_block, Q):
    return quantized_block * Q


def zigzag_scan(block):
    return [block[pos[0], pos[1]] for pos in ZIGZAG_ORDER]


def inverse_zigzag(arr):
    block = np.zeros((8, 8), dtype=float)
    for i, pos in enumerate(ZIGZAG_ORDER):
        block[pos[0], pos[1]] = arr[i]
    return block


def run_length_encode(zigzag_arr):
    encoded = []
    current_val, count = zigzag_arr[0], 1
    for val in zigzag_arr[1:]:
        if val == current_val:
            count += 1
        else:
            encoded.append((current_val, count))
            current_val, count = val, 1
    encoded.append((current_val, count))
    return encoded


def run_length_decode(encoded):
    result = []
    for val, count in encoded:
        result.extend([val] * count)
    return result


def jpeg_encode(img, Q):
    width, height = img.size
    img_array = np.array(img, dtype=float) - 128
    
    encoded_blocks = []
    for by in range(height // 8):
        for bx in range(width // 8):
            block = img_array[by*8:(by+1)*8, bx*8:(bx+1)*8]
            dct_block = dct_2d(block)
            quantized = quantize(dct_block, Q)
            zigzag = zigzag_scan(quantized)
            rle = run_length_encode(zigzag)
            encoded_blocks.append(rle)
    
    return encoded_blocks


def jpeg_decode(encoded_blocks, width, height, Q):
    img_array = np.zeros((height, width), dtype=float)
    
    block_idx = 0
    for by in range(height // 8):
        for bx in range(width // 8):
            zigzag = run_length_decode(encoded_blocks[block_idx])
            quantized = inverse_zigzag(zigzag)
            dct_block = dequantize(quantized, Q)
            block = idct_2d(dct_block)
            img_array[by*8:(by+1)*8, bx*8:(bx+1)*8] = block
            block_idx += 1
    
    img_array = np.clip(img_array + 128, 0, 255)
    return Image.fromarray(img_array.astype(np.uint8), mode='L')


def calculate_compression_ratio(encoded_blocks, width, height):
    original_bits = width * height * 8
    compressed_bits = sum(
        sum(max(1, int(math.log2(abs(val)+1)) + 2) + max(1, int(math.log2(count+1))) 
            for val, count in block)
        for block in encoded_blocks
    )
    return original_bits / compressed_bits


def create_correlated_image(size=256):
    img = Image.new('L', (size, size))
    for x in range(size):
        for y in range(size):
            val = int((x + y) / 2 + 50 * math.sin(x/30) * math.cos(y/30))
            img.putpixel((x, y), max(0, min(255, val)))
    return img


def create_decorrelated_image(size=256):
    np.random.seed(42)
    return Image.fromarray(np.random.randint(0, 256, (size, size), dtype=np.uint8), mode='L')

correlated_img = create_correlated_image(256)
decorrelated_img = create_decorrelated_image(256)

correlated_img.save("Assignment 11 - JPEG Image Data Comprression/correlated_test.bmp")
decorrelated_img.save("Assignment 11 - JPEG Image Data Comprression/decorrelated_test.bmp")

Q = QUANTIZATION_MATRIX

corr_encoded = jpeg_encode(correlated_img, Q)
corr_decoded = jpeg_decode(corr_encoded, 256, 256, Q)
corr_ratio = calculate_compression_ratio(corr_encoded, 256, 256)

decorr_encoded = jpeg_encode(decorrelated_img, Q)
decorr_decoded = jpeg_decode(decorr_encoded, 256, 256, Q)
decorr_ratio = calculate_compression_ratio(decorr_encoded, 256, 256)

corr_decoded.save("Assignment 11 - JPEG Image Data Comprression/correlated_reconstructed.bmp")
decorr_decoded.save("Assignment 11 - JPEG Image Data Comprression/decorrelated_reconstructed.bmp")

print(f"Correlated Image Compression Ratio:   {corr_ratio:.2f}:1")
print(f"Decorrelated Image Compression Ratio: {decorr_ratio:.2f}:1")
print(f"\nCorrelated achieves {corr_ratio/decorr_ratio:.1f}x better compression")
