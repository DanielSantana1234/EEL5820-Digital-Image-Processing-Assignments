from PIL import Image
import numpy as np

image_path = "Images/Natural/512_lena_gray.bmp"
img = Image.open(image_path).convert('L')
img_array = np.array(img, dtype=np.float64)

def convolve2d(image, kernel):
    kernel = np.array(kernel, dtype=np.float64)
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    output = np.zeros_like(image, dtype=np.float64)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output[i, j] = np.sum(padded[i:i+k_h, j:j+k_w] * kernel)
    return output

def edge(image, method='sobel', threshold=None):
    if method == 'sobel':
        gx = convolve2d(image, np.array([[-1,0,1],[-2,0,2],[-1,0,1]]))
        gy = convolve2d(image, np.array([[-1,-2,-1],[0,0,0],[1,2,1]]))
        magnitude = np.sqrt(gx**2 + gy**2)
    elif method == 'kirsch':
        kernels = [
            np.array([[5,5,5],[-3,0,-3],[-3,-3,-3]]),
            np.array([[-3,5,5],[-3,0,5],[-3,-3,-3]]),
            np.array([[-3,-3,5],[-3,0,5],[-3,-3,5]]),
            np.array([[-3,-3,-3],[-3,0,5],[-3,5,5]]),
            np.array([[-3,-3,-3],[-3,0,-3],[5,5,5]]),
            np.array([[-3,-3,-3],[5,0,-3],[5,5,-3]]),
            np.array([[5,-3,-3],[5,0,-3],[5,-3,-3]]),
            np.array([[5,5,-3],[5,0,-3],[-3,-3,-3]])
        ]
        magnitude = np.max([convolve2d(image, k) for k in kernels], axis=0)
    
    magnitude = magnitude / magnitude.max() * 255
    if threshold is not None:
        binary = np.zeros_like(magnitude, dtype=np.uint8)
        binary[magnitude >= threshold] = 255
        return binary
    return magnitude.astype(np.uint8)


for thresh in [50, 100, 150, 200]:
    Image.fromarray(edge(img_array, 'sobel', thresh)).save(f"sobel_thresh_{thresh}.bmp")
    Image.fromarray(edge(img_array, 'kirsch', thresh)).save(f"kirsch_thresh_{thresh}.bmp")
Image.fromarray(edge(img_array, 'sobel')).save("Assignment 10 - Edge Detection/sobel_magnitude.bmp")
Image.fromarray(edge(img_array, 'kirsch')).save("Assignment 10 - Edge Detection/kirsch_magnitude.bmp")