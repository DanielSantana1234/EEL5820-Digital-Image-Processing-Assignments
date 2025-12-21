from PIL import Image

image_path = "Images/Synthetic/512_synthetic.bmp"
img = Image.open(image_path)
width, height = img.size
negative_image = Image.new("L", img.size, 0xffffff)

for x in range(width):
    for y in range(height):
        gray = img.getpixel((x, y))
        negative_gray = 255 - gray
        negative_image.putpixel((x, y), negative_gray)
        negative_image.save('Assignment 1 - Negative of an image/negative.bmp')