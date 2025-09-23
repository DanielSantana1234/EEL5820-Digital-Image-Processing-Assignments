from PIL import Image

image_path = "Assignment 2 - Image Size Reduction/lena_gray.bmp"
img = Image.open(image_path)
width, height = img.size
image_reduction = Image.new("L", (img.size[0]//2, img.size[1]//2), 0xffffff)

for x in range(0, width, 2):
    for y in range(0, height, 2):
        # get the next four pixels
        current_pixel1 = img.getpixel((x,y))
        current_pixel2 = img.getpixel((x+1, y))
        current_pixel3 = img.getpixel((x, y+1))
        current_pixel4 = img.getpixel((x+1, y+1))

        # sum them up now then divide by four to get the average
        average_pixel = (current_pixel1 + current_pixel2+ current_pixel3 + current_pixel4) / 4
        image_reduction.putpixel((x//2, y//2), int(average_pixel))
        image_reduction.save('Assignment 2 - Image Size Reduction/reduced_lena_gray.bmp')
