from skimage.filters import gaussian
from skimage.filters import sobel
from skimage import color
from skimage.color import rgb2gray
from skimage.filters import try_all_threshold
from skimage.filters import threshold_local
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import numpy as np
from skimage import data, color
rocket_image = data.rocket()
coffee_image = data.coffee()
coin_image = data.coins()
print(rocket_image.shape)  # (427, 640, 3)
print(coffee_image.shape)  # (400, 600, 3)
print(coin_image.shape)  # (303, 384)

grayscale = color.rgb2gray(rocket_image)
rgb = color.gray2rgb(grayscale)


def show_image(image, title='Image', cmap_type='gray'):
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()


show_image(grayscale, "Grayscale")


# Numpy for image

image = plt.imread('./images/ev1.JPG')
print(type(image))  # <class 'numpy.ndarray'>

# obtaining the red values of the image

red = image[:, :, 0]

# obtaining the green values of the image

green = image[:, :, 1]

# obtaining the blue values of the image

blue = image[:, :, 2]

# display the red nuances in gray

plt.imshow(red, cmap="gray")
plt.title('Red')
plt.axis('off')
plt.show()

# plt.imshow(green, cmap="gray")
# plt.imshow(blue, cmap="gray")


# get the shape
print(image.shape)  # (823, 1545, 3)
print(image.size)  # 3814605

# flipping the image

vertically_flipped = np.flipud(image)
show_image(vertically_flipped, 'Vertically Flipped')

horizontally_flipped = np.fliplr(image)
show_image(horizontally_flipped, 'Horizontally Flipped')

# flip vertically then horizontally and finaly put the image in gray
flip_image = np.flip(image)
show_image(flip_image, 'Flipped')


# show the histogram of red image

plt.hist(red.ravel(), bins=256)
plt.show()

# show the histogram of the image
plt.hist(image.ravel(), bins=256)
plt.show()

# Thresholding

thresh = 127

binary = grayscale > thresh
print(type(binary))  # <class 'numpy.ndarray'>
print(type(image))  # <class 'numpy.ndarray'>
inverted_binary = image <= thresh
show_image(rocket_image, 'Original')
show_image(binary, 'Thresholded')


# Import the otsu threshold function


chess_pieces_image = data.rocket()
# Make the image grayscale using rgb2gray
chess_pieces_image_gray = color.rgb2gray(chess_pieces_image)

# Obtain the optimal threshold value with otsu
thresh = threshold_otsu(chess_pieces_image_gray)

# Apply thresholding to the image
binary = chess_pieces_image_gray > thresh

# Show the image
show_image(binary, 'Binary image')

"""
Awesome! You just converted the image to binary and we can
separate the foreground from the background.






When the background isn't that obvious

Sometimes, it isn't that obvious to identify the background.
If the image background is relatively uniform, then you can use
a global threshold value as we practiced before, using threshold_otsu().
However, if there's uneven background illumination, adaptive thresholding 
threshold_local() (a.k.a. local thresholding) may produce better results.
"""
# Import the otsu threshold function

# Obtain the optimal otsu global thresh value
block_size = 35
global_thresh = threshold_local(grayscale, block_size, offset=10)

# Obtain the binary image by applying global thresholding
binary_global = grayscale > global_thresh

# Show the binary image obtained
show_image(binary_global, 'Global thresholding')


"""
When we are not sure on what algorithm to use
"""

# Import the try all function

# Import the rgb to gray convertor function

# Turn the fruits_image to grayscale
grayscale = rgb2gray(rocket_image)

# Use the try all method on the resulting grayscale image
fig, ax = try_all_threshold(grayscale, verbose=False)

# Show the resulting plots
plt.show()

"""
Apply thresholding
In this exercise, you will decide what type of thresholding is best
used to binarize an image of knitting and craft tools. In doing so,
you will be able to see the shapes of the objects, from paper hearts
to scissors more clearly.
"""


"""
Edge detection
In this exercise, you'll detect edges in an image by applying the Sobel filter.
"""
# Import the color module

# Import the filters module and sobel function

# Make the image grayscale
soaps_image_gray = color.rgb2gray(rocket_image)

# Apply edge detection filter
edge_sobel = sobel(soaps_image_gray)

# Show original and resulting image to compare
show_image(rocket_image, "Original")
show_image(edge_sobel, "Edges with Sobel")


"""
Blurring to reduce noise
In this exercise you will reduce the sharpness of an image of a building taken
during a London trip, through filtering.
"""

# Import Gaussian filter

# Apply filter
gaussian_image = gaussian(rocket_image, channel_axis=-1)

# Show original and resulting image to compare
show_image(gaussian_image, "Original")
show_image(gaussian_image, "Reduced sharpness Gaussian")


"""

# Import the required module
from skimage import exposure

# Show original x-ray image and its histogram
show_image(chest_xray_image, 'Original x-ray')

plt.title('Histogram of image')
plt.hist(chest_xray_image.ravel(), bins=256)
plt.show()

# Use histogram equalization to improve the contrast
xray_image_eq =  exposure.equalize_hist(chest_xray_image)

show_image(xray_image_eq)






Aerial image
In this exercise, we will improve the quality of an aerial image of a city.
The image has low contrast and therefore we can not distinguish all the 
elements in it.

Image loaded as image_aerial


- Import the required module from scikit-image.
- Use the histogram equalization function from the module previously imported.
- Show the resulting image


# Import the required module
from skimage import exposure

# Use histogram equalization to improve the contrast
image_eq =  exposure.equalize_hist(image_aerial)

# Show the original and resulting image
show_image(image_aerial, 'Original')
show_image(image_eq, 'Resulting image')






Let's add some impact and contrast
Have you ever wanted to enhance the contrast of your photos so that they appear
more dramatic?

In this exercise, you'll increase the contrast of a cup of coffee.
Something you could share with your friends on social media. Don't 
forget to use #ImageProcessingDatacamp as hashtag!

Even though this is not our Sunday morning coffee cup, you can still 
apply the same methods to any of our photos.

A function called show_image(), that displays an image using Matplotlib,
has already been defined. It has the arguments image and title, with title 
being 'Original' by default.



Import the module that includes the Contrast Limited Adaptive Histogram Equalization
(CLAHE) function.

Obtain the image you'll work on, with a cup of coffee in it, from the module that holds
all the images for testing purposes.

From the previously imported module, call the function to apply the adaptive
equalization method on the original image and set the clip limit to 0.03.
"""
# exercise to show contrast

# Import the required module
from skimage import exposure

# load the image
chest_xray_image = plt.imread('./images/contrast_plus.png')

# Show original x-ray image and its histogram
show_image(chest_xray_image, 'Original x-ray')

plt.title('Histogram of image')
plt.hist(chest_xray_image.ravel(), bins=256)
plt.show()

# Use histogram equalization to improve the contrast
xray_image_eq =  exposure.equalize_hist(chest_xray_image)

show_image(xray_image_eq)



# Exercise : Add contrast to make the image more ddramatic

# Import the necessary modules
from skimage import data, exposure

# Load the image
original_image = data.coffee()

# Apply the adaptive equalization on the original image
adapthist_eq_image = exposure.equalize_adapthist(original_image, clip_limit=0.03)

# Compare the original image to the equalized
show_image(original_image)
show_image(adapthist_eq_image, '#ImageProcessingDatacamp')

# Perfect!!!!!


"""
Aliasing, rotating and rescaling
Let's look at the impact of aliasing on images.

Remember that aliasing is an effect that causes different signals,
in this case pixels, to become indistinguishable or distorted.

You'll make this cat image upright by rotating it 90 degrees and 
then rescaling it two times. Once with the anti aliasing filter 
applied before rescaling and a second time without it, so you 
can compare them.
"""

# Import the module and the rotate and rescale functions
from skimage.transform import rotate, rescale

image_cat = data.cat()
show_image(image_cat, 'Cat!')
# Rotate the image 90 degrees clockwise 
rotated_cat_image = rotate(image_cat, -90)

# Rescale with anti aliasing
# on remplace multichannel=True avec channel_axis=-1
rescaled_with_aa = rescale(rotated_cat_image, 1/4, anti_aliasing=True, channel_axis=-1)

# Rescale without anti aliasing
rescaled_without_aa = rescale(rotated_cat_image, 1/4, anti_aliasing=False, channel_axis=-1)

# Show the resulting images
show_image(rescaled_with_aa, "Transformed with anti aliasing")
show_image(rescaled_without_aa, "Transformed without anti aliasing")


"""
Enlarging images
Have you ever tried resizing an image to make it larger? This usually results
in loss of quality, with the enlarged image looking blurry.

The good news is that the algorithm used by scikit-image works very well 
for enlarging images up to a certain point.

In this exercise you'll enlarge an image three times!!

You'll do this by rescaling the image of a rocket, that will be loaded from 
the data module.


Import the module and function needed to enlarge images, you'll do this by rescaling.
Import the data module.
Load the rocket() image from data.
Enlarge the rocket_image so it is 3 times bigger, with the anti aliasing filter applied.
Make sure to set multichannel to True or you risk your session timing out!

"""


# Import the module and function to enlarge images
from skimage.transform import rescale



# Load the image from data
rocket_image = data.rocket()

# Enlarge the image so it is 3 times bigger
enlarged_rocket_image = rescale(rocket_image, 3, anti_aliasing=True, channel_axis=-1)

# Show original and resulting image
show_image(rocket_image)
show_image(enlarged_rocket_image, "3 times enlarged image")


"""
Proportionally resizing
We want to downscale the images of a veterinary blog website so all of them have the same 
compressed size.

It's important that you do this proportionally, meaning that these are not distorted.

First, you'll try it out for one image so you know what code to test later in the rest 
of the pictures.

Import the module and function to resize.

Set the proportional height and width so it is half the image's height size.

Resize using the calculated proportional height and width.
"""

# Import the module and function
from skimage.transform import resize

dogs_banner = data.cat()

# Set proportional height so its half its size
height = int(dogs_banner.shape[0]/ 2)
width = int(dogs_banner.shape[1] / 2)

# Resize using the calculated proportional height and width
image_resized = resize(dogs_banner, (height, width),
                       anti_aliasing=True)

# Show the original and resized image
show_image(dogs_banner, 'Original')
show_image(image_resized, 'Resized image')