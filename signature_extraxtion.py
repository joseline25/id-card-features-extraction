"""
Implementing signature extraction

A signature extraction system can be developed in two ways: traditional computer vision
using OpenCV and object detection with deep learning. In this tutorial, you'll be 
implementing the first solution using Python 3.9 
"""

import cv2
import matplotlib.pyplot as plt
from skimage import measure, morphology
from skimage.color import label2rgb
from skimage.measure import regionprops
import numpy as np


"""
Read the input image file from the local path and apply preprocessing that will help
in the identification of the signature area
"""

img = cv2.imread('../id card dataset/Aadhaar/id.jpg', 0) # the front of our id card

# crop the image with opencv and show it

y=250
x=400
h=550
w=700
crop_img = img[x:w, y:h]
cv2.imshow("Cropped", crop_img)
cv2.waitKey(0)
plt.show()

thresh_crop_image = cv2.threshold(crop_img, 127, 255, cv2.THRESH_BINARY)[1]

# apply thresholding to partition the background and foreground of grayscale 
# image by essentially making them black and white
# the 0 parameter in cv.imread() indicates that the image has one color channel;
# in other words, it’s a black and white or grayscale image
img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]

# Binary thresholding is the process of converting image pixels to black or white
# given a threshold, in this case 127

# we compare each pixel to a given threshold value. if the pixel is less than that value,
# we turn it to white else, black

def show_image(image, title='Image', cmap_type='gray'):
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()


show_image(img, "Front ID Card")
show_image(thresh_crop_image, "Cropped  ID Card")

# Work with the cropped image 
img = thresh_crop_image

"""
Now that the image is ready, connected component analysis must be applied to 
detect the connected regions in the image. This helps in identifying the 
signature area, as signature characters are coupled together. skimage provides
a function to do this
"""

# connected component analysis by scikit-learn framework

# identify blobs whose size is greater than the image pixel average
blobs = img > img.mean()

# measure the size of each blob
blobs_labels = measure.label(blobs, background=1)

# the blob labels are converted to RGB and are overlaid on the original image
# for better visualization
image_label_overlay = label2rgb(blobs_labels, image=img)

"""

A blob is a set of pixel values that generally distinguishes an object from
its background. In this case, the text and signature are blobs on a background
of white pixels.
 
"""

# draw image
# fix the figure size to (10, 6)
fig, ax = plt.subplots(figsize=(10, 6))

# plot the connected components (for debugging)
ax.imshow(image_label_overlay)
ax.set_axis_off()
plt.tight_layout()
plt.show()

"""
Generally, a signature will be bigger than other text areas in a document, 
so you need to do some measurements. Using component analysis, find the biggest
component among the blobs
"""

# initialize the variables to get the biggest component
the_biggest_component = 0
total_area = 0
counter = 0
average = 0.0

# iterate over each blob and get the highest size component
for region in regionprops(blobs_labels):
    # if blob size is greater than 10 then add it to the total area
    if (region.area > 10):
        total_area = total_area + region.area
        counter = counter + 1

    # take regions with large enough areas and filter the highest component
    if (region.area >= 250):
        if (region.area > the_biggest_component):
                the_biggest_component = region.area

# calculate the average of the blob regions
average = (total_area/counter)
print("the_biggest_component: " + str(the_biggest_component))
print("average: " + str(average))


# Next, you need to filter out some outliers that might get confused 
# with the signature blob

# the parameters are used to remove outliers of small size connected pixels
constant_parameter_1 = 84
constant_parameter_2 = 250
constant_parameter_3 = 100

# the parameter is used to remove outliers of large size connected pixels
constant_parameter_4 = 18

# experimental-based ratio calculation, modify it for your cases
a4_small_size_outlier_constant = ((average/constant_parameter_1)*constant_parameter_2)+constant_parameter_3
print("a4_small_size_outlier_constant: " + str(a4_small_size_outlier_constant))



# experimental-based ratio calculation, modify it for your cases
a4_big_size_outlier_constant = a4_small_size_outlier_constant*constant_parameter_4
print("a4_big_size_outlier_constant: " + str(a4_big_size_outlier_constant))


# remove the connected pixels that are smaller than threshold a4_small_size_outlier_constant
pre_version = morphology.remove_small_objects(blobs_labels, a4_small_size_outlier_constant)
# remove the connected pixels that are bigger than threshold a4_big_size_outlier_constant
component_sizes = np.bincount(pre_version.ravel())
too_small = component_sizes > (a4_big_size_outlier_constant)
too_small_mask = too_small[pre_version]
pre_version[too_small_mask] = 0
# save the pre-version, which is the image with color labels after connected component analysis
plt.imsave('pre_version.png', pre_version)


# read the pre-version
img = cv2.imread('pre_version.png', 0)
# ensure a binary image with Otsu’s method
img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]


cv2.imwrite("./images/output.png", img)