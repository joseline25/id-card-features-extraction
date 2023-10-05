"""
pip install opencv-python
pip install pytesseract


OpenCV (Open source computer vision) is a library of programming functions mainly
aimed at real-time computer vision. OpenCV in python helps to process an image and 
apply various functions like resizing image, pixel manipulations, object detection,
etc. In this article, we will learn how to use contours to detect the text in an 
image and save it to a text file.

After the necessary imports, a sample image is read using the imread function of opencv.


I - image processing for the image

    - The colorspace of the image is first changed and stored in a variable.
    - For color conversion we use the function cv2.cvtColor(input_image, flag)
    The second parameter flag determines the type of conversion. We can chose 
    among cv2.COLOR_BGR2GRAY and cv2.COLOR_BGR2HSV.
    
    cv2.COLOR_BGR2GRAY helps us to convert an RGB image to gray scale image and 
    cv2.COLOR_BGR2HSV is used to convert an RGB image to HSV (Hue, Saturation, Value)
    color-space image. Here, we use cv2.COLOR_BGR2GRAY. A threshold is applied to the
    converted image using cv2.threshold function.
    
    There are 3 types of thresholding: 
 

        Simple Thresholding
        Adaptive Thresholding
        Otsu's Binarization
        
    - Get a rectangular structure
    - Finding Contours
    - Applying OCR

"""


# Import required packages
import cv2
import pytesseract

# Mention the installed location of Tesseract-OCR in your system
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
#'C:/Program Files/Tesseract-OCR'

# Read image from which text needs to be extracted
img = cv2.imread('../id card dataset/Aadhaar/id_back.jpg') 

## I - Preprocessing the image starts

# 1 - Convert the image to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # the image become an array of array


# 2 - Performing OTSU threshold
ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

"""
cv2.threshold() has 4 parameters, first parameter being the color-space changed image,
followed by the minimum threshold value, the maximum threshold value and the type of
thresholding that needs to be applied.
"""

# Specify structure shape and kernel size.
# Kernel size increases or decreases the area
# of the rectangle to be detected.
# A smaller value like (10, 10) will detect
# each word instead of a sentence.
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
 
 
# Applying dilation on the threshold image
dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)


# Finding contours
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_NONE)


# Creating a copy of image
im2 = img.copy()
 
# A text file is created and flushed
file = open("recognized.txt", "w+")
file.write("")
file.close()
 

# Looping through the identified contours
# Then rectangular part is cropped and passed on
# to pytesseract for extracting text from it
# Extracted text is then written into the text file
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
     
    # Drawing a rectangle on copied image
    rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
     
    # Cropping the text block for giving input to OCR
    cropped = im2[y:y + h, x:x + w]
     
    # Open the file in append mode
    file = open("recognized.txt", "a")
     
    # Apply OCR on the cropped image
    text = pytesseract.image_to_string(cropped)
     
    # Appending the text into file
    file.write(text)
    file.write("\n")
     
    # Close the file
    file.close