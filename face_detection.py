"""
Face recognition using Artificial Intelligence



Face recognition using Artificial Intelligence(AI) is a computer vision
technology that is used to identify a person or object from an image or
video. It uses a combination of techniques including deep learning,  
computer vision algorithms, and Image processing. These technologies 
are used to enable a system to detect, recognize, and verify faces in
digital images or videos.

Face recognition  vs Face detection 


Face recognition is the process of identifying a person from an image or
video feed and face detection is the process of detecting a face in an 
image or video feed. In the case of  Face recognition, someone's face is
recognized and differentiated based on their facial features. It involves
more advanced processing techniques to identify a person's identity based
on feature point extraction, and comparison algorithms.


While Face detection is a much simpler process and can be used for applications
such as image tagging or altering the angle of a photo bas ed on the face detected.
it is the initial step in the face recognition process and is a simpler process 
that simply identifies a face in an image or video feed.

Process

- Image reading
    For any color image, there are 3 primary colors – Red, green, and blue.
    A matrix is formed for every primary color and later these matrices
    combine to provide a Pixel value for the individual R, G, and B colors.
    Each element of the matrices provide data about the intensity of the 
    brightness of the pixel.
    
- 
"""

# Face Detection using OpenCv

"""
we will learn to apply a popular face detection approach called Haar Cascade 
for face detection using OpenCV and Python.


This method was first introduced in the paper Rapid Object Detection Using a 
Boosted Cascade of Simple Features, written by Paul Viola and Michael Jones.

The idea behind this technique involves using a cascade of classifiers to detect
different features in an image. These classifiers are then combined into one strong
classifier that can accurately distinguish between samples that contain a human 
face from those that don't.

The Haar Cascade classifier that is built into OpenCV has already been trained on
a large dataset of human faces, so no further training is required. We just need
to load the classifier from the library and use it to perform face detection on 
an input image.


"""
# read pdf file

from pdf2image import convert_from_path
import numpy as np

# Step 1: Import the OpenCV Package
import cv2

imagePath = '../id card dataset/Aadhaar/id_back.jpg'

# Step 2: Read the Image

img = cv2.imread(imagePath)
# or from a pdf
# pdfPath = '../id card dataset/Aadhaar/MaCYBSEC_02 (6).pdf'

# pages = convert_from_path(pdfPath)
# img = np.array(pages[0])

# This will load the image from the specified file path and return it
# in the form of a Numpy array. 

#  get the dimensions of the image

print(img.shape) # (634, 1024, 3)

"""
Notice that this is a 3-dimensional array. The array's values represent the picture's 
height, width, and channels respectively. Since this is a color image, there are three
channels used to depict it - blue, green, and red (BGR). 



Note that while the conventional sequence used to represent images is RGB 
(Red, Blue, Green), the OpenCV library uses the opposite layout (Blue, Green, Red).
"""

# Step 3: Convert the Image to Grayscale

"""
To improve computational efficiency, we first need to convert this image to
grayscale before performing face detection on it
"""
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(gray_image.shape) # (634, 1024)

"""
Notice that this array only has two values since the image is grayscale 
and no longer has the third color channel.
"""
# Step 4: Load the Classifier

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

"""
Notice that we are using a file called haarcascade_frontalface_default.xml.
This classifier is designed specifically for detecting frontal faces in
visual input. 


OpenCV also provides other pre-trained models to detect different objects
within an image - such as a person's eyes, smile, upper body, and even a 
vehicle's license plate. You can learn more about the different classifiers 
built into OpenCV by examining the library's GitHub repository.
"""
# Step 5: Perform the Face Detection

face = face_classifier.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
)

# Step 6: Drawing a Bounding Box

for (x, y, w, h) in face:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
    
# Step 7: Displaying the Image

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
import matplotlib.pyplot as plt



plt.figure(figsize=(20,10))
plt.imshow(img_rgb)
plt.axis('off')
plt.show()

# It works


"""
One way to expand this project is to identify human faces in different types
of input data, such as PDF files or surveillance images. 

Also, you can create a face detection model on large datasets.

Detecting whether a person is wearing masks in image datasets.
"""


# convert pdf to image then to a numpy array

pdfPath = '../id card dataset/Aadhaar/MaCYBSEC_02 (6).pdf'

pages = convert_from_path(pdfPath)
img = np.array(pages[0])

# opencv code to view image
img = cv2.resize(img, None, fx=0.5, fy=0.5)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# or 

plt.figure(figsize=(15,10))
plt.imshow(img)
plt.axis('off')
plt.show()


