import pytesseract
from PIL import Image
import cv2

# Open the image file
image = Image.open('../id card dataset/Aadhaar/id.jpg')

# Perform OCR using PyTesseract
text = pytesseract.image_to_string(image)


# Print the extracted text
print(text)


# ************************ tesseract tutorial ************************ # 

my_config = r"--psm 6 --oem 3"
text = pytesseract.image_to_string(Image.open('./images/id.jpg'), config=my_config)

print(text)
