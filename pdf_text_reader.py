# importing required modules
from PyPDF2 import PdfReader
import random
from googletrans import Translator

# creating a pdf reader object
reader = PdfReader(
    './Les oiseaux se cachent pour mou - Colleen McCullough.pdf')

# printing number of pages in pdf file
print(len(reader.pages))

# getting a specific page from the pdf file
page = reader.pages[5]

# extracting text from page
text = page.extract_text()
# print(text)

# create a text file and append the whole pdf

with open('mypdf.txt', 'w+') as f:
    f.write('My Pdf file reader!!')

for page in reader.pages:
    text = page.extract_text()
    with open('mypdf.txt', 'a+', encoding='utf-8') as f:
        f.write(text)
        f.write("\n")


# Close the file
f.close()


# ******************************** the method ************************************ #


def pdf_to_txt(path='./Les oiseaux se cachent pour mou - Colleen McCullough.pdf'):
    # create a translator object
    translator = Translator()
    
    
    reader = PdfReader(path)
    # create a text file and append the whole pdf
    val = str(random.randint(1, 2000))
    with open('./pdf/mypdf'+val+'.txt', 'a+', encoding='utf-8') as f:
        f.write('My Pdf file reader!!')

    for page in reader.pages:
        text = page.extract_text()
        
        # translate 
        text_tranlated = translator.translate(text, dest='fr')
        with open('./pdf/mypdf'+val+'.txt', 'a+', encoding='utf-8') as f:
            f.write(text)
            f.write("\n")
    f.close()


pdf_to_txt('./pdf/pdfcoffee.com_larisarenarthecircleoffemininepowertheenergyoftheelementspdf-pdf-free (1).pdf')
