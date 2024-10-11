import pytesseract
from PIL import Image
import os
import sys

# Optional: Print Python executable to verify environment
print("Python Executable:", sys.executable)

# Optional: Print current PATH to verify Tesseract's presence
print("Current PATH:")
for path in os.environ.get('PATH', '').split(os.pathsep):
    print(path)

# Specify the exact path to the tesseract executable
# Update this path if Tesseract is installed in a different directory
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Path to your image file
image_path = r'E:\Doc_Analyser\sample.jpg'  # Update with your actual image path

# Verify that the image exists
if not os.path.isfile(image_path):
    raise FileNotFoundError(f"The image file was not found: {image_path}")

# Load the image using PIL
try:
    image = Image.open(image_path)
except Exception as e:
    raise IOError(f"Error opening image file: {e}")

# Perform OCR using pytesseract
try:
    text = pytesseract.image_to_string(image)
    print("Extracted Text:")
    print(text)
except pytesseract.pytesseract.TesseractNotFoundError:
    print("Tesseract OCR executable not found. Please ensure it is installed and added to PATH.")
except Exception as e:
    print(f"An error occurred during OCR: {e}")

# For layout-aware output
try:
    hocr = pytesseract.image_to_pdf_or_hocr(image, extension='hocr')
    with open('output.hocr', 'w', encoding='utf-8') as f:
        f.write(hocr.decode('utf-8'))
    print("HOCR output saved to 'output.hocr'.")
except Exception as e:
    print(f"An error occurred while generating HOCR output: {e}")