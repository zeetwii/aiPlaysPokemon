from PIL import Image # needed for image processing

import pytesseract # needed for OCR

while True:
    imagePath = input("Enter the path to the image file (or 'exit' to quit): ")
    if imagePath.lower() == 'exit':
        break

    try:
        # Open the image using PIL
        image = Image.open(imagePath)

        # Perform OCR using pytesseract
        text = pytesseract.image_to_string(image)

        print("Extracted Text:")
        print(text)
        print("\nPress Enter to continue or type 'exit' to quit.\n")
    except Exception as e:
        print(f"An error occurred: {e}")