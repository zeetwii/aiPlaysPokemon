import easyocr # needed for text extraction
import cv2 # needed for image preprocessing
import numpy as np # needed for image manipulation

class TextAnalyzer:
    def __init__(self, language='en', debug=False):
        """
        Basic Initalization method for Text analysis

        Args:
            language (str, optional): The language to use for OCR. Defaults to 'en'.
            debug (bool, optional): If True, saves preprocessed images for inspection. Defaults to False.
        """
        self.reader = easyocr.Reader([language]) # initialize the OCR reader for the specified language
        self.debug = debug

    def preprocessImage(self, imagePath, scaleFactor=4, threshold=180, debug=False):
        """
        Preprocesses a GBA screenshot for better OCR accuracy.
        Upscales using nearest-neighbor interpolation to preserve pixel art edges,
        then converts to grayscale and applies binary thresholding.

        Args:
            imagePath (str): The path to the image file to preprocess.
            scaleFactor (int, optional): How much to upscale the image. Defaults to 4.
            threshold (int, optional): The binary threshold value (0-255). Pixels above
                this become white, below become black. Defaults to 180.
            debug (bool, optional): If True, saves the preprocessed image alongside
                the original with a '_debug' suffix. Defaults to False.

        Returns:
            numpy.ndarray: The preprocessed image ready for OCR.
        """
        image = cv2.imread(imagePath)

        if image is None:
            raise FileNotFoundError(f"Could not read image at: {imagePath}")

        # Upscale with nearest-neighbor to keep pixel art edges sharp
        height, width = image.shape[:2]
        upscaled = cv2.resize(
            image,
            (width * scaleFactor, height * scaleFactor),
            interpolation=cv2.INTER_NEAREST
        )

        # Convert to grayscale
        #gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)

        # Apply binary threshold: dark text on light background
        #_, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        if debug:
            import os
            name, ext = os.path.splitext(imagePath)
            debugPath = f"{name}_debug{ext}"
            #cv2.imwrite(debugPath, binary) # Save the binary image for debugging
            #cv2.imwrite(debugPath, gray) # Save the grayscale image for debugging
            cv2.imwrite(debugPath, upscaled) # Save the upscaled image for debugging
            print(f"[DEBUG] Saved preprocessed image to: {debugPath}")

        #return binary
        #return gray
        return upscaled

    def extractText(self, imagePath):
        """
        Extracts text from the given image using easyocr, with preprocessing
        to improve accuracy on GBA screenshots.

        Args:
            imagePath (str): The path to the image file from which text is to be extracted.

        Returns:
            list: A list of tuples containing the extracted text and confidence levels.
        """

        # Preprocess the image before OCR
        processedImage = self.preprocessImage(imagePath, debug=self.debug)

        # Use easyocr to extract text from the preprocessed image
        result = self.reader.readtext(processedImage)

        # Stores the found text and confidence levels
        foundText = []

        # Loop through the results and print the text and confidence levels
        for (bbox, text, confidence) in result:
            foundText.append((text, confidence))

        return foundText
    
if __name__ == "__main__":
    # Example usage
    textAnalyzer = TextAnalyzer(language='en', debug=True)

    print("Text Analyzer Initialized. You can now extract text from images.\n")

    while True:
        imagePath = input("Enter the path to the image file (default: '../screenshot.png'): ") or '../screenshot.png'
        extractedText = textAnalyzer.extractText(imagePath)
        print("Extracted Text and Confidence Levels:")
        for text, confidence in extractedText:
            print(f'Text: {text}, Confidence: {confidence}')
        print("\nPress Ctrl+C to exit or continue to extract text from another image.\n")