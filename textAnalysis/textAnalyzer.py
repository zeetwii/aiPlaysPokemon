import easyocr # needed for text extraction

class TextAnalyzer:
    def __init__(self, language='en'):
        """
        Basic Initalization method for Text analysis

        Args:
            language (str, optional): The language to use for OCR. Defaults to 'en'.
        """
        self.reader = easyocr.Reader([language]) # initialize the OCR reader for the specified language

    def extractText(self, imagePath):
        """
        Extracts text from the given image using easyocr

        Args:
            imagePath (str): The path to the image file from which text is to be extracted.

        Returns:
            list: A list of duples containing the extracted text and confidence levels.
        """

        # Use easyocr to extract text from the image
        result = self.reader.readtext(imagePath)

        # Stores the found text and confidence levels
        foundText = []

        # Loop through the results and print the text and confidence levels
        for (bbox, text, confidence) in result:
            print(f'Text: {text}, Confidence: {confidence}')
            foundText.append((text, confidence))

        return foundText
    
if __name__ == "__main__":
    # Example usage
    textAnalyzer = TextAnalyzer(language='en')

    print ("Extracting text from screenshot.png...")
    extractedText = textAnalyzer.extractText('../screenshot.png')