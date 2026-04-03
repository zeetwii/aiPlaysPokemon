import ollama # needed to run LLM
import time # needed for sleep

from pathlib import Path # needed for file path handling

class OCRtest:
    """
    A class for test Gemma4 OCR capabilities
    """

    def __init__(self):
        """
        Basic initalization function
        """

        self.pullImages()

        # preload the ollama model
        print("Preloading Ollama model...")
        response = ollama.chat(model='gemma4:e4b', messages=[{'role': 'system', 'content': f'Say boot up successful'}])
        print(response.message.content)


    def pullImages(self):
        """
        Pulls all images from the testPhotos library and stores their paths
        """
        self.imagePaths = list(Path('./testPhotos').glob('*.png'))
        print(f"Found {len(self.imagePaths)} images in the testPhotos library.")

    def testOCR(self, imagePath):
        """
        Method for testing the OCR capabilities of Gemma4
        """

        if not Path(imagePath).is_file():
            print(f"Error: File {imagePath} does not exist.")
            return

        response = ollama.chat(model='gemma4:e4b', messages=[{'role': 'system', 'content': f'You are an OCR engine. Extract all text from the image and return it as a string.'}, {'role': 'user', 'images': [f"{imagePath}"]}])
        
        print("Extracted text:")
        print(response.message.content)

if __name__ == "__main__":
    print("Starting OCR test...")
    
    ocrTest = OCRtest()
    
    while True:

        print("\nSelect an image to test OCR on:")
        for i in range(len(ocrTest.imagePaths)):
            print(f"Image {i}: {ocrTest.imagePaths[i]}")
        
        choice = input("Enter the number of the image you want to test (or 'q' to quit): ")

        if choice.lower() == 'q' or choice.lower() == 'quit':
            print("Exiting OCR test.")
            break
        elif choice.isdigit() and int(choice) < len(ocrTest.imagePaths):
            ocrTest.testOCR(ocrTest.imagePaths[int(choice)])
        else:
            print("Invalid choice. Please try again.")