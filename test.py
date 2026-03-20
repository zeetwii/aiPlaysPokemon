import requests # needed for talking to mGBA-Http
import ollama # needed to run LLM
import time # needed for sleep

from textAnalysis.textAnalyzer import TextAnalyzer # needed for text analysis
from locationTracking.locationTracker import LocationTracker


class AIplayer:
    """
    The base class for the AI to make decisions about the game
    """

    def __init__(self):
        """
        Basic initalization function
        """

        # preload the ollama model
        print("Preloading Ollama model...")
        response = ollama.chat(model='gemma3:4b-it-qat', messages=[{'role': 'system', 'content': f'Say boot up successful'}])
        print(response.message.content)

        print("Initializing Text Analyzer...")
        self.textAnalyzer = TextAnalyzer(language='en')
        print("Text Analyzer initialized.")

        print("Initializing Location Tracker...")
        self.locationTracker = LocationTracker()
        print("Location Tracker initialized.")



    def makeChoice(self):
        """
        Method for having the model make decisions about what to do next
        """

        locationResult = self.locationTracker.locatePlayer('./screenshot.png')

        locationString = f"The location tracker has determined that the player is currently at {locationResult['position']} on the map {locationResult['mapName']} with confidence {locationResult['confidence']}.  "

        print(locationString)



        response = ollama.chat(
            model='gemma3:4b-it-qat',
            messages=[{
                'role': 'user',
                'content': f'You are playing Pokemon Leaf Green.  Attached is the current screenshot of the game.  {locationString}  You can interact and control what is happening on the screen by sending back any combination of the following commands: Left, Right, Up, Down, A, B, Start, Select.  You can chain together commands but can only do a single command per line.  For example to move up and to the right you would respond with: Up\nRight\n',
                'images': ['./screenshot.png']
            }]
        )
        print(response['message']['content'])
        return response['message']['content']


    def getScreenShot(self):
        """
        Gets the current screenshot from the game and saves it locally
        """

        try:
            response = requests.post('http://localhost:5000/core/screenshot', params={
                'path': r'C:\Users\zeetw\Documents\GitHub\aiPlaysPokemon\screenshot.png'
            }, timeout=5)
            print(f'Screenshot saved with status code: {response.status_code}, response: {response.text}')
        except requests.exceptions.Timeout:
            print('Screenshot request timed out - check that mGBA has the Lua script loaded')

    def sendInput(self, inputCommand):
        """
        Sends an input command to the game in the form of a button press

        Args:
            inputCommand (str): The input command to send to the game
        """

        # do formating corrections for mGBA-Http

        formattedCommand = ''
        if inputCommand is None or inputCommand == '': # skip empty commands
            return
        
        if inputCommand.lower() == 'left':
            formattedCommand = 'Left'
        elif inputCommand.lower() == 'right':
            formattedCommand = 'Right'
        elif inputCommand.lower() == 'up':
            formattedCommand = 'Up'
        elif inputCommand.lower() == 'down':
            formattedCommand = 'Down'
        elif inputCommand.lower() == 'a':
            formattedCommand = 'A'
        elif inputCommand.lower() == 'b':
            formattedCommand = 'B'
        elif inputCommand.lower() == 'start':
            formattedCommand = 'Start'
        elif inputCommand.lower() == 'select':
            formattedCommand = 'Select'
        else:
            print('Invalid input command: ' + inputCommand)
            return
        
        response = requests.post('http://localhost:5000/mgba-http/button/tap', params={
            'button': formattedCommand
        })

        print(f'Sent input command: {formattedCommand} with status code: {response.status_code}, response: {response.text}')




if __name__ == '__main__':

    aiPlayer = AIplayer()

    while True:

        aiPlayer.getScreenShot()
        time.sleep(0.1)
        aiPlayer.textAnalyzer.extractText('./screenshot.png')
        choice = aiPlayer.makeChoice()
        time.sleep(0.1)
        commands = choice.split('\n')
        for command in commands:
            aiPlayer.sendInput(command.strip())
            time.sleep(0.1)