import ollama # needed to run LLM
import time # needed for sleep

import socket # needed for lua script communication with mGBA
import requests # needed for http communication with mGBA

from textAnalysis.textAnalyzer import TextAnalyzer # needed for text analysis
from locationTracking.locationTracker import LocationTracker # needed for location tracking


class AIplayer:
    """
    The base class for the AI to make decisions about the game
    """

    def __init__(self):
        """
        Basic initalization function
        """

        # TODO: turn these into a yaml config file or something similar for easier editing and readability
        self.mgbaHost = 'localhost'
        self.mgbaTCPPort = 54321
        self.mgbaRequestPort = 5000
        self.mgbaCommunication = 'tcp' # options are 'tcp' or 'http'

        if self.mgbaCommunication == 'tcp':
            print("Initializing TCP connection to mGBA Lua script...")
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.mgbaHost, self.mgbaTCPPort))
            print("TCP connection established.")

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
    

    def tcp_send_command(self, command: str) -> str:
        """
        Handles sending commands over TCP to the Lua script controlling mGBA

        Args:
            command (str): The command string to send to the Lua script

        Raises:
            ConnectionError: An error occurred while trying to send the command or receive the response

        Returns:
            str: The response from the Lua script
        """
        self.sock.sendall((command + "\n").encode("utf-8"))
        # Read until we get a newline (the response header)
        buf = b""
        while b"\n" not in buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("Server closed connection")
            buf += chunk
        nl = buf.index(b"\n")
        header = buf[:nl].decode("utf-8")
        remainder = buf[nl + 1:]
        return header, remainder


    def tcp_tap(self, button: str, frames: int = None):
        """
        Sends a button tap over TCP to the mGBA Lua script

        Args:
            button (str): The button to tap (e.g. 'A', 'B', 'Left', 'Right', etc.)
            frames (int, optional): How many frames to hold the button press. Defaults to None.
        """

        cmd = f"TAP|{button}"
        if frames is not None:
            cmd += f"|{frames}"
        header, _ = self.tcp_send_command(cmd)
        print(f"TAP {button}: {header}")

    
    def tcpScreenshot(self, output_path: str = "screenshot.png"):
        """
        Handles saving a screenshot when over TCP and talking directly to the lua script

        Args:
            sock (socket.socket): _description_
            output_path (str, optional): _description_. Defaults to "screenshot.png".

        Raises:
            ConnectionError: _description_
            ConnectionError: _description_
        """

        self.sock.sendall(b"SCREENSHOT\n")

        # Read response header: OK|<byte_length>\n
        buf = b""
        while b"\n" not in buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("Server closed connection")
            buf += chunk

        nl = buf.index(b"\n")
        header = buf[:nl].decode("utf-8")
        remainder = buf[nl + 1:]

        if header.startswith("ERR"):
            print(f"Screenshot failed: {header}")
            return

        # Parse byte length
        parts = header.split("|")
        byte_length = int(parts[1])

        # Read the PNG data
        png_data = remainder
        while len(png_data) < byte_length:
            chunk = self.sock.recv(min(65536, byte_length - len(png_data)))
            if not chunk:
                raise ConnectionError("Server closed connection during transfer")
            png_data += chunk

        with open(output_path, "wb") as f:
            f.write(png_data[:byte_length])
        print(f"Screenshot saved to {output_path} ({byte_length} bytes)")


    def getScreenShot(self):
        """
        Gets the current screenshot from the game and saves it locally
        """

        try:
            if self.mgbaCommunication == 'tcp':
                self.tcpScreenshot("./screenshot.png")
            elif self.mgbaCommunication == 'http':
                response = requests.post(f'http://{self.mgbaHost}:{self.mgbaRequestPort}/core/screenshot', params={
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
        
        if self.mgbaCommunication == 'tcp':
            self.tcp_tap(formattedCommand)
        elif self.mgbaCommunication == 'http':
        
            response = requests.post(f'http://{self.mgbaHost}:{self.mgbaRequestPort}/mgba-http/button/tap', params={
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