import cv2 # needed for template matching
import os # needed for file path operations
import numpy as np # needed for image processing

class LocationTracker:
    """
    Tracks the player's location in Pokemon Leaf Green by template matching
    game screenshots against known map images using OpenCV.

    Designed to be extended for navigation and pathing.
    """

    # The player sprite sits at the center of the 240x160 GBA screen.
    # These offsets convert a template match origin to player map coordinates.
    SCREEN_WIDTH = 240
    SCREEN_HEIGHT = 160
    PLAYER_OFFSET_X = SCREEN_WIDTH // 2
    PLAYER_OFFSET_Y = SCREEN_HEIGHT // 2

    def __init__(self, mapsDirectory=None):
        """
        Loads all map images from the maps directory.

        Args:
            mapsDirectory (str, optional): Path to the folder containing map PNGs.
                Defaults to the maps folder next to this file.
        """

        if mapsDirectory is None:
            mapsDirectory = os.path.join(os.path.dirname(__file__), 'maps')

        self.maps = {}  # dict of mapName -> cv2 image (BGR)
        self._loadMaps(mapsDirectory)

        # Cached result from the last successful locate call
        self.currentMap = None
        self.currentPosition = None  # (x, y) pixel coords of player on the map
        self.currentConfidence = 0.0

        # When set, this map is checked first for a faster match
        self._lastMapName = None

    def _loadMaps(self, mapsDirectory):
        """
        Loads every image file in the maps directory into memory.

        Args:
            mapsDirectory (str): Path to the maps folder.
        """

        supportedExtensions = ('.png', '.jpg', '.jpeg', '.bmp')

        for filename in os.listdir(mapsDirectory):
            if filename.lower().endswith(supportedExtensions):
                filepath = os.path.join(mapsDirectory, filename)
                image = cv2.imread(filepath)
                if image is not None:
                    mapName = os.path.splitext(filename)[0]
                    self.maps[mapName] = image

        print(f'LocationTracker: Loaded {len(self.maps)} maps.')

    def locatePlayer(self, screenshotPath):
        """
        Finds which map the screenshot belongs to and where the player is on it.

        Args:
            screenshotPath (str): Path to the current game screenshot.

        Returns:
            dict: {
                'mapName': str,         # name of the matched map
                'position': (int, int), # (x, y) player pixel coords on the map
                'confidence': float     # match confidence 0-1
            } or None if no match found.
        """

        screenshot = cv2.imread(screenshotPath)
        if screenshot is None:
            print(f'LocationTracker: Could not read screenshot at {screenshotPath}')
            return None

        bestMatch = None
        bestConfidence = -1.0
        bestLocation = None

        # Check the last known map first for speed
        orderedMaps = self._getOrderedMaps()

        for mapName, mapImage in orderedMaps:
            # Skip maps smaller than the screenshot in either dimension
            if (mapImage.shape[0] < screenshot.shape[0] or
                    mapImage.shape[1] < screenshot.shape[1]):
                continue

            result = cv2.matchTemplate(mapImage, screenshot, cv2.TM_CCOEFF_NORMED)
            _, maxVal, _, maxLoc = cv2.minMaxLoc(result)

            if maxVal > bestConfidence:
                bestConfidence = maxVal
                bestMatch = mapName
                bestLocation = maxLoc  # (x, y) of the top-left corner of the match

        if bestMatch is None:
            print('LocationTracker: No matching map found.')
            return None

        # Convert match origin to player position (center of screen)
        playerX = bestLocation[0] + self.PLAYER_OFFSET_X
        playerY = bestLocation[1] + self.PLAYER_OFFSET_Y

        # Cache the result
        self.currentMap = bestMatch
        self.currentPosition = (playerX, playerY)
        self.currentConfidence = bestConfidence
        self._lastMapName = bestMatch

        result = {
            'mapName': bestMatch,
            'position': (playerX, playerY),
            'confidence': bestConfidence
        }

        print(f'LocationTracker: Found player on {bestMatch} at ({playerX}, {playerY}) '
              f'with confidence {bestConfidence:.4f}')

        return result

    def _getOrderedMaps(self):
        """
        Returns map items with the last matched map first for faster re-matching.

        Returns:
            list: List of (mapName, mapImage) tuples.
        """

        if self._lastMapName and self._lastMapName in self.maps:
            # Yield the last map first, then the rest
            ordered = [(self._lastMapName, self.maps[self._lastMapName])]
            ordered.extend(
                (name, img) for name, img in self.maps.items()
                if name != self._lastMapName
            )
            return ordered

        return list(self.maps.items())

    def getMapNames(self):
        """
        Returns a list of all loaded map names.

        Returns:
            list: List of map name strings.
        """
        return list(self.maps.keys())


if __name__ == '__main__':
    tracker = LocationTracker()

    result = tracker.locatePlayer('../screenshot.png')

    if result:
        print(f"\nMap:        {result['mapName']}")
        print(f"Position:   {result['position']}")
        print(f"Confidence: {result['confidence']:.4f}")
    else:
        print('\nCould not determine player location.')
