# Location Tracking in Pokemon

To figure out where the player character is in the game, we're using template matching to take advantage of how each map is is just a collection of tile objects.  This approch works better for 2D maps instead of 3D ones, since in the early games we never have to worry about the map being rotated.  

# Resources

The maps for every location in the game are found in the [maps folder](./maps/).  These were taken from [vgmaps.com](https://www.vgmaps.com/atlas/GBA/index.htm), which was a huge help in making this possible.  They have maps for tons of different games there, so if you want to do something similar for a different franchise, check them out.  