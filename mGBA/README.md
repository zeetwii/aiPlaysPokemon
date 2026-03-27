# mGBA

This folder contains the helper scripts for mGBA, which is the emulator we use to run the Pokémon games. The main script is ['mgba_server.lua'](./mgba_server.lua), which is a Lua script that runs inside mGBA and listens for commands from the Python code. It can read and write memory, and it can also send notifications when certain events happen in the game.

To test that everything is working, you can run the pythong script ['mgba_client.py'](./mgba_client.py), which will connect to the TCP server and let you send and test commands.  