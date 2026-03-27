"""
mgba_client.py — Example Python client for mgba_server.lua

Usage:
    python mgba_client.py              # interactive mode
    python mgba_client.py tap A        # tap A button (default 8 frames)
    python mgba_client.py tap A 16     # tap A button for 16 frames
    python mgba_client.py screenshot   # save screenshot to screenshot.png
    python mgba_client.py ping         # health check

Multiple clients can connect simultaneously.
"""

import socket
import sys
import struct

HOST = "127.0.0.1"
PORT = 54321


def send_command(sock: socket.socket, command: str) -> str:
    """Send a newline-terminated command string and return the response line."""
    sock.sendall((command + "\n").encode("utf-8"))
    # Read until we get a newline (the response header)
    buf = b""
    while b"\n" not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("Server closed connection")
        buf += chunk
    nl = buf.index(b"\n")
    header = buf[:nl].decode("utf-8")
    remainder = buf[nl + 1:]
    return header, remainder


def tap(sock: socket.socket, button: str, frames: int = None):
    """Send a button tap command."""
    cmd = f"TAP|{button}"
    if frames is not None:
        cmd += f"|{frames}"
    header, _ = send_command(sock, cmd)
    print(f"TAP {button}: {header}")


def screenshot(sock: socket.socket, output_path: str = "screenshot.png"):
    """Request a screenshot and save it to a file."""
    sock.sendall(b"SCREENSHOT\n")

    # Read response header: OK|<byte_length>\n
    buf = b""
    while b"\n" not in buf:
        chunk = sock.recv(4096)
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
        chunk = sock.recv(min(65536, byte_length - len(png_data)))
        if not chunk:
            raise ConnectionError("Server closed connection during transfer")
        png_data += chunk

    with open(output_path, "wb") as f:
        f.write(png_data[:byte_length])
    print(f"Screenshot saved to {output_path} ({byte_length} bytes)")


def ping(sock: socket.socket):
    """Send a ping command."""
    header, _ = send_command(sock, "PING")
    print(f"PING: {header}")


def interactive(sock: socket.socket):
    """Simple interactive REPL."""
    print("Connected to mGBA server. Commands:")
    print("  tap <button> [frames]   - e.g. tap A, tap START 16")
    print("  screenshot [filename]   - save screenshot")
    print("  ping                    - health check")
    print("  quit                    - exit")
    print()

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue

        parts = line.split()
        cmd = parts[0].lower()

        try:
            if cmd == "tap":
                if len(parts) < 2:
                    print("Usage: tap <button> [frames]")
                    continue
                frames = int(parts[2]) if len(parts) > 2 else None
                tap(sock, parts[1], frames)
            elif cmd == "screenshot":
                path = parts[1] if len(parts) > 1 else "screenshot.png"
                screenshot(sock, path)
            elif cmd == "ping":
                ping(sock)
            elif cmd in ("quit", "exit", "q"):
                break
            else:
                print(f"Unknown command: {cmd}")
        except Exception as e:
            print(f"Error: {e}")
            break

    print("Disconnected.")


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))

    try:
        if len(sys.argv) > 1:
            cmd = sys.argv[1].lower()
            if cmd == "tap":
                button = sys.argv[2] if len(sys.argv) > 2 else "A"
                frames = int(sys.argv[3]) if len(sys.argv) > 3 else None
                tap(sock, button, frames)
            elif cmd == "screenshot":
                path = sys.argv[2] if len(sys.argv) > 2 else "screenshot.png"
                screenshot(sock, path)
            elif cmd == "ping":
                ping(sock)
            else:
                print(f"Unknown command: {cmd}")
        else:
            interactive(sock)
    finally:
        sock.close()


if __name__ == "__main__":
    main()
