from pathlib import Path
from uuid import uuid4

from PIL import Image

import time
from typing import NoReturn, Tuple
from urllib.parse import urlparse
# from .base import ToolError

from vncdotool import api

TYPING_DELAY_MS = 20

OUTPUT_DIR = "./tmp/outputs"

class VNCClient:
    vnc_connection_string: str
    vnc_password: str
    vnc_api: api.ThreadedVNCClientProxy

    def __init__(self, url: str):
        parsed_url = urlparse(url)
        self.vnc_connection_string = f"{parsed_url.hostname}::{parsed_url.port or '5900'}"
        self.vnc_password = parsed_url.password
        self.vnc_api = None
        self.reconnect()

    def reconnect(self) -> NoReturn:
        if self.vnc_api is not None:
            self.vnc_api.disconnect()
        self.vnc_api = api.connect(self.vnc_connection_string, password=self.vnc_password)

    def shutdown(self) -> NoReturn:
        self.vnc_api.shutdown()

    def screenshot(self) -> Tuple[Image.Image, str]:
        self.reconnect() # TODO Find a better way to ensure we are getting an updated screenshot without reconnecting

        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"screenshot_{uuid4().hex}.png"

        # self.vnc_api.refreshScreen(False)
        self.vnc_api.captureScreen(path)
        screenshot = Image.open(path)
        print(f"Captured screenshot to {path}")

        return screenshot, path

    def current_screen_size(self) -> Tuple[int, int]:
        """Return width and height of the screen"""
        screenshot, _path = self.screenshot()
        return screenshot.size

    def mouse_move(self, x: int, y: int) -> NoReturn:
        self.vnc_api.mouseMove(x, y)

    def mouse_drag(self, x: int, y: int) -> NoReturn:
        self.vnc_api.mouseDrag(x, y)

    def mouse_left_click(self) -> NoReturn:
        self.vnc_api.mousePress(1)

    def mouse_double_click(self) -> NoReturn:
        self.mouse_left_click()
        self.mouse_left_click()

    def mouse_right_click(self) -> NoReturn:
        self.vnc_api.mousePress(2)

    def mouse_middle_click(self) -> NoReturn:
        self.vnc_api.mousePress(3)

    def mouse_left_press(self) -> NoReturn:
        self.vnc_api.mouseDown(1)
        time.sleep(1)
        self.vnc_api.mouseUp(1)

    def scroll_up(self) -> NoReturn:
        self.vnc_api.mousePress(4) # Button #4 is SCROLL_WHEEL_UP

    def scroll_down(self) -> NoReturn:
        self.vnc_api.mousePress(5) # Button #5 is SCROLL_WHEEL_DOWN

    def key_down(self, key: str) -> NoReturn:
        self.vnc_api.keyDown(key)

    def key_up(self, key: str) -> NoReturn:
        self.vnc_api.keyUp(key)

    def key_press(self, key: str) -> NoReturn:
        self.key_down(key)
        self.key_up(key)

    def type(self, text: str) -> NoReturn:
        for char in text:
            self.key_press(char)
            self.vnc_api.pause(TYPING_DELAY_MS / 1000.0)
