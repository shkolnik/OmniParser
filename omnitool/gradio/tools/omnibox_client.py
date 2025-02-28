import base64
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from PIL import Image

import requests
import time
import re
from typing import NoReturn, Tuple

from .base import ToolError

TYPING_DELAY_MS = 12

def send_to_vm(base_url: str, action: str):
        """
        Executes a python command on the server. Only return tuple of x,y when action is "pyautogui.position()"
        """
        prefix = "import pyautogui; pyautogui.FAILSAFE = False;"
        command_list = ["python", "-c", f"{prefix} {action}"]
        parse = action == "pyautogui.position()"
        if parse:
            command_list[-1] = f"{prefix} print({action})"

        try:
            print(f"sending to vm: {command_list}")
            response = requests.post(
                base_url + '/execute',
                headers={'Content-Type': 'application/json'},
                json={"command": command_list},
                timeout=90
            )
            time.sleep(0.7) # avoid async error as actions take time to complete
            print(f"action executed")
            if response.status_code != 200:
                raise ToolError(f"Failed to execute command. Status code: {response.status_code}")
            if parse:
                output = response.json()['output'].strip()
                match = re.search(r'Point\(x=(\d+),\s*y=(\d+)\)', output)
                if not match:
                    raise ToolError(f"Could not parse coordinates from output: {output}")
                x, y = map(int, match.groups())
                return x, y
        except requests.exceptions.RequestException as e:
            raise ToolError(f"An error occurred while trying to execute the command: {str(e)}")

OUTPUT_DIR = "./tmp/outputs"

def get_screenshot(base_url: str, resize: bool = False, target_width: int = 1920, target_height: int = 1080):
    """Capture screenshot by requesting from HTTP endpoint - returns native resolution unless resized"""
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"screenshot_{uuid4().hex}.png"

    try:
        response = requests.get(base_url + '/screenshot')
        if response.status_code != 200:
            raise ToolError(f"Failed to capture screenshot: HTTP {response.status_code}")

        # (1280, 800)
        screenshot = Image.open(BytesIO(response.content))

        if resize and screenshot.size != (target_width, target_height):
            screenshot = screenshot.resize((target_width, target_height))
        screenshot.save(path)
        return screenshot, path
    except Exception as e:
        raise ToolError(f"Failed to capture screenshot: {str(e)}")

def get_screen_size(base_url: str):
    """Return width and height of the screen"""
    try:
        response = requests.post(
            base_url + '/execute',
            headers={'Content-Type': 'application/json'},
            json={"command": ["python", "-c", "import pyautogui; print(pyautogui.size())"]},
            timeout=90
        )
        if response.status_code != 200:
            raise ToolError(f"Failed to get screen size. Status code: {response.status_code}")

        output = response.json()['output'].strip()
        match = re.search(r'Size\(width=(\d+),\s*height=(\d+)\)', output)
        if not match:
            raise ToolError(f"Could not parse screen size from output: {output}")
        width, height = map(int, match.groups())
        return width, height
    except requests.exceptions.RequestException as e:
        raise ToolError(f"An error occurred while trying to get screen size: {str(e)}")


class OmniboxClient:
    base_url: str

    def __init__(self, base_url: str):
        self.base_url = base_url

    def screenshot(self):
        return get_screenshot(self.base_url)

    async def scaled_screenshot(self, target_width, target_height):
        return get_screenshot(self.base_url, resize=True, target_width=target_width, target_height=target_height)

    def current_screen_size(self) -> Tuple[int, int]:
        return get_screen_size(self.base_url)

    def current_mouse_coordinates(self) -> Tuple[int, int]:
        return send_to_vm(self.base_url, "pyautogui.position()")

    def mouse_move(self, x: int, y: int) -> NoReturn:
        send_to_vm(self.base_url, f"pyautogui.moveTo({x}, {y})")

    def mouse_drag(self, x: int, y: int) -> NoReturn:
        send_to_vm(self.base_url, f"pyautogui.dragTo({x}, {y}, duration=0.5)")

    def mouse_left_click(self, ) -> NoReturn:
        send_to_vm(self.base_url, "pyautogui.click()")

    def mouse_double_click(self, ) -> NoReturn:
        send_to_vm(self.base_url, "pyautogui.doubleClick()")

    def mouse_right_click(self, ) -> NoReturn:
        send_to_vm(self.base_url, "pyautogui.rightClick()")

    def mouse_middle_click(self, ) -> NoReturn:
        send_to_vm(self.base_url, "pyautogui.middleClick()")

    def mouse_left_press(self, ) -> NoReturn:
        send_to_vm(self.base_url, "pyautogui.mouseDown()")
        time.sleep(1)
        send_to_vm(self.base_url, "pyautogui.mouseUp()")

    def scroll_up(self, ) -> NoReturn:
        send_to_vm(self.base_url, "pyautogui.scroll(100)")

    def scroll_down(self, ) -> NoReturn:
        send_to_vm(self.base_url, "pyautogui.scroll(-100)")

    def key_down(self, key: str) -> NoReturn:
        send_to_vm(self.base_url, f"pyautogui.keyDown('{key}')")

    def key_up(self, key: str) -> NoReturn:
        send_to_vm(self.base_url, f"pyautogui.keyUp('{key}')")

    def key_press(self, key: str) -> NoReturn:
        self.key_down(key)
        self.key_up(key)

    def type(self, text: str) -> NoReturn:
        send_to_vm(self.base_url, f"pyautogui.typewrite('{text}', interval={TYPING_DELAY_MS / 1000})")
