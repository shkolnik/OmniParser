import base64
import requests
import time
import re
from typing import NoReturn, Tuple

from .screen_capture import get_screenshot

from .base import ToolError, ToolResult

TYPING_DELAY_MS = 12

def send_to_vm(url: str, action: str):
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
                url,
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


def get_screen_size(url: str):
    """Return width and height of the screen"""
    try:
        response = requests.post(
            url,
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


class VMClient:
    vm_url: str = f"http://192.168.64.5:5000/execute"

    async def screenshot(self, target_width, target_height):
        return get_screenshot(resize=True, target_width=target_width, target_height=target_height)

    def current_screen_size(self) -> Tuple[int, int]:
        return get_screen_size(self.vm_url);

    def current_mouse_coordinates(self) -> Tuple[int, int]:
        return send_to_vm(self.vm_url, "pyautogui.position()")

    def mouse_move(self, x: int, y: int) -> NoReturn:
        send_to_vm(self.vm_url, f"pyautogui.moveTo({x}, {y})")

    def mouse_drag(self, x: int, y: int) -> NoReturn:
        send_to_vm(self.vm_url, f"pyautogui.dragTo({x}, {y}, duration=0.5)")

    def mouse_left_click(self, ) -> NoReturn:
        send_to_vm(self.vm_url, "pyautogui.click()")

    def mouse_double_click(self, ) -> NoReturn:
        send_to_vm(self.vm_url, "pyautogui.doubleClick()")

    def mouse_right_click(self, ) -> NoReturn:
        send_to_vm(self.vm_url, "pyautogui.rightClick()")

    def mouse_middle_click(self, ) -> NoReturn:
        send_to_vm(self.vm_url, "pyautogui.middleClick()")

    def mouse_left_press(self, ) -> NoReturn:
        send_to_vm(self.vm_url, "pyautogui.mouseDown()")
        time.sleep(1)
        send_to_vm(self.vm_url, "pyautogui.mouseUp()")

    def scroll_up(self, ) -> NoReturn:
        send_to_vm(self.vm_url, "pyautogui.scroll(100)")

    def scroll_down(self, ) -> NoReturn:
        send_to_vm(self.vm_url, "pyautogui.scroll(-100)")

    def key_down(self, key: str) -> NoReturn:
        send_to_vm(self.vm_url, f"pyautogui.keyDown('{key}')")

    def key_up(self, key: str) -> NoReturn:
        send_to_vm(self.vm_url, f"pyautogui.keyUp('{key}')")

    def key_press(self, key: str) -> NoReturn:
        self.key_down(key)
        self.key_up(key)

    def type(self, text: str) -> NoReturn:
        send_to_vm(self.vm_url, f"pyautogui.typewrite('{text}', interval={TYPING_DELAY_MS / 1000})")
