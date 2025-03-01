import base64
import time
from enum import StrEnum
from typing import Literal, TypedDict

from anthropic.types.beta import BetaToolComputerUse20241022Param

from .omnibox_client import OmniboxClient

from .base import BaseAnthropicTool, ToolError, ToolResult

OUTPUT_DIR = "./tmp/outputs"

Action = Literal[
    "key",
    "type",
    "mouse_move",
    "left_click",
    "left_click_drag",
    "right_click",
    "middle_click",
    "double_click",
    "screenshot",
    "cursor_position",
    "hover",
    "wait"
]


class Resolution(TypedDict):
    width: int
    height: int


MAX_SCALING_TARGETS: dict[str, Resolution] = {
    "XGA": Resolution(width=1024, height=768),  # 4:3
    "WXGA": Resolution(width=1280, height=800),  # 16:10
    "FWXGA": Resolution(width=1366, height=768),  # ~16:9
}


class ScalingSource(StrEnum):
    COMPUTER = "computer"
    API = "api"


class ComputerToolOptions(TypedDict):
    display_height_px: int
    display_width_px: int
    display_number: int | None

class ComputerTool(BaseAnthropicTool):
    """
    A tool that allows the agent to interact with the screen, keyboard, and mouse of the current computer.
    Adapted for Windows using 'pyautogui'.
    """

    name: Literal["computer"] = "computer"
    api_type: Literal["computer_20241022"] = "computer_20241022"
    width: int
    height: int
    display_num: int | None
    computer_client: OmniboxClient

    @property
    def options(self) -> ComputerToolOptions:
        return {
            "display_width_px": self.width,
            "display_height_px": self.height,
            "display_number": self.display_num,
        }

    def to_params(self) -> BetaToolComputerUse20241022Param:
        return {"name": self.name, "type": self.api_type, **self.options}

    def __init__(self, computer_client: OmniboxClient):
        super().__init__()
        self.computer_client = computer_client

        # Get screen width and height using Windows command
        self.display_num = None
        self.width, self.height = self.computer_client.current_screen_size()
        print(f"screen size: {self.width}, {self.height}")

        self.key_conversion = {"Page_Down": "pagedown",
                               "Page_Up": "pageup",
                               "Super_L": "win",
                               "Escape": "esc"}


    async def __call__(
        self,
        *,
        action: Action,
        text: str | None = None,
        coordinate: tuple[int, int] | None = None,
        **kwargs,
    ):
        print(f"action: {action}, text: {text}, coordinate: {coordinate}")
        if action in ("mouse_move", "left_click_drag"):
            if coordinate is None:
                raise ToolError(f"coordinate is required for {action}")
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 2:
                raise ToolError(f"{coordinate} must be a tuple of length 2")
            # if not all(isinstance(i, int) and i >= 0 for i in coordinate):
            if not all(isinstance(i, int) for i in coordinate):
                raise ToolError(f"{coordinate} must be a tuple of non-negative ints")

            x, y = coordinate
            print(f"mouse move to {x}, {y}")

            if action == "mouse_move":
                self.computer_client.mouse_move(x, y)
                return ToolResult(output=f"Moved mouse to ({x}, {y})")
            elif action == "left_click_drag":
                current_x, current_y = self.computer_client.current_mouse_coordinates()
                self.computer_client.mouse_drag(x, y)
                return ToolResult(output=f"Dragged mouse from ({current_x}, {current_y}) to ({x}, {y})")

        if action in ("key", "type"):
            if text is None:
                raise ToolError(f"text is required for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")
            if not isinstance(text, str):
                raise ToolError(output=f"{text} must be a string")

            if action == "key":
                # Handle key combinations
                keys = text.split('+')
                for key in keys:
                    key = self.key_conversion.get(key.strip(), key.strip())
                    key = key.lower()
                    self.computer_client.key_down(key)  # Press down each key
                for key in reversed(keys):
                    key = self.key_conversion.get(key.strip(), key.strip())
                    key = key.lower()
                    self.computer_client.key_up(key)    # Release each key in reverse order
                return ToolResult(output=f"Pressed keys: {text}")

            elif action == "type":
                # default click before type TODO: check if this is needed
                self.computer_client.mouse_left_click()
                self.computer_client.type(text)
                self.computer_client.key_press('enter')
                _screenshot, path = self.computer_client.screenshot()
                return ToolResult(output=text, base64_image=base64.b64encode(path.read_bytes()).decode())

        if action in (
            "left_click",
            "right_click",
            "double_click",
            "middle_click",
            "screenshot",
            "cursor_position",
            "left_press",
        ):
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")

            if action == "screenshot":
                _screenshot, path = self.computer_client.screenshot()
                return ToolResult(base64_image=base64.b64encode(path.read_bytes()).decode())
            elif action == "cursor_position":
                x, y = self.computer_client.current_mouse_coordinates()
                return ToolResult(output=f"X={x},Y={y}")
            else:
                if action == "left_click":
                    self.computer_client.mouse_left_click()
                elif action == "right_click":
                    self.computer_client.mouse_right_click()
                elif action == "middle_click":
                    self.computer_client.mouse_middle_click()
                elif action == "double_click":
                    self.computer_client.mouse_double_click()
                elif action == "left_press":
                    self.computer_client.mouse_left_press()
                return ToolResult(output=f"Performed {action}")
        if action in ("scroll_up", "scroll_down"):
            if action == "scroll_up":
                self.computer_client.scroll_up()
            elif action == "scroll_down":
                self.computer_client.scroll_down()
            return ToolResult(output=f"Performed {action}")
        if action == "hover":
            return ToolResult(output=f"Performed {action}")
        if action == "wait":
            time.sleep(1)
            return ToolResult(output=f"Performed {action}")
        raise ToolError(f"Invalid action: {action}")

