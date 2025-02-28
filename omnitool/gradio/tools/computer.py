import base64
import time
from enum import StrEnum
from typing import Literal, TypedDict

from anthropic.types.beta import BetaToolComputerUse20241022Param

from .vm_client import VMClient

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


# def chunks(s: str, chunk_size: int) -> list[str]:
#     return [s[i : i + chunk_size] for i in range(0, len(s), chunk_size)]

# def padding_image(screenshot):
#     """Pad the screenshot to 16:10 aspect ratio, when the aspect ratio is not 16:10."""
#     _, height = screenshot.size
#     new_width = height * 16 // 10

#     padding_image = Image.new("RGB", (new_width, height), (255, 255, 255))
#     # padding to top left
#     padding_image.paste(screenshot, (0, 0))
#     return padding_image

client = VMClient(f"http://192.168.64.5:5000")

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

    _screenshot_delay = 2.0
    _scaling_enabled = True

    @property
    def options(self) -> ComputerToolOptions:
        width, height = self.scale_coordinates(
            ScalingSource.COMPUTER, self.width, self.height
        )
        return {
            "display_width_px": width,
            "display_height_px": height,
            "display_number": self.display_num,
        }

    def to_params(self) -> BetaToolComputerUse20241022Param:
        return {"name": self.name, "type": self.api_type, **self.options}

    def __init__(self, is_scaling: bool = False):
        super().__init__()

        # Get screen width and height using Windows command
        self.display_num = None
        self.offset_x = 0
        self.offset_y = 0
        self.is_scaling = is_scaling
        self.width, self.height = client.current_screen_size()
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
        print(f"action: {action}, text: {text}, coordinate: {coordinate}, is_scaling: {self.is_scaling}")
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

            if self.is_scaling:
                x, y = self.scale_coordinates(
                    ScalingSource.API, coordinate[0], coordinate[1]
                )
            else:
                x, y = coordinate

            # print(f"scaled_coordinates: {x}, {y}")
            # print(f"offset: {self.offset_x}, {self.offset_y}")

            # x += self.offset_x # TODO - check if this is needed
            # y += self.offset_y

            print(f"mouse move to {x}, {y}")

            if action == "mouse_move":
                client.mouse_move(x, y)
                return ToolResult(output=f"Moved mouse to ({x}, {y})")
            elif action == "left_click_drag":
                current_x, current_y = client.current_mouse_coordinates()
                client.mouse_drag(x, y)
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
                    client.key_down(key)  # Press down each key
                for key in reversed(keys):
                    key = self.key_conversion.get(key.strip(), key.strip())
                    key = key.lower()
                    client.key_up(key)    # Release each key in reverse order
                return ToolResult(output=f"Pressed keys: {text}")

            elif action == "type":
                # default click before type TODO: check if this is needed
                client.mouse_left_click()
                client.type(text)
                client.key_press('enter')
                screenshot_base64 = (await self.screenshot()).base64_image
                return ToolResult(output=text, base64_image=screenshot_base64)

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
                return await self.screenshot()
            elif action == "cursor_position":
                x, y = client.current_mouse_coordinates()
                x, y = self.scale_coordinates(ScalingSource.COMPUTER, x, y)
                return ToolResult(output=f"X={x},Y={y}")
            else:
                if action == "left_click":
                    client.mouse_left_click()
                elif action == "right_click":
                    client.mouse_right_click()
                elif action == "middle_click":
                    client.mouse_middle_click()
                elif action == "double_click":
                    client.mouse_double_click()
                elif action == "left_press":
                    client.mouse_left_press()
                return ToolResult(output=f"Performed {action}")
        if action in ("scroll_up", "scroll_down"):
            if action == "scroll_up":
                client.scroll_up()
            elif action == "scroll_down":
                client.scroll_down()
            return ToolResult(output=f"Performed {action}")
        if action == "hover":
            return ToolResult(output=f"Performed {action}")
        if action == "wait":
            time.sleep(1)
            return ToolResult(output=f"Performed {action}")
        raise ToolError(f"Invalid action: {action}")


    async def screenshot(self):
        if not hasattr(self, 'target_dimension'):
            raise 'Expected target_dimensions to be set'
        width, height = self.target_dimension["width"], self.target_dimension["height"]
        _screenshot, path = await client.scaled_screenshot(width, height)
        return ToolResult(base64_image=base64.b64encode(path.read_bytes()).decode())

    def set_target_dimensions(self):
        ratio = self.width / self.height
        target_dimension = None

        for target_name, dimension in MAX_SCALING_TARGETS.items():
            # allow some error in the aspect ratio - not ratios are exactly 16:9
            if abs(dimension["width"] / dimension["height"] - ratio) < 0.02:
                if dimension["width"] < self.width:
                    target_dimension = dimension
                    self.target_dimension = target_dimension
                    # print(f"target_dimension: {target_dimension}")
                break

        if target_dimension is None:
            # TODO: currently we force the target to be WXGA (16:10), when it cannot find a match
            target_dimension = MAX_SCALING_TARGETS["WXGA"]
            self.target_dimension = MAX_SCALING_TARGETS["WXGA"]


    def scale_coordinates(self, source: ScalingSource, x: int, y: int):
        """Scale coordinates to a target maximum resolution."""
        if not self._scaling_enabled:
            return x, y

        self.set_target_dimensions()

        # should be less than 1
        x_scaling_factor = self.target_dimension["width"] / self.width
        y_scaling_factor = self.target_dimension["height"] / self.height
        if source == ScalingSource.API:
            if x > self.width or y > self.height:
                raise ToolError(f"Coordinates {x}, {y} are out of bounds")
            # scale up
            return round(x / x_scaling_factor), round(y / y_scaling_factor)
        # scale down
        return round(x * x_scaling_factor), round(y * y_scaling_factor)

