import base64
from PIL import Image
from anthropic.types.beta import BetaToolUseBlock

def is_image_path(text):
    image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif")
    if text.endswith(image_extensions):
        return True
    else:
        return False

def encode_image(image_path):
    """Encode image file to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class UserMessage:
    def __init__(self, content):
        self.content: str = content

class ParsedScreenshot:
    def __init__(self, parsed_elements, original_image, annotated_image, parse_time_sec):
        self.parsed_elements = parsed_elements
        self.original_image: Image.Image = original_image
        self.annotated_image: Image.Image = annotated_image
        self.parse_time_sec: float = parse_time_sec

    def width(self) -> int:
        return self.original_image.width

    def height(self) -> int:
        return self.original_image.height

    def parsed_elements_formatted(self) -> str:
        screen_info = ""
        for idx, element in enumerate(self.parsed_elements):
            element['idx'] = idx
            if element['type'] == 'text':
                screen_info += f'ID: {idx}, Text: {element["content"]}\n'
            elif element['type'] == 'icon':
                screen_info += f'ID: {idx}, Icon: {element["content"]}\n'
        return screen_info

class BotMessage:
    def __init__(self, content, request_content, requested_actions):
        self.content: str = content
        self.request_content = request_content
        self.requested_actions: list[BetaToolUseBlock] = requested_actions

class ToolResultMessage:
    def __init__(self, content):
        self.content: str = content
# LLM Message
#   - content
#   - tool_calls
#   - formatted_request