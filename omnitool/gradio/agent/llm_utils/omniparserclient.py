from io import BytesIO
import requests
import base64
from agent.llm_utils.utils import encode_image, ParsedScreenshot
from PIL import Image

OUTPUT_DIR = "./tmp/outputs"

class OmniParserClient:
    def __init__(self,
                 url: str,
                 computer_client) -> None:
        self.url = url
        self.computer_client = computer_client

    def __call__(self,):
        screenshot, screenshot_path = self.computer_client.screenshot()
        screenshot_path = str(screenshot_path)
        image_base64 = encode_image(screenshot_path)
        response = requests.post(self.url, json={"base64_image": image_base64})
        response_json = response.json()
        print('omniparser latency:', response_json['latency'])

        som_image_data = base64.b64decode(response_json['som_image_base64'])
        annotated_image = Image.open(BytesIO(som_image_data))

        response_json = self.reformat_messages(response_json)

        return ParsedScreenshot(
            response_json['parsed_content_list'],
            screenshot,
            annotated_image,
            response_json['latency'],
        )

    def reformat_messages(self, response_json: dict):
        screen_info = ""
        for idx, element in enumerate(response_json["parsed_content_list"]):
            element['idx'] = idx
            if element['type'] == 'text':
                screen_info += f'ID: {idx}, Text: {element["content"]}\n'
            elif element['type'] == 'icon':
                screen_info += f'ID: {idx}, Icon: {element["content"]}\n'
        response_json['screen_info'] = screen_info
        return response_json