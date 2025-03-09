import json
from collections.abc import Callable
from typing import cast, Callable
import uuid
from PIL import ImageDraw
import base64
from io import BytesIO

from anthropic import APIResponse
from anthropic.types.beta import BetaToolUseBlock, BetaMessageParam

from agent.llm_utils.oaiclient import OpenAIClient, QwenClient
from agent.llm_utils.groqclient import GroqClient
from agent.llm_utils.utils import UserMessage, BotMessage, ToolResultMessage, ParsedScreenshot
from agent.vlm_instructions import get_vlm_instructions
import time
import re

OUTPUT_DIR = "./tmp/outputs"

def extract_data(input_string, data_type):
    # Regular expression to extract content starting from '```python' until the end if there are no closing backticks
    pattern = f"```{data_type}" + r"(.*?)(```|$)"
    # Extract content
    # re.DOTALL allows '.' to match newlines as well
    matches = re.findall(pattern, input_string, re.DOTALL)
    # Return the first match if exists, trimming whitespace and ignoring potential closing backticks
    return matches[0][0].strip() if matches else input_string

def format_messages_for_llm(messages, screenshots_to_keep: int):
    messages_for_llm = []
    screenshot_count = 0
    for message in reversed(messages):
        if type(message) is UserMessage:
            messages_for_llm.append({
                "role": 'user',
                "content": [{"text": message.content, "type": "text"}]
            })
        elif type(message) is BotMessage:
            content = [{"text": message.content, "type": "text"}]
            for requested_action in message.requested_actions:
                content.append(requested_action.to_dict())
            messages_for_llm.append({
                "role": 'assistant',
                "content": content
            })
        elif type(message) is ToolResultMessage:
            None
            # messages_for_llm.append({
            #     "role": 'tool',
            #     "content": message.content
            # })
        elif type(message) is ParsedScreenshot:
            if screenshot_count >= screenshots_to_keep:
                continue
            buffer = BytesIO()
            message.annotated_image.save(buffer, format="PNG")
            base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
            # messages_for_llm.append({
            #     "role": 'tool',
            #     "content": {
            #         "type": "image_url",
            #         "image_url": {
            #             "url": f"data:image/png;base64,{base64_image}"
            #         },
            #     },
            # })
            screenshot_count += 1
    messages_for_llm.reverse()
    return messages_for_llm


class VLMAgent:
    def __init__(
        self,
        model: str,
        provider: str,
        api_key: str,
        output_callback: Callable,
        max_tokens: int = 4096,
        only_n_most_recent_images: int | None = None,
    ):
        if model == "omniparser + gpt-4o":
            self.model = "gpt-4o-2024-11-20"
            self.llm_client = OpenAIClient(self.model, api_key, max_tokens, 0)
        elif model == "omniparser + R1":
            self.model = "deepseek-r1-distill-llama-70b"
            self.llm_client = GroqClient(self.model, api_key, max_tokens)
        elif model == "omniparser + qwen2.5vl":
            self.model = "qwen2.5-vl-72b-instruct"
            self.llm_client = QwenClient(self.model, api_key, max_tokens, 0)
        elif model == "omniparser + o1":
            self.model = "o1"
            self.llm_client = OpenAIClient(self.model, api_key, max_tokens, 0)
        elif model == "omniparser + o3-mini":
            self.model = "o3-mini"
            self.llm_client = OpenAIClient(self.model, api_key, max_tokens, 0)
        else:
            raise ValueError(f"Model {model} not supported")

        self.provider = provider
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.only_n_most_recent_images = only_n_most_recent_images
        self.output_callback = output_callback

        self.step_count = 0

    def __call__(self, messages: list):
        parsed_screenshot = None
        for message in reversed(messages):
            if type(message) is ParsedScreenshot:
                parsed_screenshot = message
                break

        if parsed_screenshot is None:
            raise 'Request to the LLM must include a parsed screenshot.'

        self.step_count += 1
        self.output_callback(f'-- Step {self.step_count}: --', sender="bot")
        boxids_and_labels = parsed_screenshot.parsed_elements_formatted()
        screen_width, screen_height = parsed_screenshot.width(), parsed_screenshot.height()

        is_thinking_model = ("r1" in self.model)
        system = get_vlm_instructions(boxids_and_labels, is_thinking_model)
        messages_for_llm = format_messages_for_llm(messages, self.only_n_most_recent_images)

        start = time.time()
        vlm_response = self.llm_client.request(system, messages_for_llm)
        latency_vlm = time.time() - start
        self.output_callback(f"LLM: {latency_vlm:.2f}s, OmniParser: {parsed_screenshot.parse_time_sec:.2f}s", sender="bot")

        print(f"{vlm_response}")
        print(f"Total token so far: {self.llm_client.total_token_usage}. Total cost so far: $USD{self.llm_client.total_cost():.5f}")

        vlm_response_json = extract_data(vlm_response, "json")
        vlm_response_json = json.loads(vlm_response_json)

        if "Box ID" in vlm_response_json:
            try:
                box_id = int(vlm_response_json["Box ID"])
                bbox = parsed_screenshot.parsed_elements[box_id]["bbox"]
                vlm_response_json["box_centroid_coordinate"] = [int((bbox[0] + bbox[2]) / 2 * screen_width), int((bbox[1] + bbox[3]) / 2 * screen_height)]
                img_to_show = parsed_screenshot.annotated_image.copy()

                draw = ImageDraw.Draw(img_to_show)
                x, y = vlm_response_json["box_centroid_coordinate"]
                radius = 10
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='red')
                draw.ellipse((x - radius*3, y - radius*3, x + radius*3, y + radius*3), fill=None, outline='red', width=2)

                buffered = BytesIO()
                img_to_show.save(buffered, format="PNG")
                img_to_show_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                self.output_callback(f'<img src="data:image/png;base64,{img_to_show_base64}">', sender="bot")
            except:
                print(f"Error parsing: {vlm_response_json}")
                pass

        vlm_plan_str = ""
        for key, value in vlm_response_json.items():
            if key == "Reasoning":
                vlm_plan_str += f'{value}'
            else:
                vlm_plan_str += f'\n{key}: {value}'

        # construct the response so that anthropicExcutor can execute the tool
        requested_actions: list[BetaToolUseBlock] = []
        if 'box_centroid_coordinate' in vlm_response_json:
            move_cursor_block = BetaToolUseBlock(id=f'toolu_{uuid.uuid4()}',
                                            input={'action': 'mouse_move', 'coordinate': vlm_response_json["box_centroid_coordinate"]},
                                            name='computer', type='tool_use')
            requested_actions.append(move_cursor_block)

        if vlm_response_json["Next Action"] == "None":
            print("Task paused/completed.")
        elif vlm_response_json["Next Action"] == "type":
            sim_content_block = BetaToolUseBlock(id=f'toolu_{uuid.uuid4()}',
                                        input={'action': vlm_response_json["Next Action"], 'text': vlm_response_json["value"]},
                                        name='computer', type='tool_use')
            requested_actions.append(sim_content_block)
        else:
            sim_content_block = BetaToolUseBlock(id=f'toolu_{uuid.uuid4()}',
                                            input={'action': vlm_response_json["Next Action"]},
                                            name='computer', type='tool_use')
            requested_actions.append(sim_content_block)

        return BotMessage(
            vlm_plan_str,
            {'system_prompt': system, 'messages': messages_for_llm},
            requested_actions
        )

