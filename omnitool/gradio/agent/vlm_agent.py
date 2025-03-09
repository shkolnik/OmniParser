import json
from collections.abc import Callable
from typing import cast, Callable
import uuid
from PIL import ImageDraw
import base64
from io import BytesIO

from anthropic import APIResponse
from anthropic.types.beta import BetaToolUseBlock, BetaMessageParam

from agent.llm_utils.oaiclient import run_oai_interleaved
from agent.llm_utils.groqclient import run_groq_interleaved
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

class VLMAgent:
    def __init__(
        self,
        model: str,
        provider: str,
        api_key: str,
        output_callback: Callable,
        max_tokens: int = 4096,
        only_n_most_recent_images: int | None = None,
        print_usage: bool = True,
    ):
        if model == "omniparser + gpt-4o":
            self.model = "gpt-4o-2024-11-20"
        elif model == "omniparser + R1":
            self.model = "deepseek-r1-distill-llama-70b"
        elif model == "omniparser + qwen2.5vl":
            self.model = "qwen2.5-vl-72b-instruct"
        elif model == "omniparser + o1":
            self.model = "o1"
        elif model == "omniparser + o3-mini":
            self.model = "o3-mini"
        else:
            raise ValueError(f"Model {model} not supported")


        self.provider = provider
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.only_n_most_recent_images = only_n_most_recent_images
        self.output_callback = output_callback

        self.print_usage = print_usage
        self.total_token_usage = 0
        self.total_cost = 0
        self.step_count = 0

        self.system = ''

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

        thinking_model = "r1" in self.model
        system = get_vlm_instructions(boxids_and_labels, thinking_model)

        messages_for_llm = []
        screenshot_count = 0
        for message in reversed(messages):
            if type(message) is UserMessage:
                messages_for_llm.append({
                    "role": 'user',
                    "content": message.content
                })
            elif type(message) is BotMessage:
                messages_for_llm.append({
                    "role": 'assistant',
                    "content": message.content
                })
            elif type(message) is ToolResultMessage:
                messages_for_llm.append({
                    "role": 'tool',
                    "content": message.content
                })
            elif type(message) is ParsedScreenshot:
                if screenshot_count >= self.only_n_most_recent_images:
                    continue
                buffer = BytesIO()
                message.annotated_image.save(buffer, format="PNG")
                base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
                messages_for_llm.append({
                    "role": 'tool',
                    "content": {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        },
                    },
                })
                screenshot_count += 1
        messages_for_llm.reverse()

        start = time.time()
        if "gpt" in self.model or "o1" in self.model or "o3-mini" in self.model:
            vlm_response, token_usage = run_oai_interleaved(
                messages=messages_for_llm,
                system=system,
                model_name=self.model,
                api_key=self.api_key,
                max_tokens=self.max_tokens,
                provider_base_url="https://api.openai.com/v1",
                temperature=0,
            )
            print(f"oai token usage: {token_usage}")
            self.total_token_usage += token_usage
            if 'gpt' in self.model:
                self.total_cost += (token_usage * 2.5 / 1000000)  # https://openai.com/api/pricing/
            elif 'o1' in self.model:
                self.total_cost += (token_usage * 15 / 1000000)  # https://openai.com/api/pricing/
            elif 'o3-mini' in self.model:
                self.total_cost += (token_usage * 1.1 / 1000000)  # https://openai.com/api/pricing/
        elif "r1" in self.model:
            vlm_response, token_usage = run_groq_interleaved(
                messages=messages_for_llm,
                system=system,
                model_name=self.model,
                api_key=self.api_key,
                max_tokens=self.max_tokens,
            )
            print(f"groq token usage: {token_usage}")
            self.total_token_usage += token_usage
            self.total_cost += (token_usage * 0.99 / 1000000)
        elif "qwen" in self.model:
            vlm_response, token_usage = run_oai_interleaved(
                messages=messages_for_llm,
                system=system,
                model_name=self.model,
                api_key=self.api_key,
                max_tokens=min(2048, self.max_tokens),
                provider_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                temperature=0,
            )
            print(f"qwen token usage: {token_usage}")
            self.total_token_usage += token_usage
            self.total_cost += (token_usage * 2.2 / 1000000)  # https://help.aliyun.com/zh/model-studio/getting-started/models?spm=a2c4g.11186623.0.0.74b04823CGnPv7#fe96cfb1a422a
        else:
            raise ValueError(f"Model {self.model} not supported")
        latency_vlm = time.time() - start
        self.output_callback(f"LLM: {latency_vlm:.2f}s, OmniParser: {parsed_screenshot.parse_time_sec:.2f}s", sender="bot")

        print(f"{vlm_response}")

        if self.print_usage:
            print(f"Total token so far: {self.total_token_usage}. Total cost so far: $USD{self.total_cost:.5f}")

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
        self.output_callback(
                    f'<details>'
                    f'  <summary>Parsed Screen elemetns by OmniParser</summary>'
                    f'  <pre>{boxids_and_labels}</pre>'
                    f'</details>',
                    sender="bot"
                )
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

