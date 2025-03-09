"""
Agentic sampling loop that calls the Anthropic API and local implenmentation of anthropic-defined computer use tools.
"""
from collections.abc import Callable
from enum import StrEnum
import time

from anthropic import APIResponse
from anthropic.types import (
    TextBlock,
)
from anthropic.types.beta import (
    BetaContentBlock,
    BetaMessage,
    BetaMessageParam
)
from tools import ToolResult, VNCClient

from agent.llm_utils.omniparserclient import OmniParserClient
from agent.llm_utils.utils import BotMessage, ToolResultMessage, UserMessage, ParsedScreenshot
from agent.anthropic_agent import AnthropicActor
from agent.vlm_agent import VLMAgent
from executor.anthropic_executor import AnthropicExecutor


BETA_FLAG = "computer-use-2024-10-22"

class APIProvider(StrEnum):
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    VERTEX = "vertex"
    OPENAI = "openai"


PROVIDER_TO_DEFAULT_MODEL_NAME: dict[APIProvider, str] = {
    APIProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
    APIProvider.BEDROCK: "anthropic.claude-3-5-sonnet-20241022-v2:0",
    APIProvider.VERTEX: "claude-3-5-sonnet-v2@20241022",
    APIProvider.OPENAI: "gpt-4o",
}

def sampling_loop_sync(
    *,
    model: str,
    provider: APIProvider | None,
    messages: list[BetaMessageParam],
    output_callback: Callable[[BetaContentBlock], None],
    api_key: str,
    only_n_most_recent_images: int | None = 2,
    max_tokens: int = 4096,
    vm_url: str,
    omniparser_url: str
):
    """
    Synchronous agentic sampling loop for the assistant/tool interaction of computer use.
    """
    print('in sampling_loop_sync, model:', model)
    assert(len(messages) == 1)
    initial_request = UserMessage(messages[0]['content'][0].text)
    session_history = [initial_request]

    computer_client = VNCClient(vm_url)
    omniparser_client = OmniParserClient(url=f"http://{omniparser_url}/parse/", computer_client=computer_client)
    if model == "claude-3-5-sonnet-20241022":
        # Register Actor and Executor
        actor = AnthropicActor(
            computer_client=computer_client,
            model=model,
            provider=provider,
            api_key=api_key,
            max_tokens=max_tokens,
            only_n_most_recent_images=only_n_most_recent_images
        )
    elif model in set(["omniparser + gpt-4o", "omniparser + o1", "omniparser + o3-mini", "omniparser + R1", "omniparser + qwen2.5vl"]):
        actor = VLMAgent(
            model=model,
            provider=provider,
            api_key=api_key,
            output_callback=output_callback,
            max_tokens=max_tokens,
            only_n_most_recent_images=only_n_most_recent_images
        )
    else:
        raise ValueError(f"Model {model} not supported")
    executor = AnthropicExecutor(
        computer_client=computer_client,
        output_callback=output_callback,
    )
    print(f"Model Inited: {model}, Provider: {provider}")

    # tool_result_content = None

    print(f"Start the message loop. User messages: {messages}")

    # Main loop
    while True:
        parsed_screen: ParsedScreenshot = omniparser_client()
        session_history.append(parsed_screen)
        yield(parsed_screen)
        # tools_use_needed = actor(messages=messages, parsed_screen=parsed_screen)
        bot_message: BotMessage = actor(session_history)
        session_history.append(bot_message)
        yield(bot_message)

        if len(bot_message.requested_actions) == 0:
            return []

        for requested_action in bot_message.requested_actions:
            tool_result_message: ToolResultMessage = executor(requested_action)
            session_history.append(tool_result_message)
            yield(tool_result_message)

        time.sleep(1)
        # for message, tool_result_content in executor(tools_use_needed, messages):
        #     yield message

        # if not tool_result_content:
        #     return messages

        # if model == "claude-3-5-sonnet-20241022":
        #     messages.append({"content": tool_result_content, "role": "user"})

    # computer_client.shutdown()