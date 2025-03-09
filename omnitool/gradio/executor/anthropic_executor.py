import asyncio
from typing import Any, cast
from collections.abc import Callable
from anthropic.types.beta import BetaToolUseBlock
from agent.llm_utils.utils import ToolResultMessage
from tools import ComputerTool, ToolCollection


class AnthropicExecutor:
    def __init__(
        self,
        computer_client,
        output_callback: Callable[[Any], None],
    ):
        self.tool_collection = ToolCollection(
            ComputerTool(computer_client)
        )
        self.output_callback = output_callback

    def __call__(self, requested_action: BetaToolUseBlock): #, messages: list[BetaMessageParam]):
        self.output_callback(f"Next I will perform the following action: {requested_action.input}", sender="bot")

        # Run the asynchronous tool execution in a synchronous context
        result = asyncio.run(self.tool_collection.run(
            name=requested_action.name,
            tool_input=cast(dict[str, Any], requested_action.input),
        ))

        self.output_callback(result, sender="bot")

        return ToolResultMessage(result.output or result.error)

# def _message_display_callback(messages):
#     display_messages = []
#     for msg in messages:
#         try:
#             if isinstance(msg["content"][0], TextBlock):
#                 display_messages.append((msg["content"][0].text, None))  # User message
#             elif isinstance(msg["content"][0], BetaTextBlock):
#                 display_messages.append((None, msg["content"][0].text))  # Bot message
#             elif isinstance(msg["content"][0], BetaToolUseBlock):
#                 display_messages.append((None, f"Tool Use: {msg['content'][0].name}\nInput: {msg['content'][0].input}"))  # Bot message
#             elif isinstance(msg["content"][0], Dict) and msg["content"][0]["content"][-1]["type"] == "image":
#                 display_messages.append((None, f'<img src="data:image/png;base64,{msg["content"][0]["content"][-1]["source"]["data"]}">'))  # Bot message
#             else:
#                 print(msg["content"][0])
#         except Exception as e:
#             print("error", e)
#             pass
#     return display_messages

# def _make_api_tool_result(
#     result: ToolResult, tool_use_id: str
# ) -> BetaToolResultBlockParam:
#     """Convert an agent ToolResult to an API ToolResultBlockParam."""
#     tool_result_content: list[BetaTextBlockParam | BetaImageBlockParam] | str = []
#     is_error = False
#     if result.error:
#         is_error = True
#         tool_result_content = _maybe_prepend_system_tool_result(result, result.error)
#     else:
#         if result.output:
#             tool_result_content.append(
#                 {
#                     "type": "text",
#                     "text": _maybe_prepend_system_tool_result(result, result.output),
#                 }
#             )
#         if result.base64_image:
#             tool_result_content.append(
#                 {
#                     "type": "image",
#                     "source": {
#                         "type": "base64",
#                         "media_type": "image/png",
#                         "data": result.base64_image,
#                     },
#                 }
#             )
#     return {
#         "type": "tool_result",
#         "content": tool_result_content,
#         "tool_use_id": tool_use_id,
#         "is_error": is_error,
#     }


# def _maybe_prepend_system_tool_result(result: ToolResult, result_text: str):
#     if result.system:
#         result_text = f"<system>{result.system}</system>\n{result_text}"
#     return result_text