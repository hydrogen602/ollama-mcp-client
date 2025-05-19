from typing import Literal, TypedDict
import ollama
from ollama_toolmanager import OllamaToolManager
from mcp.types import CallToolResult


class Message(TypedDict):
    role: Literal["user", "assistant", "tool"]
    content: str


class OllamaAgent:
    def __init__(
        self,
        model: str,
        tool_manager: OllamaToolManager,
        default_prompt="You are a helpful assistant who can use available tools to solve problems",
    ) -> None:
        self.model = model
        self.default_prompt = default_prompt
        self.messages: list[Message] = []
        self.tool_manager = tool_manager

    async def get_response(self, content: str) -> str:
        self.messages.append({"role": "user", "content": content})

        query = ollama.chat(
            model=self.model,
            messages=self.messages,
            tools=self.tool_manager.get_tools(),
        )

        return await self.handle_response(query)

    async def handle_response(self, response: ollama.ChatResponse) -> str:
        try:
            tool_calls = response.message.tool_calls
            self.messages.append({"role": "tool", "content": str(response)})

            if not tool_calls:
                return ""

            results: list[CallToolResult] = []
            for tool_call in tool_calls:
                results.append(await self.tool_manager.execute_tool(tool_call))

            tool_response = []
            for result in results:
                for content in result.content:
                    if not hasattr(content, "text"):
                        raise NotImplementedError(
                            "Tool response content does not have 'text' attribute"
                        )
                    tool_response.append(
                        content.text  # pyright: ignore[reportAttributeAccessIssue] - we just checked this
                    )

            return "".join(tool_response)
        except Exception as e:
            print(e)
            return f'Function calling failed with error: "{e}"'
