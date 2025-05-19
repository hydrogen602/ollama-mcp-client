from typing import Literal, TypedDict
import ollama
from ollama_toolmanager import OllamaToolManager
from mcp.types import CallToolResult


class OllamaAgent:
    def __init__(
        self,
        model: str,
        tool_manager: OllamaToolManager,
        default_prompt="You are a helpful assistant who can use available tools to solve problems",
    ) -> None:
        self.model = model
        self.default_prompt = default_prompt
        self.messages: list[ollama.Message] = []
        self.tool_manager = tool_manager

    async def get_response(self, content: str) -> str:
        self.messages.append(ollama.Message(role="user", content=content))

        run_idx = 0
        while run_idx < 5:  # Limit iterations to prevent infinite loops
            run_idx += 1
            query = ollama.chat(
                model=self.model,
                messages=self.messages,
                tools=self.tool_manager.get_tools(),
            )

            tools_ran = await self.handle_response(query)
            if not tools_ran:
                break

        return "\n".join(
            f"> {e.content}" or "" for e in self.messages if e.role == "assistant"
        )

    async def handle_response(self, response: ollama.ChatResponse):
        """
        Returns true if tool calls were executed & messages updated.
        """
        if response.message.role == "assistant" and response.message.content:
            self.messages.append(
                ollama.Message(role="assistant", content=response.message.content)
            )

        try:
            tool_calls = response.message.tool_calls

            if not tool_calls:
                return False

            for tool_call in tool_calls:
                result = await self.tool_manager.execute_tool(tool_call)
                text_result = ""
                for content in result.content:
                    if hasattr(content, "text"):
                        text_result += (
                            content.text  # pyright: ignore[reportAttributeAccessIssue] - we just checked this
                        )
                    else:
                        raise NotImplementedError(
                            "Tool response content does not have 'text' attribute"
                        )

                self.messages.append(ollama.Message(role="tool", content=text_result))

            return True

        except Exception as e:
            print("Error", type(e).__name__, e)
            self.messages.append(
                ollama.Message(
                    role="tool", content=f"Error: {type(e).__name__}: {str(e)}"
                )
            )
            return True
