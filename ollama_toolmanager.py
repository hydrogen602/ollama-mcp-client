from collections.abc import Awaitable
from typing import Any, Callable, Literal, TypedDict
from dataclasses import dataclass
from mcp.types import CallToolResult, TextContent
import ollama


@dataclass
class OllamaTool:
    name: str
    function: Callable[..., Awaitable[CallToolResult]]
    description: str
    properties: dict[str, Any]
    required: list[str]


class FunctionDetails(TypedDict):
    name: str
    description: str
    properties: dict[str, Any]
    required: list[str]


class FunctionInfo(TypedDict):
    type: Literal["function"]
    function: FunctionDetails


class OllamaToolManager:
    def __init__(self):
        self.tools: dict[str, OllamaTool] = {}

    def register_tool(
        self,
        name: str,
        function: Callable[..., Awaitable[CallToolResult]],
        description: str,
        inputSchema: dict[str, Any],
    ) -> None:
        """
        Register a function as a tool.
        """
        properties = inputSchema["properties"]
        required = inputSchema["required"]
        tool = OllamaTool(name, function, description, properties, required)
        self.tools[name] = tool

    def get_tools(self) -> list[FunctionInfo]:
        """
        Generate the tools specification.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.description,
                    "properties": tool.properties,
                    "required": tool.required,
                },
            }
            for name, tool in self.tools.items()
        ]

    async def execute_tool(self, payload: ollama.Message.ToolCall) -> CallToolResult:
        """
        Execute a tool based on the agent's request, handling name translation
        """
        function = payload.function
        name = function.name
        tool_input = function.arguments

        if name not in self.tools:
            raise ValueError(f"Unknown tool: {name}")
        try:
            tool_func = self.tools[name].function
            print(f"Tool call: {name}({tool_input})")
            return await tool_func(name, tool_input)
        except Exception as e:
            result = TextContent(type="text", text=f"Error executing tool: {str(e)}")
            return CallToolResult(
                content=[result],
                isError=True,
            )

    def clear_tools(self) -> None:
        """Clear all registered tools"""
        self.tools.clear()
