from typing import Any, Self
from mcp import ClientSession, StdioServerParameters, Tool
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult


class MCPClient:
    def __init__(self, server_params: StdioServerParameters):
        self.server_params = server_params
        self.session = None
        self._client = None

    async def __aenter__(self) -> Self:
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.__aexit__(exc_type, exc_val, exc_tb)
        if self._client:
            await self._client.__aexit__(exc_type, exc_val, exc_tb)

    async def connect(self):
        """Establishes connection to MCP server"""
        self._client = stdio_client(self.server_params)
        self.read, self.write = await self._client.__aenter__()
        session = ClientSession(self.read, self.write)
        self.session = await session.__aenter__()
        await self.session.initialize()

    async def get_available_tools(self) -> list[Tool]:
        """List available tools"""
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        tools = await self.session.list_tools()
        if tools.nextCursor:
            raise NotImplementedError(
                "Pagination is not supported yet. Please implement pagination handling."
            )
        return tools.tools

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any] | None
    ) -> CallToolResult:
        """Call a tool with given arguments"""
        if not self.session:
            raise RuntimeError("Not connected to MCP server")
        result = await self.session.call_tool(tool_name, arguments=arguments)
        return result
