import asyncio
from mcp import StdioServerParameters
from mcpclient import MCPClient
from ollama_toolmanager import OllamaToolManager
from agent import OllamaAgent


async def main(user_prompt: str = "get weather for texas"):
    # Git server configuration
    git_server_params = StdioServerParameters(
        command="bash", args=["../mcp-client/run.sh"], env=None
    )

    # Update model in OllamaAgent
    # List of local models supporting tool usage: https://ollama.com/search?c=tools
    agent = OllamaAgent("PetrosStav/gemma3-tools:4b", OllamaToolManager())

    async with MCPClient(git_server_params) as mcpclient:
        print("Fetching available tools from the MCP server")
        tools_list = await mcpclient.get_available_tools()
        for tool in tools_list:
            agent.tool_manager.register_tool(
                name=tool.name,
                function=mcpclient.call_tool,
                description=tool.description or "",
                inputSchema=tool.inputSchema,
            )
        print("Tools registered successfully.")

        try:
            res = await agent.get_response(user_prompt)
            print(res)
        except Exception as e:
            print(f"\nError occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
