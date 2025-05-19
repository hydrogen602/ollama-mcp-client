import asyncio
from mcp import StdioServerParameters
from mcpclient import MCPClient
from ollama_toolmanager import OllamaToolManager
from agent import OllamaAgent


async def main():
    # Update model in OllamaAgent
    # List of local models supporting tool usage: https://ollama.com/search?c=tools
    agent = OllamaAgent("PetrosStav/gemma3-tools:4b", OllamaToolManager())

    async with (
        MCPClient(StdioServerParameters(command="mcp-fs")) as mcpclient,
        MCPClient(
            StdioServerParameters(
                command="npx", args=["-y", "@modelcontextprotocol/server-memory"]
            )
        ) as memory,
    ):
        print("Fetching available tools from the MCP server")
        await agent.add_tools(mcpclient)
        await agent.add_tools(memory)

        while True:
            user_prompt = input("prompt> ")
            if user_prompt.lower() in ["exit", "quit", "q", ":q"]:
                break

            print(await agent.get_response(user_prompt))


if __name__ == "__main__":
    asyncio.run(main())
