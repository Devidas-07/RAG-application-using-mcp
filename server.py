from fastmcp import FastMCP
mcp = FastMCP("rag-app")

@mcp.tool()
def test_tool ():
    return "health is ok"
if __name__ == "__main__":
    # Start the server using standard input/output transport
    mcp.run()