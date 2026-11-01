import json
import asyncio
import logging
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

_session: ClientSession | None = None
_tools_cache: list[dict] | None = None
_ready = asyncio.Event()
_shutdown = asyncio.Event()


async def _run_session():
    """Runs inside a background task — keeps stdio_client context alive."""
    global _session, _tools_cache
    server_params = StdioServerParameters(command="uv", args=["run", "python", "server.py"])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            _tools_cache = [
                {"name": t.name, "description": t.description, "schema": t.input_schema}
                for t in result.tools
            ]
            _session = session
            logger.info("[MCP] Connected. Tools: %s", [t["name"] for t in _tools_cache])
            _ready.set()
            await _shutdown.wait()   # stay alive until shutdown is signalled
    logger.info("[MCP] Session closed.")


async def connect():
    """Start the background session task and wait until it's ready."""
    asyncio.create_task(_run_session())
    await _ready.wait()


async def disconnect():
    """Signal the background task to shut down."""
    _shutdown.set()


async def list_tools() -> list[dict]:
    if _tools_cache is None:
        raise RuntimeError("MCP session not initialized.")
    return _tools_cache


async def call_tool(tool_name: str, arguments: dict):
    if _session is None:
        raise RuntimeError("MCP session not initialized.")
    result = await _session.call_tool(tool_name, arguments=arguments)
    parts = [block.text for block in result.content if hasattr(block, "text")]
    if not parts:
        return None
    parsed = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            obj = json.loads(part)
            parsed.extend(obj) if isinstance(obj, list) else parsed.append(obj)
        except json.JSONDecodeError:
            try:
                decoder = json.JSONDecoder()
                idx = 0
                while idx < len(part):
                    part = part[idx:].lstrip()
                    if not part:
                        break
                    obj, end = decoder.raw_decode(part)
                    parsed.append(obj)
                    idx = end
            except json.JSONDecodeError:
                # plain string response (e.g. "done", "ok") — return as-is
                parsed.append(part)
    return parsed if len(parsed) != 1 else parsed[0]
