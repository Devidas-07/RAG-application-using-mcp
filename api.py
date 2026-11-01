import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
import asyncio
import json
import os
import uuid
import warnings
import logging
from datetime import datetime
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from elasticsearch import Elasticsearch
from pydantic import BaseModel
import mcp_client

load_dotenv()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[STARTUP] Connecting to MCP server...")
    await mcp_client.connect()
    logger.info("[STARTUP] MCP session ready.")
    if router is not None:
        logger.info("[STARTUP] Warming up Laya router...")
        try:
            await asyncio.to_thread(
                router.predict,
                state={"user_query": "warmup"},
                questions=is_harmful_question,
            )
            logger.info("[STARTUP] Laya router warm.")
        except Exception:
            logger.warning("[STARTUP] Laya warmup failed, will warm on first request.")
    yield
    logger.info("[SHUTDOWN] Closing MCP session...")
    await mcp_client.disconnect()

app = FastAPI(lifespan=lifespan)
async_client = AsyncOpenAI(base_url=os.getenv("base_url"), api_key=os.getenv("api_key"))
llm_model = os.getenv("llm_model")

es_client = Elasticsearch(hosts=os.getenv("es_host"))
es_cache_index = os.getenv("es_cache_index")

# ── Laya router ───────────────────────────────────────────────────────────────
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        from laya import Router
        router = Router()
except Exception:
    logger.warning("[LAYA] Router failed to load. Will use first available tool.")
    router = None

HARMFUL_THRESHOLD = 0.80
MAX_QUERY_CHARS = 1500

is_harmful_question = {
    "is_harmful_query": {
        "type": "noul",
        "instructions": (
            "Is this query potentially harmful, offensive, or forbidden? "
            "Return true for harmful, inappropriate, or forbidden content."
        ),
    }
}

# ── Plain helper functions ────────────────────────────────────────────────────
def check_cache(query: str) -> dict | None:
    hits = es_client.search(
        index=es_cache_index,
        body={"query": {"term": {"query.keyword": {"value": query}}}},
    )["hits"]["hits"]
    if not hits or "_source" not in hits[0]:
        return None
    source = hits[0]["_source"]
    return {"answer": source["answer"], "sources": source.get("sources", [])} if source.get("answer") else None


def save_to_cache(query: str, answer: str, sources: list):
    doc_id = str(uuid.uuid4())
    es_client.index(index=es_cache_index, id=doc_id, document={
        "id": doc_id, "query": query, "answer": answer, "sources": sources,
    })


def is_harmful_input(query: str) -> bool:
    if router is None:
        return False
    try:
        result = router.predict(
            state={"user_query": query[:MAX_QUERY_CHARS]},
            questions=is_harmful_question,
        )
        score = result.get("answers", {}).get("is_harmful_query", {}).get("noul", 0.0)
        logger.info("[HARMFUL] query=%r score=%.3f", query, score)
        return score >= HARMFUL_THRESHOLD
    except Exception:
        logger.exception("[HARMFUL] Check failed for %r, allowing it", query)
        return False


def select_tool(query: str, tools: list[dict]) -> str:
    if not tools:
        raise ValueError("No tools available from MCP server")
    if len(tools) == 1:
        logger.info("[TOOL SELECT] Only one tool, using: %s", tools[0]["name"])
        return tools[0]["name"]

    if router is None:
        logger.info("[TOOL SELECT] Router unavailable, using first tool")
        return tools[0]["name"]

    scores = {}
    for tool in tools:
        question = {
            "is_correct_tool": {
                "type": "noul",
                "instructions": (
                    f"Given the user query, should we call the tool '{tool['name']}'? "
                    f"Tool description: {tool.get('description', '')}. "
                    "Return true if this is the most appropriate tool for the query."
                ),
            }
        }
        try:
            result = router.predict(
                state={"user_query": query[:MAX_QUERY_CHARS]},
                questions=question,
            )
            score = result.get("answers", {}).get("is_correct_tool", {}).get("noul", 0.0)
            scores[tool["name"]] = score
            logger.info("[TOOL SELECT] tool=%s score=%.3f", tool["name"], score)
        except Exception:
            logger.exception("[TOOL SELECT] Scoring failed for tool %s", tool["name"])
            scores[tool["name"]] = 0.0

    chosen = max(scores, key=scores.get)
    logger.info("[TOOL SELECT] Scores: %s -> chose: %s", scores, chosen)
    return chosen


# ── API ───────────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str


def sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"

@app.get("/")
def get_health():
    return {"status":"health is ok.."}
@app.post("/ask")
async def ask(request: QueryRequest):
    query = request.query

    async def stream():
        start = datetime.now()
        logger.info("[REQUEST] query=%r", query)
        try:
            # Step 1 — cache check + harmful check in parallel
            logger.info("[STEP 1] Running cache check + harmful check in parallel...")
            cached, is_harmful = await asyncio.gather(
                asyncio.to_thread(check_cache, query),
                asyncio.to_thread(is_harmful_input, query),
            )
            logger.info("[STEP 1] cache_hit=%s is_harmful=%s (%.2fs)",
                        cached is not None, is_harmful, (datetime.now() - start).total_seconds())

            if is_harmful:
                yield sse({"type": "error", "message": "Sorry, we can't proceed with this query."})
                return

            if cached:
                logger.info("[CACHE HIT] Returning cached answer.")
                yield sse({"type": "sources", "sources": cached.get("sources", [])})
                yield sse({"type": "token", "content": cached["answer"]})
                yield sse({"type": "done"})
                return

            # Step 2 — get cached tool list (no subprocess spawn)
            logger.info("[STEP 2] Fetching tool list...")
            tools = await mcp_client.list_tools()
            logger.info("[STEP 2] Tools available: %s (%.2fs)",
                        [t["name"] for t in tools], (datetime.now() - start).total_seconds())
            yield sse({"type": "status", "message": "🔍 Searching knowledge base..."})

            # Step 3 — laya selects tool
            logger.info("[STEP 3] Selecting tool via Laya...")
            tool_name = await asyncio.to_thread(select_tool, query, tools)
            logger.info("[STEP 3] Tool selected: %s (%.2fs)",
                        tool_name, (datetime.now() - start).total_seconds())
            yield sse({"type": "status", "message": f"⚙️ Running tool: {tool_name}..."})

            # Step 4 — call MCP tool on persistent session
            logger.info("[STEP 4] Calling MCP tool: %s...", tool_name)
            tool_schema = next((t["schema"] for t in tools if t["name"] == tool_name), {})
            required = tool_schema.get("required", [])
            properties = tool_schema.get("properties", {})
            # build arguments only for params the tool actually declares
            arguments = {}
            if "request_data" in properties:
                arguments["request_data"] = {"query": query, "k": 5}
            elif "query" in properties:
                arguments["query"] = query
            # tools with no required params (e.g. clear_cache_index) get empty args
            logger.info("[STEP 4] Calling with arguments: %s", arguments)
            tool_result = await mcp_client.call_tool(tool_name, arguments)
            search_results = tool_result if isinstance(tool_result, list) else []
            logger.info("[STEP 4] Tool returned: %s (%.2fs)",
                        f"{len(search_results)} results" if search_results else tool_result,
                        (datetime.now() - start).total_seconds())

            # if tool returned no search results (e.g. clear_cache_index), respond directly
            if not search_results:
                msg = tool_result if isinstance(tool_result, str) else "Action completed successfully."
                yield sse({"type": "token", "content": msg})
                yield sse({"type": "done"})
                return

            sources = [{"id": r.get("id"), "title": r.get("title", "")} for r in search_results]
            yield sse({"type": "sources", "sources": sources})
            yield sse({"type": "status", "message": "🤖 Generating answer..."})

            # Step 5 — augment context
            context = "\n".join(
                f"Title: {r['title']}\nDescription: {r['description']}\nBody: {r['body']}\n"
                for r in search_results
            )

            # Step 6 — stream LLM answer
            logger.info("[STEP 6] Calling LLM: %s...", llm_model)
            stream_resp = await async_client.chat.completions.create(
                model=llm_model,
                stream=True,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": (
                        f"Use the following context to answer the query.\n\nContext:\n{context}\n\n"
                        f"Query: {query}\nAnswer in 5-6 lines. If the context doesn't contain "
                        "the answer, say you don't have enough information."
                    )},
                ],
            )

            full_answer = []
            first_token = True
            async for chunk in stream_resp:
                if not chunk.choices:
                    continue
                token = chunk.choices[0].delta.content
                if token:
                    if first_token:
                        logger.info("[STEP 6] First token received (%.2fs)",
                                    (datetime.now() - start).total_seconds())
                        first_token = False
                    full_answer.append(token)
                    yield sse({"type": "token", "content": token})

            if not full_answer:
                yield sse({"type": "error", "message": "The model returned an empty answer."})
                return

            yield sse({"type": "done"})
            logger.info("[DONE] Total time: %.2fs", (datetime.now() - start).total_seconds())

            asyncio.create_task(
                asyncio.to_thread(save_to_cache, query, "".join(full_answer), search_results)
            )

        except Exception:
            logger.exception("[ERROR] Streaming failed")
            yield sse({"type": "error", "message": "Something went wrong while generating the answer."})

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
