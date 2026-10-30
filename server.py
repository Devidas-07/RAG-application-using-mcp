import os
import sys
import asyncio
from dotenv import load_dotenv
from fastmcp import FastMCP
from elasticsearch import Elasticsearch
from openai import OpenAI
from pydantic import BaseModel
import math

os.environ["TQDM_DISABLE"] = "1"
load_dotenv()

if sys.platform == "win32" and sys.version_info < (3, 12):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

mcp = FastMCP("rag-app")

es_client = Elasticsearch(hosts=os.getenv("es_host"))
es_index = os.getenv("es_index")
cache_index = os.getenv("es_cache_index")
openai_client = OpenAI(base_url=os.getenv("base_url"), api_key=os.getenv("api_key"))
embedding_model = os.getenv("embedding_model")


class SearchRequest(BaseModel):
    query: str
    k: int = 5


@mcp.tool()
def hybrid_search(request_data: SearchRequest) -> list[dict]:
    """Search documents using hybrid BM25 + vector search."""
    query_embedding = openai_client.embeddings.create(
        model=embedding_model,
        input=request_data.query,
        encoding_format="float",
    )
    knn_query = query_embedding.data[0].embedding
    

    K = request_data.rank_constant if hasattr(request_data, "rank_constant") else 60
    WINDOW = 50

    # ---- 1. Keyword / BM25 retriever (standard query) ----
    kw_response = es_client.search(
        index=es_index,
        size=WINDOW,
        query={
            "bool": {
                "should": [
                    { "term":  { "title": { "value": request_data.query, "boost": 10.0 } } },
                    { "match": { "title": { "query": request_data.query, "boost": 5.0 } } },
                    { "match": { "body":  { "query": request_data.query, "boost": 3.0 } } },
                    { "match": { "description": { "query": request_data.query, "boost": 2.0 } } },
                ]
            }
        },
    )

    # ---- 2. kNN / vector retriever ----
    knn_response = es_client.search(
        index=es_index,
        size=WINDOW,
        knn={
            "field": "embedding_field",
            "query_vector": knn_query,
            "k": 50,
            "num_candidates": 100,
        },
    )



    fused = {}
    docs = {}
    for response in (kw_response, knn_response):
        for doc_id, score in rrf_scores(response).items():
            fused[doc_id] = fused.get(doc_id, 0.0) + score
        for hit in response["hits"]["hits"]:
            docs.setdefault(doc_id := hit["_id"], hit["_source"]) if False else docs.__setitem__(doc_id, hit) if False else None
            docs[hit["_id"]] = hit

    # Order by combined score desc, limit to rank_window_size
    ranked_ids = sorted(fused, key=lambda d: fused[d], reverse=True)[:WINDOW]

    return [
        {
            "id": doc_id,
            "title": docs[doc_id]["_source"].get("title", ""),
            "description": docs[doc_id]["_source"].get("description", ""),
            "body": docs[doc_id]["_source"].get("body", ""),
            "created_by": docs[doc_id]["_source"].get("created_by", ""),
            "score": fused[doc_id],  # combined RRF score, same formula as native
        }
        for doc_id in ranked_ids
    ]
# ---- 3. Reciprocal Rank Fusion (same formula as ES native RRF) ----
def rrf_scores(response):
    """Map doc_id -> 1 / (rank_constant + rank), rank starting at 1."""
    out = {}
    for rank, hit in enumerate(response["hits"]["hits"], start=1):
        out[hit["_id"]] = 1.0 / (60 + rank)
    return out

@mcp.tool()
def clear_cache_index():
    """Clear the Elasticsearch index cache."""
    query_body = {
        "query": {
            "match_all": {}
        }
    }
    es_client.delete_by_query(index=cache_index, body=query_body)
    return {"message": "Cache cleared."}
if __name__ == "__main__":
    mcp.run()
