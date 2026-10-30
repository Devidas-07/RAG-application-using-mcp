import os
import sys
import asyncio
from dotenv import load_dotenv
from fastmcp import FastMCP
from elasticsearch import Elasticsearch
from openai import OpenAI
from pydantic import BaseModel

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
    response = es_client.search(
        index=es_index,
        body={
            "size": request_data.k,
            "query": {
                "bool": {
                    "should": [{
                        "multi_match": {
                            "query": request_data.query,
                            "fields": ["title", "body", "description", "created_by"],
                            "fuzziness": "AUTO",
                            "boost": 1.0,
                        }
                    }],
                    "minimum_should_match": 0,
                }
            },
            "knn": {
                "field": "embedding_field",
                "query_vector": query_embedding.data[0].embedding,
                "k": request_data.k,
                "num_candidates": 100,
                "boost": 2.0,
            },
        },
    )
    return [
        {
            "id": hit["_id"],
            "title": hit["_source"].get("title", ""),
            "description": hit["_source"].get("description", ""),
            "body": hit["_source"].get("body", ""),
            "created_by": hit["_source"].get("created_by", ""),
            "score": hit["_score"],
        }
        for hit in response["hits"]["hits"]
    ]

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
