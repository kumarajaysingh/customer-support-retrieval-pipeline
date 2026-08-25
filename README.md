# Customer Support Retrieval Pipeline

Retrieval service for the customer-support RAG pipeline. It exposes a single
FastAPI endpoint, `POST /retrieve`, that turns a natural-language query into a
ranked list of document chunks pulled from Weaviate. It does **not** generate
answers — this is retrieval only, meant to be called by an agent/LLM layer
that consumes the returned chunks as context.

## How a request flows

`POST /retrieve` runs through the following stages (see
[src/api/pipeline.py](src/api/pipeline.py)):

1. **Embed** — the query is embedded locally with the same Hugging Face model
   used at ingestion time (`BAAI/bge-base-en-v1.5` by default), prefixed with
   a BGE asymmetric-retrieval instruction string
   ([src/embeddings/embedder.py](src/embeddings/embedder.py)).
2. **Hybrid search** — Weaviate is queried with both the query vector and
   BM25 keyword matching (`collection.query.hybrid`), restricted to the
   `chunk_text` and `section_title` properties. The blend between keyword and
   vector search (`alpha`) is tuned per category
   ([src/api/search_strategy.py](src/api/search_strategy.py)):
   - `product` / `refund`: alpha `0.4` (keyword-leaning — these collections
     contain exact identifiers like product IDs and category names).
   - `technical`: alpha `0.7` (vector-leaning — customers describe symptoms
     in their own words rather than document phrasing).

   The candidate pool size is `top_k * CANDIDATE_K_MULTIPLIER`, over-fetching
   so the reranker has enough to work with.
3. **Score threshold** — candidates below the category's calibrated hybrid
   score cutoff (`SCORE_THRESHOLD_*`) are dropped before reranking.
4. **Rerank** — surviving candidates are reranked with a cross-encoder
   (`BAAI/bge-reranker-base` by default) that scores the query and each
   chunk's text together, correcting cases where hybrid search over/under-
   ranked a chunk on keyword or embedding overlap alone
   ([src/api/rerank.py](src/api/rerank.py)). A `NoopReranker` passthrough
   exists behind the same `Reranker` protocol for testing or for disabling
   reranking.
5. **Truncate & respond** — the top `top_k` survivors are mapped to the
   response schema and returned, alongside per-stage latency and score
   logging for observability.

## Project layout

```
main.py                        Entry point — `python main.py` starts uvicorn
src/
  config.py                    Loads settings once from .env
  api/
    app.py                     FastAPI app + lifespan (loads models, verifies
                                collections exist, wires rate limiting)
    routes.py                  POST /retrieve — the only endpoint
    pipeline.py                Orchestrates embed -> search -> filter ->
                                rerank -> truncate
    search_strategy.py         Hybrid search params (alpha, thresholds) per
                                category
    rerank.py                  Reranker protocol + NoopReranker /
                                CrossEncoderReranker implementations
    schemas.py                 Request/response Pydantic models
    dependencies.py            FastAPI Depends() providers + API-key auth
    errors.py                  Exception -> HTTP response mapping
  embeddings/
    embedder.py                Query embedding (Hugging Face, local model)
  weaviate_store/
    client.py                  Async Weaviate client (long-lived, gRPC)
  utils/
    logger.py                  Shared logger (console + file)
logs/retrieval_api.log         Log output (path configurable via .env)
```

## Prerequisites

- Python 3.10+
- A running Weaviate instance (local Docker by default) with the three
  collections below already populated by the ingestion pipeline — this
  service fails fast at startup if any collection is missing:
  - `product_specs`
  - `technical_specs`
  - `refund_specs`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirement.txt
cp .env.example .env             # then edit as needed
```

The embedding and reranker models (`sentence-transformers` /
`transformers`) download from Hugging Face on first run and are cached
locally afterward, so the first startup needs network access.

## Configuration

All configuration is loaded once from `.env` at startup
([src/config.py](src/config.py)). See [.env.example](.env.example) for the
full list of variables and their defaults:

| Group | Variables |
|---|---|
| Weaviate connection | `WEAVIATE_HTTP_HOST`, `WEAVIATE_HTTP_PORT`, `WEAVIATE_GRPC_HOST`, `WEAVIATE_GRPC_PORT`, `WEAVIATE_API_KEY` |
| Models | `EMBEDDING_MODEL_NAME`, `RERANKER_MODEL_NAME` |
| Collections | `PRODUCT_SPECS_COLLECTION`, `TECHNICAL_SPECS_COLLECTION`, `REFUND_SPECS_COLLECTION` |
| API | `API_HOST`, `API_PORT`, `API_KEY`, `REQUEST_TIMEOUT_SECONDS`, `MAX_QUERY_CHARS`, `RATE_LIMIT_PER_MINUTE` |
| Retrieval | `DEFAULT_TOP_K`, `MAX_TOP_K`, `CANDIDATE_K_MULTIPLIER` |
| Hybrid search alpha | `HYBRID_ALPHA_PRODUCT`, `HYBRID_ALPHA_TECHNICAL`, `HYBRID_ALPHA_REFUND` |
| Score thresholds | `SCORE_THRESHOLD_PRODUCT`, `SCORE_THRESHOLD_TECHNICAL`, `SCORE_THRESHOLD_REFUND` |
| Logging | `LOG_FILE_PATH`, `LOG_LEVEL` |

> **Important:** `EMBEDDING_MODEL_NAME` must match whatever model the
> ingestion pipeline used to embed documents — query and document vectors
> have to come from the same model space.

`API_KEY` is optional; if unset, the `x-api-key` header check is skipped.

## Running

```bash
python main.py
```

Starts uvicorn on `API_HOST:API_PORT` (`0.0.0.0:8089` by default) with
auto-reload enabled. On startup the service loads the embedding and
reranker models and verifies all three collections exist in Weaviate before
accepting traffic.

## API

### `POST /retrieve`

Headers:
- `x-api-key: <API_KEY>` — required only if `API_KEY` is set in `.env`.

Request body:

```json
{
  "query": "How do I get a refund for a damaged item?",
  "category": "refund",
  "top_k": 5
}
```

- `query` — required, non-empty, max `MAX_QUERY_CHARS` characters.
- `category` — required, one of `product`, `technical`, `refund`.
- `top_k` — optional, `1..MAX_TOP_K`; defaults to `DEFAULT_TOP_K`.

Response body:

```json
{
  "results": [
    {
      "chunk_id": "…",
      "chunk_text": "…",
      "file_name": "…",
      "page_no": 3,
      "is_table": false,
      "section_title": "…",
      "product_name": "…",
      "ingested_at": "…",
      "score": 0.87
    }
  ],
  "result_count": 1,
  "category": "refund",
  "query": "How do I get a refund for a damaged item?"
}
```

`score` is always the score of the reranker that was actually used, so it
matches the returned order.

Error responses:
- `401` — missing/invalid `x-api-key` (when `API_KEY` is configured).
- `422` — request validation failure (empty/too-long query, invalid category, `top_k` out of range).
- `429` — rate limit exceeded (`RATE_LIMIT_PER_MINUTE` per client IP).
- `504` — request exceeded `REQUEST_TIMEOUT_SECONDS`.
- `500` — unhandled server error.

## Logging

Every stage of a request logs to both the console and `LOG_FILE_PATH`
(`logs/retrieval_api.log` by default): candidate/survivor/returned counts,
per-stage latency (embed/search/rerank), and the final scores — the same
data future retrieval-quality tracking (e.g. Recall@k, MRR) would depend on.
