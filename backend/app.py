import os
import re
import uuid
from collections import deque
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load configuration from .env
load_dotenv()

# Azure OpenAI configuration
def _require_env(var: str) -> str:
    value = os.getenv(var, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable {var}")
    return value


def _clean_endpoint(value: str, var: str) -> str:
    """Ensure endpoint is valid and does not keep placeholders or missing hosts."""
    parsed = urlparse(value)
    if parsed.scheme not in {"https", "http"} or not parsed.netloc:
        raise RuntimeError(f"{var} is invalid. Provide the full Azure endpoint URL.")
    if "<" in parsed.netloc or ">" in parsed.netloc:
        raise RuntimeError(f"{var} still contains a placeholder. Replace it with the real Azure endpoint.")
    return value.rstrip("/")


AOAI_ENDPOINT = _clean_endpoint(_require_env("AZURE_OPENAI_ENDPOINT"), "AZURE_OPENAI_ENDPOINT")
AOAI_DEPLOYMENT = _require_env("AZURE_OPENAI_DEPLOYMENT")
AOAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
if not AOAI_API_VERSION:
    raise RuntimeError("Missing required environment variable AZURE_OPENAI_API_VERSION")
AOAI_KEY = _require_env("AZURE_OPENAI_API_KEY")

EMBED_ENDPOINT = os.getenv("AZURE_OPENAI_EMBED_ENDPOINT", "").strip() or AOAI_ENDPOINT
EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "").strip()
EMBED_API_VERSION = os.getenv("AZURE_OPENAI_EMBED_API_VERSION", "").strip() or AOAI_API_VERSION
EMBED_KEY = os.getenv("AZURE_OPENAI_EMBED_API_KEY", "").strip() or AOAI_KEY

# Azure Search configuration (optional)
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT","").rstrip("/")
SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX","")
SEARCH_KEY = os.getenv("AZURE_SEARCH_API_KEY","")
SEARCH_API_VERSION = os.getenv("AZURE_SEARCH_API_VERSION", "2023-11-01").strip()

SEARCH_FIELDS = [f.strip() for f in os.getenv("AZURE_SEARCH_FIELDS", "").split(",") if f.strip()]
SEARCH_VECTOR_FIELD = os.getenv("AZURE_SEARCH_VECTOR_FIELD", "").strip()

STOPWORDS = {
    "a","an","the","and","or","but","so","to","for","with","about",
    "we","you","i","me","my","our","your","they","them","their","it",
    "is","are","was","were","be","am","of","at","on","in","as","by",
    "from","all","well","just","like","want","need","can","could",
    "please","hey","hi","hello","good","evening","morning","afternoon",
}


def _extract_keywords(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    return [t for t in tokens if len(t) > 2 and t not in STOPWORDS]


def _get_embedding(text: str) -> list[float] | None:
    if not (SEARCH_VECTOR_FIELD and EMBED_DEPLOYMENT):
        return None
    url = f"{EMBED_ENDPOINT}/openai/deployments/{EMBED_DEPLOYMENT}/embeddings?api-version={EMBED_API_VERSION}"
    headers = {"api-key": EMBED_KEY, "Content-Type": "application/json"}
    payload = {"input": text}
    try:
        with httpx.Client(timeout=20) as c:
            r = c.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
        return data["data"][0]["embedding"]
    except Exception as e:
        print(f"Embedding request failed: {e}")
        return None

app = FastAPI(title="RAG Netflix Agent")

# Simple in-memory conversation history so we do not lose recent context
SESSION_HISTORY: dict[str, deque[dict[str, str]]] = {}
MAX_HISTORY_MESSAGES = 10  # keep the latest 10 messages (user + assistant)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None

def _append_vector_query(payload: dict, embedding: list[float], k: int) -> None:
    if not SEARCH_VECTOR_FIELD:
        return
    version = SEARCH_API_VERSION.lower()
    if any(y in version for y in ("2024", "2025")):
        payload["vectorQueries"] = [{
            "kind": "vector",
            "vector": embedding,
            "fields": SEARCH_VECTOR_FIELD,
            "k": k,
        }]
        return
    payload["vector"] = {
        "value": embedding,
        "fields": SEARCH_VECTOR_FIELD,
        "k": k,
    }


def _run_search(query: str, k: int, embedding: list[float] | None = None) -> list[dict]:
    url = f"{SEARCH_ENDPOINT}/indexes/{SEARCH_INDEX}/docs/search?api-version={SEARCH_API_VERSION}"
    headers = {"api-key": SEARCH_KEY, "Content-Type":"application/json"}
    payload = {
        "search": query,
        "queryType": "simple",
        "searchMode": "any",
        "top": k,
    }
    if SEARCH_FIELDS:
        payload["searchFields"] = ",".join(SEARCH_FIELDS)
    if embedding is not None and SEARCH_VECTOR_FIELD:
        _append_vector_query(payload, embedding, k)
        payload["search"] = "*"
    with httpx.Client(timeout=20) as c:
        r = c.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
    hits = []
    for d in data.get("value", []):
        chunk_text = d.get("chunk") or d.get("content") or d.get("description") or ""
        hits.append({
            "title": d.get("title", "Untitled"),
            "chunk": chunk_text[:800],
            "url": d.get("url", ""),
        })
    return hits


def search_top_k(query:str, k:int=5):
    """Fetch the top K documents from Azure Cognitive Search."""
    if not (SEARCH_ENDPOINT and SEARCH_INDEX and SEARCH_KEY):
        return []
    try:
        embedding = _get_embedding(query)
        hits = _run_search(query, k, embedding)
        if hits:
            return hits
        keywords = _extract_keywords(query)
        if keywords:
            fallback_query = " ".join(keywords[:3])
            if fallback_query and fallback_query != query:
                hits = _run_search(fallback_query, k)
    except httpx.HTTPStatusError as e:
        detail = e.response.text
        print(f"Azure Search request failed: {e}; detail={detail}")
        return []
    except httpx.RequestError as e:
        # If search is temporarily unavailable or authentication fails, fall back to empty context.
        print(f"Azure Search request failed: {e}")
        return []
    return hits

SYSTEM = (
    "You are a warm Netflix recommender. "
    "Whenever the user asks for concrete recommendations, choose titles strictly from CONTEXT. "
    "If CONTEXT lacks relevant options, politely ask the user for more details. "
    "For greetings, follow-up questions, or extra commentary, feel free to speak naturally."
)

def build_user_msg(q:str, passages:list[dict]):
    if not passages:
        ctx = "No external context."
    else:
        ctx = "\n".join([f"- {p['title']}: {p['chunk']}" for p in passages])
    return (
        f"USER QUESTION:\n{q}\n\n"
        f"CONTEXT:\n{ctx}\n"
        "If you recommend titles, make sure they appear in CONTEXT."
        "You may add brief friendly commentary beyond the list."
    )

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat")
def chat(req: ChatRequest):
    q = req.question.strip()
    if not q: raise HTTPException(400, "Empty question.")
    sid = req.session_id or str(uuid.uuid4())
    history = SESSION_HISTORY.setdefault(sid, deque(maxlen=MAX_HISTORY_MESSAGES))
    passages = search_top_k(q, k=5)
    user_msg = build_user_msg(q, passages)

    url = f"{AOAI_ENDPOINT}/openai/deployments/{AOAI_DEPLOYMENT}/chat/completions?api-version={AOAI_API_VERSION}"
    headers = {"api-key": AOAI_KEY, "Content-Type": "application/json"}
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM},
            *history,
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.3,
        "max_tokens": 600,
    }

    try:
        with httpx.Client(timeout=60) as c:
            r = c.post(url, headers=headers, json=payload)
            # Bubble up Azure OpenAI 4xx/5xx responses for easier debugging.
            r.raise_for_status()
            data = r.json()
        answer = data["choices"][0]["message"]["content"]
    except httpx.HTTPStatusError as e:
        # Return the raw Azure response to aid troubleshooting.
        status = e.response.status_code
        detail = e.response.text
        raise HTTPException(status, detail)
    except httpx.RequestError as e:
        raise HTTPException(502, f"Unable to reach Azure OpenAI: {e}") from e
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(500, f"Internal error: {e}")

    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": answer})
    return {
        "session_id": sid,
        "answer": answer,
        "sources": []
    }
