"""
Semantic Cache trên Redis — trả lời ngay câu hỏi trùng/ngữ nghĩa gần.
Embedding local: sentence-transformers/all-MiniLM-L6-v2 (mặc định).
"""
import json
import math
import os
from typing import Optional

from dotenv import load_dotenv

from redis_client import get_redis

load_dotenv()

SIMILARITY_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.92"))
CACHE_TTL_SECONDS = int(os.getenv("SEMANTIC_CACHE_TTL", "86400"))  # 24h
MAX_ENTRIES_PER_SESSION = int(os.getenv("SEMANTIC_CACHE_MAX_ENTRIES", "50"))

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        model_name = os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        _embedder = SentenceTransformer(model_name)
    return _embedder


def embed_text(text: str) -> list[float]:
    model = _get_embedder()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _cache_key(session_id: str) -> str:
    return f"semantic_cache:{session_id}"


async def lookup_cache(session_id: str, query: str) -> Optional[str]:
    """
    Tìm câu trả lời đã cache theo cosine similarity.
    Returns answer string nếu hit, None nếu miss.
    """
    redis_conn = await get_redis()
    raw = await redis_conn.get(_cache_key(session_id))
    if not raw:
        return None

    entries = json.loads(raw)
    query_vec = embed_text(query)
    best_score = 0.0
    best_answer: Optional[str] = None

    for entry in entries:
        score = cosine_similarity(query_vec, entry["embedding"])
        if score >= SIMILARITY_THRESHOLD and score > best_score:
            best_score = score
            best_answer = entry["answer"]

    return best_answer


async def store_cache(session_id: str, query: str, answer: str) -> None:
    redis_conn = await get_redis()
    key = _cache_key(session_id)
    raw = await redis_conn.get(key)
    entries = json.loads(raw) if raw else []

    entries.append(
        {
            "query": query,
            "embedding": embed_text(query),
            "answer": answer,
        }
    )
    if len(entries) > MAX_ENTRIES_PER_SESSION:
        entries = entries[-MAX_ENTRIES_PER_SESSION:]

    await redis_conn.set(key, json.dumps(entries), ex=CACHE_TTL_SECONDS)


async def invalidate_session_cache(session_id: str) -> None:
    """Gọi khi user upload PDF mới — xóa cache session."""
    redis_conn = await get_redis()
    await redis_conn.delete(_cache_key(session_id))
