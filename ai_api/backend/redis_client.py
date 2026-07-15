"""
Redis connection pool + distributed rate limiting.
Thay thế slowapi in-memory bằng sliding-window counter trên Redis.
"""
import os
import time
from typing import Optional

from dotenv import load_dotenv
import redis.asyncio as aioredis

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
REDIS_MAX_CONNECTIONS = int(os.getenv("REDIS_MAX_CONNECTIONS", "20"))

_redis_pool: Optional[aioredis.Redis] = None

# Sliding window rate limit — Lua script (atomic trên Redis)
_RATE_LIMIT_SCRIPT = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
local count = redis.call('ZCARD', key)
if count >= limit then
    return 0
end
redis.call('ZADD', key, now, now .. ':' .. math.random(1000000))
redis.call('EXPIRE', key, window)
return 1
"""


async def get_redis() -> aioredis.Redis:
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = aioredis.from_url(
            REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            max_connections=REDIS_MAX_CONNECTIONS,
            socket_connect_timeout=5,
            socket_keepalive=True,
            health_check_interval=30,
        )
    return _redis_pool


async def close_redis() -> None:
    global _redis_pool
    if _redis_pool is not None:
        await _redis_pool.aclose()
        _redis_pool = None


async def check_rate_limit(
    identifier: str,
    limit: int = 60,
    window_seconds: int = 60,
    prefix: str = "ratelimit",
) -> tuple[bool, int]:
    """
    Kiểm tra rate limit theo identifier (mặc định user_id).
    Returns (allowed, remaining_approx).
    """
    redis_conn = await get_redis()
    key = f"{prefix}:{identifier}"
    now = time.time()
    allowed = await redis_conn.eval(
        _RATE_LIMIT_SCRIPT, 1, key, now, window_seconds, limit
    )
    if allowed:
        current = await redis_conn.zcard(key)
        remaining = max(0, limit - current)
        return True, remaining
    return False, 0


async def redis_health_check() -> bool:
    try:
        redis_conn = await get_redis()
        return await redis_conn.ping()
    except Exception:
        return False
