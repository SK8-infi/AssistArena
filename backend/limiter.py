"""Rate limiting mechanisms for AssistArena.

Provides in-memory thread-safe rate limiting (RateLimitBucket) and atomic Redis-backed
distributed rate limiting (RedisRateLimitBucket) using a token-bucket algorithm.
"""

import logging
import os
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import redis

logger = logging.getLogger("assistarena.limiter")

RATE_LIMIT_CEILING = 20
_PRUNE_LIMIT = 1024


class RateLimitBucket:
    """Thread-safe in-memory rate-limiter bucket using monotonic time checks."""

    def __init__(
        self,
        capacity: int,
        refill_window: float,
        prune_limit: int = _PRUNE_LIMIT,
    ) -> None:
        """Initialize bucket state details."""
        self.capacity = float(capacity)
        self._refill_rate = capacity / refill_window
        self._prune_limit = prune_limit
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def _prune_expired(self, now: float) -> None:
        """Prune fully refilled limit buckets (caller must hold lock)."""
        expired = [
            k
            for k, (tokens, last) in self._buckets.items()
            if tokens + (now - last) * self._refill_rate >= self.capacity
        ]
        for k in expired:
            del self._buckets[k]

    def acquire(self, client_key: str) -> bool:
        """Attempt to consume 1 token for a client key."""
        now = time.monotonic()
        with self._lock:
            if client_key not in self._buckets and len(self._buckets) >= self._prune_limit:
                self._prune_expired(now)
            tokens, last = self._buckets.get(client_key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self._refill_rate)
            if tokens < 1.0:
                self._buckets[client_key] = (tokens, now)
                return False
            self._buckets[client_key] = (tokens - 1.0, now)
            return True

    def flush(self) -> None:
        """Clear all in-memory limits."""
        with self._lock:
            self._buckets.clear()


class RedisRateLimitBucket:
    """Atomic distributed rate-limiter backed by Redis storage."""

    _SCRIPT = """
    local cap  = tonumber(ARGV[1])
    local rate = tonumber(ARGV[2])
    local now  = tonumber(ARGV[3])
    local ttl  = tonumber(ARGV[4])
    local b = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
    local tokens = tonumber(b[1])
    local ts = tonumber(b[2])
    if tokens == nil then tokens = cap; ts = now end
    local elapsed = now - ts
    if elapsed < 0 then elapsed = 0 end
    tokens = math.min(cap, tokens + elapsed * rate)
    local allowed = 0
    if tokens >= 1 then tokens = tokens - 1; allowed = 1 end
    redis.call('HMSET', KEYS[1], 'tokens', tokens, 'ts', now)
    redis.call('EXPIRE', KEYS[1], ttl)
    return allowed
    """

    def __init__(
        self,
        redis_client: "redis.Redis",
        capacity: int,
        refill_window: float,
        namespace: str = "assistarena:rl:",
    ) -> None:
        """Initialize Redis connection details."""
        self.capacity = float(capacity)
        self._refill_rate = capacity / refill_window
        self._ttl = int(refill_window) + 10
        self._client = redis_client
        self._namespace = namespace
        self._script = redis_client.register_script(self._SCRIPT)

    def acquire(self, client_key: str) -> bool:
        """Execute atomic Lua routine to test rate limit."""
        allowed = self._script(
            keys=[self._namespace + client_key],
            args=[self.capacity, self._refill_rate, time.time(), self._ttl],
        )
        return bool(int(allowed))

    def flush(self) -> None:
        """Remove keys corresponding to the limiter's namespace."""
        keys = list(self._client.scan_iter(match=self._namespace + "*"))
        if keys:
            self._client.delete(*keys)


def initialize_rate_limiter() -> RateLimitBucket | RedisRateLimitBucket:
    """Locate rate limiter configuration and instantiate backend."""
    redis_url = os.environ.get("REDIS_URL")
    if not redis_url:
        return RateLimitBucket(RATE_LIMIT_CEILING, 60.0)
    try:
        import redis  # noqa: PLC0415

        client = redis.Redis.from_url(redis_url, decode_responses=True)
        client.ping()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "REDIS_URL configured but connection failed (%s). "
            "Falling back to local in-memory rate limiting.",
            exc,
        )
        return RateLimitBucket(RATE_LIMIT_CEILING, 60.0)
    logger.info("Distributed Redis rate limiting connected.")
    return RedisRateLimitBucket(client, RATE_LIMIT_CEILING, 60.0)
