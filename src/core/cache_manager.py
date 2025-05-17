import redis
import json
import hashlib
import os
import pickle
from typing import Optional, Any
from pathlib import Path

class CacheManager:
    def __init__(self, config: dict):
        self.config = config
        self.use_redis = False
        self.cache_dir = Path(config['paths']['cache_dir'])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self.redis_client = redis.from_url(config['cache']['redis_url'])
            self.redis_client.ping()  # Test connection
            self.use_redis = True
        except (redis.ConnectionError, redis.TimeoutError):
            self.redis_client = None
            self.use_redis = False
            
        self.ttl = config['cache']['ttl']

    def _generate_key(self, query: str) -> str:
        """生成缓存键"""
        return f"rag:{hashlib.md5(query.encode()).hexdigest()}"

    def get(self, query: str) -> Optional[dict]:
        """获取缓存的结果"""
        key = self._generate_key(query)
        
        if self.use_redis:
            cached_result = self.redis_client.get(key)
            if cached_result:
                return json.loads(cached_result)
        else:
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        return None

    def set(self, query: str, result: Any):
        """设置缓存"""
        key = self._generate_key(query)
        if self.use_redis:
            self.redis_client.setex(
                key,
                self.ttl,
                json.dumps(result)
            )
        else:
            cache_file = self.cache_dir / f"{key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)

    def clear(self):
        """清除所有缓存"""
        if self.use_redis:
            for key in self.redis_client.scan_iter("rag:*"):
                self.redis_client.delete(key)
        else:
            for cache_file in self.cache_dir.glob("rag_*.pkl"):
                cache_file.unlink()
