from datetime import datetime
import os
from typing import Dict, Any
from src.utils.logger import setup_logger

class RAGMonitor:
    def __init__(self, log_dir: str):
        # 确保日志目录存在
        try:
            os.makedirs(log_dir, exist_ok=True)
        except FileExistsError:
            pass  # 如果目录已存在则忽略
        self.logger = setup_logger('RAGMonitor', log_dir)
        # 初始化指标容器
        self.metrics = {}
        
    def start_monitoring(self):
        """初始化监控指标并记录启动时间
        
        监控指标说明:
        - query_count: 累计处理的查询请求数量
        - total_response_time: 所有查询的总响应时间(秒)
        - average_response_time: 平均响应时间(秒)
        - error_count: 累计发生的错误数量  
        - cache_hits: 缓存命中次数
        - start_time: 监控启动时间
        """
        self.metrics = {
            'query_count': 0,          # 查询计数器
            'total_response_time': 0.0, # 总响应时间(秒)
            'average_response_time': 0.0, # 平均响应时间(秒)
            'error_count': 0,          # 错误计数器
            'cache_hits': 0,           # 缓存命中数
            'start_time': datetime.now() # 监控启动时间
        }
        self.logger.info("Monitoring started with metrics initialized")

    def log_query(self, query: str, response_time: float):
        if not hasattr(self, 'metrics') or not self.metrics:
            self.start_monitoring()
            
        self.metrics['query_count'] += 1
        self.metrics['total_response_time'] += response_time
        self.metrics['average_response_time'] = round(
            self.metrics['total_response_time'] / self.metrics['query_count'], 
            4
        )
        self.logger.info(f"Query: {query}, Response Time: {response_time:.4f}s")

    def log_error(self, error: str):
        if not hasattr(self, 'metrics') or not self.metrics:
            self.start_monitoring()
        self.metrics['error_count'] += 1
        self.logger.error(f"Error: {error}")

    def log_cache_hit(self):
        if not hasattr(self, 'metrics') or not self.metrics:
            self.start_monitoring()
        self.metrics['cache_hits'] += 1

    def get_metrics(self) -> Dict[str, Any]:
        if not hasattr(self, 'metrics') or not self.metrics:
            self.start_monitoring()
        metrics = self.metrics.copy()
        # 计算运行时间
        metrics['uptime'] = str(datetime.now() - metrics['start_time'])
        return metrics
