"""
指标收集器
统一收集和管理系统各项指标
"""

import time
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict, deque
import threading
import numpy as np
import pandas as pd
import sqlite3
from contextlib import contextmanager
import sys

sys.path.append("/Users/pluviophile/chi2025")
from team_c.config import EvaluationConfig

logger = logging.getLogger(__name__)

@dataclass
class MetricPoint:
    """指标数据点"""
    timestamp: float
    metric_name: str
    value: float
    labels: Dict[str, str]
    source: str

@dataclass
class AggregatedMetric:
    """聚合指标"""
    metric_name: str
    time_window: str
    count: int
    mean: float
    median: float
    min_value: float
    max_value: float
    std: float
    p95: float
    p99: float

class MetricsCollector:
    """
    指标收集器
    负责收集、存储和分析系统运行指标
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        
        # 内存缓存
        self.metric_buffer = deque(maxlen=10000)
        self.metric_cache = defaultdict(list)
        
        # 数据库路径
        self.db_path = config.results_dir / "metrics.db"
        
        # 线程安全
        self.lock = threading.Lock()
        
        # 预定义指标类型
        self.metric_types = {
            # 性能指标
            'response_time': {'unit': 'ms', 'type': 'histogram'},
            'throughput': {'unit': 'req/s', 'type': 'gauge'},
            'error_rate': {'unit': '%', 'type': 'gauge'},
            'cpu_usage': {'unit': '%', 'type': 'gauge'},
            'memory_usage': {'unit': 'MB', 'type': 'gauge'},
            
            # 准确率指标
            'overall_accuracy': {'unit': 'ratio', 'type': 'gauge'},
            'concept_accuracy': {'unit': 'ratio', 'type': 'gauge'},
            'classification_accuracy': {'unit': 'ratio', 'type': 'gauge'},
            
            # 用户体验指标
            'user_satisfaction': {'unit': 'score', 'type': 'gauge'},
            'session_duration': {'unit': 's', 'type': 'histogram'},
            'task_completion_rate': {'unit': '%', 'type': 'gauge'},
            
            # 系统健康指标
            'service_availability': {'unit': '%', 'type': 'gauge'},
            'request_queue_size': {'unit': 'count', 'type': 'gauge'},
            'database_connections': {'unit': 'count', 'type': 'gauge'}
        }
        
        # 初始化数据库
        self._init_database()
        
        logger.info("指标收集器初始化完成")
    
    def _init_database(self):
        """初始化SQLite数据库"""
        with self._get_db_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    metric_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    labels TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_metrics_name_time 
                ON metrics(metric_name, timestamp)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_metrics_source_time 
                ON metrics(source, timestamp)
            ''')
        
        logger.info(f"指标数据库初始化完成: {self.db_path}")
    
    @contextmanager
    def _get_db_connection(self):
        """获取数据库连接"""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def record_metric(self,
                     metric_name: str,
                     value: float,
                     labels: Optional[Dict[str, str]] = None,
                     source: str = 'unknown',
                     timestamp: Optional[float] = None):
        """
        记录单个指标
        
        Args:
            metric_name: 指标名称
            value: 指标值
            labels: 标签（用于分组和过滤）
            source: 数据源
            timestamp: 时间戳，默认为当前时间
        """
        if timestamp is None:
            timestamp = time.time()
        
        if labels is None:
            labels = {}
        
        metric_point = MetricPoint(
            timestamp=timestamp,
            metric_name=metric_name,
            value=value,
            labels=labels,
            source=source
        )
        
        with self.lock:
            # 添加到内存缓存
            self.metric_buffer.append(metric_point)
            self.metric_cache[metric_name].append(metric_point)
            
            # 限制内存缓存大小
            if len(self.metric_cache[metric_name]) > 1000:
                self.metric_cache[metric_name] = self.metric_cache[metric_name][-500:]
        
        logger.debug(f"记录指标: {metric_name}={value}, 来源={source}")
    
    def record_batch_metrics(self, metrics_batch: List[Dict[str, Any]]):
        """
        批量记录指标
        
        Args:
            metrics_batch: 指标批次，每个元素包含metric_name, value, labels, source等字段
        """
        timestamp = time.time()
        
        with self.lock:
            for metric_data in metrics_batch:
                metric_point = MetricPoint(
                    timestamp=metric_data.get('timestamp', timestamp),
                    metric_name=metric_data['metric_name'],
                    value=metric_data['value'],
                    labels=metric_data.get('labels', {}),
                    source=metric_data.get('source', 'batch')
                )
                
                self.metric_buffer.append(metric_point)
                self.metric_cache[metric_point.metric_name].append(metric_point)
        
        logger.debug(f"批量记录 {len(metrics_batch)} 个指标")
    
    def flush_to_database(self):
        """将缓存的指标刷新到数据库"""
        with self.lock:
            if not self.metric_buffer:
                return
            
            metrics_to_flush = list(self.metric_buffer)
            self.metric_buffer.clear()
        
        # 批量插入数据库
        with self._get_db_connection() as conn:
            metrics_data = [
                (
                    metric.timestamp,
                    metric.metric_name,
                    metric.value,
                    json.dumps(metric.labels),
                    metric.source
                )
                for metric in metrics_to_flush
            ]
            
            conn.executemany('''
                INSERT INTO metrics (timestamp, metric_name, value, labels, source)
                VALUES (?, ?, ?, ?, ?)
            ''', metrics_data)
        
        logger.info(f"刷新 {len(metrics_to_flush)} 个指标到数据库")
    
    def query_metrics(self,
                     metric_name: str,
                     start_time: Optional[float] = None,
                     end_time: Optional[float] = None,
                     labels: Optional[Dict[str, str]] = None,
                     source: Optional[str] = None,
                     limit: int = 1000) -> List[MetricPoint]:
        """
        查询指标数据
        
        Args:
            metric_name: 指标名称
            start_time: 开始时间戳
            end_time: 结束时间戳
            labels: 标签过滤条件
            source: 数据源过滤
            limit: 结果数量限制
            
        Returns:
            指标数据列表
        """
        # 构建查询条件
        where_conditions = ["metric_name = ?"]
        params = [metric_name]
        
        if start_time is not None:
            where_conditions.append("timestamp >= ?")
            params.append(start_time)
        
        if end_time is not None:
            where_conditions.append("timestamp <= ?")
            params.append(end_time)
        
        if source is not None:
            where_conditions.append("source = ?")
            params.append(source)
        
        query = f'''
            SELECT timestamp, metric_name, value, labels, source
            FROM metrics
            WHERE {" AND ".join(where_conditions)}
            ORDER BY timestamp DESC
            LIMIT ?
        '''
        params.append(limit)
        
        results = []
        with self._get_db_connection() as conn:
            cursor = conn.execute(query, params)
            
            for row in cursor:
                timestamp, name, value, labels_json, source = row
                
                try:
                    parsed_labels = json.loads(labels_json)
                except:
                    parsed_labels = {}
                
                # 标签过滤
                if labels:
                    if not all(parsed_labels.get(k) == v for k, v in labels.items()):
                        continue
                
                metric_point = MetricPoint(
                    timestamp=timestamp,
                    metric_name=name,
                    value=value,
                    labels=parsed_labels,
                    source=source
                )
                results.append(metric_point)
        
        return results
    
    def aggregate_metrics(self,
                         metric_name: str,
                         time_window: int = 3600,  # 1小时
                         start_time: Optional[float] = None,
                         end_time: Optional[float] = None) -> AggregatedMetric:
        """
        聚合指标数据
        
        Args:
            metric_name: 指标名称
            time_window: 时间窗口（秒）
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            聚合指标结果
        """
        if end_time is None:
            end_time = time.time()
        
        if start_time is None:
            start_time = end_time - time_window
        
        # 查询原始数据
        metrics = self.query_metrics(
            metric_name=metric_name,
            start_time=start_time,
            end_time=end_time,
            limit=10000
        )
        
        if not metrics:
            return AggregatedMetric(
                metric_name=metric_name,
                time_window=f"{time_window}s",
                count=0,
                mean=0.0,
                median=0.0,
                min_value=0.0,
                max_value=0.0,
                std=0.0,
                p95=0.0,
                p99=0.0
            )
        
        values = [m.value for m in metrics]
        
        return AggregatedMetric(
            metric_name=metric_name,
            time_window=f"{time_window}s",
            count=len(values),
            mean=float(np.mean(values)),
            median=float(np.median(values)),
            min_value=float(np.min(values)),
            max_value=float(np.max(values)),
            std=float(np.std(values)),
            p95=float(np.percentile(values, 95)),
            p99=float(np.percentile(values, 99))
        )
    
    def get_metric_summary(self, hours_back: int = 24) -> Dict[str, AggregatedMetric]:
        """获取指标汇总"""
        end_time = time.time()
        start_time = end_time - (hours_back * 3600)
        
        # 获取所有指标名称
        with self._get_db_connection() as conn:
            cursor = conn.execute('''
                SELECT DISTINCT metric_name 
                FROM metrics 
                WHERE timestamp >= ?
            ''', [start_time])
            
            metric_names = [row[0] for row in cursor]
        
        summary = {}
        for metric_name in metric_names:
            try:
                aggregated = self.aggregate_metrics(
                    metric_name=metric_name,
                    time_window=hours_back * 3600,
                    start_time=start_time,
                    end_time=end_time
                )
                summary[metric_name] = aggregated
            except Exception as e:
                logger.error(f"聚合指标 {metric_name} 失败: {e}")
        
        return summary
    
    def record_performance_metrics(self,
                                 response_time: float,
                                 endpoint: str,
                                 status_code: int,
                                 user_id: Optional[str] = None):
        """记录性能指标"""
        labels = {
            'endpoint': endpoint,
            'status_code': str(status_code)
        }
        
        if user_id:
            labels['user_id'] = user_id
        
        # 记录响应时间
        self.record_metric(
            metric_name='response_time',
            value=response_time,
            labels=labels,
            source='api'
        )
        
        # 记录成功/失败
        success = 1.0 if 200 <= status_code < 400 else 0.0
        self.record_metric(
            metric_name='request_success',
            value=success,
            labels=labels,
            source='api'
        )
    
    def record_accuracy_metrics(self,
                              overall_accuracy: float,
                              concept_accuracies: Dict[str, float],
                              test_id: Optional[str] = None):
        """记录准确率指标"""
        labels = {}
        if test_id:
            labels['test_id'] = test_id
        
        # 整体准确率
        self.record_metric(
            metric_name='overall_accuracy',
            value=overall_accuracy,
            labels=labels,
            source='evaluation'
        )
        
        # 各概念准确率
        for concept, accuracy in concept_accuracies.items():
            concept_labels = labels.copy()
            concept_labels['concept'] = concept
            
            self.record_metric(
                metric_name='concept_accuracy',
                value=accuracy,
                labels=concept_labels,
                source='evaluation'
            )
    
    def record_user_experience_metrics(self,
                                     user_id: str,
                                     session_duration: float,
                                     satisfaction_score: Optional[float] = None,
                                     task_completed: bool = True):
        """记录用户体验指标"""
        labels = {'user_id': user_id}
        
        # 会话时长
        self.record_metric(
            metric_name='session_duration',
            value=session_duration,
            labels=labels,
            source='user_experience'
        )
        
        # 用户满意度
        if satisfaction_score is not None:
            self.record_metric(
                metric_name='user_satisfaction',
                value=satisfaction_score,
                labels=labels,
                source='user_experience'
            )
        
        # 任务完成率
        completion = 1.0 if task_completed else 0.0
        self.record_metric(
            metric_name='task_completion',
            value=completion,
            labels=labels,
            source='user_experience'
        )
    
    def export_metrics(self,
                      start_time: Optional[float] = None,
                      end_time: Optional[float] = None,
                      format: str = 'csv') -> Path:
        """
        导出指标数据
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            format: 导出格式 ('csv', 'json')
            
        Returns:
            导出文件路径
        """
        if end_time is None:
            end_time = time.time()
        
        if start_time is None:
            start_time = end_time - (24 * 3600)  # 默认24小时
        
        # 查询数据
        query = '''
            SELECT timestamp, metric_name, value, labels, source
            FROM metrics
            WHERE timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp
        '''
        
        with self._get_db_connection() as conn:
            df = pd.read_sql_query(query, conn, params=[start_time, end_time])
        
        # 导出文件
        timestamp_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        if format == 'csv':
            filepath = self.config.results_dir / f"metrics_export_{timestamp_str}.csv"
            df.to_csv(filepath, index=False)
        elif format == 'json':
            filepath = self.config.results_dir / f"metrics_export_{timestamp_str}.json"
            df.to_json(filepath, orient='records', indent=2)
        else:
            raise ValueError(f"不支持的导出格式: {format}")
        
        logger.info(f"指标数据导出完成: {filepath}")
        return filepath
    
    def generate_metrics_dashboard_data(self) -> Dict[str, Any]:
        """生成指标仪表板数据"""
        summary = self.get_metric_summary(hours_back=1)  # 最近1小时
        
        dashboard_data = {
            'timestamp': time.time(),
            'overview': {},
            'performance': {},
            'accuracy': {},
            'user_experience': {},
            'system_health': {}
        }
        
        # 性能概览
        if 'response_time' in summary:
            rt_summary = summary['response_time']
            dashboard_data['performance']['response_time'] = {
                'mean': rt_summary.mean,
                'p95': rt_summary.p95,
                'p99': rt_summary.p99,
                'target_met': rt_summary.mean <= self.config.latency_target * 1000
            }
        
        # 准确率概览
        if 'overall_accuracy' in summary:
            acc_summary = summary['overall_accuracy']
            dashboard_data['accuracy']['overall'] = {
                'current': acc_summary.mean,
                'target': self.config.accuracy_threshold,
                'target_met': acc_summary.mean >= self.config.accuracy_threshold
            }
        
        # 用户体验概览
        if 'user_satisfaction' in summary:
            satisfaction_summary = summary['user_satisfaction']
            dashboard_data['user_experience']['satisfaction'] = {
                'mean': satisfaction_summary.mean,
                'count': satisfaction_summary.count
            }
        
        # 系统健康概览
        success_rate = 0.0
        if 'request_success' in summary:
            success_summary = summary['request_success']
            success_rate = success_summary.mean * 100
        
        dashboard_data['system_health']['availability'] = {
            'success_rate': success_rate,
            'target': 99.0,
            'status': 'healthy' if success_rate >= 99.0 else 'warning' if success_rate >= 95.0 else 'critical'
        }
        
        return dashboard_data
    
    def cleanup_old_metrics(self, days_to_keep: int = 30):
        """清理旧的指标数据"""
        cutoff_time = time.time() - (days_to_keep * 24 * 3600)
        
        with self._get_db_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM metrics WHERE timestamp < ?",
                [cutoff_time]
            )
            deleted_count = cursor.rowcount
        
        logger.info(f"清理了 {deleted_count} 条旧指标数据（保留 {days_to_keep} 天）")
    
    def __del__(self):
        """析构时确保数据已刷新"""
        try:
            self.flush_to_database()
        except:
            pass

def main():
    """测试指标收集器"""
    from team_c.config import get_evaluation_config
    
    config = get_evaluation_config()
    collector = MetricsCollector(config)
    
    # 模拟记录一些指标
    for i in range(50):
        # 性能指标
        collector.record_performance_metrics(
            response_time=np.random.normal(400, 100),
            endpoint='/api/cognitive-state',
            status_code=200 if np.random.random() > 0.05 else 500,
            user_id=f"user_{i % 10}"
        )
        
        # 准确率指标
        if i % 10 == 0:
            overall_acc = np.random.uniform(0.7, 0.9)
            concept_accs = {
                'attention': np.random.uniform(0.6, 0.9),
                'creativity': np.random.uniform(0.6, 0.9),
                'memory': np.random.uniform(0.6, 0.9)
            }
            
            collector.record_accuracy_metrics(overall_acc, concept_accs, f"test_{i}")
        
        # 用户体验指标
        if i % 5 == 0:
            collector.record_user_experience_metrics(
                user_id=f"user_{i % 10}",
                session_duration=np.random.normal(300, 100),
                satisfaction_score=np.random.uniform(3.0, 5.0),
                task_completed=np.random.random() > 0.1
            )
        
        time.sleep(0.01)  # 模拟时间间隔
    
    # 刷新到数据库
    collector.flush_to_database()
    
    # 获取汇总
    summary = collector.get_metric_summary(hours_back=1)
    print("指标汇总:")
    for metric_name, agg in summary.items():
        print(f"  {metric_name}: 平均={agg.mean:.2f}, P95={agg.p95:.2f}, 样本数={agg.count}")
    
    # 生成仪表板数据
    dashboard = collector.generate_metrics_dashboard_data()
    print(f"\n仪表板数据: {json.dumps(dashboard, indent=2)}")
    
    # 导出数据
    export_path = collector.export_metrics(format='csv')
    print(f"数据导出: {export_path}")

if __name__ == "__main__":
    main()