"""
性能监控系统
Prometheus集成和实时监控告警
"""

import asyncio
import logging
import time
import psutil
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from prometheus_client import Counter, Histogram, Gauge, Summary, Info
from prometheus_client import start_http_server as start_prometheus_server

from ..storage.mongodb_service import CognitiveStateStorage
from ..cognitive_service import CognitiveAnalysisService

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """告警严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AlertRule:
    """告警规则"""
    name: str
    description: str
    metric_name: str
    threshold: float
    operator: str  # >, <, >=, <=, ==
    severity: AlertSeverity
    duration: int = 60  # 持续时间（秒）
    callback: Optional[Callable] = None
    enabled: bool = True
    triggered_count: int = 0
    last_triggered: Optional[datetime] = None


@dataclass
class Alert:
    """告警实例"""
    id: str
    rule_name: str
    severity: AlertSeverity
    message: str
    metric_value: float
    threshold: float
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class PrometheusMetrics:
    """Prometheus指标定义"""
    
    def __init__(self):
        # API指标
        self.api_requests_total = Counter(
            'cognitive_api_requests_total',
            'Total API requests',
            ['method', 'endpoint', 'status_code']
        )
        
        self.api_request_duration = Histogram(
            'cognitive_api_request_duration_seconds',
            'API request duration in seconds',
            ['method', 'endpoint']
        )
        
        # 认知分析指标
        self.cognitive_analysis_total = Counter(
            'cognitive_analysis_total',
            'Total cognitive analyses performed',
            ['status']
        )
        
        self.cognitive_processing_time = Histogram(
            'cognitive_processing_duration_seconds',
            'Cognitive analysis processing time',
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        )
        
        self.difcm_prediction_accuracy = Gauge(
            'difcm_prediction_accuracy',
            'DIFCM model prediction accuracy'
        )
        
        # 系统资源指标
        self.system_memory_usage = Gauge(
            'system_memory_usage_bytes',
            'System memory usage in bytes'
        )
        
        self.system_cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage'
        )
        
        self.active_connections = Gauge(
            'active_database_connections',
            'Number of active database connections'
        )
        
        # 数据存储指标
        self.cognitive_states_stored = Counter(
            'cognitive_states_stored_total',
            'Total cognitive states stored in database'
        )
        
        self.database_query_duration = Histogram(
            'database_query_duration_seconds',
            'Database query duration',
            ['operation', 'collection']
        )
        
        # 队列指标
        self.processing_queue_size = Gauge(
            'processing_queue_size',
            'Number of items in processing queue'
        )
        
        # 服务健康指标
        self.service_up = Gauge(
            'service_up',
            'Service availability',
            ['service_name']
        )
        
        # 业务指标
        self.active_user_sessions = Gauge(
            'active_user_sessions',
            'Number of active user sessions'
        )
        
        self.cognitive_state_distribution = Histogram(
            'cognitive_state_values',
            'Distribution of cognitive state values',
            ['concept'],
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )


class PerformanceMonitor:
    """
    性能监控器
    负责收集系统指标和触发告警
    """
    
    def __init__(self, 
                 cognitive_service: CognitiveAnalysisService,
                 storage_service: CognitiveStateStorage,
                 prometheus_port: int = 9090):
        self.cognitive_service = cognitive_service
        self.storage_service = storage_service
        self.prometheus_port = prometheus_port
        
        # Prometheus指标
        self.metrics = PrometheusMetrics()
        
        # 告警系统
        self.alert_rules: List[AlertRule] = []
        self.active_alerts: List[Alert] = []
        self.alert_history: List[Alert] = []
        
        # 监控状态
        self.monitoring_active = False
        self.start_time = time.time()
        
        # 性能历史
        self.performance_history = {
            'cpu_usage': [],
            'memory_usage': [],
            'response_times': [],
            'error_rates': []
        }
        
        self._setup_default_alert_rules()
        logger.info("性能监控器初始化完成")
    
    def _setup_default_alert_rules(self):
        """设置默认告警规则"""
        default_rules = [
            AlertRule(
                name="high_response_time",
                description="API响应时间过高",
                metric_name="api_response_time",
                threshold=5.0,
                operator=">",
                severity=AlertSeverity.HIGH,
                duration=120
            ),
            AlertRule(
                name="high_error_rate",
                description="错误率过高",
                metric_name="error_rate",
                threshold=0.1,
                operator=">",
                severity=AlertSeverity.CRITICAL,
                duration=60
            ),
            AlertRule(
                name="high_memory_usage",
                description="内存使用率过高",
                metric_name="memory_usage",
                threshold=0.85,
                operator=">",
                severity=AlertSeverity.HIGH,
                duration=300
            ),
            AlertRule(
                name="high_cpu_usage",
                description="CPU使用率过高",
                metric_name="cpu_usage",
                threshold=0.80,
                operator=">",
                severity=AlertSeverity.MEDIUM,
                duration=180
            ),
            AlertRule(
                name="difcm_low_accuracy",
                description="DIFCM模型准确率过低",
                metric_name="difcm_accuracy",
                threshold=0.75,
                operator="<",
                severity=AlertSeverity.HIGH,
                duration=600
            ),
            AlertRule(
                name="database_disconnected",
                description="数据库连接断开",
                metric_name="database_connected",
                threshold=1,
                operator="<",
                severity=AlertSeverity.CRITICAL,
                duration=30
            )
        ]
        
        self.alert_rules.extend(default_rules)
    
    async def start_monitoring(self):
        """启动监控"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        # 启动Prometheus HTTP服务器
        try:
            start_prometheus_server(self.prometheus_port)
            logger.info(f"Prometheus指标服务器启动在端口 {self.prometheus_port}")
        except Exception as e:
            logger.error(f"Prometheus服务器启动失败: {e}")
        
        # 启动监控任务
        asyncio.create_task(self._monitoring_loop())
        logger.info("性能监控开始运行")
    
    async def stop_monitoring(self):
        """停止监控"""
        self.monitoring_active = False
        logger.info("性能监控已停止")
    
    async def _monitoring_loop(self):
        """监控主循环"""
        while self.monitoring_active:
            try:
                # 收集系统指标
                await self._collect_system_metrics()
                
                # 收集服务指标
                await self._collect_service_metrics()
                
                # 检查告警
                await self._check_alerts()
                
                # 清理过期数据
                await self._cleanup_expired_data()
                
                # 等待下一次检查
                await asyncio.sleep(30)  # 30秒间隔
                
            except Exception as e:
                logger.error(f"监控循环异常: {e}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(10)
    
    async def _collect_system_metrics(self):
        """收集系统资源指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics.system_cpu_usage.set(cpu_percent)
            self.performance_history['cpu_usage'].append({
                'timestamp': time.time(),
                'value': cpu_percent
            })
            
            # 内存使用
            memory = psutil.virtual_memory()
            self.metrics.system_memory_usage.set(memory.used)
            memory_percent = memory.percent
            self.performance_history['memory_usage'].append({
                'timestamp': time.time(),
                'value': memory_percent
            })
            
            # 保持历史数据在合理范围内
            max_history = 1000
            for key in self.performance_history:
                if len(self.performance_history[key]) > max_history:
                    self.performance_history[key] = self.performance_history[key][-max_history:]
            
        except Exception as e:
            logger.error(f"收集系统指标失败: {e}")
    
    async def _collect_service_metrics(self):
        """收集服务相关指标"""
        try:
            # 认知服务状态
            if self.cognitive_service.is_ready():
                self.metrics.service_up.labels(service_name="cognitive_service").set(1)
                
                # 获取认知服务性能指标
                perf_metrics = self.cognitive_service.get_performance_metrics()
                if perf_metrics['total_requests'] > 0:
                    error_rate = perf_metrics['error_rate']
                    self.performance_history['error_rates'].append({
                        'timestamp': time.time(),
                        'value': error_rate
                    })
            else:
                self.metrics.service_up.labels(service_name="cognitive_service").set(0)
            
            # 存储服务状态
            if self.storage_service.is_connected():
                self.metrics.service_up.labels(service_name="storage_service").set(1)
                
                # 获取数据库统计
                stats = await self.storage_service.get_statistics()
                if stats:
                    cognitive_states_count = stats.get('cognitive_states', 0)
                    self.metrics.cognitive_states_stored._value._value = cognitive_states_count
            else:
                self.metrics.service_up.labels(service_name="storage_service").set(0)
            
        except Exception as e:
            logger.error(f"收集服务指标失败: {e}")
    
    async def _check_alerts(self):
        """检查告警条件"""
        current_metrics = await self._get_current_metrics()
        
        for rule in self.alert_rules:
            if not rule.enabled:
                continue
            
            try:
                metric_value = current_metrics.get(rule.metric_name)
                if metric_value is None:
                    continue
                
                # 检查阈值
                triggered = self._evaluate_alert_condition(
                    metric_value, rule.threshold, rule.operator
                )
                
                if triggered:
                    await self._handle_alert_triggered(rule, metric_value)
                else:
                    await self._handle_alert_resolved(rule)
                    
            except Exception as e:
                logger.error(f"检查告警规则 {rule.name} 失败: {e}")
    
    def _evaluate_alert_condition(self, value: float, threshold: float, operator: str) -> bool:
        """评估告警条件"""
        if operator == ">":
            return value > threshold
        elif operator == "<":
            return value < threshold
        elif operator == ">=":
            return value >= threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == "==":
            return value == threshold
        else:
            return False
    
    async def _handle_alert_triggered(self, rule: AlertRule, metric_value: float):
        """处理告警触发"""
        # 检查是否已有活跃告警
        existing_alert = next(
            (alert for alert in self.active_alerts 
             if alert.rule_name == rule.name and not alert.resolved),
            None
        )
        
        if existing_alert:
            return  # 已有活跃告警，不重复创建
        
        # 创建新告警
        alert = Alert(
            id=f"{rule.name}_{int(time.time())}",
            rule_name=rule.name,
            severity=rule.severity,
            message=f"{rule.description}: {metric_value:.3f} (阈值: {rule.threshold})",
            metric_value=metric_value,
            threshold=rule.threshold,
            timestamp=datetime.now()
        )
        
        self.active_alerts.append(alert)
        self.alert_history.append(alert)
        
        rule.triggered_count += 1
        rule.last_triggered = datetime.now()
        
        logger.warning(f"🚨 告警触发: {alert.message}")
        
        # 执行回调
        if rule.callback:
            try:
                await rule.callback(alert)
            except Exception as e:
                logger.error(f"告警回调执行失败: {e}")
    
    async def _handle_alert_resolved(self, rule: AlertRule):
        """处理告警解决"""
        # 查找并解决活跃告警
        for alert in self.active_alerts:
            if alert.rule_name == rule.name and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                logger.info(f"✅ 告警已解决: {alert.message}")
                break
    
    async def _get_current_metrics(self) -> Dict[str, float]:
        """获取当前指标值"""
        metrics = {}
        
        try:
            # 系统指标
            metrics['cpu_usage'] = psutil.cpu_percent() / 100.0
            metrics['memory_usage'] = psutil.virtual_memory().percent / 100.0
            
            # 服务指标
            if self.cognitive_service.is_ready():
                perf = self.cognitive_service.get_performance_metrics()
                metrics['api_response_time'] = perf.get('avg_processing_time', 0)
                metrics['error_rate'] = perf.get('error_rate', 0)
                metrics['difcm_accuracy'] = 0.85  # TODO: 实现真实的准确率计算
            
            # 数据库连接状态
            metrics['database_connected'] = 1 if self.storage_service.is_connected() else 0
            
        except Exception as e:
            logger.error(f"获取当前指标失败: {e}")
        
        return metrics
    
    async def _cleanup_expired_data(self):
        """清理过期数据"""
        try:
            # 清理告警历史（保留7天）
            cutoff_time = datetime.now() - timedelta(days=7)
            self.alert_history = [
                alert for alert in self.alert_history
                if alert.timestamp > cutoff_time
            ]
            
            # 清理已解决的活跃告警（保留1小时）
            cutoff_time = datetime.now() - timedelta(hours=1)
            self.active_alerts = [
                alert for alert in self.active_alerts
                if not alert.resolved or alert.resolved_at > cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"清理过期数据失败: {e}")
    
    def record_api_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """记录API请求指标"""
        self.metrics.api_requests_total.labels(
            method=method, endpoint=endpoint, status_code=str(status_code)
        ).inc()
        
        self.metrics.api_request_duration.labels(
            method=method, endpoint=endpoint
        ).observe(duration)
    
    def record_cognitive_analysis(self, processing_time: float, success: bool):
        """记录认知分析指标"""
        status = "success" if success else "failure"
        self.metrics.cognitive_analysis_total.labels(status=status).inc()
        self.metrics.cognitive_processing_time.observe(processing_time)
    
    def record_cognitive_state(self, cognitive_state: Dict[str, float]):
        """记录认知状态分布"""
        for concept, value in cognitive_state.items():
            self.metrics.cognitive_state_distribution.labels(concept=concept).observe(value)
    
    def get_performance_summary(self) -> Dict:
        """获取性能摘要"""
        uptime = time.time() - self.start_time
        
        # 计算平均值
        recent_cpu = [m['value'] for m in self.performance_history['cpu_usage'][-60:]]
        recent_memory = [m['value'] for m in self.performance_history['memory_usage'][-60:]]
        recent_errors = [m['value'] for m in self.performance_history['error_rates'][-60:]]
        
        return {
            'uptime_seconds': uptime,
            'avg_cpu_usage': np.mean(recent_cpu) if recent_cpu else 0,
            'avg_memory_usage': np.mean(recent_memory) if recent_memory else 0,
            'avg_error_rate': np.mean(recent_errors) if recent_errors else 0,
            'active_alerts_count': len([a for a in self.active_alerts if not a.resolved]),
            'total_alerts_triggered': sum(rule.triggered_count for rule in self.alert_rules),
            'service_status': {
                'cognitive_service': self.cognitive_service.is_ready(),
                'storage_service': self.storage_service.is_connected()
            }
        }
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        return [alert for alert in self.active_alerts if not alert.resolved]
    
    def add_alert_rule(self, rule: AlertRule):
        """添加自定义告警规则"""
        self.alert_rules.append(rule)
        logger.info(f"添加告警规则: {rule.name}")
    
    def disable_alert_rule(self, rule_name: str):
        """禁用告警规则"""
        for rule in self.alert_rules:
            if rule.name == rule_name:
                rule.enabled = False
                logger.info(f"禁用告警规则: {rule_name}")
                break