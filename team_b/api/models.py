"""
数据模型定义
与Team A的DIFCM模型和Team C的数据集完全兼容
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


class CognitiveConceptName(str, Enum):
    """写作认知概念枚举 - 基于权威教育心理学研究框架"""
    DEPTH_THINKING = "depth_thinking"         # 深刻性：主题本质挖掘
    FLEXIBLE_THINKING = "flexible_thinking"   # 灵活性：多角度思维  
    CRITICAL_THINKING = "critical_thinking"   # 批判性：错误识别与自评
    ORIGINALITY = "originality"               # 独创性：新颖独特程度
    FLUENCY = "fluency"                       # 流畅性：思维表达连贯性
    MOTIVATION = "motivation"                 # 动机水平：写作驱动力
    EMOTION_REGULATION = "emotion_regulation" # 情绪调节：情感状态管理
    COGNITIVE_LOAD = "cognitive_load"         # 认知负载：思维负担程度


class KeystrokeEvent(BaseModel):
    """击键事件 - 与Team A特征提取器输入格式一致"""
    key: str = Field(..., description="按键名称")
    timestamp: float = Field(..., description="时间戳")
    type: str = Field(default="keydown", description="事件类型")


class MouseEvent(BaseModel):
    """鼠标事件 - 与Team A特征提取器输入格式一致"""
    x: int = Field(..., ge=0, description="X坐标")
    y: int = Field(..., ge=0, description="Y坐标")
    timestamp: float = Field(..., description="时间戳")
    type: str = Field(default="mousemove", description="事件类型")


class TaskContext(BaseModel):
    """任务上下文"""
    scenario: Optional[str] = Field(None, description="场景类型")
    session_duration: float = Field(default=0.0, description="会话时长")
    task_type: str = Field(default="creative_writing", description="任务类型")


class UserBehaviorData(BaseModel):
    """用户行为数据 - 与Team A和Team C格式完全兼容"""
    user_id: str = Field(..., min_length=1, description="用户ID")
    session_id: str = Field(..., min_length=1, description="会话ID")
    timestamp: float = Field(..., description="数据时间戳")
    keystrokes: List[KeystrokeEvent] = Field(default_factory=list, description="击键数据")
    mouse_moves: List[MouseEvent] = Field(default_factory=list, description="鼠标移动数据")
    dwell_times: List[float] = Field(default_factory=list, description="停留时间")
    response_times: List[float] = Field(default_factory=list, description="响应时间")
    task_context: TaskContext = Field(default_factory=TaskContext, description="任务上下文")
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        if v <= 0:
            raise ValueError('时间戳必须为正数')
        return v
    
    @validator('keystrokes')
    def validate_keystrokes(cls, v):
        if len(v) > 10000:  # 防止超大数据
            raise ValueError('击键事件数量不能超过10000')
        return v


class CognitiveState(BaseModel):
    """写作认知状态 - 基于权威写作思维能力框架"""
    depth_thinking: float = Field(..., ge=0.0, le=1.0, description="深刻性：主题本质挖掘能力")
    flexible_thinking: float = Field(..., ge=0.0, le=1.0, description="灵活性：多角度辩证思维")
    critical_thinking: float = Field(..., ge=0.0, le=1.0, description="批判性：错误识别与自我评价")
    originality: float = Field(..., ge=0.0, le=1.0, description="独创性：新颖独特程度")
    fluency: float = Field(..., ge=0.0, le=1.0, description="流畅性：思维表达连贯性")
    motivation: float = Field(..., ge=0.0, le=1.0, description="动机水平：写作驱动力")
    emotion_regulation: float = Field(..., ge=0.0, le=1.0, description="情绪调节：情感状态管理")
    cognitive_load: float = Field(..., ge=0.0, le=1.0, description="认知负载：思维负担程度")
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典格式"""
        return {
            "attention": self.attention,
            "memory": self.memory,
            "comprehension": self.comprehension,
            "creativity": self.creativity,
            "motivation": self.motivation,
            "emotion": self.emotion,
            "confidence": self.confidence,
            "fatigue": self.fatigue
        }
    
    def to_vector(self) -> List[float]:
        """转换为向量格式 - 与Team A的DIFCM兼容"""
        return [
            self.attention,
            self.memory,
            self.comprehension,
            self.creativity,
            self.motivation,
            self.emotion,
            self.confidence,
            self.fatigue
        ]


class ProcessingStatus(str, Enum):
    """处理状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class CognitiveAnalysisResult(BaseModel):
    """认知分析结果"""
    request_id: str = Field(..., description="请求ID")
    user_id: str = Field(..., description="用户ID")
    session_id: str = Field(..., description="会话ID")
    timestamp: float = Field(..., description="分析时间戳")
    cognitive_state: CognitiveState = Field(..., description="认知状态")
    extracted_features: Dict[str, float] = Field(..., description="提取的特征")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="置信度")
    processing_time: float = Field(..., description="处理时间（秒）")
    status: ProcessingStatus = Field(default=ProcessingStatus.COMPLETED, description="处理状态")
    error_message: Optional[str] = Field(None, description="错误信息")


class DataReceiveResponse(BaseModel):
    """数据接收响应"""
    status: str = Field(..., description="处理状态")
    data_id: str = Field(..., description="数据ID")
    request_id: str = Field(..., description="请求ID")
    cognitive_state: Optional[CognitiveState] = Field(None, description="认知状态")
    message: str = Field(default="数据处理成功", description="状态消息")


class HealthCheckResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="服务状态")
    timestamp: datetime = Field(default_factory=datetime.now, description="检查时间")
    version: str = Field(default="1.0.0", description="服务版本")
    database_connected: bool = Field(..., description="数据库连接状态")
    difcm_model_loaded: bool = Field(..., description="DIFCM模型加载状态")
    feature_extractor_ready: bool = Field(..., description="特征提取器就绪状态")


class QueryFilters(BaseModel):
    """查询过滤器"""
    user_id: Optional[str] = Field(None, description="用户ID过滤")
    session_id: Optional[str] = Field(None, description="会话ID过滤")
    start_time: Optional[float] = Field(None, description="开始时间戳")
    end_time: Optional[float] = Field(None, description="结束时间戳")
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="最小置信度")
    status: Optional[ProcessingStatus] = Field(None, description="处理状态过滤")


class PaginationParams(BaseModel):
    """分页参数"""
    page: int = Field(default=1, ge=1, description="页码")
    page_size: int = Field(default=20, ge=1, le=100, description="每页大小")
    sort_by: str = Field(default="timestamp", description="排序字段")
    sort_order: str = Field(default="desc", pattern=r"^(asc|desc)$", description="排序方向")


class PaginatedResponse(BaseModel):
    """分页响应"""
    items: List[CognitiveAnalysisResult] = Field(..., description="数据项")
    total: int = Field(..., ge=0, description="总数量")
    page: int = Field(..., ge=1, description="当前页码")
    page_size: int = Field(..., ge=1, description="每页大小")
    total_pages: int = Field(..., ge=0, description="总页数")
    has_next: bool = Field(..., description="是否有下一页")
    has_prev: bool = Field(..., description="是否有上一页")


class PerformanceMetrics(BaseModel):
    """性能指标"""
    timestamp: datetime = Field(default_factory=datetime.now, description="指标时间")
    avg_processing_time: float = Field(..., description="平均处理时间")
    success_rate: float = Field(..., ge=0.0, le=1.0, description="成功率")
    error_rate: float = Field(..., ge=0.0, le=1.0, description="错误率")
    requests_per_minute: float = Field(..., description="每分钟请求数")
    active_sessions: int = Field(..., ge=0, description="活跃会话数")
    memory_usage: float = Field(..., description="内存使用率")
    cpu_usage: float = Field(..., description="CPU使用率")


class MonitoringAlert(BaseModel):
    """监控告警"""
    alert_id: str = Field(..., description="告警ID")
    alert_type: str = Field(..., description="告警类型")
    severity: str = Field(..., pattern=r"^(low|medium|high|critical)$", description="严重程度")
    message: str = Field(..., description="告警消息")
    timestamp: datetime = Field(default_factory=datetime.now, description="告警时间")
    resolved: bool = Field(default=False, description="是否已解决")


class SystemStatus(BaseModel):
    """系统状态"""
    overall_health: str = Field(..., description="整体健康状态")
    services: Dict[str, str] = Field(..., description="各服务状态")
    performance: PerformanceMetrics = Field(..., description="性能指标")
    active_alerts: List[MonitoringAlert] = Field(default_factory=list, description="活跃告警")
    uptime: float = Field(..., description="运行时间（秒）")