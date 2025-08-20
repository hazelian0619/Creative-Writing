"""
FastAPI主应用程序
提供完整的认知状态追踪API服务
"""

import asyncio
import logging
import time
import traceback
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, generate_latest

from .config import get_settings
from .models import (
    UserBehaviorData, CognitiveAnalysisResult, DataReceiveResponse,
    HealthCheckResponse, QueryFilters, PaginationParams, PaginatedResponse,
    PerformanceMetrics, SystemStatus
)
from .cognitive_service import CognitiveAnalysisService
from .storage.mongodb_service import CognitiveStateStorage

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus指标
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'API request duration', ['method', 'endpoint'])
PROCESSING_TIME = Histogram('cognitive_processing_duration_seconds', 'Cognitive analysis processing time')
ACTIVE_SESSIONS = Gauge('active_sessions_total', 'Number of active sessions')
COGNITIVE_STATES_STORED = Counter('cognitive_states_stored_total', 'Total cognitive states stored')

# 全局服务实例
cognitive_service: Optional[CognitiveAnalysisService] = None
storage_service: Optional[CognitiveStateStorage] = None
app_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用程序生命周期管理"""
    # 启动时初始化
    await startup_event()
    try:
        yield
    finally:
        # 关闭时清理
        await shutdown_event()


# 创建FastAPI应用
app = FastAPI(
    title="认知状态追踪API",
    description="师范生创意写作认知状态实时追踪系统 - Team B 后端服务",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# 添加中间件
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Prometheus指标中间件"""
    start_time = time.time()
    method = request.method
    endpoint = request.url.path
    
    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as e:
        logger.error(f"请求处理异常: {e}")
        status_code = 500
        response = JSONResponse(
            status_code=500,
            content={"error": "内部服务器错误"}
        )
    
    # 记录指标
    duration = time.time() - start_time
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status_code).inc()
    REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
    
    return response


async def startup_event():
    """应用启动事件"""
    global cognitive_service, storage_service
    
    logger.info("🚀 启动认知状态追踪API服务...")
    
    try:
        # 初始化存储服务
        logger.info("初始化存储服务...")
        storage_service = CognitiveStateStorage()
        await storage_service.connect()
        
        # 初始化认知分析服务
        logger.info("初始化认知分析服务...")
        cognitive_service = CognitiveAnalysisService()
        await cognitive_service.initialize()
        
        logger.info("✅ 所有服务初始化完成")
        
    except Exception as e:
        logger.error(f"❌ 服务初始化失败: {e}")
        raise


async def shutdown_event():
    """应用关闭事件"""
    global storage_service
    
    logger.info("🛑 关闭认知状态追踪API服务...")
    
    if storage_service:
        await storage_service.disconnect()
    
    logger.info("✅ 服务关闭完成")


def get_cognitive_service() -> CognitiveAnalysisService:
    """获取认知分析服务实例"""
    if cognitive_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="认知分析服务未初始化"
        )
    return cognitive_service


def get_storage_service() -> CognitiveStateStorage:
    """获取存储服务实例"""
    if storage_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="存储服务未初始化"
        )
    return storage_service


# ======================== API 端点 ========================

@app.get("/", response_model=dict)
async def root():
    """根端点"""
    return {
        "service": "认知状态追踪API",
        "version": "1.0.0",
        "team": "Team B - Backend Engineering",
        "status": "运行中",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check(
    cognitive_svc: CognitiveAnalysisService = Depends(get_cognitive_service),
    storage_svc: CognitiveStateStorage = Depends(get_storage_service)
):
    """健康检查端点"""
    try:
        # 检查各个服务状态
        database_connected = storage_svc.is_connected()
        difcm_model_loaded = cognitive_svc.is_ready()
        feature_extractor_ready = cognitive_svc.feature_extractor is not None
        
        overall_status = "healthy" if all([
            database_connected, difcm_model_loaded, feature_extractor_ready
        ]) else "unhealthy"
        
        return HealthCheckResponse(
            status=overall_status,
            database_connected=database_connected,
            difcm_model_loaded=difcm_model_loaded,
            feature_extractor_ready=feature_extractor_ready
        )
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"健康检查失败: {str(e)}"
        )


@app.post("/api/v1/behavior-data", response_model=DataReceiveResponse)
async def receive_behavior_data(
    data: UserBehaviorData,
    background_tasks: BackgroundTasks,
    cognitive_svc: CognitiveAnalysisService = Depends(get_cognitive_service),
    storage_svc: CognitiveStateStorage = Depends(get_storage_service)
):
    """
    接收用户行为数据并进行认知状态分析
    这是核心API端点，与Team A的模型和Team C的数据格式完全兼容
    """
    try:
        start_time = time.time()
        
        # 1. 异步存储原始行为数据
        background_tasks.add_task(storage_svc.store_behavior_data, data)
        
        # 2. 进行认知分析
        analysis_result = await cognitive_svc.analyze_behavior(data)
        
        # 3. 存储分析结果
        data_id = await storage_svc.store_cognitive_analysis(analysis_result)
        
        # 4. 记录性能指标
        processing_time = time.time() - start_time
        PROCESSING_TIME.observe(processing_time)
        COGNITIVE_STATES_STORED.inc()
        
        # 5. 构建响应
        response = DataReceiveResponse(
            status="success",
            data_id=data_id,
            request_id=analysis_result.request_id,
            cognitive_state=analysis_result.cognitive_state,
            message=f"数据处理成功，置信度: {analysis_result.confidence_score:.3f}"
        )
        
        logger.info(f"行为数据处理完成: user={data.user_id}, time={processing_time:.3f}s")
        return response
        
    except Exception as e:
        logger.error(f"行为数据处理失败: {e}")
        logger.error(traceback.format_exc())
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"数据处理失败: {str(e)}"
        )


@app.get("/api/v1/cognitive-states", response_model=PaginatedResponse)
async def query_cognitive_states(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    min_confidence: Optional[float] = None,
    page: int = 1,
    page_size: int = 20,
    sort_by: str = "timestamp",
    sort_order: str = "desc",
    storage_svc: CognitiveStateStorage = Depends(get_storage_service)
):
    """查询认知状态历史数据（支持分页和过滤）"""
    try:
        # 构建查询条件
        filters = QueryFilters(
            user_id=user_id,
            session_id=session_id,
            start_time=start_time,
            end_time=end_time,
            min_confidence=min_confidence
        )
        
        # 构建分页参数
        pagination = PaginationParams(
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        # 执行查询
        result = await storage_svc.query_cognitive_states(filters, pagination)
        return result
        
    except Exception as e:
        logger.error(f"查询认知状态失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"查询失败: {str(e)}"
        )


@app.get("/api/v1/latest-state/{user_id}", response_model=Optional[CognitiveAnalysisResult])
async def get_user_latest_state(
    user_id: str,
    storage_svc: CognitiveStateStorage = Depends(get_storage_service)
):
    """获取用户最新的认知状态"""
    try:
        result = await storage_svc.get_user_latest_state(user_id)
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"未找到用户 {user_id} 的认知状态数据"
            )
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取用户最新状态失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取最新状态失败: {str(e)}"
        )


@app.get("/api/v1/session-history/{session_id}", response_model=List[CognitiveAnalysisResult])
async def get_session_history(
    session_id: str,
    limit: int = 100,
    storage_svc: CognitiveStateStorage = Depends(get_storage_service)
):
    """获取会话的认知状态历史"""
    try:
        history = await storage_svc.get_session_history(session_id, limit)
        return history
        
    except Exception as e:
        logger.error(f"获取会话历史失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取会话历史失败: {str(e)}"
        )


@app.get("/api/v1/performance", response_model=PerformanceMetrics)
async def get_performance_metrics(
    cognitive_svc: CognitiveAnalysisService = Depends(get_cognitive_service),
    storage_svc: CognitiveStateStorage = Depends(get_storage_service)
):
    """获取系统性能指标"""
    try:
        # 获取认知服务性能指标
        cognitive_metrics = cognitive_svc.get_performance_metrics()
        
        # 获取存储统计
        storage_stats = await storage_svc.get_statistics()
        
        # 计算系统指标
        current_time = datetime.now()
        uptime = time.time() - app_start_time
        
        return PerformanceMetrics(
            timestamp=current_time,
            avg_processing_time=cognitive_metrics['avg_processing_time'],
            success_rate=cognitive_metrics['success_rate'],
            error_rate=cognitive_metrics['error_rate'],
            requests_per_minute=cognitive_metrics['total_requests'] / max(uptime / 60, 1),
            active_sessions=storage_stats.get('user_sessions', 0),
            memory_usage=0.0,  # TODO: 实现内存监控
            cpu_usage=0.0      # TODO: 实现CPU监控
        )
        
    except Exception as e:
        logger.error(f"获取性能指标失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取性能指标失败: {str(e)}"
        )


@app.get("/api/v1/model-info", response_model=dict)
async def get_model_info(
    cognitive_svc: CognitiveAnalysisService = Depends(get_cognitive_service)
):
    """获取DIFCM模型信息"""
    try:
        model_info = await cognitive_svc.get_model_info()
        return model_info
        
    except Exception as e:
        logger.error(f"获取模型信息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取模型信息失败: {str(e)}"
        )


@app.get("/metrics")
async def get_prometheus_metrics():
    """Prometheus指标端点"""
    return generate_latest(prometheus_client.REGISTRY)


@app.get("/api/v1/system-status", response_model=SystemStatus)
async def get_system_status(
    cognitive_svc: CognitiveAnalysisService = Depends(get_cognitive_service),
    storage_svc: CognitiveStateStorage = Depends(get_storage_service)
):
    """获取完整的系统状态"""
    try:
        # 检查各服务状态
        services_status = {
            "cognitive_service": "healthy" if cognitive_svc.is_ready() else "unhealthy",
            "storage_service": "healthy" if storage_svc.is_connected() else "unhealthy",
            "database": "connected" if storage_svc.is_connected() else "disconnected"
        }
        
        # 获取性能指标
        performance = await get_performance_metrics(cognitive_svc, storage_svc)
        
        # 确定整体健康状态
        overall_health = "healthy" if all(
            status in ["healthy", "connected"] for status in services_status.values()
        ) else "unhealthy"
        
        return SystemStatus(
            overall_health=overall_health,
            services=services_status,
            performance=performance,
            active_alerts=[],  # TODO: 实现告警系统
            uptime=time.time() - app_start_time
        )
        
    except Exception as e:
        logger.error(f"获取系统状态失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取系统状态失败: {str(e)}"
        )


# ======================== 错误处理 ========================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器"""
    logger.error(f"未处理的异常: {exc}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "内部服务器错误",
            "detail": str(exc),
            "path": str(request.url),
            "timestamp": datetime.now().isoformat()
        }
    )


# ======================== 开发服务器 ========================

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "team_b.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower()
    )