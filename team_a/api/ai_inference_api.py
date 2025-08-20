"""
AI推理服务API接口
与team B的存储和API服务对接
"""

import asyncio
import time
import uuid
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import logging
import json
from datetime import datetime
import httpx
from contextlib import asynccontextmanager

# 导入team A的核心模块
from team_a.models.algorithm_coordinator import (
    AlgorithmCoordinator, CoordinatorConfig, InferenceRequest, InferenceResult
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API数据模型
class UserBehaviorData(BaseModel):
    """用户行为数据模型"""
    user_id: str = Field(..., description="用户ID")
    session_id: str = Field(..., description="会话ID")
    timestamp: float = Field(default_factory=time.time, description="时间戳")
    input_text: Optional[str] = Field(None, description="用户输入文本")
    keystrokes: List[Dict[str, Any]] = Field(default_factory=list, description="击键数据")
    mouse_moves: List[Dict[str, Any]] = Field(default_factory=list, description="鼠标移动数据")
    dwell_times: List[float] = Field(default_factory=list, description="停留时间")
    response_times: List[float] = Field(default_factory=list, description="响应时间")
    task_context: Dict[str, Any] = Field(default_factory=dict, description="任务上下文")
    writing_goal: Optional[str] = Field("创意写作", description="写作目标")
    genre: Optional[str] = Field("通用", description="文体类型")
    difficulty_level: Optional[str] = Field("medium", description="难度级别")

class CognitiveStateResponse(BaseModel):
    """认知状态响应模型"""
    user_id: str
    session_id: str
    request_id: str
    cognitive_state: Dict[str, float]
    prompts: List[str]
    cognitive_feedback: Dict[str, Any]
    system_info: Dict[str, Any]
    processing_time: float
    timestamp: float

class HealthCheckResponse(BaseModel):
    """健康检查响应模型"""
    status: str
    timestamp: float
    components: Dict[str, str]
    performance: Dict[str, float]

class ServiceStatus(BaseModel):
    """服务状态模型"""
    service_name: str = "ai-inference"
    version: str = "1.0.0"
    status: str = "healthy"
    uptime: float
    requests_processed: int
    avg_response_time: float

# 服务配置
class AIInferenceConfig:
    def __init__(self):
        self.service_name = "ai-inference-service"
        self.version = "1.0.0"
        self.max_concurrent_requests = 50
        self.request_timeout = 30.0
        self.team_b_base_url = "http://user-management-service:8002"  # team B服务地址
        self.storage_service_url = "http://storage-service:8003"
        self.monitoring_enabled = True

# 全局变量
config = AIInferenceConfig()
coordinator = None
service_start_time = time.time()
request_stats = {
    'total_requests': 0,
    'successful_requests': 0,
    'failed_requests': 0,
    'total_processing_time': 0.0
}

# 服务客户端
class TeamBServiceClient:
    """Team B服务客户端"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=10.0)
    
    async def get_user_info(self, user_id: str) -> Optional[Dict]:
        """获取用户信息"""
        try:\n            response = await self.client.get(f\"{self.base_url}/users/{user_id}\")\n            if response.status_code == 200:\n                return response.json()\n            else:\n                logger.warning(f\"获取用户信息失败: {user_id}, 状态码: {response.status_code}\")\n                return None\n        except Exception as e:\n            logger.error(f\"调用用户管理服务失败: {e}\")\n            return None\n    \n    async def get_user_learning_history(self, user_id: str, limit: int = 10) -> List[Dict]:\n        \"\"\"获取用户学习历史\"\"\"\n        try:\n            response = await self.client.get(\n                f\"{self.base_url}/users/{user_id}/learning-history\",\n                params={'limit': limit}\n            )\n            if response.status_code == 200:\n                return response.json().get('interactions', [])\n            else:\n                return []\n        except Exception as e:\n            logger.error(f\"获取学习历史失败: {e}\")\n            return []\n    \n    async def store_cognitive_state(self, user_id: str, session_id: str, \n                                  cognitive_data: Dict) -> bool:\n        \"\"\"存储认知状态\"\"\"\n        try:\n            payload = {\n                'user_id': user_id,\n                'session_id': session_id,\n                'timestamp': time.time(),\n                'cognitive_state': cognitive_data['cognitive_state'],\n                'prompts': cognitive_data['prompts'],\n                'metadata': cognitive_data.get('metadata', {})\n            }\n            \n            response = await self.client.post(\n                f\"{config.storage_service_url}/cognitive-states\",\n                json=payload\n            )\n            \n            return response.status_code == 200\n            \n        except Exception as e:\n            logger.error(f\"存储认知状态失败: {e}\")\n            return False\n    \n    async def close(self):\n        \"\"\"关闭客户端\"\"\"\n        await self.client.aclose()\n\n# 初始化服务客户端\nteam_b_client = TeamBServiceClient(config.team_b_base_url)\n\n# 应用程序生命周期管理\n@asynccontextmanager\nasync def lifespan(app: FastAPI):\n    \"\"\"应用程序生命周期管理\"\"\"\n    # 启动时初始化\n    global coordinator\n    try:\n        coordinator_config = CoordinatorConfig(\n            max_concurrent_requests=config.max_concurrent_requests,\n            timeout_seconds=config.request_timeout,\n            cache_enabled=True,\n            performance_monitoring=True\n        )\n        coordinator = AlgorithmCoordinator(coordinator_config)\n        logger.info(\"AI推理服务启动成功\")\n        yield\n    except Exception as e:\n        logger.error(f\"服务启动失败: {e}\")\n        raise\n    finally:\n        # 关闭时清理\n        if coordinator:\n            await coordinator.shutdown()\n        await team_b_client.close()\n        logger.info(\"AI推理服务已关闭\")\n\n# 创建FastAPI应用\napp = FastAPI(\n    title=\"师范生创意写作AI推理服务\",\n    description=\"基于DIFCM、T-S模糊推理、GraphSAGE的AI辅助写作系统\",\n    version=config.version,\n    lifespan=lifespan\n)\n\n# 添加中间件\napp.add_middleware(\n    CORSMiddleware,\n    allow_origins=[\"*\"],\n    allow_credentials=True,\n    allow_methods=[\"*\"],\n    allow_headers=[\"*\"],\n)\napp.add_middleware(GZipMiddleware, minimum_size=1000)\n\n# 依赖注入函数\nasync def get_user_context(user_id: str) -> Dict[str, Any]:\n    \"\"\"获取用户上下文信息\"\"\"\n    user_info = await team_b_client.get_user_info(user_id)\n    learning_history = await team_b_client.get_user_learning_history(user_id)\n    \n    return {\n        'user_info': user_info,\n        'learning_history': learning_history\n    }\n\n# API路由\n@app.post(\"/api/v1/inference/process\", response_model=CognitiveStateResponse)\nasync def process_user_interaction(\n    data: UserBehaviorData,\n    background_tasks: BackgroundTasks,\n    user_context: Dict = Depends(get_user_context)\n):\n    \"\"\"\n    处理用户交互推理请求\n    \n    这是核心API接口，接收用户行为数据，返回AI推理结果\n    \"\"\"\n    start_time = time.time()\n    request_id = str(uuid.uuid4())\n    \n    try:\n        # 更新请求统计\n        request_stats['total_requests'] += 1\n        \n        # 构建推理请求\n        inference_request = InferenceRequest(\n            request_id=request_id,\n            user_id=data.user_id,\n            session_id=data.session_id,\n            input_data={\n                'text': data.input_text or '',\n                'keystrokes': data.keystrokes,\n                'mouse_moves': data.mouse_moves,\n                'dwell_times': data.dwell_times,\n                'response_times': data.response_times,\n                'task_context': data.task_context,\n                'writing_goal': data.writing_goal,\n                'genre': data.genre,\n                'difficulty_level': data.difficulty_level\n            },\n            context_history=user_context['learning_history'],\n            timestamp=data.timestamp\n        )\n        \n        # 执行AI推理\n        result = await coordinator.process_request(inference_request)\n        \n        if not result.success:\n            request_stats['failed_requests'] += 1\n            raise HTTPException(\n                status_code=500,\n                detail=f\"AI推理失败: {result.error_message}\"\n            )\n        \n        # 构建响应\n        response = CognitiveStateResponse(\n            user_id=data.user_id,\n            session_id=data.session_id,\n            request_id=request_id,\n            cognitive_state={\n                'attention': float(result.cognitive_state[0]),\n                'memory': float(result.cognitive_state[1]),\n                'comprehension': float(result.cognitive_state[2]),\n                'creativity': float(result.cognitive_state[3]),\n                'motivation': float(result.cognitive_state[4]),\n                'emotion': float(result.cognitive_state[5]),\n                'confidence': float(result.cognitive_state[6]),\n                'fatigue': float(result.cognitive_state[7])\n            },\n            prompts=result.ai_response.get('prompts', []),\n            cognitive_feedback=result.ai_response.get('cognitive_feedback', {}),\n            system_info=result.ai_response.get('system_info', {}),\n            processing_time=result.processing_time,\n            timestamp=time.time()\n        )\n        \n        # 异步存储结果到team B服务\n        background_tasks.add_task(\n            store_result_async,\n            data.user_id,\n            data.session_id,\n            result\n        )\n        \n        # 更新统计\n        request_stats['successful_requests'] += 1\n        request_stats['total_processing_time'] += result.processing_time\n        \n        return response\n        \n    except HTTPException:\n        raise\n    except Exception as e:\n        request_stats['failed_requests'] += 1\n        logger.error(f\"处理推理请求失败: {e}\")\n        raise HTTPException(\n            status_code=500,\n            detail=f\"内部服务错误: {str(e)}\"\n        )\n\nasync def store_result_async(user_id: str, session_id: str, result: InferenceResult):\n    \"\"\"异步存储推理结果\"\"\"\n    try:\n        cognitive_data = {\n            'cognitive_state': result.cognitive_state.tolist(),\n            'prompts': result.ai_response.get('prompts', []),\n            'metadata': result.inference_metadata\n        }\n        \n        success = await team_b_client.store_cognitive_state(\n            user_id, session_id, cognitive_data\n        )\n        \n        if not success:\n            logger.warning(f\"存储认知状态失败: {user_id}\")\n            \n    except Exception as e:\n        logger.error(f\"异步存储失败: {e}\")\n\n@app.get(\"/api/v1/inference/status\", response_model=ServiceStatus)\nasync def get_service_status():\n    \"\"\"获取服务状态\"\"\"\n    uptime = time.time() - service_start_time\n    avg_response_time = (\n        request_stats['total_processing_time'] / max(1, request_stats['successful_requests'])\n    )\n    \n    return ServiceStatus(\n        uptime=uptime,\n        requests_processed=request_stats['total_requests'],\n        avg_response_time=avg_response_time\n    )\n\n@app.get(\"/api/v1/inference/health\", response_model=HealthCheckResponse)\nasync def health_check():\n    \"\"\"健康检查接口\"\"\"\n    try:\n        # 检查协调器状态\n        health_status = coordinator.get_health_status() if coordinator else {'coordinator': 'error'}\n        \n        # 检查team B服务连通性\n        try:\n            test_response = await team_b_client.client.get(\n                f\"{config.team_b_base_url}/health\",\n                timeout=5.0\n            )\n            team_b_status = 'healthy' if test_response.status_code == 200 else 'error'\n        except:\n            team_b_status = 'error'\n        \n        health_status['team_b_service'] = team_b_status\n        \n        # 获取性能指标\n        performance_report = coordinator.get_performance_report() if coordinator else {}\n        \n        overall_status = \"healthy\" if all(\n            status == 'healthy' for status in health_status.values()\n        ) else \"degraded\"\n        \n        return HealthCheckResponse(\n            status=overall_status,\n            timestamp=time.time(),\n            components=health_status,\n            performance={\n                'avg_processing_time': performance_report.get('timing', {}).get('avg_processing_time', 0.0),\n                'success_rate': performance_report.get('requests', {}).get('success_rate', 0.0),\n                'active_requests': performance_report.get('system', {}).get('active_requests', 0)\n            }\n        )\n        \n    except Exception as e:\n        logger.error(f\"健康检查失败: {e}\")\n        return HealthCheckResponse(\n            status=\"error\",\n            timestamp=time.time(),\n            components={'coordinator': 'error'},\n            performance={'error': str(e)}\n        )\n\n@app.get(\"/api/v1/inference/performance\")\nasync def get_performance_metrics():\n    \"\"\"获取详细性能指标\"\"\"\n    if not coordinator:\n        raise HTTPException(status_code=503, detail=\"协调器未初始化\")\n    \n    performance_report = coordinator.get_performance_report()\n    \n    # 添加请求统计\n    performance_report['request_stats'] = request_stats\n    performance_report['service_uptime'] = time.time() - service_start_time\n    \n    return performance_report\n\n@app.post(\"/api/v1/inference/feedback\")\nasync def submit_user_feedback(\n    feedback_data: Dict[str, Any]\n):\n    \"\"\"接收用户反馈\"\"\"\n    try:\n        user_id = feedback_data.get('user_id')\n        session_id = feedback_data.get('session_id')\n        satisfaction_score = feedback_data.get('satisfaction', 0.5)\n        \n        if not user_id or not session_id:\n            raise HTTPException(status_code=400, detail=\"缺少必需的用户ID或会话ID\")\n        \n        # 更新模糊推理引擎的自适应参数\n        if coordinator and coordinator.fuzzy_engine:\n            await asyncio.create_task(\n                asyncio.to_thread(\n                    coordinator.fuzzy_engine.adaptive_rule_adjustment,\n                    {'satisfaction': satisfaction_score}\n                )\n            )\n        \n        return {\n            \"status\": \"success\",\n            \"message\": \"反馈已接收并用于模型优化\"\n        }\n        \n    except HTTPException:\n        raise\n    except Exception as e:\n        logger.error(f\"处理用户反馈失败: {e}\")\n        raise HTTPException(status_code=500, detail=str(e))\n\n@app.get(\"/api/v1/inference/models/stats\")\nasync def get_model_statistics():\n    \"\"\"获取模型统计信息\"\"\"\n    if not coordinator:\n        raise HTTPException(status_code=503, detail=\"协调器未初始化\")\n    \n    try:\n        stats = {}\n        \n        # DIFCM模型统计\n        if coordinator.difcm_engine:\n            stats['difcm'] = {\n                'concept_states': coordinator.difcm_engine.get_interpretation(),\n                'state_history_length': len(coordinator.difcm_engine.state_history)\n            }\n        \n        # 模糊推理统计\n        if coordinator.fuzzy_engine:\n            stats['fuzzy'] = coordinator.fuzzy_engine.get_rule_statistics()\n        \n        # GraphSAGE统计\n        if coordinator.prompt_generator:\n            stats['graphsage'] = {\n                'generation_history_length': len(coordinator.prompt_generator.generation_history)\n            }\n        \n        return stats\n        \n    except Exception as e:\n        logger.error(f\"获取模型统计失败: {e}\")\n        raise HTTPException(status_code=500, detail=str(e))\n\n# 错误处理器\n@app.exception_handler(500)\nasync def internal_error_handler(request, exc):\n    logger.error(f\"内部服务器错误: {exc}\")\n    return {\n        \"error\": \"内部服务器错误\",\n        \"detail\": \"服务暂时不可用，请稍后重试\",\n        \"timestamp\": time.time()\n    }\n\n@app.exception_handler(404)\nasync def not_found_handler(request, exc):\n    return {\n        \"error\": \"接口不存在\",\n        \"detail\": f\"请求的接口 {request.url.path} 不存在\",\n        \"timestamp\": time.time()\n    }\n\nif __name__ == \"__main__\":\n    import uvicorn\n    uvicorn.run(\n        \"ai_inference_api:app\",\n        host=\"0.0.0.0\",\n        port=8001,\n        reload=True,\n        log_level=\"info\"\n    )"