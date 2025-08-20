"""
MongoDB存储服务
负责认知状态数据的持久化存储和查询
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pymongo
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from pymongo import IndexModel, ASCENDING, DESCENDING
import time

from .models import (
    CognitiveAnalysisResult, UserBehaviorData, QueryFilters, 
    PaginationParams, PaginatedResponse, ProcessingStatus
)
from .config import get_settings

logger = logging.getLogger(__name__)


class CognitiveStateStorage:
    """
    认知状态存储服务
    提供高性能的MongoDB存储和查询功能
    """
    
    def __init__(self):
        """初始化存储服务"""
        self.settings = get_settings()
        self.client: Optional[AsyncIOMotorClient] = None
        self.database: Optional[AsyncIOMotorDatabase] = None
        self.collections: Dict[str, AsyncIOMotorCollection] = {}
        self.connected = False
        
        # 集合名称定义
        self.COGNITIVE_STATES_COLLECTION = "cognitive_states"
        self.BEHAVIOR_DATA_COLLECTION = "behavior_data"
        self.PERFORMANCE_METRICS_COLLECTION = "performance_metrics"
        self.USER_SESSIONS_COLLECTION = "user_sessions"
        
        logger.info("认知状态存储服务初始化")
    
    async def connect(self):
        """连接到MongoDB"""
        try:
            self.client = AsyncIOMotorClient(
                self.settings.mongodb_url,
                serverSelectionTimeoutMS=5000,
                maxPoolSize=50,
                minPoolSize=10
            )
            
            # 验证连接
            await self.client.admin.command('ping')
            
            self.database = self.client[self.settings.database_name]
            
            # 初始化集合
            await self._initialize_collections()
            
            # 创建索引
            await self._create_indexes()
            
            self.connected = True
            logger.info(f"MongoDB连接成功: {self.settings.mongodb_url}")
            
        except Exception as e:
            logger.error(f"MongoDB连接失败: {e}")
            raise
    
    async def disconnect(self):
        """断开MongoDB连接"""
        if self.client:
            self.client.close()
            self.connected = False
            logger.info("MongoDB连接已断开")
    
    async def _initialize_collections(self):
        """初始化集合"""
        collection_names = [
            self.COGNITIVE_STATES_COLLECTION,
            self.BEHAVIOR_DATA_COLLECTION,
            self.PERFORMANCE_METRICS_COLLECTION,
            self.USER_SESSIONS_COLLECTION
        ]
        
        for collection_name in collection_names:
            self.collections[collection_name] = self.database[collection_name]
        
        logger.info(f"集合初始化完成: {list(self.collections.keys())}")
    
    async def _create_indexes(self):
        """创建数据库索引以优化查询性能"""
        try:
            # 认知状态集合索引
            cognitive_indexes = [
                IndexModel([("user_id", ASCENDING), ("timestamp", DESCENDING)]),
                IndexModel([("session_id", ASCENDING), ("timestamp", DESCENDING)]),
                IndexModel([("timestamp", DESCENDING)]),
                IndexModel([("status", ASCENDING)]),
                IndexModel([("confidence_score", DESCENDING)]),
                IndexModel([("user_id", ASCENDING), ("session_id", ASCENDING), ("timestamp", DESCENDING)]),
                # 复合索引用于常见查询
                IndexModel([
                    ("user_id", ASCENDING), 
                    ("status", ASCENDING), 
                    ("timestamp", DESCENDING)
                ])
            ]
            
            await self.collections[self.COGNITIVE_STATES_COLLECTION].create_indexes(cognitive_indexes)
            
            # 行为数据集合索引
            behavior_indexes = [
                IndexModel([("user_id", ASCENDING), ("timestamp", DESCENDING)]),
                IndexModel([("session_id", ASCENDING)]),
                IndexModel([("timestamp", DESCENDING)])
            ]
            
            await self.collections[self.BEHAVIOR_DATA_COLLECTION].create_indexes(behavior_indexes)
            
            # 性能指标集合索引
            metrics_indexes = [
                IndexModel([("timestamp", DESCENDING)]),
                IndexModel([("metric_type", ASCENDING), ("timestamp", DESCENDING)])
            ]
            
            await self.collections[self.PERFORMANCE_METRICS_COLLECTION].create_indexes(metrics_indexes)
            
            # 用户会话集合索引
            session_indexes = [
                IndexModel([("user_id", ASCENDING), ("start_time", DESCENDING)]),
                IndexModel([("session_id", ASCENDING)], unique=True),
                IndexModel([("start_time", DESCENDING)])
            ]
            
            await self.collections[self.USER_SESSIONS_COLLECTION].create_indexes(session_indexes)
            
            logger.info("数据库索引创建完成")
            
        except Exception as e:
            logger.error(f"索引创建失败: {e}")
            raise
    
    async def store_cognitive_analysis(self, analysis_result: CognitiveAnalysisResult) -> str:
        """
        存储认知分析结果
        
        Args:
            analysis_result: 认知分析结果
            
        Returns:
            存储的文档ID
        """
        try:
            document = {
                "request_id": analysis_result.request_id,
                "user_id": analysis_result.user_id,
                "session_id": analysis_result.session_id,
                "timestamp": analysis_result.timestamp,
                "cognitive_state": analysis_result.cognitive_state.dict(),
                "extracted_features": analysis_result.extracted_features,
                "confidence_score": analysis_result.confidence_score,
                "processing_time": analysis_result.processing_time,
                "status": analysis_result.status.value,
                "error_message": analysis_result.error_message,
                "created_at": datetime.utcnow(),
                # 添加便于查询的字段
                "date": datetime.utcfromtimestamp(analysis_result.timestamp).date().isoformat(),
                "hour": datetime.utcfromtimestamp(analysis_result.timestamp).hour
            }
            
            result = await self.collections[self.COGNITIVE_STATES_COLLECTION].insert_one(document)
            
            logger.debug(f"认知分析结果已存储: {result.inserted_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"存储认知分析结果失败: {e}")
            raise
    
    async def store_behavior_data(self, behavior_data: UserBehaviorData) -> str:
        """
        存储原始行为数据
        
        Args:
            behavior_data: 用户行为数据
            
        Returns:
            存储的文档ID
        """
        try:
            document = {
                "user_id": behavior_data.user_id,
                "session_id": behavior_data.session_id,
                "timestamp": behavior_data.timestamp,
                "keystrokes": [ks.dict() for ks in behavior_data.keystrokes],
                "mouse_moves": [mm.dict() for mm in behavior_data.mouse_moves],
                "dwell_times": behavior_data.dwell_times,
                "response_times": behavior_data.response_times,
                "task_context": behavior_data.task_context.dict(),
                "created_at": datetime.utcnow(),
                "date": datetime.utcfromtimestamp(behavior_data.timestamp).date().isoformat()
            }
            
            result = await self.collections[self.BEHAVIOR_DATA_COLLECTION].insert_one(document)
            
            logger.debug(f"行为数据已存储: {result.inserted_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"存储行为数据失败: {e}")
            raise
    
    async def query_cognitive_states(self, 
                                   filters: QueryFilters,
                                   pagination: PaginationParams) -> PaginatedResponse:
        """
        查询认知状态数据
        
        Args:
            filters: 查询过滤器
            pagination: 分页参数
            
        Returns:
            分页查询结果
        """
        try:
            # 构建查询条件
            query = await self._build_query(filters)
            
            # 构建排序条件
            sort_field = pagination.sort_by
            sort_direction = DESCENDING if pagination.sort_order == "desc" else ASCENDING
            sort_criteria = [(sort_field, sort_direction)]
            
            # 计算总数
            total = await self.collections[self.COGNITIVE_STATES_COLLECTION].count_documents(query)
            
            # 计算分页
            skip = (pagination.page - 1) * pagination.page_size
            total_pages = (total + pagination.page_size - 1) // pagination.page_size
            
            # 执行查询
            cursor = self.collections[self.COGNITIVE_STATES_COLLECTION].find(query)\
                .sort(sort_criteria)\
                .skip(skip)\
                .limit(pagination.page_size)
            
            documents = await cursor.to_list(length=pagination.page_size)
            
            # 转换为分析结果对象
            items = []
            for doc in documents:
                analysis_result = await self._document_to_analysis_result(doc)
                items.append(analysis_result)
            
            # 构建分页响应
            response = PaginatedResponse(
                items=items,
                total=total,
                page=pagination.page,
                page_size=pagination.page_size,
                total_pages=total_pages,
                has_next=pagination.page < total_pages,
                has_prev=pagination.page > 1
            )
            
            logger.debug(f"查询完成: {len(items)} 项，总计 {total} 项")
            return response
            
        except Exception as e:
            logger.error(f"查询认知状态失败: {e}")
            raise
    
    async def _build_query(self, filters: QueryFilters) -> Dict:
        """构建MongoDB查询条件"""
        query = {}
        
        if filters.user_id:
            query["user_id"] = filters.user_id
        
        if filters.session_id:
            query["session_id"] = filters.session_id
        
        if filters.start_time or filters.end_time:
            time_query = {}
            if filters.start_time:
                time_query["$gte"] = filters.start_time
            if filters.end_time:
                time_query["$lte"] = filters.end_time
            query["timestamp"] = time_query
        
        if filters.min_confidence is not None:
            query["confidence_score"] = {"$gte": filters.min_confidence}
        
        if filters.status:
            query["status"] = filters.status.value
        
        return query
    
    async def _document_to_analysis_result(self, doc: Dict) -> CognitiveAnalysisResult:
        """将MongoDB文档转换为分析结果对象"""
        from .models import CognitiveState
        
        cognitive_state = CognitiveState(**doc["cognitive_state"])
        
        return CognitiveAnalysisResult(
            request_id=doc["request_id"],
            user_id=doc["user_id"],
            session_id=doc["session_id"],
            timestamp=doc["timestamp"],
            cognitive_state=cognitive_state,
            extracted_features=doc["extracted_features"],
            confidence_score=doc["confidence_score"],
            processing_time=doc["processing_time"],
            status=ProcessingStatus(doc["status"]),
            error_message=doc.get("error_message")
        )
    
    async def get_user_latest_state(self, user_id: str) -> Optional[CognitiveAnalysisResult]:
        """获取用户最新的认知状态"""
        try:
            doc = await self.collections[self.COGNITIVE_STATES_COLLECTION].find_one(
                {"user_id": user_id, "status": ProcessingStatus.COMPLETED.value},
                sort=[("timestamp", DESCENDING)]
            )
            
            if doc:
                return await self._document_to_analysis_result(doc)
            return None
            
        except Exception as e:
            logger.error(f"获取用户最新状态失败: {e}")
            return None
    
    async def get_session_history(self, session_id: str, limit: int = 100) -> List[CognitiveAnalysisResult]:
        """获取会话历史记录"""
        try:
            cursor = self.collections[self.COGNITIVE_STATES_COLLECTION].find(
                {"session_id": session_id}
            ).sort([("timestamp", ASCENDING)]).limit(limit)
            
            documents = await cursor.to_list(length=limit)
            
            results = []
            for doc in documents:
                result = await self._document_to_analysis_result(doc)
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"获取会话历史失败: {e}")
            return []
    
    async def store_performance_metrics(self, metrics: Dict) -> str:
        """存储性能指标"""
        try:
            document = {
                **metrics,
                "timestamp": datetime.utcnow(),
                "metric_type": "system_performance"
            }
            
            result = await self.collections[self.PERFORMANCE_METRICS_COLLECTION].insert_one(document)
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"存储性能指标失败: {e}")
            raise
    
    async def get_recent_performance(self, hours: int = 24) -> List[Dict]:
        """获取最近的性能指标"""
        try:
            start_time = datetime.utcnow() - timedelta(hours=hours)
            
            cursor = self.collections[self.PERFORMANCE_METRICS_COLLECTION].find(
                {"timestamp": {"$gte": start_time}}
            ).sort([("timestamp", DESCENDING)])
            
            return await cursor.to_list(length=1000)
            
        except Exception as e:
            logger.error(f"获取性能指标失败: {e}")
            return []
    
    async def cleanup_old_data(self, days: int = 90):
        """清理过期数据"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=days)
            
            # 清理过期的行为数据
            behavior_result = await self.collections[self.BEHAVIOR_DATA_COLLECTION].delete_many(
                {"created_at": {"$lt": cutoff_time}}
            )
            
            # 清理过期的性能指标
            metrics_result = await self.collections[self.PERFORMANCE_METRICS_COLLECTION].delete_many(
                {"timestamp": {"$lt": cutoff_time}}
            )
            
            logger.info(f"数据清理完成: 行为数据 {behavior_result.deleted_count} 条, "
                       f"性能指标 {metrics_result.deleted_count} 条")
            
        except Exception as e:
            logger.error(f"数据清理失败: {e}")
    
    async def get_statistics(self) -> Dict:
        """获取存储统计信息"""
        try:
            stats = {}
            
            for collection_name, collection in self.collections.items():
                count = await collection.count_documents({})
                stats[collection_name] = count
            
            return stats
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}
    
    def is_connected(self) -> bool:
        """检查连接状态"""
        return self.connected and self.client is not None