"""
认知分析服务
保持DIFCM核心算法，集成工业级数据源和辅助模块
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Optional, Tuple
import numpy as np
import sys
import os
import torch
from transformers import pipeline  # 仅用于数据预处理
import datasets
from datasets import load_dataset

# 保持Team A的核心DIFCM算法
sys.path.append("/Users/pluviophile/chi2025")
from team_a.models.difcm_core import DIFCMModel, DIFCMConfig, CognitiveStateInterpreter
from team_a.features.cognitive_features import CognitiveFeatureExtractor, FeaturePreprocessor

from .models import (
    UserBehaviorData, CognitiveState, CognitiveAnalysisResult, 
    ProcessingStatus, CognitiveConceptName
)
from .industrial_data import IndustrialDatasetLoader, EnhancedFeatureExtractor

logger = logging.getLogger(__name__)


class CognitiveAnalysisService:
    """
    认知分析服务
    整合DIFCM模型和特征提取，提供工业级的认知状态分析
    """
    
    def __init__(self):
        """初始化认知分析服务 - 保持DIFCM核心，增加工业数据"""
        self.difcm_model = None
        self.feature_extractor = None
        self.feature_preprocessor = None
        self.state_interpreter = None
        self.model_loaded = False
        
        # 新增：工业级数据增强
        self.dataset_loader = IndustrialDatasetLoader()
        self.enhanced_extractor = None
        
        self.processing_queue = asyncio.Queue(maxsize=1000)
        self.performance_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_processing_time': 0.0
        }
        
        logger.info("认知分析服务初始化开始 - 保持DIFCM算法，集成工业数据")
        
    async def initialize(self):
        """异步初始化模型和组件 - 保持DIFCM，增加工业数据"""
        try:
            # 1. 初始化核心DIFCM模型 (保持不变)
            config = DIFCMConfig(
                n_concepts=8,
                learning_rate=0.01,
                decay_rate=0.95,
                device='cpu'
            )
            self.difcm_model = DIFCMModel(config)
            
            # 2. 初始化原有特征提取器 (保持不变)
            self.feature_extractor = CognitiveFeatureExtractor()
            
            # 3. 初始化特征预处理器 (保持不变)
            self.feature_preprocessor = FeaturePreprocessor()
            
            # 4. 初始化状态解释器 (保持不变)
            self.state_interpreter = CognitiveStateInterpreter()
            
            # 5. 新增：初始化工业级数据集
            await self.dataset_loader.initialize()
            self.enhanced_extractor = EnhancedFeatureExtractor(self.dataset_loader)
            
            # 6. 加载预训练模型（如果存在）
            await self._load_pretrained_model()
            
            self.model_loaded = True
            logger.info("认知分析服务初始化完成 - DIFCM核心算法 + 工业数据增强")
            
        except Exception as e:
            logger.error(f"认知分析服务初始化失败: {e}")
            raise
    
    async def _load_pretrained_model(self):
        """加载预训练的DIFCM模型"""
        model_path = "/Users/pluviophile/chi2025/team_a/models/pretrained_difcm.npy"
        try:
            if os.path.exists(model_path):
                self.difcm_model.load_state(model_path)
                logger.info(f"预训练模型已加载: {model_path}")
            else:
                logger.info("未找到预训练模型，使用随机初始化")
        except Exception as e:
            logger.warning(f"预训练模型加载失败: {e}")
    
    async def analyze_behavior(self, behavior_data: UserBehaviorData) -> CognitiveAnalysisResult:
        """
        分析用户行为数据，返回认知状态
        
        Args:
            behavior_data: 用户行为数据
            
        Returns:
            认知分析结果
        """
        if not self.model_loaded:
            raise RuntimeError("认知分析服务未初始化")
        
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # 1. 数据预处理和验证
            validated_data = await self._validate_behavior_data(behavior_data)
            
            # 2. 增强特征提取 - 保持DIFCM算法，增加工业数据
            if self.enhanced_extractor:
                extracted_features = await self.enhanced_extractor.extract_enhanced_features(
                    validated_data, 
                    text_content=validated_data.get('task_context', {}).get('text_content', '')
                )
            else:
                # 降级到原有特征提取
                extracted_features = await self._extract_features(validated_data)
            
            # 3. DIFCM认知状态预测
            cognitive_state = await self._predict_cognitive_state(extracted_features)
            
            # 4. 计算置信度
            confidence_score = await self._calculate_confidence(
                extracted_features, cognitive_state
            )
            
            # 5. 构建结果
            processing_time = time.time() - start_time
            
            result = CognitiveAnalysisResult(
                request_id=request_id,
                user_id=behavior_data.user_id,
                session_id=behavior_data.session_id,
                timestamp=behavior_data.timestamp,
                cognitive_state=cognitive_state,
                extracted_features=extracted_features,
                confidence_score=confidence_score,
                processing_time=processing_time,
                status=ProcessingStatus.COMPLETED
            )
            
            # 6. 更新性能指标
            await self._update_performance_metrics(processing_time, True)
            
            logger.info(f"认知分析完成: user={behavior_data.user_id}, time={processing_time:.3f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"认知分析失败: {e}")
            
            # 更新性能指标
            await self._update_performance_metrics(processing_time, False)
            
            # 返回错误结果
            return CognitiveAnalysisResult(
                request_id=request_id,
                user_id=behavior_data.user_id,
                session_id=behavior_data.session_id,
                timestamp=behavior_data.timestamp,
                cognitive_state=CognitiveState(
                    attention=0.5, memory=0.5, comprehension=0.5, creativity=0.5,
                    motivation=0.5, emotion=0.5, confidence=0.5, fatigue=0.5
                ),
                extracted_features={},
                confidence_score=0.0,
                processing_time=processing_time,
                status=ProcessingStatus.FAILED,
                error_message=str(e)
            )
    
    async def _validate_behavior_data(self, data: UserBehaviorData) -> Dict:
        """验证和预处理行为数据"""
        # 转换为Team A特征提取器所需的格式
        raw_data = {
            'keystrokes': [ks.dict() for ks in data.keystrokes],
            'mouse_moves': [mm.dict() for mm in data.mouse_moves],
            'dwell_times': data.dwell_times,
            'response_times': data.response_times,
            'task_context': data.task_context.dict()
        }
        
        # 数据质量检查
        if not raw_data['keystrokes'] and not raw_data['mouse_moves']:
            raise ValueError("行为数据不能为空")
        
        return raw_data
    
    async def _extract_features(self, raw_data: Dict) -> Dict[str, float]:
        """使用Team A的特征提取器提取特征"""
        try:
            features = self.feature_extractor.extract_all_features(raw_data)
            
            # 确保所有8个认知维度都有值
            required_features = [
                'attention', 'memory', 'comprehension', 'creativity',
                'motivation', 'emotion', 'confidence', 'fatigue'
            ]
            
            for feature_name in required_features:
                if feature_name not in features:
                    features[feature_name] = 0.5  # 默认值
            
            logger.debug(f"特征提取完成: {features}")
            return features
            
        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            # 返回默认特征
            return {name: 0.5 for name in [
                'attention', 'memory', 'comprehension', 'creativity',
                'motivation', 'emotion', 'confidence', 'fatigue'
            ]}
    
    async def _predict_cognitive_state(self, features: Dict[str, float]) -> CognitiveState:
        """使用DIFCM模型预测认知状态"""
        try:
            # 转换特征为DIFCM输入格式
            feature_vector = np.array([
                features['attention'], features['memory'], features['comprehension'],
                features['creativity'], features['motivation'], features['emotion'],
                features['confidence'], features['fatigue']
            ])
            
            # DIFCM预测
            import torch
            input_features = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                predicted_state = self.difcm_model.forward(input_features)
                state_array = predicted_state.squeeze().numpy()
            
            # 更新DIFCM内部状态
            self.difcm_model.update_concepts(state_array)
            
            # 构建认知状态对象
            cognitive_state = CognitiveState(
                attention=float(state_array[0]),
                memory=float(state_array[1]),
                comprehension=float(state_array[2]),
                creativity=float(state_array[3]),
                motivation=float(state_array[4]),
                emotion=float(state_array[5]),
                confidence=float(state_array[6]),
                fatigue=float(state_array[7])
            )
            
            logger.debug(f"DIFCM预测完成: {cognitive_state.dict()}")
            return cognitive_state
            
        except Exception as e:
            logger.error(f"DIFCM预测失败: {e}")
            # 返回基于特征的简单映射
            return CognitiveState(
                attention=features['attention'],
                memory=features['memory'],
                comprehension=features['comprehension'],
                creativity=features['creativity'],
                motivation=features['motivation'],
                emotion=features['emotion'],
                confidence=features['confidence'],
                fatigue=features['fatigue']
            )
    
    async def _calculate_confidence(self, features: Dict[str, float], 
                                  cognitive_state: CognitiveState) -> float:
        """计算预测置信度"""
        try:
            # 基于特征质量和模型一致性计算置信度
            feature_quality = self._assess_feature_quality(features)
            model_consistency = self._assess_model_consistency(features, cognitive_state)
            
            # 综合置信度
            confidence = (feature_quality + model_consistency) / 2
            return max(0.1, min(0.95, confidence))
            
        except Exception as e:
            logger.error(f"置信度计算失败: {e}")
            return 0.5
    
    def _assess_feature_quality(self, features: Dict[str, float]) -> float:
        """评估特征质量"""
        # 检查特征值的合理性
        quality_score = 1.0
        
        for name, value in features.items():
            if not (0.0 <= value <= 1.0):
                quality_score *= 0.8  # 惩罚异常值
            if value == 0.5:  # 默认值表示特征提取可能有问题
                quality_score *= 0.9
        
        return quality_score
    
    def _assess_model_consistency(self, features: Dict[str, float], 
                                cognitive_state: CognitiveState) -> float:
        """评估模型一致性"""
        # 计算特征和预测状态的一致性
        feature_vector = np.array(list(features.values()))
        state_vector = np.array(cognitive_state.to_vector())
        
        # 使用余弦相似度衡量一致性
        try:
            dot_product = np.dot(feature_vector, state_vector)
            norm_features = np.linalg.norm(feature_vector)
            norm_state = np.linalg.norm(state_vector)
            
            if norm_features == 0 or norm_state == 0:
                return 0.5
            
            similarity = dot_product / (norm_features * norm_state)
            return (similarity + 1) / 2  # 归一化到[0,1]
            
        except Exception:
            return 0.5
    
    async def _update_performance_metrics(self, processing_time: float, success: bool):
        """更新性能指标"""
        self.performance_metrics['total_requests'] += 1
        
        if success:
            self.performance_metrics['successful_requests'] += 1
        else:
            self.performance_metrics['failed_requests'] += 1
        
        # 更新平均处理时间
        total_time = (self.performance_metrics['avg_processing_time'] * 
                     (self.performance_metrics['total_requests'] - 1) + processing_time)
        self.performance_metrics['avg_processing_time'] = total_time / self.performance_metrics['total_requests']
    
    def get_performance_metrics(self) -> Dict:
        """获取性能指标"""
        total = self.performance_metrics['total_requests']
        if total == 0:
            return {
                'success_rate': 1.0,
                'error_rate': 0.0,
                'avg_processing_time': 0.0,
                'total_requests': 0
            }
        
        return {
            'success_rate': self.performance_metrics['successful_requests'] / total,
            'error_rate': self.performance_metrics['failed_requests'] / total,
            'avg_processing_time': self.performance_metrics['avg_processing_time'],
            'total_requests': total
        }
    
    def is_ready(self) -> bool:
        """检查服务是否就绪"""
        return (self.model_loaded and 
                self.difcm_model is not None and 
                self.feature_extractor is not None)
    
    async def get_model_info(self) -> Dict:
        """获取模型信息"""
        if not self.is_ready():
            return {"status": "not_ready"}
        
        return {
            "status": "ready",
            "difcm_config": self.difcm_model.config.__dict__,
            "concept_relationships": self.difcm_model.get_concept_relationships(),
            "current_state": self.difcm_model.get_state_vector().tolist()
        }