"""
增强版DIFCM模型
实现区间值运算、动态权重更新、思维漂移检测和认知负载评估
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from collections import deque
import time

@dataclass
class IntervalValue:
    """区间值数据结构"""
    lower: float
    upper: float
    
    def __post_init__(self):
        if self.lower > self.upper:
            self.lower, self.upper = self.upper, self.lower
    
    def midpoint(self) -> float:
        return (self.lower + self.upper) / 2
    
    def width(self) -> float:
        return self.upper - self.lower
    
    def __add__(self, other):
        if isinstance(other, IntervalValue):
            return IntervalValue(self.lower + other.lower, self.upper + other.upper)
        return IntervalValue(self.lower + other, self.upper + other)
    
    def __mul__(self, other):
        if isinstance(other, IntervalValue):
            products = [
                self.lower * other.lower,
                self.lower * other.upper,
                self.upper * other.lower,
                self.upper * other.upper
            ]
            return IntervalValue(min(products), max(products))
        return IntervalValue(self.lower * other, self.upper * other)
    
    def __repr__(self):
        return f"[{self.lower:.3f}, {self.upper:.3f}]"

class IntervalMath:
    """区间值数学运算工具"""
    
    @staticmethod
    def sigmoid(interval: IntervalValue) -> IntervalValue:
        """区间值Sigmoid函数"""
        lower_sig = 1 / (1 + np.exp(-interval.lower))
        upper_sig = 1 / (1 + np.exp(-interval.upper))
        return IntervalValue(lower_sig, upper_sig)
    
    @staticmethod
    def distance(iv1: IntervalValue, iv2: IntervalValue) -> float:
        """计算两个区间值的Hausdorff距离"""
        return max(abs(iv1.lower - iv2.lower), abs(iv1.upper - iv2.upper))
    
    @staticmethod
    def weighted_sum(intervals: List[IntervalValue], weights: List[float]) -> IntervalValue:
        """区间值加权求和"""
        result = IntervalValue(0.0, 0.0)
        for interval, weight in zip(intervals, weights):
            result = result + (interval * weight)
        return result

class DynamicWeightMatrix:
    """动态权重矩阵"""
    
    def __init__(self, n_concepts: int, learning_rate: float = 0.01):
        self.n_concepts = n_concepts
        self.learning_rate = learning_rate
        
        # 权重矩阵使用区间值
        self.weights = np.array([
            [IntervalValue(np.random.normal(0, 0.1), np.random.normal(0, 0.1))
             for _ in range(n_concepts)]
            for _ in range(n_concepts)
        ])
        
        # 权重更新历史
        self.update_history = deque(maxlen=100)
        
    def update_weights(self, from_concept: int, to_concept: int, delta: float):
        """更新特定概念间的权重"""
        current_weight = self.weights[from_concept, to_concept]
        
        # 区间值权重更新
        new_lower = current_weight.lower + self.learning_rate * delta * 0.8
        new_upper = current_weight.upper + self.learning_rate * delta * 1.2
        
        # 约束权重范围
        new_lower = np.clip(new_lower, -1.0, 1.0)
        new_upper = np.clip(new_upper, -1.0, 1.0)
        
        self.weights[from_concept, to_concept] = IntervalValue(new_lower, new_upper)
        
        # 记录更新历史
        self.update_history.append({
            'timestamp': time.time(),
            'from_concept': from_concept,
            'to_concept': to_concept,
            'delta': delta
        })
    
    def get_weight_matrix_tensor(self) -> torch.Tensor:
        """获取权重矩阵的中点值作为PyTorch张量"""
        matrix = np.zeros((self.n_concepts, self.n_concepts))
        for i in range(self.n_concepts):
            for j in range(self.n_concepts):
                matrix[i, j] = self.weights[i, j].midpoint()
        return torch.FloatTensor(matrix)

class CognitiveDriftDetector:
    """思维漂移检测器"""
    
    def __init__(self, window_size: int = 50, threshold: float = 0.3):
        self.window_size = window_size
        self.threshold = threshold
        self.state_history = deque(maxlen=window_size)
        
    def detect_drift(self, current_state: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        检测认知状态的思维漂移
        
        Returns:
            drift_score: 漂移程度 [0,1]
            drift_direction: 漂移方向向量
        """
        if len(self.state_history) < 10:
            self.state_history.append(current_state)
            return 0.0, np.zeros_like(current_state)
        
        # 计算状态变化序列
        state_changes = []
        for i in range(1, len(self.state_history)):
            change = np.array(self.state_history[i]) - np.array(self.state_history[i-1])
            state_changes.append(change)
        
        # 指数平滑
        smoothed_changes = self._exponential_smoothing(state_changes, alpha=0.3)
        
        # 当前变化
        current_change = current_state - self.state_history[-1]
        
        # 预期变化
        expected_change = self._predict_next_change(smoothed_changes)
        
        # 漂移分数
        drift_vector = current_change - expected_change
        drift_score = np.linalg.norm(drift_vector) / self.threshold
        drift_score = min(drift_score, 1.0)
        
        # 更新历史
        self.state_history.append(current_state)
        
        return drift_score, drift_vector
    
    def _exponential_smoothing(self, changes: List[np.ndarray], alpha: float) -> List[np.ndarray]:
        """指数平滑处理"""
        if not changes:
            return []
        
        smoothed = [changes[0]]
        for i in range(1, len(changes)):
            smooth_change = alpha * changes[i] + (1 - alpha) * smoothed[-1]
            smoothed.append(smooth_change)
        
        return smoothed
    
    def _predict_next_change(self, smoothed_changes: List[np.ndarray]) -> np.ndarray:
        """预测下一个变化"""
        if len(smoothed_changes) < 3:
            return np.zeros_like(smoothed_changes[0]) if smoothed_changes else np.zeros(8)
        
        # 简单趋势预测
        recent_changes = smoothed_changes[-3:]
        trend = np.mean(recent_changes, axis=0)
        return trend

class CognitiveLoadAssessor:
    """认知负载评估器"""
    
    def __init__(self):
        self.assessment_history = deque(maxlen=20)
        
    def assess_cognitive_load(self, 
                            difcm_state: np.ndarray,
                            interaction_data: Dict,
                            time_window: int = 300) -> Tuple[float, Dict[str, float]]:
        """
        评估认知负载水平
        
        Args:
            difcm_state: DIFCM当前状态
            interaction_data: 交互数据
            time_window: 时间窗口(秒)
            
        Returns:
            cognitive_load: 认知负载值 [0,1]
            load_components: 负载分解
        """
        # 1. 思维复杂度指标
        complexity_score = self._calculate_thinking_complexity(difcm_state)
        
        # 2. 时间压力指标
        time_pressure = self._calculate_time_pressure(interaction_data, time_window)
        
        # 3. 信息超载指标
        info_overload = self._calculate_information_overload(interaction_data)
        
        # 4. 综合负载计算
        weights = [0.4, 0.3, 0.3]
        cognitive_load = (weights[0] * complexity_score + 
                         weights[1] * time_pressure + 
                         weights[2] * info_overload)
        
        load_components = {
            'thinking_complexity': complexity_score,
            'time_pressure': time_pressure,
            'information_overload': info_overload
        }
        
        # 记录历史
        self.assessment_history.append({
            'timestamp': time.time(),
            'cognitive_load': cognitive_load,
            'components': load_components
        })
        
        return cognitive_load, load_components
    
    def _calculate_thinking_complexity(self, state: np.ndarray) -> float:
        """计算思维复杂度"""
        # 基于状态向量的方差
        complexity = np.var(state) * 2
        return min(complexity, 1.0)
    
    def _calculate_time_pressure(self, interaction_data: Dict, time_window: int) -> float:
        """计算时间压力"""
        response_times = interaction_data.get('response_times', [])
        if not response_times:
            return 0.5
        
        # 最近时间窗口内的响应时间趋势
        recent_times = response_times[-min(len(response_times), 10):]
        if len(recent_times) > 2:
            trend = np.polyfit(range(len(recent_times)), recent_times, 1)[0]
            pressure = max(0, min(1, trend / 10))  # 归一化
        else:
            pressure = 0.5
            
        return pressure
    
    def _calculate_information_overload(self, interaction_data: Dict) -> float:
        """计算信息超载"""
        keystrokes = interaction_data.get('keystrokes', [])
        if not keystrokes:
            return 0.5
        
        # 基于击键复杂度和频率
        key_diversity = len(set([k.get('key', '') for k in keystrokes]))
        total_keys = len(keystrokes)
        
        if total_keys > 0:
            diversity_ratio = key_diversity / total_keys
            # 过高的多样性可能表示信息超载
            overload = max(0, min(1, (diversity_ratio - 0.3) / 0.4))
        else:
            overload = 0.5
            
        return overload

class IndustrialDataQualityAssessor:
    """工业数据质量评估器"""
    
    def __init__(self):
        self.quality_thresholds = {
            'completeness': 0.8,
            'consistency': 0.75,
            'freshness': 0.9,
            'accuracy': 0.85
        }
        self.quality_history = deque(maxlen=50)
    
    def assess_data_quality(self, industrial_data: Dict[str, Any]) -> Dict[str, float]:
        """评估工业数据质量"""
        quality_scores = {}
        
        # 1. 完整性评估
        quality_scores['completeness'] = self._assess_completeness(industrial_data)
        
        # 2. 一致性评估
        quality_scores['consistency'] = self._assess_consistency(industrial_data)
        
        # 3. 新鲜度评估
        quality_scores['freshness'] = self._assess_freshness(industrial_data)
        
        # 4. 准确性评估（基于已知模式）
        quality_scores['accuracy'] = self._assess_accuracy(industrial_data)
        
        # 总体质量评分
        overall_quality = np.mean(list(quality_scores.values()))
        quality_scores['overall'] = overall_quality
        
        # 记录质量历史
        self.quality_history.append({
            'timestamp': time.time(),
            'scores': quality_scores.copy()
        })
        
        return quality_scores
    
    def _assess_completeness(self, data: Dict[str, Any]) -> float:
        """评估数据完整性"""
        expected_fields = ['text_features', 'behavior_features', 'emotion_features', 'temporal_features']
        present_fields = sum(1 for field in expected_fields if field in data and data[field] is not None)
        return present_fields / len(expected_fields)
    
    def _assess_consistency(self, data: Dict[str, Any]) -> float:
        """评估数据一致性"""
        # 检查数据格式和值范围的一致性
        consistency_score = 1.0
        
        # 检查文本特征维度
        if 'text_features' in data:
            text_feat = data['text_features']
            if hasattr(text_feat, 'shape') and text_feat.shape[-1] != 768:
                consistency_score -= 0.3
        
        # 检查情感特征维度
        if 'emotion_features' in data:
            emotion_feat = data['emotion_features']
            if hasattr(emotion_feat, 'shape') and emotion_feat.shape[-1] != 7:
                consistency_score -= 0.2
        
        return max(0.0, consistency_score)
    
    def _assess_freshness(self, data: Dict[str, Any]) -> float:
        """评估数据新鲜度"""
        current_time = time.time()
        data_timestamp = data.get('timestamp', current_time)
        
        time_diff = current_time - data_timestamp
        # 数据在5分钟内为新鲜，超过30分钟认为过时
        if time_diff < 300:  # 5分钟
            return 1.0
        elif time_diff < 1800:  # 30分钟
            return 1.0 - (time_diff - 300) / 1500
        else:
            return 0.1  # 保留最小分数
    
    def _assess_accuracy(self, data: Dict[str, Any]) -> float:
        """评估数据准确性（基于已知模式）"""
        accuracy_score = 0.8  # 基准分数
        
        # 检查数值范围的合理性
        if 'behavior_features' in data:
            behavior_feat = data['behavior_features']
            if hasattr(behavior_feat, 'min') and hasattr(behavior_feat, 'max'):
                if behavior_feat.min() < 0 or behavior_feat.max() > 1:
                    accuracy_score -= 0.2
        
        # 检查情感特征的概率分布
        if 'emotion_features' in data:
            emotion_feat = data['emotion_features']
            if hasattr(emotion_feat, 'sum') and abs(emotion_feat.sum() - 1.0) > 0.1:
                accuracy_score -= 0.1
        
        return max(0.1, accuracy_score)
    
    def get_quality_recommendations(self, quality_scores: Dict[str, float]) -> List[str]:
        """基于质量评估提供改进建议"""
        recommendations = []
        
        for metric, score in quality_scores.items():
            if metric == 'overall':
                continue
                
            threshold = self.quality_thresholds.get(metric, 0.8)
            if score < threshold:
                if metric == 'completeness':
                    recommendations.append(f"数据完整性不足({score:.2f}): 检查数据源连接和字段映射")
                elif metric == 'consistency':
                    recommendations.append(f"数据一致性问题({score:.2f}): 验证数据格式和预处理流程")
                elif metric == 'freshness':
                    recommendations.append(f"数据新鲜度低({score:.2f}): 增加数据更新频率")
                elif metric == 'accuracy':
                    recommendations.append(f"数据准确性待提升({score:.2f}): 检查数据清洗和验证规则")
        
        return recommendations

class EnhancedDIFCMModel(nn.Module):
    """增强版DIFCM模型 - 集成工业级数据处理"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 写作思维能力8维概念定义（基于权威教育心理学研究）
        self.concept_names = [
            'depth_thinking',      # 深刻性：主题本质挖掘、论证周密性
            'flexible_thinking',   # 灵活性：多角度思维、辩证表达
            'critical_thinking',   # 批判性：错误识别、自我评价
            'originality',         # 独创性：新颖独特程度
            'fluency',            # 流畅性：思维表达连贯性
            'motivation',         # 动机水平：写作驱动力
            'emotion_regulation', # 情绪调节：情感状态管理
            'cognitive_load'      # 认知负载：思维负担程度
        ]
        
        # 概念状态（区间值）
        self.concept_states = {
            name: IntervalValue(0.5, 0.5) for name in self.concept_names
        }
        
        # 动态权重矩阵
        self.dynamic_weights = DynamicWeightMatrix(
            config.n_concepts, 
            config.learning_rate
        )
        
        # 工业数据适配层 - 处理多源异构数据
        self.industrial_adapter = nn.ModuleDict({
            'text_encoder': nn.Sequential(
                nn.Linear(768, 256),  # BERT embeddings -> compressed
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 64)
            ),
            'behavior_encoder': nn.Sequential(
                nn.Linear(50, 32),    # 行为特征
                nn.ReLU(),
                nn.Linear(32, 16)
            ),
            'emotion_encoder': nn.Sequential(
                nn.Linear(7, 16),     # 情感分类结果
                nn.ReLU(),
                nn.Linear(16, 8)
            ),
            'temporal_encoder': nn.Sequential(
                nn.Linear(10, 16),    # 时间序列特征
                nn.ReLU(),
                nn.Linear(16, 8)
            )
        })
        
        # 多模态融合层
        total_encoded_dim = 64 + 16 + 8 + 8  # 96
        self.multimodal_fusion = nn.Sequential(
            nn.Linear(total_encoded_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, config.n_concepts)
        )
        
        # 工业数据质量评估
        self.data_quality_assessor = IndustrialDataQualityAssessor()
        
        # 思维漂移检测器
        self.drift_detector = CognitiveDriftDetector()
        
        # 认知负载评估器
        self.load_assessor = CognitiveLoadAssessor()
        
        # 历史状态
        self.state_history = deque(maxlen=100)
        
        logging.info("增强版DIFCM模型初始化完成 - 支持工业级数据")
    
    def forward(self, input_data: Dict[str, torch.Tensor], 
                interaction_data: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """
        增强版前向传播 - 支持工业级多模态数据
        
        Args:
            input_data: 工业数据字典 {
                'text_features': torch.Tensor [batch_size, 768],
                'behavior_features': torch.Tensor [batch_size, 50],
                'emotion_features': torch.Tensor [batch_size, 7],
                'temporal_features': torch.Tensor [batch_size, 10]
            }
            interaction_data: 交互数据字典
            
        Returns:
            包含认知状态、数据质量、漂移检测、负载评估的结果字典
        """
        batch_size = input_data['text_features'].size(0) if 'text_features' in input_data else 1
        
        # 1. 工业数据质量评估
        quality_scores = self.data_quality_assessor.assess_data_quality(
            {k: v.detach().numpy() if hasattr(v, 'detach') else v for k, v in input_data.items()}
        )
        
        # 2. 工业数据编码 - 多模态特征提取
        encoded_features = self._encode_industrial_data(input_data)
        
        # 3. 数据质量权重调制
        quality_weight = min(1.0, quality_scores['overall'] + 0.2)  # 保证最小0.2权重
        encoded_features = encoded_features * quality_weight
        
        # 4. 多模态融合到概念空间
        concept_activations = self.multimodal_fusion(encoded_features)
        concept_activations = torch.sigmoid(concept_activations)
        
        # 5. 获取当前权重矩阵和状态
        weight_matrix = self.dynamic_weights.get_weight_matrix_tensor()
        current_state = self._get_current_state_tensor(batch_size)
        
        # 6. DIFCM核心：概念间动态影响计算
        concept_influences = torch.matmul(current_state, weight_matrix)
        
        # 7. 区间值融合的状态更新
        alpha = 0.6  # 平衡新输入和历史状态
        new_state = torch.sigmoid(
            alpha * concept_activations + 
            (1 - alpha) * current_state + 
            0.2 * concept_influences
        )
        new_state = torch.clamp(new_state, 0.01, 0.99)
        
        # 8. 思维漂移检测（工业数据增强）
        drift_results = self._detect_drift_batch_enhanced(
            new_state.detach().numpy(), quality_scores
        )
        
        # 9. 认知负载评估（工业数据上下文）
        load_results = self._assess_load_batch_enhanced(
            new_state.detach().numpy(), interaction_data, quality_scores
        )
        
        # 10. 更新概念状态
        self._update_concept_states(new_state[0].detach().numpy())
        
        return {
            'cognitive_state': new_state,
            'data_quality': quality_scores,
            'quality_recommendations': self.data_quality_assessor.get_quality_recommendations(quality_scores),
            'drift_scores': torch.FloatTensor([r[0] for r in drift_results]),
            'drift_directions': torch.FloatTensor([r[1] for r in drift_results]),
            'cognitive_loads': torch.FloatTensor([r[0] for r in load_results]),
            'load_components': [r[1] for r in load_results],
            'encoded_features': encoded_features,
            'concept_activations': concept_activations
        }
    
    def _encode_industrial_data(self, input_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """工业数据多模态编码"""
        encoded_components = []
        
        # 文本特征编码
        if 'text_features' in input_data:
            text_encoded = self.industrial_adapter['text_encoder'](input_data['text_features'])
            encoded_components.append(text_encoded)
        else:
            # 填充零特征
            batch_size = next(iter(input_data.values())).size(0)
            text_encoded = torch.zeros(batch_size, 64)
            encoded_components.append(text_encoded)
        
        # 行为特征编码
        if 'behavior_features' in input_data:
            behavior_encoded = self.industrial_adapter['behavior_encoder'](input_data['behavior_features'])
            encoded_components.append(behavior_encoded)
        else:
            batch_size = next(iter(input_data.values())).size(0)
            behavior_encoded = torch.zeros(batch_size, 16)
            encoded_components.append(behavior_encoded)
        
        # 情感特征编码
        if 'emotion_features' in input_data:
            emotion_encoded = self.industrial_adapter['emotion_encoder'](input_data['emotion_features'])
            encoded_components.append(emotion_encoded)
        else:
            batch_size = next(iter(input_data.values())).size(0)
            emotion_encoded = torch.zeros(batch_size, 8)
            encoded_components.append(emotion_encoded)
        
        # 时间特征编码
        if 'temporal_features' in input_data:
            temporal_encoded = self.industrial_adapter['temporal_encoder'](input_data['temporal_features'])
            encoded_components.append(temporal_encoded)
        else:
            batch_size = next(iter(input_data.values())).size(0)
            temporal_encoded = torch.zeros(batch_size, 8)
            encoded_components.append(temporal_encoded)
        
        # 拼接所有编码特征
        concatenated_features = torch.cat(encoded_components, dim=1)
        return concatenated_features
    
    def _detect_drift_batch_enhanced(self, states: np.ndarray, 
                                   quality_scores: Dict[str, float]) -> List[Tuple[float, np.ndarray]]:
        """增强版批量思维漂移检测 - 考虑数据质量"""
        results = []
        quality_factor = quality_scores.get('overall', 0.8)
        
        for state in states:
            drift_score, drift_direction = self.drift_detector.detect_drift(state)
            
            # 数据质量低时，降低漂移检测的敏感度
            if quality_factor < 0.6:
                drift_score *= 0.7  # 降低漂移分数
            elif quality_factor < 0.8:
                drift_score *= 0.85
            
            results.append((drift_score, drift_direction))
        return results
    
    def _assess_load_batch_enhanced(self, states: np.ndarray, 
                                  interaction_data: Optional[Dict],
                                  quality_scores: Dict[str, float]) -> List[Tuple[float, Dict]]:
        """增强版批量认知负载评估 - 工业数据上下文"""
        results = []
        quality_factor = quality_scores.get('overall', 0.8)
        
        for state in states:
            if interaction_data:
                load, components = self.load_assessor.assess_cognitive_load(
                    state, interaction_data
                )
                
                # 基于数据质量调整负载评估
                if quality_factor < 0.7:
                    # 数据质量低可能导致额外的认知负载
                    load = min(1.0, load + (0.7 - quality_factor) * 0.5)
                    components['data_quality_stress'] = (0.7 - quality_factor) * 0.5
                
            else:
                load, components = 0.5, {'default_load': 0.5}
            
            results.append((load, components))
        return results
    
    def _get_current_state_tensor(self, batch_size: int) -> torch.Tensor:
        """获取当前状态张量"""
        state_vector = np.array([
            self.concept_states[name].midpoint() 
            for name in self.concept_names
        ])
        return torch.FloatTensor(state_vector).unsqueeze(0).repeat(batch_size, 1)
    
    def _detect_drift_batch(self, states: np.ndarray) -> List[Tuple[float, np.ndarray]]:
        """批量思维漂移检测"""
        results = []
        for state in states:
            drift_score, drift_direction = self.drift_detector.detect_drift(state)
            results.append((drift_score, drift_direction))
        return results
    
    def _assess_load_batch(self, states: np.ndarray, 
                          interaction_data: Optional[Dict]) -> List[Tuple[float, Dict]]:
        """批量认知负载评估"""
        results = []
        for state in states:
            if interaction_data:
                load, components = self.load_assessor.assess_cognitive_load(
                    state, interaction_data
                )
            else:
                load, components = 0.5, {}
            results.append((load, components))
        return results
    
    def _update_concept_states(self, new_state: np.ndarray):
        """更新概念状态"""
        for i, name in enumerate(self.concept_names):
            old_value = self.concept_states[name].midpoint()
            new_value = new_state[i]
            
            # 自适应区间更新
            uncertainty = abs(new_value - old_value)
            lower_bound = new_value - uncertainty * 0.1
            upper_bound = new_value + uncertainty * 0.1
            
            self.concept_states[name] = IntervalValue(
                max(0.01, lower_bound),
                min(0.99, upper_bound)
            )
        
        # 记录状态历史
        self.state_history.append(new_state.copy())
    
    def update_weights_from_feedback(self, feedback_data: Dict):
        """基于用户反馈更新权重"""
        if 'concept_correlations' in feedback_data:
            correlations = feedback_data['concept_correlations']
            for (from_idx, to_idx), correlation in correlations.items():
                # 将相关性转换为权重更新
                delta = correlation * 0.1  # 调节更新幅度
                self.dynamic_weights.update_weights(from_idx, to_idx, delta)
    
    def get_interpretation(self) -> Dict[str, str]:
        """获取当前认知状态解释"""
        interpretation = {}
        for name, interval_value in self.concept_states.items():
            value = interval_value.midpoint()
            if value > 0.7:
                level = "高"
            elif value > 0.4:
                level = "中"
            else:
                level = "低"
            interpretation[name] = level
        return interpretation
    
    def save_enhanced_state(self, filepath: str):
        """保存增强模型状态"""
        state_dict = {
            'concept_states': {
                name: (iv.lower, iv.upper) 
                for name, iv in self.concept_states.items()
            },
            'weight_matrix': self.dynamic_weights.weights,
            'state_history': list(self.state_history),
            'config': self.config.__dict__
        }
        np.save(filepath, state_dict)
        logging.info(f"增强模型状态已保存到: {filepath}")