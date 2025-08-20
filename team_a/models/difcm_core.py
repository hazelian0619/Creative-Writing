"""
Dynamic Interval Fuzzy Cognitive Map (DIFCM) Model
师范生创意写作认知状态追踪核心模型
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

@dataclass
class DIFCMConfig:
    """DIFCM模型配置"""
    n_concepts: int = 8
    learning_rate: float = 0.01
    decay_rate: float = 0.95
    influence_threshold: float = 0.1
    update_interval: int = 10
    device: str = 'cpu'

class DIFCMConcept:
    """DIFCM认知概念"""
    def __init__(self, name: str, initial_value: float = 0.5):
        self.name = name
        self.value = initial_value
        self.history = [initial_value]
        self.interval = [0.0, 1.0]
        
    def update_interval(self, new_value: float):
        """动态更新概念值区间"""
        self.history.append(new_value)
        if len(self.history) > 50:
            self.history = self.history[-50:]
            
        # 使用滑动窗口更新区间
        recent_values = self.history[-20:]
        self.interval = [
            np.percentile(recent_values, 10),
            np.percentile(recent_values, 90)
        ]
        
    def normalize_value(self, value: float) -> float:
        """基于动态区间归一化值"""
        if self.interval[1] - self.interval[0] < 0.01:
            return value
        return (value - self.interval[0]) / (self.interval[1] - self.interval[0])

class DIFCMModel(nn.Module):
    """
    动态区间模糊认知图模型
    用于师范生创意写作认知状态追踪
    """
    
    def __init__(self, config: DIFCMConfig):
        super(DIFCMModel, self).__init__()
        self.config = config
        
        # 写作思维能力8维概念定义（基于权威教育心理学研究）
        self.concepts = {
            'depth_thinking': DIFCMConcept('深刻性'),
            'flexible_thinking': DIFCMConcept('灵活性'),
            'critical_thinking': DIFCMConcept('批判性'),
            'originality': DIFCMConcept('独创性'),
            'fluency': DIFCMConcept('流畅性'),
            'motivation': DIFCMConcept('动机水平'),
            'emotion_regulation': DIFCMConcept('情绪调节'),
            'cognitive_load': DIFCMConcept('认知负载')
        }
        
        # 概念名称列表
        self.concept_names = list(self.concepts.keys())
        
        # 权重矩阵 (概念间影响关系)
        self.weight_matrix = nn.Parameter(
            torch.randn(config.n_concepts, config.n_concepts) * 0.1
        )
        
        # 动态阈值
        self.register_buffer('influence_threshold', 
                           torch.tensor(config.influence_threshold))
        
        # 历史状态记录
        self.state_history = []
        self.prediction_history = []
        
        logging.info(f"DIFCM模型初始化完成，概念数量: {config.n_concepts}")
        
    def forward(self, 
                input_features: torch.Tensor,
                current_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播计算认知状态
        
        Args:
            input_features: 输入特征 [batch_size, feature_dim]
            current_state: 当前认知状态 [batch_size, n_concepts]
            
        Returns:
            预测的认知状态 [batch_size, n_concepts]
        """
        batch_size = input_features.size(0)
        
        if current_state is None:
            current_state = torch.zeros(batch_size, self.config.n_concepts)
            
        # 特征到概念的映射
        feature_mapping = nn.Linear(input_features.size(-1), self.config.n_concepts)
        external_influence = torch.sigmoid(feature_mapping(input_features))
        
        # 概念间影响计算
        concept_influences = torch.matmul(current_state, self.weight_matrix)
        
        # 应用动态阈值过滤
        concept_influences = torch.where(
            torch.abs(concept_influences) > self.influence_threshold,
            concept_influences,
            torch.zeros_like(concept_influences)
        )
        
        # 状态更新
        new_state = torch.sigmoid(current_state + concept_influences + external_influence)
        
        # 确保值在合理范围内
        new_state = torch.clamp(new_state, 0.01, 0.99)
        
        return new_state
    
    def update_concepts(self, predicted_state: np.ndarray):
        """更新概念值和动态区间"""
        for i, concept_name in enumerate(self.concept_names):
            concept = self.concepts[concept_name]
            new_value = float(predicted_state[i])
            concept.update_interval(new_value)
            concept.value = new_value
    
    def get_state_vector(self) -> np.ndarray:
        """获取当前认知状态向量"""
        return np.array([concept.value for concept in self.concepts.values()])
    
    def get_normalized_state(self) -> np.ndarray:
        """获取归一化的认知状态"""
        state = self.get_state_vector()
        normalized = []
        for i, concept_name in enumerate(self.concept_names):
            concept = self.concepts[concept_name]
            normalized.append(concept.normalize_value(state[i]))
        return np.array(normalized)
    
    def save_state(self, filepath: str):
        """保存模型状态"""
        state_dict = {
            'weight_matrix': self.weight_matrix.data.cpu().numpy(),
            'concepts': {
                name: {
                    'value': concept.value,
                    'history': concept.history,
                    'interval': concept.interval
                }
                for name, concept in self.concepts.items()
            },
            'config': self.config.__dict__
        }
        np.save(filepath, state_dict)
        logging.info(f"模型状态已保存到: {filepath}")
    
    def load_state(self, filepath: str):
        """加载模型状态"""
        state_dict = np.load(filepath, allow_pickle=True).item()
        self.weight_matrix.data = torch.from_numpy(state_dict['weight_matrix'])
        
        for name, data in state_dict['concepts'].items():
            if name in self.concepts:
                self.concepts[name].value = data['value']
                self.concepts[name].history = data['history']
                self.concepts[name].interval = data['interval']
                
        logging.info(f"模型状态已从 {filepath} 加载")
    
    def get_concept_relationships(self) -> Dict[str, Dict[str, float]]:
        """获取概念间关系权重"""
        weights = self.weight_matrix.data.cpu().numpy()
        relationships = {}
        
        for i, from_concept in enumerate(self.concept_names):
            relationships[from_concept] = {}
            for j, to_concept in enumerate(self.concept_names):
                relationships[from_concept][to_concept] = float(weights[i, j])
                
        return relationships
    
    def __repr__(self) -> str:
        return f"DIFCMModel(n_concepts={self.config.n_concepts}, learning_rate={self.config.learning_rate})"

# 认知状态解释器
class CognitiveStateInterpreter:
    """认知状态解释器"""
    
    def __init__(self):
        self.interpretation_rules = {
            'attention': {
                'high': lambda x: x > 0.7,
                'medium': lambda x: 0.3 <= x <= 0.7,
                'low': lambda x: x < 0.3
            },
            'creativity': {
                'high': lambda x: x > 0.8,
                'medium': lambda x: 0.4 <= x <= 0.8,
                'low': lambda x: x < 0.4
            }
        }
    
    def interpret_state(self, state_vector: np.ndarray) -> Dict[str, str]:
        """解释认知状态"""
        concept_names = ['attention', 'memory', 'comprehension', 'creativity',
                        'motivation', 'emotion', 'confidence', 'fatigue']
        
        interpretation = {}
        for i, concept in enumerate(concept_names):
            value = state_vector[i]
            if concept in self.interpretation_rules:
                for level, rule in self.interpretation_rules[concept].items():
                    if rule(value):
                        interpretation[concept] = level
                        break
            else:
                interpretation[concept] = 'medium'
                
        return interpretation
    
    def generate_recommendation(self, state: Dict[str, str]) -> str:
        """基于认知状态生成建议"""
        if state.get('attention') == 'low':
            return "建议先进行注意力集中训练"
        elif state.get('creativity') == 'low':
            return "建议使用启发式写作技巧"
        else:
            return "当前认知状态良好，可以继续创作"