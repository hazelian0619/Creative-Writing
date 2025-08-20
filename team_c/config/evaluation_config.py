"""
Team C 评估与验证配置
与团队A和团队B的配置保持一致
"""

from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import os

@dataclass
class EvaluationConfig:
    """评估配置"""
    
    # 数据集配置
    dataset_size: int = 1000
    validation_split: float = 0.2
    test_split: float = 0.2
    synthetic_data_ratio: float = 0.7  # 合成数据占比
    
    # 准确率计算配置
    accuracy_threshold: float = 0.75  # Phase1目标准确率
    tolerance: float = 0.1  # 认知状态预测容差
    concept_weights: Dict[str, float] = None
    
    # 性能测试配置
    latency_target: float = 0.5  # 500ms目标响应时间
    concurrent_users: int = 20
    test_duration: int = 300  # 5分钟测试
    
    # A/B测试配置
    ab_test_groups: List[str] = None
    statistical_significance: float = 0.05
    minimum_sample_size: int = 100
    
    # 路径配置
    base_dir: Path = Path("/Users/pluviophile/chi2025/team_c")
    data_dir: Path = None
    results_dir: Path = None
    models_dir: Path = None
    
    def __post_init__(self):
        """初始化默认值"""
        if self.concept_weights is None:
            # 与团队A的8维认知概念对齐
            self.concept_weights = {
                'attention': 0.15,      # 注意力
                'memory': 0.15,         # 工作记忆  
                'comprehension': 0.15,  # 理解力
                'creativity': 0.20,     # 创造力 (重点)
                'motivation': 0.10,     # 动机水平
                'emotion': 0.10,        # 情绪状态
                'confidence': 0.10,     # 自信心
                'fatigue': 0.05         # 疲劳度
            }
        
        if self.ab_test_groups is None:
            self.ab_test_groups = ['control', 'experimental']
            
        # 设置路径
        if self.data_dir is None:
            self.data_dir = self.base_dir / "datasets" / "data"
        if self.results_dir is None:
            self.results_dir = self.base_dir / "evaluation" / "results"
        if self.models_dir is None:
            self.models_dir = Path("/Users/pluviophile/chi2025/team_a/checkpoints")
            
        # 创建目录
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

@dataclass 
class APITestConfig:
    """API测试配置，与团队B对齐"""
    
    # API服务配置 - 与team_b/api/config.py保持一致
    api_host: str = "localhost"
    api_port: int = 8000
    api_timeout: float = 30.0
    
    # 认知状态API端点
    behavior_data_endpoint: str = "/api/v1/behavior-data"
    latest_state_endpoint: str = "/api/v1/latest-state"
    health_check_endpoint: str = "/health"
    
    # 请求配置
    max_retries: int = 3
    retry_delay: float = 1.0
    request_timeout: float = 10.0

@dataclass
class CognitiveStateSpec:
    """认知状态规格，与团队A的DIFCM模型完全对齐"""
    
    # 概念定义 - 必须与team_a/models/difcm_core.py一致
    concept_names: List[str] = None
    n_concepts: int = 8
    value_range: tuple = (0.1, 0.9)  # 与DIFCM的clamp范围一致
    
    # 特征定义 - 与team_a/features/cognitive_features.py对齐
    feature_names: List[str] = None
    
    def __post_init__(self):
        if self.concept_names is None:
            self.concept_names = [
                'attention',      # 注意力
                'memory',         # 工作记忆
                'comprehension',  # 理解力
                'creativity',     # 创造力
                'motivation',     # 动机水平
                'emotion',        # 情绪状态
                'confidence',     # 自信心
                'fatigue'         # 疲劳度
            ]
        
        if self.feature_names is None:
            self.feature_names = [
                'attention', 'memory', 'comprehension', 'creativity',
                'motivation', 'emotion', 'confidence', 'fatigue'
            ]

class TestScenarios:
    """测试场景定义"""
    
    # 认知状态测试场景
    COGNITIVE_SCENARIOS = {
        'high_attention': {
            'description': '高注意力状态测试',
            'expected_state': {
                'attention': 0.8, 'memory': 0.7, 'comprehension': 0.8,
                'creativity': 0.6, 'motivation': 0.8, 'emotion': 0.7,
                'confidence': 0.8, 'fatigue': 0.3
            }
        },
        'low_attention': {
            'description': '低注意力状态测试',  
            'expected_state': {
                'attention': 0.3, 'memory': 0.4, 'comprehension': 0.3,
                'creativity': 0.3, 'motivation': 0.4, 'emotion': 0.5,
                'confidence': 0.4, 'fatigue': 0.8
            }
        },
        'high_creativity': {
            'description': '高创造力状态测试',
            'expected_state': {
                'attention': 0.6, 'memory': 0.7, 'comprehension': 0.8,
                'creativity': 0.9, 'motivation': 0.8, 'emotion': 0.7,
                'confidence': 0.9, 'fatigue': 0.2
            }
        },
        'balanced_state': {
            'description': '平衡认知状态测试',
            'expected_state': {
                'attention': 0.6, 'memory': 0.6, 'comprehension': 0.6,
                'creativity': 0.6, 'motivation': 0.6, 'emotion': 0.6,
                'confidence': 0.6, 'fatigue': 0.4
            }
        }
    }
    
    # 性能测试场景
    PERFORMANCE_SCENARIOS = {
        'light_load': {
            'concurrent_users': 5,
            'requests_per_user': 10,
            'duration': 60
        },
        'normal_load': {
            'concurrent_users': 20,
            'requests_per_user': 20,
            'duration': 300
        },
        'stress_load': {
            'concurrent_users': 50,
            'requests_per_user': 50,
            'duration': 600
        }
    }

def get_evaluation_config() -> EvaluationConfig:
    """获取评估配置实例"""
    return EvaluationConfig()

def get_api_test_config() -> APITestConfig:
    """获取API测试配置实例"""
    return APITestConfig()

def get_cognitive_state_spec() -> CognitiveStateSpec:
    """获取认知状态规格实例"""
    return CognitiveStateSpec()