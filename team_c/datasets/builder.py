"""
评估数据集构建器
与团队A的DIFCM模型和特征提取器完全对齐
"""

import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from dataclasses import dataclass
import time
import random
from datetime import datetime, timedelta

# 导入团队A的模块以确保完全兼容
import sys
sys.path.append("/Users/pluviophile/chi2025")
from team_a.features.cognitive_features import CognitiveFeatureExtractor
from team_c.config import CognitiveStateSpec, EvaluationConfig

logger = logging.getLogger(__name__)

@dataclass
class LabeledSample:
    """标注样本"""
    user_id: str
    session_id: str
    timestamp: float
    raw_behavior: Dict
    cognitive_state: Dict[str, float]
    scenario: str
    quality_score: float

class CognitiveDatasetBuilder:
    """
    认知评估数据集构建器
    生成高质量的标注数据用于模型验证
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.spec = CognitiveStateSpec()
        self.feature_extractor = CognitiveFeatureExtractor()
        
        # 标注数据映射 - 基于专家知识和文献
        self.scenario_mappings = {
            'high_attention_focused': {
                'cognitive_state': [0.8, 0.7, 0.9, 0.6, 0.8, 0.7, 0.8, 0.3],
                'behavior_pattern': {
                    'keystroke_stability': 'high',
                    'mouse_consistency': 'high', 
                    'response_pattern': 'fast_consistent'
                }
            },
            'low_attention_distracted': {
                'cognitive_state': [0.3, 0.4, 0.2, 0.3, 0.2, 0.5, 0.3, 0.8],
                'behavior_pattern': {
                    'keystroke_stability': 'low',
                    'mouse_consistency': 'erratic',
                    'response_pattern': 'slow_inconsistent' 
                }
            },
            'high_creativity_flow': {
                'cognitive_state': [0.6, 0.7, 0.8, 0.9, 0.8, 0.7, 0.9, 0.2],
                'behavior_pattern': {
                    'keystroke_stability': 'medium',
                    'mouse_consistency': 'complex',
                    'response_pattern': 'burst_activity'
                }
            },
            'low_creativity_blocked': {
                'cognitive_state': [0.3, 0.4, 0.3, 0.2, 0.3, 0.4, 0.2, 0.7],
                'behavior_pattern': {
                    'keystroke_stability': 'low',
                    'mouse_consistency': 'minimal',
                    'response_pattern': 'slow_sparse'
                }
            },
            'balanced_normal': {
                'cognitive_state': [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.4],
                'behavior_pattern': {
                    'keystroke_stability': 'medium',
                    'mouse_consistency': 'medium',
                    'response_pattern': 'steady'
                }
            },
            'fatigue_tired': {
                'cognitive_state': [0.3, 0.3, 0.4, 0.3, 0.2, 0.4, 0.3, 0.9],
                'behavior_pattern': {
                    'keystroke_stability': 'declining',
                    'mouse_consistency': 'slow',
                    'response_pattern': 'delayed_increasing'
                }
            }
        }
        
        logger.info(f"数据集构建器初始化完成，目标样本数: {config.dataset_size}")
    
    def generate_synthetic_behavior(self, scenario: str, base_time: float = None) -> Dict:
        """
        生成符合认知状态的合成行为数据
        与团队A的特征提取器输入格式完全一致
        """
        if base_time is None:
            base_time = time.time()
            
        pattern = self.scenario_mappings[scenario]['behavior_pattern']
        
        # 生成击键数据
        keystrokes = self._generate_keystrokes(pattern, base_time)
        
        # 生成鼠标数据
        mouse_moves = self._generate_mouse_moves(pattern, base_time)
        
        # 生成停留时间
        dwell_times = self._generate_dwell_times(pattern)
        
        # 生成响应时间
        response_times = self._generate_response_times(pattern)
        
        return {
            'keystrokes': keystrokes,
            'mouse_moves': mouse_moves,
            'dwell_times': dwell_times,
            'response_times': response_times,
            'task_context': {
                'scenario': scenario,
                'session_duration': max(response_times) if response_times else 60,
                'task_type': 'creative_writing'
            }
        }
    
    def _generate_keystrokes(self, pattern: Dict, base_time: float) -> List[Dict]:
        """生成击键序列"""
        keystrokes = []
        current_time = base_time
        
        if pattern['keystroke_stability'] == 'high':
            # 高稳定性：规律的间隔，少量删除
            interval_mean, interval_std = 0.15, 0.03
            delete_ratio = 0.05
            key_count = random.randint(80, 120)
        elif pattern['keystroke_stability'] == 'medium':
            interval_mean, interval_std = 0.20, 0.08
            delete_ratio = 0.12
            key_count = random.randint(60, 100)
        elif pattern['keystroke_stability'] == 'low':
            interval_mean, interval_std = 0.35, 0.15
            delete_ratio = 0.25
            key_count = random.randint(40, 80)
        elif pattern['keystroke_stability'] == 'declining':
            # 疲劳模式：间隔逐渐增大
            interval_mean, interval_std = 0.18, 0.06
            delete_ratio = 0.15
            key_count = random.randint(50, 90)
        
        common_keys = ['a', 'e', 'i', 'o', 'u', 'n', 't', 'r', 's', 'l', 'd']
        special_keys = ['Space', 'Backspace', 'Delete', 'Enter']
        
        for i in range(key_count):
            # 疲劳模式的特殊处理
            if pattern['keystroke_stability'] == 'declining':
                interval_mean += 0.002  # 逐渐减慢
                interval_std += 0.001
            
            interval = max(0.05, np.random.normal(interval_mean, interval_std))
            current_time += interval
            
            # 选择按键
            if random.random() < delete_ratio:
                key = random.choice(['Backspace', 'Delete'])
            elif random.random() < 0.15:
                key = 'Space'
            elif random.random() < 0.05:
                key = 'Enter'
            else:
                key = random.choice(common_keys)
            
            keystrokes.append({
                'key': key,
                'timestamp': current_time,
                'type': 'keydown'
            })
        
        return keystrokes
    
    def _generate_mouse_moves(self, pattern: Dict, base_time: float) -> List[Dict]:
        """生成鼠标移动数据"""
        mouse_moves = []
        current_time = base_time
        
        # 模拟屏幕区域
        screen_width, screen_height = 1920, 1080
        x, y = screen_width // 2, screen_height // 2
        
        if pattern['mouse_consistency'] == 'high':
            move_count = random.randint(30, 60)
            velocity_var = 0.3
        elif pattern['mouse_consistency'] == 'medium':
            move_count = random.randint(50, 100)
            velocity_var = 0.6
        elif pattern['mouse_consistency'] == 'complex':
            move_count = random.randint(80, 150)
            velocity_var = 1.2
        elif pattern['mouse_consistency'] == 'erratic':
            move_count = random.randint(100, 200)
            velocity_var = 2.0
        elif pattern['mouse_consistency'] == 'minimal':
            move_count = random.randint(10, 30)
            velocity_var = 0.8
        elif pattern['mouse_consistency'] == 'slow':
            move_count = random.randint(20, 50)
            velocity_var = 0.4
        
        for i in range(move_count):
            # 生成移动
            dx = np.random.normal(0, 50 * velocity_var)
            dy = np.random.normal(0, 50 * velocity_var)
            
            x = max(0, min(screen_width, x + dx))
            y = max(0, min(screen_height, y + dy))
            
            interval = max(0.01, np.random.exponential(0.1))
            current_time += interval
            
            mouse_moves.append({
                'x': int(x),
                'y': int(y),
                'timestamp': current_time,
                'type': 'mousemove'
            })
        
        return mouse_moves
    
    def _generate_dwell_times(self, pattern: Dict) -> List[float]:
        """生成停留时间"""
        if pattern['response_pattern'] == 'fast_consistent':
            return [np.random.normal(2.0, 0.5) for _ in range(random.randint(8, 15))]
        elif pattern['response_pattern'] == 'slow_inconsistent':
            return [np.random.exponential(8.0) for _ in range(random.randint(3, 8))]
        elif pattern['response_pattern'] == 'burst_activity':
            # 创造力高：短暂停留 + 长时间思考
            dwell_times = []
            for _ in range(random.randint(10, 20)):
                if random.random() < 0.3:
                    dwell_times.append(np.random.normal(15.0, 5.0))  # 深度思考
                else:
                    dwell_times.append(np.random.normal(1.5, 0.3))   # 快速响应
            return dwell_times
        elif pattern['response_pattern'] == 'steady':
            return [np.random.normal(4.0, 1.0) for _ in range(random.randint(6, 12))]
        elif pattern['response_pattern'] == 'delayed_increasing':
            # 疲劳：停留时间逐渐增加
            base_time = 3.0
            dwell_times = []
            for i in range(random.randint(5, 10)):
                base_time += 0.5  # 逐渐增加
                dwell_times.append(max(1.0, np.random.normal(base_time, 1.0)))
            return dwell_times
        else:
            return [np.random.normal(5.0, 2.0) for _ in range(random.randint(4, 10))]
    
    def _generate_response_times(self, pattern: Dict) -> List[float]:
        """生成响应时间"""
        if pattern['response_pattern'] == 'fast_consistent':
            return [np.random.normal(1.2, 0.2) for _ in range(random.randint(10, 20))]
        elif pattern['response_pattern'] == 'slow_inconsistent':
            return [np.random.exponential(3.0) for _ in range(random.randint(5, 12))]
        elif pattern['response_pattern'] == 'burst_activity':
            response_times = []
            for _ in range(random.randint(8, 18)):
                if random.random() < 0.4:
                    response_times.append(np.random.normal(0.8, 0.2))  # 快速爆发
                else:
                    response_times.append(np.random.normal(2.5, 0.8))  # 正常响应
            return response_times
        elif pattern['response_pattern'] == 'steady':
            return [np.random.normal(2.0, 0.5) for _ in range(random.randint(8, 15))]
        elif pattern['response_pattern'] == 'delayed_increasing':
            # 疲劳：响应时间递增
            base_time = 1.5
            response_times = []
            for i in range(random.randint(6, 12)):
                base_time += 0.3
                response_times.append(max(0.5, np.random.normal(base_time, 0.5)))
            return response_times
        elif pattern['response_pattern'] == 'slow_sparse':
            return [np.random.exponential(5.0) for _ in range(random.randint(3, 8))]
        else:
            return [np.random.normal(3.0, 1.0) for _ in range(random.randint(5, 12))]
    
    def build_labeled_dataset(self, target_size: int = None) -> List[LabeledSample]:
        """构建标注数据集"""
        if target_size is None:
            target_size = self.config.dataset_size
            
        logger.info(f"开始构建标注数据集，目标大小: {target_size}")
        
        samples = []
        scenarios = list(self.scenario_mappings.keys())
        
        # 确保每种场景都有足够的样本
        samples_per_scenario = target_size // len(scenarios)
        extra_samples = target_size % len(scenarios)
        
        for i, scenario in enumerate(scenarios):
            scenario_samples = samples_per_scenario
            if i < extra_samples:
                scenario_samples += 1
                
            logger.info(f"生成场景 '{scenario}' 的 {scenario_samples} 个样本")
            
            for j in range(scenario_samples):
                sample = self._generate_single_sample(scenario, f"user_{i*1000+j}")
                samples.append(sample)
        
        # 随机打乱
        random.shuffle(samples)
        
        logger.info(f"数据集构建完成，总样本数: {len(samples)}")
        return samples
    
    def _generate_single_sample(self, scenario: str, user_id: str) -> LabeledSample:
        """生成单个标注样本"""
        session_id = f"session_{random.randint(1000, 9999)}"
        timestamp = time.time() + random.uniform(-86400, 86400)  # ±1天
        
        # 生成行为数据
        raw_behavior = self.generate_synthetic_behavior(scenario, timestamp)
        
        # 获取对应的认知状态
        cognitive_values = self.scenario_mappings[scenario]['cognitive_state']
        
        # 添加适当的噪声
        noisy_values = []
        for value in cognitive_values:
            noise = np.random.normal(0, 0.05)  # 5%标准差的噪声
            noisy_value = max(0.1, min(0.9, value + noise))
            noisy_values.append(noisy_value)
        
        cognitive_state = dict(zip(self.spec.concept_names, noisy_values))
        
        # 验证特征提取器兼容性
        try:
            extracted_features = self.feature_extractor.extract_all_features(raw_behavior)
            quality_score = self._calculate_quality_score(extracted_features, cognitive_state)
        except Exception as e:
            logger.warning(f"特征提取失败: {e}")
            quality_score = 0.5
        
        return LabeledSample(
            user_id=user_id,
            session_id=session_id,
            timestamp=timestamp,
            raw_behavior=raw_behavior,
            cognitive_state=cognitive_state,
            scenario=scenario,
            quality_score=quality_score
        )
    
    def _calculate_quality_score(self, features: Dict[str, float], 
                                cognitive_state: Dict[str, float]) -> float:
        """计算样本质量分数"""
        # 检查特征与认知状态的一致性
        consistency_scores = []
        
        for concept in self.spec.concept_names:
            if concept in features and concept in cognitive_state:
                feature_val = features[concept]
                state_val = cognitive_state[concept]
                
                # 计算一致性 (值越接近，一致性越高)
                consistency = 1.0 - abs(feature_val - state_val)
                consistency_scores.append(consistency)
        
        if not consistency_scores:
            return 0.5
            
        return np.mean(consistency_scores)
    
    def save_dataset(self, samples: List[LabeledSample], filename: str = None) -> Path:
        """保存数据集"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cognitive_dataset_{timestamp}.json"
        
        filepath = self.config.data_dir / filename
        
        # 转换为可序列化格式
        dataset_dict = {
            'metadata': {
                'total_samples': len(samples),
                'concept_names': self.spec.concept_names,
                'scenarios': list(self.scenario_mappings.keys()),
                'created_at': datetime.now().isoformat(),
                'config': {
                    'dataset_size': self.config.dataset_size,
                    'concept_weights': self.config.concept_weights
                }
            },
            'samples': []
        }
        
        for sample in samples:
            sample_dict = {
                'user_id': sample.user_id,
                'session_id': sample.session_id,
                'timestamp': sample.timestamp,
                'raw_behavior': sample.raw_behavior,
                'cognitive_state': sample.cognitive_state,
                'scenario': sample.scenario,
                'quality_score': sample.quality_score
            }
            dataset_dict['samples'].append(sample_dict)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"数据集已保存到: {filepath}")
        return filepath
    
    def load_dataset(self, filepath: Path) -> List[LabeledSample]:
        """加载数据集"""
        with open(filepath, 'r', encoding='utf-8') as f:
            dataset_dict = json.load(f)
        
        samples = []
        for sample_dict in dataset_dict['samples']:
            sample = LabeledSample(
                user_id=sample_dict['user_id'],
                session_id=sample_dict['session_id'],
                timestamp=sample_dict['timestamp'],
                raw_behavior=sample_dict['raw_behavior'],
                cognitive_state=sample_dict['cognitive_state'],
                scenario=sample_dict['scenario'],
                quality_score=sample_dict['quality_score']
            )
            samples.append(sample)
        
        logger.info(f"从 {filepath} 加载了 {len(samples)} 个样本")
        return samples
    
    def validate_dataset_quality(self, samples: List[LabeledSample]) -> Dict[str, float]:
        """验证数据集质量"""
        quality_metrics = {
            'average_quality_score': np.mean([s.quality_score for s in samples]),
            'scenario_balance': self._calculate_scenario_balance(samples),
            'cognitive_range_coverage': self._calculate_range_coverage(samples),
            'feature_extraction_success_rate': self._test_feature_extraction(samples)
        }
        
        logger.info("数据集质量验证结果:")
        for metric, value in quality_metrics.items():
            logger.info(f"  {metric}: {value:.3f}")
        
        return quality_metrics
    
    def _calculate_scenario_balance(self, samples: List[LabeledSample]) -> float:
        """计算场景平衡度"""
        scenario_counts = {}
        for sample in samples:
            scenario_counts[sample.scenario] = scenario_counts.get(sample.scenario, 0) + 1
        
        counts = list(scenario_counts.values())
        if not counts:
            return 0.0
            
        # 计算均匀度 (标准差越小越均匀)
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        
        if mean_count == 0:
            return 0.0
            
        balance_score = 1.0 - (std_count / mean_count)
        return max(0.0, balance_score)
    
    def _calculate_range_coverage(self, samples: List[LabeledSample]) -> float:
        """计算认知状态范围覆盖度"""
        all_values = []
        for sample in samples:
            all_values.extend(sample.cognitive_state.values())
        
        if not all_values:
            return 0.0
            
        min_val, max_val = min(all_values), max(all_values)
        target_range = self.spec.value_range[1] - self.spec.value_range[0]
        actual_range = max_val - min_val
        
        coverage = actual_range / target_range
        return min(1.0, coverage)
    
    def _test_feature_extraction(self, samples: List[LabeledSample]) -> float:
        """测试特征提取成功率"""
        success_count = 0
        
        for sample in samples:
            try:
                features = self.feature_extractor.extract_all_features(sample.raw_behavior)
                if len(features) == len(self.spec.feature_names):
                    success_count += 1
            except Exception:
                continue
        
        return success_count / len(samples) if samples else 0.0

def main():
    """测试数据集构建器"""
    from team_c.config import get_evaluation_config
    
    config = get_evaluation_config()
    builder = CognitiveDatasetBuilder(config)
    
    # 构建小规模测试数据集
    samples = builder.build_labeled_dataset(50)
    
    # 验证质量
    quality_metrics = builder.validate_dataset_quality(samples)
    
    # 保存数据集
    filepath = builder.save_dataset(samples, "test_dataset.json")
    
    print(f"测试数据集创建完成，保存于: {filepath}")
    print(f"质量指标: {quality_metrics}")

if __name__ == "__main__":
    main()