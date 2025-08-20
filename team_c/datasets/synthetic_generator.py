"""
合成数据生成器
为A/B测试和压力测试生成大量合成数据
"""

import numpy as np
import random
import json
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

from team_c.config import EvaluationConfig, CognitiveStateSpec

logger = logging.getLogger(__name__)

@dataclass
class SyntheticPattern:
    """合成数据模式"""
    name: str
    cognitive_state_range: Dict[str, Tuple[float, float]]
    behavior_characteristics: Dict[str, str]
    probability_weights: Dict[str, float]

class SyntheticDataGenerator:
    """
    合成数据生成器
    基于统计模型生成大量测试数据
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.spec = CognitiveStateSpec()
        
        # 定义合成数据模式
        self.patterns = {
            # 正常使用模式
            'normal_user': SyntheticPattern(
                name='正常用户',
                cognitive_state_range={
                    'attention': (0.4, 0.8),
                    'memory': (0.4, 0.8),
                    'comprehension': (0.5, 0.8),
                    'creativity': (0.3, 0.9),
                    'motivation': (0.4, 0.8),
                    'emotion': (0.3, 0.8),
                    'confidence': (0.3, 0.8),
                    'fatigue': (0.2, 0.7)
                },
                behavior_characteristics={
                    'keystroke_pattern': 'normal',
                    'mouse_pattern': 'normal',
                    'response_pattern': 'normal'
                },
                probability_weights={
                    'attention': 1.0,
                    'creativity': 1.2,  # 稍微偏向创造力
                    'motivation': 1.0
                }
            ),
            
            # 高表现用户
            'high_performer': SyntheticPattern(
                name='高表现用户',
                cognitive_state_range={
                    'attention': (0.7, 0.9),
                    'memory': (0.6, 0.9),
                    'comprehension': (0.7, 0.9),
                    'creativity': (0.6, 0.9),
                    'motivation': (0.7, 0.9),
                    'emotion': (0.6, 0.8),
                    'confidence': (0.7, 0.9),
                    'fatigue': (0.1, 0.4)
                },
                behavior_characteristics={
                    'keystroke_pattern': 'efficient',
                    'mouse_pattern': 'precise',
                    'response_pattern': 'fast'
                },
                probability_weights={
                    'attention': 1.5,
                    'creativity': 1.5,
                    'confidence': 1.3
                }
            ),
            
            # 疲劳用户
            'fatigued_user': SyntheticPattern(
                name='疲劳用户',
                cognitive_state_range={
                    'attention': (0.1, 0.4),
                    'memory': (0.2, 0.5),
                    'comprehension': (0.2, 0.5),
                    'creativity': (0.1, 0.4),
                    'motivation': (0.1, 0.4),
                    'emotion': (0.2, 0.6),
                    'confidence': (0.2, 0.5),
                    'fatigue': (0.6, 0.9)
                },
                behavior_characteristics={
                    'keystroke_pattern': 'slow',
                    'mouse_pattern': 'sluggish', 
                    'response_pattern': 'delayed'
                },
                probability_weights={
                    'fatigue': 2.0,
                    'attention': 0.5,
                    'motivation': 0.5
                }
            ),
            
            # 创造力爆发用户
            'creative_burst': SyntheticPattern(
                name='创造力爆发用户',
                cognitive_state_range={
                    'attention': (0.5, 0.8),
                    'memory': (0.5, 0.8),
                    'comprehension': (0.6, 0.9),
                    'creativity': (0.8, 0.9),
                    'motivation': (0.7, 0.9),
                    'emotion': (0.6, 0.9),
                    'confidence': (0.7, 0.9),
                    'fatigue': (0.1, 0.3)
                },
                behavior_characteristics={
                    'keystroke_pattern': 'burst',
                    'mouse_pattern': 'active',
                    'response_pattern': 'burst'
                },
                probability_weights={
                    'creativity': 3.0,
                    'motivation': 1.5,
                    'emotion': 1.3
                }
            ),
            
            # 困难用户
            'struggling_user': SyntheticPattern(
                name='困难用户',
                cognitive_state_range={
                    'attention': (0.2, 0.5),
                    'memory': (0.2, 0.5),
                    'comprehension': (0.1, 0.4),
                    'creativity': (0.1, 0.3),
                    'motivation': (0.2, 0.5),
                    'emotion': (0.2, 0.5),
                    'confidence': (0.1, 0.4),
                    'fatigue': (0.4, 0.8)
                },
                behavior_characteristics={
                    'keystroke_pattern': 'hesitant',
                    'mouse_pattern': 'erratic',
                    'response_pattern': 'inconsistent'
                },
                probability_weights={
                    'comprehension': 0.3,
                    'creativity': 0.3,
                    'confidence': 0.3
                }
            )
        }
        
        logger.info(f"合成数据生成器初始化完成，支持 {len(self.patterns)} 种用户模式")
    
    def generate_batch_data(self, 
                          batch_size: int,
                          pattern_distribution: Optional[Dict[str, float]] = None) -> List[Dict]:
        """
        批量生成合成数据
        
        Args:
            batch_size: 批次大小
            pattern_distribution: 用户模式分布 {pattern_name: probability}
        
        Returns:
            合成数据列表
        """
        if pattern_distribution is None:
            # 默认分布
            pattern_distribution = {
                'normal_user': 0.4,
                'high_performer': 0.2,
                'fatigued_user': 0.15,
                'creative_burst': 0.15,
                'struggling_user': 0.1
            }
        
        # 验证分布
        total_prob = sum(pattern_distribution.values())
        if abs(total_prob - 1.0) > 0.01:
            logger.warning(f"模式分布概率和不为1: {total_prob}")
            # 归一化
            pattern_distribution = {k: v/total_prob for k, v in pattern_distribution.items()}
        
        logger.info(f"开始生成 {batch_size} 个合成样本")
        
        synthetic_samples = []
        for i in range(batch_size):
            # 根据分布选择模式
            pattern_name = self._sample_pattern(pattern_distribution)
            pattern = self.patterns[pattern_name]
            
            # 生成样本
            sample = self._generate_single_synthetic_sample(pattern, f"synthetic_user_{i}")
            synthetic_samples.append(sample)
            
            if (i + 1) % 100 == 0:
                logger.info(f"已生成 {i + 1}/{batch_size} 个样本")
        
        logger.info(f"合成数据生成完成，总计 {len(synthetic_samples)} 个样本")
        return synthetic_samples
    
    def _sample_pattern(self, distribution: Dict[str, float]) -> str:
        """根据分布采样用户模式"""
        patterns = list(distribution.keys())
        probabilities = list(distribution.values())
        
        return np.random.choice(patterns, p=probabilities)
    
    def _generate_single_synthetic_sample(self, pattern: SyntheticPattern, user_id: str) -> Dict:
        """生成单个合成样本"""
        # 生成认知状态
        cognitive_state = {}
        for concept in self.spec.concept_names:
            if concept in pattern.cognitive_state_range:
                min_val, max_val = pattern.cognitive_state_range[concept]
                # 使用模式权重影响采样
                weight = pattern.probability_weights.get(concept, 1.0)
                
                # 加权采样 - 权重越高，越倾向于高值
                if weight > 1.0:
                    # Beta分布偏向高值
                    beta_val = np.random.beta(2 * weight, 2)
                elif weight < 1.0:
                    # Beta分布偏向低值
                    beta_val = np.random.beta(2, 2 / weight)
                else:
                    # 均匀分布
                    beta_val = np.random.uniform(0, 1)
                
                value = min_val + beta_val * (max_val - min_val)
                cognitive_state[concept] = max(0.1, min(0.9, value))
            else:
                # 默认值
                cognitive_state[concept] = np.random.uniform(0.3, 0.7)
        
        # 生成对应的行为数据
        behavior_data = self._generate_synthetic_behavior(pattern, cognitive_state)
        
        # 计算质量分数
        quality_score = self._estimate_quality_score(pattern, cognitive_state)
        
        return {
            'user_id': user_id,
            'session_id': f"session_{random.randint(1000, 9999)}",
            'timestamp': time.time() + random.uniform(-7200, 7200),  # ±2小时
            'pattern': pattern.name,
            'cognitive_state': cognitive_state,
            'behavior_data': behavior_data,
            'quality_score': quality_score,
            'synthetic': True
        }
    
    def _generate_synthetic_behavior(self, 
                                   pattern: SyntheticPattern, 
                                   cognitive_state: Dict[str, float]) -> Dict:
        """根据模式和认知状态生成行为数据"""
        behavior_chars = pattern.behavior_characteristics
        
        # 生成击键数据
        keystrokes = self._generate_pattern_keystrokes(
            behavior_chars['keystroke_pattern'], 
            cognitive_state
        )
        
        # 生成鼠标数据
        mouse_moves = self._generate_pattern_mouse_moves(
            behavior_chars['mouse_pattern'],
            cognitive_state
        )
        
        # 生成时间数据
        dwell_times = self._generate_pattern_dwell_times(
            behavior_chars['response_pattern'],
            cognitive_state
        )
        
        response_times = self._generate_pattern_response_times(
            behavior_chars['response_pattern'],
            cognitive_state
        )
        
        return {
            'keystrokes': keystrokes,
            'mouse_moves': mouse_moves,
            'dwell_times': dwell_times,
            'response_times': response_times,
            'task_context': {
                'pattern': pattern.name,
                'session_duration': max(response_times) if response_times else 60,
                'task_type': 'synthetic_creative_writing'
            }
        }
    
    def _generate_pattern_keystrokes(self, 
                                   pattern_type: str, 
                                   cognitive_state: Dict[str, float]) -> List[Dict]:
        """根据模式生成击键数据"""
        attention = cognitive_state.get('attention', 0.5)
        fatigue = cognitive_state.get('fatigue', 0.5)
        confidence = cognitive_state.get('confidence', 0.5)
        
        base_time = time.time()
        keystrokes = []
        
        if pattern_type == 'normal':
            key_count = random.randint(60, 100)
            interval_base = 0.18
            delete_ratio = 0.12
        elif pattern_type == 'efficient':
            key_count = random.randint(80, 120)
            interval_base = 0.12 / max(0.3, attention)  # 注意力越高越快
            delete_ratio = 0.05 / max(0.3, confidence)  # 自信越高删除越少
        elif pattern_type == 'slow':
            key_count = random.randint(30, 60)
            interval_base = 0.25 + fatigue * 0.3  # 疲劳影响速度
            delete_ratio = 0.25
        elif pattern_type == 'burst':
            key_count = random.randint(70, 130)
            interval_base = 0.08  # 爆发模式很快
            delete_ratio = 0.08
        elif pattern_type == 'hesitant':
            key_count = random.randint(40, 70)
            interval_base = 0.35
            delete_ratio = 0.35  # 犹豫用户删除很多
        else:
            key_count = random.randint(50, 80)
            interval_base = 0.20
            delete_ratio = 0.15
        
        current_time = base_time
        common_keys = ['a', 'e', 'i', 'o', 'u', 'n', 't', 'r', 's', 'l', 'd', 'c', 'm', 'h']
        
        for i in range(key_count):
            # 动态调整间隔
            if pattern_type == 'burst':
                # 爆发模式：短爆发 + 长间隔
                if i % 10 < 7:
                    interval = np.random.exponential(0.08)
                else:
                    interval = np.random.exponential(1.5)
            else:
                interval_std = interval_base * 0.3
                interval = max(0.05, np.random.normal(interval_base, interval_std))
            
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
    
    def _generate_pattern_mouse_moves(self,
                                    pattern_type: str,
                                    cognitive_state: Dict[str, float]) -> List[Dict]:
        """根据模式生成鼠标移动数据"""
        attention = cognitive_state.get('attention', 0.5)
        creativity = cognitive_state.get('creativity', 0.5)
        
        base_time = time.time()
        mouse_moves = []
        
        screen_width, screen_height = 1920, 1080
        x, y = screen_width // 2, screen_height // 2
        
        if pattern_type == 'normal':
            move_count = random.randint(50, 100)
            movement_variance = 50
        elif pattern_type == 'precise':
            move_count = random.randint(30, 70)
            movement_variance = 20 / max(0.3, attention)
        elif pattern_type == 'sluggish':
            move_count = random.randint(20, 50)
            movement_variance = 30
        elif pattern_type == 'active':
            move_count = random.randint(80, 150)
            movement_variance = 60 * creativity  # 创造力影响移动复杂度
        elif pattern_type == 'erratic':
            move_count = random.randint(100, 200)
            movement_variance = 100
        else:
            move_count = random.randint(40, 80)
            movement_variance = 40
        
        current_time = base_time
        
        for i in range(move_count):
            # 生成移动
            dx = np.random.normal(0, movement_variance)
            dy = np.random.normal(0, movement_variance)
            
            x = max(0, min(screen_width, x + dx))
            y = max(0, min(screen_height, y + dy))
            
            if pattern_type == 'sluggish':
                interval = np.random.exponential(0.2)
            elif pattern_type == 'active':
                interval = np.random.exponential(0.05)
            else:
                interval = np.random.exponential(0.1)
            
            current_time += interval
            
            mouse_moves.append({
                'x': int(x),
                'y': int(y),
                'timestamp': current_time,
                'type': 'mousemove'
            })
        
        return mouse_moves
    
    def _generate_pattern_dwell_times(self,
                                    pattern_type: str,
                                    cognitive_state: Dict[str, float]) -> List[float]:
        """根据模式生成停留时间"""
        comprehension = cognitive_state.get('comprehension', 0.5)
        creativity = cognitive_state.get('creativity', 0.5)
        
        if pattern_type == 'normal':
            count = random.randint(6, 12)
            base_time = 3.0
        elif pattern_type == 'fast':
            count = random.randint(8, 16)
            base_time = 1.5 / max(0.3, comprehension)
        elif pattern_type == 'delayed':
            count = random.randint(4, 8)
            base_time = 6.0
        elif pattern_type == 'burst':
            count = random.randint(10, 20)
            # 爆发模式：短停留 + 长思考
            dwell_times = []
            for _ in range(count):
                if random.random() < 0.3:
                    # 深度思考时间，受创造力影响
                    think_time = np.random.normal(15.0 * creativity, 5.0)
                    dwell_times.append(max(5.0, think_time))
                else:
                    # 快速响应
                    quick_time = np.random.normal(1.0, 0.3)
                    dwell_times.append(max(0.5, quick_time))
            return dwell_times
        elif pattern_type == 'inconsistent':
            count = random.randint(3, 10)
            base_time = 4.0
        else:
            count = random.randint(5, 10)
            base_time = 4.0
        
        dwell_times = []
        for _ in range(count):
            if pattern_type == 'inconsistent':
                # 不一致模式：大幅变动
                variation = np.random.uniform(0.5, 3.0)
                time_val = base_time * variation
            else:
                time_val = np.random.normal(base_time, base_time * 0.3)
            
            dwell_times.append(max(0.5, time_val))
        
        return dwell_times
    
    def _generate_pattern_response_times(self,
                                       pattern_type: str,
                                       cognitive_state: Dict[str, float]) -> List[float]:
        """根据模式生成响应时间"""
        attention = cognitive_state.get('attention', 0.5)
        motivation = cognitive_state.get('motivation', 0.5)
        
        if pattern_type == 'normal':
            count = random.randint(8, 15)
            base_time = 2.0
        elif pattern_type == 'fast':
            count = random.randint(10, 20)
            base_time = 1.0 / max(0.3, (attention + motivation) / 2)
        elif pattern_type == 'delayed':
            count = random.randint(5, 10)
            base_time = 4.0
        elif pattern_type == 'burst':
            count = random.randint(12, 25)
            response_times = []
            for _ in range(count):
                if random.random() < 0.4:
                    # 快速爆发
                    fast_time = np.random.normal(0.6, 0.2)
                    response_times.append(max(0.3, fast_time))
                else:
                    # 正常响应
                    normal_time = np.random.normal(2.0, 0.8)
                    response_times.append(max(0.5, normal_time))
            return response_times
        elif pattern_type == 'inconsistent':
            count = random.randint(5, 12)
            base_time = 3.0
        else:
            count = random.randint(6, 12)
            base_time = 2.5
        
        response_times = []
        for _ in range(count):
            if pattern_type == 'inconsistent':
                # 不一致：指数分布
                time_val = np.random.exponential(base_time)
            else:
                time_val = np.random.normal(base_time, base_time * 0.4)
            
            response_times.append(max(0.2, time_val))
        
        return response_times
    
    def _estimate_quality_score(self, 
                              pattern: SyntheticPattern, 
                              cognitive_state: Dict[str, float]) -> float:
        """估计样本质量分数"""
        # 基于模式一致性和认知状态合理性评分
        quality_factors = []
        
        # 1. 认知状态范围一致性
        range_consistency = 0
        valid_concepts = 0
        
        for concept, value in cognitive_state.items():
            if concept in pattern.cognitive_state_range:
                min_val, max_val = pattern.cognitive_state_range[concept]
                if min_val <= value <= max_val:
                    range_consistency += 1
                valid_concepts += 1
        
        if valid_concepts > 0:
            range_score = range_consistency / valid_concepts
            quality_factors.append(range_score)
        
        # 2. 认知状态内部一致性
        # 例如：高注意力 + 高创造力 + 低疲劳 是一致的
        attention = cognitive_state.get('attention', 0.5)
        creativity = cognitive_state.get('creativity', 0.5)
        fatigue = cognitive_state.get('fatigue', 0.5)
        confidence = cognitive_state.get('confidence', 0.5)
        
        # 计算内部一致性
        consistency_score = 0
        
        # 注意力与疲劳负相关
        attention_fatigue_consistency = 1 - abs((attention + fatigue) - 1.0)
        consistency_score += attention_fatigue_consistency
        
        # 创造力与自信正相关
        creativity_confidence_consistency = 1 - abs(creativity - confidence) / 2
        consistency_score += creativity_confidence_consistency
        
        internal_consistency = consistency_score / 2
        quality_factors.append(internal_consistency)
        
        # 3. 模式特征一致性
        pattern_score = 0
        if pattern.name == 'high_performer':
            # 高表现用户应该有高注意力、低疲劳
            if attention > 0.6 and fatigue < 0.5:
                pattern_score = 1.0
            else:
                pattern_score = 0.6
        elif pattern.name == 'fatigued_user':
            # 疲劳用户应该有高疲劳、低注意力
            if fatigue > 0.6 and attention < 0.5:
                pattern_score = 1.0
            else:
                pattern_score = 0.6
        elif pattern.name == 'creative_burst':
            # 创造力爆发用户应该有高创造力
            if creativity > 0.7:
                pattern_score = 1.0
            else:
                pattern_score = 0.7
        else:
            pattern_score = 0.8  # 其他模式默认分数
        
        quality_factors.append(pattern_score)
        
        # 综合质量分数
        final_quality = np.mean(quality_factors)
        return max(0.1, min(1.0, final_quality))
    
    def save_synthetic_data(self, samples: List[Dict], filename: str = None) -> Path:
        """保存合成数据"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"synthetic_data_{timestamp}.json"
        
        filepath = self.config.data_dir / filename
        
        # 构建数据字典
        data_dict = {
            'metadata': {
                'total_samples': len(samples),
                'patterns': list(self.patterns.keys()),
                'created_at': time.strftime("%Y-%m-%d %H:%M:%S"),
                'data_type': 'synthetic',
                'config': {
                    'concept_names': self.spec.concept_names,
                    'value_range': self.spec.value_range
                }
            },
            'samples': samples
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"合成数据已保存到: {filepath}")
        return filepath
    
    def get_pattern_statistics(self, samples: List[Dict]) -> Dict:
        """获取模式统计信息"""
        pattern_counts = {}
        pattern_quality = {}
        
        for sample in samples:
            pattern = sample.get('pattern', 'unknown')
            quality = sample.get('quality_score', 0.0)
            
            if pattern not in pattern_counts:
                pattern_counts[pattern] = 0
                pattern_quality[pattern] = []
            
            pattern_counts[pattern] += 1
            pattern_quality[pattern].append(quality)
        
        # 计算统计
        statistics = {}
        for pattern in pattern_counts:
            statistics[pattern] = {
                'count': pattern_counts[pattern],
                'percentage': pattern_counts[pattern] / len(samples) * 100,
                'avg_quality': np.mean(pattern_quality[pattern]),
                'quality_std': np.std(pattern_quality[pattern])
            }
        
        return statistics

def main():
    """测试合成数据生成器"""
    from team_c.config import get_evaluation_config
    
    config = get_evaluation_config()
    generator = SyntheticDataGenerator(config)
    
    # 生成测试数据
    samples = generator.generate_batch_data(100)
    
    # 获取统计信息
    stats = generator.get_pattern_statistics(samples)
    
    print("合成数据统计:")
    for pattern, stat in stats.items():
        print(f"  {pattern}: {stat['count']} 样本 ({stat['percentage']:.1f}%), 平均质量: {stat['avg_quality']:.3f}")
    
    # 保存数据
    filepath = generator.save_synthetic_data(samples, "test_synthetic.json")
    print(f"测试数据保存于: {filepath}")

if __name__ == "__main__":
    main()