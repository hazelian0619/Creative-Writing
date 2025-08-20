"""
认知特征提取算法
从用户行为数据中提取8维认知特征
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler

class CognitiveFeatureExtractor:
    """
    认知特征提取器
    从用户行为数据中提取8个认知维度的特征
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        # 写作思维能力8维特征名称（基于权威教育心理学研究）
        self.feature_names = [
            'depth_thinking', 'flexible_thinking', 'critical_thinking', 'originality',
            'fluency', 'motivation', 'emotion_regulation', 'cognitive_load'
        ]
        
    def extract_all_features(self, raw_data: Dict) -> Dict[str, float]:
        """
        从原始行为数据提取所有认知特征
        
        Args:
            raw_data: 包含用户行为数据的字典
            {
                'keystrokes': List[Dict],      # 击键数据
                'mouse_moves': List[Dict],     # 鼠标移动数据
                'dwell_times': List[float],    # 停留时间
                'response_times': List[float], # 响应时间
                'task_context': Dict           # 任务上下文
            }
            
        Returns:
            8维认知特征向量
        """
        features = {}
        
        # 1. 深刻性特征 - 主题本质挖掘能力
        features['depth_thinking'] = self._extract_depth_thinking_features(raw_data)
        
        # 2. 灵活性特征 - 多角度辩证思维
        features['flexible_thinking'] = self._extract_flexible_thinking_features(raw_data)
        
        # 3. 批判性特征 - 错误识别与自我评价
        features['critical_thinking'] = self._extract_critical_thinking_features(raw_data)
        
        # 4. 独创性特征 - 新颖独特程度
        features['originality'] = self._extract_originality_features(raw_data)
        
        # 5. 流畅性特征 - 思维表达连贯性
        features['fluency'] = self._extract_fluency_features(raw_data)
        
        # 6. 动机水平特征 - 写作驱动力
        features['motivation'] = self._extract_motivation_features(raw_data)
        
        # 7. 情绪调节特征 - 情感状态管理
        features['emotion_regulation'] = self._extract_emotion_regulation_features(raw_data)
        
        # 8. 认知负载特征 - 思维负担程度
        features['cognitive_load'] = self._extract_cognitive_load_features(raw_data)
        
        return features
    
    def _extract_depth_thinking_features(self, data: Dict) -> float:
        """
        提取深刻性特征 - 主题本质挖掘、论证周密性
        通过分析思考停顿时间、修改深度等判断深度思维
        """
        dwell_times = data.get('dwell_times', [])
        keystrokes = data.get('keystrokes', [])
        task_context = data.get('task_context', {})
        
        if not dwell_times and not keystrokes:
            return 0.5
        
        # 1. 长时间思考停顿表示深度思考
        deep_thinking_score = 0.0
        if dwell_times:
            long_pauses = [t for t in dwell_times if t > 3.0]  # 3秒以上停顿
            deep_thinking_score = min(1.0, len(long_pauses) / max(len(dwell_times), 1) * 2)
        
        # 2. 删除重写模式表示深度思考
        revision_score = 0.0
        if keystrokes:
            delete_keys = [k for k in keystrokes if k.get('key') in ['Backspace', 'Delete']]
            revision_score = min(1.0, len(delete_keys) / max(len(keystrokes), 1) * 3)
        
        # 3. 任务复杂度调节
        complexity_factor = task_context.get('complexity', 0.5)
        
        # 综合深刻性评分
        depth_score = (deep_thinking_score * 0.4 + revision_score * 0.4 + complexity_factor * 0.2)
        return np.clip(depth_score, 0.0, 1.0)
    
    def _extract_flexible_thinking_features(self, data: Dict) -> float:
        """
        提取灵活性特征 - 多角度思维、辍证表达
        通过分析思维跳跃、多样性表达等判断灵活思维
        """
        keystrokes = data.get('keystrokes', [])
        mouse_moves = data.get('mouse_moves', [])
        
        if not keystrokes and not mouse_moves:
            return 0.5
        
        # 1. 击键节奏变化表示思维灵活
        rhythm_flexibility = 0.0
        if keystrokes and len(keystrokes) > 5:
            intervals = []
            for i in range(1, len(keystrokes)):
                interval = keystrokes[i]['timestamp'] - keystrokes[i-1]['timestamp']
                intervals.append(interval)
            
            # 节奏变化多样性
            rhythm_flexibility = min(1.0, np.std(intervals) / (np.mean(intervals) + 1e-6) * 0.5)
        
        # 2. 鼠标移动模式多样性
        movement_flexibility = 0.0
        if mouse_moves and len(mouse_moves) > 10:
            directions = []
            for i in range(1, len(mouse_moves)):
                dx = mouse_moves[i]['x'] - mouse_moves[i-1]['x']
                dy = mouse_moves[i]['y'] - mouse_moves[i-1]['y']
                if abs(dx) > 1 or abs(dy) > 1:  # 过滤微小移动
                    angle = np.arctan2(dy, dx)
                    directions.append(angle)
            
            if directions:
                # 方向多样性
                direction_changes = sum(1 for i in range(1, len(directions)) 
                                      if abs(directions[i] - directions[i-1]) > 0.5)
                movement_flexibility = min(1.0, direction_changes / len(directions) * 2)
        
        # 综合灵活性评分
        flexibility_score = (rhythm_flexibility * 0.6 + movement_flexibility * 0.4)
        return np.clip(flexibility_score, 0.0, 1.0)
            
            if velocities:
                mouse_consistency = 1.0 - np.std(velocities) / (np.mean(velocities) + 1e-6)
                attention_score = (attention_score + mouse_consistency) / 2
                
        return max(0.1, min(1.0, attention_score))
    
    def _extract_memory_features(self, data: Dict) -> float:
        """提取工作记忆特征"""
        response_times = data.get('response_times', [])
        dwell_times = data.get('dwell_times', [])
        
        if not response_times:
            return 0.5
            
        # 工作记忆负荷指标
        # 1. 响应时间的一致性（记忆检索效率）
        rt_consistency = 1.0 - np.std(response_times) / (np.mean(response_times) + 1e-6)
        
        # 2. 停留时间与响应时间的比例
        if dwell_times:
            dwell_rt_ratio = np.mean(dwell_times) / (np.mean(response_times) + 1e-6)
            memory_efficiency = max(0, 1.0 - abs(dwell_rt_ratio - 1.0))
        else:
            memory_efficiency = 0.5
            
        # 3. 工作记忆指
        memory_score = (rt_consistency + memory_efficiency) / 2
        
        return max(0.1, min(1.0, memory_score))
    
    def _extract_comprehension_features(self, data: Dict) -> float:
        """提取理解力特征"""
        keystrokes = data.get('keystrokes', [])
        dwell_times = data.get('dwell_times', [])
        
        if not keystrokes:
            return 0.5
            
        # 理解力指标
        # 1. 击键模式的复杂性（反映思考深度）
        key_sequences = [k['key'] for k in keystrokes]
        unique_keys = len(set(key_sequences))
        total_keys = len(key_sequences)
        
        if total_keys > 0:
            complexity = unique_keys / total_keys
        else:
            complexity = 0.5
            
        # 2. 停留时间的分布（反映理解过程）
        if dwell_times:
            dwell_variance = np.std(dwell_times)
            dwell_mean = np.mean(dwell_times)
            
            # 合理的停留时间变化反映深度理解
            if dwell_mean > 0:
                understanding_depth = min(1.0, dwell_variance / dwell_mean)
            else:
                understanding_depth = 0.5
        else:
            understanding_depth = 0.5
            
        comprehension_score = (complexity + understanding_depth) / 2
        
        return max(0.1, min(1.0, comprehension_score))
    
    def _extract_creativity_features(self, data: Dict) -> float:
        """提取创造力特征"""
        keystrokes = data.get('keystrokes', [])
        mouse_moves = data.get('mouse_moves', [])
        
        # 创造力指标
        creativity_indicators = []
        
        # 1. 击键多样性（创造性思维活跃度）
        if keystrokes:
            key_types = [k['key'] for k in keystrokes]
            key_entropy = self._calculate_entropy(key_types)
            creativity_indicators.append(key_entropy)
            
        # 2. 鼠标轨迹的复杂度（创造性探索行为）
        if mouse_moves and len(mouse_moves) > 10:
            path_complexity = self._calculate_path_complexity(mouse_moves)
            creativity_indicators.append(path_complexity)
            
        # 3. 行为模式的非线性程度
        if keystrokes:
            temporal_patterns = [k['timestamp'] for k in keystrokes]
            nonlinearity = self._calculate_nonlinearity(temporal_patterns)
            creativity_indicators.append(nonlinearity)
            
        if not creativity_indicators:
            return 0.5
            
        creativity_score = np.mean(creativity_indicators)
        return max(0.1, min(1.0, creativity_score))
    
    def _extract_motivation_features(self, data: Dict) -> float:
        """提取动机水平特征"""
        keystrokes = data.get('keystrokes', [])
        response_times = data.get('response_times', [])
        
        if not keystrokes:
            return 0.5
            
        # 动机指标
        # 1. 活动强度（击键频率）
        key_times = [k['timestamp'] for k in keystrokes]
        if len(key_times) > 1:
            duration = max(key_times) - min(key_times)
            activity_rate = len(keystrokes) / max(duration, 1)
            intensity = min(1.0, activity_rate / 10)  # 归一化
        else:
            intensity = 0.5
            
        # 2. 响应积极性（响应时间趋势）
        if response_times and len(response_times) > 5:
            # 响应时间递减表示积极性提高
            trend = np.polyfit(range(len(response_times)), response_times, 1)[0]
            responsiveness = max(0, min(1, -trend / 100))
        else:
            responsiveness = 0.5
            
        motivation_score = (intensity + responsiveness) / 2
        return max(0.1, min(1.0, motivation_score))
    
    def _extract_emotion_features(self, data: Dict) -> float:
        """提取情绪状态特征"""
        keystrokes = data.get('keystrokes', [])
        mouse_moves = data.get('mouse_moves', [])
        
        # 情绪指标 - 基于行为模式的情感分析
        emotion_indicators = []
        
        # 1. 击键模式的情绪特征
        if keystrokes:
            # 快速连续击键可能表示兴奋或焦虑
            key_times = [k['timestamp'] for k in keystrokes]
            if len(key_times) > 1:
                intervals = np.diff(key_times)
                rapid_typing = np.mean(intervals) < 0.1
                emotion_indicators.append(1.0 if rapid_typing else 0.5)
            
        # 2. 鼠标移动的情绪特征
        if mouse_moves:
            # 鼠标抖动可能表示紧张或兴奋
            mouse_velocities = []
            for i in range(1, len(mouse_moves)):
                dx = mouse_moves[i]['x'] - mouse_moves[i-1]['x']
                dy = mouse_moves[i]['y'] - mouse_moves[i-1]['y']
                velocity = np.sqrt(dx**2 + dy**2)
                mouse_velocities.append(velocity)
            
            if mouse_velocities:
                jitter = np.std(mouse_velocities) / (np.mean(mouse_velocities) + 1e-6)
                emotional_intensity = min(1.0, jitter / 5)
                emotion_indicators.append(emotional_intensity)
        
        if not emotion_indicators:
            return 0.5
            
        emotion_score = np.mean(emotion_indicators)
        return max(0.1, min(1.0, emotion_score))
    
    def _extract_confidence_features(self, data: Dict) -> float:
        """提取自信心特征"""
        keystrokes = data.get('keystrokes', [])
        mouse_moves = data.get('mouse_moves', [])
        
        # 自信心指标
        confidence_indicators = []
        
        # 1. 击键的确定性（删除键比例）
        if keystrokes:
            total_keys = len(keystrokes)
            delete_keys = len([k for k in keystrokes if k['key'] in ['Backspace', 'Delete']])
            
            if total_keys > 0:
                certainty = 1.0 - (delete_keys / total_keys)
                confidence_indicators.append(certainty)
            
        # 2. 鼠标移动的流畅性（路径效率）
        if mouse_moves and len(mouse_moves) > 5:
            path_efficiency = self._calculate_path_efficiency(mouse_moves)
            confidence_indicators.append(path_efficiency)
            
        if not confidence_indicators:
            return 0.5
            
        confidence_score = np.mean(confidence_indicators)
        return max(0.1, min(1.0, confidence_score))
    
    def _extract_fatigue_features(self, data: Dict) -> float:
        """提取疲劳度特征"""
        keystrokes = data.get('keystrokes', [])
        response_times = data.get('response_times', [])
        
        # 疲劳度指标
        fatigue_indicators = []
        
        # 1. 响应时间递增趋势
        if response_times and len(response_times) > 5:
            trend = np.polyfit(range(len(response_times)), response_times, 1)[0]
            fatigue_trend = max(0, min(1, trend / 50))
            fatigue_indicators.append(fatigue_trend)
        
        # 2. 击键频率递减
        if keystrokes:
            key_times = [k['timestamp'] for k in keystrokes]
            if len(key_times) > 10:
                # 计算前后半段的击键频率
                mid_point = len(key_times) // 2
                first_half_freq = mid_point / max(key_times[mid_point] - key_times[0], 1)
                second_half_freq = (len(key_times) - mid_point) / max(key_times[-1] - key_times[mid_point], 1)
                
                if second_half_freq > 0:
                    fatigue_ratio = max(0, min(1, 1 - second_half_freq / first_half_freq))
                    fatigue_indicators.append(fatigue_ratio)
        
        if not fatigue_indicators:
            return 0.5
            
        fatigue_score = np.mean(fatigue_indicators)
        return max(0.1, min(1.0, fatigue_score))
    
    def _calculate_entropy(self, items: List[str]) -> float:
        """计算信息熵"""
        if not items:
            return 0.0
            
        unique, counts = np.unique(items, return_counts=True)
        probabilities = counts / len(items)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        # 归一化到[0,1]
        max_entropy = np.log2(len(unique) + 1e-10)
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _calculate_path_complexity(self, mouse_moves: List[Dict]) -> float:
        """计算鼠标路径复杂度"""
        if len(mouse_moves) < 3:
            return 0.5
            
        x_coords = [m['x'] for m in mouse_moves]
        y_coords = [m['y'] for m in mouse_moves]
        
        # 计算路径长度
        path_length = 0
        for i in range(1, len(mouse_moves)):
            dx = x_coords[i] - x_coords[i-1]
            dy = y_coords[i] - y_coords[i-1]
            path_length += np.sqrt(dx**2 + dy**2)
        
        # 计算直线距离
        straight_distance = np.sqrt((x_coords[-1] - x_coords[0])**2 + 
                                  (y_coords[-1] - y_coords[0])**2)
        
        # 复杂度 = 路径长度 / 直线距离
        if straight_distance > 0:
            complexity = min(1.0, path_length / straight_distance)
        else:
            complexity = 0.5
            
        return complexity
    
    def _calculate_nonlinearity(self, timestamps: List[float]) -> float:
        """计算时间序列的非线性程度"""
        if len(timestamps) < 3:
            return 0.5
            
        # 计算相邻时间间隔
        intervals = np.diff(timestamps)
        if len(intervals) < 2:
            return 0.5
            
        # 计算变异系数
        cv = np.std(intervals) / (np.mean(intervals) + 1e-6)
        return min(1.0, cv / 2)
    
    def _calculate_path_efficiency(self, mouse_moves: List[Dict]) -> float:
        """计算鼠标路径效率"""
        if len(mouse_moves) < 2:
            return 0.5
            
        # 计算实际路径长度
        actual_length = 0
        for i in range(1, len(mouse_moves)):
            dx = mouse_moves[i]['x'] - mouse_moves[i-1]['x']
            dy = mouse_moves[i]['y'] - mouse_moves[i-1]['y']
            actual_length += np.sqrt(dx**2 + dy**2)
        
        # 计算理想路径长度（直线）
        start = mouse_moves[0]
        end = mouse_moves[-1]
        ideal_length = np.sqrt((end['x'] - start['x'])**2 + (end['y'] - start['y'])**2)
        
        if ideal_length > 0:
            efficiency = ideal_length / actual_length
        else:
            efficiency = 1.0
            
        return max(0.1, min(1.0, efficiency))

class FeaturePreprocessor:
    """特征预处理器"""
    
    def __init__(self):
        self.scalers = {}
        self.feature_stats = {}
    
    def fit(self, features_list: List[Dict[str, float]]):
        """拟合特征标准化器"""
        if not features_list:
            return
            
        # 收集所有特征值
        feature_arrays = {}
        for feature_name in features_list[0].keys():
            feature_arrays[feature_name] = []
            
        for features in features_list:
            for name, value in features.items():
                feature_arrays[name].append(value)
        
        # 计算统计信息
        for name, values in feature_arrays.items():
            values = np.array(values)
            self.feature_stats[name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
    
    def transform(self, features: Dict[str, float]) -> np.ndarray:
        """标准化特征"""
        normalized_features = []
        
        for feature_name in self.feature_names:
            if feature_name in features:
                value = features[feature_name]
                if feature_name in self.feature_stats:
                    stats = self.feature_stats[feature_name]
                    if stats['std'] > 0:
                        normalized = (value - stats['mean']) / stats['std']
                        normalized = np.clip(normalized, -3, 3)
                        normalized = (normalized + 3) / 6  # 归一化到[0,1]
                    else:
                        normalized = 0.5
                else:
                    normalized = value
                normalized_features.append(normalized)
            else:
                normalized_features.append(0.5)
                
        return np.array(normalized_features)
    
    def inverse_transform(self, normalized_features: np.ndarray) -> Dict[str, float]:
        """逆标准化特征"""
        original_features = {}
        
        for i, feature_name in enumerate(self.feature_names):
            if i < len(normalized_features) and feature_name in self.feature_stats:
                stats = self.feature_stats[feature_name]
                normalized = normalized_features[i] * 6 - 3
                original = normalized * stats['std'] + stats['mean']
                original_features[feature_name] = float(np.clip(original, 0, 1))
            else:
                original_features[feature_name] = 0.5
                
        return original_features