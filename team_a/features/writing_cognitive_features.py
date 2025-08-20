"""
写作认知特征提取算法
从用户行为数据中提取8维写作思维特征
基于权威教育心理学研究框架
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler

class WritingCognitiveFeatureExtractor:
    """
    写作认知特征提取器
    基于权威写作思维能力框架提取8个维度的特征
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
        从原始行为数据提取所有写作认知特征
        
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
            8维写作认知特征向量
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
        提取灵活性特征 - 多角度思维、辩证表达
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
    
    def _extract_critical_thinking_features(self, data: Dict) -> float:
        """
        提取批判性特征 - 错误识别、自我评价
        通过分析自我修正行为、检查模式等判断批判思维
        """
        keystrokes = data.get('keystrokes', [])
        response_times = data.get('response_times', [])
        
        if not keystrokes and not response_times:
            return 0.5
        
        # 1. 自我修正频率（删除-重写模式）
        correction_score = 0.0
        if keystrokes and len(keystrokes) > 3:
            corrections = 0
            for i in range(len(keystrokes) - 1):
                if (keystrokes[i].get('key') in ['Backspace', 'Delete'] and
                    i + 1 < len(keystrokes) and
                    keystrokes[i + 1].get('key') not in ['Backspace', 'Delete']):
                    corrections += 1
            correction_score = min(1.0, corrections / (len(keystrokes) / 10))
        
        # 2. 检查性停顿（短暂停顿后继续）
        review_score = 0.0
        if response_times:
            # 中等长度停顿表示检查思考
            review_pauses = [t for t in response_times if 1.0 < t < 3.0]
            review_score = min(1.0, len(review_pauses) / max(len(response_times), 1) * 2)
        
        # 综合批判性评分
        critical_score = (correction_score * 0.6 + review_score * 0.4)
        return np.clip(critical_score, 0.0, 1.0)
    
    def _extract_originality_features(self, data: Dict) -> float:
        """
        提取独创性特征 - 新颖独特程度
        通过分析创新性行为模式、非常规操作等判断独创性
        """
        keystrokes = data.get('keystrokes', [])
        mouse_moves = data.get('mouse_moves', [])
        task_context = data.get('task_context', {})
        
        if not keystrokes and not mouse_moves:
            return 0.5
        
        # 1. 非常规击键模式（创新性表达）
        novelty_score = 0.0
        if keystrokes:
            special_keys = [k for k in keystrokes if k.get('key') in 
                           ['Tab', 'Enter', 'Space', 'Shift']]  # 格式控制键使用
            novelty_score = min(1.0, len(special_keys) / max(len(keystrokes), 1) * 5)
        
        # 2. 探索性鼠标行为
        exploration_score = 0.0
        if mouse_moves:
            # 大范围移动表示探索性思考
            large_movements = 0
            for i in range(1, len(mouse_moves)):
                dx = abs(mouse_moves[i]['x'] - mouse_moves[i-1]['x'])
                dy = abs(mouse_moves[i]['y'] - mouse_moves[i-1]['y'])
                distance = np.sqrt(dx**2 + dy**2)
                if distance > 100:  # 大距离移动
                    large_movements += 1
            exploration_score = min(1.0, large_movements / max(len(mouse_moves), 1) * 10)
        
        # 3. 任务创新倾向
        innovation_factor = task_context.get('creativity_level', 0.5)
        
        # 综合独创性评分
        originality_score = (novelty_score * 0.4 + exploration_score * 0.3 + innovation_factor * 0.3)
        return np.clip(originality_score, 0.0, 1.0)
    
    def _extract_fluency_features(self, data: Dict) -> float:
        """
        提取流畅性特征 - 思维表达连贯性
        通过分析输入连贯性、节奏稳定性等判断流畅度
        """
        keystrokes = data.get('keystrokes', [])
        response_times = data.get('response_times', [])
        
        if not keystrokes and not response_times:
            return 0.5
        
        # 1. 击键节奏稳定性
        rhythm_fluency = 0.0
        if keystrokes and len(keystrokes) > 5:
            intervals = []
            for i in range(1, len(keystrokes)):
                interval = keystrokes[i]['timestamp'] - keystrokes[i-1]['timestamp']
                intervals.append(interval)
            
            # 稳定的节奏表示流畅
            if intervals:
                rhythm_fluency = 1.0 - min(1.0, np.std(intervals) / (np.mean(intervals) + 1e-6))
        
        # 2. 响应连贯性
        response_fluency = 0.0
        if response_times:
            # 适中且稳定的响应时间表示流畅
            optimal_times = [t for t in response_times if 0.5 < t < 2.0]
            response_fluency = len(optimal_times) / max(len(response_times), 1)
        
        # 综合流畅性评分
        fluency_score = (rhythm_fluency * 0.6 + response_fluency * 0.4)
        return np.clip(fluency_score, 0.0, 1.0)
    
    def _extract_motivation_features(self, data: Dict) -> float:
        """
        提取动机水平特征 - 写作驱动力
        通过分析持续性、投入度等判断动机水平
        """
        keystrokes = data.get('keystrokes', [])
        dwell_times = data.get('dwell_times', [])
        task_context = data.get('task_context', {})
        
        if not keystrokes and not dwell_times:
            return 0.5
        
        # 1. 持续投入度（总体活动时长）
        persistence_score = 0.0
        if keystrokes:
            total_time = keystrokes[-1]['timestamp'] - keystrokes[0]['timestamp']
            active_time = len(keystrokes) * 0.1  # 假设每次击键0.1秒
            persistence_score = min(1.0, active_time / max(total_time, 1) * 5)
        
        # 2. 任务投入强度
        intensity_score = 0.0
        if keystrokes:
            keystroke_rate = len(keystrokes) / max(
                (keystrokes[-1]['timestamp'] - keystrokes[0]['timestamp']) / 60, 1)
            intensity_score = min(1.0, keystroke_rate / 50)  # 50击/分钟为满分
        
        # 3. 主观动机调节
        motivation_factor = task_context.get('motivation_level', 0.5)
        
        # 综合动机评分
        motivation_score = (persistence_score * 0.4 + intensity_score * 0.3 + motivation_factor * 0.3)
        return np.clip(motivation_score, 0.0, 1.0)
    
    def _extract_emotion_regulation_features(self, data: Dict) -> float:
        """
        提取情绪调节特征 - 情感状态管理
        通过分析行为稳定性、压力指标等判断情绪调节能力
        """
        keystrokes = data.get('keystrokes', [])
        mouse_moves = data.get('mouse_moves', [])
        
        if not keystrokes and not mouse_moves:
            return 0.5
        
        # 1. 行为稳定性（压力指标）
        stability_score = 0.0
        if keystrokes and len(keystrokes) > 10:
            # 计算击键强度变化（通过按键持续时间）
            durations = []
            for i in range(len(keystrokes) - 1):
                if keystrokes[i].get('type') == 'keydown' and keystrokes[i+1].get('type') == 'keyup':
                    duration = keystrokes[i+1]['timestamp'] - keystrokes[i]['timestamp']
                    durations.append(duration)
            
            if durations and len(durations) > 5:
                # 稳定的击键强度表示良好的情绪调节
                stability_score = 1.0 - min(1.0, np.std(durations) / (np.mean(durations) + 1e-6))
        
        # 2. 压力恢复能力
        recovery_score = 0.0
        if mouse_moves:
            # 平滑的鼠标轨迹表示放松状态
            smoothness = 0
            for i in range(2, len(mouse_moves)):
                # 计算轨迹曲率
                p1 = (mouse_moves[i-2]['x'], mouse_moves[i-2]['y'])
                p2 = (mouse_moves[i-1]['x'], mouse_moves[i-1]['y'])
                p3 = (mouse_moves[i]['x'], mouse_moves[i]['y'])
                
                # 简化的曲率计算
                angle_change = self._calculate_angle_change(p1, p2, p3)
                if abs(angle_change) < 0.2:  # 平滑移动
                    smoothness += 1
            
            if mouse_moves:
                recovery_score = smoothness / max(len(mouse_moves) - 2, 1)
        
        # 综合情绪调节评分
        emotion_regulation_score = (stability_score * 0.7 + recovery_score * 0.3)
        return np.clip(emotion_regulation_score, 0.0, 1.0)
    
    def _extract_cognitive_load_features(self, data: Dict) -> float:
        """
        提取认知负载特征 - 思维负担程度
        通过分析多任务处理、反应延迟等判断认知负载
        """
        response_times = data.get('response_times', [])
        keystrokes = data.get('keystrokes', [])
        mouse_moves = data.get('mouse_moves', [])
        
        if not response_times and not keystrokes:
            return 0.5
        
        # 1. 反应时间延迟
        latency_load = 0.0
        if response_times:
            avg_response_time = np.mean(response_times)
            # 响应时间越长，认知负载越高
            latency_load = min(1.0, avg_response_time / 5.0)  # 5秒为高负载阈值
        
        # 2. 多任务处理复杂度
        multitask_load = 0.0
        if keystrokes and mouse_moves:
            # 键鼠交替频率表示任务复杂度
            kb_times = [k['timestamp'] for k in keystrokes]
            mouse_times = [m['timestamp'] for m in mouse_moves]
            
            all_events = sorted(kb_times + mouse_times)
            transitions = 0
            
            last_type = None
            for event_time in all_events:
                current_type = 'kb' if event_time in kb_times else 'mouse'
                if last_type and last_type != current_type:
                    transitions += 1
                last_type = current_type
            
            multitask_load = min(1.0, transitions / max(len(all_events), 1) * 2)
        
        # 3. 错误修正负载
        error_load = 0.0
        if keystrokes:
            error_keys = [k for k in keystrokes if k.get('key') in ['Backspace', 'Delete']]
            error_load = min(1.0, len(error_keys) / max(len(keystrokes), 1) * 5)
        
        # 综合认知负载评分
        cognitive_load_score = (latency_load * 0.4 + multitask_load * 0.3 + error_load * 0.3)
        return np.clip(cognitive_load_score, 0.0, 1.0)
    
    def _calculate_angle_change(self, p1, p2, p3):
        """计算三点间的角度变化"""
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        # 避免除零错误
        norm1 = np.sqrt(v1[0]**2 + v1[1]**2) + 1e-6
        norm2 = np.sqrt(v2[0]**2 + v2[1]**2) + 1e-6
        
        cos_angle = (v1[0]*v2[0] + v1[1]*v2[1]) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1, 1)
        
        return np.arccos(cos_angle)

# 为了向后兼容，创建别名
CognitiveFeatureExtractor = WritingCognitiveFeatureExtractor