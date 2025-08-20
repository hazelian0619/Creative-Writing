"""
T-S模糊推理引擎
师范生创意写作AI辅助系统的决策核心
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable
import logging
from dataclasses import dataclass
from collections import defaultdict
import time

@dataclass
class FuzzySet:
    """模糊集定义"""
    name: str
    membership_func: Callable[[float], float]
    center: float
    spread: float

class MembershipFunction:
    """隶属函数库"""
    
    @staticmethod
    def triangular(x: float, a: float, b: float, c: float) -> float:
        """三角形隶属函数"""
        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        else:
            return (c - x) / (c - b)
    
    @staticmethod
    def gaussian(x: float, center: float, sigma: float) -> float:
        """高斯隶属函数"""
        return np.exp(-0.5 * ((x - center) / sigma) ** 2)
    
    @staticmethod
    def trapezoidal(x: float, a: float, b: float, c: float, d: float) -> float:
        """梯形隶属函数"""
        if x <= a or x >= d:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        elif b < x <= c:
            return 1.0
        else:
            return (d - x) / (d - c)

@dataclass
class FuzzyRule:
    """T-S模糊规则"""
    rule_id: int
    conditions: List[Tuple[str, str]]  # [(变量名, 模糊集名称)]
    consequent_func: Callable[[Dict[str, float]], float]  # 后件函数
    weight: float = 1.0
    activation_count: int = 0
    performance_score: float = 0.5

class TSFuzzyInferenceEngine:
    """T-S模糊推理引擎"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # 输入变量定义
        self.input_variables = self._define_input_variables()
        
        # 模糊规则库
        self.fuzzy_rules = self._initialize_fuzzy_rules()
        
        # 自适应参数
        self.adaptive_params = {
            'learning_rate': 0.01,
            'rule_weights_decay': 0.995,
            'performance_threshold': 0.3
        }
        
        # 推理历史
        self.inference_history = []
        
        # 性能统计
        self.performance_stats = defaultdict(list)
        
        logging.info("T-S模糊推理引擎初始化完成")
    
    def _define_input_variables(self) -> Dict[str, Dict[str, FuzzySet]]:
        """定义输入变量及其模糊集"""
        variables = {}
        
        # 1. 认知负载 (CognitiveLoad): [0, 1]
        variables['cognitive_load'] = {
            'low': FuzzySet(
                'low',
                lambda x: max(0, (0.4 - x) / 0.4),
                center=0.2,
                spread=0.2
            ),
            'medium': FuzzySet(
                'medium',
                lambda x: MembershipFunction.gaussian(x, 0.5, 0.15),
                center=0.5,
                spread=0.15
            ),
            'high': FuzzySet(
                'high',
                lambda x: max(0, (x - 0.6) / 0.4),
                center=0.8,
                spread=0.2
            )
        }
        
        # 2. 思维漂移程度 (DriftLevel): [0, 1]
        variables['drift_level'] = {
            'stable': FuzzySet(
                'stable',
                lambda x: np.exp(-(x**2) / 0.1),
                center=0.0,
                spread=0.2
            ),
            'light_drift': FuzzySet(
                'light_drift',
                lambda x: MembershipFunction.gaussian(x, 0.3, 0.1),
                center=0.3,
                spread=0.1
            ),
            'severe_drift': FuzzySet(
                'severe_drift',
                lambda x: MembershipFunction.gaussian(x, 0.8, 0.1),
                center=0.8,
                spread=0.1
            )
        }
        
        # 3. 语义覆盖率 (SemanticCoverage): [0, 1]
        variables['semantic_coverage'] = {
            'insufficient': FuzzySet(
                'insufficient',
                lambda x: max(0, (0.5 - x) / 0.5),
                center=0.25,
                spread=0.25
            ),
            'adequate': FuzzySet(
                'adequate',
                lambda x: MembershipFunction.gaussian(x, 0.75, 0.15),
                center=0.75,
                spread=0.15
            ),
            'sufficient': FuzzySet(
                'sufficient',
                lambda x: max(0, (x - 0.8) / 0.2),
                center=0.9,
                spread=0.1
            )
        }
        
        # 4. 响应时间 (ResponseTime): [0, 30] 秒
        variables['response_time'] = {
            'fast': FuzzySet(
                'fast',
                lambda x: max(0, (8 - x) / 8),
                center=4,
                spread=4
            ),
            'normal': FuzzySet(
                'normal',
                lambda x: MembershipFunction.gaussian(x, 12, 3),
                center=12,
                spread=3
            ),
            'slow': FuzzySet(
                'slow',
                lambda x: max(0, (x - 15) / 15),
                center=22,
                spread=7
            )
        }
        
        return variables
    
    def _initialize_fuzzy_rules(self) -> List[FuzzyRule]:
        """初始化8条核心模糊规则"""
        rules = []
        
        # 规则1: 低负载+稳定 -> 轻度提示
        rules.append(FuzzyRule(
            rule_id=1,
            conditions=[('cognitive_load', 'low'), ('drift_level', 'stable')],
            consequent_func=lambda inputs: 0.2 * inputs['cognitive_load'] + 0.1 * inputs['semantic_coverage'],
            weight=1.0
        ))
        
        # 规则2: 低负载+轻微漂移 -> 中度提示
        rules.append(FuzzyRule(
            rule_id=2,
            conditions=[('cognitive_load', 'low'), ('drift_level', 'light_drift')],
            consequent_func=lambda inputs: 0.3 * inputs['cognitive_load'] + 0.2 * inputs['drift_level'],
            weight=1.0
        ))
        
        # 规则3: 中负载+语义不足 -> 强化提示
        rules.append(FuzzyRule(
            rule_id=3,
            conditions=[('cognitive_load', 'medium'), ('semantic_coverage', 'insufficient')],
            consequent_func=lambda inputs: 0.6 + 0.2 * (1 - inputs['semantic_coverage']),
            weight=1.0
        ))
        
        # 规则4: 高负载+响应慢 -> 减少提示
        rules.append(FuzzyRule(
            rule_id=4,
            conditions=[('cognitive_load', 'high'), ('response_time', 'slow')],
            consequent_func=lambda inputs: max(0.1, 0.8 - 0.3 * inputs['cognitive_load']),
            weight=1.0
        ))
        
        # 规则5: 严重漂移 -> 重度校准
        rules.append(FuzzyRule(
            rule_id=5,
            conditions=[('drift_level', 'severe_drift')],
            consequent_func=lambda inputs: 0.9,
            weight=1.2
        ))
        
        # 规则6: 语义充分+低负载 -> 维持提示
        rules.append(FuzzyRule(
            rule_id=6,
            conditions=[('semantic_coverage', 'sufficient'), ('cognitive_load', 'low')],
            consequent_func=lambda inputs: 0.1 + 0.1 * inputs['cognitive_load'],
            weight=0.8
        ))
        
        # 规则7: 快速响应+语义适中 -> 适度提示
        rules.append(FuzzyRule(
            rule_id=7,
            conditions=[('response_time', 'fast'), ('semantic_coverage', 'adequate')],
            consequent_func=lambda inputs: 0.4 * inputs['semantic_coverage'],
            weight=1.0
        ))
        
        # 规则8: 高负载+稳定 -> 轻度提示
        rules.append(FuzzyRule(
            rule_id=8,
            conditions=[('cognitive_load', 'high'), ('drift_level', 'stable')],
            consequent_func=lambda inputs: 0.2,
            weight=0.9
        ))
        
        return rules
    
    def fuzzify_inputs(self, inputs: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        模糊化输入变量
        
        Args:
            inputs: 输入变量值字典
            
        Returns:
            各变量的模糊集隶属度
        """
        memberships = {}
        
        for var_name, value in inputs.items():
            if var_name in self.input_variables:
                memberships[var_name] = {}
                for fuzzy_set_name, fuzzy_set in self.input_variables[var_name].items():
                    membership_value = fuzzy_set.membership_func(value)
                    memberships[var_name][fuzzy_set_name] = membership_value
        
        return memberships
    
    def calculate_rule_activation(self, rule: FuzzyRule, 
                                memberships: Dict[str, Dict[str, float]]) -> float:
        """计算规则激活强度"""
        activation = 1.0
        
        for var_name, fuzzy_set_name in rule.conditions:
            if var_name in memberships and fuzzy_set_name in memberships[var_name]:
                membership_value = memberships[var_name][fuzzy_set_name]
                activation = min(activation, membership_value)  # AND操作使用最小值
            else:
                activation = 0.0
                break
        
        return activation * rule.weight
    
    def ts_inference(self, inputs: Dict[str, float], 
                    memberships: Dict[str, Dict[str, float]]) -> float:
        """
        T-S推理计算提示强度
        
        Args:
            inputs: 原始输入值
            memberships: 模糊化结果
            
        Returns:
            推理得到的提示强度 [0,1]
        """
        numerator = 0.0
        denominator = 0.0
        
        active_rules = []
        
        for rule in self.fuzzy_rules:
            activation = self.calculate_rule_activation(rule, memberships)
            
            if activation > 0.01:  # 激活阈值
                # 计算规则后件
                consequent_value = rule.consequent_func(inputs)
                
                numerator += activation * consequent_value
                denominator += activation
                
                # 记录激活的规则
                rule.activation_count += 1
                active_rules.append((rule.rule_id, activation, consequent_value))
        
        # 计算最终输出
        if denominator > 0:
            output = numerator / denominator
        else:
            output = 0.5  # 默认值
        
        # 记录推理过程
        self.inference_history.append({
            'timestamp': time.time(),
            'inputs': inputs.copy(),
            'output': output,
            'active_rules': active_rules
        })
        
        return max(0.0, min(1.0, output))
    
    def determine_prompt_type(self, cognitive_load: float, 
                            semantic_coverage: float, 
                            drift_level: float) -> str:
        """确定提示类型"""
        if drift_level > 0.6:
            return "结构化"  # 严重漂移需要结构化引导
        elif semantic_coverage < 0.5:
            return "示例化"  # 语义覆盖不足需要具体示例
        elif cognitive_load < 0.3:
            return "启发式"  # 认知负载低可以用启发性提示
        else:
            return "结构化"  # 默认结构化
    
    def determine_calibration_level(self, drift_level: float, 
                                  cognitive_load: float) -> str:
        """确定校准级别"""
        if drift_level > 0.7 or cognitive_load > 0.8:
            return "重度"
        elif drift_level > 0.4 or cognitive_load > 0.6:
            return "中度"
        else:
            return "轻度"
    
    def infer(self, cognitive_load: float, drift_level: float, 
            semantic_coverage: float, response_time: float) -> Dict[str, any]:
        """
        主推理函数
        
        Args:
            cognitive_load: 认知负载 [0,1]
            drift_level: 思维漂移程度 [0,1]
            semantic_coverage: 语义覆盖率 [0,1]
            response_time: 响应时间 [0,30]秒
            
        Returns:
            推理结果字典
        """
        start_time = time.time()
        
        # 1. 构建输入字典
        inputs = {
            'cognitive_load': cognitive_load,
            'drift_level': drift_level,
            'semantic_coverage': semantic_coverage,
            'response_time': response_time
        }
        
        # 2. 模糊化
        memberships = self.fuzzify_inputs(inputs)
        
        # 3. T-S推理
        prompt_intensity = self.ts_inference(inputs, memberships)
        
        # 4. 确定提示类型和校准级别
        prompt_type = self.determine_prompt_type(
            cognitive_load, semantic_coverage, drift_level
        )
        calibration_level = self.determine_calibration_level(
            drift_level, cognitive_load
        )
        
        inference_time = time.time() - start_time
        
        # 5. 记录性能统计
        self.performance_stats['inference_times'].append(inference_time)
        
        result = {
            'prompt_intensity': prompt_intensity,
            'prompt_type': prompt_type,
            'calibration_level': calibration_level,
            'inference_time': inference_time,
            'memberships': memberships,
            'active_rules_count': len([h for h in self.inference_history[-1]['active_rules'] if h[1] > 0.01])
        }
        
        return result
    
    def update_rule_performance(self, rule_id: int, performance_score: float):
        """更新规则性能分数"""
        for rule in self.fuzzy_rules:
            if rule.rule_id == rule_id:
                # 指数移动平均更新
                alpha = 0.1
                rule.performance_score = (alpha * performance_score + 
                                        (1 - alpha) * rule.performance_score)
                break
    
    def adaptive_rule_adjustment(self, user_feedback: Dict[str, float]):
        """基于用户反馈的自适应规则调整"""
        if not self.inference_history:
            return
        
        # 获取最近的推理
        recent_inference = self.inference_history[-1]
        feedback_score = user_feedback.get('satisfaction', 0.5)
        
        # 更新激活规则的性能
        for rule_id, activation, _ in recent_inference['active_rules']:
            weighted_feedback = feedback_score * activation
            self.update_rule_performance(rule_id, weighted_feedback)
        
        # 自适应权重调整
        learning_rate = self.adaptive_params['learning_rate']
        
        for rule in self.fuzzy_rules:
            if rule.performance_score < self.adaptive_params['performance_threshold']:
                # 降低表现差的规则权重
                rule.weight *= self.adaptive_params['rule_weights_decay']
            else:
                # 提升表现好的规则权重
                rule.weight = min(2.0, rule.weight * (1 + learning_rate))
        
        logging.info(f"规则权重已根据反馈调整，满意度: {feedback_score:.3f}")
    
    def get_rule_statistics(self) -> Dict[str, any]:
        """获取规则使用统计"""
        stats = {}
        total_activations = sum(rule.activation_count for rule in self.fuzzy_rules)
        
        for rule in self.fuzzy_rules:
            usage_rate = rule.activation_count / max(total_activations, 1)
            stats[f'rule_{rule.rule_id}'] = {
                'activation_count': rule.activation_count,
                'usage_rate': usage_rate,
                'performance_score': rule.performance_score,
                'current_weight': rule.weight
            }
        
        # 系统统计
        stats['system'] = {
            'total_inferences': len(self.inference_history),
            'avg_inference_time': np.mean(self.performance_stats['inference_times'][-100:]) if self.performance_stats['inference_times'] else 0,
            'total_activations': total_activations
        }
        
        return stats
    
    def explain_decision(self, inputs: Dict[str, float]) -> str:
        """解释推理决策过程"""
        if not self.inference_history:
            return "暂无推理历史"
        
        recent = self.inference_history[-1]
        explanation = f"基于输入参数：\\n"
        
        for var, value in inputs.items():
            explanation += f"- {var}: {value:.3f}\\n"
        
        explanation += f"\\n激活的规则：\\n"
        for rule_id, activation, consequent in recent['active_rules']:
            explanation += f"- 规则{rule_id}: 激活度={activation:.3f}, 输出={consequent:.3f}\\n"
        
        explanation += f"\\n最终提示强度: {recent['output']:.3f}"
        
        return explanation
    
    def save_engine_state(self, filepath: str):
        """保存推理引擎状态"""
        state = {
            'fuzzy_rules': [{
                'rule_id': rule.rule_id,
                'weight': rule.weight,
                'performance_score': rule.performance_score,
                'activation_count': rule.activation_count
            } for rule in self.fuzzy_rules],
            'adaptive_params': self.adaptive_params,
            'performance_stats': dict(self.performance_stats),
            'config': self.config
        }
        
        np.save(filepath, state)
        logging.info(f"模糊推理引擎状态已保存到: {filepath}")
    
    def load_engine_state(self, filepath: str):
        """加载推理引擎状态"""
        state = np.load(filepath, allow_pickle=True).item()
        
        # 恢复规则状态
        rule_data = {r['rule_id']: r for r in state['fuzzy_rules']}
        
        for rule in self.fuzzy_rules:
            if rule.rule_id in rule_data:
                data = rule_data[rule.rule_id]
                rule.weight = data['weight']
                rule.performance_score = data['performance_score']
                rule.activation_count = data['activation_count']
        
        # 恢复参数
        self.adaptive_params.update(state['adaptive_params'])
        self.performance_stats.update(state['performance_stats'])
        
        logging.info(f"模糊推理引擎状态已从 {filepath} 加载")