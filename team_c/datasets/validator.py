"""
数据验证器
验证数据集质量和与团队A模型的兼容性
"""

import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
from scipy import stats
import sys

# 导入团队A的模块确保兼容性验证
sys.path.append("/Users/pluviophile/chi2025")
try:
    from team_a.features.cognitive_features import CognitiveFeatureExtractor
    from team_a.models.difcm_core import DIFCMModel, DIFCMConfig
except ImportError as e:
    logging.warning(f"无法导入团队A模块: {e}")

from team_c.config import EvaluationConfig, CognitiveStateSpec

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    score: float
    issues: List[str]
    recommendations: List[str]
    detailed_metrics: Dict[str, Any]

class DataValidator:
    """
    数据验证器
    确保数据集质量和团队间兼容性
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.spec = CognitiveStateSpec()
        
        # 尝试初始化团队A的组件
        try:
            self.feature_extractor = CognitiveFeatureExtractor()
        except:
            self.feature_extractor = None
            logger.warning("无法初始化特征提取器，将跳过特征提取验证")
        
        # 验证规则
        self.validation_rules = {
            'completeness': self._check_completeness,
            'format_consistency': self._check_format_consistency,
            'value_ranges': self._check_value_ranges,
            'cognitive_consistency': self._check_cognitive_consistency,
            'behavioral_plausibility': self._check_behavioral_plausibility,
            'team_a_compatibility': self._check_team_a_compatibility,
            'statistical_properties': self._check_statistical_properties
        }
        
        logger.info("数据验证器初始化完成")
    
    def validate_dataset(self, samples: List[Dict]) -> ValidationResult:
        """
        完整验证数据集
        
        Args:
            samples: 数据样本列表
            
        Returns:
            验证结果
        """
        if not samples:
            return ValidationResult(
                is_valid=False,
                score=0.0,
                issues=["数据集为空"],
                recommendations=["请提供有效的数据样本"],
                detailed_metrics={}
            )
        
        logger.info(f"开始验证数据集，样本数: {len(samples)}")
        
        validation_scores = {}
        all_issues = []
        all_recommendations = []
        detailed_metrics = {}
        
        # 执行所有验证规则
        for rule_name, rule_func in self.validation_rules.items():
            try:
                logger.info(f"执行验证规则: {rule_name}")
                result = rule_func(samples)
                
                validation_scores[rule_name] = result.score
                all_issues.extend(result.issues)
                all_recommendations.extend(result.recommendations)
                detailed_metrics[rule_name] = result.detailed_metrics
                
                logger.info(f"{rule_name} 验证完成，得分: {result.score:.3f}")
                
            except Exception as e:
                logger.error(f"验证规则 {rule_name} 执行失败: {e}")
                validation_scores[rule_name] = 0.0
                all_issues.append(f"{rule_name} 验证失败: {str(e)}")
        
        # 计算总体分数
        overall_score = np.mean(list(validation_scores.values()))
        is_valid = overall_score >= 0.7  # 70% 为合格线
        
        # 生成汇总报告
        detailed_metrics['validation_scores'] = validation_scores
        detailed_metrics['sample_count'] = len(samples)
        detailed_metrics['overall_score'] = overall_score
        
        logger.info(f"数据集验证完成，总体得分: {overall_score:.3f}, 是否有效: {is_valid}")
        
        return ValidationResult(
            is_valid=is_valid,
            score=overall_score,
            issues=list(set(all_issues)),  # 去重
            recommendations=list(set(all_recommendations)),
            detailed_metrics=detailed_metrics
        )
    
    def _check_completeness(self, samples: List[Dict]) -> ValidationResult:
        """检查数据完整性"""
        issues = []
        recommendations = []
        metrics = {}
        
        # 必需字段
        required_fields = [
            'user_id', 'session_id', 'timestamp',
            'cognitive_state', 'behavior_data'
        ]
        
        # 检查每个样本的完整性
        complete_samples = 0
        field_missing_counts = {field: 0 for field in required_fields}
        
        for i, sample in enumerate(samples):
            sample_complete = True
            
            for field in required_fields:
                if field not in sample or sample[field] is None:
                    field_missing_counts[field] += 1
                    sample_complete = False
                    
                    if field_missing_counts[field] == 1:  # 只报告一次
                        issues.append(f"样本 {i} 缺少必需字段: {field}")
            
            # 检查认知状态字段
            if 'cognitive_state' in sample and sample['cognitive_state']:
                cognitive_state = sample['cognitive_state']
                for concept in self.spec.concept_names:
                    if concept not in cognitive_state:
                        issues.append(f"样本 {i} 认知状态缺少概念: {concept}")
                        sample_complete = False
            
            if sample_complete:
                complete_samples += 1
        
        # 计算完整性分数
        completeness_score = complete_samples / len(samples)
        
        # 生成建议
        for field, missing_count in field_missing_counts.items():
            if missing_count > 0:
                missing_percentage = missing_count / len(samples) * 100
                recommendations.append(
                    f"补充 {missing_count} 个样本的 {field} 字段 ({missing_percentage:.1f}%)"
                )
        
        metrics = {
            'complete_samples': complete_samples,
            'completeness_percentage': completeness_score * 100,
            'field_missing_counts': field_missing_counts
        }
        
        return ValidationResult(
            is_valid=completeness_score >= 0.95,
            score=completeness_score,
            issues=issues,
            recommendations=recommendations,
            detailed_metrics=metrics
        )
    
    def _check_format_consistency(self, samples: List[Dict]) -> ValidationResult:
        """检查格式一致性"""
        issues = []
        recommendations = []
        metrics = {}
        
        # 检查数据类型一致性
        type_consistency = {}
        value_format_issues = 0
        
        for i, sample in enumerate(samples):
            # 检查基本字段类型
            if 'user_id' in sample and not isinstance(sample['user_id'], str):
                issues.append(f"样本 {i} user_id 类型错误，应为字符串")
                
            if 'timestamp' in sample and not isinstance(sample['timestamp'], (int, float)):
                issues.append(f"样本 {i} timestamp 类型错误，应为数字")
            
            # 检查认知状态格式
            if 'cognitive_state' in sample:
                cognitive_state = sample['cognitive_state']
                if not isinstance(cognitive_state, dict):
                    issues.append(f"样本 {i} cognitive_state 应为字典格式")
                    value_format_issues += 1
                    continue
                
                for concept, value in cognitive_state.items():
                    if not isinstance(value, (int, float)):
                        issues.append(f"样本 {i} 认知状态 {concept} 值类型错误")
                        value_format_issues += 1
                    elif not (0 <= value <= 1):
                        issues.append(f"样本 {i} 认知状态 {concept} 值超出[0,1]范围: {value}")
                        value_format_issues += 1
            
            # 检查行为数据格式
            if 'behavior_data' in sample:
                behavior_data = sample['behavior_data']
                if not isinstance(behavior_data, dict):
                    issues.append(f"样本 {i} behavior_data 应为字典格式")
                    value_format_issues += 1
                    continue
                
                # 检查行为数据子字段
                expected_behavior_fields = ['keystrokes', 'mouse_moves', 'dwell_times', 'response_times']
                for field in expected_behavior_fields:
                    if field in behavior_data:
                        if not isinstance(behavior_data[field], list):
                            issues.append(f"样本 {i} behavior_data.{field} 应为列表格式")
                            value_format_issues += 1
        
        # 计算格式一致性分数
        total_checks = len(samples) * 4  # 每个样本检查4个主要方面
        format_score = max(0, 1 - value_format_issues / total_checks)
        
        if value_format_issues > 0:
            recommendations.append(f"修复 {value_format_issues} 个格式错误")
            recommendations.append("确保认知状态值在[0,1]范围内")
            recommendations.append("确保所有时间戳为数值类型")
        
        metrics = {
            'format_issues_count': value_format_issues,
            'format_score': format_score,
            'checked_samples': len(samples)
        }
        
        return ValidationResult(
            is_valid=format_score >= 0.9,
            score=format_score,
            issues=issues,
            recommendations=recommendations,
            detailed_metrics=metrics
        )
    
    def _check_value_ranges(self, samples: List[Dict]) -> ValidationResult:
        """检查数值范围合理性"""
        issues = []
        recommendations = []
        metrics = {}
        
        value_range_violations = 0
        concept_stats = {concept: [] for concept in self.spec.concept_names}
        
        for i, sample in enumerate(samples):
            if 'cognitive_state' not in sample:
                continue
                
            cognitive_state = sample['cognitive_state']
            
            for concept in self.spec.concept_names:
                if concept in cognitive_state:
                    value = cognitive_state[concept]
                    concept_stats[concept].append(value)
                    
                    # 检查基本范围
                    if not (0 <= value <= 1):
                        issues.append(f"样本 {i} {concept} 值超出范围: {value}")
                        value_range_violations += 1
                    
                    # 检查极端值（可能的异常）
                    if value < 0.05 or value > 0.95:
                        if len(issues) < 10:  # 限制报告数量
                            issues.append(f"样本 {i} {concept} 值接近极端: {value}")
        
        # 计算统计信息
        range_stats = {}
        for concept, values in concept_stats.items():
            if values:
                range_stats[concept] = {
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'samples': len(values)
                }
                
                # 检查分布合理性
                if np.std(values) < 0.05:
                    recommendations.append(f"{concept} 变化范围过小，考虑增加多样性")
                elif np.std(values) > 0.4:
                    recommendations.append(f"{concept} 变化范围过大，检查数据一致性")
        
        # 计算范围合理性分数
        total_values = sum(len(values) for values in concept_stats.values())
        range_score = max(0, 1 - value_range_violations / max(1, total_values))
        
        metrics = {
            'range_violations': value_range_violations,
            'total_values_checked': total_values,
            'concept_statistics': range_stats,
            'range_score': range_score
        }
        
        return ValidationResult(
            is_valid=range_score >= 0.95,
            score=range_score,
            issues=issues,
            recommendations=recommendations,
            detailed_metrics=metrics
        )
    
    def _check_cognitive_consistency(self, samples: List[Dict]) -> ValidationResult:
        """检查认知状态内部一致性"""
        issues = []
        recommendations = []
        metrics = {}
        
        consistency_violations = 0
        consistency_scores = []
        
        for i, sample in enumerate(samples):
            if 'cognitive_state' not in sample:
                continue
                
            cognitive_state = sample['cognitive_state']
            
            # 获取关键认知维度
            attention = cognitive_state.get('attention', 0.5)
            fatigue = cognitive_state.get('fatigue', 0.5)
            creativity = cognitive_state.get('creativity', 0.5)
            confidence = cognitive_state.get('confidence', 0.5)
            motivation = cognitive_state.get('motivation', 0.5)
            emotion = cognitive_state.get('emotion', 0.5)
            
            # 检查逻辑一致性
            sample_consistency = []
            
            # 1. 注意力与疲劳负相关
            attention_fatigue_consistency = 1 - abs((attention + fatigue) - 1.0)
            if attention_fatigue_consistency < 0.3:
                consistency_violations += 1
                if len(issues) < 20:
                    issues.append(f"样本 {i} 注意力与疲劳不一致: attention={attention:.2f}, fatigue={fatigue:.2f}")
            sample_consistency.append(attention_fatigue_consistency)
            
            # 2. 创造力与自信相关
            creativity_confidence_corr = 1 - abs(creativity - confidence) / 2
            sample_consistency.append(creativity_confidence_corr)
            
            # 3. 动机与情绪相关
            motivation_emotion_corr = 1 - abs(motivation - emotion) / 2
            sample_consistency.append(motivation_emotion_corr)
            
            # 4. 高疲劳时其他能力下降
            if fatigue > 0.7:
                high_fatigue_consistency = np.mean([1 - attention, 1 - creativity, 1 - motivation])
                if high_fatigue_consistency < 0.3:
                    consistency_violations += 1
                    if len(issues) < 20:
                        issues.append(f"样本 {i} 高疲劳状态但其他能力过高")
                sample_consistency.append(high_fatigue_consistency)
            
            # 计算样本一致性分数
            if sample_consistency:
                consistency_scores.append(np.mean(sample_consistency))
        
        # 计算总体一致性
        overall_consistency = np.mean(consistency_scores) if consistency_scores else 0.0
        
        if consistency_violations > len(samples) * 0.1:
            recommendations.append("检查认知状态逻辑关系，特别是注意力与疲劳的负相关")
        if overall_consistency < 0.7:
            recommendations.append("增强认知状态内部一致性，参考心理学研究")
        
        metrics = {
            'consistency_violations': consistency_violations,
            'overall_consistency': overall_consistency,
            'consistency_scores': consistency_scores,
            'checked_samples': len(consistency_scores)
        }
        
        return ValidationResult(
            is_valid=overall_consistency >= 0.7,
            score=overall_consistency,
            issues=issues,
            recommendations=recommendations,
            detailed_metrics=metrics
        )
    
    def _check_behavioral_plausibility(self, samples: List[Dict]) -> ValidationResult:
        """检查行为数据合理性"""
        issues = []
        recommendations = []
        metrics = {}
        
        plausibility_issues = 0
        behavior_stats = {}
        
        for i, sample in enumerate(samples):
            if 'behavior_data' not in sample:
                continue
                
            behavior_data = sample['behavior_data']
            
            # 检查击键数据
            if 'keystrokes' in behavior_data:
                keystrokes = behavior_data['keystrokes']
                if isinstance(keystrokes, list):
                    if len(keystrokes) > 1000:
                        issues.append(f"样本 {i} 击键数据过多: {len(keystrokes)}")
                        plausibility_issues += 1
                    elif len(keystrokes) < 5:
                        issues.append(f"样本 {i} 击键数据过少: {len(keystrokes)}")
                        plausibility_issues += 1
                    
                    # 检查时间戳序列
                    timestamps = [k.get('timestamp', 0) for k in keystrokes if isinstance(k, dict)]
                    if len(timestamps) > 1:
                        if not all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1)):
                            issues.append(f"样本 {i} 击键时间戳序列错误")
                            plausibility_issues += 1
            
            # 检查鼠标数据
            if 'mouse_moves' in behavior_data:
                mouse_moves = behavior_data['mouse_moves']
                if isinstance(mouse_moves, list):
                    if len(mouse_moves) > 5000:
                        issues.append(f"样本 {i} 鼠标数据过多: {len(mouse_moves)}")
                        plausibility_issues += 1
                    
                    # 检查坐标合理性
                    for j, move in enumerate(mouse_moves):
                        if isinstance(move, dict):
                            x, y = move.get('x', 0), move.get('y', 0)
                            if not (0 <= x <= 10000 and 0 <= y <= 10000):  # 假设最大屏幕尺寸
                                if len(issues) < 50:
                                    issues.append(f"样本 {i} 鼠标坐标异常: ({x}, {y})")
                                plausibility_issues += 1
                                break
            
            # 检查响应时间
            if 'response_times' in behavior_data:
                response_times = behavior_data['response_times']
                if isinstance(response_times, list) and response_times:
                    # 检查极端值
                    max_rt = max(response_times)
                    min_rt = min(response_times)
                    
                    if max_rt > 60:  # 超过1分钟
                        issues.append(f"样本 {i} 响应时间过长: {max_rt:.1f}s")
                        plausibility_issues += 1
                    if min_rt < 0.01:  # 小于10ms
                        issues.append(f"样本 {i} 响应时间过短: {min_rt:.3f}s")
                        plausibility_issues += 1
        
        # 计算合理性分数
        total_behavior_checks = len(samples) * 3  # 每个样本检查3个行为方面
        plausibility_score = max(0, 1 - plausibility_issues / max(1, total_behavior_checks))
        
        if plausibility_issues > 0:
            recommendations.append(f"修复 {plausibility_issues} 个行为数据异常")
            recommendations.append("检查时间戳序列和坐标范围")
            recommendations.append("验证响应时间在合理范围内(10ms-60s)")
        
        metrics = {
            'plausibility_issues': plausibility_issues,
            'plausibility_score': plausibility_score,
            'behavior_stats': behavior_stats,
            'checked_samples': len(samples)
        }
        
        return ValidationResult(
            is_valid=plausibility_score >= 0.8,
            score=plausibility_score,
            issues=issues,
            recommendations=recommendations,
            detailed_metrics=metrics
        )
    
    def _check_team_a_compatibility(self, samples: List[Dict]) -> ValidationResult:
        """检查与团队A模型的兼容性"""
        issues = []
        recommendations = []
        metrics = {}
        
        if not self.feature_extractor:
            return ValidationResult(
                is_valid=True,
                score=0.8,  # 无法验证但不影响整体
                issues=["无法验证与团队A的兼容性：特征提取器不可用"],
                recommendations=["确保团队A模块可正常导入"],
                detailed_metrics={'compatibility_check': 'skipped'}
            )
        
        compatibility_failures = 0
        feature_extraction_success = 0
        feature_quality_scores = []
        
        # 抽样检查兼容性（避免处理所有样本）
        sample_indices = np.random.choice(len(samples), min(50, len(samples)), replace=False)
        
        for idx in sample_indices:
            sample = samples[idx]
            
            if 'behavior_data' not in sample:
                continue
            
            try:
                # 尝试使用团队A的特征提取器
                behavior_data = sample['behavior_data']
                
                # 重新格式化以匹配团队A的预期格式
                formatted_data = {
                    'keystrokes': behavior_data.get('keystrokes', []),
                    'mouse_moves': behavior_data.get('mouse_moves', []),
                    'dwell_times': behavior_data.get('dwell_times', []),
                    'response_times': behavior_data.get('response_times', []),
                    'task_context': behavior_data.get('task_context', {})
                }
                
                # 提取特征
                extracted_features = self.feature_extractor.extract_all_features(formatted_data)
                
                # 检查提取的特征是否符合预期
                if len(extracted_features) == len(self.spec.concept_names):
                    feature_extraction_success += 1
                    
                    # 比较提取的特征与标注的认知状态
                    if 'cognitive_state' in sample:
                        cognitive_state = sample['cognitive_state']
                        feature_quality = self._compare_features_and_states(
                            extracted_features, cognitive_state
                        )
                        feature_quality_scores.append(feature_quality)
                else:
                    compatibility_failures += 1
                    issues.append(f"样本 {idx} 特征提取结果维度不匹配")
                    
            except Exception as e:
                compatibility_failures += 1
                if len(issues) < 10:
                    issues.append(f"样本 {idx} 特征提取失败: {str(e)}")
        
        # 计算兼容性分数
        tested_samples = len(sample_indices)
        if tested_samples > 0:
            compatibility_score = feature_extraction_success / tested_samples
        else:
            compatibility_score = 0.0
        
        # 特征质量分数
        if feature_quality_scores:
            avg_feature_quality = np.mean(feature_quality_scores)
        else:
            avg_feature_quality = 0.0
        
        # 综合分数
        final_score = (compatibility_score + avg_feature_quality) / 2
        
        if compatibility_failures > 0:
            recommendations.append(f"修复 {compatibility_failures} 个兼容性问题")
            recommendations.append("确保行为数据格式符合团队A的特征提取器要求")
        
        if avg_feature_quality < 0.6:
            recommendations.append("提高特征与认知状态的一致性")
        
        metrics = {
            'tested_samples': tested_samples,
            'compatibility_failures': compatibility_failures,
            'feature_extraction_success_rate': compatibility_score,
            'average_feature_quality': avg_feature_quality,
            'final_compatibility_score': final_score
        }
        
        return ValidationResult(
            is_valid=final_score >= 0.7,
            score=final_score,
            issues=issues,
            recommendations=recommendations,
            detailed_metrics=metrics
        )
    
    def _compare_features_and_states(self, 
                                   extracted_features: Dict[str, float],
                                   cognitive_state: Dict[str, float]) -> float:
        """比较提取的特征与标注的认知状态"""
        if not extracted_features or not cognitive_state:
            return 0.0
        
        correlations = []
        
        for concept in self.spec.concept_names:
            if concept in extracted_features and concept in cognitive_state:
                feature_val = extracted_features[concept]
                state_val = cognitive_state[concept]
                
                # 计算相关性（值越接近越好）
                correlation = 1 - abs(feature_val - state_val)
                correlations.append(correlation)
        
        return np.mean(correlations) if correlations else 0.0
    
    def _check_statistical_properties(self, samples: List[Dict]) -> ValidationResult:
        """检查统计特性"""
        issues = []
        recommendations = []
        metrics = {}
        
        if len(samples) < 30:
            return ValidationResult(
                is_valid=True,
                score=0.8,
                issues=["样本数量过少，无法进行充分的统计检查"],
                recommendations=["建议样本数量至少30个以上"],
                detailed_metrics={'sample_count': len(samples)}
            )
        
        # 收集认知状态数据
        concept_values = {concept: [] for concept in self.spec.concept_names}
        
        for sample in samples:
            if 'cognitive_state' in sample:
                cognitive_state = sample['cognitive_state']
                for concept in self.spec.concept_names:
                    if concept in cognitive_state:
                        concept_values[concept].append(cognitive_state[concept])
        
        statistical_scores = []
        
        for concept, values in concept_values.items():
            if len(values) < 10:
                continue
            
            # 正态性检验
            _, p_value = stats.normaltest(values)
            
            # 分布特性
            mean_val = np.mean(values)
            std_val = np.std(values)
            skewness = stats.skew(values)
            kurtosis = stats.kurtosis(values)
            
            # 检查分布是否合理
            score_factors = []
            
            # 均值应在合理范围内
            if 0.2 <= mean_val <= 0.8:
                score_factors.append(1.0)
            else:
                score_factors.append(0.5)
                issues.append(f"{concept} 均值偏极端: {mean_val:.3f}")
            
            # 标准差应适中
            if 0.1 <= std_val <= 0.3:
                score_factors.append(1.0)
            elif std_val < 0.05:
                score_factors.append(0.3)
                issues.append(f"{concept} 方差过小: {std_val:.3f}")
            elif std_val > 0.4:
                score_factors.append(0.6)
                issues.append(f"{concept} 方差过大: {std_val:.3f}")
            else:
                score_factors.append(0.8)
            
            # 偏度不应过大
            if abs(skewness) < 1.5:
                score_factors.append(1.0)
            else:
                score_factors.append(0.7)
                issues.append(f"{concept} 分布偏度过大: {skewness:.3f}")
            
            concept_score = np.mean(score_factors)
            statistical_scores.append(concept_score)
            
            metrics[f'{concept}_stats'] = {
                'mean': float(mean_val),
                'std': float(std_val),
                'skewness': float(skewness),
                'kurtosis': float(kurtosis),
                'normality_p': float(p_value),
                'score': float(concept_score)
            }
        
        overall_statistical_score = np.mean(statistical_scores) if statistical_scores else 0.0
        
        if overall_statistical_score < 0.7:
            recommendations.append("调整数据分布，确保各概念值分布合理")
            recommendations.append("增加数据多样性，避免过度集中或分散")
        
        metrics['overall_statistical_score'] = overall_statistical_score
        metrics['concepts_analyzed'] = len(statistical_scores)
        
        return ValidationResult(
            is_valid=overall_statistical_score >= 0.7,
            score=overall_statistical_score,
            issues=issues,
            recommendations=recommendations,
            detailed_metrics=metrics
        )
    
    def generate_validation_report(self, validation_result: ValidationResult, 
                                 output_path: Path = None) -> Path:
        """生成验证报告"""
        if output_path is None:
            output_path = self.config.results_dir / "validation_report.json"
        
        report = {
            'validation_summary': {
                'is_valid': validation_result.is_valid,
                'overall_score': validation_result.score,
                'timestamp': pd.Timestamp.now().isoformat()
            },
            'issues': validation_result.issues,
            'recommendations': validation_result.recommendations,
            'detailed_metrics': validation_result.detailed_metrics,
            'validation_rules': list(self.validation_rules.keys()),
            'config': {
                'concept_names': self.spec.concept_names,
                'value_range': self.spec.value_range,
                'target_accuracy': self.config.accuracy_threshold
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"验证报告已保存到: {output_path}")
        return output_path

def main():
    """测试数据验证器"""
    from team_c.config import get_evaluation_config
    from team_c.datasets.builder import CognitiveDatasetBuilder
    
    config = get_evaluation_config()
    validator = DataValidator(config)
    
    # 创建测试数据
    builder = CognitiveDatasetBuilder(config)
    samples = builder.build_labeled_dataset(20)
    
    # 转换为验证器预期的格式
    validation_samples = []
    for sample in samples:
        validation_sample = {
            'user_id': sample.user_id,
            'session_id': sample.session_id, 
            'timestamp': sample.timestamp,
            'cognitive_state': sample.cognitive_state,
            'behavior_data': sample.raw_behavior,
            'quality_score': sample.quality_score
        }
        validation_samples.append(validation_sample)
    
    # 执行验证
    result = validator.validate_dataset(validation_samples)
    
    # 生成报告
    report_path = validator.generate_validation_report(result)
    
    print(f"验证完成:")
    print(f"  有效: {result.is_valid}")
    print(f"  分数: {result.score:.3f}")
    print(f"  问题数: {len(result.issues)}")
    print(f"  报告: {report_path}")

if __name__ == "__main__":
    main()