"""
A/B测试框架
支持对比测试不同的模型版本和配置
"""

import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time
from enum import Enum
import sys

sys.path.append("/Users/pluviophile/chi2025")
from team_c.config import EvaluationConfig, CognitiveStateSpec

logger = logging.getLogger(__name__)

class TestStatus(Enum):
    """测试状态"""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ABTestGroup:
    """A/B测试组"""
    group_id: str
    group_name: str
    description: str
    config: Dict[str, Any]
    sample_size: int
    participants: List[str]
    results: List[Dict[str, Any]]
    
@dataclass 
class ABTestResult:
    """A/B测试结果"""
    test_id: str
    control_group: str
    treatment_groups: List[str]
    primary_metric: str
    control_mean: float
    treatment_means: Dict[str, float]
    p_values: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    effect_sizes: Dict[str, float]
    statistical_power: float
    sample_sizes: Dict[str, int]
    is_significant: bool
    recommendations: List[str]
    detailed_analysis: Dict[str, Any]

class ABTestFramework:
    """
    A/B测试框架
    支持多组对比测试和统计分析
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.spec = CognitiveStateSpec()
        
        # 测试配置
        self.significance_level = config.statistical_significance
        self.minimum_sample_size = config.minimum_sample_size
        
        # 活跃测试
        self.active_tests: Dict[str, Dict] = {}
        
        # 预定义指标
        self.available_metrics = {
            'accuracy': self._calculate_accuracy_metric,
            'latency': self._calculate_latency_metric,
            'user_satisfaction': self._calculate_satisfaction_metric,
            'engagement': self._calculate_engagement_metric,
            'cognitive_improvement': self._calculate_cognitive_improvement_metric
        }
        
        logger.info("A/B测试框架初始化完成")
    
    def create_test(self,
                   test_id: str,
                   test_name: str,
                   description: str,
                   primary_metric: str,
                   groups: List[Dict[str, Any]],
                   duration_days: int = 7) -> str:
        """
        创建A/B测试
        
        Args:
            test_id: 测试ID
            test_name: 测试名称
            description: 测试描述
            primary_metric: 主要指标
            groups: 测试组配置列表
            duration_days: 测试持续天数
            
        Returns:
            创建的测试ID
        """
        if test_id in self.active_tests:
            raise ValueError(f"测试ID {test_id} 已存在")
        
        if primary_metric not in self.available_metrics:
            raise ValueError(f"不支持的指标: {primary_metric}")
        
        # 验证测试组配置
        if len(groups) < 2:
            raise ValueError("至少需要2个测试组")
        
        # 创建测试组对象
        test_groups = {}
        for group_config in groups:
            group_id = group_config['group_id']
            test_groups[group_id] = ABTestGroup(
                group_id=group_id,
                group_name=group_config['group_name'],
                description=group_config.get('description', ''),
                config=group_config.get('config', {}),
                sample_size=group_config.get('sample_size', self.minimum_sample_size),
                participants=[],
                results=[]
            )
        
        # 创建测试记录
        test_record = {
            'test_id': test_id,
            'test_name': test_name,
            'description': description,
            'primary_metric': primary_metric,
            'groups': test_groups,
            'status': TestStatus.CREATED,
            'created_at': time.time(),
            'started_at': None,
            'ended_at': None,
            'duration_days': duration_days,
            'results': None
        }
        
        self.active_tests[test_id] = test_record
        
        logger.info(f"A/B测试创建成功: {test_id}")
        logger.info(f"  测试名称: {test_name}")
        logger.info(f"  主要指标: {primary_metric}")
        logger.info(f"  测试组数: {len(groups)}")
        
        return test_id
    
    def start_test(self, test_id: str):
        """开始A/B测试"""
        if test_id not in self.active_tests:
            raise ValueError(f"测试ID {test_id} 不存在")
        
        test_record = self.active_tests[test_id]
        
        if test_record['status'] != TestStatus.CREATED:
            raise ValueError(f"测试 {test_id} 当前状态为 {test_record['status']}，无法启动")
        
        test_record['status'] = TestStatus.RUNNING
        test_record['started_at'] = time.time()
        
        logger.info(f"A/B测试启动: {test_id}")
    
    def assign_participant(self, test_id: str, participant_id: str, group_id: str = None) -> str:
        """
        分配参与者到测试组
        
        Args:
            test_id: 测试ID
            participant_id: 参与者ID
            group_id: 指定的组ID，None则随机分配
            
        Returns:
            分配到的组ID
        """
        if test_id not in self.active_tests:
            raise ValueError(f"测试ID {test_id} 不存在")
        
        test_record = self.active_tests[test_id]
        groups = test_record['groups']
        
        # 检查参与者是否已分配
        for gid, group in groups.items():
            if participant_id in group.participants:
                return gid  # 已分配，返回现有组
        
        # 分配逻辑
        if group_id is None:
            # 随机分配到样本量未满的组
            available_groups = [
                gid for gid, group in groups.items()
                if len(group.participants) < group.sample_size
            ]
            
            if not available_groups:
                raise ValueError("所有测试组已满")
            
            group_id = np.random.choice(available_groups)
        
        if group_id not in groups:
            raise ValueError(f"组ID {group_id} 不存在")
        
        # 检查组容量
        group = groups[group_id]
        if len(group.participants) >= group.sample_size:
            raise ValueError(f"组 {group_id} 已满")
        
        # 添加参与者
        group.participants.append(participant_id)
        
        logger.debug(f"参与者 {participant_id} 分配到组 {group_id}")
        return group_id
    
    def record_result(self,
                     test_id: str,
                     participant_id: str,
                     metrics: Dict[str, float],
                     additional_data: Optional[Dict] = None):
        """
        记录测试结果
        
        Args:
            test_id: 测试ID
            participant_id: 参与者ID
            metrics: 指标值
            additional_data: 额外数据
        """
        if test_id not in self.active_tests:
            raise ValueError(f"测试ID {test_id} 不存在")
        
        test_record = self.active_tests[test_id]
        groups = test_record['groups']
        
        # 找到参与者所在组
        participant_group = None
        for group_id, group in groups.items():
            if participant_id in group.participants:
                participant_group = group
                break
        
        if participant_group is None:
            raise ValueError(f"参与者 {participant_id} 未分配到任何组")
        
        # 记录结果
        result_record = {
            'participant_id': participant_id,
            'timestamp': time.time(),
            'metrics': metrics,
            'additional_data': additional_data or {}
        }
        
        participant_group.results.append(result_record)
        
        logger.debug(f"记录测试结果: 测试={test_id}, 参与者={participant_id}, 组={participant_group.group_id}")
    
    def analyze_test(self, test_id: str) -> ABTestResult:
        """
        分析A/B测试结果
        
        Args:
            test_id: 测试ID
            
        Returns:
            测试分析结果
        """
        if test_id not in self.active_tests:
            raise ValueError(f"测试ID {test_id} 不存在")
        
        test_record = self.active_tests[test_id]
        groups = test_record['groups']
        primary_metric = test_record['primary_metric']
        
        logger.info(f"开始分析A/B测试: {test_id}")
        
        # 收集各组数据
        group_data = {}
        for group_id, group in groups.items():
            group_metrics = []
            
            for result in group.results:
                if primary_metric in result['metrics']:
                    group_metrics.append(result['metrics'][primary_metric])
            
            group_data[group_id] = group_metrics
            logger.info(f"组 {group_id}: {len(group_metrics)} 个有效样本")
        
        # 确定对照组（通常是第一个或名为'control'的组）
        control_group_id = self._determine_control_group(groups)
        treatment_group_ids = [gid for gid in groups.keys() if gid != control_group_id]
        
        # 统计检验
        control_data = group_data[control_group_id]
        control_mean = np.mean(control_data) if control_data else 0.0
        
        treatment_means = {}
        p_values = {}
        confidence_intervals = {}
        effect_sizes = {}
        sample_sizes = {gid: len(data) for gid, data in group_data.items()}
        
        for treatment_id in treatment_group_ids:
            treatment_data = group_data[treatment_id]
            
            if len(control_data) < 2 or len(treatment_data) < 2:
                logger.warning(f"组 {treatment_id} 或对照组样本不足，跳过统计检验")
                treatment_means[treatment_id] = np.mean(treatment_data) if treatment_data else 0.0
                p_values[treatment_id] = 1.0
                confidence_intervals[treatment_id] = (0.0, 0.0)
                effect_sizes[treatment_id] = 0.0
                continue
            
            treatment_mean = np.mean(treatment_data)
            treatment_means[treatment_id] = treatment_mean
            
            # t检验
            t_stat, p_value = stats.ttest_ind(control_data, treatment_data)
            p_values[treatment_id] = p_value
            
            # 效应量 (Cohen's d)
            pooled_std = np.sqrt(((np.std(control_data, ddof=1)**2) + 
                                (np.std(treatment_data, ddof=1)**2)) / 2)
            cohens_d = (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0.0
            effect_sizes[treatment_id] = cohens_d
            
            # 置信区间
            se = np.sqrt((np.var(control_data, ddof=1)/len(control_data)) + 
                        (np.var(treatment_data, ddof=1)/len(treatment_data)))
            dof = len(control_data) + len(treatment_data) - 2
            t_critical = stats.t.ppf(1 - self.significance_level/2, dof)
            margin_error = t_critical * se
            
            confidence_intervals[treatment_id] = (
                (treatment_mean - control_mean) - margin_error,
                (treatment_mean - control_mean) + margin_error
            )
        
        # 统计功效
        statistical_power = self._calculate_statistical_power(group_data)
        
        # 显著性判断
        is_significant = any(p < self.significance_level for p in p_values.values())
        
        # 生成建议
        recommendations = self._generate_recommendations(
            control_mean, treatment_means, p_values, effect_sizes, sample_sizes
        )
        
        # 详细分析
        detailed_analysis = self._detailed_ab_analysis(test_record, group_data)
        
        result = ABTestResult(
            test_id=test_id,
            control_group=control_group_id,
            treatment_groups=treatment_group_ids,
            primary_metric=primary_metric,
            control_mean=control_mean,
            treatment_means=treatment_means,
            p_values=p_values,
            confidence_intervals=confidence_intervals,
            effect_sizes=effect_sizes,
            statistical_power=statistical_power,
            sample_sizes=sample_sizes,
            is_significant=is_significant,
            recommendations=recommendations,
            detailed_analysis=detailed_analysis
        )
        
        # 更新测试记录
        test_record['results'] = result
        
        self._log_ab_result(result)
        return result
    
    def _determine_control_group(self, groups: Dict[str, ABTestGroup]) -> str:
        """确定对照组"""
        # 优先选择名为'control'的组
        for group_id in groups.keys():
            if 'control' in group_id.lower():
                return group_id
        
        # 否则选择第一个组
        return list(groups.keys())[0]
    
    def _calculate_statistical_power(self, group_data: Dict[str, List[float]]) -> float:
        """计算统计功效"""
        # 简化的功效计算
        min_sample_size = min(len(data) for data in group_data.values() if data)
        
        if min_sample_size < 10:
            return 0.2
        elif min_sample_size < 30:
            return 0.5
        elif min_sample_size < 100:
            return 0.8
        else:
            return 0.95
    
    def _generate_recommendations(self,
                                control_mean: float,
                                treatment_means: Dict[str, float],
                                p_values: Dict[str, float],
                                effect_sizes: Dict[str, float],
                                sample_sizes: Dict[str, int]) -> List[str]:
        """生成测试建议"""
        recommendations = []
        
        # 样本量检查
        min_sample_size = min(sample_sizes.values())
        if min_sample_size < self.minimum_sample_size:
            recommendations.append(f"样本量不足，建议至少 {self.minimum_sample_size} 个样本")
        
        # 显著性分析
        significant_groups = [gid for gid, p in p_values.items() if p < self.significance_level]
        
        if significant_groups:
            for group_id in significant_groups:
                improvement = ((treatment_means[group_id] - control_mean) / control_mean * 100) if control_mean > 0 else 0
                recommendations.append(f"组 {group_id} 显示显著差异，改进 {improvement:.1f}%")
        else:
            recommendations.append("未检测到显著差异，考虑延长测试时间或增加样本量")
        
        # 效应量分析
        for group_id, effect_size in effect_sizes.items():
            if abs(effect_size) < 0.2:
                recommendations.append(f"组 {group_id} 效应量较小 ({effect_size:.3f})，可能没有实际意义")
            elif abs(effect_size) > 0.8:
                recommendations.append(f"组 {group_id} 效应量很大 ({effect_size:.3f})，建议重点关注")
        
        return recommendations
    
    def _detailed_ab_analysis(self, test_record: Dict, group_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """详细A/B分析"""
        analysis = {
            'test_duration': time.time() - test_record['started_at'] if test_record['started_at'] else 0,
            'group_statistics': {},
            'distribution_analysis': {},
            'trend_analysis': {}
        }
        
        # 各组统计
        for group_id, data in group_data.items():
            if data:
                analysis['group_statistics'][group_id] = {
                    'count': len(data),
                    'mean': float(np.mean(data)),
                    'median': float(np.median(data)),
                    'std': float(np.std(data)),
                    'min': float(np.min(data)),
                    'max': float(np.max(data)),
                    'q25': float(np.percentile(data, 25)),
                    'q75': float(np.percentile(data, 75))
                }
                
                # 正态性检验
                _, normality_p = stats.normaltest(data)
                analysis['distribution_analysis'][group_id] = {
                    'normality_p_value': float(normality_p),
                    'is_normal': normality_p > 0.05
                }
        
        return analysis
    
    def _calculate_accuracy_metric(self, result_data: Dict) -> float:
        """计算准确率指标"""
        return result_data.get('accuracy', 0.0)
    
    def _calculate_latency_metric(self, result_data: Dict) -> float:
        """计算延迟指标"""
        return result_data.get('latency', 0.0)
    
    def _calculate_satisfaction_metric(self, result_data: Dict) -> float:
        """计算用户满意度指标"""
        return result_data.get('user_satisfaction', 0.0)
    
    def _calculate_engagement_metric(self, result_data: Dict) -> float:
        """计算用户参与度指标"""
        return result_data.get('engagement', 0.0)
    
    def _calculate_cognitive_improvement_metric(self, result_data: Dict) -> float:
        """计算认知改善指标"""
        return result_data.get('cognitive_improvement', 0.0)
    
    def _log_ab_result(self, result: ABTestResult):
        """记录A/B测试结果"""
        logger.info(f"=== A/B测试结果: {result.test_id} ===")
        logger.info(f"主要指标: {result.primary_metric}")
        logger.info(f"对照组 {result.control_group}: {result.control_mean:.4f}")
        
        for treatment_id in result.treatment_groups:
            treatment_mean = result.treatment_means[treatment_id]
            p_value = result.p_values[treatment_id]
            effect_size = result.effect_sizes[treatment_id]
            
            logger.info(f"实验组 {treatment_id}: {treatment_mean:.4f} (p={p_value:.4f}, d={effect_size:.3f})")
        
        logger.info(f"统计显著性: {'是' if result.is_significant else '否'}")
        logger.info(f"统计功效: {result.statistical_power:.2f}")
    
    def plot_ab_results(self, result: ABTestResult, save_path: Path = None) -> Path:
        """绘制A/B测试结果"""
        if save_path is None:
            save_path = self.config.results_dir / f"ab_test_{result.test_id}.png"
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 组间对比
        groups = [result.control_group] + result.treatment_groups
        means = [result.control_mean] + [result.treatment_means[gid] for gid in result.treatment_groups]
        
        ax1 = axes[0, 0]
        bars = ax1.bar(groups, means, alpha=0.7)
        ax1.set_title(f'组间 {result.primary_metric} 对比')
        ax1.set_ylabel(result.primary_metric)
        
        # 添加数值标签
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(means) * 0.01,
                    f'{mean:.3f}', ha='center', va='bottom')
        
        # 2. P值和显著性
        treatment_groups = result.treatment_groups
        p_values = [result.p_values[gid] for gid in treatment_groups]
        
        ax2 = axes[0, 1]
        bars = ax2.bar(treatment_groups, p_values, alpha=0.7)
        ax2.axhline(y=self.significance_level, color='r', linestyle='--', 
                   label=f'显著性水平 ({self.significance_level})')
        ax2.set_title('P值分析')
        ax2.set_ylabel('P值')
        ax2.set_yscale('log')
        ax2.legend()
        
        # 添加显著性标注
        for bar, p_val in zip(bars, p_values):
            height = bar.get_height()
            significance = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            ax2.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                    significance, ha='center', va='bottom')
        
        # 3. 效应量
        effect_sizes = [result.effect_sizes[gid] for gid in treatment_groups]
        
        ax3 = axes[1, 0]
        bars = ax3.bar(treatment_groups, effect_sizes, alpha=0.7)
        ax3.axhline(y=0.2, color='orange', linestyle='--', alpha=0.7, label='小效应 (0.2)')
        ax3.axhline(y=0.5, color='green', linestyle='--', alpha=0.7, label='中效应 (0.5)')
        ax3.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='大效应 (0.8)')
        ax3.set_title("效应量 (Cohen's d)")
        ax3.set_ylabel("Cohen's d")
        ax3.legend()
        
        # 4. 样本量和功效
        sample_sizes = [result.sample_sizes[gid] for gid in groups]
        
        ax4 = axes[1, 1]
        ax4.bar(groups, sample_sizes, alpha=0.7, label='样本量')
        ax4.axhline(y=self.minimum_sample_size, color='r', linestyle='--', 
                   label=f'最小样本量 ({self.minimum_sample_size})')
        ax4.set_title('样本量分布')
        ax4.set_ylabel('样本量')
        ax4.legend()
        
        # 添加统计功效信息
        ax4.text(0.02, 0.98, f'统计功效: {result.statistical_power:.2f}',
                transform=ax4.transAxes, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"A/B测试图表已保存到: {save_path}")
        return save_path
    
    def save_ab_report(self, result: ABTestResult, filepath: Path = None) -> Path:
        """保存A/B测试报告"""
        if filepath is None:
            filepath = self.config.results_dir / f"ab_test_report_{result.test_id}.json"
        
        report = {
            'test_summary': {
                'test_id': result.test_id,
                'primary_metric': result.primary_metric,
                'control_group': result.control_group,
                'treatment_groups': result.treatment_groups,
                'is_significant': result.is_significant,
                'statistical_power': result.statistical_power
            },
            'results': {
                'control_mean': result.control_mean,
                'treatment_means': result.treatment_means,
                'p_values': result.p_values,
                'confidence_intervals': {k: list(v) for k, v in result.confidence_intervals.items()},
                'effect_sizes': result.effect_sizes,
                'sample_sizes': result.sample_sizes
            },
            'recommendations': result.recommendations,
            'detailed_analysis': result.detailed_analysis,
            'configuration': {
                'significance_level': self.significance_level,
                'minimum_sample_size': self.minimum_sample_size
            },
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"A/B测试报告已保存到: {filepath}")
        return filepath
    
    def get_test_status(self, test_id: str) -> Dict[str, Any]:
        """获取测试状态"""
        if test_id not in self.active_tests:
            raise ValueError(f"测试ID {test_id} 不存在")
        
        test_record = self.active_tests[test_id]
        groups = test_record['groups']
        
        status = {
            'test_id': test_id,
            'status': test_record['status'].value,
            'created_at': test_record['created_at'],
            'started_at': test_record['started_at'],
            'duration_days': test_record['duration_days'],
            'groups_status': {}
        }
        
        for group_id, group in groups.items():
            status['groups_status'][group_id] = {
                'participants': len(group.participants),
                'target_sample_size': group.sample_size,
                'results_collected': len(group.results),
                'completion_rate': len(group.results) / group.sample_size if group.sample_size > 0 else 0
            }
        
        return status

def main():
    """测试A/B测试框架"""
    from team_c.config import get_evaluation_config
    
    config = get_evaluation_config()
    ab_framework = ABTestFramework(config)
    
    # 创建测试
    test_id = ab_framework.create_test(
        test_id="accuracy_test_001",
        test_name="模型准确率对比测试",
        description="对比基线模型和改进模型的准确率",
        primary_metric="accuracy",
        groups=[
            {
                'group_id': 'control',
                'group_name': '基线模型',
                'description': 'DIFCM基线版本',
                'sample_size': 50
            },
            {
                'group_id': 'treatment',
                'group_name': '改进模型',
                'description': 'DIFCM改进版本',
                'sample_size': 50
            }
        ]
    )
    
    # 启动测试
    ab_framework.start_test(test_id)
    
    # 模拟参与者分配和结果记录
    for i in range(80):
        participant_id = f"user_{i:03d}"
        group_id = ab_framework.assign_participant(test_id, participant_id)
        
        # 模拟结果数据
        if group_id == 'control':
            accuracy = np.random.normal(0.75, 0.1)  # 基线准确率
        else:
            accuracy = np.random.normal(0.80, 0.1)  # 改进准确率
        
        accuracy = np.clip(accuracy, 0.0, 1.0)
        
        ab_framework.record_result(
            test_id, participant_id, 
            {'accuracy': accuracy, 'latency': np.random.normal(300, 50)}
        )
    
    # 分析结果
    result = ab_framework.analyze_test(test_id)
    
    # 生成图表和报告
    plot_path = ab_framework.plot_ab_results(result)
    report_path = ab_framework.save_ab_report(result)
    
    print(f"A/B测试完成:")
    print(f"  测试ID: {test_id}")
    print(f"  显著差异: {'是' if result.is_significant else '否'}")
    print(f"  对照组准确率: {result.control_mean:.3f}")
    print(f"  实验组准确率: {result.treatment_means['treatment']:.3f}")
    print(f"  P值: {result.p_values['treatment']:.4f}")
    print(f"  图表: {plot_path}")
    print(f"  报告: {report_path}")

if __name__ == "__main__":
    main()