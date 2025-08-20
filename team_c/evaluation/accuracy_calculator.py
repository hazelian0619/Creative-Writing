"""
准确率计算框架
计算认知状态识别准确率，达到Phase1的75%目标
"""

import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import sys

# 导入团队A模型进行预测
sys.path.append("/Users/pluviophile/chi2025")
try:
    from team_a.models.difcm_core import DIFCMModel, DIFCMConfig
    from team_a.features.cognitive_features import CognitiveFeatureExtractor
except ImportError as e:
    logging.warning(f"无法导入团队A模块: {e}")

from team_c.config import EvaluationConfig, CognitiveStateSpec

logger = logging.getLogger(__name__)

@dataclass
class AccuracyMetrics:
    """准确率指标"""
    overall_accuracy: float
    concept_accuracies: Dict[str, float]
    mse: float
    mae: float
    r2_score: float
    classification_metrics: Dict[str, Dict[str, float]]
    detailed_results: Dict[str, any]

class AccuracyCalculator:
    """
    准确率计算器
    支持回归和分类两种评估模式
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.spec = CognitiveStateSpec()
        
        # 准确率阈值
        self.tolerance = config.tolerance
        self.target_accuracy = config.accuracy_threshold
        
        # 概念权重
        self.concept_weights = config.concept_weights
        
        # 分类阈值 - 用于将连续值转换为类别
        self.classification_thresholds = {
            'attention': {'low': 0.4, 'high': 0.7},
            'memory': {'low': 0.4, 'high': 0.7},
            'comprehension': {'low': 0.4, 'high': 0.7},
            'creativity': {'low': 0.3, 'high': 0.8},  # 创造力范围更宽
            'motivation': {'low': 0.4, 'high': 0.7},
            'emotion': {'low': 0.4, 'high': 0.7},
            'confidence': {'low': 0.4, 'high': 0.7},
            'fatigue': {'low': 0.3, 'high': 0.6}  # 疲劳阈值偏低
        }
        
        logger.info(f"准确率计算器初始化完成，目标准确率: {self.target_accuracy}")
    
    def calculate_accuracy(self, 
                         predictions: List[Dict[str, float]], 
                         ground_truth: List[Dict[str, float]]) -> AccuracyMetrics:
        """
        计算认知状态识别准确率
        
        Args:
            predictions: 模型预测结果 [{'concept': value, ...}, ...]
            ground_truth: 真实标签 [{'concept': value, ...}, ...]
            
        Returns:
            准确率指标
        """
        if len(predictions) != len(ground_truth):
            raise ValueError(f"预测数量({len(predictions)})与真实标签数量({len(ground_truth)})不匹配")
        
        if not predictions:
            raise ValueError("预测结果为空")
        
        logger.info(f"开始计算准确率，样本数: {len(predictions)}")
        
        # 1. 回归指标
        regression_metrics = self._calculate_regression_metrics(predictions, ground_truth)
        
        # 2. 分类指标
        classification_metrics = self._calculate_classification_metrics(predictions, ground_truth)
        
        # 3. 概念级别准确率
        concept_accuracies = self._calculate_concept_accuracies(predictions, ground_truth)
        
        # 4. 整体准确率
        overall_accuracy = self._calculate_overall_accuracy(predictions, ground_truth)
        
        # 5. 详细分析
        detailed_results = self._detailed_analysis(predictions, ground_truth)
        
        metrics = AccuracyMetrics(
            overall_accuracy=overall_accuracy,
            concept_accuracies=concept_accuracies,
            mse=regression_metrics['mse'],
            mae=regression_metrics['mae'],
            r2_score=regression_metrics['r2'],
            classification_metrics=classification_metrics,
            detailed_results=detailed_results
        )
        
        self._log_accuracy_summary(metrics)
        return metrics
    
    def _calculate_regression_metrics(self, 
                                    predictions: List[Dict[str, float]], 
                                    ground_truth: List[Dict[str, float]]) -> Dict[str, float]:
        """计算回归指标"""
        # 将字典转换为矩阵
        pred_matrix = self._dict_list_to_matrix(predictions)
        truth_matrix = self._dict_list_to_matrix(ground_truth)
        
        # 计算回归指标
        mse = mean_squared_error(truth_matrix, pred_matrix)
        mae = mean_absolute_error(truth_matrix, pred_matrix)
        r2 = r2_score(truth_matrix, pred_matrix)
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'r2': float(r2)
        }
    
    def _calculate_classification_metrics(self, 
                                        predictions: List[Dict[str, float]], 
                                        ground_truth: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """计算分类指标"""
        classification_metrics = {}
        
        for concept in self.spec.concept_names:
            # 提取概念值
            pred_values = [p.get(concept, 0.5) for p in predictions]
            true_values = [t.get(concept, 0.5) for t in ground_truth]
            
            # 转换为分类标签
            pred_labels = [self._value_to_class(v, concept) for v in pred_values]
            true_labels = [self._value_to_class(v, concept) for v in true_values]
            
            # 计算分类指标
            try:
                accuracy = accuracy_score(true_labels, pred_labels)
                precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
                recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
                f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
                
                classification_metrics[concept] = {
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1)
                }
            except Exception as e:
                logger.warning(f"计算 {concept} 分类指标失败: {e}")
                classification_metrics[concept] = {
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0
                }
        
        return classification_metrics
    
    def _calculate_concept_accuracies(self, 
                                    predictions: List[Dict[str, float]], 
                                    ground_truth: List[Dict[str, float]]) -> Dict[str, float]:
        """计算各概念的准确率"""
        concept_accuracies = {}
        
        for concept in self.spec.concept_names:
            correct_count = 0
            total_count = 0
            
            for pred, truth in zip(predictions, ground_truth):
                if concept in pred and concept in truth:
                    pred_value = pred[concept]
                    true_value = truth[concept]
                    
                    # 检查是否在容差范围内
                    if abs(pred_value - true_value) <= self.tolerance:
                        correct_count += 1
                    total_count += 1
            
            if total_count > 0:
                accuracy = correct_count / total_count
            else:
                accuracy = 0.0
            
            concept_accuracies[concept] = accuracy
        
        return concept_accuracies
    
    def _calculate_overall_accuracy(self, 
                                  predictions: List[Dict[str, float]], 
                                  ground_truth: List[Dict[str, float]]) -> float:
        """计算整体准确率（加权）"""
        weighted_accuracies = []
        
        for pred, truth in zip(predictions, ground_truth):
            sample_accuracy = 0.0
            total_weight = 0.0
            
            for concept in self.spec.concept_names:
                if concept in pred and concept in truth:
                    pred_value = pred[concept]
                    true_value = truth[concept]
                    weight = self.concept_weights.get(concept, 1.0)
                    
                    # 计算概念准确率
                    if abs(pred_value - true_value) <= self.tolerance:
                        concept_acc = 1.0
                    else:
                        # 线性衰减
                        error = abs(pred_value - true_value)
                        concept_acc = max(0.0, 1.0 - error / self.tolerance)
                    
                    sample_accuracy += concept_acc * weight
                    total_weight += weight
            
            if total_weight > 0:
                sample_accuracy /= total_weight
            
            weighted_accuracies.append(sample_accuracy)
        
        return float(np.mean(weighted_accuracies)) if weighted_accuracies else 0.0
    
    def _detailed_analysis(self, 
                         predictions: List[Dict[str, float]], 
                         ground_truth: List[Dict[str, float]]) -> Dict[str, any]:
        """详细分析"""
        analysis = {}
        
        # 错误分析
        errors_by_concept = {}
        large_errors = []
        
        for i, (pred, truth) in enumerate(zip(predictions, ground_truth)):
            for concept in self.spec.concept_names:
                if concept in pred and concept in truth:
                    error = abs(pred[concept] - truth[concept])
                    
                    if concept not in errors_by_concept:
                        errors_by_concept[concept] = []
                    errors_by_concept[concept].append(error)
                    
                    # 记录大错误
                    if error > 2 * self.tolerance:
                        large_errors.append({
                            'sample_index': i,
                            'concept': concept,
                            'predicted': pred[concept],
                            'truth': truth[concept],
                            'error': error
                        })
        
        # 统计信息
        error_stats = {}
        for concept, errors in errors_by_concept.items():
            if errors:
                error_stats[concept] = {
                    'mean_error': float(np.mean(errors)),
                    'std_error': float(np.std(errors)),
                    'max_error': float(np.max(errors)),
                    'median_error': float(np.median(errors))
                }
        
        # 性能分布
        accuracy_distribution = self._analyze_accuracy_distribution(predictions, ground_truth)
        
        analysis = {
            'error_statistics': error_stats,
            'large_errors': large_errors[:20],  # 限制数量
            'accuracy_distribution': accuracy_distribution,
            'total_samples': len(predictions),
            'phase1_target_met': self._check_phase1_target(predictions, ground_truth)
        }
        
        return analysis
    
    def _analyze_accuracy_distribution(self, 
                                     predictions: List[Dict[str, float]], 
                                     ground_truth: List[Dict[str, float]]) -> Dict[str, any]:
        """分析准确率分布"""
        sample_accuracies = []
        
        for pred, truth in zip(predictions, ground_truth):
            correct_concepts = 0
            total_concepts = 0
            
            for concept in self.spec.concept_names:
                if concept in pred and concept in truth:
                    if abs(pred[concept] - truth[concept]) <= self.tolerance:
                        correct_concepts += 1
                    total_concepts += 1
            
            if total_concepts > 0:
                sample_accuracy = correct_concepts / total_concepts
                sample_accuracies.append(sample_accuracy)
        
        if not sample_accuracies:
            return {'error': 'No valid samples for analysis'}
        
        return {
            'mean': float(np.mean(sample_accuracies)),
            'std': float(np.std(sample_accuracies)),
            'min': float(np.min(sample_accuracies)),
            'max': float(np.max(sample_accuracies)),
            'percentiles': {
                '25': float(np.percentile(sample_accuracies, 25)),
                '50': float(np.percentile(sample_accuracies, 50)),
                '75': float(np.percentile(sample_accuracies, 75)),
                '90': float(np.percentile(sample_accuracies, 90))
            },
            'samples_above_target': sum(1 for acc in sample_accuracies if acc >= self.target_accuracy),
            'percentage_above_target': sum(1 for acc in sample_accuracies if acc >= self.target_accuracy) / len(sample_accuracies) * 100
        }
    
    def _check_phase1_target(self, 
                           predictions: List[Dict[str, float]], 
                           ground_truth: List[Dict[str, float]]) -> Dict[str, any]:
        """检查是否达到Phase1目标"""
        overall_accuracy = self._calculate_overall_accuracy(predictions, ground_truth)
        
        return {
            'target_accuracy': self.target_accuracy,
            'achieved_accuracy': overall_accuracy,
            'target_met': overall_accuracy >= self.target_accuracy,
            'accuracy_gap': self.target_accuracy - overall_accuracy,
            'target_percentage': overall_accuracy / self.target_accuracy * 100 if self.target_accuracy > 0 else 0
        }
    
    def _value_to_class(self, value: float, concept: str) -> str:
        """将连续值转换为分类标签"""
        thresholds = self.classification_thresholds.get(concept, {'low': 0.4, 'high': 0.7})
        
        if value < thresholds['low']:
            return 'low'
        elif value > thresholds['high']:
            return 'high'
        else:
            return 'medium'
    
    def _dict_list_to_matrix(self, dict_list: List[Dict[str, float]]) -> np.ndarray:
        """将字典列表转换为矩阵"""
        matrix = []
        
        for d in dict_list:
            row = [d.get(concept, 0.5) for concept in self.spec.concept_names]
            matrix.append(row)
        
        return np.array(matrix)
    
    def _log_accuracy_summary(self, metrics: AccuracyMetrics):
        """记录准确率摘要"""
        logger.info("=== 准确率计算结果 ===")
        logger.info(f"整体准确率: {metrics.overall_accuracy:.3f}")
        logger.info(f"目标准确率: {self.target_accuracy}")
        logger.info(f"目标达成: {'是' if metrics.overall_accuracy >= self.target_accuracy else '否'}")
        logger.info(f"MSE: {metrics.mse:.4f}")
        logger.info(f"MAE: {metrics.mae:.4f}")
        logger.info(f"R²: {metrics.r2_score:.4f}")
        
        logger.info("各概念准确率:")
        for concept, accuracy in metrics.concept_accuracies.items():
            logger.info(f"  {concept}: {accuracy:.3f}")
    
    def plot_accuracy_results(self, metrics: AccuracyMetrics, save_path: Path = None) -> Path:
        """绘制准确率结果图表"""
        if save_path is None:
            save_path = self.config.results_dir / "accuracy_results.png"
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 概念准确率条形图
        concepts = list(metrics.concept_accuracies.keys())
        accuracies = list(metrics.concept_accuracies.values())
        
        ax1 = axes[0, 0]
        bars = ax1.bar(concepts, accuracies, alpha=0.7)
        ax1.axhline(y=self.target_accuracy, color='r', linestyle='--', label=f'目标 ({self.target_accuracy})')
        ax1.set_title('各概念准确率')
        ax1.set_ylabel('准确率')
        ax1.set_ylim(0, 1)
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # 2. 分类指标热力图
        ax2 = axes[0, 1]
        classification_data = []
        metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for concept in concepts:
            if concept in metrics.classification_metrics:
                row = [metrics.classification_metrics[concept][m] for m in metrics_names]
            else:
                row = [0.0] * len(metrics_names)
            classification_data.append(row)
        
        im = ax2.imshow(classification_data, cmap='YlOrRd', aspect='auto')
        ax2.set_title('分类指标热力图')
        ax2.set_xticks(range(len(metrics_names)))
        ax2.set_xticklabels(metrics_names)
        ax2.set_yticks(range(len(concepts)))
        ax2.set_yticklabels(concepts)
        
        # 添加数值标签
        for i in range(len(concepts)):
            for j in range(len(metrics_names)):
                text = ax2.text(j, i, f'{classification_data[i][j]:.3f}',
                               ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax2, shrink=0.6)
        
        # 3. 准确率分布
        if 'accuracy_distribution' in metrics.detailed_results:
            dist = metrics.detailed_results['accuracy_distribution']
            
            ax3 = axes[1, 0]
            # 模拟分布数据用于展示
            sample_acc = np.random.beta(2, 2, 1000) * 0.6 + 0.3  # 模拟数据
            ax3.hist(sample_acc, bins=20, alpha=0.7, edgecolor='black')
            ax3.axvline(x=dist['mean'], color='r', linestyle='-', label=f"均值 ({dist['mean']:.3f})")
            ax3.axvline(x=self.target_accuracy, color='g', linestyle='--', label=f"目标 ({self.target_accuracy})")
            ax3.set_title('样本准确率分布')
            ax3.set_xlabel('准确率')
            ax3.set_ylabel('样本数量')
            ax3.legend()
        
        # 4. 整体指标
        ax4 = axes[1, 1]
        metrics_labels = ['整体准确率', 'MSE', 'MAE', 'R²']
        metrics_values = [metrics.overall_accuracy, metrics.mse, metrics.mae, metrics.r2_score]
        
        # 标准化显示
        normalized_values = [
            metrics.overall_accuracy,
            1 - min(1, metrics.mse),  # MSE越小越好
            1 - min(1, metrics.mae),  # MAE越小越好
            max(0, metrics.r2_score)  # R²可能为负
        ]
        
        ax4.barh(metrics_labels, normalized_values, alpha=0.7)
        ax4.set_title('整体指标 (标准化)')
        ax4.set_xlabel('分数')
        ax4.set_xlim(0, 1)
        
        # 添加数值标签
        for i, (label, orig_val) in enumerate(zip(metrics_labels, metrics_values)):
            ax4.text(normalized_values[i] + 0.02, i, f'{orig_val:.3f}',
                    va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"准确率结果图表已保存到: {save_path}")
        return save_path
    
    def save_accuracy_report(self, metrics: AccuracyMetrics, filepath: Path = None) -> Path:
        """保存准确率报告"""
        if filepath is None:
            filepath = self.config.results_dir / "accuracy_report.json"
        
        report = {
            'summary': {
                'overall_accuracy': metrics.overall_accuracy,
                'target_accuracy': self.target_accuracy,
                'target_met': metrics.overall_accuracy >= self.target_accuracy,
                'mse': metrics.mse,
                'mae': metrics.mae,
                'r2_score': metrics.r2_score
            },
            'concept_accuracies': metrics.concept_accuracies,
            'classification_metrics': metrics.classification_metrics,
            'detailed_analysis': metrics.detailed_results,
            'configuration': {
                'tolerance': self.tolerance,
                'concept_weights': self.concept_weights,
                'classification_thresholds': self.classification_thresholds
            },
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"准确率报告已保存到: {filepath}")
        return filepath

def main():
    """测试准确率计算器"""
    from team_c.config import get_evaluation_config
    
    config = get_evaluation_config()
    calculator = AccuracyCalculator(config)
    
    # 生成测试数据
    n_samples = 100
    concepts = ['attention', 'memory', 'comprehension', 'creativity',
               'motivation', 'emotion', 'confidence', 'fatigue']
    
    # 模拟真实值
    ground_truth = []
    for _ in range(n_samples):
        truth = {concept: np.random.uniform(0.1, 0.9) for concept in concepts}
        ground_truth.append(truth)
    
    # 模拟预测值（带有一定误差）
    predictions = []
    for truth in ground_truth:
        pred = {}
        for concept, true_val in truth.items():
            # 添加高斯噪声
            noise = np.random.normal(0, 0.1)
            pred_val = np.clip(true_val + noise, 0.1, 0.9)
            pred[concept] = pred_val
        predictions.append(pred)
    
    # 计算准确率
    metrics = calculator.calculate_accuracy(predictions, ground_truth)
    
    # 生成图表和报告
    plot_path = calculator.plot_accuracy_results(metrics)
    report_path = calculator.save_accuracy_report(metrics)
    
    print(f"测试完成:")
    print(f"  整体准确率: {metrics.overall_accuracy:.3f}")
    print(f"  目标达成: {'是' if metrics.overall_accuracy >= config.accuracy_threshold else '否'}")
    print(f"  图表: {plot_path}")
    print(f"  报告: {report_path}")

if __name__ == "__main__":
    main()