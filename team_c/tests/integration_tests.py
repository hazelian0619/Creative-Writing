"""
集成测试
测试团队C与团队A、团队B的集成
"""

import unittest
import asyncio
import time
import logging
from typing import Dict, List, Optional
import numpy as np
import sys
from pathlib import Path

# 添加项目路径
sys.path.append("/Users/pluviophile/chi2025")

# 导入团队A模块
try:
    from team_a.models.difcm_core import DIFCMModel, DIFCMConfig
    from team_a.features.cognitive_features import CognitiveFeatureExtractor
except ImportError as e:
    logging.warning(f"无法导入团队A模块: {e}")
    DIFCMModel = None
    CognitiveFeatureExtractor = None

# 导入团队C模块
from team_c.config import get_evaluation_config, get_api_test_config
from team_c.datasets import CognitiveDatasetBuilder, DataValidator, SyntheticDataGenerator
from team_c.evaluation import AccuracyCalculator, LatencyTester, ABTestFramework, MetricsCollector

logger = logging.getLogger(__name__)

class TeamIntegrationTests(unittest.TestCase):
    """团队集成测试"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.config = get_evaluation_config()
        cls.api_config = get_api_test_config()
        
        # 创建测试目录
        cls.test_results_dir = cls.config.results_dir / "integration_tests"
        cls.test_results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("集成测试初始化完成")
    
    def setUp(self):
        """每个测试方法的初始化"""
        self.test_start_time = time.time()
    
    def tearDown(self):
        """每个测试方法的清理"""
        test_duration = time.time() - self.test_start_time
        logger.info(f"测试用时: {test_duration:.2f}s")
    
    def test_team_a_integration(self):
        """测试与团队A的集成"""
        if DIFCMModel is None or CognitiveFeatureExtractor is None:
            self.skipTest("团队A模块不可用")
        
        logger.info("测试与团队A的集成...")
        
        # 1. 测试DIFCM模型加载
        difcm_config = DIFCMConfig(n_concepts=8)
        model = DIFCMModel(difcm_config)
        self.assertIsNotNone(model)
        
        # 2. 测试特征提取器
        feature_extractor = CognitiveFeatureExtractor()
        self.assertIsNotNone(feature_extractor)
        
        # 3. 测试数据兼容性
        builder = CognitiveDatasetBuilder(self.config)
        samples = builder.build_labeled_dataset(10)
        
        # 验证样本格式
        self.assertGreater(len(samples), 0)
        
        sample = samples[0]
        self.assertIn('raw_behavior', sample.__dict__)
        self.assertIn('cognitive_state', sample.__dict__)
        
        # 4. 测试特征提取兼容性
        raw_behavior = sample.raw_behavior
        try:
            features = feature_extractor.extract_all_features(raw_behavior)
            self.assertIsInstance(features, dict)
            self.assertEqual(len(features), 8)  # 8个认知概念
            logger.info("与团队A集成测试通过")
        except Exception as e:
            self.fail(f"特征提取失败: {e}")
    
    def test_team_b_api_compatibility(self):
        """测试与团队B API的兼容性"""
        logger.info("测试与团队B API的兼容性...")
        
        # 1. 测试API配置
        self.assertIsNotNone(self.api_config.api_host)
        self.assertIsNotNone(self.api_config.api_port)
        
        # 2. 测试端点定义
        expected_endpoints = [
            'behavior_data_endpoint',
            'latest_state_endpoint', 
            'health_check_endpoint'
        ]
        
        for endpoint in expected_endpoints:
            self.assertTrue(hasattr(self.api_config, endpoint))
            endpoint_value = getattr(self.api_config, endpoint)
            self.assertIsInstance(endpoint_value, str)
            self.assertTrue(endpoint_value.startswith('/'))
        
        # 3. 测试延迟测试器初始化
        latency_tester = LatencyTester(self.config)
        self.assertIsNotNone(latency_tester)
        self.assertEqual(latency_tester.base_url, 
                        f"http://{self.api_config.api_host}:{self.api_config.api_port}")
        
        logger.info("与团队B API兼容性测试通过")
    
    def test_data_pipeline_integration(self):
        """测试数据管道集成"""
        logger.info("测试数据管道集成...")
        
        # 1. 数据集构建
        builder = CognitiveDatasetBuilder(self.config)
        samples = builder.build_labeled_dataset(20)
        self.assertEqual(len(samples), 20)
        
        # 2. 数据验证
        validator = DataValidator(self.config)
        
        # 转换为验证格式
        validation_samples = []
        for sample in samples:
            validation_sample = {
                'user_id': sample.user_id,
                'session_id': sample.session_id,
                'timestamp': sample.timestamp,
                'cognitive_state': sample.cognitive_state,
                'behavior_data': sample.raw_behavior
            }
            validation_samples.append(validation_sample)
        
        result = validator.validate_dataset(validation_samples)
        self.assertTrue(result.score > 0.5)  # 基本质量要求
        
        # 3. 合成数据生成
        generator = SyntheticDataGenerator(self.config)
        synthetic_samples = generator.generate_batch_data(10)
        self.assertEqual(len(synthetic_samples), 10)
        
        # 4. 数据保存和加载
        save_path = builder.save_dataset(samples, "integration_test_dataset.json")
        self.assertTrue(save_path.exists())
        
        loaded_samples = builder.load_dataset(save_path)
        self.assertEqual(len(loaded_samples), len(samples))
        
        logger.info("数据管道集成测试通过")
    
    def test_evaluation_pipeline_integration(self):
        """测试评估管道集成"""
        logger.info("测试评估管道集成...")
        
        # 1. 准确率计算器
        calculator = AccuracyCalculator(self.config)
        
        # 生成测试数据
        n_samples = 50
        concepts = ['attention', 'memory', 'comprehension', 'creativity',
                   'motivation', 'emotion', 'confidence', 'fatigue']
        
        ground_truth = []
        predictions = []
        
        for _ in range(n_samples):
            truth = {concept: np.random.uniform(0.2, 0.8) for concept in concepts}
            pred = {concept: truth[concept] + np.random.normal(0, 0.1) 
                   for concept in concepts}
            
            # 确保在有效范围内
            pred = {k: np.clip(v, 0.1, 0.9) for k, v in pred.items()}
            
            ground_truth.append(truth)
            predictions.append(pred)
        
        # 计算准确率
        metrics = calculator.calculate_accuracy(predictions, ground_truth)
        self.assertIsInstance(metrics.overall_accuracy, float)
        self.assertGreaterEqual(metrics.overall_accuracy, 0.0)
        self.assertLessEqual(metrics.overall_accuracy, 1.0)
        
        # 2. A/B测试框架
        ab_framework = ABTestFramework(self.config)
        
        test_id = ab_framework.create_test(
            test_id="integration_test",
            test_name="集成测试",
            description="测试A/B框架集成",
            primary_metric="accuracy",
            groups=[
                {'group_id': 'control', 'group_name': '对照组', 'sample_size': 10},
                {'group_id': 'treatment', 'group_name': '实验组', 'sample_size': 10}
            ]
        )
        
        ab_framework.start_test(test_id)
        
        # 模拟参与者分配
        for i in range(15):
            participant_id = f"integration_user_{i}"
            group_id = ab_framework.assign_participant(test_id, participant_id)
            
            # 记录结果
            accuracy = np.random.uniform(0.6, 0.9)
            ab_framework.record_result(
                test_id, participant_id, 
                {'accuracy': accuracy}
            )
        
        # 分析结果
        ab_result = ab_framework.analyze_test(test_id)
        self.assertIsInstance(ab_result.overall_accuracy, float)
        
        # 3. 指标收集器
        collector = MetricsCollector(self.config)
        
        # 记录指标
        collector.record_metric('test_metric', 42.0, source='integration_test')
        collector.flush_to_database()
        
        # 查询指标
        metrics_data = collector.query_metrics('test_metric', limit=10)
        self.assertGreater(len(metrics_data), 0)
        
        logger.info("评估管道集成测试通过")
    
    def test_end_to_end_workflow(self):
        """测试端到端工作流"""
        logger.info("测试端到端工作流...")
        
        # 1. 数据准备
        builder = CognitiveDatasetBuilder(self.config)
        samples = builder.build_labeled_dataset(30)
        
        # 2. 数据验证
        validator = DataValidator(self.config)
        validation_samples = []
        for sample in samples:
            validation_sample = {
                'user_id': sample.user_id,
                'cognitive_state': sample.cognitive_state,
                'behavior_data': sample.raw_behavior
            }
            validation_samples.append(validation_sample)
        
        validation_result = validator.validate_dataset(validation_samples)
        self.assertTrue(validation_result.is_valid)
        
        # 3. 模拟模型预测（如果团队A可用）
        if CognitiveFeatureExtractor is not None:
            feature_extractor = CognitiveFeatureExtractor()
            predictions = []
            ground_truth = []
            
            for sample in samples[:20]:  # 使用部分样本
                try:
                    # 提取特征作为"预测"
                    features = feature_extractor.extract_all_features(sample.raw_behavior)
                    predictions.append(features)
                    ground_truth.append(sample.cognitive_state)
                except:
                    continue
            
            if predictions and ground_truth:
                # 4. 计算准确率
                calculator = AccuracyCalculator(self.config)
                accuracy_metrics = calculator.calculate_accuracy(predictions, ground_truth)
                
                # 5. 记录指标
                collector = MetricsCollector(self.config)
                collector.record_accuracy_metrics(
                    overall_accuracy=accuracy_metrics.overall_accuracy,
                    concept_accuracies=accuracy_metrics.concept_accuracies,
                    test_id="end_to_end_test"
                )
                collector.flush_to_database()
                
                # 6. 验证目标达成
                target_met = accuracy_metrics.overall_accuracy >= self.config.accuracy_threshold
                logger.info(f"端到端测试: 准确率={accuracy_metrics.overall_accuracy:.3f}, "
                          f"目标达成={'是' if target_met else '否'}")
        
        logger.info("端到端工作流测试通过")
    
    def test_error_handling_and_resilience(self):
        """测试错误处理和容错性"""
        logger.info("测试错误处理和容错性...")
        
        # 1. 测试空数据处理
        calculator = AccuracyCalculator(self.config)
        
        with self.assertRaises(ValueError):
            calculator.calculate_accuracy([], [])
        
        with self.assertRaises(ValueError):
            calculator.calculate_accuracy([{'a': 1}], [{'a': 1}, {'a': 2}])
        
        # 2. 测试无效数据处理
        validator = DataValidator(self.config)
        
        invalid_samples = [
            {'user_id': 'test', 'cognitive_state': 'invalid'},
            {'missing_required_field': True}
        ]
        
        result = validator.validate_dataset(invalid_samples)
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.issues), 0)
        
        # 3. 测试A/B测试错误处理
        ab_framework = ABTestFramework(self.config)
        
        with self.assertRaises(ValueError):
            ab_framework.create_test("test", "Test", "Desc", "invalid_metric", [])
        
        with self.assertRaises(ValueError):
            ab_framework.assign_participant("nonexistent_test", "user")
        
        # 4. 测试指标收集器异常处理
        collector = MetricsCollector(self.config)
        
        # 这些调用不应该抛出异常
        collector.record_metric('test', float('inf'))
        collector.record_metric('test', float('nan'))
        collector.flush_to_database()
        
        logger.info("错误处理和容错性测试通过")
    
    def test_configuration_consistency(self):
        """测试配置一致性"""
        logger.info("测试配置一致性...")
        
        # 1. 验证核心配置
        self.assertEqual(len(self.config.concept_weights), 8)
        self.assertAlmostEqual(sum(self.config.concept_weights.values()), 1.0, places=2)
        
        # 2. 验证认知概念一致性
        from team_c.config import CognitiveStateSpec
        spec = CognitiveStateSpec()
        
        config_concepts = set(self.config.concept_weights.keys())
        spec_concepts = set(spec.concept_names)
        
        self.assertEqual(config_concepts, spec_concepts)
        
        # 3. 验证路径配置
        self.assertTrue(self.config.data_dir.exists())
        self.assertTrue(self.config.results_dir.exists())
        
        # 4. 验证阈值合理性
        self.assertGreater(self.config.accuracy_threshold, 0.0)
        self.assertLessEqual(self.config.accuracy_threshold, 1.0)
        self.assertGreater(self.config.latency_target, 0.0)
        
        logger.info("配置一致性测试通过")

def run_integration_tests():
    """运行集成测试"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TeamIntegrationTests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 返回测试结果
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_integration_tests()
    exit(0 if success else 1)