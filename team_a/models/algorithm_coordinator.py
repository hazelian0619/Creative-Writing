"""
算法协调框架
集成DIFCM、T-S模糊推理、GraphSAGE三大核心算法
"""

import asyncio
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import deque
import json
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import traceback

# 导入自定义模块
from team_a.models.difcm_enhanced import EnhancedDIFCMModel, DIFCMConfig
from team_a.models.ts_fuzzy_engine import TSFuzzyInferenceEngine
from team_a.models.graphsage_reasoning import (
    CreativeWritingGraphSAGE, ProgressivePromptGenerator, KnowledgeGraph
)
from team_a.features.cognitive_features import CognitiveFeatureExtractor

@dataclass
class InferenceRequest:
    """推理请求数据结构"""
    request_id: str
    user_id: str
    session_id: str
    input_data: Dict[str, Any]
    context_history: List[Dict[str, Any]]
    timestamp: float
    priority: int = 1  # 1=高, 2=中, 3=低

@dataclass
class InferenceResult:
    """推理结果数据结构"""
    request_id: str
    user_id: str
    ai_response: Dict[str, Any]
    cognitive_state: np.ndarray
    inference_metadata: Dict[str, Any]
    processing_time: float
    success: bool
    error_message: Optional[str] = None

@dataclass
class CoordinatorConfig:
    """协调器配置"""
    max_concurrent_requests: int = 10
    timeout_seconds: float = 30.0
    cache_enabled: bool = True
    performance_monitoring: bool = True
    error_recovery: bool = True
    
class AlgorithmCoordinator:
    """算法协调器 - 统筹三个核心算法的协作"""
    
    def __init__(self, config: CoordinatorConfig):
        self.config = config
        
        # 初始化核心算法模块
        self._initialize_algorithms()
        
        # 协调策略
        self.coordination_strategy = 'adaptive'
        
        # 性能监控
        self.performance_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_processing_time': 0.0,
            'algorithm_times': {
                'difcm': deque(maxlen=100),
                'fuzzy': deque(maxlen=100),
                'graphsage': deque(maxlen=100),
                'coordination': deque(maxlen=100)
            }
        }
        
        # 请求队列和缓存
        self.request_queue = asyncio.Queue(maxsize=100)
        self.result_cache = {}
        self.active_requests = {}
        
        # 线程池执行器
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_requests)
        
        logging.info("算法协调器初始化完成")
    
    def _initialize_algorithms(self):
        """初始化核心算法"""
        try:
            # 1. 初始化DIFCM模型
            difcm_config = DIFCMConfig(
                n_concepts=8,
                learning_rate=0.01,
                decay_rate=0.95
            )
            self.difcm_engine = EnhancedDIFCMModel(difcm_config)
            
            # 2. 初始化模糊推理引擎
            self.fuzzy_engine = TSFuzzyInferenceEngine()
            
            # 3. 初始化GraphSAGE推理网络
            self.graphsage_model = CreativeWritingGraphSAGE(
                input_dim=256,
                hidden_dim=128,
                output_dim=64,
                num_layers=3
            )
            
            # 4. 初始化知识图谱和提示生成器
            self.knowledge_graph = KnowledgeGraph()
            self.prompt_generator = ProgressivePromptGenerator(
                self.graphsage_model, self.knowledge_graph
            )
            
            # 5. 初始化特征提取器
            self.feature_extractor = CognitiveFeatureExtractor()
            
            logging.info("所有核心算法模块初始化成功")
            
        except Exception as e:
            logging.error(f"算法初始化失败: {e}")
            raise
    
    async def process_request(self, request: InferenceRequest) -> InferenceResult:
        """
        处理推理请求的主要流程
        
        Args:
            request: 推理请求
            
        Returns:
            推理结果
        """
        start_time = time.time()
        
        try:
            # 更新请求统计
            self.performance_metrics['total_requests'] += 1
            self.active_requests[request.request_id] = request
            
            # 检查缓存
            if self.config.cache_enabled:
                cached_result = self._check_cache(request)
                if cached_result:
                    return cached_result
            
            # 执行推理管道
            result = await self._execute_inference_pipeline(request)
            
            # 缓存结果
            if self.config.cache_enabled and result.success:
                self._cache_result(request, result)
            
            # 更新性能指标
            processing_time = time.time() - start_time
            self._update_performance_metrics(processing_time, result.success)
            
            return result
            
        except asyncio.TimeoutError:
            error_msg = f"推理请求超时: {request.request_id}"
            logging.warning(error_msg)
            return self._create_error_result(request, error_msg, time.time() - start_time)
            
        except Exception as e:
            error_msg = f"推理请求失败: {request.request_id}, 错误: {str(e)}"
            logging.error(error_msg)
            logging.error(traceback.format_exc())
            return self._create_error_result(request, error_msg, time.time() - start_time)
            
        finally:
            # 清理活跃请求
            if request.request_id in self.active_requests:
                del self.active_requests[request.request_id]
    
    async def _execute_inference_pipeline(self, request: InferenceRequest) -> InferenceResult:
        """执行完整的推理管道"""
        start_time = time.time()
        pipeline_results = {}
        
        # 1. 认知特征提取
        feature_start = time.time()
        cognitive_features = await self._extract_cognitive_features(request.input_data)
        pipeline_results['feature_extraction_time'] = time.time() - feature_start
        
        # 2. DIFCM认知状态更新
        difcm_start = time.time()
        difcm_result = await self._run_difcm_inference(
            request.user_id, cognitive_features, request.context_history
        )
        difcm_time = time.time() - difcm_start
        self.performance_metrics['algorithm_times']['difcm'].append(difcm_time)
        pipeline_results['difcm_time'] = difcm_time
        
        # 3. 模糊推理决策
        fuzzy_start = time.time()
        fuzzy_result = await self._run_fuzzy_inference(difcm_result)
        fuzzy_time = time.time() - fuzzy_start
        self.performance_metrics['algorithm_times']['fuzzy'].append(fuzzy_time)
        pipeline_results['fuzzy_time'] = fuzzy_time
        
        # 4. GraphSAGE知识推理
        graph_start = time.time()
        graph_result = await self._run_graphsage_inference(
            request.input_data, difcm_result['cognitive_state']
        )
        graph_time = time.time() - graph_start
        self.performance_metrics['algorithm_times']['graphsage'].append(graph_time)
        pipeline_results['graphsage_time'] = graph_time
        
        # 5. 算法协调与输出生成
        coord_start = time.time()
        final_result = await self._coordinate_output(
            graph_result['layered_prompts'],
            fuzzy_result['prompt_intensity'],
            fuzzy_result['prompt_type'],
            fuzzy_result['calibration_level'],
            difcm_result
        )\n        coord_time = time.time() - coord_start\n        self.performance_metrics['algorithm_times']['coordination'].append(coord_time)\n        pipeline_results['coordination_time'] = coord_time\n        \n        total_time = time.time() - start_time\n        \n        # 构建最终结果\n        return InferenceResult(\n            request_id=request.request_id,\n            user_id=request.user_id,\n            ai_response=final_result,\n            cognitive_state=difcm_result['cognitive_state'],\n            inference_metadata={\n                'pipeline_results': pipeline_results,\n                'fuzzy_metadata': fuzzy_result,\n                'difcm_metadata': difcm_result,\n                'graph_metadata': graph_result\n            },\n            processing_time=total_time,\n            success=True\n        )\n    \n    async def _extract_cognitive_features(self, input_data: Dict[str, Any]) -> Dict[str, float]:\n        \"\"\"提取认知特征\"\"\"\n        loop = asyncio.get_event_loop()\n        \n        # 构建特征提取的输入数据\n        behavioral_data = {\n            'keystrokes': input_data.get('keystrokes', []),\n            'mouse_moves': input_data.get('mouse_moves', []),\n            'dwell_times': input_data.get('dwell_times', []),\n            'response_times': input_data.get('response_times', []),\n            'task_context': input_data.get('task_context', {})\n        }\n        \n        # 在线程池中执行CPU密集的特征提取\n        features = await loop.run_in_executor(\n            self.executor,\n            self.feature_extractor.extract_all_features,\n            behavioral_data\n        )\n        \n        return features\n    \n    async def _run_difcm_inference(self, user_id: str, features: Dict[str, float], \n                                  context_history: List[Dict]) -> Dict[str, Any]:\n        \"\"\"运行DIFCM认知状态推理\"\"\"\n        loop = asyncio.get_event_loop()\n        \n        def _difcm_forward():\n            # 准备输入特征\n            feature_vector = np.array(list(features.values()))\n            input_tensor = torch.FloatTensor(feature_vector).unsqueeze(0)\n            \n            # 构建交互数据\n            interaction_data = {\n                'response_times': [ctx.get('response_time', 1.0) for ctx in context_history[-5:]],\n                'keystrokes': [{'key': 'text', 'timestamp': time.time()}]  # 简化版本\n            }\n            \n            # DIFCM前向传播\n            with torch.no_grad():\n                result = self.difcm_engine(input_tensor, interaction_data)\n            \n            return {\n                'cognitive_state': result['cognitive_state'][0].numpy(),\n                'drift_score': result['drift_scores'][0].item(),\n                'drift_direction': result['drift_directions'][0].numpy(),\n                'cognitive_load': result['cognitive_loads'][0].item(),\n                'load_components': result['load_components'][0]\n            }\n        \n        return await loop.run_in_executor(self.executor, _difcm_forward)\n    \n    async def _run_fuzzy_inference(self, difcm_result: Dict[str, Any]) -> Dict[str, Any]:\n        \"\"\"运行模糊推理\"\"\"\n        loop = asyncio.get_event_loop()\n        \n        def _fuzzy_inference():\n            # 准备模糊推理输入\n            cognitive_load = difcm_result['cognitive_load']\n            drift_level = difcm_result['drift_score']\n            \n            # 简化的语义覆盖率计算\n            cognitive_state = difcm_result['cognitive_state']\n            semantic_coverage = np.mean(cognitive_state[[1, 2, 3]])  # memory, comprehension, creativity\n            \n            # 简化的响应时间\n            response_time = 5.0  # 默认值\n            \n            # 执行模糊推理\n            result = self.fuzzy_engine.infer(\n                cognitive_load, drift_level, semantic_coverage, response_time\n            )\n            \n            return result\n        \n        return await loop.run_in_executor(self.executor, _fuzzy_inference)\n    \n    async def _run_graphsage_inference(self, input_data: Dict[str, Any], \n                                     cognitive_state: np.ndarray) -> Dict[str, Any]:\n        \"\"\"运行GraphSAGE知识推理\"\"\"\n        loop = asyncio.get_event_loop()\n        \n        def _graph_inference():\n            # 提取主题概念\n            topic_text = input_data.get('text', input_data.get('input_text', ''))\n            topic_concepts = self._extract_topic_concepts(topic_text)\n            \n            # 构建用户上下文\n            user_context = {\n                'writing_goal': input_data.get('writing_goal', '创意写作'),\n                'genre': input_data.get('genre', '通用'),\n                'difficulty_level': input_data.get('difficulty_level', 'medium')\n            }\n            \n            # 生成分层提示\n            layered_prompts = self.prompt_generator.generate_layered_prompts(\n                topic_concepts, user_context, cognitive_state\n            )\n            \n            return {\n                'layered_prompts': layered_prompts,\n                'topic_concepts': topic_concepts,\n                'stats': {\n                    'concepts_found': len(topic_concepts),\n                    'prompts_generated': sum(len(prompts) for prompts in layered_prompts.values())\n                }\n            }\n        \n        return await loop.run_in_executor(self.executor, _graph_inference)\n    \n    async def _coordinate_output(self, layered_prompts: Dict[str, List[str]], \n                               intensity: float, prompt_type: str, \n                               calibration_level: str, difcm_result: Dict) -> Dict[str, Any]:\n        \"\"\"协调算法输出生成最终响应\"\"\"\n        loop = asyncio.get_event_loop()\n        \n        def _coordinate():\n            # 1. 基于强度选择合适的提示层级\n            selected_prompts = self._select_prompt_layer(layered_prompts, intensity)\n            \n            # 2. 根据类型调整提示风格\n            styled_prompts = self._apply_prompt_style(selected_prompts, prompt_type)\n            \n            # 3. 根据校准级别进行精度调节\n            calibrated_prompts = self._apply_calibration(styled_prompts, calibration_level)\n            \n            # 4. 生成最终响应\n            final_response = self._generate_final_response(\n                calibrated_prompts, difcm_result, intensity\n            )\n            \n            return final_response\n        \n        return await loop.run_in_executor(self.executor, _coordinate)\n    \n    def _select_prompt_layer(self, layered_prompts: Dict[str, List[str]], \n                           intensity: float) -> List[str]:\n        \"\"\"基于强度选择提示层级\"\"\"\n        if intensity < 0.3:\n            return layered_prompts.get('layer1', [])  # 轻度引导\n        elif intensity < 0.7:\n            layer1 = layered_prompts.get('layer1', [])\n            layer2 = layered_prompts.get('layer2', [])\n            return layer1 + layer2[:2]  # 中度，联想+结构\n        else:\n            layer2 = layered_prompts.get('layer2', [])\n            layer3 = layered_prompts.get('layer3', [])\n            return layer2 + layer3  # 高强度，结构+操作\n    \n    def _apply_prompt_style(self, prompts: List[str], style_type: str) -> List[str]:\n        \"\"\"应用提示风格\"\"\"\n        style_prefixes = {\n            '启发式': '💡 ',\n            '结构化': '📋 ',\n            '示例化': '📝 '\n        }\n        \n        prefix = style_prefixes.get(style_type, '')\n        return [f\"{prefix}{prompt}\" for prompt in prompts]\n    \n    def _apply_calibration(self, prompts: List[str], level: str) -> List[str]:\n        \"\"\"应用校准级别\"\"\"\n        if level == '轻度':\n            return prompts[:2]  # 只保留前两个提示\n        elif level == '中度':\n            return prompts[:3]  # 保留前三个提示\n        else:  # 重度\n            emergency_prompts = [\n                \"🚨 让我们重新整理思路，从最简单的一个想法开始。\",\n                \"🎯 建议暂停创作，先明确你的核心目标。\"\n            ]\n            return prompts + emergency_prompts\n    \n    def _generate_final_response(self, prompts: List[str], \n                               difcm_result: Dict, intensity: float) -> Dict[str, Any]:\n        \"\"\"生成最终AI响应\"\"\"\n        cognitive_state = difcm_result['cognitive_state']\n        \n        # 认知状态解释\n        interpretation = self.difcm_engine.get_interpretation()\n        \n        # 构建响应\n        response = {\n            'prompts': prompts,\n            'cognitive_feedback': {\n                'current_state': interpretation,\n                'cognitive_load': difcm_result['cognitive_load'],\n                'drift_detected': difcm_result['drift_score'] > 0.5,\n                'recommendations': self._generate_recommendations(cognitive_state)\n            },\n            'system_info': {\n                'prompt_intensity': intensity,\n                'response_confidence': min(1.0, 1.0 - difcm_result['drift_score']),\n                'processing_quality': 'high' if intensity > 0.6 else 'standard'\n            }\n        }\n        \n        return response\n    \n    def _generate_recommendations(self, cognitive_state: np.ndarray) -> List[str]:\n        \"\"\"基于认知状态生成建议\"\"\"\n        recommendations = []\n        \n        if len(cognitive_state) >= 8:\n            if cognitive_state[0] < 0.4:  # 注意力低\n                recommendations.append(\"建议适当休息，提高注意力集中度\")\n            if cognitive_state[3] < 0.4:  # 创造力低\n                recommendations.append(\"尝试一些创意激发练习\")\n            if cognitive_state[7] > 0.7:  # 疲劳度高\n                recommendations.append(\"注意劳逸结合，避免过度疲劳\")\n        \n        if not recommendations:\n            recommendations.append(\"当前状态良好，继续保持\")\n        \n        return recommendations\n    \n    def _extract_topic_concepts(self, text: str) -> List[str]:\n        \"\"\"从文本中提取主题概念（简化版）\"\"\"\n        # 简化的关键词提取\n        keywords = ['写作', '创意', '故事', '人物', '情节', '描写', '叙事']\n        found_concepts = []\n        \n        text_lower = text.lower()\n        for keyword in keywords:\n            if keyword in text_lower:\n                found_concepts.append(keyword)\n        \n        return found_concepts[:3]  # 限制数量\n    \n    def _check_cache(self, request: InferenceRequest) -> Optional[InferenceResult]:\n        \"\"\"检查结果缓存\"\"\"\n        # 简化的缓存键生成\n        cache_key = f\"{request.user_id}_{hash(str(request.input_data))}\"\n        \n        if cache_key in self.result_cache:\n            cached_data = self.result_cache[cache_key]\n            # 检查缓存时效性（5分钟）\n            if time.time() - cached_data['timestamp'] < 300:\n                logging.info(f\"使用缓存结果: {request.request_id}\")\n                return cached_data['result']\n            else:\n                del self.result_cache[cache_key]\n        \n        return None\n    \n    def _cache_result(self, request: InferenceRequest, result: InferenceResult):\n        \"\"\"缓存推理结果\"\"\"\n        cache_key = f\"{request.user_id}_{hash(str(request.input_data))}\"\n        self.result_cache[cache_key] = {\n            'timestamp': time.time(),\n            'result': result\n        }\n        \n        # 限制缓存大小\n        if len(self.result_cache) > 1000:\n            # 删除最旧的条目\n            oldest_key = min(self.result_cache.keys(), \n                            key=lambda k: self.result_cache[k]['timestamp'])\n            del self.result_cache[oldest_key]\n    \n    def _create_error_result(self, request: InferenceRequest, \n                           error_message: str, processing_time: float) -> InferenceResult:\n        \"\"\"创建错误结果\"\"\"\n        self.performance_metrics['failed_requests'] += 1\n        \n        return InferenceResult(\n            request_id=request.request_id,\n            user_id=request.user_id,\n            ai_response={'error': error_message, 'prompts': ['系统暂时不可用，请稍后重试']},\n            cognitive_state=np.array([0.5] * 8),\n            inference_metadata={'error': error_message},\n            processing_time=processing_time,\n            success=False,\n            error_message=error_message\n        )\n    \n    def _update_performance_metrics(self, processing_time: float, success: bool):\n        \"\"\"更新性能指标\"\"\"\n        if success:\n            self.performance_metrics['successful_requests'] += 1\n        \n        # 更新平均处理时间\n        total_requests = self.performance_metrics['total_requests']\n        current_avg = self.performance_metrics['avg_processing_time']\n        self.performance_metrics['avg_processing_time'] = (\n            (current_avg * (total_requests - 1) + processing_time) / total_requests\n        )\n    \n    def get_performance_report(self) -> Dict[str, Any]:\n        \"\"\"获取性能报告\"\"\"\n        algo_times = self.performance_metrics['algorithm_times']\n        \n        report = {\n            'requests': {\n                'total': self.performance_metrics['total_requests'],\n                'successful': self.performance_metrics['successful_requests'],\n                'failed': self.performance_metrics['failed_requests'],\n                'success_rate': self.performance_metrics['successful_requests'] / max(1, self.performance_metrics['total_requests'])\n            },\n            'timing': {\n                'avg_processing_time': self.performance_metrics['avg_processing_time'],\n                'algorithm_avg_times': {\n                    name: np.mean(times) if times else 0.0\n                    for name, times in algo_times.items()\n                }\n            },\n            'system': {\n                'active_requests': len(self.active_requests),\n                'cache_size': len(self.result_cache),\n                'queue_size': self.request_queue.qsize()\n            }\n        }\n        \n        return report\n    \n    def get_health_status(self) -> Dict[str, str]:\n        \"\"\"获取系统健康状态\"\"\"\n        try:\n            # 检查各组件状态\n            status = {\n                'coordinator': 'healthy',\n                'difcm_engine': 'healthy' if self.difcm_engine else 'error',\n                'fuzzy_engine': 'healthy' if self.fuzzy_engine else 'error',\n                'graphsage_model': 'healthy' if self.graphsage_model else 'error',\n                'knowledge_graph': 'healthy' if self.knowledge_graph else 'error'\n            }\n            \n            # 检查性能指标\n            if self.performance_metrics['avg_processing_time'] > 10.0:\n                status['performance'] = 'warning'\n            else:\n                status['performance'] = 'good'\n            \n            return status\n            \n        except Exception as e:\n            logging.error(f\"健康检查失败: {e}\")\n            return {'coordinator': 'error', 'error': str(e)}\n    \n    async def shutdown(self):\n        \"\"\"优雅关闭协调器\"\"\"\n        logging.info(\"正在关闭算法协调器...\")\n        \n        # 等待活跃请求完成\n        while self.active_requests:\n            await asyncio.sleep(0.1)\n        \n        # 关闭线程池\n        self.executor.shutdown(wait=True)\n        \n        logging.info(\"算法协调器已关闭\")"