"""
响应时间测试器
测试API响应时间，确保达到Phase1的500ms目标
"""

import time
import asyncio
import aiohttp
import json
import logging
import statistics
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

sys.path.append("/Users/pluviophile/chi2025")
from team_c.config import EvaluationConfig, APITestConfig, TestScenarios

logger = logging.getLogger(__name__)

@dataclass
class LatencyResult:
    """延迟测试结果"""
    endpoint: str
    mean_latency: float
    median_latency: float
    p95_latency: float
    p99_latency: float
    max_latency: float
    min_latency: float
    std_latency: float
    success_rate: float
    total_requests: int
    failed_requests: int
    latencies: List[float]

@dataclass
class LoadTestResult:
    """负载测试结果"""
    concurrent_users: int
    requests_per_user: int
    test_duration: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_latency: float
    throughput: float  # 请求/秒
    error_rate: float
    latency_distribution: Dict[str, float]
    endpoint_results: Dict[str, LatencyResult]

class LatencyTester:
    """
    响应时间测试器
    支持单端点测试和负载测试
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.api_config = APITestConfig()
        
        # 目标响应时间
        self.target_latency = config.latency_target * 1000  # 转换为毫秒
        
        # 测试端点
        self.test_endpoints = {
            'behavior_data': self.api_config.behavior_data_endpoint,
            'latest_state': self.api_config.latest_state_endpoint,
            'health_check': self.api_config.health_check_endpoint
        }
        
        # 基础URL
        self.base_url = f"http://{self.api_config.api_host}:{self.api_config.api_port}"
        
        logger.info(f"响应时间测试器初始化完成，目标延迟: {self.target_latency}ms")
    
    async def test_single_endpoint(self, 
                                 endpoint: str, 
                                 n_requests: int = 100,
                                 method: str = 'GET',
                                 payload: Optional[Dict] = None) -> LatencyResult:
        """
        测试单个端点的响应时间
        
        Args:
            endpoint: 端点路径
            n_requests: 请求次数
            method: HTTP方法
            payload: 请求载荷
            
        Returns:
            延迟测试结果
        """
        logger.info(f"开始测试端点 {endpoint}，请求次数: {n_requests}")
        
        url = f"{self.base_url}{endpoint}"
        latencies = []
        failed_requests = 0
        
        # 创建会话
        timeout = aiohttp.ClientTimeout(total=self.api_config.request_timeout)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = []
            
            for i in range(n_requests):
                task = self._single_request(session, url, method, payload, i)
                tasks.append(task)
            
            # 执行所有请求
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            for result in results:
                if isinstance(result, Exception):
                    failed_requests += 1
                    logger.debug(f"请求失败: {result}")
                elif isinstance(result, dict) and 'latency' in result:
                    latencies.append(result['latency'])
                else:
                    failed_requests += 1
        
        # 计算统计信息
        if latencies:
            mean_latency = statistics.mean(latencies)
            median_latency = statistics.median(latencies)
            std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
            min_latency = min(latencies)
            max_latency = max(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
        else:
            mean_latency = median_latency = std_latency = 0.0
            min_latency = max_latency = p95_latency = p99_latency = 0.0
        
        success_rate = (n_requests - failed_requests) / n_requests if n_requests > 0 else 0.0
        
        result = LatencyResult(
            endpoint=endpoint,
            mean_latency=mean_latency,
            median_latency=median_latency,
            p95_latency=p95_latency,
            p99_latency=p99_latency,
            max_latency=max_latency,
            min_latency=min_latency,
            std_latency=std_latency,
            success_rate=success_rate,
            total_requests=n_requests,
            failed_requests=failed_requests,
            latencies=latencies
        )
        
        self._log_latency_result(result)
        return result
    
    async def _single_request(self, 
                            session: aiohttp.ClientSession,
                            url: str,
                            method: str,
                            payload: Optional[Dict],
                            request_id: int) -> Dict:
        """执行单个请求并测量延迟"""
        try:
            start_time = time.time()
            
            if method.upper() == 'GET':
                async with session.get(url) as response:
                    await response.text()  # 确保完全接收响应
                    status = response.status
            elif method.upper() == 'POST':
                async with session.post(url, json=payload) as response:
                    await response.text()
                    status = response.status
            else:
                raise ValueError(f"不支持的HTTP方法: {method}")
            
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # 转换为毫秒
            
            return {
                'latency': latency,
                'status': status,
                'request_id': request_id,
                'success': 200 <= status < 300
            }
            
        except Exception as e:
            logger.debug(f"请求 {request_id} 失败: {e}")
            return {'error': str(e), 'request_id': request_id}
    
    async def test_all_endpoints(self, n_requests: int = 100) -> Dict[str, LatencyResult]:
        """测试所有端点"""
        logger.info(f"开始测试所有端点，每个端点 {n_requests} 次请求")
        
        results = {}
        
        for endpoint_name, endpoint_path in self.test_endpoints.items():
            try:
                # 为不同端点准备不同的测试参数
                if endpoint_name == 'behavior_data':
                    # POST请求，需要载荷
                    payload = self._generate_test_payload()
                    result = await self.test_single_endpoint(
                        endpoint_path, n_requests, 'POST', payload
                    )
                else:
                    # GET请求
                    result = await self.test_single_endpoint(
                        endpoint_path, n_requests, 'GET'
                    )
                
                results[endpoint_name] = result
                
            except Exception as e:
                logger.error(f"测试端点 {endpoint_name} 失败: {e}")
                # 创建失败结果
                results[endpoint_name] = LatencyResult(
                    endpoint=endpoint_path,
                    mean_latency=0.0,
                    median_latency=0.0,
                    p95_latency=0.0,
                    p99_latency=0.0,
                    max_latency=0.0,
                    min_latency=0.0,
                    std_latency=0.0,
                    success_rate=0.0,
                    total_requests=n_requests,
                    failed_requests=n_requests,
                    latencies=[]
                )
        
        return results
    
    def run_load_test(self, scenario: str = 'normal_load') -> LoadTestResult:
        """
        运行负载测试
        
        Args:
            scenario: 测试场景名称
            
        Returns:
            负载测试结果
        """
        if scenario not in TestScenarios.PERFORMANCE_SCENARIOS:
            raise ValueError(f"未知测试场景: {scenario}")
        
        scenario_config = TestScenarios.PERFORMANCE_SCENARIOS[scenario]
        concurrent_users = scenario_config['concurrent_users']
        requests_per_user = scenario_config['requests_per_user']
        duration = scenario_config['duration']
        
        logger.info(f"开始负载测试场景: {scenario}")
        logger.info(f"并发用户: {concurrent_users}, 每用户请求: {requests_per_user}, 持续时间: {duration}s")
        
        start_time = time.time()
        
        # 使用线程池执行负载测试
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = []
            
            for user_id in range(concurrent_users):
                future = executor.submit(
                    self._user_load_test,
                    user_id,
                    requests_per_user,
                    duration
                )
                futures.append(future)
            
            # 收集结果
            all_results = []
            for future in as_completed(futures):
                try:
                    user_results = future.result()
                    all_results.extend(user_results)
                except Exception as e:
                    logger.error(f"用户负载测试失败: {e}")
        
        end_time = time.time()
        actual_duration = end_time - start_time
        
        # 分析结果
        load_result = self._analyze_load_test_results(
            all_results, concurrent_users, requests_per_user, actual_duration
        )
        
        self._log_load_test_result(load_result)
        return load_result
    
    def _user_load_test(self, 
                       user_id: int, 
                       requests_per_user: int,
                       duration: int) -> List[Dict]:
        """单用户负载测试"""
        results = []
        start_time = time.time()
        request_count = 0
        
        while (time.time() - start_time) < duration and request_count < requests_per_user:
            # 随机选择端点
            endpoint_name = np.random.choice(list(self.test_endpoints.keys()))
            endpoint_path = self.test_endpoints[endpoint_name]
            
            try:
                # 执行请求
                result = self._sync_request(endpoint_name, endpoint_path)
                result['user_id'] = user_id
                result['request_count'] = request_count
                result['timestamp'] = time.time()
                results.append(result)
                
            except Exception as e:
                results.append({
                    'user_id': user_id,
                    'request_count': request_count,
                    'endpoint': endpoint_name,
                    'error': str(e),
                    'success': False,
                    'timestamp': time.time()
                })
            
            request_count += 1
            
            # 短暂休息以模拟真实用户行为
            time.sleep(np.random.exponential(0.1))
        
        return results
    
    def _sync_request(self, endpoint_name: str, endpoint_path: str) -> Dict:
        """同步请求（用于线程池）"""
        import requests
        
        url = f"{self.base_url}{endpoint_path}"
        start_time = time.time()
        
        try:
            if endpoint_name == 'behavior_data':
                payload = self._generate_test_payload()
                response = requests.post(
                    url, 
                    json=payload, 
                    timeout=self.api_config.request_timeout
                )
            else:
                response = requests.get(
                    url, 
                    timeout=self.api_config.request_timeout
                )
            
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # ms
            
            return {
                'endpoint': endpoint_name,
                'latency': latency,
                'status': response.status_code,
                'success': 200 <= response.status_code < 300
            }
            
        except requests.exceptions.Timeout:
            return {
                'endpoint': endpoint_name,
                'error': 'timeout',
                'success': False,
                'latency': self.api_config.request_timeout * 1000
            }
        except Exception as e:
            return {
                'endpoint': endpoint_name,
                'error': str(e),
                'success': False,
                'latency': 0.0
            }
    
    def _analyze_load_test_results(self, 
                                 results: List[Dict],
                                 concurrent_users: int,
                                 requests_per_user: int,
                                 duration: float) -> LoadTestResult:
        """分析负载测试结果"""
        if not results:
            return LoadTestResult(
                concurrent_users=concurrent_users,
                requests_per_user=requests_per_user,
                test_duration=duration,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                average_latency=0.0,
                throughput=0.0,
                error_rate=1.0,
                latency_distribution={},
                endpoint_results={}
            )
        
        # 总体统计
        total_requests = len(results)
        successful_results = [r for r in results if r.get('success', False)]
        failed_requests = total_requests - len(successful_results)
        
        # 延迟统计
        successful_latencies = [r['latency'] for r in successful_results if 'latency' in r]
        
        if successful_latencies:
            average_latency = statistics.mean(successful_latencies)
            latency_distribution = {
                'mean': average_latency,
                'median': statistics.median(successful_latencies),
                'p95': np.percentile(successful_latencies, 95),
                'p99': np.percentile(successful_latencies, 99),
                'max': max(successful_latencies),
                'min': min(successful_latencies),
                'std': statistics.stdev(successful_latencies) if len(successful_latencies) > 1 else 0.0
            }
        else:
            average_latency = 0.0
            latency_distribution = {}
        
        # 吞吐量
        throughput = total_requests / duration if duration > 0 else 0.0
        
        # 错误率
        error_rate = failed_requests / total_requests if total_requests > 0 else 0.0
        
        # 按端点分析
        endpoint_results = {}
        for endpoint_name in self.test_endpoints.keys():
            endpoint_data = [r for r in results if r.get('endpoint') == endpoint_name]
            
            if endpoint_data:
                endpoint_latencies = [r['latency'] for r in endpoint_data if r.get('success', False) and 'latency' in r]
                endpoint_success = len([r for r in endpoint_data if r.get('success', False)])
                endpoint_total = len(endpoint_data)
                
                if endpoint_latencies:
                    endpoint_results[endpoint_name] = LatencyResult(
                        endpoint=self.test_endpoints[endpoint_name],
                        mean_latency=statistics.mean(endpoint_latencies),
                        median_latency=statistics.median(endpoint_latencies),
                        p95_latency=np.percentile(endpoint_latencies, 95),
                        p99_latency=np.percentile(endpoint_latencies, 99),
                        max_latency=max(endpoint_latencies),
                        min_latency=min(endpoint_latencies),
                        std_latency=statistics.stdev(endpoint_latencies) if len(endpoint_latencies) > 1 else 0.0,
                        success_rate=endpoint_success / endpoint_total,
                        total_requests=endpoint_total,
                        failed_requests=endpoint_total - endpoint_success,
                        latencies=endpoint_latencies
                    )
        
        return LoadTestResult(
            concurrent_users=concurrent_users,
            requests_per_user=requests_per_user,
            test_duration=duration,
            total_requests=total_requests,
            successful_requests=len(successful_results),
            failed_requests=failed_requests,
            average_latency=average_latency,
            throughput=throughput,
            error_rate=error_rate,
            latency_distribution=latency_distribution,
            endpoint_results=endpoint_results
        )
    
    def _generate_test_payload(self) -> Dict:
        """生成测试载荷"""
        return {
            "user_id": f"test_user_{np.random.randint(1000, 9999)}",
            "session_id": f"test_session_{np.random.randint(1000, 9999)}",
            "timestamp": time.time(),
            "keystrokes": [
                {
                    "key": "a",
                    "timestamp": time.time(),
                    "type": "keydown"
                }
            ],
            "mouse_data": [
                {
                    "x": 100,
                    "y": 200,
                    "timestamp": time.time(),
                    "type": "mousemove"
                }
            ],
            "task_context": {
                "task_type": "latency_test"
            }
        }
    
    def _log_latency_result(self, result: LatencyResult):
        """记录延迟测试结果"""
        logger.info(f"=== 端点 {result.endpoint} 延迟测试结果 ===")
        logger.info(f"平均延迟: {result.mean_latency:.2f}ms")
        logger.info(f"中位延迟: {result.median_latency:.2f}ms")
        logger.info(f"P95延迟: {result.p95_latency:.2f}ms")
        logger.info(f"P99延迟: {result.p99_latency:.2f}ms")
        logger.info(f"最大延迟: {result.max_latency:.2f}ms")
        logger.info(f"成功率: {result.success_rate*100:.1f}%")
        logger.info(f"目标达成: {'是' if result.mean_latency <= self.target_latency else '否'}")
    
    def _log_load_test_result(self, result: LoadTestResult):
        """记录负载测试结果"""
        logger.info(f"=== 负载测试结果 ===")
        logger.info(f"并发用户: {result.concurrent_users}")
        logger.info(f"总请求数: {result.total_requests}")
        logger.info(f"成功请求: {result.successful_requests}")
        logger.info(f"失败请求: {result.failed_requests}")
        logger.info(f"平均延迟: {result.average_latency:.2f}ms")
        logger.info(f"吞吐量: {result.throughput:.2f} 请求/秒")
        logger.info(f"错误率: {result.error_rate*100:.2f}%")
        logger.info(f"目标达成: {'是' if result.average_latency <= self.target_latency else '否'}")
    
    def plot_latency_results(self, 
                           results: Dict[str, LatencyResult],
                           save_path: Path = None) -> Path:
        """绘制延迟测试结果"""
        if save_path is None:
            save_path = self.config.results_dir / "latency_results.png"
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 端点延迟对比
        endpoints = list(results.keys())
        mean_latencies = [results[ep].mean_latency for ep in endpoints]
        p95_latencies = [results[ep].p95_latency for ep in endpoints]
        
        ax1 = axes[0, 0]
        x_pos = np.arange(len(endpoints))
        ax1.bar(x_pos - 0.2, mean_latencies, 0.4, label='平均延迟', alpha=0.7)
        ax1.bar(x_pos + 0.2, p95_latencies, 0.4, label='P95延迟', alpha=0.7)
        ax1.axhline(y=self.target_latency, color='r', linestyle='--', label=f'目标 ({self.target_latency}ms)')
        ax1.set_title('端点延迟对比')
        ax1.set_ylabel('延迟 (ms)')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(endpoints, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 成功率
        success_rates = [results[ep].success_rate * 100 for ep in endpoints]
        
        ax2 = axes[0, 1]
        bars = ax2.bar(endpoints, success_rates, alpha=0.7)
        ax2.axhline(y=95, color='r', linestyle='--', label='目标 (95%)')
        ax2.set_title('端点成功率')
        ax2.set_ylabel('成功率 (%)')
        ax2.set_ylim(0, 105)
        ax2.legend()
        
        # 添加数值标签
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rate:.1f}%', ha='center', va='bottom')
        
        # 3. 延迟分布（选择第一个端点）
        if endpoints and results[endpoints[0]].latencies:
            ax3 = axes[1, 0]
            latencies = results[endpoints[0]].latencies
            ax3.hist(latencies, bins=30, alpha=0.7, edgecolor='black')
            ax3.axvline(x=np.mean(latencies), color='r', linestyle='-', label=f'平均 ({np.mean(latencies):.1f}ms)')
            ax3.axvline(x=self.target_latency, color='g', linestyle='--', label=f'目标 ({self.target_latency}ms)')
            ax3.set_title(f'{endpoints[0]} 延迟分布')
            ax3.set_xlabel('延迟 (ms)')
            ax3.set_ylabel('频次')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. 延迟统计汇总
        ax4 = axes[1, 1]
        metrics = ['平均', '中位', 'P95', 'P99', '最大']
        
        if endpoints:
            first_endpoint = endpoints[0]
            result = results[first_endpoint]
            values = [
                result.mean_latency,
                result.median_latency,
                result.p95_latency,
                result.p99_latency,
                result.max_latency
            ]
            
            bars = ax4.barh(metrics, values, alpha=0.7)
            ax4.axvline(x=self.target_latency, color='r', linestyle='--', label=f'目标 ({self.target_latency}ms)')
            ax4.set_title(f'{first_endpoint} 延迟统计')
            ax4.set_xlabel('延迟 (ms)')
            ax4.legend()
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                width = bar.get_width()
                ax4.text(width + max(values) * 0.01, bar.get_y() + bar.get_height()/2,
                        f'{value:.1f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"延迟测试图表已保存到: {save_path}")
        return save_path
    
    def save_latency_report(self, 
                          results: Dict[str, LatencyResult],
                          load_results: Optional[LoadTestResult] = None,
                          filepath: Path = None) -> Path:
        """保存延迟测试报告"""
        if filepath is None:
            filepath = self.config.results_dir / "latency_report.json"
        
        report = {
            'summary': {
                'target_latency_ms': self.target_latency,
                'endpoints_tested': len(results),
                'overall_success': all(r.success_rate >= 0.95 for r in results.values()),
                'target_met': all(r.mean_latency <= self.target_latency for r in results.values())
            },
            'endpoint_results': {},
            'configuration': {
                'base_url': self.base_url,
                'request_timeout': self.api_config.request_timeout,
                'target_latency': self.target_latency
            },
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # 端点结果
        for endpoint_name, result in results.items():
            report['endpoint_results'][endpoint_name] = {
                'endpoint': result.endpoint,
                'mean_latency': result.mean_latency,
                'median_latency': result.median_latency,
                'p95_latency': result.p95_latency,
                'p99_latency': result.p99_latency,
                'max_latency': result.max_latency,
                'min_latency': result.min_latency,
                'std_latency': result.std_latency,
                'success_rate': result.success_rate,
                'total_requests': result.total_requests,
                'failed_requests': result.failed_requests,
                'target_met': result.mean_latency <= self.target_latency
            }
        
        # 负载测试结果
        if load_results:
            report['load_test'] = {
                'concurrent_users': load_results.concurrent_users,
                'total_requests': load_results.total_requests,
                'successful_requests': load_results.successful_requests,
                'failed_requests': load_results.failed_requests,
                'average_latency': load_results.average_latency,
                'throughput': load_results.throughput,
                'error_rate': load_results.error_rate,
                'latency_distribution': load_results.latency_distribution,
                'target_met': load_results.average_latency <= self.target_latency
            }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"延迟测试报告已保存到: {filepath}")
        return filepath

async def main():
    """测试延迟测试器"""
    from team_c.config import get_evaluation_config
    
    config = get_evaluation_config()
    tester = LatencyTester(config)
    
    print("注意：此测试需要API服务运行在配置的地址上")
    print(f"测试地址: {tester.base_url}")
    
    try:
        # 测试所有端点
        results = await tester.test_all_endpoints(n_requests=20)
        
        # 生成图表和报告
        plot_path = tester.plot_latency_results(results)
        report_path = tester.save_latency_report(results)
        
        print(f"测试完成:")
        for endpoint, result in results.items():
            print(f"  {endpoint}: {result.mean_latency:.1f}ms (成功率: {result.success_rate*100:.1f}%)")
        print(f"图表: {plot_path}")
        print(f"报告: {report_path}")
        
    except Exception as e:
        print(f"测试失败: {e}")
        print("请确保API服务正在运行")

if __name__ == "__main__":
    asyncio.run(main())