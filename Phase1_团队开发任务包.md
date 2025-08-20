# Phase 1 团队开发任务包 (Week 1-2)

## 项目背景
Phase 1目标：验证DIFCM认知追踪能力，实现基础认知状态识别，准确率>75%，响应时间<500ms

## 团队分工架构

### Team A: 数据与模型团队
**负责人**：算法工程师
**任务**：DIFCM核心算法实现与训练

### Team B: 接口与存储团队  
**负责人**：后端工程师
**任务**：实时数据接口与状态存

### Team C: 评估与验证团队
**负责人**：测试工程师
**任务**：认知状态评估框架与指标验证

### Team D: 前端交互团队
**负责人**：前端工程师
**任务**：用户行为数据采集界面

---

## Team A: 数据与模型团队任务包

### 任务清单
- [ ] 1.1 DIFCM模型核心实现
- [ ] 1.2 认知特征提取算法
- [ ] 1.3 模型训练管道
- [ ] 1.4 模型评估与保存

### 详细规范

#### 1.1 DIFCM模型核心实现
```python
# 文件: models/difcm_core.py
class DIFCModel:
    def __init__(self, n_concepts=8, learning_rate=0.01):
        """
        DIFCM模型初始化
        n_concepts: 认知概念数量 (注意力, 记忆, 理解, 创造, 动机, 情绪, 自信, 疲劳)
        """
        self.n_concepts = n_concepts
        self.concepts = ['attention', 'memory', 'comprehension', 'creativity', 
                        'motivation', 'emotion', 'confidence', 'fatigue']
        self.weight_matrix = np.random.randn(n_concepts, n_concepts) * 0.1
        self.concept_values = np.zeros(n_concepts)
        
    def forward(self, input_features):
        """前向传播计算认知状态"""
        # TODO: 实现DIFCM状态更新方程
        pass
        
    def update_weights(self, target_state, predicted_state):
        """基于预测的权重更新"""
        # TODO: 实现权重自适应调整
        pass
```

#### 1.2 认知特征提取算法
```python
# 文件: features/cognitive_features.py
def extract_behavioral_features(raw_data):
    """
    从用户行为数据提取认知特征
    输入: {
        'keystrokes': [...],      # 击键序列
        'mouse_moves': [...],     # 鼠标轨迹
        'dwell_times': [...],     # 停留时间
        'response_times': [...]   # 响应时间
    }
    输出: 8维认知特征向量
    """
    features = {
        'attention': calculate_attention_score(raw_data),
        'memory': calculate_memory_load(raw_data),
        'comprehension': calculate_comprehension_level(raw_data),
        'creativity': calculate_creativity_index(raw_data),
        'motivation': calculate_motivation_level(raw_data),
        'emotion': detect_emotional_state(raw_data),
        'confidence': estimate_confidence_level(raw_data),
        'fatigue': detect_fatigue_signs(raw_data)
    }
    return features
```

#### 1.3 模型训练管道
```python
# 文件: training/train_difcm.py
class DIFCMTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
    def train_epoch(self, dataloader):
        """单轮训练"""
        for batch in dataloader:
            features, labels = batch
            predictions = self.model(features)
            loss = self.criterion(predictions, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
    def validate(self, dataloader):
        """验证模型性能"""
        # TODO: 计算准确率、召回率等指标
        pass
```

### 交付物要求
1. **代码交付**：完整的DIFCM模型实现
2. **模型文件**：训练好的.pth模型文件
3. **测试报告**：包含准确率、响应时间指标
4. **API文档**：模型调用接口说明

---

## Team B: 接口与存储团队任务包

### 任务清单
- [ ] 2.1 实时数据接收API
- [ ] 2.2 认知状态存储服务
- [ ] 2.3 历史数据查询接口
- [ ] 2.4 性能监控接口

### 详细规范

#### 2.1 实时数据接收API
```python
# 文件: api/data_receiver.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

class UserBehaviorData(BaseModel):
    user_id: str
    session_id: str
    timestamp: float
    keystrokes: List[dict]
    mouse_data: List[dict]
    task_context: dict

app = FastAPI()

@app.post("/api/v1/behavior-data")
async def receive_behavior_data(data: UserBehaviorData):
    """接收用户行为数据"""
    try:
        # 数据验证
        validate_data_format(data)
        
        # 存储到数据库
        data_id = store_raw_data(data)
        
        # 触发模型预测
        cognitive_state = await trigger_prediction(data_id)
        
        return {
            "status": "success",
            "data_id": data_id,
            "cognitive_state": cognitive_state
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

#### 2.2 认知状态存储服务
```python
# 文件: storage/state_storage.py
class CognitiveStateStorage:
    def __init__(self, db_config):
        self.db = self.init_database(db_config)
        
    def store_state(self, user_id, session_id, state_data):
        """存储认知状态"""
        record = {
            'user_id': user_id,
            'session_id': session_id,
            'timestamp': datetime.now(),
            'cognitive_state': state_data['state_vector'],
            'confidence': state_data['confidence'],
            'raw_features': state_data['features']
        }
        
        self.db.cognitive_states.insert_one(record)
        return record['_id']
        
    def get_user_history(self, user_id, limit=100):
        """获取用户历史认知状态"""
        return list(self.db.cognitive_states
                   .find({'user_id': user_id})
                   .sort('timestamp', -1)
                   .limit(limit))
```

#### 2.3 性能监控接口
```python
# 文件: monitoring/performance_tracker.py
class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'prediction_latency': [],
            'accuracy_scores': [],
            'system_load': []
        }
        
    def record_prediction(self, latency, accuracy):
        """记录预测性能"""
        self.metrics['prediction_latency'].append(latency)
        self.metrics['accuracy_scores'].append(accuracy)
        
    def get_performance_report(self):
        """获取性能报告"""
        return {
            'avg_latency': np.mean(self.metrics['prediction_latency']),
            'accuracy': np.mean(self.metrics['accuracy_scores']),
            'p99_latency': np.percentile(self.metrics['prediction_latency'], 99)
        }
```

### 交付物要求
1. **API服务**：完整的RESTful API实现
2. **数据库设计**：MongoDB集合设计文档
3. **监控面板**：实时性能监控界面
4. **部署脚本**：Docker容器化部署

---

## Team C: 评估与验证团队任务包

### 任务清单
- [ ] 3.1 评估数据集构建
- [ ] 3.2 准确率计算框架
- [ ] 3.3 响应时间测试
- [ ] 3.4 A/B测试框架

### 详细规范

#### 3.1 评估数据集构建
```python
# 文件: evaluation/dataset_builder.py
class CognitiveDatasetBuilder:
    def __init__(self):
        self.label_mapping = {
            '高注意力': [0.8, 0.7, 0.9, 0.6, 0.8, 0.7, 0.8, 0.3],
            '低注意力': [0.3, 0.4, 0.2, 0.3, 0.2, 0.5, 0.3, 0.8],
            '高创造力': [0.6, 0.7, 0.8, 0.9, 0.8, 0.7, 0.9, 0.2],
            '低创造力': [0.3, 0.4, 0.3, 0.2, 0.3, 0.4, 0.2, 0.7]
        }
        
    def generate_synthetic_data(self, n_samples=1000):
        """生成评估用合成数据"""
        data = []
        for label, base_state in self.label_mapping.items():
            for _ in range(n_samples // len(self.label_mapping)):
                # 添加噪声
                noisy_state = base_state + np.random.normal(0, 0.1, len(base_state))
                noisy_state = np.clip(noisy_state, 0, 1)
                
                # 生成对应的行为特征
                features = self.generate_behavioral_features(noisy_state)
                
                data.append({
                    'features': features,
                    'label': label,
                    'cognitive_state': noisy_state
                })
        return data
```

#### 3.2 准确率计算框架
```python
# 文件: evaluation/accuracy_calculator.py
class AccuracyCalculator:
    def __init__(self):
        self.thresholds = {
            'attention': 0.5,
            'creativity': 0.5,
            'comprehension': 0.5
        }
        
    def calculate_accuracy(self, predictions, ground_truth):
        """计算认知状态识别准确率"""
        correct = 0
        total = len(predictions)
        
        for pred, truth in zip(predictions, ground_truth):
            # 计算每个维度的准确率
            attention_correct = abs(pred['attention'] - truth['attention']) < 0.1
            creativity_correct = abs(pred['creativity'] - truth['creativity']) < 0.1
            
            if attention_correct and creativity_correct:
                correct += 1
                
        return correct / total
        
    def calculate_f1_score(self, predictions, ground_truth, threshold=0.5):
        """计算F1分数"""
        # TODO: 实现多类别F1计算
        pass
```

#### 3.3 响应时间测试
```python
# 文件: evaluation/latency_tester.py
class LatencyTester:
    def __init__(self, api_endpoint):
        self.endpoint = api_endpoint
        
    async def test_response_time(self, n_requests=1000):
        """测试API响应时间"""
        latencies = []
        
        async with aiohttp.ClientSession() as session:
            for i in range(n_requests):
                start_time = time.time()
                
                async with session.post(self.endpoint, json=self.generate_test_data()) as response:
                    await response.json()
                    
                latency = (time.time() - start_time) * 1000  # ms
                latencies.append(latency)
                
        return {
            'mean_latency': np.mean(latencies),
            'p95_latency': np.percentile(latencies, 95),
            'p99_latency': np.percentile(latencies, 99),
            'max_latency': np.max(latencies)
        }
```

### 交付物要求
1. **测试数据集**：1000+标注样本
2. **评估报告**：准确率、响应时间完整测试
3. **测试脚本**：自动化测试工具
4. **基准指标**：Phase 1验收标准文档

---

## Team D: 前端交互团队任务包

### 任务清单
- [ ] 4.1 用户行为数据采集界面
- [ ] 4.2 实时认知状态可视化
- [ ] 4.3 用户测试环境搭建
- [ ] 4.4 数据导出工具

### 详细规范

#### 4.1 用户行为数据采集界面
```javascript
// 文件: frontend/src/components/BehaviorTracker.jsx
import React, { useEffect, useState } from 'react';
import { BehaviorCollector } from '../utils/behaviorCollector';

const BehaviorTracker = ({ userId, onDataCollected }) => {
    const [collector, setCollector] = useState(null);
    
    useEffect(() => {
        const newCollector = new BehaviorCollector(userId);
        newCollector.start();
        setCollector(newCollector);
        
        return () => newCollector.stop();
    }, [userId]);
    
    const handleSubmitData = async () => {
        const data = collector.getCollectedData();
        await fetch('/api/v1/behavior-data', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        onDataCollected(data);
    };
    
    return (
        <div>
            <h3>认知状态追踪测试</h3>
            <div>用户ID: {userId}</div>
            <button onClick={handleSubmitData}>提交行为数据</button>
        </div>
    );
};

// 文件: frontend/src/utils/behaviorCollector.js
class BehaviorCollector {
    constructor(userId) {
        this.userId = userId;
        this.sessionId = this.generateSessionId();
        this.keystrokes = [];
        this.mouseData = [];
        this.startTime = Date.now();
        
        this.setupEventListeners();
    }
    
    setupEventListeners() {
        // 键盘事件监听
        document.addEventListener('keydown', (e) => {
            this.keystrokes.push({
                key: e.key,
                timestamp: Date.now() - this.startTime,
                type: 'keydown'
            });
        });
        
        // 鼠标事件监听
        document.addEventListener('mousemove', (e) => {
            this.mouseData.push({
                x: e.clientX,
                y: e.clientY,
                timestamp: Date.now() - this.startTime,
                type: 'mousemove'
            });
        });
    }
    
    getCollectedData() {
        return {
            userId: this.userId,
            sessionId: this.sessionId,
            timestamp: Date.now(),
            keystrokes: this.keystrokes,
            mouseData: this.mouseData,
            taskContext: this.getTaskContext()
        };
    }
}
```

#### 4.2 实时认知状态可视化
```javascript
// 文件: frontend/src/components/CognitiveStateDisplay.jsx
const CognitiveStateDisplay = ({ stateData }) => {
    const [state, setState] = useState(null);
    
    useEffect(() => {
        const interval = setInterval(async () => {
            const response = await fetch('/api/v1/latest-state');
            const data = await response.json();
            setState(data);
        }, 1000);
        
        return () => clearInterval(interval);
    }, []);
    
    return (
        <div>
            <h4>当前认知状态</h4>
            <div>注意力: {state?.attention?.toFixed(2)}</div>
            <div>创造力: {state?.creativity?.toFixed(2)}</div>
            <div>理解力: {state?.comprehension?.toFixed(2)}</div>
        </div>
    );
};
```

### 交付物要求
1. **前端界面**：完整的用户测试界面
2. **数据采集**：行为数据收集工具
3. **可视化组件**：实时状态展示
4. **测试报告**：用户交互体验评估

---

## 团队协作接口规范

### API接口定义
```yaml
# 团队间接口契约
paths:
  /api/v1/behavior-data:
    post:
      summary: 接收行为数据
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/UserBehaviorData'
      responses:
        200:
          description: 认知状态预测结果
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/CognitiveState'

  /api/v1/latest-state:
    get:
      summary: 获取最新认知状态
      responses:
        200:
          description: 当前认知状态
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/CognitiveState'
```

### 数据格式标准
```json
{
  "user_id": "string",
  "session_id": "string", 
  "timestamp": "float",
  "cognitive_state": {
    "attention": "float[0-1]",
    "memory": "float[0-1]",
    "comprehension": "float[0-1]",
    "creativity": "float[0-1]",
    "motivation": "float[0-1]",
    "emotion": "float[0-1]",
    "confidence": "float[0-1]",
    "fatigue": "float[0-1]"
  },
  "confidence": "float[0-1]",
  "raw_features": "dict"
}
```

### 每周同步会议
- **时间**：每周五下午2:00
- **时长**：30分钟
- **议程**：
  1. 各团队进度汇报（5分钟/团队）
  2. 接口对接问题讨论（10分钟）
  3. 下周计划协调（5分钟）

---

## 验收标准

### Phase 1 完成标准
1. **技术指标**：
   - 认知状态识别准确率 ≥ 75%
   - 平均响应时间 ≤ 500ms
   - 系统可用性 ≥ 99%

2. **功能指标**：
   - 所有API接口正常工作
   - 数据采集完整性 ≥ 95%
   - 用户面可用性 ≥ 80%

3. **文档指标**：
   - 所有代码有完整注释
   - API文档完整准确
   - 测试报告详细完整

**完成时间：Week 2结束（14天内）**