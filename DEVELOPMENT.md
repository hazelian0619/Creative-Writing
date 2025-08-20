# 开发协作指南

## 分支策略

### 主要分支
- **`main`** - 稳定发布版本，只接受来自develop的合并
- **`develop`** - 开发集成分支，各团队功能开发的汇总

### 团队开发分支
各团队基于develop创建自己的开发分支：

```bash
# Team A - 算法开发分支
git checkout develop
git checkout -b team-a/algorithm-enhancement

# Team B - API服务分支  
git checkout develop
git checkout -b team-b/api-services

# Team C - 评估系统分支
git checkout develop  
git checkout -b team-c/evaluation-system

# Team D - 前端开发分支
git checkout develop
git checkout -b team-d/frontend-ui
```

### 功能开发分支
针对具体功能创建feature分支：

```bash
# 示例：优化DIFCM算法
git checkout team-a/algorithm-enhancement
git checkout -b feature/difcm-optimization

# 开发完成后合并到团队分支
git checkout team-a/algorithm-enhancement
git merge feature/difcm-optimization

# 团队分支定期合并到develop
git checkout develop  
git merge team-a/algorithm-enhancement
```

## 当前代码状态

### ✅ 已完成模块（主分支稳定版本）

#### Team A - 核心算法
- `team_a/models/difcm_enhanced.py` - **主线算法**，DIFCM模型核心实现
- `team_a/models/ts_fuzzy_engine.py` - **主线算法**，T-S模糊推理引擎  
- `team_a/models/graphsage_reasoning.py` - **主线算法**，GraphSAGE知识推理
- `team_a/models/algorithm_coordinator.py` - **主线算法**，三算法协调机制
- `team_a/api/ai_inference_api.py` - 算法推理API接口

#### Team B - API服务
- `team_b/api/cognitive_service.py` - **主线服务**，认知状态处理核心
- `team_b/api/industrial_data.py` - **主线服务**，工业数据集成
- `team_b/api/main.py` - API服务入口
- `team_b/storage/mongodb_service.py` - 数据存储服务

#### Team C - 评估系统  
- `team_c/evaluation/metrics_collector.py` - **主线评估**，指标收集核心
- `team_c/evaluation/accuracy_calculator.py` - 准确率计算
- `team_c/evaluation/latency_tester.py` - 延迟测试

#### Team D - 前端界面
- `team_d/frontend/src/` - React前端框架
- `team_d/cognitive_writing_assistant_v2.html` - **主线界面**，完整版交互界面

### 🔄 可更新开发版本（develop分支）

以下文件可以在develop分支上继续迭代：

#### 算法优化
- `team_a/models/difcm_enhanced.py` - 可优化模型性能、增加新特性
- `team_a/features/writing_cognitive_features.py` - 可增强特征提取
- `team_a/training/train_difcm.py` - 可优化训练流程

#### 服务增强  
- `team_b/api/cognitive_service.py` - 可增加新的API端点
- `team_b/monitoring/performance_monitor.py` - 可增强监控功能
- `team_b/docker/` - 可优化部署配置

#### 评估扩展
- `team_c/evaluation/` - 可增加新的评估指标
- `team_c/tests/integration_tests.py` - 可扩展集成测试

#### 前端改进
- `team_d/frontend/src/` - 可优化用户体验
- `team_d/frontend/src/types/index.ts` - 可扩展类型定义

## 协作工作流

### 1. 日常开发
```bash
# 1. 从develop拉取最新代码
git checkout develop
git pull origin develop

# 2. 创建功能分支
git checkout -b feature/your-feature

# 3. 开发并提交
git add .
git commit -m "feat: 描述你的功能"

# 4. 推送并创建PR
git push origin feature/your-feature
```

### 2. 代码审查
- 所有功能分支必须通过PR合并到团队分支
- 团队分支合并到develop需要至少2人审查
- develop合并到main需要项目负责人审批

### 3. 持续集成
```bash
# 运行测试
python -m pytest team_c/tests/

# 检查代码风格
flake8 team_a/ team_b/ team_c/

# 运行完整系统测试
./run_integration_tests.sh
```

## 提交规范

### 提交消息格式
```
type(scope): 简短描述

详细描述（可选）

Footer（可选，如关闭的issue）
```

### 类型说明
- `feat`: 新功能
- `fix`: Bug修复
- `docs`: 文档更新
- `style`: 格式调整
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建或工具相关

### 示例
```bash
git commit -m "feat(team-a): 优化DIFCM算法性能

- 增加批处理支持
- 优化内存使用
- 提升推理速度30%

Closes #123"
```

## 发布流程

### Phase 2 发布准备
1. 功能开发在各团队分支完成
2. 合并到develop分支集成测试
3. 通过所有测试后合并到main
4. 创建release tag
5. 部署到生产环境

### 版本号规范
- v1.0.0 - Phase 1 核心算法版本（当前main分支）
- v1.1.0 - Phase 2 系统集成版本（计划中）
- v1.x.y - 后续迭代版本

## 注意事项

### 🚨 主分支保护
- `main`分支受保护，直接push被禁止
- 必须通过PR方式合并
- 需要通过所有CI检查

### 🎯 开发重点
- Team A：专注算法优化，不要偏离核心职责
- Team B：API稳定性和性能优化
- Team C：评估系统完善和指标扩展
- Team D：用户体验和界面优化

### 📝 文档同步
- 代码变更必须同步更新相关文档
- API变更必须更新接口文档
- 新功能需要添加使用示例