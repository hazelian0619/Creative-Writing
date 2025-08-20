# 师范生创意写作AI辅助系统

基于"模糊性引导精确度理论"的师范教育创新项目

## 项目概述

本项目专门为师范生创意写作教学设计，通过AI算法实现从模糊创意到精确表达的引导过程。

### 核心理论
**模糊性引导精确度理论 (Ambiguity-Guided Precision Theory)**
- 阶段1：模糊启发阶段 - 开放性创意激发
- 阶段2：概念聚焦阶段 - 逐步明确方向  
- 阶段3：精确实现阶段 - 具体可操作指导

### 技术架构

#### 核心算法创新
1. **DIFCM (Dynamic Interval-valued Fuzzy Cognitive Map)** - 动态区间模糊认知图
2. **T-S模糊推理引擎** - 多维信号融合推理
3. **GraphSAGE知识推理** - 创意写作知识图谱深度联想

#### 团队分工
- **Team A**: 核心算法实现 (DIFCM + T-S模糊推理 + GraphSAGE)
- **Team B**: API接口与数据服务
- **Team C**: 评估系统与性能监控
- **Team D**: 前端交互界面

## 验收标准
- ✅ 认知状态追踪准确率 > 75%
- ✅ 系统响应时间 < 500ms
- ✅ 专门服务师范生创意写作场景

## 快速开始

### 环境要求
- Python 3.8+
- PyTorch 1.12+
- Node.js 16+ (前端)
- PostgreSQL (数据存储)

### 安装依赖
```bash
# Team A - 算法模块
cd team_a
pip install torch transformers torch-geometric datasets

# Team B - API服务
cd ../team_b  
pip install fastapi uvicorn sqlalchemy psycopg2

# Team C - 评估模块
cd ../team_c
pip install scikit-learn matplotlib

# Team D - 前端界面
cd ../team_d
npm install
```

### 运行服务
```bash
# 启动Team A算法推理服务
cd team_a
python -m api.ai_inference_api

# 启动Team B数据服务
cd ../team_b
python -m api.user_management_api

# 启动前端
cd ../team_d
npm start
```

## 项目状态

### Phase 1 - 核心算法实现 ✅
- [x] DIFCM动态认知图模型
- [x] T-S模糊推理引擎  
- [x] GraphSAGE知识推理网络
- [x] 算法协调框架
- [x] 工业级数据集成
- [x] API接口对接

### Phase 2 - 系统集成 (进行中)
- [ ] 完整数据流测试
- [ ] 性能优化
- [ ] 用户体验优化

## 开发指南

### 代码规范
- 遵循PEP 8 Python代码风格
- 使用类型提示
- 完善的docstring文档
- 单元测试覆盖率 > 80%

### 提交规范
```
feat: 新功能
fix: 修复bug  
docs: 文档更新
refactor: 代码重构
test: 测试相关
```

### 分支策略
- `main`: 稳定主分支
- `develop`: 开发分支
- `feature/*`: 功能开发分支
- `hotfix/*`: 紧急修复分支

## 核心特性

### 🧠 智能认知追踪
实时分析师范生的创作认知状态，包含8个核心维度：
- 深刻性思维
- 灵活性思维  
- 批判性思维
- 独创性
- 流畅性
- 动机水平
- 情绪调节
- 认知负载

### 🎯 精准个性化引导
基于认知状态提供三层递进式提示：
1. **联想层** - 开放性创意激发
2. **结构层** - 框架化思维引导
3. **操作层** - 具体技法指导

### 📚 专业知识图谱
整合师范教育和创意写作领域知识：
- ConceptNet概念关系网络
- HuggingFace教育数据集
- 师范生写作特定场景库

## 技术文档

详细技术文档请参考：
- [AI算法设计规范书](./AI算法设计规范书.md)
- [系统架构设计文档](./系统架构与数据流设计文档.md)
- [团队开发任务包](./Phase1_团队开发任务包.md)

## 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 联系我们

项目负责人：团队负责人
邮箱：[联系邮箱]

---

**注意**: 本项目专注于师范教育领域的创新应用，算法设计充分考虑了师范生的认知特点和教学实际需求。