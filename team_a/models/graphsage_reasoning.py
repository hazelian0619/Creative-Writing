"""
GraphSAGE知识图谱推理模块
创意写作知识推理与分层提示生成
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import Dict, List, Tuple, Optional, Set
import logging
from dataclasses import dataclass
from collections import defaultdict, deque
import random
import json
import time

@dataclass
class NodeType:
    """节点类型枚举"""
    CONCEPT = 0
    TECHNIQUE = 1
    ELEMENT = 2
    EXAMPLE = 3
    PROMPT = 4

@dataclass
class EdgeType:
    """边类型枚举"""
    CONTAINS = 0
    APPLIES = 1
    INSPIRES = 2
    CONSTRAINS = 3
    EXEMPLIFIES = 4

@dataclass
class KnowledgeNode:
    """知识图谱节点"""
    node_id: int
    node_type: int
    name: str
    description: str
    features: np.ndarray
    attributes: Dict[str, float]  # 难度级别、适用阶段、认知负载、创新程度

class HierarchicalSampler:
    """分层邻居采样器"""
    
    def __init__(self, sample_sizes: List[int] = [15, 8], node_types: List[int] = None):
        self.sample_sizes = sample_sizes
        self.node_types = node_types or list(range(5))
        self.sampling_history = deque(maxlen=1000)
        
    def sample_neighbors(self, 
                        center_nodes: torch.Tensor,
                        edge_index: torch.Tensor,
                        node_types: torch.Tensor,
                        k_hop: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        分层邻居采样
        
        Args:
            center_nodes: 中心节点集合
            edge_index: 图的边索引 [2, num_edges]
            node_types: 节点类型 [num_nodes]
            k_hop: 采样跳数
            
        Returns:
            sampled_nodes: 采样节点集合
            sampled_edges: 采样边集合
        """
        sampled_nodes = set(center_nodes.tolist())
        sampled_edges = []
        current_frontier = center_nodes
        
        for hop in range(k_hop):
            next_frontier = []
            
            # 按节点类型分层采样
            for node_type in self.node_types:
                type_neighbors = self._get_typed_neighbors(
                    current_frontier, edge_index, node_types, node_type
                )
                
                # 采样指定数量的邻居
                sample_size = self.sample_sizes[min(hop, len(self.sample_sizes)-1)]
                if len(type_neighbors) > sample_size:
                    # 基于节点重要性的采样
                    sampled_neighbors = self._importance_sampling(
                        type_neighbors, node_types, sample_size
                    )
                else:
                    sampled_neighbors = type_neighbors
                
                next_frontier.extend(sampled_neighbors.tolist())
                sampled_nodes.update(sampled_neighbors.tolist())
                
                # 收集相关边
                type_edges = self._get_edges_for_nodes(
                    current_frontier, sampled_neighbors, edge_index
                )
                sampled_edges.extend(type_edges)
            
            current_frontier = torch.tensor(list(set(next_frontier)))
        
        # 记录采样历史
        self.sampling_history.append({
            'timestamp': time.time(),
            'center_nodes': center_nodes.tolist(),
            'sampled_count': len(sampled_nodes),
            'hops': k_hop
        })
        
        return torch.tensor(list(sampled_nodes)), torch.tensor(sampled_edges).t()
    
    def _get_typed_neighbors(self, nodes: torch.Tensor, edge_index: torch.Tensor,
                           node_types: torch.Tensor, target_type: int) -> torch.Tensor:
        """获取特定类型的邻居节点"""
        neighbors = []
        
        for node in nodes:
            # 找到该节点的所有邻居
            node_neighbors = edge_index[1][edge_index[0] == node]
            # 筛选特定类型的邻居
            typed_neighbors = node_neighbors[node_types[node_neighbors] == target_type]
            neighbors.extend(typed_neighbors.tolist())
        
        return torch.tensor(list(set(neighbors)))
    
    def _importance_sampling(self, neighbors: torch.Tensor, 
                           node_types: torch.Tensor, sample_size: int) -> torch.Tensor:
        """基于重要性的采样"""
        if len(neighbors) <= sample_size:
            return neighbors
        
        # 简单的重要性评分（可以基于度、PageRank等）
        importance_scores = torch.rand(len(neighbors))  # 简化版本
        
        # 按重要性排序并采样
        _, indices = torch.topk(importance_scores, sample_size)
        return neighbors[indices]
    
    def _get_edges_for_nodes(self, source_nodes: torch.Tensor,
                           target_nodes: torch.Tensor, 
                           edge_index: torch.Tensor) -> List[Tuple[int, int]]:
        """获取节点间的边"""
        edges = []
        source_set = set(source_nodes.tolist())
        target_set = set(target_nodes.tolist())
        
        for i in range(edge_index.size(1)):
            src, tgt = edge_index[0, i].item(), edge_index[1, i].item()
            if src in source_set and tgt in target_set:
                edges.append((src, tgt))
        
        return edges

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.output_linear = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, embed_dim] 或 [num_nodes, num_layers, embed_dim]
        """
        batch_size, seq_len, embed_dim = x.size()
        
        # 生成Q, K, V
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)
        
        # 重塑为多头格式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力
        attention = self._scaled_dot_product_attention(Q, K, V, mask)
        
        # 合并多头
        attention = attention.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        
        # 输出变换
        output = self.output_linear(attention)
        
        # 残差连接和层归一化
        return self.layer_norm(x + self.dropout(output))
    
    def _scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, 
                                    V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """缩放点积注意力"""
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attention_weights, V)

class CreativeWritingGraphSAGE(nn.Module):
    """创意写作GraphSAGE模型"""
    
    def __init__(self, 
                 input_dim: int = 256,
                 hidden_dim: int = 128,
                 output_dim: int = 64,
                 num_layers: int = 3,
                 num_heads: int = 4,
                 dropout: float = 0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # 节点类型嵌入
        self.node_type_embedding = nn.Embedding(5, 32)
        
        # GraphSAGE层
        self.sage_layers = nn.ModuleList()
        self.sage_layers.append(SAGEConv(input_dim + 32, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.sage_layers.append(SAGEConv(hidden_dim, hidden_dim))
        
        self.sage_layers.append(SAGEConv(hidden_dim, output_dim))
        
        # 多头注意力
        self.attention = MultiHeadAttention(output_dim, num_heads, dropout)
        
        # 分层采样器
        self.hierarchical_sampler = HierarchicalSampler()
        
        # 投影层
        self.projection = nn.Linear(output_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        logging.info(f\"GraphSAGE模型初始化完成: {num_layers}层, 隐藏维度{hidden_dim}\")\n    \n    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:\n        \"\"\"\n        前向传播\n        \n        Args:\n            data: PyTorch Geometric Data对象\n            \n        Returns:\n            node_embeddings: 节点嵌入 [num_nodes, output_dim]\n            graph_embedding: 图级别嵌入 [1, output_dim]\n        \"\"\"\n        x, edge_index = data.x, data.edge_index\n        node_types = data.node_type if hasattr(data, 'node_type') else torch.zeros(x.size(0), dtype=torch.long)\n        \n        # 1. 节点类型嵌入融合\n        type_emb = self.node_type_embedding(node_types)\n        x = torch.cat([x, type_emb], dim=1)\n        \n        # 2. 多层GraphSAGE传播\n        layer_outputs = []\n        for i, sage_layer in enumerate(self.sage_layers):\n            x = sage_layer(x, edge_index)\n            if i < len(self.sage_layers) - 1:\n                x = F.relu(x)\n                x = self.dropout(x)\n            layer_outputs.append(x)\n        \n        # 3. 注意力聚合多层信息\n        if len(layer_outputs) > 1:\n            # 将多层输出堆叠为 [num_nodes, num_layers, output_dim]\n            stacked_outputs = torch.stack(layer_outputs, dim=1)\n            x = self.attention(stacked_outputs).mean(dim=1)  # 平均池化\n        else:\n            x = layer_outputs[0]\n        \n        # 4. 最终投影\n        x = self.projection(x)\n        \n        # 5. 图级别嵌入\n        batch = getattr(data, 'batch', None)\n        if batch is not None:\n            graph_emb = global_mean_pool(x, batch)\n        else:\n            graph_emb = torch.mean(x, dim=0, keepdim=True)\n        \n        return x, graph_emb\n    \n    def extract_subgraph(self, center_nodes: List[int], \n                        full_data: Data, radius: int = 2) -> Data:\n        \"\"\"\n        提取以中心节点为核心的子图\n        \n        Args:\n            center_nodes: 中心节点列表\n            full_data: 完整图数据\n            radius: 子图半径\n            \n        Returns:\n            子图数据\n        \"\"\"\n        center_tensor = torch.tensor(center_nodes)\n        \n        # 分层采样\n        sampled_nodes, sampled_edges = self.hierarchical_sampler.sample_neighbors(\n            center_tensor, full_data.edge_index, \n            getattr(full_data, 'node_type', torch.zeros(full_data.x.size(0))),\n            k_hop=radius\n        )\n        \n        # 重新映射节点索引\n        node_mapping = {old_id.item(): new_id for new_id, old_id in enumerate(sampled_nodes)}\n        \n        # 构建子图\n        subgraph_x = full_data.x[sampled_nodes]\n        subgraph_edge_index = []\n        \n        for edge in sampled_edges.t():\n            src, tgt = edge[0].item(), edge[1].item()\n            if src in node_mapping and tgt in node_mapping:\n                subgraph_edge_index.append([node_mapping[src], node_mapping[tgt]])\n        \n        if subgraph_edge_index:\n            subgraph_edge_index = torch.tensor(subgraph_edge_index).t()\n        else:\n            subgraph_edge_index = torch.empty((2, 0), dtype=torch.long)\n        \n        # 构建子图数据对象\n        subgraph_data = Data(\n            x=subgraph_x,\n            edge_index=subgraph_edge_index,\n            node_type=getattr(full_data, 'node_type', torch.zeros(len(sampled_nodes)))[sampled_nodes]\n        )\n        \n        return subgraph_data\n\nclass ProgressivePromptGenerator:\n    \"\"\"三层渐进式提示生成器\"\"\"\n    \n    def __init__(self, graphsage_model: CreativeWritingGraphSAGE, \n                 knowledge_graph: 'KnowledgeGraph'):\n        self.model = graphsage_model\n        self.kg = knowledge_graph\n        self.prompt_templates = self._load_prompt_templates()\n        self.generation_history = deque(maxlen=100)\n        \n    def _load_prompt_templates(self) -> Dict[str, Dict[str, str]]:\n        \"\"\"加载提示模板\"\"\"\n        return {\n            'associative': {\n                'creative': \"🎨 让我们从'{concept}'开始联想：{description}。这个概念如何激发你的创意思考？\",\n                'structured': \"基于'{concept}'概念：{description}，请考虑以下关联方向...\"\n            },\n            'structured': {\n                'detailed': \"📖 写作框架：{framework}\\n具体步骤：{steps}\\n结合你的{context}，如何应用这个框架？\",\n                'brief': \"写作结构建议：{framework} - {steps}\"\n            },\n            'actionable': {\n                'high': \"✍️ 具体技法：{technique}\\n操作步骤：{steps}\\n参考示例：{examples}\",\n                'medium': \"建议使用{technique}技法，关键步骤：{steps}\"\n            }\n        }\n    \n    def generate_layered_prompts(self, \n                               topic_concepts: List[str],\n                               user_context: Dict[str, any],\n                               cognitive_state: np.ndarray) -> Dict[str, List[str]]:\n        \"\"\"\n        三层渐进式提示生成\n        \n        Args:\n            topic_concepts: 主题概念列表\n            user_context: 用户上下文\n            cognitive_state: 认知状态\n            \n        Returns:\n            分层提示字典\n        \"\"\"\n        start_time = time.time()\n        \n        # 1. 获取主题相关子图\n        topic_nodes = self.kg.find_concept_nodes(topic_concepts)\n        if not topic_nodes:\n            return self._generate_fallback_prompts()\n        \n        subgraph = self.model.extract_subgraph(topic_nodes, self.kg.get_graph_data())\n        \n        # 2. GraphSAGE推理生成节点嵌入\n        with torch.no_grad():\n            node_embeddings, graph_embedding = self.model(subgraph)\n        \n        # 3. 基于认知状态的动态权重计算\n        cognitive_weights = self._compute_cognitive_weights(cognitive_state)\n        \n        # 4. 三层提示生成\n        layered_prompts = {\n            'layer1': self._generate_associative_prompts(\n                node_embeddings, subgraph, cognitive_weights['creativity']\n            ),\n            'layer2': self._generate_structured_prompts(\n                graph_embedding, user_context, cognitive_weights['structure']\n            ),\n            'layer3': self._generate_actionable_prompts(\n                subgraph, cognitive_state, cognitive_weights['precision']\n            )\n        }\n        \n        generation_time = time.time() - start_time\n        \n        # 5. 记录生成历史\n        self.generation_history.append({\n            'timestamp': time.time(),\n            'topic_concepts': topic_concepts,\n            'cognitive_state': cognitive_state.tolist(),\n            'generation_time': generation_time,\n            'prompt_counts': {k: len(v) for k, v in layered_prompts.items()}\n        })\n        \n        return layered_prompts\n    \n    def _generate_associative_prompts(self, embeddings: torch.Tensor, \n                                    subgraph: Data, creativity_weight: float) -> List[str]:\n        \"\"\"生成开放性联想提示\"\"\"\n        prompts = []\n        \n        # 基于embedding相似性找到相关概念\n        similarities = torch.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)\n        \n        # 加权随机采样\n        num_nodes = embeddings.size(0)\n        sampling_weights = similarities.mean(dim=1) * creativity_weight + torch.rand(num_nodes) * (1 - creativity_weight)\n        \n        top_nodes = torch.topk(sampling_weights, k=min(3, num_nodes)).indices\n        \n        for node_idx in top_nodes:\n            node_info = self.kg.get_node_info(node_idx.item())\n            if node_info:\n                style = 'creative' if creativity_weight > 0.6 else 'structured'\n                prompt = self.prompt_templates['associative'][style].format(\n                    concept=node_info.get('name', '未知概念'),\n                    description=node_info.get('description', '')\n                )\n                prompts.append(prompt)\n        \n        return prompts\n    \n    def _generate_structured_prompts(self, graph_emb: torch.Tensor, \n                                   context: Dict, structure_weight: float) -> List[str]:\n        \"\"\"生成结构化引导提示\"\"\"\n        prompts = []\n        \n        # 基于图嵌入检索相关写作框架\n        frameworks = self._retrieve_writing_frameworks(graph_emb)\n        \n        for framework in frameworks[:2]:  # 取前2个框架\n            detail_level = 'detailed' if structure_weight > 0.7 else 'brief'\n            \n            prompt = self.prompt_templates['structured'][detail_level].format(\n                framework=framework.get('name', ''),\n                steps=framework.get('steps', ''),\n                context=context.get('writing_goal', '创意写作')\n            )\n            prompts.append(prompt)\n        \n        return prompts\n    \n    def _generate_actionable_prompts(self, subgraph: Data, \n                                   cognitive_state: np.ndarray, \n                                   precision_weight: float) -> List[str]:\n        \"\"\"生成具体操作提示\"\"\"\n        prompts = []\n        \n        # 分析认知瓶颈\n        bottleneck = self._identify_cognitive_bottleneck(cognitive_state)\n        \n        # 找到针对性技法\n        techniques = self._find_relevant_techniques(subgraph, bottleneck)\n        \n        for technique in techniques[:2]:  # 取前2个技法\n            specificity = 'high' if precision_weight > 0.6 else 'medium'\n            \n            prompt = self.prompt_templates['actionable'][specificity].format(\n                technique=technique.get('name', ''),\n                steps=technique.get('steps', ''),\n                examples=technique.get('examples', '')\n            )\n            prompts.append(prompt)\n        \n        return prompts\n    \n    def _compute_cognitive_weights(self, cognitive_state: np.ndarray) -> Dict[str, float]:\n        \"\"\"基于认知状态计算权重\"\"\"\n        if len(cognitive_state) < 8:\n            return {'creativity': 0.5, 'structure': 0.5, 'precision': 0.5}\n        \n        creativity_activation = cognitive_state[3]  # 创造力\n        structure_activation = cognitive_state[2]   # 理解力\n        cognitive_load = cognitive_state[7]         # 疲劳度\n        \n        weights = {\n            'creativity': min(1.0, creativity_activation + 0.3 * (1 - cognitive_load)),\n            'structure': structure_activation,\n            'precision': max(0.2, 1.0 - cognitive_load)\n        }\n        \n        return weights\n    \n    def _retrieve_writing_frameworks(self, graph_emb: torch.Tensor) -> List[Dict]:\n        \"\"\"检索写作框架\"\"\"\n        # 简化版本，返回预定义框架\n        frameworks = [\n            {\n                'name': '三幕式结构',\n                'steps': '1. 设置情境和冲突 2. 发展和高潮 3. 解决和结尾',\n                'similarity': 0.8\n            },\n            {\n                'name': '英雄之旅',\n                'steps': '1. 平凡世界 2. 冒险召唤 3. 拒绝召唤 4. 遇见导师',\n                'similarity': 0.7\n            }\n        ]\n        return frameworks\n    \n    def _identify_cognitive_bottleneck(self, cognitive_state: np.ndarray) -> str:\n        \"\"\"识别认知瓶颈\"\"\"\n        if len(cognitive_state) < 8:\n            return 'attention'\n        \n        concept_names = ['attention', 'memory', 'comprehension', 'creativity',\n                        'motivation', 'emotion', 'confidence', 'fatigue']\n        \n        min_idx = np.argmin(cognitive_state[:7])  # 不包括疲劳度\n        return concept_names[min_idx]\n    \n    def _find_relevant_techniques(self, subgraph: Data, bottleneck: str) -> List[Dict]:\n        \"\"\"找到相关技法\"\"\"\n        techniques_map = {\n            'attention': [{\n                'name': '焦点写作法',\n                'steps': '1. 设定5分钟专注时间 2. 只写一个场景 3. 忽略语法错误',\n                'examples': '专注描述一个房间的细节'\n            }],\n            'creativity': [{\n                'name': '自由联想法',\n                'steps': '1. 写下关键词 2. 快速联想相关概念 3. 不做判断记录',\n                'examples': '从\"雨\"联想到\"眼泪\"再到\"告别\"'\n            }],\n            'comprehension': [{\n                'name': '概念梳理法',\n                'steps': '1. 列出主要概念 2. 建立概念关系图 3. 逐步展开',\n                'examples': '主题-人物-情节-环境的关系梳理'\n            }]\n        }\n        \n        return techniques_map.get(bottleneck, [{\n            'name': '通用写作法',\n            'steps': '1. 确定目标 2. 构思大纲 3. 逐步实现',\n            'examples': '根据具体情况调整'\n        }])\n    \n    def _generate_fallback_prompts(self) -> Dict[str, List[str]]:\n        \"\"\"生成后备提示\"\"\"\n        return {\n            'layer1': [\"让我们从一个简单的想法开始，你最近有什么感兴趣的话题吗？\"],\n            'layer2': [\"试试这个结构：开始-发展-转折-结尾，每部分用一段话描述。\"],\n            'layer3': [\"具体建议：先写下三个关键词，然后为每个词写一句话。\"]\n        }\n\n# 简化的知识图谱接口\nclass KnowledgeGraph:\n    \"\"\"知识图谱接口（简化版）\"\"\"\n    \n    def __init__(self):\n        self.nodes = {}\n        self.edges = []\n        self._build_sample_graph()\n    \n    def _build_sample_graph(self):\n        \"\"\"构建示例知识图谱\"\"\"\n        # 添加一些示例节点\n        sample_nodes = [\n            KnowledgeNode(0, NodeType.CONCEPT, \"叙事结构\", \"故事的组织方式\", \n                         np.random.randn(256), {'difficulty': 0.6, 'innovation': 0.7}),\n            KnowledgeNode(1, NodeType.TECHNIQUE, \"对话描写\", \"通过对话推进情节\", \n                         np.random.randn(256), {'difficulty': 0.4, 'innovation': 0.5}),\n            KnowledgeNode(2, NodeType.ELEMENT, \"人物塑造\", \"创造生动的角色\", \n                         np.random.randn(256), {'difficulty': 0.7, 'innovation': 0.8})\n        ]\n        \n        for node in sample_nodes:\n            self.nodes[node.node_id] = node\n    \n    def find_concept_nodes(self, concepts: List[str]) -> List[int]:\n        \"\"\"根据概念名称查找节点ID\"\"\"\n        found_nodes = []\n        for concept in concepts:\n            for node_id, node in self.nodes.items():\n                if concept.lower() in node.name.lower():\n                    found_nodes.append(node_id)\n        return found_nodes[:5]  # 限制数量\n    \n    def get_node_info(self, node_id: int) -> Optional[Dict]:\n        \"\"\"获取节点信息\"\"\"\n        if node_id in self.nodes:\n            node = self.nodes[node_id]\n            return {\n                'name': node.name,\n                'description': node.description,\n                'type': node.node_type,\n                'attributes': node.attributes\n            }\n        return None\n    \n    def get_graph_data(self) -> Data:\n        \"\"\"获取图数据对象\"\"\"\n        if not self.nodes:\n            # 返回空图\n            return Data(x=torch.randn(1, 256), edge_index=torch.empty((2, 0), dtype=torch.long))\n        \n        # 构建特征矩阵\n        node_features = []\n        node_types = []\n        \n        for node_id in sorted(self.nodes.keys()):\n            node = self.nodes[node_id]\n            node_features.append(node.features)\n            node_types.append(node.node_type)\n        \n        x = torch.FloatTensor(np.array(node_features))\n        node_type = torch.LongTensor(node_types)\n        \n        # 简单的边连接（全连接图的简化版）\n        num_nodes = len(self.nodes)\n        if num_nodes > 1:\n            edge_list = []\n            for i in range(num_nodes):\n                for j in range(i+1, min(i+3, num_nodes)):  # 每个节点连接2个邻居\n                    edge_list.extend([(i, j), (j, i)])\n            \n            if edge_list:\n                edge_index = torch.LongTensor(edge_list).t()\n            else:\n                edge_index = torch.empty((2, 0), dtype=torch.long)\n        else:\n            edge_index = torch.empty((2, 0), dtype=torch.long)\n        \n        return Data(x=x, edge_index=edge_index, node_type=node_type)"