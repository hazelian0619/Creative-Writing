"""
DIFCM模型训练管道
师范生创意写作认知状态训练系统
"""

import os
import json
import time
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 导入自定义模块
from team_a.models.difcm_core import DIFCMModel, DIFCMConfig
from team_a.features.cognitive_features import CognitiveFeatureExtractor, FeaturePreprocessor

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    validation_split: float = 0.2
    patience: int = 15
    min_delta: float = 0.001
    model_save_path: str = "checkpoints/difcm_model.pth"
    log_dir: str = "logs"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class CognitiveDataset(Dataset):
    """认知状态数据集"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Args:
            features: 行为特征 [n_samples, feature_dim]
            labels: 认知状态标签 [n_samples, n_concepts]
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class DIFCMTrainer:
    """DIFCM模型训练器"""
    
    def __init__(self, model: DIFCMModel, config: TrainingConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # 优化器和损失函数
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.learning_rate,
            weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        self.criterion = nn.MSELoss()
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        
        # 创建目录
        os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        logger.info(f"训练器初始化完成，设备: {self.device}")
    
    def create_synthetic_data(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """创建合成的训练数据"""
        np.random.seed(42)
        
        # 特征维度
        feature_dim = 20
        
        # 生成特征
        features = np.random.randn(n_samples, feature_dim)
        
        # 生成对应的认知状态标签
        # 基于特征生成合理的认知状态
        labels = np.zeros((n_samples, self.model.config.n_concepts))
        
        for i in range(n_samples):
            # 模拟认知状态与行为特征的关联
            attention_base = 0.5 + 0.3 * np.tanh(features[i, 0])
            creativity_base = 0.5 + 0.2 * np.sin(features[i, 1]) + 0.1 * features[i, 2]
            memory_base = 0.6 + 0.2 * np.exp(-abs(features[i, 3]))
            
            labels[i] = [
                max(0.1, min(0.9, attention_base)),      # 注意力
                max(0.1, min(0.9, memory_base)),         # 工作记忆
                max(0.1, min(0.9, 0.5 + 0.2 * features[i, 4])),  # 理解力
                max(0.1, min(0.9, creativity_base)),     # 创造力
                max(0.1, min(0.9, 0.7 + 0.2 * np.tanh(features[i, 5]))),  # 动机
                max(0.1, min(0.9, 0.5 + 0.3 * np.sin(features[i, 6])),   # 情绪
                max(0.1, min(0.9, 0.6 + 0.2 * features[i, 7]),   # 自信
                max(0.1, min(0.9, 0.3 + 0.2 * abs(features[i, 8]))),    # 疲劳
            ]
        
        return features, labels
    
    def prepare_data(self, features: np.ndarray, labels: np.ndarray) -> Tuple[DataLoader, DataLoader]:
        """准备训练和验证数据"""
        dataset = CognitiveDataset(features, labels)
        
        # 分割数据集
        n_samples = len(dataset)
        n_train = int(n_samples * (1 - self.config.validation_split))
        
        train_dataset, val_dataset = random_split(
            dataset, [n_train, n_samples - n_train]
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False
        )
        
        logger.info(f"数据准备完成: 训练样本 {len(train_dataset)}, 验证样本 {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            predictions = self.model(features)
            loss = self.criterion(predictions, labels)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.debug(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        """验证模型"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                
                predictions = self.model(features)
                loss = self.criterion(predictions, labels)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """计算评估指标"""
        metrics = {}
        
        # 整体指标
        metrics['mse'] = mean_squared_error(targets, predictions)
        metrics['mae'] = mean_absolute_error(targets, predictions)
        metrics['r2'] = r2_score(targets, predictions)
        
        # 按概念算指标
        concept_names = ['attention', 'memory', 'comprehension', 'creativity',
                        'motivation', 'emotion', 'confidence', 'fatigue']
        
        for i, concept in enumerate(concept_names):
            mse = mean_squared_error(targets[:, i], predictions[:, i])
            mae = mean_absolute_error(targets[:, i], predictions[:, i])
            r2 = r2_score(targets[:, i], predictions[:, i])
            
            metrics[f'{concept}_mse'] = mse
            metrics[f'{concept}_mae'] = mae
            metrics[f'{concept}_r2'] = r2
        
        return metrics
    
    def save_model(self, filepath: str, metrics: Optional[Dict] = None):
        """保存模型"""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'best_val_loss': self.best_val_loss,
            'metrics': metrics
        }
        
        torch.save(save_dict, filepath)
        logger.info(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        logger.info(f"模型已从 {filepath} 加载")
    
    def plot_training_curves(self, save_path: str):
        """绘制训练曲线"""
        plt.figure(figsize=(12, 4))
        
        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='训练损失')
        plt.plot(self.val_losses, label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('训练与验证损失')
        plt.legend()
        plt.grid(True)
        
        # 学习率曲线
        plt.subplot(1, 2, 2)
        lrs = [self.optimizer.param_groups[0]['lr']] * len(self.train_losses)
        plt.plot(lrs)
        plt.xlabel('Epoch')
        plt.ylabel('学习率')
        plt.title('学习率变化')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def train(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """完整训练流程"""
        logger.info("开始训练DIFCM模型...")
        
        # 准备数据
        train_loader, val_loader = self.prepare_data(features, labels)
        
        # 训练循环
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            # 训练
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 早停检查
            if val_loss < self.best_val_loss - self.config.min_delta:
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
                # 保存最佳模型
                self.save_model(self.config.model_save_path)
                logger.info(f"Epoch {epoch+1}: 保存最佳模型 (val_loss: {val_loss:.4f})")
            else:
                self.early_stopping_counter += 1
            
            # 日志
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}")
                logger.info(f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
                logger.info(f"学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 早停
            if self.early_stopping_counter >= self.config.patience:
                logger.info(f"早停于第 {epoch+1} 轮")
                break
        
        training_time = time.time() - start_time
        logger.info(f"训练完成，耗时: {training_time:.2f}秒")
        
        # 评估最终性能
        final_metrics = self.evaluate_model(train_loader, val_loader)
        
        # 保存训练曲线
        self.plot_training_curves(os.path.join(self.config.log_dir, 'training_curves.png'))
        
        return final_metrics
    
    def evaluate_model(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, float]:
        """评估模型性能"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for loader in [train_loader, val_loader]:
                for features, labels in loader:
                    features = features.to(self.device)
                    predictions = self.model(features)
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(labels.numpy())
        
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        metrics = self.calculate_metrics(predictions, targets)
        
        # 保存评估结果
        eval_path = os.path.join(self.config.log_dir, 'evaluation_results.json')
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        logger.info("=== 模型评估结果 ===")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        return metrics

def main():
    """主函数：演示训练流程"""
    
    # 配置
    difcm_config = DIFCMConfig(
        n_concepts=8,
        learning_rate=0.01,
        decay_rate=0.95
    )
    
    training_config = TrainingConfig(
        batch_size=32,
        learning_rate=0.001,
        num_epochs=100,
        validation_split=0.2,
        patience=15
    )
    
    # 初始化模型
    model = DIFCMModel(difcm_config)
    trainer = DIFCMTrainer(model, training_config)
    
    # 创建合成数据
    features, labels = trainer.create_synthetic_data(n_samples=2000)
    
    # 训练模型
    metrics = trainer.train(features, labels)
    
    # 测试预测
    test_features = features[:5]
    model.eval()
    with torch.no_grad():
        predictions = model(torch.FloatTensor(test_features))
        predictions = predictions.numpy()
    
    print("\n=== 预测示例 ===")
    concept_names = ['attention', 'memory', 'comprehension', 'creativity',
                    'motivation', 'emotion', 'confidence', 'fatigue']
    
    for i, pred in enumerate(predictions):
        print(f"样本 {i+1}:")
        for j, concept in enumerate(concept_names):
            print(f"  {concept}: {pred[j]:.3f}")

if __name__ == "__main__":
    main()