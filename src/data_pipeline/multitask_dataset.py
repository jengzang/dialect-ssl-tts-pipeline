"""
多任务数据集模块

结合翻译和分类任务的数据集。

支持的任务：
1. 方言翻译（Dialect Translation）
2. 口音识别（Accent Classification）
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import random
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class MultitaskDialectDataset(Dataset):
    """多任务方言数据集"""

    def __init__(
        self,
        translation_data_path: Optional[str] = None,
        classification_data_path: Optional[str] = None,
        tokenizer=None,
        max_length: int = 512,
        task_sampling: str = "balanced"
    ):
        """
        初始化多任务数据集

        Args:
            translation_data_path: 翻译数据路径（JSON）
            classification_data_path: 分类数据路径（JSON）
            tokenizer: Tokenizer
            max_length: 最大序列长度
            task_sampling: 任务采样策略
                - "balanced": 平衡采样
                - "proportional": 按数据集大小比例采样
                - "translation_heavy": 偏向翻译任务
                - "classification_heavy": 偏向分类任务
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_sampling = task_sampling

        # 加载数据
        self.translation_data = []
        self.classification_data = []

        if translation_data_path:
            self.translation_data = self._load_json(translation_data_path)
            logger.info(f"Loaded {len(self.translation_data)} translation samples")

        if classification_data_path:
            self.classification_data = self._load_json(classification_data_path)
            logger.info(f"Loaded {len(self.classification_data)} classification samples")

        # 创建任务索引
        self.samples = self._create_task_samples()

        logger.info(f"Total samples: {len(self.samples)}")
        logger.info(f"Task sampling strategy: {task_sampling}")

    def _load_json(self, path: str) -> List[Dict[str, Any]]:
        """加载 JSON 数据"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _create_task_samples(self) -> List[Dict[str, Any]]:
        """创建任务样本索引"""
        samples = []

        # 添加翻译任务样本
        for idx, item in enumerate(self.translation_data):
            samples.append({
                'task': 'translation',
                'task_id': 0,
                'data_idx': idx,
                'data': item
            })

        # 添加分类任务样本
        for idx, item in enumerate(self.classification_data):
            samples.append({
                'task': 'classification',
                'task_id': 1,
                'data_idx': idx,
                'data': item
            })

        # 根据采样策略调整
        if self.task_sampling == "balanced":
            # 平衡采样：确保两个任务样本数相同
            min_samples = min(
                len(self.translation_data),
                len(self.classification_data)
            )
            if min_samples > 0:
                translation_samples = [s for s in samples if s['task'] == 'translation'][:min_samples]
                classification_samples = [s for s in samples if s['task'] == 'classification'][:min_samples]
                samples = translation_samples + classification_samples

        elif self.task_sampling == "translation_heavy":
            # 翻译任务占 70%
            translation_samples = [s for s in samples if s['task'] == 'translation']
            classification_samples = [s for s in samples if s['task'] == 'classification']
            target_classification = int(len(translation_samples) * 0.3 / 0.7)
            classification_samples = classification_samples[:target_classification]
            samples = translation_samples + classification_samples

        elif self.task_sampling == "classification_heavy":
            # 分类任务占 70%
            translation_samples = [s for s in samples if s['task'] == 'translation']
            classification_samples = [s for s in samples if s['task'] == 'classification']
            target_translation = int(len(classification_samples) * 0.3 / 0.7)
            translation_samples = translation_samples[:target_translation]
            samples = translation_samples + classification_samples

        # 打乱顺序
        random.shuffle(samples)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        task = sample['task']
        data = sample['data']

        if task == 'translation':
            return self._process_translation(data, sample['task_id'])
        else:
            return self._process_classification(data, sample['task_id'])

    def _process_translation(
        self,
        data: Dict[str, str],
        task_id: int
    ) -> Dict[str, torch.Tensor]:
        """
        处理翻译任务样本

        Args:
            data: 包含 'dialect' 和 'mandarin' 的字典
            task_id: 任务 ID

        Returns:
            处理后的样本
        """
        dialect_text = data['dialect']
        mandarin_text = data['mandarin']

        # 构建输入
        input_text = f"[翻译任务] 请将以下方言翻译成普通话：\n方言：{dialect_text}\n普通话："
        target_text = f"{mandarin_text}"

        # Tokenize
        full_text = input_text + target_text

        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        # 创建标签（只计算目标部分的损失）
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True
        )
        input_length = len(input_encoding['input_ids'])

        labels = input_ids.clone()
        labels[:input_length] = -100  # 忽略输入部分的损失

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'task_id': torch.tensor(task_id, dtype=torch.long),
            'task_type': 'translation'
        }

    def _process_classification(
        self,
        data: Dict[str, Any],
        task_id: int
    ) -> Dict[str, torch.Tensor]:
        """
        处理分类任务样本

        Args:
            data: 包含 'text' 和 'label' 的字典
            task_id: 任务 ID

        Returns:
            处理后的样本
        """
        text = data.get('text', data.get('dialect', ''))
        label = data.get('label', data.get('accent', 0))

        # 如果 label 是字符串，转换为整数
        if isinstance(label, str):
            # 简单的标签映射
            label_map = {
                'shanghai': 0,
                'cantonese': 1,
                'mandarin': 2,
                'other': 3
            }
            label = label_map.get(label.lower(), 0)

        # 构建输入
        input_text = f"[分类任务] 请识别以下文本的方言类型：\n文本：{text}\n方言类型："

        # Tokenize
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'classification_label': torch.tensor(label, dtype=torch.long),
            'task_id': torch.tensor(task_id, dtype=torch.long),
            'task_type': 'classification'
        }

    def get_task_distribution(self) -> Dict[str, int]:
        """
        获取任务分布

        Returns:
            任务分布字典
        """
        distribution = {}
        for sample in self.samples:
            task = sample['task']
            distribution[task] = distribution.get(task, 0) + 1

        return distribution


def create_multitask_dataloaders(
    translation_train_path: str,
    translation_val_path: Optional[str],
    classification_train_path: Optional[str],
    classification_val_path: Optional[str],
    tokenizer,
    batch_size: int = 4,
    max_length: int = 512,
    task_sampling: str = "balanced"
) -> Tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader]]:
    """
    创建多任务数据加载器的便捷函数

    Args:
        translation_train_path: 翻译训练数据路径
        translation_val_path: 翻译验证数据路径
        classification_train_path: 分类训练数据路径
        classification_val_path: 分类验证数据路径
        tokenizer: Tokenizer
        batch_size: 批次大小
        max_length: 最大序列长度
        task_sampling: 任务采样策略

    Returns:
        训练和验证数据加载器
    """
    # 训练集
    train_dataset = MultitaskDialectDataset(
        translation_data_path=translation_train_path,
        classification_data_path=classification_train_path,
        tokenizer=tokenizer,
        max_length=max_length,
        task_sampling=task_sampling
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Windows 兼容性
    )

    # 验证集
    val_loader = None
    if translation_val_path or classification_val_path:
        val_dataset = MultitaskDialectDataset(
            translation_data_path=translation_val_path,
            classification_data_path=classification_val_path,
            tokenizer=tokenizer,
            max_length=max_length,
            task_sampling=task_sampling
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

    logger.info(f"Train dataset: {len(train_dataset)} samples")
    if val_loader:
        logger.info(f"Val dataset: {len(val_dataset)} samples")

    # 打印任务分布
    train_dist = train_dataset.get_task_distribution()
    logger.info(f"Train task distribution: {train_dist}")

    return train_loader, val_loader
