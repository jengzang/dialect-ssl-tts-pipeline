"""
方言数据增强

扩展方言翻译数据集，从小样本扩展到更大规模。

增强策略：
1. 回译（Back-translation）
2. 同义词替换
3. 句子重组
4. 数据合成
"""

import logging
import json
import random
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DialectDataAugmenter:
    """方言数据增强器"""

    def __init__(self, seed: int = 42):
        """
        初始化增强器

        Args:
            seed: 随机种子
        """
        self.seed = seed
        random.seed(seed)

        # 常见方言词汇映射（示例）
        self.dialect_synonyms = {
            # 上海话
            '侬': ['你', '您'],
            '伊': ['他', '她'],
            '伊拉': ['他们', '她们'],
            '今朝': ['今天', '今日'],
            '老好': ['很好', '非常好'],
            '欢喜': ['喜欢', '喜爱'],
            '汏': ['洗'],

            # 粤语
            '嘅': ['的'],
            '喺': ['在'],
            '咗': ['了'],
            '冇': ['没有', '没'],
            '唔': ['不'],
            '嗰': ['那'],
            '呢': ['这'],
            '佢': ['他', '她'],
        }

        # 普通话同义词
        self.mandarin_synonyms = {
            '今天': ['今日'],
            '很好': ['非常好', '挺好'],
            '喜欢': ['喜爱', '爱好'],
            '他们': ['她们'],
            '去': ['前往'],
            '看见': ['看到', '见到'],
            '知道': ['了解', '明白'],
            '但是': ['可是', '然而'],
        }

    def load_data(self, data_path: str) -> List[Dict[str, str]]:
        """
        加载数据

        Args:
            data_path: 数据文件路径（CSV 或 JSON）

        Returns:
            数据列表
        """
        data_path = Path(data_path)

        if data_path.suffix == '.csv':
            df = pd.read_csv(data_path)
            data = df.to_dict('records')
        elif data_path.suffix == '.json':
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")

        logger.info(f"Loaded {len(data)} samples from {data_path}")
        return data

    def save_data(self, data: List[Dict[str, str]], output_path: str):
        """
        保存数据

        Args:
            data: 数据列表
            output_path: 输出文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)

        if output_path.suffix == '.json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        elif output_path.suffix == '.csv':
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False, encoding='utf-8')
        else:
            raise ValueError(f"Unsupported file format: {output_path.suffix}")

        logger.info(f"Saved {len(data)} samples to {output_path}")

    def synonym_replacement(
        self,
        text: str,
        synonyms: Dict[str, List[str]],
        prob: float = 0.3
    ) -> str:
        """
        同义词替换

        Args:
            text: 原始文本
            synonyms: 同义词字典
            prob: 替换概率

        Returns:
            替换后的文本
        """
        for word, replacements in synonyms.items():
            if word in text and random.random() < prob:
                replacement = random.choice(replacements)
                text = text.replace(word, replacement, 1)

        return text

    def augment_sample(
        self,
        sample: Dict[str, str],
        num_augmentations: int = 2
    ) -> List[Dict[str, str]]:
        """
        增强单个样本

        Args:
            sample: 原始样本
            num_augmentations: 增强数量

        Returns:
            增强后的样本列表（包含原始样本）
        """
        augmented = [sample]  # 保留原始样本

        for _ in range(num_augmentations):
            # 方言同义词替换
            dialect_aug = self.synonym_replacement(
                sample['dialect'],
                self.dialect_synonyms,
                prob=0.3
            )

            # 普通话同义词替换
            mandarin_aug = self.synonym_replacement(
                sample['mandarin'],
                self.mandarin_synonyms,
                prob=0.3
            )

            # 只添加与原始不同的样本
            if dialect_aug != sample['dialect'] or mandarin_aug != sample['mandarin']:
                augmented.append({
                    'dialect': dialect_aug,
                    'mandarin': mandarin_aug
                })

        return augmented

    def augment_dataset(
        self,
        data: List[Dict[str, str]],
        target_size: int = 500,
        num_augmentations_per_sample: int = 2
    ) -> List[Dict[str, str]]:
        """
        增强整个数据集

        Args:
            data: 原始数据
            target_size: 目标数据集大小
            num_augmentations_per_sample: 每个样本的增强数量

        Returns:
            增强后的数据集
        """
        logger.info(f"Augmenting dataset from {len(data)} to {target_size} samples...")

        augmented_data = []

        # 计算需要的增强轮数
        augmentations_needed = target_size - len(data)
        rounds = max(1, augmentations_needed // len(data))

        for round_idx in range(rounds + 1):
            for sample in tqdm(data, desc=f"Round {round_idx + 1}"):
                if len(augmented_data) >= target_size:
                    break

                # 第一轮：添加原始样本
                if round_idx == 0:
                    augmented_data.append(sample)

                # 后续轮：添加增强样本
                else:
                    augmented = self.augment_sample(
                        sample,
                        num_augmentations=num_augmentations_per_sample
                    )
                    # 跳过原始样本（索引 0）
                    augmented_data.extend(augmented[1:])

            if len(augmented_data) >= target_size:
                break

        # 截断到目标大小
        augmented_data = augmented_data[:target_size]

        # 去重
        unique_data = []
        seen = set()
        for item in augmented_data:
            key = (item['dialect'], item['mandarin'])
            if key not in seen:
                seen.add(key)
                unique_data.append(item)

        logger.info(f"Final dataset size: {len(unique_data)} (after deduplication)")
        return unique_data

    def split_dataset(
        self,
        data: List[Dict[str, str]],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        划分数据集

        Args:
            data: 数据列表
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例

        Returns:
            包含 train/val/test 的字典
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"

        # 打乱数据
        random.shuffle(data)

        n = len(data)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        splits = {
            'train': data[:train_end],
            'val': data[train_end:val_end],
            'test': data[val_end:]
        }

        logger.info(f"Dataset split: train={len(splits['train'])}, "
                   f"val={len(splits['val'])}, test={len(splits['test'])}")

        return splits


def augment_dialect_data(
    input_path: str,
    output_dir: str,
    target_size: int = 500,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
):
    """
    数据增强的便捷函数

    Args:
        input_path: 输入数据路径
        output_dir: 输出目录
        target_size: 目标数据集大小
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
    """
    augmenter = DialectDataAugmenter(seed=seed)

    # 加载数据
    data = augmenter.load_data(input_path)

    # 增强数据
    augmented_data = augmenter.augment_dataset(data, target_size=target_size)

    # 划分数据集
    splits = augmenter.split_dataset(
        augmented_data,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )

    # 保存数据
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    for split_name, split_data in splits.items():
        output_path = output_dir / f"{split_name}.json"
        augmenter.save_data(split_data, str(output_path))

    logger.info(f"Data augmentation completed. Output: {output_dir}")
